use anyhow::{Context, Result, anyhow};
use ash::vk;

use crate::render::camera::{compute_pixel_to_ray, compute_view_proj};
use crate::render::capture::{CaptureMetadata, RenderCapture, cmd_copy_image_to_buffer};
use crate::render::device::RenderDevice;
#[cfg(not(target_os = "android"))]
use crate::render::egui_renderer::{EguiFrame, EguiRenderer};
use crate::render::frame::FrameContext;
use crate::render::graph::RenderGraph;
use crate::render::passes::blit_to_swapchain;
use crate::render::passes::rt_direct_lighting::RtDirectLightingShaders;
use crate::render::passes::rt_direct_lighting::{RtDirectLightingCreateInfo, RtDirectLightingPass};
use crate::render::passes::rt_resolve::{RtResolveCreateInfo, RtResolvePass};
use crate::render::passes::rt_restir_di::{RtRestirDiCreateInfo, RtRestirDiPass};
use crate::render::passes::rt_restir_gi::{
    RtRestirGiCreateInfo, RtRestirGiPass, RtRestirGiShaders,
};
use crate::render::passes::rt_surface::{RtSurfaceCreateInfo, RtSurfacePass, RtSurfaceShaders};
use crate::render::passes::rt_temporal::{RtTemporalCreateInfo, RtTemporalPass};
use crate::render::resource::{AccessKind, QueueType};
use crate::render::restir_di::build_direct_lights_from_ucvh;
use crate::render::rt_history::{
    GpuRtHistoryUniforms, RT_HISTORY_FLAG_AS_REBUILT, RT_HISTORY_FLAG_CAMERA_CUT,
    RT_HISTORY_FLAG_LIGHTS_INVALIDATED, RT_HISTORY_FLAG_RESIZE, RT_HISTORY_FLAG_SCENE_INVALIDATED,
};
use crate::render::rt_scene::RtSceneBackend;
use crate::render::rt_settings::{RtDebugView, RtSettings};
use crate::render::scene_ubo::{
    LightingSettings, RenderMode, SceneUniformBuffer, SceneUniformInputs, VptDebugView,
    build_scene_uniforms,
};
use crate::render::vpt_pipeline::{VptCameraFrame, VptFrameRecordResult};
use crate::voxel::gpu_upload::UcvhGpuResources;
use crate::voxel::ucvh::Ucvh;

const RT_SCENE_KEY_WORDS: usize = 19;

pub struct RtFrameInputs<'a> {
    pub scene_ubo: &'a SceneUniformBuffer,
    pub camera: VptCameraFrame,
    pub sun_direction: glam::Vec3,
    pub sun_intensity: glam::Vec3,
    pub elapsed_seconds: f32,
    pub lighting_settings: LightingSettings,
    pub rt_settings: RtSettings,
    pub capture: Option<&'a mut RenderCapture>,
    pub ucvh_ready: bool,
    pub ucvh: Option<&'a Ucvh>,
    pub ucvh_gpu: Option<&'a UcvhGpuResources>,
    pub external_history_reset_generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtFrameSkipReason {
    UcvhUploadPending,
    CpuUcvhSceneMissing,
    AccelerationStructureLoaderMissing,
    AccelerationStructureRebuildFailed,
    AccelerationStructureMissing,
    UcvhGpuDescriptorsMissing,
    RequiredPassesMissing,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RtFrameStatus {
    pub frame_resources_ready: bool,
    pub surface_ready: bool,
    pub restir_di_history_ready: bool,
    pub restir_gi_history_ready: bool,
    pub restir_di_rendered: bool,
    pub restir_gi_rendered: bool,
    pub direct_lighting_ready: bool,
    pub temporal_ready: bool,
    pub resolve_ready: bool,
    pub skip_reason: Option<RtFrameSkipReason>,
}

#[derive(Default)]
pub struct RtPipelineFrameState {
    pub surface_initialized: bool,
    pub restir_di_history_initialized: bool,
    pub restir_gi_history_initialized: bool,
    pub restir_di_rendered: bool,
    pub restir_gi_rendered: bool,
    pub direct_lighting_initialized: bool,
    pub temporal_initialized: bool,
    pub resolve_initialized: bool,
    pub skip_reason: Option<RtFrameSkipReason>,
    pub history_reset_generation: u32,
    pub as_rebuild_generation: u32,
    last_scene_key: Option<[u32; RT_SCENE_KEY_WORDS]>,
    previous_view_proj: Option<glam::Mat4>,
    previous_resolution: Option<[u32; 2]>,
}

impl RtPipelineFrameState {
    fn reset_history(&mut self, generation: u32) {
        self.history_reset_generation = self.history_reset_generation.max(generation);
        self.history_reset_generation = self.history_reset_generation.wrapping_add(1);
        self.surface_initialized = false;
        self.restir_di_history_initialized = false;
        self.restir_gi_history_initialized = false;
        self.restir_di_rendered = false;
        self.restir_gi_rendered = false;
        self.direct_lighting_initialized = false;
        self.temporal_initialized = false;
        self.resolve_initialized = false;
        self.skip_reason = None;
        self.previous_view_proj = None;
        self.previous_resolution = None;
    }
}

pub struct RtRuntimePipeline {
    rt_surface_pass: Option<RtSurfacePass>,
    rt_restir_di_pass: Option<RtRestirDiPass>,
    rt_restir_gi_pass: Option<RtRestirGiPass>,
    rt_direct_lighting_pass: Option<RtDirectLightingPass>,
    rt_temporal_pass: Option<RtTemporalPass>,
    rt_resolve_pass: Option<RtResolvePass>,
    rt_scene: RtSceneBackend,
    frame_state: RtPipelineFrameState,
}

impl Default for RtRuntimePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl RtRuntimePipeline {
    pub fn new() -> Self {
        Self {
            rt_surface_pass: None,
            rt_restir_di_pass: None,
            rt_restir_gi_pass: None,
            rt_direct_lighting_pass: None,
            rt_temporal_pass: None,
            rt_resolve_pass: None,
            rt_scene: RtSceneBackend::default(),
            frame_state: RtPipelineFrameState::default(),
        }
    }

    pub fn reset_history(&mut self, generation: u32) {
        self.frame_state.reset_history(generation);
    }

    pub fn as_rebuild_generation(&self) -> u32 {
        self.frame_state.as_rebuild_generation
    }

    pub fn has_frame_resources(&self) -> bool {
        self.rt_surface_pass.is_some()
            || self.rt_restir_di_pass.is_some()
            || self.rt_restir_gi_pass.is_some()
            || self.rt_direct_lighting_pass.is_some()
            || self.rt_temporal_pass.is_some()
            || self.rt_resolve_pass.is_some()
    }

    pub fn frame_status(&self) -> RtFrameStatus {
        RtFrameStatus {
            frame_resources_ready: self.has_frame_resources(),
            surface_ready: self.frame_state.surface_initialized,
            restir_di_history_ready: self.frame_state.restir_di_history_initialized,
            restir_gi_history_ready: self.frame_state.restir_gi_history_initialized,
            restir_di_rendered: self.frame_state.restir_di_rendered,
            restir_gi_rendered: self.frame_state.restir_gi_rendered,
            direct_lighting_ready: self.frame_state.direct_lighting_initialized,
            temporal_ready: self.frame_state.temporal_initialized,
            resolve_ready: self.frame_state.resolve_initialized,
            skip_reason: self.frame_state.skip_reason,
        }
    }

    pub fn ensure_passes(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh: Option<&Ucvh>,
        ucvh_gpu: Option<&UcvhGpuResources>,
        rt_settings: RtSettings,
    ) {
        let extent = renderer.swapchain_extent();
        let frame_count = scene_ubo.frame_count();
        self.ensure_rt_surface_pass(renderer, scene_ubo, ucvh_gpu, extent.width, extent.height);
        self.ensure_rt_restir_di_pass(
            renderer,
            scene_ubo,
            ucvh,
            rt_settings.restir_di_enabled,
            extent.width,
            extent.height,
        );
        self.ensure_rt_restir_gi_pass(
            renderer,
            scene_ubo,
            ucvh_gpu,
            rt_settings.restir_gi_enabled,
            extent.width,
            extent.height,
        );
        self.ensure_rt_direct_lighting_pass(
            renderer,
            frame_count,
            ucvh_gpu,
            extent.width,
            extent.height,
        );
        self.ensure_rt_temporal_pass(renderer, frame_count, extent.width, extent.height);
        self.ensure_rt_resolve_pass(renderer, frame_count, extent.width, extent.height);
        if frame_count == 0 {
            tracing::warn!("RT pipeline cannot render without scene UBO frame slots");
        }
    }

    pub fn resize(
        &mut self,
        renderer: &RenderDevice,
        _scene_ubo: &SceneUniformBuffer,
        width: u32,
        height: u32,
    ) -> Result<()> {
        renderer.wait_idle()?;
        if let Some(pass) = &mut self.rt_surface_pass {
            pass.resize_images(renderer.device(), renderer.allocator(), width, height)
                .context("failed to resize RT surface image")?;
        }
        if let Some(pass) = &mut self.rt_restir_di_pass {
            pass.resize_buffers(renderer.device(), renderer.allocator(), width, height)
                .context("failed to resize RT ReSTIR-DI reservoirs")?;
        }
        if let Some(pass) = &mut self.rt_restir_gi_pass {
            pass.resize_buffers(renderer.device(), renderer.allocator(), width, height)
                .context("failed to resize RT ReSTIR-GI reservoirs")?;
        }
        if let Some(pass) = &mut self.rt_direct_lighting_pass {
            pass.resize_images(renderer.device(), renderer.allocator(), width, height)
                .context("failed to resize RT direct-lighting image")?;
        }
        if let Some(pass) = &mut self.rt_temporal_pass {
            pass.resize_images(renderer.device(), renderer.allocator(), width, height)
                .context("failed to resize RT temporal image")?;
        }
        if let Some(pass) = &mut self.rt_resolve_pass {
            pass.resize_images(renderer.device(), renderer.allocator(), width, height)
                .context("failed to resize RT resolve image")?;
        }
        self.frame_state
            .reset_history(self.frame_state.history_reset_generation);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_and_execute_frame(
        &mut self,
        renderer: &RenderDevice,
        frame: &FrameContext,
        #[cfg(not(target_os = "android"))] _egui_renderer: Option<&mut EguiRenderer>,
        #[cfg(not(target_os = "android"))] _egui_frame: Option<&EguiFrame>,
        mut inputs: RtFrameInputs<'_>,
    ) -> Result<VptFrameRecordResult> {
        let mut graph = RenderGraph::new();
        let mut pending_capture = None;
        if inputs.external_history_reset_generation > self.frame_state.history_reset_generation {
            self.frame_state
                .reset_history(inputs.external_history_reset_generation);
        }

        let scene_key = Self::make_scene_key(
            inputs.sun_direction,
            inputs.sun_intensity,
            inputs.lighting_settings,
            inputs.rt_settings,
        );
        let scene_changed = self.frame_state.last_scene_key != Some(scene_key);
        if scene_changed {
            self.frame_state
                .reset_history(self.frame_state.history_reset_generation);
            self.frame_state.last_scene_key = Some(scene_key);
        }

        let mut skip_reason = None;
        let as_rebuilt = if inputs.ucvh_ready {
            if let Some(ucvh) = inputs.ucvh {
                if let Some(acceleration_structure_loader) =
                    renderer.acceleration_structure_loader()
                {
                    renderer.wait_for_other_frame_fences(frame.in_flight_fence)?;
                    let scratch_alignment = renderer
                        .acceleration_structure_properties()
                        .min_acceleration_structure_scratch_offset_alignment
                        as vk::DeviceSize;
                    match self.rt_scene.rebuild_gpu(
                        renderer.device(),
                        renderer.allocator(),
                        acceleration_structure_loader,
                        frame.command_buffer,
                        scratch_alignment,
                        ucvh,
                    ) {
                        Ok(()) => {
                            let rebuilt = self.rt_scene.build_generation
                                != self.frame_state.as_rebuild_generation;
                            self.frame_state.as_rebuild_generation = self.rt_scene.build_generation;
                            rebuilt
                        }
                        Err(error) => {
                            skip_reason.get_or_insert(
                                RtFrameSkipReason::AccelerationStructureRebuildFailed,
                            );
                            tracing::error!(
                                %error,
                                "failed to rebuild RT scene acceleration structures"
                            );
                            false
                        }
                    }
                } else {
                    skip_reason
                        .get_or_insert(RtFrameSkipReason::AccelerationStructureLoaderMissing);
                    tracing::warn!(
                        "skipping RT scene AS rebuild without acceleration structure loader"
                    );
                    false
                }
            } else {
                skip_reason.get_or_insert(RtFrameSkipReason::CpuUcvhSceneMissing);
                false
            }
        } else {
            skip_reason.get_or_insert(RtFrameSkipReason::UcvhUploadPending);
            tracing::warn!("rendering RT fallback output until UCVH data is ready");
            false
        };

        let camera = inputs.camera;
        let pixel_to_ray = compute_pixel_to_ray(
            camera.position,
            camera.forward,
            camera.up,
            camera.fov_y_radians,
            frame.swapchain_extent.width,
            frame.swapchain_extent.height,
        );
        let scene_data = build_scene_uniforms(SceneUniformInputs {
            pixel_to_ray,
            resolution: [frame.swapchain_extent.width, frame.swapchain_extent.height],
            camera_right: camera.up.cross(camera.forward).normalize(),
            camera_up: camera.up.normalize(),
            camera_forward: camera.forward.normalize(),
            aperture_radius: camera.aperture_radius,
            focal_distance: camera.focal_distance,
            sun_direction: inputs.sun_direction,
            sun_intensity: inputs.sun_intensity,
            sky_color: [0.4, 0.5, 0.7],
            ground_color: [0.15, 0.1, 0.08],
            time: inputs.elapsed_seconds,
            lighting_settings: inputs.lighting_settings,
            vpt_sample_index: 0,
            history_reset_generation: self.frame_state.history_reset_generation,
        });
        inputs.scene_ubo.update(frame.frame_slot, &scene_data);

        let current_view_proj = compute_view_proj(
            camera.position,
            camera.forward,
            camera.up,
            camera.fov_y_radians,
            frame.swapchain_extent.width,
            frame.swapchain_extent.height,
        );
        let previous_view_proj = self
            .frame_state
            .previous_view_proj
            .unwrap_or(current_view_proj);
        let previous_resolution = self
            .frame_state
            .previous_resolution
            .unwrap_or([frame.swapchain_extent.width, frame.swapchain_extent.height]);
        let mut history_flags = 0;
        if self.frame_state.previous_view_proj.is_none() {
            history_flags |= RT_HISTORY_FLAG_CAMERA_CUT;
        }
        if self.frame_state.previous_resolution.is_none()
            || previous_resolution != [frame.swapchain_extent.width, frame.swapchain_extent.height]
        {
            history_flags |= RT_HISTORY_FLAG_RESIZE;
        }
        if scene_changed {
            history_flags |= RT_HISTORY_FLAG_SCENE_INVALIDATED | RT_HISTORY_FLAG_LIGHTS_INVALIDATED;
        }
        if as_rebuilt {
            history_flags |= RT_HISTORY_FLAG_AS_REBUILT;
        }
        let history_uniforms = GpuRtHistoryUniforms {
            current_view_proj: current_view_proj.transpose().to_cols_array_2d(),
            previous_view_proj: previous_view_proj.transpose().to_cols_array_2d(),
            current_resolution: [frame.swapchain_extent.width, frame.swapchain_extent.height],
            previous_resolution,
            current_jitter: [0.0, 0.0],
            previous_jitter: [0.0, 0.0],
            frame_index: frame.frame_index as u32,
            history_reset_generation: self.frame_state.history_reset_generation,
            as_rebuild_generation: self.frame_state.as_rebuild_generation,
            scene_generation: self.rt_scene.dirty_generation,
            lights_generation: if scene_changed {
                self.frame_state.history_reset_generation
            } else {
                0
            },
            temporal_denoise_enabled: inputs.rt_settings.temporal_denoise_enabled as u32,
            flags: history_flags,
            debug_view: inputs.rt_settings.debug_view.as_gpu_value(),
            history_length: inputs.rt_settings.history_length,
            normal_threshold: inputs.rt_settings.normal_threshold,
            depth_threshold: inputs.rt_settings.depth_threshold,
            _pad0: 0,
        };

        let mut rt_graph_rendered = false;
        let mut rt_restir_di_rendered = false;
        let mut rt_restir_gi_rendered = false;
        if let (Some(rt_surface), Some(rt_direct_lighting), Some(rt_temporal), Some(rt_resolve)) = (
            &self.rt_surface_pass,
            &self.rt_direct_lighting_pass,
            &self.rt_temporal_pass,
            &self.rt_resolve_pass,
        ) {
            match (self.rt_scene.tlas_handle(), self.rt_scene.aabb_buffer()) {
                (Some(tlas), Some(aabb_buffer)) => {
                    if let Some(ucvh_gpu) = inputs.ucvh_gpu {
                        rt_surface.update_tlas_descriptor(
                            renderer.device(),
                            frame.frame_slot,
                            tlas,
                        );
                        rt_surface.update_aabb_descriptor(
                            renderer.device(),
                            frame.frame_slot,
                            aabb_buffer,
                        );
                        rt_surface.update_ucvh_descriptors(
                            renderer.device(),
                            frame.frame_slot,
                            ucvh_gpu,
                        );
                        rt_direct_lighting.update_tlas_descriptor(
                            renderer.device(),
                            frame.frame_slot,
                            tlas,
                        );
                        rt_direct_lighting.update_aabb_descriptor(
                            renderer.device(),
                            frame.frame_slot,
                            aabb_buffer,
                        );
                        rt_direct_lighting.update_ucvh_descriptors(
                            renderer.device(),
                            frame.frame_slot,
                            ucvh_gpu,
                        );
                        if let Some(rt_restir_gi) = &self.rt_restir_gi_pass {
                            rt_restir_gi.update_tlas_descriptor(
                                renderer.device(),
                                frame.frame_slot,
                                tlas,
                            );
                            rt_restir_gi.update_aabb_descriptor(
                                renderer.device(),
                                frame.frame_slot,
                                aabb_buffer,
                            );
                            rt_restir_gi.update_ucvh_descriptors(
                                renderer.device(),
                                frame.frame_slot,
                                ucvh_gpu,
                            );
                        }
                        rt_surface.update_history_uniforms(frame.frame_slot, &history_uniforms);
                        rt_temporal.update_history_uniforms(frame.frame_slot, &history_uniforms);
                        let rt_surface_outputs = rt_surface.register_graph(
                            &mut graph,
                            frame.frame_slot,
                            self.frame_state.surface_initialized,
                        );
                        let (rt_restir_di_reservoir_resource, rt_restir_di_reservoir_buffer) =
                            if inputs.rt_settings.restir_di_enabled {
                                if let Some(rt_restir_di) = &self.rt_restir_di_pass {
                                    rt_restir_di.update_history_uniforms(
                                        frame.frame_slot,
                                        &history_uniforms,
                                    );
                                    rt_restir_di.update_uniforms(
                                        frame.frame_slot,
                                        inputs.rt_settings,
                                        frame.frame_index,
                                        self.frame_state.restir_di_history_initialized,
                                    );
                                    rt_restir_di.update_frame_descriptors(
                                        renderer.device(),
                                        frame.frame_slot,
                                        frame.frame_index,
                                        rt_surface.surface_buffer(),
                                        inputs.rt_settings.restir_di_spatial_enabled
                                            && inputs.rt_settings.restir_di_spatial_sample_count
                                                > 0,
                                    );
                                    let rt_restir_di_outputs = rt_restir_di.register_graph(
                                        &mut graph,
                                        frame.frame_slot,
                                        frame.frame_index,
                                        rt_surface_outputs.surface,
                                        self.frame_state.restir_di_history_initialized,
                                        inputs.rt_settings.restir_di_spatial_enabled
                                            && inputs.rt_settings.restir_di_spatial_sample_count
                                                > 0,
                                    );
                                    (
                                        Some(rt_restir_di_outputs.reservoirs),
                                        Some(
                                            rt_restir_di.output_reservoir_buffer(frame.frame_slot),
                                        ),
                                    )
                                } else {
                                    tracing::warn!(
                                        "RT ReSTIR-DI enabled but pass is not initialized"
                                    );
                                    (None, None)
                                }
                            } else {
                                (None, None)
                            };
                        rt_restir_di_rendered = rt_restir_di_reservoir_resource.is_some();
                        let (rt_restir_gi_reservoir_resource, rt_restir_gi_reservoir_buffer) =
                            if inputs.rt_settings.restir_gi_enabled {
                                if let Some(rt_restir_gi) = &self.rt_restir_gi_pass {
                                    rt_restir_gi.update_history_uniforms(
                                        frame.frame_slot,
                                        &history_uniforms,
                                    );
                                    rt_restir_gi.update_uniforms(
                                        frame.frame_slot,
                                        inputs.rt_settings,
                                        frame.frame_index,
                                        self.frame_state.restir_gi_history_initialized,
                                    );
                                    rt_restir_gi.update_frame_descriptors(
                                        renderer.device(),
                                        frame.frame_slot,
                                        frame.frame_index,
                                        rt_surface.surface_buffer(),
                                    );
                                    let rt_restir_gi_reservoir_buffer =
                                        rt_restir_gi.output_reservoir_buffer(frame.frame_slot);
                                    let rt_restir_gi_outputs = rt_restir_gi.register_graph(
                                        &mut graph,
                                        frame.frame_slot,
                                        frame.frame_index,
                                        rt_surface_outputs.surface,
                                        self.frame_state.restir_gi_history_initialized,
                                    );
                                    (
                                        Some(rt_restir_gi_outputs.reservoirs),
                                        Some(rt_restir_gi_reservoir_buffer),
                                    )
                                } else {
                                    tracing::warn!(
                                        "RT ReSTIR-GI enabled but pass is not initialized"
                                    );
                                    (None, None)
                                }
                            } else {
                                (None, None)
                            };
                        rt_restir_gi_rendered = rt_restir_gi_reservoir_resource.is_some();
                        let restir_di_active = rt_restir_di_rendered;
                        let restir_gi_active = rt_restir_gi_rendered;
                        rt_direct_lighting.update_uniforms(
                            frame.frame_slot,
                            inputs.rt_settings,
                            restir_di_active,
                            restir_gi_active,
                            inputs.lighting_settings.shadows_enabled,
                        );
                        rt_direct_lighting.update_frame_descriptors(
                            renderer.device(),
                            frame.frame_slot,
                            rt_surface.surface_buffer(),
                            rt_restir_di_reservoir_buffer,
                            rt_restir_gi_reservoir_buffer,
                        );
                        let rt_direct_lighting_outputs = rt_direct_lighting.register_graph(
                            &mut graph,
                            frame.frame_slot,
                            rt_surface_outputs.surface,
                            rt_restir_di_reservoir_resource,
                            rt_restir_gi_reservoir_resource,
                            self.frame_state.direct_lighting_initialized,
                        );
                        rt_temporal.update_frame_descriptors(
                            renderer.device(),
                            frame.frame_slot,
                            frame.frame_index,
                            rt_surface.surface_buffer(),
                            rt_direct_lighting.current_radiance_image(),
                        );
                        let rt_temporal_outputs = rt_temporal.register_graph(
                            &mut graph,
                            frame.frame_slot,
                            frame.frame_index,
                            rt_surface_outputs.surface,
                            rt_direct_lighting_outputs.current_radiance,
                            self.frame_state.temporal_initialized,
                        );
                        rt_resolve.update_input_descriptor(
                            renderer.device(),
                            frame.frame_slot,
                            rt_temporal.current_temporal_image(frame.frame_index),
                        );
                        let rt_resolve_outputs = rt_resolve.register_graph(
                            &mut graph,
                            rt_temporal_outputs.temporal_radiance,
                            frame.frame_slot,
                            self.frame_state.resolve_initialized,
                        );
                        rt_graph_rendered = true;
                        let mut capture_dependency = None;
                        let capture_frame = inputs
                            .capture
                            .as_ref()
                            .is_some_and(|capture| capture.should_capture(frame.frame_index));
                        if capture_frame && let Some(capture) = inputs.capture.as_deref_mut() {
                            let output_image = rt_resolve.output_image.handle;
                            let output_extent = rt_resolve.output_image.extent;
                            let readback = capture.ensure_readback(
                                renderer.device(),
                                renderer.allocator(),
                                output_extent.width,
                                output_extent.height,
                            )?;
                            let readback_buffer = readback.handle;
                            let readback_size = readback.size;
                            let readback_usage = readback.usage;
                            let readback_resource = graph.import_buffer_with_access(
                                readback_buffer,
                                readback_size,
                                readback_usage,
                                AccessKind::Undefined,
                            );
                            let capture_writes = graph.add_pass(
                                "capture_rt_resolve",
                                QueueType::Transfer,
                                |builder| {
                                    builder.read_as(
                                        rt_resolve_outputs.output,
                                        AccessKind::TransferRead,
                                    );
                                    builder.write_as(readback_resource, AccessKind::TransferWrite);
                                    Box::new(move |ctx| {
                                        cmd_copy_image_to_buffer(
                                            ctx.device,
                                            ctx.command_buffer,
                                            output_image,
                                            output_extent,
                                            readback_buffer,
                                        );
                                    })
                                },
                            );
                            capture_dependency = Some(capture_writes[0]);
                            let paths = capture.config().paths_for_frame(frame.frame_index);
                            pending_capture = Some(CaptureMetadata {
                                frame_index: frame.frame_index,
                                vpt_sample_index: 0,
                                width: output_extent.width,
                                height: output_extent.height,
                                source: "rt_resolve_output",
                                ppm_path: paths.ppm_path,
                                json_path: paths.json_path,
                                render_backend: "rt",
                                render_mode: capture_render_mode_name(
                                    inputs.lighting_settings.render_mode,
                                ),
                                rt_debug_view: capture_rt_debug_view_name(
                                    inputs.rt_settings.debug_view,
                                ),
                                rt_restir_di_enabled: inputs.rt_settings.restir_di_enabled,
                                rt_restir_di_spatial_enabled: inputs
                                    .rt_settings
                                    .restir_di_spatial_enabled,
                                rt_restir_di_spatial_sample_count: inputs
                                    .rt_settings
                                    .restir_di_spatial_sample_count,
                                rt_restir_gi_enabled: inputs.rt_settings.restir_gi_enabled,
                                rt_temporal_denoise_enabled: inputs
                                    .rt_settings
                                    .temporal_denoise_enabled,
                                rt_frame_rendered: rt_graph_rendered,
                                rt_restir_di_rendered,
                                rt_restir_gi_rendered,
                                rt_resolve_ready: true,
                                restir_di_enabled: false,
                                restir_di_temporal_enabled: false,
                                restir_di_spatial_enabled: false,
                                area_restir_enabled: false,
                                area_restir_temporal_enabled: false,
                                area_restir_spatial_enabled: false,
                                vpt_debug_view: capture_vpt_debug_view_name(
                                    inputs.lighting_settings.vpt_debug_view,
                                ),
                                denoiser_enabled: inputs.lighting_settings.denoiser_enabled(),
                                denoiser_mode: inputs.lighting_settings.denoiser_mode_name(),
                                effective_denoiser_mode: inputs
                                    .lighting_settings
                                    .effective_denoiser_mode_name(),
                            });
                            tracing::info!(
                                frame_index = frame.frame_index,
                                width = output_extent.width,
                                height = output_extent.height,
                                "queued RT resolve capture"
                            );
                        }
                        add_blit_to_swapchain_pass(
                            &mut graph,
                            rt_resolve_outputs.output,
                            rt_resolve.output_image.handle,
                            rt_resolve.output_image.extent,
                            frame,
                            capture_dependency,
                        )?;
                    } else {
                        skip_reason.get_or_insert(RtFrameSkipReason::UcvhGpuDescriptorsMissing);
                        tracing::warn!("skipping RT surface trace without UCVH GPU descriptors");
                        add_swapchain_clear_present_pass(&mut graph, frame)?;
                    }
                }
                _ => {
                    skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureMissing);
                    tracing::warn!("skipping RT surface trace without built TLAS and AABB buffer");
                    add_swapchain_clear_present_pass(&mut graph, frame)?;
                }
            }
        } else {
            skip_reason.get_or_insert(RtFrameSkipReason::RequiredPassesMissing);
            tracing::warn!(
                rt_surface = self.rt_surface_pass.is_some(),
                rt_direct_lighting = self.rt_direct_lighting_pass.is_some(),
                rt_temporal = self.rt_temporal_pass.is_some(),
                rt_resolve = self.rt_resolve_pass.is_some(),
                "skipping RT graph until required passes are initialized"
            );
            add_swapchain_clear_present_pass(&mut graph, frame)?;
        }

        graph.compile()?;
        graph.execute(renderer.device(), frame.command_buffer, frame.frame_index);

        self.frame_state.previous_view_proj = Some(current_view_proj);
        self.frame_state.previous_resolution =
            Some([frame.swapchain_extent.width, frame.swapchain_extent.height]);
        self.frame_state.surface_initialized = rt_graph_rendered;
        self.frame_state.restir_di_history_initialized = rt_graph_rendered && rt_restir_di_rendered;
        self.frame_state.restir_gi_history_initialized = rt_graph_rendered && rt_restir_gi_rendered;
        self.frame_state.restir_di_rendered = rt_graph_rendered && rt_restir_di_rendered;
        self.frame_state.restir_gi_rendered = rt_graph_rendered && rt_restir_gi_rendered;
        self.frame_state.direct_lighting_initialized = rt_graph_rendered;
        self.frame_state.temporal_initialized = rt_graph_rendered;
        self.frame_state.resolve_initialized = rt_graph_rendered;
        self.frame_state.skip_reason = if rt_graph_rendered { None } else { skip_reason };

        Ok(VptFrameRecordResult {
            pending_capture,
            submitted_fence: frame.in_flight_fence,
            rendered_vpt: false,
            traversal_stats_requested: false,
            traversal_stats: None,
        })
    }

    pub fn destroy(
        self,
        device: &ash::Device,
        allocator: &crate::render::allocator::GpuAllocator,
        acceleration_structure_loader: Option<&ash::khr::acceleration_structure::Device>,
    ) {
        if let Some(pass) = self.rt_surface_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.rt_restir_di_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.rt_restir_gi_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.rt_direct_lighting_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.rt_temporal_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.rt_resolve_pass {
            pass.destroy(device, allocator);
        }
        self.rt_scene
            .destroy(device, allocator, acceleration_structure_loader);
    }

    fn ensure_rt_surface_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh_gpu: Option<&UcvhGpuResources>,
        width: u32,
        height: u32,
    ) {
        let Some(ucvh_gpu) = ucvh_gpu else {
            if let Some(pass) = self.rt_surface_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            tracing::warn!("skipping RT surface pass creation without UCVH GPU descriptors");
            return;
        };
        if self
            .rt_surface_pass
            .as_ref()
            .is_some_and(|pass| pass.width() == width && pass.height() == height)
        {
            return;
        }
        if let Some(pass) = self.rt_surface_pass.take() {
            pass.destroy(renderer.device(), renderer.allocator());
        }
        let Some(ray_tracing_pipeline_loader) = renderer.ray_tracing_pipeline_loader() else {
            tracing::error!("failed to create RT surface pass without KHR ray tracing loader");
            return;
        };
        let shaders = RtSurfaceShaders {
            raygen: include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt_surface.rgen.spv")),
            miss: include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt_surface.rmiss.spv")),
            closest_hit: include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt_surface.rchit.spv")),
            intersection: include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt_surface.rint.spv")),
        };
        match RtSurfacePass::new(
            renderer.device(),
            renderer.allocator(),
            ray_tracing_pipeline_loader,
            RtSurfaceCreateInfo {
                rt_pipeline_properties: renderer.rt_pipeline_properties(),
                width,
                height,
                scene_ubo,
                ucvh_gpu,
                shaders,
            },
        ) {
            Ok(pass) => self.rt_surface_pass = Some(pass),
            Err(error) => tracing::error!(%error, "failed to create RT surface pass"),
        }
    }

    fn ensure_rt_temporal_pass(
        &mut self,
        renderer: &RenderDevice,
        frame_count: usize,
        width: u32,
        height: u32,
    ) {
        if self
            .rt_temporal_pass
            .as_ref()
            .is_some_and(|pass| pass.width() == width && pass.height() == height)
        {
            return;
        }
        if let Some(pass) = self.rt_temporal_pass.take() {
            pass.destroy(renderer.device(), renderer.allocator());
        }
        let Some(ray_tracing_pipeline_loader) = renderer.ray_tracing_pipeline_loader() else {
            tracing::error!("failed to create RT temporal pass without KHR ray tracing loader");
            return;
        };
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt_temporal.rgen.spv"));
        match RtTemporalPass::new(
            renderer.device(),
            renderer.allocator(),
            RtTemporalCreateInfo {
                ray_tracing_pipeline_loader,
                rt_pipeline_properties: renderer.rt_pipeline_properties(),
                width,
                height,
                frame_count,
                raygen_spirv: spirv,
            },
        ) {
            Ok(pass) => self.rt_temporal_pass = Some(pass),
            Err(error) => tracing::error!(%error, "failed to create RT temporal pass"),
        }
    }

    fn ensure_rt_direct_lighting_pass(
        &mut self,
        renderer: &RenderDevice,
        frame_count: usize,
        ucvh_gpu: Option<&UcvhGpuResources>,
        width: u32,
        height: u32,
    ) {
        let Some(ucvh_gpu) = ucvh_gpu else {
            if let Some(pass) = self.rt_direct_lighting_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            return;
        };
        if self
            .rt_direct_lighting_pass
            .as_ref()
            .is_some_and(|pass| pass.width() == width && pass.height() == height)
        {
            return;
        }
        if let Some(pass) = self.rt_direct_lighting_pass.take() {
            pass.destroy(renderer.device(), renderer.allocator());
        }
        let Some(ray_tracing_pipeline_loader) = renderer.ray_tracing_pipeline_loader() else {
            tracing::error!(
                "failed to create RT direct-lighting pass without KHR ray tracing loader"
            );
            return;
        };
        let spirv = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/rt_direct_lighting.rgen.spv"
        ));
        let miss_spirv = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/rt_direct_lighting.rmiss.spv"
        ));
        let closest_hit_spirv = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/rt_direct_lighting.rchit.spv"
        ));
        let intersection_spirv = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/rt_direct_lighting.rint.spv"
        ));
        match RtDirectLightingPass::new(
            renderer.device(),
            renderer.allocator(),
            RtDirectLightingCreateInfo {
                ray_tracing_pipeline_loader,
                rt_pipeline_properties: renderer.rt_pipeline_properties(),
                width,
                height,
                frame_count,
                ucvh_gpu,
                shaders: RtDirectLightingShaders {
                    raygen: spirv,
                    miss: miss_spirv,
                    closest_hit: closest_hit_spirv,
                    intersection: intersection_spirv,
                },
            },
        ) {
            Ok(pass) => self.rt_direct_lighting_pass = Some(pass),
            Err(error) => tracing::error!(%error, "failed to create RT direct-lighting pass"),
        }
    }

    fn ensure_rt_restir_di_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh: Option<&Ucvh>,
        restir_di_enabled: bool,
        width: u32,
        height: u32,
    ) {
        if !restir_di_enabled {
            if let Some(pass) = self.rt_restir_di_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            return;
        }
        if self
            .rt_restir_di_pass
            .as_ref()
            .is_some_and(|pass| pass.width() == width && pass.height() == height)
        {
            return;
        }
        if let Some(pass) = self.rt_restir_di_pass.take() {
            pass.destroy(renderer.device(), renderer.allocator());
        }
        let Some(ucvh) = ucvh else {
            tracing::warn!("skipping RT ReSTIR-DI pass creation without CPU UCVH scene");
            return;
        };
        let Some(ray_tracing_pipeline_loader) = renderer.ray_tracing_pipeline_loader() else {
            tracing::error!("failed to create RT ReSTIR-DI pass without KHR ray tracing loader");
            return;
        };
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt_restir_di.rgen.spv"));
        let spatial_spirv = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/rt_restir_di_spatial.rgen.spv"
        ));
        if spirv.is_empty() || spatial_spirv.is_empty() {
            tracing::warn!("RT ReSTIR-DI shader is empty; slangc may not be installed");
            return;
        }
        let direct_lights = build_direct_lights_from_ucvh(ucvh, 4096);
        match RtRestirDiPass::new(
            renderer.device(),
            renderer.allocator(),
            RtRestirDiCreateInfo {
                ray_tracing_pipeline_loader,
                rt_pipeline_properties: renderer.rt_pipeline_properties(),
                width,
                height,
                frame_count: scene_ubo.frame_count(),
                raygen_spirv: spirv,
                spatial_raygen_spirv: spatial_spirv,
                direct_lights: &direct_lights,
            },
        ) {
            Ok(pass) => {
                tracing::info!(
                    width,
                    height,
                    direct_lights = direct_lights.len(),
                    "initialized RT ReSTIR-DI initial reservoir pass"
                );
                self.rt_restir_di_pass = Some(pass);
            }
            Err(error) => tracing::error!(%error, "failed to create RT ReSTIR-DI pass"),
        }
    }

    fn ensure_rt_restir_gi_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh_gpu: Option<&UcvhGpuResources>,
        restir_gi_enabled: bool,
        width: u32,
        height: u32,
    ) {
        if !restir_gi_enabled {
            if let Some(pass) = self.rt_restir_gi_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            return;
        }
        let Some(ucvh_gpu) = ucvh_gpu else {
            if let Some(pass) = self.rt_restir_gi_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            tracing::warn!("skipping RT ReSTIR-GI pass creation without UCVH GPU descriptors");
            return;
        };
        if self
            .rt_restir_gi_pass
            .as_ref()
            .is_some_and(|pass| pass.width() == width && pass.height() == height)
        {
            return;
        }
        if let Some(pass) = self.rt_restir_gi_pass.take() {
            pass.destroy(renderer.device(), renderer.allocator());
        }
        let Some(ray_tracing_pipeline_loader) = renderer.ray_tracing_pipeline_loader() else {
            tracing::error!("failed to create RT ReSTIR-GI pass without KHR ray tracing loader");
            return;
        };
        let shaders = RtRestirGiShaders {
            raygen: include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt_restir_gi.rgen.spv")),
            miss: include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt_restir_gi.rmiss.spv")),
            closest_hit: include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/rt_restir_gi.rchit.spv"
            )),
            intersection: include_bytes!(concat!(
                env!("OUT_DIR"),
                "/shaders/rt_restir_gi.rint.spv"
            )),
        };
        if [
            shaders.raygen,
            shaders.miss,
            shaders.closest_hit,
            shaders.intersection,
        ]
        .iter()
        .any(|spirv| spirv.is_empty())
        {
            tracing::warn!("RT ReSTIR-GI shaders are empty; slangc may not be installed");
            return;
        }
        match RtRestirGiPass::new(
            renderer.device(),
            renderer.allocator(),
            RtRestirGiCreateInfo {
                ray_tracing_pipeline_loader,
                rt_pipeline_properties: renderer.rt_pipeline_properties(),
                width,
                height,
                frame_count: scene_ubo.frame_count(),
                ucvh_gpu,
                shaders,
            },
        ) {
            Ok(pass) => {
                tracing::info!(width, height, "initialized RT ReSTIR-GI reservoir pass");
                self.rt_restir_gi_pass = Some(pass);
            }
            Err(error) => tracing::error!(%error, "failed to create RT ReSTIR-GI pass"),
        }
    }

    fn ensure_rt_resolve_pass(
        &mut self,
        renderer: &RenderDevice,
        frame_count: usize,
        width: u32,
        height: u32,
    ) {
        if self
            .rt_resolve_pass
            .as_ref()
            .is_some_and(|pass| pass.width() == width && pass.height() == height)
        {
            return;
        }
        if let Some(pass) = self.rt_resolve_pass.take() {
            pass.destroy(renderer.device(), renderer.allocator());
        }
        let Some(ray_tracing_pipeline_loader) = renderer.ray_tracing_pipeline_loader() else {
            tracing::error!("failed to create RT resolve pass without KHR ray tracing loader");
            return;
        };
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt_resolve.rgen.spv"));
        match RtResolvePass::new(
            renderer.device(),
            renderer.allocator(),
            RtResolveCreateInfo {
                ray_tracing_pipeline_loader,
                rt_pipeline_properties: renderer.rt_pipeline_properties(),
                width,
                height,
                frame_count,
                raygen_spirv: spirv,
            },
        ) {
            Ok(pass) => self.rt_resolve_pass = Some(pass),
            Err(error) => tracing::error!(%error, "failed to create RT resolve pass"),
        }
    }

    fn make_scene_key(
        sun_direction: glam::Vec3,
        sun_intensity: glam::Vec3,
        lighting_settings: LightingSettings,
        rt_settings: RtSettings,
    ) -> [u32; RT_SCENE_KEY_WORDS] {
        [
            sun_direction.x.to_bits(),
            sun_direction.y.to_bits(),
            sun_direction.z.to_bits(),
            sun_intensity.x.to_bits(),
            sun_intensity.y.to_bits(),
            sun_intensity.z.to_bits(),
            lighting_settings.shadows_enabled as u32,
            lighting_settings.skip_backface_shadows as u32,
            lighting_settings.sun_angular_radius.to_bits(),
            lighting_settings.debug_view.as_gpu_value(),
            rt_settings.restir_di_enabled as u32,
            rt_settings.restir_gi_enabled as u32,
            rt_settings.temporal_denoise_enabled as u32,
            rt_settings.restir_di_spatial_enabled as u32,
            rt_settings.restir_di_spatial_sample_count,
            rt_settings.history_length,
            rt_settings.normal_threshold.to_bits(),
            rt_settings.depth_threshold.to_bits(),
            rt_settings.debug_view.as_gpu_value(),
        ]
    }
}

fn capture_render_mode_name(render_mode: RenderMode) -> &'static str {
    match render_mode {
        RenderMode::Auto => "auto",
        RenderMode::Vpt => "vpt",
        RenderMode::Rt => "rt",
    }
}

fn capture_rt_debug_view_name(debug_view: RtDebugView) -> &'static str {
    match debug_view {
        RtDebugView::Off => "off",
        RtDebugView::Surface => "surface",
        RtDebugView::HitDistance => "hit_distance",
        RtDebugView::HistoryValid => "history_valid",
        RtDebugView::DirectReservoir => "direct_reservoir",
        RtDebugView::IndirectReservoir => "indirect_reservoir",
        RtDebugView::Temporal => "temporal",
    }
}

fn capture_vpt_debug_view_name(debug_view: VptDebugView) -> &'static str {
    match debug_view {
        VptDebugView::Final => "final",
        VptDebugView::Raw => "raw",
        VptDebugView::Temporal => "temporal",
        VptDebugView::Variance => "variance",
        VptDebugView::HistoryValid => "history_valid",
        VptDebugView::Motion => "motion",
        VptDebugView::Normal => "normal",
        VptDebugView::Depth => "depth",
        VptDebugView::ReservoirWeight => "reservoir_weight",
        VptDebugView::Direct => "direct",
        VptDebugView::Indirect => "indirect",
        VptDebugView::AreaSubpixel => "area_subpixel",
        VptDebugView::AreaLens => "area_lens",
        VptDebugView::AreaWeight => "area_weight",
        VptDebugView::AreaHistoryValid => "area_history_valid",
        VptDebugView::AreaRejection => "area_rejection",
        VptDebugView::AreaJacobian => "area_jacobian",
        VptDebugView::VoxelBrick => "voxel_brick",
        VptDebugView::VoxelLocal => "voxel_local",
        VptDebugView::VoxelHit => "voxel_hit",
        VptDebugView::NrdNormalRoughness => "nrd_normal_roughness",
        VptDebugView::NrdViewZ => "nrd_viewz",
        VptDebugView::NrdMotion => "nrd_motion",
        VptDebugView::NrdMotionZ => "nrd_motion_z",
        VptDebugView::NrdValidation => "nrd_validation",
    }
}

fn add_blit_to_swapchain_pass<'a>(
    graph: &mut RenderGraph<'a>,
    src_resource: crate::render::resource::ResourceHandle,
    src_image: vk::Image,
    src_extent: vk::Extent3D,
    frame: &'a FrameContext,
    capture_dependency: Option<crate::render::resource::ResourceHandle>,
) -> Result<()> {
    let swapchain = graph.import_image_with_access(
        frame.swapchain_image,
        frame.swapchain_extent.width,
        frame.swapchain_extent.height,
        frame.swapchain_format,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        swapchain_access_from_layout(frame.swapchain_image_layout)?,
    );
    graph.add_pass("blit_to_swapchain", QueueType::Graphics, |builder| {
        builder.read_as(src_resource, AccessKind::TransferRead);
        if let Some(capture_dependency) = capture_dependency {
            builder.depend_on(capture_dependency);
        }
        builder.write_as(swapchain, AccessKind::TransferWrite);
        builder.finish_as(swapchain, AccessKind::Present);
        Box::new(move |ctx| {
            blit_to_swapchain::record_blit_core(
                ctx.device,
                ctx.command_buffer,
                src_image,
                src_extent,
                frame.swapchain_image,
                frame.swapchain_extent,
            );
        })
    });
    Ok(())
}

fn add_swapchain_clear_present_pass<'a>(
    graph: &mut RenderGraph<'a>,
    frame: &'a FrameContext,
) -> Result<()> {
    let swapchain = graph.import_image_with_access(
        frame.swapchain_image,
        frame.swapchain_extent.width,
        frame.swapchain_extent.height,
        frame.swapchain_format,
        vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        swapchain_access_from_layout(frame.swapchain_image_layout)?,
    );
    graph.add_pass("rt_clear_fallback", QueueType::Transfer, |builder| {
        builder.write_as(swapchain, AccessKind::TransferWrite);
        builder.finish_as(swapchain, AccessKind::Present);
        Box::new(move |ctx| {
            let clear = vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            };
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            unsafe {
                ctx.device.cmd_clear_color_image(
                    ctx.command_buffer,
                    frame.swapchain_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &clear,
                    &[range],
                );
            }
        })
    });
    Ok(())
}

fn swapchain_access_from_layout(layout: vk::ImageLayout) -> Result<AccessKind> {
    AccessKind::from_swapchain_layout(layout).ok_or_else(|| {
        anyhow!(
            "unsupported swapchain image layout for RT graph import: {:?}",
            layout
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_pipeline_frame_status_defaults_to_not_ready() {
        let pipeline = RtRuntimePipeline::new();

        assert_eq!(pipeline.frame_status(), RtFrameStatus::default());
    }

    #[test]
    fn rt_pipeline_frame_status_snapshots_internal_frame_state() {
        let mut pipeline = RtRuntimePipeline::new();
        pipeline.frame_state.surface_initialized = true;
        pipeline.frame_state.restir_di_history_initialized = true;
        pipeline.frame_state.restir_gi_history_initialized = true;
        pipeline.frame_state.restir_di_rendered = true;
        pipeline.frame_state.restir_gi_rendered = true;
        pipeline.frame_state.direct_lighting_initialized = true;
        pipeline.frame_state.temporal_initialized = true;
        pipeline.frame_state.resolve_initialized = true;

        assert_eq!(
            pipeline.frame_status(),
            RtFrameStatus {
                frame_resources_ready: false,
                surface_ready: true,
                restir_di_history_ready: true,
                restir_gi_history_ready: true,
                restir_di_rendered: true,
                restir_gi_rendered: true,
                direct_lighting_ready: true,
                temporal_ready: true,
                resolve_ready: true,
                skip_reason: None,
            }
        );
    }

    #[test]
    fn rt_pipeline_frame_status_snapshots_skip_reason() {
        let mut pipeline = RtRuntimePipeline::new();
        pipeline.frame_state.skip_reason = Some(RtFrameSkipReason::UcvhUploadPending);

        assert_eq!(
            pipeline.frame_status().skip_reason,
            Some(RtFrameSkipReason::UcvhUploadPending)
        );
    }

    #[test]
    fn rt_pipeline_history_reset_clears_skip_reason() {
        let mut pipeline = RtRuntimePipeline::new();
        pipeline.frame_state.skip_reason = Some(RtFrameSkipReason::RequiredPassesMissing);

        pipeline.reset_history(7);

        assert_eq!(pipeline.frame_status().skip_reason, None);
    }

    #[test]
    fn rt_pipeline_history_reset_clears_active_restir_pass_state() {
        let mut pipeline = RtRuntimePipeline::new();
        pipeline.frame_state.restir_di_rendered = true;
        pipeline.frame_state.restir_gi_rendered = true;

        pipeline.reset_history(7);

        let status = pipeline.frame_status();
        assert!(!status.restir_di_rendered);
        assert!(!status.restir_gi_rendered);
    }

    #[test]
    fn rt_pipeline_frame_status_reports_frame_resource_presence_from_passes() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT pipeline implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        assert!(compact.contains("pubfnframe_status(&self)->RtFrameStatus"));
        assert!(
            compact.contains("frame_resources_ready:self.has_frame_resources()"),
            "RT frame status must report actual RT pass resource presence"
        );
    }

    #[test]
    fn rt_pipeline_records_structured_skip_reasons_for_fallbacks() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        for token in [
            "skip_reason.get_or_insert(RtFrameSkipReason::UcvhUploadPending)",
            "skip_reason.get_or_insert(RtFrameSkipReason::CpuUcvhSceneMissing)",
            "skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureLoaderMissing)",
            "skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureRebuildFailed,);",
            "skip_reason.get_or_insert(RtFrameSkipReason::RequiredPassesMissing)",
            "skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureMissing)",
            "skip_reason.get_or_insert(RtFrameSkipReason::UcvhGpuDescriptorsMissing)",
            "self.frame_state.skip_reason=ifrt_graph_rendered{None}else{skip_reason};",
        ] {
            assert!(
                compact.contains(token),
                "RT pipeline missing structured skip reason token {token}"
            );
        }

        let early_root_cause = compact
            .find("skip_reason.get_or_insert(RtFrameSkipReason::UcvhUploadPending)")
            .expect("RT pipeline should record UCVH upload pending as an early root cause");
        let downstream_required_passes = compact
            .find("skip_reason.get_or_insert(RtFrameSkipReason::RequiredPassesMissing)")
            .expect("RT pipeline should record missing required passes");
        let downstream_as_missing = compact
            .find("skip_reason.get_or_insert(RtFrameSkipReason::AccelerationStructureMissing)")
            .expect("RT pipeline should record missing acceleration structures");
        let downstream_gpu_descriptors = compact
            .find("skip_reason.get_or_insert(RtFrameSkipReason::UcvhGpuDescriptorsMissing)")
            .expect("RT pipeline should record missing UCVH GPU descriptors");

        assert!(early_root_cause < downstream_required_passes);
        assert!(early_root_cause < downstream_as_missing);
        assert!(early_root_cause < downstream_gpu_descriptors);
    }

    #[test]
    fn rt_pipeline_records_active_restir_pass_state() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        for token in [
            "letmutrt_restir_di_rendered=false;",
            "letmutrt_restir_gi_rendered=false;",
            "rt_restir_di_rendered=rt_restir_di_reservoir_resource.is_some();",
            "rt_restir_gi_rendered=rt_restir_gi_reservoir_resource.is_some();",
            "self.frame_state.restir_di_rendered=rt_graph_rendered&&rt_restir_di_rendered;",
            "self.frame_state.restir_gi_rendered=rt_graph_rendered&&rt_restir_gi_rendered;",
        ] {
            assert!(
                compact.contains(token),
                "RT pipeline missing active RT ReSTIR pass state token {token}"
            );
        }
    }

    #[test]
    fn rt_pipeline_registers_surface_direct_lighting_temporal_then_resolve_in_order() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let surface = source
            .find("let rt_surface_outputs")
            .expect("RT surface pass should be registered");
        let direct_lighting = source
            .find("let rt_direct_lighting_outputs")
            .expect("RT direct-lighting pass should be registered");
        let temporal = source
            .find("let rt_temporal_outputs")
            .expect("RT temporal pass should be registered");
        let resolve = source
            .find("let rt_resolve_outputs")
            .expect("RT resolve pass should be registered");
        assert!(surface < temporal);
        assert!(surface < direct_lighting);
        assert!(direct_lighting < temporal);
        assert!(temporal < resolve);
    }

    #[test]
    fn rt_pipeline_queues_capture_from_resolve_output() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let frame_inputs = source
            .split("pub struct RtFrameInputs")
            .nth(1)
            .expect("RtFrameInputs should exist")
            .split("pub struct RtPipelineFrameState")
            .next()
            .expect("RtFrameInputs should end before frame state");
        assert!(
            frame_inputs.contains("capture: Option<&'a mut RenderCapture>"),
            "RT frame inputs must carry RenderCapture for RT resolve readback"
        );

        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        for token in [
            "letmutpending_capture=None;",
            "inputs.capture.as_ref().is_some_and(|capture|capture.should_capture(frame.frame_index))",
            "capture.ensure_readback(",
            "graph.add_pass(\"capture_rt_resolve\",QueueType::Transfer,|builder|{",
            "cmd_copy_image_to_buffer(",
            "source:\"rt_resolve_output\"",
            "render_backend:\"rt\"",
            "pending_capture=Some(CaptureMetadata{",
            "pending_capture,",
        ] {
            assert!(
                compact.contains(token),
                "RT capture path missing compact token {token}"
            );
        }
    }

    #[test]
    fn rt_pipeline_capture_metadata_records_active_rt_passes() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        for token in [
            "rt_frame_rendered:rt_graph_rendered",
            "rt_restir_di_rendered",
            "rt_restir_gi_rendered",
            "rt_resolve_ready:true",
        ] {
            assert!(
                compact.contains(token),
                "RT capture metadata missing active pass token {token}"
            );
        }

        let mark_rendered = compact
            .find("rt_graph_rendered=true")
            .expect("RT graph should be marked rendered once resolve is registered");
        let metadata = compact
            .find("pending_capture=Some(CaptureMetadata{")
            .expect("RT capture metadata should be populated");
        assert!(
            mark_rendered < metadata,
            "RT capture metadata must read rt_graph_rendered after it is marked true"
        );
    }

    #[test]
    fn rt_pipeline_includes_build_output_names_for_raygen_only_rt_shaders() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");

        for token in ["rt_temporal.rgen.spv", "rt_resolve.rgen.spv"] {
            assert!(
                source.contains(token),
                "RT pipeline must include build output {token}"
            );
        }
        for forbidden in [
            "\"/shaders/rt_temporal.spv\"",
            "\"/shaders/rt_resolve.spv\"",
        ] {
            assert!(
                !source.contains(forbidden),
                "RT pipeline must not include stale build output {forbidden}"
            );
        }
    }

    #[test]
    fn rt_pipeline_includes_build_output_name_for_rt_restir_di_raygen() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT pipeline implementation should precede tests");

        assert!(
            implementation.contains("rt_restir_di.rgen.spv"),
            "RT ReSTIR-DI must include the raygen build output name"
        );
        assert!(
            !implementation.contains("\"/shaders/rt_restir_di.spv\""),
            "RT ReSTIR-DI must not include a stale non-stage-suffixed shader output"
        );
    }

    #[test]
    fn rt_pipeline_wires_optional_rt_restir_di_initial_after_surface() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT pipeline implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "RtRestirDiPass",
            "rt_restir_di_pass:Option<RtRestirDiPass>",
            "fnensure_rt_restir_di_pass",
            "build_direct_lights_from_ucvh",
            "RtRestirDiCreateInfo",
            "rt_restir_di.update_uniforms(frame.frame_slot,inputs.rt_settings,frame.frame_index,self.frame_state.restir_di_history_initialized,",
            "rt_restir_di.update_frame_descriptors(renderer.device(),frame.frame_slot,frame.frame_index,rt_surface.surface_buffer(),",
            "rt_restir_di.register_graph(&mutgraph,frame.frame_slot,frame.frame_index,rt_surface_outputs.surface,self.frame_state.restir_di_history_initialized,",
            "RtDirectLightingPass",
            "rt_direct_lighting.update_frame_descriptors(",
            "rt_direct_lighting.register_graph(&mutgraph,frame.frame_slot,rt_surface_outputs.surface,",
        ] {
            assert!(
                compact.contains(token),
                "RT pipeline must wire optional RT ReSTIR-DI initial pass with {token}"
            );
        }

        let surface = compact
            .find("letrt_surface_outputs")
            .expect("RT surface output should exist");
        let restir = compact
            .find("letrt_restir_di_outputs")
            .expect("RT ReSTIR-DI output should exist");
        let direct_lighting = compact
            .find("letrt_direct_lighting_outputs")
            .expect("RT direct-lighting output should exist");
        let temporal = compact
            .find("letrt_temporal_outputs")
            .expect("RT temporal output should exist");
        assert!(surface < restir);
        assert!(restir < direct_lighting);
        assert!(direct_lighting < temporal);
        assert!(
            compact.contains(
                "rt_temporal.register_graph(&mutgraph,frame.frame_slot,frame.frame_index,rt_surface_outputs.surface,rt_direct_lighting_outputs.current_radiance,"
            ),
            "RT temporal must consume current radiance resolved from RT direct lighting"
        );
    }

    #[test]
    fn rt_pipeline_gates_rt_restir_di_temporal_history_after_rendered_frame() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT pipeline implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "pubrestir_di_history_initialized:bool",
            "self.restir_di_history_initialized=false",
            "self.frame_state.restir_di_history_initialized",
            "rt_restir_di.update_history_uniforms(frame.frame_slot,&history_uniforms,)",
            "rt_restir_di.update_uniforms(frame.frame_slot,inputs.rt_settings,frame.frame_index,self.frame_state.restir_di_history_initialized,",
            "rt_restir_di.update_frame_descriptors(renderer.device(),frame.frame_slot,frame.frame_index,rt_surface.surface_buffer(),",
            "rt_restir_di.register_graph(&mutgraph,frame.frame_slot,frame.frame_index,rt_surface_outputs.surface,self.frame_state.restir_di_history_initialized,",
            "self.frame_state.restir_di_history_initialized=rt_graph_rendered&&rt_restir_di_rendered",
        ] {
            assert!(
                compact.contains(token),
                "RT pipeline must gate RT ReSTIR-DI temporal history with {token}"
            );
        }
    }

    #[test]
    fn rt_pipeline_wires_optional_rt_restir_gi_after_surface_before_direct_lighting() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT pipeline implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "RtRestirGiPass",
            "rt_restir_gi_pass:Option<RtRestirGiPass>",
            "pubrestir_gi_history_initialized:bool",
            "self.restir_gi_history_initialized=false",
            "fnensure_rt_restir_gi_pass",
            "rt_restir_gi.rgen.spv",
            "inputs.rt_settings.restir_gi_enabled",
            "rt_restir_gi.update_history_uniforms(",
            "rt_restir_gi.update_uniforms(frame.frame_slot,inputs.rt_settings,frame.frame_index,self.frame_state.restir_gi_history_initialized,",
            "rt_restir_gi.update_frame_descriptors(",
            "rt_restir_gi.update_tlas_descriptor(renderer.device(),frame.frame_slot,tlas,)",
            "rt_restir_gi.update_aabb_descriptor(renderer.device(),frame.frame_slot,aabb_buffer,)",
            "rt_restir_gi.update_ucvh_descriptors(renderer.device(),frame.frame_slot,ucvh_gpu,)",
            "frame.frame_index",
            "rt_surface.surface_buffer()",
            "letrt_restir_gi_outputs=rt_restir_gi.register_graph(&mutgraph,frame.frame_slot,frame.frame_index,rt_surface_outputs.surface,self.frame_state.restir_gi_history_initialized,",
            "self.frame_state.restir_gi_history_initialized=rt_graph_rendered&&rt_restir_gi_rendered",
        ] {
            assert!(
                compact.contains(token),
                "RT pipeline must wire optional RT ReSTIR-GI with {token}"
            );
        }

        let surface = compact
            .find("letrt_surface_outputs")
            .expect("RT surface output should exist");
        let restir_gi = compact
            .find("letrt_restir_gi_outputs")
            .expect("RT ReSTIR-GI output should exist");
        let direct_lighting = compact
            .find("letrt_direct_lighting_outputs")
            .expect("RT direct-lighting output should exist");
        let temporal = compact
            .find("letrt_temporal_outputs")
            .expect("RT temporal output should exist");
        assert!(surface < restir_gi);
        assert!(restir_gi < direct_lighting);
        assert!(direct_lighting < temporal);
    }

    #[test]
    fn rt_pipeline_feeds_rt_restir_gi_reservoirs_into_final_lighting_resolve() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT pipeline implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "letrt_restir_gi_reservoir_buffer=rt_restir_gi.output_reservoir_buffer(frame.frame_slot)",
            "Some(rt_restir_gi_outputs.reservoirs)",
            "Some(rt_restir_gi_reservoir_buffer)",
            "rt_direct_lighting.update_uniforms(frame.frame_slot,inputs.rt_settings,restir_di_active,restir_gi_active,inputs.lighting_settings.shadows_enabled,)",
            "rt_direct_lighting.update_frame_descriptors(renderer.device(),frame.frame_slot,rt_surface.surface_buffer(),rt_restir_di_reservoir_buffer,rt_restir_gi_reservoir_buffer,)",
            "rt_direct_lighting.register_graph(&mutgraph,frame.frame_slot,rt_surface_outputs.surface,rt_restir_di_reservoir_resource,rt_restir_gi_reservoir_resource,self.frame_state.direct_lighting_initialized,)",
        ] {
            assert!(
                compact.contains(token),
                "RT pipeline must feed RT ReSTIR-GI reservoirs into final lighting resolve with {token}"
            );
        }
    }

    #[test]
    fn rt_pipeline_updates_temporal_history_and_resolve_descriptors_before_registering_passes() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        for token in [
            "lethistory_uniforms=GpuRtHistoryUniforms{",
            "rt_temporal.update_history_uniforms(frame.frame_slot,&history_uniforms)",
            "rt_temporal.update_frame_descriptors(renderer.device(),frame.frame_slot,frame.frame_index,",
            "rt_resolve.update_input_descriptor(renderer.device(),frame.frame_slot,",
        ] {
            assert!(
                compact.contains(token),
                "RT pipeline must wire temporal/resolve descriptor state with {token}"
            );
        }

        let update_history = compact
            .find("rt_temporal.update_history_uniforms(")
            .expect("RT temporal history uniforms should be updated");
        let update_temporal = compact
            .find("rt_temporal.update_frame_descriptors(")
            .expect("RT temporal descriptors should be refreshed");
        let register_temporal = compact
            .find("rt_temporal.register_graph(")
            .expect("RT temporal pass should be registered");
        let update_resolve = compact
            .find("rt_resolve.update_input_descriptor(")
            .expect("RT resolve input descriptor should be refreshed");
        let register_resolve = compact
            .find("rt_resolve.register_graph(")
            .expect("RT resolve pass should be registered");

        assert!(update_history < register_temporal);
        assert!(update_temporal < register_temporal);
        assert!(register_temporal < update_resolve);
        assert!(update_resolve < register_resolve);
    }

    #[test]
    fn rt_pipeline_copies_rt_temporal_settings_into_history_uniforms() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("let mut rt_graph_rendered")
            .next()
            .expect("history uniforms should be built before graph registration");
        let compact = crate::render::source_checks::compact(record);

        for token in [
            "temporal_denoise_enabled:inputs.rt_settings.temporal_denoise_enabledasu32",
            "debug_view:inputs.rt_settings.debug_view.as_gpu_value()",
            "history_length:inputs.rt_settings.history_length",
            "normal_threshold:inputs.rt_settings.normal_threshold",
            "depth_threshold:inputs.rt_settings.depth_threshold",
        ] {
            assert!(
                compact.contains(token),
                "RT history uniforms must copy temporal setting token {token}"
            );
        }
    }

    #[test]
    fn rt_scene_key_tracks_rt_temporal_and_debug_settings() {
        let base = RtRuntimePipeline::make_scene_key(
            glam::Vec3::X,
            glam::Vec3::splat(2.0),
            LightingSettings::default(),
            RtSettings::default(),
        );
        let changed = RtRuntimePipeline::make_scene_key(
            glam::Vec3::X,
            glam::Vec3::splat(2.0),
            LightingSettings::default(),
            RtSettings {
                history_length: 32,
                ..RtSettings::default()
            },
        );

        assert_ne!(base, changed);
    }

    #[test]
    fn rt_pipeline_rebuilds_gpu_acceleration_structures_before_surface_trace() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        assert!(compact.contains("renderer.acceleration_structure_loader()"));
        assert!(compact.contains("self.rt_scene.rebuild_gpu("));
        assert!(
            !compact.contains("self.rt_scene.rebuild(ucvh);"),
            "RT runtime should record GPU AS builds instead of CPU-only scene rebuilds"
        );

        let as_rebuild = compact
            .find("self.rt_scene.rebuild_gpu(")
            .expect("RT scene GPU rebuild should be recorded");
        let graph_execute = compact
            .find("graph.execute(")
            .expect("render graph should execute after AS rebuild");
        assert!(
            as_rebuild < graph_execute,
            "GPU AS build commands must be recorded before RT surface trace graph execution"
        );
    }

    #[test]
    fn rt_pipeline_waits_for_other_frames_before_mutating_shared_rt_scene_resources() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        let wait = compact
            .find("renderer.wait_for_other_frame_fences(frame.in_flight_fence)?")
            .expect("RT pipeline must wait for other in-flight frames before AS rebuild/update");
        let rebuild = compact
            .find("self.rt_scene.rebuild_gpu(")
            .expect("RT scene GPU rebuild should be recorded");
        assert!(
            wait < rebuild,
            "shared RT scene AS resources and AABB input buffers must not be mutated while older frames can still read them"
        );
    }

    #[test]
    fn rt_pipeline_passes_acceleration_structure_scratch_alignment_to_scene_rebuild() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        assert!(compact.contains("renderer.acceleration_structure_properties().min_acceleration_structure_scratch_offset_alignmentasvk::DeviceSize"));
        assert!(compact.contains("scratch_alignment,ucvh,"));
    }

    #[test]
    fn rt_pipeline_refreshes_surface_tlas_descriptor_after_scene_rebuild() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        assert!(compact.contains("self.rt_scene.tlas_handle()"));
        assert!(compact.contains("rt_surface.update_tlas_descriptor("));

        let rebuild = compact
            .find("self.rt_scene.rebuild_gpu(")
            .expect("RT scene GPU rebuild should be recorded");
        let update_descriptor = compact
            .find("rt_surface.update_tlas_descriptor(")
            .expect("RT surface pass should update its TLAS descriptor");
        let register_graph = compact
            .find("rt_surface.register_graph(")
            .expect("RT surface graph registration should exist");

        assert!(rebuild < update_descriptor);
        assert!(update_descriptor < register_graph);
    }

    #[test]
    fn rt_pipeline_refreshes_surface_aabb_descriptor_after_scene_rebuild() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        assert!(compact.contains("self.rt_scene.aabb_buffer()"));
        assert!(compact.contains("rt_surface.update_aabb_descriptor("));

        let rebuild = compact
            .find("self.rt_scene.rebuild_gpu(")
            .expect("RT scene GPU rebuild should be recorded");
        let update_descriptor = compact
            .find("rt_surface.update_aabb_descriptor(")
            .expect("RT surface pass should update its AABB descriptor");
        let register_graph = compact
            .find("rt_surface.register_graph(")
            .expect("RT surface graph registration should exist");

        assert!(rebuild < update_descriptor);
        assert!(update_descriptor < register_graph);
    }

    #[test]
    fn rt_pipeline_refreshes_surface_ucvh_descriptor_before_surface_trace() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let frame_inputs = source
            .split("pub struct RtFrameInputs")
            .nth(1)
            .expect("RtFrameInputs should exist")
            .split("pub struct RtPipelineFrameState")
            .next()
            .expect("RtFrameInputs should end before frame state");
        assert!(
            frame_inputs.contains("ucvh_gpu: Option<&'a UcvhGpuResources>"),
            "RT frame inputs must carry GPU UCVH resources for RT surface descriptors"
        );

        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        assert!(compact.contains("inputs.ucvh_gpu"));
        assert!(compact.contains("rt_surface.update_ucvh_descriptors("));
        assert!(
            compact
                .contains("rt_surface.update_history_uniforms(frame.frame_slot,&history_uniforms)")
        );
        let update_tlas = compact
            .find("rt_surface.update_tlas_descriptor(")
            .expect("RT surface pass should update its TLAS descriptor");
        let update_ucvh = compact
            .find("rt_surface.update_ucvh_descriptors(")
            .expect("RT surface pass should update its UCVH descriptors");
        let update_history = compact
            .find("rt_surface.update_history_uniforms(")
            .expect("RT surface pass should update its history descriptors");
        let register_graph = compact
            .find("rt_surface.register_graph(")
            .expect("RT surface graph registration should exist");

        assert!(update_tlas < update_ucvh);
        assert!(update_ucvh < update_history);
        assert!(update_history < register_graph);
    }

    #[test]
    fn rt_pipeline_creates_surface_pass_only_when_ucvh_gpu_descriptors_are_available() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let ensure_passes = source
            .split("pub fn ensure_passes")
            .nth(1)
            .expect("RtRuntimePipeline::ensure_passes should exist")
            .split("pub fn resize")
            .next()
            .expect("ensure_passes should end before resize");
        assert!(ensure_passes.contains("ucvh_gpu: Option<&UcvhGpuResources>"));
        assert!(
            ensure_passes.contains("self.ensure_rt_surface_pass(renderer, scene_ubo, ucvh_gpu")
        );

        let ensure_surface = source
            .split("fn ensure_rt_surface_pass")
            .nth(1)
            .expect("ensure_rt_surface_pass should exist")
            .split("fn ensure_rt_temporal_pass")
            .next()
            .expect("ensure_rt_surface_pass should end before temporal helper");
        assert!(ensure_surface.contains("ucvh_gpu: Option<&UcvhGpuResources>"));
        assert!(ensure_surface.contains("let Some(ucvh_gpu) = ucvh_gpu else"));
        assert!(ensure_surface.contains("ucvh_gpu,"));
    }

    #[test]
    fn rt_pipeline_skips_surface_trace_without_built_tlas_and_aabbs() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let record = source
            .split("pub fn record_and_execute_frame")
            .nth(1)
            .expect("RtRuntimePipeline::record_and_execute_frame should exist")
            .split("pub fn destroy")
            .next()
            .expect("record_and_execute_frame should end before destroy");
        let compact = crate::render::source_checks::compact(record);

        assert!(compact.contains("match(self.rt_scene.tlas_handle(),self.rt_scene.aabb_buffer())"));
        assert!(compact.contains("add_swapchain_clear_present_pass(&mutgraph,frame)?"));
        assert!(compact.contains("self.frame_state.surface_initialized=rt_graph_rendered"));
    }

    #[test]
    fn rt_pipeline_destroy_releases_rt_scene_acceleration_structures() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let destroy = source
            .split("pub fn destroy")
            .nth(1)
            .expect("RtRuntimePipeline::destroy should exist")
            .split("fn ensure_rt_surface_pass")
            .next()
            .expect("destroy should end before pass creation helpers");

        assert!(destroy.contains("acceleration_structure_loader"));
        let compact = crate::render::source_checks::compact(destroy);
        assert!(compact.contains("self.rt_scene.destroy("));
    }

    #[test]
    fn rt_pipeline_exposes_frame_resource_presence_for_runtime_resize_routing() {
        let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT pipeline implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "pubfnhas_frame_resources(&self)->bool",
            "self.rt_surface_pass.is_some()",
            "self.rt_restir_di_pass.is_some()",
            "self.rt_restir_gi_pass.is_some()",
            "self.rt_direct_lighting_pass.is_some()",
            "self.rt_temporal_pass.is_some()",
            "self.rt_resolve_pass.is_some()",
        ] {
            assert!(
                compact.contains(token),
                "RT frame-resource helper missing {token}"
            );
        }
    }
}
