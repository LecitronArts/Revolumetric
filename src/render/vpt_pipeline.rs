use anyhow::{Context, Result};
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::area_restir::AreaRestirSettings;
use crate::render::camera::{compute_pixel_to_ray, compute_view_proj};
use crate::render::capture::{CaptureMetadata, RenderCapture, cmd_copy_image_to_buffer};
use crate::render::device::RenderDevice;
#[cfg(not(target_os = "android"))]
use crate::render::egui_renderer::{EguiFrame, EguiRenderer};
use crate::render::frame::FrameContext;
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::{RenderGraph, RenderGraphTransientResources};
use crate::render::passes::area_restir::{AreaRestirPass, AreaRestirPassCreateInfo};
use crate::render::passes::blit_to_swapchain;
use crate::render::passes::postprocess::{PostprocessGraphInputs, PostprocessPass};
use crate::render::passes::restir_di::{RestirDiPass, RestirDiPassCreateInfo};
use crate::render::passes::vpt::{VptGraphInputs, VptPass, VptPassCreateInfo, VptPassResizeInfo};
use crate::render::passes::vpt_atrous::{
    VptAtrousGraphInputs, VptAtrousPass, VptAtrousPassCreateInfo, VptAtrousPassResizeInfo,
};
use crate::render::passes::vpt_nrd_adapter::{
    VptNrdAdapterGraphInputs, VptNrdAdapterPass, VptNrdAdapterPassCreateInfo,
    VptNrdAdapterPassImageRefs, VptNrdAdapterPassResizeInfo, VptNrdFrameSettings,
    VptNrdFrameSettingsInputs,
};
use crate::render::passes::vpt_nrd_confidence::{
    VptNrdConfidenceGraphInputs, VptNrdConfidencePass, VptNrdConfidencePassCreateInfo,
    VptNrdConfidencePassResizeInfo,
};
use crate::render::passes::vpt_nrd_frontend::{
    VptNrdFrontendGraphInputs, VptNrdFrontendPass, VptNrdFrontendPassCreateInfo,
    VptNrdFrontendPassResizeInfo,
};
use crate::render::passes::vpt_nrd_resolve::{
    VptNrdResolveGraphInputs, VptNrdResolvePass, VptNrdResolvePassCreateInfo,
    VptNrdResolvePassResizeInfo,
};
use crate::render::passes::vpt_surface::{
    VptSurfacePass, VptSurfacePassCreateInfo, VptSurfacePassResizeInfo,
};
use crate::render::passes::vpt_temporal::{
    VptTemporalGraphInputs, VptTemporalPass, VptTemporalPassCreateInfo, VptTemporalPassResizeInfo,
};
use crate::render::resource::{AccessKind, QueueType};
use crate::render::restir_di::RestirDiSettings;
use crate::render::restir_di::build_direct_lights_from_ucvh;
use crate::render::scene_ubo::{
    LightingSettings, SceneUniformBuffer, SceneUniformInputs, VptDebugView, VptDenoiserMode,
    build_scene_uniforms,
};
use crate::render::traversal_stats::{VptTraversalStatsBuffer, VptTraversalStatsSnapshot};
use crate::render::vpt_history::{
    GpuVptHistoryUniforms, VPT_HISTORY_FLAG_CAMERA_CUT, VPT_HISTORY_FLAG_LIGHTS_INVALIDATED,
    VPT_HISTORY_FLAG_RESIZE, VPT_HISTORY_FLAG_SCENE_INVALIDATED,
};
use crate::voxel::gpu_upload::UcvhGpuResources;
use crate::voxel::ucvh::{Ucvh, UcvhInvalidationRegion, UcvhMotionEvent};

const VPT_SCENE_KEY_WORDS: usize = 35;

#[derive(Debug, Clone, Copy)]
pub struct VptCameraFrame {
    pub position: glam::Vec3,
    pub forward: glam::Vec3,
    pub up: glam::Vec3,
    pub fov_y_radians: f32,
    pub aperture_radius: f32,
    pub focal_distance: f32,
}

pub struct VptFrameInputs<'a> {
    pub scene_ubo: &'a SceneUniformBuffer,
    pub camera: VptCameraFrame,
    pub sun_direction: glam::Vec3,
    pub sun_intensity: glam::Vec3,
    pub elapsed_seconds: f32,
    pub lighting_settings: LightingSettings,
    pub restir_di_settings: RestirDiSettings,
    pub area_restir_settings: AreaRestirSettings,
    pub restir_di_enabled: bool,
    pub area_restir_enabled: bool,
    pub ucvh_ready: bool,
    pub ucvh_frame_changes: UcvhFrameChanges,
    pub ucvh_motion_event_count: u32,
    pub capture: Option<&'a mut RenderCapture>,
    pub profiler: Option<&'a GpuProfiler>,
}

#[derive(Debug, Default)]
pub struct UcvhFrameChanges {
    pub invalidation_regions: Vec<UcvhInvalidationRegion>,
    pub motion_events: Vec<UcvhMotionEvent>,
}

impl UcvhFrameChanges {
    pub fn new(
        invalidation_regions: Vec<UcvhInvalidationRegion>,
        motion_events: Vec<UcvhMotionEvent>,
    ) -> Self {
        Self {
            invalidation_regions,
            motion_events,
        }
    }
}

pub struct VptFrameRecordResult {
    pub pending_capture: Option<CaptureMetadata>,
    pub submitted_fence: vk::Fence,
    pub rendered_vpt: bool,
    pub traversal_stats_requested: bool,
    pub traversal_stats: Option<VptTraversalStatsSnapshot>,
}

pub struct VptPipelineFrameState {
    pub vpt_sample_index: u32,
    pub last_vpt_camera_key: Option<[u32; 15]>,
    pub last_vpt_scene_key: Option<[u32; VPT_SCENE_KEY_WORDS]>,
    pub history_reset_generation: u32,
    pub vpt_accumulation_needs_init: bool,
    pub vpt_temporal_history_initialized: bool,
    pub postprocess_output_initialized: bool,
    pub vpt_nrd_texture_pools_initialized: bool,
    pub area_restir_history_initialized: bool,
    pub restir_di_history_initialized: bool,
    pub previous_vpt_view_proj: Option<glam::Mat4>,
    pub previous_vpt_resolution: Option<[u32; 2]>,
    pub previous_nrd_world_to_view: Option<glam::Mat4>,
    pub previous_nrd_view_to_clip: Option<glam::Mat4>,
    pub previous_nrd_elapsed_seconds: Option<f32>,
}

impl Default for VptPipelineFrameState {
    fn default() -> Self {
        Self {
            vpt_sample_index: 0,
            last_vpt_camera_key: None,
            last_vpt_scene_key: None,
            history_reset_generation: 0,
            vpt_accumulation_needs_init: true,
            vpt_temporal_history_initialized: false,
            postprocess_output_initialized: false,
            vpt_nrd_texture_pools_initialized: false,
            area_restir_history_initialized: false,
            restir_di_history_initialized: false,
            previous_vpt_view_proj: None,
            previous_vpt_resolution: None,
            previous_nrd_world_to_view: None,
            previous_nrd_view_to_clip: None,
            previous_nrd_elapsed_seconds: None,
        }
    }
}

impl VptPipelineFrameState {
    pub fn reset_for_resize_or_camera_cut(&mut self) {
        self.vpt_sample_index = 0;
        self.last_vpt_camera_key = None;
        self.history_reset_generation = self.history_reset_generation.wrapping_add(1);
        self.vpt_accumulation_needs_init = true;
        self.vpt_temporal_history_initialized = false;
        self.postprocess_output_initialized = false;
        self.vpt_nrd_texture_pools_initialized = false;
        self.area_restir_history_initialized = false;
        self.restir_di_history_initialized = false;
        self.previous_vpt_view_proj = None;
        self.previous_vpt_resolution = None;
        self.previous_nrd_world_to_view = None;
        self.previous_nrd_view_to_clip = None;
        self.previous_nrd_elapsed_seconds = None;
    }

    pub fn reset_for_scene_change(&mut self) {
        self.vpt_sample_index = 0;
        self.last_vpt_camera_key = None;
        self.last_vpt_scene_key = None;
        self.history_reset_generation = self.history_reset_generation.wrapping_add(1);
        self.vpt_accumulation_needs_init = true;
        self.vpt_temporal_history_initialized = false;
        self.postprocess_output_initialized = false;
        self.vpt_nrd_texture_pools_initialized = false;
        self.area_restir_history_initialized = false;
        self.restir_di_history_initialized = false;
    }
}

fn compute_nrd_world_to_view(
    camera_pos: glam::Vec3,
    camera_forward: glam::Vec3,
    camera_up: glam::Vec3,
) -> glam::Mat4 {
    let forward = camera_forward.normalize();
    let right = camera_up.cross(forward).normalize();
    let up = forward.cross(right);
    glam::Mat4::from_cols(
        glam::Vec4::new(right.x, up.x, -forward.x, 0.0),
        glam::Vec4::new(right.y, up.y, -forward.y, 0.0),
        glam::Vec4::new(right.z, up.z, -forward.z, 0.0),
        glam::Vec4::new(
            -right.dot(camera_pos),
            -up.dot(camera_pos),
            forward.dot(camera_pos),
            1.0,
        ),
    )
}

fn compute_nrd_view_to_clip(fov_y: f32, width: u32, height: u32) -> glam::Mat4 {
    glam::Mat4::perspective_rh(fov_y, width as f32 / height.max(1) as f32, 0.01, 10_000.0)
}

pub struct VptRuntimePipeline {
    pub postprocess_pass: Option<PostprocessPass>,
    pub vpt_surface_pass: Option<VptSurfacePass>,
    pub vpt_nrd_confidence_pass: Option<VptNrdConfidencePass>,
    pub vpt_pass: Option<VptPass>,
    pub vpt_nrd_frontend_pass: Option<VptNrdFrontendPass>,
    pub vpt_nrd_adapter_pass: Option<VptNrdAdapterPass>,
    pub vpt_nrd_resolve_pass: Option<VptNrdResolvePass>,
    pub vpt_temporal_pass: Option<VptTemporalPass>,
    pub vpt_atrous_pass: Option<VptAtrousPass>,
    pub area_restir_pass: Option<AreaRestirPass>,
    pub restir_di_pass: Option<RestirDiPass>,
    pub traversal_stats_buffers: Vec<VptTraversalStatsBuffer>,
    pub render_graph_transients: Vec<RenderGraphTransientResources>,
    pub frame_state: VptPipelineFrameState,
}

impl Default for VptRuntimePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl VptRuntimePipeline {
    fn make_scene_key(
        sun_direction: glam::Vec3,
        sun_intensity: glam::Vec3,
        lighting_settings: LightingSettings,
        restir_di_settings: RestirDiSettings,
        area_restir_settings: AreaRestirSettings,
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) -> [u32; VPT_SCENE_KEY_WORDS] {
        [
            sun_direction.x.to_bits(),
            sun_direction.y.to_bits(),
            sun_direction.z.to_bits(),
            sun_intensity.x.to_bits(),
            sun_intensity.y.to_bits(),
            sun_intensity.z.to_bits(),
            lighting_settings.shadows_enabled as u32,
            lighting_settings.skip_backface_shadows as u32,
            lighting_settings.vpt_max_bounces,
            lighting_settings.sun_angular_radius.to_bits(),
            lighting_settings.denoiser_mode.as_scene_key_value(),
            lighting_settings.denoiser_atrous_iterations,
            lighting_settings.vpt_debug_view.as_gpu_value(),
            lighting_settings.debug_view.as_gpu_value(),
            restir_di_enabled as u32,
            restir_di_settings.enabled as u32,
            restir_di_settings.temporal_enabled as u32,
            restir_di_settings.spatial_enabled as u32,
            restir_di_settings.initial_candidate_count,
            restir_di_settings.spatial_sample_count,
            restir_di_settings.history_length,
            restir_di_settings.debug_view.as_gpu_value(),
            area_restir_enabled as u32,
            area_restir_settings.enabled as u32,
            area_restir_settings.temporal_enabled as u32,
            area_restir_settings.spatial_enabled as u32,
            area_restir_settings.subpixel_enabled as u32,
            area_restir_settings.lens_enabled as u32,
            area_restir_settings.initial_candidate_count,
            area_restir_settings.spatial_sample_count,
            area_restir_settings.history_length,
            area_restir_settings.normal_threshold.to_bits(),
            area_restir_settings.depth_threshold.to_bits(),
            area_restir_settings.spatial_radius.to_bits(),
            area_restir_settings.debug_view.as_gpu_value(),
        ]
    }

    pub fn new() -> Self {
        Self {
            postprocess_pass: None,
            vpt_surface_pass: None,
            vpt_nrd_confidence_pass: None,
            vpt_pass: None,
            vpt_nrd_frontend_pass: None,
            vpt_nrd_adapter_pass: None,
            vpt_nrd_resolve_pass: None,
            vpt_temporal_pass: None,
            vpt_atrous_pass: None,
            area_restir_pass: None,
            restir_di_pass: None,
            traversal_stats_buffers: Vec::new(),
            render_graph_transients: Vec::new(),
            frame_state: VptPipelineFrameState::default(),
        }
    }

    pub fn has_frame_resources(&self) -> bool {
        self.postprocess_pass.is_some()
            || self.vpt_surface_pass.is_some()
            || self.vpt_nrd_confidence_pass.is_some()
            || self.vpt_pass.is_some()
            || self.vpt_nrd_frontend_pass.is_some()
            || self.vpt_nrd_adapter_pass.is_some()
            || self.vpt_nrd_resolve_pass.is_some()
            || self.vpt_temporal_pass.is_some()
            || self.vpt_atrous_pass.is_some()
            || self.area_restir_pass.is_some()
            || self.restir_di_pass.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ensure_passes(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh: Option<&Ucvh>,
        ucvh_gpu: Option<&UcvhGpuResources>,
        lighting_settings: LightingSettings,
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) {
        if !self.ensure_traversal_stats_buffers(renderer, scene_ubo.frame_count()) {
            return;
        }
        self.ensure_render_graph_transients(renderer, scene_ubo.frame_count());
        self.ensure_vpt_surface_pass(renderer, scene_ubo, ucvh_gpu);
        self.ensure_vpt_nrd_confidence_pass(renderer, scene_ubo);
        self.ensure_vpt_pass(renderer, scene_ubo, ucvh_gpu);
        self.ensure_vpt_nrd_frontend_pass(renderer, scene_ubo);
        self.ensure_vpt_nrd_adapter_pass(renderer, scene_ubo, lighting_settings);
        self.ensure_vpt_nrd_resolve_pass(renderer, scene_ubo);
        self.ensure_restir_di_pass(renderer, scene_ubo, ucvh, restir_di_enabled);
        self.ensure_area_restir_pass(renderer, scene_ubo, ucvh_gpu, area_restir_enabled);
        self.ensure_vpt_temporal_pass(renderer, scene_ubo);
        self.ensure_vpt_atrous_pass(renderer, scene_ubo);
        self.ensure_postprocess_pass(renderer, scene_ubo);
    }

    fn ensure_traversal_stats_buffers(
        &mut self,
        renderer: &RenderDevice,
        frame_count: usize,
    ) -> bool {
        if self.traversal_stats_buffers.len() == frame_count {
            return true;
        }

        for buffer in std::mem::take(&mut self.traversal_stats_buffers) {
            buffer.destroy(renderer.device(), renderer.allocator());
        }

        let mut buffers = Vec::with_capacity(frame_count);
        for _slot in 0..frame_count {
            match VptTraversalStatsBuffer::new(renderer.device(), renderer.allocator()) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    tracing::error!(%error, "failed to create VPT traversal stats buffer");
                    for buffer in buffers {
                        buffer.destroy(renderer.device(), renderer.allocator());
                    }
                    return false;
                }
            }
        }
        self.traversal_stats_buffers = buffers;
        self.sync_traversal_stats_descriptors(renderer.device(), frame_count);
        true
    }

    fn ensure_render_graph_transients(&mut self, renderer: &RenderDevice, frame_count: usize) {
        if self.render_graph_transients.len() == frame_count {
            return;
        }

        for transients in std::mem::take(&mut self.render_graph_transients) {
            transients.destroy(renderer.device(), renderer.allocator());
        }

        self.render_graph_transients = (0..frame_count)
            .map(|_| RenderGraphTransientResources::new())
            .collect();
    }

    pub fn traversal_stats_buffer(&self, frame_slot: usize) -> Option<&VptTraversalStatsBuffer> {
        self.traversal_stats_buffers.get(frame_slot)
    }

    pub fn traversal_stats_snapshot(
        &self,
        frame_slot: usize,
    ) -> Result<Option<VptTraversalStatsSnapshot>> {
        self.traversal_stats_buffer(frame_slot)
            .map(VptTraversalStatsBuffer::snapshot)
            .transpose()
    }

    fn sync_traversal_stats_descriptors(&self, device: &ash::Device, frame_count: usize) {
        for slot in 0..frame_count {
            let Some(stats_buffer) = self.traversal_stats_buffer(slot) else {
                continue;
            };
            if let Some(vpt_surface) = &self.vpt_surface_pass {
                vpt_surface.update_traversal_stats_descriptor(device, slot, stats_buffer);
            }
            if let Some(area_restir) = &self.area_restir_pass {
                area_restir.update_traversal_stats_descriptors(device, slot, stats_buffer);
            }
            if let Some(vpt) = &self.vpt_pass {
                vpt.update_traversal_stats_descriptor(device, slot, stats_buffer);
            }
        }
    }

    fn ensure_vpt_surface_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh_gpu: Option<&UcvhGpuResources>,
    ) {
        if self.vpt_surface_pass.is_some() {
            return;
        }
        let Some(ucvh_gpu) = ucvh_gpu else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt_surface.spv"));
        if spirv.is_empty() {
            tracing::warn!("vpt_surface.spv is empty; slangc may not be installed");
            return;
        }

        match VptSurfacePass::new(
            renderer.device(),
            renderer.allocator(),
            VptSurfacePassCreateInfo {
                width: extent.width,
                height: extent.height,
                spirv_bytes: spirv,
                ucvh_gpu,
                scene_ubo,
                traversal_stats_buffers: &self.traversal_stats_buffers,
            },
        ) {
            Ok(pass) => {
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized VPT surface pass"
                );
                self.vpt_surface_pass = Some(pass);
            }
            Err(error) => {
                tracing::error!(%error, "failed to create VPT surface pass");
            }
        }
    }

    fn ensure_vpt_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh_gpu: Option<&UcvhGpuResources>,
    ) {
        if self.vpt_pass.is_some() {
            return;
        }
        let Some(ucvh_gpu) = ucvh_gpu else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt.spv"));
        if spirv.is_empty() {
            tracing::warn!("vpt.spv is empty; slangc may not be installed");
            return;
        }

        match VptPass::new(
            renderer.device(),
            renderer.allocator(),
            VptPassCreateInfo {
                width: extent.width,
                height: extent.height,
                spirv_bytes: spirv,
                ucvh_gpu,
                scene_ubo,
                traversal_stats_buffers: &self.traversal_stats_buffers,
            },
        ) {
            Ok(pass) => {
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized VPT pass"
                );
                self.vpt_pass = Some(pass);
                self.frame_state.vpt_accumulation_needs_init = true;
            }
            Err(error) => {
                tracing::error!(%error, "failed to create VPT pass");
            }
        }
    }

    fn ensure_vpt_nrd_confidence_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
    ) {
        if self.vpt_nrd_confidence_pass.is_some() {
            return;
        }
        let Some(vpt_surface) = &self.vpt_surface_pass else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt_nrd_confidence.spv"));
        if spirv.is_empty() {
            tracing::warn!("vpt_nrd_confidence.spv is empty; slangc may not be installed");
            return;
        }

        match VptNrdConfidencePass::new(
            renderer.device(),
            renderer.allocator(),
            VptNrdConfidencePassCreateInfo {
                width: extent.width,
                height: extent.height,
                spirv_bytes: spirv,
                scene_ubo,
                vpt_surface,
            },
        ) {
            Ok(pass) => {
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized VPT NRD confidence pass"
                );
                self.vpt_nrd_confidence_pass = Some(pass);
            }
            Err(error) => {
                tracing::error!(%error, "failed to create VPT NRD confidence pass");
            }
        }
    }

    fn ensure_vpt_nrd_frontend_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
    ) {
        if self.vpt_nrd_frontend_pass.is_some() {
            return;
        }
        let Some(vpt) = &self.vpt_pass else {
            return;
        };
        let Some(vpt_surface) = &self.vpt_surface_pass else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt_nrd_frontend.spv"));
        if spirv.is_empty() {
            tracing::warn!("vpt_nrd_frontend.spv is empty; slangc may not be installed");
            return;
        }

        match VptNrdFrontendPass::new(
            renderer.device(),
            renderer.allocator(),
            VptNrdFrontendPassCreateInfo {
                width: extent.width,
                height: extent.height,
                spirv_bytes: spirv,
                scene_ubo,
                vpt,
                vpt_surface,
            },
        ) {
            Ok(pass) => {
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized VPT NRD frontend pass"
                );
                self.vpt_nrd_frontend_pass = Some(pass);
            }
            Err(error) => {
                tracing::error!(%error, "failed to create VPT NRD frontend pass");
            }
        }
    }

    fn ensure_vpt_nrd_adapter_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        lighting_settings: LightingSettings,
    ) {
        if !matches!(
            lighting_settings.denoiser_mode,
            VptDenoiserMode::Relax | VptDenoiserMode::Reblur
        ) {
            let _ = self.destroy_vpt_nrd_adapter_chain(renderer);
            return;
        }
        if let Some(pass) = self.vpt_nrd_adapter_pass.as_ref() {
            if pass.denoiser_mode() == lighting_settings.denoiser_mode {
                return;
            }
        }
        if !self.destroy_vpt_nrd_adapter_chain(renderer) {
            return;
        }
        if self.vpt_surface_pass.is_none()
            || self.vpt_nrd_confidence_pass.is_none()
            || self.vpt_nrd_frontend_pass.is_none()
        {
            return;
        }
        let (Some(vpt_surface), Some(vpt_nrd_confidence), Some(vpt_nrd_frontend)) = (
            self.vpt_surface_pass.as_ref(),
            self.vpt_nrd_confidence_pass.as_ref(),
            self.vpt_nrd_frontend_pass.as_ref(),
        ) else {
            return;
        };

        let extent = renderer.swapchain_extent();
        match VptNrdAdapterPass::new(
            renderer.device(),
            renderer.allocator(),
            VptNrdAdapterPassCreateInfo {
                width: extent.width,
                height: extent.height,
                denoiser_mode: lighting_settings.denoiser_mode,
                relax_atrous_iteration_num: lighting_settings.denoiser_atrous_iterations,
                constant_buffer_alignment: renderer
                    .physical_device_properties()
                    .limits
                    .min_uniform_buffer_offset_alignment,
                scene_ubo,
                image_refs: VptNrdAdapterPassImageRefs {
                    frontend: vpt_nrd_frontend,
                    confidence: vpt_nrd_confidence,
                    surface: vpt_surface,
                },
            },
        ) {
            Ok(pass) => {
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized VPT NRD adapter pass"
                );
                self.vpt_nrd_adapter_pass = Some(pass);
            }
            Err(error) => {
                tracing::error!(%error, "failed to create VPT NRD adapter pass");
            }
        }
    }

    fn destroy_vpt_nrd_adapter_chain(&mut self, renderer: &RenderDevice) -> bool {
        if self.vpt_nrd_adapter_pass.is_none() && self.vpt_nrd_resolve_pass.is_none() {
            return true;
        }
        if let Err(error) = renderer.wait_idle() {
            tracing::error!(%error, "failed to idle Vulkan device before destroying VPT NRD adapter resources");
            return false;
        }
        if let Some(pass) = self.vpt_nrd_resolve_pass.take() {
            pass.destroy(renderer.device(), renderer.allocator());
        }
        if let Some(pass) = self.vpt_nrd_adapter_pass.take() {
            pass.destroy(renderer.device(), renderer.allocator());
        }
        self.frame_state.vpt_nrd_texture_pools_initialized = false;
        self.frame_state.postprocess_output_initialized = false;
        true
    }

    fn ensure_vpt_nrd_resolve_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
    ) {
        if self.vpt_nrd_resolve_pass.is_some() {
            return;
        }
        let (Some(vpt_nrd_adapter), Some(vpt_nrd_frontend)) = (
            self.vpt_nrd_adapter_pass.as_ref(),
            self.vpt_nrd_frontend_pass.as_ref(),
        ) else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt_nrd_resolve.spv"));
        if spirv.is_empty() {
            tracing::warn!("vpt_nrd_resolve.spv is empty; slangc may not be installed");
            return;
        }

        match VptNrdResolvePass::new(
            renderer.device(),
            renderer.allocator(),
            VptNrdResolvePassCreateInfo {
                width: extent.width,
                height: extent.height,
                spirv_bytes: spirv,
                scene_ubo,
                denoised_diff_radiance_hitdist: &vpt_nrd_adapter.nrd_diff_radiance_hitdist,
                frontend: vpt_nrd_frontend,
            },
        ) {
            Ok(pass) => {
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized VPT NRD resolve pass"
                );
                self.vpt_nrd_resolve_pass = Some(pass);
            }
            Err(error) => {
                tracing::error!(%error, "failed to create VPT NRD resolve pass");
            }
        }
    }

    fn ensure_restir_di_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh: Option<&Ucvh>,
        restir_di_enabled: bool,
    ) {
        if self.restir_di_pass.is_some() || !restir_di_enabled {
            return;
        }
        let Some(ucvh) = ucvh else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let initial_spirv =
            include_bytes!(concat!(env!("OUT_DIR"), "/shaders/restir_di_initial.spv"));
        let temporal_spirv =
            include_bytes!(concat!(env!("OUT_DIR"), "/shaders/restir_di_temporal.spv"));
        let spatial_spirv =
            include_bytes!(concat!(env!("OUT_DIR"), "/shaders/restir_di_spatial.spv"));
        if initial_spirv.is_empty() || temporal_spirv.is_empty() || spatial_spirv.is_empty() {
            tracing::warn!("ReSTIR-DI shaders are empty; slangc may not be installed");
            return;
        }

        let direct_lights = build_direct_lights_from_ucvh(ucvh, 4096);
        match RestirDiPass::new(
            renderer.device(),
            renderer.allocator(),
            RestirDiPassCreateInfo {
                width: extent.width,
                height: extent.height,
                frame_count: scene_ubo.frame_count(),
                initial_spirv,
                temporal_spirv,
                spatial_spirv,
                direct_lights: &direct_lights,
            },
        ) {
            Ok(pass) => {
                if let Some(vpt_surface) = &self.vpt_surface_pass {
                    pass.update_surface_descriptors(renderer.device(), vpt_surface);
                }
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    direct_lights = direct_lights.len(),
                    "initialized ReSTIR-DI VPT pass skeleton"
                );
                self.restir_di_pass = Some(pass);
            }
            Err(error) => {
                tracing::error!(%error, "failed to create ReSTIR-DI pass");
            }
        }
    }

    fn ensure_area_restir_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh_gpu: Option<&UcvhGpuResources>,
        area_restir_enabled: bool,
    ) {
        if self.area_restir_pass.is_some() || !area_restir_enabled {
            return;
        }
        let Some(ucvh_gpu) = ucvh_gpu else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let initial_spirv =
            include_bytes!(concat!(env!("OUT_DIR"), "/shaders/area_restir_initial.spv"));
        let temporal_spirv = include_bytes!(concat!(
            env!("OUT_DIR"),
            "/shaders/area_restir_temporal.spv"
        ));
        let spatial_spirv =
            include_bytes!(concat!(env!("OUT_DIR"), "/shaders/area_restir_spatial.spv"));
        if initial_spirv.is_empty() || temporal_spirv.is_empty() || spatial_spirv.is_empty() {
            tracing::warn!("Area ReSTIR shaders are empty; slangc may not be installed");
            return;
        }

        match AreaRestirPass::new(
            renderer.device(),
            renderer.allocator(),
            AreaRestirPassCreateInfo {
                width: extent.width,
                height: extent.height,
                frame_count: scene_ubo.frame_count(),
                initial_spirv,
                temporal_spirv,
                spatial_spirv,
                scene_ubo,
                ucvh_gpu,
                traversal_stats_buffers: &self.traversal_stats_buffers,
            },
        ) {
            Ok(pass) => {
                if let Some(vpt_surface) = &self.vpt_surface_pass {
                    pass.update_surface_descriptors(renderer.device(), vpt_surface);
                }
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized Area ReSTIR VPT sample-area pass"
                );
                self.area_restir_pass = Some(pass);
                self.sync_area_restir_descriptors(renderer.device(), scene_ubo.frame_count());
                self.frame_state.area_restir_history_initialized = false;
            }
            Err(error) => {
                tracing::error!(%error, "failed to create Area ReSTIR pass");
            }
        }
    }

    fn sync_area_restir_descriptors(&self, device: &ash::Device, frame_count: usize) {
        let (Some(area_restir), Some(vpt), Some(vpt_surface)) = (
            &self.area_restir_pass,
            &self.vpt_pass,
            &self.vpt_surface_pass,
        ) else {
            return;
        };

        for slot in 0..frame_count {
            let (area_uniform_buffer, _, _) = area_restir.uniform_buffer(slot);
            let (area_selected_current_buffer, _, _) = area_restir.selected_current_buffer(slot);
            vpt.update_area_restir_descriptors(
                device,
                slot,
                area_uniform_buffer,
                area_selected_current_buffer,
            );
            vpt_surface.update_area_restir_descriptors(
                device,
                slot,
                area_uniform_buffer,
                area_selected_current_buffer,
            );
        }
    }

    fn ensure_vpt_temporal_pass(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
    ) {
        if self.vpt_temporal_pass.is_some() {
            return;
        }
        let (Some(vpt), Some(vpt_surface)) = (&self.vpt_pass, &self.vpt_surface_pass) else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt_temporal.spv"));
        if spirv.is_empty() {
            tracing::warn!("vpt_temporal.spv is empty; slangc may not be installed");
            return;
        }

        match VptTemporalPass::new(
            renderer.device(),
            renderer.allocator(),
            VptTemporalPassCreateInfo {
                width: extent.width,
                height: extent.height,
                spirv_bytes: spirv,
                scene_ubo,
                vpt,
                vpt_surface,
            },
        ) {
            Ok(pass) => {
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized VPT temporal pass"
                );
                self.vpt_temporal_pass = Some(pass);
                self.frame_state.vpt_temporal_history_initialized = false;
            }
            Err(error) => {
                tracing::error!(%error, "failed to create VPT temporal pass");
            }
        }
    }

    fn ensure_vpt_atrous_pass(&mut self, renderer: &RenderDevice, scene_ubo: &SceneUniformBuffer) {
        if self.vpt_atrous_pass.is_some() {
            return;
        }
        let (Some(vpt_temporal), Some(vpt_surface)) =
            (&self.vpt_temporal_pass, &self.vpt_surface_pass)
        else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt_atrous.spv"));
        if spirv.is_empty() {
            tracing::warn!("vpt_atrous.spv is empty; slangc may not be installed");
            return;
        }

        match VptAtrousPass::new(
            renderer.device(),
            renderer.allocator(),
            VptAtrousPassCreateInfo {
                width: extent.width,
                height: extent.height,
                spirv_bytes: spirv,
                scene_ubo,
                temporal: vpt_temporal,
                vpt_surface,
            },
        ) {
            Ok(pass) => {
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized VPT atrous pass"
                );
                self.vpt_atrous_pass = Some(pass);
            }
            Err(error) => {
                tracing::error!(%error, "failed to create VPT atrous pass");
            }
        }
    }

    fn ensure_postprocess_pass(&mut self, renderer: &RenderDevice, scene_ubo: &SceneUniformBuffer) {
        if self.postprocess_pass.is_some() {
            return;
        }
        let Some(vpt_atrous) = &self.vpt_atrous_pass else {
            return;
        };

        let extent = renderer.swapchain_extent();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/postprocess.spv"));
        if spirv.is_empty() {
            tracing::warn!("postprocess.spv is empty; slangc may not be installed");
            return;
        }

        match PostprocessPass::new(
            renderer.device(),
            renderer.allocator(),
            extent.width,
            extent.height,
            spirv,
            vpt_atrous.output_image(),
            scene_ubo,
        ) {
            Ok(pass) => {
                tracing::info!(
                    width = extent.width,
                    height = extent.height,
                    "initialized postprocess pass from VPT output"
                );
                self.postprocess_pass = Some(pass);
            }
            Err(error) => {
                tracing::error!(%error, "failed to create VPT postprocess pass");
            }
        }
    }

    pub fn record_and_execute_frame(
        &mut self,
        renderer: &RenderDevice,
        frame: &FrameContext,
        #[cfg(not(target_os = "android"))] egui_renderer: Option<&mut EguiRenderer>,
        #[cfg(not(target_os = "android"))] egui_frame: Option<&EguiFrame>,
        mut inputs: VptFrameInputs<'_>,
    ) -> Result<VptFrameRecordResult> {
        let mut graph = RenderGraph::new();
        self.ensure_render_graph_transients(renderer, inputs.scene_ubo.frame_count());
        let mut pending_capture = None;
        let mut rendered_vpt = false;
        let mut vpt_accumulation_written = false;
        let mut restir_di_selected_written = false;
        let mut area_restir_selected_written = false;
        let mut current_vpt_view_proj = None;
        let mut current_nrd_frame_state = None;
        let mut nrd_adapter_pass_recorded = false;
        let traversal_stats_requested = inputs.lighting_settings.vpt_traversal_stats_enabled;
        let scene_key = Self::make_scene_key(
            inputs.sun_direction,
            inputs.sun_intensity,
            inputs.lighting_settings,
            inputs.restir_di_settings,
            inputs.area_restir_settings,
            inputs.restir_di_enabled,
            inputs.area_restir_enabled,
        );
        let scene_changed = self.frame_state.last_vpt_scene_key != Some(scene_key);
        if scene_changed {
            self.frame_state.reset_for_scene_change();
            self.frame_state.last_vpt_scene_key = Some(scene_key);
        }

        if inputs.ucvh_ready {
            let camera = inputs.camera;
            let pixel_to_ray = compute_pixel_to_ray(
                camera.position,
                camera.forward,
                camera.up,
                camera.fov_y_radians,
                frame.swapchain_extent.width,
                frame.swapchain_extent.height,
            );
            let scene_vpt_sample_index = if self.frame_state.vpt_accumulation_needs_init {
                0
            } else {
                self.frame_state.vpt_sample_index
            };

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
                vpt_sample_index: scene_vpt_sample_index,
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
            current_vpt_view_proj = Some(current_view_proj);
            let previous_view_proj = self
                .frame_state
                .previous_vpt_view_proj
                .unwrap_or(current_view_proj);
            let previous_resolution = self
                .frame_state
                .previous_vpt_resolution
                .unwrap_or([frame.swapchain_extent.width, frame.swapchain_extent.height]);
            let history_flags = if self.frame_state.previous_vpt_view_proj.is_none() {
                VPT_HISTORY_FLAG_CAMERA_CUT
            } else if self.frame_state.previous_vpt_resolution.is_none()
                || previous_resolution
                    != [frame.swapchain_extent.width, frame.swapchain_extent.height]
            {
                VPT_HISTORY_FLAG_RESIZE
            } else {
                0
            };
            let scene_history_flags = if scene_changed {
                VPT_HISTORY_FLAG_SCENE_INVALIDATED | VPT_HISTORY_FLAG_LIGHTS_INVALIDATED
            } else {
                0
            };
            let history_uniforms = GpuVptHistoryUniforms {
                current_view_proj: current_view_proj.transpose().to_cols_array_2d(),
                previous_view_proj: previous_view_proj.transpose().to_cols_array_2d(),
                current_resolution: [frame.swapchain_extent.width, frame.swapchain_extent.height],
                previous_resolution,
                current_jitter: [0.0, 0.0],
                previous_jitter: [0.0, 0.0],
                frame_index: frame.frame_index as u32,
                history_reset_generation: self.frame_state.history_reset_generation,
                flags: history_flags | scene_history_flags,
                _pad0: 0,
            };
            if let Some(vpt_surface) = &mut self.vpt_surface_pass {
                vpt_surface.update_history_uniforms(frame.frame_slot, &history_uniforms);
                vpt_surface
                    .update_motion_guide_state(frame.frame_slot, inputs.ucvh_motion_event_count);
            }
            let current_nrd_world_to_view =
                compute_nrd_world_to_view(camera.position, camera.forward, camera.up);
            let current_nrd_view_to_clip = compute_nrd_view_to_clip(
                camera.fov_y_radians,
                frame.swapchain_extent.width,
                frame.swapchain_extent.height,
            );
            let nrd_reset_history = self.frame_state.vpt_accumulation_needs_init
                || history_flags != 0
                || scene_history_flags != 0;
            let previous_nrd_world_to_view = if nrd_reset_history {
                current_nrd_world_to_view
            } else {
                self.frame_state
                    .previous_nrd_world_to_view
                    .unwrap_or(current_nrd_world_to_view)
            };
            let previous_nrd_view_to_clip = if nrd_reset_history {
                current_nrd_view_to_clip
            } else {
                self.frame_state
                    .previous_nrd_view_to_clip
                    .unwrap_or(current_nrd_view_to_clip)
            };
            let previous_nrd_elapsed_seconds = self.frame_state.previous_nrd_elapsed_seconds;
            let nrd_time_delta_seconds = previous_nrd_elapsed_seconds
                .map(|previous| inputs.elapsed_seconds - previous)
                .unwrap_or(0.0);
            let nrd_frame_settings = if matches!(
                inputs.lighting_settings.denoiser_mode,
                VptDenoiserMode::Relax | VptDenoiserMode::Reblur
            ) {
                let settings = VptNrdFrameSettings::from_inputs(VptNrdFrameSettingsInputs {
                    current_world_to_view: current_nrd_world_to_view,
                    previous_world_to_view: previous_nrd_world_to_view,
                    current_view_to_clip: current_nrd_view_to_clip,
                    previous_view_to_clip: previous_nrd_view_to_clip,
                    current_resolution: [
                        frame.swapchain_extent.width,
                        frame.swapchain_extent.height,
                    ],
                    previous_resolution,
                    frame_index: frame.frame_index as u32,
                    time_delta_seconds: nrd_time_delta_seconds,
                    reset_history: nrd_reset_history,
                    history_confidence_available: true,
                    relax_atrous_iteration_num: inputs.lighting_settings.denoiser_atrous_iterations,
                    enable_validation: matches!(
                        inputs.lighting_settings.vpt_debug_view,
                        VptDebugView::NrdValidation
                    ),
                })?;
                current_nrd_frame_state = Some((
                    current_nrd_world_to_view,
                    current_nrd_view_to_clip,
                    inputs.elapsed_seconds,
                ));
                Some(settings)
            } else {
                None
            };

            if let (
                Some(vpt_surface),
                Some(vpt),
                Some(vpt_temporal),
                Some(vpt_atrous),
                Some(postprocess),
            ) = (
                &self.vpt_surface_pass,
                &self.vpt_pass,
                &self.vpt_temporal_pass,
                &self.vpt_atrous_pass,
                &self.postprocess_pass,
            ) {
                let slot = frame.frame_slot;
                let profiler = inputs.profiler;
                let traversal_stats_resource = if traversal_stats_requested {
                    self.traversal_stats_buffer(frame.frame_slot).map(|buffer| {
                        buffer.clear_cpu();
                        graph.import_buffer_with_access(
                            buffer.handle(),
                            buffer.size(),
                            buffer.usage(),
                            AccessKind::Undefined,
                        )
                    })
                } else {
                    None
                };
                let surface_graph = vpt_surface.register_bootstrap_graph(
                    &mut graph,
                    self.frame_state.vpt_temporal_history_initialized,
                    slot,
                    traversal_stats_resource,
                    profiler,
                );
                let bootstrap_surface_resources = surface_graph.surface_writes;
                let mut final_surface_writes = bootstrap_surface_resources;
                let previous_surface_resources = surface_graph.previous_surface_resources;
                let mut traversal_stats_resource = surface_graph.traversal_stats_resource;
                let mut vpt_area_restir_reads = None;
                if inputs.area_restir_enabled
                    && let Some(area_restir) = &self.area_restir_pass
                {
                    let area_graph = area_restir.register_graph(
                        &mut graph,
                        renderer.device(),
                        vpt,
                        vpt_surface,
                        frame.frame_slot,
                        frame.frame_index,
                        inputs.area_restir_settings,
                        self.frame_state.area_restir_history_initialized,
                        bootstrap_surface_resources,
                        previous_surface_resources,
                        traversal_stats_resource,
                        inputs.profiler,
                    );
                    final_surface_writes = area_graph.final_surface_writes;
                    traversal_stats_resource = area_graph.traversal_stats_resource;
                    vpt_area_restir_reads = Some((
                        area_graph.uniform_resource,
                        area_graph.selected_current_resource,
                    ));
                    area_restir_selected_written = true;
                }

                let mut vpt_restir_reads = None;
                if inputs.restir_di_enabled
                    && let Some(restir_di) = &self.restir_di_pass
                {
                    let restir_graph = restir_di.register_graph(
                        &mut graph,
                        renderer.device(),
                        vpt,
                        frame.frame_slot,
                        frame.frame_index,
                        inputs.restir_di_settings,
                        self.frame_state.restir_di_history_initialized,
                        final_surface_writes,
                        previous_surface_resources,
                        inputs.profiler,
                    );
                    vpt_restir_reads = Some((
                        restir_graph.uniform_resource,
                        restir_graph.selected_current_resource,
                    ));
                    restir_di_selected_written = true;
                }

                let nrd_confidence_outputs = if matches!(
                    inputs.lighting_settings.denoiser_mode,
                    VptDenoiserMode::Relax | VptDenoiserMode::Reblur
                ) {
                    self.vpt_nrd_confidence_pass
                        .as_ref()
                        .map(|vpt_nrd_confidence| {
                            vpt_nrd_confidence.register_graph(
                                &mut graph,
                                VptNrdConfidenceGraphInputs {
                                    frame_slot: slot,
                                    surface_inputs: final_surface_writes,
                                    previous_surface_inputs: previous_surface_resources,
                                    profiler,
                                },
                            )
                        })
                } else {
                    None
                };

                vpt_accumulation_written = true;
                rendered_vpt = true;

                let vpt_outputs = vpt.register_graph(
                    &mut graph,
                    VptGraphInputs {
                        frame_slot: slot,
                        accumulation_needs_init: self.frame_state.vpt_accumulation_needs_init,
                        restir_reads: vpt_restir_reads,
                        area_restir_reads: vpt_area_restir_reads,
                        traversal_stats_resource,
                        profiler,
                    },
                );
                let noisy_radiance_dep = vpt_outputs.noisy_radiance;
                let noisy_moments_dep = vpt_outputs.noisy_moments;
                let nrd_frontend_outputs = if matches!(
                    inputs.lighting_settings.denoiser_mode,
                    VptDenoiserMode::Relax | VptDenoiserMode::Reblur
                ) {
                    self.vpt_nrd_frontend_pass.as_ref().map(|vpt_nrd_frontend| {
                        vpt_nrd_frontend.register_graph(
                            &mut graph,
                            VptNrdFrontendGraphInputs {
                                frame_slot: slot,
                                raw_noisy: vpt_outputs.nrd_noisy,
                                surface_inputs: final_surface_writes,
                                profiler,
                            },
                        )
                    })
                } else {
                    None
                };
                let nrd_adapter_outputs = if matches!(
                    inputs.lighting_settings.denoiser_mode,
                    VptDenoiserMode::Relax | VptDenoiserMode::Reblur
                ) {
                    if let (
                        Some(vpt_nrd_adapter),
                        Some(vpt_nrd_frontend),
                        Some(vpt_nrd_confidence),
                        Some(nrd_frame_settings),
                        Some(nrd_frontend_outputs),
                        Some(nrd_confidence_outputs),
                    ) = (
                        self.vpt_nrd_adapter_pass.as_mut(),
                        self.vpt_nrd_frontend_pass.as_ref(),
                        self.vpt_nrd_confidence_pass.as_ref(),
                        nrd_frame_settings,
                        nrd_frontend_outputs,
                        nrd_confidence_outputs,
                    ) {
                        if vpt_nrd_adapter.is_ready() {
                            match vpt_nrd_adapter.update_frame_settings(
                                renderer,
                                nrd_frame_settings,
                                VptNrdAdapterPassImageRefs {
                                    frontend: vpt_nrd_frontend,
                                    confidence: vpt_nrd_confidence,
                                    surface: vpt_surface,
                                },
                                slot,
                            ) {
                                Ok(()) => Some(vpt_nrd_adapter.register_graph(
                                    &mut graph,
                                    VptNrdAdapterGraphInputs {
                                        frame_slot: slot,
                                        packed: nrd_frontend_outputs.packed,
                                        confidence: nrd_confidence_outputs.confidence,
                                        surface_inputs: final_surface_writes,
                                        texture_pools_initialized:
                                            self.frame_state.vpt_nrd_texture_pools_initialized,
                                        profiler,
                                    },
                                )),
                                Err(error) => {
                                    tracing::error!(%error, "failed to update VPT NRD frame settings");
                                    None
                                }
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                nrd_adapter_pass_recorded = nrd_adapter_outputs.is_some();
                let nrd_resolve_outputs = if matches!(
                    inputs.lighting_settings.denoiser_mode,
                    VptDenoiserMode::Relax | VptDenoiserMode::Reblur
                ) {
                    if let (
                        Some(vpt_nrd_resolve),
                        Some(nrd_adapter_outputs),
                        Some(nrd_frontend_outputs),
                    ) = (
                        self.vpt_nrd_resolve_pass.as_ref(),
                        nrd_adapter_outputs,
                        nrd_frontend_outputs,
                    ) {
                        let _ = nrd_adapter_outputs.resources.validation;
                        Some(vpt_nrd_resolve.register_graph(
                            &mut graph,
                            VptNrdResolveGraphInputs {
                                frame_slot: slot,
                                denoised_diff_radiance_hitdist:
                                    nrd_adapter_outputs.resources.diff_radiance_hitdist,
                                packed: nrd_frontend_outputs.packed,
                                profiler,
                            },
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let nrd_resolve_available = nrd_resolve_outputs.is_some();
                let actual_effective_denoiser_mode_name = capture_effective_denoiser_mode_name(
                    inputs.lighting_settings,
                    nrd_resolve_available,
                );

                let temporal_outputs = vpt_temporal.register_graph(
                    &mut graph,
                    VptTemporalGraphInputs {
                        frame_slot: slot,
                        history_initialized: self.frame_state.vpt_temporal_history_initialized,
                        noisy_inputs: [noisy_radiance_dep, noisy_moments_dep],
                        surface_inputs: final_surface_writes,
                        previous_surface_inputs: previous_surface_resources,
                        profiler,
                    },
                );
                let temporal_radiance_dep = temporal_outputs.accumulated_radiance;
                let temporal_moments_dep = temporal_outputs.accumulated_moments;

                let atrous_outputs = vpt_atrous.register_graph(
                    &mut graph,
                    VptAtrousGraphInputs {
                        frame_slot: slot,
                        lighting_settings: inputs.lighting_settings,
                        temporal_radiance: temporal_radiance_dep,
                        temporal_moments: temporal_moments_dep,
                        surface_inputs: final_surface_writes,
                        profiler,
                    },
                );
                let atrous_filtered_dep = atrous_outputs.filtered_radiance;

                vpt_temporal.register_history_update_graph(
                    &mut graph,
                    vpt_surface,
                    temporal_outputs,
                    final_surface_writes,
                    previous_surface_resources,
                );

                let postprocess_output = postprocess.output_image.handle;
                let postprocess_extent = postprocess.output_image.extent;
                let (postprocess_input_radiance, postprocess_hdr_image) = if matches!(
                    inputs.lighting_settings.vpt_debug_view,
                    VptDebugView::NrdValidation
                ) {
                    if let Some(nrd_adapter_outputs) = nrd_adapter_outputs {
                        (
                            nrd_adapter_outputs.resources.validation,
                            nrd_adapter_outputs.validation_image,
                        )
                    } else {
                        (atrous_filtered_dep, vpt_atrous.output_image())
                    }
                } else if inputs.lighting_settings.vpt_debug_view == VptDebugView::Final {
                    if let Some(nrd_resolve_outputs) = nrd_resolve_outputs {
                        if let Some(vpt_nrd_resolve) = self.vpt_nrd_resolve_pass.as_ref() {
                            (
                                nrd_resolve_outputs.resolved_radiance,
                                vpt_nrd_resolve.output_image(),
                            )
                        } else {
                            (atrous_filtered_dep, vpt_atrous.output_image())
                        }
                    } else {
                        (atrous_filtered_dep, vpt_atrous.output_image())
                    }
                } else {
                    (atrous_filtered_dep, vpt_atrous.output_image())
                };
                let postprocess_outputs = postprocess.register_graph(
                    &mut graph,
                    PostprocessGraphInputs {
                        device: renderer.device(),
                        frame_slot: frame.frame_slot,
                        input_radiance: postprocess_input_radiance,
                        hdr_image: postprocess_hdr_image,
                        output_initialized: self.frame_state.postprocess_output_initialized,
                        profiler,
                    },
                );
                let src_image = postprocess_output;
                let src_extent = postprocess_extent;
                let dst_image = frame.swapchain_image;
                let dst_extent = frame.swapchain_extent;
                let dep_handle = postprocess_outputs.output;
                #[cfg(not(target_os = "android"))]
                let has_egui_overlay =
                    egui_renderer.is_some() && egui_frame.is_some_and(|frame| !frame.is_empty());
                #[cfg(target_os = "android")]
                let has_egui_overlay = false;
                let mut capture_dependency = None;
                let capture_frame = inputs
                    .capture
                    .as_ref()
                    .is_some_and(|capture| capture.should_capture(frame.frame_index));
                if capture_frame && let Some(capture) = inputs.capture.as_deref_mut() {
                    let restir_di_temporal_enabled =
                        inputs.restir_di_enabled && inputs.restir_di_settings.temporal_enabled;
                    let restir_di_spatial_enabled =
                        restir_di_temporal_enabled && inputs.restir_di_settings.spatial_enabled;
                    let area_restir_temporal_enabled =
                        inputs.area_restir_enabled && inputs.area_restir_settings.temporal_enabled;
                    let area_restir_spatial_enabled =
                        area_restir_temporal_enabled && inputs.area_restir_settings.spatial_enabled;
                    let readback = capture.ensure_readback(
                        renderer.device(),
                        renderer.allocator(),
                        postprocess_extent.width,
                        postprocess_extent.height,
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
                    let capture_writes =
                        graph.add_pass("capture_postprocess", QueueType::Transfer, |builder| {
                            builder.read_as(dep_handle, AccessKind::TransferRead);
                            builder.write_as(readback_resource, AccessKind::TransferWrite);
                            Box::new(move |ctx| {
                                cmd_copy_image_to_buffer(
                                    ctx.device,
                                    ctx.command_buffer,
                                    src_image,
                                    src_extent,
                                    readback_buffer,
                                );
                            })
                        });
                    let capture_dep = capture_writes[0];
                    let paths = capture.config().paths_for_frame(frame.frame_index);
                    pending_capture = Some(CaptureMetadata {
                        frame_index: frame.frame_index,
                        vpt_sample_index: scene_vpt_sample_index,
                        width: postprocess_extent.width,
                        height: postprocess_extent.height,
                        source: "postprocess_output",
                        ppm_path: paths.ppm_path,
                        json_path: paths.json_path,
                        restir_di_enabled: inputs.restir_di_enabled,
                        restir_di_temporal_enabled,
                        restir_di_spatial_enabled,
                        area_restir_enabled: inputs.area_restir_enabled,
                        area_restir_temporal_enabled,
                        area_restir_spatial_enabled,
                        vpt_debug_view: vpt_debug_view_name(
                            inputs.lighting_settings.vpt_debug_view,
                        ),
                        denoiser_enabled: inputs.lighting_settings.denoiser_enabled(),
                        denoiser_mode: inputs.lighting_settings.denoiser_mode_name(),
                        effective_denoiser_mode: actual_effective_denoiser_mode_name,
                    });
                    tracing::info!(
                        frame_index = frame.frame_index,
                        width = postprocess_extent.width,
                        height = postprocess_extent.height,
                        "queued postprocess capture"
                    );
                    capture_dependency = Some(capture_dep);
                }

                let swapchain_dep = graph.import_image_with_access(
                    dst_image,
                    dst_extent.width,
                    dst_extent.height,
                    frame.swapchain_format,
                    vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                    swapchain_access_from_layout(frame.swapchain_image_layout)?,
                );
                let blit_writes =
                    graph.add_pass("blit_to_swapchain", QueueType::Graphics, |builder| {
                        builder.read_as(dep_handle, AccessKind::TransferRead);
                        if let Some(capture_dep) = capture_dependency {
                            builder.depend_on(capture_dep);
                        }
                        builder.write_as(swapchain_dep, AccessKind::TransferWrite);
                        if !has_egui_overlay {
                            builder.finish_as(swapchain_dep, AccessKind::Present);
                        }
                        Box::new(move |ctx| {
                            if let Some(profiler) = profiler {
                                profiler.begin_scope(
                                    ctx.device,
                                    ctx.command_buffer,
                                    slot,
                                    GpuProfileScope::BlitToSwapchain,
                                );
                            }
                            blit_to_swapchain::record_blit_core(
                                ctx.device,
                                ctx.command_buffer,
                                src_image,
                                src_extent,
                                dst_image,
                                dst_extent,
                            );
                            if let Some(profiler) = profiler {
                                profiler.end_scope(
                                    ctx.device,
                                    ctx.command_buffer,
                                    slot,
                                    GpuProfileScope::BlitToSwapchain,
                                );
                            }
                        })
                    });
                #[cfg(not(target_os = "android"))]
                if has_egui_overlay {
                    let swapchain_after_blit = blit_writes[0];
                    if let (Some(egui_renderer), Some(egui_frame)) = (egui_renderer, egui_frame) {
                        graph.add_pass("egui_overlay", QueueType::Graphics, |builder| {
                            builder
                                .write_as(swapchain_after_blit, AccessKind::ColorAttachmentWrite);
                            builder.finish_as(swapchain_after_blit, AccessKind::Present);
                            Box::new(move |_ctx| {
                                if let Err(error) =
                                    egui_renderer.record(renderer, frame, egui_frame)
                                {
                                    tracing::error!(%error, "failed to record egui overlay");
                                }
                            })
                        });
                    }
                }
            } else {
                self.frame_state.vpt_sample_index = 0;
                self.frame_state.last_vpt_camera_key = None;
                tracing::warn!(
                    vpt_ready = self.vpt_pass.is_some(),
                    vpt_nrd_confidence_ready = self.vpt_nrd_confidence_pass.is_some(),
                    vpt_nrd_frontend_ready = self.vpt_nrd_frontend_pass.is_some(),
                    vpt_nrd_adapter_ready = self.vpt_nrd_adapter_pass.is_some(),
                    vpt_nrd_resolve_ready = self.vpt_nrd_resolve_pass.is_some(),
                    vpt_temporal_ready = self.vpt_temporal_pass.is_some(),
                    vpt_atrous_ready = self.vpt_atrous_pass.is_some(),
                    postprocess_ready = self.postprocess_pass.is_some(),
                    "skipping VPT frame until required passes are initialized"
                );
            }
        } else {
            tracing::warn!("skipping UCVH render passes until GPU upload succeeds");
        }

        if !graph.has_final_access(AccessKind::Present) {
            tracing::warn!(
                "render graph produced no presentable output; clearing swapchain fallback"
            );
            add_swapchain_clear_present_pass(
                &mut graph,
                frame.swapchain_image,
                frame.swapchain_extent,
                frame.swapchain_format,
                frame.swapchain_image_layout,
            )?;
        }
        let transients = self
            .render_graph_transients
            .get_mut(frame.frame_slot)
            .context("missing RenderGraph transient resources for frame slot")?;
        graph.execute_with_transient_resources(
            renderer.device(),
            renderer.allocator(),
            transients,
            frame.command_buffer,
            frame.frame_index,
        )?;

        if let Some(current_view_proj) = current_vpt_view_proj {
            self.frame_state.previous_vpt_view_proj = Some(current_view_proj);
            self.frame_state.previous_vpt_resolution =
                Some([frame.swapchain_extent.width, frame.swapchain_extent.height]);
        }
        if let Some((world_to_view, view_to_clip, elapsed_seconds)) = current_nrd_frame_state {
            self.frame_state.previous_nrd_world_to_view = Some(world_to_view);
            self.frame_state.previous_nrd_view_to_clip = Some(view_to_clip);
            self.frame_state.previous_nrd_elapsed_seconds = Some(elapsed_seconds);
        }
        if nrd_adapter_pass_recorded {
            self.frame_state.vpt_nrd_texture_pools_initialized = true;
        }
        if vpt_accumulation_written {
            self.frame_state.vpt_sample_index = self.frame_state.vpt_sample_index.saturating_add(1);
            self.frame_state.vpt_accumulation_needs_init = false;
            self.frame_state.vpt_temporal_history_initialized = true;
            self.frame_state.postprocess_output_initialized = true;
        }
        if restir_di_selected_written {
            self.frame_state.restir_di_history_initialized = true;
        }
        if area_restir_selected_written {
            self.frame_state.area_restir_history_initialized = true;
        }

        Ok(VptFrameRecordResult {
            pending_capture,
            submitted_fence: frame.in_flight_fence,
            rendered_vpt,
            traversal_stats_requested,
            traversal_stats: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn resize(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh_gpu: &UcvhGpuResources,
        width: u32,
        height: u32,
        lighting_settings: LightingSettings,
        _restir_di_enabled: bool,
        _area_restir_enabled: bool,
    ) -> Result<()> {
        let device = renderer.device().clone();
        let allocator = renderer.allocator();
        self.frame_state.reset_for_resize_or_camera_cut();
        renderer
            .wait_idle()
            .context("failed to idle Vulkan device before resizing VPT resources")?;

        if let Some(vpt) = &mut self.vpt_pass {
            vpt.resize_images(
                &device,
                allocator,
                VptPassResizeInfo {
                    width,
                    height,
                    scene_ubo,
                    ucvh_gpu,
                    traversal_stats_buffers: &self.traversal_stats_buffers,
                },
            )
            .context("failed to resize VPT images")?;
        }
        if let Some(vpt_surface) = &mut self.vpt_surface_pass {
            vpt_surface
                .resize_images(
                    &device,
                    allocator,
                    VptSurfacePassResizeInfo {
                        width,
                        height,
                        scene_ubo,
                        ucvh_gpu,
                        traversal_stats_buffers: &self.traversal_stats_buffers,
                    },
                )
                .context("failed to resize VPT surface images")?;
        }
        if let (Some(restir_di), Some(vpt_surface)) = (&self.restir_di_pass, &self.vpt_surface_pass)
        {
            restir_di.update_surface_descriptors(&device, vpt_surface);
        }
        if let (Some(area_restir), Some(vpt_surface)) =
            (&self.area_restir_pass, &self.vpt_surface_pass)
        {
            area_restir.update_surface_descriptors(&device, vpt_surface);
            area_restir.update_scene_descriptors(&device, scene_ubo);
            area_restir.update_ucvh_descriptors(&device, ucvh_gpu);
        }
        if let (Some(vpt_nrd_frontend), Some(vpt), Some(vpt_surface)) = (
            &mut self.vpt_nrd_frontend_pass,
            &self.vpt_pass,
            &self.vpt_surface_pass,
        ) {
            vpt_nrd_frontend
                .resize_images(
                    &device,
                    allocator,
                    VptNrdFrontendPassResizeInfo {
                        width,
                        height,
                        scene_ubo,
                        vpt,
                        vpt_surface,
                    },
                )
                .context("failed to resize VPT NRD frontend images")?;
        }
        if let (Some(vpt_nrd_confidence), Some(vpt_surface)) =
            (&mut self.vpt_nrd_confidence_pass, &self.vpt_surface_pass)
        {
            vpt_nrd_confidence
                .resize_images(
                    &device,
                    allocator,
                    VptNrdConfidencePassResizeInfo {
                        width,
                        height,
                        scene_ubo,
                        vpt_surface,
                    },
                )
                .context("failed to resize VPT NRD confidence images")?;
        }
        if let (
            Some(vpt_nrd_adapter),
            Some(vpt_nrd_frontend),
            Some(vpt_nrd_confidence),
            Some(vpt_surface),
        ) = (
            &mut self.vpt_nrd_adapter_pass,
            &self.vpt_nrd_frontend_pass,
            &self.vpt_nrd_confidence_pass,
            &self.vpt_surface_pass,
        ) {
            vpt_nrd_adapter
                .resize_images(
                    &device,
                    allocator,
                    VptNrdAdapterPassResizeInfo {
                        width,
                        height,
                        denoiser_mode: lighting_settings.denoiser_mode,
                        relax_atrous_iteration_num: lighting_settings.denoiser_atrous_iterations,
                        constant_buffer_alignment: renderer
                            .physical_device_properties()
                            .limits
                            .min_uniform_buffer_offset_alignment,
                        scene_ubo,
                        image_refs: VptNrdAdapterPassImageRefs {
                            frontend: vpt_nrd_frontend,
                            confidence: vpt_nrd_confidence,
                            surface: vpt_surface,
                        },
                    },
                )
                .context("failed to resize VPT NRD adapter images")?;
        }
        if let (Some(vpt_nrd_resolve), Some(vpt_nrd_adapter), Some(vpt_nrd_frontend)) = (
            &mut self.vpt_nrd_resolve_pass,
            &self.vpt_nrd_adapter_pass,
            &self.vpt_nrd_frontend_pass,
        ) {
            vpt_nrd_resolve
                .resize_images(
                    &device,
                    allocator,
                    VptNrdResolvePassResizeInfo {
                        width,
                        height,
                        scene_ubo,
                        denoised_diff_radiance_hitdist: &vpt_nrd_adapter.nrd_diff_radiance_hitdist,
                        frontend: vpt_nrd_frontend,
                    },
                )
                .context("failed to resize VPT NRD resolve images")?;
        }
        if let Some(restir_di) = &mut self.restir_di_pass {
            restir_di
                .resize_buffers(&device, allocator, width, height)
                .context("failed to resize ReSTIR-DI buffers")?;
        }
        if let Some(area_restir) = &mut self.area_restir_pass {
            area_restir
                .resize_buffers(&device, allocator, width, height)
                .context("failed to resize Area ReSTIR buffers")?;
            area_restir.update_scene_descriptors(&device, scene_ubo);
            area_restir.update_ucvh_descriptors(&device, ucvh_gpu);
            if let Some(vpt) = &self.vpt_pass {
                for slot in 0..scene_ubo.frame_count() {
                    let (area_uniform_buffer, _, _) = area_restir.uniform_buffer(slot);
                    let (area_selected_current_buffer, _, _) =
                        area_restir.selected_current_buffer(slot);
                    vpt.update_area_restir_descriptors(
                        &device,
                        slot,
                        area_uniform_buffer,
                        area_selected_current_buffer,
                    );
                }
            }
        }
        if let (Some(vpt_temporal), Some(vpt), Some(vpt_surface)) = (
            &mut self.vpt_temporal_pass,
            &self.vpt_pass,
            &self.vpt_surface_pass,
        ) {
            vpt_temporal
                .resize_images(
                    &device,
                    allocator,
                    VptTemporalPassResizeInfo {
                        width,
                        height,
                        scene_ubo,
                        vpt,
                        vpt_surface,
                    },
                )
                .context("failed to resize VPT temporal images")?;
        }
        if let (Some(vpt_atrous), Some(vpt_temporal), Some(vpt_surface)) = (
            &mut self.vpt_atrous_pass,
            &self.vpt_temporal_pass,
            &self.vpt_surface_pass,
        ) {
            vpt_atrous
                .resize_images(
                    &device,
                    allocator,
                    VptAtrousPassResizeInfo {
                        width,
                        height,
                        scene_ubo,
                        temporal: vpt_temporal,
                        vpt_surface,
                    },
                )
                .context("failed to resize VPT atrous images")?;
        }
        if let (Some(postprocess), Some(vpt_atrous)) =
            (&mut self.postprocess_pass, &self.vpt_atrous_pass)
        {
            postprocess
                .resize_images(
                    &device,
                    allocator,
                    width,
                    height,
                    vpt_atrous.output_image(),
                    scene_ubo,
                )
                .context("failed to resize VPT postprocess images")?;
        }

        Ok(())
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        if let Some(pass) = self.postprocess_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_atrous_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_nrd_resolve_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_nrd_confidence_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_nrd_frontend_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_nrd_adapter_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_temporal_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.area_restir_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_surface_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.restir_di_pass {
            pass.destroy(device, allocator);
        }
        for buffer in self.traversal_stats_buffers {
            buffer.destroy(device, allocator);
        }
        for transients in self.render_graph_transients {
            transients.destroy(device, allocator);
        }
    }
}

fn swapchain_access_from_layout(layout: vk::ImageLayout) -> Result<AccessKind> {
    AccessKind::from_swapchain_layout(layout)
        .with_context(|| format!("unsupported tracked swapchain image layout: {layout:?}"))
}

fn add_swapchain_clear_present_pass(
    graph: &mut RenderGraph<'_>,
    dst_image: vk::Image,
    dst_extent: vk::Extent2D,
    dst_format: vk::Format,
    current_layout: vk::ImageLayout,
) -> Result<()> {
    let current_access = swapchain_access_from_layout(current_layout)?;
    let swapchain = graph.import_image_with_access(
        dst_image,
        dst_extent.width,
        dst_extent.height,
        dst_format,
        vk::ImageUsageFlags::TRANSFER_DST,
        current_access,
    );

    graph.add_pass("clear_swapchain", QueueType::Graphics, |builder| {
        builder.write_as(swapchain, AccessKind::TransferWrite);
        builder.finish_as(swapchain, AccessKind::Present);
        Box::new(move |ctx| {
            let color = vk::ClearColorValue {
                float32: [0.015, 0.018, 0.022, 1.0],
            };
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            unsafe {
                ctx.device.cmd_clear_color_image(
                    ctx.command_buffer,
                    dst_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &color,
                    std::slice::from_ref(&range),
                );
            }
        })
    });
    Ok(())
}

fn vpt_debug_view_name(debug_view: VptDebugView) -> &'static str {
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

fn capture_effective_denoiser_mode_name(
    lighting_settings: LightingSettings,
    nrd_resolve_available: bool,
) -> &'static str {
    match lighting_settings.denoiser_mode {
        VptDenoiserMode::Relax | VptDenoiserMode::Reblur => {
            if nrd_resolve_available {
                lighting_settings.denoiser_mode.as_config_value()
            } else {
                VptDenoiserMode::Svgf.as_config_value()
            }
        }
        _ => lighting_settings.denoiser_mode_name(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::area_restir::AreaRestirDebugView;
    use crate::render::restir_di::RestirDiDebugView;
    use crate::render::scene_ubo::{
        LightingDebugView, LightingSettings, RenderMode, VptDebugView, VptDenoiserMode,
    };
    use crate::voxel::ucvh::{UcvhInvalidationRegion, UcvhMotionEvent};

    fn assert_mat4_near(actual: glam::Mat4, expected: glam::Mat4, epsilon: f32) {
        for (actual, expected) in actual.to_cols_array().iter().zip(expected.to_cols_array()) {
            assert!(
                (*actual - expected).abs() <= epsilon,
                "matrix element mismatch: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn frame_state_reset_clears_history_and_accumulation() {
        let mut state = VptPipelineFrameState {
            vpt_sample_index: 7,
            last_vpt_camera_key: Some([1; 15]),
            last_vpt_scene_key: Some([2; VPT_SCENE_KEY_WORDS]),
            history_reset_generation: 9,
            vpt_accumulation_needs_init: false,
            vpt_temporal_history_initialized: true,
            postprocess_output_initialized: true,
            vpt_nrd_texture_pools_initialized: true,
            area_restir_history_initialized: true,
            restir_di_history_initialized: true,
            previous_vpt_view_proj: Some(glam::Mat4::IDENTITY),
            previous_vpt_resolution: Some([1280, 720]),
            previous_nrd_world_to_view: Some(glam::Mat4::from_scale(glam::Vec3::splat(2.0))),
            previous_nrd_view_to_clip: Some(glam::Mat4::from_scale(glam::Vec3::splat(3.0))),
            previous_nrd_elapsed_seconds: Some(12.0),
        };

        state.reset_for_resize_or_camera_cut();

        assert_eq!(state.vpt_sample_index, 0);
        assert_eq!(state.last_vpt_camera_key, None);
        assert_eq!(state.last_vpt_scene_key, Some([2; VPT_SCENE_KEY_WORDS]));
        assert_eq!(state.history_reset_generation, 10);
        assert!(state.vpt_accumulation_needs_init);
        assert!(!state.vpt_temporal_history_initialized);
        assert!(!state.postprocess_output_initialized);
        assert!(!state.vpt_nrd_texture_pools_initialized);
        assert!(!state.area_restir_history_initialized);
        assert!(!state.restir_di_history_initialized);
        assert_eq!(state.previous_vpt_view_proj, None);
        assert_eq!(state.previous_vpt_resolution, None);
        assert_eq!(state.previous_nrd_world_to_view, None);
        assert_eq!(state.previous_nrd_view_to_clip, None);
        assert_eq!(state.previous_nrd_elapsed_seconds, None);
    }

    #[test]
    fn frame_state_scene_reset_clears_output_history_and_restir_history_without_touching_camera_state()
     {
        let mut state = VptPipelineFrameState {
            vpt_sample_index: 11,
            last_vpt_camera_key: Some([1; 15]),
            last_vpt_scene_key: Some([2; VPT_SCENE_KEY_WORDS]),
            history_reset_generation: 3,
            vpt_accumulation_needs_init: false,
            vpt_temporal_history_initialized: true,
            postprocess_output_initialized: true,
            vpt_nrd_texture_pools_initialized: true,
            area_restir_history_initialized: true,
            restir_di_history_initialized: true,
            previous_vpt_view_proj: Some(glam::Mat4::IDENTITY),
            previous_vpt_resolution: Some([1280, 720]),
            previous_nrd_world_to_view: Some(glam::Mat4::from_scale(glam::Vec3::splat(2.0))),
            previous_nrd_view_to_clip: Some(glam::Mat4::from_scale(glam::Vec3::splat(3.0))),
            previous_nrd_elapsed_seconds: Some(12.0),
        };

        state.reset_for_scene_change();

        assert_eq!(state.vpt_sample_index, 0);
        assert_eq!(state.last_vpt_camera_key, None);
        assert_eq!(state.last_vpt_scene_key, None);
        assert_eq!(state.history_reset_generation, 4);
        assert!(state.vpt_accumulation_needs_init);
        assert!(!state.vpt_temporal_history_initialized);
        assert!(!state.postprocess_output_initialized);
        assert!(!state.vpt_nrd_texture_pools_initialized);
        assert!(!state.area_restir_history_initialized);
        assert!(!state.restir_di_history_initialized);
        assert_eq!(state.previous_vpt_view_proj, Some(glam::Mat4::IDENTITY));
        assert_eq!(state.previous_vpt_resolution, Some([1280, 720]));
        assert_eq!(
            state.previous_nrd_world_to_view,
            Some(glam::Mat4::from_scale(glam::Vec3::splat(2.0)))
        );
        assert_eq!(
            state.previous_nrd_view_to_clip,
            Some(glam::Mat4::from_scale(glam::Vec3::splat(3.0)))
        );
        assert_eq!(state.previous_nrd_elapsed_seconds, Some(12.0));
    }

    #[test]
    fn scene_key_reset_clears_restir_histories() {
        let mut state = VptPipelineFrameState {
            vpt_sample_index: 11,
            last_vpt_camera_key: Some([1; 15]),
            last_vpt_scene_key: Some([2; VPT_SCENE_KEY_WORDS]),
            history_reset_generation: 3,
            vpt_accumulation_needs_init: false,
            vpt_temporal_history_initialized: true,
            postprocess_output_initialized: true,
            vpt_nrd_texture_pools_initialized: true,
            area_restir_history_initialized: true,
            restir_di_history_initialized: true,
            previous_vpt_view_proj: Some(glam::Mat4::IDENTITY),
            previous_vpt_resolution: Some([1280, 720]),
            previous_nrd_world_to_view: Some(glam::Mat4::from_scale(glam::Vec3::splat(2.0))),
            previous_nrd_view_to_clip: Some(glam::Mat4::from_scale(glam::Vec3::splat(3.0))),
            previous_nrd_elapsed_seconds: Some(12.0),
        };

        state.reset_for_scene_change();

        assert!(!state.vpt_temporal_history_initialized);
        assert!(!state.postprocess_output_initialized);
        assert!(!state.vpt_nrd_texture_pools_initialized);
        assert!(!state.area_restir_history_initialized);
        assert!(!state.restir_di_history_initialized);
    }

    #[test]
    fn scene_uniform_inputs_forward_history_reset_generation_from_frame_state() {
        let source = crate::render::source_checks::read_source("src/render/vpt_pipeline.rs");
        let scene_uniforms = source
            .split("let scene_data = build_scene_uniforms(SceneUniformInputs {")
            .nth(1)
            .expect("scene uniform construction should exist")
            .split("});")
            .next()
            .expect("scene uniform construction should terminate");
        let compact = crate::render::source_checks::compact(scene_uniforms);

        assert!(
            compact.contains("history_reset_generation:self.frame_state.history_reset_generation"),
            "VPT scene uniforms must receive the frame state's history reset generation"
        );
    }

    #[test]
    fn nrd_camera_matrices_match_existing_view_projection_path() {
        let position = glam::Vec3::new(3.0, 4.0, -5.0);
        let forward = glam::Vec3::new(0.2, -0.1, 1.0).normalize();
        let up = glam::Vec3::Y;
        let fov_y = 1.1;
        let width = 1920;
        let height = 1080;

        let world_to_view = compute_nrd_world_to_view(position, forward, up);
        let view_to_clip = compute_nrd_view_to_clip(fov_y, width, height);
        let expected = compute_view_proj(position, forward, up, fov_y, width, height);

        assert_mat4_near(view_to_clip * world_to_view, expected, 1.0e-5);
    }

    #[test]
    fn capture_effective_denoiser_mode_name_reports_svgf_when_nrd_path_is_unavailable() {
        for mode in [VptDenoiserMode::Relax, VptDenoiserMode::Reblur] {
            let settings = LightingSettings {
                denoiser_mode: mode,
                ..LightingSettings::default()
            };

            assert_eq!(
                capture_effective_denoiser_mode_name(settings, true),
                mode.as_config_value()
            );
            assert_eq!(
                capture_effective_denoiser_mode_name(settings, false),
                VptDenoiserMode::Svgf.as_config_value()
            );
        }

        let settings = LightingSettings {
            denoiser_mode: VptDenoiserMode::Off,
            ..LightingSettings::default()
        };

        assert_eq!(
            capture_effective_denoiser_mode_name(settings, false),
            VptDenoiserMode::Off.as_config_value()
        );
    }

    #[test]
    fn ucvh_frame_changes_preserve_motion_guide_events_without_implying_reset() {
        let no_changes = UcvhFrameChanges::default();
        assert!(no_changes.invalidation_regions.is_empty());
        assert!(no_changes.motion_events.is_empty());

        let invalidated = UcvhFrameChanges::new(
            vec![UcvhInvalidationRegion {
                brick_min: glam::UVec3::new(1, 2, 3),
                brick_max_exclusive: glam::UVec3::new(2, 3, 4),
                generation: 7,
            }],
            Vec::new(),
        );
        assert_eq!(invalidated.invalidation_regions.len(), 1);
        assert!(invalidated.motion_events.is_empty());

        let moved = UcvhFrameChanges::new(
            Vec::new(),
            vec![UcvhMotionEvent {
                region_min: glam::UVec3::new(8, 8, 8),
                region_max_exclusive: glam::UVec3::new(16, 16, 16),
                world_delta_current_from_previous: glam::IVec3::new(1, 0, 0),
                generation: 8,
            }],
        );
        assert!(moved.invalidation_regions.is_empty());
        assert_eq!(moved.motion_events.len(), 1);
    }

    #[test]
    fn ucvh_frame_changes_do_not_request_frame_wide_history_reset() {
        let source = crate::render::source_checks::read_source("src/render/vpt_pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("implementation section should exist");

        assert!(
            !implementation.contains("reset_for_ucvh_content_change"),
            "UCVH content and motion events must be handled by per-pixel motion guide generation checks"
        );
        assert!(
            !implementation.contains("invalidates_history"),
            "UcvhFrameChanges must not be used as a frame-wide history invalidation trigger"
        );
    }

    #[test]
    fn scene_key_changes_when_light_and_reuse_settings_change() {
        let base = VptRuntimePipeline::make_scene_key(
            glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
            glam::Vec3::new(2.0, 1.5, 1.25),
            LightingSettings {
                shadows_enabled: true,
                skip_backface_shadows: false,
                render_mode: RenderMode::Vpt,
                vpt_max_bounces: 2,
                sun_angular_radius: 0.02,
                debug_view: LightingDebugView::Final,
                exposure: 1.0,
                denoiser_mode: VptDenoiserMode::Svgf,
                denoiser_atrous_iterations: 4,
                vpt_debug_view: VptDebugView::Final,
                ..LightingSettings::default()
            },
            RestirDiSettings::default(),
            AreaRestirSettings::default(),
            false,
            false,
        );
        let changed = VptRuntimePipeline::make_scene_key(
            glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
            glam::Vec3::new(3.0, 2.0, 1.5),
            LightingSettings {
                shadows_enabled: false,
                skip_backface_shadows: true,
                render_mode: RenderMode::Vpt,
                vpt_max_bounces: 4,
                sun_angular_radius: 0.05,
                debug_view: LightingDebugView::Final,
                exposure: 1.0,
                denoiser_mode: VptDenoiserMode::Off,
                denoiser_atrous_iterations: 2,
                vpt_debug_view: VptDebugView::Final,
                ..LightingSettings::default()
            },
            RestirDiSettings::default(),
            AreaRestirSettings::default(),
            true,
            true,
        );

        assert_ne!(base, changed);
    }

    #[test]
    fn scene_key_tracks_requested_denoiser_mode() {
        let base_settings = LightingSettings {
            denoiser_mode: VptDenoiserMode::Svgf,
            ..LightingSettings::default()
        };
        let relax_settings = LightingSettings {
            denoiser_mode: VptDenoiserMode::Relax,
            ..LightingSettings::default()
        };

        let base = VptRuntimePipeline::make_scene_key(
            glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
            glam::Vec3::new(2.0, 1.5, 1.25),
            base_settings,
            RestirDiSettings::default(),
            AreaRestirSettings::default(),
            false,
            false,
        );
        let relax = VptRuntimePipeline::make_scene_key(
            glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
            glam::Vec3::new(2.0, 1.5, 1.25),
            relax_settings,
            RestirDiSettings::default(),
            AreaRestirSettings::default(),
            false,
            false,
        );

        assert_ne!(base, relax);
    }

    #[test]
    fn nrd_adapter_rebuild_waits_for_submitted_gpu_work_before_destroying_old_pass() {
        let source = crate::render::source_checks::read_source("src/render/vpt_pipeline.rs");
        let ensure_adapter = source
            .split("fn ensure_vpt_nrd_adapter_pass")
            .nth(1)
            .expect("VptRuntimePipeline::ensure_vpt_nrd_adapter_pass should exist")
            .split("fn ensure_vpt_nrd_resolve_pass")
            .next()
            .expect("NRD adapter ensure function should end before resolve ensure");
        let compact = crate::render::source_checks::compact(ensure_adapter);

        assert!(compact.contains("self.destroy_vpt_nrd_adapter_chain(renderer);"));
        assert!(compact.contains("if!self.destroy_vpt_nrd_adapter_chain(renderer){return;}"));

        let teardown = source
            .split("fn destroy_vpt_nrd_adapter_chain")
            .nth(1)
            .expect("VptRuntimePipeline::destroy_vpt_nrd_adapter_chain should exist")
            .split("fn ensure_vpt_nrd_resolve_pass")
            .next()
            .expect("NRD adapter teardown helper should end before resolve ensure");
        let compact_teardown = crate::render::source_checks::compact(teardown);
        let wait = compact_teardown
            .find("renderer.wait_idle(")
            .expect("NRD adapter teardown must wait for submitted GPU work");
        let destroy_resolve = compact_teardown
            .find("pass.destroy(renderer.device(),renderer.allocator());")
            .expect("NRD adapter teardown should destroy stale pass resources");

        assert!(
            wait < destroy_resolve,
            "NRD adapter pass rebuild must wait for submitted GPU work before destroying pipelines/images/views referenced by earlier frame command buffers"
        );
        assert!(
            compact_teardown.contains("self.vpt_nrd_resolve_pass.take()"),
            "NRD resolve pass descriptors point at adapter images and must be torn down with the adapter"
        );
        assert!(
            compact_teardown.contains("self.vpt_nrd_adapter_pass.take()"),
            "NRD adapter pass should be destroyed only through the synchronized teardown helper"
        );
    }

    #[test]
    fn resize_waits_for_submitted_gpu_work_before_replacing_pass_resources() {
        let source = crate::render::source_checks::read_source("src/render/vpt_pipeline.rs");
        let resize = source
            .split("pub fn resize(")
            .nth(1)
            .expect("VptRuntimePipeline::resize should exist")
            .split("pub fn destroy(")
            .next()
            .expect("resize should end before destroy");
        let compact = crate::render::source_checks::compact(resize);
        let wait = compact
            .find("renderer.wait_idle(")
            .expect("VptRuntimePipeline::resize must wait for submitted GPU work");
        let first_resize = compact
            .find(".resize_images(")
            .expect("VptRuntimePipeline::resize should resize pass-owned images");

        assert!(
            wait < first_resize,
            "pass resize replaces and destroys old images/buffers, so it must wait before any resize_images/resize_buffers call"
        );
    }

    #[test]
    fn resize_refreshes_existing_restir_passes_even_when_temporarily_disabled() {
        let source = crate::render::source_checks::read_source("src/render/vpt_pipeline.rs");
        let resize = source
            .split("pub fn resize(")
            .nth(1)
            .expect("VptRuntimePipeline::resize should exist")
            .split("pub fn destroy(")
            .next()
            .expect("resize should end before destroy");
        let compact = crate::render::source_checks::compact(resize);

        assert!(
            compact.contains("ifletSome(restir_di)=&mutself.restir_di_pass{")
                && compact.contains("restir_di.resize_buffers(&device,allocator,width,height)")
                && compact.contains("restir_di.update_surface_descriptors(&device,vpt_surface);")
                && !compact
                    .contains("ifrestir_di_enabled&&letSome(restir_di)=&mutself.restir_di_pass{"),
            "resize must refresh an existing ReSTIR-DI pass even when the pass is temporarily disabled"
        );

        assert!(
            compact.contains("ifletSome(area_restir)=&mutself.area_restir_pass{")
                && compact.contains("area_restir.resize_buffers(&device,allocator,width,height)")
                && compact.contains("area_restir.update_surface_descriptors(&device,vpt_surface);")
                && compact.contains("area_restir.update_scene_descriptors(&device,scene_ubo);")
                && compact.contains("area_restir.update_ucvh_descriptors(&device,ucvh_gpu);")
                && !compact.contains(
                    "ifarea_restir_enabled&&letSome(area_restir)=&mutself.area_restir_pass{"
                ),
            "resize must refresh an existing Area ReSTIR pass even when the pass is temporarily disabled"
        );
    }

    #[test]
    fn scene_key_tracks_vpt_debug_view_changes() {
        let final_key = VptRuntimePipeline::make_scene_key(
            glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
            glam::Vec3::new(2.0, 1.5, 1.25),
            LightingSettings::default(),
            RestirDiSettings::default(),
            AreaRestirSettings::default(),
            false,
            false,
        );
        let debug_key = VptRuntimePipeline::make_scene_key(
            glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
            glam::Vec3::new(2.0, 1.5, 1.25),
            LightingSettings {
                vpt_debug_view: VptDebugView::ReservoirWeight,
                ..LightingSettings::default()
            },
            RestirDiSettings::default(),
            AreaRestirSettings::default(),
            false,
            false,
        );

        assert_ne!(
            final_key, debug_key,
            "switching VPT debug views must invalidate accumulated history from the previous output mode"
        );
    }

    #[test]
    fn scene_key_tracks_restir_di_runtime_tuning() {
        let base = VptRuntimePipeline::make_scene_key(
            glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
            glam::Vec3::new(2.0, 1.5, 1.25),
            LightingSettings::default(),
            RestirDiSettings {
                enabled: true,
                ..RestirDiSettings::default()
            },
            AreaRestirSettings::default(),
            true,
            false,
        );

        for changed_restir in [
            RestirDiSettings {
                enabled: true,
                temporal_enabled: false,
                ..RestirDiSettings::default()
            },
            RestirDiSettings {
                enabled: true,
                spatial_enabled: true,
                ..RestirDiSettings::default()
            },
            RestirDiSettings {
                enabled: true,
                initial_candidate_count: 8,
                ..RestirDiSettings::default()
            },
            RestirDiSettings {
                enabled: true,
                spatial_sample_count: 7,
                ..RestirDiSettings::default()
            },
            RestirDiSettings {
                enabled: true,
                history_length: 32,
                ..RestirDiSettings::default()
            },
            RestirDiSettings {
                enabled: true,
                debug_view: RestirDiDebugView::ReservoirWeight,
                ..RestirDiSettings::default()
            },
        ] {
            let changed = VptRuntimePipeline::make_scene_key(
                glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
                glam::Vec3::new(2.0, 1.5, 1.25),
                LightingSettings::default(),
                changed_restir,
                AreaRestirSettings::default(),
                true,
                false,
            );
            assert_ne!(
                base, changed,
                "ReSTIR-DI UI tuning {changed_restir:?} must be part of the VPT scene key"
            );
        }
    }

    #[test]
    fn scene_key_tracks_area_restir_runtime_tuning() {
        let base = VptRuntimePipeline::make_scene_key(
            glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
            glam::Vec3::new(2.0, 1.5, 1.25),
            LightingSettings::default(),
            RestirDiSettings::default(),
            AreaRestirSettings {
                enabled: true,
                ..AreaRestirSettings::default()
            },
            false,
            true,
        );

        for changed_area in [
            AreaRestirSettings {
                enabled: true,
                temporal_enabled: false,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                spatial_enabled: false,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                subpixel_enabled: false,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                lens_enabled: false,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                initial_candidate_count: 8,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                spatial_sample_count: 8,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                history_length: 32,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                normal_threshold: 0.5,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                depth_threshold: 0.08,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                spatial_radius: 48.0,
                ..AreaRestirSettings::default()
            },
            AreaRestirSettings {
                enabled: true,
                debug_view: AreaRestirDebugView::Weight,
                ..AreaRestirSettings::default()
            },
        ] {
            let changed = VptRuntimePipeline::make_scene_key(
                glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
                glam::Vec3::new(2.0, 1.5, 1.25),
                LightingSettings::default(),
                RestirDiSettings::default(),
                changed_area,
                false,
                true,
            );
            assert_ne!(
                base, changed,
                "Area ReSTIR UI tuning {changed_area:?} must be part of the VPT scene key"
            );
        }
    }

    #[test]
    fn vpt_pipeline_exposes_frame_resource_presence_for_runtime_resize_routing() {
        let source = crate::render::source_checks::read_source("src/render/vpt_pipeline.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("VPT pipeline implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "pubfnhas_frame_resources(&self)->bool",
            "self.postprocess_pass.is_some()",
            "self.vpt_surface_pass.is_some()",
            "self.vpt_nrd_confidence_pass.is_some()",
            "self.vpt_pass.is_some()",
            "self.vpt_nrd_frontend_pass.is_some()",
            "self.vpt_nrd_adapter_pass.is_some()",
            "self.vpt_nrd_resolve_pass.is_some()",
            "self.vpt_temporal_pass.is_some()",
            "self.vpt_atrous_pass.is_some()",
            "self.area_restir_pass.is_some()",
            "self.restir_di_pass.is_some()",
        ] {
            assert!(
                compact.contains(token),
                "VPT frame-resource helper missing {token}"
            );
        }
    }
}
