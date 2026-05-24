use anyhow::{Context, Result};
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::area_restir::AreaRestirSettings;
use crate::render::camera::{compute_pixel_to_ray, compute_view_proj};
use crate::render::capture::{CaptureMetadata, RenderCapture, cmd_copy_image_to_buffer};
use crate::render::device::RenderDevice;
use crate::render::frame::FrameContext;
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::passes::area_restir::{AreaRestirPass, AreaRestirPassCreateInfo};
use crate::render::passes::blit_to_swapchain;
use crate::render::passes::postprocess::{PostprocessGraphInputs, PostprocessPass};
use crate::render::passes::restir_di::{RestirDiPass, RestirDiPassCreateInfo};
use crate::render::passes::vpt::VptPass;
use crate::render::passes::vpt_atrous::{
    VptAtrousGraphInputs, VptAtrousPass, VptAtrousPassCreateInfo, VptAtrousPassResizeInfo,
};
use crate::render::passes::vpt_nrd_adapter::{
    VptNrdAdapterGraphInputs, VptNrdAdapterPass, VptNrdAdapterPassCreateInfo,
    VptNrdAdapterPassImageRefs, VptNrdAdapterPassResizeInfo,
};
use crate::render::passes::vpt_nrd_confidence::{
    VptNrdConfidenceGraphInputs, VptNrdConfidencePass, VptNrdConfidencePassCreateInfo,
    VptNrdConfidencePassResizeInfo,
};
use crate::render::passes::vpt_nrd_frontend::{
    VptNrdFrontendGraphInputs, VptNrdFrontendPass, VptNrdFrontendPassCreateInfo,
    VptNrdFrontendPassResizeInfo,
};
use crate::render::passes::vpt_surface::VptSurfacePass;
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
use crate::render::vpt_history::{
    GpuVptHistoryUniforms, VPT_HISTORY_FLAG_CAMERA_CUT, VPT_HISTORY_FLAG_LIGHTS_INVALIDATED,
    VPT_HISTORY_FLAG_RESIZE, VPT_HISTORY_FLAG_SCENE_INVALIDATED,
};
use crate::voxel::gpu_upload::UcvhGpuResources;
use crate::voxel::ucvh::{Ucvh, UcvhInvalidationRegion, UcvhMotionEvent};

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
}

pub struct VptPipelineFrameState {
    pub vpt_sample_index: u32,
    pub last_vpt_camera_key: Option<[u32; 15]>,
    pub last_vpt_scene_key: Option<[u32; 14]>,
    pub history_reset_generation: u32,
    pub vpt_accumulation_needs_init: bool,
    pub vpt_temporal_history_initialized: bool,
    pub postprocess_output_initialized: bool,
    pub area_restir_history_initialized: bool,
    pub restir_di_history_initialized: bool,
    pub previous_vpt_view_proj: Option<glam::Mat4>,
    pub previous_vpt_resolution: Option<[u32; 2]>,
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
            area_restir_history_initialized: false,
            restir_di_history_initialized: false,
            previous_vpt_view_proj: None,
            previous_vpt_resolution: None,
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
        self.area_restir_history_initialized = false;
        self.restir_di_history_initialized = false;
        self.previous_vpt_view_proj = None;
        self.previous_vpt_resolution = None;
    }

    pub fn reset_for_scene_change(&mut self) {
        self.vpt_sample_index = 0;
        self.last_vpt_camera_key = None;
        self.last_vpt_scene_key = None;
        self.history_reset_generation = self.history_reset_generation.wrapping_add(1);
        self.vpt_accumulation_needs_init = true;
        self.vpt_temporal_history_initialized = false;
        self.postprocess_output_initialized = false;
        self.area_restir_history_initialized = false;
        self.restir_di_history_initialized = false;
    }
}

pub struct VptRuntimePipeline {
    pub postprocess_pass: Option<PostprocessPass>,
    pub vpt_surface_pass: Option<VptSurfacePass>,
    pub vpt_nrd_confidence_pass: Option<VptNrdConfidencePass>,
    pub vpt_pass: Option<VptPass>,
    pub vpt_nrd_frontend_pass: Option<VptNrdFrontendPass>,
    pub vpt_nrd_adapter_pass: Option<VptNrdAdapterPass>,
    pub vpt_temporal_pass: Option<VptTemporalPass>,
    pub vpt_atrous_pass: Option<VptAtrousPass>,
    pub area_restir_pass: Option<AreaRestirPass>,
    pub restir_di_pass: Option<RestirDiPass>,
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
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) -> [u32; 14] {
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
            restir_di_enabled as u32,
            area_restir_enabled as u32,
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
            vpt_temporal_pass: None,
            vpt_atrous_pass: None,
            area_restir_pass: None,
            restir_di_pass: None,
            frame_state: VptPipelineFrameState::default(),
        }
    }

    pub fn ensure_passes(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh: Option<&Ucvh>,
        ucvh_gpu: Option<&UcvhGpuResources>,
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) {
        self.ensure_vpt_surface_pass(renderer, scene_ubo, ucvh_gpu);
        self.ensure_vpt_nrd_confidence_pass(renderer, scene_ubo);
        self.ensure_vpt_pass(renderer, scene_ubo, ucvh_gpu);
        self.ensure_vpt_nrd_frontend_pass(renderer, scene_ubo);
        self.ensure_vpt_nrd_adapter_pass(renderer, scene_ubo);
        self.ensure_restir_di_pass(renderer, scene_ubo, ucvh, restir_di_enabled);
        self.ensure_area_restir_pass(renderer, scene_ubo, ucvh_gpu, area_restir_enabled);
        self.ensure_vpt_temporal_pass(renderer, scene_ubo);
        self.ensure_vpt_atrous_pass(renderer, scene_ubo);
        self.ensure_postprocess_pass(renderer, scene_ubo);
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
            extent.width,
            extent.height,
            spirv,
            ucvh_gpu,
            scene_ubo,
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
            extent.width,
            extent.height,
            spirv,
            ucvh_gpu,
            scene_ubo,
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
    ) {
        if self.vpt_nrd_adapter_pass.is_some() {
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
        mut inputs: VptFrameInputs<'_>,
    ) -> Result<VptFrameRecordResult> {
        let mut graph = RenderGraph::new();
        let mut pending_capture = None;
        let mut rendered_vpt = false;
        let mut vpt_accumulation_written = false;
        let mut restir_di_selected_written = false;
        let mut area_restir_selected_written = false;
        let mut current_vpt_view_proj = None;
        let scene_key = Self::make_scene_key(
            inputs.sun_direction,
            inputs.sun_intensity,
            inputs.lighting_settings,
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
            let camera_key = [
                camera.position.x.to_bits(),
                camera.position.y.to_bits(),
                camera.position.z.to_bits(),
                camera.forward.x.to_bits(),
                camera.forward.y.to_bits(),
                camera.forward.z.to_bits(),
                camera.up.x.to_bits(),
                camera.up.y.to_bits(),
                camera.up.z.to_bits(),
                camera.fov_y_radians.to_bits(),
                camera.aperture_radius.to_bits(),
                camera.focal_distance.to_bits(),
                frame.swapchain_extent.width,
                frame.swapchain_extent.height,
                inputs.lighting_settings.vpt_max_bounces,
            ];
            if self.frame_state.last_vpt_camera_key == Some(camera_key) {
                self.frame_state.vpt_sample_index =
                    self.frame_state.vpt_sample_index.saturating_add(1);
            } else {
                self.frame_state.vpt_sample_index = 0;
                self.frame_state.last_vpt_camera_key = Some(camera_key);
            }
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
                let surface_graph = vpt_surface.register_bootstrap_graph(
                    &mut graph,
                    self.frame_state.vpt_temporal_history_initialized,
                    slot,
                    profiler,
                );
                let bootstrap_surface_resources = surface_graph.surface_writes;
                let mut final_surface_writes = bootstrap_surface_resources;
                let previous_surface_resources = surface_graph.previous_surface_resources;
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
                        inputs.profiler,
                    );
                    final_surface_writes = area_graph.final_surface_writes;
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
                    slot,
                    self.frame_state.vpt_accumulation_needs_init,
                    vpt_restir_reads,
                    vpt_area_restir_reads,
                    profiler,
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
                                profiler,
                            },
                        )
                    })
                } else {
                    None
                };
                let nrd_adapter_outputs = if matches!(
                    inputs.lighting_settings.denoiser_mode,
                    VptDenoiserMode::Relax
                ) {
                    match (
                        self.vpt_nrd_adapter_pass.as_ref(),
                        nrd_frontend_outputs,
                        nrd_confidence_outputs,
                    ) {
                        (
                            Some(vpt_nrd_adapter),
                            Some(nrd_frontend_outputs),
                            Some(nrd_confidence_outputs),
                        ) => Some(vpt_nrd_adapter.register_graph(
                            &mut graph,
                            VptNrdAdapterGraphInputs {
                                frame_slot: slot,
                                packed: nrd_frontend_outputs.packed,
                                confidence: nrd_confidence_outputs.confidence,
                                surface_inputs: final_surface_writes,
                                profiler,
                            },
                        )),
                        _ => None,
                    }
                } else {
                    None
                };
                let _ = nrd_adapter_outputs.as_ref().map(|outputs| {
                    (
                        outputs.resources.diff_radiance_hitdist,
                        outputs.resources.validation,
                    )
                });

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
                let postprocess_outputs = postprocess.register_graph(
                    &mut graph,
                    PostprocessGraphInputs {
                        device: renderer.device(),
                        frame_slot: frame.frame_slot,
                        input_radiance: atrous_filtered_dep,
                        hdr_image: vpt_atrous.output_image(),
                        output_initialized: self.frame_state.postprocess_output_initialized,
                        profiler,
                    },
                );
                let src_image = postprocess_output;
                let src_extent = postprocess_extent;
                let dst_image = frame.swapchain_image;
                let dst_extent = frame.swapchain_extent;
                let dep_handle = postprocess_outputs.output;
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
                        effective_denoiser_mode: inputs
                            .lighting_settings
                            .effective_denoiser_mode_name(),
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
                    vk::ImageUsageFlags::TRANSFER_DST,
                    swapchain_access_from_layout(frame.swapchain_image_layout)?,
                );
                graph.add_pass("blit_to_swapchain", QueueType::Graphics, |builder| {
                    builder.read_as(dep_handle, AccessKind::TransferRead);
                    if let Some(capture_dep) = capture_dependency {
                        builder.depend_on(capture_dep);
                    }
                    builder.write_as(swapchain_dep, AccessKind::TransferWrite);
                    builder.finish_as(swapchain_dep, AccessKind::Present);
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
            } else {
                self.frame_state.vpt_sample_index = 0;
                self.frame_state.last_vpt_camera_key = None;
                tracing::warn!(
                    vpt_ready = self.vpt_pass.is_some(),
                    vpt_nrd_confidence_ready = self.vpt_nrd_confidence_pass.is_some(),
                    vpt_nrd_frontend_ready = self.vpt_nrd_frontend_pass.is_some(),
                    vpt_nrd_adapter_ready = self.vpt_nrd_adapter_pass.is_some(),
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
        graph.compile()?;
        graph.execute(renderer.device(), frame.command_buffer, frame.frame_index);

        if let Some(current_view_proj) = current_vpt_view_proj {
            self.frame_state.previous_vpt_view_proj = Some(current_view_proj);
            self.frame_state.previous_vpt_resolution =
                Some([frame.swapchain_extent.width, frame.swapchain_extent.height]);
        }
        if vpt_accumulation_written {
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
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) -> Result<()> {
        let device = renderer.device().clone();
        let allocator = renderer.allocator();
        self.frame_state.reset_for_resize_or_camera_cut();

        if let Some(vpt) = &mut self.vpt_pass {
            vpt.resize_images(&device, allocator, width, height, scene_ubo, ucvh_gpu)
                .context("failed to resize VPT images")?;
        }
        if let (Some(vpt_nrd_frontend), Some(vpt)) =
            (&mut self.vpt_nrd_frontend_pass, &self.vpt_pass)
        {
            vpt_nrd_frontend
                .resize_images(
                    &device,
                    allocator,
                    VptNrdFrontendPassResizeInfo {
                        width,
                        height,
                        scene_ubo,
                        vpt,
                    },
                )
                .context("failed to resize VPT NRD frontend images")?;
        }
        if let Some(vpt_surface) = &mut self.vpt_surface_pass {
            vpt_surface
                .resize_images(&device, allocator, width, height, scene_ubo, ucvh_gpu)
                .context("failed to resize VPT surface images")?;
            if restir_di_enabled && let Some(restir_di) = &self.restir_di_pass {
                restir_di.update_surface_descriptors(&device, vpt_surface);
            }
            if area_restir_enabled && let Some(area_restir) = &self.area_restir_pass {
                area_restir.update_surface_descriptors(&device, vpt_surface);
            }
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
        if restir_di_enabled && let Some(restir_di) = &mut self.restir_di_pass {
            restir_di
                .resize_buffers(&device, allocator, width, height)
                .context("failed to resize ReSTIR-DI buffers")?;
        }
        if area_restir_enabled && let Some(area_restir) = &mut self.area_restir_pass {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::scene_ubo::{
        LightingDebugView, LightingSettings, RenderMode, VptDebugView, VptDenoiserMode,
    };
    use crate::voxel::ucvh::{UcvhInvalidationRegion, UcvhMotionEvent};

    #[test]
    fn frame_state_reset_clears_history_and_accumulation() {
        let mut state = VptPipelineFrameState {
            vpt_sample_index: 7,
            last_vpt_camera_key: Some([1; 15]),
            last_vpt_scene_key: Some([2; 14]),
            history_reset_generation: 9,
            vpt_accumulation_needs_init: false,
            vpt_temporal_history_initialized: true,
            postprocess_output_initialized: true,
            area_restir_history_initialized: true,
            restir_di_history_initialized: true,
            previous_vpt_view_proj: Some(glam::Mat4::IDENTITY),
            previous_vpt_resolution: Some([1280, 720]),
        };

        state.reset_for_resize_or_camera_cut();

        assert_eq!(state.vpt_sample_index, 0);
        assert_eq!(state.last_vpt_camera_key, None);
        assert_eq!(state.last_vpt_scene_key, Some([2; 14]));
        assert_eq!(state.history_reset_generation, 10);
        assert!(state.vpt_accumulation_needs_init);
        assert!(!state.vpt_temporal_history_initialized);
        assert!(!state.postprocess_output_initialized);
        assert!(!state.area_restir_history_initialized);
        assert!(!state.restir_di_history_initialized);
        assert_eq!(state.previous_vpt_view_proj, None);
        assert_eq!(state.previous_vpt_resolution, None);
    }

    #[test]
    fn frame_state_scene_reset_clears_history_without_touching_camera_history() {
        let mut state = VptPipelineFrameState {
            vpt_sample_index: 11,
            last_vpt_camera_key: Some([1; 15]),
            last_vpt_scene_key: Some([2; 14]),
            history_reset_generation: 3,
            vpt_accumulation_needs_init: false,
            vpt_temporal_history_initialized: true,
            postprocess_output_initialized: true,
            area_restir_history_initialized: true,
            restir_di_history_initialized: true,
            previous_vpt_view_proj: Some(glam::Mat4::IDENTITY),
            previous_vpt_resolution: Some([1280, 720]),
        };

        state.reset_for_scene_change();

        assert_eq!(state.vpt_sample_index, 0);
        assert_eq!(state.last_vpt_camera_key, None);
        assert_eq!(state.last_vpt_scene_key, None);
        assert_eq!(state.history_reset_generation, 4);
        assert!(state.vpt_accumulation_needs_init);
        assert!(!state.vpt_temporal_history_initialized);
        assert!(!state.postprocess_output_initialized);
        assert!(!state.area_restir_history_initialized);
        assert!(!state.restir_di_history_initialized);
        assert_eq!(state.previous_vpt_view_proj, Some(glam::Mat4::IDENTITY));
        assert_eq!(state.previous_vpt_resolution, Some([1280, 720]));
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
            },
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
            },
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
            false,
            false,
        );
        let relax = VptRuntimePipeline::make_scene_key(
            glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
            glam::Vec3::new(2.0, 1.5, 1.25),
            relax_settings,
            false,
            false,
        );

        assert_ne!(base, relax);
    }
}
