use anyhow::{Context, Result};

use crate::render::allocator::GpuAllocator;
use crate::render::device::RenderDevice;
use crate::render::passes::area_restir::{AreaRestirPass, AreaRestirPassCreateInfo};
use crate::render::passes::postprocess::PostprocessPass;
use crate::render::passes::restir_di::{RestirDiPass, RestirDiPassCreateInfo};
use crate::render::passes::vpt::VptPass;
use crate::render::passes::vpt_atrous::{
    VptAtrousPass, VptAtrousPassCreateInfo, VptAtrousPassResizeInfo,
};
use crate::render::passes::vpt_surface::VptSurfacePass;
use crate::render::passes::vpt_temporal::{
    VptTemporalPass, VptTemporalPassCreateInfo, VptTemporalPassResizeInfo,
};
use crate::render::restir_di::build_direct_lights_from_ucvh;
use crate::render::scene_ubo::SceneUniformBuffer;
use crate::voxel::gpu_upload::UcvhGpuResources;
use crate::voxel::ucvh::Ucvh;

pub struct VptPipelineFrameState {
    pub vpt_sample_index: u32,
    pub last_vpt_camera_key: Option<[u32; 15]>,
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
        self.vpt_accumulation_needs_init = true;
        self.vpt_temporal_history_initialized = false;
        self.postprocess_output_initialized = false;
        self.area_restir_history_initialized = false;
        self.restir_di_history_initialized = false;
        self.previous_vpt_view_proj = None;
        self.previous_vpt_resolution = None;
    }
}

pub struct VptRuntimePipeline {
    pub postprocess_pass: Option<PostprocessPass>,
    pub vpt_surface_pass: Option<VptSurfacePass>,
    pub vpt_pass: Option<VptPass>,
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
    pub fn new() -> Self {
        Self {
            postprocess_pass: None,
            vpt_surface_pass: None,
            vpt_pass: None,
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
        self.ensure_vpt_pass(renderer, scene_ubo, ucvh_gpu);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_state_reset_clears_history_and_accumulation() {
        let mut state = VptPipelineFrameState {
            vpt_sample_index: 7,
            last_vpt_camera_key: Some([1; 15]),
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
        assert!(state.vpt_accumulation_needs_init);
        assert!(!state.vpt_temporal_history_initialized);
        assert!(!state.postprocess_output_initialized);
        assert!(!state.area_restir_history_initialized);
        assert!(!state.restir_di_history_initialized);
        assert_eq!(state.previous_vpt_view_proj, None);
        assert_eq!(state.previous_vpt_resolution, None);
    }
}
