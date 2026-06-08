use anyhow::Result;
use winit::window::Window;

use crate::render::area_restir::AreaRestirSettings;
use crate::render::capture::RenderCapture;
use crate::render::device::RenderDevice;
#[cfg(not(target_os = "android"))]
use crate::render::egui_renderer::{EguiFrame, EguiRenderer};
use crate::render::gpu_profiler::{GpuProfiler, GpuProfilerConfig};
use crate::render::restir_di::RestirDiSettings;
use crate::render::scene_ubo::{LightingSettings, SceneUniformBuffer};
use crate::render::vpt_pipeline::{
    UcvhFrameChanges, VptCameraFrame, VptFrameInputs, VptRuntimePipeline,
};
use crate::voxel::gpu_upload::UcvhGpuResources;
use crate::voxel::ucvh::Ucvh;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeSettings {
    pub lighting: LightingSettings,
    pub restir_di: RestirDiSettings,
    pub area_restir: AreaRestirSettings,
}

pub struct RenderFrameInput<'a> {
    pub camera: VptCameraFrame,
    pub sun_direction: glam::Vec3,
    pub sun_intensity: glam::Vec3,
    pub elapsed_seconds: f32,
    pub settings: RuntimeSettings,
    pub restir_di_enabled: bool,
    pub area_restir_enabled: bool,
    pub ucvh: Option<&'a mut Ucvh>,
    #[cfg(not(target_os = "android"))]
    pub egui_frame: Option<EguiFrame>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderFrameOutcome {
    pub began_frame: bool,
    pub rendered: bool,
    pub uploaded_ucvh: bool,
    pub uploaded_motion_events: u32,
    pub wrote_capture: bool,
}

pub struct RenderRuntime {
    renderer: RenderDevice,
    gpu_profiler: Option<GpuProfiler>,
    capture: Option<RenderCapture>,
    scene_ubo: Option<SceneUniformBuffer>,
    ucvh_gpu: Option<UcvhGpuResources>,
    ucvh_uploaded: bool,
    vpt_pipeline: VptRuntimePipeline,
    #[cfg(not(target_os = "android"))]
    egui_renderer: Option<EguiRenderer>,
}

impl RenderRuntime {
    pub fn new(window: &Window, settings: RuntimeSettings, ucvh: Option<&Ucvh>) -> Result<Self> {
        let renderer = RenderDevice::new(window)?;

        tracing::info!(
            renderer = %renderer.backend_name(),
            physical_device = %renderer.physical_device_name(),
            graphics_queue_family = renderer.graphics_queue_family_index(),
            present_queue_family = renderer.present_queue_family_index(),
            swapchain_format = ?renderer.swapchain_format(),
            swapchain_extent = ?renderer.swapchain_extent(),
            swapchain_images = renderer.swapchain_image_count(),
            surface = ?renderer.surface(),
            "initialized renderer bootstrap"
        );

        let gpu_profiler = match GpuProfiler::new(
            renderer.device(),
            renderer
                .physical_device_properties()
                .limits
                .timestamp_period,
            renderer.graphics_queue_timestamp_valid_bits(),
            renderer.frame_slot_count(),
            GpuProfilerConfig::from_env(),
        ) {
            Ok(profiler) => profiler,
            Err(error) => {
                tracing::warn!(%error, "failed to initialize GPU profiler; continuing without profiling");
                None
            }
        };

        let capture = match RenderCapture::from_env() {
            Ok(capture) => {
                if let Some(capture) = &capture {
                    tracing::info!(
                        target_frame = ?capture.config().target_frame,
                        output_dir = %capture.config().output_dir.display(),
                        prefix = %capture.config().prefix,
                        "enabled postprocess capture"
                    );
                }
                capture
            }
            Err(error) => {
                tracing::warn!(%error, "invalid postprocess capture configuration; capture disabled");
                None
            }
        };

        let scene_ubo = match SceneUniformBuffer::new(
            renderer.device(),
            renderer.allocator(),
            renderer.swapchain_image_count(),
        ) {
            Ok(ubo) => {
                tracing::info!(
                    frame_count = renderer.swapchain_image_count(),
                    "created scene UBO"
                );
                Some(ubo)
            }
            Err(error) => {
                tracing::error!(%error, "failed to create scene UBO");
                None
            }
        };

        let ucvh_gpu = match ucvh {
            Some(ucvh) => {
                match UcvhGpuResources::new(renderer.device(), renderer.allocator(), ucvh) {
                    Ok(gpu) => {
                        tracing::info!("created UCVH GPU resources");
                        Some(gpu)
                    }
                    Err(error) => {
                        tracing::error!(%error, "failed to create UCVH GPU resources");
                        None
                    }
                }
            }
            None => None,
        };

        #[cfg(not(target_os = "android"))]
        let egui_renderer = match EguiRenderer::new(&renderer) {
            Ok(renderer) => Some(renderer),
            Err(error) => {
                tracing::warn!(%error, "failed to initialize egui renderer; continuing without editor overlay");
                None
            }
        };

        let mut runtime = Self {
            renderer,
            gpu_profiler,
            capture,
            scene_ubo,
            ucvh_gpu,
            ucvh_uploaded: false,
            vpt_pipeline: VptRuntimePipeline::new(),
            #[cfg(not(target_os = "android"))]
            egui_renderer,
        };
        runtime.ensure_passes(
            ucvh,
            settings,
            settings.restir_di.enabled,
            settings.area_restir.enabled,
        );
        Ok(runtime)
    }

    pub fn device(&self) -> &RenderDevice {
        &self.renderer
    }

    pub fn ensure_passes(
        &mut self,
        ucvh: Option<&Ucvh>,
        settings: RuntimeSettings,
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) {
        if let Some(scene_ubo) = self.scene_ubo.as_ref() {
            self.vpt_pipeline.ensure_passes(
                &self.renderer,
                scene_ubo,
                ucvh,
                self.ucvh_gpu.as_ref(),
                settings.lighting,
                restir_di_enabled,
                area_restir_enabled,
            );
        }
    }

    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
        ucvh: Option<&Ucvh>,
        settings: RuntimeSettings,
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) -> Result<()> {
        self.renderer.handle_resize(width, height)?;
        self.resize_pipeline_to_swapchain(ucvh, settings, restir_di_enabled, area_restir_enabled)
    }

    fn resize_pipeline_to_swapchain(
        &mut self,
        ucvh: Option<&Ucvh>,
        settings: RuntimeSettings,
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) -> Result<()> {
        self.ensure_passes(ucvh, settings, restir_di_enabled, area_restir_enabled);
        let extent = self.renderer.swapchain_extent();
        if let (Some(scene_ubo), Some(ucvh_gpu)) = (self.scene_ubo.as_ref(), self.ucvh_gpu.as_ref())
        {
            self.vpt_pipeline.resize(
                &self.renderer,
                scene_ubo,
                ucvh_gpu,
                extent.width,
                extent.height,
                settings.lighting,
                restir_di_enabled,
                area_restir_enabled,
            )?;
        }
        Ok(())
    }

    pub fn render_frame(&mut self, mut input: RenderFrameInput<'_>) -> Result<RenderFrameOutcome> {
        let mut outcome = RenderFrameOutcome::default();
        let frame = self.renderer.begin_frame()?;
        outcome.began_frame = true;

        if frame.swapchain_recreated {
            self.resize_pipeline_to_swapchain(
                input.ucvh.as_deref(),
                input.settings,
                input.restir_di_enabled,
                input.area_restir_enabled,
            )?;
        }
        if !frame.should_render {
            return Ok(outcome);
        }

        if let Some(profiler) = &mut self.gpu_profiler {
            profiler.begin_frame(
                self.renderer.device(),
                frame.command_buffer,
                frame.frame_slot,
                frame.frame_index,
            );
        }

        if !self.ucvh_uploaded {
            if let (Some(ucvh), Some(gpu)) = (input.ucvh.as_deref_mut(), &self.ucvh_gpu) {
                match gpu.upload_all(self.renderer.device(), frame.command_buffer, ucvh) {
                    Ok(()) => {
                        self.ucvh_uploaded = true;
                        outcome.uploaded_ucvh = true;
                        tracing::info!("uploaded UCVH data to GPU");
                    }
                    Err(error) => {
                        tracing::error!(%error, "failed to upload UCVH data to GPU");
                    }
                }
            }
        }

        let ucvh_frame_changes = if self.ucvh_uploaded {
            input
                .ucvh
                .as_deref()
                .map(Self::snapshot_ucvh_frame_changes)
                .unwrap_or_default()
        } else {
            UcvhFrameChanges::default()
        };

        let mut ucvh_motion_event_count = 0u32;
        if self.ucvh_uploaded
            && let (Some(ucvh), Some(gpu)) = (input.ucvh.as_deref(), &self.ucvh_gpu)
        {
            match gpu.upload_motion_guide(
                self.renderer.device(),
                frame.command_buffer,
                ucvh,
                &ucvh_frame_changes.motion_events,
            ) {
                Ok(count) => {
                    ucvh_motion_event_count = count;
                    outcome.uploaded_motion_events = count;
                }
                Err(error) => tracing::error!(%error, "failed to upload UCVH motion guide"),
            }
        }

        let Some(scene_ubo) = self.scene_ubo.as_ref() else {
            tracing::warn!("skipping render frame until scene UBO is initialized");
            let completion = self.renderer.end_frame(frame)?;
            if completion.swapchain_recreated {
                self.resize_pipeline_to_swapchain(
                    input.ucvh.as_deref(),
                    input.settings,
                    input.restir_di_enabled,
                    input.area_restir_enabled,
                )?;
            }
            return Ok(outcome);
        };

        let record_result = self.vpt_pipeline.record_and_execute_frame(
            &self.renderer,
            &frame,
            #[cfg(not(target_os = "android"))]
            self.egui_renderer.as_mut(),
            #[cfg(not(target_os = "android"))]
            input.egui_frame.as_ref(),
            VptFrameInputs {
                scene_ubo,
                camera: input.camera,
                sun_direction: input.sun_direction,
                sun_intensity: input.sun_intensity,
                elapsed_seconds: input.elapsed_seconds,
                lighting_settings: input.settings.lighting,
                restir_di_settings: input.settings.restir_di,
                area_restir_settings: input.settings.area_restir,
                restir_di_enabled: input.restir_di_enabled,
                area_restir_enabled: input.area_restir_enabled,
                ucvh_ready: self.ucvh_uploaded,
                ucvh_frame_changes,
                ucvh_motion_event_count,
                capture: self.capture.as_mut(),
                profiler: self.gpu_profiler.as_ref(),
            },
        )?;
        let submitted_fence = record_result.submitted_fence;
        let mut pending_capture = record_result.pending_capture;
        outcome.rendered = true;
        let completion = self.renderer.end_frame(frame)?;
        if completion.swapchain_recreated {
            self.resize_pipeline_to_swapchain(
                input.ucvh.as_deref(),
                input.settings,
                input.restir_di_enabled,
                input.area_restir_enabled,
            )?;
        }

        if self.ucvh_uploaded
            && let Some(ucvh) = input.ucvh.as_deref_mut()
        {
            Self::clear_ucvh_frame_changes(ucvh);
        }
        if let Some(metadata) = pending_capture.take() {
            self.renderer.wait_for_fence(submitted_fence)?;
            if let Some(capture) = &self.capture {
                capture.write_rgba8_capture(&metadata)?;
                outcome.wrote_capture = true;
                tracing::info!(
                    frame_index = metadata.frame_index,
                    ppm = %metadata.ppm_path.display(),
                    json = %metadata.json_path.display(),
                    "wrote postprocess capture"
                );
            }
        }

        Ok(outcome)
    }

    fn snapshot_ucvh_frame_changes(ucvh: &Ucvh) -> UcvhFrameChanges {
        UcvhFrameChanges::new(
            ucvh.invalidation_regions().to_vec(),
            ucvh.motion_events().to_vec(),
        )
    }

    fn clear_ucvh_frame_changes(ucvh: &mut Ucvh) {
        let _ = ucvh.take_invalidation_regions();
        let _ = ucvh.take_motion_events();
    }
}

impl Drop for RenderRuntime {
    fn drop(&mut self) {
        unsafe { self.renderer.device().device_wait_idle().ok() };
        if let Some(profiler) = self.gpu_profiler.take() {
            profiler.destroy(self.renderer.device());
        }
        if let Some(capture) = self.capture.take() {
            capture.destroy(self.renderer.device(), self.renderer.allocator());
        }
        #[cfg(not(target_os = "android"))]
        if let Some(egui_renderer) = self.egui_renderer.take() {
            egui_renderer.destroy(self.renderer.device(), self.renderer.allocator());
        }
        let vpt_pipeline = std::mem::take(&mut self.vpt_pipeline);
        vpt_pipeline.destroy(self.renderer.device(), self.renderer.allocator());
        if let Some(gpu) = self.ucvh_gpu.take() {
            gpu.destroy(self.renderer.device(), self.renderer.allocator());
        }
        if let Some(scene_ubo) = self.scene_ubo.take() {
            scene_ubo.destroy(self.renderer.device(), self.renderer.allocator());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::brick::VoxelCell;
    use crate::voxel::ucvh::{UcvhConfig, UcvhMotionEvent};

    #[test]
    fn snapshotting_ucvh_frame_changes_returns_render_visible_change_summary_without_consuming() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(glam::UVec3::splat(32)));
        assert!(ucvh.set_voxel(glam::UVec3::new(1, 2, 3), VoxelCell::new(1, 0, [0; 3])));
        assert!(ucvh.push_motion_event(UcvhMotionEvent {
            region_min: glam::UVec3::new(8, 8, 8),
            region_max_exclusive: glam::UVec3::new(16, 16, 16),
            world_delta_current_from_previous: glam::IVec3::new(1, 0, 0),
            generation: 2,
        }));

        let changes = RenderRuntime::snapshot_ucvh_frame_changes(&ucvh);

        assert_eq!(changes.invalidation_regions.len(), 1);
        assert_eq!(changes.motion_events.len(), 1);
        assert_eq!(ucvh.invalidation_regions().len(), 1);
        assert_eq!(ucvh.motion_events().len(), 1);
    }

    #[test]
    fn clearing_ucvh_frame_changes_discards_initial_generation_metadata() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(glam::UVec3::splat(32)));
        assert!(ucvh.set_voxel(glam::UVec3::new(1, 2, 3), VoxelCell::new(1, 0, [0; 3])));

        RenderRuntime::clear_ucvh_frame_changes(&mut ucvh);

        assert!(ucvh.invalidation_regions().is_empty());
        assert!(ucvh.motion_events().is_empty());
    }

    #[test]
    fn app_delegates_gpu_runtime_ownership_to_render_runtime() {
        let source = crate::render::source_checks::read_source("src/app.rs");
        let app_struct = source
            .split("struct RevolumetricApp")
            .nth(1)
            .expect("RevolumetricApp struct should exist")
            .split("impl RevolumetricApp")
            .next()
            .expect("RevolumetricApp struct should end before impl");

        assert!(app_struct.contains("render_runtime: Option<RenderRuntime>"));
        for forbidden in [
            "renderer: Option<RenderDevice>",
            "gpu_profiler: Option<GpuProfiler>",
            "capture: Option<RenderCapture>",
            "vpt_pipeline: VptRuntimePipeline",
            "ucvh_gpu: Option<UcvhGpuResources>",
            "ucvh_uploaded: bool",
            "scene_ubo: Option<SceneUniformBuffer>",
        ] {
            assert!(
                !app_struct.contains(forbidden),
                "app.rs should not own GPU runtime field {forbidden}"
            );
        }

        for forbidden in [
            "record_and_execute_frame(",
            ".begin_frame(",
            ".end_frame(",
            ".upload_all(",
            ".upload_motion_guide(",
            ".handle_resize(",
        ] {
            assert!(
                !source.contains(forbidden),
                "app.rs should delegate GPU runtime call {forbidden}"
            );
        }
        assert!(source.contains(".render_frame("));
        assert!(source.contains(".resize("));
    }

    #[test]
    fn render_runtime_owns_gpu_resources_and_frame_orchestration() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let runtime_struct = source
            .split("pub struct RenderRuntime")
            .nth(1)
            .expect("RenderRuntime struct should exist")
            .split("impl RenderRuntime")
            .next()
            .expect("RenderRuntime struct should end before impl");

        for token in [
            "renderer: RenderDevice",
            "gpu_profiler: Option<GpuProfiler>",
            "capture: Option<RenderCapture>",
            "scene_ubo: Option<SceneUniformBuffer>",
            "ucvh_gpu: Option<UcvhGpuResources>",
            "ucvh_uploaded: bool",
            "vpt_pipeline: VptRuntimePipeline",
            "egui_renderer: Option<EguiRenderer>",
        ] {
            assert!(
                runtime_struct.contains(token),
                "RenderRuntime must own GPU runtime field {token}"
            );
        }

        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before UCVH helpers");
        for token in [
            ".begin_frame(",
            ".upload_all(",
            ".upload_motion_guide(",
            ".record_and_execute_frame(",
            ".end_frame(",
            ".wait_for_fence(",
            ".write_rgba8_capture(",
        ] {
            assert!(
                render_frame.contains(token),
                "RenderRuntime::render_frame must own frame orchestration call {token}"
            );
        }

        let runtime_drop = source
            .split("impl Drop for RenderRuntime")
            .nth(1)
            .expect("RenderRuntime Drop impl should exist")
            .split("#[cfg(test)]")
            .next()
            .expect("Drop impl should end before tests");
        for token in [
            ".device_wait_idle()",
            "profiler.destroy(",
            "capture.destroy(",
            "vpt_pipeline.destroy(",
            "gpu.destroy(",
            "scene_ubo.destroy(",
        ] {
            assert!(
                runtime_drop.contains(token),
                "RenderRuntime Drop must destroy GPU resource with {token}"
            );
        }
    }

    #[test]
    fn render_runtime_observes_swapchain_recreate_signals() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let runtime_impl = source
            .split("impl RenderRuntime")
            .nth(1)
            .expect("RenderRuntime impl should exist");

        assert!(runtime_impl.contains("fn resize_pipeline_to_swapchain("));
        assert!(runtime_impl.contains("self.renderer.swapchain_extent()"));
        assert!(runtime_impl.contains("if frame.swapchain_recreated"));
        assert!(runtime_impl.contains("if completion.swapchain_recreated"));
    }
}
