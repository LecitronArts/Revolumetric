use anyhow::Result;
use winit::window::Window;

use crate::render::area_restir::AreaRestirSettings;
use crate::render::capture::RenderCapture;
use crate::render::device::RenderDevice;
#[cfg(not(target_os = "android"))]
use crate::render::egui_renderer::{EguiFrame, EguiRenderer};
use crate::render::gpu_profiler::{GpuProfiler, GpuProfilerConfig};
use crate::render::restir_di::RestirDiSettings;
use crate::render::rt_capabilities::{RenderBackend, RtCapabilities, resolve_render_backend};
use crate::render::rt_pipeline::{RtFrameInputs, RtFrameStatus, RtRuntimePipeline};
use crate::render::rt_settings::RtSettings;
use crate::render::scene_ubo::{LightingSettings, RenderMode, SceneUniformBuffer};
use crate::render::vpt_pipeline::{
    UcvhFrameChanges, VptCameraFrame, VptFrameInputs, VptRuntimePipeline,
};
use crate::voxel::gpu_upload::UcvhGpuResources;
use crate::voxel::ucvh::Ucvh;

#[derive(Debug, Clone, Copy)]
pub struct RuntimeSettings {
    pub lighting: LightingSettings,
    pub rt: RtSettings,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderRuntimeStatus {
    pub actual_backend: RenderBackend,
    pub rt_supported: bool,
    pub rt_frame_status: Option<RtFrameStatus>,
}

pub struct RenderRuntime {
    renderer: RenderDevice,
    rt_capabilities: RtCapabilities,
    requested_render_mode: RenderMode,
    render_backend: RenderBackend,
    last_render_backend: RenderBackend,
    rt_history_reset_generation: u32,
    gpu_profiler: Option<GpuProfiler>,
    capture: Option<RenderCapture>,
    scene_ubo: Option<SceneUniformBuffer>,
    ucvh_gpu: Option<UcvhGpuResources>,
    ucvh_uploaded: bool,
    rt_pipeline: RtRuntimePipeline,
    vpt_pipeline: VptRuntimePipeline,
    #[cfg(not(target_os = "android"))]
    egui_renderer: Option<EguiRenderer>,
}

impl RenderRuntime {
    pub fn new(window: &Window, settings: RuntimeSettings, ucvh: Option<&Ucvh>) -> Result<Self> {
        let renderer = RenderDevice::new(window)?;
        let rt_capabilities = renderer.rt_capabilities();
        let render_backend =
            resolve_render_backend(settings.lighting.render_mode, rt_capabilities.supported());

        if settings.lighting.render_mode == RenderMode::Rt && render_backend == RenderBackend::Vpt {
            tracing::warn!(
                device = %renderer.physical_device_name(),
                "requested RT backend but hardware support was unavailable; falling back to VPT"
            );
        }

        tracing::info!(
            renderer = %renderer.backend_name(),
            render_backend = ?render_backend,
            rt_supported = rt_capabilities.supported(),
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
            rt_capabilities,
            requested_render_mode: settings.lighting.render_mode,
            render_backend,
            last_render_backend: render_backend,
            rt_history_reset_generation: 0,
            gpu_profiler,
            capture,
            scene_ubo,
            ucvh_gpu,
            ucvh_uploaded: false,
            rt_pipeline: RtRuntimePipeline::new(),
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

    pub fn render_backend(&self) -> RenderBackend {
        self.render_backend
    }

    pub fn rt_capabilities(&self) -> RtCapabilities {
        self.rt_capabilities
    }

    pub fn status(&self) -> RenderRuntimeStatus {
        RenderRuntimeStatus {
            actual_backend: self.render_backend,
            rt_supported: self.rt_capabilities.supported(),
            rt_frame_status: (self.render_backend == RenderBackend::Rt
                || self.rt_pipeline.has_frame_resources())
            .then(|| self.rt_pipeline.frame_status()),
        }
    }

    fn refresh_render_backend(&mut self, requested: RenderMode) {
        let previous_requested = self.requested_render_mode;
        let previous_backend = self.render_backend;
        let resolved = resolve_render_backend(requested, self.rt_capabilities.supported());

        if requested == RenderMode::Rt
            && resolved == RenderBackend::Vpt
            && previous_requested != RenderMode::Rt
        {
            tracing::warn!(
                device = %self.renderer.physical_device_name(),
                "requested RT backend but hardware support was unavailable; falling back to VPT"
            );
        }
        if previous_requested != requested || previous_backend != resolved {
            tracing::debug!(
                requested = ?requested,
                render_backend = ?resolved,
                rt_supported = self.rt_capabilities.supported(),
                "updated render backend selection"
            );
        }

        self.requested_render_mode = requested;
        self.render_backend = resolved;
    }

    pub fn ensure_passes(
        &mut self,
        ucvh: Option<&Ucvh>,
        settings: RuntimeSettings,
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) {
        if let Some(scene_ubo) = self.scene_ubo.as_ref() {
            match self.render_backend {
                RenderBackend::Rt => {
                    self.rt_pipeline.ensure_passes(
                        &self.renderer,
                        scene_ubo,
                        ucvh,
                        self.ucvh_gpu.as_ref(),
                        settings.rt,
                    );
                }
                RenderBackend::Vpt => {
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
        self.refresh_render_backend(settings.lighting.render_mode);
        self.ensure_passes(ucvh, settings, restir_di_enabled, area_restir_enabled);
        let extent = self.renderer.swapchain_extent();
        let Some(scene_ubo) = self.scene_ubo.take() else {
            return Ok(());
        };
        let result = {
            let scene_ubo = &scene_ubo;
            (|| {
                self.resize_rt_pipeline_to_swapchain(scene_ubo, extent.width, extent.height)?;
                self.resize_vpt_pipeline_to_swapchain(
                    scene_ubo,
                    extent.width,
                    extent.height,
                    settings,
                    restir_di_enabled,
                    area_restir_enabled,
                )?;
                Ok(())
            })()
        };
        self.scene_ubo = Some(scene_ubo);
        result
    }

    fn resize_rt_pipeline_to_swapchain(
        &mut self,
        scene_ubo: &SceneUniformBuffer,
        width: u32,
        height: u32,
    ) -> Result<()> {
        if !(self.render_backend == RenderBackend::Rt || self.rt_pipeline.has_frame_resources()) {
            return Ok(());
        }

        self.rt_history_reset_generation = self.rt_history_reset_generation.wrapping_add(1);
        self.rt_pipeline
            .resize(&self.renderer, scene_ubo, width, height)?;
        Ok(())
    }

    fn resize_vpt_pipeline_to_swapchain(
        &mut self,
        scene_ubo: &SceneUniformBuffer,
        width: u32,
        height: u32,
        settings: RuntimeSettings,
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) -> Result<()> {
        if !(self.render_backend == RenderBackend::Vpt || self.vpt_pipeline.has_frame_resources()) {
            return Ok(());
        }
        let Some(ucvh_gpu) = self.ucvh_gpu.as_ref() else {
            return Ok(());
        };

        self.vpt_pipeline.resize(
            &self.renderer,
            scene_ubo,
            ucvh_gpu,
            width,
            height,
            settings.lighting,
            restir_di_enabled,
            area_restir_enabled,
        )?;
        Ok(())
    }

    pub fn render_frame(&mut self, mut input: RenderFrameInput<'_>) -> Result<RenderFrameOutcome> {
        let mut outcome = RenderFrameOutcome::default();
        let frame = self.renderer.begin_frame()?;
        outcome.began_frame = true;
        self.refresh_render_backend(input.settings.lighting.render_mode);

        if self.last_render_backend != self.render_backend {
            self.rt_history_reset_generation = self.rt_history_reset_generation.wrapping_add(1);
            self.rt_pipeline
                .reset_history(self.rt_history_reset_generation);
            self.last_render_backend = self.render_backend;
        }

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

        if self.scene_ubo.is_none() {
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
        }

        self.ensure_passes(
            input.ucvh.as_deref(),
            input.settings,
            input.restir_di_enabled,
            input.area_restir_enabled,
        );
        let scene_ubo = self
            .scene_ubo
            .as_ref()
            .expect("scene UBO was checked before ensuring render passes");

        let as_rebuild_generation = self.rt_pipeline.as_rebuild_generation();
        let record_result = match self.render_backend {
            RenderBackend::Rt => self.rt_pipeline.record_and_execute_frame(
                &self.renderer,
                &frame,
                #[cfg(not(target_os = "android"))]
                self.egui_renderer.as_mut(),
                #[cfg(not(target_os = "android"))]
                input.egui_frame.as_ref(),
                RtFrameInputs {
                    scene_ubo,
                    camera: input.camera,
                    sun_direction: input.sun_direction,
                    sun_intensity: input.sun_intensity,
                    elapsed_seconds: input.elapsed_seconds,
                    lighting_settings: input.settings.lighting,
                    rt_settings: input.settings.rt,
                    capture: self.capture.as_mut(),
                    ucvh_ready: self.ucvh_uploaded,
                    ucvh: input.ucvh.as_deref(),
                    ucvh_gpu: self.ucvh_gpu.as_ref(),
                    external_history_reset_generation: self
                        .rt_history_reset_generation
                        .max(as_rebuild_generation),
                },
            )?,
            RenderBackend::Vpt => self.vpt_pipeline.record_and_execute_frame(
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
                    rt_settings: input.settings.rt,
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
            )?,
        };
        let frame_slot = frame.frame_slot;
        let submitted_fence = record_result.submitted_fence;
        let traversal_stats_requested = record_result.traversal_stats_requested;
        let traversal_stats = record_result.traversal_stats;
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
        if traversal_stats_requested || pending_capture.is_some() {
            self.renderer.wait_for_fence(submitted_fence)?;
        }
        if let Some(snapshot) = traversal_stats {
            tracing::info!("{}", snapshot.format_log_line());
        } else if traversal_stats_requested {
            match self.vpt_pipeline.traversal_stats_snapshot(frame_slot)? {
                Some(snapshot) => tracing::info!("{}", snapshot.format_log_line()),
                None => tracing::warn!("TraversalStats requested but no stats buffer exists"),
            }
        }
        if let Some(metadata) = pending_capture.take() {
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
        let rt_pipeline = std::mem::take(&mut self.rt_pipeline);
        rt_pipeline.destroy(
            self.renderer.device(),
            self.renderer.allocator(),
            self.renderer.acceleration_structure_loader(),
        );
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
    use crate::render::rt_capabilities::{RenderBackend, resolve_render_backend};
    use crate::render::rt_settings::{RtDebugView, RtSettings};
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
    fn rt_settings_parse_valid_overrides() {
        let parsed = RtSettings::from_values(
            Some("on"),
            Some("off"),
            Some("true"),
            Some("32"),
            Some("0.85"),
            Some("0.02"),
            Some("surface"),
            Some("off"),
            Some("4"),
        );

        assert!(parsed.settings.restir_di_enabled);
        assert!(!parsed.settings.restir_gi_enabled);
        assert!(parsed.settings.temporal_denoise_enabled);
        assert!(!parsed.settings.restir_di_spatial_enabled);
        assert_eq!(parsed.settings.restir_di_spatial_sample_count, 4);
        assert_eq!(parsed.settings.history_length, 32);
        assert_eq!(parsed.settings.normal_threshold, 0.85);
        assert_eq!(parsed.settings.depth_threshold, 0.02);
        assert_eq!(parsed.settings.debug_view, RtDebugView::Surface);
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn runtime_resolves_rt_to_vpt_when_hardware_support_is_missing() {
        assert_eq!(
            resolve_render_backend(crate::render::scene_ubo::RenderMode::Rt, false),
            RenderBackend::Vpt
        );
    }

    #[test]
    fn render_runtime_owns_gpu_resources_and_frame_orchestration() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let runtime_struct = source
            .split("pub struct RenderRuntime {")
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

    #[test]
    fn render_frame_revalidates_runtime_toggled_passes_before_recording_vpt() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before UCVH helpers");
        let compact = crate::render::source_checks::compact(render_frame);
        let scene_ubo_guard = compact
            .find("ifself.scene_ubo.is_none(){")
            .expect("render_frame must guard missing scene UBO before rebuilding passes");
        let ensure_passes = compact
            .find(
                "self.ensure_passes(input.ucvh.as_deref(),input.settings,input.restir_di_enabled,input.area_restir_enabled,);",
            )
            .expect("render_frame must ensure passes before recording");
        let record_vpt = compact
            .find("self.vpt_pipeline.record_and_execute_frame(")
            .expect("render_frame must call the VPT recorder");

        assert!(
            scene_ubo_guard < ensure_passes,
            "render_frame must wait until scene UBO exists before rebuilding GPU passes"
        );
        assert!(
            ensure_passes < record_vpt,
            "render_frame must ensure passes with current UI settings before recording so live denoiser/ReSTIR toggles can instantiate their GPU passes"
        );
    }

    #[test]
    fn render_runtime_routes_selected_rt_backend_to_rt_pipeline() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let runtime_struct = source
            .split("pub struct RenderRuntime {")
            .nth(1)
            .expect("RenderRuntime struct should exist")
            .split("impl RenderRuntime")
            .next()
            .expect("RenderRuntime struct should end before impl");
        assert!(
            runtime_struct.contains("rt_pipeline: RtRuntimePipeline"),
            "RenderRuntime must own the hardware RT pipeline beside the VPT fallback"
        );

        let ensure_passes = source
            .split("pub fn ensure_passes(")
            .nth(1)
            .expect("RenderRuntime::ensure_passes should exist")
            .split("pub fn resize(")
            .next()
            .expect("ensure_passes should end before resize");
        assert!(ensure_passes.contains("RenderBackend::Rt"));
        assert!(ensure_passes.contains("self.rt_pipeline.ensure_passes"));

        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before helpers");
        let compact = crate::render::source_checks::compact(render_frame);
        assert!(compact.contains("matchself.render_backend{"));
        assert!(compact.contains("RenderBackend::Rt=>self.rt_pipeline.record_and_execute_frame("));
        assert!(
            compact.contains("RenderBackend::Vpt=>self.vpt_pipeline.record_and_execute_frame(")
        );
    }

    #[test]
    fn render_runtime_passes_capture_to_rt_and_vpt_pipelines() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before helpers");
        let compact = crate::render::source_checks::compact(render_frame);
        let rt_branch_start = compact
            .find("RenderBackend::Rt=>")
            .expect("render_frame must contain RT branch");
        let vpt_branch_start = compact
            .find("RenderBackend::Vpt=>")
            .expect("render_frame must contain VPT branch");
        let rt_branch = &compact[rt_branch_start..vpt_branch_start];
        let vpt_branch = &compact[vpt_branch_start..];

        assert!(
            rt_branch.contains("RtFrameInputs{")
                && rt_branch.contains("capture:self.capture.as_mut(),"),
            "RT frame inputs must receive RenderCapture for RT resolve readback"
        );
        assert!(
            vpt_branch.contains("VptFrameInputs{")
                && vpt_branch.contains("rt_settings:input.settings.rt,")
                && vpt_branch.contains("capture:self.capture.as_mut(),"),
            "VPT frame inputs must receive RT settings and RenderCapture for fallback metadata"
        );
    }

    #[test]
    fn render_runtime_passes_ucvh_gpu_resources_to_rt_pipeline() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");

        let ensure_passes = source
            .split("pub fn ensure_passes(")
            .nth(1)
            .expect("RenderRuntime::ensure_passes should exist")
            .split("pub fn resize(")
            .next()
            .expect("ensure_passes should end before resize");
        let ensure_compact = crate::render::source_checks::compact(ensure_passes);
        assert!(
            ensure_compact.contains(
                "self.rt_pipeline.ensure_passes(&self.renderer,scene_ubo,ucvh,self.ucvh_gpu.as_ref(),settings.rt,"
            ),
            "RT pass creation must receive CPU UCVH, GPU UCVH resources, and RT settings"
        );

        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before helpers");
        let compact = crate::render::source_checks::compact(render_frame);
        assert!(
            compact.contains("ucvh_gpu:self.ucvh_gpu.as_ref()"),
            "RT frame inputs must receive GPU UCVH resources for descriptor refresh"
        );
    }

    #[test]
    fn runtime_resets_rt_history_when_backend_or_scene_generation_changes() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");

        assert!(source.contains("history_reset_generation"));
        assert!(source.contains("as_rebuild_generation"));
        assert!(source.contains("rt_history_reset_generation"));
    }

    #[test]
    fn render_runtime_status_exposes_backend_and_rt_support_for_editor() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let status_struct = source
            .split("pub struct RenderRuntimeStatus")
            .nth(1)
            .expect("RenderRuntimeStatus should exist")
            .split("pub struct RenderRuntime {")
            .next()
            .expect("RenderRuntimeStatus should be declared before RenderRuntime");

        for token in [
            "pub actual_backend: RenderBackend",
            "pub rt_supported: bool",
            "pub rt_frame_status: Option<RtFrameStatus>",
        ] {
            assert!(
                status_struct.contains(token),
                "RenderRuntimeStatus missing {token}"
            );
        }

        let runtime_impl = source
            .split("impl RenderRuntime")
            .nth(1)
            .expect("RenderRuntime impl should exist");
        let runtime_impl_compact = crate::render::source_checks::compact(runtime_impl);
        for token in [
            "pub fn status(&self) -> RenderRuntimeStatus",
            "actual_backend: self.render_backend",
            "rt_supported: self.rt_capabilities.supported()",
        ] {
            assert!(
                runtime_impl.contains(token),
                "RenderRuntime::status missing {token}"
            );
        }
        for token in [
            "rt_frame_status:",
            "self.render_backend==RenderBackend::Rt",
            "self.rt_pipeline.has_frame_resources()",
            "self.rt_pipeline.frame_status()",
        ] {
            assert!(
                runtime_impl_compact.contains(token),
                "RenderRuntime::status missing compact RT frame status token {token}"
            );
        }
    }

    #[test]
    fn render_frame_refreshes_backend_from_current_settings_before_pass_selection() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before UCVH helpers");
        let compact = crate::render::source_checks::compact(render_frame);

        let refresh = compact
            .find("self.refresh_render_backend(input.settings.lighting.render_mode);")
            .expect("render_frame must refresh backend from current frame settings");
        let reset = compact
            .find("ifself.last_render_backend!=self.render_backend{")
            .expect("render_frame must retain backend-change history reset");
        let ensure_passes = compact
            .find("self.ensure_passes(")
            .expect("render_frame must ensure passes");
        let record = compact
            .find("matchself.render_backend{")
            .expect("render_frame must select pass recorder from refreshed backend");

        assert!(refresh < reset);
        assert!(refresh < ensure_passes);
        assert!(refresh < record);
    }

    #[test]
    fn resize_refreshes_backend_from_current_settings() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let resize_pipeline = source
            .split("fn resize_pipeline_to_swapchain")
            .nth(1)
            .expect("resize_pipeline_to_swapchain should exist")
            .split("pub fn render_frame")
            .next()
            .expect("resize helper should end before render_frame");
        let compact = crate::render::source_checks::compact(resize_pipeline);

        let refresh = compact
            .find("self.refresh_render_backend(settings.lighting.render_mode);")
            .expect("resize must refresh backend from current settings");
        let ensure_passes = compact
            .find("self.ensure_passes(")
            .expect("resize must ensure passes");
        let rt_resize = compact
            .find("self.resize_rt_pipeline_to_swapchain(")
            .expect("resize must route RT resources through refreshed backend state");
        let vpt_resize = compact
            .find("self.resize_vpt_pipeline_to_swapchain(")
            .expect("resize must route VPT resources through refreshed backend state");

        assert!(refresh < ensure_passes);
        assert!(refresh < rt_resize);
        assert!(refresh < vpt_resize);
        assert!(ensure_passes < rt_resize);
        assert!(ensure_passes < vpt_resize);
    }

    #[test]
    fn render_runtime_has_backend_refresh_helper() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let runtime_impl = source
            .split("impl RenderRuntime")
            .nth(1)
            .expect("RenderRuntime impl should exist");

        for token in [
            "fn refresh_render_backend(&mut self, requested: RenderMode)",
            "resolve_render_backend(requested, self.rt_capabilities.supported())",
            "self.render_backend = resolved",
        ] {
            assert!(
                runtime_impl.contains(token),
                "backend refresh helper missing {token}"
            );
        }
    }

    #[test]
    fn resize_pipeline_to_swapchain_resizes_selected_and_existing_inactive_backends() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let resize_pipeline = source
            .split("fn resize_pipeline_to_swapchain")
            .nth(1)
            .expect("resize_pipeline_to_swapchain should exist")
            .split("pub fn render_frame")
            .next()
            .expect("resize helper should end before render_frame");
        let compact = crate::render::source_checks::compact(resize_pipeline);

        for token in [
            "self.resize_rt_pipeline_to_swapchain(scene_ubo,extent.width,extent.height)?",
            "self.resize_vpt_pipeline_to_swapchain(scene_ubo,extent.width,extent.height,settings,restir_di_enabled,area_restir_enabled,)?",
        ] {
            assert!(
                compact.contains(token),
                "runtime resize must route through backend resize helper {token}"
            );
        }

        assert!(
            !compact.contains("matchself.render_backend{"),
            "resize must not resize only the currently selected backend because inactive backend resources can become stale"
        );
    }

    #[test]
    fn rt_resize_helper_resizes_selected_or_existing_rt_resources() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let helper = source
            .split("fn resize_rt_pipeline_to_swapchain")
            .nth(1)
            .expect("resize_rt_pipeline_to_swapchain should exist")
            .split("fn resize_vpt_pipeline_to_swapchain")
            .next()
            .expect("RT resize helper should end before VPT resize helper");
        let compact = crate::render::source_checks::compact(helper);

        for token in [
            "self.render_backend==RenderBackend::Rt||self.rt_pipeline.has_frame_resources()",
            "self.rt_history_reset_generation=self.rt_history_reset_generation.wrapping_add(1)",
            "self.rt_pipeline.resize(&self.renderer,scene_ubo,width,height)?",
        ] {
            assert!(compact.contains(token), "RT resize helper missing {token}");
        }
    }

    #[test]
    fn vpt_resize_helper_resizes_selected_or_existing_vpt_resources() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let helper = source
            .split("fn resize_vpt_pipeline_to_swapchain")
            .nth(1)
            .expect("resize_vpt_pipeline_to_swapchain should exist")
            .split("pub fn render_frame")
            .next()
            .expect("VPT resize helper should end before render_frame");
        let compact = crate::render::source_checks::compact(helper);

        for token in [
            "self.render_backend==RenderBackend::Vpt||self.vpt_pipeline.has_frame_resources()",
            "letSome(ucvh_gpu)=self.ucvh_gpu.as_ref()else",
            "self.vpt_pipeline.resize(&self.renderer,scene_ubo,ucvh_gpu,width,height,settings.lighting,restir_di_enabled,area_restir_enabled,)?",
        ] {
            assert!(compact.contains(token), "VPT resize helper missing {token}");
        }
    }
}
