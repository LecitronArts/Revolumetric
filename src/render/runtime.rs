use anyhow::Result;
use winit::window::Window;

use crate::render::area_restir::AreaRestirSettings;
use crate::render::capture::{CaptureCameraPathMetadata, RenderCapture};
use crate::render::device::RenderDevice;
#[cfg(not(target_os = "android"))]
use crate::render::egui_renderer::{EguiFrame, EguiRenderer};
use crate::render::gpu_profiler::{GpuProfiler, GpuProfilerConfig};
use crate::render::restir_di::RestirDiSettings;
use crate::render::rt_capabilities::{RenderBackend, RtCapabilities, resolve_render_backend};
use crate::render::rt_page_blas::{
    RtCompactBlasCreateInfo, RtCompactBlasBuildResources, RtCompactBlasRetirementQueue,
    RtCompactBlasScratchFreeQueue,
};
use crate::render::rt_page_geometry::RtCompactPageGeometry;
use crate::render::rt_page_gpu::{
    RtPageGpuConfig, RtPageGpuResources, shared_rt_page_lattice_vertices,
};
use crate::render::rt_page_registry::{
    RtPageBuildStartError, RtPageQueueReport, RtPageRegistry, RtPageRepresentation,
};
use crate::render::rt_page_tlas::{
    RtPageCompactTlasBinding, RtPageTlasFrameReferences, RtPageTlasGpuConfig,
    RtPageTlasGpuResources, RtPageTlasInstanceStore,
};
use crate::render::rt_pipeline::{RtFrameInputs, RtFrameStatus, RtRuntimePipeline};
use crate::render::rt_settings::RtSettings;
use crate::render::rt_surface_mask::SurfaceMaskPage;
use crate::render::scene_ubo::{LightingSettings, RenderMode, SceneUniformBuffer};
use crate::render::vpt_pipeline::{
    UcvhFrameChanges, VptCameraFrame, VptFrameInputs, VptRuntimePipeline,
};
use crate::voxel::gpu_upload::{
    INITIAL_UCVH_UPLOAD_FRAME_BUDGET_BYTES, UcvhGpuResources, UcvhInitialUploadProgress,
};
use crate::voxel::ucvh::Ucvh;

const MAX_AUTO_RT_UCVH_BRICKS: u32 = 100_000;
const RT_PAGE_DIRTY_QUEUE_CAPACITY: usize = 16_384;
const RT_PAGE_INITIAL_TLAS_CAPACITY: u32 = 16_384;
/// GPU geometry arena: 4 MiB for index buffer.
const RT_PAGE_GPU_INDEX_CAPACITY_BYTES: u64 = 4 * 1024 * 1024;
/// GPU geometry arena: 65536 face records (each 4 bytes).
const RT_PAGE_GPU_FACE_CAPACITY_RECORDS: u32 = 65536;
/// Per-frame staging buffer: large enough for lattice upload + N pages/frame.
const RT_PAGE_GPU_STAGING_BYTES_PER_FRAME: u64 = 1024 * 1024;
/// Maximum dirty pages to build per frame (geometry upload + BLAS build).
const RT_PAGE_BUILDS_PER_FRAME: usize = 4;

struct RtPageTlasRuntime {
    gpu: RtPageTlasGpuResources,
    instances: RtPageTlasInstanceStore,
    frame_references: RtPageTlasFrameReferences,
}

fn frame_render_backend(
    requested: RenderMode,
    rt_supported: bool,
    ucvh_brick_count: Option<u32>,
) -> RenderBackend {
    let resolved = resolve_render_backend(requested, rt_supported);
    if requested == RenderMode::Auto
        && resolved == RenderBackend::Rt
        && ucvh_brick_count.is_none_or(|count| count > MAX_AUTO_RT_UCVH_BRICKS)
    {
        return RenderBackend::Vpt;
    }
    resolved
}

fn ucvh_gpu_requires_recreation(gpu_capacity: usize, cpu_storage_len: usize) -> bool {
    gpu_capacity < cpu_storage_len
}

#[derive(Debug, Clone, Copy)]
pub struct RuntimeSettings {
    pub lighting: LightingSettings,
    pub rt: RtSettings,
    pub restir_di: RestirDiSettings,
    pub area_restir: AreaRestirSettings,
}

pub struct RenderFrameInput<'a> {
    pub camera: VptCameraFrame,
    pub camera_path: CaptureCameraPathMetadata,
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
    ucvh_initial_upload: UcvhInitialUploadProgress,
    ucvh_initial_upload_batch_id: Option<u64>,
    ucvh_initial_upload_snapshot_taken: bool,
    ucvh_initial_upload_committed: bool,
    ucvh_uploaded: bool,
    /// True when the RT backend skipped uploading changed L1-L4 occupancy hierarchy
    /// levels (T1-C). The GPU L1-L4 buffers then lag the CPU hierarchy; before the
    /// next VPT trace (only VPT reads L1-L4) the runtime records a full L1-L4
    /// re-upload and clears this. Kept correct across the auto RT↔VPT backend switch.
    hierarchy_ln_gpu_stale: bool,
    rt_page_registry: RtPageRegistry,
    rt_page_registry_bootstrapped: bool,
    rt_page_tlas: Option<RtPageTlasRuntime>,
    /// GPU geometry/index/face/page-record buffers for the CompactExact triangle path.
    /// Created on first frame that has ucvh data and an acceleration structure loader.
    rt_page_gpu: Option<crate::render::rt_page_gpu::RtPageGpuResources>,
    /// Whether the shared 9×9×9 lattice has been uploaded to the GPU.
    rt_page_lattice_uploaded: bool,
    /// Deferred-retire queue for old CompactExact BLAS resources.
    rt_page_retirement_queue: RtCompactBlasRetirementQueue,
    /// Fence-gated free-queue for BLAS build scratch buffers (T1-E). Scratch is dead
    /// after its build submission signals, so we free it without waiting for the
    /// resident BLAS to retire, reclaiming per-page AS scratch VRAM.
    rt_page_scratch_free_queue: RtCompactBlasScratchFreeQueue,
    /// Installed CompactExact BLASes still resident in the TLAS.
    rt_page_installed_blas: Vec<crate::render::rt_page_blas::RtInstalledCompactBlas>,
    rt_pipeline: RtRuntimePipeline,
    vpt_pipeline: VptRuntimePipeline,
    #[cfg(not(target_os = "android"))]
    egui_renderer: Option<EguiRenderer>,
}

impl RenderRuntime {
    pub fn new(window: &Window, settings: RuntimeSettings, ucvh: Option<&Ucvh>) -> Result<Self> {
        let renderer = RenderDevice::new(window)?;
        let rt_capabilities = renderer.rt_capabilities();
        let render_backend = frame_render_backend(
            settings.lighting.render_mode,
            rt_capabilities.supported(),
            ucvh.map(|ucvh| ucvh.pool.allocated_count()),
        );

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
                match UcvhGpuResources::new(
                    renderer.device(),
                    renderer.allocator(),
                    ucvh,
                    renderer.frame_slot_count(),
                ) {
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
            ucvh_initial_upload: UcvhInitialUploadProgress::default(),
            ucvh_initial_upload_batch_id: None,
            ucvh_initial_upload_snapshot_taken: false,
            ucvh_initial_upload_committed: false,
            ucvh_uploaded: false,
            hierarchy_ln_gpu_stale: false,
            rt_page_registry: RtPageRegistry::new(RT_PAGE_DIRTY_QUEUE_CAPACITY),
            rt_page_registry_bootstrapped: false,
            rt_page_tlas: None,
            rt_page_gpu: None,
            rt_page_lattice_uploaded: false,
            rt_page_retirement_queue: RtCompactBlasRetirementQueue::default(),
            rt_page_scratch_free_queue: RtCompactBlasScratchFreeQueue::default(),
            rt_page_installed_blas: Vec::new(),
            rt_pipeline: RtRuntimePipeline::new(),
            vpt_pipeline: VptRuntimePipeline::new(),
            #[cfg(not(target_os = "android"))]
            egui_renderer,
        };
        runtime.ensure_rt_page_registry_bootstrapped(ucvh);
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

    #[cfg(not(target_os = "android"))]
    pub fn egui_font_texture_ready(&self) -> bool {
        self.egui_renderer
            .as_ref()
            .is_some_and(EguiRenderer::font_texture_ready)
    }

    fn refresh_render_backend(&mut self, requested: RenderMode, ucvh_brick_count: Option<u32>) {
        let previous_requested = self.requested_render_mode;
        let previous_backend = self.render_backend;
        let resolved = frame_render_backend(
            requested,
            self.rt_capabilities.supported(),
            ucvh_brick_count,
        );

        if requested == RenderMode::Rt
            && resolved == RenderBackend::Vpt
            && previous_requested != RenderMode::Rt
        {
            tracing::warn!(
                device = %self.renderer.physical_device_name(),
                "requested RT backend but hardware support was unavailable; falling back to VPT"
            );
        }
        if requested == RenderMode::Auto
            && self.rt_capabilities.supported()
            && resolved == RenderBackend::Vpt
            && ucvh_brick_count.is_some_and(|count| count > MAX_AUTO_RT_UCVH_BRICKS)
            && previous_backend != RenderBackend::Vpt
        {
            tracing::warn!(
                bricks = ucvh_brick_count.unwrap_or_default(),
                max_auto_rt_bricks = MAX_AUTO_RT_UCVH_BRICKS,
                "large UCVH scene is using VPT in Auto mode to avoid startup RT acceleration-structure stall"
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

    fn ensure_ucvh_gpu_resources(&mut self, ucvh: Option<&Ucvh>) {
        if self.ucvh_gpu.is_some() {
            return;
        }
        let Some(ucvh) = ucvh else {
            return;
        };
        match UcvhGpuResources::new(
            self.renderer.device(),
            self.renderer.allocator(),
            ucvh,
            self.renderer.frame_slot_count(),
        ) {
            Ok(gpu) => {
                self.ucvh_gpu = Some(gpu);
                self.reset_ucvh_initial_upload_state();
                tracing::info!("created UCVH GPU resources");
            }
            Err(error) => {
                tracing::error!(%error, "failed to create UCVH GPU resources");
            }
        }
    }

    fn reset_ucvh_initial_upload_state(&mut self) {
        self.ucvh_initial_upload = UcvhInitialUploadProgress::default();
        self.ucvh_initial_upload_batch_id = None;
        self.ucvh_initial_upload_snapshot_taken = false;
        self.ucvh_initial_upload_committed = false;
        self.ucvh_uploaded = false;
        // The initial upload path re-sends the full hierarchy (all levels), so any
        // prior RT-mode L1-L4 skip is superseded (T1-C).
        self.hierarchy_ln_gpu_stale = false;
    }

    fn ensure_ucvh_gpu_capacity(&mut self, ucvh: Option<&Ucvh>) -> Result<()> {
        self.ensure_ucvh_gpu_resources(ucvh);
        let Some(ucvh) = ucvh else {
            return Ok(());
        };
        let Some(gpu_capacity) = self.ucvh_gpu.as_ref().map(UcvhGpuResources::brick_capacity)
        else {
            return Ok(());
        };
        let cpu_storage_len = ucvh.pool.occupancy_pool().len();
        if !ucvh_gpu_requires_recreation(gpu_capacity, cpu_storage_len) {
            return Ok(());
        }

        tracing::info!(
            gpu_capacity,
            cpu_storage_len,
            "recreating UCVH GPU resources after CPU brick storage growth"
        );
        self.renderer.wait_idle()?;

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

        self.reset_ucvh_initial_upload_state();
        self.rt_history_reset_generation = self.rt_history_reset_generation.wrapping_add(1);
        let gpu = UcvhGpuResources::new(
            self.renderer.device(),
            self.renderer.allocator(),
            ucvh,
            self.renderer.frame_slot_count(),
        )?;
        self.ucvh_gpu = Some(gpu);
        Ok(())
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
        self.refresh_render_backend(
            settings.lighting.render_mode,
            ucvh.map(|ucvh| ucvh.pool.allocated_count()),
        );
        let extent = self.renderer.swapchain_extent();
        let Some(scene_ubo) = self.scene_ubo.take() else {
            return Ok(());
        };
        let result: Result<()> = {
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
        result?;
        self.ensure_passes(ucvh, settings, restir_di_enabled, area_restir_enabled);
        Ok(())
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
        self.ensure_rt_page_registry_bootstrapped(input.ucvh.as_deref());
        self.ensure_ucvh_gpu_capacity(input.ucvh.as_deref())?;
        let frame = self.renderer.begin_frame()?;
        outcome.began_frame = true;
        self.refresh_render_backend(
            input.settings.lighting.render_mode,
            input
                .ucvh
                .as_deref()
                .map(|ucvh| ucvh.pool.allocated_count()),
        );
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

        let mut initial_upload_completed_this_frame = false;
        if !self.ucvh_initial_upload_committed {
            if let (Some(ucvh), Some(gpu)) = (input.ucvh.as_deref_mut(), &self.ucvh_gpu) {
                if !self.ucvh_initial_upload_snapshot_taken {
                    let batch = ucvh.snapshot_render_change_batch();
                    let report = self
                        .rt_page_registry
                        .ingest_render_change_batch(&batch, frame.frame_index);
                    Self::log_rt_page_queue_report("initial authority snapshot", &report);
                    self.ucvh_initial_upload_batch_id = (!batch.is_empty()).then_some(batch.id);
                    self.ucvh_initial_upload_snapshot_taken = true;
                }
                match gpu.upload_initial_incremental(
                    self.renderer.device(),
                    frame.command_buffer,
                    frame.frame_slot,
                    ucvh,
                    &mut self.ucvh_initial_upload,
                    INITIAL_UCVH_UPLOAD_FRAME_BUDGET_BYTES,
                ) {
                    Ok(upload) => {
                        if upload.bytes_uploaded > 0 {
                            tracing::debug!(
                                bytes = upload.bytes_uploaded,
                                completed = upload.completed,
                                "advanced initial UCVH GPU upload"
                            );
                        }
                        if upload.completed {
                            initial_upload_completed_this_frame = true;
                        }
                    }
                    Err(error) => {
                        tracing::error!(%error, "failed to upload UCVH data to GPU");
                    }
                }
            }
        }

        let mut uploaded_authority_batch_id = None;
        if self.ucvh_initial_upload_committed {
            if let Some(ucvh) = input.ucvh.as_deref_mut() {
                let batch = ucvh.snapshot_render_change_batch();
                if !batch.is_empty() {
                    // T1-B Phase 1: grow material staging for this frame slot if the
                    // batch exceeds the current (budget-sized) staging capacity. Uses a
                    // mutable borrow of self.ucvh_gpu; Phase 2 (immutable) follows.
                    // Fence-safe: the runtime waits this slot's fence before reusing its
                    // command buffer, so no in-flight GPU transfer reads staging here.
                    let device = self.renderer.device();
                    let allocator = self.renderer.allocator();
                    if let Some(gpu) = &mut self.ucvh_gpu {
                        if let Err(e) = gpu.ensure_material_staging_capacity(
                            device,
                            allocator,
                            frame.frame_slot,
                            batch.bricks.len(),
                        ) {
                            tracing::warn!(
                                %e,
                                bricks = batch.bricks.len(),
                                "failed to grow material staging for incremental batch"
                            );
                        }
                    }

                    let report = self
                        .rt_page_registry
                        .ingest_render_change_batch(&batch, frame.frame_index);
                    Self::log_rt_page_queue_report("incremental authority snapshot", &report);
                    // T1-C: on the RT backend, skip uploading changed L1-L4 hierarchy
                    // levels — only VPT's software traversal reads them. L0 is always
                    // uploaded (RT's procedural intersection reads it).
                    let skip_ln_hierarchy_upload = self.render_backend == RenderBackend::Rt;
                    // T1-B Phase 2: upload with budget-sized staging (compact slots).
                    if let Some(gpu) = &self.ucvh_gpu {
                        match gpu.upload_incremental_changes(
                            self.renderer.device(),
                            frame.command_buffer,
                            frame.frame_slot,
                            ucvh,
                            &batch,
                            skip_ln_hierarchy_upload,
                        ) {
                            Ok(upload) => {
                                uploaded_authority_batch_id = Some(batch.id);
                                if upload.skipped_ln_hierarchy {
                                    self.hierarchy_ln_gpu_stale = true;
                                }
                                tracing::debug!(
                                    batch_id = batch.id,
                                    changed_bricks = upload.changed_bricks,
                                    bytes = upload.bytes_uploaded,
                                    skipped_ln_hierarchy = upload.skipped_ln_hierarchy,
                                    "uploaded incremental UCVH authority changes"
                                );
                            }
                            Err(error) => {
                                tracing::error!(
                                    %error,
                                    batch_id = batch.id,
                                    "failed to upload incremental UCVH authority changes; retaining batch"
                                );
                            }
                        }
                    }
                }
            }
        }

        // T1-C: heal a stale GPU L1-L4 hierarchy before the VPT backend traces it.
        // The RT backend skips incremental L1-L4 uploads; when the auto backend
        // switches to VPT (large scene) the CPU L1-L4 is current but the GPU lags, so
        // re-upload all four levels in one pass. Robust to missing the exact switch
        // frame — any VPT frame with the flag set heals it before tracing. Capacity
        // recreation re-uploads everything via the initial path and clears the flag.
        if self.render_backend == RenderBackend::Vpt
            && self.hierarchy_ln_gpu_stale
            && self.ucvh_initial_upload_committed
        {
            let mut resynced = false;
            if let (Some(ucvh), Some(gpu)) = (input.ucvh.as_deref(), self.ucvh_gpu.as_ref()) {
                match gpu.record_full_ln_hierarchy_upload(
                    self.renderer.device(),
                    frame.command_buffer,
                    frame.frame_slot,
                    ucvh,
                ) {
                    Ok(bytes) => {
                        tracing::info!(
                            bytes,
                            "resynchronized full L1-L4 hierarchy after RT→VPT backend switch"
                        );
                        resynced = true;
                    }
                    Err(error) => tracing::error!(
                        %error,
                        "failed to resync L1-L4 hierarchy on RT→VPT backend switch"
                    ),
                }
            }
            if resynced {
                self.hierarchy_ln_gpu_stale = false;
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
        let uploaded_invalidation_regions = ucvh_frame_changes.invalidation_regions.clone();
        let uploaded_motion_events = ucvh_frame_changes.motion_events.clone();

        let mut ucvh_motion_event_count = 0u32;
        let mut motion_events_uploaded = false;
        if self.ucvh_uploaded
            && let Some(gpu) = self.ucvh_gpu.as_ref()
        {
            match gpu.upload_motion_guide(
                self.renderer.device(),
                frame.command_buffer,
                frame.frame_slot,
                &uploaded_motion_events,
            ) {
                Ok(count) => {
                    ucvh_motion_event_count = count;
                    outcome.uploaded_motion_events = count;
                    motion_events_uploaded = true;
                }
                Err(error) => tracing::error!(%error, "failed to upload UCVH motion guide"),
            }
        }

        let rt_page_tlas_submission = if self.render_backend == RenderBackend::Rt {
            // Record CompactExact BLAS builds before the TLAS update so that
            // AS barriers within the same command buffer order build → TLAS → trace.
            if let Some(ucvh) = input.ucvh.as_deref() {
                if let Err(e) = self.record_rt_page_builds(
                    frame.command_buffer,
                    frame.frame_slot,
                    frame.frame_index,
                    ucvh,
                ) {
                    tracing::warn!(%e, "RT page build recording failed");
                }
            }
            Some((
                frame.frame_slot,
                frame.frame_index,
                self.record_rt_page_tlas_before_trace(frame.command_buffer, frame.frame_slot)?,
            ))
        } else {
            None
        };

        let record_result = if self.scene_ubo.is_none() {
            tracing::warn!("skipping render frame until scene UBO is initialized");
            None
        } else {
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
                        camera_path: input.camera_path.clone(),
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
                        profiler: self.gpu_profiler.as_ref(),
                        rt_page_tlas_handle: self
                            .rt_page_tlas
                            .as_ref()
                            .and_then(|pt| pt.gpu.tlas_handle(frame.frame_slot)),
                        rt_page_face_buffer: self
                            .rt_page_gpu
                            .as_ref()
                            .map(|pg| pg.face_buffer()),
                        rt_page_record_buffer: self
                            .rt_page_gpu
                            .as_ref()
                            .map(|pg| pg.page_record_buffer()),
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
                        camera_path: input.camera_path,
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
            outcome.rendered = true;
            Some(record_result)
        };
        let frame_slot = frame.frame_slot;
        // Record the TLAS submission *before* end_frame submits the command buffer to the GPU.
        // This ensures can_retire() never sees the resources as free during the window between
        // vkQueueSubmit (end_frame) and the CPU-side in-flight stamp (record_submission).
        if let Some((submission_frame_slot, frame_index, resource_versions)) =
            rt_page_tlas_submission.as_ref()
            && let Some(page_tlas) = self.rt_page_tlas.as_mut()
        {
            page_tlas.frame_references.record_submission(
                *submission_frame_slot,
                *frame_index,
                resource_versions.clone(),
            )?;
        }
        let completion = self.renderer.end_frame(frame)?;
        if completion.swapchain_recreated {
            self.resize_pipeline_to_swapchain(
                input.ucvh.as_deref(),
                input.settings,
                input.restir_di_enabled,
                input.area_restir_enabled,
            )?;
        }

        if initial_upload_completed_this_frame {
            self.ucvh_initial_upload_committed = true;
        }
        if let Some(ucvh) = input.ucvh.as_deref_mut() {
            if self.ucvh_initial_upload_committed
                && let Some(batch_id) = self.ucvh_initial_upload_batch_id
            {
                if ucvh.ack_render_change_batch(batch_id) {
                    self.ucvh_initial_upload_batch_id = None;
                } else {
                    tracing::error!(
                        batch_id,
                        "initial UCVH upload submitted but render change acknowledgement was rejected"
                    );
                }
            }
            if let Some(batch_id) = uploaded_authority_batch_id
                && !ucvh.ack_render_change_batch(batch_id)
            {
                tracing::error!(
                    batch_id,
                    "incremental UCVH upload submitted but render change acknowledgement was rejected"
                );
            }
            if motion_events_uploaded && !ucvh.ack_motion_events(&uploaded_motion_events) {
                tracing::warn!(
                    "UCVH motion event acknowledgement did not match the uploaded snapshot"
                );
            }
            if uploaded_authority_batch_id.is_some()
                && !ucvh.ack_invalidation_regions(&uploaded_invalidation_regions)
            {
                tracing::warn!(
                    "UCVH invalidation acknowledgement did not match the uploaded authority snapshot"
                );
            }
            if self.ucvh_initial_upload_committed
                && ucvh.snapshot_render_change_batch().is_empty()
                && !self.ucvh_uploaded
            {
                self.ucvh_uploaded = true;
                outcome.uploaded_ucvh = true;
                tracing::info!("uploaded UCVH data to GPU and committed incremental catch-up");
            }
        }
        let Some(record_result) = record_result else {
            return Ok(outcome);
        };
        let submitted_fence = record_result.submitted_fence;
        let traversal_stats_requested = record_result.traversal_stats_requested;
        let traversal_stats = record_result.traversal_stats;
        let mut pending_capture = record_result.pending_capture;
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

    fn record_rt_page_tlas_before_trace(
        &mut self,
        command_buffer: ash::vk::CommandBuffer,
        frame_slot: usize,
    ) -> Result<Vec<u64>> {
        if let Some(completed_epoch) = self.renderer.completed_frame_epoch(frame_slot)
            && let Some(page_tlas) = self.rt_page_tlas.as_mut()
            && page_tlas
                .frame_references
                .in_flight_generation(frame_slot)
                .is_some()
        {
            page_tlas
                .frame_references
                .complete_through(frame_slot, completed_epoch)?;
        }

        let required = u32::try_from(self.rt_page_registry.record_count())
            .map_err(|_| anyhow::anyhow!("RT page registry exceeds u32 slot capacity"))?;
        let needs_rebuild = self
            .rt_page_tlas
            .as_ref()
            .is_none_or(|page_tlas| required as usize > page_tlas.instances.instances().len());
        let rebuilt = needs_rebuild;
        if rebuilt {
            if self.rt_page_tlas.is_some() {
                self.renderer
                    .wait_for_other_frame_fences_and_mark_completed(frame_slot)?;
                if let Some(page_tlas) = self.rt_page_tlas.as_mut() {
                    for slot in 0..self.renderer.frame_slot_count() {
                        if page_tlas
                            .frame_references
                            .in_flight_generation(slot)
                            .is_some()
                        {
                            let completed_epoch =
                                self.renderer.completed_frame_epoch(slot).ok_or_else(|| {
                                    anyhow::anyhow!("waited frame slot has no completed epoch")
                                })?;
                            page_tlas
                                .frame_references
                                .complete_through(slot, completed_epoch)?;
                        }
                    }
                    if !page_tlas.frame_references.all_quiescent() {
                        anyhow::bail!(
                            "RT page TLAS frame references remained live after fence waits"
                        );
                    }
                }
            }
            self.rebuild_rt_page_tlas(command_buffer, frame_slot, required)?;
        }

        let compact_bindings = (0..self.rt_page_registry.record_count() as u32)
            .map(|slot| (slot, self.compact_tlas_binding_for_slot(slot)))
            .collect::<Vec<_>>();
        let page_tlas = self
            .rt_page_tlas
            .as_mut()
            .expect("RT page TLAS must exist after lazy initialization or rebuild");
        for (slot, compact_binding) in compact_bindings {
            page_tlas
                .instances
                .sync_registry_slot(&self.rt_page_registry, slot, compact_binding)?;
        }
        if !rebuilt {
            page_tlas.gpu.record_frame_slot_update(
                self.renderer.device(),
                self.renderer
                    .acceleration_structure_loader()
                    .expect("RT support was checked before page TLAS recording"),
                command_buffer,
                self.gpu_profiler.as_ref(),
                frame_slot,
                page_tlas.instances.instances(),
            )?;
        }
        Ok(page_tlas.instances.resource_versions().collect())
    }

    fn compact_tlas_binding_for_slot(
        &self,
        slot: crate::render::rt_page_registry::RtPageSlot,
    ) -> Option<RtPageCompactTlasBinding> {
        let resource_version = match self.rt_page_registry.state_for_slot(slot) {
            Some(crate::render::rt_page_registry::RtPageState::Resident {
                representation: crate::render::rt_page_registry::RtPageRepresentation::CompactExact,
                resource_version,
                ..
            }) => resource_version,
            _ => return None,
        };
        self.rt_page_installed_blas
            .iter()
            .find(|installed| {
                installed.resident.slot == slot
                    && installed.resident.resource_version == resource_version
            })
            .map(|installed| RtPageCompactTlasBinding {
                resource_version,
                blas_address: installed.resources.blas_device_address(),
            })
    }

    fn rebuild_rt_page_tlas(
        &mut self,
        command_buffer: ash::vk::CommandBuffer,
        current_frame_slot: usize,
        required: u32,
    ) -> Result<()> {
        let acceleration_structure_loader = self
            .renderer
            .acceleration_structure_loader()
            .ok_or_else(|| anyhow::anyhow!("RT page TLAS requires acceleration structures"))?;
        let properties = self.renderer.acceleration_structure_properties();
        let device_limit = crate::render::rt_page_tlas::RtPageTlasCapacity::device_limit(
            properties.max_instance_count,
        );
        let capacity = required
            .max(RT_PAGE_INITIAL_TLAS_CAPACITY.min(device_limit))
            .next_power_of_two()
            .min(device_limit);
        if required > capacity {
            anyhow::bail!(
                "RT page registry exceeds device TLAS capacity: required={required} limit={device_limit}"
            );
        }
        let compact_bindings = (0..self.rt_page_registry.record_count() as u32)
            .map(|slot| (slot, self.compact_tlas_binding_for_slot(slot)))
            .collect::<Vec<_>>();
        let mut gpu = RtPageTlasGpuResources::new(
            self.renderer.device(),
            self.renderer.allocator(),
            acceleration_structure_loader,
            RtPageTlasGpuConfig {
                frame_slot_count: self.renderer.frame_slot_count(),
                instance_capacity: capacity,
                max_instance_count: properties.max_instance_count,
                min_acceleration_structure_scratch_offset_alignment: properties
                    .min_acceleration_structure_scratch_offset_alignment
                    .into(),
            },
        )?;
        let build_result = (|| -> Result<(RtPageTlasInstanceStore, RtPageTlasFrameReferences)> {
            let mut instances = RtPageTlasInstanceStore::new(
                capacity,
                gpu.dummy_blas_address(acceleration_structure_loader),
                gpu.reference_blas_address(acceleration_structure_loader),
            )?;
            for &(slot, compact_binding) in &compact_bindings {
                instances.sync_registry_slot(&self.rt_page_registry, slot, compact_binding)?;
            }
            gpu.record_initial_builds(
                self.renderer.device(),
                acceleration_structure_loader,
                command_buffer,
                current_frame_slot,
                instances.instances(),
            )?;
            let frame_references =
                RtPageTlasFrameReferences::new(self.renderer.frame_slot_count())?;
            Ok((instances, frame_references))
        })();
        let (instances, frame_references) = match build_result {
            Ok(resources) => resources,
            Err(error) => {
                gpu.destroy(
                    self.renderer.device(),
                    self.renderer.allocator(),
                    acceleration_structure_loader,
                );
                return Err(error);
            }
        };
        let replacement = RtPageTlasRuntime {
            gpu,
            instances,
            frame_references,
        };
        if let Some(previous) = self.rt_page_tlas.replace(replacement) {
            previous.gpu.destroy(
                self.renderer.device(),
                self.renderer.allocator(),
                acceleration_structure_loader,
            );
        }
        Ok(())
    }

    fn ensure_rt_page_registry_bootstrapped(&mut self, ucvh: Option<&Ucvh>) {
        if self.rt_page_registry_bootstrapped {
            return;
        }
        let Some(ucvh) = ucvh else {
            return;
        };

        let report = self
            .rt_page_registry
            .bootstrap_pages(ucvh.allocated_brick_positions(), 0);
        self.rt_page_registry_bootstrapped = true;
        Self::log_rt_page_queue_report("initial sparse page bootstrap", &report);
    }

    fn log_rt_page_queue_report(context: &'static str, report: &RtPageQueueReport) {
        if report.overflowed() {
            tracing::warn!(
                context,
                invalidated_pages = report.invalidated_pages,
                enqueued_pages = report.enqueued_pages,
                overflowed_pages = report.overflowed_pages,
                "RT page work queue reached capacity; page identities remain durably pending"
            );
        } else if report.invalidated_pages != 0 {
            tracing::debug!(
                context,
                invalidated_pages = report.invalidated_pages,
                enqueued_pages = report.enqueued_pages,
                already_pending_pages = report.already_pending_pages,
                "queued durable RT page topology work"
            );
        }
    }

    /// Process at most `RT_PAGE_BUILDS_PER_FRAME` dirty pages: compute SurfaceMask →
    /// build CompactExact geometry → upload → BLAS BUILD → install → update TLAS slot.
    ///
    /// Must be called BEFORE `record_rt_page_tlas_before_trace` in the same command buffer
    /// so that the BLAS builds are ordered before the TLAS UPDATE via the AS barriers
    /// already emitted by `record_geometry_upload` and `record_build`.
    fn record_rt_page_builds(
        &mut self,
        command_buffer: ash::vk::CommandBuffer,
        frame_slot: usize,
        frame_index: u64,
        ucvh: &Ucvh,
    ) -> Result<()> {
        use crate::render::rt_page_blas::{RtCompactBlasCreateInfo, RtInstalledCompactBlas};
        use crate::render::rt_page_gpu::RtPageGpuConfig;
        use crate::render::rt_page_registry::{RtPageBuildStartError, RtPageRepresentation};
        use crate::render::rt_surface_mask::SurfaceMaskPage;
        use crate::render::rt_page_geometry::RtCompactPageGeometry;

        let Some(accel_loader) = self.renderer.acceleration_structure_loader() else {
            return Ok(());
        };
        let scratch_alignment = self
            .renderer
            .acceleration_structure_properties()
            .min_acceleration_structure_scratch_offset_alignment as ash::vk::DeviceSize;
        let frame_slot_count = self.renderer.frame_slot_count();

        // ── Lazy initialization ──────────────────────────────────────────────────
        if self.rt_page_gpu.is_none() {
            match RtPageGpuResources::new(
                self.renderer.device(),
                self.renderer.allocator(),
                RtPageGpuConfig {
                    index_capacity_bytes: RT_PAGE_GPU_INDEX_CAPACITY_BYTES,
                    face_capacity_records: RT_PAGE_GPU_FACE_CAPACITY_RECORDS,
                    page_record_capacity: RT_PAGE_INITIAL_TLAS_CAPACITY,
                    staging_bytes_per_frame: RT_PAGE_GPU_STAGING_BYTES_PER_FRAME,
                    frame_slot_count,
                },
            ) {
                Ok(gpu) => { self.rt_page_gpu = Some(gpu); }
                Err(e) => {
                    tracing::error!(%e, "failed to create RT page GPU resources");
                    return Ok(());
                }
            }
        }

        // Advance staging tracker for this frame slot.
        let page_gpu = self.rt_page_gpu.as_mut().expect("just initialized");
        if let Err(e) = page_gpu.begin_frame_slot(frame_slot, frame_index) {
            tracing::warn!(%e, "RT page GPU begin_frame_slot failed");
            return Ok(());
        }

        // ── Drain completed retirements (fence-gated) ────────────────────────────
        if let Some(completed_epoch) = self.renderer.completed_frame_epoch(frame_slot) {
            let page_gpu = self.rt_page_gpu.as_mut().expect("initialized");
            if let Err(e) = self.rt_page_retirement_queue.drain_completed(
                completed_epoch,
                self.renderer.device(),
                self.renderer.allocator(),
                &accel_loader,
                page_gpu,
            ) {
                tracing::warn!(%e, "RT page BLAS retirement drain failed");
            }
            // T1-E: free BLAS build scratch buffers whose build submission completed.
            self.rt_page_scratch_free_queue.drain_completed(
                completed_epoch,
                self.renderer.device(),
                self.renderer.allocator(),
            );
        }

        // ── Upload shared lattice once ────────────────────────────────────────────
        let page_gpu = self.rt_page_gpu.as_mut().expect("initialized");
        if !self.rt_page_lattice_uploaded {
            match page_gpu.record_lattice_upload(
                self.renderer.device(),
                command_buffer,
                frame_slot,
                frame_index,
            ) {
                Ok(_) => { self.rt_page_lattice_uploaded = true; }
                Err(e) => { tracing::warn!(%e, "RT page lattice upload failed"); }
            }
            // Lattice not yet on GPU — BLAS builds would fail; wait until next frame.
            return Ok(());
        }

        // ── Build loop: up to RT_PAGE_BUILDS_PER_FRAME pages ─────────────────────
        let mut built = 0;
        while built < RT_PAGE_BUILDS_PER_FRAME {
            let Some(dirty) = self.rt_page_registry.peek_dirty_page() else { break };

            let mask = SurfaceMaskPage::from_ucvh(ucvh, dirty.page);
            let source = mask.source_stamp();
            if !source.matches_ucvh(ucvh) {
                break; // UCVH has already advanced; try again next frame
            }

            let geometry = RtCompactPageGeometry::from_surface_mask(&mask);
            if geometry.faces.is_empty() {
                // No exposed faces — advance registry without GPU work.
                match self.rt_page_registry.begin_build(
                    dirty.page,
                    RtPageRepresentation::CompactExact,
                    source,
                ) {
                    Ok(ticket) => { self.rt_page_registry.fail_build(ticket, frame_index); }
                    Err(RtPageBuildStartError::NotPending | RtPageBuildStartError::NotQueued) => {}
                    Err(e) => { tracing::debug!(?e, page = ?dirty.page, "empty-page build skip"); }
                }
                built += 1;
                continue;
            }

            let face_count = geometry.faces.len() as u32;
            let alloc = match self.rt_page_gpu.as_mut().expect("initialized").allocate_geometry(face_count) {
                Ok(a) => a,
                Err(e) => {
                    tracing::debug!(%e, "geometry arena full; deferring page builds");
                    break;
                }
            };

            let ticket = match self.rt_page_registry.begin_build(
                dirty.page,
                RtPageRepresentation::CompactExact,
                source,
            ) {
                Ok(t) => t,
                Err(e) => {
                    let _ = self.rt_page_gpu.as_mut().expect("initialized").free_geometry(alloc);
                    tracing::debug!(?e, page = ?dirty.page, "begin_build failed");
                    built += 1;
                    continue;
                }
            };

            let page_gpu = self.rt_page_gpu.as_mut().expect("initialized");
            if let Err(e) = page_gpu.record_geometry_upload(
                self.renderer.device(),
                command_buffer,
                frame_slot,
                frame_index,
                alloc,
                &geometry,
            ) {
                tracing::warn!(%e, page = ?dirty.page, "geometry upload failed");
                self.rt_page_registry.fail_build(ticket, frame_index);
                let _ = page_gpu.free_geometry(alloc);
                built += 1;
                continue;
            }

            let blas_result = RtCompactBlasBuildResources::new(
                self.renderer.device(),
                self.renderer.allocator(),
                &accel_loader,
                RtCompactBlasCreateInfo {
                    page_gpu: self.rt_page_gpu.as_ref().expect("initialized"),
                    geometry_allocation: alloc,
                    geometry: &geometry,
                    source,
                    build_generation: ticket.generation,
                    resource_version: alloc.allocation_id,
                    min_acceleration_structure_scratch_offset_alignment: scratch_alignment,
                },
            );

            let mut blas = match blas_result {
                Ok(Some(b)) => b,
                Ok(None) => {
                    self.rt_page_registry.fail_build(ticket, frame_index);
                    built += 1;
                    continue;
                }
                Err(e) => {
                    tracing::warn!(%e, page = ?dirty.page, "BLAS creation failed");
                    self.rt_page_registry.fail_build(ticket, frame_index);
                    let _ = self.rt_page_gpu.as_mut().expect("initialized").free_geometry(alloc);
                    built += 1;
                    continue;
                }
            };

            if let Err(e) = blas.record_build(
                self.renderer.device(),
                &accel_loader,
                command_buffer,
                self.gpu_profiler.as_ref(),
                frame_slot,
            ) {
                tracing::warn!(%e, page = ?dirty.page, "BLAS record_build failed");
                self.rt_page_registry.fail_build(ticket, frame_index);
                built += 1;
                continue;
            }
            blas.mark_submitted().expect("allocated BLAS transitions to submitted");

            // T1-E: the build scratch is consumed only by the build command just
            // recorded into this frame's command buffer, so it is dead once
            // frame_index's fence signals. Hand it to the fence-gated free-queue now
            // instead of letting the resident BLAS pin it for its whole lifetime.
            if let Some(scratch) = blas.take_scratch() {
                self.rt_page_scratch_free_queue.enqueue(scratch, frame_index);
            }

            let blas_address = blas.blas_device_address();
            match blas.install_or_retire(
                &mut self.rt_page_registry,
                ticket,
                source,
                frame_index,
                &mut self.rt_page_retirement_queue,
            ) {
                Ok(installed) => {
                    let slot = installed.resident.slot;
                    let rv   = installed.resident.resource_version;
                    if let Some(page_tlas) = self.rt_page_tlas.as_mut() {
                        let _ = page_tlas.instances.sync_registry_slot(
                            &self.rt_page_registry,
                            slot,
                            Some(crate::render::rt_page_tlas::RtPageCompactTlasBinding {
                                resource_version: rv,
                                blas_address,
                            }),
                        );
                    }
                    tracing::debug!(
                        page = ?dirty.page,
                        faces = geometry.faces.len(),
                        "CompactExact BLAS installed"
                    );
                    self.rt_page_installed_blas.push(installed);
                }
                Err(e) => {
                    tracing::debug!(?e, page = ?dirty.page, "install_or_retire failed");
                }
            }
            built += 1;
        }
        Ok(())
    }

    fn snapshot_ucvh_frame_changes(ucvh: &Ucvh) -> UcvhFrameChanges {
        UcvhFrameChanges::new(
            ucvh.invalidation_regions().to_vec(),
            ucvh.motion_events().to_vec(),
        )
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
        if let Some(page_tlas) = self.rt_page_tlas.take()
            && let Some(acceleration_structure_loader) =
                self.renderer.acceleration_structure_loader()
        {
            page_tlas.gpu.destroy(
                self.renderer.device(),
                self.renderer.allocator(),
                acceleration_structure_loader,
            );
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
    fn rt_page_tlas_sync_keeps_resident_compact_exact_bindings() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let sync = source
            .split("fn record_rt_page_tlas_before_trace")
            .nth(1)
            .and_then(|source| source.split("fn rebuild_rt_page_tlas").next())
            .expect("RT page TLAS synchronization must be present");

        assert!(
            sync.contains("compact_tlas_binding_for_slot(slot)"),
            "resident CompactExact pages must be rebound to their matching BLAS during TLAS synchronization"
        );
    }

    #[test]
    fn ucvh_gpu_capacity_recreation_decision_tracks_cpu_storage_growth() {
        assert!(ucvh_gpu_requires_recreation(3, 4));
        assert!(!ucvh_gpu_requires_recreation(4, 4));
        assert!(!ucvh_gpu_requires_recreation(5, 4));
    }

    #[test]
    fn rt_page_tlas_is_synchronized_after_authority_invalidation_and_before_trace() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn ensure_rt_page_registry_bootstrapped")
            .next()
            .expect("render_frame should end before page-registry helpers");
        let compact = crate::render::source_checks::compact(render_frame);
        let invalidation = compact
            .find("ingest_render_change_batch")
            .expect("authority invalidations must enter the registry");
        let tlas = compact
            .find("self.record_rt_page_tlas_before_trace")
            .expect("page TLAS must be synchronized before tracing");
        let trace = compact
            .find("self.rt_pipeline.record_and_execute_frame")
            .expect("RT trace recording must exist");

        assert!(invalidation < tlas && tlas < trace);
    }

    #[test]
    fn rt_page_tlas_capacity_growth_waits_for_slots_and_rebuilds_instead_of_failing() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let helper = source
            .split("fn record_rt_page_tlas_before_trace")
            .nth(1)
            .expect("page TLAS recording helper must exist")
            .split("fn ensure_rt_page_registry_bootstrapped")
            .next()
            .expect("page TLAS helper must end before registry bootstrap");
        let compact = crate::render::source_checks::compact(helper);

        assert!(compact.contains("wait_for_other_frame_fences_and_mark_completed"));
        assert!(compact.contains("frame_references.all_quiescent()"));
        assert!(compact.contains("rebuild_rt_page_tlas"));
        assert!(!compact.contains("capacitygrowthrequiresaquiescentrebuild"));
    }

    #[test]
    fn rt_page_tlas_rebuild_cleans_up_a_partially_prepared_replacement() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let rebuild = source
            .split("fn rebuild_rt_page_tlas")
            .nth(1)
            .expect("page TLAS rebuild helper must exist")
            .split("fn ensure_rt_page_registry_bootstrapped")
            .next()
            .expect("page TLAS rebuild must end before registry bootstrap");
        let compact = crate::render::source_checks::compact(rebuild);

        assert!(compact.contains("letbuild_result="));
        assert!(compact.contains("Err(error)=>"));
        assert!(compact.contains("gpu.destroy("));
        assert!(compact.contains("returnErr(error)"));
    }

    #[test]
    fn rt_page_tlas_rebuild_frame_does_not_update_the_just_built_tlas() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let helper = source
            .split("fn record_rt_page_tlas_before_trace")
            .nth(1)
            .expect("page TLAS recording helper must exist")
            .split("fn rebuild_rt_page_tlas")
            .next()
            .expect("record helper must end before rebuild helper");
        let compact = crate::render::source_checks::compact(helper);

        assert!(compact.contains("if!rebuilt"));
        let condition = compact.find("if!rebuilt").unwrap();
        let update = compact.find("record_frame_slot_update").unwrap();
        assert!(condition < update);
    }

    #[test]
    fn ucvh_gpu_capacity_is_recovered_before_beginning_a_frame() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before UCVH helpers");
        let compact = crate::render::source_checks::compact(render_frame);
        let recovery = compact
            .find("self.ensure_ucvh_gpu_capacity(input.ucvh.as_deref())?")
            .expect("render_frame must recover undersized UCVH GPU resources");
        let begin = compact
            .find("self.renderer.begin_frame()?")
            .expect("render_frame must begin a frame");

        assert!(
            recovery < begin,
            "UCVH GPU resources must be recovered before begin_frame acquires frame resources"
        );
    }

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
    fn acknowledged_ucvh_frame_change_snapshot_does_not_clear_newer_events() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(glam::UVec3::splat(32)));
        assert!(ucvh.set_voxel(glam::UVec3::new(1, 2, 3), VoxelCell::new(1, 0, [0; 3])));
        let snapshot = RenderRuntime::snapshot_ucvh_frame_changes(&ucvh);
        assert!(ucvh.set_voxel(glam::UVec3::new(2, 2, 3), VoxelCell::new(2, 0, [0; 3])));

        assert!(!ucvh.ack_invalidation_regions(&snapshot.invalidation_regions));
        assert_eq!(ucvh.invalidation_regions().len(), 1);
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
            Some("6"),
            Some("off"),
            Some("3"),
        );

        assert!(parsed.settings.restir_di_enabled);
        assert!(!parsed.settings.restir_gi_enabled);
        assert!(parsed.settings.temporal_denoise_enabled);
        assert!(!parsed.settings.restir_di_spatial_enabled);
        assert_eq!(parsed.settings.restir_di_spatial_sample_count, 4);
        assert_eq!(parsed.settings.restir_gi_initial_candidate_count, 6);
        assert!(!parsed.settings.restir_gi_spatial_enabled);
        assert_eq!(parsed.settings.restir_gi_spatial_sample_count, 3);
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
    fn auto_backend_routes_large_ucvh_scenes_to_vpt_to_avoid_startup_rt_as_stall() {
        assert_eq!(
            frame_render_backend(RenderMode::Auto, true, Some(405_563)),
            RenderBackend::Vpt
        );
        assert_eq!(
            frame_render_backend(RenderMode::Auto, true, Some(4_096)),
            RenderBackend::Rt
        );
        assert_eq!(
            frame_render_backend(RenderMode::Auto, true, None),
            RenderBackend::Vpt
        );
        assert_eq!(
            frame_render_backend(RenderMode::Rt, true, Some(405_563)),
            RenderBackend::Rt
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

        assert!(
            runtime_struct.contains("ucvh_initial_upload"),
            "RenderRuntime must retain initial UCVH upload progress across frames"
        );

        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before UCVH helpers");
        for token in [
            ".begin_frame(",
            ".upload_initial_incremental(",
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
        assert!(
            !render_frame.contains(".upload_all("),
            "RenderRuntime::render_frame must not block a startup frame with the whole Vintessa upload"
        );

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
    fn render_runtime_can_create_ucvh_gpu_resources_after_async_scene_load() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("RenderRuntime::render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before UCVH helpers");

        assert!(
            source.contains("fn ensure_ucvh_gpu_resources"),
            "RenderRuntime needs a lazy UCVH GPU resource path for background scene loading"
        );
        assert!(
            render_frame.contains("self.ensure_ucvh_gpu_capacity("),
            "render_frame should create or resize UCVH GPU resources when the async scene becomes available"
        );
    }

    #[test]
    fn render_runtime_uploads_and_acknowledges_durable_render_change_batches() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before helpers");
        let compact = crate::render::source_checks::compact(render_frame);

        for token in [
            "snapshot_render_change_batch()",
            ".ingest_render_change_batch(&batch,frame.frame_index)",
            ".upload_incremental_changes(",
            "frame.frame_slot",
            "uploaded_authority_batch_id=Some(batch.id)",
            "ack_render_change_batch(batch_id)",
        ] {
            assert!(
                compact.contains(token),
                "runtime must drive durable incremental UCVH upload with {token}"
            );
        }
        assert!(
            !compact.contains("Self::clear_ucvh_frame_changes(ucvh);"),
            "frame-end cleanup must not acknowledge render authority changes"
        );
        let submitted = compact
            .find("letcompletion=self.renderer.end_frame(frame)?;")
            .expect("render frame must submit before acknowledging UCVH work");
        let acknowledged = compact
            .find("ack_render_change_batch(batch_id)")
            .expect("render frame must acknowledge submitted UCVH batch");
        let durable_page_copy = compact
            .find(".ingest_render_change_batch(&batch,frame.frame_index)")
            .expect("RT page invalidations must enter durable registry state");
        assert!(
            compact
                .match_indices(".ingest_render_change_batch(&batch,frame.frame_index)")
                .count()
                >= 2,
            "initial and incremental authority snapshots must both preserve RT page invalidations"
        );
        assert!(
            submitted < acknowledged,
            "UCVH batch acknowledgement must wait until end_frame accepted the submission"
        );
        assert!(
            durable_page_copy < acknowledged,
            "RT page invalidations must be durable before authority acknowledgement"
        );
    }

    #[test]
    fn render_runtime_bootstraps_sparse_rt_pages_before_beginning_a_frame() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let runtime_struct = crate::render::source_checks::compact(
            source
                .split("pub struct RenderRuntime {")
                .nth(1)
                .expect("RenderRuntime should exist")
                .split("impl RenderRuntime")
                .next()
                .expect("RenderRuntime struct should end before its implementation"),
        );
        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before helpers");
        let compact = crate::render::source_checks::compact(render_frame);

        for token in [
            "rt_page_registry:RtPageRegistry",
            "rt_page_registry_bootstrapped:bool",
        ] {
            assert!(
                runtime_struct.contains(token),
                "runtime must retain sparse RT bootstrap state with {token}"
            );
        }
        assert!(
            crate::render::source_checks::compact(&source)
                .contains("bootstrap_pages(ucvh.allocated_brick_positions(),0)"),
            "bootstrap must enumerate allocated sparse brick coordinates even without a change batch"
        );
        let bootstrap = compact
            .find("self.ensure_rt_page_registry_bootstrapped(input.ucvh.as_deref());")
            .expect("render_frame must ensure RT page bootstrap");
        let begin_frame = compact
            .find("letframe=self.renderer.begin_frame()?;")
            .expect("render_frame must begin a frame");
        assert!(
            bootstrap < begin_frame,
            "RT page bootstrap must not depend on a renderable swapchain frame"
        );
    }

    #[test]
    fn initial_ucvh_upload_waits_for_submission_and_incremental_catch_up_before_ready() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let runtime_struct = crate::render::source_checks::compact(
            source
                .split("pub struct RenderRuntime {")
                .nth(1)
                .expect("RenderRuntime should exist")
                .split("impl RenderRuntime")
                .next()
                .expect("RenderRuntime struct should end before its implementation"),
        );
        let render_frame = source
            .split("pub fn render_frame")
            .nth(1)
            .expect("render_frame should exist")
            .split("fn snapshot_ucvh_frame_changes")
            .next()
            .expect("render_frame should end before helpers");
        let compact = crate::render::source_checks::compact(render_frame);

        for token in [
            "ucvh_initial_upload_committed:bool",
            "ucvh_initial_upload_snapshot_taken:bool",
            "initial_upload_completed_this_frame",
            "self.ucvh_initial_upload_committed=true",
            "self.renderer.end_frame(frame)?",
            "snapshot_render_change_batch()",
        ] {
            assert!(
                runtime_struct.contains(token) || compact.contains(token),
                "initial upload lifecycle must include {token}"
            );
        }
        let submitted = compact
            .find("letcompletion=self.renderer.end_frame(frame)?;")
            .expect("initial upload must submit the command buffer");
        let committed = compact
            .find("self.ucvh_initial_upload_committed=true")
            .expect("initial upload must record committed state only after submission");
        assert!(
            submitted < committed,
            "initial upload must not become committed before end_frame succeeds"
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
            .find("self.refresh_render_backend(input.settings.lighting.render_mode,input.ucvh.as_deref().map(|ucvh|ucvh.pool.allocated_count()),);")
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
            .find("self.refresh_render_backend(settings.lighting.render_mode,ucvh.map(|ucvh|ucvh.pool.allocated_count()),);")
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

        assert!(refresh < rt_resize);
        assert!(refresh < vpt_resize);
        assert!(
            rt_resize < ensure_passes,
            "resize must resize existing RT resources before ensuring passes so size changes do not recreate-and-resize the same pass in one turn"
        );
        assert!(
            vpt_resize < ensure_passes,
            "resize must resize existing VPT resources before ensuring passes so size changes do not recreate-and-resize the same pass in one turn"
        );
    }

    #[test]
    fn render_runtime_has_backend_refresh_helper() {
        let source = crate::render::source_checks::read_source("src/render/runtime.rs");
        let runtime_impl = source
            .split("impl RenderRuntime")
            .nth(1)
            .expect("RenderRuntime impl should exist");

        for token in [
            "fn refresh_render_backend(&mut self, requested: RenderMode, ucvh_brick_count: Option<u32>)",
            "frame_render_backend(",
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
