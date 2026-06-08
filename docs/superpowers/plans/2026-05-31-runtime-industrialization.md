# Runtime Industrialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move GPU runtime orchestration from `app.rs` into a focused `RenderRuntime` while preserving current rendering behavior.

**Architecture:** `app.rs` keeps window, input, ECS, camera, and CPU demo scene coordination. `src/render/runtime.rs` owns `RenderDevice`, profiler, capture, scene UBO, UCVH GPU resources, VPT pipeline, frame rendering, resize handling, and GPU teardown ordering. App-to-runtime communication uses explicit settings/input/outcome structs.

**Tech Stack:** Rust, winit, ash/Vulkan, anyhow, existing Revolumetric ECS/render modules, source-level regression tests.

---

## File Structure

- Create `src/render/runtime.rs`: `RuntimeSettings`, `RenderFrameInput`, `RenderFrameOutcome`, `RenderRuntime`, runtime initialization, frame rendering, resize, teardown, and runtime source tests.
- Modify `src/render/mod.rs`: export the new `runtime` module.
- Modify `src/app.rs`: replace direct GPU resource fields with `render_runtime: Option<RenderRuntime>`, delegate frame and resize work, keep app-level settings and CPU UCVH ownership.
- Modify `docs/superpowers/specs/2026-05-31-runtime-industrialization-design.md`: already done during design.
- Modify `docs/superpowers/plans/2026-05-31-runtime-industrialization.md`: mark steps as complete while implementing.

---

### Task 1: Add Runtime Source Guard Tests

**Files:**
- Modify: `src/app.rs`
- Create: `src/render/runtime.rs`
- Modify: `src/render/mod.rs`

- [ ] **Step 1: Create the runtime module shell**

Add this module export to `src/render/mod.rs`:

```rust
pub mod runtime;
```

Create `src/render/runtime.rs` with source-level tests first:

```rust
#[cfg(test)]
mod tests {
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
}
```

- [ ] **Step 2: Update the existing app ownership test expectation**

In `src/app.rs`, update `app_delegates_vpt_pass_ownership_to_runtime_pipeline` to expect the new runtime boundary:

```rust
assert!(app_struct.contains("render_runtime: Option<RenderRuntime>"));
assert!(!app_struct.contains("vpt_pipeline: VptRuntimePipeline"));
assert!(!source.contains("self.vpt_pipeline.record_and_execute_frame("));
```

Keep the existing forbidden pass assertions.

- [ ] **Step 3: Run the focused failing tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::runtime::tests::app_delegates_gpu_runtime_ownership_to_render_runtime app::tests::app_delegates_vpt_pass_ownership_to_runtime_pipeline; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: fail because `RenderRuntime` is not implemented and `app.rs` still owns the old fields.

---

### Task 2: Implement RenderRuntime API and Ownership

**Files:**
- Modify: `src/render/runtime.rs`

- [ ] **Step 1: Add imports and public data structs**

Implement these definitions at the top of `src/render/runtime.rs`:

```rust
use anyhow::{Context, Result};
use winit::window::Window;

use crate::render::area_restir::AreaRestirSettings;
use crate::render::capture::RenderCapture;
use crate::render::device::RenderDevice;
use crate::render::gpu_profiler::{GpuProfiler, GpuProfilerConfig};
use crate::render::restir_di::RestirDiSettings;
use crate::render::scene_ubo::LightingSettings;
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
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderFrameOutcome {
    pub began_frame: bool,
    pub rendered: bool,
    pub uploaded_ucvh: bool,
    pub uploaded_motion_events: u32,
    pub wrote_capture: bool,
}
```

- [ ] **Step 2: Add RenderRuntime fields and helpers**

Add:

```rust
pub struct RenderRuntime {
    renderer: RenderDevice,
    gpu_profiler: Option<GpuProfiler>,
    capture: Option<RenderCapture>,
    scene_ubo: Option<crate::render::scene_ubo::SceneUniformBuffer>,
    ucvh_gpu: Option<UcvhGpuResources>,
    ucvh_uploaded: bool,
    vpt_pipeline: VptRuntimePipeline,
}

impl RenderRuntime {
    pub fn device(&self) -> &RenderDevice {
        &self.renderer
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

    fn scene_ubo(&self) -> &crate::render::scene_ubo::SceneUniformBuffer {
        self.scene_ubo
            .as_ref()
            .expect("scene UBO should exist while RenderRuntime is active")
    }
}
```

- [ ] **Step 3: Run compile check**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::runtime::tests::app_delegates_gpu_runtime_ownership_to_render_runtime; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile succeeds for the new types, test still fails until app migration.

---

### Task 3: Move Initialization Into RenderRuntime

**Files:**
- Modify: `src/render/runtime.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Implement `RenderRuntime::new`**

Add the initialization method:

```rust
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
            renderer.physical_device_properties().limits.timestamp_period,
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

        let scene_ubo = match crate::render::scene_ubo::SceneUniformBuffer::new(
            renderer.device(),
            renderer.allocator(),
            renderer.swapchain_image_count(),
        ) {
            Ok(ubo) => {
                tracing::info!(frame_count = renderer.swapchain_image_count(), "created scene UBO");
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

        let mut runtime = Self {
            renderer,
            gpu_profiler,
            capture,
            scene_ubo: Some(scene_ubo),
            ucvh_gpu,
            ucvh_uploaded: false,
            vpt_pipeline: VptRuntimePipeline::new(),
        };
        runtime.ensure_passes(ucvh, settings, settings.restir_di.enabled, settings.area_restir.enabled);
        Ok(runtime)
    }
}
```

- [ ] **Step 2: Replace app imports and fields**

In `src/app.rs`, remove imports for direct GPU runtime types and add:

```rust
use crate::render::runtime::{RenderFrameInput, RenderRuntime, RuntimeSettings};
use crate::render::vpt_pipeline::VptCameraFrame;
```

Change the app struct fields from direct GPU resources to:

```rust
render_runtime: Option<RenderRuntime>,
ucvh: Option<Ucvh>,
lighting_settings: LightingSettings,
area_restir_settings: AreaRestirSettings,
restir_di_settings: RestirDiSettings,
```

Initialize with:

```rust
render_runtime: None,
```

- [ ] **Step 3: Add an app settings helper**

Add to `impl RevolumetricApp`:

```rust
fn runtime_settings(&self) -> RuntimeSettings {
    RuntimeSettings {
        lighting: self.lighting_settings,
        restir_di: self.restir_di_settings,
        area_restir: self.area_restir_settings,
    }
}
```

- [ ] **Step 4: Migrate `resumed` to construct `RenderRuntime`**

After CPU UCVH generation and settings parsing, replace direct renderer/profiler/capture/scene UBO/UCVH GPU/pass initialization with:

```rust
let render_runtime = match RenderRuntime::new(&window, self.runtime_settings(), self.ucvh.as_ref())
{
    Ok(runtime) => runtime,
    Err(error) => {
        tracing::error!(%error, "failed to initialize render runtime");
        event_loop.exit();
        return;
    }
};

self.render_runtime = Some(render_runtime);
self.window = Some(window);
self.window_id = Some(window_id);
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::runtime::tests::app_delegates_gpu_runtime_ownership_to_render_runtime app::tests::app_delegates_vpt_pass_ownership_to_runtime_pipeline; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile may fail until frame and resize migration are complete.

---

### Task 4: Move Frame Rendering and Resize Into RenderRuntime

**Files:**
- Modify: `src/render/runtime.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Implement `ensure_passes` and `resize`**

Add:

```rust
impl RenderRuntime {
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
        self.ensure_passes(ucvh, settings, restir_di_enabled, area_restir_enabled);
        if let (Some(scene_ubo), Some(ucvh_gpu)) = (self.scene_ubo.as_ref(), self.ucvh_gpu.as_ref()) {
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
        }
        Ok(())
    }
}
```

- [ ] **Step 2: Implement `render_frame`**

Move the existing frame body from `app.rs` into runtime form:

```rust
impl RenderRuntime {
    pub fn render_frame(&mut self, input: RenderFrameInput<'_>) -> Result<RenderFrameOutcome> {
        let mut outcome = RenderFrameOutcome::default();
        let frame = self.renderer.begin_frame()?;
        outcome.began_frame = true;

        if frame.should_render {
            if let Some(profiler) = &mut self.gpu_profiler {
                profiler.begin_frame(
                    self.renderer.device(),
                    frame.command_buffer,
                    frame.frame_slot,
                    frame.frame_index,
                );
            }

            let mut ucvh_ref = input.ucvh;
            if !self.ucvh_uploaded {
                if let (Some(ucvh), Some(gpu)) = (ucvh_ref.as_deref_mut(), &self.ucvh_gpu) {
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
                ucvh_ref
                    .as_deref()
                    .map(Self::snapshot_ucvh_frame_changes)
                    .unwrap_or_default()
            } else {
                UcvhFrameChanges::default()
            };

            let mut ucvh_motion_event_count = 0u32;
            if self.ucvh_uploaded
                && let (Some(ucvh), Some(gpu)) = (ucvh_ref.as_deref(), &self.ucvh_gpu)
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
                self.renderer.end_frame(frame)?;
                return Ok(outcome);
            };

            let record_result = self.vpt_pipeline.record_and_execute_frame(
                &self.renderer,
                &frame,
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
            self.renderer.end_frame(frame)?;

            if self.ucvh_uploaded && let Some(ucvh) = ucvh_ref.as_deref_mut() {
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
        }

        Ok(outcome)
    }
}
```

- [ ] **Step 3: Simplify `tick_frame`**

Replace the direct render block in `app.rs` with:

```rust
if let Some(runtime) = self.render_runtime.as_mut() {
    runtime.render_frame(RenderFrameInput {
        camera,
        sun_direction,
        sun_intensity,
        elapsed_seconds,
        settings: self.runtime_settings(),
        restir_di_enabled,
        area_restir_enabled,
        ucvh: self.ucvh.as_mut(),
    })?;
}
```

- [ ] **Step 4: Simplify resize event handling**

Replace direct resize calls in `WindowEvent::Resized` with:

```rust
if let Some(runtime) = self.render_runtime.as_mut() {
    if let Err(error) = runtime.resize(
        size.width,
        size.height,
        self.ucvh.as_ref(),
        self.runtime_settings(),
        self.restir_di_vpt_enabled(),
        self.area_restir_vpt_enabled(),
    ) {
        tracing::error!(%error, "failed to resize render runtime");
        event_loop.exit();
        return;
    }
}
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::runtime::tests::app_delegates_gpu_runtime_ownership_to_render_runtime app::tests::app_delegates_vpt_pass_ownership_to_runtime_pipeline; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: both focused tests pass.

---

### Task 5: Move GPU Teardown and Final Verification

**Files:**
- Modify: `src/render/runtime.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Implement runtime `Drop`**

Add:

```rust
impl Drop for RenderRuntime {
    fn drop(&mut self) {
        unsafe { self.renderer.device().device_wait_idle().ok() };
        if let Some(profiler) = self.gpu_profiler.take() {
            profiler.destroy(self.renderer.device());
        }
        if let Some(capture) = self.capture.take() {
            capture.destroy(self.renderer.device(), self.renderer.allocator());
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
```

- [ ] **Step 2: Remove app `Drop` GPU teardown**

Delete the custom `Drop for RevolumetricApp` or reduce it to app-owned cleanup only. `RenderRuntime` should handle all GPU resources through its own drop ordering.

- [ ] **Step 3: Run formatting**

Run:

```powershell
cargo fmt
```

Expected: exits successfully.

- [ ] **Step 4: Run full library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all library tests pass.

- [ ] **Step 5: Inspect diff for scope**

Run:

```powershell
git diff -- src/app.rs src/render/mod.rs src/render/runtime.rs docs/superpowers/specs/2026-05-31-runtime-industrialization-design.md docs/superpowers/plans/2026-05-31-runtime-industrialization.md
```

Expected: only runtime-boundary files and docs changed by this phase.

---

## Execution Results

- Runtime boundary implemented in `src/render/runtime.rs`.
- App orchestration migrated to `render_runtime: Option<RenderRuntime>`.
- Source-level tests updated so app tests guard app ownership and render/pass tests guard runtime or pipeline ownership.
- Verification command used after implementation:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

- Result observed during execution after the final runtime guard test was added: 441 passed, 0 failed.
