# Runtime Industrialization Design

## Goal

Move GPU runtime orchestration out of `app.rs` and into a focused render runtime boundary. The first industrialization phase should make the app loop thinner, centralize GPU lifecycle ownership, and create a stable seam for subsequent RenderGraph/resource lifecycle work without changing rendering algorithms, shader behavior, or pass scheduling semantics.

## Current Facts

- `RevolumetricApp` currently owns `RenderDevice`, `GpuProfiler`, `RenderCapture`, `VptRuntimePipeline`, `UcvhGpuResources`, the UCVH upload flag, and `SceneUniformBuffer`.
- `tick_frame` directly begins frames, starts profiling, uploads UCVH data and motion guides, snapshots/clears UCVH changes, calls `VptRuntimePipeline::record_and_execute_frame`, ends the frame, and writes captures.
- `resumed` directly initializes the Vulkan device, profiler, capture, scene UBO, UCVH GPU resources, and VPT passes.
- `README.md` already identifies that `app.rs` owns too much runtime orchestration and that RenderGraph still lacks full transient allocation and descriptor automation.
- Existing source checks already assert that `app.rs` does not own individual VPT frame graph passes.

## Phase Scope

This phase creates the runtime boundary and moves orchestration into it. It must not change:

- VPT, ReSTIR, NRD, denoising, or postprocess algorithms.
- Shader source or native NRD adapter behavior.
- RenderGraph pass order, resource barriers, descriptor layout, or execution semantics.
- ECS stages, window creation policy, input handling, or camera controls.

Later phases can use this boundary to tackle RenderGraph transient allocation, descriptor automation, async compute scheduling, and shader/native build boundaries.

## Architecture

Add `src/render/runtime.rs` with a `RenderRuntime` that owns the GPU-side runtime:

- `RenderDevice`
- `Option<GpuProfiler>`
- `Option<RenderCapture>`
- `SceneUniformBuffer`
- `Option<UcvhGpuResources>`
- `VptRuntimePipeline`
- UCVH upload state

`app.rs` remains responsible for:

- Window and event-loop ownership.
- ECS world and schedule execution.
- Input, touch, camera update, and app-level settings.
- CPU-side demo UCVH generation.
- Exit-after-frame policy.

The runtime receives app-owned state through explicit input structs and returns explicit outcomes. It should not reach into `World`, `Schedule`, `WindowDescriptor`, or input resources.

## Public API

`RuntimeSettings` centralizes environment-derived render settings:

```rust
pub struct RuntimeSettings {
    pub lighting: LightingSettings,
    pub restir_di: RestirDiSettings,
    pub area_restir: AreaRestirSettings,
}
```

`RenderFrameInput` captures one frame's app-provided render inputs:

```rust
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
```

`RenderFrameOutcome` reports what happened without exposing pass internals:

```rust
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderFrameOutcome {
    pub began_frame: bool,
    pub rendered: bool,
    pub uploaded_ucvh: bool,
    pub uploaded_motion_events: u32,
    pub wrote_capture: bool,
}
```

The core methods are:

```rust
impl RenderRuntime {
    pub fn new(window: &winit::window::Window, settings: RuntimeSettings, ucvh: Option<&Ucvh>) -> anyhow::Result<Self>;
    pub fn ensure_passes(&mut self, ucvh: Option<&Ucvh>, settings: RuntimeSettings, restir_di_enabled: bool, area_restir_enabled: bool);
    pub fn render_frame(&mut self, input: RenderFrameInput<'_>) -> anyhow::Result<RenderFrameOutcome>;
    pub fn resize(&mut self, width: u32, height: u32, ucvh: Option<&Ucvh>, settings: RuntimeSettings, restir_di_enabled: bool, area_restir_enabled: bool) -> anyhow::Result<()>;
    pub fn device(&self) -> &RenderDevice;
}
```

`Drop` for `RenderRuntime` owns GPU teardown ordering: wait for the device, destroy profiler, capture, VPT pipeline, UCVH GPU resources, scene UBO, and then let `RenderDevice` drop.

## Initialization Flow

`app.rs::resumed` should:

1. Create the window.
2. Parse lighting/ReSTIR settings and apply the Area ReSTIR debug view mapping.
3. Generate the CPU demo UCVH if missing.
4. Create `RenderRuntime::new(&window, settings, self.ucvh.as_ref())`.
5. Store the runtime, window, and window id.
6. Run the startup stage once.

`RenderRuntime::new` should:

1. Create `RenderDevice`.
2. Log renderer properties from the device.
3. Create `GpuProfiler` from `GpuProfilerConfig::from_env()`.
4. Create `RenderCapture` from environment configuration.
5. Try to create `SceneUniformBuffer`; log and continue without rendering if creation fails, matching current app behavior.
6. Try to create optional `UcvhGpuResources` when a CPU UCVH is available; log and continue without GPU UCVH upload if creation fails, matching current app behavior.
7. Create `VptRuntimePipeline` and ensure its passes.

Initialization errors that are fatal today remain fatal to app startup. Optional profiler/capture errors keep the existing behavior: log a warning and continue disabled.

## Frame Flow

`app.rs::tick_frame` should keep the ECS and camera flow:

1. Advance time.
2. Run `PreUpdate`, `Update`, `PostUpdate`.
3. Update camera.
4. Run `ExtractRender`, `PrepareRender`.
5. Build `RenderFrameInput`.
6. Call `runtime.render_frame(input)`.
7. Clear per-frame input.
8. Run `ExecuteRender`.

`RenderRuntime::render_frame` should:

1. Begin the Vulkan frame.
2. Start profiler timestamps if profiling is enabled and the frame should render.
3. Upload UCVH data once when both CPU and GPU UCVH resources exist.
4. Snapshot UCVH invalidation and motion events only after the first upload.
5. Upload the motion guide and report its count.
6. Call `VptRuntimePipeline::record_and_execute_frame`.
7. End the frame.
8. Clear UCVH frame changes after successful render work.
9. Wait for capture fence and write capture metadata/output when requested.

If the frame is acquired but no render work should be submitted, the runtime returns an outcome with `began_frame = true` and `rendered = false`.

## Resize Flow

`app.rs::window_event` should keep the zero-size minimized guard, then delegate non-zero resize work to `RenderRuntime::resize`. The runtime should call `RenderDevice::handle_resize` and then resize the VPT pipeline when UCVH GPU resources exist. This preserves the current behavior while removing direct renderer and pass ownership from `app.rs`.

## Error Handling

- `RenderDevice::new` remains fatal through `RenderRuntime::new`.
- `SceneUniformBuffer::new` and `UcvhGpuResources::new` remain non-fatal: log the error and keep the runtime alive, matching current app behavior.
- UCVH upload and motion-guide upload errors remain non-fatal and are logged, matching current app behavior.
- `record_and_execute_frame`, `end_frame`, fence wait, and capture writes remain fallible and propagate as `anyhow::Result`.
- Optional profiler/capture setup failures log warnings and disable those features.

## Testing

The phase should add or update source-level regression tests because most runtime ownership changes require Vulkan objects that unit tests cannot instantiate in CI:

- `app.rs` must own `render_runtime: Option<RenderRuntime>` and must not own `RenderDevice`, `GpuProfiler`, `RenderCapture`, `VptRuntimePipeline`, `UcvhGpuResources`, `ucvh_uploaded`, or `SceneUniformBuffer` fields.
- `app.rs` must call `render_frame(` and must not call `record_and_execute_frame(`, `begin_frame(`, `end_frame(`, `upload_all(`, or `upload_motion_guide(`.
- `app.rs` must call `resize(` through the runtime and must not call `handle_resize(` or `VptRuntimePipeline::resize`.
- `render/runtime.rs` must own `VptRuntimePipeline`, call `record_and_execute_frame`, and handle UCVH upload and capture write paths.
- Existing RenderGraph and shader source tests must continue passing.

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

## Acceptance Criteria

- `app.rs` is materially smaller and no longer directly orchestrates GPU frame recording.
- GPU runtime resources have a single teardown owner.
- Runtime inputs and outputs are explicit structs, not ad hoc argument lists spread through `app.rs`.
- Existing behavior and logs remain equivalent unless the log ownership name changes from app to runtime.
- The full library test suite passes with shader compilation skipped.

## Known Deferred Risk

`RenderDevice::begin_frame` and `RenderDevice::end_frame` can recreate the swapchain internally on out-of-date or suboptimal presentation paths. This phase preserved explicit window-resize behavior by routing `WindowEvent::Resized` through `RenderRuntime::resize`; the follow-up `Swapchain Lifecycle Signal` phase makes internal swapchain recreation observable and routes those signals through the same runtime resize synchronization path.
