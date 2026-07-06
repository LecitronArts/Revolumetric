# Hardware RT + ReSTIR-DI/GI + Temporal History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add a hardware ray tracing backend that becomes the primary renderer on RT-capable devices, then layer ReSTIR-DI and ReSTIR-GI on top, while keeping the final image denoiser temporal-only for this phase.

**Architecture:** Keep the existing VPT path intact as fallback and reference. Add RT capability probing and backend selection to `RenderDevice` and `RenderRuntime`, then implement a separate RT scene backend over occupied brick AABBs, RT surface/hit/history ABI, ray-tracing passes, and a conservative temporal-only accumulation path. The render graph continues to own ordering and barrier validation; RT-specific passes and shaders stay isolated so the first working RT output can be validated before ReSTIR is turned on.

**Tech Stack:** Rust 2024, Vulkan via `ash`, Slang ray tracing shaders, existing `RenderGraph`, `GpuBuffer`/`GpuImage`, `cargo test`, `cargo clippy`, strict shader compilation.

**Completion status:** Implemented and verified in the final consolidated plan commit. The per-task commit steps below were consolidated because implementation happened in one long-running workspace with interdependent RT build, shader, runtime, and validation changes. Final verification covered formatting, whitespace, skip/strict shader test suites, all-target clippy, desktop build, explicit RT smoke, default `auto` smoke, and residual scans for NRD/SER/path-guiding exclusions.

---

## File Structure

- Create `src/render/rt_capabilities.rs`: RT extension/feature/property detection and a pure resolver for requested-vs-supported RT backend selection.
- Create `src/render/rt_settings.rs`: RT temporal settings, RT debug views, env parsing, and RT-specific GPU ABI for the runtime.
- Create `src/render/rt_scene.rs`: procedural brick-AABB acceleration-structure backend, dirty-brick tracking, and AS rebuild generation.
- Create `src/render/rt_history.rs`: surface/hit/history GPU ABI and reset flags for the RT temporal pass.
- Create `src/render/rt_pipeline.rs`: RT backend orchestration, pass ordering, SBT ownership, and frame recording.
- Create `src/render/restir_gi.rs`: RT GI settings, ABI, and CPU-side helpers.
- Modify `src/render/device.rs`, `src/render/runtime.rs`, `src/render/scene_ubo.rs`, `src/render/resource.rs`, `src/render/pass_context.rs`, `src/render/graph.rs`, `src/render/descriptor.rs`, `src/assets/shader_reflect.rs`, `src/build_support.rs`, `build.rs`, `src/app.rs`, `src/render/passes/mod.rs`.
- Create `src/render/passes/rt_surface.rs`, `rt_temporal.rs`, `rt_resolve.rs`, `rt_restir_di.rs`, `rt_restir_gi.rs`.
- Create `assets/shaders/shared/rt_common.slang`, `rt_history_common.slang`, `restir_gi_common.slang`.
- Create `assets/shaders/passes/rt_surface.rgen.slang`, `rt_surface.rmiss.slang`, `rt_surface.rchit.slang`, `rt_surface.rint.slang`, `rt_temporal.rgen.slang`, `rt_resolve.rgen.slang`, `rt_restir_di.rgen.slang`, `rt_restir_gi.rgen.slang`.
- Modify `README.md` for runtime environment documentation.

## Source Facts To Preserve

- The shipped runtime is still VPT-first; RT must be additive, not a rewrite of VPT.
- `RenderRuntime` already owns app-to-render delegation, so the new RT backend should plug into that boundary instead of bypassing it.
- Existing compute ReSTIR-DI and Area ReSTIR code are useful references for reservoir layout and history conventions, but they are not the hardware RT backend.
- NRD stays out of this phase; temporal-only history is the only stabilization pass for now.
- SER and path guiding remain explicitly deferred.

## Recommendation

Do this in the following order:

1. RT capability probing and render-mode routing.
2. RT build/reflection/graph support.
3. RT scene backend and surface/history ABI.
4. RT pipeline skeleton and backend routing.
5. ReSTIR-DI on the RT backend.
6. ReSTIR-GI on the RT backend.
7. Temporal-only accumulation and final resolve.

That sequencing keeps the first visible result debuggable before lighting reuse is enabled.

### Task 1: RT Capability Probe and Render-Mode Routing

**Files:**
- Create: `src/render/rt_capabilities.rs`
- Create: `src/render/rt_settings.rs`
- Modify: `src/render/scene_ubo.rs`
- Modify: `src/render/device.rs`
- Modify: `src/render/runtime.rs`
- Modify: `src/app.rs`
- Test: `src/render/scene_ubo.rs`, `src/render/rt_capabilities.rs`, `src/render/runtime.rs`

- [x] **Step 1: Write the failing settings and backend-selection tests**

Add these tests first:

```rust
#[test]
fn lighting_settings_parse_rt_render_mode() {
    let parsed = LightingSettings::from_values_report(
        None,
        None,
        Some("rt"),
        None,
        None,
        None,
    );
    assert_eq!(parsed.settings.render_mode, RenderMode::Rt);
    assert!(parsed.warnings.is_empty());
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
    );
    assert!(parsed.settings.restir_di_enabled);
    assert!(!parsed.settings.restir_gi_enabled);
    assert!(parsed.settings.temporal_denoise_enabled);
    assert_eq!(parsed.settings.history_length, 32);
    assert_eq!(parsed.settings.normal_threshold, 0.85);
    assert_eq!(parsed.settings.depth_threshold, 0.02);
    assert_eq!(parsed.settings.debug_view, RtDebugView::Surface);
    assert!(parsed.warnings.is_empty());
}

#[test]
fn runtime_resolves_rt_to_vpt_when_hardware_support_is_missing() {
    assert_eq!(
        resolve_render_backend(RenderMode::Rt, false),
        RenderBackend::Vpt
    );
}
```

- [x] **Step 2: Run the focused tests to verify they fail**

Run:

```powershell
cargo test render::scene_ubo::tests::lighting_settings_parse_rt_render_mode --lib
cargo test render::rt_settings::tests::rt_settings_parse_valid_overrides --lib
cargo test render::runtime::tests::runtime_resolves_rt_to_vpt_when_hardware_support_is_missing --lib
```

Expected: compile or test failure because `RenderMode::Rt`, `RtSettings`, and backend resolution do not exist yet.

- [x] **Step 3: Implement the minimal RT mode and settings plumbing**

Add the RT-facing types and routing helpers:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Vpt,
    Rt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtDebugView {
    Off,
    Surface,
    HitDistance,
    HistoryValid,
    DirectReservoir,
    IndirectReservoir,
    Temporal,
    Final,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtSettings {
    pub restir_di_enabled: bool,
    pub restir_gi_enabled: bool,
    pub temporal_denoise_enabled: bool,
    pub history_length: u32,
    pub normal_threshold: f32,
    pub depth_threshold: f32,
    pub debug_view: RtDebugView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBackend {
    Vpt,
    Rt,
}

pub fn resolve_render_backend(requested: RenderMode, rt_supported: bool) -> RenderBackend {
    match (requested, rt_supported) {
        (RenderMode::Rt, true) => RenderBackend::Rt,
        _ => RenderBackend::Vpt,
    }
}
```

Wire `RenderDevice` to report RT support without requiring it, and make `RenderRuntime` choose the backend from `LightingSettings::render_mode` plus the RT capability flag.

- [x] **Step 4: Run the focused tests to verify they pass**

Run:

```powershell
cargo test render::scene_ubo::tests::lighting_settings_parse_rt_render_mode --lib
cargo test render::rt_settings::tests::rt_settings_parse_valid_overrides --lib
cargo test render::runtime::tests::runtime_resolves_rt_to_vpt_when_hardware_support_is_missing --lib
```

Expected: all three tests pass.

- [x] **Step 5: Commit**

```powershell
git add src/render/scene_ubo.rs src/render/rt_capabilities.rs src/render/rt_settings.rs src/render/device.rs src/render/runtime.rs src/app.rs
git commit -m "feat: add RT mode routing and settings plumbing"
```

### Task 2: Shader Build, Reflection, and Descriptor Support

**Files:**
- Modify: `src/build_support.rs`
- Modify: `build.rs`
- Modify: `src/assets/shader_reflect.rs`
- Modify: `src/render/descriptor.rs`
- Modify: `src/render/resource.rs`
- Modify: `src/render/pass_context.rs`
- Modify: `src/render/graph.rs`
- Test: `src/build_support.rs`, `src/assets/shader_reflect.rs`, `src/render/descriptor.rs`, `src/render/resource.rs`, `src/render/pass_context.rs`, `src/render/graph.rs`

- [x] **Step 1: Write the failing stage-mapping, reflection, and graph tests**

Add these tests first:

```rust
#[test]
fn rt_shader_jobs_map_suffixes_to_ray_tracing_stages() {
    let jobs = rt_shader_jobs(Path::new("assets/shaders"));
    assert!(jobs.iter().any(|job| job.path.ends_with("rt_surface.rgen.slang")
        && job.stage == "raygeneration"));
    assert!(jobs.iter().any(|job| job.path.ends_with("rt_surface.rmiss.slang")
        && job.stage == "miss"));
    assert!(jobs.iter().any(|job| job.path.ends_with("rt_surface.rchit.slang")
        && job.stage == "closesthit"));
}

#[test]
fn shader_reflection_parses_acceleration_structure_descriptors() {
    let source = r#"
[[vk::binding(0, 0)]]
AccelerationStructureKHR scene_tlas;
"#;
    let reflection = ShaderReflection::from_slang_source("main", source).unwrap();
    assert_eq!(reflection.bindings[0].kind, DescriptorKind::AccelerationStructure);
}

#[test]
fn create_image_as_records_custom_access_kind() {
    let mut builder = PassBuilder::new("rt_surface", QueueType::RayTracing, 0);
    let handle = builder.create_image_as(
        128,
        128,
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE,
        AccessKind::RayTracingShaderWrite,
    );
    assert_eq!(builder.accesses[0].kind, AccessKind::RayTracingShaderWrite);
    assert_eq!(handle.id, 0);
}

#[test]
fn compile_plans_ray_tracing_barriers_between_trace_and_temporal_passes() {
    let mut graph = RenderGraph::new();
    let surface = graph.add_pass("rt_surface", QueueType::RayTracing, |builder| {
        builder.create_image_as(
            128,
            128,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            AccessKind::RayTracingShaderWrite,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(surface, fake_image(7));
    graph.add_pass("rt_temporal", QueueType::RayTracing, |builder| {
        builder.read_as(surface, AccessKind::RayTracingShaderRead);
        Box::new(|_ctx| {})
    });
    graph.compile().unwrap();
    let barriers = graph.barrier_plan();
    assert_eq!(barriers.len(), 1);
    assert_eq!(barriers[0].from, AccessKind::RayTracingShaderWrite);
    assert_eq!(barriers[0].to, AccessKind::RayTracingShaderRead);
}
```

- [x] **Step 2: Run the focused tests to verify they fail**

Run:

```powershell
cargo test build_support::tests::rt_shader_jobs_map_suffixes_to_ray_tracing_stages --lib
cargo test assets::shader_reflect::tests::shader_reflection_parses_acceleration_structure_descriptors --lib
cargo test render::descriptor::tests::assert_specs_match_shader_bindings_accepts_acceleration_structure_khr --lib
cargo test render::pass_context::tests::create_image_as_records_custom_access_kind --lib
cargo test render::graph::tests::compile_plans_ray_tracing_barriers_between_trace_and_temporal_passes --lib
```

Expected: failures because stage-aware RT shader jobs, acceleration-structure descriptor parsing, RT access kinds, and RT-aware graph builders do not exist yet.

- [x] **Step 3: Implement the RT-aware build and descriptor plumbing**

Add the missing RT support:

```rust
pub enum DescriptorKind {
    UniformBuffer,
    StorageBuffer,
    StorageImage,
    SampledImage,
    Sampler,
    AccelerationStructure,
}

pub fn rt_shader_jobs(shader_dir: &Path) -> Vec<ShaderJobSpec> {
    // Detect `.rgen.slang`, `.rmiss.slang`, `.rchit.slang`, `.rahit.slang`,
    // `.rint.slang`, and `.rcall.slang` files and map them to the matching
    // Slang stage names.
}

pub fn create_image_as(
    &mut self,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    kind: AccessKind,
) -> ResourceHandle {
    // Same as create_image, but records the supplied access kind.
}
```

Update `build.rs` so RT shader files compile with the correct Slang stages, update shader reflection to understand `AccelerationStructureKHR`, and extend `AccessKind` with RT shader and acceleration-structure build stage/access masks.

- [x] **Step 4: Run the focused tests to verify they pass**

Run:

```powershell
cargo test build_support::tests::rt_shader_jobs_map_suffixes_to_ray_tracing_stages --lib
cargo test assets::shader_reflect::tests::shader_reflection_parses_acceleration_structure_descriptors --lib
cargo test render::descriptor::tests::assert_specs_match_shader_bindings_accepts_acceleration_structure_khr --lib
cargo test render::pass_context::tests::create_image_as_records_custom_access_kind --lib
cargo test render::graph::tests::compile_plans_ray_tracing_barriers_between_trace_and_temporal_passes --lib
```

Expected: all five tests pass.

- [x] **Step 5: Commit**

```powershell
git add build.rs src/build_support.rs src/assets/shader_reflect.rs src/render/descriptor.rs src/render/resource.rs src/render/pass_context.rs src/render/graph.rs
git commit -m "feat: add RT shader build and graph support"
```

### Task 3: RT Scene Backend and Surface/History ABI

**Files:**
- Create: `src/render/rt_scene.rs`
- Create: `src/render/rt_history.rs`
- Modify: `src/render/runtime.rs`
- Test: `src/render/rt_scene.rs`, `src/render/rt_history.rs`

- [x] **Step 1: Write the failing scene and history ABI tests**

Add these tests first:

```rust
#[test]
fn collect_occupied_brick_bounds_deduplicates_bricks() {
    let mut ucvh = Ucvh::new(UcvhConfig::new(glam::UVec3::splat(32)));
    assert!(ucvh.set_voxel(glam::UVec3::new(1, 2, 3), VoxelCell::new(1, 0, [0; 3])));
    assert!(ucvh.set_voxel(glam::UVec3::new(9, 2, 3), VoxelCell::new(1, 0, [0; 3])));
    ucvh.rebuild_hierarchy();
    let bounds = collect_occupied_brick_bounds(&ucvh);
    assert_eq!(bounds.len(), 2);
}

#[test]
fn rt_history_uniforms_layout_is_stable() {
    assert_eq!(std::mem::size_of::<GpuRtHistoryUniforms>(), 192);
    assert_eq!(std::mem::offset_of!(GpuRtHistoryUniforms, current_view_proj), 0);
    assert_eq!(std::mem::offset_of!(GpuRtHistoryUniforms, frame_index), 160);
    assert_eq!(std::mem::offset_of!(GpuRtHistoryUniforms, flags), 184);
}

#[test]
fn rt_history_flags_cover_camera_cut_resize_scene_change_and_as_rebuild() {
    let all = RT_HISTORY_FLAG_CAMERA_CUT
        | RT_HISTORY_FLAG_RESIZE
        | RT_HISTORY_FLAG_SCENE_INVALIDATED
        | RT_HISTORY_FLAG_AS_REBUILT
        | RT_HISTORY_FLAG_LIGHTS_INVALIDATED;
    assert_eq!(all.count_ones(), 5);
}
```

- [x] **Step 2: Run the focused tests to verify they fail**

Run:

```powershell
cargo test render::rt_scene::tests::collect_occupied_brick_bounds_deduplicates_bricks --lib
cargo test render::rt_history::tests::rt_history_uniforms_layout_is_stable --lib
cargo test render::rt_history::tests::rt_history_flags_cover_camera_cut_resize_scene_change_and_as_rebuild --lib
```

Expected: failures because the RT scene backend and history ABI do not exist yet.

- [x] **Step 3: Implement the procedural AS backend and RT history ABI**

Add the RT scene backend over occupied brick AABBs and a dedicated history ABI:

```rust
pub struct RtSceneBackend {
    pub build_generation: u32,
    // Own the TLAS/BLAS handles, scratch buffers, and per-frame rebuild state.
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRtHistoryUniforms {
    pub current_view_proj: [[f32; 4]; 4],
    pub previous_view_proj: [[f32; 4]; 4],
    pub current_resolution: [u32; 2],
    pub previous_resolution: [u32; 2],
    pub frame_index: u32,
    pub history_reset_generation: u32,
    pub as_rebuild_generation: u32,
    pub flags: u32,
}
```

Keep the AS backend outside the render graph, but feed its rebuild generation into the RT history ABI so camera cut, resize, scene invalidation, light changes, and AS rebuilds can invalidate accumulation explicitly.

- [x] **Step 4: Run the focused tests to verify they pass**

Run:

```powershell
cargo test render::rt_scene::tests::collect_occupied_brick_bounds_deduplicates_bricks --lib
cargo test render::rt_history::tests::rt_history_uniforms_layout_is_stable --lib
cargo test render::rt_history::tests::rt_history_flags_cover_camera_cut_resize_scene_change_and_as_rebuild --lib
```

Expected: all three tests pass.

- [x] **Step 5: Commit**

```powershell
git add src/render/rt_scene.rs src/render/rt_history.rs src/render/runtime.rs
git commit -m "feat: add RT scene backend and history ABI"
```

### Task 4: RT Pipeline Skeleton and Runtime Integration

**Files:**
- Create: `src/render/rt_pipeline.rs`
- Create: `src/render/passes/rt_surface.rs`
- Create: `src/render/passes/rt_temporal.rs`
- Create: `src/render/passes/rt_resolve.rs`
- Modify: `src/render/passes/mod.rs`
- Modify: `src/render/runtime.rs`
- Test: `src/render/rt_pipeline.rs`, `src/render/passes/rt_surface.rs`, `src/render/passes/rt_temporal.rs`

- [x] **Step 1: Write the failing pipeline and shader-source tests**

Add these tests first:

```rust
#[test]
fn rt_pipeline_registers_surface_then_temporal_then_resolve_in_order() {
    let source = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
    let surface = source.find("rt_surface");
    let temporal = source.find("rt_temporal");
    let resolve = source.find("rt_resolve");
    assert!(surface.is_some() && temporal.is_some() && resolve.is_some());
    assert!(surface.unwrap() < temporal.unwrap());
    assert!(temporal.unwrap() < resolve.unwrap());
}

#[test]
fn rt_surface_shader_source_declares_tlas_and_rt_history_outputs() {
    let source = std::fs::read_to_string("assets/shaders/passes/rt_surface.rgen.slang")
        .expect("rt_surface.rgen.slang should be readable");
    assert!(source.contains("rt_history_common.slang"));
    assert!(source.contains("AccelerationStructureKHR"));
    assert!(source.contains("GpuRtSurfacePixel"));
    assert!(source.contains("GpuRtHistoryUniforms"));
}

#[test]
fn rt_temporal_shader_source_uses_conservative_reprojection_and_clamp() {
    let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
        .expect("rt_temporal.rgen.slang should be readable");
    assert!(source.contains("reproject"));
    assert!(source.contains("clamp"));
    assert!(source.contains("history_reset_generation"));
}
```

- [x] **Step 2: Run the focused tests to verify they fail**

Run:

```powershell
cargo test render::rt_pipeline::tests::rt_pipeline_registers_surface_then_temporal_then_resolve_in_order --lib
cargo test render::passes::rt_surface::shader_source_tests --lib
cargo test render::passes::rt_temporal::shader_source_tests --lib
```

Expected: failures because the RT pipeline and shader files do not exist yet.

- [x] **Step 3: Implement the RT pipeline shell and no-op shaders**

Create the RT backend shell and the first compileable shaders:

```rust
pub struct RtRuntimePipeline {
    rt_surface_pass: Option<RtSurfacePass>,
    rt_temporal_pass: Option<RtTemporalPass>,
    rt_resolve_pass: Option<RtResolvePass>,
}

impl RtRuntimePipeline {
    pub fn record_and_execute_frame(&mut self, /* rt inputs */) -> Result<()> {
        // Surface -> temporal -> resolve
        Ok(())
    }
}
```

The first shader pass should only prove that `vkCmdTraceRaysKHR` is wired and that the RT output images and history buffers compile. Keep VPT untouched as the fallback/reference path.

- [x] **Step 4: Run the focused tests to verify they pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test render::rt_pipeline::tests::rt_pipeline_registers_surface_then_temporal_then_resolve_in_order --lib; cargo test render::passes::rt_surface::shader_source_tests --lib; cargo test render::passes::rt_temporal::shader_source_tests --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: the RT pipeline skeleton tests and shader source tests pass.

- [x] **Step 5: Commit**

```powershell
git add src/render/rt_pipeline.rs src/render/passes/mod.rs src/render/passes/rt_surface.rs src/render/passes/rt_temporal.rs src/render/passes/rt_resolve.rs assets/shaders/passes/rt_surface.rgen.slang assets/shaders/passes/rt_temporal.rgen.slang assets/shaders/passes/rt_resolve.rgen.slang
git commit -m "feat: add RT pipeline skeleton"
```

### Task 5: ReSTIR-DI on the RT Backend

**Files:**
- Create: `src/render/passes/rt_restir_di.rs`
- Modify: `src/render/passes/mod.rs`
- Create: `assets/shaders/shared/restir_di_common.slang` if the RT backend needs a shared variant
- Create: `assets/shaders/passes/rt_restir_di.rgen.slang`
- Test: `src/render/passes/rt_restir_di.rs`, `src/render/restir_di.rs`

- [x] **Step 1: Write the failing RT DI tests**

Add these tests first:

```rust
#[test]
fn rt_restir_di_shader_source_uses_restir_di_common_and_rt_surface_inputs() {
    let source = std::fs::read_to_string("assets/shaders/passes/rt_restir_di.rgen.slang")
        .expect("rt_restir_di.rgen.slang should be readable");
    assert!(source.contains("restir_di_common.slang"));
    assert!(source.contains("rt_history_common.slang"));
    assert!(source.contains("GpuDirectLight"));
    assert!(source.contains("GpuRestirDiReservoir"));
}

#[test]
fn rt_restir_di_pass_keeps_spatial_reuse_disabled_in_phase_one() {
    let source = crate::render::source_checks::read_source("src/render/passes/rt_restir_di.rs");
    assert!(!source.contains("spatial reuse"));
    assert!(!source.contains("spatial_pass"));
}
```

- [x] **Step 2: Run the focused tests to verify they fail**

Run:

```powershell
cargo test render::passes::rt_restir_di::shader_source_tests --lib
```

Expected: failure because the RT DI pass and shader do not exist yet.

- [x] **Step 3: Implement RT DI using the existing direct-light table and reservoir ABI**

Create the RT DI pass and keep it temporal-only in this phase:

```rust
pub struct RtRestirDiPass {
    // direct-light table, reservoirs, per-frame uniforms, and RT raygen pipeline
}

impl RtRestirDiPass {
    pub fn record(&self, /* rt inputs */) {
        // Generate candidates, trace visibility, and reuse the previous frame temporally.
    }
}
```

Reuse the existing `GpuDirectLight`, `GpuRestirDiReservoir`, and
`RestirDiSettings` CPU ABI. The initial plan deferred spatial reuse, but the
current implementation includes RT ReSTIR-DI spatial reuse and the default RT
startup profile enables it unless explicitly disabled; keep it isolated from the
temporal-only GI and final denoise path.

- [x] **Step 4: Run the focused tests to verify they pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test render::passes::rt_restir_di::shader_source_tests --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: RT DI source tests pass.

- [x] **Step 5: Commit**

```powershell
git add src/render/passes/rt_restir_di.rs assets/shaders/passes/rt_restir_di.rgen.slang assets/shaders/shared/restir_di_common.slang
git commit -m "feat: add RT ReSTIR-DI pass"
```

### Task 6: ReSTIR-GI on the RT Backend

**Files:**
- Create: `src/render/restir_gi.rs`
- Create: `src/render/passes/rt_restir_gi.rs`
- Modify: `src/render/passes/mod.rs`
- Create: `assets/shaders/shared/restir_gi_common.slang`
- Create: `assets/shaders/passes/rt_restir_gi.rgen.slang`
- Test: `src/render/restir_gi.rs`, `src/render/passes/rt_restir_gi.rs`

- [x] **Step 1: Write the failing GI settings and shader-source tests**

Add these tests first:

```rust
#[test]
fn restir_gi_settings_default_to_conservative_values() {
    let settings = RestirGiSettings::default();
    assert!(!settings.enabled);
    assert!(!settings.temporal_enabled);
    assert_eq!(settings.history_length, 20);
    assert_eq!(settings.debug_view, RestirGiDebugView::Off);
}

#[test]
fn rt_restir_gi_shader_source_uses_gi_common_and_no_path_guiding() {
    let source = std::fs::read_to_string("assets/shaders/passes/rt_restir_gi.rgen.slang")
        .expect("rt_restir_gi.rgen.slang should be readable");
    assert!(source.contains("restir_gi_common.slang"));
    assert!(source.contains("rt_history_common.slang"));
    assert!(!source.contains("path guiding"));
    assert!(!source.contains("SER"));
}
```

- [x] **Step 2: Run the focused tests to verify they fail**

Run:

```powershell
cargo test render::restir_gi::tests::restir_gi_settings_default_to_conservative_values --lib
cargo test render::passes::rt_restir_gi::shader_source_tests --lib
```

Expected: failure because GI settings, ABI, and RT GI shaders do not exist yet.

- [x] **Step 3: Implement the GI settings and RT GI pass**

Add a dedicated GI settings/ABI module and a temporal-only RT GI pass:

```rust
pub struct RestirGiSettings {
    pub enabled: bool,
    pub temporal_enabled: bool,
    pub initial_candidate_count: u32,
    pub history_length: u32,
    pub max_bounces: u32,
    pub debug_view: RestirGiDebugView,
}

pub struct RtRestirGiPass {
    // indirect reservoirs, per-frame uniforms, and RT raygen pipeline
}
```

The first version should stay conservative: no SER, no path guiding, and no spatial reuse.

- [x] **Step 4: Run the focused tests to verify they pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test render::restir_gi::tests::restir_gi_settings_default_to_conservative_values --lib; cargo test render::passes::rt_restir_gi::shader_source_tests --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: GI settings and RT GI shader source tests pass.

- [x] **Step 5: Commit**

```powershell
git add src/render/restir_gi.rs src/render/passes/rt_restir_gi.rs assets/shaders/passes/rt_restir_gi.rgen.slang assets/shaders/shared/restir_gi_common.slang
git commit -m "feat: add RT ReSTIR-GI pass"
```

### Task 7: Temporal-Only Resolve and History Invalidation

**Files:**
- Modify: `src/render/rt_history.rs`
- Modify: `src/render/rt_temporal.rs`
- Modify: `src/render/rt_pipeline.rs`
- Modify: `src/render/runtime.rs`
- Modify: `src/app.rs`
- Test: `src/render/rt_history.rs`, `src/render/rt_temporal.rs`, `src/render/runtime.rs`

- [x] **Step 1: Write the failing temporal and invalidation tests**

Add these tests first:

```rust
#[test]
fn rt_temporal_invalidates_on_camera_cut_resize_scene_change_and_as_rebuild() {
    let flags = RT_HISTORY_FLAG_CAMERA_CUT
        | RT_HISTORY_FLAG_RESIZE
        | RT_HISTORY_FLAG_SCENE_INVALIDATED
        | RT_HISTORY_FLAG_AS_REBUILT;
    assert_eq!(flags.count_ones(), 4);
}

#[test]
fn rt_temporal_accumulates_hdr_without_claiming_nrd_behavior() {
    let source = crate::render::source_checks::read_source("src/render/rt_temporal.rs");
    assert!(!source.contains("NRD"));
    assert!(!source.contains("ReBLUR"));
    assert!(!source.contains("RELAX"));
}

#[test]
fn runtime_resets_rt_history_when_backend_or_scene_generation_changes() {
    let source = crate::render::source_checks::read_source("src/render/runtime.rs");
    assert!(source.contains("history_reset_generation"));
    assert!(source.contains("as_rebuild_generation"));
}
```

- [x] **Step 2: Run the focused tests to verify they fail**

Run:

```powershell
cargo test render::rt_history::tests::rt_temporal_invalidates_on_camera_cut_resize_scene_change_and_as_rebuild --lib
cargo test render::rt_temporal::tests::rt_temporal_accumulates_hdr_without_claiming_nrd_behavior --lib
cargo test render::runtime::tests::runtime_resets_rt_history_when_backend_or_scene_generation_changes --lib
```

Expected: failures because the RT temporal pass and invalidation plumbing are not complete yet.

- [x] **Step 3: Implement conservative reprojection, clamp, and accumulation**

Keep the RT temporal pass deliberately conservative:

```rust
pub fn should_reset_rt_history(flags: u32, camera_cut: bool, resize: bool, scene_change: bool, as_rebuild: bool) -> bool {
    camera_cut
        || resize
        || scene_change
        || as_rebuild
        || (flags & RT_HISTORY_FLAG_LIGHTS_INVALIDATED) != 0
}

pub fn accumulate_rt_history(/* inputs */) {
    // Reproject, reject mismatched history, clamp growth, and accumulate HDR.
}
```

Wire the RT backend so a backend switch, AS rebuild, or scene change resets history explicitly.

- [x] **Step 4: Run the focused tests to verify they pass**

Run:

```powershell
cargo test render::rt_history::tests::rt_temporal_invalidates_on_camera_cut_resize_scene_change_and_as_rebuild --lib
cargo test render::rt_temporal::tests::rt_temporal_accumulates_hdr_without_claiming_nrd_behavior --lib
cargo test render::runtime::tests::runtime_resets_rt_history_when_backend_or_scene_generation_changes --lib
```

Expected: all three tests pass.

- [x] **Step 5: Commit**

```powershell
git add src/render/rt_history.rs src/render/rt_temporal.rs src/render/rt_pipeline.rs src/render/runtime.rs src/app.rs
git commit -m "feat: add RT temporal-only resolve"
```

### Task 8: End-to-End Validation and Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-28-hardware-rt-restir-temporal-design.md` only if the spec needs a correction after implementation
- No code changes unless validation exposes a regression

- [x] **Step 1: Update runtime docs**

Document:

- `REVOLUMETRIC_RENDER_MODE=auto|rt|vpt`
- `REVOLUMETRIC_RT_RESTIR_DI=on|off`
- `REVOLUMETRIC_RT_RESTIR_GI=on|off`
- `REVOLUMETRIC_RT_TEMPORAL_DENOISE=on|off`
- `REVOLUMETRIC_RT_TEMPORAL_HISTORY_LENGTH=1..64`
- `REVOLUMETRIC_RT_TEMPORAL_NORMAL_THRESHOLD`
- `REVOLUMETRIC_RT_TEMPORAL_DEPTH_THRESHOLD`
- `REVOLUMETRIC_RT_DEBUG_VIEW`

- [x] **Step 2: Run the full validation suite**

Run:

```powershell
cargo fmt
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
git diff --check
```

Expected:

- Unit tests pass.
- Clippy has no warnings.
- Strict shader compilation succeeds.
- `git diff --check` has no whitespace or patch-format errors.

- [x] **Step 3: Run the runtime smoke checks**

Run on RT-capable hardware:

```powershell
$env:REVOLUMETRIC_RENDER_MODE='rt'
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'
.\target\debug\revolumetric.exe
Remove-Item Env:\REVOLUMETRIC_RENDER_MODE
Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
```

Run the fallback smoke when RT is unavailable:

```powershell
$env:REVOLUMETRIC_RENDER_MODE='rt'
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'
.\target\debug\revolumetric.exe
Remove-Item Env:\REVOLUMETRIC_RENDER_MODE
Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
```

Expected:

- RT-capable hardware runs the RT backend.
- Non-RT hardware falls back to VPT with a warning and keeps rendering.

- [x] **Step 4: Run residual scans**

Run:

```powershell
rg -n -i "nrd|reblur|relax|path guiding|SER" src/render assets/shaders README.md docs
rg -n "cmd_pipeline_barrier|ImageMemoryBarrier|BufferMemoryBarrier" src/render/passes assets/shaders
```

Expected:

- RT files do not introduce NRD, SER, or path-guiding tokens.
- RT passes do not bypass the render graph with ad hoc barriers.
