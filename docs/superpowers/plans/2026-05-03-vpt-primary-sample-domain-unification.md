# VPT Primary Sample Domain Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VPT surface history, Area ReSTIR selected samples, ReSTIR-DI surface validation, VPT tracing, and temporal denoising use one coherent primary-ray sample domain.

**Architecture:** Keep Area ReSTIR as the VPT primary sample-area selector, but stop using a pixel-center `vpt_surface` as the final guide after Area ReSTIR changes the actual primary ray. The frame graph becomes `vpt_surface_bootstrap -> area_restir_* -> vpt_surface_selected -> restir_di_* -> vpt -> vpt_temporal -> surface/history update -> postprocess`, where bootstrap surface is only the current-frame target guide for Area ReSTIR and selected surface is the authoritative guide for DI, VPT, and temporal reuse. Valid Area ReSTIR selected samples are replayed through the same film/lens sample contract in `vpt_surface_selected` and `vpt`; invalid/disabled surface fallback remains a stable center ray so guide and motion buffers do not jitter every frame.

**Tech Stack:** Rust 2024, Vulkan through `ash`, existing `RenderGraph`, Slang compute shaders, source tests, strict shader compilation, Area ReSTIR/RTXDI/SVGF/NRD reference constraints.

---

## Research Constraints

- Area ReSTIR reservoirs live in 4D primary ray space: 2D film sample plus 2D lens sample. Public project: <https://research.nvidia.com/labs/rtr/publication/zhang2024area/> and local `target/research/Area-ReSTIR`.
- Area ReSTIR temporal reuse uses fractional motion footprints and validates shifted samples in the current target domain. Local evidence: `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/TemporalResampling_FloatMotion.cs.slang`.
- RTXDI temporal reuse requires previous G-buffer/surface data, motion, depth threshold, normal threshold, and current-domain normalization. Public SDK: <https://github.com/NVIDIA-RTX/RTXDI>; local docs under `target/research/RTXDI/Doc/`.
- SVGF/NRD history accumulation depends on guide buffers matching the surface being shaded: motion, normal, depth, material, history length, and disocclusion rejection. NRD public repo: <https://github.com/NVIDIA-RTX/NRD>.

## Current Root Cause

`assets/shaders/passes/vpt_surface.slang` currently traces the center pixel ray:

```hlsl
float3 origin = scene.pixel_to_ray[3].xyz;
float3 ray_dir = primary_ray_direction(scene, pixel);
```

`assets/shaders/passes/vpt.slang` traces the actual VPT ray from Area ReSTIR when available:

```hlsl
area_restir_pixel_sample(pixel, reservoir.sample_state)
reservoir.sample_state.lens_uv
```

Therefore the surface/motion buffers used by temporal reuse can describe a different primary hit than the ray used for final radiance. That explains screen-wide noise refresh, block-flow artifacts, and vertical/patch instability when Area ReSTIR is enabled.

## File Structure

- Modify `assets/shaders/shared/vpt_primary_sample_common.slang`: add a shared VPT primary-sample contract for Area ReSTIR selected samples and fallback jitter.
- Modify `assets/shaders/passes/vpt.slang`: use the shared primary-sample contract.
- Modify `assets/shaders/passes/vpt_surface.slang`: bind Area ReSTIR selected reservoirs, replay the same valid selected primary ray as VPT, and keep invalid/disabled fallback deterministic for history guides.
- Modify `src/render/passes/vpt_surface.rs`: add bootstrap and selected descriptor sets, disabled Area ReSTIR fallback resources, selected Area ReSTIR descriptor update, and explicit record methods.
- Modify `src/app.rs`: split surface graph usage into bootstrap and selected passes; move ReSTIR-DI after selected surface; make VPT temporal and surface history update consume selected surface.
- Modify `src/render/passes/vpt.rs`: update source tests for the shared primary-sample contract and graph order.
- Modify `src/render/passes/area_restir.rs`: update graph/source tests to distinguish bootstrap and selected surface.

## Task 1: Lock The Contract With RED Tests

**Files:**
- Modify: `src/render/passes/vpt.rs`
- Modify: `src/render/passes/area_restir.rs`

- [ ] **Step 1: Add source tests for shared primary-sample contract**

Add tests that assert:

```rust
#[test]
fn vpt_surface_and_trace_share_area_restir_primary_sample_contract() {
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang").unwrap();
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang").unwrap();
    let common = std::fs::read_to_string("assets/shaders/shared/vpt_primary_sample_common.slang").unwrap();

    for source in [&surface, &vpt] {
        assert!(source.contains("#include \"vpt_primary_sample_common.slang\""));
        assert!(!source.contains("float3 primary_ray_direction(SceneUniforms scene, uint2 pixel)"));
    }
    assert!(surface.contains("vpt_resolve_surface_primary_ray"));
    assert!(vpt.contains("vpt_resolve_area_restir_primary_ray"));
    assert!(common.contains("area_restir_pixel_sample(pixel, reservoir.sample_state)"));
    assert!(common.contains("scene_primary_ray_from_area_sample"));
    assert!(common.contains("vpt_primary_rng_seed"));
    assert!(common.contains("vpt_center_primary_ray"));
}
```

Expected RED: shared file and surface integration do not exist yet.

- [ ] **Step 2: Add source tests for surface pass dual descriptors**

Add tests that assert `VptSurfacePass` has `bootstrap_descriptor_sets`, `selected_descriptor_sets`, `update_area_restir_descriptors`, `record_bootstrap`, `record_selected`, disabled Area ReSTIR fallback resources, and descriptor bindings `10`/`11`.

Expected RED: pass only owns one descriptor set array and has no Area ReSTIR descriptors.

- [ ] **Step 3: Add graph-order tests**

Add tests that assert `src/app.rs` contains:

```text
"vpt_surface_bootstrap"
"vpt_surface_selected"
"vpt_surface.record_bootstrap"
"vpt_surface.record_selected"
"vpt_surface.update_area_restir_descriptors"
```

and that `restir_di_initial`, `vpt`, `vpt_temporal`, and `vpt_surface_history_update` are found after `vpt_surface_selected`, not after the bootstrap surface.

Expected RED: current graph has only `vpt_surface`.

## Task 2: Implement Shared Primary-Sample Shader Contract

**Files:**
- Create: `assets/shaders/shared/vpt_primary_sample_common.slang`
- Modify: `assets/shaders/passes/vpt.slang`

- [ ] **Step 1: Create common helper**

Implement:

```hlsl
#pragma once
#include "scene_common.slang"
#include "area_restir_common.slang"

uint vpt_primary_hash_u32(uint x) { ... }
float vpt_primary_rand01(inout uint state) { ... }
uint vpt_primary_rng_seed(uint2 pixel, SceneUniforms scene) { ... }
ScenePrimaryRay vpt_fallback_primary_ray(uint2 pixel, SceneUniforms scene, inout uint rng_state) { ... }
ScenePrimaryRay vpt_primary_ray_from_area_reservoir(uint2 pixel, SceneUniforms scene, AreaRestirReservoir reservoir) { ... }
ScenePrimaryRay vpt_resolve_area_restir_primary_ray(uint2 pixel, SceneUniforms scene, AreaRestirUniforms area, AreaRestirReservoir reservoir, uint index, inout uint rng_state) { ... }
```

Fallback jitter must match the existing VPT jitter distribution. Area ReSTIR selected samples must call `area_restir_pixel_sample(pixel, reservoir.sample_state)` and `scene_primary_ray_from_area_sample(...)`.

- [ ] **Step 2: Update VPT trace shader**

Replace the local Area ReSTIR resolver with the common helper. Initialize the RNG with `vpt_primary_rng_seed(tid.xy, scene)` so surface and trace use the same fallback ray when no selected reservoir is valid.

- [ ] **Step 3: Run targeted test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt::shader_source_tests::vpt_surface_and_trace_share_area_restir_primary_sample_contract; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 3: Add Area ReSTIR Selected Primary Ray To VPT Surface Pass

**Files:**
- Modify: `assets/shaders/passes/vpt_surface.slang`
- Modify: `src/render/passes/vpt_surface.rs`

- [ ] **Step 1: Update shader resources**

Add:

```hlsl
#include "area_restir_common.slang"
#include "vpt_primary_sample_common.slang"

[[vk::binding(10, 0)]]
ConstantBuffer<AreaRestirUniforms> area_restir;

[[vk::binding(11, 0)]]
StructuredBuffer<AreaRestirReservoir> area_restir_reservoirs;
```

Replace center-ray generation with:

```hlsl
uint rng_state = vpt_primary_rng_seed(pixel, scene);
uint index = pixel.y * scene.resolution.x + pixel.x;
AreaRestirReservoir reservoir = index < area_restir.reservoir_count
    ? area_restir_reservoirs[index]
    : area_restir_invalid_reservoir();
ScenePrimaryRay primary_ray = vpt_resolve_area_restir_primary_ray(pixel, scene, area_restir, reservoir, index, rng_state);
HitResult hit = trace_primary_ray(make_ray(primary_ray.origin, primary_ray.direction), ...);
```

- [ ] **Step 2: Add dual descriptor set arrays in Rust**

`VptSurfacePass` should own:

```rust
bootstrap_descriptor_sets: Vec<vk::DescriptorSet>,
selected_descriptor_sets: Vec<vk::DescriptorSet>,
disabled_area_restir_uniform_buffers: Vec<GpuBuffer>,
disabled_area_restir_reservoir_buffer: GpuBuffer,
```

Bootstrap descriptor sets always bind disabled Area ReSTIR buffers. Selected descriptor sets are initialized with the same disabled buffers and can be rebound to the live Area ReSTIR selected buffer.

- [ ] **Step 3: Add explicit record methods**

Implement:

```rust
pub fn update_area_restir_descriptors(&self, device: &ash::Device, frame_slot: usize, uniform: &GpuBuffer, reservoirs: &GpuBuffer)
pub fn record_bootstrap(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize)
pub fn record_selected(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize)
pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) { self.record_bootstrap(device, cmd, frame_slot); }
```

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt::shader_source_tests::vpt_surface_pass_binds_area_restir_selected_primary_sample; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 4: Rewire The Frame Graph Around Selected Surface

**Files:**
- Modify: `src/app.rs`

- [ ] **Step 1: Rename the first surface pass**

Rename graph pass `"vpt_surface"` to `"vpt_surface_bootstrap"` and record it with `vpt_surface.record_bootstrap(...)`. Area ReSTIR continues to read this bootstrap surface.

- [ ] **Step 2: Add selected surface pass after Area ReSTIR**

When Area ReSTIR is enabled, after the selected reservoir is produced:

```rust
vpt_surface.update_area_restir_descriptors(renderer.device(), frame.frame_slot, area_uniform_buffer, area_selected_current_buffer);
let selected_surface_writes = graph.add_pass("vpt_surface_selected", QueueType::Compute, |builder| {
    builder.read_as(area_uniform_resource, AccessKind::ComputeShaderRead);
    builder.read_as(area_selected_reservoir_resource, AccessKind::ComputeShaderRead);
    builder.write_as(surface_position_dep, AccessKind::ComputeShaderWrite);
    builder.write_as(surface_normal_dep, AccessKind::ComputeShaderWrite);
    builder.write_as(surface_albedo_dep, AccessKind::ComputeShaderWrite);
    builder.write_as(motion_history_dep, AccessKind::ComputeShaderWrite);
    Box::new(move |ctx| vpt_surface.record_selected(ctx.device, ctx.command_buffer, slot))
});
```

When Area ReSTIR is disabled, use bootstrap writes as the final surface writes.

- [ ] **Step 3: Move ReSTIR-DI after final surface**

ReSTIR-DI initial/temporal/spatial must read `final_surface_writes`, not bootstrap writes. This makes direct-light reservoirs validate the same primary hit that VPT traces.

- [ ] **Step 4: Move VPT temporal and history update after final surface**

`vpt_temporal` and `vpt_surface_history_update` must read/copy `final_surface_writes`. Previous surface images remain the previous selected surface because history update runs after temporal.

- [ ] **Step 5: Run targeted graph tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt::shader_source_tests::app_uses_selected_vpt_surface_after_area_restir_for_di_trace_and_temporal; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN.

## Task 5: Full Verification

**Files:** all changed files.

- [ ] **Step 1: Format**

Run:

```powershell
cargo fmt
```

Expected: exit code 0.

- [ ] **Step 2: Source tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: exit code 0.

- [ ] **Step 3: Strict shader compile**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: exit code 0.

- [ ] **Step 4: Clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: exit code 0.

- [ ] **Step 5: Build**

Run:

```powershell
cargo build --bin revolumetric
```

Expected: exit code 0.

- [ ] **Step 6: Runtime smoke if local Vulkan path is available**

Run:

```powershell
$env:REVOLUMETRIC_RENDER_MODE='vpt'
$env:REVOLUMETRIC_VPT_RESTIR_DI='on'
$env:REVOLUMETRIC_RESTIR_DI_SPATIAL='on'
$env:REVOLUMETRIC_AREA_RESTIR='on'
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'
cargo run --bin revolumetric
Remove-Item Env:\REVOLUMETRIC_RENDER_MODE,Env:\REVOLUMETRIC_VPT_RESTIR_DI,Env:\REVOLUMETRIC_RESTIR_DI_SPATIAL,Env:\REVOLUMETRIC_AREA_RESTIR,Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
```

Expected: exits after 3 frames without Vulkan validation or descriptor errors.

## Self-Review

- Spec coverage: primary sample domain, Area ReSTIR, ReSTIR-DI, temporal surface history, and graph barriers are covered by tasks 1-5.
- Placeholder scan: no `TBD`, `TODO`, or "implement later" placeholders remain in the execution steps.
- Type consistency: Rust method names are `update_area_restir_descriptors`, `record_bootstrap`, and `record_selected`; graph pass names are `vpt_surface_bootstrap` and `vpt_surface_selected`.
