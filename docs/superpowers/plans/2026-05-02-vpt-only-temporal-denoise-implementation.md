# VPT-Only Temporal Denoise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make VPT the only active renderer, then add explicit surface history, temporal reprojection, ReSTIR-DI validation, and SVGF-style denoising.

**Architecture:** The runtime becomes `vpt_surface -> restir_di_initial -> restir_di_temporal -> restir_di_spatial -> vpt_trace -> vpt_temporal -> vpt_atrous -> postprocess -> blit`. VCT-era primary-ray and lighting passes are deleted from active startup, graph construction, shader compilation, settings, profiling, and README. VPT owns all surface, motion, radiance, moments, history, and denoiser resources.

**Tech Stack:** Rust, Vulkan through `ash`, RenderGraph-managed barriers, Slang compute shaders, UCVH GPU buffers, source tests, strict shader compilation.

---

## Research And Open-Source Basis

- ReSTIR DI: Bitterli et al. 2020 defines candidate generation, spatial/temporal reservoir reuse, confidence through sample count/weight, and final unbiased or biased estimators. In this repo the direct-light reuse layer must stay separate from denoising and must validate history against surface state.
- Rearchitecting Spatiotemporal Resampling for Production: Wyman and Panteleev 2021 shows that production ReSTIR needs cache-friendly pass decomposition, fewer rays, and explicit quality/performance parameters. In this repo that maps to split initial/temporal/spatial passes and conservative history caps.
- RTXDI: the open-source SDK includes integrable shader/host sources plus both minimal single-pass and full multi-pass ReSTIR DI samples. In this repo we copy the pass shape and validation discipline, not implementation code.
- SVGF: Schied et al. 2017 uses temporal accumulation, luminance moments/variance, and hierarchical atrous filtering guided by geometry. In this repo the denoiser must operate on HDR radiance/moments before tone mapping.
- NRD: NVIDIA NRD documents production spatiotemporal denoisers that require normal, roughness, viewZ, and motion vector guides for one-path-per-pixel signals. In this repo those guides come from `vpt_surface`, not from legacy VCT G-buffer state.
- Volumetric ReSTIR and volumetric VPT denoise papers: volume rendering work confirms that spatiotemporal reuse and denoising are useful for VPT, but this voxel renderer should first implement opaque-surface-compatible VPT guides because the current UCVH path produces primary voxel hits and material IDs.
- BMFR/SVGF open implementations: use them only as reference for history rejection, feature buffers, and reconstruction staging. Do not import source code or new dependencies.
- Area ReSTIR: useful later for subpixel/lens-area reuse, antialiasing, depth of field, and high-frequency primary-ray details, but it is not a replacement for the current VPT temporal denoise work. See `docs/superpowers/specs/2026-05-02-area-restir-reference-design.md`.

Reference URLs:

- https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/
- https://research.nvidia.com/labs/rtr/publication/wyman2021rearchitecting/
- https://github.com/NVIDIA-RTX/RTXDI
- https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/
- https://github.com/NVIDIA-RTX/NRD
- https://research.nvidia.com/publication/2021-11_fast-volume-rendering-spatiotemporal-reservoir-resampling
- https://arxiv.org/abs/2106.08034
- https://github.com/gztong/BMFR-DXR-Denoiser
- https://research.nvidia.com/labs/rtr/publication/zhang2024area/
- https://github.com/guiqi134/Area-ReSTIR

## File Map

- Modify `src/render/scene_ubo.rs`: remove VCT settings and constants, keep VPT-only render mode value, add denoiser/debug fields later.
- Modify `assets/shaders/shared/scene_common.slang`: mirror the Rust ABI, remove VCT flags/mode constants.
- Modify `src/app.rs`: delete VCT branch, `PrimaryRayPass`, `LightingPass`, VCT resize/drop/startup, and always use VPT startup/postprocess.
- Modify `src/render/passes/mod.rs`: remove `lighting` and `primary_ray`; later add `vpt_surface`, `vpt_temporal`, and `vpt_atrous`.
- Modify `src/render/gpu_profiler.rs`: replace VCT-era scopes with VPT stages.
- Delete active files after code no longer imports them: `src/render/passes/lighting.rs`, `src/render/passes/primary_ray.rs`, `assets/shaders/passes/lighting.slang`, `assets/shaders/passes/primary_ray.slang`, `assets/shaders/shared/vct_common.slang`.
- Modify `build.rs`: either rely on file deletion or add source tests proving deleted shaders no longer produce active `.spv` includes.
- Modify `README.md`: document VPT-only launch/settings and remove VCT-current wording.
- Create `src/render/passes/vpt_surface.rs` and `assets/shaders/passes/vpt_surface.slang`.
- Create `src/render/vpt_history.rs`: Rust ABI structs for surface/history/denoiser uniforms.
- Create `src/render/passes/vpt_temporal.rs`, `src/render/passes/vpt_atrous.rs`, `assets/shaders/passes/vpt_temporal.slang`, and `assets/shaders/passes/vpt_atrous.slang`.
- Modify `src/render/passes/restir_di.rs` and `assets/shaders/passes/restir_di_*.slang`: add surface resources and compatibility rejection.
- Modify `assets/shaders/passes/vpt.slang`: emit noisy radiance/moments and consume final reservoir with visibility validation.
- Modify `assets/shaders/passes/postprocess.slang` and `src/render/passes/postprocess.rs`: input becomes filtered HDR when denoiser is enabled.

## Phase 1: Remove Active VCT Runtime

### Task 1: RED Tests For VPT-Only Settings

**Files:**
- Modify: `src/render/scene_ubo.rs`
- Modify: `assets/shaders/shared/scene_common.slang`

- [x] **Step 1: Add failing settings tests**

Add tests that assert:

```rust
#[test]
fn lighting_settings_default_is_vpt_only() {
    let settings = LightingSettings::default();
    assert_eq!(settings.render_mode, RenderMode::Vpt);
    assert_eq!(settings.gpu_flags() & LIGHTING_FLAG_SHADOWS_ENABLED, LIGHTING_FLAG_SHADOWS_ENABLED);
}

#[test]
fn vct_render_mode_is_rejected() {
    let result = LightingSettings::from_values_report(None, None, Some("vct"), Some("vct"), None, None);
    assert_eq!(result.settings.render_mode, RenderMode::Vpt);
    assert!(result.warnings.iter().any(|warning| warning.variable == "REVOLUMETRIC_RENDER_MODE"));
    assert!(result.warnings.iter().any(|warning| warning.variable == "REVOLUMETRIC_VCT"));
}
```

Expected RED: current defaults are `RenderMode::Vct`, VCT env is accepted, and `REVOLUMETRIC_VCT` is parsed.

- [ ] **Step 2: Remove VCT settings production code**

Remove `LIGHTING_FLAG_VCT_ENABLED`, `RENDER_MODE_VCT`, `RenderMode::Vct`, `LightingSettings::vct_enabled`, `REVOLUMETRIC_VCT` parsing, and `parse_render_mode("vct")`.

- [ ] **Step 3: Update ABI defaults**

Set default render mode to VPT in `GpuSceneUniforms` construction and Slang `scene_common.slang`.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::scene_ubo; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN: scene UBO tests pass and VCT settings are rejected.

### Task 2: RED Tests For Active VCT Absence

**Files:**
- Modify: `src/render/passes/vpt.rs` or new source-test module
- Modify: `src/app.rs`
- Modify: `src/render/passes/mod.rs`

- [ ] **Step 1: Add failing source tests**

Add a source test that reads active source roots and rejects:

```rust
for forbidden in [
    "RenderMode::Vct",
    "LIGHTING_FLAG_VCT_ENABLED",
    "LightingPass",
    "PrimaryRayPass",
    "graph.add_pass(\"lighting\"",
    "graph.add_pass(\"primary_ray\"",
    "vct_common.slang",
] {
    assert!(!active_source.contains(forbidden), "forbidden active VCT token: {forbidden}");
}
```

Expected RED: current `src/app.rs`, `scene_ubo.rs`, `passes/mod.rs`, and shaders contain these tokens.

- [ ] **Step 2: Delete active VCT imports and fields**

Remove `LightingPass` and `PrimaryRayPass` imports, fields, resize handling, drop handling, and startup initialization from `src/app.rs`.

- [ ] **Step 3: Collapse graph construction to VPT-only**

Remove the non-VPT `primary_ray -> lighting -> postprocess` branch. The only normal graph branch should import VPT output, run optional ReSTIR-DI, run `vpt`, run `postprocess`, then blit.

- [ ] **Step 4: Remove pass modules**

Remove `pub mod lighting;` and `pub mod primary_ray;` from `src/render/passes/mod.rs`. Delete the pass source files only after imports are gone.

- [ ] **Step 5: Run targeted source tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected GREEN: active VPT source tests pass.

### Task 3: VPT-Only Startup And README

**Files:**
- Modify: `src/app.rs`
- Modify: `README.md`

- [ ] **Step 1: Add failing tests for startup wording and README**

Source tests should assert:

```rust
assert!(app.contains("initialized VPT pass"));
assert!(!app.contains("initialized primary ray pass"));
assert!(!app.contains("initialized lighting pass"));
assert!(!readme.contains("VCT-first"));
assert!(!readme.contains("REVOLUMETRIC_RENDER_MODE=vct"));
```

- [ ] **Step 2: Update startup messages and postprocess ownership**

Initialize VPT before postprocess, then initialize postprocess from `vpt.output_image` unconditionally. Remove wording that calls VPT a reference/debug-only renderer.

- [ ] **Step 3: Update README**

Document:

```text
REVOLUMETRIC_RENDER_MODE=vpt is accepted for compatibility but VPT is always the renderer.
REVOLUMETRIC_VPT_RESTIR_DI controls direct-light reuse.
REVOLUMETRIC_DENOISER controls the temporal/atrous chain once enabled.
```

No active README line may describe VCT as current/default/supported.

- [ ] **Step 4: Run targeted tests and diff check**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
git diff --check
```

Expected GREEN: CPU/source tests pass and no whitespace errors.

## Phase 2: VPT Surface And History ABI

### Task 4: Surface/History ABI

**Files:**
- Create: `src/render/vpt_history.rs`
- Modify: `src/render/mod.rs`
- Modify: `assets/shaders/shared/vpt_history_common.slang`

- [ ] **Step 1: Add failing ABI tests**

Define and test `GpuVptHistoryUniforms`, `GpuVptSurfacePixel`, and enum values for debug views. Required fields:

```rust
pub struct GpuVptHistoryUniforms {
    pub current_view_proj: [[f32; 4]; 4],
    pub previous_view_proj: [[f32; 4]; 4],
    pub current_resolution: [u32; 2],
    pub previous_resolution: [u32; 2],
    pub current_jitter: [f32; 2],
    pub previous_jitter: [f32; 2],
    pub frame_index: u32,
    pub reset_generation: u32,
    pub flags: u32,
    pub _pad0: u32,
}
```

Expected RED: module does not exist.

- [ ] **Step 2: Implement ABI and Slang mirror**

Use `#[repr(C)]`, `Pod`, `Zeroable`, and offset tests. Slang mirror must include the same field order.

- [ ] **Step 3: Run ABI tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::vpt_history; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

### Task 5: `VptSurfacePass`

**Files:**
- Create: `src/render/passes/vpt_surface.rs`
- Create: `assets/shaders/passes/vpt_surface.slang`
- Modify: `src/render/passes/mod.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Add failing source tests**

Assert resources exist and no legacy primary pass is referenced:

```rust
assert!(shader.contains("RWTexture2D<float4> surface_position_depth"));
assert!(shader.contains("RWTexture2D<float4> surface_normal_roughness"));
assert!(shader.contains("RWTexture2D<float4> surface_albedo_material"));
assert!(shader.contains("RWTexture2D<float4> motion_history"));
assert!(shader.contains("trace_primary_ray("));
assert!(!rust.contains("PrimaryRayPass"));
```

- [ ] **Step 2: Implement pass resources and descriptors**

Create `rgba32f` position/depth, `rgba16f` or `rgba32f` normal/roughness, `rgba16f` albedo/material, and `rgba32f` motion/history images. Bind scene UBO and UCVH buffers exactly as VPT does.

- [ ] **Step 3: Implement shader**

Trace primary voxel hit, write miss as `depth=-1`, valid motion only when previous camera data exists, and write normal/material guides.

- [ ] **Step 4: Insert graph pass**

Declare `vpt_surface` before ReSTIR-DI and VPT. All image barriers are RenderGraph-owned.

## Phase 3: Surface-Aware ReSTIR-DI

### Task 6: Surface Inputs And Compatibility

**Files:**
- Modify: `src/render/passes/restir_di.rs`
- Modify: `assets/shaders/passes/restir_di_initial.slang`
- Modify: `assets/shaders/passes/restir_di_temporal.slang`
- Modify: `assets/shaders/passes/restir_di_spatial.slang`
- Modify: `assets/shaders/shared/restir_di_common.slang`

- [ ] **Step 1: Add source tests**

Tests must detect surface bindings, miss rejection, normal threshold, position threshold, material check, and capped history length.

- [ ] **Step 2: Bind surface images**

Initial reads current surface. Temporal reads current surface plus previous/history surface or motion-history. Spatial reads current surface for neighbor compatibility.

- [ ] **Step 3: Shader rejection**

Reject history/neighbors when miss states differ, previous pixel is out of bounds, normal dot is below threshold, position delta exceeds voxel-scale threshold, or material differs.

- [ ] **Step 4: Update history ownership**

Add explicit history copy/swap after spatial pass. Do not update history implicitly on CPU.

## Phase 4: Temporal Radiance Accumulation

Status 2026-05-02: baseline implemented and runtime-smoke validated. VPT now writes current-frame noisy radiance plus moments, `vpt_temporal` performs history-compatible temporal accumulation, and postprocess reads temporal radiance. Follow-up quality work remains in Phase 5 atrous/debug views.

### Task 7: Noisy Radiance And Moments

**Files:**
- Modify: `src/render/passes/vpt.rs`
- Modify: `assets/shaders/passes/vpt.slang`
- Create: `src/render/passes/vpt_temporal.rs`
- Create: `assets/shaders/passes/vpt_temporal.slang`

- [x] **Step 1: Add failing source/ABI tests**

Assert VPT writes noisy HDR and moments separately, and temporal pass reads current/previous surface, current noisy radiance, previous accumulated radiance, previous moments, and history length.

- [x] **Step 2: Split VPT output**

Rename current accumulation image to `vpt_noisy_radiance` and add `vpt_noisy_moments`. Remove progressive averaging as the quality mechanism.

- [x] **Step 3: Implement temporal accumulation**

Reproject current surface into previous frame, reject incompatible history, accumulate with bounded alpha, update moments and history length.

Additional runtime hardening completed with RED/GREEN tests:

- VPT temporal descriptor rebinding no longer rewrites all frame-slot descriptor sets from the per-frame render path.
- Persistent VPT images now declare previous-frame final layouts correctly: temporal radiance resumes from postprocess read layout, temporal moments resumes from history-copy read layout, and postprocess output is imported as a persistent image with tracked layout state.

## Phase 5: Atrous Denoising And Debug Views

### Task 8: Atrous Chain

**Files:**
- Create: `src/render/passes/vpt_atrous.rs`
- Create: `assets/shaders/passes/vpt_atrous.slang`
- Modify: `src/app.rs`
- Modify: `src/render/scene_ubo.rs`
- Modify: `assets/shaders/shared/scene_common.slang`

- [x] **Step 1: Add settings/debug tests**

Assert `REVOLUMETRIC_DENOISER`, `REVOLUMETRIC_DENOISER_ATROUS_ITERATIONS`, and `REVOLUMETRIC_VPT_DEBUG_VIEW` parse with warnings on invalid values.

Status 2026-05-02: Rust settings and `SceneUniforms`/Slang ABI are in place. The spatial atrous pass and postprocess selection are still pending.

- [ ] **Step 2: Implement ping-pong atrous resources**

Use two HDR images and run 0..5 iterations. Step width should be `1,2,4,8,16`.

- [ ] **Step 3: Implement edge stops**

Weights include normal, depth/world position, material ID, and variance. Debug views expose final/raw/temporal/variance/history/motion/rejection/reservoir.

- [ ] **Step 4: Postprocess filtered output**

Postprocess reads filtered HDR by default, raw/temporal only through explicit debug settings.

## Phase 6: Final Deletion And Verification

### Task 9: Delete Stale VCT Artifacts

**Files:**
- Delete: `src/render/passes/lighting.rs`
- Delete: `src/render/passes/primary_ray.rs`
- Delete: `assets/shaders/passes/lighting.slang`
- Delete: `assets/shaders/passes/primary_ray.slang`
- Delete: `assets/shaders/shared/vct_common.slang`

- [ ] **Step 1: Delete files after active source tests pass**

Use `apply_patch` delete hunks. Do not delete historical docs under `docs/superpowers/plans/2026-04-*`.

- [ ] **Step 2: Run full verification**

Run:

```powershell
cargo fmt
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --lib; cargo build --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
git diff --check
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; cargo run; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; $env:REVOLUMETRIC_VPT_DEBUG_VIEW='history_valid'; cargo run; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES; Remove-Item Env:\REVOLUMETRIC_VPT_DEBUG_VIEW
```

Expected: all commands exit 0. If runtime cannot launch because no display/Vulkan device is available, record the exact error and treat visual validation as still open.

## Completion Criteria

- Active source roots `src`, `assets/shaders`, and `README.md` no longer describe VCT as supported/current/default.
- VPT launches without `REVOLUMETRIC_RENDER_MODE`.
- ReSTIR-DI movement noise is addressed by surface-aware temporal rejection, not by color smoothing.
- Denoiser debug views can show why history is accepted or rejected.
- RenderGraph owns all pass ordering and barriers.
- Full verification is freshly run before any completion claim.
