# VPT NRD Guide Debug Views Phase 2D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the completed VPT NRD guide contract through visible debug views before building the NRD noisy frontend.

**Architecture:** Extend the existing `REVOLUMETRIC_VPT_DEBUG_VIEW` path with NRD guide aliases, route them through `vpt_temporal`, and bind only the additional guide image needed for visualization. The debug output remains unfiltered and uses the already produced guide resources rather than reconstructing values from legacy packed lanes.

**Tech Stack:** Rust scene settings, Rust Vulkan descriptor layout/writes, Slang compute shaders, shader source contract tests.

---

### Task 1: Red Tests

**Files:**
- Modify: `src/render/scene_ubo.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [x] **Step 1: Add parse and routing assertions**

Add a scene settings test that parses:

```rust
let cases = [
    ("nrd_normal_roughness", 20),
    ("nrd_viewz", 21),
    ("nrd_motion", 22),
    ("nrd_motion_z", 23),
];
```

For each case, call `LightingSettings::from_values_report_with_denoiser(..., Some(raw), ...)`, apply the settings to `GpuSceneUniforms`, and assert there are no warnings and `uniforms.vpt_debug_view == expected_gpu_value`.

Add shader source assertions that require:

```rust
"VPT_DEBUG_VIEW_NRD_NORMAL_ROUGHNESS"
"VPT_DEBUG_VIEW_NRD_VIEWZ"
"VPT_DEBUG_VIEW_NRD_MOTION"
"VPT_DEBUG_VIEW_NRD_MOTION_Z"
"RWTexture2D<float> surface_view_z"
"visualize_nrd_normal_roughness"
"visualize_nrd_motion_z"
"surface_view_z[pixel]"
"motion.z"
```

- [x] **Step 2: Run tests to verify failure**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib lighting_settings_parse_nrd_guide_debug_view_aliases vpt_nrd_guide_debug_views_are_routed_to_temporal_output -- --exact
```

Expected: FAIL because the aliases, constants, binding, and temporal routing do not exist yet.

### Task 2: Settings And Shader Constants

**Files:**
- Modify: `src/render/scene_ubo.rs`
- Modify: `assets/shaders/shared/scene_common.slang`

- [x] **Step 1: Add stable debug constants and enum variants**

Append the guide constants after `VPT_DEBUG_VIEW_VOXEL_HIT` without renumbering existing values:

```rust
pub const VPT_DEBUG_VIEW_NRD_NORMAL_ROUGHNESS: u32 = 20;
pub const VPT_DEBUG_VIEW_NRD_VIEWZ: u32 = 21;
pub const VPT_DEBUG_VIEW_NRD_MOTION: u32 = 22;
pub const VPT_DEBUG_VIEW_NRD_MOTION_Z: u32 = 23;
```

Add matching `VptDebugView` variants, `as_gpu_value` arms, parser aliases, and expected-value help text. Add matching Slang constants in `scene_common.slang`.

- [x] **Step 2: Verify parse test passes**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib lighting_settings_parse_nrd_guide_debug_view_aliases -- --exact
```

Expected: PASS.

### Task 3: Temporal Binding And Routing

**Files:**
- Modify: `assets/shaders/passes/vpt_temporal.slang`
- Modify: `src/render/passes/vpt_temporal.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [x] **Step 1: Bind the standalone viewZ guide**

Add `surface_view_z` as a single-channel storage image after `surface_material_roughness` in `vpt_temporal.slang`. Update `VptTemporalPass::descriptor_binding_specs`, descriptor pool storage-image count, and `write_descriptor_sets` image refs to bind `vpt_surface.surface_view_z` at the matching binding.

- [x] **Step 2: Route unfiltered NRD guide views**

Add helpers:

```slang
float3 visualize_nrd_normal_roughness(uint2 pixel, bool valid_surface)
float3 visualize_nrd_motion_z(float motion_z, bool valid_motion)
```

Route:

```slang
VPT_DEBUG_VIEW_NRD_NORMAL_ROUGHNESS -> surface normal xy plus roughness
VPT_DEBUG_VIEW_NRD_VIEWZ -> surface_view_z
VPT_DEBUG_VIEW_NRD_MOTION -> motion.xy
VPT_DEBUG_VIEW_NRD_MOTION_Z -> signed motion.z
```

Each path writes `accumulated_radiance_image` directly and returns before temporal accumulation.

- [x] **Step 3: Verify focused tests pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib lighting_settings_parse_nrd_guide_debug_view_aliases vpt_nrd_guide_debug_views_are_routed_to_temporal_output vpt_temporal_shader_binding_manifest_matches_motion_guide_resources vpt_svgf_descriptor_specs_match_shader_manifests
```

Expected: PASS.

### Task 4: Regression And Commit

**Files:**
- Modify only files from Tasks 1-3.

- [x] **Step 1: Run verification**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
git diff --check
```

Expected: all exit 0.

- [x] **Step 2: Commit relevant files only**

Stage:

```powershell
git add docs/superpowers/plans/2026-05-23-vpt-nrd-guide-debug-views-phase2d.md assets/shaders/shared/scene_common.slang assets/shaders/passes/vpt_temporal.slang src/render/scene_ubo.rs src/render/passes/vpt_temporal.rs src/render/passes/vpt/shader_source_tests.rs
git commit -m "feat: expose VPT NRD guide debug views"
```
