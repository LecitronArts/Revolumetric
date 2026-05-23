# VPT NRD Guide Roughness Phase 2A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit surface material roughness guide for VPT denoising without breaking the existing SVGF emissive-firefly handling.

**Architecture:** Keep `surface_normal_roughness.w` as the current SVGF emissive hint for this slice, and add a separate `surface_material_roughness` current/previous guide image. Surface generation writes deterministic material roughness from shader material helpers; temporal and A-trous consume that guide for compatibility and edge weights. This creates a true roughness source for later NRD normal/roughness packing while avoiding a semantic flip that would make every rough diffuse surface bypass firefly clamping.

**Tech Stack:** Rust, Vulkan/ash render graph resources, Slang compute shaders, Cargo tests with `REVOLUMETRIC_SHADER_COMPILE=skip`.

---

## Scope

In scope:
- Add deterministic material roughness helpers in `assets/shaders/shared/material_common.slang`.
- Add `surface_material_roughness` and `previous_surface_material_roughness` images to `VptSurfacePass`.
- Expand surface graph resource arrays from current 6 to 7 and previous 4 to 5.
- Bind `surface_material_roughness` in `vpt_surface.slang`, `vpt_temporal.slang`, and `vpt_atrous.slang`.
- Use roughness in temporal surface compatibility and A-trous guide weighting.
- Preserve current `surface_normal_roughness.w` emissive hint until the SVGF firefly clamp has a separate emissive signal.

Out of scope:
- `viewZ` output.
- Moving `motion_id` out of `motion_history.z`.
- NRD packed guide pass.
- NRD SDK integration.
- Diffuse/specular signal split.

## File Structure

- Modify `assets/shaders/shared/material_common.slang`
  - Owns material roughness table and helper functions.
- Modify `assets/shaders/passes/vpt_surface.slang`
  - Writes the explicit material roughness guide.
- Modify `assets/shaders/passes/vpt_temporal.slang`
  - Reads current and previous roughness guides and rejects incompatible history.
- Modify `assets/shaders/passes/vpt_atrous.slang`
  - Reads current roughness guide and includes roughness in spatial guide weight.
- Modify `src/render/passes/vpt_surface.rs`
  - Allocates, binds, registers, copies, resizes, and destroys roughness guide images.
- Modify `src/render/passes/vpt_temporal.rs`
  - Binds roughness guide images to the temporal shader and graph.
- Modify `src/render/passes/vpt_atrous.rs`
  - Binds roughness guide images to the A-trous shader and graph.
- Modify `src/render/passes/vpt/shader_source_tests.rs`
  - Adds source-contract tests before implementation.

## Review Cadence

- Review after Task 1: material helper semantics.
- Review after Task 2: surface pass resource/descriptor/graph contract.
- Review after Task 3: temporal and A-trous consumers.
- Full review after Task 4 verification before continuing to viewZ/motion work.

## Task 1: Material Roughness Helpers

**Files:**
- Modify: `assets/shaders/shared/material_common.slang`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Write failing source test**

Add this test to `src/render/passes/vpt/shader_source_tests.rs`:

```rust
#[test]
fn material_common_declares_deterministic_roughness_helpers() {
    let material = source("assets/shaders/shared/material_common.slang");

    for token in [
        "static const float MATERIAL_ROUGHNESS[8]",
        "float material_roughness(uint material_id)",
        "float material_cell_roughness(VoxelCell cell)",
        "float material_emissive_luminance(VoxelCell cell)",
        "return MATERIAL_ROUGHNESS[min(material_id, 7u)]",
    ] {
        assert!(material.contains(token), "material common missing {token}");
    }
}
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib material_common_declares_deterministic_roughness_helpers; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: FAIL because the material roughness helpers do not exist.

- [ ] **Step 3: Add roughness helpers**

In `assets/shaders/shared/material_common.slang`, after `MATERIAL_ALBEDO`, add:

```slang
static const float MATERIAL_ROUGHNESS[8] = {
    1.00, // 0: default diffuse
    0.95, // 1: stone
    0.90, // 2: red cloth
    0.90, // 3: green cloth
    0.85, // 4: blue
    0.88, // 5: brick
    0.98, // 6: dark stone
    0.72, // 7: wood
};
```

After `material_albedo`, add:

```slang
float material_roughness(uint material_id) {
    return MATERIAL_ROUGHNESS[min(material_id, 7u)];
}
```

After `material_cell_albedo`, add:

```slang
float material_cell_roughness(VoxelCell cell) {
    return material_roughness(voxel_material(cell));
}
```

After `material_emissive`, add:

```slang
float material_emissive_luminance(VoxelCell cell) {
    return dot(material_emissive(cell), float3(0.2126, 0.7152, 0.0722));
}
```

- [ ] **Step 4: Verify the test passes**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib material_common_declares_deterministic_roughness_helpers; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: PASS.

- [ ] **Step 5: Review and commit**

Run:

```powershell
git diff -- assets/shaders/shared/material_common.slang src/render/passes/vpt/shader_source_tests.rs
git diff --check -- assets/shaders/shared/material_common.slang src/render/passes/vpt/shader_source_tests.rs
git add assets/shaders/shared/material_common.slang src/render/passes/vpt/shader_source_tests.rs
git commit -m "feat: add VPT material roughness helpers"
```

## Task 2: Surface Roughness Guide Resource

**Files:**
- Modify: `assets/shaders/passes/vpt_surface.slang`
- Modify: `src/render/passes/vpt_surface.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Write failing shader/resource contract test**

Add this test to `src/render/passes/vpt/shader_source_tests.rs`:

```rust
#[test]
fn vpt_surface_writes_explicit_material_roughness_guide() {
    let shader = source("assets/shaders/passes/vpt_surface.slang");
    let rust = source("src/render/passes/vpt_surface.rs");

    for token in [
        "RWTexture2D<float4> surface_material_roughness",
        "surface_material_roughness[pixel] = float4(material_cell_roughness(hit.cell), emissive_luma, 0.0, 1.0);",
        "surface_material_roughness[pixel] = float4(1.0, 0.0, 0.0, 0.0);",
        "material_emissive_luminance(hit.cell)",
    ] {
        assert!(shader.contains(token), "VPT surface shader missing {token}");
    }

    for token in [
        "pub surface_material_roughness: GpuImage",
        "pub previous_surface_material_roughness: GpuImage",
        "pub surface_writes: [ResourceHandle; 7]",
        "pub previous_surface_resources: [ResourceHandle; 5]",
        "vpt_surface_material_roughness",
        "vpt_previous_surface_material_roughness",
    ] {
        assert!(rust.contains(token), "VPT surface pass missing {token}");
    }
}
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_surface_writes_explicit_material_roughness_guide; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: FAIL because neither the shader binding nor Rust image fields exist.

- [ ] **Step 3: Add shader binding and writes**

In `assets/shaders/passes/vpt_surface.slang`, insert after `surface_albedo_material`:

```slang
[[vk::binding(4, 0)]]
[[vk::image_format("rgba32f")]]
RWTexture2D<float4> surface_material_roughness;
```

Shift the existing `motion_history` and later bindings up by one. The first motion guide image becomes binding 17, brick generation becomes 18, brick generation buffer becomes 19, motion events becomes 20.

In `write_miss`, add:

```slang
surface_material_roughness[pixel] = float4(1.0, 0.0, 0.0, 0.0);
```

Replace the local emissive computation with:

```slang
float emissive_luma = material_emissive_luminance(hit.cell);
```

After `surface_albedo_material[pixel] = float4(albedo, material_id);`, add:

```slang
surface_material_roughness[pixel] = float4(material_cell_roughness(hit.cell), emissive_luma, 0.0, 1.0);
```

- [ ] **Step 4: Add Rust surface images and graph resources**

In `src/render/passes/vpt_surface.rs`:
- Change `descriptor_binding_specs()` array from 20 to 21 entries.
- Insert a storage image binding for `surface_material_roughness` at binding 4.
- Shift bindings after it by one.
- Add `pub surface_material_roughness: GpuImage` and `pub previous_surface_material_roughness: GpuImage`.
- Add both fields to `VptSurfaceImages`, `VptSurfaceImageRefs`, creation, resize replacement, descriptor writes, graph import, history update copy, destroy paths, and return structs.
- Change `VptSurfaceBootstrapGraph.surface_writes` to `[ResourceHandle; 7]`.
- Change `VptSurfaceBootstrapGraph.previous_surface_resources` to `[ResourceHandle; 5]`.
- The current surface order must be:

```rust
[
    surface_position_resource,
    surface_normal_resource,
    surface_albedo_resource,
    surface_material_roughness_resource,
    motion_history_resource,
    motion_flags_resource,
    surface_brick_generation_resource,
]
```

- The previous surface order must be:

```rust
[
    previous_surface_position_resource,
    previous_surface_normal_resource,
    previous_surface_albedo_resource,
    previous_surface_material_roughness_resource,
    previous_surface_brick_generation_resource,
]
```

- [ ] **Step 5: Verify the focused test passes**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_surface_writes_explicit_material_roughness_guide; $code=$LASTEXITCODE; cargo test --lib vpt_surface_descriptor_specs_match_shader_manifest; if ($LASTEXITCODE -ne 0) { $code=$LASTEXITCODE }; cargo test --lib vpt_surface_shader_binding_manifest_matches_expected_resources; if ($LASTEXITCODE -ne 0) { $code=$LASTEXITCODE }; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: PASS for the surface guide tests.

- [ ] **Step 6: Review and commit**

Run:

```powershell
git diff -- assets/shaders/passes/vpt_surface.slang src/render/passes/vpt_surface.rs src/render/passes/vpt/shader_source_tests.rs
git diff --check -- assets/shaders/passes/vpt_surface.slang src/render/passes/vpt_surface.rs src/render/passes/vpt/shader_source_tests.rs
git add assets/shaders/passes/vpt_surface.slang src/render/passes/vpt_surface.rs src/render/passes/vpt/shader_source_tests.rs
git commit -m "feat: add VPT surface roughness guide"
```

## Task 3: Temporal and A-trous Roughness Consumers

**Files:**
- Modify: `assets/shaders/passes/vpt_temporal.slang`
- Modify: `assets/shaders/passes/vpt_atrous.slang`
- Modify: `src/render/passes/vpt_temporal.rs`
- Modify: `src/render/passes/vpt_atrous.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Write failing consumer test**

Add this test to `src/render/passes/vpt/shader_source_tests.rs`:

```rust
#[test]
fn vpt_temporal_and_atrous_consume_material_roughness_guide() {
    let temporal = source("assets/shaders/passes/vpt_temporal.slang");
    let atrous = source("assets/shaders/passes/vpt_atrous.slang");
    let temporal_rs = source("src/render/passes/vpt_temporal.rs");
    let atrous_rs = source("src/render/passes/vpt_atrous.rs");

    for token in [
        "RWTexture2D<float4> surface_material_roughness",
        "RWTexture2D<float4> previous_surface_material_roughness",
        "roughness_delta",
        "surface_material_roughness[pixel]",
        "previous_surface_material_roughness[previous_pixel]",
    ] {
        assert!(temporal.contains(token), "temporal shader missing {token}");
    }

    for token in [
        "RWTexture2D<float4> surface_material_roughness",
        "float roughness_weight(",
        "surface_material_roughness[neighbor_pixel]",
        "center_material_roughness",
    ] {
        assert!(atrous.contains(token), "A-trous shader missing {token}");
    }

    assert!(temporal_rs.contains("surface_inputs: [ResourceHandle; 7]"));
    assert!(temporal_rs.contains("previous_surface_inputs: [ResourceHandle; 5]"));
    assert!(atrous_rs.contains("surface_inputs: [ResourceHandle; 7]"));
}
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_temporal_and_atrous_consume_material_roughness_guide; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: FAIL because temporal and A-trous do not bind or consume roughness guide images.

- [ ] **Step 3: Update temporal shader and Rust pass**

In `assets/shaders/passes/vpt_temporal.slang`:
- Add `surface_material_roughness` after `surface_albedo_material`.
- Add `previous_surface_material_roughness` after `previous_surface_albedo_material`.
- Shift later bindings to keep descriptor order aligned.
- In `compatible_current_surface`, load center/neighbor roughness guide and reject when:

```slang
float roughness_delta = abs(surface_material_roughness[pixel].x - surface_material_roughness[neighbor_pixel].x);
if (roughness_delta > 0.35) {
    return false;
}
```

- In `compatible_history`, reject when:

```slang
float roughness_delta = abs(surface_material_roughness[pixel].x - previous_surface_material_roughness[previous_pixel].x);
if (roughness_delta > 0.35) {
    return false;
}
```

In `src/render/passes/vpt_temporal.rs`:
- Increase descriptor count for storage images.
- Change input array types to `[ResourceHandle; 7]` and `[ResourceHandle; 5]`.
- Bind current roughness from `vpt_surface.surface_material_roughness`.
- Bind previous roughness from `vpt_surface.previous_surface_material_roughness`.
- Read the new graph resources in `register_graph`.

- [ ] **Step 4: Update A-trous shader and Rust pass**

In `assets/shaders/passes/vpt_atrous.slang`:
- Add `surface_material_roughness` after `surface_albedo_material`.
- Shift output/UBO bindings after it.
- Add:

```slang
float roughness_weight(float center_roughness, float neighbor_roughness) {
    return exp(-abs(center_roughness - neighbor_roughness) * 6.0);
}
```

- In guide weight computation, multiply by roughness weight using:

```slang
float4 center_material_roughness = surface_material_roughness[pixel];
float4 neighbor_material_roughness = surface_material_roughness[neighbor_pixel];
```

In `src/render/passes/vpt_atrous.rs`:
- Increase descriptor image count.
- Change `surface_inputs` to `[ResourceHandle; 7]`.
- Bind `vpt_surface.surface_material_roughness`.
- Read `surface_inputs[3]` in `register_graph`.

- [ ] **Step 5: Verify the focused tests pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_temporal_and_atrous_consume_material_roughness_guide; $code=$LASTEXITCODE; cargo test --lib vpt_temporal_shader_binding_manifest_matches_motion_guide_resources; if ($LASTEXITCODE -ne 0) { $code=$LASTEXITCODE }; cargo test --lib vpt_atrous_pass_declares_svgf_edge_aware_filter_contract; if ($LASTEXITCODE -ne 0) { $code=$LASTEXITCODE }; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: PASS for consumer source tests.

- [ ] **Step 6: Review and commit**

Run:

```powershell
git diff -- assets/shaders/passes/vpt_temporal.slang assets/shaders/passes/vpt_atrous.slang src/render/passes/vpt_temporal.rs src/render/passes/vpt_atrous.rs src/render/passes/vpt/shader_source_tests.rs
git diff --check -- assets/shaders/passes/vpt_temporal.slang assets/shaders/passes/vpt_atrous.slang src/render/passes/vpt_temporal.rs src/render/passes/vpt_atrous.rs src/render/passes/vpt/shader_source_tests.rs
git add assets/shaders/passes/vpt_temporal.slang assets/shaders/passes/vpt_atrous.slang src/render/passes/vpt_temporal.rs src/render/passes/vpt_atrous.rs src/render/passes/vpt/shader_source_tests.rs
git commit -m "feat: consume VPT roughness guide in SVGF passes"
```

## Task 4: Full Verification and Review

**Files:**
- Check all files touched in Tasks 1-3.

- [ ] **Step 1: Format**

Run:

```powershell
cargo fmt
```

Expected: exit code 0.

- [ ] **Step 2: Run library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: exit code 0.

- [ ] **Step 3: Run clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: exit code 0.

- [ ] **Step 4: Check diff scope**

Run:

```powershell
git diff --check
git status --short
```

Expected:
- No whitespace errors.
- Only roughness-guide files are staged or unstaged for this plan.
- Pre-existing unrelated dirty files remain unstaged.

- [ ] **Step 5: Request code review**

Dispatch review for the commit range starting at the parent of Task 1 and ending at Task 3. The reviewer must check:
- roughness is explicit and deterministic
- `surface_normal_roughness.w` emissive hint behavior is preserved
- descriptor binding order matches shader manifests
- temporal/A-trous consume the new guide
- no unrelated dirty files are included

Address Critical and Important findings before proceeding to viewZ/motion work.

## Self-Review

Spec coverage:
- Covers the roughness/material guide part of Phase 2.
- Does not cover `viewZ` or 2.5D motion; those need separate Phase 2B/2C plans because they touch projection math and motion identity semantics.
- Keeps legacy SVGF emissive handling intact while adding a true roughness source for future NRD packing.

Placeholder scan:
- No placeholder markers.
- Every task has concrete files, code snippets, commands, and expected outcomes.

Type consistency:
- Current surface guide array has 7 resources.
- Previous surface guide array has 5 resources.
- The roughness image name is consistently `surface_material_roughness`.
- The history image name is consistently `previous_surface_material_roughness`.
