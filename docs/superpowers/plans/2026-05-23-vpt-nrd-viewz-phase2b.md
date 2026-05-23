# VPT NRD ViewZ Phase 2B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit VPT primary-surface `viewZ` guide for the NRD/ReLAX/ReBLUR input contract while preserving the existing SVGF depth and roughness semantics.

**Architecture:** Keep `surface_position_depth.w` as the voxel DDA hit distance `hit.t`; add separate current and previous `surface_view_z` images in `R32_SFLOAT`. The surface shader writes linear camera-forward depth for hit pixels and a large positive miss sentinel for miss pixels. The render graph carries `viewZ` through current/previous surface bundles and history copy, but SVGF temporal and A-trous do not consume it yet.

**Tech Stack:** Rust, Vulkan/ash render graph resources, Slang compute shaders, Cargo source tests with `REVOLUMETRIC_SHADER_COMPILE=skip`.

---

## Scope

In scope:
- Add `surface_view_z` and `previous_surface_view_z` images to `VptSurfacePass`.
- Write `surface_view_z[pixel]` from world position and camera forward depth in `vpt_surface.slang`.
- Preserve `surface_position_depth[pixel] = float4(hit.position, hit.t);`.
- Use a large positive miss sentinel for `surface_view_z`, while keeping miss `surface_position_depth.w = -1.0`.
- Expand current surface graph resources from 7 to 8 and previous surface resources from 5 to 6.
- Update Area ReSTIR, ReSTIR-DI, temporal, and A-trous graph signatures to accept the expanded surface bundles.
- Add source-contract and descriptor-reflection tests that lock the new resource order.

Out of scope:
- `motion_history.z = viewZprev - viewZ`.
- Moving `motion_id` out of `motion_history.z`.
- NRD packed normal/roughness/viewZ frontend pass.
- SVGF temporal or A-trous guide weighting with `viewZ`.
- Runtime debug view routing for `nrd_viewz`.

## File Structure

- Modify `assets/shaders/shared/scene_common.slang`
  - Owns the small camera-space depth helper and miss sentinel constant.
- Modify `assets/shaders/passes/vpt_surface.slang`
  - Adds the `surface_view_z` output binding and writes hit/miss values.
- Modify `src/render/passes/vpt_surface.rs`
  - Allocates, binds, registers, resizes, copies, and destroys current/previous `viewZ` images.
- Modify `src/render/passes/vpt_temporal.rs`
  - Accepts expanded surface graph arrays and includes `viewZ` in the history-copy dependency contract.
- Modify `src/render/passes/vpt_atrous.rs`
  - Accepts the expanded current surface graph array without binding `viewZ` to the shader.
- Modify `src/render/passes/area_restir.rs`
  - Carries expanded current/previous surface graph arrays through selected-surface registration.
- Modify `src/render/passes/restir_di.rs`
  - Carries expanded current/previous surface graph arrays through temporal dependency registration.
- Modify `src/render/passes/vpt/shader_source_tests.rs`
  - Adds failing tests first and updates binding/resource-order contracts.

## Review Cadence

- Review after Task 1: shader semantics, especially `hit.t` preservation and `viewZ` miss sentinel.
- Review after Task 2: Rust image lifetime, descriptor order, graph resource order, and history copy.
- Full review after Task 3 verification before proceeding to Phase 2C motion `.z`.

## Task 1: Shader ViewZ Contract

**Files:**
- Modify: `assets/shaders/shared/scene_common.slang`
- Modify: `assets/shaders/passes/vpt_surface.slang`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Write failing source test**

Add this test to `src/render/passes/vpt/shader_source_tests.rs`:

```rust
#[test]
fn vpt_surface_writes_independent_view_z_guide() {
    let scene_common = source("assets/shaders/shared/scene_common.slang");
    let surface = source("assets/shaders/passes/vpt_surface.slang");

    for token in [
        "static const float VPT_VIEW_Z_MISS_SENTINEL = 1.0e20;",
        "float scene_view_z(float3 world_position, SceneUniforms scene)",
        "return max(dot(world_position - scene.pixel_to_ray[3].xyz, normalize(scene.camera_forward)), 0.0);",
    ] {
        assert!(scene_common.contains(token), "scene common missing {token}");
    }

    for token in [
        "RWTexture2D<float> surface_view_z",
        "surface_view_z[pixel] = VPT_VIEW_Z_MISS_SENTINEL;",
        "float view_z = scene_view_z(hit.position, scene);",
        "surface_view_z[pixel] = view_z;",
        "surface_position_depth[pixel] = float4(hit.position, hit.t);",
        "motion.z = float(hit.motion_id);",
    ] {
        assert!(surface.contains(token), "VPT surface shader missing {token}");
    }

    assert!(
        !surface.contains("surface_position_depth[pixel] = float4(hit.position, view_z);"),
        "viewZ must not replace the legacy hit.t lane in surface_position_depth"
    );
}
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_surface_writes_independent_view_z_guide; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: FAIL because `surface_view_z` and `scene_view_z` do not exist yet.

- [ ] **Step 3: Add shared helper**

In `assets/shaders/shared/scene_common.slang`, after the debug constants, add:

```slang
static const float VPT_VIEW_Z_MISS_SENTINEL = 1.0e20;
```

After `ScenePrimaryRay`, add:

```slang
float scene_view_z(float3 world_position, SceneUniforms scene) {
    return max(dot(world_position - scene.pixel_to_ray[3].xyz, normalize(scene.camera_forward)), 0.0);
}
```

- [ ] **Step 4: Add surface shader output**

In `assets/shaders/passes/vpt_surface.slang`, insert after `surface_material_roughness`:

```slang
[[vk::binding(5, 0)]]
[[vk::image_format("r32f")]]
RWTexture2D<float> surface_view_z;
```

Shift later bindings up by one. `motion_history` becomes binding 6; `ucvh_config` starts at binding 7; `vpt_history` becomes binding 15; `area_restir` becomes binding 16; `area_restir_reservoirs` becomes binding 17; `motion_flags` becomes binding 18; `surface_brick_generation` becomes binding 19; `brick_generations` becomes binding 20; `ucvh_motion_events` becomes binding 21.

In `write_miss`, add:

```slang
surface_view_z[pixel] = VPT_VIEW_Z_MISS_SENTINEL;
```

In the hit path, immediately before surface writes, add:

```slang
float view_z = scene_view_z(hit.position, scene);
```

After `surface_material_roughness[pixel] = ...`, add:

```slang
surface_view_z[pixel] = view_z;
```

Keep:

```slang
surface_position_depth[pixel] = float4(hit.position, hit.t);
motion.z = float(hit.motion_id);
```

- [ ] **Step 5: Verify focused shader test passes**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_surface_writes_independent_view_z_guide; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: PASS.

- [ ] **Step 6: Review and commit**

Run:

```powershell
git diff -- assets/shaders/shared/scene_common.slang assets/shaders/passes/vpt_surface.slang src/render/passes/vpt/shader_source_tests.rs
git diff --check -- assets/shaders/shared/scene_common.slang assets/shaders/passes/vpt_surface.slang src/render/passes/vpt/shader_source_tests.rs
git add assets/shaders/shared/scene_common.slang assets/shaders/passes/vpt_surface.slang src/render/passes/vpt/shader_source_tests.rs
git commit -m "feat: write VPT surface viewZ guide"
```

## Task 2: Rust Surface Resource And Graph Contract

**Files:**
- Modify: `src/render/passes/vpt_surface.rs`
- Modify: `src/render/passes/vpt_temporal.rs`
- Modify: `src/render/passes/vpt_atrous.rs`
- Modify: `src/render/passes/area_restir.rs`
- Modify: `src/render/passes/restir_di.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Write failing Rust/resource-order test**

Add this test to `src/render/passes/vpt/shader_source_tests.rs`:

```rust
#[test]
fn vpt_surface_view_z_resource_graph_contract_is_ordered() {
    let surface_rs = source("src/render/passes/vpt_surface.rs");
    let temporal_rs = source("src/render/passes/vpt_temporal.rs");
    let atrous_rs = source("src/render/passes/vpt_atrous.rs");
    let area_rs = source("src/render/passes/area_restir.rs");
    let restir_rs = source("src/render/passes/restir_di.rs");
    let compact_surface = surface_rs.split_whitespace().collect::<String>();
    let compact_temporal = temporal_rs.split_whitespace().collect::<String>();

    for token in [
        "pub surface_view_z: GpuImage",
        "pub previous_surface_view_z: GpuImage",
        "pub surface_writes: [ResourceHandle; 8]",
        "pub previous_surface_resources: [ResourceHandle; 6]",
        "vpt_surface_view_z",
        "vpt_previous_surface_view_z",
        "vk::Format::R32_SFLOAT",
    ] {
        assert!(surface_rs.contains(token), "VPT surface pass missing {token}");
    }

    for token in [
        "surface_view_z_resource",
        "previous_surface_view_z_resource",
        "letsurface_images=[self.surface_position_depth.handle,self.surface_normal_roughness.handle,self.surface_albedo_material.handle,self.surface_material_roughness.handle,self.surface_view_z.handle,self.motion_history.handle,self.motion_flags.handle,self.surface_brick_generation.handle,];",
        "previous_surface_resources:[previous_surface_position_resource,previous_surface_normal_resource,previous_surface_albedo_resource,previous_surface_material_roughness_resource,previous_surface_view_z_resource,previous_surface_brick_generation_resource,]",
        "copy_surface_image(device,cmd,&self.surface_material_roughness,&self.previous_surface_material_roughness,);copy_surface_image(device,cmd,&self.surface_view_z,&self.previous_surface_view_z,);copy_surface_image(device,cmd,&self.surface_brick_generation,&self.previous_surface_brick_generation,);",
    ] {
        assert!(compact_surface.contains(token), "surface graph order missing {token}");
    }

    for token in [
        "builder.read_as(surface_inputs[0],AccessKind::TransferRead);builder.read_as(surface_inputs[1],AccessKind::TransferRead);builder.read_as(surface_inputs[2],AccessKind::TransferRead);builder.read_as(surface_inputs[3],AccessKind::TransferRead);builder.read_as(surface_inputs[4],AccessKind::TransferRead);builder.read_as(surface_inputs[7],AccessKind::TransferRead);",
        "builder.write_as(previous_surface_inputs[0],AccessKind::TransferWrite);builder.write_as(previous_surface_inputs[1],AccessKind::TransferWrite);builder.write_as(previous_surface_inputs[2],AccessKind::TransferWrite);builder.write_as(previous_surface_inputs[3],AccessKind::TransferWrite);builder.write_as(previous_surface_inputs[4],AccessKind::TransferWrite);builder.write_as(previous_surface_inputs[5],AccessKind::TransferWrite);",
    ] {
        assert!(compact_temporal.contains(token), "history copy order missing {token}");
    }

    for (name, source) in [
        ("temporal", temporal_rs.as_str()),
        ("atrous", atrous_rs.as_str()),
        ("area_restir", area_rs.as_str()),
        ("restir_di", restir_rs.as_str()),
    ] {
        assert!(
            source.contains("[ResourceHandle; 8]"),
            "{name} must accept expanded current surface resources"
        );
        assert!(
            name == "atrous" || source.contains("[ResourceHandle; 6]"),
            "{name} must accept expanded previous surface resources"
        );
    }
}
```

- [ ] **Step 2: Verify the test fails**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_surface_view_z_resource_graph_contract_is_ordered; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: FAIL because Rust resources and graph arrays are still 7/5.

- [ ] **Step 3: Add `viewZ` images to `VptSurfacePass`**

In `src/render/passes/vpt_surface.rs`:
- Change `descriptor_binding_specs()` from 21 to 22 entries.
- Insert `DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_IMAGE)`.
- Shift later descriptor binding numbers by one.
- Change surface storage-image descriptor pool count from `14 * frame_count` to `16 * frame_count`.
- Add:

```rust
pub surface_view_z: GpuImage,
pub previous_surface_view_z: GpuImage,
```

- Add `surface_view_z` to `VptSurfaceImageRefs`.
- Add `surface_view_z` to `output_images` after `surface_material_roughness`.
- Change UCVH buffer descriptor writes to start at binding `7`.
- Change history UBO write to binding `15`.
- Change Area ReSTIR disabled descriptor writes to bindings `16` and `17`.
- Change motion guide image writes to start at binding `18`.
- Change motion guide buffer writes to start at binding `20`.

Create images with:

```rust
create_surface_image(device, allocator, width, height, vk::Format::R32_SFLOAT, "vpt_surface_view_z")
create_surface_image(device, allocator, width, height, vk::Format::R32_SFLOAT, "vpt_previous_surface_view_z")
```

Add both images to creation rollback, resize replacement, destroy paths, descriptor writes, and `record_history_update`.

- [ ] **Step 4: Expand graph resource bundles**

In `src/render/passes/vpt_surface.rs`:
- Change `VptSurfaceBootstrapGraph.surface_writes` to `[ResourceHandle; 8]`.
- Change `previous_surface_resources` to `[ResourceHandle; 6]`.
- Import current `surface_view_z` as `vk::Format::R32_SFLOAT`, storage plus transfer src.
- Import previous `surface_view_z` as `vk::Format::R32_SFLOAT`, storage plus transfer src/dst.
- Add current order:

```rust
[
    surface_position_resource,
    surface_normal_resource,
    surface_albedo_resource,
    surface_material_roughness_resource,
    surface_view_z_resource,
    motion_history_resource,
    motion_flags_resource,
    surface_brick_generation_resource,
]
```

- Add previous order:

```rust
[
    previous_surface_position_resource,
    previous_surface_normal_resource,
    previous_surface_albedo_resource,
    previous_surface_material_roughness_resource,
    previous_surface_view_z_resource,
    previous_surface_brick_generation_resource,
]
```

In `src/render/passes/vpt_temporal.rs`:
- Change surface input arrays to `[ResourceHandle; 8]` and `[ResourceHandle; 6]`.
- In `register_history_update_graph`, read current indices `0, 1, 2, 3, 4, 7` and write previous indices `0, 1, 2, 3, 4, 5`.

In `src/render/passes/vpt_atrous.rs`:
- Change `surface_inputs` to `[ResourceHandle; 8]`.
- Keep shader descriptor bindings unchanged; A-trous still reads only indices `0..=3`.

In `src/render/passes/area_restir.rs`:
- Change `final_surface_writes` and `bootstrap_surface_writes` to `[ResourceHandle; 8]`.
- Change `previous_surface_resources` to `[ResourceHandle; 6]`.
- Return all 8 selected surface write handles.

In `src/render/passes/restir_di.rs`:
- Change `final_surface_writes` to `[ResourceHandle; 8]`.
- Change `previous_surface_resources` to `[ResourceHandle; 6]`.
- Keep shader descriptors unchanged; the extra reads are graph dependencies, not shader inputs.

- [ ] **Step 5: Verify focused graph test passes**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_surface_view_z_resource_graph_contract_is_ordered; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: PASS.

- [ ] **Step 6: Review and commit**

Run:

```powershell
git diff -- src/render/passes/vpt_surface.rs src/render/passes/vpt_temporal.rs src/render/passes/vpt_atrous.rs src/render/passes/area_restir.rs src/render/passes/restir_di.rs src/render/passes/vpt/shader_source_tests.rs
git diff --check -- src/render/passes/vpt_surface.rs src/render/passes/vpt_temporal.rs src/render/passes/vpt_atrous.rs src/render/passes/area_restir.rs src/render/passes/restir_di.rs src/render/passes/vpt/shader_source_tests.rs
git add src/render/passes/vpt_surface.rs src/render/passes/vpt_temporal.rs src/render/passes/vpt_atrous.rs src/render/passes/area_restir.rs src/render/passes/restir_di.rs src/render/passes/vpt/shader_source_tests.rs
git commit -m "feat: add VPT surface viewZ resources"
```

## Task 3: Descriptor Reflection And Full Verification

**Files:**
- Modify: `src/render/passes/vpt/shader_source_tests.rs`
- Check all files touched in Tasks 1-2.

- [ ] **Step 1: Update binding manifest expectations**

In `vpt_surface_shader_binding_manifest_matches_expected_resources`, insert:

```rust
binding(5, DescriptorKind::StorageImage, "surface_view_z"),
```

Then shift expected `vpt_surface.slang` bindings after it by one:

```rust
binding(6, DescriptorKind::StorageImage, "motion_history"),
binding(7, DescriptorKind::StorageBuffer, "ucvh_config"),
binding(8, DescriptorKind::StorageBuffer, "hierarchy_l0"),
binding(9, DescriptorKind::StorageBuffer, "hierarchy_l1"),
binding(10, DescriptorKind::StorageBuffer, "hierarchy_l2"),
binding(11, DescriptorKind::StorageBuffer, "hierarchy_l3"),
binding(12, DescriptorKind::StorageBuffer, "hierarchy_l4"),
binding(13, DescriptorKind::StorageBuffer, "brick_occupancy"),
binding(14, DescriptorKind::StorageBuffer, "brick_materials"),
binding(15, DescriptorKind::UniformBuffer, "vpt_history"),
binding(16, DescriptorKind::UniformBuffer, "area_restir"),
binding(17, DescriptorKind::StorageBuffer, "area_restir_reservoirs"),
binding(18, DescriptorKind::StorageImage, "motion_flags"),
binding(19, DescriptorKind::StorageImage, "surface_brick_generation"),
binding(20, DescriptorKind::StorageBuffer, "brick_generations"),
binding(21, DescriptorKind::StorageBuffer, "ucvh_motion_events"),
```

Update older roughness resource tests to expect `[ResourceHandle; 8]`, `[ResourceHandle; 6]`, and the new order tokens.

- [ ] **Step 2: Run focused reflection and contract tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_surface_writes_independent_view_z_guide; $code=$LASTEXITCODE; cargo test --lib vpt_surface_view_z_resource_graph_contract_is_ordered; if ($LASTEXITCODE -ne 0) { $code=$LASTEXITCODE }; cargo test --lib vpt_svgf_descriptor_specs_match_shader_manifests; if ($LASTEXITCODE -ne 0) { $code=$LASTEXITCODE }; cargo test --lib vpt_surface_shader_binding_manifest_matches_expected_resources; if ($LASTEXITCODE -ne 0) { $code=$LASTEXITCODE }; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: PASS.

- [ ] **Step 3: Format**

Run:

```powershell
cargo fmt
```

Expected: exit code 0.

- [ ] **Step 4: Run library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: exit code 0.

- [ ] **Step 5: Run clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; $code=$LASTEXITCODE; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE; exit $code
```

Expected: exit code 0.

- [ ] **Step 6: Check diff scope**

Run:

```powershell
git diff --check
git status --short
```

Expected:
- No whitespace errors.
- Only Phase 2B files are staged or unstaged for this plan.
- Pre-existing unrelated dirty files remain unstaged.

- [ ] **Step 7: Request code review**

Dispatch review for the Phase 2B commit range. The reviewer must check:
- `viewZ` is independent from `hit.t`.
- Miss pixels use a large positive `viewZ` sentinel while legacy miss depth stays negative.
- `surface_position_depth.w` remains `hit.t`.
- Current/previous surface graph order is stable and history copy includes `viewZ`.
- Area ReSTIR and ReSTIR-DI still receive coherent surface graph dependencies.
- Descriptor specs match shader reflection.
- No unrelated dirty files are included.

Address Critical and Important findings before proceeding to Phase 2C.

## Self-Review

Spec coverage:
- Covers Phase 2 `viewZ` output.
- Does not cover 2.5D motion or `motion_id` relocation; those remain Phase 2C because they change motion ABI semantics.
- Keeps SVGF behavior unchanged because neither temporal nor A-trous consumes `viewZ` in this slice.

Placeholder scan:
- No placeholder markers.
- Every task lists concrete files, code snippets, commands, and expected outcomes.

Type consistency:
- Current surface graph resources are consistently `[ResourceHandle; 8]`.
- Previous surface graph resources are consistently `[ResourceHandle; 6]`.
- Current order is position, normal, albedo, material roughness, viewZ, motion history, motion flags, brick generation.
- Previous order is position, normal, albedo, material roughness, viewZ, brick generation.
