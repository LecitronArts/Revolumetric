# Radiance Cascades Quality Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Radiance Cascades visual stability, optional lighting performance, and shader/Rust ABI maintainability without changing the default high-quality path.

**Architecture:** Keep the existing 3-level RC probe layout and per-frame-slot double buffering. Add a cheap probe integration quality tier behind `REVOLUMETRIC_RC_PROBE_QUALITY`, add temporal radiance blending in `rc_trace`, and remove duplicated material LUTs by reusing shared shader material helpers. Use source-level regression tests for shader behavior and Rust unit tests for settings/ABI behavior.

**Tech Stack:** Rust 2024, Vulkan via `ash`, Slang compute shaders, existing Cargo unit tests and strict shader compilation.

---

## File Ownership

- Rust RC settings and UBO tests: `src/render/scene_ubo.rs`.
- Shader behavior and shader source tests: `assets/shaders/shared/scene_common.slang`, `assets/shaders/shared/radiance_cascade.slang`, `assets/shaders/passes/primary_ray.slang`, `assets/shaders/passes/rc_trace.slang`, `assets/shaders/passes/lighting.slang`, `src/render/passes/primary_ray.rs`, `src/render/passes/radiance_cascade_trace.rs`, `src/render/passes/lighting.rs`.
- Descriptor UBO range hardening: `src/render/passes/primary_ray.rs`, `src/render/passes/lighting.rs`, `src/render/passes/radiance_cascade_trace.rs`.

Do not edit unrelated voxel, Vulkan device, swapchain, allocator, or build script files in this plan.

---

### Task 1: Rust Quality Tier Settings

**Files:**
- Modify: `src/render/scene_ubo.rs`

- [ ] **Step 1: Add failing tests for the new RC quality tier**

Add tests in `src/render/scene_ubo.rs` under the existing `tests` module:

```rust
#[test]
fn lighting_settings_parse_fast_probe_quality() {
    let settings = LightingSettings::from_values(None, None, None, None, Some("fast"));

    assert_eq!(settings.rc_probe_quality, RcProbeQuality::Fast);
}

#[test]
fn lighting_settings_encode_fast_probe_quality() {
    let settings = LightingSettings {
        shadows_enabled: true,
        skip_backface_shadows: false,
        debug_view: LightingDebugView::Final,
        rc_normal_strategy: RcNormalStrategy::AxisNormal,
        rc_probe_quality: RcProbeQuality::Fast,
    };

    let uniforms = build_scene_uniforms(SceneUniformInputs {
        pixel_to_ray: glam::Mat4::IDENTITY,
        resolution: [640, 480],
        sun_direction: glam::Vec3::Y,
        sun_intensity: glam::Vec3::ONE,
        sky_color: [0.4, 0.5, 0.7],
        ground_color: [0.15, 0.1, 0.08],
        time: 0.0,
        rc_enabled: true,
        lighting_settings: settings,
    });

    assert_eq!(uniforms.rc_probe_quality, RC_PROBE_QUALITY_FAST);
}
```

- [ ] **Step 2: Verify the new tests fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::scene_ubo::tests::lighting_settings_parse_fast_probe_quality render::scene_ubo::tests::lighting_settings_encode_fast_probe_quality
```

Expected: fails because `RcProbeQuality::Fast` and `RC_PROBE_QUALITY_FAST` do not exist.

- [ ] **Step 3: Implement the Rust setting**

Update `src/render/scene_ubo.rs`:

```rust
pub const RC_PROBE_QUALITY_FULL: u32 = 0;
pub const RC_PROBE_QUALITY_FAST: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RcProbeQuality {
    Full,
    Fast,
}

impl RcProbeQuality {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Full => RC_PROBE_QUALITY_FULL,
            Self::Fast => RC_PROBE_QUALITY_FAST,
        }
    }
}

fn parse_rc_probe_quality(value: &str) -> Option<RcProbeQuality> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("full") {
        Some(RcProbeQuality::Full)
    } else if value.eq_ignore_ascii_case("fast")
        || value.eq_ignore_ascii_case("cheap")
        || value.eq_ignore_ascii_case("low")
    {
        Some(RcProbeQuality::Fast)
    } else {
        None
    }
}
```

Update the expected string in `from_values_report` from `"full"` to `"full|fast|cheap|low"`.

- [ ] **Step 4: Verify Rust settings tests pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::scene_ubo::tests::lighting_settings_parse_fast_probe_quality render::scene_ubo::tests::lighting_settings_encode_fast_probe_quality
```

Expected: both tests pass.

---

### Task 2: Shader Quality Path, Temporal Stability, And ABI Hardening

**Files:**
- Modify: `assets/shaders/shared/scene_common.slang`
- Modify: `assets/shaders/shared/radiance_cascade.slang`
- Modify: `assets/shaders/passes/primary_ray.slang`
- Modify: `assets/shaders/passes/rc_trace.slang`
- Modify: `assets/shaders/passes/lighting.slang`
- Modify: `src/render/passes/primary_ray.rs`
- Modify: `src/render/passes/radiance_cascade_trace.rs`
- Modify: `src/render/passes/lighting.rs`

- [ ] **Step 1: Add failing shader source tests**

Add source tests that fail before implementation:

In `src/render/passes/radiance_cascade_trace.rs`:

```rust
#[test]
fn rc_trace_temporally_blends_radiance_but_keeps_current_distance() {
    let source =
        include_str!("../../../assets/shaders/passes/rc_trace.slang").replace("\r\n", "\n");

    assert!(
        source.contains("float4 rc_temporal_blend(float4 current, float4 previous)")
    );
    assert!(
        source.contains("return float4(lerp(previous.xyz, current.xyz, RC_TEMPORAL_BLEND), current.w);")
    );
    assert!(
        source.contains("probe_write[idx] = rc_temporal_blend(result, probe_read[idx]);")
    );
}

#[test]
fn rc_trace_uses_shared_material_helpers() {
    let source =
        include_str!("../../../assets/shaders/passes/rc_trace.slang").replace("\r\n", "\n");

    assert!(source.contains("#include \"material_common.slang\""));
    assert!(source.contains("material_emissive(hit.cell)"));
    assert!(source.contains("material_cell_albedo(hit.cell)"));
    assert!(!source.contains("if (mat_id == 1u)"));
}
```

In `src/render/passes/lighting.rs`:

```rust
#[test]
fn lighting_shader_uses_rc_probe_quality_for_indirect_integration() {
    let source =
        include_str!("../../../assets/shaders/passes/lighting.slang").replace("\r\n", "\n");
    let shared = include_str!("../../../assets/shaders/shared/radiance_cascade.slang")
        .replace("\r\n", "\n");

    assert!(shared.contains("float3 integrate_probe_fast("));
    assert!(shared.contains("float3 integrate_probe_quality("));
    assert!(source.contains("scene.rc_probe_quality"));
    assert!(source.contains("integrate_probe_quality(position, rc_normal, rc_probes, scene.rc_c0_offset, c0_grid, scene.rc_probe_quality)"));
}

#[test]
fn descriptor_ubo_range_uses_rust_uniform_size() {
    let source = std::fs::read_to_string("src/render/passes/lighting.rs")
        .expect("lighting pass source should be readable");

    assert!(source.contains("std::mem::size_of::<GpuSceneUniforms>() as u64"));
    assert!(!source.contains(".range(176)"));
}
```

In `src/render/passes/primary_ray.rs`:

```rust
#[test]
fn primary_ray_uses_shared_material_helpers_and_dynamic_ubo_range() {
    let shader =
        include_str!("../../../assets/shaders/passes/primary_ray.slang").replace("\r\n", "\n");
    let rust = std::fs::read_to_string("src/render/passes/primary_ray.rs")
        .expect("primary ray pass source should be readable");

    assert!(shader.contains("#include \"material_common.slang\""));
    assert!(shader.contains("material_cell_albedo(hit.cell)"));
    assert!(!shader.contains("static const float3 MATERIAL_ALBEDO"));
    assert!(rust.contains("std::mem::size_of::<GpuSceneUniforms>() as u64"));
    assert!(!rust.contains(".range(176)"));
}
```

- [ ] **Step 2: Verify the source tests fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::passes::radiance_cascade_trace::tests::rc_trace_temporally_blends_radiance_but_keeps_current_distance render::passes::radiance_cascade_trace::tests::rc_trace_uses_shared_material_helpers render::passes::lighting::shader_source_tests::lighting_shader_uses_rc_probe_quality_for_indirect_integration render::passes::lighting::shader_source_tests::descriptor_ubo_range_uses_rust_uniform_size render::passes::primary_ray::shader_source_tests::primary_ray_uses_shared_material_helpers_and_dynamic_ubo_range
```

Expected: tests fail because the helpers and dynamic range are missing.

- [ ] **Step 3: Add shader constants and quality integration helpers**

Update `assets/shaders/shared/scene_common.slang`:

```slang
static const uint RC_PROBE_QUALITY_FULL = 0u;
static const uint RC_PROBE_QUALITY_FAST = 1u;
```

Update `assets/shaders/shared/radiance_cascade.slang` with a fast integration path:

```slang
float3 integrate_probe_fast(float3 world_pos, float3 normal,
                            StructuredBuffer<float4> rc_probes,
                            uint c0_offset, uint3 c0_grid_size) {
    float3 grid_pos = rc_world_to_c0_grid_position(world_pos);
    uint3 coord = uint3(clamp(int3(round(grid_pos)), int3(0), int3(c0_grid_size) - 1));
    uint face = normal_to_face(normal);

    uint probe_base = c0_offset +
        ((coord.z * c0_grid_size.y + coord.y) * c0_grid_size.x + coord.x)
        * 6u * 9u;
    if (rc_probes[probe_base].w < -0.5) {
        return float3(0.0);
    }

    return read_probe_face(coord, face, c0_grid_size, c0_offset, rc_probes);
}

float3 integrate_probe_quality(float3 world_pos, float3 normal,
                               StructuredBuffer<float4> rc_probes,
                               uint c0_offset, uint3 c0_grid_size,
                               uint quality) {
    if (quality == RC_PROBE_QUALITY_FAST) {
        return integrate_probe_fast(world_pos, normal, rc_probes, c0_offset, c0_grid_size);
    }
    return integrate_probe(world_pos, normal, rc_probes, c0_offset, c0_grid_size);
}
```

- [ ] **Step 4: Use shared material helpers**

Update `assets/shaders/passes/primary_ray.slang`:

```slang
#include "voxel_traverse.slang"
#include "material_common.slang"
```

Replace local material LUT usage with:

```slang
float3 albedo = material_cell_albedo(hit.cell);
```

Update `assets/shaders/passes/rc_trace.slang`:

```slang
#include "material_common.slang"
```

Replace local emissive/albedo logic with:

```slang
float3 emissive = material_emissive(hit.cell);
if (material_has_emissive(emissive)) {
    result.xyz = emissive * 8.0;
} else {
    float3 albedo = material_cell_albedo(hit.cell);
    ...
    result.xyz = (direct + indirect) * albedo;
}
```

- [ ] **Step 5: Add temporal radiance blending in RC trace**

Update `assets/shaders/passes/rc_trace.slang`:

```slang
static const float RC_TEMPORAL_BLEND = 0.18;

bool rc_probe_history_is_uninitialized(float4 previous) {
    return abs(previous.w) < 0.00001 && dot(abs(previous.xyz), float3(1.0)) < 0.00001;
}

float4 rc_temporal_blend(float4 current, float4 previous) {
    if (current.w < -0.5) return current;
    if (previous.w < -0.5) return current;
    if (rc_probe_history_is_uninitialized(previous)) return current;
    return float4(lerp(previous.xyz, current.xyz, RC_TEMPORAL_BLEND), current.w);
}
```

At the final probe write:

```slang
probe_write[idx] = rc_temporal_blend(result, probe_read[idx]);
```

Keep `current.w` so visibility and merge decisions use the current ray distance rather than lagged distance.

- [ ] **Step 6: Route lighting through quality-aware integration**

Update `assets/shaders/passes/lighting.slang`:

```slang
float3 indirect = integrate_probe_quality(
    position,
    rc_normal,
    rc_probes,
    scene.rc_c0_offset,
    c0_grid,
    scene.rc_probe_quality
);
```

Leave RC trace bounce integration on the full path in this task to avoid changing multi-bounce energy and temporal behavior at the same time.

- [ ] **Step 7: Replace hardcoded UBO descriptor ranges**

In `src/render/passes/primary_ray.rs`, `src/render/passes/lighting.rs`, and `src/render/passes/radiance_cascade_trace.rs`, import `GpuSceneUniforms` beside `SceneUniformBuffer` and replace:

```rust
.range(176);
```

with:

```rust
.range(std::mem::size_of::<GpuSceneUniforms>() as u64);
```

- [ ] **Step 8: Verify shader source tests pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test render::passes::radiance_cascade_trace::tests::rc_trace_temporally_blends_radiance_but_keeps_current_distance render::passes::radiance_cascade_trace::tests::rc_trace_uses_shared_material_helpers render::passes::lighting::shader_source_tests::lighting_shader_uses_rc_probe_quality_for_indirect_integration render::passes::lighting::shader_source_tests::descriptor_ubo_range_uses_rust_uniform_size render::passes::primary_ray::shader_source_tests::primary_ray_uses_shared_material_helpers_and_dynamic_ubo_range
```

Expected: source tests pass.

---

### Task 3: Integration Verification And Commit

**Files:**
- Review all modified files from Task 1 and Task 2.

- [ ] **Step 1: Run CPU test suite**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test
```

Expected: all tests pass.

- [ ] **Step 2: Run strict shader compilation**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib
```

Expected: Slang compiles all pass shaders and all library tests pass.

- [ ] **Step 3: Run clippy**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
```

Expected: no warnings or errors.

- [ ] **Step 4: Review diff scope**

```powershell
git -c safe.directory=E:/.Codes/Revolumetric diff -- assets/shaders/shared/scene_common.slang assets/shaders/shared/radiance_cascade.slang assets/shaders/passes/primary_ray.slang assets/shaders/passes/rc_trace.slang assets/shaders/passes/lighting.slang src/render/scene_ubo.rs src/render/passes/primary_ray.rs src/render/passes/radiance_cascade_trace.rs src/render/passes/lighting.rs docs/superpowers/plans/2026-04-30-radiance-cascades-quality-performance.md
```

Expected: diff only contains RC quality/performance changes and plan documentation.

- [ ] **Step 5: Commit only this task scope**

```powershell
git -c safe.directory=E:/.Codes/Revolumetric add assets/shaders/shared/scene_common.slang assets/shaders/shared/radiance_cascade.slang assets/shaders/passes/primary_ray.slang assets/shaders/passes/rc_trace.slang assets/shaders/passes/lighting.slang src/render/scene_ubo.rs src/render/passes/primary_ray.rs src/render/passes/radiance_cascade_trace.rs src/render/passes/lighting.rs docs/superpowers/plans/2026-04-30-radiance-cascades-quality-performance.md
git -c safe.directory=E:/.Codes/Revolumetric commit -m "feat: improve radiance cascades quality controls"
```

Expected: one commit is created. Existing unrelated dirty files remain unstaged.
