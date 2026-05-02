# ReSTIR-DI VPT MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a VPT-only ReSTIR-DI MVP that builds a direct-light table, stores per-pixel reservoirs, and prepares the VPT path to use reservoir-resampled direct lighting.

**Architecture:** Keep VCT as the default renderer and keep ReSTIR-DI behind VPT-specific settings. Implement prerequisites first: configuration/ABI, direct-light table construction, graph-owned buffer barriers, and pass skeletons with tests. The first renderable integration should use a VPT-owned `vpt_surface` stage or an intentionally changed VPT graph; no pass may silently assume the current VPT branch has primary-ray G-buffer outputs.

**Tech Stack:** Rust 2024, Vulkan via `ash`, Slang compute shaders, existing `RenderGraph`, `GpuBuffer`, UCVH `BrickPool`, `cargo test`, `cargo clippy`, strict shader compilation.

---

## Source Facts To Preserve

- Current VPT graph is `vpt -> postprocess -> blit`, not `primary_ray -> vpt -> postprocess`.
- Current VPT descriptor set only has Scene UBO, accumulation image, and UCVH config/L0/occupancy/material buffers.
- Current UCVH shader path uses `StructuredBuffer<T>` and ordinary indexing.
- Current RenderGraph emits image and buffer barriers for declared single-queue accesses.
- Current scene has sun fields and emissive voxel materials; `src/render/restir_di.rs` now builds a sampleable direct-light table from the sun and brick-clustered emissive voxels.
- ReSTIR-DI is direct-light reuse only; ReSTIR GI/PT is out of scope.

## File Structure

- Create `src/render/restir_di.rs`: CPU-side settings, GPU ABI structs, parse helpers, direct-light table construction, and unit tests.
- Modify `src/render/mod.rs`: export `restir_di`.
- Modify `src/render/scene_ubo.rs`: no required changes for the first implementation; ReSTIR-DI uses a dedicated `RestirDiSettings` and `GpuRestirDiUniforms` path in `src/render/restir_di.rs`.
- Modify `assets/shaders/shared/restir_di_common.slang`: Slang ABI mirror and reservoir helper functions.
- Modify `src/render/resource.rs`, `src/render/graph.rs`: graph-owned buffer barrier planning/emission.
- Create `assets/shaders/passes/restir_di_initial.slang`, `restir_di_temporal.slang`, `restir_di_spatial.slang`, and `vpt_surface.slang` if the implementation needs reusable primary-surface state.
- Create `src/render/passes/restir_di.rs`: pass/resource owner for ReSTIR-DI buffers and descriptor sets.
- Modify `src/render/passes/vpt.rs` and `assets/shaders/passes/vpt.slang`: consume final reservoir after pass skeleton is validated.
- Modify `src/render/gpu_profiler.rs`: add optional ReSTIR-DI scopes once passes exist.
- Modify `src/app.rs`: wire settings, resources, resize/reset, graph passes, and runtime smoke.
- Modify `README.md` and `docs/superpowers/specs/2026-05-02-restir-di-vpt-design.md`: keep runtime docs and design constraints current.

---

### Task 1: ReSTIR-DI Settings And GPU ABI

**Files:**
- Create: `src/render/restir_di.rs`
- Modify: `src/render/mod.rs`
- Test: `src/render/restir_di.rs`

- [x] **Step 1: Write failing settings and ABI tests**

Add `src/render/restir_di.rs` with the tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_defaults_keep_restir_disabled() {
        let settings = RestirDiSettings::default();
        assert!(!settings.enabled);
        assert!(settings.temporal_enabled);
        assert!(settings.spatial_enabled);
        assert_eq!(settings.initial_candidate_count, 1);
        assert_eq!(settings.spatial_sample_count, 4);
        assert_eq!(settings.history_length, 20);
        assert_eq!(settings.debug_view, RestirDiDebugView::Off);
    }

    #[test]
    fn settings_parse_valid_overrides() {
        let parsed = RestirDiSettings::from_values(
            Some("on"),
            Some("off"),
            Some("true"),
            Some("8"),
            Some("3"),
            Some("32"),
            Some("reservoir_weight"),
        );
        assert!(parsed.settings.enabled);
        assert!(!parsed.settings.temporal_enabled);
        assert!(parsed.settings.spatial_enabled);
        assert_eq!(parsed.settings.initial_candidate_count, 8);
        assert_eq!(parsed.settings.spatial_sample_count, 3);
        assert_eq!(parsed.settings.history_length, 32);
        assert_eq!(parsed.settings.debug_view, RestirDiDebugView::ReservoirWeight);
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn settings_reject_invalid_values_without_changing_defaults() {
        let parsed = RestirDiSettings::from_values(
            Some("maybe"),
            Some("later"),
            Some("sometimes"),
            Some("0"),
            Some("99"),
            Some("0"),
            Some("heatmap"),
        );
        assert_eq!(parsed.settings, RestirDiSettings::default());
        assert_eq!(parsed.warnings.len(), 7);
    }

    #[test]
    fn gpu_struct_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuRestirDiUniforms>(), 48);
        assert_eq!(std::mem::size_of::<GpuDirectLight>(), 48);
        assert_eq!(std::mem::size_of::<GpuRestirDiReservoir>(), 64);
        assert_eq!(std::mem::offset_of!(GpuRestirDiUniforms, enabled), 0);
        assert_eq!(std::mem::offset_of!(GpuRestirDiUniforms, light_count), 36);
        assert_eq!(std::mem::offset_of!(GpuDirectLight, color_power), 32);
        assert_eq!(std::mem::offset_of!(GpuRestirDiReservoir, sample_position_pdf), 32);
    }
}
```

- [x] **Step 2: Verify red**

Run:

```powershell
cargo test render::restir_di::tests --lib
```

Expected: compile failure because `render::restir_di` and the tested types do not exist.

- [x] **Step 3: Implement settings and ABI types**

Create `src/render/restir_di.rs`:

```rust
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestirDiDebugView {
    Off,
    ReservoirWeight,
    LightId,
    Visibility,
    TemporalValid,
    SpatialNeighbors,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestirDiSettings {
    pub enabled: bool,
    pub temporal_enabled: bool,
    pub spatial_enabled: bool,
    pub initial_candidate_count: u32,
    pub spatial_sample_count: u32,
    pub history_length: u32,
    pub debug_view: RestirDiDebugView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestirDiParseWarning {
    pub variable: &'static str,
    pub expected: &'static str,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestirDiSettingsParse {
    pub settings: RestirDiSettings,
    pub warnings: Vec<RestirDiParseWarning>,
}

impl Default for RestirDiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            temporal_enabled: true,
            spatial_enabled: true,
            initial_candidate_count: 1,
            spatial_sample_count: 4,
            history_length: 20,
            debug_view: RestirDiDebugView::Off,
        }
    }
}

impl RestirDiSettings {
    pub fn from_env() -> RestirDiSettingsParse {
        Self::from_values(
            std::env::var("REVOLUMETRIC_VPT_RESTIR_DI").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_TEMPORAL").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_SPATIAL").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_INITIAL_CANDIDATES").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_HISTORY_LENGTH").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RESTIR_DI_DEBUG").ok().as_deref(),
        )
    }

    pub fn from_values(
        enabled: Option<&str>,
        temporal: Option<&str>,
        spatial: Option<&str>,
        initial_candidates: Option<&str>,
        spatial_samples: Option<&str>,
        history_length: Option<&str>,
        debug_view: Option<&str>,
    ) -> RestirDiSettingsParse {
        let mut settings = Self::default();
        let mut warnings = Vec::new();
        parse_bool("REVOLUMETRIC_VPT_RESTIR_DI", enabled, &mut settings.enabled, &mut warnings);
        parse_bool("REVOLUMETRIC_RESTIR_DI_TEMPORAL", temporal, &mut settings.temporal_enabled, &mut warnings);
        parse_bool("REVOLUMETRIC_RESTIR_DI_SPATIAL", spatial, &mut settings.spatial_enabled, &mut warnings);
        parse_u32_range("REVOLUMETRIC_RESTIR_DI_INITIAL_CANDIDATES", initial_candidates, 1, 16, &mut settings.initial_candidate_count, &mut warnings);
        parse_u32_range("REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES", spatial_samples, 0, 8, &mut settings.spatial_sample_count, &mut warnings);
        parse_u32_range("REVOLUMETRIC_RESTIR_DI_HISTORY_LENGTH", history_length, 1, 64, &mut settings.history_length, &mut warnings);
        parse_debug_view(debug_view, &mut settings.debug_view, &mut warnings);
        RestirDiSettingsParse { settings, warnings }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRestirDiUniforms {
    pub enabled: u32,
    pub temporal_enabled: u32,
    pub spatial_enabled: u32,
    pub debug_view: u32,
    pub initial_candidate_count: u32,
    pub spatial_sample_count: u32,
    pub history_length: u32,
    pub frame_index: u32,
    pub reservoir_count: u32,
    pub light_count: u32,
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuDirectLight {
    pub position_radius: [f32; 4],
    pub normal_type: [f32; 4],
    pub color_power: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRestirDiReservoir {
    pub sample_light_id: u32,
    pub sample_flags: u32,
    pub sample_count_m: u32,
    pub pad0: u32,
    pub target_pdf: f32,
    pub weight_sum: f32,
    pub selected_weight: f32,
    pub confidence: f32,
    pub sample_position_pdf: [f32; 4],
    pub sample_radiance: [f32; 4],
}
```

Add these private parse helpers in the same file:

```rust
fn parse_bool(
    variable: &'static str,
    value: Option<&str>,
    target: &mut bool,
    warnings: &mut Vec<RestirDiParseWarning>,
) {
    let Some(value) = value else { return; };
    if matches!(value, "1" | "on" | "true" | "yes") {
        *target = true;
    } else if matches!(value, "0" | "off" | "false" | "no") {
        *target = false;
    } else {
        warnings.push(RestirDiParseWarning {
            variable,
            expected: "on|off|1|0|true|false|yes|no",
            value: value.to_string(),
        });
    }
}

fn parse_u32_range(
    variable: &'static str,
    value: Option<&str>,
    min: u32,
    max: u32,
    target: &mut u32,
    warnings: &mut Vec<RestirDiParseWarning>,
) {
    let Some(value) = value else { return; };
    match value.parse::<u32>().ok().filter(|parsed| (min..=max).contains(parsed)) {
        Some(parsed) => *target = parsed,
        None => warnings.push(RestirDiParseWarning {
            variable,
            expected: "integer in configured range",
            value: value.to_string(),
        }),
    }
}

fn parse_debug_view(
    value: Option<&str>,
    target: &mut RestirDiDebugView,
    warnings: &mut Vec<RestirDiParseWarning>,
) {
    let Some(value) = value else { return; };
    *target = if value.eq_ignore_ascii_case("off") {
        RestirDiDebugView::Off
    } else if value.eq_ignore_ascii_case("reservoir_weight") {
        RestirDiDebugView::ReservoirWeight
    } else if value.eq_ignore_ascii_case("light_id") {
        RestirDiDebugView::LightId
    } else if value.eq_ignore_ascii_case("visibility") {
        RestirDiDebugView::Visibility
    } else if value.eq_ignore_ascii_case("temporal_valid") {
        RestirDiDebugView::TemporalValid
    } else if value.eq_ignore_ascii_case("spatial_neighbors") {
        RestirDiDebugView::SpatialNeighbors
    } else {
        warnings.push(RestirDiParseWarning {
            variable: "REVOLUMETRIC_RESTIR_DI_DEBUG",
            expected: "off|reservoir_weight|light_id|visibility|temporal_valid|spatial_neighbors",
            value: value.to_string(),
        });
        *target
    };
}
```

Add to `src/render/mod.rs`:

```rust
pub mod restir_di;
```

- [x] **Step 4: Verify green**

Run:

```powershell
cargo test render::restir_di::tests --lib
```

Expected: all `render::restir_di::tests` pass.

---

### Task 2: Slang ReSTIR-DI ABI Mirror

**Files:**
- Create: `assets/shaders/shared/restir_di_common.slang`
- Modify: `src/render/restir_di.rs`

- [x] **Step 1: Write failing shader source test**

Add this test to `src/render/restir_di.rs`:

```rust
#[test]
fn slang_restir_di_common_declares_matching_abi() {
    let source = std::fs::read_to_string("assets/shaders/shared/restir_di_common.slang")
        .expect("restir_di_common.slang should be readable");
    assert!(source.contains("struct RestirDiUniforms"));
    assert!(source.contains("uint enabled;"));
    assert!(source.contains("uint light_count;"));
    assert!(source.contains("struct DirectLight"));
    assert!(source.contains("float4 color_power;"));
    assert!(source.contains("struct RestirDiReservoir"));
    assert!(source.contains("uint sample_count_m;"));
    assert!(source.contains("float4 sample_radiance;"));
}
```

- [x] **Step 2: Verify red**

Run:

```powershell
cargo test render::restir_di::tests::slang_restir_di_common_declares_matching_abi --lib
```

Expected: failure because `restir_di_common.slang` does not exist.

- [x] **Step 3: Create Slang ABI file**

Create `assets/shaders/shared/restir_di_common.slang`:

```hlsl
struct RestirDiUniforms {
    uint enabled;
    uint temporal_enabled;
    uint spatial_enabled;
    uint debug_view;
    uint initial_candidate_count;
    uint spatial_sample_count;
    uint history_length;
    uint frame_index;
    uint reservoir_count;
    uint light_count;
    uint width;
    uint height;
};

struct DirectLight {
    float4 position_radius;
    float4 normal_type;
    float4 color_power;
};

struct RestirDiReservoir {
    uint sample_light_id;
    uint sample_flags;
    uint sample_count_m;
    uint pad0;
    float target_pdf;
    float weight_sum;
    float selected_weight;
    float confidence;
    float4 sample_position_pdf;
    float4 sample_radiance;
};
```

- [x] **Step 4: Verify green**

Run:

```powershell
cargo test render::restir_di::tests::slang_restir_di_common_declares_matching_abi --lib
```

Expected: test passes.

---

### Task 3: Direct-Light Table CPU Construction From UCVH

**Files:**
- Modify: `src/render/restir_di.rs`
- Test: `src/render/restir_di.rs`

- [x] **Step 1: Write failing light table tests**

Add tests:

```rust
#[test]
fn direct_light_table_includes_sun_when_power_is_positive() {
    let lights = build_direct_lights_for_test(
        [0.0, -1.0, 0.0],
        2.0,
        &[],
    );
    assert_eq!(lights.len(), 1);
    assert_eq!(lights[0].normal_type[3], 0.0);
    assert!(lights[0].color_power[3] > 0.0);
}

#[test]
fn direct_light_table_clusters_emissive_voxels_by_brick() {
    let voxels = [
        EmissiveVoxelForTest { brick_id: 7, world_position: [1.0, 2.0, 3.0], emissive: [255, 128, 0] },
        EmissiveVoxelForTest { brick_id: 7, world_position: [3.0, 2.0, 1.0], emissive: [255, 128, 0] },
        EmissiveVoxelForTest { brick_id: 8, world_position: [8.0, 0.0, 0.0], emissive: [0, 0, 0] },
    ];
    let lights = build_direct_lights_for_test(
        [0.0, -1.0, 0.0],
        0.0,
        &voxels,
    );
    assert_eq!(lights.len(), 1);
    assert_eq!(lights[0].normal_type[3], 1.0);
    assert_eq!(lights[0].position_radius[0], 2.0);
    assert_eq!(lights[0].position_radius[1], 2.0);
    assert_eq!(lights[0].position_radius[2], 2.0);
}

#[test]
fn direct_light_table_can_be_built_from_ucvh_sponza_emissives() {
    let mut ucvh = crate::voxel::sponza_generator::SponzaGenerator::generate();
    ucvh.rebuild_hierarchy();
    let lights = build_direct_lights_from_ucvh(
        &ucvh,
        [0.0, -1.0, 0.0],
        0.0,
        4096,
    );
    assert!(
        lights.iter().any(|light| light.normal_type[3] == 1.0 && light.color_power[3] > 0.0),
        "sponza emissive voxels should produce at least one emissive direct-light cluster"
    );
}
```

- [ ] **Step 2: Verify red**

Run:

```powershell
cargo test render::restir_di::tests::direct_light_table --lib
```

Expected: compile failure because the helper types/functions do not exist.

- [x] **Step 3: Implement testable table builder**

Add a CPU builder in `src/render/restir_di.rs`:

```rust
#[derive(Debug, Clone, Copy)]
pub struct EmissiveVoxelForTest {
    pub brick_id: u32,
    pub world_position: [f32; 3],
    pub emissive: [u8; 3],
}

pub fn build_direct_lights_for_test(
    sun_direction: [f32; 3],
    sun_intensity: f32,
    emissive_voxels: &[EmissiveVoxelForTest],
) -> Vec<GpuDirectLight> {
    build_direct_lights_from_emissive_iter(sun_direction, sun_intensity, emissive_voxels.iter().copied())
}

pub fn build_direct_lights_from_ucvh(
    ucvh: &crate::voxel::ucvh::Ucvh,
    sun_direction: [f32; 3],
    sun_intensity: f32,
    max_lights: usize,
) -> Vec<GpuDirectLight> {
    let mut emissive_voxels = Vec::new();
    let grid = ucvh.config.brick_grid_size;
    let materials = ucvh.pool.material_pool();
    for z in 0..grid.z {
        for y in 0..grid.y {
            for x in 0..grid.x {
                let brick_pos = glam::UVec3::new(x, y, z);
                let node = ucvh.hierarchy.get_l0(brick_pos);
                if node.flags & 1 == 0 || node.brick_id == u32::MAX {
                    continue;
                }
                let base = node.brick_id as usize * crate::voxel::brick::BRICK_VOLUME;
                for morton_index in 0..crate::voxel::brick::BRICK_VOLUME {
                    let cell = materials[base + morton_index];
                    if cell.emissive == [0; 3] {
                        continue;
                    }
                    let local = crate::voxel::morton::decode(morton_index as u32);
                    let world = brick_pos * crate::voxel::brick::BRICK_EDGE + local;
                    emissive_voxels.push(EmissiveVoxelForTest {
                        brick_id: node.brick_id,
                        world_position: [world.x as f32, world.y as f32, world.z as f32],
                        emissive: cell.emissive,
                    });
                }
            }
        }
    }
    let mut lights = build_direct_lights_from_emissive_iter(
        sun_direction,
        sun_intensity,
        emissive_voxels.into_iter(),
    );
    lights.sort_by(|a, b| b.color_power[3].total_cmp(&a.color_power[3]));
    lights.truncate(max_lights);
    lights
}
```

Implement `build_direct_lights_from_emissive_iter()` so it:

- Adds one sun light when `sun_intensity > 0.0`.
- Ignores emissive voxels whose RGB sum is zero.
- Groups emissive voxels by `brick_id`.
- Averages grouped positions.
- Sums RGB power into `color_power`. Use the current shader convention that emissive RGB decoded from `0..255` becomes `0..1` and is multiplied by `3.0` before contributing to lighting.
- Writes `normal_type.w = 1.0` for emissive voxel clusters.
- Clamps the final list to `max_lights` in `build_direct_lights_from_ucvh()` after sorting by `color_power.w` descending.

- [x] **Step 4: Verify green**

Run:

```powershell
cargo test render::restir_di::tests::direct_light_table --lib
```

Expected: direct-light table tests pass.

---

### Task 4: RenderGraph Buffer Barrier Support

**Files:**
- Modify: `src/render/graph.rs`
- Modify: `src/render/resource.rs` only if a buffer-specific access helper is needed
- Test: `src/render/graph.rs`

- [x] **Step 1: Replace fail-closed test with failing positive tests**

Change the existing buffer barrier test expectation into positive tests:

```rust
#[test]
fn compile_plans_buffer_barriers_between_compute_writes_and_reads() {
    let mut graph = RenderGraph::new();
    let writer = graph.add_pass("writer", QueueType::Compute, |builder| {
        builder.create_buffer(4096, vk::BufferUsageFlags::STORAGE_BUFFER)
    })[0];
    let reader = graph.add_pass("reader", QueueType::Compute, |builder| {
        builder.read_as(writer, AccessKind::ComputeShaderRead);
    });
    graph.bind_buffer(writer, fake_buffer(44));
    graph.compile().unwrap();
    let barriers = graph.planned_barriers();
    assert_eq!(barriers.len(), 1);
    assert_eq!(barriers[0].pass_index, reader);
    assert_eq!(barriers[0].resource, writer);
    assert_eq!(barriers[0].from, AccessKind::ComputeShaderWrite);
    assert_eq!(barriers[0].to, AccessKind::ComputeShaderRead);
}

#[test]
fn imported_buffer_with_access_can_transition_to_compute_write() {
    let mut graph = RenderGraph::new();
    let buffer = graph.import_buffer_with_access(
        fake_buffer(45),
        4096,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        AccessKind::ComputeShaderRead,
    );
    graph.add_pass("writer", QueueType::Compute, |builder| {
        builder.write_as(buffer, AccessKind::ComputeShaderWrite);
    });
    graph.compile().unwrap();
    let barriers = graph.planned_barriers();
    assert_eq!(barriers.len(), 1);
    assert_eq!(barriers[0].resource, buffer);
    assert_eq!(barriers[0].from, AccessKind::ComputeShaderRead);
    assert_eq!(barriers[0].to, AccessKind::ComputeShaderWrite);
}

#[test]
fn compile_rejects_null_buffer_barrier_binding() {
    let mut graph = RenderGraph::new();
    let writer = graph.add_pass("writer", QueueType::Compute, |builder| {
        builder.create_buffer(4096, vk::BufferUsageFlags::STORAGE_BUFFER)
    })[0];
    graph.add_pass("reader", QueueType::Compute, |builder| {
        builder.read_as(writer, AccessKind::ComputeShaderRead);
    });
    graph.bind_buffer(writer, vk::Buffer::null());
    let err = graph.compile().unwrap_err().to_string();
    assert!(err.contains("buffer resource id"));
    assert!(err.contains("non-null bound vk::Buffer"));
}

#[test]
fn compile_rejects_transfer_read_without_transfer_src_buffer_usage() {
    let mut graph = RenderGraph::new();
    let buffer = graph.import_buffer_with_access(
        fake_buffer(46),
        4096,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        AccessKind::ComputeShaderWrite,
    );
    graph.add_pass("copy", QueueType::Transfer, |builder| {
        builder.read_as(buffer, AccessKind::TransferRead);
    });
    let err = graph.compile().unwrap_err().to_string();
    assert!(err.contains("requires buffer usage"));
}
```

- [x] **Step 2: Verify red**

Run:

```powershell
cargo test render::graph::tests::compile_plans_buffer_barriers_between_compute_writes_and_reads --lib
cargo test render::graph::tests::imported_buffer_with_access_can_transition_to_compute_write --lib
cargo test render::graph::tests::compile_rejects_null_buffer_barrier_binding --lib
cargo test render::graph::tests::compile_rejects_transfer_read_without_transfer_src_buffer_usage --lib
```

Expected: failures because graph-owned buffer barriers, `bind_buffer`, and `import_buffer_with_access` do not exist yet.

- [x] **Step 3: Implement buffer binding and barrier emission**

Implement:

- `RenderGraph::bind_buffer(handle: ResourceHandle, buffer: vk::Buffer)`.
- `RenderGraph::import_buffer_with_access(buffer: vk::Buffer, size: vk::DeviceSize, usage: vk::BufferUsageFlags, access: AccessKind)`.
- Non-null validation for accessed buffer resources.
- `vk::BufferMemoryBarrier` emission in `record_barriers()` using `AccessKind::stage_flags()` and `AccessKind::access_flags()`.
- Usage validation:
  - `ComputeShaderRead`, `ComputeShaderWrite`, `ComputeShaderReadWrite` require `vk::BufferUsageFlags::STORAGE_BUFFER`.
  - `TransferRead` requires `TRANSFER_SRC`.
  - `TransferWrite` requires `TRANSFER_DST`.

The old error string `graph-owned buffer barriers are not implemented` should disappear from active tests.

- [x] **Step 4: Verify green**

Run:

```powershell
cargo test render::graph::tests::compile_plans_buffer_barriers_between_compute_writes_and_reads --lib
cargo test render::graph::tests::imported_buffer_with_access_can_transition_to_compute_write --lib
cargo test render::graph::tests::compile_rejects_null_buffer_barrier_binding --lib
cargo test render::graph::tests::compile_rejects_transfer_read_without_transfer_src_buffer_usage --lib
```

Expected: test passes.

---

### Task 5: ReSTIR-DI Pass Skeleton And Shader Compilation

**Files:**
- Create: `src/render/passes/restir_di.rs`
- Modify: `src/render/passes/mod.rs`
- Create: `assets/shaders/passes/restir_di_initial.slang`
- Create: `assets/shaders/passes/restir_di_temporal.slang`
- Create: `assets/shaders/passes/restir_di_spatial.slang`
- Test: `src/render/passes/restir_di.rs`

- [x] **Step 1: Write failing source tests**

Create `src/render/passes/restir_di.rs` with source tests:

```rust
#[cfg(test)]
mod shader_source_tests {
    fn source(path: &str) -> String {
        std::fs::read_to_string(path).expect("shader source should be readable")
    }

    #[test]
    fn restir_di_shaders_declare_expected_entry_points_and_resources() {
        let initial = source("assets/shaders/passes/restir_di_initial.slang");
        let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");
        for shader in [&initial, &temporal, &spatial] {
            assert!(shader.contains("#include \"restir_di_common.slang\""));
            assert!(shader.contains("[shader(\"compute\")]"));
            assert!(shader.contains("RestirDiUniforms"));
            assert!(shader.contains("RestirDiReservoir"));
        }
        assert!(initial.contains("StructuredBuffer<DirectLight>"));
        assert!(temporal.contains("history_reservoirs"));
        assert!(spatial.contains("temporal_reservoirs"));
    }

    #[test]
    fn restir_di_pass_does_not_issue_pass_local_barriers() {
        let implementation = std::fs::read_to_string("src/render/passes/restir_di.rs")
            .expect("restir pass source should be readable");
        assert!(!implementation.contains("cmd_pipeline_barrier"));
        assert!(!implementation.contains("ImageMemoryBarrier"));
        assert!(!implementation.contains("BufferMemoryBarrier"));
    }
}
```

- [x] **Step 2: Verify red**

Run:

```powershell
cargo test render::passes::restir_di::shader_source_tests --lib
```

Expected: compile or test failure because module/shaders are missing.

- [x] **Step 3: Add shader skeletons and module export**

Add `pub mod restir_di;` in `src/render/passes/mod.rs`.

Create each shader with the expected includes/resources and a no-op compute body that preserves compilation:

```hlsl
#include "restir_di_common.slang"

[[vk::binding(0, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(1, 0)]]
StructuredBuffer<DirectLight> direct_lights;

[[vk::binding(2, 0)]]
RWStructuredBuffer<RestirDiReservoir> output_reservoirs;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.y * restir.width + tid.x;
    if (tid.x >= restir.width || tid.y >= restir.height || index >= restir.reservoir_count) return;
    RestirDiReservoir reservoir = (RestirDiReservoir)0;
    reservoir.sample_light_id = 0xffffffffu;
    reservoir.sample_count_m = 0u;
    output_reservoirs[index] = reservoir;
}
```

Use similar no-op bodies for temporal/spatial with correctly named resources.

- [x] **Step 4: Verify green**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test render::passes::restir_di::shader_source_tests --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: source tests pass and strict shader compile includes the ReSTIR-DI shader files.

---

### Task 6: App Wiring Behind Disabled Defaults

**Files:**
- Modify: `src/app.rs`
- Modify: `src/render/passes/vpt.rs`
- Modify: `src/render/passes/restir_di.rs`
- Modify: `src/render/gpu_profiler.rs`
- Test: `src/render/passes/vpt.rs`, `src/render/gpu_profiler.rs`

- [x] **Step 1: Write failing wiring tests**

Add tests that assert:

```rust
#[test]
fn app_keeps_restir_di_behind_vpt_setting() {
    let source = std::fs::read_to_string("src/app.rs").expect("app source should be readable");
    assert!(source.contains("RestirDiSettings::from_env"));
    assert!(source.contains("self.lighting_settings.render_mode == RenderMode::Vpt"));
    assert!(source.contains("restir_di_pass"));
}

#[test]
fn vpt_does_not_assume_primary_gbuffer_for_restir_di() {
    let design = std::fs::read_to_string("docs/superpowers/specs/2026-05-02-restir-di-vpt-design.md")
        .expect("design doc should be readable");
    assert!(design.contains("Do not write a ReSTIR-DI pass that silently assumes `gbuffer_pos`"));
}
```

- [x] **Step 2: Verify red**

Run:

```powershell
cargo test render::passes::vpt::shader_source_tests::app_keeps_restir_di_behind_vpt_setting --lib
```

Expected: failure because app wiring does not exist.

- [x] **Step 3: Wire disabled resources only**

Add app fields:

```rust
restir_di_settings: RestirDiSettings,
restir_di_pass: Option<RestirDiPass>,
```

Initialize settings from env. Do not dispatch any ReSTIR-DI pass unless:

```rust
self.lighting_settings.render_mode == RenderMode::Vpt && self.restir_di_settings.enabled
```

When disabled, VPT output and smoke tests must remain byte-for-byte path-equivalent at the graph level: `vpt -> postprocess -> blit`.

- [x] **Step 4: Verify green**

Run:

```powershell
cargo test render::passes::vpt::shader_source_tests --lib
```

Expected: VPT source tests pass, including the new ReSTIR-DI disabled-default checks.

---

### Task 7: VPT Resolve Uses Reservoir When Enabled

**Files:**
- Modify: `assets/shaders/passes/vpt.slang`
- Modify: `src/render/passes/vpt.rs`
- Modify: `src/render/passes/restir_di.rs`

- [ ] **Step 1: Write failing shader source test**

Add a test that requires the VPT shader to include ReSTIR-DI and branch on enabled state:

```rust
#[test]
fn vpt_shader_can_resolve_restir_di_direct_light_when_enabled() {
    let source = normalized_source(include_str!("../../../assets/shaders/passes/vpt.slang"));
    assert!(source.contains("#include \"restir_di_common.slang\""));
    assert!(source.contains("resolve_restir_di_direct_light"));
    assert!(source.contains("if (restir.enabled != 0u)"));
}
```

- [ ] **Step 2: Verify red**

Run:

```powershell
cargo test render::passes::vpt::shader_source_tests::vpt_shader_can_resolve_restir_di_direct_light_when_enabled --lib
```

Expected: failure because VPT does not include or use ReSTIR-DI.

- [ ] **Step 3: Implement shader-side gated resolve**

Add ReSTIR-DI descriptor bindings after existing VPT bindings. Keep disabled behavior equivalent:

```hlsl
[[vk::binding(6, 0)]]
ConstantBuffer<RestirDiUniforms> restir;

[[vk::binding(7, 0)]]
StructuredBuffer<RestirDiReservoir> restir_reservoirs;
```

Add:

```hlsl
float3 resolve_restir_di_direct_light(uint2 pixel, HitResult hit, SceneUniforms scene) {
    uint index = pixel.y * scene.resolution.x + pixel.x;
    RestirDiReservoir reservoir = restir_reservoirs[index];
    if (reservoir.sample_light_id == 0xffffffffu || reservoir.weight_sum <= 0.0) {
        return float3(0.0);
    }
    return reservoir.sample_radiance.rgb * reservoir.selected_weight;
}
```

Replace the direct sun heuristic with:

```hlsl
if (restir.enabled != 0u && bounce == 0u) {
    radiance += throughput * resolve_restir_di_direct_light(uint2(pixel), hit, scene);
} else {
    float sun_term = max(dot(hit.normal, scene.sun_direction), 0.0);
    radiance += throughput * material_cell_albedo(hit.cell) * scene.sun_intensity * sun_term * 0.2;
}
```

- [ ] **Step 4: Verify green**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test render::passes::vpt::shader_source_tests --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: VPT tests pass and strict shader compile succeeds.

---

### Task 8: Full Verification And Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-05-02-restir-di-vpt-design.md`
- Modify: this plan file as checkboxes are completed

- [ ] **Step 1: Update runtime docs**

Document:

- `REVOLUMETRIC_VPT_RESTIR_DI`
- `REVOLUMETRIC_RESTIR_DI_TEMPORAL`
- `REVOLUMETRIC_RESTIR_DI_SPATIAL`
- `REVOLUMETRIC_RESTIR_DI_INITIAL_CANDIDATES`
- `REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES`
- `REVOLUMETRIC_RESTIR_DI_HISTORY_LENGTH`
- `REVOLUMETRIC_RESTIR_DI_DEBUG`

- [ ] **Step 2: Run full validation**

Run:

```powershell
cargo fmt
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --lib; cargo build --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
git diff --check
$env:REVOLUMETRIC_RENDER_MODE='vpt'; $env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; .\target\debug\revolumetric.exe; Remove-Item Env:\REVOLUMETRIC_RENDER_MODE; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
```

Expected:

- Unit tests pass.
- Clippy has no warnings.
- Strict shader compile succeeds.
- `git diff --check` has no whitespace errors. LF/CRLF warnings are acceptable if no error is reported.
- VPT smoke exits by itself after 3 frames.

- [ ] **Step 3: Run residual scans**

Run:

```powershell
rg -n -i "radiance cascade|radiance_cascade|rc_trace|rc_merge|rc_probe|REVOLUMETRIC_RC|LIGHTING_DEBUG_VIEW_RC|\bRC\b" README.md docs src assets/shaders reference
rg -n "cmd_pipeline_barrier|ImageMemoryBarrier|BufferMemoryBarrier" src/render/passes assets/shaders
rg -n "RESTIR|restir|reservoir|REVOLUMETRIC_VPT_RESTIR" README.md docs src assets/shaders
```

Expected:

- RC only appears in migration/deletion-boundary docs or negative tests.
- ReSTIR-DI pass files do not contain pass-local barrier calls.
- ReSTIR-DI docs, settings, shaders, and tests are discoverable by grep.
