# Phase 5: Radiance Cascades GI + Demo Scene — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 3-level world-space Radiance Cascades GI and a Sponza-style demo scene to the voxel engine, replacing the current hemisphere ambient with physically-based indirect lighting.

**Architecture:** 3 sub-phases — (5A) material system + demo scene, (5B) RC trace/merge compute passes with double-buffered temporal probes, (5C) lighting integration. All RC data stored in a single ~12 MB double-buffered StructuredBuffer at brick resolution (8-voxel probe spacing).

**Tech Stack:** Rust + ash (Vulkan 1.3), Slang shaders, bytemuck for GPU structs

**Spec:** `docs/superpowers/specs/2026-04-05-radiance-cascades-design.md`

---

## File Structure

### New files:
- `src/voxel/sponza_generator.rs` — SponzaGenerator implementing VoxelGenerator trait
- `src/render/rc_probe_buffer.rs` — RcProbeBuffer (double-buffered GPU storage buffer)
- `src/render/passes/radiance_cascade_trace.rs` — RcTracePass Vulkan pass
- `src/render/passes/radiance_cascade_merge.rs` — RcMergePass Vulkan pass
- `assets/shaders/passes/rc_trace.slang` — probe ray tracing compute shader
- `assets/shaders/passes/rc_merge.slang` — cascade merging compute shader

### Modified files:
- `assets/shaders/shared/ray.slang` — add VoxelCell field to HitResult
- `assets/shaders/shared/voxel_traverse.slang` — populate hit.cell in trace_primary_ray
- `assets/shaders/shared/radiance_cascade.slang` — fill placeholder with ComputeDir, ProjectDir, integrate_probe
- `assets/shaders/shared/scene_common.slang` — extend SceneUniforms to 176 bytes
- `assets/shaders/passes/primary_ray.slang` — material LUT, write real albedo/emissive to G-buffer
- `assets/shaders/passes/lighting.slang` — add RC probe buffer binding, replace hemisphere_ambient
- `src/render/scene_ubo.rs` — extend GpuSceneUniforms to 176 bytes
- `src/voxel/generator.rs` — add generate_sponza_scene() function
- `src/voxel/mod.rs` — add `pub mod sponza_generator;`
- `src/render/mod.rs` — add `pub mod rc_probe_buffer;`
- `src/render/passes/lighting.rs` — add binding 9 for RC probe buffer
- `src/app.rs` — create RC passes, integrate into render graph, swap double-buffer

---

## Sub-Phase 5A: Material System + Demo Scene

### Task 1: Extend HitResult with VoxelCell

**Files:**
- Modify: `assets/shaders/shared/ray.slang:23-32`
- Modify: `assets/shaders/shared/voxel_traverse.slang:165-172`

- [ ] **Step 1: Add VoxelCell field to HitResult**

In `assets/shaders/shared/ray.slang`, replace the HitResult struct and NO_HIT:

```slang
struct HitResult {
    bool  hit;
    float t;         // distance along ray
    float3 position; // world-space hit point
    float3 normal;   // face normal at hit (axis-aligned, from DDA step)
    uint  brick_id;  // which brick was hit
    uint3 local;     // local voxel coordinate within brick [0,7]³
    VoxelCell cell;  // material + emissive data for the hit voxel
};

static const HitResult NO_HIT = { false, 1e30, float3(0), float3(0), 0xFFFFFFFFu, uint3(0), {0, 0} };
```

Note: `ray.slang` includes `math.slang` but NOT `voxel_common.slang`. Since `VoxelCell` is defined in `voxel_common.slang`, add an include. Check that `ray.slang` doesn't already include it — if not, add `#include "voxel_common.slang"` after the existing `#include "math.slang"`.

- [ ] **Step 2: Populate hit.cell in trace_primary_ray**

In `assets/shaders/shared/voxel_traverse.slang`, after line 171 (`result.local = hit_local;`), add:

```slang
                    result.cell = material_buf[node.brick_id * 512u + morton_encode(hit_local)];
```

- [ ] **Step 3: Build and verify compilation**

Run: `cargo build 2>&1 | head -30`
Expected: Build succeeds (or slangc compiles without errors). The new `cell` field is populated but not yet used — existing shaders are unaffected.

- [ ] **Step 4: Commit**

```bash
git add assets/shaders/shared/ray.slang assets/shaders/shared/voxel_traverse.slang
git commit -m "feat(shader): add VoxelCell to HitResult for material access"
```

---

### Task 2: Material LUT in primary_ray.slang

**Files:**
- Modify: `assets/shaders/passes/primary_ray.slang:57-65`

- [ ] **Step 1: Add material LUT and write real albedo/emissive**

In `assets/shaders/passes/primary_ray.slang`, add the material LUT before the `main` function:

```slang
static const float3 MATERIAL_ALBEDO[6] = {
    float3(1.0, 1.0, 1.0),       // 0: default white
    float3(0.95, 0.925, 0.9),    // 1: stone
    float3(0.99, 0.4, 0.4),     // 2: red cloth
    float3(0.4, 0.99, 0.4),     // 3: green cloth
    float3(0.5, 0.6, 0.9),      // 4: blue
    float3(0.75, 0.65, 0.5),    // 5: brick
};
```

Then replace the hit branch (lines 57-60) from:

```slang
    if (hit.hit) {
        gbuffer_pos[tid.xy] = float4(hit.position, hit.t);
        gbuffer0[tid.xy] = float4(1.0, 1.0, 1.0, 1.0);  // white albedo, no AO
        gbuffer1[tid.xy] = uint4(encode_normal_id(hit.normal), 128, 0, 0x01);
```

To:

```slang
    if (hit.hit) {
        gbuffer_pos[tid.xy] = float4(hit.position, hit.t);
        uint mat_id = voxel_material(hit.cell);
        float3 albedo = MATERIAL_ALBEDO[min(mat_id, 5u)];
        gbuffer0[tid.xy] = float4(albedo, 1.0);
        float3 emissive = voxel_emissive(hit.cell);
        uint3 emissive_raw = uint3(emissive * 255.0);
        gbuffer1[tid.xy] = uint4(encode_normal_id(hit.normal), emissive_raw.r, emissive_raw.g, emissive_raw.b);
```

- [ ] **Step 2: Build and verify**

Run: `cargo build 2>&1 | head -30`
Expected: Compiles. The existing sphere scene uses material=1 (stone), so it should now render with slightly off-white stone color instead of pure white.

- [ ] **Step 3: Commit**

```bash
git add assets/shaders/passes/primary_ray.slang
git commit -m "feat(shader): material LUT + real albedo/emissive in G-buffer"
```

---

### Task 3: Emissive handling in lighting.slang

**Files:**
- Modify: `assets/shaders/passes/lighting.slang:67-92`

- [ ] **Step 1: Add emissive early-out in lighting shader**

In `assets/shaders/passes/lighting.slang`, after reading the G-buffer data (after line 72 `float3 normal = NORMAL_TABLE[min(normal_id, 5u)];`), add emissive handling:

```slang
    // Emissive early-out: if any emissive channel is non-zero, output it directly
    uint3 emissive_raw = uint3(gb1.g, gb1.b, gb1.a);
    if (emissive_raw.x + emissive_raw.y + emissive_raw.z > 0u) {
        float3 emissive = float3(emissive_raw) / 255.0;
        // Apply gamma only (emissive is already HDR-ish from raw values)
        output_image[tid.xy] = float4(pow(emissive, float3(1.0 / 2.2)), 1.0);
        return;
    }
```

- [ ] **Step 2: Build and verify**

Run: `cargo build 2>&1 | head -30`
Expected: Compiles. No emissive voxels in current scene yet, so behavior is unchanged.

- [ ] **Step 3: Commit**

```bash
git add assets/shaders/passes/lighting.slang
git commit -m "feat(shader): emissive voxel early-out in lighting pass"
```

---

### Task 4: SponzaGenerator — Rust-side scene generation

**Files:**
- Create: `src/voxel/sponza_generator.rs`
- Modify: `src/voxel/mod.rs`
- Modify: `src/voxel/generator.rs`

- [ ] **Step 1: Create sponza_generator.rs**

Create `src/voxel/sponza_generator.rs`:

```rust
use glam::{UVec3, Vec3};
use crate::voxel::brick::{BrickData, VoxelCell, BRICK_EDGE};
use crate::voxel::ucvh::UcvhConfig;
use crate::voxel::generator::VoxelGenerator;

/// Material IDs matching the shader-side MATERIAL_ALBEDO LUT.
const MAT_STONE: u16 = 1;
const MAT_RED_CLOTH: u16 = 2;
const MAT_GREEN_CLOTH: u16 = 3;
const MAT_BLUE: u16 = 4;
const MAT_BRICK: u16 = 5;

/// Sponza-inspired architectural scene for 128³ world.
/// Occupies approximately 96×96×120 voxels centered in the world.
pub struct SponzaGenerator;

impl SponzaGenerator {
    /// Evaluate a single voxel position and return (material_id, emissive_rgb) or None if air.
    fn eval_voxel(p: Vec3) -> Option<(u16, [u8; 3])> {
        // Scene is centered at (64, 0, 64), extends roughly 96×96×120
        // Coordinate system: x=width, y=height, z=depth
        let offset = Vec3::new(16.0, 0.0, 4.0);
        let v = p - offset;

        // Bounding box / outer walls (96×96×120)
        if v.x < 0.0 || v.x > 96.0 || v.y < 0.0 || v.y > 96.0 || v.z < 0.0 || v.z > 120.0 {
            return None;
        }

        // --- Floor (y=0..2) ---
        if v.y < 2.0 {
            // Checkered floor pattern
            let cx = ((v.x / 4.0).floor() as i32) & 1;
            let cz = ((v.z / 4.0).floor() as i32) & 1;
            if cx ^ cz == 0 {
                return Some((MAT_STONE, [0; 3]));
            }
            return Some((MAT_STONE, [0; 3]));
        }

        // --- Ceiling (y=88..92) ---
        if v.y > 88.0 && v.y < 92.0 {
            return Some((MAT_STONE, [0; 3]));
        }

        // --- Outer walls (thickness=2) ---
        let wall_inner = v.x > 2.0 && v.x < 94.0 && v.z > 2.0 && v.z < 118.0;
        if !wall_inner && v.y < 92.0 {
            // X walls
            if v.x < 2.0 || v.x > 94.0 {
                // Brick pattern on X+ wall
                if v.x < 2.0 {
                    let by = ((v.y / 4.0).floor() as i32) & 1;
                    let bz_off = if by == 0 { 0.0 } else { 2.0 };
                    if ((v.z + bz_off) % 4.0) > 3.0 || (v.y % 4.0) > 3.0 {
                        return Some((MAT_STONE, [0; 3]));
                    }
                    return Some((MAT_BRICK, [0; 3]));
                }
                return Some((MAT_STONE, [0; 3]));
            }
            // Z walls
            if v.z < 2.0 || v.z > 118.0 {
                return Some((MAT_STONE, [0; 3]));
            }
        }

        // --- Columns (stone, r=5, every 24 units along Z, at x=24 and x=72) ---
        for &cx in &[24.0_f32, 72.0] {
            let mut cz = 18.0;
            while cz < 110.0 {
                let dx = v.x - cx;
                let dz = v.z - cz;
                let dist_sq = dx * dx + dz * dz;
                if dist_sq < 25.0 && v.y < 88.0 {
                    return Some((MAT_STONE, [0; 3]));
                }
                cz += 24.0;
            }
        }

        // --- Arches between columns (simplified: stone slabs at top of each column pair) ---
        for &cx in &[24.0_f32, 72.0] {
            let mut cz = 18.0;
            while cz + 24.0 < 110.0 {
                let mid_z = cz + 12.0;
                if v.x > cx - 6.0 && v.x < cx + 6.0 && v.z > cz - 1.0 && v.z < cz + 25.0 {
                    // Arch shape: semicircle at y = 75..85
                    if v.y > 75.0 && v.y < 85.0 {
                        let arch_r = 12.0;
                        let dy = v.y - 75.0;
                        let dz = (v.z - mid_z).abs();
                        if dy * dy + dz * dz > arch_r * arch_r {
                            return Some((MAT_STONE, [0; 3]));
                        }
                    }
                }
                cz += 24.0;
            }
        }

        // --- Second floor (y=46..48) at x=0..48 ---
        if v.x < 48.0 && v.y > 46.0 && v.y < 48.0 {
            return Some((MAT_STONE, [0; 3]));
        }

        // --- Red cloth banner (thin, at z=40, x=30..36, y=20..70) ---
        if (v.z - 40.0).abs() < 0.6 && v.x > 30.0 && v.x < 36.0 && v.y > 20.0 && v.y < 70.0 {
            return Some((MAT_RED_CLOTH, [0; 3]));
        }

        // --- Green cloth banner (thin, at z=72, x=30..36, y=20..70) ---
        if (v.z - 72.0).abs() < 0.6 && v.x > 30.0 && v.x < 36.0 && v.y > 20.0 && v.y < 70.0 {
            return Some((MAT_GREEN_CLOTH, [0; 3]));
        }

        // --- Fountain base (cylindrical, at x=48, z=100) ---
        {
            let dx = v.x - 48.0;
            let dz = v.z - 100.0;
            let dist_sq = dx * dx + dz * dz;
            // Base ring
            if v.y < 8.0 && dist_sq < 64.0 && (v.y < 3.0 || dist_sq > 36.0) {
                return Some((MAT_STONE, [0; 3]));
            }
            // Central pillar
            if v.y < 14.0 && dist_sq < 4.0 {
                return Some((MAT_STONE, [0; 3]));
            }
        }

        // --- Emissive lamps ---
        // Lamp 1: hanging above fountain (x=48, y=20, z=100)
        {
            let d = (p - Vec3::new(64.0, 20.0, 104.0)).length();
            if d < 2.5 {
                return Some((MAT_STONE, [255, 150, 50]));
            }
        }
        // Lamp 2: wall sconce (x=16.5, y=30, z=60)
        {
            let d = (p - Vec3::new(16.5, 30.0, 64.0)).length();
            if d < 2.0 {
                return Some((MAT_STONE, [200, 140, 60]));
            }
        }
        // Lamp 3: wall sconce other side (x=110.5, y=30, z=60)
        {
            let d = (p - Vec3::new(110.5, 30.0, 64.0)).length();
            if d < 2.0 {
                return Some((MAT_STONE, [200, 140, 60]));
            }
        }

        // --- Blue sphere (static, at x=72, y=58, z=30) ---
        {
            let d = (p - Vec3::new(88.0, 58.0, 34.0)).length();
            if d < 8.0 {
                return Some((MAT_BLUE, [0; 3]));
            }
        }

        // --- Decorative spherical wall bumps on Z+ wall ---
        if v.z > 116.0 && v.z < 118.0 {
            let mut sx = 12.0;
            while sx < 90.0 {
                let mut sy = 12.0;
                while sy < 80.0 {
                    let dx = v.x - sx;
                    let dy = v.y - sy;
                    if dx * dx + dy * dy < 9.0 {
                        return Some((MAT_STONE, [0; 3]));
                    }
                    sy += 16.0;
                }
                sx += 16.0;
            }
        }

        None
    }
}

impl VoxelGenerator for SponzaGenerator {
    fn generate_brick(&self, brick_pos: UVec3, _config: &UcvhConfig) -> Option<BrickData> {
        let base = brick_pos * BRICK_EDGE;
        let mut data = BrickData::new();
        let mut any_solid = false;

        for lz in 0..BRICK_EDGE {
            for ly in 0..BRICK_EDGE {
                for lx in 0..BRICK_EDGE {
                    let world = Vec3::new(
                        (base.x + lx) as f32 + 0.5,
                        (base.y + ly) as f32 + 0.5,
                        (base.z + lz) as f32 + 0.5,
                    );
                    if let Some((material, emissive)) = Self::eval_voxel(world) {
                        data.set_voxel(lx, ly, lz, VoxelCell::new(material, 1, emissive));
                        any_solid = true;
                    }
                }
            }
        }

        if any_solid { Some(data) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::ucvh::UcvhConfig;

    #[test]
    fn floor_is_solid() {
        // Voxel at (64, 0.5, 64) should be floor
        let result = SponzaGenerator::eval_voxel(Vec3::new(64.0, 0.5, 64.0));
        assert!(result.is_some(), "floor voxel should be solid");
        let (mat, _) = result.unwrap();
        assert_eq!(mat, MAT_STONE);
    }

    #[test]
    fn center_air_is_empty() {
        // Voxel at (64, 40, 64) should be air (inside the building, not on any structure)
        let result = SponzaGenerator::eval_voxel(Vec3::new(64.0, 40.0, 64.0));
        assert!(result.is_none(), "center should be air");
    }

    #[test]
    fn lamp_is_emissive() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(64.0, 20.0, 104.0));
        assert!(result.is_some());
        let (_, emissive) = result.unwrap();
        assert!(emissive[0] > 0, "lamp should have emissive");
    }

    #[test]
    fn red_cloth_has_correct_material() {
        let result = SponzaGenerator::eval_voxel(Vec3::new(33.0, 40.0, 40.0));
        assert!(result.is_some());
        let (mat, _) = result.unwrap();
        assert_eq!(mat, MAT_RED_CLOTH);
    }

    #[test]
    fn sponza_generates_bricks() {
        let config = UcvhConfig::new(UVec3::splat(128));
        let gen = SponzaGenerator;
        // Floor brick at (2, 0, 2) should have content
        let data = gen.generate_brick(UVec3::new(2, 0, 8), &config);
        assert!(data.is_some(), "floor brick should be non-empty");
    }
}
```

- [ ] **Step 2: Add VoxelCell::new constructor if missing**

Check `src/voxel/brick.rs` for the VoxelCell impl. The current code has `VoxelCell::AIR` but may not have a `new()` constructor. If missing, add to the existing `impl VoxelCell` block in `src/voxel/brick.rs`:

```rust
    pub fn new(material: u16, flags: u16, emissive: [u8; 3]) -> Self {
        Self { material, flags, emissive, _pad: 0 }
    }
```

- [ ] **Step 3: Add module to mod.rs**

In `src/voxel/mod.rs`, add:

```rust
pub mod sponza_generator;
```

- [ ] **Step 4: Add generate_sponza_scene to generator.rs**

In `src/voxel/generator.rs`, add a new function after `generate_demo_scene`:

```rust
/// Generate a Sponza-inspired demo scene in a 128³ world.
pub fn generate_sponza_scene(ucvh: &mut Ucvh) -> u32 {
    let gen = crate::voxel::sponza_generator::SponzaGenerator;
    let bgs = ucvh.config.brick_grid_size;
    let mut count = 0u32;
    for bz in 0..bgs.z {
        for by in 0..bgs.y {
            for bx in 0..bgs.x {
                let bp = UVec3::new(bx, by, bz);
                if let Some(data) = gen.generate_brick(bp, &ucvh.config) {
                    if ucvh.write_brick(bp, &data) {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p revolumetric -- sponza`
Expected: All 5 sponza tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/voxel/sponza_generator.rs src/voxel/mod.rs src/voxel/generator.rs src/voxel/brick.rs
git commit -m "feat(voxel): Sponza-style demo scene generator"
```

---

### Task 5: Switch app.rs to Sponza scene

**Files:**
- Modify: `src/app.rs:428-448`

- [ ] **Step 1: Replace generate_demo_scene with generate_sponza_scene**

In `src/app.rs`, line 432, change:

```rust
            let brick_count = generator::generate_demo_scene(&mut ucvh);
```

To:

```rust
            let brick_count = generator::generate_sponza_scene(&mut ucvh);
```

Also update the log message on line 437:

```rust
                "generated sponza demo scene"
```

- [ ] **Step 2: Build and run**

Run: `cargo build && cargo run`
Expected: The app renders the Sponza scene with colored materials (stone walls, red/green banners, blue sphere, glowing lamps). The existing shadow + hemisphere ambient lighting is applied.

- [ ] **Step 3: Commit**

```bash
git add src/app.rs
git commit -m "feat(app): switch to Sponza demo scene"
```

---

## Sub-Phase 5B: Radiance Cascades Core

### Task 6: Extend SceneUniforms to 176 bytes

**Files:**
- Modify: `src/render/scene_ubo.rs:13-25`
- Modify: `assets/shaders/shared/scene_common.slang:1-16`
- Modify: `src/render/passes/lighting.rs:78` (UBO range)
- Modify: `src/render/passes/primary_ray.rs` (UBO range — find `.range(144)`)
- Modify: `src/app.rs:213-225` (scene_data construction)

- [ ] **Step 1: Extend Rust GpuSceneUniforms**

In `src/render/scene_ubo.rs`, replace the struct (lines 13-25):

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSceneUniforms {
    pub pixel_to_ray: [[f32; 4]; 4], // 64B — col 0-2: direction matrix, col 3: camera origin
    pub resolution: [u32; 2],         // 8B
    pub _pad0: [u32; 2],             // 8B
    pub sun_direction: [f32; 3],     // 12B — normalized, world space, points TOWARD sun
    pub _pad1: f32,                  // 4B
    pub sun_intensity: [f32; 3],     // 12B — HDR color * intensity
    pub _pad2: f32,                  // 4B
    pub sky_color: [f32; 3],         // 12B — hemisphere ambient upper
    pub _pad3: f32,                  // 4B
    pub ground_color: [f32; 3],      // 12B — hemisphere ambient lower
    pub time: f32,                   // 4B
    // --- RC fields (new) ---
    pub rc_c0_grid: [u32; 3],       // 12B — C0 probe grid dimensions (16,16,16)
    pub rc_c0_offset: u32,          // 4B  — offset into probe buffer for C0
    pub rc_enabled: u32,             // 4B  — 0 or 1
    pub _pad4: [u32; 3],            // 12B — pad to 176B
}
```

Update the test on line 92-93:

```rust
    fn gpu_scene_uniforms_size_is_176_bytes() {
        assert_eq!(std::mem::size_of::<GpuSceneUniforms>(), 176);
    }
```

- [ ] **Step 2: Extend Slang SceneUniforms**

In `assets/shaders/shared/scene_common.slang`, replace the struct:

```slang
// Scene-wide uniforms shared by all passes.
// Must match Rust GpuSceneUniforms in src/render/scene_ubo.rs exactly (176 bytes).

struct SceneUniforms {
    float4x4 pixel_to_ray;   // 64B — col 0-2: direction matrix, col 3: camera origin
    uint2    resolution;      // 8B
    uint2    _pad0;           // 8B
    float3   sun_direction;   // 12B — normalized, world space, points TOWARD sun
    float    _pad1;           // 4B
    float3   sun_intensity;   // 12B — HDR color * intensity
    float    _pad2;           // 4B
    float3   sky_color;       // 12B — hemisphere ambient upper
    float    _pad3;           // 4B
    float3   ground_color;    // 12B — hemisphere ambient lower
    float    time;            // 4B
    // --- RC fields ---
    uint3    rc_c0_grid;      // 12B — next uint packs into remaining 4B of this 16B row
    uint     rc_c0_offset;    // 4B
    uint     rc_enabled;      // 4B
    uint3    _pad4;           // 12B — pad to 176B
};                            // total: 176B, 16-byte aligned
```

- [ ] **Step 3: Update UBO range in descriptor writes**

Search all Rust files for `.range(144)` and replace with `.range(176)`. Expected locations:
- `src/render/passes/lighting.rs` line ~78
- `src/render/passes/primary_ray.rs` (search for `.range(144)`)

- [ ] **Step 4: Update scene_data construction in app.rs**

In `src/app.rs`, update the `GpuSceneUniforms` construction (line ~213) to include new fields:

```rust
                    let scene_data = GpuSceneUniforms {
                        pixel_to_ray: pixel_to_ray.transpose().to_cols_array_2d(),
                        resolution: [frame.swapchain_extent.width, frame.swapchain_extent.height],
                        _pad0: [0; 2],
                        sun_direction: sun_dir.to_array(),
                        _pad1: 0.0,
                        sun_intensity: sun_intensity.to_array(),
                        _pad2: 0.0,
                        sky_color: [0.4, 0.5, 0.7],
                        _pad3: 0.0,
                        ground_color: [0.15, 0.1, 0.08],
                        time: self.world.resource::<Time>().map_or(0.0, |t| t.elapsed_seconds),
                        rc_c0_grid: [16, 16, 16],
                        rc_c0_offset: 0,
                        rc_enabled: 0,  // disabled until RC passes are wired up
                        _pad4: [0; 3],
                    };
```

- [ ] **Step 5: Run tests and build**

Run: `cargo test -- scene_uniforms && cargo build`
Expected: Size test passes with 176, build succeeds.

- [ ] **Step 6: Commit**

```bash
git add src/render/scene_ubo.rs assets/shaders/shared/scene_common.slang src/render/passes/lighting.rs src/render/passes/primary_ray.rs src/app.rs
git commit -m "feat(ubo): extend SceneUniforms to 176B with RC fields"
```

---

### Task 7: RcProbeBuffer — double-buffered GPU storage

**Files:**
- Create: `src/render/rc_probe_buffer.rs`
- Modify: `src/render/mod.rs`

- [ ] **Step 1: Create rc_probe_buffer.rs**

Create `src/render/rc_probe_buffer.rs`:

```rust
use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

/// Cascade layout constants.
/// C0: 16³ probes × 6 faces × 9 dirs = 221,184 entries
/// C1: 8³ probes × 6 faces × 36 dirs = 110,592 entries
/// C2: 4³ probes × 6 faces × 144 dirs = 55,296 entries
pub const RC_C0_ENTRIES: u32 = 4096 * 6 * 9;       // 221,184
pub const RC_C1_ENTRIES: u32 = 512 * 6 * 36;        // 110,592
pub const RC_C2_ENTRIES: u32 = 64 * 6 * 144;        // 55,296
pub const RC_TOTAL_ENTRIES: u32 = RC_C0_ENTRIES + RC_C1_ENTRIES + RC_C2_ENTRIES; // 387,072

pub const RC_C0_OFFSET: u32 = 0;
pub const RC_C1_OFFSET: u32 = RC_C0_ENTRIES;        // 221,184
pub const RC_C2_OFFSET: u32 = RC_C0_ENTRIES + RC_C1_ENTRIES; // 331,776

/// Entry size: float4 = 16 bytes (radiance.rgb + ray_distance).
const ENTRY_SIZE: vk::DeviceSize = 16;

/// Double-buffered probe storage for Radiance Cascades.
/// Frame N reads buffer[prev], writes buffer[curr]. Swap at end of frame.
pub struct RcProbeBuffer {
    pub buffers: [GpuBuffer; 2],
    pub current: usize,
}

impl RcProbeBuffer {
    pub fn new(device: &ash::Device, allocator: &GpuAllocator) -> Result<Self> {
        let size = RC_TOTAL_ENTRIES as vk::DeviceSize * ENTRY_SIZE;
        let mut bufs: Vec<GpuBuffer> = Vec::with_capacity(2);
        for i in 0..2 {
            let buf = GpuBuffer::new(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                &format!("rc_probe_buffer_{i}"),
            )?;
            bufs.push(buf);
        }

        // Zero-initialize both buffers
        // Note: actual clearing happens via vkCmdFillBuffer during first frame upload
        // (GpuOnly memory can't be mapped)

        Ok(Self {
            buffers: [bufs.remove(0), bufs.remove(0)],
            current: 0,
        })
    }

    /// Buffer that rc_trace WRITES to this frame.
    pub fn write_buffer(&self) -> vk::Buffer {
        self.buffers[self.current].handle
    }

    /// Buffer that rc_trace READS from this frame (previous frame's data).
    pub fn read_buffer(&self) -> vk::Buffer {
        self.buffers[1 - self.current].handle
    }

    /// Size of the entire probe buffer in bytes.
    pub fn buffer_size(&self) -> vk::DeviceSize {
        self.buffers[0].size
    }

    /// Swap current/previous at end of frame.
    pub fn swap(&mut self) {
        self.current ^= 1;
    }

    /// Record vkCmdFillBuffer to zero-initialize both buffers.
    /// Call once during first frame before any RC dispatch.
    pub fn record_clear(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
    ) {
        let size = self.buffer_size();
        unsafe {
            device.cmd_fill_buffer(cmd, self.buffers[0].handle, 0, size, 0);
            device.cmd_fill_buffer(cmd, self.buffers[1].handle, 0, size, 0);
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        let [b0, b1] = self.buffers;
        b0.destroy(device, allocator);
        b1.destroy(device, allocator);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_offsets_are_correct() {
        assert_eq!(RC_C0_OFFSET, 0);
        assert_eq!(RC_C1_OFFSET, 221_184);
        assert_eq!(RC_C2_OFFSET, 331_776);
        assert_eq!(RC_TOTAL_ENTRIES, 387_072);
    }

    #[test]
    fn buffer_size_is_about_6mb() {
        let size = RC_TOTAL_ENTRIES as u64 * 16;
        assert_eq!(size, 6_193_152); // 6.2 MB
    }
}
```

- [ ] **Step 2: Add module to render/mod.rs**

In `src/render/mod.rs`, add:

```rust
pub mod rc_probe_buffer;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -- rc_probe_buffer`
Expected: Both tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/render/rc_probe_buffer.rs src/render/mod.rs
git commit -m "feat(render): RcProbeBuffer with double-buffered GPU storage"
```

---

### Task 8: Hemisphere direction functions in radiance_cascade.slang

**Files:**
- Modify: `assets/shaders/shared/radiance_cascade.slang` (replace placeholder)

- [ ] **Step 1: Port ComputeDir, ComputeDirEven, ProjectDir from M3ycWt**

Replace the placeholder content of `assets/shaders/shared/radiance_cascade.slang` with the ported functions. Reference: `reference/shadertoy_M3ycWt/common.glsl` lines 96-136.

```slang
// Radiance Cascade shared helpers.
// Ported from Shadertoy M3ycWt (common.glsl).

#include "voxel_common.slang"

static const float RC_PI = 3.141592653;

// --- Probe direction mapping ---

// Concentric-square → hemisphere direction (for probeSize > 4).
float3 ComputeDirEven(float2 uv, float probeSize) {
    float2 probeRel = uv - probeSize * 0.5;
    float probeThetai = max(abs(probeRel.x), abs(probeRel.y));
    float probeTheta = probeThetai / probeSize * RC_PI;
    float probePhi = 0.0;
    if (probeRel.x + 0.5 > probeThetai && probeRel.y - 0.5 > -probeThetai) {
        probePhi = probeRel.x - probeRel.y;
    } else if (probeRel.y - 0.5 < -probeThetai && probeRel.x - 0.5 > -probeThetai) {
        probePhi = probeThetai * 2.0 - probeRel.y - probeRel.x;
    } else if (probeRel.x - 0.5 < -probeThetai && probeRel.y + 0.5 < probeThetai) {
        probePhi = probeThetai * 4.0 - probeRel.x + probeRel.y;
    } else if (probeRel.y + 0.5 > probeThetai && probeRel.x + 0.5 < probeThetai) {
        probePhi = probeThetai * 8.0 - (probeRel.y - probeRel.x);
    }
    probePhi = probePhi * RC_PI * 2.0 / (4.0 + 8.0 * floor(probeThetai));
    return float3(float2(sin(probePhi), cos(probePhi)) * sin(probeTheta), cos(probeTheta));
}

// Map 2D grid position to hemisphere direction.
// probeSize <= 3: simple 8-direction + center.
// probeSize > 4: concentric-square even distribution.
float3 ComputeDir(float2 uv, float probeSize) {
    if (probeSize > 4.5) return ComputeDirEven(uv, probeSize);
    float2 probeRel = uv - 1.5;
    if (length(probeRel) < 0.1) return float3(0.0, 0.0, 1.0);
    float probePhi = atan2(probeRel.x, probeRel.y) + RC_PI * 1.75;
    float probeTheta = RC_PI * 0.25;
    return float3(float2(sin(probePhi), cos(probePhi)) * sin(probeTheta), cos(probeTheta));
}

// Inverse: map 3D direction to probe grid UV.
// Returns (-1,-1) if direction points away from hemisphere.
float2 ProjectDir(float3 dir, float probeSize) {
    if (dir.z <= 0.0) return float2(-1.0);
    float thetai = min(floor((1.0 - acos(length(dir.xy) / length(dir)) / (RC_PI * 0.5)) * (probeSize * 0.5)), probeSize * 0.5 - 1.0);
    float phiF = atan2(-dir.x, -dir.y);
    float phiI = floor((phiF / RC_PI * 0.5 + 0.5) * (4.0 + 8.0 * thetai) + 0.5) + 0.5;
    float2 phiUV;
    float phiLen = 2.0 * thetai + 1.0;
    float sideLen = phiLen + 1.0;
    if (phiI < phiLen) phiUV = float2(sideLen - 0.5, sideLen - phiI);
    else if (phiI < phiLen * 2.0) phiUV = float2(sideLen - (phiI - phiLen), 0.5);
    else if (phiI < phiLen * 3.0) phiUV = float2(0.5, phiI - phiLen * 2.0);
    else phiUV = float2(phiI - phiLen * 3.0, sideLen - 0.5);
    return float2((probeSize - sideLen) * 0.5) + phiUV;
}

// --- Probe face TBN ---
// Canonical face ordering (matches encode_normal_id):
//   0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z

void get_face_tbn(uint face, out float3 normal, out float3 tangent, out float3 bitangent) {
    if (face == 0)      { normal = float3( 1, 0, 0); tangent = float3(0, 0, 1); bitangent = float3(0, 1, 0); }
    else if (face == 1) { normal = float3(-1, 0, 0); tangent = float3(0, 0, 1); bitangent = float3(0, 1, 0); }
    else if (face == 2) { normal = float3(0,  1, 0); tangent = float3(1, 0, 0); bitangent = float3(0, 0, 1); }
    else if (face == 3) { normal = float3(0, -1, 0); tangent = float3(1, 0, 0); bitangent = float3(0, 0, 1); }
    else if (face == 4) { normal = float3(0, 0,  1); tangent = float3(1, 0, 0); bitangent = float3(0, 1, 0); }
    else                { normal = float3(0, 0, -1); tangent = float3(1, 0, 0); bitangent = float3(0, 1, 0); }
}

// --- Probe buffer indexing ---

uint rc_probe_index(uint3 probe_coord, uint grid_dim, uint dirs_per_face, uint face, uint dir_index, uint buffer_offset) {
    uint probe_linear = (probe_coord.z * grid_dim + probe_coord.y) * grid_dim + probe_coord.x;
    return buffer_offset + probe_linear * 6u * dirs_per_face + face * dirs_per_face + dir_index;
}

// --- Geometry offset (find nearby empty voxel for probe placement) ---

bool rc_outside_geo(float3 sp,
                    StructuredBuffer<NodeL0> hierarchy_l0,
                    StructuredBuffer<BrickOccupancy> occupancy_buf,
                    uint grid_dim_l0) {
    int3 p = int3(floor(sp));
    if (any(p < int3(0)) || any(p >= int3(128))) return true;
    int3 brick_coord = p / 8;
    uint l0_idx = uint(brick_coord.x) + uint(brick_coord.y) * grid_dim_l0 + uint(brick_coord.z) * grid_dim_l0 * grid_dim_l0;
    NodeL0 node = hierarchy_l0[l0_idx];
    if (node_l0_is_empty(node)) return true;
    BrickOccupancy occ = occupancy_buf[node.brick_id];
    uint3 local = uint3(p) % 8u;
    return !read_occupancy_bit(occ, local);
}

float3 rc_geo_offset(float3 vertex,
                     StructuredBuffer<NodeL0> hierarchy_l0,
                     StructuredBuffer<BrickOccupancy> occupancy_buf,
                     uint grid_dim_l0) {
    for (float x = -0.5; x < 1.0; x += 1.0) {
        for (float y = -0.5; y < 1.0; y += 1.0) {
            for (float z = -0.5; z < 1.0; z += 1.0) {
                if (rc_outside_geo(vertex + float3(x, y, z), hierarchy_l0, occupancy_buf, grid_dim_l0))
                    return float3(x, y, z) * 0.1;
            }
        }
    }
    return float3(0.001);
}

// --- integrate_probe (for lighting pass, visibility-weighted trilinear) ---

uint normal_to_face(float3 n) {
    return encode_normal_id(n);
}

float3 read_probe_face(uint3 probe_coord, uint face, uint3 grid_size,
                       uint c0_offset, StructuredBuffer<float4> rc_probes) {
    uint base = c0_offset +
        ((probe_coord.z * grid_size.y + probe_coord.y) * grid_size.x + probe_coord.x)
        * 6u * 9u + face * 9u;
    float3 total = float3(0.0);
    for (uint i = 0; i < 9; i++) {
        total += rc_probes[base + i].xyz;
    }
    return total;
}

float probe_visibility(uint3 probe_coord, float3 world_pos, uint face,
                       uint3 grid_size, uint c0_offset, float probe_spacing,
                       StructuredBuffer<float4> rc_probes) {
    float3 probe_world = (float3(probe_coord) + 0.5) * probe_spacing;
    float dist = length(world_pos - probe_world);
    uint center_idx = c0_offset +
        ((probe_coord.z * grid_size.y + probe_coord.y) * grid_size.x + probe_coord.x)
        * 6u * 9u + face * 9u + 4u;
    float ray_dist = rc_probes[center_idx].w;
    return (ray_dist < dist - 0.5) ? 0.00001 : 1.0;
}

float3 integrate_probe(float3 world_pos, float3 normal,
                       StructuredBuffer<float4> rc_probes,
                       uint c0_offset, uint3 c0_grid_size) {
    uint face = normal_to_face(normal);
    float spacing = 8.0;

    float3 grid_pos = world_pos / spacing - 0.5;
    int3 base_coord = int3(floor(grid_pos));
    float3 frac_pos = grid_pos - float3(base_coord);

    float3 total_radiance = float3(0.0);
    float total_weight = 0.0;

    for (uint i = 0; i < 8; i++) {
        int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        int3 coord = base_coord + offset;
        uint3 clamped = uint3(clamp(coord, int3(0), int3(c0_grid_size) - 1));

        float3 w = lerp(1.0 - frac_pos, frac_pos, float3(offset));
        float trilinear_w = w.x * w.y * w.z;

        float vis_w = probe_visibility(clamped, world_pos, face,
                                       c0_grid_size, c0_offset, spacing, rc_probes);

        float weight = trilinear_w * vis_w;
        total_radiance += read_probe_face(clamped, face, c0_grid_size, c0_offset, rc_probes) * weight;
        total_weight += weight;
    }

    return (total_weight > 0.0001) ? total_radiance / total_weight : float3(0.0);
}
```

- [ ] **Step 2: Build to verify Slang compilation**

Run: `cargo build 2>&1 | head -30`
Expected: Compiles (this file is a shared include, not a pass entry point, so it won't be compiled standalone — but any pass that includes it will compile).

- [ ] **Step 3: Commit**

```bash
git add assets/shaders/shared/radiance_cascade.slang
git commit -m "feat(shader): RC direction mapping, probe indexing, integrate_probe"
```

---

### Task 9: rc_trace.slang — probe ray tracing compute shader

**Files:**
- Create: `assets/shaders/passes/rc_trace.slang`

- [ ] **Step 1: Write rc_trace.slang**

Create `assets/shaders/passes/rc_trace.slang`:

```slang
// Radiance Cascade probe ray tracing.
// One invocation per (probe, face, direction). Traces a ray through UCVH,
// writes float4(weighted_radiance, ray_distance) to the probe buffer.

#include "voxel_traverse.slang"
#include "lighting_common.slang"
#include "radiance_cascade.slang"

// Push constants
struct RcTracePush {
    uint cascade_level;
    uint probe_grid_dim;
    uint probe_size;       // directions per face edge (3, 6, or 12)
    uint buffer_offset;
};
[[vk::push_constant]] RcTracePush push;

// Binding 0: Scene UBO
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

// Binding 1-4: UCVH buffers
[[vk::binding(1, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

[[vk::binding(2, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(3, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(4, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

// Binding 5: Probe buffer READ (previous frame)
[[vk::binding(5, 0)]]
StructuredBuffer<float4> probe_read;

// Binding 6: Probe buffer WRITE (current frame)
[[vk::binding(6, 0)]]
RWStructuredBuffer<float4> probe_write;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    uint grid_dim = push.probe_grid_dim;
    uint probe_size = push.probe_size;
    uint dirs_per_face = probe_size * probe_size;

    // Total invocations = grid_dim³ × 6 × dirs_per_face
    uint total = grid_dim * grid_dim * grid_dim * 6u * dirs_per_face;
    uint tid = dtid.x;
    if (tid >= total) return;

    // Decode thread ID
    uint dir_index = tid % dirs_per_face;
    uint remainder = tid / dirs_per_face;
    uint face = remainder % 6u;
    uint probe_linear = remainder / 6u;
    uint3 probe_coord = uint3(
        probe_linear % grid_dim,
        (probe_linear / grid_dim) % grid_dim,
        probe_linear / (grid_dim * grid_dim)
    );

    // Probe world position
    float lod_factor = float(1u << push.cascade_level); // 1, 2, or 4
    float probe_spacing = 8.0 * lod_factor;
    float3 probe_pos = (float3(probe_coord) + 0.5) * probe_spacing;

    // Probe offset: LOD 0 offsets by -face_normal*0.25, LOD>0 uses GeoOffset
    float3 face_normal, face_tangent, face_bitangent;
    get_face_tbn(face, face_normal, face_tangent, face_bitangent);

    if (push.cascade_level > 0u) {
        probe_pos += rc_geo_offset(probe_pos, hierarchy_l0, brick_occupancy, 16u);
    } else {
        probe_pos -= face_normal * 0.25;
    }

    // Check if probe is inside solid geometry
    if (!rc_outside_geo(probe_pos, hierarchy_l0, brick_occupancy, 16u)) {
        uint idx = rc_probe_index(probe_coord, grid_dim, dirs_per_face, face, dir_index, push.buffer_offset);
        probe_write[idx] = float4(0.0, 0.0, 0.0, -1.0);
        return;
    }

    // Compute ray direction in hemisphere
    float2 dir_uv = float2(float(dir_index % probe_size) + 0.5, float(dir_index / probe_size) + 0.5);
    float3 local_dir = ComputeDir(dir_uv, float(probe_size));
    float3 world_dir = face_tangent * local_dir.x + face_bitangent * local_dir.y + face_normal * local_dir.z;

    // Trace ray
    Ray ray = make_ray(probe_pos, world_dir);
    HitResult hit = trace_primary_ray(ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);

    float4 result;
    if (hit.hit) {
        result.w = hit.t;

        // Check emissive
        float3 emissive = voxel_emissive(hit.cell);
        if (emissive.x + emissive.y + emissive.z > 0.001) {
            result.xyz = emissive;
        } else {
            uint mat_id = voxel_material(hit.cell);
            float3 albedo = float3(0.9, 0.9, 0.9); // fallback; RC trace doesn't need full LUT
            if (mat_id == 1u) albedo = float3(0.95, 0.925, 0.9);
            else if (mat_id == 2u) albedo = float3(0.99, 0.4, 0.4);
            else if (mat_id == 3u) albedo = float3(0.4, 0.99, 0.4);
            else if (mat_id == 4u) albedo = float3(0.5, 0.6, 0.9);
            else if (mat_id == 5u) albedo = float3(0.75, 0.65, 0.5);

            // Direct sun
            float3 hit_normal = hit.normal;
            float ndotl = max(dot(hit_normal, scene.sun_direction), 0.0);
            float3 direct = float3(0.0);
            if (ndotl > 0.0) {
                float3 shadow_origin = hit.position + hit_normal * 0.51;
                Ray shadow_ray = make_ray(shadow_origin, scene.sun_direction);
                HitResult shadow_hit = trace_primary_ray(shadow_ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);
                float shadow = shadow_hit.hit ? 0.0 : 1.0;
                direct = scene.sun_intensity * ndotl * shadow;
            }

            // Indirect from previous frame's probes (read higher cascade or same)
            float3 indirect = float3(0.0);
            if (scene.rc_enabled > 0u) {
                // Read from probe_read (previous frame), using C0 data
                indirect = integrate_probe(hit.position, hit_normal, probe_read, scene.rc_c0_offset, scene.rc_c0_grid);
            } else {
                indirect = hemisphere_ambient(hit_normal, scene.sky_color, scene.ground_color);
            }

            result.xyz = (direct + indirect) * albedo;
        }
    } else {
        result.w = 1e7;
        result.xyz = sky_color_for_dir(world_dir, scene);
    }

    // Weight by solid_angle × cos(theta)
    float cos_theta = dot(world_dir, face_normal);
    if (push.cascade_level == 0u) {
        if (length(dir_uv - float2(1.5, 1.5)) < 0.75) {
            result.xyz *= (1.0 - cos(0.25 * RC_PI));
        } else {
            result.xyz *= cos(0.25 * RC_PI) / 8.0;
        }
    } else {
        float2 probeRel = dir_uv - float(probe_size) * 0.5;
        float probeThetai = max(abs(probeRel.x), abs(probeRel.y));
        float probeTheta = acos(cos_theta);
        result.xyz *= (cos(probeTheta - 0.5 * RC_PI / float(probe_size)) -
                       cos(probeTheta + 0.5 * RC_PI / float(probe_size))) /
                      (4.0 + 8.0 * floor(probeThetai));
    }
    result.xyz *= cos_theta;

    uint idx = rc_probe_index(probe_coord, grid_dim, dirs_per_face, face, dir_index, push.buffer_offset);
    probe_write[idx] = result;
}
```

- [ ] **Step 2: Build to verify Slang compilation**

Run: `cargo build 2>&1 | head -40`
Expected: `slangc` compiles `rc_trace.slang` to `rc_trace.spv` without errors.

- [ ] **Step 3: Commit**

```bash
git add assets/shaders/passes/rc_trace.slang
git commit -m "feat(shader): rc_trace compute shader for probe ray tracing"
```

---

### Task 10: rc_merge.slang — cascade merging compute shader

**Files:**
- Create: `assets/shaders/passes/rc_merge.slang`

- [ ] **Step 1: Write rc_merge.slang**

Create `assets/shaders/passes/rc_merge.slang`:

```slang
// Radiance Cascade merge pass.
// Merges higher cascade data into lower cascade using visibility-weighted trilinear interpolation.
// Distance-based blend: near hits use own data, far hits blend with higher cascade.

#include "radiance_cascade.slang"

struct RcMergePush {
    uint cascade_level;      // level being merged INTO (0 or 1)
    uint own_grid_dim;       // 16 or 8
    uint own_probe_size;     // 3 or 6
    uint own_offset;         // buffer offset for this cascade
    uint higher_grid_dim;    // 8 or 4
    uint higher_probe_size;  // 6 or 12
    uint higher_offset;      // buffer offset for higher cascade
    uint _pad;
};
[[vk::push_constant]] RcMergePush push;

// Single probe buffer (read-write). Merge operates in-place.
[[vk::binding(0, 0)]]
RWStructuredBuffer<float4> probe_buffer;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint own_dirs = push.own_probe_size * push.own_probe_size;
    uint total = push.own_grid_dim * push.own_grid_dim * push.own_grid_dim * 6u * own_dirs;
    uint tid = dtid.x;
    if (tid >= total) return;

    // Decode
    uint dir_index = tid % own_dirs;
    uint remainder = tid / own_dirs;
    uint face = remainder % 6u;
    uint probe_linear = remainder / 6u;
    uint grid_dim = push.own_grid_dim;
    uint3 probe_coord = uint3(
        probe_linear % grid_dim,
        (probe_linear / grid_dim) % grid_dim,
        probe_linear / (grid_dim * grid_dim)
    );

    // Read own data
    uint own_idx = rc_probe_index(probe_coord, grid_dim, own_dirs, face, dir_index, push.own_offset);
    float4 own_data = probe_buffer[own_idx];

    // If this probe is inside geometry, skip merge
    if (own_data.w < -0.5) return;

    // Higher cascade lookup
    float lod_factor = float(1u << push.cascade_level); // 1 or 2
    float higher_lod_factor = lod_factor * 2.0;
    float probe_spacing = 8.0 * lod_factor;
    float3 voxel_pos = (float3(probe_coord) + 0.5) * probe_spacing;

    // Position in higher cascade grid
    uint h_grid = push.higher_grid_dim;
    uint h_dirs = push.higher_probe_size * push.higher_probe_size;
    float3 h_pos = clamp(voxel_pos / (8.0 * higher_lod_factor), float3(0.5), float3(float(h_grid)) - 0.5);

    // Map own direction to higher cascade direction index
    float2 own_dir_uv = float2(float(dir_index % push.own_probe_size), float(dir_index / push.own_probe_size));
    float h_dir_index0 = floor(own_dir_uv.x) * 2.0 + floor(own_dir_uv.y) * float(push.higher_probe_size) * 2.0 + 0.5;

    // Trilinear interpolation with visibility weighting
    float3 fPos = clamp(floor(h_pos - 0.5), float3(0.0), float3(float(h_grid) - 2.0));
    float3 frPos = min(float3(1.0), h_pos - 0.5 - fPos);

    float h_pow4 = pow(4.0, float(push.cascade_level + 1u));
    float h_probe_size_f = float(push.higher_probe_size);
    float3 h_size = float3(float(h_grid));

    // Precompute direction UV offsets for 2×2 bilinear in direction space
    float4 lDIUV0 = float4(
        floor(fmod(h_dir_index0, h_pow4)) * h_size.x * h_size.y,
        floor(h_dir_index0 / h_pow4) * h_size.z,
        floor(fmod(h_dir_index0 + 1.0, h_pow4)) * h_size.x * h_size.y,
        floor((h_dir_index0 + 1.0) / h_pow4) * h_size.z
    );
    float4 lDIUV1 = float4(
        floor(fmod(h_dir_index0 + h_probe_size_f, h_pow4)) * h_size.x * h_size.y,
        floor((h_dir_index0 + h_probe_size_f) / h_pow4) * h_size.z,
        floor(fmod(h_dir_index0 + h_probe_size_f + 1.0, h_pow4)) * h_size.x * h_size.y,
        floor((h_dir_index0 + h_probe_size_f + 1.0) / h_pow4) * h_size.z
    );

    // Sample 8 trilinear neighbors with visibility
    float3 merged_radiance = float3(0.0);
    float merged_weight = 0.0;

    for (uint i = 0; i < 8; i++) {
        int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        uint3 neighbor = uint3(int3(fPos) + offset);
        neighbor = clamp(neighbor, uint3(0), uint3(h_grid - 1));

        // Visibility test: check if this higher probe can see our position
        uint vis_dir_idx = uint(h_dir_index0);
        uint vis_idx = rc_probe_index(neighbor, h_grid, h_dirs, face, min(vis_dir_idx, h_dirs - 1u), push.higher_offset);
        float vis_ray_dist = probe_buffer[vis_idx].w;
        float3 neighbor_world = (float3(neighbor) + 0.5) * 8.0 * higher_lod_factor;
        float dist_to_us = length(voxel_pos - neighbor_world);
        float vis_w = (vis_ray_dist < dist_to_us - 0.5) ? 0.00001 : 1.0;

        // Read 4 direction samples (2×2 bilinear)
        float4 sum = float4(0.0);
        uint base = rc_probe_index(neighbor, h_grid, h_dirs, face, 0u, push.higher_offset);
        uint d0 = min(uint(h_dir_index0), h_dirs - 1u);
        uint d1 = min(uint(h_dir_index0 + 1.0), h_dirs - 1u);
        uint d2 = min(uint(h_dir_index0 + h_probe_size_f), h_dirs - 1u);
        uint d3 = min(uint(h_dir_index0 + h_probe_size_f + 1.0), h_dirs - 1u);
        sum += probe_buffer[base + d0];
        sum += probe_buffer[base + d1];
        sum += probe_buffer[base + d2];
        sum += probe_buffer[base + d3];

        float3 w = lerp(1.0 - frPos, frPos, float3(offset));
        float tri_w = w.x * w.y * w.z;
        float weight = tri_w * vis_w;

        if (sum.w < -0.5) {
            merged_radiance += sum.xyz * 0.001 * weight;
        } else {
            merged_radiance += sum.xyz * weight;
        }
        merged_weight += weight;
    }

    if (merged_weight > 0.0001) {
        merged_radiance /= merged_weight;
    }

    // Distance-based blend
    float inv_lod_factor = 1.0 / lod_factor;
    float dist_interp = clamp((own_data.w - lod_factor) * inv_lod_factor * 0.5, 0.0, 1.0);
    float3 final_radiance = lerp(own_data.xyz, merged_radiance, dist_interp);

    probe_buffer[own_idx] = float4(final_radiance, own_data.w);
}
```

- [ ] **Step 2: Build and verify**

Run: `cargo build 2>&1 | head -40`
Expected: `rc_merge.spv` compiles.

- [ ] **Step 3: Commit**

```bash
git add assets/shaders/passes/rc_merge.slang
git commit -m "feat(shader): rc_merge compute shader for cascade merging"
```

---

### Task 11: RcTracePass — Rust-side pass

**Files:**
- Modify: `src/render/passes/radiance_cascade_trace.rs` (replace placeholder)

- [ ] **Step 1: Implement RcTracePass**

Replace the placeholder in `src/render/passes/radiance_cascade_trace.rs`:

```rust
use anyhow::Result;
use ash::vk;
use std::ffi::CStr;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::pipeline::{create_shader_module, ComputePipeline};
use crate::render::rc_probe_buffer::{self, RcProbeBuffer};
use crate::render::scene_ubo::SceneUniformBuffer;
use crate::voxel::gpu_upload::UcvhGpuResources;

/// Push constant layout matching RcTracePush in rc_trace.slang.
#[repr(C)]
#[derive(Clone, Copy)]
struct RcTracePushConstants {
    cascade_level: u32,
    probe_grid_dim: u32,
    probe_size: u32,
    buffer_offset: u32,
}

/// Cascade dispatch parameters.
struct CascadeParams {
    level: u32,
    grid_dim: u32,
    probe_size: u32,
    offset: u32,
    total_invocations: u32,
}

const CASCADE_PARAMS: [CascadeParams; 3] = [
    CascadeParams { level: 0, grid_dim: 16, probe_size: 3,  offset: rc_probe_buffer::RC_C0_OFFSET, total_invocations: 16*16*16*6*9 },
    CascadeParams { level: 1, grid_dim: 8,  probe_size: 6,  offset: rc_probe_buffer::RC_C1_OFFSET, total_invocations: 8*8*8*6*36 },
    CascadeParams { level: 2, grid_dim: 4,  probe_size: 12, offset: rc_probe_buffer::RC_C2_OFFSET, total_invocations: 4*4*4*6*144 },
];

pub struct RcTracePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RcTracePass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        spirv_bytes: &[u8],
        ucvh_gpu: &UcvhGpuResources,
        scene_ubo: &SceneUniformBuffer,
        rc_probes: &RcProbeBuffer,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(1, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1) // ucvh config
            .add_binding(2, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1) // hierarchy_l0
            .add_binding(3, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1) // occupancy
            .add_binding(4, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1) // materials
            .add_binding(5, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1) // probe read
            .add_binding(6, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1) // probe write
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: frame_count as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 6 * frame_count as u32 },
        ];
        let descriptor_pool = DescriptorPool::new(device, frame_count as u32, &pool_sizes)?;
        let layouts: Vec<_> = (0..frame_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = descriptor_pool.allocate(device, &layouts)?;

        let ucvh_buffers = [
            &ucvh_gpu.config_buffer,
            &ucvh_gpu.hierarchy_l0_buffer,
            &ucvh_gpu.occupancy_buffer,
            &ucvh_gpu.material_buffer,
        ];

        for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(scene_ubo.buffer_handle(set_idx))
                .offset(0)
                .range(176);

            let ubo_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info));

            let ucvh_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers.iter().map(|buf| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buf.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            }).collect();

            let ucvh_writes: Vec<vk::WriteDescriptorSet> = ucvh_infos.iter().enumerate().map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding((i + 1) as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            }).collect();

            let read_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.read_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let read_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&read_info));

            let write_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&write_info));

            let mut all_writes = vec![ubo_write];
            all_writes.extend(ucvh_writes);
            all_writes.push(read_write);
            all_writes.push(write_write);
            unsafe { device.update_descriptor_sets(&all_writes, &[]) };
        }

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<RcTracePushConstants>() as u32);

        let shader_module = create_shader_module(device, spirv_bytes)?;
        let pipeline = ComputePipeline::new(
            device,
            shader_module,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") },
            &[descriptor_set_layout],
            &[push_range],
        )?;
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self { pipeline, descriptor_set_layout, descriptor_pool, descriptor_sets })
    }

    /// Update descriptor sets when probe buffer swaps (call each frame before record).
    pub fn update_probe_descriptors(
        &self,
        device: &ash::Device,
        rc_probes: &RcProbeBuffer,
    ) {
        for &ds in &self.descriptor_sets {
            let read_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.read_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let read_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&read_info));

            let write_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&write_info));

            unsafe { device.update_descriptor_sets(&[read_write, write_write], &[]) };
        }
    }

    /// Record all 3 cascade trace dispatches. No barriers needed between them.
    pub fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
    ) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.descriptor_sets[frame_slot]],
                &[],
            );
        }

        // Dispatch all 3 cascades (C2, C1, C0 — order doesn't matter with temporal reads)
        for params in &CASCADE_PARAMS {
            let push = RcTracePushConstants {
                cascade_level: params.level,
                probe_grid_dim: params.grid_dim,
                probe_size: params.probe_size,
                buffer_offset: params.offset,
            };
            let push_bytes = unsafe {
                std::slice::from_raw_parts(
                    &push as *const RcTracePushConstants as *const u8,
                    std::mem::size_of::<RcTracePushConstants>(),
                )
            };
            unsafe {
                device.cmd_push_constants(
                    cmd,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_bytes,
                );
                device.cmd_dispatch(cmd, (params.total_invocations + 63) / 64, 1, 1);
            }
        }
    }

    pub fn destroy(self, device: &ash::Device, _allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}
```

- [ ] **Step 2: Build**

Run: `cargo build 2>&1 | head -40`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add src/render/passes/radiance_cascade_trace.rs
git commit -m "feat(render): RcTracePass with 3-cascade dispatch"
```

---

### Task 12: RcMergePass — Rust-side pass

**Files:**
- Modify: `src/render/passes/radiance_cascade_merge.rs` (replace placeholder)

- [ ] **Step 1: Implement RcMergePass**

Replace the placeholder in `src/render/passes/radiance_cascade_merge.rs`:

```rust
use anyhow::Result;
use ash::vk;
use std::ffi::CStr;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::pipeline::{create_shader_module, ComputePipeline};
use crate::render::rc_probe_buffer::{self, RcProbeBuffer};

#[repr(C)]
#[derive(Clone, Copy)]
struct RcMergePushConstants {
    cascade_level: u32,
    own_grid_dim: u32,
    own_probe_size: u32,
    own_offset: u32,
    higher_grid_dim: u32,
    higher_probe_size: u32,
    higher_offset: u32,
    _pad: u32,
}

struct MergeParams {
    cascade_level: u32,
    own_grid_dim: u32,
    own_probe_size: u32,
    own_offset: u32,
    higher_grid_dim: u32,
    higher_probe_size: u32,
    higher_offset: u32,
    total_invocations: u32,
}

/// Merge C2→C1 first, then C1→C0.
const MERGE_PARAMS: [MergeParams; 2] = [
    MergeParams {
        cascade_level: 1, own_grid_dim: 8, own_probe_size: 6, own_offset: rc_probe_buffer::RC_C1_OFFSET,
        higher_grid_dim: 4, higher_probe_size: 12, higher_offset: rc_probe_buffer::RC_C2_OFFSET,
        total_invocations: 8*8*8*6*36,
    },
    MergeParams {
        cascade_level: 0, own_grid_dim: 16, own_probe_size: 3, own_offset: rc_probe_buffer::RC_C0_OFFSET,
        higher_grid_dim: 8, higher_probe_size: 6, higher_offset: rc_probe_buffer::RC_C1_OFFSET,
        total_invocations: 16*16*16*6*9,
    },
];

pub struct RcMergePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RcMergePass {
    pub fn new(
        device: &ash::Device,
        spirv_bytes: &[u8],
        rc_probes: &RcProbeBuffer,
        frame_count: usize,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .build(device)?;

        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: frame_count as u32 },
        ];
        let descriptor_pool = DescriptorPool::new(device, frame_count as u32, &pool_sizes)?;
        let layouts: Vec<_> = (0..frame_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = descriptor_pool.allocate(device, &layouts)?;

        // All frame-slot sets point to the write buffer (updated each frame)
        for &ds in &descriptor_sets {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<RcMergePushConstants>() as u32);

        let shader_module = create_shader_module(device, spirv_bytes)?;
        let pipeline = ComputePipeline::new(
            device,
            shader_module,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") },
            &[descriptor_set_layout],
            &[push_range],
        )?;
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self { pipeline, descriptor_set_layout, descriptor_pool, descriptor_sets })
    }

    /// Update descriptor to point to current frame's write buffer.
    pub fn update_probe_descriptor(&self, device: &ash::Device, rc_probes: &RcProbeBuffer) {
        for &ds in &self.descriptor_sets {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }
    }

    /// Record merge dispatches: C2→C1 barrier C1→C0. Inserts barrier between.
    pub fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        probe_buffer_handle: vk::Buffer,
    ) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.descriptor_sets[frame_slot]],
                &[],
            );
        }

        for (i, params) in MERGE_PARAMS.iter().enumerate() {
            if i > 0 {
                // Barrier between merge dispatches
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .buffer(probe_buffer_handle)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[], &[barrier], &[],
                    );
                }
            }

            let push = RcMergePushConstants {
                cascade_level: params.cascade_level,
                own_grid_dim: params.own_grid_dim,
                own_probe_size: params.own_probe_size,
                own_offset: params.own_offset,
                higher_grid_dim: params.higher_grid_dim,
                higher_probe_size: params.higher_probe_size,
                higher_offset: params.higher_offset,
                _pad: 0,
            };
            let push_bytes = unsafe {
                std::slice::from_raw_parts(
                    &push as *const RcMergePushConstants as *const u8,
                    std::mem::size_of::<RcMergePushConstants>(),
                )
            };
            unsafe {
                device.cmd_push_constants(cmd, self.pipeline.layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
                device.cmd_dispatch(cmd, (params.total_invocations + 63) / 64, 1, 1);
            }
        }
    }

    pub fn destroy(self, device: &ash::Device, _allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}
```

- [ ] **Step 2: Build**

Run: `cargo build 2>&1 | head -40`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add src/render/passes/radiance_cascade_merge.rs
git commit -m "feat(render): RcMergePass with C2→C1→C0 merge dispatch"
```

---

## Sub-Phase 5C: Lighting Integration + App Wiring

### Task 13: Integrate RC into lighting.slang

**Files:**
- Modify: `assets/shaders/passes/lighting.slang`

- [ ] **Step 1: Add RC probe buffer binding and include**

In `assets/shaders/passes/lighting.slang`, add after the existing includes (line 5):

```slang
#include "radiance_cascade.slang"
```

Add binding 9 after binding 8 (line 38):

```slang
// Binding 9: RC probe buffer (read-only for indirect lighting)
[[vk::binding(9, 0)]]
StructuredBuffer<float4> rc_probes;
```

- [ ] **Step 2: Add emissive early-out**

After `float3 normal = NORMAL_TABLE[min(normal_id, 5u)];` (line 72), add:

```slang
    // Emissive early-out
    uint3 emissive_raw = uint3(gb1.g, gb1.b, gb1.a);
    if (emissive_raw.x + emissive_raw.y + emissive_raw.z > 0u) {
        float3 emissive = float3(emissive_raw) / 255.0;
        output_image[tid.xy] = float4(pow(emissive, float3(1.0 / 2.2)), 1.0);
        return;
    }
```

- [ ] **Step 3: Replace hemisphere_ambient with RC indirect**

Replace line 85:

```slang
    float3 ambient = base_color * hemisphere_ambient(normal, scene.sky_color, scene.ground_color);
```

With:

```slang
    float3 ambient;
    if (scene.rc_enabled > 0u) {
        float3 indirect = integrate_probe(position, normal, rc_probes, scene.rc_c0_offset, scene.rc_c0_grid);
        ambient = base_color * indirect;
    } else {
        ambient = base_color * hemisphere_ambient(normal, scene.sky_color, scene.ground_color);
    }
```

- [ ] **Step 4: Build and verify**

Run: `cargo build 2>&1 | head -40`
Expected: Compiles. With `rc_enabled=0`, behavior is identical to before.

- [ ] **Step 5: Commit**

```bash
git add assets/shaders/passes/lighting.slang
git commit -m "feat(shader): integrate RC probes into lighting pass"
```

---

### Task 14: Add binding 9 to lighting.rs

**Files:**
- Modify: `src/render/passes/lighting.rs`

- [ ] **Step 1: Add RC buffer descriptor binding**

In `src/render/passes/lighting.rs`, modify `new()`:

1. Add `rc_probes: &RcProbeBuffer` parameter.

2. Add binding 9 in the descriptor layout (after `.add_binding(8, ...)`):

```rust
            .add_binding(9, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
```

3. Update pool sizes to include one more SSBO:

```rust
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 5 * frame_count as u32 },
```

4. In the descriptor write loop, after the UCVH buffer writes, add the RC probe buffer write:

```rust
            // Binding 9: RC probe buffer (read current frame's merged data)
            let rc_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let rc_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&rc_info));
            all_writes.push(rc_write);
```

5. Add a method to update the RC descriptor each frame (since buffer swaps):

```rust
    pub fn update_rc_descriptor(&self, device: &ash::Device, rc_probes: &RcProbeBuffer) {
        for &ds in &self.descriptor_sets {
            let rc_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let rc_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&rc_info));
            unsafe { device.update_descriptor_sets(&[rc_write], &[]) };
        }
    }
```

6. Update `new()` signature import and callers.

- [ ] **Step 2: Build**

Run: `cargo build 2>&1 | head -40`
Expected: Build fails at `app.rs` because `LightingPass::new()` now requires `rc_probes`. That's expected — fixed in Task 15.

- [ ] **Step 3: Commit**

```bash
git add src/render/passes/lighting.rs
git commit -m "feat(render): add RC probe buffer binding to LightingPass"
```

---

### Task 15: Wire everything into app.rs

**Files:**
- Modify: `src/app.rs`

This is the largest integration task. It wires together all the new passes.

- [ ] **Step 1: Add new fields to RevolumetricApp**

Add imports at the top:

```rust
use crate::render::rc_probe_buffer::RcProbeBuffer;
use crate::render::passes::radiance_cascade_trace::RcTracePass;
use crate::render::passes::radiance_cascade_merge::RcMergePass;
```

Add fields to `RevolumetricApp` struct (after `lighting_pass`):

```rust
    rc_probes: Option<RcProbeBuffer>,
    rc_trace_pass: Option<RcTracePass>,
    rc_merge_pass: Option<RcMergePass>,
    rc_cleared: bool,
```

Initialize in `new()`:

```rust
            rc_probes: None,
            rc_trace_pass: None,
            rc_merge_pass: None,
            rc_cleared: false,
```

- [ ] **Step 2: Create RC resources in resumed()**

After the lighting pass creation block (line ~520), add:

```rust
        // Create RC probe buffer
        if self.rc_probes.is_none() {
            let renderer = self.renderer.as_ref().unwrap();
            match RcProbeBuffer::new(renderer.device(), renderer.allocator()) {
                Ok(buf) => {
                    tracing::info!("created RC probe buffer (double-buffered, ~12 MB)");
                    self.rc_probes = Some(buf);
                }
                Err(e) => tracing::error!(%e, "failed to create RC probe buffer"),
            }
        }

        // Create RC trace pass
        if self.rc_trace_pass.is_none() {
            if let (Some(ucvh_gpu), Some(scene_ubo_ref), Some(rc_probes)) =
                (&self.ucvh_gpu, &self.scene_ubo, &self.rc_probes)
            {
                let renderer = self.renderer.as_ref().unwrap();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rc_trace.spv"));
                if !spirv.is_empty() {
                    match RcTracePass::new(
                        renderer.device(), renderer.allocator(), spirv,
                        ucvh_gpu, scene_ubo_ref, rc_probes,
                    ) {
                        Ok(pass) => {
                            tracing::info!("initialized RC trace pass");
                            self.rc_trace_pass = Some(pass);
                        }
                        Err(e) => tracing::error!(%e, "failed to create RC trace pass"),
                    }
                }
            }
        }

        // Create RC merge pass
        if self.rc_merge_pass.is_none() {
            if let (Some(rc_probes), Some(scene_ubo_ref)) = (&self.rc_probes, &self.scene_ubo) {
                let renderer = self.renderer.as_ref().unwrap();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rc_merge.spv"));
                if !spirv.is_empty() {
                    match RcMergePass::new(
                        renderer.device(), spirv, rc_probes, scene_ubo_ref.frame_count(),
                    ) {
                        Ok(pass) => {
                            tracing::info!("initialized RC merge pass");
                            self.rc_merge_pass = Some(pass);
                        }
                        Err(e) => tracing::error!(%e, "failed to create RC merge pass"),
                    }
                }
            }
        }
```

Also update `LightingPass::new()` call to pass `rc_probes`:

```rust
                    match LightingPass::new(
                        renderer.device(),
                        renderer.allocator(),
                        extent.width,
                        extent.height,
                        spirv,
                        primary,
                        ucvh_gpu,
                        scene_ubo_ref,
                        rc_probes,  // NEW parameter
                    ) {
```

This requires `rc_probes` to be created before the lighting pass. Reorder the initialization blocks accordingly: rc_probes → lighting_pass.

- [ ] **Step 3: Update scene_data to enable RC**

In `tick_frame()`, update the scene_data construction to enable RC when passes are ready:

```rust
                        rc_c0_grid: [16, 16, 16],
                        rc_c0_offset: 0,
                        rc_enabled: if self.rc_trace_pass.is_some() { 1 } else { 0 },
                        _pad4: [0; 3],
```

- [ ] **Step 4: Add RC passes to render graph**

In `tick_frame()`, after the `primary_ray` render graph pass and before the `lighting` pass, add:

```rust
                    // RC trace + merge passes
                    if let (Some(rc_trace), Some(rc_merge), Some(rc_probes)) =
                        (&self.rc_trace_pass, &self.rc_merge_pass, &self.rc_probes)
                    {
                        // Zero-init on first frame
                        if !self.rc_cleared {
                            rc_probes.record_clear(renderer.device(), frame.command_buffer);
                            // Barrier: TRANSFER → COMPUTE
                            let barrier = vk::BufferMemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                                .buffer(rc_probes.write_buffer())
                                .size(vk::WHOLE_SIZE);
                            let barrier2 = vk::BufferMemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                                .buffer(rc_probes.read_buffer())
                                .size(vk::WHOLE_SIZE);
                            unsafe {
                                renderer.device().cmd_pipeline_barrier(
                                    frame.command_buffer,
                                    vk::PipelineStageFlags::TRANSFER,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::DependencyFlags::empty(),
                                    &[], &[barrier, barrier2], &[],
                                );
                            }
                            self.rc_cleared = true;
                        }

                        // Update descriptors for swapped buffers
                        rc_trace.update_probe_descriptors(renderer.device(), rc_probes);
                        rc_merge.update_probe_descriptor(renderer.device(), rc_probes);
                        if let Some(lighting) = &self.lighting_pass {
                            lighting.update_rc_descriptor(renderer.device(), rc_probes);
                        }

                        // RC trace (all 3 cascades, no inter-dispatch barriers)
                        rc_trace.record(renderer.device(), frame.command_buffer, frame.frame_slot);

                        // Barrier: rc_trace writes → rc_merge reads
                        let barrier = vk::BufferMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                            .buffer(rc_probes.write_buffer())
                            .size(vk::WHOLE_SIZE);
                        unsafe {
                            renderer.device().cmd_pipeline_barrier(
                                frame.command_buffer,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::DependencyFlags::empty(),
                                &[], &[barrier], &[],
                            );
                        }

                        // RC merge (C2→C1, barrier, C1→C0)
                        rc_merge.record(
                            renderer.device(), frame.command_buffer,
                            frame.frame_slot, rc_probes.write_buffer(),
                        );

                        // Barrier: rc_merge writes → lighting reads
                        let barrier = vk::BufferMemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                            .dst_access_mask(vk::AccessFlags::SHADER_READ)
                            .buffer(rc_probes.write_buffer())
                            .size(vk::WHOLE_SIZE);
                        unsafe {
                            renderer.device().cmd_pipeline_barrier(
                                frame.command_buffer,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::PipelineStageFlags::COMPUTE_SHADER,
                                vk::DependencyFlags::empty(),
                                &[], &[barrier], &[],
                            );
                        }
                    }
```

Note: The RC passes record directly to the command buffer (not through the RenderGraph) because they need fine-grained buffer barriers. Place this code after the graph is compiled but before end_frame. Alternatively, integrate into the graph system — implementer's judgment.

- [ ] **Step 5: Swap probe buffer at end of frame**

After `renderer.end_frame(frame)?;`, add:

```rust
                if let Some(rc_probes) = &mut self.rc_probes {
                    rc_probes.swap();
                }
```

- [ ] **Step 6: Update Drop impl**

In the `Drop` impl, add cleanup before existing passes (order matters — destroy RC passes first):

```rust
            if let Some(pass) = self.rc_merge_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(pass) = self.rc_trace_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(buf) = self.rc_probes.take() {
                buf.destroy(renderer.device(), renderer.allocator());
            }
```

- [ ] **Step 7: Build and run**

Run: `cargo build && cargo run`
Expected: The Sponza scene renders with Radiance Cascades indirect lighting. Color bleeding should be visible (red/green cloth tinting nearby stone surfaces). Emissive lamps should cast warm indirect glow. The GI fades in over the first 3-4 frames as cascades converge.

- [ ] **Step 8: Commit**

```bash
git add src/app.rs
git commit -m "feat(app): wire RC trace/merge/lighting into render loop"
```

---

### Task 16: Final verification and cleanup

**Files:**
- All files modified in Phase 5

- [ ] **Step 1: Run full test suite**

Run: `cargo test`
Expected: All tests pass (including new sponza_generator tests, rc_probe_buffer tests, scene_ubo size test).

- [ ] **Step 2: Run with Vulkan validation**

Run: `RUST_LOG=debug cargo run 2>&1 | grep -i "validation\|error\|warning" | head -20`
Expected: No Vulkan validation errors. Check for storage buffer format mismatches or descriptor issues.

- [ ] **Step 3: Visual verification**

Verify:
- Color bleeding: red/green cloth banners tint nearby stone
- Emissive GI: warm glow from lamps illuminates surroundings
- Shadow rays: direct sunlight with proper hard shadows
- Sky: miss rays show gradient sky
- No artifacts: no black spots, no flickering, no obvious light leaking

- [ ] **Step 4: Commit any fixes**

If any issues found, fix and commit with appropriate message.

- [ ] **Step 5: Final commit for Phase 5**

```bash
git log --oneline -15  # review Phase 5 commit history
```
