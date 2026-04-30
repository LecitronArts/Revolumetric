# Phase 3: Voxel Ray Tracing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement two-level hierarchical DDA ray tracing through the UCVH data structure, producing a flat-color rendered image of the voxel scene.

**Architecture:** A compute shader (`primary_ray.slang`) traces rays from camera through the UCVH brick grid. Level 1: Amanatides-Woo DDA steps through the brick grid (L0 hierarchy), skipping empty bricks. Level 2: texel-cached DDA through the 8³ occupancy bitmask within each occupied brick. On hit, the face normal determines a flat-shaded color. A Rust-side `PrimaryRayPass` struct (following the `TestPatternPass` pattern) manages descriptors, pipeline, and push constants. Camera `pixel_to_ray` matrix is computed CPU-side each frame.

**Tech Stack:** Rust, ash (Vulkan 1.3), Slang → SPIR-V, glam (math), bytemuck

**Depends on:** Phase 2 (UCVH Core) — all Phase 2 commits must be present.

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/render/camera.rs` | `compute_pixel_to_ray()` pure math + `PrimaryRayPushConstants` repr(C) struct |
| Create | `assets/shaders/shared/ray.slang` | Ray struct, AABB intersection, ray-box entry/exit helpers |
| Modify | `assets/shaders/shared/voxel_common.slang` | GPU struct definitions matching Rust: `UcvhConfig`, `BrickOccupancy`, `VoxelCell`, `NodeL0`, `NodeLN` + occupancy read helpers |
| Create | `assets/shaders/shared/voxel_traverse.slang` | Two-level DDA: brick-grid DDA + texel-cached brick DDA + `trace_primary_ray()` |
| Create | `assets/shaders/passes/primary_ray.slang` | Compute entry point: push constants, descriptor bindings, per-pixel ray gen + trace + output |
| Create | `src/render/passes/primary_ray.rs` | `PrimaryRayPass` struct: descriptor layout (1 image + 8 SSBOs), pipeline, `record()` |
| Modify | `src/render/passes/mod.rs:1-8` | Add `pub mod primary_ray;` |
| Modify | `src/render/mod.rs` | Add `pub mod camera;` |
| Modify | `src/app.rs` | Replace `test_pattern_pass` with `primary_ray_pass`, compute `pixel_to_ray` each frame |

---

## Task 1: Camera Math Module

**Files:**
- Create: `src/render/camera.rs`
- Modify: `src/render/mod.rs:1-14` — add `pub mod camera;`

This task creates a pure-math module for computing the `pixel_to_ray` matrix used by the primary ray shader. The matrix packs camera origin + ray direction computation into a single 4×4 matrix. Fully unit-testable with no GPU dependency.

**Key math:** For pixel (px, py), the shader computes:
- `origin = pixel_to_ray[3].xyz` (column 3 = camera position)
- `direction = normalize(mat3(pixel_to_ray) * float3(px, py, 1.0))`

The 3×3 submatrix encodes: pixel→NDC→view-space→world-space direction.

- [ ] **Step 1: Write failing tests for camera math**

Create `src/render/camera.rs` with the struct + function signature + tests:

```rust
// src/render/camera.rs
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

/// Push constants for the primary ray pass.
/// Sent to GPU each frame — must match the Slang `PrimaryRayPC` layout exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PrimaryRayPushConstants {
    /// Column-major 4×4 matrix.
    /// - Columns 0-2 (3×3): maps (pixel_x, pixel_y, 1) → unnormalized world-space ray direction
    /// - Column 3 (xyz):   camera world position (ray origin)
    pub pixel_to_ray: [[f32; 4]; 4],
    pub resolution: [u32; 2],
    pub _pad: [u32; 2],
}

/// Compute the pixel_to_ray matrix for a pinhole camera.
///
/// Convention: Y-up, camera looks along its `forward` direction.
/// The shader normalizes the direction, so the 3×3 part need not produce unit vectors.
pub fn compute_pixel_to_ray(
    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_up: Vec3,
    fov_y_rad: f32,
    width: u32,
    height: u32,
) -> Mat4 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_pixel_looks_along_forward() {
        let m = compute_pixel_to_ray(
            Vec3::ZERO, Vec3::Z, Vec3::Y,
            std::f32::consts::FRAC_PI_2, 800, 600,
        );
        let origin = Vec3::new(m.col(3).x, m.col(3).y, m.col(3).z);
        assert!((origin - Vec3::ZERO).length() < 1e-5);
        let mat3 = glam::Mat3::from_cols(
            m.col(0).truncate(), m.col(1).truncate(), m.col(2).truncate(),
        );
        let dir = (mat3 * Vec3::new(400.0, 300.0, 1.0)).normalize();
        assert!(dir.z > 0.5, "center ray should point along +Z, got {dir}");
    }

    #[test]
    fn origin_matches_camera_position() {
        let pos = Vec3::new(10.0, 20.0, 30.0);
        let m = compute_pixel_to_ray(pos, Vec3::Z, Vec3::Y, 1.0, 1920, 1080);
        let origin = Vec3::new(m.col(3).x, m.col(3).y, m.col(3).z);
        assert!((origin - pos).length() < 1e-5);
    }

    #[test]
    fn horizontal_ray_divergence() {
        let m = compute_pixel_to_ray(Vec3::ZERO, Vec3::Z, Vec3::Y, 1.0, 800, 600);
        let mat3 = glam::Mat3::from_cols(
            m.col(0).truncate(), m.col(1).truncate(), m.col(2).truncate(),
        );
        let left = (mat3 * Vec3::new(0.0, 300.0, 1.0)).normalize();
        let right = (mat3 * Vec3::new(799.0, 300.0, 1.0)).normalize();
        assert!(left.x < right.x, "left.x={} < right.x={}", left.x, right.x);
    }

    #[test]
    fn push_constants_size() {
        assert_eq!(std::mem::size_of::<PrimaryRayPushConstants>(), 80);
    }
}
```

- [ ] **Step 2: Add module declaration**

In `src/render/mod.rs`, add `pub mod camera;` (alphabetically after `pub mod buffer;`).

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test render::camera`
Expected: FAIL — `not yet implemented`

- [ ] **Step 4: Implement `compute_pixel_to_ray`**

Replace the `todo!()` in `src/render/camera.rs`:

```rust
pub fn compute_pixel_to_ray(
    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_up: Vec3,
    fov_y_rad: f32,
    width: u32,
    height: u32,
) -> Mat4 {
    let w = width as f32;
    let h = height as f32;
    let aspect = w / h;
    let t = (fov_y_rad * 0.5).tan();

    // Build orthonormal camera basis (right-handed)
    let forward = camera_forward.normalize();
    let right = forward.cross(camera_up).normalize();
    let up = right.cross(forward);

    // For pixel (px, py), the view-space direction is:
    //   vx = aspect*t * ((2*(px+0.5)/w) - 1)
    //   vy = t * (1 - (2*(py+0.5)/h))
    //   vz = 1.0
    // direction = right*vx + up*vy + forward*vz
    // This maps to: direction = mat3_cols * (px, py, 1)
    let sx = 2.0 * aspect * t / w;
    let sy = -2.0 * t / h;
    let ox = aspect * t * (1.0 / w - 1.0);
    let oy = t * (1.0 - 1.0 / h);

    let col0 = right * sx;
    let col1 = up * sy;
    let col2 = right * ox + up * oy + forward;
    let col3 = camera_pos;

    Mat4::from_cols(
        Vec4::new(col0.x, col0.y, col0.z, 0.0),
        Vec4::new(col1.x, col1.y, col1.z, 0.0),
        Vec4::new(col2.x, col2.y, col2.z, 0.0),
        Vec4::new(col3.x, col3.y, col3.z, 1.0),
    )
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test render::camera`
Expected: 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/render/camera.rs src/render/mod.rs
git commit -m "feat(phase3): add camera math module with pixel_to_ray matrix"
```

---

## Task 2: GPU Struct Definitions (voxel_common.slang)

**Files:**
- Modify: `assets/shaders/shared/voxel_common.slang` (currently a 1-line placeholder)

Populate the shared Slang module with GPU struct definitions that exactly mirror the Rust-side types. These are consumed by all voxel shader passes.

**Critical alignment rules** (must match Rust `#[repr(C)]` + `bytemuck::Pod`):
- `BrickOccupancy`: 80 bytes = `uint bits[16]` + `uint count` + `uint _pad[3]`
- `VoxelCell`: 8 bytes = `uint packed` (material:16 | flags:16) + `uint emissive_pad` (emissive:24 | pad:8)
- `NodeL0`: 8 bytes = `uint brick_id` + `uint flags_pad` (flags:16 | pad:16)
- `NodeLN`: 4 bytes = `uint packed` (child_mask:8 | flags:8 | pad:16)
- `UcvhGpuConfig`: 48 bytes

- [ ] **Step 1: Write the voxel_common.slang shader**

Replace the content of `assets/shaders/shared/voxel_common.slang`:

```slang
// GPU struct definitions for UCVH data — must match Rust repr(C) layouts exactly.
// Consumed by all voxel shader passes via #include "voxel_common.slang"

struct UcvhConfig {
    uint4 world_size;       // xyz + pad
    uint4 brick_grid_size;  // xyz + pad
    uint  brick_capacity;
    uint  allocated_bricks;
    uint  _pad0;
    uint  _pad1;
};

// 80 bytes per brick. bits[] indexed by Morton order.
struct BrickOccupancy {
    uint bits[16];  // 512 bits = 16 × 32
    uint count;     // number of solid voxels
    uint _pad[3];   // pad to 80 bytes (16-byte alignment)
};

// 8 bytes per voxel, stored in Morton order within each brick.
struct VoxelCell {
    uint material_flags; // low 16: material id, high 16: flags
    uint emissive_pad;   // low 24: emissive RGB (8 bits each), high 8: padding
};

// 8 bytes per L0 node.
struct NodeL0 {
    uint brick_id;    // index into BrickPool, or 0xFFFFFFFF if empty
    uint flags_pad;   // low 16: flags (bit 0 = has_solid), high 16: padding
};

// 4 bytes per L1-L4 node.
struct NodeLN {
    uint packed; // low 8: child_mask, next 8: flags (bit 0 = any_solid), high 16: padding
};

// --- Accessor helpers ---

uint voxel_material(VoxelCell cell) {
    return cell.material_flags & 0xFFFFu;
}

uint voxel_flags(VoxelCell cell) {
    return cell.material_flags >> 16u;
}

float3 voxel_emissive(VoxelCell cell) {
    float r = float(cell.emissive_pad & 0xFFu) / 255.0;
    float g = float((cell.emissive_pad >> 8u) & 0xFFu) / 255.0;
    float b = float((cell.emissive_pad >> 16u) & 0xFFu) / 255.0;
    return float3(r, g, b);
}

bool node_l0_is_empty(NodeL0 node) {
    return node.brick_id == 0xFFFFFFFFu;
}

uint node_ln_child_mask(NodeLN node) {
    return node.packed & 0xFFu;
}

// Morton encode for 8^3 brick (3-bit per axis, interleaved x-y-z)
uint morton_encode(uint3 p) {
    uint m = 0u;
    for (uint bit = 0; bit < 3; bit++) {
        m |= ((p.x >> bit) & 1u) << (3u * bit);
        m |= ((p.y >> bit) & 1u) << (3u * bit + 1u);
        m |= ((p.z >> bit) & 1u) << (3u * bit + 2u);
    }
    return m;
}

// Read a single voxel's occupancy bit from a BrickOccupancy.
// Uses Morton-order bit indexing.
bool read_occupancy_bit(BrickOccupancy brick, uint3 local) {
    uint m = morton_encode(local);
    uint word = m / 32u;
    uint bit = m % 32u;
    return (brick.bits[word] & (1u << bit)) != 0u;
}
```

- [ ] **Step 2: Verify shader compiles**

Run: `slangc assets/shaders/shared/voxel_common.slang -target spirv -o /dev/null -entry main -stage compute 2>&1 || echo "Expected: no main entry, but syntax should be valid"`

This file has no entry point, so slangc will warn — that's fine. The real compilation test happens when `primary_ray.slang` includes it. For now, verify there are no syntax errors by checking the output doesn't contain "error".

Alternatively, just proceed — the build.rs only compiles files in `passes/`, not `shared/`.

- [ ] **Step 3: Commit**

```bash
git add assets/shaders/shared/voxel_common.slang
git commit -m "feat(phase3): populate voxel_common.slang with GPU struct definitions"
```

---

## Task 3: Ray Math Shared Shader (ray.slang)

**Files:**
- Create: `assets/shaders/shared/ray.slang`
- Modify: `assets/shaders/shared/math.slang` (currently a 1-line placeholder)

Shared ray tracing utilities: Ray struct, AABB intersection, ray-box entry/exit distance. Used by the DDA traversal and any future ray-based pass.

- [ ] **Step 1: Write `math.slang` with basic AABB utilities**

Replace content of `assets/shaders/shared/math.slang`:

```slang
// Shared math utilities for voxel ray tracing.

// Axis-aligned bounding box.
struct AABB {
    float3 mn; // min corner
    float3 mx; // max corner
};

// Ray-AABB slab intersection. Returns (tmin, tmax).
// If tmin > tmax or tmax < 0, ray misses the box.
float2 intersect_aabb(float3 origin, float3 inv_dir, AABB box) {
    float3 t0 = (box.mn - origin) * inv_dir;
    float3 t1 = (box.mx - origin) * inv_dir;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float enter = max(max(tmin.x, tmin.y), tmin.z);
    float exit  = min(min(tmax.x, tmax.y), tmax.z);
    return float2(enter, exit);
}
```

- [ ] **Step 2: Write `ray.slang`**

Create `assets/shaders/shared/ray.slang`:

```slang
// Ray tracing shared utilities.
#include "math.slang"

struct Ray {
    float3 origin;
    float3 direction;
    float3 inv_dir; // 1.0 / direction, precomputed
};

Ray make_ray(float3 origin, float3 direction) {
    Ray r;
    r.origin = origin;
    r.direction = direction;
    // Avoid division by zero: use a large reciprocal for near-zero components
    r.inv_dir = float3(
        abs(direction.x) > 1e-8 ? 1.0 / direction.x : sign(direction.x) * 1e8,
        abs(direction.y) > 1e-8 ? 1.0 / direction.y : sign(direction.y) * 1e8,
        abs(direction.z) > 1e-8 ? 1.0 / direction.z : sign(direction.z) * 1e8
    );
    return r;
}

struct HitResult {
    bool  hit;
    float t;         // distance along ray
    float3 position; // world-space hit point
    float3 normal;   // face normal at hit (axis-aligned, from DDA step)
    uint  brick_id;  // which brick was hit
    uint3 local;     // local voxel coordinate within brick [0,7]³
};

static const HitResult NO_HIT = { false, 1e30, float3(0), float3(0), 0xFFFFFFFFu, uint3(0) };
```

- [ ] **Step 3: Commit**

```bash
git add assets/shaders/shared/math.slang assets/shaders/shared/ray.slang
git commit -m "feat(phase3): add ray.slang and math.slang shared shader utilities"
```

---

## Task 4: Voxel Traversal Library (voxel_traverse.slang)

**Files:**
- Create: `assets/shaders/shared/voxel_traverse.slang`

The core two-level DDA algorithm. This is the most complex shader code in Phase 3.

**Level 1 — Brick-grid DDA:** Standard Amanatides-Woo DDA stepping through the L0 hierarchy grid. Each cell maps to an 8³ brick. Skip cells where `node_l0_is_empty()`.

**Level 2 — Brick-internal DDA:** Amanatides-Woo DDA through the 512-bit occupancy bitmask. Reads occupancy via `read_occupancy_bit()` (Morton-order indexing, defined in `voxel_common.slang`). Branching DDA (faster than branchless per spec §5.4, from DeadlockCode/voxel_ray_traversal).

**Note:** The spec §5.3 describes a texel-cached 4×4×8 sub-block layout for amortized loads. Our `BrickOccupancy` currently stores bits in **Morton order** (matching the material storage), which is incompatible with that spatial caching scheme. For Phase 3 we use direct Morton bit reads. A future optimization pass can restructure the occupancy layout if profiling shows it's a bottleneck.

The traversal function takes buffer references as parameters — it does NOT declare descriptor bindings (those live in the pass shader).

- [ ] **Step 1: Write `voxel_traverse.slang`**

Create `assets/shaders/shared/voxel_traverse.slang`:

```slang
// Two-level hierarchical DDA for UCVH voxel traversal.
// Level 1: Brick-grid DDA (L0 hierarchy)
// Level 2: Brick-internal DDA (8^3 occupancy bitmask, Morton-order bit indexing)
//
// Key algorithms from references:
// - Amanatides-Woo DDA stepping (standard)
// - Branching DDA over branchless (from DeadlockCode/voxel_ray_traversal, ~5% faster)
// - occupancy_count skip (from GDVoxelPlayground — if count==0, skip entire brick)
// - pixelToRay single-matrix ray gen (from DeadlockCode/voxel_ray_traversal)

#include "ray.slang"
#include "voxel_common.slang"

// --- Brick-internal DDA (Level 2) ---
// Amanatides-Woo DDA through 8^3 brick occupancy.
// Uses read_occupancy_bit() from voxel_common.slang (Morton-order bit indexing).
// Returns true if a voxel is hit, populating hit_local and face normal.
bool brick_dda(
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    uint brick_id,
    float3 ray_origin,    // in brick-local space [0, 8]
    float3 ray_dir,
    out uint3 hit_local,
    out float3 hit_normal,
    out float hit_t
) {
    hit_local = uint3(0);
    hit_normal = float3(0);
    hit_t = 0.0;

    float3 inv_dir = float3(
        abs(ray_dir.x) > 1e-8 ? 1.0 / ray_dir.x : sign(ray_dir.x) * 1e8,
        abs(ray_dir.y) > 1e-8 ? 1.0 / ray_dir.y : sign(ray_dir.y) * 1e8,
        abs(ray_dir.z) > 1e-8 ? 1.0 / ray_dir.z : sign(ray_dir.z) * 1e8
    );

    // Entry into [0, 8] box
    float2 box_t = intersect_aabb(ray_origin, inv_dir, AABB(float3(0), float3(8)));
    float t_enter = max(box_t.x, 0.0);
    if (t_enter >= box_t.y) return false; // ray misses brick

    float3 pos = ray_origin + ray_dir * (t_enter + 0.001);
    int3 coord = int3(clamp(floor(pos), float3(0), float3(7)));
    int3 step_dir = int3(sign(ray_dir));

    // Determine entry face normal from which axis boundary the ray crossed
    float3 t_entry = (float3(coord) - ray_origin) * inv_dir;
    if (step_dir.x > 0) t_entry.x = (float(coord.x) - ray_origin.x) * inv_dir.x;
    if (step_dir.y > 0) t_entry.y = (float(coord.y) - ray_origin.y) * inv_dir.y;
    if (step_dir.z > 0) t_entry.z = (float(coord.z) - ray_origin.z) * inv_dir.z;
    // Entry normal: the axis with the largest t_entry was crossed last (= entry face)
    if (t_entry.x > t_entry.y && t_entry.x > t_entry.z)
        hit_normal = float3(-float(step_dir.x), 0, 0);
    else if (t_entry.y > t_entry.z)
        hit_normal = float3(0, -float(step_dir.y), 0);
    else
        hit_normal = float3(0, 0, -float(step_dir.z));
    float3 t_delta = abs(inv_dir);   // distance in t to cross one voxel
    // t_max: distance to next voxel boundary along each axis
    float3 t_max = float3(
        ((step_dir.x > 0 ? coord.x + 1 : coord.x) - pos.x) * inv_dir.x,
        ((step_dir.y > 0 ? coord.y + 1 : coord.y) - pos.y) * inv_dir.y,
        ((step_dir.z > 0 ? coord.z + 1 : coord.z) - pos.z) * inv_dir.z
    );

    // Load the full BrickOccupancy once (80 bytes, fits in registers)
    BrickOccupancy occ = occupancy_buf[brick_id];

    // Max steps: 8+8+8 = 24 (diagonal through 8^3)
    for (int i = 0; i < 24; i++) {
        if (any(coord < int3(0)) || any(coord >= int3(8))) break;

        // Morton-order bit read (matches Rust BrickOccupancy storage)
        if (read_occupancy_bit(occ, uint3(coord))) {
            hit_local = uint3(coord);
            hit_t = t_enter + min(min(t_max.x, t_max.y), t_max.z);
            return true;
        }

        // Branching DDA (faster than branchless per DeadlockCode benchmarks)
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                hit_normal = float3(-float(step_dir.x), 0, 0);
                coord.x += step_dir.x;
                t_max.x += t_delta.x;
            } else {
                hit_normal = float3(0, 0, -float(step_dir.z));
                coord.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                hit_normal = float3(0, -float(step_dir.y), 0);
                coord.y += step_dir.y;
                t_max.y += t_delta.y;
            } else {
                hit_normal = float3(0, 0, -float(step_dir.z));
                coord.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        }
    }
    return false;
}

// --- Primary trace through UCVH (Level 1 + Level 2) ---
// DDA through brick grid, descending into occupied bricks.
HitResult trace_primary_ray(
    Ray ray,
    StructuredBuffer<UcvhConfig> config_buf,
    StructuredBuffer<NodeL0> hierarchy_l0,
    StructuredBuffer<BrickOccupancy> occupancy_buf,
    StructuredBuffer<VoxelCell> material_buf
) {
    UcvhConfig config = config_buf[0];
    float3 grid_size = float3(config.brick_grid_size.xyz);
    float3 world_size = grid_size * 8.0; // each brick = 8 voxels

    // Intersect ray with world AABB [0, world_size]
    float2 world_t = intersect_aabb(ray.origin, ray.inv_dir, AABB(float3(0), world_size));
    if (world_t.x > world_t.y || world_t.y < 0.0)
        return NO_HIT;

    float t_enter = max(world_t.x, 0.0);
    float3 pos = ray.origin + ray.direction * (t_enter + 0.001);

    // Brick-grid DDA setup
    float3 inv_dir = ray.inv_dir;
    int3 brick_coord = int3(clamp(floor(pos / 8.0), float3(0), grid_size - 1.0));
    int3 step_dir = int3(sign(ray.direction));
    float3 t_delta = abs(inv_dir) * 8.0; // distance in t to cross one brick (8 voxels)
    float3 t_max = float3(
        ((step_dir.x > 0 ? (brick_coord.x + 1) * 8.0 : brick_coord.x * 8.0) - pos.x) * inv_dir.x,
        ((step_dir.y > 0 ? (brick_coord.y + 1) * 8.0 : brick_coord.y * 8.0) - pos.y) * inv_dir.y,
        ((step_dir.z > 0 ? (brick_coord.z + 1) * 8.0 : brick_coord.z * 8.0) - pos.z) * inv_dir.z
    );

    int3 igrid = int3(grid_size);
    float3 brick_normal = float3(0);

    // Max brick steps: grid diagonal
    int max_steps = igrid.x + igrid.y + igrid.z;
    for (int i = 0; i < max_steps && i < 256; i++) {
        if (any(brick_coord < int3(0)) || any(brick_coord >= igrid))
            break;

        // Lookup L0 node
        uint l0_idx = uint(brick_coord.x)
                    + uint(brick_coord.y) * uint(igrid.x)
                    + uint(brick_coord.z) * uint(igrid.x) * uint(igrid.y);
        NodeL0 node = hierarchy_l0[l0_idx];

        if (!node_l0_is_empty(node)) {
            // Quick check: is brick occupied at all?
            BrickOccupancy occ = occupancy_buf[node.brick_id];
            if (occ.count > 0u) {
                // Compute ray in brick-local space
                float3 brick_origin = float3(brick_coord) * 8.0;
                float3 local_origin = ray.origin - brick_origin;

                uint3 hit_local;
                float3 hit_normal;
                float hit_t;
                if (brick_dda(occupancy_buf, node.brick_id, local_origin, ray.direction,
                              hit_local, hit_normal, hit_t)) {
                    HitResult result;
                    result.hit = true;
                    result.t = hit_t;
                    result.position = brick_origin + float3(hit_local) + 0.5;
                    result.normal = hit_normal;
                    result.brick_id = node.brick_id;
                    result.local = hit_local;
                    return result;
                }
            }
        }

        // Step to next brick (branching DDA)
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                brick_coord.x += step_dir.x;
                t_max.x += t_delta.x;
            } else {
                brick_coord.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        } else {
            if (t_max.y < t_max.z) {
                brick_coord.y += step_dir.y;
                t_max.y += t_delta.y;
            } else {
                brick_coord.z += step_dir.z;
                t_max.z += t_delta.z;
            }
        }
    }

    return NO_HIT;
}
```

- [ ] **Step 2: Commit**

```bash
git add assets/shaders/shared/voxel_traverse.slang
git commit -m "feat(phase3): add two-level DDA voxel traversal shader library"
```

---

## Task 5: Primary Ray Compute Shader (primary_ray.slang)

**Files:**
- Create: `assets/shaders/passes/primary_ray.slang`

The compute entry point. Each thread traces one pixel. Push constants provide the `pixel_to_ray` matrix and resolution. Descriptor bindings provide the output image and all UCVH SSBOs.

**Output:** Normal-based flat shading on hit (maps face normal to RGB), sky gradient on miss.

- [ ] **Step 1: Write `primary_ray.slang`**

Create `assets/shaders/passes/primary_ray.slang`:

```slang
// Primary ray compute shader — screen-space voxel ray tracing.
// Traces one ray per pixel through the UCVH brick grid.
// Output: flat-shaded normal color on hit, sky gradient on miss.

#include "voxel_traverse.slang"

struct PrimaryRayPC {
    float4x4 pixel_to_ray; // col 0-2: direction matrix, col 3: camera origin
    uint2 resolution;
    uint2 _pad;
};

[[vk::push_constant]]
PrimaryRayPC pc;

// Binding 0: output image
[[vk::binding(0, 0)]]
RWTexture2D<float4> output_image;

// Binding 1: UCVH config (single element)
[[vk::binding(1, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

// Binding 2: brick occupancy pool
[[vk::binding(2, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

// Binding 3: brick material pool
[[vk::binding(3, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

// Binding 4: hierarchy L0
[[vk::binding(4, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

// Bindings 5-8: hierarchy L1-L4 (reserved for future hierarchy skip optimization)
// Not used in the basic two-level DDA, but bound for forward compatibility.
[[vk::binding(5, 0)]] StructuredBuffer<NodeLN> hierarchy_l1;
[[vk::binding(6, 0)]] StructuredBuffer<NodeLN> hierarchy_l2;
[[vk::binding(7, 0)]] StructuredBuffer<NodeLN> hierarchy_l3;
[[vk::binding(8, 0)]] StructuredBuffer<NodeLN> hierarchy_l4;

// Sky gradient (simple procedural sky)
float3 sky_color(float3 dir) {
    float t = dir.y * 0.5 + 0.5; // -1..1 → 0..1
    float3 horizon = float3(0.7, 0.8, 1.0);
    float3 zenith  = float3(0.2, 0.4, 0.9);
    return lerp(horizon, zenith, saturate(t));
}

// Normal to flat color: maps face normal to visible RGB
float3 normal_color(float3 n, float3 position) {
    // Base color from normal direction
    float3 base = abs(n) * 0.5 + 0.3;
    // Simple directional lighting (sun from upper-right)
    float3 sun_dir = normalize(float3(0.5, 0.8, 0.3));
    float ndotl = max(dot(n, sun_dir), 0.0);
    float ambient = 0.3;
    return base * (ambient + (1.0 - ambient) * ndotl);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.resolution.x || tid.y >= pc.resolution.y) return;

    // Generate ray from pixel_to_ray matrix.
    // Rust side stores column-major via glam Mat4::from_cols + to_cols_array_2d.
    // Slang float4x4 is ROW-major by default, so m[i] = row i.
    // When Rust writes column-major data, Slang reads it transposed:
    //   Slang m[i] = Rust column i.
    // So: origin = m[3].xyz (Slang row 3 = Rust col 3 = camera position) ✓
    //     dir = m * (px, py, 1, 0) applies the 3×3 correctly.
    float2 pixel = float2(tid.xy);
    float3 origin = pc.pixel_to_ray[3].xyz;
    float3x3 dir_mat = float3x3(
        pc.pixel_to_ray[0].xyz,
        pc.pixel_to_ray[1].xyz,
        pc.pixel_to_ray[2].xyz
    );
    float3 dir = normalize(mul(dir_mat, float3(pixel, 1.0)));

    Ray ray = make_ray(origin, dir);
    HitResult hit = trace_primary_ray(ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);

    float3 color;
    if (hit.hit) {
        color = normal_color(hit.normal, hit.position);
    } else {
        color = sky_color(dir);
    }

    output_image[tid.xy] = float4(color, 1.0);
}
```

- [ ] **Step 2: Verify shader compiles (if slangc available)**

Run: `slangc assets/shaders/passes/primary_ray.slang -target spirv -entry main -stage compute -I assets/shaders/shared -o /dev/null`
Expected: compiles without errors (warnings about unused bindings are OK).

If slangc is not installed, `cargo build` will emit a warning but still succeed (build.rs writes an empty placeholder .spv).

- [ ] **Step 3: Commit**

```bash
git add assets/shaders/passes/primary_ray.slang
git commit -m "feat(phase3): add primary_ray.slang compute shader with two-level DDA"
```

---

## Task 6: PrimaryRayPass Rust Struct

**Files:**
- Create: `src/render/passes/primary_ray.rs`
- Modify: `src/render/passes/mod.rs:1-8` — add `pub mod primary_ray;`

Follows the `TestPatternPass` pattern exactly (see `src/render/passes/test_pattern.rs`). The key difference: 9 descriptor bindings (1 storage image + 8 SSBOs) instead of 1, and the push constants carry a 4×4 matrix + resolution instead of time + dimensions.

**Descriptor layout (all in set 0):**

| Binding | Type | Buffer |
|---------|------|--------|
| 0 | STORAGE_IMAGE | output_image (RWTexture2D) |
| 1 | STORAGE_BUFFER | ucvh_config |
| 2 | STORAGE_BUFFER | brick_occupancy |
| 3 | STORAGE_BUFFER | brick_materials |
| 4 | STORAGE_BUFFER | hierarchy_l0 |
| 5 | STORAGE_BUFFER | hierarchy_l1 |
| 6 | STORAGE_BUFFER | hierarchy_l2 |
| 7 | STORAGE_BUFFER | hierarchy_l3 |
| 8 | STORAGE_BUFFER | hierarchy_l4 |

- [ ] **Step 1: Add module declaration**

In `src/render/passes/mod.rs`, add `pub mod primary_ray;` (alphabetically after `pub mod debug_views;`).

- [ ] **Step 2: Write `primary_ray.rs`**

Create `src/render/passes/primary_ray.rs`:

```rust
use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;

use crate::render::allocator::GpuAllocator;
use crate::render::camera::PrimaryRayPushConstants;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{create_shader_module, ComputePipeline};
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct PrimaryRayPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pub output_image: GpuImage,
}

impl PrimaryRayPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        spirv_bytes: &[u8],
        ucvh_gpu: &UcvhGpuResources,
    ) -> Result<Self> {
        // Descriptor set layout: 1 storage image + 8 storage buffers
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(1, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(2, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(3, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(4, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(5, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(6, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(7, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(8, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .build(device)?;

        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 8 },
        ];
        let descriptor_pool = DescriptorPool::new(device, 1, &pool_sizes)?;
        let descriptor_set = descriptor_pool.allocate(device, &[descriptor_set_layout])?[0];

        // Output image
        let output_image = GpuImage::new(
            device, allocator,
            &GpuImageDesc {
                width, height, depth: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "primary_ray_output",
            },
        )?;

        // Write descriptor set
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(output_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let image_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_info));

        // SSBO buffer infos: config, occupancy, materials, l0, l1, l2, l3, l4
        let buffer_handles = [
            &ucvh_gpu.config_buffer,
            &ucvh_gpu.occupancy_buffer,
            &ucvh_gpu.material_buffer,
            &ucvh_gpu.hierarchy_l0_buffer,
            &ucvh_gpu.hierarchy_ln_buffers[0],
            &ucvh_gpu.hierarchy_ln_buffers[1],
            &ucvh_gpu.hierarchy_ln_buffers[2],
            &ucvh_gpu.hierarchy_ln_buffers[3],
        ];

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffer_handles
            .iter()
            .map(|buf| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buf.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        let mut buffer_writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        for (i, info) in buffer_infos.iter().enumerate() {
            buffer_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding((i + 1) as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            );
        }

        let mut all_writes = vec![image_write];
        all_writes.extend(buffer_writes);
        unsafe { device.update_descriptor_sets(&all_writes, &[]) };

        // Pipeline
        let shader_module = create_shader_module(device, spirv_bytes)?;
        let push_constant_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<PrimaryRayPushConstants>() as u32,
        }];
        let pipeline = ComputePipeline::new(
            device,
            shader_module,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") },
            &[descriptor_set_layout],
            &push_constant_ranges,
        )?;
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            output_image,
        })
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, pc: &PrimaryRayPushConstants) {
        let extent = self.output_image.extent;

        // Transition output image to GENERAL for compute write
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .image(self.output_image.handle)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier],
            );
        }

        // Bind pipeline and descriptor set
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.descriptor_set],
                &[],
            );
        }

        // Push constants
        unsafe {
            let pc_bytes = std::slice::from_raw_parts(
                pc as *const _ as *const u8,
                std::mem::size_of::<PrimaryRayPushConstants>(),
            );
            device.cmd_push_constants(
                cmd, self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE, 0, pc_bytes,
            );
        }

        // Dispatch (8×8 workgroups)
        let groups_x = (extent.width + 7) / 8;
        let groups_y = (extent.height + 7) / 8;
        unsafe { device.cmd_dispatch(cmd, groups_x, groups_y, 1) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.output_image.destroy(device, allocator);
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check`
Expected: compiles (shader .spv may be empty if slangc not installed, but Rust code compiles).

- [ ] **Step 4: Commit**

```bash
git add src/render/passes/primary_ray.rs src/render/passes/mod.rs
git commit -m "feat(phase3): add PrimaryRayPass Rust struct with UCVH descriptor bindings"
```

---

## Task 7: App Integration — Replace Test Pattern with Primary Ray

**Files:**
- Modify: `src/app.rs`

Replace the `TestPatternPass` with `PrimaryRayPass` in the application loop. The `test_pattern_pass` field becomes `primary_ray_pass`. Camera is hardcoded for now (Phase 9 adds FPS camera).

**Camera setup:** Position the camera outside the 128³ voxel world, looking at the center. The demo scene is a sphere at the center, so we place the camera at `(64, 80, -40)` looking toward `(64, 64, 64)` — this should produce a visible sphere with normal-based shading.

- [ ] **Step 1: Update imports in `app.rs`**

In `src/app.rs`, replace:
```rust
use crate::render::passes::test_pattern::TestPatternPass;
```
with:
```rust
use crate::render::camera::{compute_pixel_to_ray, PrimaryRayPushConstants};
use crate::render::passes::primary_ray::PrimaryRayPass;
```

- [ ] **Step 2: Replace struct field**

In the `RevolumetricApp` struct, replace:
```rust
    test_pattern_pass: Option<TestPatternPass>,
```
with:
```rust
    primary_ray_pass: Option<PrimaryRayPass>,
```

- [ ] **Step 3: Update `new()` constructor**

In `RevolumetricApp::new()`, replace:
```rust
            test_pattern_pass: None,
```
with:
```rust
            primary_ray_pass: None,
```

- [ ] **Step 4: Update `tick_frame()` render loop**

Replace the entire `if let Some(pass) = &self.test_pattern_pass { ... }` block (approx lines 104-146) with:

```rust
                if let Some(pass) = &self.primary_ray_pass {
                    // Hardcoded camera (Phase 9 adds FPS controller)
                    let camera_pos = glam::Vec3::new(64.0, 80.0, -40.0);
                    let camera_target = glam::Vec3::new(64.0, 64.0, 64.0);
                    let camera_forward = (camera_target - camera_pos).normalize();
                    let camera_up = glam::Vec3::Y;
                    let fov_y = std::f32::consts::FRAC_PI_4; // 45° FOV

                    let pixel_to_ray = compute_pixel_to_ray(
                        camera_pos, camera_forward, camera_up, fov_y,
                        frame.swapchain_extent.width, frame.swapchain_extent.height,
                    );

                    let pc = PrimaryRayPushConstants {
                        pixel_to_ray: pixel_to_ray.to_cols_array_2d(),
                        resolution: [frame.swapchain_extent.width, frame.swapchain_extent.height],
                        _pad: [0; 2],
                    };

                    let output_extent = pass.output_image.extent;
                    let output_img = pass.output_image.handle;

                    let primary_ray_writes = graph.add_pass(
                        "primary_ray",
                        QueueType::Compute,
                        |builder| {
                            let _output = builder.create_image(
                                frame.swapchain_extent.width,
                                frame.swapchain_extent.height,
                                vk::Format::R8G8B8A8_UNORM,
                                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                            );
                            Box::new(move |ctx| {
                                pass.record(ctx.device, ctx.command_buffer, &pc);
                            })
                        },
                    );

                    let src_image = output_img;
                    let src_extent = output_extent;
                    let dst_image = frame.swapchain_image;
                    let dst_extent = frame.swapchain_extent;
                    let dep_handle = primary_ray_writes[0];
                    graph.add_pass(
                        "blit_to_swapchain",
                        QueueType::Graphics,
                        |builder| {
                            builder.read(dep_handle);
                            Box::new(move |ctx| {
                                blit_to_swapchain::record_blit(
                                    ctx.device, ctx.command_buffer,
                                    src_image, src_extent, dst_image, dst_extent,
                                );
                            })
                        },
                    );
                }
```

- [ ] **Step 5: Update `resumed()` — replace pass initialization**

Replace the `// Initialize test pattern pass` block (lines 216-242) with:

```rust
        // Initialize primary ray pass (requires UCVH GPU resources)
        if self.primary_ray_pass.is_none() {
            if let Some(ucvh_gpu) = &self.ucvh_gpu {
                let renderer = self.renderer.as_ref().unwrap();
                let extent = renderer.swapchain_extent();
                let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/primary_ray.spv"));
                if spirv.is_empty() {
                    tracing::warn!("primary_ray.spv is empty — slangc may not be installed");
                } else {
                    match PrimaryRayPass::new(
                        renderer.device(),
                        renderer.allocator(),
                        extent.width,
                        extent.height,
                        spirv,
                        ucvh_gpu,
                    ) {
                        Ok(pass) => {
                            tracing::info!(
                                width = extent.width,
                                height = extent.height,
                                "initialized primary ray pass"
                            );
                            self.primary_ray_pass = Some(pass);
                        }
                        Err(error) => {
                            tracing::error!(%error, "failed to create primary ray pass");
                        }
                    }
                }
            }
        }
```

**Important:** Move this block AFTER the UCVH generation block (because it needs `ucvh_gpu` to exist). The order in `resumed()` should be:
1. Create window + renderer
2. Generate UCVH demo scene + create GPU resources
3. Initialize primary ray pass (needs GPU resources)
4. Run startup stage

- [ ] **Step 6: Update `Drop` impl**

Replace:
```rust
            if let Some(pass) = self.test_pattern_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
```
with:
```rust
            if let Some(pass) = self.primary_ray_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
```

- [ ] **Step 7: Remove unused import**

Remove the old import line:
```rust
use crate::render::passes::test_pattern::TestPatternPass;
```

Also add `use glam;` if not already imported (it is used via `glam::Vec3` in tick_frame).

- [ ] **Step 8: Verify compilation**

Run: `cargo check`
Expected: compiles cleanly. The old `test_pattern` module is still in `passes/mod.rs` (not removed — other code may reference it later or it can be cleaned up separately).

- [ ] **Step 9: Commit**

```bash
git add src/app.rs
git commit -m "feat(phase3): integrate PrimaryRayPass into app, replacing test pattern"
```

---

## Task 8: End-to-End Build & Visual Verification

**Files:** None (verification only)

This task ensures the full pipeline works: Slang → SPIR-V → Vulkan compute → visible sphere.

- [ ] **Step 1: Run all unit tests**

Run: `cargo test`
Expected: All existing tests pass (32+ from Phase 2 + 4 new camera tests from Task 1).

- [ ] **Step 2: Build with shader compilation**

Run: `cargo build 2>&1`
Expected: `Compiled assets/shaders/passes/primary_ray.slang` in build output (if slangc is installed). No compilation errors.

If slangc reports errors in the .slang files, fix them in the relevant shader file and re-commit.

- [ ] **Step 3: Run the application**

Run: `cargo run`
Expected: A window opens showing a shaded sphere (the demo scene from Phase 2's `SphereGenerator`) against a blue sky gradient. The sphere should have visible face normals as flat-shaded colors (different colors for faces pointing in X, Y, Z directions) with simple directional lighting.

**If the screen is black:**
- Check tracing output for "uploaded UCVH data to GPU" and "initialized primary ray pass"
- Verify `primary_ray.spv` is non-empty: `ls -la target/debug/build/*/out/shaders/primary_ray.spv`
- Check that the camera position is outside the world bounds and looking inward

**If there are visual artifacts:**
- DDA stepping bugs often show as "striped" or "checkerboard" patterns
- Brick boundary issues show as gaps between bricks (8-voxel aligned lines)
- Morton order mismatches show as scrambled/rotated voxels

- [ ] **Step 4: Final commit (if shader fixes were needed)**

```bash
git add -A
git commit -m "fix(phase3): shader compilation fixes for primary ray tracing"
```

---

## Summary

| Task | Description | Tests | Files |
|------|-------------|-------|-------|
| 1 | Camera math (`pixel_to_ray` matrix) | 4 unit tests | `camera.rs`, `mod.rs` |
| 2 | GPU struct definitions | compile-time | `voxel_common.slang` |
| 3 | Ray math shared shader | compile-time | `ray.slang`, `math.slang` |
| 4 | Voxel traversal library (two-level DDA) | compile-time | `voxel_traverse.slang` |
| 5 | Primary ray compute shader | compile-time | `primary_ray.slang` |
| 6 | PrimaryRayPass Rust struct | `cargo check` | `primary_ray.rs`, `mod.rs` |
| 7 | App integration | `cargo check` | `app.rs` |
| 8 | E2E verification | `cargo test` + visual | — |

**Expected result:** A window displaying a ray-traced voxel sphere with normal-based flat shading and sky gradient background. This proves the full UCVH → GPU → DDA → pixel pipeline works end-to-end.
