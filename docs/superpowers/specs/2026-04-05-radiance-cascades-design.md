# Phase 5: Radiance Cascades GI + Demo Scene — Design Specification

## 1. Overview

Phase 5 adds world-space 3D Radiance Cascades (RC) global illumination to the Revolumetric voxel engine, replacing the current hemisphere ambient approximation with physically-based multi-bounce indirect lighting. It also includes a Sponza-style demo scene to properly exercise and validate the GI system.

### 1.1 Goals

- Rich demo scene with diverse materials, emissives, and architectural features for GI testing
- 3-level cascade probe system at brick resolution (8-voxel spacing)
- Visibility-weighted trilinear interpolation for smooth indirect lighting
- Distance-based cascade merging (coarse-to-fine)
- UCVH-accelerated probe ray tracing
- Total probe buffer memory ~6 MB

### 1.2 Reference

Based on Shadertoy M3ycWt — a volumetric RC implementation with:
- 32×32×48 voxel grid, 5 LOD cascade levels
- 6 hemispheres per probe, (3×3) × 4^N rays per hemisphere
- Visibility-weighted trilinear interpolation merging
- DDA voxel ray tracing

### 1.3 Scope

Split into three sub-phases:
- **5A**: Demo scene generator + material system completion
- **5B**: RC trace + merge compute passes
- **5C**: Lighting integration + optimization

## 2. Demo Scene (5A)

### 2.1 Scene Description

A Sponza-inspired architectural scene in the 128³ world, occupying approximately 96×96×120 voxels centered in the world. Scale factor ~3-4x relative to M3ycWt's 32×32×48 scene.

Elements:

| Element | Approx Size | Material | GI Test Purpose |
|---------|------------|----------|-----------------|
| Stone building shell | 90×96×120 | Stone (0.95, 0.925, 0.9) | Enclosed indirect light |
| Checkered floor | Full ground | Stone | Diffuse reflection |
| Stone columns + arches | r=6, spacing=40 | Stone | Complex shadows + light penetration |
| Red cloth banner | Thin sheet | Red (0.99, 0.4, 0.4) | Color bleeding test |
| Green cloth banner | Thin sheet | Green (0.4, 0.99, 0.4) | Color bleeding test |
| Second floor + vaults | y≈48 | Stone | Multi-floor light transport |
| Fountain | r=6-10 | Stone | Curved occlusion |
| Emissive lamps (2-3) | Small | Warm (255, 150, 50) | Emissive GI test |
| Brick wall | One face | Brick (0.75, 0.65, 0.5) | Material diversity |
| Decorative wall | One face | Stone | Geometric detail |
| Ceiling | y≈90 | Stone | Enclosed reflection |
| Blue sphere | r≈10 | Blue (0.5, 0.6, 0.9) | Static colored object |

### 2.2 Implementation

New `SponzaGenerator` implementing the existing `VoxelGenerator` trait:

```rust
pub struct SponzaGenerator {
    pub scale: f32,
    pub offset: UVec3,
}

impl VoxelGenerator for SponzaGenerator {
    fn generate_brick(&self, brick_pos: UVec3, config: &UcvhConfig) -> Option<BrickData> {
        // Per-voxel SDF evaluation (DFBox, cylinder, sphere combinations)
        // Mirrors buffer_b.glsl logic scaled to 128³
    }
}
```

Called from `app.rs` to replace `generate_demo_scene()`.

### 2.3 Material System

Current `VoxelCell` already has `material: u16` and `emissive: [u8; 3]`. Material ID mapping:

| ID | Name | Albedo RGB |
|----|------|-----------|
| 0 | Default white | (1.0, 1.0, 1.0) |
| 1 | Stone | (0.95, 0.925, 0.9) |
| 2 | Red cloth | (0.99, 0.4, 0.4) |
| 3 | Green cloth | (0.4, 0.99, 0.4) |
| 4 | Blue | (0.5, 0.6, 0.9) |
| 5 | Brick | (0.75, 0.65, 0.5) |

Shader-side LUT in `primary_ray.slang`:

```slang
static const float3 MATERIAL_ALBEDO[6] = {
    float3(1.0, 1.0, 1.0),
    float3(0.95, 0.925, 0.9),
    float3(0.99, 0.4, 0.4),
    float3(0.4, 0.99, 0.4),
    float3(0.5, 0.6, 0.9),
    float3(0.75, 0.65, 0.5),
};
```

### 2.4 HitResult Extension

`voxel_traverse.slang` `HitResult` gains material data:

```slang
struct HitResult {
    bool hit;
    float3 position;
    float3 normal;
    float t;
    VoxelCell cell;  // NEW: material + emissive from brick_materials
};
```

### 2.5 G-Buffer Changes

`primary_ray.slang` writes real albedo and emissive:

```slang
// Hit path:
uint mat_id = hit.cell.material;
float3 albedo = MATERIAL_ALBEDO[min(mat_id, 5u)];
gbuffer0[tid.xy] = float4(albedo, 1.0);
gbuffer1[tid.xy] = uint4(
    encode_normal_id(hit.normal),
    hit.cell.emissive_r,
    hit.cell.emissive_g,
    hit.cell.emissive_b
);
```

Emissive voxels: lighting pass reads `gbuffer1.gba` and outputs emissive color directly (skip shadow + indirect).

## 3. Radiance Cascades Core (5B)

### 3.1 Cascade Level Design

3-level cascade system at brick resolution, leveraging the 16³ brick grid:

| Level | Probe Spacing | Probe Grid | Dirs/Face | Total Probes | Per Probe | Memory |
|-------|--------------|-----------|----------|-------------|----------|--------|
| C0 | 8 voxels (1 brick) | 16³ = 4,096 | 3×3 = 9 | 4,096 | 864B | 3.5 MB |
| C1 | 16 voxels (2 bricks) | 8³ = 512 | 6×6 = 36 | 512 | 3,456B | 1.7 MB |
| C2 | 32 voxels (4 bricks) | 4³ = 64 | 12×12 = 144 | 64 | 13,824B | 0.9 MB |

Total probe buffer: ~6 MB in a single `StructuredBuffer<float4>`.

Each probe has 6 hemisphere faces (±X, ±Y, ±Z). Each face stores `probe_size²` direction samples as `float4(weighted_radiance.rgb, ray_distance)`.

### 3.2 Buffer Layout

Single contiguous `StructuredBuffer<float4>`, cascades concatenated:

```
Offset 0:          C0 data — 4096 probes × 6 faces × 9 dirs = 221,184 entries
Offset 221,184:    C1 data — 512 probes × 6 faces × 36 dirs = 110,592 entries
Offset 331,776:    C2 data — 64 probes × 6 faces × 144 dirs = 55,296 entries
Total:             387,072 entries × 16 bytes = 6.2 MB
```

Probe indexing: `offset + ((probe_z * grid_y + probe_y) * grid_x + probe_x) * 6 * dirs_per_face + face * dirs_per_face + dir_index`

### 3.3 Hemisphere Direction Mapping

Ported from M3ycWt's `ComputeDir` / `ComputeDirEven` / `ProjectDir` (common.glsl):

- **ComputeDir(uv, probeSize)**: Maps 2D grid position to hemisphere direction. For probeSize ≤ 3 uses simple 8-direction + center layout. For larger probes uses concentric-square mapping for even distribution.
- **ComputeDirEven(uv, probeSize)**: Concentric square → hemisphere mapping with theta/phi parameterization. Ensures uniform solid angle coverage.
- **ProjectDir(dir, probeSize)**: Inverse of ComputeDir — maps a 3D direction to the nearest probe direction index. Used during merge for visibility lookup.

These go in `assets/shaders/shared/rc_common.slang`.

### 3.4 rc_trace Pass

Compute shader `assets/shaders/passes/rc_trace.slang`.

**Dispatch**: One dispatch per cascade level, C2 → C1 → C0, with barriers between.

**Workgroup**: `[numthreads(64, 1, 1)]` — each thread handles one (probe, face, direction) tuple.

**Per-invocation logic**:

```
1. Decode thread ID → (cascade_level, probe_xyz, face, dir_index)
2. Compute probe world position = probe_xyz * probe_spacing + spacing/2
3. If LOD > 0: apply GeoOffset (find nearest empty voxel near probe center)
   If LOD == 0: offset by -normal * 0.25
4. Check OutsideGeo — if probe is inside solid, write (0,0,0,-1) and return
5. Compute ray direction via ComputeDir, transform by face TBN
6. Trace ray through UCVH (trace_primary_ray)
7. If hit:
   a. ray_dist = hit.t
   b. If emissive: radiance = emissive_color
   c. Else:
      - Direct: shadow test → sun contribution
      - Indirect: IntegrateVoxel from HIGHER cascade (C1 reads C2, C0 reads C1)
      - radiance = (direct + indirect) * albedo
8. If miss:
   a. ray_dist = FAR
   b. radiance = sky_color(ray_dir)
9. Weight radiance by solid_angle × cos(theta)  (matching M3ycWt weighting)
10. Write float4(radiance, ray_dist) to probe buffer
```

**Cascade dependency**: C2 traces independently (uses sky only for indirect). C1 reads merged C2 data. C0 reads merged C1 data. This requires C2 trace → C2 portion of merge → C1 trace → C1 portion of merge → C0 trace.

Revised dispatch order:
```
Dispatch rc_trace C2 → barrier → Dispatch rc_merge C2→C1 → barrier →
Dispatch rc_trace C1 → barrier → Dispatch rc_merge C1→C0 → barrier →
Dispatch rc_trace C0
```

Actually, simpler: trace all levels reading the **previous frame's** probe data (temporal). This avoids cascade dependency within a single frame:

```
Frame N: rc_trace reads Frame N-1 probe buffer, writes Frame N probe buffer
```

This requires double-buffering the probe buffer (2 × 6.2 MB = 12.4 MB total), but eliminates all intra-frame cascade barriers and allows a single dispatch per level with no ordering constraint.

**Final decision: Double-buffered temporal approach.**

### 3.5 rc_merge Pass

Compute shader `assets/shaders/passes/rc_merge.slang`.

Operates on the **current frame's** probe buffer after all rc_trace dispatches complete.

**Dispatch**: One dispatch for C0 probes, one for C1 probes. Each thread handles one (probe, face, direction).

**Per-invocation logic (for cascade level L, merging from level L+1)**:

```
1. Read own traced radiance + ray_dist from probe buffer
2. Compute position in higher cascade's probe grid: lPos = probe_xyz * 0.5
3. Find 8 trilinear neighbors in higher cascade (clamp to grid bounds)
4. For each neighbor:
   a. Compute relative direction from neighbor probe to current probe world pos
   b. Use ProjectDir to find closest ray in neighbor's data
   c. Read neighbor's ray_dist for that direction
   d. Visibility test: if neighbor_ray_dist < distance_to_current_probe - 0.5
      → weight = 0.00001 (occluded)
      Else → weight = 1.0
   e. Read neighbor's 4 surrounding direction samples (2×2 bilinear in direction space)
   f. Accumulate weighted radiance
5. Trilinear interpolate all 8 neighbors using visibility weights
6. Distance-based blend:
   distInterp = clamp((own_ray_dist - lod_factor) * inv_lod_factor * 0.5, 0, 1)
   final = mix(own_radiance, merged_radiance, distInterp)
7. Write back to probe buffer (in-place, same cascade slot)
```

### 3.6 Rust-Side Pass Structures

**RcProbeBuffer** (new, in `src/render/`):

```rust
pub struct RcProbeBuffer {
    pub buffer: [GpuBuffer; 2],  // double-buffered
    pub current_frame: usize,    // 0 or 1, toggled each frame
    pub c0_offset: u32,          // 0
    pub c1_offset: u32,          // 221184
    pub c2_offset: u32,          // 331776
    pub total_entries: u32,      // 387072
}
```

**RcTracePass** (`src/render/passes/radiance_cascade_trace.rs`):

```rust
pub struct RcTracePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,  // per frame-slot
}
```

Bindings: scene UBO, UCVH buffers (4), probe buffer read (prev frame), probe buffer write (curr frame), push constants (cascade level, grid size, probe size, offsets).

**RcMergePass** (`src/render/passes/radiance_cascade_merge.rs`):

```rust
pub struct RcMergePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}
```

Bindings: probe buffer (read-write), push constants (cascade level, grid sizes, offsets).

## 4. Lighting Integration (5C)

### 4.1 integrate_probe Function

New function in `lighting_common.slang` or `rc_common.slang`:

```slang
float3 integrate_probe(float3 world_pos, float3 normal,
                       StructuredBuffer<float4> rc_probes,
                       uint c0_offset, uint3 c0_grid_size) {
    // Convert world position to C0 probe grid coordinate
    uint3 probe_coord = uint3(floor(world_pos / 8.0));
    probe_coord = clamp(probe_coord, uint3(0), c0_grid_size - 1);

    // Select face from normal (same as NORMAL_TABLE mapping)
    uint face = normal_to_face(normal);

    // Accumulate 9 direction samples for this face
    float3 total = float3(0.0);
    uint base = c0_offset +
        ((probe_coord.z * c0_grid_size.y + probe_coord.y) * c0_grid_size.x + probe_coord.x)
        * 6 * 9 + face * 9;
    for (uint i = 0; i < 9; i++) {
        total += rc_probes[base + i].xyz;
    }
    return total;
}
```

### 4.2 lighting.slang Changes

Add binding 9 for RC probe buffer. Replace `hemisphere_ambient` call:

```slang
// Before:
float3 ambient = base_color * hemisphere_ambient(normal, scene.sky_color, scene.ground_color);

// After:
float3 indirect = integrate_probe(position, normal, rc_probes, scene.rc_c0_offset, scene.rc_c0_grid);
float3 ambient = base_color * indirect;
```

Emissive handling:
```slang
uint3 emissive_raw = uint3(gb1.g, gb1.b, gb1.a);
if (emissive_raw.x + emissive_raw.y + emissive_raw.z > 0) {
    float3 emissive = float3(emissive_raw) / 255.0;
    output_image[tid.xy] = float4(pow(emissive, float3(1.0 / 2.2)), 1.0);
    return;
}
```

### 4.3 SceneUniforms Extension

Add RC-related uniforms to `GpuSceneUniforms`:

```rust
pub rc_c0_offset: u32,
pub rc_c0_grid: [u32; 3],  // 16, 16, 16
pub rc_enabled: u32,        // 0 or 1, for fallback
```

This grows the UBO. Needs padding review to stay 16-byte aligned.

## 5. Render Graph

### 5.1 Final Pass Order

```
voxel_upload
    ↓
primary_ray  ──→  gbuffer_pos, gbuffer0, gbuffer1
    ↓
rc_trace C2  ──→  probe_buffer[current] (C2 region)
    ↓ barrier        reads: UCVH, scene UBO, probe_buffer[previous]
rc_trace C1  ──→  probe_buffer[current] (C1 region)
    ↓ barrier
rc_trace C0  ──→  probe_buffer[current] (C0 region)
    ↓ barrier
rc_merge C1  ──→  probe_buffer[current] (C1 region, in-place)
    ↓ barrier        reads: probe_buffer[current] C2 region
rc_merge C0  ──→  probe_buffer[current] (C0 region, in-place)
    ↓ barrier        reads: probe_buffer[current] C1 region
lighting     ──→  output_image
    ↓               reads: gbuffers, UCVH, scene UBO, probe_buffer[current]
blit_to_swapchain
```

Note: rc_trace reads previous frame's buffer (temporal), rc_merge reads current frame's buffer (just written by rc_trace). After all passes, swap current/previous index.

### 5.2 Synchronization

- Between rc_trace dispatches: `COMPUTE_SHADER → COMPUTE_SHADER` buffer memory barrier on probe_buffer[current]
- Between rc_trace and rc_merge: `COMPUTE_SHADER → COMPUTE_SHADER` barrier
- Between rc_merge dispatches: `COMPUTE_SHADER → COMPUTE_SHADER` barrier
- rc_merge → lighting: `COMPUTE_SHADER → COMPUTE_SHADER` barrier on probe_buffer[current]

### 5.3 Double Buffer Swap

In `app.rs` at end of frame: `rc_probes.current_frame ^= 1;`

## 6. Optimizations

All optimizations preserve visual quality:

| Optimization | Mechanism | Impact |
|-------------|-----------|--------|
| UCVH two-level DDA | Existing hierarchy skips empty bricks | 2-4x faster ray trace |
| Probe skip (empty regions) | Check occupancy before tracing; probes far from geometry write sky directly | ~50-70% fewer traces |
| Temporal double-buffer | Read previous frame, no intra-frame cascade dependency | Eliminates ordering constraints; 1-frame latency (imperceptible for static) |
| Workgroup shared memory | Share DDA setup for rays in similar directions | Reduced register pressure |

Future optimization (not in Phase 5): sub-brick cascade (1-voxel spacing) for higher spatial precision, sparsely allocated only in occupied bricks.

## 7. File Structure

```
assets/shaders/
  shared/
    rc_common.slang              ← ComputeDir, ProjectDir, probe indexing, integrate_probe
    scene_common.slang           ← SceneUniforms (extended with RC fields)
    voxel_traverse.slang         ← HitResult gains VoxelCell
    lighting_common.slang        ← unchanged
  passes/
    primary_ray.slang            ← material LUT, real albedo/emissive output
    rc_trace.slang               ← probe ray tracing compute shader
    rc_merge.slang               ← cascade merging compute shader
    lighting.slang               ← integrate_probe replaces hemisphere_ambient

src/render/
    rc_probe_buffer.rs           ← RcProbeBuffer (double-buffered GPU storage)
  passes/
    radiance_cascade_trace.rs    ← RcTracePass
    radiance_cascade_merge.rs    ← RcMergePass
    primary_ray.rs               ← unchanged (G-buffer formats stay same)
    lighting.rs                  ← add RC buffer binding (binding 9)

src/voxel/
    generator.rs                 ← add SponzaGenerator
```
