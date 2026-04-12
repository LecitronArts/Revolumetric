# Voxel Engine Optimization Roadmap — Design Specification

## 1. Overview

Post-Phase-5 optimization plan covering GPU profiling, material system upgrade, soft shadows, performance optimization, and advanced GI. Excludes large world/LOD/streaming.

### 1.1 Goals

- Establish GPU performance baseline with per-pass timing
- Upgrade VoxelCell to 64-bit with per-voxel color, roughness, metallic
- Two-layer material system: per-voxel attributes + material template defaults
- Soft shadows replacing binary shadow rays
- Data-driven performance optimization
- Multi-bounce GI, temporal stability, per-face vertex AO

### 1.2 Phases

| Phase | Name | Deliverable | Depends On |
|-------|------|-------------|------------|
| P1 | GPU Profiling | Per-pass GPU timing + console readout | — |
| P2 | Material System | 64-bit VoxelCell + two-layer materials + G-buffer redesign | P1 |
| P3 | Soft Shadows | Stochastic shadow rays + temporal accumulation | P2 |
| P4 | Performance Optimization | Bottleneck-targeted optimization (informed by P1 data) | P1, P2, P3 |
| P5 | Advanced GI | Multi-bounce + temporal probe stability + vertex AO | P2, P3 |

## 2. Phase P1: GPU Profiling

### 2.1 Design

New `GpuProfiler` struct wrapping a Vulkan `VkQueryPool` (timestamp type).

**Measurement points (8 scopes):**
1. Primary Ray Pass
2. RC Trace C0
3. RC Trace C1
4. RC Trace C2
5. RC Merge C2→C1
6. RC Merge C1→C0
7. Lighting Pass
8. Blit to Swapchain

**API:**
```rust
pub struct GpuProfiler {
    query_pool: vk::QueryPool,
    query_count: u32,
    timestamp_period: f64,  // ns per tick
    frame_count: u64,
    accumulators: Vec<f64>,
    names: Vec<&'static str>,
}

impl GpuProfiler {
    fn new(device: &ash::Device, physical: vk::PhysicalDevice, scope_count: u32) -> Self;
    fn begin_scope(&self, device: &ash::Device, cmd: vk::CommandBuffer, scope: usize);
    fn end_scope(&self, device: &ash::Device, cmd: vk::CommandBuffer, scope: usize);
    fn read_back_and_print(&mut self, device: &ash::Device);  // every 60 frames
}
```

**Output format (every 60 frames):**
```
[GPU] PrimaryRay: 1.23ms | RC-C0: 0.45ms | RC-C1: 0.89ms | RC-C2: 0.34ms | Merge: 0.12ms | Lighting: 0.67ms | Blit: 0.05ms | Total: 3.75ms
```

### 2.2 Files

- New: `src/render/gpu_profiler.rs`
- Modified: `src/app.rs` (insert begin/end calls around each pass)

### 2.3 Scope

~150 lines new code. No shader changes. No visual impact.

## 3. Phase P2: Material System Upgrade

### 3.1 VoxelCell Layout (64-bit)

Current (48-bit):
```
material_id: u16 | occupancy: u8 | emissive_r: u8 | emissive_g: u8 | emissive_b: u8
```

New (64-bit):
```
color_r:    u8  // per-voxel base color R
color_g:    u8  // per-voxel base color G
color_b:    u8  // per-voxel base color B
roughness:  u8  // 0=mirror, 255=fully rough
metallic:   u8  // 0=dielectric, 255=full metal
emissive:   u8  // scalar intensity (emissive_color = base_color * intensity/255)
material_id:u8  // template index (0-255) for upper-layer defaults
flags:      u8  // bit 0: occupied, bits 1-3: reserved
```

**Design rationale:**
- Per-voxel RGB replaces material_id→albedo LUT lookup
- 8-bit roughness/metallic gives 256 levels (4-bit is too coarse for metallic transitions)
- Emissive becomes scalar intensity × base color (saves 16 bits vs RGB emissive)
- Material template ID provides defaults; per-voxel values override when non-zero
- 64-bit alignment is GPU-optimal for structured buffers

**Memory impact:** Brick data grows from 3072B (512×6B) to 4096B (512×8B) per brick. +33% brick memory but better GPU alignment.

### 3.2 Material Template Table

```rust
pub struct MaterialTemplate {
    pub name: &'static str,
    pub default_roughness: u8,
    pub default_metallic: u8,
    pub subsurface: f32,        // future: subsurface scattering
    pub ior: f32,               // future: index of refraction
}
```

Templates are CPU-side only for now. The procedural generator uses them to set per-voxel defaults:
```rust
// Stone template: rough, non-metallic
let stone = MaterialTemplate { roughness: 200, metallic: 5, .. };
// Metal template: smooth, metallic
let iron = MaterialTemplate { roughness: 80, metallic: 230, .. };
```

### 3.3 G-Buffer Redesign

Current:
- `gbuffer0` (RGBA8): rgb = base_color (from LUT), a = ao
- `gbuffer1` (RGBA8UI): r = normal_id, g = emissive_r, b = emissive_g, a = emissive_b

New:
- `gbuffer0` (RGBA8): rgb = per-voxel color (direct from VoxelCell), a = roughness (normalized)
- `gbuffer1` (RGBA8UI): r = normal_id, g = metallic, b = emissive_intensity, a = flags

**Impact:** Removes `MATERIAL_ALBEDO` LUT from `primary_ray.slang` and the if-else chain from `rc_trace.slang`. Colors come directly from voxel data.

### 3.4 Shader Changes

**primary_ray.slang:**
- Read color_rgb directly from VoxelCell instead of LUT lookup
- Write roughness to gbuffer0.a, metallic to gbuffer1.g

**rc_trace.slang:**
- Read albedo directly from VoxelCell.color_rgb (remove hardcoded if-else chain)
- Emissive = VoxelCell.color * VoxelCell.emissive / 255.0

**lighting.slang:**
- Read roughness from gbuffer0.a, metallic from gbuffer1.g
- Apply Cook-Torrance BRDF for specular (metallic surfaces)
- Fresnel term: `F0 = lerp(0.04, base_color, metallic)`
- Emissive path: `base_color * emissive_intensity / 255.0 * HDR_SCALE`

### 3.5 Sponza Generator Update

Update `SponzaGenerator::eval_voxel` to return the new VoxelCell format:
- Stone: color(165,160,155), roughness=200, metallic=5
- Metal fixtures: color(180,175,170), roughness=60, metallic=220
- Wood: color(115,76,46), roughness=180, metallic=3
- Cloth: color varies, roughness=240, metallic=0
- Water: color(30,50,80), roughness=20, metallic=0, emissive=100

### 3.6 Files

- Modified: `src/voxel/brick.rs` (VoxelCell struct)
- Modified: `src/voxel/sponza_generator.rs` (new format)
- Modified: `assets/shaders/passes/primary_ray.slang` (direct color, roughness/metallic)
- Modified: `assets/shaders/passes/rc_trace.slang` (remove albedo LUT)
- Modified: `assets/shaders/passes/lighting.slang` (Cook-Torrance BRDF, read roughness/metallic)
- Modified: `assets/shaders/shared/voxel_common.slang` (VoxelCell layout)

## 4. Phase P3: Soft Shadows

### 4.1 Approach: Stochastic Shadow Rays + Temporal Accumulation

Instead of 1 binary shadow ray, trace 1 shadow ray per frame with random angular offset (simulating finite sun disk), then temporally blend with previous frames.

**Why not multi-sample per frame:** RC trace already dominates GPU time. Adding 4-9 shadow rays per pixel would be expensive. 1 stochastic ray + temporal is cheaper and gives comparable quality.

### 4.2 Implementation

**lighting.slang:**
```slang
// Jitter sun direction to simulate finite sun disk (angular radius ~0.5 degrees)
float sun_radius = 0.015;  // ~0.86 degrees
uint2 seed = tid.xy ^ uint2(scene.frame_count * 1791, scene.frame_count * 3571);
float2 xi = hash22(seed);  // [0,1]^2
float3 jittered_sun = normalize(scene.sun_direction + sun_tangent * (xi.x - 0.5) * sun_radius
                                                    + sun_bitangent * (xi.y - 0.5) * sun_radius);
```

**Temporal accumulation (new history buffer):**
- New RGBA16F texture: `shadow_history`
- Blend: `shadow_out = lerp(shadow_history[tid.xy], current_shadow, 0.1)`
- On camera move: increase blend factor to 0.3 (faster convergence)
- Reproject using motion vectors (or reset on large camera motion)

### 4.3 SceneUniforms Extension

Add to SceneUniforms:
- `frame_count: u32` (for hash seed)
- `sun_tangent: float3` (for jitter basis)
- `sun_bitangent: float3`

### 4.4 Files

- Modified: `assets/shaders/passes/lighting.slang` (stochastic shadow + temporal blend)
- Modified: `assets/shaders/shared/scene_common.slang` (new uniforms)
- Modified: `src/render/scene_ubo.rs` (new fields)
- New: shadow history texture allocation in `LightingPass`

## 5. Phase P4: Performance Optimization

### 5.1 Approach

Data-driven: use P1 profiling results to identify the bottleneck, then apply targeted optimization. Potential directions:

**If RC Trace is bottleneck (likely):**
- Reduce C2 trace frequency (trace every 2nd frame, interpolate)
- Early-out for probes with low variance (skip stable probes)
- Occupancy-based skip: don't trace probes in fully empty regions

**If Primary Ray is bottleneck:**
- Hierarchical DDA with mipmap occupancy (skip empty 2×2×2 / 4×4×4 regions)
- Brick-level bounding box early-out

**If Lighting is bottleneck:**
- Reduce integrate_probe cost: cache face irradiance per probe instead of per-pixel sum
- Fewer rc_outside_geo calls in gradient normal (use LUT or precompute)

### 5.2 SVO Evaluation

Evaluate whether SVO (from `reference/SparseVoxelOctree`) would improve traversal speed vs current UCVH. This is a larger refactor — only pursue if profiling shows primary ray is the dominant cost AND the scene is sparse enough to benefit.

### 5.3 Files

Depends on profiling results. No files can be specified in advance.

## 6. Phase P5: Advanced GI

### 6.1 Temporal Probe Stability

Current RC probes are fully recomputed each frame → flicker on moving light/objects.

**Fix:** Exponential moving average on probe data:
```slang
// In rc_trace, after computing result:
float4 prev = probe_read[idx];
float blend = 0.15;  // 85% old, 15% new
probe_write[idx] = lerp(prev, result, blend);
```

Skip blending for inside-geo probes (w=-1) and first frame.

### 6.2 Multi-Bounce Enhancement

Current: 1 indirect bounce in rc_trace (reads probe_read for hit surface irradiance).
Already functional via temporal accumulation — each frame's trace reads last frame's probes which contain previous bounce data. Over ~10 frames, this naturally converges to multi-bounce.

**Enhancement:** Use gradient smooth normal for the indirect bounce in rc_trace (currently uses DDA normal). This improves bounce quality on curved surfaces.

### 6.3 Per-Face Vertex AO

From Phase 12 design. Cheap screen-space AO from occupancy lookups:
- For each hit face, sample 8 neighboring voxels (4 edge + 4 corner)
- Compute per-vertex AO: `ao = (side1 + side2 + max(corner, side1 * side2)) / 3`
- Bilinear interpolate across the face using hit position within voxel
- Apply as multiplier on ambient term

**Cost:** 8 voxel lookups + 4 interpolations per pixel. Negligible vs current shadow ray cost.

### 6.4 Files

- Modified: `assets/shaders/passes/rc_trace.slang` (temporal blend, smooth normal for bounce)
- Modified: `assets/shaders/passes/lighting.slang` (vertex AO)
- Modified: `assets/shaders/passes/primary_ray.slang` (output hit UV within voxel for AO interpolation)

## 7. Success Criteria

- P1: GPU timing visible in console, all passes measured
- P2: Voxels render with per-voxel color, roughness/metallic affect specular highlights
- P3: Shadow edges are soft, no temporal flickering after convergence
- P4: Identified bottleneck has measurable improvement (>20% speedup in target pass)
- P5: No visible flicker in static scenes, AO darkens corners/crevices naturally

## 8. Reference Projects

Cloned to `reference/` for implementation guidance:
- `amitabha` — Holographic RC (Rust), probe merge strategies
- `three-rc` — 3D RC, cascade layout
- `SparseVoxelOctree` — GPU SVO traversal
- `RTXGI` — DDGI probe management best practices
- `HashDAG` — Compact voxel storage
- `vk_voxel_cone_tracing` — Vulkan VCT pipeline
- `Q2RTX` — ASVGF denoiser shaders
