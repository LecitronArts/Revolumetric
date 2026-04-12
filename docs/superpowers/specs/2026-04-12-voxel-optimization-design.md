# Voxel Engine Optimization Roadmap — Design Specification

## 1. Overview

Post-Phase-5 optimization plan covering GPU profiling, material system upgrade, soft shadows, performance optimization, and advanced GI. Excludes large world/LOD/streaming.

### 1.1 Goals

- Establish GPU performance baseline with per-pass timing
- Redesign VoxelCell with per-voxel color, roughness, metallic (same 64-bit size)
- Two-layer material system: per-voxel attributes + material template defaults
- Soft shadows replacing binary shadow rays
- Data-driven performance optimization
- Multi-bounce GI, temporal stability, per-face vertex AO

### 1.2 Phases

| Phase | Name | Deliverable | Depends On |
|-------|------|-------------|------------|
| P1 | GPU Profiling | Per-pass GPU timing + console readout | — |
| P2 | Material System | VoxelCell redesign + two-layer materials + G-buffer redesign | P1 |
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

### 3.1 VoxelCell Layout (64-bit → 64-bit redesign)

**Current (64-bit / 8 bytes):**
```
Rust:  material: u16 | flags: u16 | emissive: [u8; 3] | _pad: u8
GPU:   uint material_flags (low16: material, high16: flags)
       uint emissive_pad   (low24: emissive RGB, high8: pad)
```

**New (64-bit / 8 bytes):**
```
Rust:
  color:       [u8; 3]   // per-voxel base color RGB
  roughness:   u8        // 0=mirror, 255=fully rough
  metallic:    u8        // 0=dielectric, 255=full metal
  emissive:    u8        // scalar intensity (emissive_color = base_color * intensity/255)
  material_id: u8        // template index (0-255) for upper-layer defaults
  flags:       u8        // bit 0: override_roughness, bit 1: override_metallic, bits 2-7: reserved

GPU (two uint32 words):
  uint word0:  bits[7:0]   = color_r
               bits[15:8]  = color_g
               bits[23:16] = color_b
               bits[31:24] = roughness
  uint word1:  bits[7:0]   = metallic
               bits[15:8]  = emissive
               bits[23:16] = material_id
               bits[31:24] = flags
```

**Accessor helpers (voxel_common.slang):**
```slang
float3 voxel_color(VoxelCell cell) {
    return float3(cell.word0 & 0xFF, (cell.word0 >> 8) & 0xFF, (cell.word0 >> 16) & 0xFF) / 255.0;
}
float voxel_roughness(VoxelCell cell) { return float((cell.word0 >> 24) & 0xFF) / 255.0; }
float voxel_metallic(VoxelCell cell)  { return float(cell.word1 & 0xFF) / 255.0; }
float voxel_emissive_intensity(VoxelCell cell) { return float((cell.word1 >> 8) & 0xFF) / 255.0; }
uint  voxel_material_id(VoxelCell cell) { return (cell.word1 >> 16) & 0xFF; }
uint  voxel_flags(VoxelCell cell) { return (cell.word1 >> 24) & 0xFF; }
```

**Design rationale:**
- Per-voxel RGB replaces material_id→albedo LUT lookup
- 8-bit roughness/metallic gives 256 levels (4-bit is too coarse for metallic transitions)
- Emissive becomes scalar intensity × base color (saves 16 bits vs RGB emissive)
- Same 64-bit size — no memory impact change (512×8B = 4096B per brick, unchanged)
- material_id shrinks from u16 to u8 (256 templates). Current usage: 8 materials. 256 is sufficient for procedural generation; a u16 can be reconsidered if file import is added later.

**Template override resolution (flags-based, avoids roughness=0 ambiguity):**
- `flags bit 0` (override_roughness): if set, use per-voxel roughness; if clear, use template default
- `flags bit 1` (override_metallic): if set, use per-voxel metallic; if clear, use template default
- This allows roughness=0 to be a valid mirror surface value, not conflated with "use default"
- The procedural generator always sets both override bits when writing per-voxel PBR values

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

Templates are CPU-side only. The procedural generator resolves templates at voxel write time:
```rust
fn write_voxel(cell: &mut VoxelCell, color: [u8; 3], template: &MaterialTemplate,
               roughness_override: Option<u8>, metallic_override: Option<u8>) {
    cell.color = color;
    cell.emissive = 0;
    cell.material_id = template.id;
    if let Some(r) = roughness_override {
        cell.roughness = r;
        cell.flags |= 0x01; // override_roughness
    } else {
        cell.roughness = template.default_roughness;
        cell.flags |= 0x01; // always set — template resolved at write time
    }
    // same for metallic with bit 1
}
```

Since templates are resolved at write time, the GPU never sees template lookups. This keeps the shader path simple: always read roughness/metallic directly from VoxelCell.

### 3.3 G-Buffer Redesign

Current:
- `gbuffer0` (RGBA8): rgb = base_color (from LUT), a = ao
- `gbuffer1` (RGBA8UI): r = normal_id, g = emissive_r, b = emissive_g, a = emissive_b

New:
- `gbuffer0` (RGBA8): rgb = per-voxel color (direct from VoxelCell), a = roughness (normalized)
- `gbuffer1` (RGBA8UI): r = normal_id, g = metallic, b = emissive_intensity, a = reserved (0)

**Impact:** Removes `MATERIAL_ALBEDO` LUT from `primary_ray.slang` and the if-else chain from `rc_trace.slang`. Colors come directly from voxel data.

### 3.4 Shader Changes

**voxel_common.slang:**
- Replace `struct VoxelCell { uint material_flags; uint emissive_pad; }` with `struct VoxelCell { uint word0; uint word1; }`
- Replace old accessors (`voxel_material`, `voxel_emissive`) with new ones (see 3.1)

**primary_ray.slang:**
- Read color_rgb via `voxel_color(hit.cell)` instead of LUT lookup
- Write roughness to gbuffer0.a via `voxel_roughness(hit.cell)`
- Write metallic to gbuffer1.g via `uint(voxel_metallic(hit.cell) * 255.0)`
- Write emissive_intensity to gbuffer1.b

**rc_trace.slang:**
- Read albedo via `voxel_color(hit.cell)` (remove hardcoded if-else chain at lines 113-121)
- Emissive = `voxel_color(hit.cell) * voxel_emissive_intensity(hit.cell) * HDR_SCALE`

**lighting.slang:**
- Read roughness from gbuffer0.a, metallic from gbuffer1.g / 255.0
- Apply Cook-Torrance BRDF (see §3.5)
- Emissive path: `base_color * emissive_intensity * HDR_SCALE`

### 3.5 Cook-Torrance BRDF Specification

For metallic/rough surfaces, replace the current Lambert-only direct lighting with a microfacet BRDF:

**Components:**
- **NDF:** GGX/Trowbridge-Reitz: `D = α² / (π * (NdotH² * (α²-1) + 1)²)` where `α = roughness²`
- **Geometry:** Smith-Schlick-GGX: `G = G1(N,V) * G1(N,L)` where `G1(N,X) = NdotX / (NdotX * (1-k) + k)`, `k = (roughness+1)² / 8`
- **Fresnel:** Schlick approximation: `F = F0 + (1-F0) * (1-VdotH)⁵` where `F0 = lerp(0.04, base_color, metallic)`

**View vector reconstruction:** `V = normalize(camera_origin - position)` where `camera_origin = scene.pixel_to_ray[3].xyz` (stored in column 3 of the 4×4 matrix) and `position` from gbuffer_pos.

**Final direct lighting:**
```slang
float3 F0 = lerp(float3(0.04), base_color, metallic);
float3 H = normalize(V + L);
float NdotL = max(dot(N, L), 0.0);
float NdotV = max(dot(N, V), 0.001);
float NdotH = max(dot(N, H), 0.0);
float VdotH = max(dot(V, H), 0.0);

float D = DistributionGGX(NdotH, roughness);
float G = GeometrySmith(NdotV, NdotL, roughness);
float3 F = FresnelSchlick(VdotH, F0);

float3 specular = D * G * F / max(4.0 * NdotV * NdotL, 0.001);
float3 kD = (1.0 - F) * (1.0 - metallic);
float3 diffuse = kD * base_color / PI;
float3 direct = (diffuse + specular) * sun_intensity * NdotL * shadow;
```

### 3.6 Sponza Generator Update

Update `SponzaGenerator::eval_voxel` to return the new VoxelCell format:
- Stone: color(165,160,155), roughness=200, metallic=5
- Metal fixtures: color(180,175,170), roughness=60, metallic=220
- Wood: color(115,76,46), roughness=180, metallic=3
- Cloth: color varies, roughness=240, metallic=0
- Water: color(30,50,80), roughness=20, metallic=0, emissive=100

All voxels set both override flags (bit 0 and bit 1 in flags) since templates are resolved at write time.

### 3.7 Files

- Modified: `src/voxel/brick.rs` (VoxelCell struct — same size, new fields)
- Modified: `src/voxel/sponza_generator.rs` (new format)
- Modified: `assets/shaders/passes/primary_ray.slang` (direct color, roughness/metallic)
- Modified: `assets/shaders/passes/rc_trace.slang` (remove albedo LUT)
- Modified: `assets/shaders/passes/lighting.slang` (Cook-Torrance BRDF, read roughness/metallic)
- Modified: `assets/shaders/shared/voxel_common.slang` (VoxelCell layout + new accessors)

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
- New **R16F** texture: `shadow_history` (single-channel — shadow is a scalar 0/1 value)
- Blend: `shadow_out = lerp(shadow_history[tid.xy], current_shadow, 0.1)`
- On camera move: increase blend factor to 0.3 (faster convergence)
- Reprojection strategy: use `prev_view_proj` matrix to find previous screen position. If previous pixel is invalid (off-screen or depth mismatch > threshold), reset with blend factor 1.0 (no history).

**Reprojection detail:**
```slang
// In lighting.slang:
float4 prev_clip = mul(scene.prev_view_proj, float4(position, 1.0));
float2 prev_uv = prev_clip.xy / prev_clip.w * 0.5 + 0.5;
bool valid = all(prev_uv >= 0.0) && all(prev_uv <= 1.0);
float blend = valid ? 0.1 : 1.0;
// On large camera motion (detected CPU-side), set scene.shadow_reset = 1 → blend = 1.0
```

### 4.3 SceneUniforms Extension

**New fields added to SceneUniforms (Rust + Slang must match):**
- `frame_count: u32` (for hash seed)
- `shadow_reset: u32` (1 = force reset temporal history)
- `prev_view_proj: float4x4` (64 bytes, for shadow reprojection)
- `sun_tangent: float3` + pad (16 bytes, for jitter basis)
- `sun_bitangent: float3` + pad (16 bytes, for jitter basis)

**UBO size change:** 176B → 176 + 4 + 4 + 64 + 16 + 16 = **280B** (padded to 288B for 16-byte alignment).

**All files referencing UBO size that must be updated:**
- `src/render/scene_ubo.rs` — struct definition + size test (`gpu_scene_uniforms_size_is_176_bytes` → 288)
- `assets/shaders/shared/scene_common.slang` — struct definition + size comment
- Any descriptor set layout that specifies UBO range (verify in `src/render/passes/*.rs`)

### 4.4 Files

- Modified: `assets/shaders/passes/lighting.slang` (stochastic shadow + temporal blend + reprojection)
- Modified: `assets/shaders/shared/scene_common.slang` (new uniforms, updated size)
- Modified: `src/render/scene_ubo.rs` (new fields, updated size, updated test)
- Modified: `src/app.rs` (compute prev_view_proj, sun_tangent/bitangent, frame_count, shadow_reset)
- New: shadow history texture allocation in `LightingPass` (R16F, same resolution as output)
- New: `hash22()` utility function in `lighting_common.slang`

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

Cheap screen-space AO from occupancy lookups:
- For each hit face, sample 8 neighboring voxels (4 edge + 4 corner)
- Compute per-vertex AO: `ao = (side1 + side2 + max(corner, side1 * side2)) / 3`
- Bilinear interpolate across the face using hit position within voxel
- Apply as multiplier on ambient term

**Hit UV computation (no extra G-buffer channel needed):**
The hit position within the voxel is derived in `lighting.slang` from `gbuffer_pos` and the normal:
```slang
float3 frac_pos = fract(position);  // position within voxel [0,1)^3
// Project onto hit face based on normal_id to get 2D UV:
// +X/-X face: uv = (frac_pos.z, frac_pos.y)
// +Y/-Y face: uv = (frac_pos.x, frac_pos.z)
// +Z/-Z face: uv = (frac_pos.x, frac_pos.y)
```
No additional G-buffer channel required — `gbuffer_pos.xyz` already contains sub-voxel precision positions from the DDA ray intersection.

**Cost:** 8 voxel lookups + 4 interpolations per pixel. Negligible vs current shadow ray cost.

### 6.4 Files

- Modified: `assets/shaders/passes/rc_trace.slang` (temporal blend, smooth normal for bounce)
- Modified: `assets/shaders/passes/lighting.slang` (vertex AO with computed hit UV)

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
