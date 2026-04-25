# Phase 4: Shadow & Lighting — Design Spec

## Goal

Replace flat normal shading with proper shadow ray tracing + directional lighting + hemisphere ambient + ACES tonemapping. Introduce G-buffer architecture and Scene UBO as shared data pipeline for all passes.

## Architecture Overview

Two compute passes with G-buffer intermediate:

```
Primary Ray Pass (compute)
  reads:  Scene UBO (ConstantBuffer), UCVH buffers
  writes: gbuffer_pos (RGBA32F), gbuffer0 (RGBA8), gbuffer1 (RGBA8_UINT)

Lighting Pass (compute)
  reads:  Scene UBO (ConstantBuffer), gbuffer_pos, gbuffer0, gbuffer1, UCVH buffers
  writes: output_image (RGBA8)

Blit to Swapchain
```

## Constraints

- No new dependencies
- Reuse existing DDA traversal for shadow rays
- Remove all push constants — pure Scene UBO
- Build.rs already compiles `assets/shaders/passes/*.slang` to SPIR-V

## Scene UBO

Single uniform buffer shared by all passes. Written once per frame on CPU, read by every pass via descriptor binding.

### GPU Layout (Slang — `assets/shaders/shared/scene_common.slang`)

Both passes `#include "scene_common.slang"`. The struct uses Slang's default RowMajor layout for `float4x4` in constant buffers, matching the Rust-side transpose convention.

```slang
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
};                            // total: 144B, 16-byte aligned
```

### Rust Side

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuSceneUniforms {
    pub pixel_to_ray: [[f32; 4]; 4],
    pub resolution: [u32; 2],
    pub _pad0: [u32; 2],
    pub sun_direction: [f32; 3],
    pub _pad1: f32,
    pub sun_intensity: [f32; 3],
    pub _pad2: f32,
    pub sky_color: [f32; 3],
    pub _pad3: f32,
    pub ground_color: [f32; 3],
    pub time: f32,
}
```

Size: 144 bytes. Static assert in tests.

### Scene UBO Management

New struct `SceneUniformBuffer` in `src/render/scene_ubo.rs`:
- Owns **one GPU buffer per frame slot** (N × 144 bytes, where N = `swapchain.images.len()`, `UNIFORM_BUFFER` usage, host-visible + `HOST_COHERENT`). One buffer per frame slot prevents CPU overwriting a buffer the GPU is still reading.
- `update(&self, device, frame_index: usize, data: &GpuSceneUniforms)` — memcpy via persistent mapped pointer to the buffer for the current frame slot
- `destroy(device, allocator)` — cleanup

Created once at renderer init. Each pass owns **N descriptor sets** (one per frame slot, where N = `swapchain.images.len()`). Descriptor set `k` has binding 0 permanently pointing to UBO buffer `k`. At record time, bind the descriptor set for the current frame slot. This avoids `vkUpdateDescriptorSets` on in-flight descriptor sets. Pass-specific bindings (G-buffer images, UCVH buffers) are identical across all N sets since those resources are not double-buffered.

### Filling Per Frame (app.rs)

Each frame in `tick_frame`, after `update_camera`:

```
1. Read CameraRig → compute pixel_to_ray (with transpose for Slang RowMajor)
2. Read DirectionalLight → sun_direction, sun_intensity
3. Fill sky_color = (0.4, 0.5, 0.7), ground_color = (0.15, 0.1, 0.08)  (hardcoded ambient for now)
4. Write GpuSceneUniforms → SceneUniformBuffer::update()
```

## G-buffer

### Format (Industrial Standard, Voxel-Adapted)

| Image | Format | Content | Size/px |
|-------|--------|---------|---------|
| `gbuffer_pos` | `R32G32B32A32_SFLOAT` | xyz = voxel center position, w = hit_t (negative = miss) | 16B |
| `gbuffer0` | `R8G8B8A8_UNORM` | rgb = base_color, a = ao | 4B |
| `gbuffer1` | `R8G8B8A8_UINT` | r = normal_id (0-5), g = roughness_u8, b = metallic_u8, a = flags | 4B |

Total: 24 bytes/pixel.

Why store position explicitly instead of reconstructing from depth: The DDA's `hit_t` is the parametric distance to the voxel EXIT face (not entry). Reconstructing `origin + dir * hit_t` places the point inside or past the solid voxel, causing shadow ray self-intersection. Storing the voxel center (`brick_origin + float3(hit_local) + 0.5`, already computed by `trace_primary_ray`) avoids this issue entirely.

### Normal ID Encoding

```
0 = +X    1 = -X
2 = +Y    3 = -Y
4 = +Z    5 = -Z
```

Lighting pass reconstructs via lookup table:

```slang
static const float3 NORMAL_TABLE[6] = {
    float3(1,0,0), float3(-1,0,0),
    float3(0,1,0), float3(0,-1,0),
    float3(0,0,1), float3(0,0,-1),
};
```

### Flags Byte (gbuffer1.a)

- Bit 0: hit (1 = geometry hit, 0 = sky/miss)
- Bit 1: emissive
- Bits 2-7: reserved

### Default Values (Phase 4)

No material system yet. Primary ray writes:
- base_color: `(255, 255, 255)` — white
- ao: `255` — no occlusion
- roughness: `128` — 0.5
- metallic: `0` — non-metallic
- flags: `0x01` on hit, `0x00` on miss

### Shadow Ray Origin

Lighting pass reads voxel center position directly from `gbuffer_pos.xyz` and offsets by `normal * 0.51` (half voxel width + epsilon) to guarantee exiting the solid voxel through the entry face:

```slang
float3 shadow_origin = gbuffer_pos.xyz + normal * 0.51;
```

This is robust because voxels are unit cubes and the position is always the voxel center.

## Primary Ray Pass Refactor

### Changes from Current

| Aspect | Before | After |
|--------|--------|-------|
| Data input | Push constants (80B) | Scene UBO binding (`ConstantBuffer<SceneUniforms>`) |
| Output | 1 RGBA8 image (final color) | 3 images (gbuffer_pos + gbuffer0 + gbuffer1) |
| Shader | Computes color inline | Writes G-buffer only, no lighting |
| Pipeline | Push constant range declared | No push constant range |

### Descriptor Bindings (Primary Ray)

| Binding | Type | Resource |
|---------|------|----------|
| 0 | ConstantBuffer | Scene UBO (`ConstantBuffer<SceneUniforms>`) |
| 1 | RWTexture2D<float4> | gbuffer_pos (RGBA32F, write) |
| 2 | RWTexture2D<float4> | gbuffer0 (RGBA8_UNORM, write) |
| 3 | RWTexture2D<uint4> | gbuffer1 (RGBA8_UINT, write) |
| 4 | StructuredBuffer<UcvhConfig> | UCVH config |
| 5 | StructuredBuffer<NodeL0> | hierarchy_l0 |
| 6 | StructuredBuffer<BrickOccupancy> | brick_occupancy |
| 7 | StructuredBuffer<VoxelCell> | brick_materials |

### Shader Output

```slang
[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Read scene uniforms (ConstantBuffer — no array index)
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    // Generate ray (same math as before)
    float2 pixel = float2(tid.xy);
    float3 origin = scene.pixel_to_ray[3].xyz;
    float3x3 dir_mat = float3x3(
        scene.pixel_to_ray[0].xyz,
        scene.pixel_to_ray[1].xyz,
        scene.pixel_to_ray[2].xyz
    );
    float3 dir = normalize(mul(float3(pixel, 1.0), dir_mat));

    Ray ray = make_ray(origin, dir);
    HitResult hit = trace_primary_ray(ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);

    if (hit.hit) {
        gbuffer_pos[tid.xy] = float4(hit.position, hit.t);
        gbuffer0[tid.xy] = float4(1.0, 1.0, 1.0, 1.0);  // white albedo, no AO
        gbuffer1[tid.xy] = uint4(encode_normal_id(hit.normal), 128, 0, 0x01);
    } else {
        gbuffer_pos[tid.xy] = float4(0, 0, 0, -1.0);  // w < 0 = miss
        gbuffer0[tid.xy] = float4(0, 0, 0, 0);
        gbuffer1[tid.xy] = uint4(0, 0, 0, 0);
    }
}
```

### `encode_normal_id` Helper

```slang
// Encodes an axis-aligned normal to a 0-5 ID.
// Assumes input is a unit axis-aligned vector from DDA face normals.
// Fallback to -Z (5) for degenerate zero normals (should not occur in practice).
uint encode_normal_id(float3 n) {
    if (n.x >  0.5) return 0;
    if (n.x < -0.5) return 1;
    if (n.y >  0.5) return 2;
    if (n.y < -0.5) return 3;
    if (n.z >  0.5) return 4;
    return 5; // -Z
}
```

Added to `voxel_common.slang` alongside existing helpers.

## Lighting Pass (New)

### Shader: `assets/shaders/passes/lighting.slang`

Required includes:
```slang
#include "scene_common.slang"       // SceneUniforms
#include "voxel_traverse.slang"     // trace_primary_ray (for shadow rays)
#include "lighting_common.slang"    // ACES tonemap, sky_color_for_dir
```

Full-screen compute shader. For each pixel:

1. Read `gbuffer_pos` — if `w < 0` (miss), reconstruct ray direction and output sky gradient:
   ```slang
   // Reconstruct ray direction for sky gradient (same math as primary ray pass)
   float2 pixel = float2(tid.xy);
   float3x3 dir_mat = float3x3(
       scene.pixel_to_ray[0].xyz,
       scene.pixel_to_ray[1].xyz,
       scene.pixel_to_ray[2].xyz
   );
   float3 ray_dir = normalize(mul(float3(pixel, 1.0), dir_mat));
   output_image[tid.xy] = float4(sky_color_for_dir(ray_dir, scene), 1.0);
   return;
   ```
2. Read voxel center position from `gbuffer_pos.xyz`
3. Read normal_id from `gbuffer1.r`, lookup normal vector via `NORMAL_TABLE`
4. Read base_color from `gbuffer0.rgb`
5. Trace shadow ray: from `gbuffer_pos.xyz + normal * 0.51` toward sun, using existing `trace_primary_ray`
6. Compute lighting:
   - `shadow = shadow_hit ? 0.0 : 1.0`
   - `ndotl = max(dot(normal, sun_direction), 0.0)`
   - `diffuse = base_color * sun_intensity * ndotl * shadow`
   - `ambient = base_color * lerp(ground_color, sky_color, normal.y * 0.5 + 0.5)`
   - `hdr_color = diffuse + ambient`
7. Tonemap (ACES Filmic):
   - `mapped = saturate((hdr_color * (2.51 * hdr_color + 0.03)) / (hdr_color * (2.43 * hdr_color + 0.59) + 0.14))`
8. Gamma correction: `output = pow(mapped, 1.0/2.2)`
9. Write to output_image

### Sky Color for Misses

```slang
float3 sky_color_for_dir(float3 dir, SceneUniforms scene) {
    float t = dir.y * 0.5 + 0.5;
    return lerp(scene.ground_color, scene.sky_color, saturate(t));
}
```

### Descriptor Bindings (Lighting Pass)

| Binding | Type | Resource |
|---------|------|----------|
| 0 | ConstantBuffer | Scene UBO (`ConstantBuffer<SceneUniforms>`) |
| 1 | RWTexture2D<float4> | gbuffer_pos (RGBA32F, read-only) |
| 2 | RWTexture2D<float4> | gbuffer0 (RGBA8_UNORM, read-only) |
| 3 | RWTexture2D<uint4> | gbuffer1 (RGBA8_UINT, read-only) |
| 4 | RWTexture2D<float4> | output_image (RGBA8_UNORM, write) |
| 5 | StructuredBuffer<UcvhConfig> | UCVH config |
| 6 | StructuredBuffer<NodeL0> | hierarchy_l0 |
| 7 | StructuredBuffer<BrickOccupancy> | brick_occupancy |
| 8 | StructuredBuffer<VoxelCell> | brick_materials |

### Rust Side: `src/render/passes/lighting.rs`

New file. Structure mirrors `PrimaryRayPass`:
- `LightingPass::new(device, allocator, width, height, spirv, ...)` — creates pipeline, descriptor set, output image
- `LightingPass::record(device, cmd)` — insert input barriers (SHADER_WRITE → SHADER_READ on all 3 G-buffer images) + bind + dispatch. Each pass owns its input barriers, following the existing `PrimaryRayPass::record()` pattern.
- `LightingPass::destroy(device, allocator)` — cleanup

## Render Pipeline in app.rs

### Pass Order

```
1. Upload UCVH (first frame)
2. Update Scene UBO (write to current frame slot)
3. Primary Ray Pass → writes gbuffer_pos, gbuffer0, gbuffer1
4. Lighting Pass → (inserts SHADER_WRITE→SHADER_READ barriers on G-buffer) → reads gbuffer + UCVH, writes output_image
5. Blit to Swapchain → (inserts SHADER_WRITE→TRANSFER_READ barrier on output_image) → copies to swapchain
```

Each pass's `record()` method inserts input barriers for its read resources before binding and dispatching, following the existing `PrimaryRayPass`/`BlitToSwapchain` pattern.

### Render Graph Wiring

```rust
let primary_writes = graph.add_pass("primary_ray", Compute, |builder| {
    // declare gbuffer_pos, gbuffer0, gbuffer1 as outputs
    ...
});

let lighting_writes = graph.add_pass("lighting", Compute, |builder| {
    // depends on ALL primary ray outputs
    builder.read(primary_writes[0]); // gbuffer_pos
    builder.read(primary_writes[1]); // gbuffer0
    builder.read(primary_writes[2]); // gbuffer1
    ...
});

graph.add_pass("blit_to_swapchain", Graphics, |builder| {
    builder.read(lighting_writes[0]); // output_image
    ...
});
```

## Files Changed Summary

| File | Change |
|------|--------|
| `src/render/scene_ubo.rs` | **New** — GpuSceneUniforms struct + SceneUniformBuffer management (double-buffered) |
| `src/render/device.rs` | Enable `shaderStorageImageExtendedFormats` in `VkPhysicalDeviceFeatures` (required for `R8G8B8A8_UINT` as storage image) |
| `src/render/passes/primary_ray.rs` | **Refactor** — remove push constants, add Scene UBO binding, output G-buffer (3 images) |
| `src/render/passes/lighting.rs` | **New** — shadow ray + Lambert + hemisphere ambient + ACES tonemap |
| `src/render/passes/composite.rs` | **Delete** — empty stub, replaced by `lighting.rs` |
| `src/render/passes/shadow_trace.rs` | **Delete** — empty stub, shadow tracing is inlined in `lighting.slang` |
| `src/render/passes/mod.rs` | Replace `pub mod composite;` / `pub mod shadow_trace;` with `pub mod lighting;` |
| `src/render/camera.rs` | Remove `PrimaryRayPushConstants` struct (replaced by GpuSceneUniforms) |
| `src/render/mod.rs` | Add `pub mod scene_ubo;` |
| `src/app.rs` | Create SceneUniformBuffer at init; fill per-frame; wire primary_ray → lighting → blit; remove push constant code; recreate G-buffer images on resize |
| `assets/shaders/shared/scene_common.slang` | **New** — `SceneUniforms` struct, shared by all passes |
| `assets/shaders/passes/primary_ray.slang` | Remove push constant, read Scene UBO via ConstantBuffer, output G-buffer |
| `assets/shaders/passes/lighting.slang` | **New** — full lighting compute shader |
| `assets/shaders/shared/voxel_common.slang` | Add `encode_normal_id()` helper |
| `assets/shaders/shared/lighting_common.slang` | Add ACES tonemap, hemisphere ambient helpers (replace placeholder `lighting.slang`); renamed to avoid collision with `passes/lighting.slang` |

## Implementation Notes

- **L1-L4 hierarchy bindings removed**: The current `primary_ray.slang` binds `hierarchy_l1` through `hierarchy_l4` at bindings 5-8 for forward compatibility. These are intentionally dropped — only L0 is used. The old bindings 5-8 are reclaimed for UCVH buffers.
- **UCVH binding order changed**: Current shader binds UCVH as config(1), occupancy(2), materials(3), l0(4). New order matches `trace_primary_ray()` parameter order: config, l0, occupancy, materials. This affects both passes.
- **`shaderStorageImageExtendedFormats`**: Must be explicitly enabled in `device.rs` (see Files Changed). The existing output image already uses `R8G8B8A8_UNORM` as a storage image, but adding `R8G8B8A8_UINT` for `gbuffer1` may require the feature to be formally enabled.
- **RowMajor matrix layout validation**: After implementing, verify with `spirv-dis` that the `pixel_to_ray` field in the `ConstantBuffer` SPIR-V output has `RowMajor` decoration, matching the existing `PushConstant` behavior. If not, adjust the Rust-side transpose accordingly.

## Out of Scope

- Soft shadows (single hard shadow ray only)
- Point/spot lights (directional only)
- PBR specular (roughness/metallic allocated but unused)
- Material colors / albedo textures (Phase 7)
- Ambient occlusion (Phase 12)
- GI (Phase 5)
