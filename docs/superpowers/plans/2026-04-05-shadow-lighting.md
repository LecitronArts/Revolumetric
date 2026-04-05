# Phase 4: Shadow & Lighting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flat normal shading with shadow ray tracing + directional lighting + hemisphere ambient + ACES tonemapping via a G-buffer architecture and shared Scene UBO.

**Architecture:** Two compute passes (primary ray → lighting) with G-buffer intermediate. Scene UBO replaces all push constants. Per-frame-slot descriptor sets prevent write-after-read hazards with frames-in-flight.

**Tech Stack:** Rust (ash, gpu-allocator, bytemuck, glam), Slang shaders (→ SPIR-V), Vulkan 1.3

**Spec:** `docs/superpowers/specs/2026-04-05-shadow-lighting-design.md`

---

## File Map

| File | Role | Change Type |
|---|---|---|
| `src/render/scene_ubo.rs` | GpuSceneUniforms struct + SceneUniformBuffer (double-buffered UBO) | **Create** |
| `src/render/mod.rs` | Module declarations | Modify |
| `src/render/device.rs` | Enable `shaderStorageImageExtendedFormats` device feature | Modify |
| `src/render/frame.rs` | Add `frame_slot` field to FrameContext | Modify |
| `src/render/camera.rs` | Remove `PrimaryRayPushConstants` (replaced by GpuSceneUniforms) | Modify |
| `src/render/passes/primary_ray.rs` | Refactor: UBO binding, 3 G-buffer images, per-frame descriptor sets, no push constants | Modify |
| `src/render/passes/lighting.rs` | New lighting pass: shadow ray + Lambert + hemisphere ambient + ACES | **Create** |
| `src/render/passes/composite.rs` | Empty stub — delete | **Delete** |
| `src/render/passes/shadow_trace.rs` | Empty stub — delete | **Delete** |
| `src/render/passes/mod.rs` | Replace composite/shadow_trace with lighting | Modify |
| `src/app.rs` | Create SceneUniformBuffer, fill per-frame, wire primary→lighting→blit, resize handling | Modify |
| `assets/shaders/shared/scene_common.slang` | SceneUniforms struct shared by all passes | **Create** |
| `assets/shaders/shared/voxel_common.slang` | Add `encode_normal_id()` helper | Modify |
| `assets/shaders/shared/lighting.slang` | Rename to `lighting_common.slang` | **Rename** |
| `assets/shaders/shared/lighting_common.slang` | ACES tonemap + sky gradient + hemisphere ambient helpers | **Create** |
| `assets/shaders/passes/primary_ray.slang` | Refactor: ConstantBuffer UBO, G-buffer output, no push constants | Modify |
| `assets/shaders/passes/lighting.slang` | New full lighting compute shader | **Create** |

---

### Task 1: GpuSceneUniforms + SceneUniformBuffer

**Files:**
- Create: `src/render/scene_ubo.rs`
- Modify: `src/render/mod.rs`

- [ ] **Step 1: Write size test for GpuSceneUniforms**

Create `src/render/scene_ubo.rs`:

```rust
use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

/// GPU-side scene uniforms. Must match Slang `SceneUniforms` in scene_common.slang exactly.
/// 144 bytes, std140-compatible (all float3 fields padded to 16-byte alignment).
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_scene_uniforms_size_is_144_bytes() {
        assert_eq!(std::mem::size_of::<GpuSceneUniforms>(), 144);
    }

    #[test]
    fn gpu_scene_uniforms_is_zeroable() {
        let u = GpuSceneUniforms::zeroed();
        assert_eq!(u.resolution, [0, 0]);
        assert_eq!(u.time, 0.0);
    }
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test --lib render::scene_ubo::tests`

Expected: 2 tests PASS.

- [ ] **Step 3: Add SceneUniformBuffer struct**

Append to `src/render/scene_ubo.rs` (before `#[cfg(test)]`):

```rust
/// Manages per-frame-slot uniform buffers for SceneUniforms.
/// One buffer per frame slot to prevent CPU/GPU write-after-read hazards.
pub struct SceneUniformBuffer {
    buffers: Vec<GpuBuffer>,
}

impl SceneUniformBuffer {
    /// Create N uniform buffers (one per frame slot).
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        frame_count: usize,
    ) -> Result<Self> {
        let size = std::mem::size_of::<GpuSceneUniforms>() as vk::DeviceSize;
        let mut buffers = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            let buf = GpuBuffer::new(
                device,
                allocator,
                size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                &format!("scene_ubo_frame_{i}"),
            )?;
            buffers.push(buf);
        }
        Ok(Self { buffers })
    }

    /// Write scene uniforms to the buffer for the given frame slot.
    pub fn update(&self, frame_slot: usize, data: &GpuSceneUniforms) {
        let buf = &self.buffers[frame_slot];
        if let Some(ptr) = buf.mapped_ptr() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data as *const GpuSceneUniforms as *const u8,
                    ptr,
                    std::mem::size_of::<GpuSceneUniforms>(),
                );
            }
        }
    }

    /// Get the VkBuffer handle for a specific frame slot (for descriptor writes).
    pub fn buffer_handle(&self, frame_slot: usize) -> vk::Buffer {
        self.buffers[frame_slot].handle
    }

    /// Number of frame slots.
    pub fn frame_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        for buf in self.buffers {
            buf.destroy(device, allocator);
        }
    }
}
```

- [ ] **Step 4: Register module in render/mod.rs**

Add to `src/render/mod.rs` after `pub mod camera;`:

```rust
pub mod scene_ubo;
```

- [ ] **Step 5: Verify build**

Run: `cargo build`

Expected: Compiles. No behavior change.

- [ ] **Step 6: Run all tests**

Run: `cargo test`

Expected: All tests pass (including new scene_ubo tests).

- [ ] **Step 7: Commit**

```bash
git add src/render/scene_ubo.rs src/render/mod.rs
git commit -m "feat(render): add GpuSceneUniforms struct and SceneUniformBuffer"
```

---

### Task 2: Infrastructure — Device Features + FrameContext

**Files:**
- Modify: `src/render/device.rs`
- Modify: `src/render/frame.rs`

- [ ] **Step 1: Enable shaderStorageImageExtendedFormats in device.rs**

In `src/render/device.rs`, find the device creation section (around line 126-132). Currently:

```rust
let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
    .buffer_device_address(true);

let device_create_info = vk::DeviceCreateInfo::default()
    .queue_create_infos(&queue_create_infos)
    .enabled_extension_names(&device_extension_names)
    .push_next(&mut bda_features);
```

Replace with:

```rust
let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
    .buffer_device_address(true);

let physical_features = vk::PhysicalDeviceFeatures::default()
    .shader_storage_image_extended_formats(true);

let device_create_info = vk::DeviceCreateInfo::default()
    .queue_create_infos(&queue_create_infos)
    .enabled_extension_names(&device_extension_names)
    .enabled_features(&physical_features)
    .push_next(&mut bda_features);
```

- [ ] **Step 2: Add frame_slot to FrameContext**

In `src/render/frame.rs`, add a new field to `FrameContext`:

```rust
pub struct FrameContext {
    pub frame_index: u64,
    pub frame_slot: usize,  // <-- ADD THIS
    pub should_render: bool,
    // ... rest unchanged
}
```

Update `FrameContext::skip()` to include:

```rust
pub fn skip(frame_index: u64) -> Self {
    Self {
        frame_index,
        frame_slot: 0,  // <-- ADD THIS
        should_render: false,
        // ... rest unchanged
    }
}
```

- [ ] **Step 3: Pass frame_slot from begin_frame in device.rs**

In `src/render/device.rs`, in `begin_frame()` (around line 254), the variable `frame_slot` is already computed. Update the `FrameContext` construction (around line 314) to include it:

```rust
Ok(FrameContext {
    frame_index: self.frame_index,
    frame_slot,  // <-- ADD THIS (uses the `frame_slot` variable from line 254)
    should_render: true,
    // ... rest unchanged
})
```

- [ ] **Step 4: Verify build**

Run: `cargo build`

Expected: Compiles. No behavior change — `frame_slot` is set but not yet read.

- [ ] **Step 5: Run all tests**

Run: `cargo test`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/render/device.rs src/render/frame.rs
git commit -m "feat(render): enable shaderStorageImageExtendedFormats; add frame_slot to FrameContext"
```

---

### Task 3: Shader Shared Files

**Files:**
- Create: `assets/shaders/shared/scene_common.slang`
- Modify: `assets/shaders/shared/voxel_common.slang`
- Delete: `assets/shaders/shared/lighting.slang`
- Create: `assets/shaders/shared/lighting_common.slang`

- [ ] **Step 1: Create scene_common.slang**

Create `assets/shaders/shared/scene_common.slang`:

```slang
// Scene-wide uniforms shared by all passes.
// Must match Rust GpuSceneUniforms in src/render/scene_ubo.rs exactly (144 bytes).

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

- [ ] **Step 2: Add encode_normal_id to voxel_common.slang**

Append to the end of `assets/shaders/shared/voxel_common.slang`:

```slang

// Encodes an axis-aligned normal to a 0-5 ID.
// Assumes input is a unit axis-aligned vector from DDA face normals.
// Fallback to -Z (5) for degenerate zero normals (should not occur in practice).
//   0 = +X    1 = -X
//   2 = +Y    3 = -Y
//   4 = +Z    5 = -Z
uint encode_normal_id(float3 n) {
    if (n.x >  0.5) return 0;
    if (n.x < -0.5) return 1;
    if (n.y >  0.5) return 2;
    if (n.y < -0.5) return 3;
    if (n.z >  0.5) return 4;
    return 5; // -Z
}
```

- [ ] **Step 3: Delete lighting.slang and create lighting_common.slang**

Delete `assets/shaders/shared/lighting.slang`.

Create `assets/shaders/shared/lighting_common.slang`:

```slang
// Lighting helper functions shared across passes.
// Includes: ACES tonemap, sky gradient, hemisphere ambient.

#include "scene_common.slang"

// ACES Filmic Tonemapping (Narkowicz 2015 approximation).
// Input: linear HDR color. Output: [0, 1] LDR color.
float3 aces_tonemap(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

// Sky color gradient based on ray direction.
// Interpolates between ground_color (horizon-down) and sky_color (zenith).
float3 sky_color_for_dir(float3 dir, SceneUniforms scene) {
    float t = dir.y * 0.5 + 0.5;
    return lerp(scene.ground_color, scene.sky_color, saturate(t));
}

// Hemisphere ambient: blend sky/ground based on surface normal Y component.
float3 hemisphere_ambient(float3 normal, float3 sky, float3 ground) {
    float t = normal.y * 0.5 + 0.5;
    return lerp(ground, sky, t);
}

// Normal ID lookup table — reconstructs float3 normal from 0-5 ID.
static const float3 NORMAL_TABLE[6] = {
    float3(1,0,0), float3(-1,0,0),
    float3(0,1,0), float3(0,-1,0),
    float3(0,0,1), float3(0,0,-1),
};
```

- [ ] **Step 4: Verify shader compilation**

Run: `cargo build`

Expected: Compiles. `build.rs` compiles `passes/primary_ray.slang` which includes `voxel_common.slang` — the new `encode_normal_id` function should not cause errors (it's added but not yet called). The new shared files are not included by any pass yet, so they won't be compiled directly.

- [ ] **Step 5: Commit**

```bash
git add assets/shaders/shared/scene_common.slang assets/shaders/shared/voxel_common.slang assets/shaders/shared/lighting_common.slang
git rm assets/shaders/shared/lighting.slang
git commit -m "feat(shaders): add scene_common, encode_normal_id, lighting_common helpers"
```

---

### Task 4: Primary Ray Pass Refactor (Shader + Rust + App Integration)

This is the largest task — the shader, Rust pass, camera.rs, and app.rs must all change atomically because the shader expects UBO while the old code sends push constants.

**Files:**
- Modify: `assets/shaders/passes/primary_ray.slang`
- Modify: `src/render/passes/primary_ray.rs`
- Modify: `src/render/camera.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Rewrite primary_ray.slang**

Replace the entire contents of `assets/shaders/passes/primary_ray.slang` with:

```slang
// Primary ray compute shader — screen-space voxel ray tracing.
// Traces one ray per pixel through the UCVH brick grid.
// Output: G-buffer (position, albedo, normal+flags) for deferred lighting.

#include "scene_common.slang"
#include "voxel_traverse.slang"

// Binding 0: Scene UBO (ConstantBuffer — maps to VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

// Binding 1-3: G-buffer outputs
[[vk::binding(1, 0)]]
RWTexture2D<float4> gbuffer_pos;    // RGBA32F: xyz = voxel center, w = hit_t (-1 = miss)

[[vk::binding(2, 0)]]
RWTexture2D<float4> gbuffer0;       // RGBA8_UNORM: rgb = base_color, a = ao

[[vk::binding(3, 0)]]
RWTexture2D<uint4> gbuffer1;        // RGBA8_UINT: r = normal_id, g = roughness, b = metallic, a = flags

// Binding 4-7: UCVH buffers (order matches trace_primary_ray parameter order)
[[vk::binding(4, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

[[vk::binding(5, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(6, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(7, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    // Read scene uniforms (ConstantBuffer — no array index)
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    // Generate ray from pixel_to_ray matrix (same math as before).
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

- [ ] **Step 2: Rewrite primary_ray.rs**

Replace the entire contents of `src/render/passes/primary_ray.rs` with:

```rust
use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{create_shader_module, ComputePipeline};
use crate::render::scene_ubo::SceneUniformBuffer;
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct PrimaryRayPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub gbuffer_pos: GpuImage,
    pub gbuffer0: GpuImage,
    pub gbuffer1: GpuImage,
}

impl PrimaryRayPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        spirv_bytes: &[u8],
        ucvh_gpu: &UcvhGpuResources,
        scene_ubo: &SceneUniformBuffer,
    ) -> Result<Self> {
        // Descriptor set layout: 1 UBO + 3 storage images + 4 storage buffers
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(1, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(2, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(3, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(4, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(5, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(6, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(7, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: frame_count as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 3 * frame_count as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 4 * frame_count as u32 },
        ];
        let descriptor_pool = DescriptorPool::new(device, frame_count as u32, &pool_sizes)?;
        let layouts: Vec<_> = (0..frame_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = descriptor_pool.allocate(device, &layouts)?;

        // G-buffer images
        let gbuffer_pos = GpuImage::new(device, allocator, &GpuImageDesc {
            width, height, depth: 1,
            format: vk::Format::R32G32B32A32_SFLOAT,
            usage: vk::ImageUsageFlags::STORAGE,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "gbuffer_pos",
        })?;

        let gbuffer0 = GpuImage::new(device, allocator, &GpuImageDesc {
            width, height, depth: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::STORAGE,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "gbuffer0",
        })?;

        let gbuffer1 = GpuImage::new(device, allocator, &GpuImageDesc {
            width, height, depth: 1,
            format: vk::Format::R8G8B8A8_UINT,
            usage: vk::ImageUsageFlags::STORAGE,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "gbuffer1",
        })?;

        // Write descriptor sets (one per frame slot)
        // UCVH buffer handles: config, l0, occupancy, materials (matches new binding order)
        let ucvh_buffers = [
            &ucvh_gpu.config_buffer,
            &ucvh_gpu.hierarchy_l0_buffer,
            &ucvh_gpu.occupancy_buffer,
            &ucvh_gpu.material_buffer,
        ];

        for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
            // Binding 0: UBO (different buffer per frame slot)
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(scene_ubo.buffer_handle(set_idx))
                .offset(0)
                .range(144); // sizeof(GpuSceneUniforms)

            let ubo_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info));

            // Bindings 1-3: G-buffer images (same for all frame slots)
            let image_infos = [
                vk::DescriptorImageInfo::default()
                    .image_view(gbuffer_pos.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(gbuffer0.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(gbuffer1.view)
                    .image_layout(vk::ImageLayout::GENERAL),
            ];

            let image_writes: Vec<vk::WriteDescriptorSet> = image_infos.iter().enumerate().map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding((i + 1) as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(info))
            }).collect();

            // Bindings 4-7: UCVH buffers (same for all frame slots)
            let buffer_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers.iter().map(|buf| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buf.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            }).collect();

            let buffer_writes: Vec<vk::WriteDescriptorSet> = buffer_infos.iter().enumerate().map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding((i + 4) as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            }).collect();

            let mut all_writes = vec![ubo_write];
            all_writes.extend(image_writes);
            all_writes.extend(buffer_writes);
            unsafe { device.update_descriptor_sets(&all_writes, &[]) };
        }

        // Pipeline (no push constant ranges)
        let shader_module = create_shader_module(device, spirv_bytes)?;
        let pipeline = ComputePipeline::new(
            device,
            shader_module,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") },
            &[descriptor_set_layout],
            &[], // no push constants
        )?;
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            gbuffer_pos,
            gbuffer0,
            gbuffer1,
        })
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let extent = self.gbuffer_pos.extent;

        // Transition all 3 G-buffer images to GENERAL for compute write
        let barriers: Vec<vk::ImageMemoryBarrier> = [
            self.gbuffer_pos.handle,
            self.gbuffer0.handle,
            self.gbuffer1.handle,
        ].iter().map(|&image| {
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                )
        }).collect();

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], &barriers,
            );
        }

        // Bind pipeline and per-frame descriptor set
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

        // Dispatch (8x8 workgroups)
        let groups_x = (extent.width + 7) / 8;
        let groups_y = (extent.height + 7) / 8;
        unsafe { device.cmd_dispatch(cmd, groups_x, groups_y, 1) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.gbuffer_pos.destroy(device, allocator);
        self.gbuffer0.destroy(device, allocator);
        self.gbuffer1.destroy(device, allocator);
    }
}
```

- [ ] **Step 3: Update camera.rs — remove PrimaryRayPushConstants**

In `src/render/camera.rs`:

1. Remove the `PrimaryRayPushConstants` struct (lines 5-19) and its `use bytemuck::{Pod, Zeroable};` import.
2. Remove the `push_constants_size` test (lines 108-110).
3. Keep `compute_pixel_to_ray` function and its other tests intact.

The file should start:

```rust
// src/render/camera.rs
use glam::{Mat4, Vec3, Vec4};

/// Compute the pixel_to_ray matrix for a pinhole camera.
// ... rest of compute_pixel_to_ray unchanged ...
```

- [ ] **Step 4: Update app.rs — SceneUniformBuffer creation and per-frame fill**

In `src/app.rs`:

**4a. Update imports** — replace the `PrimaryRayPushConstants` import:

Replace:
```rust
use crate::render::camera::{compute_pixel_to_ray, PrimaryRayPushConstants};
```
With:
```rust
use crate::render::camera::compute_pixel_to_ray;
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::scene::light::DirectionalLight;
```

**4b. Add scene_ubo field to RevolumetricApp struct:**

After `ucvh_uploaded: bool,` add:
```rust
scene_ubo: Option<SceneUniformBuffer>,
```

Initialize in `RevolumetricApp::new()`:
```rust
scene_ubo: None,
```

**4c. Create SceneUniformBuffer in `resumed()`** — after the renderer is created (after `self.renderer = Some(renderer);` around line 307), add:

```rust
// Create Scene UBO (one buffer per frame slot)
if self.scene_ubo.is_none() {
    let renderer = self.renderer.as_ref().unwrap();
    match SceneUniformBuffer::new(
        renderer.device(),
        renderer.allocator(),
        renderer.swapchain_image_count(),
    ) {
        Ok(ubo) => {
            tracing::info!(frame_count = renderer.swapchain_image_count(), "created scene UBO");
            self.scene_ubo = Some(ubo);
        }
        Err(e) => tracing::error!(%e, "failed to create scene UBO"),
    }
}
```

**4d. Update PrimaryRayPass creation** to pass scene_ubo — change the match block (around line 343):

Replace:
```rust
match PrimaryRayPass::new(
    renderer.device(),
    renderer.allocator(),
    extent.width,
    extent.height,
    spirv,
    ucvh_gpu,
) {
```
With:
```rust
let scene_ubo_ref = self.scene_ubo.as_ref().unwrap();
match PrimaryRayPass::new(
    renderer.device(),
    renderer.allocator(),
    extent.width,
    extent.height,
    spirv,
    ucvh_gpu,
    scene_ubo_ref,
) {
```

Wrap the `if let Some(ucvh_gpu)` check to also require scene_ubo:
```rust
if self.primary_ray_pass.is_none() {
    if let (Some(ucvh_gpu), Some(scene_ubo_ref)) = (&self.ucvh_gpu, &self.scene_ubo) {
        // ... existing code, but pass scene_ubo_ref to PrimaryRayPass::new
    }
}
```

**4e. Replace push constant fill with Scene UBO fill in tick_frame** — replace the entire block inside `if let Some(pass) = &self.primary_ray_pass {`:

```rust
if let Some(pass) = &self.primary_ray_pass {
    let (cam_pos, cam_forward, cam_up, fov_y) = {
        let rig = self.world.resource::<CameraRig>();
        match rig {
            Some(rig) => (
                rig.camera.position,
                rig.camera.forward,
                rig.camera.up,
                rig.camera.fov_y_radians,
            ),
            None => (
                glam::Vec3::new(64.0, 80.0, -40.0),
                glam::Vec3::Z,
                glam::Vec3::Y,
                std::f32::consts::FRAC_PI_4,
            ),
        }
    };

    let pixel_to_ray = compute_pixel_to_ray(
        cam_pos, cam_forward, cam_up, fov_y,
        frame.swapchain_extent.width, frame.swapchain_extent.height,
    );

    // Read DirectionalLight from World
    let (sun_dir, sun_intensity) = {
        let light = self.world.resource::<DirectionalLight>();
        match light {
            Some(l) => (l.direction, l.intensity),
            None => (
                glam::Vec3::new(0.5, 1.0, 0.25).normalize(),
                glam::Vec3::new(2.0, 1.5, 1.25),
            ),
        }
    };

    // Fill and upload Scene UBO
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
        time: self.world.resource::<Time>().map_or(0.0, |t| t.elapsed()),
    };

    if let Some(ubo) = &self.scene_ubo {
        ubo.update(frame.frame_slot, &scene_data);
    }

    let gbuffer_pos_img = pass.gbuffer_pos.handle;
    let gbuffer_pos_extent = pass.gbuffer_pos.extent;

    let primary_ray_writes = graph.add_pass(
        "primary_ray",
        QueueType::Compute,
        |builder| {
            let _gbp = builder.create_image(
                frame.swapchain_extent.width,
                frame.swapchain_extent.height,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageUsageFlags::STORAGE,
            );
            let _gb0 = builder.create_image(
                frame.swapchain_extent.width,
                frame.swapchain_extent.height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE,
            );
            let _gb1 = builder.create_image(
                frame.swapchain_extent.width,
                frame.swapchain_extent.height,
                vk::Format::R8G8B8A8_UINT,
                vk::ImageUsageFlags::STORAGE,
            );
            let slot = frame.frame_slot;
            Box::new(move |ctx| {
                pass.record(ctx.device, ctx.command_buffer, slot);
            })
        },
    );

    // For now, blit gbuffer_pos directly (lighting pass added in Task 5).
    // This will show raw position data, confirming G-buffer pipeline works.
    // We'll replace this blit source with the lighting pass output in Task 5.
    let src_image = gbuffer_pos_img;
    let src_extent = gbuffer_pos_extent;
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

Note: `gbuffer_pos` is `R32G32B32A32_SFLOAT` — blitting it directly to the RGBA8 swapchain will produce garbled colors (position values are > 1.0). This is expected as a temporary state. The lighting pass in Task 5 will produce the correct RGBA8 output.

**4f. Update Drop impl** to destroy scene_ubo:

In the `Drop for RevolumetricApp` impl, after destroying `ucvh_gpu`, add:
```rust
if let Some(ubo) = self.scene_ubo.take() {
    ubo.destroy(renderer.device(), renderer.allocator());
}
```

- [ ] **Step 5: Check Time::elapsed() exists**

The `Time` resource needs an `elapsed()` method. Verify it exists. If not, use `0.0` as a placeholder for `time` and add a TODO comment.

Run: `grep -n "pub fn elapsed" src/platform/time.rs` — if it doesn't exist, replace the time line with:
```rust
time: 0.0, // TODO: Time::elapsed() when available
```

- [ ] **Step 6: Verify build + shaders compile**

Run: `cargo build`

Expected: Compiles. The window will show garbled colors (raw position data blitted to swapchain) — this is expected until the lighting pass is added.

- [ ] **Step 7: Run all tests**

Run: `cargo test`

Expected: All tests pass. The `push_constants_size` test should be gone.

- [ ] **Step 8: Commit**

```bash
git add src/render/passes/primary_ray.rs src/render/camera.rs src/app.rs assets/shaders/passes/primary_ray.slang
git commit -m "feat(render): refactor primary ray pass to Scene UBO + G-buffer output"
```

---

### Task 5: Lighting Pass (Shader + Rust + App Wiring)

**Files:**
- Create: `assets/shaders/passes/lighting.slang`
- Create: `src/render/passes/lighting.rs`
- Modify: `src/render/passes/mod.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Create lighting.slang**

Create `assets/shaders/passes/lighting.slang`:

```slang
// Deferred lighting compute shader.
// Reads G-buffer, traces shadow rays, computes Lambert + hemisphere ambient + ACES tonemap.

#include "scene_common.slang"
#include "voxel_traverse.slang"
#include "lighting_common.slang"

// Binding 0: Scene UBO
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

// Binding 1-3: G-buffer inputs (read-only via RWTexture2D — no sampler needed)
[[vk::binding(1, 0)]]
RWTexture2D<float4> gbuffer_pos;

[[vk::binding(2, 0)]]
RWTexture2D<float4> gbuffer0;

[[vk::binding(3, 0)]]
RWTexture2D<uint4> gbuffer1;

// Binding 4: output image
[[vk::binding(4, 0)]]
RWTexture2D<float4> output_image;

// Binding 5-8: UCVH buffers (for shadow ray tracing)
[[vk::binding(5, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

[[vk::binding(6, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(7, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(8, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    if (tid.x >= scene.resolution.x || tid.y >= scene.resolution.y) return;

    // Read G-buffer
    float4 pos_data = gbuffer_pos[tid.xy];

    // Miss check: w < 0 means no geometry hit
    if (pos_data.w < 0.0) {
        // Reconstruct ray direction for sky gradient
        float2 pixel = float2(tid.xy);
        float3x3 dir_mat = float3x3(
            scene.pixel_to_ray[0].xyz,
            scene.pixel_to_ray[1].xyz,
            scene.pixel_to_ray[2].xyz
        );
        float3 ray_dir = normalize(mul(float3(pixel, 1.0), dir_mat));
        float3 sky = sky_color_for_dir(ray_dir, scene);
        // Apply tonemap + gamma to sky for consistency
        sky = aces_tonemap(sky);
        sky = pow(sky, float3(1.0 / 2.2));
        output_image[tid.xy] = float4(sky, 1.0);
        return;
    }

    // Read surface data
    float3 position = pos_data.xyz;
    float3 base_color = gbuffer0[tid.xy].rgb;
    uint4 gb1 = gbuffer1[tid.xy];
    uint normal_id = gb1.r;
    float3 normal = NORMAL_TABLE[min(normal_id, 5u)];

    // Shadow ray: offset from voxel center along normal to exit the solid voxel
    float3 shadow_origin = position + normal * 0.51;
    Ray shadow_ray = make_ray(shadow_origin, scene.sun_direction);
    HitResult shadow_hit = trace_primary_ray(
        shadow_ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials
    );

    // Lighting
    float shadow = shadow_hit.hit ? 0.0 : 1.0;
    float ndotl = max(dot(normal, scene.sun_direction), 0.0);
    float3 diffuse = base_color * scene.sun_intensity * ndotl * shadow;
    float3 ambient = base_color * hemisphere_ambient(normal, scene.sky_color, scene.ground_color);
    float3 hdr_color = diffuse + ambient;

    // Tonemap (ACES Filmic) + gamma correction
    float3 mapped = aces_tonemap(hdr_color);
    float3 ldr = pow(mapped, float3(1.0 / 2.2));

    output_image[tid.xy] = float4(ldr, 1.0);
}
```

- [ ] **Step 2: Create lighting.rs**

Create `src/render/passes/lighting.rs`:

```rust
use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::primary_ray::PrimaryRayPass;
use crate::render::pipeline::{create_shader_module, ComputePipeline};
use crate::render::scene_ubo::SceneUniformBuffer;
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct LightingPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_image: GpuImage,
}

impl LightingPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        spirv_bytes: &[u8],
        primary_ray: &PrimaryRayPass,
        ucvh_gpu: &UcvhGpuResources,
        scene_ubo: &SceneUniformBuffer,
    ) -> Result<Self> {
        // Descriptor layout: 1 UBO + 3 G-buffer (storage image) + 1 output (storage image) + 4 UCVH (SSBO)
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(1, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(2, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(3, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(4, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(5, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(6, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(7, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(8, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: frame_count as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 4 * frame_count as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 4 * frame_count as u32 },
        ];
        let descriptor_pool = DescriptorPool::new(device, frame_count as u32, &pool_sizes)?;
        let layouts: Vec<_> = (0..frame_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = descriptor_pool.allocate(device, &layouts)?;

        // Output image (same size as G-buffer, RGBA8 for final color)
        let output_image = GpuImage::new(device, allocator, &GpuImageDesc {
            width, height, depth: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "lighting_output",
        })?;

        // UCVH buffers: config, l0, occupancy, materials
        let ucvh_buffers = [
            &ucvh_gpu.config_buffer,
            &ucvh_gpu.hierarchy_l0_buffer,
            &ucvh_gpu.occupancy_buffer,
            &ucvh_gpu.material_buffer,
        ];

        for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
            // Binding 0: UBO
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(scene_ubo.buffer_handle(set_idx))
                .offset(0)
                .range(144);

            let ubo_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info));

            // Bindings 1-3: G-buffer inputs + Binding 4: output
            let image_infos = [
                vk::DescriptorImageInfo::default()
                    .image_view(primary_ray.gbuffer_pos.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(primary_ray.gbuffer0.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(primary_ray.gbuffer1.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(output_image.view)
                    .image_layout(vk::ImageLayout::GENERAL),
            ];

            let image_writes: Vec<vk::WriteDescriptorSet> = image_infos.iter().enumerate().map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding((i + 1) as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(info))
            }).collect();

            // Bindings 5-8: UCVH buffers
            let buffer_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers.iter().map(|buf| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buf.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            }).collect();

            let buffer_writes: Vec<vk::WriteDescriptorSet> = buffer_infos.iter().enumerate().map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding((i + 5) as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            }).collect();

            let mut all_writes = vec![ubo_write];
            all_writes.extend(image_writes);
            all_writes.extend(buffer_writes);
            unsafe { device.update_descriptor_sets(&all_writes, &[]) };
        }

        // Pipeline (no push constants)
        let shader_module = create_shader_module(device, spirv_bytes)?;
        let pipeline = ComputePipeline::new(
            device,
            shader_module,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") },
            &[descriptor_set_layout],
            &[],
        )?;
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            output_image,
        })
    }

    /// Record the lighting pass. Inserts input barriers on G-buffer images before dispatch.
    pub fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        gbuffer_images: [vk::Image; 3], // [gbuffer_pos, gbuffer0, gbuffer1]
    ) {
        let extent = self.output_image.extent;

        // Barrier: G-buffer SHADER_WRITE → SHADER_READ + output to GENERAL
        let mut barriers: Vec<vk::ImageMemoryBarrier> = gbuffer_images.iter().map(|&image| {
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                )
        }).collect();

        // Output image to GENERAL for compute write
        barriers.push(
            vk::ImageMemoryBarrier::default()
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
                )
        );

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], &barriers,
            );
        }

        // Bind pipeline and per-frame descriptor set
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

- [ ] **Step 3: Add lighting module to passes/mod.rs**

In `src/render/passes/mod.rs`, add:

```rust
pub mod lighting;
```

- [ ] **Step 4: Wire lighting pass into app.rs**

**4a. Add lighting_pass field** to `RevolumetricApp` struct:

After `primary_ray_pass: Option<PrimaryRayPass>,` add:
```rust
lighting_pass: Option<LightingPass>,
```

Initialize in `new()`:
```rust
lighting_pass: None,
```

Add import at top of app.rs:
```rust
use crate::render::passes::lighting::LightingPass;
```

**4b. Create lighting pass in `resumed()`** — after the primary_ray_pass creation block, add:

```rust
// Initialize lighting pass (requires primary ray pass + scene UBO)
if self.lighting_pass.is_none() {
    if let (Some(primary), Some(ucvh_gpu), Some(scene_ubo_ref)) =
        (&self.primary_ray_pass, &self.ucvh_gpu, &self.scene_ubo)
    {
        let renderer = self.renderer.as_ref().unwrap();
        let extent = renderer.swapchain_extent();
        let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/lighting.spv"));
        if spirv.is_empty() {
            tracing::warn!("lighting.spv is empty — slangc may not be installed");
        } else {
            match LightingPass::new(
                renderer.device(),
                renderer.allocator(),
                extent.width,
                extent.height,
                spirv,
                primary,
                ucvh_gpu,
                scene_ubo_ref,
            ) {
                Ok(pass) => {
                    tracing::info!(
                        width = extent.width,
                        height = extent.height,
                        "initialized lighting pass"
                    );
                    self.lighting_pass = Some(pass);
                }
                Err(error) => {
                    tracing::error!(%error, "failed to create lighting pass");
                }
            }
        }
    }
}
```

**4c. Update tick_frame render graph** — replace the blit section (the temporary raw G-buffer blit from Task 4) with the full pipeline:

Replace the blit section in tick_frame (after the primary_ray_writes block) with:

```rust
// Lighting pass
if let Some(lighting) = &self.lighting_pass {
    let gbuf_images = [
        pass.gbuffer_pos.handle,
        pass.gbuffer0.handle,
        pass.gbuffer1.handle,
    ];
    let lighting_output = lighting.output_image.handle;
    let lighting_extent = lighting.output_image.extent;
    let dep0 = primary_ray_writes[0];
    let dep1 = primary_ray_writes[1];
    let dep2 = primary_ray_writes[2];
    let slot = frame.frame_slot;

    let lighting_writes = graph.add_pass(
        "lighting",
        QueueType::Compute,
        |builder| {
            builder.read(dep0); // gbuffer_pos
            builder.read(dep1); // gbuffer0
            builder.read(dep2); // gbuffer1
            let _out = builder.create_image(
                frame.swapchain_extent.width,
                frame.swapchain_extent.height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            Box::new(move |ctx| {
                lighting.record(ctx.device, ctx.command_buffer, slot, gbuf_images);
            })
        },
    );

    // Blit lighting output to swapchain
    let src_image = lighting_output;
    let src_extent = lighting_extent;
    let dst_image = frame.swapchain_image;
    let dst_extent = frame.swapchain_extent;
    let dep_handle = lighting_writes[0];
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
} else {
    // Fallback: blit raw G-buffer if lighting pass not ready
    let src_image = pass.gbuffer0.handle;
    let src_extent = pass.gbuffer0.extent;
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

**4d. Update Drop impl** — destroy lighting pass before primary ray pass:

```rust
if let Some(pass) = self.lighting_pass.take() {
    pass.destroy(renderer.device(), renderer.allocator());
}
```

Add this line BEFORE `if let Some(pass) = self.primary_ray_pass.take()`.

- [ ] **Step 5: Verify build + shaders compile**

Run: `cargo build`

Expected: Compiles. Both `primary_ray.spv` and `lighting.spv` should be generated by build.rs.

- [ ] **Step 6: Run all tests**

Run: `cargo test`

Expected: All tests pass.

- [ ] **Step 7: Manual test**

Run: `cargo run`

Expected:
1. Window opens with the sphere scene
2. Sphere has directional lighting with shadows
3. Sky has gradient from ground color to sky color
4. WASD + right-click mouse look still works
5. Shadow appears on the ground plane behind the sphere (opposite to sun direction)

- [ ] **Step 8: Commit**

```bash
git add assets/shaders/passes/lighting.slang src/render/passes/lighting.rs src/render/passes/mod.rs src/app.rs
git commit -m "feat(render): add deferred lighting pass with shadow rays, Lambert + hemisphere ambient, ACES tonemap"
```

---

### Task 6: Cleanup — Remove Stubs + Final Polish

**Files:**
- Delete: `src/render/passes/composite.rs`
- Delete: `src/render/passes/shadow_trace.rs`
- Modify: `src/render/passes/mod.rs`

- [ ] **Step 1: Delete empty stubs**

Delete `src/render/passes/composite.rs` and `src/render/passes/shadow_trace.rs`.

- [ ] **Step 2: Update passes/mod.rs**

Remove these lines from `src/render/passes/mod.rs`:
```rust
pub mod composite;
pub mod shadow_trace;
```

The file should now contain:
```rust
pub mod blit_to_swapchain;
pub mod debug_views;
pub mod lighting;
pub mod primary_ray;
pub mod radiance_cascade_merge;
pub mod radiance_cascade_trace;
pub mod test_pattern;
pub mod voxel_upload;
```

- [ ] **Step 3: Check for dead code warnings**

Run: `cargo build 2>&1 | grep warning`

Fix any unused import warnings (e.g., if `PrimaryRayPushConstants` is still imported somewhere).

- [ ] **Step 4: Run full test suite**

Run: `cargo test`

Expected: All tests pass.

- [ ] **Step 5: Final manual test**

Run: `cargo run`

Verify same visual result as Task 5 Step 7 — lighting, shadows, sky gradient, camera controls all work.

- [ ] **Step 6: Commit**

```bash
git rm src/render/passes/composite.rs src/render/passes/shadow_trace.rs
git add src/render/passes/mod.rs
git commit -m "chore: remove empty composite and shadow_trace pass stubs"
```

