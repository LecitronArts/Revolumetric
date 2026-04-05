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
