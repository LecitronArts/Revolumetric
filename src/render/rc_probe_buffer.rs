use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

/// Cascade layout constants.
pub const RC_C0_ENTRIES: u32 = 4096 * 6 * 9; // 221,184
pub const RC_C1_ENTRIES: u32 = 512 * 6 * 36; // 110,592
pub const RC_C2_ENTRIES: u32 = 64 * 6 * 144; // 55,296
pub const RC_TOTAL_ENTRIES: u32 = RC_C0_ENTRIES + RC_C1_ENTRIES + RC_C2_ENTRIES; // 387,072

pub const RC_C0_OFFSET: u32 = 0;
pub const RC_C1_OFFSET: u32 = RC_C0_ENTRIES; // 221,184
pub const RC_C2_OFFSET: u32 = RC_C0_ENTRIES + RC_C1_ENTRIES; // 331,776

/// Entry size: float4 = 16 bytes (radiance.rgb + ray_distance).
const ENTRY_SIZE: vk::DeviceSize = 16;

/// Double-buffered probe storage for Radiance Cascades.
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

        Ok(Self {
            buffers: [bufs.remove(0), bufs.remove(0)],
            current: 0,
        })
    }

    pub fn write_buffer(&self) -> vk::Buffer {
        self.buffers[self.current].handle
    }

    pub fn read_buffer(&self) -> vk::Buffer {
        self.buffers[1 - self.current].handle
    }

    pub fn buffer_size(&self) -> vk::DeviceSize {
        self.buffers[0].size
    }

    pub fn swap(&mut self) {
        self.current ^= 1;
    }

    pub fn record_clear(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
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
        assert_eq!(size, 6_193_152);
    }
}
