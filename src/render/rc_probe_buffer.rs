use anyhow::{Result, ensure};
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

/// Per-frame-slot double-buffered probe indices for Radiance Cascades.
pub struct ProbeFrameIndices {
    current: Vec<usize>,
}

impl ProbeFrameIndices {
    pub fn new(frame_slot_count: usize) -> Self {
        assert!(
            frame_slot_count > 0,
            "RC probe buffer needs at least one frame slot"
        );
        Self {
            current: vec![0; frame_slot_count],
        }
    }

    pub fn frame_slot_count(&self) -> usize {
        self.current.len()
    }

    pub fn write_index(&self, frame_slot: usize) -> usize {
        self.current[frame_slot]
    }

    pub fn read_index(&self, frame_slot: usize) -> usize {
        1 - self.current[frame_slot]
    }

    pub fn swap_slot(&mut self, frame_slot: usize) {
        self.current[frame_slot] ^= 1;
    }

    fn backing_index(&self, frame_slot: usize, local_index: usize) -> usize {
        frame_slot * 2 + local_index
    }
}

/// Per-frame-slot double-buffered probe storage for Radiance Cascades.
pub struct RcProbeBuffer {
    buffers: Vec<GpuBuffer>,
    indices: ProbeFrameIndices,
}

impl RcProbeBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        frame_slot_count: usize,
    ) -> Result<Self> {
        ensure!(
            frame_slot_count > 0,
            "RC probe buffer needs at least one frame slot"
        );
        let size = RC_TOTAL_ENTRIES as vk::DeviceSize * ENTRY_SIZE;
        let mut buffers: Vec<GpuBuffer> = Vec::with_capacity(frame_slot_count * 2);
        for frame_slot in 0..frame_slot_count {
            for local_index in 0..2 {
                let backing_index = frame_slot * 2 + local_index;
                let buf = match GpuBuffer::new(
                    device,
                    allocator,
                    size,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    MemoryLocation::GpuOnly,
                    &format!("rc_probe_buffer_slot{frame_slot}_{local_index}"),
                ) {
                    Ok(buf) => buf,
                    Err(error) => {
                        for buffer in buffers {
                            buffer.destroy(device, allocator);
                        }
                        return Err(error);
                    }
                };
                debug_assert_eq!(backing_index, buffers.len());
                buffers.push(buf);
            }
        }

        Ok(Self {
            buffers,
            indices: ProbeFrameIndices::new(frame_slot_count),
        })
    }

    pub fn write_buffer(&self, frame_slot: usize) -> vk::Buffer {
        let local_index = self.indices.write_index(frame_slot);
        self.buffers[self.indices.backing_index(frame_slot, local_index)].handle
    }

    pub fn read_buffer(&self, frame_slot: usize) -> vk::Buffer {
        let local_index = self.indices.read_index(frame_slot);
        self.buffers[self.indices.backing_index(frame_slot, local_index)].handle
    }

    pub fn buffer_size(&self) -> vk::DeviceSize {
        self.buffers[0].size
    }

    pub fn swap_slot(&mut self, frame_slot: usize) {
        self.indices.swap_slot(frame_slot);
    }

    pub fn frame_slot_count(&self) -> usize {
        self.indices.frame_slot_count()
    }

    pub fn all_buffer_handles(&self) -> impl Iterator<Item = vk::Buffer> + '_ {
        self.buffers.iter().map(|buffer| buffer.handle)
    }

    pub fn record_clear(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        let size = self.buffer_size();
        unsafe {
            for buffer in &self.buffers {
                device.cmd_fill_buffer(cmd, buffer.handle, 0, size, 0);
            }
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        for buffer in self.buffers {
            buffer.destroy(device, allocator);
        }
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

    #[test]
    fn per_frame_slots_swap_independently() {
        let mut indices = ProbeFrameIndices::new(3);

        assert_eq!(indices.write_index(0), 0);
        assert_eq!(indices.read_index(0), 1);
        assert_eq!(indices.write_index(1), 0);

        indices.swap_slot(0);

        assert_eq!(indices.write_index(0), 1);
        assert_eq!(indices.read_index(0), 0);
        assert_eq!(indices.write_index(1), 0);
        assert_eq!(indices.read_index(1), 1);
    }
}
