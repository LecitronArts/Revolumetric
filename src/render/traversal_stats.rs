use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TraversalStats {
    pub ray_count: u32,
    pub brick_steps: u32,
    pub hierarchy_rejects: u32,
    pub macrocell_skips: u32,
    pub brick_tests: u32,
    pub brick_dda_steps: u32,
    pub hits: u32,
    pub _pad: u32,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug)]
pub enum TraversalStatsSlot {
    PrimaryRay = 0,
    RcTrace = 1,
    Lighting = 2,
}

pub const TRAVERSAL_STATS_SLOT_COUNT: usize = 3;
pub const TRAVERSAL_STATS_BUFFER_SIZE: usize =
    TRAVERSAL_STATS_SLOT_COUNT * std::mem::size_of::<TraversalStats>();

pub struct TraversalStatsBuffer {
    buffers: Vec<GpuBuffer>,
}

impl TraversalStatsBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        frame_count: usize,
        cpu_readback: bool,
    ) -> Result<Self> {
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let location = if cpu_readback {
            MemoryLocation::GpuToCpu
        } else {
            MemoryLocation::GpuOnly
        };
        let buffers = (0..frame_count)
            .map(|frame_slot| {
                GpuBuffer::new(
                    device,
                    allocator,
                    TRAVERSAL_STATS_BUFFER_SIZE as vk::DeviceSize,
                    usage,
                    location,
                    &format!("traversal_stats_{frame_slot}"),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let stats = Self { buffers };
        for frame_slot in 0..frame_count {
            stats.clear_frame_slot(frame_slot);
        }
        Ok(stats)
    }

    pub fn frame_count(&self) -> usize {
        self.buffers.len()
    }

    pub fn buffer_handle(&self, frame_slot: usize) -> vk::Buffer {
        self.buffers[frame_slot].handle
    }

    pub fn clear_frame_slot(&self, frame_slot: usize) {
        if let Some(ptr) = self.buffers[frame_slot].mapped_ptr() {
            unsafe {
                std::ptr::write_bytes(ptr, 0, TRAVERSAL_STATS_BUFFER_SIZE);
            }
        }
    }

    pub fn record_clear_frame_slot(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
    ) {
        let buffer = &self.buffers[frame_slot];
        unsafe {
            device.cmd_fill_buffer(cmd, buffer.handle, 0, buffer.size, 0);
        }

        let barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .buffer(buffer.handle)
            .size(vk::WHOLE_SIZE);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
    }

    pub fn snapshot_frame_slot(&self, frame_slot: usize) -> Option<[TraversalStats; 3]> {
        self.buffers[frame_slot].mapped_ptr().map(|ptr| unsafe {
            let stats = std::slice::from_raw_parts(
                ptr as *const TraversalStats,
                TRAVERSAL_STATS_SLOT_COUNT,
            );
            [stats[0], stats[1], stats[2]]
        })
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
    fn traversal_stats_abi_matches_shader_layout() {
        assert_eq!(std::mem::size_of::<TraversalStats>(), 32);
        assert_eq!(std::mem::align_of::<TraversalStats>(), 4);
        assert_eq!(std::mem::offset_of!(TraversalStats, ray_count), 0);
        assert_eq!(std::mem::offset_of!(TraversalStats, brick_steps), 4);
        assert_eq!(std::mem::offset_of!(TraversalStats, hierarchy_rejects), 8);
        assert_eq!(std::mem::offset_of!(TraversalStats, macrocell_skips), 12);
        assert_eq!(std::mem::offset_of!(TraversalStats, brick_tests), 16);
        assert_eq!(std::mem::offset_of!(TraversalStats, brick_dda_steps), 20);
        assert_eq!(std::mem::offset_of!(TraversalStats, hits), 24);
    }

    #[test]
    fn traversal_stats_buffer_has_one_counter_block_per_trace_context() {
        assert_eq!(TRAVERSAL_STATS_SLOT_COUNT, 3);
        assert_eq!(TRAVERSAL_STATS_BUFFER_SIZE, 96);
        assert_eq!(TraversalStatsSlot::PrimaryRay as usize, 0);
        assert_eq!(TraversalStatsSlot::RcTrace as usize, 1);
        assert_eq!(TraversalStatsSlot::Lighting as usize, 2);
    }
}
