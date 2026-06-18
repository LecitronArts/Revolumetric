use anyhow::{Result, anyhow};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;

pub const VPT_TRAVERSAL_STATS_COUNTERS: usize = 8;

#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VptTraversalStatCounter {
    PrimaryRays = 0,
    ShadowRays = 1,
    HierarchySkipTests = 2,
    HierarchySkipsAccepted = 3,
    BrickDdaCalls = 4,
    BrickDdaSteps = 5,
    BrickAnyHitCalls = 6,
    BrickAnyHitSteps = 7,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuVptTraversalStats {
    pub counters: [u32; VPT_TRAVERSAL_STATS_COUNTERS],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VptTraversalStatsSnapshot {
    pub primary_rays: u32,
    pub shadow_rays: u32,
    pub hierarchy_skip_tests: u32,
    pub hierarchy_skips_accepted: u32,
    pub brick_dda_calls: u32,
    pub brick_dda_steps: u32,
    pub brick_any_hit_calls: u32,
    pub brick_any_hit_steps: u32,
}

pub struct VptTraversalStatsBuffer {
    buffer: GpuBuffer,
}

impl VptTraversalStatsBuffer {
    pub fn new(device: &ash::Device, allocator: &GpuAllocator) -> Result<Self> {
        let buffer = GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GpuVptTraversalStats>() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuToCpu,
            "vpt_traversal_stats",
        )?;
        let stats = Self { buffer };
        stats.clear_cpu();
        Ok(stats)
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer.handle
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.buffer.size
    }

    pub fn usage(&self) -> vk::BufferUsageFlags {
        self.buffer.usage
    }

    pub fn clear_cpu(&self) {
        let Some(ptr) = self.buffer.mapped_ptr() else {
            return;
        };
        unsafe {
            std::ptr::write_bytes(ptr, 0, std::mem::size_of::<GpuVptTraversalStats>());
        }
    }

    pub fn snapshot(&self) -> Result<VptTraversalStatsSnapshot> {
        let mapped = self
            .buffer
            .mapped_slice()
            .ok_or_else(|| anyhow!("VPT traversal stats buffer is not host-visible"))?;
        if mapped.len() < std::mem::size_of::<GpuVptTraversalStats>() {
            return Err(anyhow!(
                "VPT traversal stats buffer is too small: has {} bytes, needs {} bytes",
                mapped.len(),
                std::mem::size_of::<GpuVptTraversalStats>()
            ));
        }

        let stats = bytemuck::from_bytes::<GpuVptTraversalStats>(
            &mapped[..std::mem::size_of::<GpuVptTraversalStats>()],
        );
        Ok(VptTraversalStatsSnapshot::from_gpu(*stats))
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.buffer.destroy(device, allocator);
    }
}

impl VptTraversalStatsSnapshot {
    pub fn from_gpu(stats: GpuVptTraversalStats) -> Self {
        Self {
            primary_rays: stats.counters[VptTraversalStatCounter::PrimaryRays as usize],
            shadow_rays: stats.counters[VptTraversalStatCounter::ShadowRays as usize],
            hierarchy_skip_tests: stats.counters
                [VptTraversalStatCounter::HierarchySkipTests as usize],
            hierarchy_skips_accepted: stats.counters
                [VptTraversalStatCounter::HierarchySkipsAccepted as usize],
            brick_dda_calls: stats.counters[VptTraversalStatCounter::BrickDdaCalls as usize],
            brick_dda_steps: stats.counters[VptTraversalStatCounter::BrickDdaSteps as usize],
            brick_any_hit_calls: stats.counters[VptTraversalStatCounter::BrickAnyHitCalls as usize],
            brick_any_hit_steps: stats.counters[VptTraversalStatCounter::BrickAnyHitSteps as usize],
        }
    }

    pub fn total_rays(self) -> u32 {
        self.primary_rays.saturating_add(self.shadow_rays)
    }

    pub fn format_log_line(self) -> String {
        format!(
            "TraversalStats: primary_rays={}, shadow_rays={}, hierarchy_skip_tests={}, hierarchy_skips_accepted={}, brick_dda_calls={}, brick_dda_steps={}, brick_any_hit_calls={}, brick_any_hit_steps={}",
            self.primary_rays,
            self.shadow_rays,
            self.hierarchy_skip_tests,
            self.hierarchy_skips_accepted,
            self.brick_dda_calls,
            self.brick_dda_steps,
            self.brick_any_hit_calls,
            self.brick_any_hit_steps
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traversal_stats_gpu_layout_is_stable_for_shader_atomic_counters() {
        assert_eq!(VPT_TRAVERSAL_STATS_COUNTERS, 8);
        assert_eq!(std::mem::size_of::<GpuVptTraversalStats>(), 32);
        assert_eq!(std::mem::align_of::<GpuVptTraversalStats>(), 4);
        assert_eq!(VptTraversalStatCounter::PrimaryRays as usize, 0);
        assert_eq!(VptTraversalStatCounter::ShadowRays as usize, 1);
        assert_eq!(VptTraversalStatCounter::HierarchySkipTests as usize, 2);
        assert_eq!(VptTraversalStatCounter::HierarchySkipsAccepted as usize, 3);
        assert_eq!(VptTraversalStatCounter::BrickDdaCalls as usize, 4);
        assert_eq!(VptTraversalStatCounter::BrickDdaSteps as usize, 5);
        assert_eq!(VptTraversalStatCounter::BrickAnyHitCalls as usize, 6);
        assert_eq!(VptTraversalStatCounter::BrickAnyHitSteps as usize, 7);
    }

    #[test]
    fn traversal_stats_snapshot_names_all_gpu_counters() {
        let mut stats = GpuVptTraversalStats::zeroed();
        for (idx, counter) in stats.counters.iter_mut().enumerate() {
            *counter = idx as u32 + 1;
        }

        let snapshot = VptTraversalStatsSnapshot::from_gpu(stats);

        assert_eq!(snapshot.primary_rays, 1);
        assert_eq!(snapshot.shadow_rays, 2);
        assert_eq!(snapshot.hierarchy_skip_tests, 3);
        assert_eq!(snapshot.hierarchy_skips_accepted, 4);
        assert_eq!(snapshot.brick_dda_calls, 5);
        assert_eq!(snapshot.brick_dda_steps, 6);
        assert_eq!(snapshot.brick_any_hit_calls, 7);
        assert_eq!(snapshot.brick_any_hit_steps, 8);
        assert_eq!(snapshot.total_rays(), 3);
        assert!(snapshot.format_log_line().contains("primary_rays=1"));
        assert!(snapshot.format_log_line().contains("brick_any_hit_steps=8"));
    }

    #[test]
    fn traversal_stats_buffer_uses_host_visible_storage_buffer_for_direct_readback() {
        let source = crate::render::source_checks::read_source("src/render/traversal_stats.rs");

        assert!(source.contains("MemoryLocation::GpuToCpu"));
        assert!(source.contains("vk::BufferUsageFlags::STORAGE_BUFFER"));
        assert!(source.contains("mapped_slice()"));
        assert!(source.contains("vpt_traversal_stats"));
    }
}
