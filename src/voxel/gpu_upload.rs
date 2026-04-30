// src/voxel/gpu_upload.rs
use anyhow::{Result, bail};
use ash::vk;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::voxel::brick::{BRICK_VOLUME, BrickOccupancy, VoxelCell};
use crate::voxel::occupancy::{NodeL0, NodeLN};
use crate::voxel::ucvh::Ucvh;

/// GPU-side config matching the shader UBO.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UcvhGpuConfig {
    pub world_size: [u32; 4],      // xyz + pad
    pub brick_grid_size: [u32; 4], // xyz + pad
    pub brick_capacity: u32,
    pub allocated_bricks: u32,
    pub _pad: [u32; 2],
}

/// All GPU buffers for UCVH data.
pub struct UcvhGpuResources {
    pub config_buffer: GpuBuffer,
    pub occupancy_buffer: GpuBuffer,
    pub material_buffer: GpuBuffer,
    pub hierarchy_l0_buffer: GpuBuffer,
    pub hierarchy_ln_buffers: [GpuBuffer; 4], // L1-L4
    // Staging buffers (host-visible, used for transfer)
    staging_occupancy: GpuBuffer,
    staging_material: GpuBuffer,
    staging_hierarchy: GpuBuffer,
    staging_config: GpuBuffer,
}

impl UcvhGpuResources {
    pub fn new(device: &ash::Device, allocator: &GpuAllocator, ucvh: &Ucvh) -> Result<Self> {
        let cap = ucvh.pool.capacity() as usize;
        let occ_size = cap * std::mem::size_of::<BrickOccupancy>();
        let mat_size = cap * BRICK_VOLUME * std::mem::size_of::<VoxelCell>();

        let ssbo_usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;

        // Device-local SSBOs
        let config_buffer = GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<UcvhGpuConfig>() as u64,
            ssbo_usage,
            MemoryLocation::GpuOnly,
            "ucvh_config",
        )?;
        let occupancy_buffer = GpuBuffer::new(
            device,
            allocator,
            occ_size as u64,
            ssbo_usage,
            MemoryLocation::GpuOnly,
            "ucvh_occupancy",
        )?;
        let material_buffer = GpuBuffer::new(
            device,
            allocator,
            mat_size as u64,
            ssbo_usage,
            MemoryLocation::GpuOnly,
            "ucvh_materials",
        )?;

        // Hierarchy buffers
        let h = &ucvh.hierarchy;
        let l0_size = h.level0.len() * std::mem::size_of::<NodeL0>();
        let ln_sizes: [usize; 4] =
            std::array::from_fn(|i| h.levels[i].len() * std::mem::size_of::<NodeLN>());

        let hierarchy_l0_buffer = GpuBuffer::new(
            device,
            allocator,
            l0_size.max(16) as u64,
            ssbo_usage,
            MemoryLocation::GpuOnly,
            "ucvh_hierarchy_l0",
        )?;
        let hierarchy_ln_buffers = [
            GpuBuffer::new(
                device,
                allocator,
                ln_sizes[0].max(16) as u64,
                ssbo_usage,
                MemoryLocation::GpuOnly,
                "ucvh_hierarchy_l1",
            )?,
            GpuBuffer::new(
                device,
                allocator,
                ln_sizes[1].max(16) as u64,
                ssbo_usage,
                MemoryLocation::GpuOnly,
                "ucvh_hierarchy_l2",
            )?,
            GpuBuffer::new(
                device,
                allocator,
                ln_sizes[2].max(16) as u64,
                ssbo_usage,
                MemoryLocation::GpuOnly,
                "ucvh_hierarchy_l3",
            )?,
            GpuBuffer::new(
                device,
                allocator,
                ln_sizes[3].max(16) as u64,
                ssbo_usage,
                MemoryLocation::GpuOnly,
                "ucvh_hierarchy_l4",
            )?,
        ];

        // Staging buffers (host-visible)
        let total_hierarchy = l0_size + ln_sizes.iter().sum::<usize>();
        let staging_occupancy = GpuBuffer::new(
            device,
            allocator,
            occ_size as u64,
            staging_usage,
            MemoryLocation::CpuToGpu,
            "staging_occupancy",
        )?;
        let staging_material = GpuBuffer::new(
            device,
            allocator,
            mat_size as u64,
            staging_usage,
            MemoryLocation::CpuToGpu,
            "staging_materials",
        )?;
        let staging_hierarchy = GpuBuffer::new(
            device,
            allocator,
            total_hierarchy.max(16) as u64,
            staging_usage,
            MemoryLocation::CpuToGpu,
            "staging_hierarchy",
        )?;
        let staging_config = GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<UcvhGpuConfig>() as u64,
            staging_usage,
            MemoryLocation::CpuToGpu,
            "staging_config",
        )?;

        Ok(Self {
            config_buffer,
            occupancy_buffer,
            material_buffer,
            hierarchy_l0_buffer,
            hierarchy_ln_buffers,
            staging_occupancy,
            staging_material,
            staging_hierarchy,
            staging_config,
        })
    }

    /// Upload all UCVH data to GPU. Call once after scene generation.
    /// Records copy commands into `cmd` — must be called between begin/end command buffer.
    pub fn upload_all(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        ucvh: &Ucvh,
    ) -> Result<()> {
        // Upload config
        let gpu_config = UcvhGpuConfig {
            world_size: [
                ucvh.config.world_size.x,
                ucvh.config.world_size.y,
                ucvh.config.world_size.z,
                0,
            ],
            brick_grid_size: [
                ucvh.config.brick_grid_size.x,
                ucvh.config.brick_grid_size.y,
                ucvh.config.brick_grid_size.z,
                0,
            ],
            brick_capacity: ucvh.pool.capacity(),
            allocated_bricks: ucvh.pool.allocated_count(),
            _pad: [0; 2],
        };
        Self::copy_to_staging(&self.staging_config, bytes_of(&gpu_config))?;

        // Upload occupancy pool
        let occ_bytes = cast_slice::<BrickOccupancy, u8>(ucvh.pool.occupancy_pool());
        Self::copy_to_staging(&self.staging_occupancy, occ_bytes)?;

        // Upload material pool
        let mat_bytes = cast_slice::<VoxelCell, u8>(ucvh.pool.material_pool());
        Self::copy_to_staging(&self.staging_material, mat_bytes)?;

        // Upload hierarchy
        let mut offset = 0u64;
        let l0_bytes = cast_slice::<NodeL0, u8>(&ucvh.hierarchy.level0);
        Self::copy_to_staging_offset(&self.staging_hierarchy, l0_bytes, offset as usize)?;
        offset += l0_bytes.len() as u64;

        let mut ln_offsets = [0u64; 4];
        for (i, ln_offset) in ln_offsets.iter_mut().enumerate() {
            *ln_offset = offset;
            let ln_bytes = cast_slice::<NodeLN, u8>(&ucvh.hierarchy.levels[i]);
            Self::copy_to_staging_offset(&self.staging_hierarchy, ln_bytes, offset as usize)?;
            offset += ln_bytes.len() as u64;
        }

        Self::record_copy(
            device,
            cmd,
            &self.staging_config,
            &self.config_buffer,
            std::mem::size_of::<UcvhGpuConfig>() as u64,
        );
        Self::record_copy(
            device,
            cmd,
            &self.staging_occupancy,
            &self.occupancy_buffer,
            occ_bytes.len() as u64,
        );
        Self::record_copy(
            device,
            cmd,
            &self.staging_material,
            &self.material_buffer,
            mat_bytes.len() as u64,
        );

        // Record copies: staging_hierarchy -> individual device-local buffers
        let l0_size = l0_bytes.len() as u64;
        Self::record_copy_region(
            device,
            cmd,
            &self.staging_hierarchy,
            &self.hierarchy_l0_buffer,
            0,
            0,
            l0_size,
        );
        for (i, ln_offset) in ln_offsets.iter().enumerate() {
            let ln_size = (ucvh.hierarchy.levels[i].len() * std::mem::size_of::<NodeLN>()) as u64;
            Self::record_copy_region(
                device,
                cmd,
                &self.staging_hierarchy,
                &self.hierarchy_ln_buffers[i],
                *ln_offset,
                0,
                ln_size,
            );
        }

        // Buffer memory barrier: ensure transfers complete before shader reads
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        Ok(())
    }

    fn copy_to_staging(buffer: &GpuBuffer, data: &[u8]) -> Result<()> {
        Self::copy_to_staging_offset(buffer, data, 0)
    }

    fn copy_to_staging_offset(buffer: &GpuBuffer, data: &[u8], offset: usize) -> Result<()> {
        Self::copy_to_mapped_staging(buffer.mapped_ptr(), buffer.size, data, offset)
    }

    fn copy_to_mapped_staging(
        mapped_ptr: Option<*mut u8>,
        buffer_size: vk::DeviceSize,
        data: &[u8],
        offset: usize,
    ) -> Result<()> {
        let size = buffer_size as usize;
        let end = match offset.checked_add(data.len()) {
            Some(end) => end,
            None => {
                bail!(
                    "staging copy exceeds staging buffer: offset={} bytes={} buffer_size={}",
                    offset,
                    data.len(),
                    buffer_size
                );
            }
        };
        if offset > size || end > size {
            bail!(
                "staging copy exceeds staging buffer: offset={} bytes={} buffer_size={}",
                offset,
                data.len(),
                buffer_size
            );
        }
        let Some(ptr) = mapped_ptr else {
            bail!(
                "staging buffer not mapped: offset={} bytes={} buffer_size={}",
                offset,
                data.len(),
                buffer_size
            );
        };
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len()) };
        Ok(())
    }

    fn record_copy(
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        size: u64,
    ) {
        Self::record_copy_region(device, cmd, src, dst, 0, 0, size);
    }

    fn record_copy_region(
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        if size == 0 {
            return;
        }
        let region = vk::BufferCopy {
            src_offset,
            dst_offset,
            size,
        };
        unsafe { device.cmd_copy_buffer(cmd, src.handle, dst.handle, &[region]) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.config_buffer.destroy(device, allocator);
        self.occupancy_buffer.destroy(device, allocator);
        self.material_buffer.destroy(device, allocator);
        self.hierarchy_l0_buffer.destroy(device, allocator);
        for buf in self.hierarchy_ln_buffers {
            buf.destroy(device, allocator);
        }
        self.staging_occupancy.destroy(device, allocator);
        self.staging_material.destroy(device, allocator);
        self.staging_hierarchy.destroy(device, allocator);
        self.staging_config.destroy(device, allocator);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unmapped_buffer(size: vk::DeviceSize) -> GpuBuffer {
        GpuBuffer {
            handle: vk::Buffer::null(),
            size,
            allocation: None,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
        }
    }

    #[test]
    fn copy_to_staging_rejects_unmapped_buffer() {
        let buffer = unmapped_buffer(4);

        let err = UcvhGpuResources::copy_to_staging(&buffer, &[1, 2])
            .expect_err("unmapped staging buffer should fail");

        assert!(err.to_string().contains("not mapped"));
    }

    #[test]
    fn copy_to_staging_rejects_data_larger_than_buffer() {
        let buffer = unmapped_buffer(2);

        let err = UcvhGpuResources::copy_to_staging(&buffer, &[1, 2, 3])
            .expect_err("oversized staging copy should fail");

        assert!(err.to_string().contains("exceeds staging buffer"));
    }

    #[test]
    fn copy_to_staging_offset_rejects_write_past_buffer_end() {
        let buffer = unmapped_buffer(4);

        let err = UcvhGpuResources::copy_to_staging_offset(&buffer, &[1, 2], 3)
            .expect_err("offset staging copy past the end should fail");

        assert!(err.to_string().contains("exceeds staging buffer"));
    }

    #[test]
    fn copy_to_staging_offset_rejects_offset_past_buffer_end() {
        let buffer = unmapped_buffer(4);

        let err = UcvhGpuResources::copy_to_staging_offset(&buffer, &[], 5)
            .expect_err("offset past the end should fail");

        assert!(err.to_string().contains("exceeds staging buffer"));
    }

    #[test]
    fn copy_to_mapped_staging_accepts_exact_fit() {
        let mut storage = [0_u8; 4];

        UcvhGpuResources::copy_to_mapped_staging(
            Some(storage.as_mut_ptr()),
            storage.len() as vk::DeviceSize,
            &[1, 2, 3, 4],
            0,
        )
        .expect("exact-fit staging copy should succeed");

        assert_eq!(storage, [1, 2, 3, 4]);
    }

    #[test]
    fn copy_to_mapped_staging_writes_at_offset() {
        let mut storage = [9_u8; 5];

        UcvhGpuResources::copy_to_mapped_staging(
            Some(storage.as_mut_ptr()),
            storage.len() as vk::DeviceSize,
            &[1, 2, 3],
            1,
        )
        .expect("offset staging copy should succeed");

        assert_eq!(storage, [9, 1, 2, 3, 9]);
    }
}
