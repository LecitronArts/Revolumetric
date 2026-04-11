use anyhow::Result;
use ash::vk;
use std::ffi::CStr;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::pipeline::{create_shader_module, ComputePipeline};
use crate::render::rc_probe_buffer::{self, RcProbeBuffer};

#[repr(C)]
#[derive(Clone, Copy)]
struct RcMergePushConstants {
    cascade_level: u32,
    own_grid_dim: u32,
    own_probe_size: u32,
    own_offset: u32,
    higher_grid_dim: u32,
    higher_probe_size: u32,
    higher_offset: u32,
    _pad: u32,
}

struct MergeParams {
    cascade_level: u32,
    own_grid_dim: u32,
    own_probe_size: u32,
    own_offset: u32,
    higher_grid_dim: u32,
    higher_probe_size: u32,
    higher_offset: u32,
    total_invocations: u32,
}

const MERGE_PARAMS: [MergeParams; 2] = [
    MergeParams {
        cascade_level: 1, own_grid_dim: 8, own_probe_size: 6, own_offset: rc_probe_buffer::RC_C1_OFFSET,
        higher_grid_dim: 4, higher_probe_size: 12, higher_offset: rc_probe_buffer::RC_C2_OFFSET,
        total_invocations: 8*8*8*6*36,
    },
    MergeParams {
        cascade_level: 0, own_grid_dim: 16, own_probe_size: 3, own_offset: rc_probe_buffer::RC_C0_OFFSET,
        higher_grid_dim: 8, higher_probe_size: 6, higher_offset: rc_probe_buffer::RC_C1_OFFSET,
        total_invocations: 16*16*16*6*9,
    },
];

pub struct RcMergePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RcMergePass {
    pub fn new(
        device: &ash::Device,
        spirv_bytes: &[u8],
        rc_probes: &RcProbeBuffer,
        frame_count: usize,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .build(device)?;

        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: frame_count as u32 },
        ];
        let descriptor_pool = DescriptorPool::new(device, frame_count as u32, &pool_sizes)?;
        let layouts: Vec<_> = (0..frame_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = descriptor_pool.allocate(device, &layouts)?;

        for &ds in &descriptor_sets {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<RcMergePushConstants>() as u32);

        let shader_module = create_shader_module(device, spirv_bytes)?;
        let pipeline = ComputePipeline::new(
            device,
            shader_module,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") },
            &[descriptor_set_layout],
            &[push_range],
        )?;
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self { pipeline, descriptor_set_layout, descriptor_pool, descriptor_sets })
    }

    pub fn update_probe_descriptor(&self, device: &ash::Device, rc_probes: &RcProbeBuffer, frame_slot: usize) {
        let ds = self.descriptor_sets[frame_slot];
        let buf_info = vk::DescriptorBufferInfo::default()
            .buffer(rc_probes.write_buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buf_info));
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    pub fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        probe_buffer_handle: vk::Buffer,
    ) {
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

        for (i, params) in MERGE_PARAMS.iter().enumerate() {
            if i > 0 {
                let barrier = vk::BufferMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                    .buffer(probe_buffer_handle)
                    .offset(0)
                    .size(vk::WHOLE_SIZE);
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[], &[barrier], &[],
                    );
                }
            }

            let push = RcMergePushConstants {
                cascade_level: params.cascade_level,
                own_grid_dim: params.own_grid_dim,
                own_probe_size: params.own_probe_size,
                own_offset: params.own_offset,
                higher_grid_dim: params.higher_grid_dim,
                higher_probe_size: params.higher_probe_size,
                higher_offset: params.higher_offset,
                _pad: 0,
            };
            let push_bytes = unsafe {
                std::slice::from_raw_parts(
                    &push as *const RcMergePushConstants as *const u8,
                    std::mem::size_of::<RcMergePushConstants>(),
                )
            };
            unsafe {
                device.cmd_push_constants(cmd, self.pipeline.layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
                device.cmd_dispatch(cmd, (params.total_invocations + 63) / 64, 1, 1);
            }
        }
    }

    pub fn destroy(self, device: &ash::Device, _allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}
