use anyhow::Result;
use ash::vk;
use std::ffi::CStr;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::pipeline::{create_shader_module, ComputePipeline};
use crate::render::rc_probe_buffer::{self, RcProbeBuffer};
use crate::render::scene_ubo::SceneUniformBuffer;
use crate::voxel::gpu_upload::UcvhGpuResources;

#[repr(C)]
#[derive(Clone, Copy)]
struct RcTracePushConstants {
    cascade_level: u32,
    probe_grid_dim: u32,
    probe_size: u32,
    buffer_offset: u32,
}

struct CascadeParams {
    level: u32,
    grid_dim: u32,
    probe_size: u32,
    offset: u32,
    total_invocations: u32,
}

const CASCADE_PARAMS: [CascadeParams; 3] = [
    CascadeParams {
        level: 0,
        grid_dim: 16,
        probe_size: 3,
        offset: rc_probe_buffer::RC_C0_OFFSET,
        total_invocations: 16 * 16 * 16 * 6 * 9,
    },
    CascadeParams {
        level: 1,
        grid_dim: 8,
        probe_size: 6,
        offset: rc_probe_buffer::RC_C1_OFFSET,
        total_invocations: 8 * 8 * 8 * 6 * 36,
    },
    CascadeParams {
        level: 2,
        grid_dim: 4,
        probe_size: 12,
        offset: rc_probe_buffer::RC_C2_OFFSET,
        total_invocations: 4 * 4 * 4 * 6 * 144,
    },
];

pub struct RcTracePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RcTracePass {
    pub fn new(
        device: &ash::Device,
        _allocator: &GpuAllocator,
        spirv_bytes: &[u8],
        ucvh_gpu: &UcvhGpuResources,
        scene_ubo: &SceneUniformBuffer,
        rc_probes: &RcProbeBuffer,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                1,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                2,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                3,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                4,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                5,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                6,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 6 * frame_count as u32,
            },
        ];
        let descriptor_pool = DescriptorPool::new(device, frame_count as u32, &pool_sizes)?;
        let layouts: Vec<_> = (0..frame_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = descriptor_pool.allocate(device, &layouts)?;

        let ucvh_buffers = [
            &ucvh_gpu.config_buffer,
            &ucvh_gpu.hierarchy_l0_buffer,
            &ucvh_gpu.occupancy_buffer,
            &ucvh_gpu.material_buffer,
        ];

        for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(scene_ubo.buffer_handle(set_idx))
                .offset(0)
                .range(176);

            let ubo_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info));

            let ucvh_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers
                .iter()
                .map(|buf| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.handle)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();

            let ucvh_writes: Vec<vk::WriteDescriptorSet> = ucvh_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding((i + 1) as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            let read_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.read_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let read_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&read_info));

            let write_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&write_info));

            let mut all_writes = vec![ubo_write];
            all_writes.extend(ucvh_writes);
            all_writes.push(read_write);
            all_writes.push(write_write);
            unsafe { device.update_descriptor_sets(&all_writes, &[]) };
        }

        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<RcTracePushConstants>() as u32);

        let shader_module = create_shader_module(device, spirv_bytes)?;
        let pipeline = ComputePipeline::new(
            device,
            shader_module,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") },
            &[descriptor_set_layout],
            &[push_range],
        )?;
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
        })
    }

    pub fn update_probe_descriptors(
        &self,
        device: &ash::Device,
        rc_probes: &RcProbeBuffer,
        frame_slot: usize,
    ) {
        let ds = self.descriptor_sets[frame_slot];
        let read_info = vk::DescriptorBufferInfo::default()
            .buffer(rc_probes.read_buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let read_write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&read_info));

        let write_info = vk::DescriptorBufferInfo::default()
            .buffer(rc_probes.write_buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let write_write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&write_info));

        unsafe { device.update_descriptor_sets(&[read_write, write_write], &[]) };
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        self.bind(device, cmd, frame_slot);

        for cascade_index in 0..CASCADE_PARAMS.len() {
            self.record_cascade(device, cmd, cascade_index);
        }
    }

    pub fn bind(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
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
    }

    pub fn record_cascade(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        cascade_index: usize,
    ) {
        let params = CASCADE_PARAMS
            .get(cascade_index)
            .expect("radiance cascade trace index out of range");
        let push = RcTracePushConstants {
            cascade_level: params.level,
            probe_grid_dim: params.grid_dim,
            probe_size: params.probe_size,
            buffer_offset: params.offset,
        };
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                &push as *const RcTracePushConstants as *const u8,
                std::mem::size_of::<RcTracePushConstants>(),
            )
        };
        unsafe {
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            device.cmd_dispatch(cmd, (params.total_invocations + 63) / 64, 1, 1);
        }
    }

    pub fn destroy(self, device: &ash::Device, _allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}
