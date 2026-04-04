use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;

use crate::render::allocator::GpuAllocator;
use crate::render::camera::PrimaryRayPushConstants;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{create_shader_module, ComputePipeline};
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct PrimaryRayPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pub output_image: GpuImage,
}

impl PrimaryRayPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        spirv_bytes: &[u8],
        ucvh_gpu: &UcvhGpuResources,
    ) -> Result<Self> {
        // Descriptor set layout: 1 storage image + 8 storage buffers
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(0, vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(1, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(2, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(3, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(4, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(5, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(6, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(7, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .add_binding(8, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .build(device)?;

        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 8 },
        ];
        let descriptor_pool = DescriptorPool::new(device, 1, &pool_sizes)?;
        let descriptor_set = descriptor_pool.allocate(device, &[descriptor_set_layout])?[0];

        // Output image
        let output_image = GpuImage::new(
            device, allocator,
            &GpuImageDesc {
                width, height, depth: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "primary_ray_output",
            },
        )?;

        // Write descriptor set
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(output_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let image_write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_info));

        // SSBO buffer infos: config, occupancy, materials, l0, l1, l2, l3, l4
        let buffer_handles = [
            &ucvh_gpu.config_buffer,
            &ucvh_gpu.occupancy_buffer,
            &ucvh_gpu.material_buffer,
            &ucvh_gpu.hierarchy_l0_buffer,
            &ucvh_gpu.hierarchy_ln_buffers[0],
            &ucvh_gpu.hierarchy_ln_buffers[1],
            &ucvh_gpu.hierarchy_ln_buffers[2],
            &ucvh_gpu.hierarchy_ln_buffers[3],
        ];

        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffer_handles
            .iter()
            .map(|buf| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buf.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        let mut buffer_writes: Vec<vk::WriteDescriptorSet> = Vec::new();
        for (i, info) in buffer_infos.iter().enumerate() {
            buffer_writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding((i + 1) as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            );
        }

        let mut all_writes = vec![image_write];
        all_writes.extend(buffer_writes);
        unsafe { device.update_descriptor_sets(&all_writes, &[]) };

        // Pipeline
        let shader_module = create_shader_module(device, spirv_bytes)?;
        let push_constant_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<PrimaryRayPushConstants>() as u32,
        }];
        let pipeline = ComputePipeline::new(
            device,
            shader_module,
            unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") },
            &[descriptor_set_layout],
            &push_constant_ranges,
        )?;
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            output_image,
        })
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, pc: &PrimaryRayPushConstants) {
        let extent = self.output_image.extent;

        // Transition output image to GENERAL for compute write
        let barrier = vk::ImageMemoryBarrier::default()
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
            );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier],
            );
        }

        // Bind pipeline and descriptor set
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.descriptor_set],
                &[],
            );
        }

        // Push constants
        unsafe {
            let pc_bytes = std::slice::from_raw_parts(
                pc as *const _ as *const u8,
                std::mem::size_of::<PrimaryRayPushConstants>(),
            );
            device.cmd_push_constants(
                cmd, self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE, 0, pc_bytes,
            );
        }

        // Dispatch (8×8 workgroups)
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
