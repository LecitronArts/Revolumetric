use anyhow::Result;
use ash::vk;
use std::ffi::CStr;


use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::primary_ray::PrimaryRayPass;
use crate::render::pipeline::{create_shader_module, ComputePipeline};
use crate::render::scene_ubo::SceneUniformBuffer;
use crate::render::rc_probe_buffer::RcProbeBuffer;
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
        rc_probes: &RcProbeBuffer,
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
            .add_binding(9, vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE, 1)
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: frame_count as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 4 * frame_count as u32 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 5 * frame_count as u32 },
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
                .range(176);

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

            // Binding 9: RC probe buffer (read current frame's merged data)
            let rc_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer())
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let rc_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&rc_info));
            all_writes.push(rc_write);

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

    /// Only updates the given frame_slot to avoid writing in-flight descriptor sets.
    pub fn update_rc_descriptor(&self, device: &ash::Device, rc_probes: &RcProbeBuffer, frame_slot: usize) {
        let ds = self.descriptor_sets[frame_slot];
        let rc_info = vk::DescriptorBufferInfo::default()
            .buffer(rc_probes.write_buffer())
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let rc_write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(9)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&rc_info));
        unsafe { device.update_descriptor_sets(&[rc_write], &[]) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.output_image.destroy(device, allocator);
    }
}
