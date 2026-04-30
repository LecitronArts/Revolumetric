use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct PrimaryRayPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub gbuffer_pos: GpuImage,
    pub gbuffer0: GpuImage,
    pub gbuffer1: GpuImage,
}

impl PrimaryRayPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        spirv_bytes: &[u8],
        ucvh_gpu: &UcvhGpuResources,
        scene_ubo: &SceneUniformBuffer,
    ) -> Result<Self> {
        // Descriptor set layout: 1 UBO + 3 storage images + 4 storage buffers
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                2,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                3,
                vk::DescriptorType::STORAGE_IMAGE,
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
            .add_binding(
                7,
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
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 3 * frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 4 * frame_count as u32,
            },
        ];
        let descriptor_pool = match DescriptorPool::new(device, frame_count as u32, &pool_sizes) {
            Ok(pool) => pool,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let layouts: Vec<_> = (0..frame_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = match descriptor_pool.allocate(device, &layouts) {
            Ok(sets) => sets,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        // G-buffer images
        let gbuffer_pos = match GpuImage::new(
            device,
            allocator,
            &GpuImageDesc {
                width,
                height,
                depth: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "gbuffer_pos",
            },
        ) {
            Ok(image) => image,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        let gbuffer0 = match GpuImage::new(
            device,
            allocator,
            &GpuImageDesc {
                width,
                height,
                depth: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "gbuffer0",
            },
        ) {
            Ok(image) => image,
            Err(error) => {
                gbuffer_pos.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        let gbuffer1 = match GpuImage::new(
            device,
            allocator,
            &GpuImageDesc {
                width,
                height,
                depth: 1,
                format: vk::Format::R8G8B8A8_UINT,
                usage: vk::ImageUsageFlags::STORAGE,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "gbuffer1",
            },
        ) {
            Ok(image) => image,
            Err(error) => {
                gbuffer0.destroy(device, allocator);
                gbuffer_pos.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        // Write descriptor sets (one per frame slot)
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
                .range(std::mem::size_of::<GpuSceneUniforms>() as u64);

            let ubo_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info));

            let image_infos = [
                vk::DescriptorImageInfo::default()
                    .image_view(gbuffer_pos.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(gbuffer0.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(gbuffer1.view)
                    .image_layout(vk::ImageLayout::GENERAL),
            ];

            let image_writes: Vec<vk::WriteDescriptorSet> = image_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding((i + 1) as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(info))
                })
                .collect();

            let buffer_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers
                .iter()
                .map(|buf| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.handle)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();

            let buffer_writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding((i + 4) as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            let mut all_writes = vec![ubo_write];
            all_writes.extend(image_writes);
            all_writes.extend(buffer_writes);
            unsafe { device.update_descriptor_sets(&all_writes, &[]) };
        }

        // Pipeline (no push constant ranges)
        let shader_module = match create_shader_module(device, spirv_bytes) {
            Ok(module) => module,
            Err(error) => {
                gbuffer1.destroy(device, allocator);
                gbuffer0.destroy(device, allocator);
                gbuffer_pos.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let pipeline = match ComputePipeline::new(
            device,
            shader_module,
            c"main",
            &[descriptor_set_layout],
            &[], // no push constants
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe { device.destroy_shader_module(shader_module, None) };
                gbuffer1.destroy(device, allocator);
                gbuffer0.destroy(device, allocator);
                gbuffer_pos.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            gbuffer_pos,
            gbuffer0,
            gbuffer1,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let new_pos = GpuImage::new(
            device,
            allocator,
            &GpuImageDesc {
                width,
                height,
                depth: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "gbuffer_pos",
            },
        )?;
        let new_gb0 = GpuImage::new(
            device,
            allocator,
            &GpuImageDesc {
                width,
                height,
                depth: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "gbuffer0",
            },
        )?;
        let new_gb1 = GpuImage::new(
            device,
            allocator,
            &GpuImageDesc {
                width,
                height,
                depth: 1,
                format: vk::Format::R8G8B8A8_UINT,
                usage: vk::ImageUsageFlags::STORAGE,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "gbuffer1",
            },
        )?;

        let old_pos = std::mem::replace(&mut self.gbuffer_pos, new_pos);
        let old_gb0 = std::mem::replace(&mut self.gbuffer0, new_gb0);
        let old_gb1 = std::mem::replace(&mut self.gbuffer1, new_gb1);
        old_pos.destroy(device, allocator);
        old_gb0.destroy(device, allocator);
        old_gb1.destroy(device, allocator);

        for &ds in &self.descriptor_sets {
            let image_infos = [
                vk::DescriptorImageInfo::default()
                    .image_view(self.gbuffer_pos.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(self.gbuffer0.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(self.gbuffer1.view)
                    .image_layout(vk::ImageLayout::GENERAL),
            ];
            let writes: Vec<vk::WriteDescriptorSet> = image_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding((i + 1) as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(info))
                })
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        Ok(())
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let extent = self.gbuffer_pos.extent;

        // Transition all 3 G-buffer images to GENERAL for compute write
        let barriers: Vec<vk::ImageMemoryBarrier> = [
            self.gbuffer_pos.handle,
            self.gbuffer0.handle,
            self.gbuffer1.handle,
        ]
        .iter()
        .map(|&image| {
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                )
        })
        .collect();

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
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

        // Dispatch (8x8 workgroups)
        let groups_x = extent.width.div_ceil(8);
        let groups_y = extent.height.div_ceil(8);
        unsafe { device.cmd_dispatch(cmd, groups_x, groups_y, 1) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.gbuffer_pos.destroy(device, allocator);
        self.gbuffer0.destroy(device, allocator);
        self.gbuffer1.destroy(device, allocator);
    }
}

#[cfg(test)]
mod shader_source_tests {
    #[test]
    fn primary_ray_uses_shared_material_helpers_and_dynamic_ubo_range() {
        let shader =
            include_str!("../../../assets/shaders/passes/primary_ray.slang").replace("\r\n", "\n");
        let rust = std::fs::read_to_string("src/render/passes/primary_ray.rs")
            .expect("primary ray pass source should be readable");
        let hardcoded_range = [".range(", "176)"].join("");

        assert!(shader.contains("#include \"material_common.slang\""));
        assert!(shader.contains("material_cell_albedo(hit.cell)"));
        assert!(!shader.contains("static const float3 MATERIAL_ALBEDO"));
        assert!(rust.contains("std::mem::size_of::<GpuSceneUniforms>() as u64"));
        assert!(!rust.contains(&hardcoded_range));
    }

    #[test]
    fn voxel_trace_returns_surface_hit_position_not_voxel_center() {
        let traverse = include_str!("../../../assets/shaders/shared/voxel_traverse.slang")
            .replace("\r\n", "\n");
        let primary =
            include_str!("../../../assets/shaders/passes/primary_ray.slang").replace("\r\n", "\n");

        assert!(
            traverse.contains("float current_t = t_enter;"),
            "brick DDA should track entry distance for the current voxel"
        );
        assert!(
            traverse.contains("hit_t = current_t;"),
            "brick DDA should return the hit voxel entry distance, not its exit distance"
        );
        assert!(
            traverse.contains("result.position = ray.origin + ray.direction * hit_t;"),
            "primary trace hit.position should be the actual ray surface hit point"
        );
        assert!(
            !traverse.contains("result.position = brick_origin + float3(hit_local) + 0.5;"),
            "primary trace must not report voxel centers as surface hit points"
        );
        assert!(
            primary.contains("xyz = surface hit point"),
            "G-buffer position semantics should document the surface hit point"
        );
    }
}
