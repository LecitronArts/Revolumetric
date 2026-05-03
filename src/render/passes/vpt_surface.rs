use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::render::vpt_history::GpuVptHistoryUniforms;
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct VptSurfacePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    history_uniform_buffers: Vec<GpuBuffer>,
    pub surface_position_depth: GpuImage,
    pub surface_normal_roughness: GpuImage,
    pub surface_albedo_material: GpuImage,
    pub previous_surface_position_depth: GpuImage,
    pub previous_surface_normal_roughness: GpuImage,
    pub previous_surface_albedo_material: GpuImage,
    pub motion_history: GpuImage,
}

impl VptSurfacePass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        spirv_bytes: &[u8],
        ucvh_gpu: &UcvhGpuResources,
        scene_ubo: &SceneUniformBuffer,
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
                vk::DescriptorType::STORAGE_IMAGE,
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
            .add_binding(
                8,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                9,
                vk::DescriptorType::UNIFORM_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 2 * frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 4 * frame_count as u32,
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

        let images = match create_surface_images(device, allocator, width, height) {
            Ok(images) => images,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let history_uniform_buffers =
            match create_history_uniform_buffers(device, allocator, frame_count) {
                Ok(buffers) => buffers,
                Err(error) => {
                    images.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };

        write_descriptor_sets(
            device,
            &descriptor_sets,
            scene_ubo,
            &images,
            ucvh_gpu,
            &history_uniform_buffers,
        );

        let shader_module = match create_shader_module(device, spirv_bytes) {
            Ok(module) => module,
            Err(error) => {
                destroy_buffers(history_uniform_buffers, device, allocator);
                images.destroy(device, allocator);
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
            &[],
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe { device.destroy_shader_module(shader_module, None) };
                destroy_buffers(history_uniform_buffers, device, allocator);
                images.destroy(device, allocator);
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
            history_uniform_buffers,
            surface_position_depth: images.surface_position_depth,
            surface_normal_roughness: images.surface_normal_roughness,
            surface_albedo_material: images.surface_albedo_material,
            previous_surface_position_depth: images.previous_surface_position_depth,
            previous_surface_normal_roughness: images.previous_surface_normal_roughness,
            previous_surface_albedo_material: images.previous_surface_albedo_material,
            motion_history: images.motion_history,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        scene_ubo: &SceneUniformBuffer,
        ucvh_gpu: &UcvhGpuResources,
    ) -> Result<()> {
        let new_images = create_surface_images(device, allocator, width, height)?;
        let old_images = VptSurfaceImages {
            surface_position_depth: std::mem::replace(
                &mut self.surface_position_depth,
                new_images.surface_position_depth,
            ),
            surface_normal_roughness: std::mem::replace(
                &mut self.surface_normal_roughness,
                new_images.surface_normal_roughness,
            ),
            surface_albedo_material: std::mem::replace(
                &mut self.surface_albedo_material,
                new_images.surface_albedo_material,
            ),
            previous_surface_position_depth: std::mem::replace(
                &mut self.previous_surface_position_depth,
                new_images.previous_surface_position_depth,
            ),
            previous_surface_normal_roughness: std::mem::replace(
                &mut self.previous_surface_normal_roughness,
                new_images.previous_surface_normal_roughness,
            ),
            previous_surface_albedo_material: std::mem::replace(
                &mut self.previous_surface_albedo_material,
                new_images.previous_surface_albedo_material,
            ),
            motion_history: std::mem::replace(&mut self.motion_history, new_images.motion_history),
        };
        old_images.destroy(device, allocator);

        let current_images = VptSurfaceImageRefs {
            surface_position_depth: &self.surface_position_depth,
            surface_normal_roughness: &self.surface_normal_roughness,
            surface_albedo_material: &self.surface_albedo_material,
            motion_history: &self.motion_history,
        };
        write_descriptor_sets_from_refs(
            device,
            &self.descriptor_sets,
            scene_ubo,
            current_images,
            ucvh_gpu,
            &self.history_uniform_buffers,
        );
        Ok(())
    }

    pub fn update_history_uniforms(&self, frame_slot: usize, uniforms: &GpuVptHistoryUniforms) {
        write_mapped(
            self.history_uniform_buffers[frame_slot].mapped_ptr(),
            uniforms,
        );
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let extent = self.surface_position_depth.extent;

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
            device.cmd_dispatch(cmd, extent.width.div_ceil(8), extent.height.div_ceil(8), 1);
        }
    }

    pub fn record_history_update(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        copy_surface_image(
            device,
            cmd,
            &self.surface_position_depth,
            &self.previous_surface_position_depth,
        );
        copy_surface_image(
            device,
            cmd,
            &self.surface_normal_roughness,
            &self.previous_surface_normal_roughness,
        );
        copy_surface_image(
            device,
            cmd,
            &self.surface_albedo_material,
            &self.previous_surface_albedo_material,
        );
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        destroy_buffers(self.history_uniform_buffers, device, allocator);
        self.surface_position_depth.destroy(device, allocator);
        self.surface_normal_roughness.destroy(device, allocator);
        self.surface_albedo_material.destroy(device, allocator);
        self.previous_surface_position_depth
            .destroy(device, allocator);
        self.previous_surface_normal_roughness
            .destroy(device, allocator);
        self.previous_surface_albedo_material
            .destroy(device, allocator);
        self.motion_history.destroy(device, allocator);
    }
}

fn copy_surface_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: &GpuImage,
    dst: &GpuImage,
) {
    let extent = src.extent;
    let region = vk::ImageCopy::default()
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .extent(extent);

    unsafe {
        device.cmd_copy_image(
            cmd,
            src.handle,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst.handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
}

struct VptSurfaceImages {
    surface_position_depth: GpuImage,
    surface_normal_roughness: GpuImage,
    surface_albedo_material: GpuImage,
    previous_surface_position_depth: GpuImage,
    previous_surface_normal_roughness: GpuImage,
    previous_surface_albedo_material: GpuImage,
    motion_history: GpuImage,
}

struct VptSurfaceImageRefs<'a> {
    surface_position_depth: &'a GpuImage,
    surface_normal_roughness: &'a GpuImage,
    surface_albedo_material: &'a GpuImage,
    motion_history: &'a GpuImage,
}

impl VptSurfaceImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.surface_position_depth.destroy(device, allocator);
        self.surface_normal_roughness.destroy(device, allocator);
        self.surface_albedo_material.destroy(device, allocator);
        self.previous_surface_position_depth
            .destroy(device, allocator);
        self.previous_surface_normal_roughness
            .destroy(device, allocator);
        self.previous_surface_albedo_material
            .destroy(device, allocator);
        self.motion_history.destroy(device, allocator);
    }
}

fn create_surface_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptSurfaceImages> {
    let surface_position_depth = create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_surface_position_depth",
    )?;
    let surface_normal_roughness = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_surface_normal_roughness",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let surface_albedo_material = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_surface_albedo_material",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let motion_history = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_motion_history",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_position_depth = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_previous_surface_position_depth",
    ) {
        Ok(image) => image,
        Err(error) => {
            motion_history.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_normal_roughness = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_previous_surface_normal_roughness",
    ) {
        Ok(image) => image,
        Err(error) => {
            previous_surface_position_depth.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_albedo_material = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_previous_surface_albedo_material",
    ) {
        Ok(image) => image,
        Err(error) => {
            previous_surface_normal_roughness.destroy(device, allocator);
            previous_surface_position_depth.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };

    Ok(VptSurfaceImages {
        surface_position_depth,
        surface_normal_roughness,
        surface_albedo_material,
        previous_surface_position_depth,
        previous_surface_normal_roughness,
        previous_surface_albedo_material,
        motion_history,
    })
}

fn create_surface_image(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
    format: vk::Format,
    name: &'static str,
) -> Result<GpuImage> {
    GpuImage::new(
        device,
        allocator,
        &GpuImageDesc {
            width,
            height,
            depth: 1,
            format,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
            name,
        },
    )
}

fn write_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    scene_ubo: &SceneUniformBuffer,
    images: &VptSurfaceImages,
    ucvh_gpu: &UcvhGpuResources,
    history_uniform_buffers: &[GpuBuffer],
) {
    write_descriptor_sets_from_refs(
        device,
        descriptor_sets,
        scene_ubo,
        VptSurfaceImageRefs {
            surface_position_depth: &images.surface_position_depth,
            surface_normal_roughness: &images.surface_normal_roughness,
            surface_albedo_material: &images.surface_albedo_material,
            motion_history: &images.motion_history,
        },
        ucvh_gpu,
        history_uniform_buffers,
    );
}

fn write_descriptor_sets_from_refs(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    scene_ubo: &SceneUniformBuffer,
    images: VptSurfaceImageRefs<'_>,
    ucvh_gpu: &UcvhGpuResources,
    history_uniform_buffers: &[GpuBuffer],
) {
    let output_images = [
        images.surface_position_depth,
        images.surface_normal_roughness,
        images.surface_albedo_material,
        images.motion_history,
    ];
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
        let history_ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(history_uniform_buffers[set_idx].handle)
            .offset(0)
            .range(std::mem::size_of::<GpuVptHistoryUniforms>() as u64);
        let image_infos: Vec<vk::DescriptorImageInfo> = output_images
            .iter()
            .map(|image| {
                vk::DescriptorImageInfo::default()
                    .image_view(image.view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect();
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info)),
        ];
        writes.extend(image_infos.iter().enumerate().map(|(idx, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding((idx + 1) as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(info))
        }));
        writes.extend(buffer_infos.iter().enumerate().map(|(idx, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding((idx + 5) as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        }));
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&history_ubo_info)),
        );
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}

fn create_history_uniform_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count);
    for slot in 0..frame_count {
        buffers.push(GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GpuVptHistoryUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("vpt_history_uniforms_{slot}"),
        )?);
    }
    Ok(buffers)
}

fn destroy_buffers(buffers: Vec<GpuBuffer>, device: &ash::Device, allocator: &GpuAllocator) {
    for buffer in buffers {
        buffer.destroy(device, allocator);
    }
}

fn write_mapped<T: Copy>(mapped_ptr: Option<*mut u8>, value: &T) {
    let Some(ptr) = mapped_ptr else {
        return;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(
            value as *const T as *const u8,
            ptr,
            std::mem::size_of::<T>(),
        );
    }
}
