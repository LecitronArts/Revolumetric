use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::vpt::VptPass;
use crate::render::passes::vpt_surface::{
    VptCurrentSurfaceResources, VptPreviousSurfaceResources, VptSurfacePass,
};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};

pub struct VptTemporalPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub accumulated_radiance: GpuImage,
    /// Stores luminance moments in xy, history_length in z, and validity in w.
    pub accumulated_moments_history: GpuImage,
    pub previous_accumulated_radiance: GpuImage,
    pub previous_accumulated_moments_history: GpuImage,
}

#[derive(Clone, Copy)]
pub struct VptTemporalGraphOutputs {
    pub accumulated_radiance: ResourceHandle,
    pub accumulated_moments: ResourceHandle,
    pub previous_accumulated_radiance: ResourceHandle,
    pub previous_accumulated_moments: ResourceHandle,
}

#[derive(Clone, Copy)]
pub struct VptTemporalGraphInputs<'a> {
    pub frame_slot: usize,
    pub history_initialized: bool,
    pub noisy_inputs: [ResourceHandle; 2],
    pub surface_inputs: VptCurrentSurfaceResources,
    pub previous_surface_inputs: VptPreviousSurfaceResources,
    pub profiler: Option<&'a GpuProfiler>,
}

pub struct VptTemporalPassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub spirv_bytes: &'a [u8],
    pub scene_ubo: &'a SceneUniformBuffer,
    pub vpt: &'a VptPass,
    pub vpt_surface: &'a VptSurfacePass,
}

pub struct VptTemporalPassResizeInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub vpt: &'a VptPass,
    pub vpt_surface: &'a VptSurfacePass,
}

impl VptTemporalPass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 19] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(6, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(7, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(8, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(9, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(10, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(11, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(12, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(13, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(14, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(15, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(16, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(17, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(18, vk::DescriptorType::STORAGE_IMAGE),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptTemporalPassCreateInfo<'_>,
    ) -> Result<Self> {
        let descriptor_set_layout = create_descriptor_set_layout(device)?;
        let frame_count = info.scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 18 * frame_count as u32,
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

        let images = match create_temporal_images(device, allocator, info.width, info.height) {
            Ok(images) => images,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        write_descriptor_sets(
            device,
            &descriptor_sets,
            info.scene_ubo,
            info.vpt,
            info.vpt_surface,
            &VptTemporalImageRefs {
                accumulated_radiance: &images.accumulated_radiance,
                accumulated_moments_history: &images.accumulated_moments_history,
                previous_accumulated_radiance: &images.previous_accumulated_radiance,
                previous_accumulated_moments_history: &images.previous_accumulated_moments_history,
            },
        );

        let shader_module = match create_shader_module(device, info.spirv_bytes) {
            Ok(module) => module,
            Err(error) => {
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
            accumulated_radiance: images.accumulated_radiance,
            accumulated_moments_history: images.accumulated_moments_history,
            previous_accumulated_radiance: images.previous_accumulated_radiance,
            previous_accumulated_moments_history: images.previous_accumulated_moments_history,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptTemporalPassResizeInfo<'_>,
    ) -> Result<()> {
        let new_images = create_temporal_images(device, allocator, info.width, info.height)?;
        let old_images = VptTemporalImages {
            accumulated_radiance: std::mem::replace(
                &mut self.accumulated_radiance,
                new_images.accumulated_radiance,
            ),
            accumulated_moments_history: std::mem::replace(
                &mut self.accumulated_moments_history,
                new_images.accumulated_moments_history,
            ),
            previous_accumulated_radiance: std::mem::replace(
                &mut self.previous_accumulated_radiance,
                new_images.previous_accumulated_radiance,
            ),
            previous_accumulated_moments_history: std::mem::replace(
                &mut self.previous_accumulated_moments_history,
                new_images.previous_accumulated_moments_history,
            ),
        };
        old_images.destroy(device, allocator);
        self.update_input_images(device, info.scene_ubo, info.vpt, info.vpt_surface);
        Ok(())
    }

    pub fn update_input_images(
        &self,
        device: &ash::Device,
        scene_ubo: &SceneUniformBuffer,
        vpt: &VptPass,
        vpt_surface: &VptSurfacePass,
    ) {
        write_descriptor_sets(
            device,
            &self.descriptor_sets,
            scene_ubo,
            vpt,
            vpt_surface,
            &VptTemporalImageRefs {
                accumulated_radiance: &self.accumulated_radiance,
                accumulated_moments_history: &self.accumulated_moments_history,
                previous_accumulated_radiance: &self.previous_accumulated_radiance,
                previous_accumulated_moments_history: &self.previous_accumulated_moments_history,
            },
        );
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let extent = self.accumulated_radiance.extent;

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
        copy_temporal_image(
            device,
            cmd,
            &self.accumulated_radiance,
            &self.previous_accumulated_radiance,
        );
        copy_temporal_image(
            device,
            cmd,
            &self.accumulated_moments_history,
            &self.previous_accumulated_moments_history,
        );
    }

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        inputs: VptTemporalGraphInputs<'a>,
    ) -> VptTemporalGraphOutputs {
        let VptTemporalGraphInputs {
            frame_slot,
            history_initialized,
            noisy_inputs,
            surface_inputs,
            previous_surface_inputs,
            profiler,
        } = inputs;
        let temporal_initial_access = if history_initialized {
            AccessKind::TransferRead
        } else {
            AccessKind::Undefined
        };
        let previous_temporal_access = if history_initialized {
            AccessKind::TransferWrite
        } else {
            AccessKind::Undefined
        };
        let temporal_radiance_resource = graph.import_image_with_access(
            self.accumulated_radiance.handle,
            self.accumulated_radiance.extent.width,
            self.accumulated_radiance.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            temporal_initial_access,
        );
        let temporal_moments_resource = graph.import_image_with_access(
            self.accumulated_moments_history.handle,
            self.accumulated_moments_history.extent.width,
            self.accumulated_moments_history.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            temporal_initial_access,
        );
        let previous_temporal_radiance_resource = graph.import_image_with_access(
            self.previous_accumulated_radiance.handle,
            self.previous_accumulated_radiance.extent.width,
            self.previous_accumulated_radiance.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            previous_temporal_access,
        );
        let previous_temporal_moments_resource = graph.import_image_with_access(
            self.previous_accumulated_moments_history.handle,
            self.previous_accumulated_moments_history.extent.width,
            self.previous_accumulated_moments_history.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            previous_temporal_access,
        );

        let temporal_writes = graph.add_pass("vpt_temporal", QueueType::Compute, |builder| {
            builder.read_as(noisy_inputs[0], AccessKind::ComputeShaderRead);
            builder.read_as(noisy_inputs[1], AccessKind::ComputeShaderRead);
            surface_inputs.for_each(|surface_input| {
                builder.read_as(surface_input, AccessKind::ComputeShaderRead);
            });
            previous_surface_inputs.for_each(|previous_surface_input| {
                builder.read_as(previous_surface_input, AccessKind::ComputeShaderRead);
            });
            builder.read_as(
                previous_temporal_radiance_resource,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(
                previous_temporal_moments_resource,
                AccessKind::ComputeShaderRead,
            );
            builder.write_as(temporal_radiance_resource, AccessKind::ComputeShaderWrite);
            builder.write_as(temporal_moments_resource, AccessKind::ComputeShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptTemporal,
                    );
                }
                self.record(ctx.device, ctx.command_buffer, frame_slot);
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptTemporal,
                    );
                }
            })
        });

        VptTemporalGraphOutputs {
            accumulated_radiance: temporal_writes[0],
            accumulated_moments: temporal_writes[1],
            previous_accumulated_radiance: previous_temporal_radiance_resource,
            previous_accumulated_moments: previous_temporal_moments_resource,
        }
    }

    pub fn register_history_update_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        vpt_surface: &'a VptSurfacePass,
        temporal_outputs: VptTemporalGraphOutputs,
        surface_inputs: VptCurrentSurfaceResources,
        previous_surface_inputs: VptPreviousSurfaceResources,
    ) {
        graph.add_pass(
            "vpt_surface_history_update",
            QueueType::Transfer,
            |builder| {
                builder.read_as(
                    temporal_outputs.accumulated_radiance,
                    AccessKind::TransferRead,
                );
                builder.read_as(
                    temporal_outputs.accumulated_moments,
                    AccessKind::TransferRead,
                );
                builder.write_as(
                    temporal_outputs.previous_accumulated_radiance,
                    AccessKind::TransferWrite,
                );
                builder.write_as(
                    temporal_outputs.previous_accumulated_moments,
                    AccessKind::TransferWrite,
                );
                builder.read_as(surface_inputs.position_depth, AccessKind::TransferRead);
                builder.read_as(surface_inputs.normal_roughness, AccessKind::TransferRead);
                builder.read_as(surface_inputs.albedo_material, AccessKind::TransferRead);
                builder.read_as(surface_inputs.material_roughness, AccessKind::TransferRead);
                builder.read_as(surface_inputs.view_z, AccessKind::TransferRead);
                builder.read_as(surface_inputs.motion_id, AccessKind::TransferRead);
                builder.read_as(surface_inputs.brick_generation, AccessKind::TransferRead);
                builder.write_as(
                    previous_surface_inputs.position_depth,
                    AccessKind::TransferWrite,
                );
                builder.write_as(
                    previous_surface_inputs.normal_roughness,
                    AccessKind::TransferWrite,
                );
                builder.write_as(
                    previous_surface_inputs.albedo_material,
                    AccessKind::TransferWrite,
                );
                builder.write_as(
                    previous_surface_inputs.material_roughness,
                    AccessKind::TransferWrite,
                );
                builder.write_as(previous_surface_inputs.view_z, AccessKind::TransferWrite);
                builder.write_as(previous_surface_inputs.motion_id, AccessKind::TransferWrite);
                builder.write_as(
                    previous_surface_inputs.brick_generation,
                    AccessKind::TransferWrite,
                );
                Box::new(move |ctx| {
                    self.record_history_update(ctx.device, ctx.command_buffer);
                    vpt_surface.record_history_update(ctx.device, ctx.command_buffer);
                })
            },
        );
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.accumulated_radiance.destroy(device, allocator);
        self.accumulated_moments_history.destroy(device, allocator);
        self.previous_accumulated_radiance
            .destroy(device, allocator);
        self.previous_accumulated_moments_history
            .destroy(device, allocator);
    }
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    DescriptorLayoutBuilder::new()
        .add_binding_specs(&VptTemporalPass::descriptor_binding_specs())
        .build(device)
}

struct VptTemporalImages {
    accumulated_radiance: GpuImage,
    accumulated_moments_history: GpuImage,
    previous_accumulated_radiance: GpuImage,
    previous_accumulated_moments_history: GpuImage,
}

struct VptTemporalImageRefs<'a> {
    accumulated_radiance: &'a GpuImage,
    accumulated_moments_history: &'a GpuImage,
    previous_accumulated_radiance: &'a GpuImage,
    previous_accumulated_moments_history: &'a GpuImage,
}

impl VptTemporalImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.accumulated_radiance.destroy(device, allocator);
        self.accumulated_moments_history.destroy(device, allocator);
        self.previous_accumulated_radiance
            .destroy(device, allocator);
        self.previous_accumulated_moments_history
            .destroy(device, allocator);
    }
}

fn create_temporal_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptTemporalImages> {
    let accumulated_radiance =
        create_temporal_image(device, allocator, width, height, "vpt_temporal_radiance")?;
    let accumulated_moments_history = match create_temporal_image(
        device,
        allocator,
        width,
        height,
        "vpt_temporal_moments_history",
    ) {
        Ok(image) => image,
        Err(error) => {
            accumulated_radiance.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_accumulated_radiance = match create_temporal_image(
        device,
        allocator,
        width,
        height,
        "vpt_previous_temporal_radiance",
    ) {
        Ok(image) => image,
        Err(error) => {
            accumulated_moments_history.destroy(device, allocator);
            accumulated_radiance.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_accumulated_moments_history = match create_temporal_image(
        device,
        allocator,
        width,
        height,
        "vpt_previous_temporal_moments_history",
    ) {
        Ok(image) => image,
        Err(error) => {
            previous_accumulated_radiance.destroy(device, allocator);
            accumulated_moments_history.destroy(device, allocator);
            accumulated_radiance.destroy(device, allocator);
            return Err(error);
        }
    };

    Ok(VptTemporalImages {
        accumulated_radiance,
        accumulated_moments_history,
        previous_accumulated_radiance,
        previous_accumulated_moments_history,
    })
}

fn create_temporal_image(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
    name: &'static str,
) -> Result<GpuImage> {
    GpuImage::new(
        device,
        allocator,
        &GpuImageDesc {
            width,
            height,
            depth: 1,
            format: vk::Format::R16G16B16A16_SFLOAT,
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
            name,
        },
    )
}

fn copy_temporal_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: &GpuImage,
    dst: &GpuImage,
) {
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
        .extent(src.extent);
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

fn write_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    scene_ubo: &SceneUniformBuffer,
    vpt: &VptPass,
    vpt_surface: &VptSurfacePass,
    temporal: &VptTemporalImageRefs<'_>,
) {
    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(scene_ubo.buffer_handle(set_idx))
            .offset(0)
            .range(std::mem::size_of::<GpuSceneUniforms>() as u64);
        let image_refs = [
            &vpt.noisy_radiance_image,
            &vpt.noisy_moments_image,
            &vpt_surface.surface_position_depth,
            &vpt_surface.surface_normal_roughness,
            &vpt_surface.surface_albedo_material,
            &vpt_surface.surface_material_roughness,
            &vpt_surface.previous_surface_position_depth,
            &vpt_surface.previous_surface_normal_roughness,
            &vpt_surface.previous_surface_albedo_material,
            &vpt_surface.previous_surface_material_roughness,
            &vpt_surface.motion_history,
            temporal.accumulated_radiance,
            temporal.accumulated_moments_history,
            temporal.previous_accumulated_radiance,
            temporal.previous_accumulated_moments_history,
            &vpt_surface.motion_flags,
            &vpt_surface.surface_brick_generation,
            &vpt_surface.previous_surface_brick_generation,
        ];
        let image_infos: Vec<_> = image_refs
            .iter()
            .map(|image| {
                vk::DescriptorImageInfo::default()
                    .image_view(image.view)
                    .image_layout(vk::ImageLayout::GENERAL)
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
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}
