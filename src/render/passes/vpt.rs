use anyhow::Result;
use ash::vk;
use bytemuck::Zeroable;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::area_restir::{GpuAreaRestirReservoir, GpuAreaRestirUniforms};
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::restir_di::{GpuRestirDiReservoir, GpuRestirDiUniforms};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct VptPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub noisy_radiance_image: GpuImage,
    pub noisy_moments_image: GpuImage,
    pub nrd_diff_radiance_hitdist: GpuImage,
    pub nrd_spec_radiance_hitdist: GpuImage,
    pub nrd_residual_radiance: GpuImage,
    pub nrd_material_factors: GpuImage,
    disabled_restir_uniform_buffers: Vec<GpuBuffer>,
    disabled_restir_reservoir_buffer: GpuBuffer,
    disabled_area_restir_uniform_buffers: Vec<GpuBuffer>,
    disabled_area_restir_reservoir_buffer: GpuBuffer,
}

#[derive(Clone, Copy)]
pub struct VptGraphOutputs {
    pub noisy_radiance: ResourceHandle,
    pub noisy_moments: ResourceHandle,
    pub nrd_noisy: VptNrdNoisyResources,
}

#[derive(Clone, Copy)]
pub struct VptNrdNoisyResources {
    pub diff_radiance_hitdist: ResourceHandle,
    pub spec_radiance_hitdist: ResourceHandle,
    pub residual_radiance: ResourceHandle,
    pub material_factors: ResourceHandle,
}

impl VptPass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 19] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(6, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(7, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(8, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(9, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(10, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(11, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(12, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(13, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(14, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(15, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(16, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(17, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(18, vk::DescriptorType::STORAGE_IMAGE),
        ]
    }

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
            .add_binding_specs(&Self::descriptor_binding_specs())
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 4 * frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 6 * frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 9 * frame_count as u32,
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

        let images = match create_vpt_images(device, allocator, width, height) {
            Ok(image) => image,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        write_descriptor_sets(
            device,
            &descriptor_sets,
            scene_ubo,
            &VptImagesRef {
                noisy_radiance_image: &images.noisy_radiance_image,
                noisy_moments_image: &images.noisy_moments_image,
                nrd_diff_radiance_hitdist: &images.nrd_diff_radiance_hitdist,
                nrd_spec_radiance_hitdist: &images.nrd_spec_radiance_hitdist,
                nrd_residual_radiance: &images.nrd_residual_radiance,
                nrd_material_factors: &images.nrd_material_factors,
            },
            ucvh_gpu,
        );

        let disabled_restir_uniform_buffers =
            match create_disabled_restir_uniform_buffers(device, allocator, frame_count) {
                Ok(buffers) => buffers,
                Err(error) => {
                    images.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let disabled_restir_reservoir_buffer =
            match create_disabled_restir_reservoir_buffer(device, allocator) {
                Ok(buffer) => buffer,
                Err(error) => {
                    destroy_buffers(disabled_restir_uniform_buffers, device, allocator);
                    images.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        write_restir_descriptor_sets(
            device,
            &descriptor_sets,
            &disabled_restir_uniform_buffers,
            &disabled_restir_reservoir_buffer,
        );

        let disabled_area_restir_uniform_buffers =
            match create_disabled_area_restir_uniform_buffers(device, allocator, frame_count) {
                Ok(buffers) => buffers,
                Err(error) => {
                    disabled_restir_reservoir_buffer.destroy(device, allocator);
                    destroy_buffers(disabled_restir_uniform_buffers, device, allocator);
                    images.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let disabled_area_restir_reservoir_buffer =
            match create_disabled_area_restir_reservoir_buffer(device, allocator) {
                Ok(buffer) => buffer,
                Err(error) => {
                    destroy_buffers(disabled_area_restir_uniform_buffers, device, allocator);
                    disabled_restir_reservoir_buffer.destroy(device, allocator);
                    destroy_buffers(disabled_restir_uniform_buffers, device, allocator);
                    images.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        write_area_restir_descriptor_sets(
            device,
            &descriptor_sets,
            &disabled_area_restir_uniform_buffers,
            &disabled_area_restir_reservoir_buffer,
        );

        let shader_module = match create_shader_module(device, spirv_bytes) {
            Ok(module) => module,
            Err(error) => {
                disabled_area_restir_reservoir_buffer.destroy(device, allocator);
                destroy_buffers(disabled_area_restir_uniform_buffers, device, allocator);
                disabled_restir_reservoir_buffer.destroy(device, allocator);
                destroy_buffers(disabled_restir_uniform_buffers, device, allocator);
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
                disabled_area_restir_reservoir_buffer.destroy(device, allocator);
                destroy_buffers(disabled_area_restir_uniform_buffers, device, allocator);
                disabled_restir_reservoir_buffer.destroy(device, allocator);
                destroy_buffers(disabled_restir_uniform_buffers, device, allocator);
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
            noisy_radiance_image: images.noisy_radiance_image,
            noisy_moments_image: images.noisy_moments_image,
            nrd_diff_radiance_hitdist: images.nrd_diff_radiance_hitdist,
            nrd_spec_radiance_hitdist: images.nrd_spec_radiance_hitdist,
            nrd_residual_radiance: images.nrd_residual_radiance,
            nrd_material_factors: images.nrd_material_factors,
            disabled_restir_uniform_buffers,
            disabled_restir_reservoir_buffer,
            disabled_area_restir_uniform_buffers,
            disabled_area_restir_reservoir_buffer,
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
        let new_images = create_vpt_images(device, allocator, width, height)?;
        let old_images = VptImages {
            noisy_radiance_image: std::mem::replace(
                &mut self.noisy_radiance_image,
                new_images.noisy_radiance_image,
            ),
            noisy_moments_image: std::mem::replace(
                &mut self.noisy_moments_image,
                new_images.noisy_moments_image,
            ),
            nrd_diff_radiance_hitdist: std::mem::replace(
                &mut self.nrd_diff_radiance_hitdist,
                new_images.nrd_diff_radiance_hitdist,
            ),
            nrd_spec_radiance_hitdist: std::mem::replace(
                &mut self.nrd_spec_radiance_hitdist,
                new_images.nrd_spec_radiance_hitdist,
            ),
            nrd_residual_radiance: std::mem::replace(
                &mut self.nrd_residual_radiance,
                new_images.nrd_residual_radiance,
            ),
            nrd_material_factors: std::mem::replace(
                &mut self.nrd_material_factors,
                new_images.nrd_material_factors,
            ),
        };
        old_images.destroy(device, allocator);
        write_descriptor_sets(
            device,
            &self.descriptor_sets,
            scene_ubo,
            &VptImagesRef {
                noisy_radiance_image: &self.noisy_radiance_image,
                noisy_moments_image: &self.noisy_moments_image,
                nrd_diff_radiance_hitdist: &self.nrd_diff_radiance_hitdist,
                nrd_spec_radiance_hitdist: &self.nrd_spec_radiance_hitdist,
                nrd_residual_radiance: &self.nrd_residual_radiance,
                nrd_material_factors: &self.nrd_material_factors,
            },
            ucvh_gpu,
        );
        write_restir_descriptor_sets(
            device,
            &self.descriptor_sets,
            &self.disabled_restir_uniform_buffers,
            &self.disabled_restir_reservoir_buffer,
        );
        write_area_restir_descriptor_sets(
            device,
            &self.descriptor_sets,
            &self.disabled_area_restir_uniform_buffers,
            &self.disabled_area_restir_reservoir_buffer,
        );
        Ok(())
    }

    pub fn update_restir_di_descriptors(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        uniforms: &GpuBuffer,
        reservoirs: &GpuBuffer,
    ) {
        let restir_uniform_info = vk::DescriptorBufferInfo::default()
            .buffer(uniforms.handle)
            .offset(0)
            .range(std::mem::size_of::<GpuRestirDiUniforms>() as u64);
        let restir_reservoir_info = vk::DescriptorBufferInfo::default()
            .buffer(reservoirs.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[frame_slot])
                .dst_binding(10)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&restir_uniform_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[frame_slot])
                .dst_binding(11)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&restir_reservoir_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn update_area_restir_descriptors(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        uniforms: &GpuBuffer,
        reservoirs: &GpuBuffer,
    ) {
        let area_uniform_info = vk::DescriptorBufferInfo::default()
            .buffer(uniforms.handle)
            .offset(0)
            .range(std::mem::size_of::<GpuAreaRestirUniforms>() as u64);
        let area_reservoir_info = vk::DescriptorBufferInfo::default()
            .buffer(reservoirs.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[frame_slot])
                .dst_binding(13)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&area_uniform_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[frame_slot])
                .dst_binding(14)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&area_reservoir_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let extent = self.noisy_radiance_image.extent;

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

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        frame_slot: usize,
        accumulation_needs_init: bool,
        restir_reads: Option<(ResourceHandle, ResourceHandle)>,
        area_restir_reads: Option<(ResourceHandle, ResourceHandle)>,
        profiler: Option<&'a GpuProfiler>,
    ) -> VptGraphOutputs {
        let noisy_initial_access = if accumulation_needs_init {
            AccessKind::Undefined
        } else {
            AccessKind::ComputeShaderRead
        };
        let noisy_image_usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let noisy_radiance_resource = graph.import_image_with_access(
            self.noisy_radiance_image.handle,
            self.noisy_radiance_image.extent.width,
            self.noisy_radiance_image.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            noisy_image_usage,
            noisy_initial_access,
        );
        let noisy_moments_resource = graph.import_image_with_access(
            self.noisy_moments_image.handle,
            self.noisy_moments_image.extent.width,
            self.noisy_moments_image.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            noisy_image_usage,
            noisy_initial_access,
        );
        let nrd_diff_radiance_hitdist_resource = graph.import_image_with_access(
            self.nrd_diff_radiance_hitdist.handle,
            self.nrd_diff_radiance_hitdist.extent.width,
            self.nrd_diff_radiance_hitdist.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            noisy_image_usage,
            noisy_initial_access,
        );
        let nrd_spec_radiance_hitdist_resource = graph.import_image_with_access(
            self.nrd_spec_radiance_hitdist.handle,
            self.nrd_spec_radiance_hitdist.extent.width,
            self.nrd_spec_radiance_hitdist.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            noisy_image_usage,
            noisy_initial_access,
        );
        let nrd_residual_radiance_resource = graph.import_image_with_access(
            self.nrd_residual_radiance.handle,
            self.nrd_residual_radiance.extent.width,
            self.nrd_residual_radiance.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            noisy_image_usage,
            noisy_initial_access,
        );
        let nrd_material_factors_resource = graph.import_image_with_access(
            self.nrd_material_factors.handle,
            self.nrd_material_factors.extent.width,
            self.nrd_material_factors.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            noisy_image_usage,
            noisy_initial_access,
        );
        let vpt_writes = graph.add_pass("vpt", QueueType::Compute, |builder| {
            builder.write_as(noisy_radiance_resource, AccessKind::ComputeShaderWrite);
            builder.write_as(noisy_moments_resource, AccessKind::ComputeShaderWrite);
            builder.write_as(
                nrd_diff_radiance_hitdist_resource,
                AccessKind::ComputeShaderWrite,
            );
            builder.write_as(
                nrd_spec_radiance_hitdist_resource,
                AccessKind::ComputeShaderWrite,
            );
            builder.write_as(
                nrd_residual_radiance_resource,
                AccessKind::ComputeShaderWrite,
            );
            builder.write_as(
                nrd_material_factors_resource,
                AccessKind::ComputeShaderWrite,
            );
            if let Some((restir_uniform_resource, restir_reservoir_resource)) = restir_reads {
                builder.read_as(restir_uniform_resource, AccessKind::ComputeShaderRead);
                builder.read_as(restir_reservoir_resource, AccessKind::ComputeShaderRead);
            }
            if let Some((area_uniform_resource, area_selected_reservoir_resource)) =
                area_restir_reads
            {
                builder.read_as(area_uniform_resource, AccessKind::ComputeShaderRead);
                builder.read_as(
                    area_selected_reservoir_resource,
                    AccessKind::ComputeShaderRead,
                );
            }
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::Vpt,
                    );
                }
                self.record(ctx.device, ctx.command_buffer, frame_slot);
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::Vpt,
                    );
                }
            })
        });

        VptGraphOutputs {
            noisy_radiance: vpt_writes[0],
            noisy_moments: vpt_writes[1],
            nrd_noisy: VptNrdNoisyResources {
                diff_radiance_hitdist: vpt_writes[2],
                spec_radiance_hitdist: vpt_writes[3],
                residual_radiance: vpt_writes[4],
                material_factors: vpt_writes[5],
            },
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.noisy_radiance_image.destroy(device, allocator);
        self.noisy_moments_image.destroy(device, allocator);
        self.nrd_diff_radiance_hitdist.destroy(device, allocator);
        self.nrd_spec_radiance_hitdist.destroy(device, allocator);
        self.nrd_residual_radiance.destroy(device, allocator);
        self.nrd_material_factors.destroy(device, allocator);
        destroy_buffers(self.disabled_restir_uniform_buffers, device, allocator);
        self.disabled_restir_reservoir_buffer
            .destroy(device, allocator);
        destroy_buffers(self.disabled_area_restir_uniform_buffers, device, allocator);
        self.disabled_area_restir_reservoir_buffer
            .destroy(device, allocator);
    }
}

struct VptImages {
    noisy_radiance_image: GpuImage,
    noisy_moments_image: GpuImage,
    nrd_diff_radiance_hitdist: GpuImage,
    nrd_spec_radiance_hitdist: GpuImage,
    nrd_residual_radiance: GpuImage,
    nrd_material_factors: GpuImage,
}

struct VptImagesRef<'a> {
    noisy_radiance_image: &'a GpuImage,
    noisy_moments_image: &'a GpuImage,
    nrd_diff_radiance_hitdist: &'a GpuImage,
    nrd_spec_radiance_hitdist: &'a GpuImage,
    nrd_residual_radiance: &'a GpuImage,
    nrd_material_factors: &'a GpuImage,
}

impl VptImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.noisy_radiance_image.destroy(device, allocator);
        self.noisy_moments_image.destroy(device, allocator);
        self.nrd_diff_radiance_hitdist.destroy(device, allocator);
        self.nrd_spec_radiance_hitdist.destroy(device, allocator);
        self.nrd_residual_radiance.destroy(device, allocator);
        self.nrd_material_factors.destroy(device, allocator);
    }
}

fn create_vpt_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptImages> {
    let noisy_radiance_image =
        create_vpt_image(device, allocator, width, height, "vpt_noisy_radiance")?;
    let noisy_moments_image =
        match create_vpt_image(device, allocator, width, height, "vpt_noisy_moments") {
            Ok(image) => image,
            Err(error) => {
                noisy_radiance_image.destroy(device, allocator);
                return Err(error);
            }
        };
    let nrd_diff_radiance_hitdist = match create_vpt_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_diff_radiance_hitdist",
    ) {
        Ok(image) => image,
        Err(error) => {
            noisy_moments_image.destroy(device, allocator);
            noisy_radiance_image.destroy(device, allocator);
            return Err(error);
        }
    };
    let nrd_spec_radiance_hitdist = match create_vpt_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_spec_radiance_hitdist",
    ) {
        Ok(image) => image,
        Err(error) => {
            nrd_diff_radiance_hitdist.destroy(device, allocator);
            noisy_moments_image.destroy(device, allocator);
            noisy_radiance_image.destroy(device, allocator);
            return Err(error);
        }
    };
    let nrd_residual_radiance = match create_vpt_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_residual_radiance",
    ) {
        Ok(image) => image,
        Err(error) => {
            nrd_spec_radiance_hitdist.destroy(device, allocator);
            nrd_diff_radiance_hitdist.destroy(device, allocator);
            noisy_moments_image.destroy(device, allocator);
            noisy_radiance_image.destroy(device, allocator);
            return Err(error);
        }
    };
    let nrd_material_factors =
        match create_vpt_image(device, allocator, width, height, "vpt_nrd_material_factors") {
            Ok(image) => image,
            Err(error) => {
                nrd_residual_radiance.destroy(device, allocator);
                nrd_spec_radiance_hitdist.destroy(device, allocator);
                nrd_diff_radiance_hitdist.destroy(device, allocator);
                noisy_moments_image.destroy(device, allocator);
                noisy_radiance_image.destroy(device, allocator);
                return Err(error);
            }
        };

    Ok(VptImages {
        noisy_radiance_image,
        noisy_moments_image,
        nrd_diff_radiance_hitdist,
        nrd_spec_radiance_hitdist,
        nrd_residual_radiance,
        nrd_material_factors,
    })
}

fn create_vpt_image(
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
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            aspect: vk::ImageAspectFlags::COLOR,
            name,
        },
    )
}

fn write_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    scene_ubo: &SceneUniformBuffer,
    images: &VptImagesRef<'_>,
    ucvh_gpu: &UcvhGpuResources,
) {
    let ucvh_buffers = [
        &ucvh_gpu.hierarchy_l0_buffer,
        &ucvh_gpu.hierarchy_ln_buffers[0],
        &ucvh_gpu.hierarchy_ln_buffers[1],
        &ucvh_gpu.hierarchy_ln_buffers[2],
        &ucvh_gpu.hierarchy_ln_buffers[3],
        &ucvh_gpu.occupancy_buffer,
        &ucvh_gpu.material_buffer,
    ];
    let config_info = vk::DescriptorBufferInfo::default()
        .buffer(ucvh_gpu.config_buffer.handle)
        .offset(0)
        .range(vk::WHOLE_SIZE);

    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(scene_ubo.buffer_handle(set_idx))
            .offset(0)
            .range(std::mem::size_of::<GpuSceneUniforms>() as u64);
        let radiance_info = vk::DescriptorImageInfo::default()
            .image_view(images.noisy_radiance_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let moments_info = vk::DescriptorImageInfo::default()
            .image_view(images.noisy_moments_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let nrd_diff_info = vk::DescriptorImageInfo::default()
            .image_view(images.nrd_diff_radiance_hitdist.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let nrd_spec_info = vk::DescriptorImageInfo::default()
            .image_view(images.nrd_spec_radiance_hitdist.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let nrd_residual_info = vk::DescriptorImageInfo::default()
            .image_view(images.nrd_residual_radiance.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let nrd_material_info = vk::DescriptorImageInfo::default()
            .image_view(images.nrd_material_factors.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers
            .iter()
            .map(|buf| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buf.handle)
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
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&radiance_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&config_info)),
        ];
        writes.extend(buffer_infos.iter().enumerate().map(|(idx, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding((idx + 3) as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        }));
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(12)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&moments_info)),
        );
        writes.extend([
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(15)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&nrd_diff_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(16)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&nrd_spec_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(17)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&nrd_residual_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(18)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&nrd_material_info)),
        ]);
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}

fn write_restir_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    disabled_restir_uniform_buffers: &[GpuBuffer],
    disabled_restir_reservoir_buffer: &GpuBuffer,
) {
    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        let restir_uniform_info = vk::DescriptorBufferInfo::default()
            .buffer(disabled_restir_uniform_buffers[set_idx].handle)
            .offset(0)
            .range(std::mem::size_of::<GpuRestirDiUniforms>() as u64);
        let restir_reservoir_info = vk::DescriptorBufferInfo::default()
            .buffer(disabled_restir_reservoir_buffer.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(10)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&restir_uniform_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(11)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&restir_reservoir_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}

fn write_area_restir_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    disabled_area_restir_uniform_buffers: &[GpuBuffer],
    disabled_area_restir_reservoir_buffer: &GpuBuffer,
) {
    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        let area_uniform_info = vk::DescriptorBufferInfo::default()
            .buffer(disabled_area_restir_uniform_buffers[set_idx].handle)
            .offset(0)
            .range(std::mem::size_of::<GpuAreaRestirUniforms>() as u64);
        let area_reservoir_info = vk::DescriptorBufferInfo::default()
            .buffer(disabled_area_restir_reservoir_buffer.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(13)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&area_uniform_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(14)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&area_reservoir_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}

fn create_disabled_restir_uniform_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count);
    for slot in 0..frame_count {
        let buffer = match GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GpuRestirDiUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("vpt_disabled_restir_uniforms_{slot}"),
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                destroy_buffers(buffers, device, allocator);
                return Err(error);
            }
        };
        let uniforms = GpuRestirDiUniforms::zeroed();
        write_mapped(buffer.mapped_ptr(), &uniforms);
        buffers.push(buffer);
    }
    Ok(buffers)
}

fn create_disabled_area_restir_uniform_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count);
    for slot in 0..frame_count {
        let buffer = match GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GpuAreaRestirUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("vpt_disabled_area_restir_uniforms_{slot}"),
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                destroy_buffers(buffers, device, allocator);
                return Err(error);
            }
        };
        let uniforms = GpuAreaRestirUniforms::zeroed();
        write_mapped(buffer.mapped_ptr(), &uniforms);
        buffers.push(buffer);
    }
    Ok(buffers)
}

fn create_disabled_area_restir_reservoir_buffer(
    device: &ash::Device,
    allocator: &GpuAllocator,
) -> Result<GpuBuffer> {
    let buffer = GpuBuffer::new(
        device,
        allocator,
        std::mem::size_of::<GpuAreaRestirReservoir>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "vpt_disabled_area_restir_reservoir",
    )?;
    let mut invalid_reservoir = GpuAreaRestirReservoir::zeroed();
    invalid_reservoir.sample_state.subpixel_uv = [-1.0, -1.0];
    invalid_reservoir.sample_state.lens_uv = [-1.0, -1.0];
    invalid_reservoir.sample_state.path_sample = u32::MAX;
    write_mapped(buffer.mapped_ptr(), &invalid_reservoir);
    Ok(buffer)
}

fn create_disabled_restir_reservoir_buffer(
    device: &ash::Device,
    allocator: &GpuAllocator,
) -> Result<GpuBuffer> {
    let buffer = GpuBuffer::new(
        device,
        allocator,
        std::mem::size_of::<GpuRestirDiReservoir>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "vpt_disabled_restir_reservoir",
    )?;
    let invalid_reservoir = GpuRestirDiReservoir {
        sample_light_id: u32::MAX,
        ..GpuRestirDiReservoir::zeroed()
    };
    write_mapped(buffer.mapped_ptr(), &invalid_reservoir);
    Ok(buffer)
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

#[cfg(test)]
mod shader_source_tests;
