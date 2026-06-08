use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::vpt_nrd_frontend::{VptNrdFrontendPass, VptNrdPackedResources};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};

pub struct VptNrdResolvePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub resolved_radiance: GpuImage,
}

#[derive(Clone, Copy)]
pub struct VptNrdResolveGraphInputs<'a> {
    pub frame_slot: usize,
    pub denoised_diff_radiance_hitdist: ResourceHandle,
    pub packed: VptNrdPackedResources,
    pub profiler: Option<&'a GpuProfiler>,
}

#[derive(Clone, Copy)]
pub struct VptNrdResolveGraphOutputs {
    pub resolved_radiance: ResourceHandle,
}

pub struct VptNrdResolvePassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub spirv_bytes: &'a [u8],
    pub scene_ubo: &'a SceneUniformBuffer,
    pub denoised_diff_radiance_hitdist: &'a GpuImage,
    pub frontend: &'a VptNrdFrontendPass,
}

pub struct VptNrdResolvePassResizeInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub denoised_diff_radiance_hitdist: &'a GpuImage,
    pub frontend: &'a VptNrdFrontendPass,
}

impl VptNrdResolvePass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 5] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_IMAGE),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdResolvePassCreateInfo<'_>,
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

        let images = match create_resolve_images(device, allocator, info.width, info.height) {
            Ok(images) => images,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let image_refs = VptNrdResolveImagesRef {
            resolved_radiance: &images.resolved_radiance,
        };
        write_descriptor_sets(
            device,
            &descriptor_sets,
            info.scene_ubo,
            info.denoised_diff_radiance_hitdist,
            info.frontend,
            &image_refs,
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
            resolved_radiance: images.resolved_radiance,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdResolvePassResizeInfo<'_>,
    ) -> Result<()> {
        let new_images = create_resolve_images(device, allocator, info.width, info.height)?;
        let old_images = VptNrdResolveImages {
            resolved_radiance: std::mem::replace(
                &mut self.resolved_radiance,
                new_images.resolved_radiance,
            ),
        };
        old_images.destroy(device, allocator);
        self.update_input_images(
            device,
            info.scene_ubo,
            info.denoised_diff_radiance_hitdist,
            info.frontend,
        );
        Ok(())
    }

    pub fn update_input_images(
        &self,
        device: &ash::Device,
        scene_ubo: &SceneUniformBuffer,
        denoised_diff_radiance_hitdist: &GpuImage,
        frontend: &VptNrdFrontendPass,
    ) {
        write_descriptor_sets(
            device,
            &self.descriptor_sets,
            scene_ubo,
            denoised_diff_radiance_hitdist,
            frontend,
            &VptNrdResolveImagesRef {
                resolved_radiance: &self.resolved_radiance,
            },
        );
    }

    pub fn output_image(&self) -> &GpuImage {
        &self.resolved_radiance
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let extent = self.resolved_radiance.extent;

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
        inputs: VptNrdResolveGraphInputs<'a>,
    ) -> VptNrdResolveGraphOutputs {
        let VptNrdResolveGraphInputs {
            frame_slot,
            denoised_diff_radiance_hitdist,
            packed,
            profiler,
        } = inputs;
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let resolved_resource = graph.import_image_with_access(
            self.resolved_radiance.handle,
            self.resolved_radiance.extent.width,
            self.resolved_radiance.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );

        let resolve_writes = graph.add_pass("vpt_nrd_resolve", QueueType::Compute, |builder| {
            builder.read_as(
                denoised_diff_radiance_hitdist,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(packed.residual_radiance, AccessKind::ComputeShaderRead);
            builder.read_as(packed.material_factors, AccessKind::ComputeShaderRead);
            builder.write_as(resolved_resource, AccessKind::ComputeShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptNrdResolve,
                    );
                }
                self.record(ctx.device, ctx.command_buffer, frame_slot);
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptNrdResolve,
                    );
                }
            })
        });

        VptNrdResolveGraphOutputs {
            resolved_radiance: resolve_writes[0],
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.resolved_radiance.destroy(device, allocator);
    }
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    DescriptorLayoutBuilder::new()
        .add_binding_specs(&VptNrdResolvePass::descriptor_binding_specs())
        .build(device)
}

struct VptNrdResolveImages {
    resolved_radiance: GpuImage,
}

struct VptNrdResolveImagesRef<'a> {
    resolved_radiance: &'a GpuImage,
}

impl VptNrdResolveImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.resolved_radiance.destroy(device, allocator);
    }
}

fn create_resolve_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptNrdResolveImages> {
    let resolved_radiance = GpuImage::new(
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
            name: "vpt_nrd_resolved_radiance",
        },
    )?;

    Ok(VptNrdResolveImages { resolved_radiance })
}

fn write_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    scene_ubo: &SceneUniformBuffer,
    denoised_diff_radiance_hitdist: &GpuImage,
    frontend: &VptNrdFrontendPass,
    images: &VptNrdResolveImagesRef<'_>,
) {
    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(scene_ubo.buffer_handle(set_idx))
            .offset(0)
            .range(std::mem::size_of::<GpuSceneUniforms>() as u64);
        let image_refs = [
            denoised_diff_radiance_hitdist,
            &frontend.residual_radiance,
            &frontend.material_factors,
            images.resolved_radiance,
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
