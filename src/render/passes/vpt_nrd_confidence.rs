use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::vpt_surface::{
    VptCurrentSurfaceResources, VptPreviousSurfaceResources, VptSurfacePass,
};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};

pub struct VptNrdConfidencePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub diff_confidence: GpuImage,
    pub spec_confidence: GpuImage,
}

pub struct VptNrdConfidenceGraphInputs<'a> {
    pub frame_slot: usize,
    pub surface_inputs: VptCurrentSurfaceResources,
    pub previous_surface_inputs: VptPreviousSurfaceResources,
    pub profiler: Option<&'a GpuProfiler>,
}

pub struct VptNrdConfidenceGraphOutputs {
    pub confidence: VptNrdConfidenceResources,
}

pub struct VptNrdConfidenceResources {
    pub diff_confidence: ResourceHandle,
    pub spec_confidence: ResourceHandle,
}

pub struct VptNrdConfidencePassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub spirv_bytes: &'a [u8],
    pub scene_ubo: &'a SceneUniformBuffer,
    pub vpt_surface: &'a VptSurfacePass,
}

pub struct VptNrdConfidencePassResizeInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub vpt_surface: &'a VptSurfacePass,
}

impl VptNrdConfidencePass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 7] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(6, vk::DescriptorType::STORAGE_IMAGE),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdConfidencePassCreateInfo<'_>,
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
                descriptor_count: 6 * frame_count as u32,
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

        let images = match create_confidence_images(device, allocator, info.width, info.height) {
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
            info.vpt_surface,
            &VptNrdConfidenceImageRefs {
                diff_confidence: &images.diff_confidence,
                spec_confidence: &images.spec_confidence,
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
            diff_confidence: images.diff_confidence,
            spec_confidence: images.spec_confidence,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdConfidencePassResizeInfo<'_>,
    ) -> Result<()> {
        let new_images = create_confidence_images(device, allocator, info.width, info.height)?;
        let old_images = VptNrdConfidenceImages {
            diff_confidence: std::mem::replace(
                &mut self.diff_confidence,
                new_images.diff_confidence,
            ),
            spec_confidence: std::mem::replace(
                &mut self.spec_confidence,
                new_images.spec_confidence,
            ),
        };
        old_images.destroy(device, allocator);
        self.update_input_images(device, info.scene_ubo, info.vpt_surface);
        Ok(())
    }

    pub fn update_input_images(
        &self,
        device: &ash::Device,
        scene_ubo: &SceneUniformBuffer,
        vpt_surface: &VptSurfacePass,
    ) {
        write_descriptor_sets(
            device,
            &self.descriptor_sets,
            scene_ubo,
            vpt_surface,
            &VptNrdConfidenceImageRefs {
                diff_confidence: &self.diff_confidence,
                spec_confidence: &self.spec_confidence,
            },
        );
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let extent = self.diff_confidence.extent;

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
        inputs: VptNrdConfidenceGraphInputs<'a>,
    ) -> VptNrdConfidenceGraphOutputs {
        let VptNrdConfidenceGraphInputs {
            frame_slot,
            surface_inputs,
            previous_surface_inputs,
            profiler,
        } = inputs;
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let diff_resource = graph.import_image_with_access(
            self.diff_confidence.handle,
            self.diff_confidence.extent.width,
            self.diff_confidence.extent.height,
            vk::Format::R16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );
        let spec_resource = graph.import_image_with_access(
            self.spec_confidence.handle,
            self.spec_confidence.extent.width,
            self.spec_confidence.extent.height,
            vk::Format::R16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );

        let diff_confidence_writes =
            graph.add_pass("vpt_nrd_confidence", QueueType::Compute, |builder| {
                builder.read_as(surface_inputs.motion_history, AccessKind::ComputeShaderRead);
                builder.read_as(surface_inputs.motion_flags, AccessKind::ComputeShaderRead);
                builder.read_as(
                    surface_inputs.brick_generation,
                    AccessKind::ComputeShaderRead,
                );
                builder.read_as(
                    previous_surface_inputs.brick_generation,
                    AccessKind::ComputeShaderRead,
                );
                builder.write_as(diff_resource, AccessKind::ComputeShaderWrite);
                builder.write_as(spec_resource, AccessKind::ComputeShaderWrite);
                Box::new(move |ctx| {
                    if let Some(profiler) = profiler {
                        profiler.begin_scope(
                            ctx.device,
                            ctx.command_buffer,
                            frame_slot,
                            GpuProfileScope::VptNrdConfidence,
                        );
                    }
                    self.record(ctx.device, ctx.command_buffer, frame_slot);
                    if let Some(profiler) = profiler {
                        profiler.end_scope(
                            ctx.device,
                            ctx.command_buffer,
                            frame_slot,
                            GpuProfileScope::VptNrdConfidence,
                        );
                    }
                })
            });

        VptNrdConfidenceGraphOutputs {
            confidence: VptNrdConfidenceResources {
                diff_confidence: diff_confidence_writes[0],
                spec_confidence: diff_confidence_writes[1],
            },
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.diff_confidence.destroy(device, allocator);
        self.spec_confidence.destroy(device, allocator);
    }
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    DescriptorLayoutBuilder::new()
        .add_binding_specs(&VptNrdConfidencePass::descriptor_binding_specs())
        .build(device)
}

struct VptNrdConfidenceImages {
    diff_confidence: GpuImage,
    spec_confidence: GpuImage,
}

struct VptNrdConfidenceImageRefs<'a> {
    diff_confidence: &'a GpuImage,
    spec_confidence: &'a GpuImage,
}

impl VptNrdConfidenceImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.diff_confidence.destroy(device, allocator);
        self.spec_confidence.destroy(device, allocator);
    }
}

fn create_confidence_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptNrdConfidenceImages> {
    let diff_confidence =
        create_confidence_image(device, allocator, width, height, "vpt_nrd_diff_confidence")?;
    let spec_confidence = match create_confidence_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_spec_confidence",
    ) {
        Ok(image) => image,
        Err(error) => {
            diff_confidence.destroy(device, allocator);
            return Err(error);
        }
    };

    Ok(VptNrdConfidenceImages {
        diff_confidence,
        spec_confidence,
    })
}

fn create_confidence_image(
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
            format: vk::Format::R16_SFLOAT,
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
    vpt_surface: &VptSurfacePass,
    images: &VptNrdConfidenceImageRefs<'_>,
) {
    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(scene_ubo.buffer_handle(set_idx))
            .offset(0)
            .range(std::mem::size_of::<GpuSceneUniforms>() as u64);
        let image_refs = [
            &vpt_surface.motion_history,
            &vpt_surface.motion_flags,
            &vpt_surface.surface_brick_generation,
            &vpt_surface.previous_surface_brick_generation,
            images.diff_confidence,
            images.spec_confidence,
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
