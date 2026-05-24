use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::vpt_nrd_confidence::VptNrdConfidenceResources;
use crate::render::passes::vpt_nrd_frontend::VptNrdPackedResources;
use crate::render::passes::vpt_surface::VptCurrentSurfaceResources;
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::SceneUniformBuffer;

pub struct VptNrdAdapterPass {
    pub nrd_diff_radiance_hitdist: GpuImage,
    pub nrd_validation: GpuImage,
}

pub struct VptNrdAdapterGraphInputs<'a> {
    pub frame_slot: usize,
    pub packed: VptNrdPackedResources,
    pub confidence: VptNrdConfidenceResources,
    pub surface_inputs: VptCurrentSurfaceResources,
    pub profiler: Option<&'a GpuProfiler>,
}

#[derive(Clone, Copy)]
pub struct VptNrdAdapterGraphOutputs {
    pub resources: VptNrdAdapterResources,
}

#[derive(Clone, Copy)]
pub struct VptNrdAdapterResources {
    pub diff_radiance_hitdist: ResourceHandle,
    pub validation: ResourceHandle,
}

pub struct VptNrdAdapterPassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
}

pub struct VptNrdAdapterPassResizeInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
}

impl VptNrdAdapterPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdAdapterPassCreateInfo<'_>,
    ) -> Result<Self> {
        let _ = info.scene_ubo.frame_count();
        let images = create_adapter_images(device, allocator, info.width, info.height)?;
        Ok(Self {
            nrd_diff_radiance_hitdist: images.nrd_diff_radiance_hitdist,
            nrd_validation: images.nrd_validation,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdAdapterPassResizeInfo<'_>,
    ) -> Result<()> {
        let _ = info.scene_ubo.frame_count();
        let new_images = create_adapter_images(device, allocator, info.width, info.height)?;
        let old_images = VptNrdAdapterImages {
            nrd_diff_radiance_hitdist: std::mem::replace(
                &mut self.nrd_diff_radiance_hitdist,
                new_images.nrd_diff_radiance_hitdist,
            ),
            nrd_validation: std::mem::replace(&mut self.nrd_validation, new_images.nrd_validation),
        };
        old_images.destroy(device, allocator);
        Ok(())
    }

    pub fn record(&self, _device: &ash::Device, _cmd: vk::CommandBuffer, _frame_slot: usize) {}

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        inputs: VptNrdAdapterGraphInputs<'a>,
    ) -> VptNrdAdapterGraphOutputs {
        let VptNrdAdapterGraphInputs {
            frame_slot,
            packed,
            confidence,
            surface_inputs,
            profiler,
        } = inputs;
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let diff_resource = graph.import_image_with_access(
            self.nrd_diff_radiance_hitdist.handle,
            self.nrd_diff_radiance_hitdist.extent.width,
            self.nrd_diff_radiance_hitdist.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );
        let validation_resource = graph.import_image_with_access(
            self.nrd_validation.handle,
            self.nrd_validation.extent.width,
            self.nrd_validation.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );

        let adapter_writes = graph.add_pass("vpt_nrd_adapter", QueueType::Compute, |builder| {
            builder.read_as(packed.diff_radiance_hitdist, AccessKind::ComputeShaderRead);
            builder.read_as(packed.spec_radiance_hitdist, AccessKind::ComputeShaderRead);
            builder.read_as(packed.residual_radiance, AccessKind::ComputeShaderRead);
            builder.read_as(packed.material_factors, AccessKind::ComputeShaderRead);
            builder.read_as(confidence.diff_confidence, AccessKind::ComputeShaderRead);
            builder.read_as(confidence.spec_confidence, AccessKind::ComputeShaderRead);
            builder.read_as(
                surface_inputs.normal_roughness,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(
                surface_inputs.material_roughness,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(surface_inputs.view_z, AccessKind::ComputeShaderRead);
            builder.read_as(surface_inputs.motion_history, AccessKind::ComputeShaderRead);
            builder.write_as(diff_resource, AccessKind::ComputeShaderWrite);
            builder.write_as(validation_resource, AccessKind::ComputeShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptNrdAdapter,
                    );
                }
                self.record(ctx.device, ctx.command_buffer, frame_slot);
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptNrdAdapter,
                    );
                }
            })
        });

        VptNrdAdapterGraphOutputs {
            resources: VptNrdAdapterResources {
                diff_radiance_hitdist: adapter_writes[0],
                validation: adapter_writes[1],
            },
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.nrd_diff_radiance_hitdist.destroy(device, allocator);
        self.nrd_validation.destroy(device, allocator);
    }
}

struct VptNrdAdapterImages {
    nrd_diff_radiance_hitdist: GpuImage,
    nrd_validation: GpuImage,
}

impl VptNrdAdapterImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.nrd_diff_radiance_hitdist.destroy(device, allocator);
        self.nrd_validation.destroy(device, allocator);
    }
}

fn create_adapter_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptNrdAdapterImages> {
    let nrd_diff_radiance_hitdist = create_adapter_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_adapter_diff_radiance_hitdist",
    )?;
    let nrd_validation = match create_adapter_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_adapter_validation",
    ) {
        Ok(image) => image,
        Err(error) => {
            nrd_diff_radiance_hitdist.destroy(device, allocator);
            return Err(error);
        }
    };
    Ok(VptNrdAdapterImages {
        nrd_diff_radiance_hitdist,
        nrd_validation,
    })
}

fn create_adapter_image(
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
