use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::vpt::{VptNrdNoisyResources, VptPass};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};

pub struct VptNrdFrontendPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub packed_diff_radiance_hitdist: GpuImage,
    pub packed_spec_radiance_hitdist: GpuImage,
    pub residual_radiance: GpuImage,
    pub material_factors: GpuImage,
}

pub struct VptNrdFrontendGraphInputs<'a> {
    pub frame_slot: usize,
    pub raw_noisy: VptNrdNoisyResources,
    pub profiler: Option<&'a GpuProfiler>,
}

pub struct VptNrdFrontendGraphOutputs {
    pub packed: VptNrdPackedResources,
}

pub struct VptNrdPackedResources {
    pub diff_radiance_hitdist: ResourceHandle,
    pub spec_radiance_hitdist: ResourceHandle,
    pub residual_radiance: ResourceHandle,
    pub material_factors: ResourceHandle,
}

pub struct VptNrdFrontendPassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub spirv_bytes: &'a [u8],
    pub scene_ubo: &'a SceneUniformBuffer,
    pub vpt: &'a VptPass,
}

pub struct VptNrdFrontendPassResizeInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub vpt: &'a VptPass,
}

impl VptNrdFrontendPass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 9] {
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
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdFrontendPassCreateInfo<'_>,
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
                descriptor_count: 8 * frame_count as u32,
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

        let images = match create_frontend_images(device, allocator, info.width, info.height) {
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
            &VptNrdFrontendImageRefs {
                packed_diff_radiance_hitdist: &images.packed_diff_radiance_hitdist,
                packed_spec_radiance_hitdist: &images.packed_spec_radiance_hitdist,
                residual_radiance: &images.residual_radiance,
                material_factors: &images.material_factors,
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
            packed_diff_radiance_hitdist: images.packed_diff_radiance_hitdist,
            packed_spec_radiance_hitdist: images.packed_spec_radiance_hitdist,
            residual_radiance: images.residual_radiance,
            material_factors: images.material_factors,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptNrdFrontendPassResizeInfo<'_>,
    ) -> Result<()> {
        let new_images = create_frontend_images(device, allocator, info.width, info.height)?;
        let old_images = VptNrdFrontendImages {
            packed_diff_radiance_hitdist: std::mem::replace(
                &mut self.packed_diff_radiance_hitdist,
                new_images.packed_diff_radiance_hitdist,
            ),
            packed_spec_radiance_hitdist: std::mem::replace(
                &mut self.packed_spec_radiance_hitdist,
                new_images.packed_spec_radiance_hitdist,
            ),
            residual_radiance: std::mem::replace(
                &mut self.residual_radiance,
                new_images.residual_radiance,
            ),
            material_factors: std::mem::replace(
                &mut self.material_factors,
                new_images.material_factors,
            ),
        };
        old_images.destroy(device, allocator);
        self.update_input_images(device, info.scene_ubo, info.vpt);
        Ok(())
    }

    pub fn update_input_images(
        &self,
        device: &ash::Device,
        scene_ubo: &SceneUniformBuffer,
        vpt: &VptPass,
    ) {
        write_descriptor_sets(
            device,
            &self.descriptor_sets,
            scene_ubo,
            vpt,
            &VptNrdFrontendImageRefs {
                packed_diff_radiance_hitdist: &self.packed_diff_radiance_hitdist,
                packed_spec_radiance_hitdist: &self.packed_spec_radiance_hitdist,
                residual_radiance: &self.residual_radiance,
                material_factors: &self.material_factors,
            },
        );
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let extent = self.packed_diff_radiance_hitdist.extent;

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
        inputs: VptNrdFrontendGraphInputs<'a>,
    ) -> VptNrdFrontendGraphOutputs {
        let VptNrdFrontendGraphInputs {
            frame_slot,
            raw_noisy,
            profiler,
        } = inputs;
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let packed_diff_resource = graph.import_image_with_access(
            self.packed_diff_radiance_hitdist.handle,
            self.packed_diff_radiance_hitdist.extent.width,
            self.packed_diff_radiance_hitdist.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );
        let packed_spec_resource = graph.import_image_with_access(
            self.packed_spec_radiance_hitdist.handle,
            self.packed_spec_radiance_hitdist.extent.width,
            self.packed_spec_radiance_hitdist.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );
        let residual_resource = graph.import_image_with_access(
            self.residual_radiance.handle,
            self.residual_radiance.extent.width,
            self.residual_radiance.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );
        let material_resource = graph.import_image_with_access(
            self.material_factors.handle,
            self.material_factors.extent.width,
            self.material_factors.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            AccessKind::Undefined,
        );

        let packed_writes = graph.add_pass("vpt_nrd_frontend", QueueType::Compute, |builder| {
            builder.read_as(
                raw_noisy.diff_radiance_hitdist,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(
                raw_noisy.spec_radiance_hitdist,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(raw_noisy.residual_radiance, AccessKind::ComputeShaderRead);
            builder.read_as(raw_noisy.material_factors, AccessKind::ComputeShaderRead);
            builder.write_as(packed_diff_resource, AccessKind::ComputeShaderWrite);
            builder.write_as(packed_spec_resource, AccessKind::ComputeShaderWrite);
            builder.write_as(residual_resource, AccessKind::ComputeShaderWrite);
            builder.write_as(material_resource, AccessKind::ComputeShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptNrdFrontend,
                    );
                }
                self.record(ctx.device, ctx.command_buffer, frame_slot);
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::VptNrdFrontend,
                    );
                }
            })
        });

        VptNrdFrontendGraphOutputs {
            packed: VptNrdPackedResources {
                diff_radiance_hitdist: packed_writes[0],
                spec_radiance_hitdist: packed_writes[1],
                residual_radiance: packed_writes[2],
                material_factors: packed_writes[3],
            },
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.packed_diff_radiance_hitdist.destroy(device, allocator);
        self.packed_spec_radiance_hitdist.destroy(device, allocator);
        self.residual_radiance.destroy(device, allocator);
        self.material_factors.destroy(device, allocator);
    }
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    DescriptorLayoutBuilder::new()
        .add_binding_specs(&VptNrdFrontendPass::descriptor_binding_specs())
        .build(device)
}

struct VptNrdFrontendImages {
    packed_diff_radiance_hitdist: GpuImage,
    packed_spec_radiance_hitdist: GpuImage,
    residual_radiance: GpuImage,
    material_factors: GpuImage,
}

struct VptNrdFrontendImageRefs<'a> {
    packed_diff_radiance_hitdist: &'a GpuImage,
    packed_spec_radiance_hitdist: &'a GpuImage,
    residual_radiance: &'a GpuImage,
    material_factors: &'a GpuImage,
}

impl VptNrdFrontendImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.packed_diff_radiance_hitdist.destroy(device, allocator);
        self.packed_spec_radiance_hitdist.destroy(device, allocator);
        self.residual_radiance.destroy(device, allocator);
        self.material_factors.destroy(device, allocator);
    }
}

fn create_frontend_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptNrdFrontendImages> {
    let packed_diff_radiance_hitdist = create_frontend_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_packed_diff_radiance_hitdist",
    )?;
    let packed_spec_radiance_hitdist = match create_frontend_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_packed_spec_radiance_hitdist",
    ) {
        Ok(image) => image,
        Err(error) => {
            packed_diff_radiance_hitdist.destroy(device, allocator);
            return Err(error);
        }
    };
    let residual_radiance = match create_frontend_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_frontend_residual_radiance",
    ) {
        Ok(image) => image,
        Err(error) => {
            packed_spec_radiance_hitdist.destroy(device, allocator);
            packed_diff_radiance_hitdist.destroy(device, allocator);
            return Err(error);
        }
    };
    let material_factors = match create_frontend_image(
        device,
        allocator,
        width,
        height,
        "vpt_nrd_frontend_material_factors",
    ) {
        Ok(image) => image,
        Err(error) => {
            residual_radiance.destroy(device, allocator);
            packed_spec_radiance_hitdist.destroy(device, allocator);
            packed_diff_radiance_hitdist.destroy(device, allocator);
            return Err(error);
        }
    };

    Ok(VptNrdFrontendImages {
        packed_diff_radiance_hitdist,
        packed_spec_radiance_hitdist,
        residual_radiance,
        material_factors,
    })
}

fn create_frontend_image(
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
    vpt: &VptPass,
    images: &VptNrdFrontendImageRefs<'_>,
) {
    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(scene_ubo.buffer_handle(set_idx))
            .offset(0)
            .range(std::mem::size_of::<GpuSceneUniforms>() as u64);
        let image_refs = [
            &vpt.nrd_diff_radiance_hitdist,
            &vpt.nrd_spec_radiance_hitdist,
            &vpt.nrd_residual_radiance,
            &vpt.nrd_material_factors,
            images.packed_diff_radiance_hitdist,
            images.packed_spec_radiance_hitdist,
            images.residual_radiance,
            images.material_factors,
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
