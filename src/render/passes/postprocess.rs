use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};

pub struct PostprocessPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_image: GpuImage,
}

#[derive(Clone, Copy)]
pub struct PostprocessGraphOutputs {
    pub output: ResourceHandle,
}

pub struct PostprocessGraphInputs<'a> {
    pub device: &'a ash::Device,
    pub frame_slot: usize,
    pub input_radiance: ResourceHandle,
    pub hdr_image: &'a GpuImage,
    pub output_initialized: bool,
    pub profiler: Option<&'a GpuProfiler>,
}

impl PostprocessPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        spirv_bytes: &[u8],
        hdr_image: &GpuImage,
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
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 2 * frame_count as u32,
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

        let output_image = match GpuImage::new(
            device,
            allocator,
            &GpuImageDesc {
                width,
                height,
                depth: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "postprocess_output",
            },
        ) {
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
            hdr_image,
            &output_image,
        );

        let shader_module = match create_shader_module(device, spirv_bytes) {
            Ok(module) => module,
            Err(error) => {
                output_image.destroy(device, allocator);
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
                output_image.destroy(device, allocator);
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
            output_image,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        hdr_image: &GpuImage,
        scene_ubo: &SceneUniformBuffer,
    ) -> Result<()> {
        let new_output = GpuImage::new(
            device,
            allocator,
            &GpuImageDesc {
                width,
                height,
                depth: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "postprocess_output",
            },
        )?;
        let old_output = std::mem::replace(&mut self.output_image, new_output);
        old_output.destroy(device, allocator);

        write_descriptor_sets(
            device,
            &self.descriptor_sets,
            scene_ubo,
            hdr_image,
            &self.output_image,
        );
        Ok(())
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        let extent = self.output_image.extent;

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
        inputs: PostprocessGraphInputs<'a>,
    ) -> PostprocessGraphOutputs {
        let PostprocessGraphInputs {
            device,
            frame_slot,
            input_radiance,
            hdr_image,
            output_initialized,
            profiler,
        } = inputs;
        self.update_input_image(device, hdr_image, frame_slot);
        let output_initial_access = if output_initialized {
            AccessKind::TransferRead
        } else {
            AccessKind::Undefined
        };
        let postprocess_output_resource = graph.import_image_with_access(
            self.output_image.handle,
            self.output_image.extent.width,
            self.output_image.extent.height,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            output_initial_access,
        );
        let postprocess_writes = graph.add_pass("postprocess", QueueType::Compute, |builder| {
            builder.read_as(input_radiance, AccessKind::ComputeShaderRead);
            builder.write_as(postprocess_output_resource, AccessKind::ComputeShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::Postprocess,
                    );
                }
                self.record(ctx.device, ctx.command_buffer, frame_slot);
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::Postprocess,
                    );
                }
            })
        });

        PostprocessGraphOutputs {
            output: postprocess_writes[0],
        }
    }

    pub fn update_input_image(
        &self,
        device: &ash::Device,
        hdr_image: &GpuImage,
        frame_slot: usize,
    ) {
        let Some(&ds) = self.descriptor_sets.get(frame_slot) else {
            return;
        };
        let hdr_info = vk::DescriptorImageInfo::default()
            .image_view(hdr_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&hdr_info));
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.output_image.destroy(device, allocator);
    }
}

fn write_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    scene_ubo: &SceneUniformBuffer,
    hdr_image: &GpuImage,
    output_image: &GpuImage,
) {
    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(scene_ubo.buffer_handle(set_idx))
            .offset(0)
            .range(std::mem::size_of::<GpuSceneUniforms>() as u64);
        let hdr_info = vk::DescriptorImageInfo::default()
            .image_view(hdr_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let output_info = vk::DescriptorImageInfo::default()
            .image_view(output_image.view)
            .image_layout(vk::ImageLayout::GENERAL);

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&hdr_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(&output_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}

#[cfg(test)]
mod shader_source_tests;
