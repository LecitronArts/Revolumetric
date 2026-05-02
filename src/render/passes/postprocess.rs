use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};

pub struct PostprocessPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_image: GpuImage,
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
mod shader_source_tests {
    fn normalized_source(path_source: &str) -> String {
        path_source.replace("\r\n", "\n")
    }

    #[test]
    fn postprocess_shader_declares_hdr_input_and_ldr_output_abi() {
        let source = normalized_source(include_str!(
            "../../../assets/shaders/passes/postprocess.slang"
        ));

        assert!(
            source.contains("[[vk::image_format(\"rgba16f\")]]\nRWTexture2D<float4> hdr_image;"),
            "postprocess input must be rgba16f HDR storage image"
        );
        assert!(
            source.contains("[[vk::image_format(\"rgba8\")]]\nRWTexture2D<float4> output_image;"),
            "postprocess output must be rgba8 LDR storage image"
        );
    }

    #[test]
    fn postprocess_shader_applies_exposure_aces_and_gamma() {
        let source = normalized_source(include_str!(
            "../../../assets/shaders/passes/postprocess.slang"
        ));

        assert!(
            source.contains("scene.exposure"),
            "postprocess shader must read exposure from SceneUniforms"
        );
        assert!(
            source.contains("aces_tonemap("),
            "postprocess shader must apply ACES tonemapping"
        );
        assert!(
            source.contains("pow(mapped, float3(1.0 / 2.2))"),
            "postprocess shader must apply gamma correction after tonemapping"
        );
    }

    #[test]
    fn postprocess_input_update_is_frame_slot_scoped() {
        let source = std::fs::read_to_string("src/render/passes/postprocess.rs")
            .expect("postprocess source should be readable");

        assert!(source.contains("update_input_image(&self, device: &ash::Device, hdr_image: &GpuImage, frame_slot: usize)"));
        assert!(source.contains("self.descriptor_sets.get(frame_slot)"));
        let start = source
            .find("pub fn update_input_image")
            .expect("update_input_image should exist");
        let end = source[start..]
            .find("pub fn destroy")
            .map(|offset| start + offset)
            .expect("destroy should follow update_input_image");
        let body = &source[start..end];
        assert!(
            !body.contains("for &ds in &self.descriptor_sets"),
            "postprocess input rebinding must not rewrite descriptor sets still in flight"
        );
    }

    #[test]
    fn app_wires_lighting_through_postprocess_before_blit() {
        let source = normalized_source(
            &std::fs::read_to_string("src/app.rs")
                .expect("app source should be readable for render-pipeline source test"),
        );

        assert!(source.contains("postprocess_pass: Option<PostprocessPass>"));
        assert!(source.contains("PostprocessPass::new"));
        assert!(
            source.contains("graph.add_pass(\"postprocess\"")
                || source.contains(
                    "graph.add_pass(\n                                        \"postprocess\""
                )
        );
        assert!(source.contains("GpuProfileScope::Postprocess"));

        let lighting_idx = source
            .find("graph.add_pass(\"lighting\"")
            .expect("lighting graph pass should exist");
        let postprocess_idx = lighting_idx
            + source[lighting_idx..]
                .find("graph.add_pass(\"postprocess\"")
                .or_else(|| {
                    source[lighting_idx..].find(
                        "graph.add_pass(\n                                        \"postprocess\"",
                    )
                })
                .expect("postprocess graph pass should exist after lighting");
        let blit_idx = postprocess_idx
            + source[postprocess_idx..]
                .find("graph.add_pass(\"blit_to_swapchain\"")
                .or_else(|| {
                    source[postprocess_idx..]
                        .find("graph.add_pass(\n                                        \"blit_to_swapchain\"")
                })
                .expect("blit graph pass should exist");

        assert!(lighting_idx < postprocess_idx);
        assert!(postprocess_idx < blit_idx);
    }
}
