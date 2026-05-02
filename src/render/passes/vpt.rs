use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct VptPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_image: GpuImage,
}

impl VptPass {
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
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .add_binding(
                3,
                vk::DescriptorType::STORAGE_BUFFER,
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
            .build(device)?;

        let frame_count = scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: frame_count as u32,
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

        let output_image = match create_accumulation_image(device, allocator, width, height) {
            Ok(image) => image,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        write_descriptor_sets(device, &descriptor_sets, scene_ubo, &output_image, ucvh_gpu);

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
        scene_ubo: &SceneUniformBuffer,
        ucvh_gpu: &UcvhGpuResources,
    ) -> Result<()> {
        let new_output = create_accumulation_image(device, allocator, width, height)?;
        let old_output = std::mem::replace(&mut self.output_image, new_output);
        old_output.destroy(device, allocator);
        write_descriptor_sets(
            device,
            &self.descriptor_sets,
            scene_ubo,
            &self.output_image,
            ucvh_gpu,
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

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.output_image.destroy(device, allocator);
    }
}

fn create_accumulation_image(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<GpuImage> {
    GpuImage::new(
        device,
        allocator,
        &GpuImageDesc {
            width,
            height,
            depth: 1,
            format: vk::Format::R16G16B16A16_SFLOAT,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "vpt_accumulation",
        },
    )
}

fn write_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    scene_ubo: &SceneUniformBuffer,
    output_image: &GpuImage,
    ucvh_gpu: &UcvhGpuResources,
) {
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
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(output_image.view)
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
                .image_info(std::slice::from_ref(&image_info)),
        ];
        writes.extend(buffer_infos.iter().enumerate().map(|(idx, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding((idx + 2) as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        }));
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
}

#[cfg(test)]
mod shader_source_tests {
    fn normalized_source(path_source: &str) -> String {
        path_source.replace("\r\n", "\n")
    }

    #[test]
    fn vpt_shader_declares_stochastic_accumulating_reference_path() {
        let source = normalized_source(include_str!("../../../assets/shaders/passes/vpt.slang"));

        assert!(source.contains("RWTexture2D<float4> accumulation_image;"));
        assert!(source.contains("hash_u32("));
        assert!(source.contains("scene.vpt_sample_index"));
        assert!(source.contains("scene.vpt_max_bounces"));
        assert!(source.contains("trace_primary_ray("));
        assert!(source.contains("if (scene.vpt_sample_index != 0u)"));
        assert!(
            source.find("float3 previous = accumulation_image[tid.xy].rgb")
                > source.find("if (scene.vpt_sample_index != 0u)")
        );
        assert!(source.contains("lerp(previous, sample_radiance, 1.0 / sample_count)"));
    }

    #[test]
    fn app_resets_vpt_accumulation_on_resize_and_key_changes() {
        let source = std::fs::read_to_string("src/app.rs")
            .expect("app source should be readable for VPT reset test");

        assert!(source.contains("self.vpt_sample_index = 0;"));
        assert!(source.contains("self.last_vpt_camera_key = None;"));
        assert!(source.contains("if self.vpt_accumulation_needs_init {"));
        assert!(source.contains("fov_y.to_bits()"));
        assert!(source.contains("frame.swapchain_extent.width"));
        assert!(source.contains("self.lighting_settings.vpt_max_bounces"));
        assert!(source.contains("initialized postprocess pass from VPT output"));
        assert!(source.contains("skipping VPT frame until required passes are initialized"));
        assert!(source.contains("graph.has_final_access(AccessKind::Present)"));
        assert!(source.contains("add_swapchain_clear_present_pass"));
        assert!(
            source.contains(
                "fn resize_render_passes(&mut self, width: u32, height: u32) -> Result<()>"
            )
        );
        assert!(source.contains("primary_ray_writes.iter().zip(primary_images.iter())"));
        assert!(source.contains("let dep_handle = primary_ray_writes[1];"));
        assert!(!source.contains("let primary_dep = primary_ray_writes[0];"));
        assert!(source.contains("graph.import_image_with_access("));
        assert!(source.contains("AccessKind::ComputeShaderReadWrite"));
    }

    #[test]
    fn app_keeps_vpt_first_use_sample_zero_until_accumulation_is_written() {
        let source = std::fs::read_to_string("src/app.rs")
            .expect("app source should be readable for VPT first-use test");

        assert!(
            source.contains("let scene_vpt_sample_index = if self.vpt_accumulation_needs_init {")
                && source.contains("vpt_sample_index: scene_vpt_sample_index"),
            "scene UBO must see sample 0 while the accumulation image is still first-use"
        );
        assert!(
            source.contains("if self.vpt_accumulation_needs_init || self.vpt_sample_index == 0"),
            "first-use VPT accumulation must be declared write-only even if internal sample state was advanced"
        );
        assert!(
            source.contains("self.last_vpt_camera_key = None;"),
            "skipped VPT frames must not advance reusable accumulation state"
        );
    }

    #[test]
    fn app_supports_frame_limited_runtime_smoke_validation() {
        let source = std::fs::read_to_string("src/app.rs")
            .expect("app source should be readable for runtime smoke validation test");

        assert!(source.contains("REVOLUMETRIC_EXIT_AFTER_FRAMES"));
        assert!(source.contains("exit_after_frames: parse_exit_after_frames()"));
        assert!(source.contains("if let Some(limit) = self.exit_after_frames"));
        assert!(source.contains("event_loop.exit();"));
    }

    #[test]
    fn app_vpt_path_does_not_register_primary_ray_graph_pass() {
        let source = std::fs::read_to_string("src/app.rs")
            .expect("app source should be readable for VPT graph purity test");
        let render_mode_idx = source
            .find("let use_vpt = self.lighting_settings.render_mode == RenderMode::Vpt;")
            .expect("render mode branch should exist");
        let vpt_branch_idx = source[render_mode_idx..]
            .find("if use_vpt {")
            .map(|idx| render_mode_idx + idx)
            .expect("VPT branch should exist");
        let primary_pass_idx = source
            .find("graph.add_pass(\"primary_ray\"")
            .expect("VCT/raw path should still register primary_ray");

        assert!(
            vpt_branch_idx < primary_pass_idx,
            "VPT graph branch must run before registering primary_ray so VPT does not declare unused G-buffer resources"
        );
        assert!(
            !source.contains("if use_vpt {\n                                        return;\n                                    }"),
            "VPT should avoid registering primary_ray instead of registering a no-op pass"
        );
    }

    #[test]
    fn app_keeps_restir_di_behind_vpt_setting() {
        let source = std::fs::read_to_string("src/app.rs")
            .expect("app source should be readable for ReSTIR-DI app wiring test");
        let compact_source = source.split_whitespace().collect::<String>();

        assert!(source.contains("RestirDiSettings::from_env"));
        assert!(source.contains("restir_di_settings: RestirDiSettings"));
        assert!(source.contains("restir_di_pass: Option<RestirDiPass>"));
        assert!(
            compact_source.contains(
                "self.lighting_settings.render_mode==RenderMode::Vpt&&self.restir_di_settings.enabled"
            ),
            "ReSTIR-DI must stay disabled unless VPT mode and the explicit setting are both enabled"
        );
        assert!(
            compact_source.find("graph.add_pass(\"restir_di_initial\"")
                > compact_source.find(
                    "self.lighting_settings.render_mode==RenderMode::Vpt&&self.restir_di_settings.enabled"
                ),
            "ReSTIR-DI graph passes must be nested behind the VPT+enabled guard"
        );
    }

    #[test]
    fn vpt_does_not_assume_primary_gbuffer_for_restir_di() {
        let design =
            std::fs::read_to_string("docs/superpowers/specs/2026-05-02-restir-di-vpt-design.md")
                .expect("ReSTIR-DI VPT design doc should be readable");

        assert!(
            design.contains(
                "Current VPT mode does not register the primary-ray graph pass before VPT"
            )
        );
        assert!(
            design.contains("must explicitly add a VPT-mode surface-state pass")
                || design
                    .contains("Do not write a ReSTIR-DI pass that silently assumes `gbuffer_pos`")
        );
    }

    #[test]
    fn vpt_accumulation_barrier_is_owned_by_render_graph() {
        let vpt_source =
            std::fs::read_to_string("src/render/passes/vpt.rs").expect("vpt source is readable");
        let app_source = std::fs::read_to_string("src/app.rs").expect("app source is readable");
        let implementation = vpt_source
            .split("#[cfg(test)]")
            .next()
            .expect("implementation section should exist");

        assert!(!implementation.contains("cmd_pipeline_barrier"));
        assert!(!implementation.contains("ImageMemoryBarrier"));
        assert!(app_source.contains("AccessKind::Undefined"));
        assert!(app_source.contains("AccessKind::ComputeShaderRead"));
        assert!(app_source.contains("AccessKind::ComputeShaderReadWrite"));
    }
}
