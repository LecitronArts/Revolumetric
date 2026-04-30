use anyhow::Result;
use ash::vk;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::primary_ray::PrimaryRayPass;
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::rc_probe_buffer::RcProbeBuffer;
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct LightingPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_image: GpuImage,
}

impl LightingPass {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        spirv_bytes: &[u8],
        primary_ray: &PrimaryRayPass,
        ucvh_gpu: &UcvhGpuResources,
        scene_ubo: &SceneUniformBuffer,
        rc_probes: &RcProbeBuffer,
    ) -> Result<Self> {
        // Descriptor layout: 1 UBO + 3 G-buffer (storage image) + 1 output (storage image) + 4 UCVH (SSBO)
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
                descriptor_count: 4 * frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 5 * frame_count as u32,
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

        // Output image (same size as G-buffer, RGBA8 for final color)
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
                name: "lighting_output",
            },
        ) {
            Ok(image) => image,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        // UCVH buffers: config, l0, occupancy, materials
        let ucvh_buffers = [
            &ucvh_gpu.config_buffer,
            &ucvh_gpu.hierarchy_l0_buffer,
            &ucvh_gpu.occupancy_buffer,
            &ucvh_gpu.material_buffer,
        ];

        for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
            // Binding 0: UBO
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(scene_ubo.buffer_handle(set_idx))
                .offset(0)
                .range(std::mem::size_of::<GpuSceneUniforms>() as u64);

            let ubo_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ubo_info));

            // Bindings 1-3: G-buffer inputs + Binding 4: output
            let image_infos = [
                vk::DescriptorImageInfo::default()
                    .image_view(primary_ray.gbuffer_pos.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(primary_ray.gbuffer0.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(primary_ray.gbuffer1.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(output_image.view)
                    .image_layout(vk::ImageLayout::GENERAL),
            ];

            let image_writes: Vec<vk::WriteDescriptorSet> = image_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding((i + 1) as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(info))
                })
                .collect();

            // Bindings 5-8: UCVH buffers
            let buffer_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers
                .iter()
                .map(|buf| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.handle)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                })
                .collect();

            let buffer_writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding((i + 5) as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            let mut all_writes = vec![ubo_write];
            all_writes.extend(image_writes);
            all_writes.extend(buffer_writes);

            // Binding 9: RC probe buffer (read current frame's merged data)
            let rc_info = vk::DescriptorBufferInfo::default()
                .buffer(rc_probes.write_buffer(set_idx))
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let rc_write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&rc_info));
            all_writes.push(rc_write);

            unsafe { device.update_descriptor_sets(&all_writes, &[]) };
        }

        // Pipeline (no push constants)
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
        primary_ray: &PrimaryRayPass,
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
                name: "lighting_output",
            },
        )?;
        let old_output = std::mem::replace(&mut self.output_image, new_output);
        old_output.destroy(device, allocator);

        for &ds in &self.descriptor_sets {
            let image_infos = [
                vk::DescriptorImageInfo::default()
                    .image_view(primary_ray.gbuffer_pos.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(primary_ray.gbuffer0.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(primary_ray.gbuffer1.view)
                    .image_layout(vk::ImageLayout::GENERAL),
                vk::DescriptorImageInfo::default()
                    .image_view(self.output_image.view)
                    .image_layout(vk::ImageLayout::GENERAL),
            ];
            let writes: Vec<vk::WriteDescriptorSet> = image_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding((i + 1) as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(info))
                })
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        Ok(())
    }

    /// Record the lighting pass. Inserts input barriers on G-buffer images before dispatch.
    pub fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        gbuffer_images: [vk::Image; 3], // [gbuffer_pos, gbuffer0, gbuffer1]
    ) {
        let extent = self.output_image.extent;

        // Barrier: G-buffer SHADER_WRITE → SHADER_READ + output to GENERAL
        let mut barriers: Vec<vk::ImageMemoryBarrier> = gbuffer_images
            .iter()
            .map(|&image| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .image(image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    )
            })
            .collect();

        // Output image to GENERAL for compute write
        barriers.push(
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .image(self.output_image.handle)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
        );

        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        // Bind pipeline and per-frame descriptor set
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
        }

        let groups_x = extent.width.div_ceil(8);
        let groups_y = extent.height.div_ceil(8);
        unsafe { device.cmd_dispatch(cmd, groups_x, groups_y, 1) };
    }

    /// Only updates the given frame_slot to avoid writing in-flight descriptor sets.
    pub fn update_rc_descriptor(
        &self,
        device: &ash::Device,
        rc_probes: &RcProbeBuffer,
        frame_slot: usize,
    ) {
        let ds = self.descriptor_sets[frame_slot];
        let rc_info = vk::DescriptorBufferInfo::default()
            .buffer(rc_probes.write_buffer(frame_slot))
            .offset(0)
            .range(vk::WHOLE_SIZE);
        let rc_write = vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(9)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&rc_info));
        unsafe { device.update_descriptor_sets(&[rc_write], &[]) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.output_image.destroy(device, allocator);
    }
}

#[cfg(test)]
mod shader_source_tests {
    fn normalized_source(path_source: &str) -> String {
        path_source.replace("\r\n", "\n")
    }

    #[test]
    fn lighting_shader_uses_rc_probe_quality_for_indirect_integration() {
        let source =
            include_str!("../../../assets/shaders/passes/lighting.slang").replace("\r\n", "\n");
        let shared = include_str!("../../../assets/shaders/shared/radiance_cascade.slang")
            .replace("\r\n", "\n");

        assert!(shared.contains("float3 integrate_probe_fast("));
        assert!(shared.contains("float3 integrate_probe_quality("));
        assert!(source.contains("scene.rc_probe_quality"));
        assert!(source.contains("integrate_probe_quality(position, rc_normal, rc_probes, scene.rc_c0_offset, c0_grid, scene.rc_probe_quality)"));
    }

    #[test]
    fn fast_probe_face_selection_uses_dominant_axis_normals() {
        let shared = include_str!("../../../assets/shaders/shared/radiance_cascade.slang")
            .replace("\r\n", "\n");
        let normal_to_face_start = shared
            .find("uint normal_to_face(float3 n)")
            .expect("radiance cascade shader should define normal_to_face");
        let face_normals_start = shared
            .find("// Face normals lookup for cosine weighting.")
            .expect("normal_to_face should appear before face normal lookup");
        let normal_to_face = &shared[normal_to_face_start..face_normals_start];

        assert!(normal_to_face.contains("float3 a = abs(n);"));
        assert!(normal_to_face.contains("if (a.x >= a.y && a.x >= a.z)"));
        assert!(normal_to_face.contains("if (a.y >= a.z)"));
        assert!(normal_to_face.contains("return (n.x >= 0.0) ? 0u : 1u;"));
        assert!(normal_to_face.contains("return (n.y >= 0.0) ? 2u : 3u;"));
        assert!(normal_to_face.contains("return (n.z >= 0.0) ? 4u : 5u;"));
        assert!(!normal_to_face.contains("return encode_normal_id(n);"));
    }

    #[test]
    fn descriptor_ubo_range_uses_rust_uniform_size() {
        let source = std::fs::read_to_string("src/render/passes/lighting.rs")
            .expect("lighting pass source should be readable");
        let hardcoded_range = [".range(", "176)"].join("");

        assert!(source.contains("std::mem::size_of::<GpuSceneUniforms>() as u64"));
        assert!(!source.contains(&hardcoded_range));
    }

    #[test]
    fn rc_gradient_normal_does_not_override_unrelated_hit_face() {
        let source = normalized_source(include_str!(
            "../../../assets/shaders/passes/lighting.slang"
        ));

        source
            .find("float3 gradient_normal = compute_occupancy_gradient_normal")
            .expect("lighting shader should compute the optional RC gradient normal");
        source
            .find("if (dot(gradient_normal, axis_normal) > 0.5)")
            .expect("gradient normal must be gated by the actual hit face normal");
        source
            .find("return axis_normal;")
            .expect("lighting shader should fall back to DDA hit face normal");
    }

    #[test]
    fn storage_image_formats_are_explicit_for_shader_abi() {
        let primary = normalized_source(include_str!(
            "../../../assets/shaders/passes/primary_ray.slang"
        ));
        let lighting = normalized_source(include_str!(
            "../../../assets/shaders/passes/lighting.slang"
        ));
        let test_pattern = normalized_source(include_str!(
            "../../../assets/shaders/passes/test_pattern.slang"
        ));

        assert!(
            primary.contains("[[vk::image_format(\"rgba32f\")]]\nRWTexture2D<float4> gbuffer_pos;"),
            "primary_ray gbuffer_pos must declare rgba32f storage image format"
        );
        assert!(
            lighting
                .contains("[[vk::image_format(\"rgba32f\")]]\nRWTexture2D<float4> gbuffer_pos;"),
            "lighting gbuffer_pos must declare rgba32f storage image format"
        );
        assert!(
            primary.contains("[[vk::image_format(\"rgba8\")]]\nRWTexture2D<float4> gbuffer0;"),
            "primary_ray gbuffer0 must declare rgba8 storage image format"
        );
        assert!(
            lighting.contains("[[vk::image_format(\"rgba8\")]]\nRWTexture2D<float4> gbuffer0;"),
            "lighting gbuffer0 must declare rgba8 storage image format"
        );
        assert!(
            lighting.contains("[[vk::image_format(\"rgba8\")]]\nRWTexture2D<float4> output_image;"),
            "lighting output_image must declare rgba8 storage image format"
        );
        assert!(
            test_pattern
                .contains("[[vk::image_format(\"rgba8\")]]\nRWTexture2D<float4> output_image;"),
            "test_pattern output_image must declare rgba8 storage image format"
        );
        assert!(
            primary.contains("[[vk::image_format(\"rgba8ui\")]]\nRWTexture2D<uint4> gbuffer1;"),
            "primary_ray gbuffer1 must keep rgba8ui storage image format"
        );
        assert!(
            lighting.contains("[[vk::image_format(\"rgba8ui\")]]\nRWTexture2D<uint4> gbuffer1;"),
            "lighting gbuffer1 must keep rgba8ui storage image format"
        );
    }

    #[test]
    fn lighting_shader_uses_runtime_debug_view_instead_of_manual_uncommenting() {
        let common = normalized_source(include_str!(
            "../../../assets/shaders/shared/scene_common.slang"
        ));
        let lighting = normalized_source(include_str!(
            "../../../assets/shaders/passes/lighting.slang"
        ));

        assert!(
            common.contains("LIGHTING_DEBUG_VIEW_SHIFT"),
            "scene common shader constants must define the debug-view bit packing"
        );
        assert!(
            common.contains("uint lighting_debug_view(uint flags)"),
            "scene common shader must expose a helper for decoding debug view bits"
        );
        assert!(
            lighting.contains("uint debug_view = lighting_debug_view(scene.lighting_flags);"),
            "lighting shader must decode debug view from the scene flags at runtime"
        );
        assert!(
            lighting.contains("debug_view == LIGHTING_DEBUG_VIEW_RC_INDIRECT"),
            "lighting shader must support RC indirect/ambient-only debug output"
        );
        assert!(
            lighting.contains("debug_view == LIGHTING_DEBUG_VIEW_DIRECT_DIFFUSE"),
            "lighting shader must support direct diffuse-only debug output"
        );
        assert!(
            lighting.contains("debug_view == LIGHTING_DEBUG_VIEW_NORMAL"),
            "lighting shader must support normal visualization output"
        );
        assert!(
            !lighting.contains("DEBUG: uncomment"),
            "debug views should be runtime-switchable, not source edits"
        );
    }
}
