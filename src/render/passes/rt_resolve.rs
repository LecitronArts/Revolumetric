use anyhow::{Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{RayTracingPipeline, ShaderBindingTable, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRtResolveUniforms {
    pub exposure: f32,
    pub _pad0: [f32; 3],
}

pub struct RtResolvePass {
    ray_tracing_pipeline_loader: ash::khr::ray_tracing_pipeline::Device,
    pipeline: RayTracingPipeline,
    shader_binding_table: ShaderBindingTable,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_buffers: Vec<GpuBuffer>,
    pub output_image: GpuImage,
}

pub struct RtResolveCreateInfo<'a> {
    pub ray_tracing_pipeline_loader: &'a ash::khr::ray_tracing_pipeline::Device,
    pub rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    pub width: u32,
    pub height: u32,
    pub frame_count: usize,
    pub raygen_spirv: &'a [u8],
}

#[derive(Clone, Copy)]
pub struct RtResolveGraphOutputs {
    pub output: ResourceHandle,
}

impl RtResolvePass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 3] {
        [
            DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::SAMPLED_IMAGE),
            DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::UNIFORM_BUFFER),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: RtResolveCreateInfo<'_>,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding_specs(&Self::descriptor_binding_specs())
            .build(device)?;
        let frame_count = info.frame_count;
        let descriptor_pool = match DescriptorPool::new(
            device,
            frame_count as u32,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: frame_count as u32,
                },
            ],
        ) {
            Ok(pool) => pool,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let descriptor_layouts = vec![descriptor_set_layout; frame_count];
        let descriptor_sets = match descriptor_pool.allocate(device, descriptor_layouts.as_slice())
        {
            Ok(sets) => sets,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let output_image = match create_output_image(device, allocator, info.width, info.height) {
            Ok(image) => image,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let uniform_buffers = match create_uniform_buffers(device, allocator, frame_count) {
            Ok(buffers) => buffers,
            Err(error) => {
                output_image.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let shader_module = match create_shader_module(device, info.raygen_spirv) {
            Ok(module) => module,
            Err(error) => {
                destroy_buffers(uniform_buffers, device, allocator);
                output_image.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error).context("failed to create rt_resolve raygen shader module");
            }
        };
        let pipeline = match RayTracingPipeline::new_raygen_only(
            device,
            info.ray_tracing_pipeline_loader,
            shader_module,
            c"main",
            &[descriptor_set_layout],
            &[],
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe { device.destroy_shader_module(shader_module, None) };
                destroy_buffers(uniform_buffers, device, allocator);
                output_image.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        unsafe { device.destroy_shader_module(shader_module, None) };
        let shader_binding_table = match ShaderBindingTable::new(
            device,
            allocator,
            info.ray_tracing_pipeline_loader,
            pipeline.handle,
            info.rt_pipeline_properties,
            pipeline.group_counts,
        ) {
            Ok(table) => table,
            Err(error) => {
                pipeline.destroy(device);
                destroy_buffers(uniform_buffers, device, allocator);
                output_image.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        Ok(Self {
            ray_tracing_pipeline_loader: info.ray_tracing_pipeline_loader.clone(),
            pipeline,
            shader_binding_table,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            uniform_buffers,
            output_image,
        })
    }

    pub fn width(&self) -> u32 {
        self.output_image.extent.width
    }

    pub fn height(&self) -> u32 {
        self.output_image.extent.height
    }

    pub fn update_uniforms(&self, frame_slot: usize, exposure: f32) {
        let exposure = if exposure.is_finite() && exposure >= 0.0 {
            exposure
        } else {
            1.0
        };
        let uniforms = GpuRtResolveUniforms {
            exposure,
            _pad0: [0.0; 3],
        };
        write_mapped(self.uniform_buffers[frame_slot].mapped_ptr(), &uniforms);
    }

    pub fn update_input_descriptor(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        temporal_radiance: &GpuImage,
    ) {
        let descriptor_set = self.descriptor_sets[frame_slot];
        write_input_descriptor(
            device,
            descriptor_set,
            temporal_radiance,
            &self.output_image,
            &self.uniform_buffers[frame_slot],
        );
    }

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        temporal_radiance: ResourceHandle,
        frame_slot: usize,
        output_initialized: bool,
        profiler: Option<&'a GpuProfiler>,
    ) -> RtResolveGraphOutputs {
        let output_resource = graph.import_image_with_access(
            self.output_image.handle,
            self.output_image.extent.width,
            self.output_image.extent.height,
            self.output_image.format,
            self.output_image_usage(),
            if output_initialized {
                AccessKind::TransferRead
            } else {
                AccessKind::Undefined
            },
        );
        let ray_tracing_pipeline_loader = self.ray_tracing_pipeline_loader.clone();
        let pipeline = self.pipeline.handle;
        let pipeline_layout = self.pipeline.layout;
        let descriptor_set = self.descriptor_sets[frame_slot];
        let sbt_regions = self.shader_binding_table.regions();
        let width = self.output_image.extent.width;
        let height = self.output_image.extent.height;

        let writes = graph.add_pass("rt_resolve", QueueType::RayTracing, |builder| {
            builder.read_as(temporal_radiance, AccessKind::RayTracingShaderRead);
            builder.write_as(output_resource, AccessKind::RayTracingShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::RtResolve,
                    );
                }
                unsafe {
                    ctx.device.cmd_bind_pipeline(
                        ctx.command_buffer,
                        vk::PipelineBindPoint::RAY_TRACING_KHR,
                        pipeline,
                    );
                    ctx.device.cmd_bind_descriptor_sets(
                        ctx.command_buffer,
                        vk::PipelineBindPoint::RAY_TRACING_KHR,
                        pipeline_layout,
                        0,
                        std::slice::from_ref(&descriptor_set),
                        &[],
                    );
                    ray_tracing_pipeline_loader.cmd_trace_rays(
                        ctx.command_buffer,
                        &sbt_regions.raygen,
                        &sbt_regions.miss,
                        &sbt_regions.hit,
                        &sbt_regions.callable,
                        width,
                        height,
                        1,
                    );
                }
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::RtResolve,
                    );
                }
            })
        });

        RtResolveGraphOutputs { output: writes[0] }
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let new_output = create_output_image(device, allocator, width, height)?;
        std::mem::replace(&mut self.output_image, new_output).destroy(device, allocator);
        Ok(())
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.shader_binding_table.destroy(device, allocator);
        self.pipeline.destroy(device);
        destroy_buffers(self.uniform_buffers, device, allocator);
        self.output_image.destroy(device, allocator);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }

    fn output_image_usage(&self) -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC
    }
}

fn create_uniform_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count);
    for slot in 0..frame_count {
        match GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GpuRtResolveUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("rt_resolve_uniforms_{slot}"),
        ) {
            Ok(buffer) => buffers.push(buffer),
            Err(error) => {
                destroy_buffers(buffers, device, allocator);
                return Err(error);
            }
        }
    }
    Ok(buffers)
}

fn create_output_image(
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
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "rt_resolve",
        },
    )
}

fn write_input_descriptor(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    temporal_radiance: &GpuImage,
    output_image: &GpuImage,
    uniform_buffer: &GpuBuffer,
) {
    let temporal_info = vk::DescriptorImageInfo::default()
        .image_view(temporal_radiance.view)
        .image_layout(vk::ImageLayout::GENERAL);
    let output_info = vk::DescriptorImageInfo::default()
        .image_view(output_image.view)
        .image_layout(vk::ImageLayout::GENERAL);
    let uniform_info = vk::DescriptorBufferInfo::default()
        .buffer(uniform_buffer.handle)
        .offset(0)
        .range(std::mem::size_of::<GpuRtResolveUniforms>() as u64);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&temporal_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&output_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&uniform_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
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
mod shader_source_tests {
    use ash::vk;

    #[test]
    fn rt_resolve_descriptor_specs_match_temporal_input_and_output_shader_resources() {
        let specs = super::RtResolvePass::descriptor_binding_specs();
        let actual = specs
            .iter()
            .map(|spec| (spec.binding, spec.descriptor_type))
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (0, vk::DescriptorType::SAMPLED_IMAGE),
                (1, vk::DescriptorType::STORAGE_IMAGE),
                (2, vk::DescriptorType::UNIFORM_BUFFER),
            ]
        );
    }

    #[test]
    fn rt_resolve_uniform_layout_is_stable() {
        assert_eq!(std::mem::size_of::<super::GpuRtResolveUniforms>(), 16);
        assert_eq!(
            std::mem::offset_of!(super::GpuRtResolveUniforms, exposure),
            0
        );
    }

    #[test]
    fn rt_resolve_shader_applies_display_transform() {
        let shader = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_resolve.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&shader);

        for token in [
            "#include\"lighting_common.slang\"",
            "structRtResolveUniforms{floatexposure;float3_pad0;};",
            "ConstantBuffer<RtResolveUniforms>rt_resolve",
            "float3hdr=max(temporal_radiance.Load(int3(launch_id.xy,0)).rgb*rt_resolve.exposure,float3(0.0));",
            "float3mapped=aces_tonemap(hdr);",
            "float3ldr=pow(mapped,float3(1.0/2.2));",
            "resolved_output[launch_id.xy]=float4(ldr,1.0);",
        ] {
            assert!(
                compact.contains(token),
                "RT resolve must apply the same display transform as postprocess; missing {token}"
            );
        }
        assert!(
            !compact.contains(
                "float3color=temporal_radiance.Load(int3(launch_id.xy,0)).rgb;resolved_output[launch_id.xy]=float4(color,1.0);"
            ),
            "RT resolve must not write linear HDR directly to an rgba8 display target"
        );
    }

    #[test]
    fn rt_resolve_pass_uses_raygen_pipeline_instead_of_clear_fallback() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_resolve.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT resolve implementation should precede tests");

        for token in [
            "RayTracingPipeline",
            "ShaderBindingTable",
            "descriptor_set_layout",
            "descriptor_pool",
            "descriptor_sets",
            "RayTracingPipeline::new_raygen_only",
            "cmd_trace_rays",
            "update_input_descriptor",
            "QueueType::RayTracing",
            "AccessKind::RayTracingShaderWrite",
        ] {
            assert!(
                implementation.contains(token),
                "RT resolve pass implementation missing {token}"
            );
        }
        assert!(
            !implementation.contains("cmd_clear_color_image"),
            "RT resolve should execute its raygen shader instead of clearing black"
        );
    }
}
