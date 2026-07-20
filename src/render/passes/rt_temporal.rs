use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{RayTracingPipeline, ShaderBindingTable, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::rt_history::{GpuRtHistoryUniforms, GpuRtSurfacePixel};

pub struct RtTemporalPass {
    ray_tracing_pipeline_loader: ash::khr::ray_tracing_pipeline::Device,
    pipeline: RayTracingPipeline,
    shader_binding_table: ShaderBindingTable,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    temporal_images: [GpuImage; 2],
    surface_history_buffers: [GpuBuffer; 2],
    history_uniform_buffers: Vec<GpuBuffer>,
}

pub struct RtTemporalCreateInfo<'a> {
    pub ray_tracing_pipeline_loader: &'a ash::khr::ray_tracing_pipeline::Device,
    pub rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    pub width: u32,
    pub height: u32,
    pub frame_count: usize,
    pub raygen_spirv: &'a [u8],
}

#[derive(Clone, Copy)]
pub struct RtTemporalGraphOutputs {
    pub temporal_radiance: ResourceHandle,
}

impl RtTemporalPass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 7] {
        [
            DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::SAMPLED_IMAGE),
            DescriptorBindingSpec::ray_tracing(3, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(4, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(5, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(6, vk::DescriptorType::SAMPLED_IMAGE),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: RtTemporalCreateInfo<'_>,
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
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: (3 * frame_count) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: (2 * frame_count) as u32,
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
        let temporal_images =
            match create_temporal_images(device, allocator, info.width, info.height) {
                Ok(images) => images,
                Err(error) => {
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let history_uniform_buffers =
            match create_history_uniform_buffers(device, allocator, frame_count) {
                Ok(buffers) => buffers,
                Err(error) => {
                    destroy_images(temporal_images, device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let surface_history_buffers =
            match create_surface_history_buffers(device, allocator, info.width, info.height) {
                Ok(buffers) => buffers,
                Err(error) => {
                    destroy_buffers(history_uniform_buffers, device, allocator);
                    destroy_images(temporal_images, device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };

        let shader_module = match create_shader_module(device, info.raygen_spirv) {
            Ok(module) => module,
            Err(error) => {
                destroy_buffers(Vec::from(surface_history_buffers), device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                destroy_images(temporal_images, device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error).context("failed to create rt_temporal raygen shader module");
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
                destroy_buffers(Vec::from(surface_history_buffers), device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                destroy_images(temporal_images, device, allocator);
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
                destroy_buffers(Vec::from(surface_history_buffers), device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                destroy_images(temporal_images, device, allocator);
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
            temporal_images,
            surface_history_buffers,
            history_uniform_buffers,
        })
    }

    pub fn width(&self) -> u32 {
        self.temporal_images[0].extent.width
    }

    pub fn height(&self) -> u32 {
        self.temporal_images[0].extent.height
    }

    pub fn current_temporal_image(&self, frame_index: u64) -> &GpuImage {
        &self.temporal_images[current_temporal_index(frame_index)]
    }

    pub fn update_history_uniforms(&self, frame_slot: usize, uniforms: &GpuRtHistoryUniforms) {
        write_mapped(
            self.history_uniform_buffers[frame_slot].mapped_ptr(),
            uniforms,
        );
    }

    pub fn update_frame_descriptors(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        frame_index: u64,
        surface_buffer: &GpuBuffer,
        current_radiance: &GpuImage,
    ) {
        let descriptor_set = self.descriptor_sets[frame_slot];
        let current = self.current_temporal_image(frame_index);
        let previous = &self.temporal_images[previous_temporal_index(frame_index)];
        let current_surface_history =
            &self.surface_history_buffers[current_temporal_index(frame_index)];
        let previous_surface_history =
            &self.surface_history_buffers[previous_temporal_index(frame_index)];
        write_frame_descriptors(
            device,
            descriptor_set,
            RtTemporalFrameDescriptors {
                surface_buffer,
                current,
                previous,
                history_uniform_buffer: &self.history_uniform_buffers[frame_slot],
                current_surface_history,
                previous_surface_history,
                current_radiance,
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        frame_slot: usize,
        frame_index: u64,
        surface: ResourceHandle,
        current_radiance: ResourceHandle,
        temporal_initialized: bool,
        profiler: Option<&'a GpuProfiler>,
    ) -> RtTemporalGraphOutputs {
        let current = self.current_temporal_image(frame_index);
        let previous = &self.temporal_images[previous_temporal_index(frame_index)];
        let current_surface_history_buffer =
            &self.surface_history_buffers[current_temporal_index(frame_index)];
        let previous_surface_history_buffer =
            &self.surface_history_buffers[previous_temporal_index(frame_index)];
        let current_resource = graph.import_image_with_access(
            current.handle,
            current.extent.width,
            current.extent.height,
            current.format,
            self.temporal_image_usage(),
            if temporal_initialized {
                AccessKind::RayTracingShaderWrite
            } else {
                AccessKind::Undefined
            },
        );
        let previous_temporal_radiance = graph.import_image_with_access(
            previous.handle,
            previous.extent.width,
            previous.extent.height,
            previous.format,
            self.temporal_image_usage(),
            if temporal_initialized {
                AccessKind::RayTracingShaderWrite
            } else {
                AccessKind::Undefined
            },
        );
        let current_surface_history = graph.import_buffer_with_access(
            current_surface_history_buffer.handle,
            current_surface_history_buffer.size,
            current_surface_history_buffer.usage,
            if temporal_initialized {
                AccessKind::RayTracingShaderWrite
            } else {
                AccessKind::Undefined
            },
        );
        let previous_surface_history = graph.import_buffer_with_access(
            previous_surface_history_buffer.handle,
            previous_surface_history_buffer.size,
            previous_surface_history_buffer.usage,
            if temporal_initialized {
                AccessKind::RayTracingShaderWrite
            } else {
                AccessKind::Undefined
            },
        );
        let ray_tracing_pipeline_loader = self.ray_tracing_pipeline_loader.clone();
        let pipeline = self.pipeline.handle;
        let pipeline_layout = self.pipeline.layout;
        let descriptor_set = self.descriptor_sets[frame_slot];
        let sbt_regions = self.shader_binding_table.regions();
        let width = current.extent.width;
        let height = current.extent.height;

        let writes = graph.add_pass("rt_temporal", QueueType::RayTracing, |builder| {
            builder.read_as(surface, AccessKind::RayTracingShaderRead);
            builder.read_as(current_radiance, AccessKind::RayTracingShaderRead);
            builder.read_as(previous_temporal_radiance, AccessKind::RayTracingShaderRead);
            builder.read_as(previous_surface_history, AccessKind::RayTracingShaderRead);
            builder.write_as(current_resource, AccessKind::RayTracingShaderWrite);
            builder.write_as(current_surface_history, AccessKind::RayTracingShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::RtTemporal,
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
                        GpuProfileScope::RtTemporal,
                    );
                }
            })
        });

        RtTemporalGraphOutputs {
            temporal_radiance: writes[0],
        }
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let new_temporal = create_temporal_images(device, allocator, width, height)?;
        let new_surface_history =
            match create_surface_history_buffers(device, allocator, width, height) {
                Ok(buffers) => buffers,
                Err(error) => {
                    destroy_images(new_temporal, device, allocator);
                    return Err(error);
                }
            };
        let old_temporal = std::mem::replace(&mut self.temporal_images, new_temporal);
        let old_surface_history =
            std::mem::replace(&mut self.surface_history_buffers, new_surface_history);
        destroy_images(old_temporal, device, allocator);
        destroy_buffers(Vec::from(old_surface_history), device, allocator);
        Ok(())
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.shader_binding_table.destroy(device, allocator);
        self.pipeline.destroy(device);
        destroy_buffers(self.history_uniform_buffers, device, allocator);
        destroy_buffers(Vec::from(self.surface_history_buffers), device, allocator);
        destroy_images(self.temporal_images, device, allocator);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }

    fn temporal_image_usage(&self) -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED
    }
}

fn current_temporal_index(frame_index: u64) -> usize {
    (frame_index as usize) & 1
}

fn previous_temporal_index(frame_index: u64) -> usize {
    current_temporal_index(frame_index) ^ 1
}

fn create_temporal_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<[GpuImage; 2]> {
    let first = create_temporal_image(device, allocator, width, height, "rt_temporal_0")?;
    let second = match create_temporal_image(device, allocator, width, height, "rt_temporal_1") {
        Ok(image) => image,
        Err(error) => {
            first.destroy(device, allocator);
            return Err(error);
        }
    };
    Ok([first, second])
}

fn create_temporal_image(
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
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
            name,
        },
    )
}

fn create_history_uniform_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count);
    for slot in 0..frame_count {
        match GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GpuRtHistoryUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("rt_history_uniforms_{slot}"),
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

fn create_surface_history_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<[GpuBuffer; 2]> {
    let first =
        create_surface_history_buffer(device, allocator, width, height, "rt_surface_history_0")?;
    let second = match create_surface_history_buffer(
        device,
        allocator,
        width,
        height,
        "rt_surface_history_1",
    ) {
        Ok(buffer) => buffer,
        Err(error) => {
            first.destroy(device, allocator);
            return Err(error);
        }
    };
    Ok([first, second])
}

fn create_surface_history_buffer(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
    name: &'static str,
) -> Result<GpuBuffer> {
    GpuBuffer::new(
        device,
        allocator,
        surface_history_buffer_size(width, height),
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
        name,
    )
}

fn surface_history_buffer_size(width: u32, height: u32) -> vk::DeviceSize {
    u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(std::mem::size_of::<GpuRtSurfacePixel>() as u64)
}

struct RtTemporalFrameDescriptors<'a> {
    surface_buffer: &'a GpuBuffer,
    current: &'a GpuImage,
    previous: &'a GpuImage,
    history_uniform_buffer: &'a GpuBuffer,
    current_surface_history: &'a GpuBuffer,
    previous_surface_history: &'a GpuBuffer,
    current_radiance: &'a GpuImage,
}

fn write_frame_descriptors(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    resources: RtTemporalFrameDescriptors<'_>,
) {
    let surface_info = vk::DescriptorBufferInfo::default()
        .buffer(resources.surface_buffer.handle)
        .offset(0)
        .range(resources.surface_buffer.size);
    let current_info = vk::DescriptorImageInfo::default()
        .image_view(resources.current.view)
        .image_layout(vk::ImageLayout::GENERAL);
    let previous_info = vk::DescriptorImageInfo::default()
        .image_view(resources.previous.view)
        .image_layout(vk::ImageLayout::GENERAL);
    let history_info = vk::DescriptorBufferInfo::default()
        .buffer(resources.history_uniform_buffer.handle)
        .offset(0)
        .range(std::mem::size_of::<GpuRtHistoryUniforms>() as u64);
    let current_surface_history_info = vk::DescriptorBufferInfo::default()
        .buffer(resources.current_surface_history.handle)
        .offset(0)
        .range(resources.current_surface_history.size);
    let previous_surface_history_info = vk::DescriptorBufferInfo::default()
        .buffer(resources.previous_surface_history.handle)
        .offset(0)
        .range(resources.previous_surface_history.size);
    let current_radiance_info = vk::DescriptorImageInfo::default()
        .image_view(resources.current_radiance.view)
        .image_layout(vk::ImageLayout::GENERAL);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&surface_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&current_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&previous_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&history_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&current_surface_history_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&previous_surface_history_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&current_radiance_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn destroy_images(images: [GpuImage; 2], device: &ash::Device, allocator: &GpuAllocator) {
    for image in images {
        image.destroy(device, allocator);
    }
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
    fn rt_temporal_shader_source_uses_conservative_reprojection_and_clamp() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        assert!(source.contains("reproject"));
        assert!(source.contains("clamp"));
        assert!(source.contains("rt_history.flags"));
    }

    #[test]
    fn rt_temporal_reprojection_uses_uploaded_row_vector_matrix_convention() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        assert!(
            compact.contains("mul(float4(world_position,1.0),rt_history.previous_view_proj)"),
            "RT temporal reprojection must match VPT's row-vector view-projection convention"
        );
        assert!(
            !compact.contains("mul(rt_history.previous_view_proj,float4(world_position,1.0))"),
            "RT temporal reprojection must not use the opposite matrix-vector convention"
        );
    }

    #[test]
    fn rt_temporal_shader_is_driven_by_history_settings_not_fixed_blend() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        assert!(source.contains("temporal_denoise_enabled"));
        assert!(source.contains("history_length"));
        assert!(source.contains("bounded_history_alpha"));
        assert!(
            compact.contains("if(!temporal_enabled"),
            "RT temporal shader must bypass accumulation when disabled"
        );
        assert!(
            !source.contains("0.9"),
            "RT temporal blend must not be a fixed 0.9 history weight"
        );
    }

    #[test]
    fn rt_temporal_shader_rejects_invalid_reprojection_and_debug_history() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "boolrt_temporal_debug_bypasses_history",
            "rt_history.debug_view",
            "previous_pixel_f.x>=0.0",
            "previous_pixel_f.x<float(previous_extent.x)",
            "previous_pixel_f.y>=0.0",
            "previous_pixel_f.y<float(previous_extent.y)",
            "surface.history_confidence>0.0",
            "surface.hit_kind==RT_SURFACE_HIT_KIND_VOXEL",
            "surface.motion_history.w>0.0",
        ] {
            assert!(
                compact.contains(token),
                "RT temporal shader must include history rejection token {token}"
            );
        }
    }

    #[test]
    fn rt_temporal_debug_view_visualizes_history_reuse() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "if(rt_history.debug_view==RT_DEBUG_VIEW_TEMPORAL)",
            "if(rt_history.debug_view==RT_DEBUG_VIEW_TEMPORAL&&!history_valid)",
            "temporal_radiance[launch_id.xy]=float4(0.0,0.0,motion_history_valid?1.0:0.0,1.0);",
            "float3temporal_debug=rt_temporal_visualize_reuse(",
            "1.0-alpha",
            "length(blended-current)",
            "motion_history_valid?1.0:0.0",
            "temporal_radiance[launch_id.xy]=float4(temporal_debug,1.0);",
        ] {
            assert!(
                compact.contains(token),
                "RT temporal debug view must expose history reuse token {token}"
            );
        }
    }

    #[test]
    fn rt_temporal_motion_debug_view_encodes_reprojection_delta() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "float3rt_temporal_visualize_motion(float4motion)",
            "motion.xy",
            "motion.w>0.0",
            "float2encoded_delta=motion_valid?motion.xy*0.05+0.5:float2(0.5,0.5)",
            "returnfloat3(encoded_delta,motion_valid?1.0:0.0)",
            "if(rt_history.debug_view==RT_DEBUG_VIEW_MOTION)",
            "temporal_radiance[launch_id.xy]=float4(rt_temporal_visualize_motion(surface.motion_history),1.0);",
        ] {
            assert!(
                compact.contains(token),
                "RT temporal motion debug view must expose reprojection delta with {token}"
            );
        }
    }

    #[test]
    fn rt_temporal_consumes_surface_motion_history_for_previous_pixel_lookup() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "float2previous_pixel_f=surface.motion_history.xy+float2(launch_id.xy)+float2(0.5)",
            "boolmotion_history_valid=surface.motion_history.w>0.0",
            "boolprevious_pixel_valid=valid_resolution&&previous_pixel_f.x>=0.0",
            "&&motion_history_valid",
            "&&previous_pixel_valid",
            "uint2previous_pixel=uint2(tap_pixel_i)",
            "previous_index=previous_pixel.y*previous_extent.x+previous_pixel.x",
        ] {
            assert!(
                compact.contains(token),
                "RT temporal must consume surface motion history token {token}"
            );
        }
    }

    #[test]
    fn rt_temporal_samples_fractional_reprojection_without_upper_left_floor_bias() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "staticconstint2rt_temporal_history_tap_offsets[4]",
            "float2previous_sample=previous_pixel_f-float2(0.5)",
            "int2previous_base_pixel=int2(floor(previous_sample))",
            "float2history_fraction=saturate(previous_sample-float2(previous_base_pixel))",
            "floatrt_temporal_history_tap_weight(uinttap,float2history_fraction)",
            "history_weight_sum+=tap_weight",
            "history/=history_weight_sum",
            "history_valid=history_weight_sum>0.0",
        ] {
            assert!(
                compact.contains(token),
                "RT temporal reprojection must bilinearly sample fractional history and avoid upper-left floor bias with {token}"
            );
        }

        assert!(
            !compact.contains("previous_pixel=uint2(clamp(previous_pixel_f"),
            "RT temporal must not truncate fractional previous pixels directly, which biases history toward the upper-left"
        );
    }

    #[test]
    fn rt_temporal_shader_uses_previous_surface_metadata_for_history_compatibility() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "StructuredBuffer<RtSurfacePixel>previous_surface_history",
            "RWStructuredBuffer<RtSurfacePixel>current_surface_history",
            "current_surface_history[index]=surface",
            "boolrt_surfaces_compatible(RtSurfacePixelcurrent,RtSurfacePixelprevious)",
            "previous.hit_kind==RT_SURFACE_HIT_KIND_VOXEL",
            "current.material_id!=previous.material_id",
            "dot(normalize(current.normal_roughness.xyz),normalize(previous.normal_roughness.xyz))",
            "rt_history.normal_threshold",
            "distance(current.position_depth.xyz,previous.position_depth.xyz)",
            "rt_history.depth_threshold",
            "rt_surfaces_compatible(surface,previous_surface)",
        ] {
            assert!(
                compact.contains(token),
                "RT temporal shader must use previous surface metadata token {token}"
            );
        }
    }

    #[test]
    fn rt_temporal_history_compatibility_uses_strict_rt_position_threshold() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let common = std::fs::read_to_string("assets/shaders/shared/rt_history_common.slang")
            .expect("rt_history_common.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);
        let common_compact = crate::render::source_checks::compact(&common);

        assert!(
            common_compact.contains("staticconstfloatRT_HISTORY_POSITION_EPSILON=0.05"),
            "RT history compatibility must use a sub-voxel epsilon instead of a one-voxel floor"
        );
        assert!(
            common_compact.contains(
                "floatrt_history_position_threshold(floatlinear_depth,floatdepth_threshold)"
            ),
            "RT history compatibility threshold should be shared across temporal and reservoir reuse"
        );
        assert!(
            compact.contains(
                "position_delta<=rt_history_position_threshold(current.position_depth.w,rt_history.depth_threshold)"
            ),
            "RT temporal must compare surface positions against the strict shared history threshold"
        );
        assert!(
            !compact.contains("position_delta<=max(1.0,depth_scale*rt_history.depth_threshold)"),
            "RT temporal must not keep accepting up to a full voxel of mismatched history"
        );
    }

    #[test]
    fn rt_temporal_descriptor_specs_match_surface_history_and_output_shader_resources() {
        let specs = super::RtTemporalPass::descriptor_binding_specs();
        let actual = specs
            .iter()
            .map(|spec| (spec.binding, spec.descriptor_type))
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (0, vk::DescriptorType::STORAGE_BUFFER),
                (1, vk::DescriptorType::STORAGE_IMAGE),
                (2, vk::DescriptorType::SAMPLED_IMAGE),
                (3, vk::DescriptorType::UNIFORM_BUFFER),
                (4, vk::DescriptorType::STORAGE_BUFFER),
                (5, vk::DescriptorType::STORAGE_BUFFER),
                (6, vk::DescriptorType::SAMPLED_IMAGE),
            ]
        );
    }

    #[test]
    fn rt_temporal_shader_reads_resolved_current_radiance_instead_of_surface_albedo() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "Texture2D<float4>current_radiance",
            "float3current=current_radiance.Load(int3(launch_id.xy,0)).rgb",
        ] {
            assert!(
                compact.contains(token),
                "RT temporal must consume resolved direct-lighting radiance token {token}"
            );
        }
        assert!(
            !compact.contains("float3current=surface.albedo_material.rgb"),
            "RT temporal must not bypass direct-lighting resolve by using raw albedo as current radiance"
        );
    }

    #[test]
    fn rt_temporal_reset_is_frame_local_not_permanent_generation_state() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_temporal.rgen.slang")
            .expect("rt_temporal.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);

        assert!(
            compact.contains("boolreset=rt_history.flags!=0u"),
            "RT temporal reset must be driven by frame-local reset flags"
        );
        assert!(
            !compact.contains("boolreset=rt_history.history_reset_generation!=0u"),
            "RT temporal must not treat the monotonic reset generation as a permanent reset flag"
        );
    }

    #[test]
    fn rt_temporal_accumulates_hdr_without_claiming_nrd_behavior() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_temporal.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT temporal implementation should precede tests");

        for forbidden in ["NRD", "ReBLUR", "RELAX"] {
            assert!(
                !implementation.contains(forbidden),
                "RT temporal pass must remain temporal-only and not claim {forbidden} behavior"
            );
        }
    }

    #[test]
    fn rt_temporal_pass_ping_pongs_surface_metadata_history_buffers() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_temporal.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT temporal implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "surface_history_buffers:[GpuBuffer;2]",
            "create_surface_history_buffers(device,allocator,info.width,info.height)",
            "current_surface_history",
            "previous_surface_history",
            "builder.read_as(previous_surface_history",
            "builder.write_as(current_surface_history",
            "destroy_buffers(Vec::from(self.surface_history_buffers),device,allocator)",
        ] {
            assert!(
                compact.contains(token),
                "RT temporal pass must ping-pong surface metadata history with {token}"
            );
        }
    }

    #[test]
    fn rt_temporal_descriptor_pool_allocates_all_surface_history_bindings() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_temporal.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT temporal implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        assert!(
            compact.contains("descriptor_count:(3*frame_count)asu32"),
            "RT temporal descriptor pool must allocate current, previous, and surface-history storage buffers per frame"
        );
    }

    #[test]
    fn rt_temporal_pass_owns_raygen_pipeline_descriptors_sbt_and_history_buffers() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_temporal.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT temporal implementation should precede tests");

        for token in [
            "RayTracingPipeline",
            "ShaderBindingTable",
            "descriptor_set_layout",
            "descriptor_pool",
            "descriptor_sets",
            "history_uniform_buffers",
            "temporal_images",
            "surface_history_buffers",
            "RayTracingPipeline::new_raygen_only",
            "cmd_trace_rays",
            "update_frame_descriptors",
            "update_history_uniforms",
        ] {
            assert!(
                implementation.contains(token),
                "RT temporal pass implementation missing {token}"
            );
        }
    }

    #[test]
    fn rt_temporal_pass_ping_pongs_current_and_previous_history_images() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_temporal.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT temporal implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "fncurrent_temporal_index(frame_index:u64)->usize",
            "fnprevious_temporal_index(frame_index:u64)->usize",
            "current_temporal_index(frame_index)",
            "previous_temporal_index(frame_index)",
            "previous_temporal_radiance",
        ] {
            assert!(
                compact.contains(token),
                "RT temporal history ping-pong missing {token}"
            );
        }
    }
}
