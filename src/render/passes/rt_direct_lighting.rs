use anyhow::{Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{RayTracingPipeline, ShaderBindingTable, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::restir_di::GpuRestirDiReservoir;
use crate::render::restir_gi::GpuRestirGiReservoir;
use crate::render::rt_settings::RtSettings;
use crate::voxel::gpu_upload::UcvhGpuResources;

pub(crate) const RT_DIRECT_LIGHTING_RAYGEN_SPV: &str = "rt_direct_lighting.rgen.spv";
pub(crate) const RT_DIRECT_LIGHTING_MISS_SPV: &str = "rt_direct_lighting.rmiss.spv";
pub(crate) const RT_DIRECT_LIGHTING_CLOSEST_HIT_SPV: &str = "rt_direct_lighting.rchit.spv";
pub(crate) const RT_DIRECT_LIGHTING_INTERSECTION_SPV: &str = "rt_direct_lighting.rint.spv";

#[derive(Clone, Copy)]
pub struct RtDirectLightingShaders<'a> {
    pub raygen: &'a [u8],
    pub miss: &'a [u8],
    pub closest_hit: &'a [u8],
    pub intersection: &'a [u8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRtDirectLightingUniforms {
    pub restir_di_enabled: u32,
    pub restir_gi_enabled: u32,
    pub debug_view: u32,
    pub width: u32,
    pub height: u32,
    pub shadows_enabled: u32,
    pub _pad0: [u32; 2],
    pub sky_color_sun_angular_radius: [f32; 4],
    pub ground_color_pad: [f32; 4],
    pub sun_direction_pad: [f32; 4],
    pub sun_intensity_pad: [f32; 4],
}

pub struct RtDirectLightingPass {
    ray_tracing_pipeline_loader: ash::khr::ray_tracing_pipeline::Device,
    pipeline: RayTracingPipeline,
    shader_binding_table: ShaderBindingTable,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_buffers: Vec<GpuBuffer>,
    fallback_direct_reservoirs: GpuBuffer,
    fallback_indirect_reservoirs: GpuBuffer,
    traversal_stats_buffer: GpuBuffer,
    current_radiance: GpuImage,
}

pub struct RtDirectLightingCreateInfo<'a> {
    pub ray_tracing_pipeline_loader: &'a ash::khr::ray_tracing_pipeline::Device,
    pub rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    pub width: u32,
    pub height: u32,
    pub frame_count: usize,
    pub ucvh_gpu: &'a UcvhGpuResources,
    pub shaders: RtDirectLightingShaders<'a>,
}

#[derive(Clone, Copy)]
pub struct RtDirectLightingFrameSettings {
    pub rt_settings: RtSettings,
    pub restir_di_active: bool,
    pub restir_gi_active: bool,
    pub shadows_enabled: bool,
    pub sun_direction: glam::Vec3,
    pub sun_intensity: glam::Vec3,
    pub sun_angular_radius: f32,
}

#[derive(Clone, Copy)]
pub struct RtDirectLightingGraphOutputs {
    pub current_radiance: ResourceHandle,
}

impl RtDirectLightingPass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 15] {
        [
            DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::ray_tracing(3, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(4, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
            DescriptorBindingSpec::ray_tracing(5, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(6, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(7, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(8, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(9, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(10, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(11, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(12, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(13, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(14, vk::DescriptorType::STORAGE_BUFFER),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: RtDirectLightingCreateInfo<'_>,
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
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    descriptor_count: frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: (12 * frame_count) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: (2 * frame_count) as u32,
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
        let uniform_buffers = match create_uniform_buffers(device, allocator, frame_count) {
            Ok(buffers) => buffers,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let fallback_direct_reservoirs =
            match create_fallback_direct_reservoirs(device, allocator, info.width, info.height) {
                Ok(buffer) => buffer,
                Err(error) => {
                    destroy_buffers(uniform_buffers, device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let traversal_stats_buffer = match create_disabled_traversal_stats_buffer(device, allocator)
        {
            Ok(buffer) => buffer,
            Err(error) => {
                fallback_direct_reservoirs.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let fallback_indirect_reservoirs =
            match create_fallback_indirect_reservoirs(device, allocator, info.width, info.height) {
                Ok(buffer) => buffer,
                Err(error) => {
                    traversal_stats_buffer.destroy(device, allocator);
                    fallback_direct_reservoirs.destroy(device, allocator);
                    destroy_buffers(uniform_buffers, device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let current_radiance =
            match create_current_radiance_image(device, allocator, info.width, info.height) {
                Ok(image) => image,
                Err(error) => {
                    fallback_indirect_reservoirs.destroy(device, allocator);
                    traversal_stats_buffer.destroy(device, allocator);
                    fallback_direct_reservoirs.destroy(device, allocator);
                    destroy_buffers(uniform_buffers, device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        for &descriptor_set in &descriptor_sets {
            write_ucvh_descriptors(device, descriptor_set, info.ucvh_gpu);
        }
        write_traversal_stats_descriptor(device, &descriptor_sets, &traversal_stats_buffer);

        let shader_modules = match create_rt_direct_lighting_shader_modules(device, info.shaders) {
            Ok(modules) => modules,
            Err(error) => {
                current_radiance.destroy(device, allocator);
                fallback_indirect_reservoirs.destroy(device, allocator);
                traversal_stats_buffer.destroy(device, allocator);
                fallback_direct_reservoirs.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let pipeline = match RayTracingPipeline::new_surface_pipeline(
            device,
            info.ray_tracing_pipeline_loader,
            shader_modules.raygen,
            shader_modules.miss,
            shader_modules.closest_hit,
            shader_modules.intersection,
            c"main",
            &[descriptor_set_layout],
            &[],
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                shader_modules.destroy(device);
                current_radiance.destroy(device, allocator);
                fallback_indirect_reservoirs.destroy(device, allocator);
                traversal_stats_buffer.destroy(device, allocator);
                fallback_direct_reservoirs.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        shader_modules.destroy(device);
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
                current_radiance.destroy(device, allocator);
                fallback_indirect_reservoirs.destroy(device, allocator);
                traversal_stats_buffer.destroy(device, allocator);
                fallback_direct_reservoirs.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
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
            fallback_direct_reservoirs,
            fallback_indirect_reservoirs,
            traversal_stats_buffer,
            current_radiance,
        })
    }

    pub fn width(&self) -> u32 {
        self.current_radiance.extent.width
    }

    pub fn height(&self) -> u32 {
        self.current_radiance.extent.height
    }

    pub fn current_radiance_image(&self) -> &GpuImage {
        &self.current_radiance
    }

    pub fn update_uniforms(&self, frame_slot: usize, settings: RtDirectLightingFrameSettings) {
        let uniforms = GpuRtDirectLightingUniforms {
            restir_di_enabled: settings.restir_di_active as u32,
            restir_gi_enabled: settings.restir_gi_active as u32,
            debug_view: settings.rt_settings.debug_view.as_gpu_value(),
            width: self.width(),
            height: self.height(),
            shadows_enabled: settings.shadows_enabled as u32,
            _pad0: [0; 2],
            sky_color_sun_angular_radius: [0.4, 0.5, 0.7, settings.sun_angular_radius],
            ground_color_pad: [0.15, 0.1, 0.08, 0.0],
            sun_direction_pad: [
                settings.sun_direction.x,
                settings.sun_direction.y,
                settings.sun_direction.z,
                0.0,
            ],
            sun_intensity_pad: [
                settings.sun_intensity.x,
                settings.sun_intensity.y,
                settings.sun_intensity.z,
                0.0,
            ],
        };
        write_mapped(self.uniform_buffers[frame_slot].mapped_ptr(), &uniforms);
    }

    pub fn update_tlas_descriptor(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        tlas: vk::AccelerationStructureKHR,
    ) {
        if let Some(&descriptor_set) = self.descriptor_sets.get(frame_slot) {
            write_tlas_descriptor(device, descriptor_set, tlas);
        }
    }

    pub fn update_aabb_descriptor(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        aabb_buffer: &GpuBuffer,
    ) {
        if let Some(&descriptor_set) = self.descriptor_sets.get(frame_slot) {
            write_aabb_descriptor(device, descriptor_set, aabb_buffer);
        }
    }

    pub fn update_ucvh_descriptors(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        ucvh_gpu: &UcvhGpuResources,
    ) {
        if let Some(&descriptor_set) = self.descriptor_sets.get(frame_slot) {
            write_ucvh_descriptors(device, descriptor_set, ucvh_gpu);
        }
    }

    pub fn update_frame_descriptors(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        surface_buffer: &GpuBuffer,
        direct_reservoir_buffer: Option<&GpuBuffer>,
        indirect_reservoir_buffer: Option<&GpuBuffer>,
    ) {
        let Some(&descriptor_set) = self.descriptor_sets.get(frame_slot) else {
            return;
        };
        let direct_reservoirs = direct_reservoir_buffer.unwrap_or(&self.fallback_direct_reservoirs);
        let indirect_reservoirs =
            indirect_reservoir_buffer.unwrap_or(&self.fallback_indirect_reservoirs);
        write_frame_descriptors(
            device,
            descriptor_set,
            RtDirectLightingFrameDescriptors {
                surface_buffer,
                direct_reservoirs,
                indirect_reservoirs,
                current_radiance: &self.current_radiance,
                uniform_buffer: &self.uniform_buffers[frame_slot],
            },
        );
    }

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        frame_slot: usize,
        surface: ResourceHandle,
        direct_reservoirs: Option<ResourceHandle>,
        indirect_reservoirs: Option<ResourceHandle>,
        output_initialized: bool,
    ) -> RtDirectLightingGraphOutputs {
        let output = graph.import_image_with_access(
            self.current_radiance.handle,
            self.current_radiance.extent.width,
            self.current_radiance.extent.height,
            self.current_radiance.format,
            self.current_radiance_usage(),
            if output_initialized {
                AccessKind::RayTracingShaderWrite
            } else {
                AccessKind::Undefined
            },
        );
        let uniform_buffer = &self.uniform_buffers[frame_slot];
        let uniform = graph.import_buffer_with_access(
            uniform_buffer.handle,
            uniform_buffer.size,
            uniform_buffer.usage,
            AccessKind::RayTracingShaderRead,
        );
        let ray_tracing_pipeline_loader = self.ray_tracing_pipeline_loader.clone();
        let pipeline = self.pipeline.handle;
        let pipeline_layout = self.pipeline.layout;
        let descriptor_set = self.descriptor_sets[frame_slot];
        let sbt_regions = self.shader_binding_table.regions();
        let width = self.current_radiance.extent.width;
        let height = self.current_radiance.extent.height;

        let writes = graph.add_pass("rt_direct_lighting", QueueType::RayTracing, |builder| {
            builder.read_as(surface, AccessKind::RayTracingShaderRead);
            builder.read_as(uniform, AccessKind::RayTracingShaderRead);
            if let Some(direct_reservoirs) = direct_reservoirs {
                builder.read_as(direct_reservoirs, AccessKind::RayTracingShaderRead);
            }
            if let Some(indirect_reservoirs) = indirect_reservoirs {
                builder.read_as(indirect_reservoirs, AccessKind::RayTracingShaderRead);
            }
            builder.write_as(output, AccessKind::RayTracingShaderWrite);
            Box::new(move |ctx| unsafe {
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
            })
        });

        RtDirectLightingGraphOutputs {
            current_radiance: writes[0],
        }
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let current_radiance = create_current_radiance_image(device, allocator, width, height)?;
        let fallback_direct_reservoirs =
            match create_fallback_direct_reservoirs(device, allocator, width, height) {
                Ok(buffer) => buffer,
                Err(error) => {
                    current_radiance.destroy(device, allocator);
                    return Err(error);
                }
            };
        let fallback_indirect_reservoirs =
            match create_fallback_indirect_reservoirs(device, allocator, width, height) {
                Ok(buffer) => buffer,
                Err(error) => {
                    fallback_direct_reservoirs.destroy(device, allocator);
                    current_radiance.destroy(device, allocator);
                    return Err(error);
                }
            };
        std::mem::replace(&mut self.current_radiance, current_radiance).destroy(device, allocator);
        std::mem::replace(
            &mut self.fallback_direct_reservoirs,
            fallback_direct_reservoirs,
        )
        .destroy(device, allocator);
        std::mem::replace(
            &mut self.fallback_indirect_reservoirs,
            fallback_indirect_reservoirs,
        )
        .destroy(device, allocator);
        Ok(())
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.shader_binding_table.destroy(device, allocator);
        self.pipeline.destroy(device);
        self.current_radiance.destroy(device, allocator);
        self.fallback_direct_reservoirs.destroy(device, allocator);
        self.fallback_indirect_reservoirs.destroy(device, allocator);
        self.traversal_stats_buffer.destroy(device, allocator);
        destroy_buffers(self.uniform_buffers, device, allocator);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }

    fn current_radiance_usage(&self) -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED
    }
}

#[derive(Clone, Copy)]
struct RtDirectLightingShaderModules {
    raygen: vk::ShaderModule,
    miss: vk::ShaderModule,
    closest_hit: vk::ShaderModule,
    intersection: vk::ShaderModule,
}

impl RtDirectLightingShaderModules {
    fn destroy(self, device: &ash::Device) {
        for module in [self.raygen, self.miss, self.closest_hit, self.intersection] {
            unsafe { device.destroy_shader_module(module, None) };
        }
    }
}

fn create_rt_direct_lighting_shader_modules(
    device: &ash::Device,
    shaders: RtDirectLightingShaders<'_>,
) -> Result<RtDirectLightingShaderModules> {
    let shader_specs = [
        (RT_DIRECT_LIGHTING_RAYGEN_SPV, shaders.raygen),
        (RT_DIRECT_LIGHTING_MISS_SPV, shaders.miss),
        (RT_DIRECT_LIGHTING_CLOSEST_HIT_SPV, shaders.closest_hit),
        (RT_DIRECT_LIGHTING_INTERSECTION_SPV, shaders.intersection),
    ];
    let mut modules = Vec::with_capacity(shader_specs.len());
    for (name, spirv) in shader_specs {
        match create_shader_module(device, spirv) {
            Ok(module) => modules.push(module),
            Err(error) => {
                for module in modules {
                    unsafe { device.destroy_shader_module(module, None) };
                }
                return Err(error)
                    .with_context(|| format!("failed to create {name} shader module"));
            }
        }
    }

    Ok(RtDirectLightingShaderModules {
        raygen: modules[0],
        miss: modules[1],
        closest_hit: modules[2],
        intersection: modules[3],
    })
}

struct RtDirectLightingFrameDescriptors<'a> {
    surface_buffer: &'a GpuBuffer,
    direct_reservoirs: &'a GpuBuffer,
    indirect_reservoirs: &'a GpuBuffer,
    current_radiance: &'a GpuImage,
    uniform_buffer: &'a GpuBuffer,
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
            std::mem::size_of::<GpuRtDirectLightingUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("rt_direct_lighting_uniforms_{slot}"),
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

fn create_fallback_direct_reservoirs(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<GpuBuffer> {
    let reservoir_count = width.saturating_mul(height).max(1) as usize;
    GpuBuffer::new(
        device,
        allocator,
        (reservoir_count * std::mem::size_of::<GpuRestirDiReservoir>()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
        "rt_direct_lighting_fallback_reservoirs",
    )
}

fn create_fallback_indirect_reservoirs(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<GpuBuffer> {
    let reservoir_count = width.saturating_mul(height).max(1) as usize;
    GpuBuffer::new(
        device,
        allocator,
        (reservoir_count * std::mem::size_of::<GpuRestirGiReservoir>()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
        "rt_direct_lighting_fallback_indirect_reservoirs",
    )
}

fn create_disabled_traversal_stats_buffer(
    device: &ash::Device,
    allocator: &GpuAllocator,
) -> Result<GpuBuffer> {
    GpuBuffer::new(
        device,
        allocator,
        16,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
        "rt_direct_lighting_disabled_traversal_stats",
    )
}

fn create_current_radiance_image(
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
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "rt_direct_lighting",
        },
    )
}

fn write_frame_descriptors(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    resources: RtDirectLightingFrameDescriptors<'_>,
) {
    let surface_info = vk::DescriptorBufferInfo::default()
        .buffer(resources.surface_buffer.handle)
        .offset(0)
        .range(resources.surface_buffer.size);
    let direct_reservoir_info = vk::DescriptorBufferInfo::default()
        .buffer(resources.direct_reservoirs.handle)
        .offset(0)
        .range(resources.direct_reservoirs.size);
    let indirect_reservoir_info = vk::DescriptorBufferInfo::default()
        .buffer(resources.indirect_reservoirs.handle)
        .offset(0)
        .range(resources.indirect_reservoirs.size);
    let current_radiance_info = vk::DescriptorImageInfo::default()
        .image_view(resources.current_radiance.view)
        .image_layout(vk::ImageLayout::GENERAL);
    let uniform_info = vk::DescriptorBufferInfo::default()
        .buffer(resources.uniform_buffer.handle)
        .offset(0)
        .range(std::mem::size_of::<GpuRtDirectLightingUniforms>() as u64);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&surface_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&direct_reservoir_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&current_radiance_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&uniform_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(14)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&indirect_reservoir_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn write_tlas_descriptor(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    tlas: vk::AccelerationStructureKHR,
) {
    let acceleration_structures = [tlas];
    let mut tlas_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
        .acceleration_structures(&acceleration_structures);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(4)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .push_next(&mut tlas_info);
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn write_aabb_descriptor(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    buffer: &GpuBuffer,
) {
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(buffer.handle)
        .offset(0)
        .range(buffer.size);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(5)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&buffer_info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn write_ucvh_descriptors(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    ucvh_gpu: &UcvhGpuResources,
) {
    let ucvh_buffers = [
        &ucvh_gpu.hierarchy_l0_buffer,
        &ucvh_gpu.hierarchy_ln_buffers[0],
        &ucvh_gpu.hierarchy_ln_buffers[1],
        &ucvh_gpu.hierarchy_ln_buffers[2],
        &ucvh_gpu.hierarchy_ln_buffers[3],
        &ucvh_gpu.occupancy_buffer,
    ];
    let config_info = vk::DescriptorBufferInfo::default()
        .buffer(ucvh_gpu.config_buffer.handle)
        .offset(0)
        .range(vk::WHOLE_SIZE);
    let buffer_infos = ucvh_buffers
        .iter()
        .map(|buffer| {
            vk::DescriptorBufferInfo::default()
                .buffer(buffer.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE)
        })
        .collect::<Vec<_>>();

    let mut writes = vec![
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&config_info)),
    ];
    writes.extend(buffer_infos.iter().enumerate().map(|(idx, info)| {
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding((idx + 7) as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(info))
    }));
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn write_traversal_stats_descriptor(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    traversal_stats_buffer: &GpuBuffer,
) {
    for &descriptor_set in descriptor_sets {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(traversal_stats_buffer.handle)
            .offset(0)
            .range(traversal_stats_buffer.size);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(13)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
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
    fn rt_direct_lighting_uniform_layout_is_stable() {
        assert_eq!(
            std::mem::size_of::<super::GpuRtDirectLightingUniforms>(),
            96
        );
        assert_eq!(
            std::mem::offset_of!(super::GpuRtDirectLightingUniforms, debug_view),
            8
        );
        assert_eq!(
            std::mem::offset_of!(super::GpuRtDirectLightingUniforms, shadows_enabled),
            20
        );
        assert_eq!(
            std::mem::offset_of!(
                super::GpuRtDirectLightingUniforms,
                sky_color_sun_angular_radius
            ),
            32
        );
        assert_eq!(
            std::mem::offset_of!(super::GpuRtDirectLightingUniforms, sun_intensity_pad),
            80
        );
    }

    #[test]
    fn rt_direct_lighting_descriptor_specs_match_shader_resources() {
        let specs = super::RtDirectLightingPass::descriptor_binding_specs();
        let actual = specs
            .iter()
            .map(|spec| (spec.binding, spec.descriptor_type))
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (0, vk::DescriptorType::STORAGE_BUFFER),
                (1, vk::DescriptorType::STORAGE_BUFFER),
                (2, vk::DescriptorType::STORAGE_IMAGE),
                (3, vk::DescriptorType::UNIFORM_BUFFER),
                (4, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
                (5, vk::DescriptorType::STORAGE_BUFFER),
                (6, vk::DescriptorType::UNIFORM_BUFFER),
                (7, vk::DescriptorType::STORAGE_BUFFER),
                (8, vk::DescriptorType::STORAGE_BUFFER),
                (9, vk::DescriptorType::STORAGE_BUFFER),
                (10, vk::DescriptorType::STORAGE_BUFFER),
                (11, vk::DescriptorType::STORAGE_BUFFER),
                (12, vk::DescriptorType::STORAGE_BUFFER),
                (13, vk::DescriptorType::STORAGE_BUFFER),
                (14, vk::DescriptorType::STORAGE_BUFFER),
            ]
        );
    }

    #[test]
    fn rt_direct_lighting_pass_owns_shadow_trace_pipeline_and_descriptors() {
        let source =
            crate::render::source_checks::read_source("src/render/passes/rt_direct_lighting.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT direct-lighting implementation should precede tests");

        for token in [
            "RtDirectLightingShaders",
            "rt_direct_lighting.rmiss.spv",
            "rt_direct_lighting.rchit.spv",
            "rt_direct_lighting.rint.spv",
            "RayTracingPipeline::new_surface_pipeline",
            "WriteDescriptorSetAccelerationStructureKHR",
            "update_tlas_descriptor",
            "update_aabb_descriptor",
            "update_ucvh_descriptors",
            "write_traversal_stats_descriptor",
            "fallback_indirect_reservoirs",
        ] {
            assert!(
                implementation.contains(token),
                "RT direct-lighting shadow pipeline plumbing missing {token}"
            );
        }
    }

    #[test]
    fn rt_direct_lighting_runtime_descriptors_are_frame_slot_scoped() {
        let source =
            crate::render::source_checks::read_source("src/render/passes/rt_direct_lighting.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT direct-lighting implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "pubfnupdate_tlas_descriptor(&self,device:&ash::Device,frame_slot:usize,tlas:vk::AccelerationStructureKHR,)",
            "pubfnupdate_aabb_descriptor(&self,device:&ash::Device,frame_slot:usize,aabb_buffer:&GpuBuffer,)",
            "pubfnupdate_ucvh_descriptors(&self,device:&ash::Device,frame_slot:usize,ucvh_gpu:&UcvhGpuResources,)",
            "self.descriptor_sets.get(frame_slot)",
        ] {
            assert!(
                compact.contains(token),
                "RT direct-lighting descriptors must be frame-slot scoped with {token}"
            );
        }
        for forbidden in [
            "for&descriptor_setin&self.descriptor_sets{write_tlas_descriptor",
            "for&descriptor_setin&self.descriptor_sets{write_aabb_descriptor",
            "write_ucvh_descriptors(device,&self.descriptor_sets,ucvh_gpu)",
        ] {
            assert!(
                !compact.contains(forbidden),
                "RT direct-lighting must not update pending descriptor sets with {forbidden}"
            );
        }
    }

    #[test]
    fn rt_direct_lighting_shaders_trace_hardware_shadow_visibility() {
        let raygen = std::fs::read_to_string("assets/shaders/passes/rt_direct_lighting.rgen.slang")
            .expect("rt_direct_lighting.rgen.slang should be readable");
        let miss = std::fs::read_to_string("assets/shaders/passes/rt_direct_lighting.rmiss.slang")
            .expect("rt_direct_lighting.rmiss.slang should be readable");
        let closest_hit =
            std::fs::read_to_string("assets/shaders/passes/rt_direct_lighting.rchit.slang")
                .expect("rt_direct_lighting.rchit.slang should be readable");
        let intersection =
            std::fs::read_to_string("assets/shaders/passes/rt_direct_lighting.rint.slang")
                .expect("rt_direct_lighting.rint.slang should be readable");

        for token in [
            "RaytracingAccelerationStructure scene_tlas",
            "RtShadowPayload",
            "TraceRay(",
            "rt_direct_light_visible",
            "rt_direct.shadows_enabled",
            "rt_direct.restir_gi_enabled",
            "surface.brick_id",
            "surface.local",
            "rt_direct_resolve_reservoir(surface, reservoir, visible)",
            "StructuredBuffer<RestirGiReservoir> indirect_reservoirs",
            "rt_direct_resolve_indirect_reservoir(surface, indirect_reservoir)",
            "current_radiance[launch_id.xy] = float4(sky_indirect + analytic_direct + direct + indirect, 1.0);",
        ] {
            assert!(
                raygen.contains(token),
                "RT direct-lighting raygen must trace visibility with {token}"
            );
        }
        assert!(miss.contains("[shader(\"miss\")]"));
        assert!(miss.contains("payload.occluded = 0u"));
        for token in [
            "[shader(\"closesthit\")]",
            "inout RtShadowPayload",
            "trace_any_hit_ray_skip_voxel",
            "payload.skip_brick_id",
            "payload.skip_local",
            "payload.max_t",
        ] {
            assert!(
                closest_hit.contains(token),
                "RT direct-lighting closest-hit shadow shader missing {token}"
            );
        }
        for token in [
            "[shader(\"intersection\")]",
            "StructuredBuffer<RtAabb> rt_aabbs",
            "ReportHit(hit_t",
        ] {
            assert!(
                intersection.contains(token),
                "RT direct-lighting intersection shader missing {token}"
            );
        }
    }

    #[test]
    fn rt_direct_lighting_primary_miss_uses_sky_ground_background() {
        let raygen = std::fs::read_to_string("assets/shaders/passes/rt_direct_lighting.rgen.slang")
            .expect("rt_direct_lighting.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&raygen);

        for token in [
            "float3rt_direct_background_color(RtSurfacePixelsurface)",
            "float3miss_dir=normalize(surface.view_direction_background.xyz);",
            "floatground_ndotl=max(rt_direct.sun_direction_pad.y,0.0);",
            "float3finite_sun_irradiance=rt_direct.sun_intensity_pad.rgb*ground_ndotl*rt_direct_sun_disk_solid_angle();",
            "returnlerp(rt_direct_sunlit_ground_color(),rt_direct.sky_color_sun_angular_radius.rgb,t);",
            "current_radiance[launch_id.xy]=float4(rt_direct_background_color(surface),1.0);",
        ] {
            assert!(
                compact.contains(token),
                "RT direct-lighting primary miss must resolve to scene background; missing {token}"
            );
        }
        assert!(
            !compact.contains("float3miss_dir=normalize(-surface.normal_roughness.xyz);"),
            "RT direct-lighting background must not infer miss ray direction from geometric normals"
        );
        assert!(
            !compact.contains("current_radiance[launch_id.xy]=float4(0.0,0.0,0.0,1.0);"),
            "RT direct-lighting must not turn primary miss pixels into a black bar"
        );
    }

    #[test]
    fn rt_direct_lighting_final_uses_scene_lighting_instead_of_flat_albedo() {
        let raygen = std::fs::read_to_string("assets/shaders/passes/rt_direct_lighting.rgen.slang")
            .expect("rt_direct_lighting.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&raygen);

        for token in [
            "float3rt_direct_sample_sky_indirect(RtSurfacePixelsurface)",
            "float3sky_indirect=rt_direct_sample_sky_indirect(surface);",
            "boolrt_direct_sun_visible(RtSurfacePixelsurface,float3sun_dir)",
            "float3rt_direct_analytic_sun_direct(RtSurfacePixelsurface)",
            "float3analytic_direct=rt_direct_analytic_sun_direct(surface);",
            "current_radiance[launch_id.xy]=float4(sky_indirect+analytic_direct+direct+indirect,1.0);",
        ] {
            assert!(
                compact.contains(token),
                "RT final lighting must keep analytic sun and ambient lighting visible; missing {token}"
            );
        }
        assert!(
            !compact.contains("float3ambient=rt_direct_hemisphere_ambient(surface);"),
            "RT final lighting must not use unoccluded hemisphere ambient as a flat albedo preview"
        );
        assert!(
            !compact.contains("current_radiance[launch_id.xy]=float4(albedo+direct+indirect,1.0);"),
            "RT final lighting must not fall back to a flat unlit albedo image"
        );
    }

    #[test]
    fn rt_direct_lighting_sky_indirect_is_ray_visibility_driven() {
        let raygen = std::fs::read_to_string("assets/shaders/passes/rt_direct_lighting.rgen.slang")
            .expect("rt_direct_lighting.rgen.slang should be readable");
        let compact = crate::render::source_checks::compact(&raygen);

        for token in [
            "float3rt_direct_background_color_for_dir(float3direction)",
            "float3rt_direct_sky_visibility_sample(RtSurfacePixelsurface,float3sky_dir)",
            "float3rt_direct_sample_sky_indirect(RtSurfacePixelsurface)",
            "float3sky_dir=normal;",
            "ray.Direction=sky_dir;",
            "RtShadowPayloadpayload=make_rt_shadow_payload(surface.brick_id,surface.local,ray.TMax);",
            "TraceRay(scene_tlas,0u,0xffu,0u,0u,0u,ray,payload);",
            "if(payload.occluded!=0u){returnfloat3(0.0);}",
            "returnsurface.albedo_material.rgb*rt_direct_sky_visibility_sample(surface,sky_dir);",
        ] {
            assert!(
                compact.contains(token),
                "RT sky indirect must be generated by an actual visibility ray; missing {token}"
            );
        }
        for forbidden in [
            "float3up_sky_dir=",
            "float3sun_sky_dir=",
            "rt_direct_sky_visibility_sample(surface,up_sky_dir)",
            "rt_direct_sky_visibility_sample(surface,sun_sky_dir)",
            "*(1.0/3.0)",
        ] {
            assert!(
                !compact.contains(forbidden),
                "RT sky indirect must not spend three visibility rays per final pixel; found {forbidden}"
            );
        }
        for forbidden in ["rt_direct_rand01", "rt_direct.frame_index"] {
            assert!(
                !raygen.contains(forbidden),
                "RT sky indirect must be stable in final output; found {forbidden}"
            );
        }
    }
}
