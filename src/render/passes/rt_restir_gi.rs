use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::graph::RenderGraph;
use crate::render::pipeline::{RayTracingPipeline, ShaderBindingTable, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::restir_gi::{
    GpuRestirGiReservoir, GpuRestirGiUniforms, RestirGiDebugView, RestirGiSettings,
};
use crate::render::rt_history::{GpuRtHistoryUniforms, GpuRtSurfacePixel};
use crate::render::rt_settings::{RtDebugView, RtSettings};
use crate::voxel::gpu_upload::UcvhGpuResources;

pub(crate) const RT_RESTIR_GI_RAYGEN_SPV: &str = "rt_restir_gi.rgen.spv";
pub(crate) const RT_RESTIR_GI_MISS_SPV: &str = "rt_restir_gi.rmiss.spv";
pub(crate) const RT_RESTIR_GI_CLOSEST_HIT_SPV: &str = "rt_restir_gi.rchit.spv";
pub(crate) const RT_RESTIR_GI_INTERSECTION_SPV: &str = "rt_restir_gi.rint.spv";

#[derive(Clone, Copy)]
pub struct RtRestirGiShaders<'a> {
    pub raygen: &'a [u8],
    pub miss: &'a [u8],
    pub closest_hit: &'a [u8],
    pub intersection: &'a [u8],
}

pub struct RtRestirGiPass {
    ray_tracing_pipeline_loader: ash::khr::ray_tracing_pipeline::Device,
    pipeline: RayTracingPipeline,
    shader_binding_table: ShaderBindingTable,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_buffers: Vec<GpuBuffer>,
    history_uniform_buffers: Vec<GpuBuffer>,
    reservoirs: Vec<GpuBuffer>,
    surface_history_buffers: [GpuBuffer; 2],
    traversal_stats_buffer: GpuBuffer,
    width: u32,
    height: u32,
    reservoir_count: u32,
}

pub struct RtRestirGiCreateInfo<'a> {
    pub ray_tracing_pipeline_loader: &'a ash::khr::ray_tracing_pipeline::Device,
    pub rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    pub width: u32,
    pub height: u32,
    pub frame_count: usize,
    pub ucvh_gpu: &'a UcvhGpuResources,
    pub shaders: RtRestirGiShaders<'a>,
}

#[derive(Clone, Copy)]
pub struct RtRestirGiGraphOutputs {
    pub reservoirs: ResourceHandle,
}

impl RtRestirGiPass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 18] {
        [
            DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(3, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(4, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(5, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(6, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(7, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
            DescriptorBindingSpec::ray_tracing(8, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(9, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(10, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(11, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(12, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(13, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(14, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(15, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(16, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(17, vk::DescriptorType::STORAGE_BUFFER),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: RtRestirGiCreateInfo<'_>,
    ) -> Result<Self> {
        let frame_count = info.frame_count;
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding_specs(&Self::descriptor_binding_specs())
            .build(device)?;
        let descriptor_pool = match DescriptorPool::new(
            device,
            frame_count as u32,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                    descriptor_count: frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: (3 * frame_count) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: (14 * frame_count) as u32,
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
        let history_uniform_buffers =
            match create_history_uniform_buffers(device, allocator, frame_count) {
                Ok(buffers) => buffers,
                Err(error) => {
                    destroy_buffers(uniform_buffers, device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let reservoir_count = info.width.saturating_mul(info.height);
        let reservoirs =
            match create_reservoir_buffers(device, allocator, frame_count, reservoir_count) {
                Ok(buffers) => buffers,
                Err(error) => {
                    destroy_buffers(history_uniform_buffers, device, allocator);
                    destroy_buffers(uniform_buffers, device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let surface_history_buffers =
            match create_surface_history_buffers(device, allocator, info.width, info.height) {
                Ok(buffers) => buffers,
                Err(error) => {
                    destroy_buffers(reservoirs, device, allocator);
                    destroy_buffers(history_uniform_buffers, device, allocator);
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
                destroy_buffers(Vec::from(surface_history_buffers), device, allocator);
                destroy_buffers(reservoirs, device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let (pipeline, shader_binding_table) = match create_raygen_pipeline(
            device,
            allocator,
            info.ray_tracing_pipeline_loader,
            info.rt_pipeline_properties,
            info.shaders,
            descriptor_set_layout,
        ) {
            Ok(resources) => resources,
            Err(error) => {
                traversal_stats_buffer.destroy(device, allocator);
                destroy_buffers(Vec::from(surface_history_buffers), device, allocator);
                destroy_buffers(reservoirs, device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        let pass = Self {
            ray_tracing_pipeline_loader: info.ray_tracing_pipeline_loader.clone(),
            pipeline,
            shader_binding_table,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            uniform_buffers,
            history_uniform_buffers,
            reservoirs,
            surface_history_buffers,
            traversal_stats_buffer,
            width: info.width,
            height: info.height,
            reservoir_count,
        };
        pass.write_static_descriptors(device);
        for &descriptor_set in &pass.descriptor_sets {
            write_ucvh_descriptors(device, descriptor_set, info.ucvh_gpu);
        }
        write_traversal_stats_descriptor(
            device,
            &pass.descriptor_sets,
            &pass.traversal_stats_buffer,
        );
        Ok(pass)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn update_uniforms(
        &self,
        frame_slot: usize,
        rt_settings: RtSettings,
        frame_index: u64,
        history_initialized: bool,
    ) {
        let settings = RestirGiSettings {
            enabled: rt_settings.restir_gi_enabled,
            temporal_enabled: rt_settings.restir_gi_enabled && history_initialized,
            initial_candidate_count: 1,
            history_length: rt_settings.history_length.max(1),
            max_bounces: 1,
            debug_view: rt_restir_gi_debug_view(rt_settings.debug_view),
        };
        let uniforms = settings.gpu_uniforms(
            frame_index as u32,
            self.reservoir_count,
            self.width,
            self.height,
        );
        write_mapped(self.uniform_buffers[frame_slot].mapped_ptr(), &uniforms);
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
    ) {
        let Some(&descriptor_set) = self.descriptor_sets.get(frame_slot) else {
            return;
        };
        write_frame_descriptors(
            device,
            descriptor_set,
            surface_buffer,
            self.history_reservoir_buffer(frame_slot),
            self.current_reservoir_buffer(frame_slot),
            self.previous_surface_history_buffer(frame_index),
            self.current_surface_history_buffer(frame_index),
        );
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

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        frame_slot: usize,
        frame_index: u64,
        surface: ResourceHandle,
        history_initialized: bool,
    ) -> RtRestirGiGraphOutputs {
        let uniform_buffer = &self.uniform_buffers[frame_slot];
        let uniform = graph.import_buffer_with_access(
            uniform_buffer.handle,
            uniform_buffer.size,
            uniform_buffer.usage,
            AccessKind::RayTracingShaderRead,
        );
        let history_uniform_buffer = &self.history_uniform_buffers[frame_slot];
        let history_uniform = graph.import_buffer_with_access(
            history_uniform_buffer.handle,
            history_uniform_buffer.size,
            history_uniform_buffer.usage,
            AccessKind::RayTracingShaderRead,
        );
        let current_reservoir_buffer = self.current_reservoir_buffer(frame_slot);
        let history_reservoir_buffer = self.history_reservoir_buffer(frame_slot);
        let output_reservoirs = graph.import_buffer_with_access(
            current_reservoir_buffer.handle,
            current_reservoir_buffer.size,
            current_reservoir_buffer.usage,
            AccessKind::Undefined,
        );
        let history_reservoirs = graph.import_buffer_with_access(
            history_reservoir_buffer.handle,
            history_reservoir_buffer.size,
            history_reservoir_buffer.usage,
            if history_initialized {
                AccessKind::RayTracingShaderWrite
            } else {
                AccessKind::Undefined
            },
        );
        let current_surface_history_buffer = self.current_surface_history_buffer(frame_index);
        let previous_surface_history_buffer = self.previous_surface_history_buffer(frame_index);
        let current_surface_history = graph.import_buffer_with_access(
            current_surface_history_buffer.handle,
            current_surface_history_buffer.size,
            current_surface_history_buffer.usage,
            AccessKind::Undefined,
        );
        let previous_surface_history = graph.import_buffer_with_access(
            previous_surface_history_buffer.handle,
            previous_surface_history_buffer.size,
            previous_surface_history_buffer.usage,
            if history_initialized {
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
        let width = self.width;
        let height = self.height;

        let writes = graph.add_pass("rt_restir_gi", QueueType::RayTracing, |builder| {
            builder.read_as(uniform, AccessKind::RayTracingShaderRead);
            builder.read_as(history_uniform, AccessKind::RayTracingShaderRead);
            builder.read_as(surface, AccessKind::RayTracingShaderRead);
            if history_initialized {
                builder.read_as(history_reservoirs, AccessKind::RayTracingShaderRead);
                builder.read_as(previous_surface_history, AccessKind::RayTracingShaderRead);
            }
            builder.write_as(output_reservoirs, AccessKind::RayTracingShaderWrite);
            builder.write_as(current_surface_history, AccessKind::RayTracingShaderWrite);
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

        RtRestirGiGraphOutputs {
            reservoirs: writes[0],
        }
    }

    pub fn resize_buffers(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let reservoir_count = width.saturating_mul(height);
        let reservoirs =
            create_reservoir_buffers(device, allocator, self.reservoirs.len(), reservoir_count)?;
        let surface_history_buffers =
            match create_surface_history_buffers(device, allocator, width, height) {
                Ok(buffers) => buffers,
                Err(error) => {
                    destroy_buffers(reservoirs, device, allocator);
                    return Err(error);
                }
            };
        destroy_buffers(
            std::mem::replace(&mut self.reservoirs, reservoirs),
            device,
            allocator,
        );
        destroy_buffers(
            Vec::from(std::mem::replace(
                &mut self.surface_history_buffers,
                surface_history_buffers,
            )),
            device,
            allocator,
        );
        self.width = width;
        self.height = height;
        self.reservoir_count = reservoir_count;
        Ok(())
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.shader_binding_table.destroy(device, allocator);
        self.pipeline.destroy(device);
        self.traversal_stats_buffer.destroy(device, allocator);
        destroy_buffers(Vec::from(self.surface_history_buffers), device, allocator);
        destroy_buffers(self.reservoirs, device, allocator);
        destroy_buffers(self.history_uniform_buffers, device, allocator);
        destroy_buffers(self.uniform_buffers, device, allocator);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }

    fn write_static_descriptors(&self, device: &ash::Device) {
        for (set_idx, &descriptor_set) in self.descriptor_sets.iter().enumerate() {
            let uniform_info = vk::DescriptorBufferInfo::default()
                .buffer(self.uniform_buffers[set_idx].handle)
                .offset(0)
                .range(std::mem::size_of::<GpuRestirGiUniforms>() as u64);
            let history_uniform_info = vk::DescriptorBufferInfo::default()
                .buffer(self.history_uniform_buffers[set_idx].handle)
                .offset(0)
                .range(std::mem::size_of::<GpuRtHistoryUniforms>() as u64);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&uniform_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&history_uniform_info)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    pub fn current_reservoir_buffer(&self, frame_slot: usize) -> &GpuBuffer {
        &self.reservoirs[self.current_reservoir_slot(frame_slot)]
    }

    pub fn output_reservoir_buffer(&self, frame_slot: usize) -> &GpuBuffer {
        self.current_reservoir_buffer(frame_slot)
    }

    pub fn history_reservoir_buffer(&self, frame_slot: usize) -> &GpuBuffer {
        &self.reservoirs[self.history_reservoir_slot(frame_slot)]
    }

    fn current_reservoir_slot(&self, frame_slot: usize) -> usize {
        frame_slot % self.reservoirs.len()
    }

    fn history_reservoir_slot(&self, frame_slot: usize) -> usize {
        (self.current_reservoir_slot(frame_slot) + self.reservoirs.len() - 1)
            % self.reservoirs.len()
    }

    fn current_surface_history_buffer(&self, frame_index: u64) -> &GpuBuffer {
        &self.surface_history_buffers[current_surface_history_index(frame_index)]
    }

    fn previous_surface_history_buffer(&self, frame_index: u64) -> &GpuBuffer {
        &self.surface_history_buffers[previous_surface_history_index(frame_index)]
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
            std::mem::size_of::<GpuRestirGiUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("rt_restir_gi_uniforms_{slot}"),
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
            &format!("rt_restir_gi_history_uniforms_{slot}"),
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

fn create_reservoir_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
    reservoir_count: u32,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count.max(2));
    for slot in 0..frame_count.max(2) {
        match create_reservoir_buffer(
            device,
            allocator,
            reservoir_count,
            &format!("rt_restir_gi_reservoirs_{slot}"),
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

fn create_reservoir_buffer(
    device: &ash::Device,
    allocator: &GpuAllocator,
    reservoir_count: u32,
    name: &str,
) -> Result<GpuBuffer> {
    let count = reservoir_count.max(1) as usize;
    GpuBuffer::new(
        device,
        allocator,
        (count * std::mem::size_of::<GpuRestirGiReservoir>()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
        name,
    )
}

fn create_surface_history_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<[GpuBuffer; 2]> {
    let first =
        create_surface_history_buffer(device, allocator, width, height, "rt_restir_gi_surface_0")?;
    let second = match create_surface_history_buffer(
        device,
        allocator,
        width,
        height,
        "rt_restir_gi_surface_1",
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

#[derive(Clone, Copy)]
struct RtRestirGiShaderModules {
    raygen: vk::ShaderModule,
    miss: vk::ShaderModule,
    closest_hit: vk::ShaderModule,
    intersection: vk::ShaderModule,
}

impl RtRestirGiShaderModules {
    fn destroy(self, device: &ash::Device) {
        for module in [self.raygen, self.miss, self.closest_hit, self.intersection] {
            unsafe { device.destroy_shader_module(module, None) };
        }
    }
}

fn create_raygen_pipeline(
    device: &ash::Device,
    allocator: &GpuAllocator,
    ray_tracing_pipeline_loader: &ash::khr::ray_tracing_pipeline::Device,
    rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    shaders: RtRestirGiShaders<'_>,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<(RayTracingPipeline, ShaderBindingTable)> {
    let shader_modules = create_rt_restir_gi_shader_modules(device, shaders)?;
    let pipeline = match RayTracingPipeline::new_surface_pipeline(
        device,
        ray_tracing_pipeline_loader,
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
            return Err(error);
        }
    };
    shader_modules.destroy(device);
    let shader_binding_table = match ShaderBindingTable::new(
        device,
        allocator,
        ray_tracing_pipeline_loader,
        pipeline.handle,
        rt_pipeline_properties,
        pipeline.group_counts,
    ) {
        Ok(table) => table,
        Err(error) => {
            pipeline.destroy(device);
            return Err(error);
        }
    };
    Ok((pipeline, shader_binding_table))
}

fn create_rt_restir_gi_shader_modules(
    device: &ash::Device,
    shaders: RtRestirGiShaders<'_>,
) -> Result<RtRestirGiShaderModules> {
    let shader_specs = [
        (RT_RESTIR_GI_RAYGEN_SPV, shaders.raygen),
        (RT_RESTIR_GI_MISS_SPV, shaders.miss),
        (RT_RESTIR_GI_CLOSEST_HIT_SPV, shaders.closest_hit),
        (RT_RESTIR_GI_INTERSECTION_SPV, shaders.intersection),
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

    Ok(RtRestirGiShaderModules {
        raygen: modules[0],
        miss: modules[1],
        closest_hit: modules[2],
        intersection: modules[3],
    })
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
        "rt_restir_gi_disabled_traversal_stats",
    )
}

fn write_frame_descriptors(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    surface_buffer: &GpuBuffer,
    history_reservoirs: &GpuBuffer,
    output_reservoirs: &GpuBuffer,
    previous_surface_history: &GpuBuffer,
    current_surface_history: &GpuBuffer,
) {
    let surface_info = vk::DescriptorBufferInfo::default()
        .buffer(surface_buffer.handle)
        .offset(0)
        .range(surface_buffer.size);
    let history_info = vk::DescriptorBufferInfo::default()
        .buffer(history_reservoirs.handle)
        .offset(0)
        .range(history_reservoirs.size);
    let output_info = vk::DescriptorBufferInfo::default()
        .buffer(output_reservoirs.handle)
        .offset(0)
        .range(output_reservoirs.size);
    let previous_surface_info = vk::DescriptorBufferInfo::default()
        .buffer(previous_surface_history.handle)
        .offset(0)
        .range(previous_surface_history.size);
    let current_surface_info = vk::DescriptorBufferInfo::default()
        .buffer(current_surface_history.handle)
        .offset(0)
        .range(current_surface_history.size);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&surface_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&history_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&output_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&previous_surface_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&current_surface_info)),
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
        .dst_binding(7)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
        .push_next(&mut tlas_info);
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn write_aabb_descriptor(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    aabb_buffer: &GpuBuffer,
) {
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(aabb_buffer.handle)
        .offset(0)
        .range(aabb_buffer.size);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(8)
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
        &ucvh_gpu.material_buffer,
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
            .dst_binding(9)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&config_info)),
    ];
    writes.extend(buffer_infos.iter().enumerate().map(|(idx, info)| {
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding((idx + 10) as u32)
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
            .dst_binding(17)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info));
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }
}

fn current_surface_history_index(frame_index: u64) -> usize {
    (frame_index as usize) & 1
}

fn previous_surface_history_index(frame_index: u64) -> usize {
    current_surface_history_index(frame_index) ^ 1
}

fn rt_restir_gi_debug_view(debug_view: RtDebugView) -> RestirGiDebugView {
    match debug_view {
        RtDebugView::IndirectReservoir => RestirGiDebugView::ReservoirWeight,
        _ => RestirGiDebugView::Off,
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
    fn rt_restir_gi_descriptor_specs_match_shader_resources() {
        let specs = super::RtRestirGiPass::descriptor_binding_specs();
        let actual = specs
            .iter()
            .map(|spec| (spec.binding, spec.descriptor_type))
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, vk::DescriptorType::UNIFORM_BUFFER),
                (2, vk::DescriptorType::STORAGE_BUFFER),
                (3, vk::DescriptorType::STORAGE_BUFFER),
                (4, vk::DescriptorType::STORAGE_BUFFER),
                (5, vk::DescriptorType::STORAGE_BUFFER),
                (6, vk::DescriptorType::STORAGE_BUFFER),
                (7, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
                (8, vk::DescriptorType::STORAGE_BUFFER),
                (9, vk::DescriptorType::UNIFORM_BUFFER),
                (10, vk::DescriptorType::STORAGE_BUFFER),
                (11, vk::DescriptorType::STORAGE_BUFFER),
                (12, vk::DescriptorType::STORAGE_BUFFER),
                (13, vk::DescriptorType::STORAGE_BUFFER),
                (14, vk::DescriptorType::STORAGE_BUFFER),
                (15, vk::DescriptorType::STORAGE_BUFFER),
                (16, vk::DescriptorType::STORAGE_BUFFER),
                (17, vk::DescriptorType::STORAGE_BUFFER),
            ]
        );
    }

    #[test]
    fn rt_restir_gi_pass_owns_raygen_pipeline_descriptors_sbt_and_reservoirs() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_restir_gi.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT ReSTIR-GI implementation should precede tests");

        for token in [
            "pub struct RtRestirGiPass",
            "RayTracingPipeline",
            "ShaderBindingTable",
            "descriptor_set_layout",
            "descriptor_pool",
            "descriptor_sets",
            "uniform_buffers",
            "history_uniform_buffers",
            "reservoirs",
            "surface_history_buffers",
            "RtRestirGiShaders",
            "rt_restir_gi.rmiss.spv",
            "rt_restir_gi.rchit.spv",
            "rt_restir_gi.rint.spv",
            "RayTracingPipeline::new_surface_pipeline",
            "cmd_trace_rays",
            "update_uniforms",
            "update_history_uniforms",
            "update_frame_descriptors",
            "update_tlas_descriptor",
            "update_aabb_descriptor",
            "update_ucvh_descriptors",
            "write_traversal_stats_descriptor",
            "rt_restir_gi",
        ] {
            assert!(
                implementation.contains(token),
                "RT ReSTIR-GI pass implementation missing {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_pass_ping_pongs_indirect_reservoir_history() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_restir_gi.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT ReSTIR-GI implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "fncurrent_reservoir_slot(&self,frame_slot:usize)->usize",
            "fnhistory_reservoir_slot(&self,frame_slot:usize)->usize",
            "self.current_reservoir_buffer(frame_slot)",
            "self.history_reservoir_buffer(frame_slot)",
            "current_surface_history_buffer",
            "previous_surface_history_buffer",
            "builder.read_as(history_reservoirs",
            "builder.write_as(output_reservoirs",
            "builder.read_as(previous_surface_history",
            "builder.write_as(current_surface_history",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI reservoir history contract missing {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_shader_source_uses_gi_common_and_rt_history_without_deferred_features() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let miss = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rmiss.slang",
        );
        let closest_hit = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rchit.slang",
        );
        let intersection = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rint.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "#include\"restir_gi_common.slang\"",
            "#include\"rt_history_common.slang\"",
            "#include\"rt_surface_common.slang\"",
            "RaytracingAccelerationStructurescene_tlas",
            "StructuredBuffer<RtSurfacePixel>surface_pixels",
            "StructuredBuffer<RestirGiReservoir>history_reservoirs",
            "RWStructuredBuffer<RestirGiReservoir>output_reservoirs",
            "StructuredBuffer<RtSurfacePixel>previous_surface_history",
            "RWStructuredBuffer<RtSurfacePixel>current_surface_history",
            "TraceRay(",
            "rt_restir_gi_trace_indirect_surface",
            "make_rt_surface_payload(indirect_direction)",
            "rt_restir_gi_generate_initial_reservoir(surface,indirect_surface,index,launch_id.xy)",
            "current_surface_history[index]=surface",
            "ConstantBuffer<RtHistoryUniforms>rt_history",
            "mul(float4(world_position,1.0),rt_history.previous_view_proj)",
            "rt_history.normal_threshold",
            "rt_history.depth_threshold",
            "restir_gi.temporal_enabled",
            "restir_gi_finalize_reservoir",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI shader missing {token}"
            );
        }
        assert!(miss.contains("[shader(\"miss\")]"));
        assert!(miss.contains("payload.hit_kind = RT_SURFACE_HIT_KIND_MISS"));
        for token in [
            "[shader(\"closesthit\")]",
            "trace_primary_ray(",
            "material_cell_albedo",
            "payload.hit_kind = RT_SURFACE_HIT_KIND_VOXEL",
        ] {
            assert!(
                closest_hit.contains(token),
                "RT ReSTIR-GI closest-hit shader missing {token}"
            );
        }
        for token in [
            "[shader(\"intersection\")]",
            "StructuredBuffer<RtAabb> rt_aabbs",
            "ReportHit(hit_t",
        ] {
            assert!(
                intersection.contains(token),
                "RT ReSTIR-GI intersection shader missing {token}"
            );
        }

        for forbidden in ["SER", "path guiding", "NRD", "ReBLUR", "RELAX"] {
            assert!(
                !source.contains(forbidden),
                "RT ReSTIR-GI shader must not contain deferred feature token {forbidden}"
            );
        }
    }
}
