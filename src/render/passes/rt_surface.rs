use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::graph::RenderGraph;
use crate::render::pipeline::{RayTracingPipeline, ShaderBindingTable, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::rt_history::{GpuRtHistoryUniforms, GpuRtSurfacePixel};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::voxel::gpu_upload::UcvhGpuResources;

const RT_SURFACE_SHADER: &str = "rt_surface.rgen.slang";
pub(crate) const RT_SURFACE_RAYGEN_SPV: &str = "rt_surface.rgen.spv";
pub(crate) const RT_SURFACE_MISS_SPV: &str = "rt_surface.rmiss.spv";
pub(crate) const RT_SURFACE_CLOSEST_HIT_SPV: &str = "rt_surface.rchit.spv";
pub(crate) const RT_SURFACE_INTERSECTION_SPV: &str = "rt_surface.rint.spv";

#[derive(Clone, Copy)]
pub struct RtSurfaceShaders<'a> {
    pub raygen: &'a [u8],
    pub miss: &'a [u8],
    pub closest_hit: &'a [u8],
    pub intersection: &'a [u8],
}

pub struct RtSurfaceCreateInfo<'a> {
    pub rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub ucvh_gpu: &'a UcvhGpuResources,
    pub shaders: RtSurfaceShaders<'a>,
}

pub struct RtSurfacePass {
    ray_tracing_pipeline_loader: ash::khr::ray_tracing_pipeline::Device,
    pipeline: RayTracingPipeline,
    shader_binding_table: ShaderBindingTable,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    surface_buffer: GpuBuffer,
    history_uniform_buffers: Vec<GpuBuffer>,
    traversal_stats_buffer: GpuBuffer,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
pub struct RtSurfaceGraphOutputs {
    pub surface: ResourceHandle,
}

impl RtSurfacePass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 14] {
        [
            DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
            DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(3, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(4, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(5, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(6, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(7, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(8, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(9, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(10, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(11, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(12, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(13, vk::DescriptorType::UNIFORM_BUFFER),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        ray_tracing_pipeline_loader: &ash::khr::ray_tracing_pipeline::Device,
        info: RtSurfaceCreateInfo<'_>,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding_specs(&Self::descriptor_binding_specs())
            .build(device)?;
        let frame_count = info.scene_ubo.frame_count();
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
                    descriptor_count: (10 * frame_count) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: (3 * frame_count) as u32,
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
        let surface_buffer = match create_surface_buffer(device, allocator, info.width, info.height)
        {
            Ok(buffer) => buffer,
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
                    surface_buffer.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let traversal_stats_buffer = match create_disabled_traversal_stats_buffer(device, allocator)
        {
            Ok(buffer) => buffer,
            Err(error) => {
                destroy_buffers(history_uniform_buffers, device, allocator);
                surface_buffer.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        write_surface_descriptors(device, &descriptor_sets, &surface_buffer);
        write_scene_descriptors(device, &descriptor_sets, info.scene_ubo);
        write_history_descriptors(device, &descriptor_sets, &history_uniform_buffers);
        for &descriptor_set in &descriptor_sets {
            write_ucvh_descriptors(device, descriptor_set, info.ucvh_gpu);
        }
        write_traversal_stats_descriptors(device, &descriptor_sets, &traversal_stats_buffer);

        let shader_modules = match create_rt_surface_shader_modules(device, info.shaders) {
            Ok(modules) => modules,
            Err(error) => {
                traversal_stats_buffer.destroy(device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                surface_buffer.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
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
                traversal_stats_buffer.destroy(device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                surface_buffer.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        shader_modules.destroy(device);
        let shader_binding_table = match ShaderBindingTable::new(
            device,
            allocator,
            ray_tracing_pipeline_loader,
            pipeline.handle,
            info.rt_pipeline_properties,
            pipeline.group_counts,
        ) {
            Ok(table) => table,
            Err(error) => {
                pipeline.destroy(device);
                traversal_stats_buffer.destroy(device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                surface_buffer.destroy(device, allocator);
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        Ok(Self {
            ray_tracing_pipeline_loader: ray_tracing_pipeline_loader.clone(),
            pipeline,
            shader_binding_table,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            surface_buffer,
            history_uniform_buffers,
            traversal_stats_buffer,
            width: info.width,
            height: info.height,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn surface_buffer(&self) -> &GpuBuffer {
        &self.surface_buffer
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

    pub fn update_history_uniforms(&self, frame_slot: usize, uniforms: &GpuRtHistoryUniforms) {
        write_mapped(
            self.history_uniform_buffers[frame_slot].mapped_ptr(),
            uniforms,
        );
    }

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        frame_slot: usize,
        surface_initialized: bool,
    ) -> RtSurfaceGraphOutputs {
        let surface_resource = graph.import_buffer_with_access(
            self.surface_buffer.handle,
            self.surface_buffer.size,
            self.surface_buffer.usage,
            if surface_initialized {
                AccessKind::RayTracingShaderWrite
            } else {
                AccessKind::Undefined
            },
        );
        let history_uniform_buffer = &self.history_uniform_buffers[frame_slot];
        let history_uniform = graph.import_buffer_with_access(
            history_uniform_buffer.handle,
            history_uniform_buffer.size,
            history_uniform_buffer.usage,
            AccessKind::RayTracingShaderRead,
        );
        let ray_tracing_pipeline_loader = self.ray_tracing_pipeline_loader.clone();
        let pipeline = self.pipeline.handle;
        let pipeline_layout = self.pipeline.layout;
        let descriptor_set = self.descriptor_sets[frame_slot];
        let sbt_regions = self.shader_binding_table.regions();
        let width = self.width;
        let height = self.height;

        let writes = graph.add_pass("rt_surface", QueueType::RayTracing, |builder| {
            builder.read_as(history_uniform, AccessKind::RayTracingShaderRead);
            builder.write_as(surface_resource, AccessKind::RayTracingShaderWrite);
            Box::new(move |ctx| {
                let _ = RT_SURFACE_SHADER;
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
            })
        });

        RtSurfaceGraphOutputs { surface: writes[0] }
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let new_surface = create_surface_buffer(device, allocator, width, height)?;
        std::mem::replace(&mut self.surface_buffer, new_surface).destroy(device, allocator);
        self.width = width;
        self.height = height;
        write_surface_descriptors(device, &self.descriptor_sets, &self.surface_buffer);
        Ok(())
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.shader_binding_table.destroy(device, allocator);
        self.pipeline.destroy(device);
        self.traversal_stats_buffer.destroy(device, allocator);
        destroy_buffers(self.history_uniform_buffers, device, allocator);
        self.surface_buffer.destroy(device, allocator);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}

#[derive(Clone, Copy)]
struct RtSurfaceShaderModules {
    raygen: vk::ShaderModule,
    miss: vk::ShaderModule,
    closest_hit: vk::ShaderModule,
    intersection: vk::ShaderModule,
}

impl RtSurfaceShaderModules {
    fn destroy(self, device: &ash::Device) {
        for module in [self.raygen, self.miss, self.closest_hit, self.intersection] {
            unsafe { device.destroy_shader_module(module, None) };
        }
    }
}

fn create_rt_surface_shader_modules(
    device: &ash::Device,
    shaders: RtSurfaceShaders<'_>,
) -> Result<RtSurfaceShaderModules> {
    let shader_specs = [
        (RT_SURFACE_RAYGEN_SPV, shaders.raygen),
        (RT_SURFACE_MISS_SPV, shaders.miss),
        (RT_SURFACE_CLOSEST_HIT_SPV, shaders.closest_hit),
        (RT_SURFACE_INTERSECTION_SPV, shaders.intersection),
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

    Ok(RtSurfaceShaderModules {
        raygen: modules[0],
        miss: modules[1],
        closest_hit: modules[2],
        intersection: modules[3],
    })
}

fn create_surface_buffer(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<GpuBuffer> {
    GpuBuffer::new(
        device,
        allocator,
        surface_buffer_size(width, height),
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
        "rt_surface",
    )
}

fn surface_buffer_size(width: u32, height: u32) -> vk::DeviceSize {
    u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(std::mem::size_of::<GpuRtSurfacePixel>() as u64)
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
            &format!("rt_surface_history_uniforms_{slot}"),
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
        "rt_surface_disabled_traversal_stats",
    )
}

fn write_surface_descriptors(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    buffer: &GpuBuffer,
) {
    for &descriptor_set in descriptor_sets {
        write_surface_descriptor(device, descriptor_set, buffer);
    }
}

fn write_surface_descriptor(
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
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&buffer_info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn write_scene_descriptors(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    scene_ubo: &SceneUniformBuffer,
) {
    for (frame_slot, &descriptor_set) in descriptor_sets.iter().enumerate() {
        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(scene_ubo.buffer_handle(frame_slot))
            .offset(0)
            .range(std::mem::size_of::<GpuSceneUniforms>() as u64);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&ubo_info));
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }
}

fn write_history_descriptors(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    history_uniform_buffers: &[GpuBuffer],
) {
    for (frame_slot, &descriptor_set) in descriptor_sets.iter().enumerate() {
        let history_info = vk::DescriptorBufferInfo::default()
            .buffer(history_uniform_buffers[frame_slot].handle)
            .offset(0)
            .range(std::mem::size_of::<GpuRtHistoryUniforms>() as u64);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(13)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&history_info));
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }
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
        .dst_binding(0)
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
        .dst_binding(3)
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
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&config_info)),
    ];
    writes.extend(buffer_infos.iter().enumerate().map(|(idx, info)| {
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding((idx + 5) as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(info))
    }));
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn write_traversal_stats_descriptors(
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
            .dst_binding(12)
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
    #[test]
    fn rt_surface_shader_source_declares_tlas_and_rt_history_outputs() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_surface.rgen.slang")
            .expect("rt_surface.rgen.slang should be readable");
        assert!(source.contains("rt_history_common.slang"));
        assert!(source.contains("scene_common.slang"));
        assert!(source.contains("AccelerationStructureKHR"));
        assert!(source.contains("RtSurfacePixel"));
        assert!(source.contains("ConstantBuffer<RtHistoryUniforms> rt_history"));
        assert!(source.contains("SceneUniforms"));
    }

    #[test]
    fn rt_surface_shader_binds_tlas_surface_scene_aabbs_and_ucvh() {
        let source = std::fs::read_to_string("assets/shaders/passes/rt_surface.rgen.slang")
            .expect("rt_surface.rgen.slang should be readable");
        let closest_hit = std::fs::read_to_string("assets/shaders/passes/rt_surface.rchit.slang")
            .expect("rt_surface.rchit.slang should be readable");
        let compact = crate::render::source_checks::compact(&source);
        let closest_hit_compact = crate::render::source_checks::compact(&closest_hit);

        assert!(compact.contains("[[vk::binding(0,0)]]RaytracingAccelerationStructurescene_tlas;"));
        assert!(
            compact
                .contains("[[vk::binding(1,0)]]RWStructuredBuffer<RtSurfacePixel>surface_pixels;")
        );
        assert!(compact.contains("[[vk::binding(2,0)]]ConstantBuffer<SceneUniforms>scene_ubo;"));
        assert!(compact.contains("[[vk::binding(3,0)]]StructuredBuffer<RtAabb>rt_aabbs;"));
        assert!(
            compact.contains("[[vk::binding(13,0)]]ConstantBuffer<RtHistoryUniforms>rt_history;")
        );
        for token in [
            "[[vk::binding(4,0)]]ConstantBuffer<UcvhConfig>ucvh_config;",
            "[[vk::binding(5,0)]]StructuredBuffer<NodeL0>hierarchy_l0;",
            "[[vk::binding(6,0)]]StructuredBuffer<NodeLN>hierarchy_l1;",
            "[[vk::binding(7,0)]]StructuredBuffer<NodeLN>hierarchy_l2;",
            "[[vk::binding(8,0)]]StructuredBuffer<NodeLN>hierarchy_l3;",
            "[[vk::binding(9,0)]]StructuredBuffer<NodeLN>hierarchy_l4;",
            "[[vk::binding(10,0)]]StructuredBuffer<BrickOccupancy>brick_occupancy;",
            "[[vk::binding(11,0)]]StructuredBuffer<VoxelCell>brick_materials;",
            "[[vk::binding(12,0)]]RWStructuredBuffer<uint>traversal_stats;",
        ] {
            assert!(
                closest_hit_compact.contains(token),
                "RT surface closest-hit shader missing UCVH binding token {token}"
            );
        }
        assert!(
            !source.contains("TLAS binding is intentionally deferred"),
            "rt_surface shader should no longer advertise a deferred TLAS binding"
        );
        assert!(
            !source.contains("no history binding yet"),
            "rt_surface shader must not advertise a deferred history binding"
        );
    }

    #[test]
    fn rt_surface_descriptor_specs_bind_tlas_surface_scene_aabbs_ucvh_and_stats() {
        let specs = super::RtSurfacePass::descriptor_binding_specs();
        let actual = specs
            .iter()
            .map(|spec| (spec.binding, spec.descriptor_type))
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (0, ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR),
                (1, ash::vk::DescriptorType::STORAGE_BUFFER),
                (2, ash::vk::DescriptorType::UNIFORM_BUFFER),
                (3, ash::vk::DescriptorType::STORAGE_BUFFER),
                (4, ash::vk::DescriptorType::UNIFORM_BUFFER),
                (5, ash::vk::DescriptorType::STORAGE_BUFFER),
                (6, ash::vk::DescriptorType::STORAGE_BUFFER),
                (7, ash::vk::DescriptorType::STORAGE_BUFFER),
                (8, ash::vk::DescriptorType::STORAGE_BUFFER),
                (9, ash::vk::DescriptorType::STORAGE_BUFFER),
                (10, ash::vk::DescriptorType::STORAGE_BUFFER),
                (11, ash::vk::DescriptorType::STORAGE_BUFFER),
                (12, ash::vk::DescriptorType::STORAGE_BUFFER),
                (13, ash::vk::DescriptorType::UNIFORM_BUFFER),
            ]
        );
    }

    #[test]
    fn rt_surface_shader_writes_motion_history_from_rt_history_uniforms() {
        let raygen = std::fs::read_to_string("assets/shaders/passes/rt_surface.rgen.slang")
            .expect("rt_surface.rgen.slang should be readable");
        let common = std::fs::read_to_string("assets/shaders/shared/rt_surface_common.slang")
            .expect("rt_surface common shader should be readable");
        let compact = crate::render::source_checks::compact(&raygen);

        for token in [
            "ConstantBuffer<RtHistoryUniforms>rt_history",
            "boolrt_surface_history_reset_active()",
            "rt_history.flags",
            "float4rt_surface_project_previous_pixel(uint2pixel,float3world_position)",
            "mul(rt_history.current_view_proj,float4(world_position,1.0))",
            "mul(rt_history.previous_view_proj,float4(world_position,1.0))",
            "float2motion_delta=previous_pixel-current_pixel",
            "surface.motion_history=rt_surface_project_previous_pixel(launch_id.xy,surface.position_depth.xyz)",
        ] {
            assert!(
                compact.contains(token),
                "RT surface raygen must write motion/history guide token {token}"
            );
        }
        assert!(
            !common.contains("pixel.motion_history = float4(0.0);"),
            "RT surface common conversion must not hard-code motion_history to a permanent zero guide"
        );
    }

    #[test]
    fn rt_surface_pass_writes_tlas_surface_scene_aabb_ucvh_and_stats_descriptors() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_surface.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT surface implementation should precede tests");

        for token in [
            "WriteDescriptorSetAccelerationStructureKHR",
            "write_tlas_descriptor",
            "update_tlas_descriptor",
            ".descriptor_count(1)",
            "write_scene_descriptors",
            "write_aabb_descriptor",
            "update_aabb_descriptor",
            "write_ucvh_descriptors",
            "update_ucvh_descriptors",
            "write_traversal_stats_descriptors",
            "history_uniform_buffers",
            "update_history_uniforms",
            "write_history_descriptors",
            "ucvh_gpu",
            "DescriptorType::ACCELERATION_STRUCTURE_KHR",
            "DescriptorType::UNIFORM_BUFFER",
            "dst_binding(0)",
            "dst_binding(1)",
            "dst_binding(2)",
            "dst_binding(3)",
            "dst_binding(4)",
            "dst_binding((idx + 5) as u32)",
            "dst_binding(12)",
            "dst_binding(13)",
        ] {
            assert!(
                implementation.contains(token),
                "RT surface descriptor plumbing missing {token}"
            );
        }
    }

    #[test]
    fn rt_surface_graph_uses_frame_slot_descriptor_sets() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_surface.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT surface implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        assert!(compact.contains("descriptor_sets:Vec<vk::DescriptorSet>"));
        assert!(compact.contains("frame_slot:usize"));
        assert!(compact.contains("self.descriptor_sets[frame_slot]"));
    }

    #[test]
    fn rt_surface_runtime_descriptors_are_frame_slot_scoped() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_surface.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT surface implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "pubfnupdate_tlas_descriptor(&self,device:&ash::Device,frame_slot:usize,tlas:vk::AccelerationStructureKHR,)",
            "pubfnupdate_aabb_descriptor(&self,device:&ash::Device,frame_slot:usize,aabb_buffer:&GpuBuffer,)",
            "pubfnupdate_ucvh_descriptors(&self,device:&ash::Device,frame_slot:usize,ucvh_gpu:&UcvhGpuResources,)",
            "self.descriptor_sets.get(frame_slot)",
        ] {
            assert!(
                compact.contains(token),
                "RT surface descriptors must be frame-slot scoped with {token}"
            );
        }
        for forbidden in [
            "for&descriptor_setin&self.descriptor_sets{write_tlas_descriptor",
            "for&descriptor_setin&self.descriptor_sets{write_aabb_descriptor",
            "write_ucvh_descriptors(device,&self.descriptor_sets,ucvh_gpu)",
        ] {
            assert!(
                !compact.contains(forbidden),
                "RT surface must not update pending descriptor sets with {forbidden}"
            );
        }
    }

    #[test]
    fn rt_surface_pass_owns_hardware_trace_pipeline_and_sbt() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_surface.rs");
        for token in [
            "RayTracingPipeline",
            "ShaderBindingTable",
            "ray_tracing_pipeline_loader",
            "cmd_trace_rays",
        ] {
            assert!(source.contains(token), "RT surface pass missing {token}");
        }
    }

    #[test]
    fn rt_surface_pass_loads_all_hardware_rt_stage_modules() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_surface.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT surface implementation should precede tests");

        for token in [
            "rt_surface.rgen.spv",
            "rt_surface.rmiss.spv",
            "rt_surface.rchit.spv",
            "rt_surface.rint.spv",
            "RayTracingPipeline::new_surface_pipeline",
        ] {
            assert!(
                implementation.contains(token),
                "RT surface pass stage module setup missing {token}"
            );
        }
    }

    #[test]
    fn rt_surface_hit_group_shader_sources_exist() {
        let miss = std::fs::read_to_string("assets/shaders/passes/rt_surface.rmiss.slang")
            .expect("rt_surface.rmiss.slang should be readable");
        let closest_hit = std::fs::read_to_string("assets/shaders/passes/rt_surface.rchit.slang")
            .expect("rt_surface.rchit.slang should be readable");
        let intersection = std::fs::read_to_string("assets/shaders/passes/rt_surface.rint.slang")
            .expect("rt_surface.rint.slang should be readable");

        assert!(miss.contains("[shader(\"miss\")]"));
        assert!(closest_hit.contains("[shader(\"closesthit\")]"));
        assert!(intersection.contains("[shader(\"intersection\")]"));
        assert!(intersection.contains("ReportHit"));
    }

    #[test]
    fn rt_surface_raygen_traces_primary_rays_into_payload() {
        let raygen = std::fs::read_to_string("assets/shaders/passes/rt_surface.rgen.slang")
            .expect("rt_surface.rgen.slang should be readable");

        for token in [
            "RtSurfacePayload",
            "RayDesc",
            "scene_primary_ray_from_area_sample",
            "TraceRay(",
            "scene_tlas",
            "surface_pixel_from_payload",
            "surface_pixels[index]",
        ] {
            assert!(
                raygen.contains(token),
                "RT surface raygen missing primary TraceRay token {token}"
            );
        }
    }

    #[test]
    fn rt_surface_miss_and_hit_shaders_fill_payload() {
        let common = std::fs::read_to_string("assets/shaders/shared/rt_surface_common.slang")
            .expect("rt_surface common shader should be readable");
        let miss = std::fs::read_to_string("assets/shaders/passes/rt_surface.rmiss.slang")
            .expect("rt_surface.rmiss.slang should be readable");
        let closest_hit = std::fs::read_to_string("assets/shaders/passes/rt_surface.rchit.slang")
            .expect("rt_surface.rchit.slang should be readable");
        let intersection = std::fs::read_to_string("assets/shaders/passes/rt_surface.rint.slang")
            .expect("rt_surface.rint.slang should be readable");

        for token in [
            "struct RtSurfacePayload",
            "struct RtAabb",
            "surface_pixel_from_payload",
            "RT_SURFACE_HIT_KIND_MISS",
            "RT_SURFACE_HIT_KIND_VOXEL",
            "uint material_id",
            "float roughness",
            "uint brick_id",
            "uint3 local",
            "pixel.material_id = payload.material_id",
            "payload.roughness",
        ] {
            assert!(
                common.contains(token),
                "RT surface common shader missing token {token}"
            );
        }
        for token in ["inout RtSurfacePayload", "RT_SURFACE_HIT_KIND_MISS"] {
            assert!(
                miss.contains(token),
                "RT surface miss shader missing token {token}"
            );
        }
        for token in [
            "inout RtSurfacePayload",
            "RtSurfaceIntersectionAttributes",
            "RayTCurrent()",
            "WorldRayOrigin()",
            "WorldRayDirection()",
            "PrimitiveIndex()",
            "RT_SURFACE_HIT_KIND_VOXEL",
        ] {
            assert!(
                closest_hit.contains(token),
                "RT surface closest-hit shader missing token {token}"
            );
        }
        for token in [
            "StructuredBuffer<RtAabb> rt_aabbs",
            "PrimitiveIndex()",
            "intersect_aabb",
            "ReportHit(hit_t",
        ] {
            assert!(
                intersection.contains(token),
                "RT surface intersection shader missing token {token}"
            );
        }
    }

    #[test]
    fn rt_surface_preserves_primary_ray_direction_for_background_misses() {
        let common = std::fs::read_to_string("assets/shaders/shared/rt_surface_common.slang")
            .expect("rt_surface common shader should be readable");
        let raygen = std::fs::read_to_string("assets/shaders/passes/rt_surface.rgen.slang")
            .expect("rt_surface raygen shader should be readable");
        let miss = std::fs::read_to_string("assets/shaders/passes/rt_surface.rmiss.slang")
            .expect("rt_surface miss shader should be readable");
        let closest_hit = std::fs::read_to_string("assets/shaders/passes/rt_surface.rchit.slang")
            .expect("rt_surface closest-hit shader should be readable");
        let compact_common = crate::render::source_checks::compact(&common);
        let compact_raygen = crate::render::source_checks::compact(&raygen);
        let compact_miss = crate::render::source_checks::compact(&miss);
        let compact_closest_hit = crate::render::source_checks::compact(&closest_hit);

        for token in [
            "float3ray_direction;",
            "pixel.view_direction_background=float4(normalize(payload.ray_direction),payload.hit_kind==RT_SURFACE_HIT_KIND_MISS?1.0:0.0);",
        ] {
            assert!(
                compact_common.contains(token),
                "RT surface payload/pixel must carry explicit background direction; missing {token}"
            );
        }

        assert!(
            compact_raygen.contains(
                "RtSurfacePayloadpayload=make_rt_surface_payload(primary_ray.direction);"
            ),
            "RT surface raygen must seed payload with primary camera ray direction"
        );
        assert!(
            compact_miss.contains("payload.normal=-payload.ray_direction;"),
            "true RT misses should keep a normal suitable for debug views"
        );
        assert!(
            compact_closest_hit.contains("payload.ray_direction=world_direction;"),
            "closest-hit local miss path must restore the original world ray direction for background shading"
        );
        assert!(
            !compact_closest_hit.contains("payload.normal=normalize(attributes.object_normal);"),
            "brick-local misses must not encode AABB normals as the background direction contract"
        );
    }

    #[test]
    fn rt_surface_closest_hit_traverses_real_voxel_materials() {
        let closest_hit = std::fs::read_to_string("assets/shaders/passes/rt_surface.rchit.slang")
            .expect("rt_surface.rchit.slang should be readable");
        let common = std::fs::read_to_string("assets/shaders/shared/rt_surface_common.slang")
            .expect("rt_surface common shader should be readable");

        for token in [
            "voxel_traverse.slang",
            "material_common.slang",
            "HitResult hit = trace_primary_ray(",
            "ucvh_config",
            "hierarchy_l0",
            "hierarchy_l4",
            "brick_occupancy",
            "brick_materials",
            "traversal_stats",
            "false",
            "payload.hit_kind = RT_SURFACE_HIT_KIND_VOXEL",
            "payload.material_id = voxel_material(hit.cell)",
            "payload.roughness = material_cell_roughness(hit.cell)",
            "payload.albedo = material_cell_albedo(hit.cell)",
            "payload.brick_id = hit.brick_id",
            "payload.local = hit.local",
        ] {
            assert!(
                closest_hit.contains(token),
                "RT surface closest-hit shader must derive voxel payload data with {token}"
            );
        }
        assert!(
            !common.contains("pixel.material_id = payload.primitive_index"),
            "RT surface pixels must no longer report AABB primitive index as material id"
        );
    }
}
