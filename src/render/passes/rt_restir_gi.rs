use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::pipeline::{
    RayTracingPipeline, RtHitGroupKind, RtShaderStageSpec, ShaderBindingTable, create_shader_module,
};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::restir_gi::{
    GpuRestirGiReservoir, GpuRestirGiUniforms, RestirGiDebugView, RestirGiLightingUniformInputs,
    RestirGiSettings,
};
use crate::render::rt_history::{GpuRtHistoryUniforms, GpuRtSurfacePixel};
use crate::render::rt_settings::{RtDebugView, RtSettings};
use crate::voxel::gpu_upload::UcvhGpuResources;

pub(crate) const RT_RESTIR_GI_RAYGEN_SPV: &str = "rt_restir_gi.rgen.spv";
pub(crate) const RT_RESTIR_GI_MISS_SPV: &str = "rt_restir_gi.rmiss.spv";
pub(crate) const RT_RESTIR_GI_CLOSEST_HIT_SPV: &str = "rt_restir_gi.rchit.spv";
pub(crate) const RT_RESTIR_GI_INTERSECTION_SPV: &str = "rt_restir_gi.rint.spv";
/// CompactExact triangle GI closest-hit (hit group 1). Uses bindings 18/19 for page data.
pub(crate) const RT_RESTIR_GI_COMPACT_EXACT_CLOSEST_HIT_SPV: &str =
    "rt_compact_exact_gi.rchit.spv";

#[derive(Clone, Copy)]
pub struct RtRestirGiShaders<'a> {
    pub raygen: &'a [u8],
    pub spatial_raygen_spirv: &'a [u8],
    pub miss: &'a [u8],
    /// Reference DDA GI closest-hit (hit group 0, procedural AABB).
    pub closest_hit: &'a [u8],
    pub intersection: &'a [u8],
    /// CompactExact triangle GI closest-hit (hit group 1, triangle BLAS).
    pub compact_exact_closest_hit: &'a [u8],
}

pub struct RtRestirGiPass {
    ray_tracing_pipeline_loader: ash::khr::ray_tracing_pipeline::Device,
    pipeline: RayTracingPipeline,
    shader_binding_table: ShaderBindingTable,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    spatial_pipeline: RayTracingPipeline,
    spatial_shader_binding_table: ShaderBindingTable,
    spatial_descriptor_set_layout: vk::DescriptorSetLayout,
    spatial_descriptor_pool: DescriptorPool,
    spatial_descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_buffers: Vec<GpuBuffer>,
    history_uniform_buffers: Vec<GpuBuffer>,
    temporal_reservoirs: GpuBuffer,
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
    pub spatial_rendered: bool,
}

#[derive(Clone, Copy)]
pub struct RtRestirGiFrameSettings {
    pub rt_settings: RtSettings,
    pub frame_index: u64,
    pub history_initialized: bool,
    pub sun_direction: glam::Vec3,
    pub sun_intensity: glam::Vec3,
    pub sun_angular_radius: f32,
}

impl RtRestirGiPass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 20] {
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
            // CompactExact page geometry buffers (rt_compact_exact_gi.rchit.slang)
            DescriptorBindingSpec::ray_tracing(18, vk::DescriptorType::STORAGE_BUFFER), // face_records
            DescriptorBindingSpec::ray_tracing(19, vk::DescriptorType::STORAGE_BUFFER), // page_records
        ]
    }

    pub(crate) fn spatial_descriptor_binding_specs() -> [DescriptorBindingSpec; 16] {
        [
            DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(3, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(4, vk::DescriptorType::UNIFORM_BUFFER),
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
                    descriptor_count: (16 * frame_count) as u32,
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
        let (spatial_descriptor_set_layout, spatial_descriptor_pool, spatial_descriptor_sets) =
            match create_descriptor_sets(
                device,
                &Self::spatial_descriptor_binding_specs(),
                frame_count,
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
                        descriptor_count: (12 * frame_count) as u32,
                    },
                ],
            ) {
                Ok(resources) => resources,
                Err(error) => {
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let uniform_buffers = match create_uniform_buffers(device, allocator, frame_count) {
            Ok(buffers) => buffers,
            Err(error) => {
                destroy_descriptor_resources(
                    device,
                    spatial_descriptor_set_layout,
                    &spatial_descriptor_pool,
                );
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
                    destroy_descriptor_resources(
                        device,
                        spatial_descriptor_set_layout,
                        &spatial_descriptor_pool,
                    );
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let reservoir_count = info.width.saturating_mul(info.height);
        let temporal_reservoirs = match create_reservoir_buffer(
            device,
            allocator,
            reservoir_count,
            "rt_restir_gi_temporal",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                destroy_buffers(history_uniform_buffers, device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                destroy_descriptor_resources(
                    device,
                    spatial_descriptor_set_layout,
                    &spatial_descriptor_pool,
                );
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let reservoirs =
            match create_reservoir_buffers(device, allocator, frame_count, reservoir_count) {
                Ok(buffers) => buffers,
                Err(error) => {
                    temporal_reservoirs.destroy(device, allocator);
                    destroy_buffers(history_uniform_buffers, device, allocator);
                    destroy_buffers(uniform_buffers, device, allocator);
                    destroy_descriptor_resources(
                        device,
                        spatial_descriptor_set_layout,
                        &spatial_descriptor_pool,
                    );
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
                    temporal_reservoirs.destroy(device, allocator);
                    destroy_buffers(history_uniform_buffers, device, allocator);
                    destroy_buffers(uniform_buffers, device, allocator);
                    destroy_descriptor_resources(
                        device,
                        spatial_descriptor_set_layout,
                        &spatial_descriptor_pool,
                    );
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
                temporal_reservoirs.destroy(device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                destroy_descriptor_resources(
                    device,
                    spatial_descriptor_set_layout,
                    &spatial_descriptor_pool,
                );
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
                temporal_reservoirs.destroy(device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                destroy_descriptor_resources(
                    device,
                    spatial_descriptor_set_layout,
                    &spatial_descriptor_pool,
                );
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let (spatial_pipeline, spatial_shader_binding_table) = match create_spatial_raygen_pipeline(
            device,
            allocator,
            info.ray_tracing_pipeline_loader,
            info.rt_pipeline_properties,
            info.shaders,
            spatial_descriptor_set_layout,
        ) {
            Ok(resources) => resources,
            Err(error) => {
                shader_binding_table.destroy(device, allocator);
                pipeline.destroy(device);
                traversal_stats_buffer.destroy(device, allocator);
                destroy_buffers(Vec::from(surface_history_buffers), device, allocator);
                destroy_buffers(reservoirs, device, allocator);
                temporal_reservoirs.destroy(device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                destroy_descriptor_resources(
                    device,
                    spatial_descriptor_set_layout,
                    &spatial_descriptor_pool,
                );
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
            spatial_pipeline,
            spatial_shader_binding_table,
            spatial_descriptor_set_layout,
            spatial_descriptor_pool,
            spatial_descriptor_sets,
            uniform_buffers,
            history_uniform_buffers,
            temporal_reservoirs,
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
        for &descriptor_set in &pass.spatial_descriptor_sets {
            write_ucvh_descriptors(device, descriptor_set, info.ucvh_gpu);
        }
        write_traversal_stats_descriptor(
            device,
            &pass.descriptor_sets,
            &pass.traversal_stats_buffer,
        );
        write_traversal_stats_descriptor(
            device,
            &pass.spatial_descriptor_sets,
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

    pub fn update_uniforms(&self, frame_slot: usize, frame_settings: RtRestirGiFrameSettings) {
        let rt_settings = frame_settings.rt_settings;
        let settings = RestirGiSettings {
            enabled: rt_settings.restir_gi_enabled,
            temporal_enabled: rt_settings.restir_gi_enabled && frame_settings.history_initialized,
            spatial_enabled: rt_settings.restir_gi_enabled
                && frame_settings.history_initialized
                && rt_settings.restir_gi_spatial_enabled
                && rt_settings.restir_gi_spatial_sample_count > 0,
            spatial_sample_count: if frame_settings.history_initialized
                && rt_settings.restir_gi_spatial_enabled
            {
                rt_settings.restir_gi_spatial_sample_count.min(8)
            } else {
                0
            },
            initial_candidate_count: rt_settings.restir_gi_initial_candidate_count.clamp(1, 16),
            history_length: rt_settings.history_length.max(1),
            max_bounces: 1,
            debug_view: rt_restir_gi_debug_view(rt_settings.debug_view),
        };
        let uniforms = settings.gpu_uniforms(
            frame_settings.frame_index as u32,
            self.reservoir_count,
            self.width,
            self.height,
            RestirGiLightingUniformInputs {
                sun_direction: frame_settings.sun_direction,
                sun_intensity: frame_settings.sun_intensity,
                sun_angular_radius: frame_settings.sun_angular_radius,
            },
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
        spatial_enabled: bool,
    ) {
        let Some(&descriptor_set) = self.descriptor_sets.get(frame_slot) else {
            return;
        };
        write_frame_descriptors(
            device,
            descriptor_set,
            surface_buffer,
            self.history_reservoir_buffer(frame_slot),
            if spatial_enabled {
                &self.temporal_reservoirs
            } else {
                self.current_reservoir_buffer(frame_slot)
            },
            self.previous_surface_history_buffer(frame_index),
            self.current_surface_history_buffer(frame_index),
        );
        if let Some(&spatial_descriptor_set) = self.spatial_descriptor_sets.get(frame_slot) {
            write_spatial_frame_descriptors(
                device,
                spatial_descriptor_set,
                &self.temporal_reservoirs,
                self.current_reservoir_buffer(frame_slot),
                surface_buffer,
            );
        }
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
        if let Some(&spatial_descriptor_set) = self.spatial_descriptor_sets.get(frame_slot) {
            write_tlas_descriptor(device, spatial_descriptor_set, tlas);
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
        if let Some(&spatial_descriptor_set) = self.spatial_descriptor_sets.get(frame_slot) {
            write_aabb_descriptor(device, spatial_descriptor_set, aabb_buffer);
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
        if let Some(&spatial_descriptor_set) = self.spatial_descriptor_sets.get(frame_slot) {
            write_ucvh_descriptors(device, spatial_descriptor_set, ucvh_gpu);
        }
    }

    /// Write CompactExact page buffers to bindings 18 (face_records) and 19 (page_records).
    pub fn update_rt_page_descriptors(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        face_buffer: &GpuBuffer,
        page_record_buffer: &GpuBuffer,
    ) {
        let Some(&descriptor_set) = self.descriptor_sets.get(frame_slot) else {
            return;
        };
        for (binding, buffer) in [(18u32, face_buffer), (19u32, page_record_buffer)] {
            let buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(buffer.handle)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        frame_slot: usize,
        frame_index: u64,
        surface: ResourceHandle,
        history_initialized: bool,
        spatial_enabled: bool,
        profiler: Option<&'a GpuProfiler>,
    ) -> RtRestirGiGraphOutputs {
        let spatial_active = spatial_enabled && history_initialized;
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
        let temporal_reservoir_buffer = &self.temporal_reservoirs;
        let output_reservoirs = graph.import_buffer_with_access(
            current_reservoir_buffer.handle,
            current_reservoir_buffer.size,
            current_reservoir_buffer.usage,
            AccessKind::Undefined,
        );
        let temporal_reservoir = graph.import_buffer_with_access(
            temporal_reservoir_buffer.handle,
            temporal_reservoir_buffer.size,
            temporal_reservoir_buffer.usage,
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
        let spatial_pipeline = self.spatial_pipeline.handle;
        let spatial_pipeline_layout = self.spatial_pipeline.layout;
        let spatial_descriptor_set = self.spatial_descriptor_sets[frame_slot];
        let spatial_sbt_regions = self.spatial_shader_binding_table.regions();
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
            let temporal_output = if spatial_active {
                temporal_reservoir
            } else {
                output_reservoirs
            };
            builder.write_as(temporal_output, AccessKind::RayTracingShaderWrite);
            builder.write_as(current_surface_history, AccessKind::RayTracingShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::RtRestirGi,
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
                        GpuProfileScope::RtRestirGi,
                    );
                }
            })
        });
        let temporal_reservoir_output = writes[0];
        let final_reservoirs = if spatial_active {
            let ray_tracing_pipeline_loader = self.ray_tracing_pipeline_loader.clone();
            let spatial_writes =
                graph.add_pass("rt_restir_gi_spatial", QueueType::RayTracing, |builder| {
                    builder.read_as(uniform, AccessKind::RayTracingShaderRead);
                    builder.read_as(history_uniform, AccessKind::RayTracingShaderRead);
                    builder.read_as(temporal_reservoir_output, AccessKind::RayTracingShaderRead);
                    builder.read_as(surface, AccessKind::RayTracingShaderRead);
                    builder.write_as(output_reservoirs, AccessKind::RayTracingShaderWrite);
                    Box::new(move |ctx| {
                        if let Some(profiler) = profiler {
                            profiler.begin_scope(
                                ctx.device,
                                ctx.command_buffer,
                                frame_slot,
                                GpuProfileScope::RtRestirGiSpatial,
                            );
                        }
                        unsafe {
                            ctx.device.cmd_bind_pipeline(
                                ctx.command_buffer,
                                vk::PipelineBindPoint::RAY_TRACING_KHR,
                                spatial_pipeline,
                            );
                            ctx.device.cmd_bind_descriptor_sets(
                                ctx.command_buffer,
                                vk::PipelineBindPoint::RAY_TRACING_KHR,
                                spatial_pipeline_layout,
                                0,
                                std::slice::from_ref(&spatial_descriptor_set),
                                &[],
                            );
                            ray_tracing_pipeline_loader.cmd_trace_rays(
                                ctx.command_buffer,
                                &spatial_sbt_regions.raygen,
                                &spatial_sbt_regions.miss,
                                &spatial_sbt_regions.hit,
                                &spatial_sbt_regions.callable,
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
                                GpuProfileScope::RtRestirGiSpatial,
                            );
                        }
                    })
                });
            spatial_writes[0]
        } else {
            writes[0]
        };

        RtRestirGiGraphOutputs {
            reservoirs: final_reservoirs,
            spatial_rendered: spatial_active,
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
        let temporal_reservoirs =
            create_reservoir_buffer(device, allocator, reservoir_count, "rt_restir_gi_temporal")?;
        let reservoirs = match create_reservoir_buffers(
            device,
            allocator,
            self.reservoirs.len(),
            reservoir_count,
        ) {
            Ok(buffers) => buffers,
            Err(error) => {
                temporal_reservoirs.destroy(device, allocator);
                return Err(error);
            }
        };
        let surface_history_buffers =
            match create_surface_history_buffers(device, allocator, width, height) {
                Ok(buffers) => buffers,
                Err(error) => {
                    destroy_buffers(reservoirs, device, allocator);
                    temporal_reservoirs.destroy(device, allocator);
                    return Err(error);
                }
            };
        std::mem::replace(&mut self.temporal_reservoirs, temporal_reservoirs)
            .destroy(device, allocator);
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
        self.spatial_shader_binding_table.destroy(device, allocator);
        self.spatial_pipeline.destroy(device);
        self.shader_binding_table.destroy(device, allocator);
        self.pipeline.destroy(device);
        self.traversal_stats_buffer.destroy(device, allocator);
        destroy_buffers(Vec::from(self.surface_history_buffers), device, allocator);
        destroy_buffers(self.reservoirs, device, allocator);
        self.temporal_reservoirs.destroy(device, allocator);
        destroy_buffers(self.history_uniform_buffers, device, allocator);
        destroy_buffers(self.uniform_buffers, device, allocator);
        self.spatial_descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.spatial_descriptor_set_layout, None) };
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
        for (set_idx, &descriptor_set) in self.spatial_descriptor_sets.iter().enumerate() {
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
                    .dst_binding(4)
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

fn create_descriptor_sets(
    device: &ash::Device,
    bindings: &[DescriptorBindingSpec],
    frame_count: usize,
    pool_sizes: &[vk::DescriptorPoolSize],
) -> Result<(
    vk::DescriptorSetLayout,
    DescriptorPool,
    Vec<vk::DescriptorSet>,
)> {
    let descriptor_set_layout = DescriptorLayoutBuilder::new()
        .add_binding_specs(bindings)
        .build(device)?;
    let descriptor_pool = match DescriptorPool::new(device, frame_count as u32, pool_sizes) {
        Ok(pool) => pool,
        Err(error) => {
            unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
            return Err(error);
        }
    };
    let descriptor_layouts = vec![descriptor_set_layout; frame_count];
    let descriptor_sets = match descriptor_pool.allocate(device, descriptor_layouts.as_slice()) {
        Ok(sets) => sets,
        Err(error) => {
            descriptor_pool.destroy(device);
            unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
            return Err(error);
        }
    };
    Ok((descriptor_set_layout, descriptor_pool, descriptor_sets))
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
    compact_exact_closest_hit: vk::ShaderModule,
}

impl RtRestirGiShaderModules {
    fn destroy(self, device: &ash::Device) {
        for module in [
            self.raygen,
            self.miss,
            self.closest_hit,
            self.intersection,
            self.compact_exact_closest_hit,
        ] {
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
    let pipeline = match {
        let stages = [
            RtShaderStageSpec { stage: vk::ShaderStageFlags::RAYGEN_KHR,      module: shader_modules.raygen,                    entry_point: c"main" },
            RtShaderStageSpec { stage: vk::ShaderStageFlags::MISS_KHR,        module: shader_modules.miss,                      entry_point: c"main" },
            RtShaderStageSpec { stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR, module: shader_modules.closest_hit,               entry_point: c"main" },
            RtShaderStageSpec { stage: vk::ShaderStageFlags::INTERSECTION_KHR,module: shader_modules.intersection,               entry_point: c"main" },
            RtShaderStageSpec { stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR, module: shader_modules.compact_exact_closest_hit,  entry_point: c"main" },
        ];
        let hit_groups = [
            RtHitGroupKind::Procedural { closest_hit_stage: 2, intersection_stage: 3 },
            RtHitGroupKind::Triangles  { closest_hit_stage: 4 },
        ];
        RayTracingPipeline::new_mixed_surface_pipeline(device, ray_tracing_pipeline_loader, &stages, &hit_groups, &[descriptor_set_layout], &[])
    } {
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

fn create_spatial_raygen_pipeline(
    device: &ash::Device,
    allocator: &GpuAllocator,
    ray_tracing_pipeline_loader: &ash::khr::ray_tracing_pipeline::Device,
    rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    shaders: RtRestirGiShaders<'_>,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<(RayTracingPipeline, ShaderBindingTable)> {
    let shader_modules = create_rt_restir_gi_spatial_shader_modules(device, shaders)?;
    let pipeline = match {
        let stages = [
            RtShaderStageSpec { stage: vk::ShaderStageFlags::RAYGEN_KHR,      module: shader_modules.raygen,                    entry_point: c"main" },
            RtShaderStageSpec { stage: vk::ShaderStageFlags::MISS_KHR,        module: shader_modules.miss,                      entry_point: c"main" },
            RtShaderStageSpec { stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR, module: shader_modules.closest_hit,               entry_point: c"main" },
            RtShaderStageSpec { stage: vk::ShaderStageFlags::INTERSECTION_KHR,module: shader_modules.intersection,               entry_point: c"main" },
            RtShaderStageSpec { stage: vk::ShaderStageFlags::CLOSEST_HIT_KHR, module: shader_modules.compact_exact_closest_hit,  entry_point: c"main" },
        ];
        let hit_groups = [
            RtHitGroupKind::Procedural { closest_hit_stage: 2, intersection_stage: 3 },
            RtHitGroupKind::Triangles  { closest_hit_stage: 4 },
        ];
        RayTracingPipeline::new_mixed_surface_pipeline(device, ray_tracing_pipeline_loader, &stages, &hit_groups, &[descriptor_set_layout], &[])
    } {
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
    let shader_specs: &[(&str, &[u8])] = &[
        (RT_RESTIR_GI_RAYGEN_SPV, shaders.raygen),
        (RT_RESTIR_GI_MISS_SPV, shaders.miss),
        (RT_RESTIR_GI_CLOSEST_HIT_SPV, shaders.closest_hit),
        (RT_RESTIR_GI_INTERSECTION_SPV, shaders.intersection),
        (
            RT_RESTIR_GI_COMPACT_EXACT_CLOSEST_HIT_SPV,
            shaders.compact_exact_closest_hit,
        ),
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
        compact_exact_closest_hit: modules[4],
    })
}

fn create_rt_restir_gi_spatial_shader_modules(
    device: &ash::Device,
    shaders: RtRestirGiShaders<'_>,
) -> Result<RtRestirGiShaderModules> {
    let shader_specs: &[(&str, &[u8])] = &[
        ("rt_restir_gi_spatial.rgen.spv", shaders.spatial_raygen_spirv),
        (RT_RESTIR_GI_MISS_SPV, shaders.miss),
        (RT_RESTIR_GI_CLOSEST_HIT_SPV, shaders.closest_hit),
        (RT_RESTIR_GI_INTERSECTION_SPV, shaders.intersection),
        (
            RT_RESTIR_GI_COMPACT_EXACT_CLOSEST_HIT_SPV,
            shaders.compact_exact_closest_hit,
        ),
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
        compact_exact_closest_hit: modules[4],
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

fn write_spatial_frame_descriptors(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    temporal_reservoirs: &GpuBuffer,
    output_reservoirs: &GpuBuffer,
    surface_buffer: &GpuBuffer,
) {
    let temporal_info = vk::DescriptorBufferInfo::default()
        .buffer(temporal_reservoirs.handle)
        .offset(0)
        .range(temporal_reservoirs.size);
    let output_info = vk::DescriptorBufferInfo::default()
        .buffer(output_reservoirs.handle)
        .offset(0)
        .range(output_reservoirs.size);
    let surface_info = vk::DescriptorBufferInfo::default()
        .buffer(surface_buffer.handle)
        .offset(0)
        .range(surface_buffer.size);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&temporal_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&output_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&surface_info)),
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
        RtDebugView::GiTemporal => RestirGiDebugView::TemporalValid,
        RtDebugView::GiSpatial => RestirGiDebugView::SpatialValid,
        _ => RestirGiDebugView::Off,
    }
}

fn destroy_buffers(buffers: Vec<GpuBuffer>, device: &ash::Device, allocator: &GpuAllocator) {
    for buffer in buffers {
        buffer.destroy(device, allocator);
    }
}

fn destroy_descriptor_resources(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: &DescriptorPool,
) {
    descriptor_pool.destroy(device);
    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
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
                (18, vk::DescriptorType::STORAGE_BUFFER), // face_records
                (19, vk::DescriptorType::STORAGE_BUFFER), // page_records
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
            "rt_compact_exact_gi.rchit.spv",
            "RayTracingPipeline::new_mixed_surface_pipeline",
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
            "#include\"restir_di_common.slang\"",
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
            "rt_restir_gi_generate_initial_reservoir(surface,index,launch_id.xy)",
            "rt_restir_gi_make_initial_candidate(surface,indirect_surface,indirect_direction,rng_state)",
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
        let compact_closest_hit = crate::render::source_checks::compact(&closest_hit);
        let compact_intersection = crate::render::source_checks::compact(&intersection);
        for token in [
            "[shader(\"closesthit\")]",
            "material_cell_albedo",
            "payload.hit_kind = RT_SURFACE_HIT_KIND_VOXEL",
        ] {
            assert!(
                closest_hit.contains(token),
                "RT ReSTIR-GI closest-hit shader missing {token}"
            );
        }
        for token in [
            "uintbrick_id=attributes.brick_id;",
            "uint3local=rt_surface_unpack_local(attributes.packed_local_normal);",
            "VoxelCellcell=brick_materials[brick_id*512u+morton_encode(local)];",
        ] {
            assert!(
                compact_closest_hit.contains(token),
                "RT ReSTIR-GI closest-hit must consume reported voxel identity token {token}"
            );
        }
        assert!(
            !closest_hit.contains("trace_primary_ray(")
                && !closest_hit.contains("voxel_traverse.slang")
                && !closest_hit.contains("hierarchy_l0")
                && !closest_hit.contains("brick_occupancy"),
            "RT ReSTIR-GI closest-hit must not retrace the UCVH after intersection found the voxel"
        );
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
        for token in [
            "brick_dda(",
            "attributes.brick_id=node.brick_id;",
            "attributes.packed_local_normal=rt_surface_pack_local_normal(hit_local,hit_normal);",
        ] {
            assert!(
                compact_intersection.contains(token),
                "RT ReSTIR-GI intersection must report a real voxel identity token {token}"
            );
        }

        for forbidden in [
            "ShaderExecutionReordering",
            "NV_shader_invocation_reorder",
            "path guiding",
            "NRD",
            "ReBLUR",
            "RELAX",
        ] {
            assert!(
                !source.contains(forbidden),
                "RT ReSTIR-GI shader must not contain deferred feature token {forbidden}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_temporal_debug_classifies_reuse_rejection_reasons() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let common = crate::render::source_checks::read_source(
            "assets/shaders/shared/restir_gi_common.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "RESTIR_GI_DEBUG_VIEW_TEMPORAL_VALID",
            "RESTIR_GI_TEMPORAL_REUSE_ACCEPTED",
            "RESTIR_GI_TEMPORAL_REUSE_RESET_OR_DISABLED",
            "RESTIR_GI_TEMPORAL_REUSE_INVALID_RESOLUTION",
            "RESTIR_GI_TEMPORAL_REUSE_REPROJECTION_FAILED",
            "RESTIR_GI_TEMPORAL_REUSE_PREVIOUS_UV_OUTSIDE",
            "RESTIR_GI_TEMPORAL_REUSE_SURFACE_INCOMPATIBLE",
            "RESTIR_GI_TEMPORAL_REUSE_HISTORY_RESERVOIR_INVALID",
            "RESTIR_GI_TEMPORAL_REUSE_CURRENT_WEIGHT_ZERO",
            "RESTIR_GI_TEMPORAL_REUSE_HISTORY_WEIGHT_ZERO",
            "RESTIR_GI_TEMPORAL_REUSE_COMBINED_WEIGHT_ZERO",
        ] {
            assert!(
                common.contains(token) || source.contains(token),
                "RT ReSTIR-GI temporal debug missing reason token {token}"
            );
        }

        for token in [
            "boolrt_restir_gi_temporal_debug_enabled()",
            "RestirGiReservoirrt_restir_gi_record_temporal_debug(RestirGiReservoirreservoir,uintreason)",
            "reservoir.sample_radiance_pdf.w=float(reason)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_RESET_OR_DISABLED)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_INVALID_RESOLUTION)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_REPROJECTION_FAILED)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_PREVIOUS_UV_OUTSIDE)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_SURFACE_INCOMPATIBLE)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_HISTORY_RESERVOIR_INVALID)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_CURRENT_WEIGHT_ZERO)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_HISTORY_WEIGHT_ZERO)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_COMBINED_WEIGHT_ZERO)",
            "returnrt_restir_gi_record_temporal_debug(reservoir,RESTIR_GI_TEMPORAL_REUSE_ACCEPTED)",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI temporal debug must classify reuse path with {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_temporal_samples_fractional_reprojection_without_upper_left_bias() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "staticconstint2rt_restir_gi_temporal_tap_offsets[4]",
            "float2previous_sample=previous_uv*float2(previous_extent)-float2(0.5)",
            "int2previous_base_pixel=int2(floor(previous_sample))",
            "float2history_fraction=saturate(previous_sample-float2(previous_base_pixel))",
            "floatrt_restir_gi_temporal_tap_weight(uinttap,float2history_fraction)",
            "boolrt_restir_gi_temporal_tap_inside(int2previous_pixel,uint2extent)",
            "floathistory_tap_weight_sum=0.0",
            "selected_history=history_reservoirs[previous_index]",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI temporal reservoir reuse must sample fractional reprojection without upper-left bias; missing {token}"
            );
        }

        assert!(
            !compact
                .contains("uint2previous_pixel=uint2(clamp(previous_uv*float2(previous_extent)"),
            "RT ReSTIR-GI temporal reservoir reuse must not truncate fractional previous pixels directly"
        );
    }

    #[test]
    fn rt_restir_gi_temporal_weights_fractional_history_taps_by_reservoir_stream_weight() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "floatcandidate_stream_weight=restir_gi_reservoir_stream_weight(",
            "floathistory_candidate_weight=candidate_stream_weight*tap_weight",
            "if(history_candidate_weight<=0.0){continue;}",
            "history_tap_weight_sum+=history_candidate_weight",
            "floatnext_selection_sum=history_tap_selection_sum+history_candidate_weight",
            "gi_rand01(rng_state)*max(next_selection_sum,1.0e-4)<=history_candidate_weight",
            "floathistory_weight=history_tap_weight_sum",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI temporal reservoir taps must be weighted by both bilinear tap and reservoir stream weight; missing {token}"
            );
        }

        assert!(
            !compact.contains("history_tap_weight_sum+=tap_weight"),
            "RT ReSTIR-GI temporal reuse must not ignore reservoir stream weight when summing history taps"
        );
        assert!(
            !compact.contains("floatnext_selection_sum=history_tap_selection_sum+tap_weight"),
            "RT ReSTIR-GI temporal reuse must not select history taps by bilinear weight alone"
        );
    }

    #[test]
    fn rt_restir_gi_temporal_compatibility_uses_strict_rt_position_threshold() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        assert!(
            compact.contains(
                "position_delta<=rt_history_position_threshold(current.position_depth.w,rt_history.depth_threshold)"
            ),
            "RT ReSTIR-GI temporal reuse must use the shared strict position threshold"
        );
        assert!(
            !compact.contains("position_delta<=max(1.0,depth_scale*rt_history.depth_threshold)"),
            "RT ReSTIR-GI temporal reuse must not accept a full voxel of mismatched history"
        );
    }

    #[test]
    fn rt_restir_gi_spatial_compatibility_uses_strict_rt_position_threshold() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi_spatial.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        assert!(
            compact.contains(
                "position_delta<=rt_history_position_threshold(center.position_depth.w,rt_history.depth_threshold)"
            ),
            "RT ReSTIR-GI spatial reuse must use the shared strict position threshold"
        );
        assert!(
            !compact.contains("position_delta<=max(1.0,depth_scale*rt_history.depth_threshold)"),
            "RT ReSTIR-GI spatial reuse must not accept a full voxel of mismatched neighbors"
        );
    }

    #[test]
    fn rt_restir_gi_temporal_debug_view_is_routed_to_visible_rt_output() {
        let rt_settings = crate::render::source_checks::read_source("src/render/rt_settings.rs");
        let rt_history = crate::render::source_checks::read_source(
            "assets/shaders/shared/rt_history_common.slang",
        );
        let pass = crate::render::source_checks::read_source("src/render/passes/rt_restir_gi.rs");
        let direct = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_direct_lighting.rgen.slang",
        );
        let temporal = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_temporal.rgen.slang",
        );
        let ui = crate::render::source_checks::read_source("src/editor/ui.rs");
        let rt_pipeline = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let vpt_pipeline = crate::render::source_checks::read_source("src/render/vpt_pipeline.rs");

        for (name, source, token) in [
            ("rt settings enum", &rt_settings, "GiTemporal"),
            (
                "rt settings gpu value",
                &rt_settings,
                "Self::GiTemporal => 7",
            ),
            ("rt settings parser", &rt_settings, "gi_temporal"),
            (
                "shared RT debug ABI",
                &rt_history,
                "RT_DEBUG_VIEW_GI_TEMPORAL = 7u",
            ),
            (
                "GI pass routing",
                &pass,
                "RtDebugView::GiTemporal => RestirGiDebugView::TemporalValid",
            ),
            (
                "direct-lighting visibility",
                &direct,
                "rt_direct.debug_view == RT_DEBUG_VIEW_GI_TEMPORAL",
            ),
            (
                "temporal bypass",
                &temporal,
                "debug_view == RT_DEBUG_VIEW_GI_TEMPORAL",
            ),
            (
                "editor option",
                &ui,
                "(RtDebugView::GiTemporal, \"GI Temporal\")",
            ),
            (
                "RT capture name",
                &rt_pipeline,
                "RtDebugView::GiTemporal => \"gi_temporal\"",
            ),
            (
                "VPT capture name",
                &vpt_pipeline,
                "RtDebugView::GiTemporal => \"gi_temporal\"",
            ),
        ] {
            assert!(
                source.contains(token),
                "{name} must route RT GI temporal debug with token {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_spatial_debug_view_is_routed_to_visible_rt_output() {
        let rt_settings = crate::render::source_checks::read_source("src/render/rt_settings.rs");
        let rt_history = crate::render::source_checks::read_source(
            "assets/shaders/shared/rt_history_common.slang",
        );
        let gi_common = crate::render::source_checks::read_source(
            "assets/shaders/shared/restir_gi_common.slang",
        );
        let pass = crate::render::source_checks::read_source("src/render/passes/rt_restir_gi.rs");
        let direct = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_direct_lighting.rgen.slang",
        );
        let temporal = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_temporal.rgen.slang",
        );
        let ui = crate::render::source_checks::read_source("src/editor/ui.rs");
        let rt_pipeline = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let vpt_pipeline = crate::render::source_checks::read_source("src/render/vpt_pipeline.rs");

        for (name, source, token) in [
            ("rt settings enum", &rt_settings, "GiSpatial"),
            (
                "rt settings gpu value",
                &rt_settings,
                "Self::GiSpatial => 8",
            ),
            ("rt settings parser", &rt_settings, "gi_spatial"),
            (
                "shared RT debug ABI",
                &rt_history,
                "RT_DEBUG_VIEW_GI_SPATIAL = 8u",
            ),
            (
                "GI common debug ABI",
                &gi_common,
                "RESTIR_GI_DEBUG_VIEW_SPATIAL_VALID = 3u",
            ),
            (
                "GI pass routing",
                &pass,
                "RtDebugView::GiSpatial => RestirGiDebugView::SpatialValid",
            ),
            (
                "direct-lighting visibility",
                &direct,
                "rt_direct.debug_view == RT_DEBUG_VIEW_GI_SPATIAL",
            ),
            (
                "temporal bypass",
                &temporal,
                "debug_view == RT_DEBUG_VIEW_GI_SPATIAL",
            ),
            (
                "editor option",
                &ui,
                "(RtDebugView::GiSpatial, \"GI Spatial\")",
            ),
            (
                "RT capture name",
                &rt_pipeline,
                "RtDebugView::GiSpatial => \"gi_spatial\"",
            ),
            (
                "VPT capture name",
                &vpt_pipeline,
                "RtDebugView::GiSpatial => \"gi_spatial\"",
            ),
        ] {
            assert!(
                source.contains(token),
                "{name} must route RT GI spatial debug with token {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_spatial_debug_classifies_reuse_reasons() {
        let common = crate::render::source_checks::read_source(
            "assets/shaders/shared/restir_gi_common.slang",
        );
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi_spatial.rgen.slang",
        );
        let direct = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_direct_lighting.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "RESTIR_GI_SPATIAL_REUSE_ACCEPTED_CENTER",
            "RESTIR_GI_SPATIAL_REUSE_DISABLED_OR_PASSTHROUGH",
            "RESTIR_GI_SPATIAL_REUSE_INVALID_SURFACE",
            "RESTIR_GI_SPATIAL_REUSE_CENTER_RESERVOIR_INVALID",
            "RESTIR_GI_SPATIAL_REUSE_NO_COMPATIBLE_NEIGHBOR",
            "RESTIR_GI_SPATIAL_REUSE_ACCEPTED_NEIGHBOR",
            "RESTIR_GI_SPATIAL_REUSE_VISIBILITY_REJECTED",
        ] {
            assert!(
                common.contains(token) || source.contains(token),
                "RT ReSTIR-GI spatial debug missing reason token {token}"
            );
        }

        for token in [
            "boolrt_restir_gi_spatial_debug_enabled()",
            "RestirGiReservoirrt_restir_gi_record_spatial_debug(RestirGiReservoirreservoir,uintreason)",
            "reservoir.sample_radiance_pdf.w=float(reason)",
            "returnrt_restir_gi_record_spatial_debug(reservoir,RESTIR_GI_SPATIAL_REUSE_DISABLED_OR_PASSTHROUGH)",
            "returnrt_restir_gi_record_spatial_debug(restir_gi_invalid_reservoir(),RESTIR_GI_SPATIAL_REUSE_INVALID_SURFACE)",
            "uintspatial_reuse_reason=RESTIR_GI_SPATIAL_REUSE_CENTER_RESERVOIR_INVALID",
            "spatial_reuse_reason=RESTIR_GI_SPATIAL_REUSE_NO_COMPATIBLE_NEIGHBOR",
            "spatial_reuse_reason=RESTIR_GI_SPATIAL_REUSE_ACCEPTED_CENTER",
            "spatial_reuse_reason=RESTIR_GI_SPATIAL_REUSE_ACCEPTED_NEIGHBOR",
            "spatial_reuse_reason=RESTIR_GI_SPATIAL_REUSE_VISIBILITY_REJECTED",
            "output_reservoirs[index]=rt_restir_gi_record_spatial_debug(reservoir,spatial_reuse_reason)",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI spatial debug must classify reuse path with {token}"
            );
        }

        assert!(
            direct.contains("rt_direct_visualize_gi_spatial_reason"),
            "RT direct lighting debug must visualize GI spatial reuse reasons"
        );
    }

    #[test]
    fn rt_restir_gi_spatial_shader_validates_neighbor_visibility() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi_spatial.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "#include\"rt_surface_common.slang\"",
            "RaytracingAccelerationStructurescene_tlas",
            "boolrt_restir_gi_spatial_sample_visible(RtSurfacePixelsurface,RestirGiReservoirreservoir)",
            "float3sample_vector=reservoir.sample_position_depth.xyz-surface.position_depth.xyz",
            "sample_direction=sample_vector*rsqrt(distance2)",
            "if(restir_gi_is_environment_sample(reservoir)){sample_direction=normalize(reservoir.sample_position_depth.xyz);sample_distance=RESTIR_GI_ENVIRONMENT_SAMPLE_DEPTH;}",
            "ray.Direction=sample_direction",
            "ray.TMax=restir_gi_is_environment_sample(reservoir)?sample_distance:max(sample_distance-0.03,0.001)",
            "RtSurfacePayloadpayload=make_rt_surface_payload(ray.Direction)",
            "TraceRay(scene_tlas",
            "returnpayload.hit_kind!=RT_SURFACE_HIT_KIND_VOXEL",
            "if(!rt_restir_gi_spatial_sample_visible(surface,neighbor)){spatial_reuse_reason=RESTIR_GI_SPATIAL_REUSE_VISIBILITY_REJECTED;continue;}",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI spatial reuse must validate neighbor visibility with {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_initial_stage_marks_spatial_debug_passthrough_when_spatial_stage_is_inactive() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "boolrt_restir_gi_spatial_debug_passthrough_enabled()",
            "returnrestir_gi.debug_view==RESTIR_GI_DEBUG_VIEW_SPATIAL_VALID",
            "RestirGiReservoirrt_restir_gi_record_spatial_debug_passthrough(RestirGiReservoirreservoir)",
            "reservoir.sample_radiance_pdf.w=float(RESTIR_GI_SPATIAL_REUSE_DISABLED_OR_PASSTHROUGH)",
            "reservoir=rt_restir_gi_record_spatial_debug_passthrough(reservoir)",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI initial/temporal stage must tag inactive spatial debug output with {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_target_pdf_and_resolve_use_cosine_sample_contribution() {
        let common = crate::render::source_checks::read_source(
            "assets/shaders/shared/restir_gi_common.slang",
        );
        let initial = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let spatial = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi_spatial.rgen.slang",
        );
        let direct = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_direct_lighting.rgen.slang",
        );
        let common_compact = crate::render::source_checks::compact(&common);
        let initial_compact = crate::render::source_checks::compact(&initial);
        let spatial_compact = crate::render::source_checks::compact(&spatial);
        let direct_compact = crate::render::source_checks::compact(&direct);

        for token in [
            "floatrestir_gi_cosine_sample_visibility(float3surface_position,float3surface_normal,RestirGiReservoirreservoir)",
            "float3sample_vector=reservoir.sample_position_depth.xyz-surface_position",
            "float3sample_direction=sample_vector*rsqrt(distance2)",
            "floatreceiver_term=dot(surface_normal,sample_direction)",
            "floatsample_term=dot(sample_normal,-sample_direction)",
            "returnreceiver_term>0.0&&sample_term>0.0?1.0:0.0",
        ] {
            assert!(
                common_compact.contains(token),
                "RT ReSTIR-GI common visibility helper missing {token}"
            );
        }

        for (name, source) in [
            ("initial/temporal", &initial_compact),
            ("spatial", &spatial_compact),
        ] {
            assert!(
                source.contains(
                    "float3contribution=restir_gi_cosine_sample_contribution(surface.albedo_material.rgb,surface.position_depth.xyz,surface_normal,reservoir)"
                ),
                "{name} target pdf must evaluate cosine-sampled contribution"
            );
        }

        for token in [
            "floattarget_pdf=rt_restir_gi_target_pdf(surface,candidate)",
            "if(target_pdf<=0.0){continue;}",
            "candidate.target_pdf=rt_restir_gi_target_pdf(surface,candidate)",
        ] {
            assert!(
                initial_compact.contains(token),
                "RT ReSTIR-GI initial sampling must reject zero-contribution candidates with {token}"
            );
        }
        assert!(
            !initial_compact.contains("max(rt_restir_gi_target_pdf(surface,candidate),1.0e-4)"),
            "RT ReSTIR-GI initial sampling must not promote zero-geometry candidates to epsilon"
        );

        for token in [
            "float3contribution=restir_gi_cosine_sample_contribution(surface.albedo_material.rgb,surface.position_depth.xyz,normal,reservoir)",
            "if(restir_gi_luma(contribution)<=1.0e-5){returnfloat3(0.0);}",
            "contribution*selected_weight",
        ] {
            assert!(
                direct_compact.contains(token),
                "RT direct lighting GI resolve must use cosine-sampled contribution with {token}"
            );
        }
        assert!(
            !direct_compact.contains(
                "*RT_LIGHTING_INV_PI*reservoir.sample_radiance_pdf.rgb*geometry*selected_weight",
            ),
            "RT direct lighting GI resolve must not add an extra Lambertian cosine/pi term"
        );
    }

    #[test]
    fn rt_restir_gi_cosine_sampled_bounce_cancels_lambertian_pdf() {
        let common = crate::render::source_checks::read_source(
            "assets/shaders/shared/restir_gi_common.slang",
        );
        let initial = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let spatial = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi_spatial.rgen.slang",
        );
        let direct = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_direct_lighting.rgen.slang",
        );
        let common_compact = crate::render::source_checks::compact(&common);
        let initial_compact = crate::render::source_checks::compact(&initial);
        let spatial_compact = crate::render::source_checks::compact(&spatial);
        let direct_compact = crate::render::source_checks::compact(&direct);

        for token in [
            "floatrestir_gi_cosine_sample_visibility(",
            "float3restir_gi_cosine_sample_contribution(",
            "returnsurface_albedo*reservoir.sample_radiance_pdf.rgb*visibility",
        ] {
            assert!(
                common_compact.contains(token),
                "RT GI must resolve cosine-sampled Lambertian bounces with the sampling PDF cancelled; missing {token}"
            );
        }

        for (name, source) in [
            ("initial/temporal target", &initial_compact),
            ("spatial target", &spatial_compact),
        ] {
            assert!(
                source.contains(
                    "float3contribution=restir_gi_cosine_sample_contribution(surface.albedo_material.rgb,surface.position_depth.xyz,surface_normal,reservoir)"
                ),
                "{name} must base ReSTIR weights on the same cosine-sampled contribution that final resolve shades"
            );
            assert!(
                source.contains("returnrestir_gi_luma(contribution)"),
                "{name} must not use an extra cosine/pi geometry factor as the reservoir target"
            );
        }

        assert!(
            direct_compact.contains(
                "float3contribution=restir_gi_cosine_sample_contribution(surface.albedo_material.rgb,surface.position_depth.xyz,normal,reservoir)"
            ),
            "RT final GI resolve must shade the cosine-sampled bounce contribution directly"
        );
        assert!(
            direct_compact.contains("returnrestir_gi_clamp_radiance_luma(contribution*selected_weight,RESTIR_GI_MAX_RESOLVED_RADIANCE_LUMA)"),
            "RT final GI resolve must apply reservoir weight after cosine/PDF cancellation"
        );
        assert!(
            !direct_compact.contains(
                "*RT_LIGHTING_INV_PI*reservoir.sample_radiance_pdf.rgb*geometry*selected_weight",
            ),
            "RT final GI resolve must not multiply cosine-sampled bounces by an extra Lambertian cosine/pi term"
        );
    }

    #[test]
    fn rt_restir_gi_initial_candidate_evaluates_second_bounce_lighting() {
        let common = crate::render::source_checks::read_source(
            "assets/shaders/shared/restir_gi_common.slang",
        );
        let shader = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let pass = crate::render::source_checks::read_source("src/render/passes/rt_restir_gi.rs");
        let pipeline = crate::render::source_checks::read_source("src/render/rt_pipeline.rs");
        let compact_shader = crate::render::source_checks::compact(&shader);
        let compact_common = crate::render::source_checks::compact(&common);
        let compact_pass = crate::render::source_checks::compact(&pass);
        let compact_pipeline = crate::render::source_checks::compact(&pipeline);

        for token in [
            "float4sky_color_sun_angular_radius",
            "float4sun_direction_pad",
            "float4sun_intensity_pad",
            "float4ground_color_pad",
        ] {
            assert!(
                compact_common.contains(token),
                "RT ReSTIR-GI uniforms must carry scene lighting for second-bounce evaluation; missing {token}"
            );
        }

        for token in [
            "float3rt_restir_gi_background_color_for_dir(float3direction)",
            "float3rt_restir_gi_sky_visibility_sample(RtSurfacePixelsurface,float3sky_dir)",
            "float3rt_restir_gi_analytic_sun_direct(RtSurfacePixelsurface,inoutuintrng_state)",
            "float3rt_restir_gi_estimate_incoming_radiance(RtSurfacePixelsurface,inoutuintrng_state)",
            "RayDescsky_ray",
            "TraceRay(scene_tlas,0u,0xffu,0u,0u,0u,sky_ray,sky_payload)",
            "RayDescsun_ray",
            "TraceRay(scene_tlas,0u,0xffu,0u,0u,0u,sun_ray,sun_payload)",
            "float3sample_radiance=rt_restir_gi_estimate_incoming_radiance(indirect_surface,rng_state)",
        ] {
            assert!(
                compact_shader.contains(token),
                "RT ReSTIR-GI initial candidate must evaluate visible second-bounce lighting with {token}"
            );
        }

        assert!(
            compact_shader.contains("if(restir_gi_luma(sample_radiance)<=0.0){returncandidate;}"),
            "RT ReSTIR-GI must reject zero-radiance second-bounce candidates instead of reserving dark samples"
        );
        assert!(
            !compact_shader.contains("max(indirect_surface.albedo_material.rgb,float3(0.0))*0.05"),
            "RT ReSTIR-GI initial candidate must not fake incoming radiance as albedo * 0.05"
        );

        for token in [
            "sky_color_sun_angular_radius",
            "sun_direction_pad",
            "sun_intensity_pad",
            "ground_color_pad",
        ] {
            assert!(
                compact_pass.contains(token),
                "RT ReSTIR-GI pass must upload scene lighting uniform field {token}"
            );
        }

        assert!(
            compact_pipeline.contains("sun_direction:inputs.sun_direction"),
            "RT pipeline must pass sun direction into RT ReSTIR-GI uniforms"
        );
        assert!(
            compact_pipeline.contains("sun_intensity:inputs.sun_intensity"),
            "RT pipeline must pass sun intensity into RT ReSTIR-GI uniforms"
        );
    }

    #[test]
    fn rt_restir_gi_sun_intensity_is_total_irradiance_not_disk_radiance() {
        let shader = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact_shader = crate::render::source_checks::compact(&shader);

        for token in [
            "float3sun_irradiance=restir_gi.sun_intensity_pad.rgb*ground_ndotl;",
            "returnrestir_gi.ground_color_pad.rgb*(1.0+RT_RESTIR_GI_INV_PI*sun_irradiance);",
            "float3direct_brdf=restir_di_direct_brdf(normal,surface.albedo_material.rgb,surface.normal_roughness.w,surface_view_dir,sun_dir);",
            "returndirect_brdf*restir_gi.sun_intensity_pad.rgb*sun_term;",
        ] {
            assert!(
                compact_shader.contains(token),
                "RT ReSTIR-GI second-bounce lighting must keep sun brightness independent of angular radius; missing {token}"
            );
        }

        for forbidden in [
            "floatrt_restir_gi_sun_disk_solid_angle()",
            "restir_gi.sun_intensity_pad.rgb*ground_ndotl*rt_restir_gi_sun_disk_solid_angle()",
            "*restir_gi.sun_intensity_pad.rgb*sun_term*solid_angle",
            "returnsurface.albedo_material.rgb*RT_RESTIR_GI_INV_PI*restir_gi.sun_intensity_pad.rgb*sun_term;",
        ] {
            assert!(
                !compact_shader.contains(forbidden),
                "RT ReSTIR-GI must not scale total sun irradiance by disk solid angle; found {forbidden}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_second_bounce_sun_samples_disk_with_shared_direct_brdf() {
        let common = crate::render::source_checks::read_source(
            "assets/shaders/shared/restir_di_common.slang",
        );
        let raygen = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact_common = crate::render::source_checks::compact(&common);
        let compact_raygen = crate::render::source_checks::compact(&raygen);

        assert!(
            compact_raygen.contains("#include\"restir_di_common.slang\""),
            "RT ReSTIR-GI second-bounce sun must share the direct BRDF helper"
        );
        assert!(
            compact_common.contains(
                "float3restir_di_direct_brdf(float3surface_normal,float3albedo,floatroughness,float3view_dir,float3light_dir)"
            ),
            "RT ReSTIR-GI second-bounce sun must use the same roughness/view-aware BRDF as RT direct lighting"
        );

        for token in [
            "float3rt_restir_gi_sample_sun_direction(inoutuintrng_state)",
            "floatsun_radius=max(restir_gi.sky_color_sun_angular_radius.w,0.0);",
            "floatcos_min=cos(sun_radius);",
            "floatcos_theta=lerp(cos_min,1.0,gi_rand01(rng_state));",
            "floatphi=6.28318530718*gi_rand01(rng_state);",
            "float3sun_forward=normalize(restir_gi.sun_direction_pad.xyz);",
            "returnnormalize(sun_right*(cos(phi)*sin_theta)+sun_up*(sin(phi)*sin_theta)+sun_forward*cos_theta);",
            "float3rt_restir_gi_analytic_sun_direct(RtSurfacePixelsurface,inoutuintrng_state)",
            "float3sun_dir=rt_restir_gi_sample_sun_direction(rng_state);",
            "float3surface_view_dir=normalize(-surface.view_direction_background.xyz);",
            "float3direct_brdf=restir_di_direct_brdf(normal,surface.albedo_material.rgb,surface.normal_roughness.w,surface_view_dir,sun_dir);",
            "returndirect_brdf*restir_gi.sun_intensity_pad.rgb*sun_term;",
            "float3rt_restir_gi_estimate_incoming_radiance(RtSurfacePixelsurface,inoutuintrng_state)",
            "returnsky+rt_restir_gi_analytic_sun_direct(surface,rng_state);",
            "RestirGiReservoirrt_restir_gi_make_initial_candidate(RtSurfacePixelsurface,RtSurfacePixelindirect_surface,float3indirect_direction,inoutuintrng_state)",
            "float3sample_radiance=rt_restir_gi_estimate_incoming_radiance(indirect_surface,rng_state)",
            "RestirGiReservoircandidate=rt_restir_gi_make_initial_candidate(surface,indirect_surface,indirect_direction,rng_state)",
        ] {
            assert!(
                compact_raygen.contains(token),
                "RT ReSTIR-GI second-bounce sun must sample finite disk shadows/highlights without changing total irradiance; missing {token}"
            );
        }

        for forbidden in [
            "float3sun_dir=normalize(sun_vector);",
            "float3rt_restir_gi_analytic_sun_direct(RtSurfacePixelsurface)",
            "float3rt_restir_gi_estimate_incoming_radiance(RtSurfacePixelsurface)",
            "returnsurface.albedo_material.rgb*RT_RESTIR_GI_INV_PI*restir_gi.sun_intensity_pad.rgb*sun_term;",
            "rt_restir_gi_make_initial_candidate(surface,indirect_surface,indirect_direction)",
        ] {
            assert!(
                !compact_raygen.contains(forbidden),
                "RT ReSTIR-GI second-bounce sun must not keep the old fixed-direction diffuse-only path; found {forbidden}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_initial_candidate_uses_emissive_hit_radiance() {
        let rt_history = crate::render::source_checks::read_source(
            "assets/shaders/shared/rt_history_common.slang",
        );
        let rt_surface_common = crate::render::source_checks::read_source(
            "assets/shaders/shared/rt_surface_common.slang",
        );
        let rt_surface_closest_hit = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_surface.rchit.slang",
        );
        let gi_closest_hit = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rchit.slang",
        );
        let gi_raygen = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact_surface_common = crate::render::source_checks::compact(&rt_surface_common);
        let compact_gi_raygen = crate::render::source_checks::compact(&gi_raygen);

        assert!(
            rt_history.contains("float4 emissive_radiance"),
            "RT surface history ABI must carry emissive radiance for GI"
        );

        for token in [
            "float3emissive_radiance",
            "payload.emissive_radiance=float3(0.0)",
            "pixel.emissive_radiance=float4(payload.emissive_radiance,0.0)",
        ] {
            assert!(
                compact_surface_common.contains(token),
                "RT surface payload conversion must preserve emissive radiance with {token}"
            );
        }

        for (name, source) in [
            ("RT surface closest hit", rt_surface_closest_hit),
            ("RT GI closest hit", gi_closest_hit),
        ] {
            assert!(
                source.contains("payload.emissive_radiance = material_emissive(cell) * 3.0;"),
                "{name} must write voxel emissive radiance into the RT surface payload"
            );
        }

        for token in [
            "float3emissive=surface.emissive_radiance.rgb",
            "if(restir_gi_luma(emissive)>0.0){returnemissive;}",
        ] {
            assert!(
                compact_gi_raygen.contains(token),
                "RT ReSTIR-GI incoming-radiance estimate must return emissive hits before sky/sun with {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_initial_candidate_accepts_environment_miss_radiance() {
        let common = crate::render::source_checks::read_source(
            "assets/shaders/shared/restir_gi_common.slang",
        );
        let raygen = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let direct = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_direct_lighting.rgen.slang",
        );
        let spatial = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi_spatial.rgen.slang",
        );
        let compact_common = crate::render::source_checks::compact(&common);
        let compact_raygen = crate::render::source_checks::compact(&raygen);
        let compact_direct = crate::render::source_checks::compact(&direct);
        let compact_spatial = crate::render::source_checks::compact(&spatial);

        for token in [
            "staticconstfloatRESTIR_GI_ENVIRONMENT_SAMPLE_DEPTH=1.0e20",
            "boolrestir_gi_is_environment_sample(RestirGiReservoirreservoir)",
            "returnreservoir.sample_position_depth.w>=RESTIR_GI_ENVIRONMENT_SAMPLE_DEPTH*0.5",
            "floatrestir_gi_environment_geometry_term(float3surface_normal,RestirGiReservoirreservoir)",
            "returnmax(dot(surface_normal,normalize(reservoir.sample_position_depth.xyz)),0.0)",
        ] {
            assert!(
                compact_common.contains(token),
                "RT ReSTIR-GI common helpers must model environment miss candidates with {token}"
            );
        }

        for token in [
            "RestirGiReservoirrt_restir_gi_make_environment_candidate(RtSurfacePixelsurface,float3indirect_direction)",
            "float3environment_radiance=rt_restir_gi_background_color_for_dir(indirect_direction)",
            "candidate.sample_position_depth=float4(indirect_direction,RESTIR_GI_ENVIRONMENT_SAMPLE_DEPTH)",
            "candidate.sample_normal_roughness=float4(-indirect_direction,1.0)",
            "if(indirect_surface.hit_kind==RT_SURFACE_HIT_KIND_MISS){returnrt_restir_gi_make_environment_candidate(surface,indirect_direction);}",
            "RestirGiReservoircandidate=rt_restir_gi_make_initial_candidate(surface,indirect_surface,indirect_direction,rng_state)",
        ] {
            assert!(
                compact_raygen.contains(token),
                "RT ReSTIR-GI initial candidates must preserve sky/environment miss radiance with {token}"
            );
        }

        assert!(
            !compact_raygen.contains(
                "if(!rt_restir_gi_surface_valid(surface)||!rt_restir_gi_surface_valid(indirect_surface)){returncandidate;}"
            ),
            "RT ReSTIR-GI must not reject environment miss candidates via the old surface-validity gate"
        );

        let token = "float3contribution=restir_gi_cosine_sample_contribution(surface.albedo_material.rgb,surface.position_depth.xyz,normal,reservoir)";
        assert!(
            compact_direct.contains(token),
            "RT direct GI resolve must handle environment miss candidates with {token}"
        );

        for token in [
            "float3contribution=restir_gi_cosine_sample_contribution(surface.albedo_material.rgb,surface.position_depth.xyz,surface_normal,reservoir)",
            "if(restir_gi_is_environment_sample(reservoir)){sample_direction=normalize(reservoir.sample_position_depth.xyz);sample_distance=RESTIR_GI_ENVIRONMENT_SAMPLE_DEPTH;}",
        ] {
            assert!(
                compact_spatial.contains(token),
                "RT ReSTIR-GI spatial reuse must treat environment samples as directions with {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_initial_candidates_clamp_sample_radiance_chroma_preserving() {
        let common = crate::render::source_checks::read_source(
            "assets/shaders/shared/restir_gi_common.slang",
        );
        let raygen = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact_common = crate::render::source_checks::compact(&common);
        let compact_raygen = crate::render::source_checks::compact(&raygen);

        for token in [
            "staticconstfloatRESTIR_GI_MAX_SAMPLE_RADIANCE_LUMA=64.0",
            "float3restir_gi_clamp_radiance_luma(float3radiance,floatmax_luma)",
            "floatluma=restir_gi_luma(radiance)",
            "returnradiance*(max_luma/luma)",
            "returnmax(radiance,float3(0.0))",
        ] {
            assert!(
                compact_common.contains(token),
                "RT ReSTIR-GI must clamp radiance by luminance while preserving chroma; missing {token}"
            );
        }

        for token in [
            "sample_radiance=restir_gi_clamp_radiance_luma(sample_radiance,RESTIR_GI_MAX_SAMPLE_RADIANCE_LUMA)",
            "candidate.sample_radiance_pdf.rgb=restir_gi_clamp_radiance_luma(candidate.sample_radiance_pdf.rgb,RESTIR_GI_MAX_SAMPLE_RADIANCE_LUMA)",
        ] {
            assert!(
                compact_raygen.contains(token),
                "RT ReSTIR-GI initial candidates must clamp extreme second-bounce radiance; missing {token}"
            );
        }

        assert!(
            compact_raygen.contains("if(restir_gi_luma(emissive)>0.0){returnemissive;}"),
            "RT ReSTIR-GI must still accept emissive hits before sky/sun estimation"
        );
    }

    #[test]
    fn rt_restir_gi_initial_candidates_do_not_apply_artificial_radiance_jitter() {
        let raygen = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&raygen);

        assert!(
            compact.contains(
                "float3sample_radiance=rt_restir_gi_estimate_incoming_radiance(indirect_surface,rng_state)"
            ),
            "RT ReSTIR-GI initial candidates must start from traced incoming radiance"
        );
        assert!(
            compact.contains("candidate.sample_radiance_pdf=float4(sample_radiance,1.0)"),
            "RT ReSTIR-GI candidates must store the traced/clamped radiance without later random energy modulation"
        );
        for forbidden in [
            "floatcandidate_jitter=0.75+gi_rand01(rng_state)*0.5",
            "candidate.sample_radiance_pdf.rgb*=candidate_jitter",
        ] {
            assert!(
                !compact.contains(forbidden),
                "RT ReSTIR-GI initial candidates must not inject artificial temporal noise into radiance; found {forbidden}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_initial_reservoir_traces_and_resamples_multiple_candidates() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "uintcandidate_count=clamp(restir_gi.initial_candidate_count,1u,16u)",
            "for(uintcandidate_index=0u;candidate_index<candidate_count;candidate_index++)",
            "float3indirect_direction=rt_restir_gi_sample_indirect_direction(normal,rng_state)",
            "RtSurfacePixelindirect_surface=rt_restir_gi_trace_indirect_surface(surface,indirect_direction)",
            "RestirGiReservoircandidate=rt_restir_gi_make_initial_candidate(surface,indirect_surface,indirect_direction,rng_state)",
            "floatcandidate_weight=target_pdf",
            "floatnext_weight_sum=weight_sum+candidate_weight",
            "if(!restir_gi_candidate_finite(next_weight_sum)){continue;}",
            "if(accepted_count==0u||gi_rand01(rng_state)*next_weight_sum<=candidate_weight)",
            "accepted_count+=1u",
            "restir_gi_finalize_reservoir(reservoir,selected_target_pdf,weight_sum,accepted_count)",
            "rt_restir_gi_generate_initial_reservoir(surface,index,launch_id.xy)",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI initial reservoir must trace and reservoir-sample candidates with {token}"
            );
        }

        assert!(
            !compact.contains("rt_restir_gi_generate_initial_reservoir(surface,indirect_surface,index,launch_id.xy)"),
            "RT ReSTIR-GI initial reservoir must not reuse one traced indirect surface for all candidates"
        );
    }

    #[test]
    fn rt_restir_gi_pass_uses_configured_initial_candidate_count() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_restir_gi.rs");
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "initial_candidate_count:rt_settings.restir_gi_initial_candidate_count.clamp(1,16)",
            "rt_settings.restir_gi_initial_candidate_count",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI pass must route configured initial candidate count with {token}"
            );
        }
    }

    #[test]
    fn rt_restir_gi_pass_spatial_stage_uses_temporal_intermediate() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_restir_gi.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT ReSTIR-GI implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "spatial_pipeline",
            "spatial_shader_binding_table",
            "spatial_descriptor_set_layout",
            "spatial_descriptor_pool",
            "spatial_descriptor_sets",
            "temporal_reservoirs",
            "spatial_raygen_spirv",
            "create_reservoir_buffer(device,allocator,reservoir_count,\"rt_restir_gi_temporal\")",
            "letspatial_active=spatial_enabled&&history_initialized",
            "rt_restir_gi_spatial",
            "builder.read_as(temporal_reservoir_output,AccessKind::RayTracingShaderRead);",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI spatial pass missing {token}"
            );
        }

        assert!(
            !compact
                .contains("builder.read_as(temporal_reservoir,AccessKind::RayTracingShaderRead);"),
            "RT ReSTIR-GI spatial graph must read the written temporal resource version"
        );
    }

    #[test]
    fn rt_restir_gi_spatial_pass_uses_surface_pipeline_for_visibility_rays() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_restir_gi.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT ReSTIR-GI implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "fnspatial_descriptor_binding_specs()->[DescriptorBindingSpec;16]",
            "DescriptorBindingSpec::ray_tracing(7,vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)",
            "DescriptorBindingSpec::ray_tracing(8,vk::DescriptorType::STORAGE_BUFFER)",
            "DescriptorBindingSpec::ray_tracing(17,vk::DescriptorType::STORAGE_BUFFER)",
            "descriptor_count:(12*frame_count)asu32",
            "create_spatial_raygen_pipeline(",
            "write_tlas_descriptor(device,spatial_descriptor_set,tlas)",
            "write_aabb_descriptor(device,spatial_descriptor_set,aabb_buffer)",
            "write_ucvh_descriptors(device,spatial_descriptor_set,ucvh_gpu)",
            "write_traversal_stats_descriptor(device,&pass.spatial_descriptor_sets,&pass.traversal_stats_buffer,)",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI spatial visibility pass must own full ray tracing resources with {token}"
            );
        }

        assert!(
            !compact.contains("create_raygen_only_pipeline("),
            "RT ReSTIR-GI spatial pass must not use a raygen-only pipeline once it traces visibility rays"
        );
    }

    #[test]
    fn rt_restir_gi_spatial_shader_reuses_compatible_neighbor_temporal_reservoirs() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_gi_spatial.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "#include\"restir_gi_common.slang\"",
            "#include\"rt_history_common.slang\"",
            "StructuredBuffer<RestirGiReservoir>temporal_reservoirs",
            "RWStructuredBuffer<RestirGiReservoir>output_reservoirs",
            "StructuredBuffer<RtSurfacePixel>surface_pixels",
            "ConstantBuffer<RtHistoryUniforms>rt_history",
            "staticconstint2rt_restir_gi_spatial_offsets[8]",
            "rt_restir_gi_spatial_surfaces_compatible",
            "restir_gi.spatial_enabled==0u",
            "restir_gi.spatial_sample_count==0u",
            "restir_gi_is_valid_reservoir(neighbor)",
            "restir_gi_reservoir_stream_weight",
            "restir_gi_finalize_reservoir",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-GI spatial shader missing {token}"
            );
        }
    }
}
