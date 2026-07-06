use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::graph::RenderGraph;
use crate::render::pipeline::{RayTracingPipeline, ShaderBindingTable, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::restir_di::{GpuDirectLight, GpuRestirDiReservoir, GpuRestirDiUniforms};
use crate::render::rt_history::{GpuRtHistoryUniforms, GpuRtSurfacePixel};
use crate::render::rt_settings::{RtDebugView, RtSettings};

pub struct RtRestirDiPass {
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
    direct_lights: GpuBuffer,
    temporal_reservoirs: GpuBuffer,
    selected_reservoirs: Vec<GpuBuffer>,
    surface_history_buffers: [GpuBuffer; 2],
    width: u32,
    height: u32,
    reservoir_count: u32,
    light_count: u32,
}

pub struct RtRestirDiCreateInfo<'a> {
    pub ray_tracing_pipeline_loader: &'a ash::khr::ray_tracing_pipeline::Device,
    pub rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    pub width: u32,
    pub height: u32,
    pub frame_count: usize,
    pub raygen_spirv: &'a [u8],
    pub spatial_raygen_spirv: &'a [u8],
    pub direct_lights: &'a [GpuDirectLight],
}

#[derive(Clone, Copy)]
pub struct RtRestirDiGraphOutputs {
    pub reservoirs: ResourceHandle,
}

impl RtRestirDiPass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 8] {
        [
            DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(3, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(4, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(5, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(6, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(7, vk::DescriptorType::UNIFORM_BUFFER),
        ]
    }

    pub(crate) fn spatial_descriptor_binding_specs() -> [DescriptorBindingSpec; 5] {
        [
            DescriptorBindingSpec::ray_tracing(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::ray_tracing(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(2, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(3, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::ray_tracing(4, vk::DescriptorType::UNIFORM_BUFFER),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: RtRestirDiCreateInfo<'_>,
    ) -> Result<Self> {
        let frame_count = info.frame_count;
        let (descriptor_set_layout, descriptor_pool, descriptor_sets) = create_descriptor_sets(
            device,
            &Self::descriptor_binding_specs(),
            frame_count,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: (2 * frame_count) as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: (6 * frame_count) as u32,
                },
            ],
        )?;
        let (spatial_descriptor_set_layout, spatial_descriptor_pool, spatial_descriptor_sets) =
            match create_descriptor_sets(
                device,
                &Self::spatial_descriptor_binding_specs(),
                frame_count,
                &[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: (2 * frame_count) as u32,
                    },
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        descriptor_count: (3 * frame_count) as u32,
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
                spatial_descriptor_pool.destroy(device);
                unsafe {
                    device.destroy_descriptor_set_layout(spatial_descriptor_set_layout, None)
                };
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
        let direct_lights = match create_direct_light_buffer(device, allocator, info.direct_lights)
        {
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
        let reservoir_count = info.width.saturating_mul(info.height);
        let temporal_reservoirs = match create_reservoir_buffer(
            device,
            allocator,
            reservoir_count,
            "rt_restir_di_temporal",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                direct_lights.destroy(device, allocator);
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
        let selected_reservoirs = match create_selected_reservoir_buffers(
            device,
            allocator,
            frame_count,
            reservoir_count,
            "rt_restir_di_selected",
        ) {
            Ok(buffers) => buffers,
            Err(error) => {
                temporal_reservoirs.destroy(device, allocator);
                direct_lights.destroy(device, allocator);
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
                    destroy_buffers(selected_reservoirs, device, allocator);
                    temporal_reservoirs.destroy(device, allocator);
                    direct_lights.destroy(device, allocator);
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
            info.raygen_spirv,
            descriptor_set_layout,
            "rt_restir_di",
        ) {
            Ok(resources) => resources,
            Err(error) => {
                destroy_buffers(Vec::from(surface_history_buffers), device, allocator);
                destroy_buffers(selected_reservoirs, device, allocator);
                temporal_reservoirs.destroy(device, allocator);
                direct_lights.destroy(device, allocator);
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
        let (spatial_pipeline, spatial_shader_binding_table) = match create_raygen_pipeline(
            device,
            allocator,
            info.ray_tracing_pipeline_loader,
            info.rt_pipeline_properties,
            info.spatial_raygen_spirv,
            spatial_descriptor_set_layout,
            "rt_restir_di_spatial",
        ) {
            Ok(resources) => resources,
            Err(error) => {
                shader_binding_table.destroy(device, allocator);
                pipeline.destroy(device);
                destroy_buffers(Vec::from(surface_history_buffers), device, allocator);
                destroy_buffers(selected_reservoirs, device, allocator);
                temporal_reservoirs.destroy(device, allocator);
                direct_lights.destroy(device, allocator);
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
            direct_lights,
            temporal_reservoirs,
            selected_reservoirs,
            surface_history_buffers,
            width: info.width,
            height: info.height,
            reservoir_count,
            light_count: info.direct_lights.len() as u32,
        };
        pass.write_static_descriptors(device);
        Ok(pass)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn output_reservoir_buffer(&self, frame_slot: usize) -> &GpuBuffer {
        self.selected_current_buffer(frame_slot)
    }

    pub fn update_uniforms(
        &self,
        frame_slot: usize,
        rt_settings: RtSettings,
        frame_index: u64,
        history_initialized: bool,
    ) {
        let temporal_enabled = rt_settings.restir_di_enabled && history_initialized;
        let spatial_enabled = temporal_enabled
            && rt_settings.restir_di_spatial_enabled
            && rt_settings.restir_di_spatial_sample_count > 0;
        let uniforms = GpuRestirDiUniforms {
            enabled: rt_settings.restir_di_enabled as u32,
            temporal_enabled: temporal_enabled as u32,
            spatial_enabled: spatial_enabled as u32,
            debug_view: rt_restir_debug_view(rt_settings.debug_view),
            initial_candidate_count: 1,
            spatial_sample_count: if spatial_enabled {
                rt_settings.restir_di_spatial_sample_count.min(8)
            } else {
                0
            },
            history_length: rt_settings.history_length.max(1),
            frame_index: frame_index as u32,
            reservoir_count: self.reservoir_count,
            light_count: self.light_count,
            width: self.width,
            height: self.height,
        };
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
            self.selected_history_buffer(frame_slot),
            if spatial_enabled {
                &self.temporal_reservoirs
            } else {
                self.selected_current_buffer(frame_slot)
            },
            self.previous_surface_history_buffer(frame_index),
            self.current_surface_history_buffer(frame_index),
        );
        if let Some(&spatial_descriptor_set) = self.spatial_descriptor_sets.get(frame_slot) {
            write_spatial_frame_descriptors(
                device,
                spatial_descriptor_set,
                &self.temporal_reservoirs,
                self.selected_current_buffer(frame_slot),
                surface_buffer,
            );
        }
    }

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        frame_slot: usize,
        frame_index: u64,
        surface: ResourceHandle,
        history_initialized: bool,
        spatial_enabled: bool,
    ) -> RtRestirDiGraphOutputs {
        let spatial_active = spatial_enabled && history_initialized;
        let uniform_buffer = &self.uniform_buffers[frame_slot];
        let uniform = graph.import_buffer_with_access(
            uniform_buffer.handle,
            uniform_buffer.size,
            uniform_buffer.usage,
            AccessKind::RayTracingShaderRead,
        );
        let direct_lights = graph.import_buffer_with_access(
            self.direct_lights.handle,
            self.direct_lights.size,
            self.direct_lights.usage,
            AccessKind::RayTracingShaderRead,
        );
        let history_uniform_buffer = &self.history_uniform_buffers[frame_slot];
        let history_uniform = graph.import_buffer_with_access(
            history_uniform_buffer.handle,
            history_uniform_buffer.size,
            history_uniform_buffer.usage,
            AccessKind::RayTracingShaderRead,
        );
        let selected_current_buffer = self.selected_current_buffer(frame_slot);
        let selected_history_buffer = self.selected_history_buffer(frame_slot);
        let temporal_reservoir_buffer = &self.temporal_reservoirs;
        let current_surface_history_buffer = self.current_surface_history_buffer(frame_index);
        let previous_surface_history_buffer = self.previous_surface_history_buffer(frame_index);
        let selected_current = graph.import_buffer_with_access(
            selected_current_buffer.handle,
            selected_current_buffer.size,
            selected_current_buffer.usage,
            AccessKind::Undefined,
        );
        let selected_history = graph.import_buffer_with_access(
            selected_history_buffer.handle,
            selected_history_buffer.size,
            selected_history_buffer.usage,
            if history_initialized {
                AccessKind::RayTracingShaderWrite
            } else {
                AccessKind::Undefined
            },
        );
        let temporal_reservoir = graph.import_buffer_with_access(
            temporal_reservoir_buffer.handle,
            temporal_reservoir_buffer.size,
            temporal_reservoir_buffer.usage,
            AccessKind::Undefined,
        );
        let current_surface_history = graph.import_buffer_with_access(
            current_surface_history_buffer.handle,
            current_surface_history_buffer.size,
            current_surface_history_buffer.usage,
            if history_initialized {
                AccessKind::RayTracingShaderWrite
            } else {
                AccessKind::Undefined
            },
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

        let writes = graph.add_pass("rt_restir_di_initial", QueueType::RayTracing, |builder| {
            builder.read_as(uniform, AccessKind::RayTracingShaderRead);
            builder.read_as(history_uniform, AccessKind::RayTracingShaderRead);
            builder.read_as(direct_lights, AccessKind::RayTracingShaderRead);
            builder.read_as(surface, AccessKind::RayTracingShaderRead);
            if history_initialized {
                builder.read_as(selected_history, AccessKind::RayTracingShaderRead);
                builder.read_as(previous_surface_history, AccessKind::RayTracingShaderRead);
            }
            let initial_output = if spatial_active {
                temporal_reservoir
            } else {
                selected_current
            };
            builder.write_as(initial_output, AccessKind::RayTracingShaderWrite);
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
        let temporal_reservoir_output = writes[0];
        let final_reservoirs = if spatial_active {
            let ray_tracing_pipeline_loader = self.ray_tracing_pipeline_loader.clone();
            let spatial_writes =
                graph.add_pass("rt_restir_di_spatial", QueueType::RayTracing, |builder| {
                    builder.read_as(uniform, AccessKind::RayTracingShaderRead);
                    builder.read_as(history_uniform, AccessKind::RayTracingShaderRead);
                    builder.read_as(temporal_reservoir_output, AccessKind::RayTracingShaderRead);
                    builder.read_as(surface, AccessKind::RayTracingShaderRead);
                    builder.write_as(selected_current, AccessKind::RayTracingShaderWrite);
                    Box::new(move |ctx| unsafe {
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
                    })
                });
            spatial_writes[0]
        } else {
            writes[0]
        };

        RtRestirDiGraphOutputs {
            reservoirs: final_reservoirs,
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
            create_reservoir_buffer(device, allocator, reservoir_count, "rt_restir_di_temporal")?;
        let selected_reservoirs = match create_selected_reservoir_buffers(
            device,
            allocator,
            self.selected_reservoirs.len(),
            reservoir_count,
            "rt_restir_di_selected",
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
                    destroy_buffers(selected_reservoirs, device, allocator);
                    temporal_reservoirs.destroy(device, allocator);
                    return Err(error);
                }
            };
        std::mem::replace(&mut self.temporal_reservoirs, temporal_reservoirs)
            .destroy(device, allocator);
        destroy_buffers(
            std::mem::replace(&mut self.selected_reservoirs, selected_reservoirs),
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
        destroy_buffers(Vec::from(self.surface_history_buffers), device, allocator);
        destroy_buffers(self.selected_reservoirs, device, allocator);
        self.temporal_reservoirs.destroy(device, allocator);
        self.direct_lights.destroy(device, allocator);
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
                .range(std::mem::size_of::<GpuRestirDiUniforms>() as u64);
            let direct_lights_info = vk::DescriptorBufferInfo::default()
                .buffer(self.direct_lights.handle)
                .offset(0)
                .range(self.direct_lights.size);
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
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&direct_lights_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&history_uniform_info)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        for (set_idx, &descriptor_set) in self.spatial_descriptor_sets.iter().enumerate() {
            let uniform_info = vk::DescriptorBufferInfo::default()
                .buffer(self.uniform_buffers[set_idx].handle)
                .offset(0)
                .range(std::mem::size_of::<GpuRestirDiUniforms>() as u64);
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

    pub fn selected_current_buffer(&self, frame_slot: usize) -> &GpuBuffer {
        &self.selected_reservoirs[self.selected_current_slot(frame_slot)]
    }

    pub fn selected_history_buffer(&self, frame_slot: usize) -> &GpuBuffer {
        &self.selected_reservoirs[self.selected_history_slot(frame_slot)]
    }

    fn selected_current_slot(&self, frame_slot: usize) -> usize {
        frame_slot % self.selected_reservoirs.len()
    }

    fn selected_history_slot(&self, frame_slot: usize) -> usize {
        (self.selected_current_slot(frame_slot) + self.selected_reservoirs.len() - 1)
            % self.selected_reservoirs.len()
    }

    fn current_surface_history_buffer(&self, frame_index: u64) -> &GpuBuffer {
        &self.surface_history_buffers[current_history_index(frame_index)]
    }

    fn previous_surface_history_buffer(&self, frame_index: u64) -> &GpuBuffer {
        &self.surface_history_buffers[previous_history_index(frame_index)]
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
            std::mem::size_of::<GpuRestirDiUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("rt_restir_di_uniforms_{slot}"),
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
            &format!("rt_restir_di_history_uniforms_{slot}"),
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

fn create_direct_light_buffer(
    device: &ash::Device,
    allocator: &GpuAllocator,
    direct_lights: &[GpuDirectLight],
) -> Result<GpuBuffer> {
    let count = direct_lights.len().max(1);
    let buffer = GpuBuffer::new(
        device,
        allocator,
        (count * std::mem::size_of::<GpuDirectLight>()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "rt_restir_di_direct_lights",
    )?;
    if !direct_lights.is_empty() {
        write_mapped_slice(buffer.mapped_ptr(), direct_lights);
    }
    Ok(buffer)
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
        (count * std::mem::size_of::<GpuRestirDiReservoir>()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::GpuOnly,
        name,
    )
}

fn create_selected_reservoir_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
    reservoir_count: u32,
    name_prefix: &str,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count.max(2));
    for slot in 0..frame_count.max(2) {
        match create_reservoir_buffer(
            device,
            allocator,
            reservoir_count,
            &format!("{name_prefix}_{slot}"),
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
        create_surface_history_buffer(device, allocator, width, height, "rt_restir_di_surface_0")?;
    let second = match create_surface_history_buffer(
        device,
        allocator,
        width,
        height,
        "rt_restir_di_surface_1",
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

fn current_history_index(frame_index: u64) -> usize {
    (frame_index as usize) & 1
}

fn previous_history_index(frame_index: u64) -> usize {
    current_history_index(frame_index) ^ 1
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

fn create_raygen_pipeline(
    device: &ash::Device,
    allocator: &GpuAllocator,
    ray_tracing_pipeline_loader: &ash::khr::ray_tracing_pipeline::Device,
    rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    raygen_spirv: &[u8],
    descriptor_set_layout: vk::DescriptorSetLayout,
    pass_name: &'static str,
) -> Result<(RayTracingPipeline, ShaderBindingTable)> {
    let shader_module = create_shader_module(device, raygen_spirv)
        .with_context(|| format!("failed to create {pass_name} raygen shader module"))?;
    let pipeline = match RayTracingPipeline::new_raygen_only(
        device,
        ray_tracing_pipeline_loader,
        shader_module,
        c"main",
        &[descriptor_set_layout],
        &[],
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe { device.destroy_shader_module(shader_module, None) };
            return Err(error);
        }
    };
    unsafe { device.destroy_shader_module(shader_module, None) };
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

fn rt_restir_debug_view(debug_view: RtDebugView) -> u32 {
    match debug_view {
        RtDebugView::DirectReservoir => 1,
        _ => 0,
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

fn write_mapped_slice<T: Copy>(mapped_ptr: Option<*mut u8>, values: &[T]) {
    let Some(ptr) = mapped_ptr else {
        return;
    };
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr() as *const u8,
            ptr,
            std::mem::size_of_val(values),
        );
    }
}

#[cfg(test)]
mod shader_source_tests {
    use ash::vk;

    #[test]
    fn rt_restir_di_descriptor_specs_match_initial_temporal_shader_resources() {
        let specs = super::RtRestirDiPass::descriptor_binding_specs();
        let actual = specs
            .iter()
            .map(|spec| (spec.binding, spec.descriptor_type))
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, vk::DescriptorType::STORAGE_BUFFER),
                (2, vk::DescriptorType::STORAGE_BUFFER),
                (3, vk::DescriptorType::STORAGE_BUFFER),
                (4, vk::DescriptorType::STORAGE_BUFFER),
                (5, vk::DescriptorType::STORAGE_BUFFER),
                (6, vk::DescriptorType::STORAGE_BUFFER),
                (7, vk::DescriptorType::UNIFORM_BUFFER),
            ]
        );
    }

    #[test]
    fn rt_restir_di_pass_ping_pongs_selected_reservoir_and_surface_history() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_restir_di.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT ReSTIR-DI implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "RayTracingPipeline",
            "ShaderBindingTable",
            "descriptor_set_layout",
            "descriptor_pool",
            "descriptor_sets",
            "uniform_buffers",
            "history_uniform_buffers",
            "direct_lights",
            "selected_reservoirs",
            "surface_history_buffers",
            "selected_current_buffer",
            "selected_history_buffer",
            "current_surface_history_buffer",
            "previous_surface_history_buffer",
            "RayTracingPipeline::new_raygen_only",
            "cmd_trace_rays",
            "update_uniforms",
            "update_history_uniforms",
            "update_frame_descriptors",
            "rt_restir_di_initial",
        ] {
            assert!(
                implementation.contains(token),
                "RT ReSTIR-DI pass implementation missing {token}"
            );
        }
        for token in [
            "fnselected_current_slot(&self,frame_slot:usize)->usize",
            "fnselected_history_slot(&self,frame_slot:usize)->usize",
            "self.selected_current_buffer(frame_slot)",
            "self.selected_history_buffer(frame_slot)",
            "builder.read_as(selected_history",
            "builder.write_as(selected_current",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-DI selected history contract missing {token}"
            );
        }
    }

    #[test]
    fn rt_restir_di_shader_temporally_reuses_history_reservoir_on_compatible_surface() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_di.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "StructuredBuffer<RestirDiReservoir>history_reservoirs",
            "RWStructuredBuffer<RestirDiReservoir>output_reservoirs",
            "StructuredBuffer<RtSurfacePixel>previous_surface_history",
            "RWStructuredBuffer<RtSurfacePixel>current_surface_history",
            "current_surface_history[index]=surface",
            "rt_restir_surfaces_compatible(surface,previous_surface)",
            "restir.temporal_enabled",
            "restir_di_reservoir_stream_weight",
            "restir_di_finalize_reservoir_on_surface_with_target",
            "selected_target_pdf=history_target_pdf",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-DI temporal shader missing {token}"
            );
        }
        assert!(
            !source.contains("spatial_reservoir"),
            "RT ReSTIR-DI must keep spatial reuse out of this phase"
        );
    }

    #[test]
    fn rt_restir_di_shader_reprojects_temporal_history_instead_of_same_pixel_lookup() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_di.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "ConstantBuffer<RtHistoryUniforms>rt_history",
            "boolrt_restir_reproject(float3world_position,outfloat2previous_uv)",
            "rt_history.previous_view_proj",
            "mul(float4(world_position,1.0),rt_history.previous_view_proj)",
            "boolrt_restir_previous_uv_inside(float2previous_uv)",
            "rt_history.previous_resolution",
            "previous_uv*float2(previous_extent)",
            "uintprevious_index=previous_pixel.y*previous_extent.x+previous_pixel.x",
            "rt_history.normal_threshold",
            "rt_history.depth_threshold",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-DI temporal reprojection missing {token}"
            );
        }
        assert!(
            !compact.contains("uintprevious_index=index;"),
            "RT ReSTIR-DI must not reuse same-pixel history after reprojection is available"
        );
    }

    #[test]
    fn rt_restir_di_pass_spatial_stage_uses_temporal_intermediate() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_restir_di.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT ReSTIR-DI implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "spatial_pipeline",
            "spatial_shader_binding_table",
            "spatial_descriptor_set_layout",
            "spatial_descriptor_pool",
            "spatial_descriptor_sets",
            "temporal_reservoirs",
            "spatial_raygen_spirv",
            "fnspatial_descriptor_binding_specs()->[DescriptorBindingSpec;5]",
            "create_reservoir_buffer(device,allocator,reservoir_count,\"rt_restir_di_temporal\")",
            "letspatial_active=spatial_enabled&&history_initialized",
            "rt_restir_di_spatial",
            "builder.read_as(temporal_reservoir",
            "builder.read_as(surface",
            "builder.write_as(selected_current",
            "cmd_trace_rays",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-DI spatial pass missing {token}"
            );
        }

        assert!(
            !compact.contains("builder.read_as(selected_current,AccessKind::RayTracingShaderRead)"),
            "RT ReSTIR-DI spatial pass must not read the selected-current buffer it writes"
        );
    }

    #[test]
    fn rt_restir_di_spatial_reads_written_temporal_resource_version() {
        let source = crate::render::source_checks::read_source("src/render/passes/rt_restir_di.rs");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("RT ReSTIR-DI implementation should precede tests");
        let compact = crate::render::source_checks::compact(implementation);

        for token in [
            "lettemporal_reservoir_output=writes[0];",
            "builder.read_as(temporal_reservoir_output,AccessKind::RayTracingShaderRead);",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-DI spatial graph must read the written temporal resource version with {token}"
            );
        }
        assert!(
            !compact
                .contains("builder.read_as(temporal_reservoir,AccessKind::RayTracingShaderRead);"),
            "RT ReSTIR-DI spatial graph must not read the imported temporal reservoir handle after it has been written"
        );
    }

    #[test]
    fn rt_restir_di_spatial_descriptor_specs_match_shader_resources() {
        let specs = super::RtRestirDiPass::spatial_descriptor_binding_specs();
        let actual = specs
            .iter()
            .map(|spec| (spec.binding, spec.descriptor_type))
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![
                (0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, vk::DescriptorType::STORAGE_BUFFER),
                (2, vk::DescriptorType::STORAGE_BUFFER),
                (3, vk::DescriptorType::STORAGE_BUFFER),
                (4, vk::DescriptorType::UNIFORM_BUFFER),
            ]
        );
    }

    #[test]
    fn rt_restir_di_spatial_shader_reuses_compatible_neighbor_temporal_reservoirs() {
        let source = crate::render::source_checks::read_source(
            "assets/shaders/passes/rt_restir_di_spatial.rgen.slang",
        );
        let compact = crate::render::source_checks::compact(&source);

        for token in [
            "#include\"restir_di_common.slang\"",
            "#include\"rt_history_common.slang\"",
            "StructuredBuffer<RestirDiReservoir>temporal_reservoirs",
            "RWStructuredBuffer<RestirDiReservoir>output_reservoirs",
            "StructuredBuffer<RtSurfacePixel>surface_pixels",
            "ConstantBuffer<RtHistoryUniforms>rt_history",
            "staticconstint2rt_restir_spatial_offsets[8]",
            "rt_restir_spatial_surfaces_compatible",
            "rt_history.normal_threshold",
            "rt_history.depth_threshold",
            "restir.spatial_enabled==0u",
            "restir_di_is_valid_reservoir(neighbor)",
            "restir_di_reservoir_stream_weight",
            "restir_di_finalize_reservoir_on_surface_with_target",
        ] {
            assert!(
                compact.contains(token),
                "RT ReSTIR-DI spatial shader missing {token}"
            );
        }
    }
}
