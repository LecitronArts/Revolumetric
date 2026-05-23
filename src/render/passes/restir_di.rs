use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::GpuImage;
use crate::render::passes::vpt::VptPass;
use crate::render::passes::vpt_surface::{
    VptCurrentSurfaceResources, VptPreviousSurfaceResources, VptSurfacePass,
};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::restir_di::{
    GpuDirectLight, GpuRestirDiReservoir, GpuRestirDiUniforms, RestirDiSettings,
};

pub struct RestirDiPass {
    initial_stage: RestirDiStage,
    temporal_stage: RestirDiStage,
    spatial_stage: RestirDiStage,
    uniform_buffers: Vec<GpuBuffer>,
    direct_lights: GpuBuffer,
    initial_reservoirs: GpuBuffer,
    temporal_reservoirs: GpuBuffer,
    selected_reservoirs: Vec<GpuBuffer>,
    width: u32,
    height: u32,
    reservoir_count: u32,
    light_count: u32,
}

pub struct RestirDiPassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub frame_count: usize,
    pub initial_spirv: &'a [u8],
    pub temporal_spirv: &'a [u8],
    pub spatial_spirv: &'a [u8],
    pub direct_lights: &'a [GpuDirectLight],
}

pub struct RestirDiGraphBuffers<'a> {
    pub uniform_buffer: &'a GpuBuffer,
    pub uniform_resource: ResourceHandle,
    pub selected_current_buffer: &'a GpuBuffer,
    pub selected_current_resource: ResourceHandle,
}

struct RestirDiBuffers {
    uniform_buffers: Vec<GpuBuffer>,
    direct_lights: GpuBuffer,
    initial_reservoirs: GpuBuffer,
    temporal_reservoirs: GpuBuffer,
    selected_reservoirs: Vec<GpuBuffer>,
}

struct RestirDiStage {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

fn restir_di_effective_settings(
    settings: RestirDiSettings,
    history_initialized: bool,
) -> RestirDiSettings {
    let mut settings = settings;
    if !history_initialized {
        settings.temporal_enabled = false;
    }
    settings.spatial_enabled = settings.temporal_enabled && settings.spatial_enabled;
    settings
}

impl RestirDiPass {
    pub(crate) fn initial_descriptor_binding_specs() -> [DescriptorBindingSpec; 6] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_IMAGE),
        ]
    }

    pub(crate) fn temporal_descriptor_binding_specs() -> [DescriptorBindingSpec; 14] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(6, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(7, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(8, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(9, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(10, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(11, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(12, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(13, vk::DescriptorType::STORAGE_IMAGE),
        ]
    }

    pub(crate) fn spatial_descriptor_binding_specs() -> [DescriptorBindingSpec; 6] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_IMAGE),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: RestirDiPassCreateInfo<'_>,
    ) -> Result<Self> {
        let light_count = info.direct_lights.len() as u32;
        let reservoir_count = info.width.saturating_mul(info.height);
        let buffers = RestirDiBuffers::new(
            device,
            allocator,
            info.frame_count,
            reservoir_count,
            info.direct_lights,
        )?;

        let initial_stage = match RestirDiStage::new(
            device,
            info.initial_spirv,
            &Self::initial_descriptor_binding_specs(),
            info.frame_count,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: info.frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 2 * info.frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 3 * info.frame_count as u32,
                },
            ],
        ) {
            Ok(stage) => stage,
            Err(error) => {
                buffers.destroy(device, allocator);
                return Err(error);
            }
        };
        let temporal_stage = match RestirDiStage::new(
            device,
            info.temporal_spirv,
            &Self::temporal_descriptor_binding_specs(),
            info.frame_count,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: info.frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 3 * info.frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 10 * info.frame_count as u32,
                },
            ],
        ) {
            Ok(stage) => stage,
            Err(error) => {
                initial_stage.destroy(device);
                buffers.destroy(device, allocator);
                return Err(error);
            }
        };
        let spatial_stage = match RestirDiStage::new(
            device,
            info.spatial_spirv,
            &Self::spatial_descriptor_binding_specs(),
            info.frame_count,
            &[
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: info.frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 2 * info.frame_count as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 3 * info.frame_count as u32,
                },
            ],
        ) {
            Ok(stage) => stage,
            Err(error) => {
                temporal_stage.destroy(device);
                initial_stage.destroy(device);
                buffers.destroy(device, allocator);
                return Err(error);
            }
        };

        let pass = Self {
            initial_stage,
            temporal_stage,
            spatial_stage,
            uniform_buffers: buffers.uniform_buffers,
            direct_lights: buffers.direct_lights,
            initial_reservoirs: buffers.initial_reservoirs,
            temporal_reservoirs: buffers.temporal_reservoirs,
            selected_reservoirs: buffers.selected_reservoirs,
            width: info.width,
            height: info.height,
            reservoir_count,
            light_count,
        };
        pass.write_descriptor_sets(device);
        Ok(pass)
    }

    pub fn update_uniforms(&self, frame_slot: usize, settings: RestirDiSettings, frame_index: u64) {
        let uniforms = settings.gpu_uniforms(
            frame_index as u32,
            self.reservoir_count,
            self.light_count,
            self.width,
            self.height,
        );
        write_mapped(self.uniform_buffers[frame_slot].mapped_ptr(), &uniforms);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        device: &ash::Device,
        vpt: &'a VptPass,
        frame_slot: usize,
        frame_index: u64,
        settings: RestirDiSettings,
        history_initialized: bool,
        final_surface_writes: VptCurrentSurfaceResources,
        previous_surface_resources: VptPreviousSurfaceResources,
        profiler: Option<&'a GpuProfiler>,
    ) -> RestirDiGraphBuffers<'a> {
        let settings = restir_di_effective_settings(settings, history_initialized);
        self.update_uniforms(frame_slot, settings, frame_index);

        let (uniform_buffer, uniform_size, uniform_usage) = self.uniform_buffer(frame_slot);
        let (direct_light_buffer, direct_light_size, direct_light_usage) =
            self.direct_light_buffer();
        let (initial_buffer, initial_size, initial_usage) = self.initial_buffer();
        let (temporal_buffer, temporal_size, temporal_usage) = self.temporal_buffer();
        let (selected_current_buffer, selected_current_size, selected_current_usage) =
            self.selected_current_buffer(frame_slot);
        let (selected_history_buffer, selected_history_size, selected_history_usage) =
            self.selected_history_buffer(frame_slot);
        let temporal_active = settings.temporal_enabled;
        let spatial_active = temporal_active && settings.spatial_enabled;
        self.update_frame_descriptors(
            device,
            frame_slot,
            selected_history_buffer,
            selected_current_buffer,
            temporal_active,
            spatial_active,
        );

        let initial_resource = graph.import_buffer_with_access(
            initial_buffer.handle,
            initial_size,
            initial_usage,
            AccessKind::Undefined,
        );
        let temporal_resource = graph.import_buffer_with_access(
            temporal_buffer.handle,
            temporal_size,
            temporal_usage,
            AccessKind::Undefined,
        );
        let selected_current_resource = graph.import_buffer_with_access(
            selected_current_buffer.handle,
            selected_current_size,
            selected_current_usage,
            AccessKind::Undefined,
        );
        let selected_history_resource = graph.import_buffer_with_access(
            selected_history_buffer.handle,
            selected_history_size,
            selected_history_usage,
            if history_initialized {
                AccessKind::ComputeShaderWrite
            } else {
                AccessKind::Undefined
            },
        );
        let uniform_resource = graph.import_buffer_with_access(
            uniform_buffer.handle,
            uniform_size,
            uniform_usage,
            AccessKind::ComputeShaderRead,
        );
        let direct_light_resource = graph.import_buffer_with_access(
            direct_light_buffer.handle,
            direct_light_size,
            direct_light_usage,
            AccessKind::ComputeShaderRead,
        );

        let initial_writes = graph.add_pass("restir_di_initial", QueueType::Compute, |builder| {
            builder.read_as(uniform_resource, AccessKind::ComputeShaderRead);
            builder.read_as(
                final_surface_writes.position_depth,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(
                final_surface_writes.normal_roughness,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(
                final_surface_writes.albedo_material,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(direct_light_resource, AccessKind::ComputeShaderRead);
            let initial_output_resource = if temporal_active {
                initial_resource
            } else {
                selected_current_resource
            };
            builder.write_as(initial_output_resource, AccessKind::ComputeShaderWrite);
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::RestirDiInitial,
                    );
                }
                self.record_initial(ctx.device, ctx.command_buffer, frame_slot);
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::RestirDiInitial,
                    );
                }
            })
        });
        let initial_dep = initial_writes[0];
        let temporal_dep = if temporal_active {
            let temporal_writes =
                graph.add_pass("restir_di_temporal", QueueType::Compute, |builder| {
                    builder.read_as(uniform_resource, AccessKind::ComputeShaderRead);
                    builder.read_as(
                        final_surface_writes.position_depth,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        final_surface_writes.normal_roughness,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        final_surface_writes.albedo_material,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        final_surface_writes.material_roughness,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(final_surface_writes.view_z, AccessKind::ComputeShaderRead);
                    builder.read_as(
                        final_surface_writes.motion_history,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        final_surface_writes.motion_flags,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        final_surface_writes.brick_generation,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        previous_surface_resources.position_depth,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        previous_surface_resources.normal_roughness,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        previous_surface_resources.albedo_material,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        previous_surface_resources.material_roughness,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        previous_surface_resources.view_z,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        previous_surface_resources.brick_generation,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(initial_dep, AccessKind::ComputeShaderRead);
                    builder.read_as(selected_history_resource, AccessKind::ComputeShaderRead);
                    let temporal_output_resource = if spatial_active {
                        temporal_resource
                    } else {
                        selected_current_resource
                    };
                    builder.write_as(temporal_output_resource, AccessKind::ComputeShaderWrite);
                    Box::new(move |ctx| {
                        if let Some(profiler) = profiler {
                            profiler.begin_scope(
                                ctx.device,
                                ctx.command_buffer,
                                frame_slot,
                                GpuProfileScope::RestirDiTemporal,
                            );
                        }
                        self.record_temporal(ctx.device, ctx.command_buffer, frame_slot);
                        if let Some(profiler) = profiler {
                            profiler.end_scope(
                                ctx.device,
                                ctx.command_buffer,
                                frame_slot,
                                GpuProfileScope::RestirDiTemporal,
                            );
                        }
                    })
                });
            temporal_writes[0]
        } else {
            initial_dep
        };
        let selected_current_dep = if spatial_active {
            let spatial_writes =
                graph.add_pass("restir_di_spatial", QueueType::Compute, |builder| {
                    builder.read_as(uniform_resource, AccessKind::ComputeShaderRead);
                    builder.read_as(
                        final_surface_writes.position_depth,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        final_surface_writes.normal_roughness,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        final_surface_writes.albedo_material,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(temporal_dep, AccessKind::ComputeShaderRead);
                    builder.write_as(selected_current_resource, AccessKind::ComputeShaderWrite);
                    Box::new(move |ctx| {
                        if let Some(profiler) = profiler {
                            profiler.begin_scope(
                                ctx.device,
                                ctx.command_buffer,
                                frame_slot,
                                GpuProfileScope::RestirDiSpatial,
                            );
                        }
                        self.record_spatial(ctx.device, ctx.command_buffer, frame_slot);
                        if let Some(profiler) = profiler {
                            profiler.end_scope(
                                ctx.device,
                                ctx.command_buffer,
                                frame_slot,
                                GpuProfileScope::RestirDiSpatial,
                            );
                        }
                    })
                });
            spatial_writes[0]
        } else {
            temporal_dep
        };

        vpt.update_restir_di_descriptors(
            device,
            frame_slot,
            uniform_buffer,
            selected_current_buffer,
        );

        RestirDiGraphBuffers {
            uniform_buffer,
            uniform_resource,
            selected_current_buffer,
            selected_current_resource: selected_current_dep,
        }
    }

    pub fn update_surface_descriptors(&self, device: &ash::Device, surface: &VptSurfacePass) {
        let current_surface_images = [
            &surface.surface_position_depth,
            &surface.surface_normal_roughness,
            &surface.surface_albedo_material,
        ];
        let temporal_surface_images = [
            &surface.surface_position_depth,
            &surface.surface_normal_roughness,
            &surface.surface_albedo_material,
            &surface.previous_surface_position_depth,
            &surface.previous_surface_normal_roughness,
            &surface.previous_surface_albedo_material,
            &surface.motion_history,
            &surface.motion_flags,
            &surface.surface_brick_generation,
            &surface.previous_surface_brick_generation,
        ];
        self.initial_stage
            .write_image_descriptors(device, 3, &current_surface_images);
        self.temporal_stage
            .write_image_descriptors(device, 4, &temporal_surface_images);
        self.spatial_stage
            .write_image_descriptors(device, 3, &current_surface_images);
    }

    pub fn resize_buffers(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let reservoir_count = width.saturating_mul(height);
        let initial_reservoirs =
            create_reservoir_buffer(device, allocator, reservoir_count, "restir_di_initial")?;
        let temporal_reservoirs =
            create_reservoir_buffer(device, allocator, reservoir_count, "restir_di_temporal")?;
        let selected_reservoirs = create_selected_reservoir_buffers(
            device,
            allocator,
            self.selected_reservoirs.len(),
            reservoir_count,
            "restir_di_selected",
        )?;

        std::mem::replace(&mut self.initial_reservoirs, initial_reservoirs)
            .destroy(device, allocator);
        std::mem::replace(&mut self.temporal_reservoirs, temporal_reservoirs)
            .destroy(device, allocator);
        destroy_buffers(
            std::mem::replace(&mut self.selected_reservoirs, selected_reservoirs),
            device,
            allocator,
        );

        self.width = width;
        self.height = height;
        self.reservoir_count = reservoir_count;
        self.write_descriptor_sets(device);
        Ok(())
    }

    pub fn uniform_buffer(
        &self,
        frame_slot: usize,
    ) -> (&GpuBuffer, vk::DeviceSize, vk::BufferUsageFlags) {
        let buffer = &self.uniform_buffers[frame_slot];
        (buffer, buffer.size, buffer.usage)
    }

    pub fn direct_light_buffer(&self) -> (&GpuBuffer, vk::DeviceSize, vk::BufferUsageFlags) {
        (
            &self.direct_lights,
            self.direct_lights.size,
            self.direct_lights.usage,
        )
    }

    pub fn initial_buffer(&self) -> (&GpuBuffer, vk::DeviceSize, vk::BufferUsageFlags) {
        (
            &self.initial_reservoirs,
            self.initial_reservoirs.size,
            self.initial_reservoirs.usage,
        )
    }

    pub fn temporal_buffer(&self) -> (&GpuBuffer, vk::DeviceSize, vk::BufferUsageFlags) {
        (
            &self.temporal_reservoirs,
            self.temporal_reservoirs.size,
            self.temporal_reservoirs.usage,
        )
    }

    pub fn selected_current_buffer(
        &self,
        frame_slot: usize,
    ) -> (&GpuBuffer, vk::DeviceSize, vk::BufferUsageFlags) {
        let buffer = &self.selected_reservoirs[self.selected_current_slot(frame_slot)];
        (buffer, buffer.size, buffer.usage)
    }

    pub fn selected_history_buffer(
        &self,
        frame_slot: usize,
    ) -> (&GpuBuffer, vk::DeviceSize, vk::BufferUsageFlags) {
        let buffer = &self.selected_reservoirs[self.selected_history_slot(frame_slot)];
        (buffer, buffer.size, buffer.usage)
    }

    pub fn update_frame_descriptors(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        selected_history: &GpuBuffer,
        selected_current: &GpuBuffer,
        temporal_enabled: bool,
        spatial_enabled: bool,
    ) {
        let initial_output = if temporal_enabled {
            &self.initial_reservoirs
        } else {
            selected_current
        };
        let temporal_output = if spatial_enabled {
            &self.temporal_reservoirs
        } else {
            selected_current
        };
        self.initial_stage.write_storage_descriptors_for_frame(
            device,
            frame_slot,
            2,
            &[initial_output],
        );
        self.temporal_stage.write_storage_descriptors_for_frame(
            device,
            frame_slot,
            1,
            &[&self.initial_reservoirs, selected_history, temporal_output],
        );
        self.spatial_stage.write_storage_descriptors_for_frame(
            device,
            frame_slot,
            1,
            &[&self.temporal_reservoirs, selected_current],
        );
    }

    pub fn record_initial(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        self.initial_stage
            .record(device, cmd, frame_slot, self.width, self.height);
    }

    pub fn record_temporal(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        self.temporal_stage
            .record(device, cmd, frame_slot, self.width, self.height);
    }

    pub fn record_spatial(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        self.spatial_stage
            .record(device, cmd, frame_slot, self.width, self.height);
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.initial_stage.destroy(device);
        self.temporal_stage.destroy(device);
        self.spatial_stage.destroy(device);
        for buffer in self.uniform_buffers {
            buffer.destroy(device, allocator);
        }
        self.direct_lights.destroy(device, allocator);
        self.initial_reservoirs.destroy(device, allocator);
        self.temporal_reservoirs.destroy(device, allocator);
        destroy_buffers(self.selected_reservoirs, device, allocator);
    }

    fn write_descriptor_sets(&self, device: &ash::Device) {
        self.initial_stage.write_descriptors(
            device,
            &self.uniform_buffers,
            &[&self.direct_lights, &self.initial_reservoirs],
        );
        self.temporal_stage.write_descriptors(
            device,
            &self.uniform_buffers,
            &[
                &self.initial_reservoirs,
                &self.selected_reservoirs[self.selected_history_slot(0)],
                &self.temporal_reservoirs,
            ],
        );
        self.spatial_stage.write_descriptors(
            device,
            &self.uniform_buffers,
            &[
                &self.temporal_reservoirs,
                &self.selected_reservoirs[self.selected_current_slot(0)],
            ],
        );
    }

    fn selected_current_slot(&self, frame_slot: usize) -> usize {
        frame_slot % self.selected_reservoirs.len()
    }

    fn selected_history_slot(&self, frame_slot: usize) -> usize {
        (self.selected_current_slot(frame_slot) + self.selected_reservoirs.len() - 1)
            % self.selected_reservoirs.len()
    }
}

impl RestirDiBuffers {
    fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        frame_count: usize,
        reservoir_count: u32,
        direct_lights: &[GpuDirectLight],
    ) -> Result<Self> {
        let uniform_buffers = create_uniform_buffers(device, allocator, frame_count)?;
        let direct_lights = match create_direct_light_buffer(device, allocator, direct_lights) {
            Ok(buffer) => buffer,
            Err(error) => {
                destroy_buffers(uniform_buffers, device, allocator);
                return Err(error);
            }
        };
        let initial_reservoirs = match create_reservoir_buffer(
            device,
            allocator,
            reservoir_count,
            "restir_di_initial",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                direct_lights.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                return Err(error);
            }
        };
        let temporal_reservoirs =
            match create_reservoir_buffer(device, allocator, reservoir_count, "restir_di_temporal")
            {
                Ok(buffer) => buffer,
                Err(error) => {
                    initial_reservoirs.destroy(device, allocator);
                    direct_lights.destroy(device, allocator);
                    destroy_buffers(uniform_buffers, device, allocator);
                    return Err(error);
                }
            };
        let selected_reservoirs = match create_selected_reservoir_buffers(
            device,
            allocator,
            frame_count,
            reservoir_count,
            "restir_di_selected",
        ) {
            Ok(buffers) => buffers,
            Err(error) => {
                temporal_reservoirs.destroy(device, allocator);
                initial_reservoirs.destroy(device, allocator);
                direct_lights.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                return Err(error);
            }
        };

        Ok(Self {
            uniform_buffers,
            direct_lights,
            initial_reservoirs,
            temporal_reservoirs,
            selected_reservoirs,
        })
    }

    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        destroy_buffers(self.uniform_buffers, device, allocator);
        self.direct_lights.destroy(device, allocator);
        self.initial_reservoirs.destroy(device, allocator);
        self.temporal_reservoirs.destroy(device, allocator);
        destroy_buffers(self.selected_reservoirs, device, allocator);
    }
}

impl RestirDiStage {
    fn new(
        device: &ash::Device,
        spirv_bytes: &[u8],
        bindings: &[DescriptorBindingSpec],
        frame_count: usize,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self> {
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
        let layouts: Vec<_> = (0..frame_count).map(|_| descriptor_set_layout).collect();
        let descriptor_sets = match descriptor_pool.allocate(device, &layouts) {
            Ok(sets) => sets,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        let shader_module = match create_shader_module(device, spirv_bytes) {
            Ok(module) => module,
            Err(error) => {
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
        })
    }

    fn write_descriptors(
        &self,
        device: &ash::Device,
        uniform_buffers: &[GpuBuffer],
        storage_buffers: &[&GpuBuffer],
    ) {
        for (set_idx, &ds) in self.descriptor_sets.iter().enumerate() {
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[set_idx].handle)
                .offset(0)
                .range(std::mem::size_of::<GpuRestirDiUniforms>() as u64);
            let storage_infos: Vec<vk::DescriptorBufferInfo> = storage_buffers
                .iter()
                .map(|buffer| {
                    vk::DescriptorBufferInfo::default()
                        .buffer(buffer.handle)
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
            ];
            writes.extend(storage_infos.iter().enumerate().map(|(idx, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding((idx + 1) as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            }));

            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    fn write_storage_descriptors_for_frame(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        first_binding: u32,
        storage_buffers: &[&GpuBuffer],
    ) {
        let Some(&ds) = self.descriptor_sets.get(frame_slot) else {
            return;
        };
        let storage_infos: Vec<_> = storage_buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        let writes: Vec<_> = storage_infos
            .iter()
            .enumerate()
            .map(|(idx, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(first_binding + idx as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    fn write_image_descriptors(
        &self,
        device: &ash::Device,
        first_binding: u32,
        images: &[&GpuImage],
    ) {
        let image_infos: Vec<vk::DescriptorImageInfo> = images
            .iter()
            .map(|image| {
                vk::DescriptorImageInfo::default()
                    .image_view(image.view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect();

        for &ds in &self.descriptor_sets {
            let writes: Vec<_> = image_infos
                .iter()
                .enumerate()
                .map(|(idx, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(first_binding + idx as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(info))
                })
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        width: u32,
        height: u32,
    ) {
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
            device.cmd_dispatch(cmd, width.div_ceil(8), height.div_ceil(8), 1);
        }
    }

    fn destroy(self, device: &ash::Device) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
    }
}

fn create_uniform_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count);
    for slot in 0..frame_count {
        buffers.push(GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GpuRestirDiUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("restir_di_uniforms_{slot}"),
        )?);
    }
    Ok(buffers)
}

fn destroy_buffers(buffers: Vec<GpuBuffer>, device: &ash::Device, allocator: &GpuAllocator) {
    for buffer in buffers {
        buffer.destroy(device, allocator);
    }
}

fn create_direct_light_buffer(
    device: &ash::Device,
    allocator: &GpuAllocator,
    direct_lights: &[GpuDirectLight],
) -> Result<GpuBuffer> {
    let buffer_len = direct_lights.len().max(1);
    let buffer = GpuBuffer::new(
        device,
        allocator,
        (buffer_len * std::mem::size_of::<GpuDirectLight>()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "restir_di_direct_lights",
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
mod shader_source_tests;
