use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::area_restir::{
    AreaRestirSettings, GpuAreaRestirReservoir, GpuAreaRestirUniforms,
};
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::vpt::VptPass;
use crate::render::passes::vpt_surface::{
    VptCurrentSurfaceResources, VptPreviousSurfaceResources, VptSurfacePass,
};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::render::traversal_stats::VptTraversalStatsBuffer;
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct AreaRestirPass {
    initial_stage: AreaRestirStage,
    temporal_stage: AreaRestirStage,
    spatial_stage: AreaRestirStage,
    uniform_buffers: Vec<GpuBuffer>,
    initial_reservoirs: GpuBuffer,
    temporal_reservoirs: GpuBuffer,
    selected_reservoirs: Vec<GpuBuffer>,
    pub debug_image: GpuImage,
    width: u32,
    height: u32,
    reservoir_count: u32,
}

pub struct AreaRestirPassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub frame_count: usize,
    pub initial_spirv: &'a [u8],
    pub temporal_spirv: &'a [u8],
    pub spatial_spirv: &'a [u8],
    pub scene_ubo: &'a SceneUniformBuffer,
    pub ucvh_gpu: &'a UcvhGpuResources,
    pub traversal_stats_buffers: &'a [VptTraversalStatsBuffer],
}

pub struct AreaRestirGraphBuffers<'a> {
    pub uniform_buffer: &'a GpuBuffer,
    pub uniform_resource: ResourceHandle,
    pub selected_current_buffer: &'a GpuBuffer,
    pub selected_current_resource: ResourceHandle,
    pub final_surface_writes: VptCurrentSurfaceResources,
    pub traversal_stats_resource: Option<ResourceHandle>,
}

struct AreaRestirBuffers {
    uniform_buffers: Vec<GpuBuffer>,
    initial_reservoirs: GpuBuffer,
    temporal_reservoirs: GpuBuffer,
    selected_reservoirs: Vec<GpuBuffer>,
    debug_image: GpuImage,
}

struct AreaRestirResizeResources {
    initial_reservoirs: GpuBuffer,
    temporal_reservoirs: GpuBuffer,
    selected_reservoirs: Vec<GpuBuffer>,
    debug_image: GpuImage,
}

struct AreaRestirStage {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

fn area_restir_effective_settings(
    settings: AreaRestirSettings,
    history_initialized: bool,
) -> AreaRestirSettings {
    let mut settings = settings;
    if !history_initialized {
        settings.temporal_enabled = false;
    }
    settings.spatial_enabled = settings.temporal_enabled && settings.spatial_enabled;
    settings
}

impl AreaRestirPass {
    pub(crate) fn initial_descriptor_binding_specs() -> [DescriptorBindingSpec; 16] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(6, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(7, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(8, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(9, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(10, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(11, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(12, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(13, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(14, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(15, vk::DescriptorType::STORAGE_BUFFER),
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

    pub(crate) fn spatial_descriptor_binding_specs() -> [DescriptorBindingSpec; 7] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(6, vk::DescriptorType::STORAGE_IMAGE),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: AreaRestirPassCreateInfo<'_>,
    ) -> Result<Self> {
        let reservoir_count = info.width.saturating_mul(info.height);
        let buffers = AreaRestirBuffers::new(
            device,
            allocator,
            info.frame_count,
            reservoir_count,
            info.width,
            info.height,
        )?;

        let initial_stage = match AreaRestirStage::new(
            device,
            info.initial_spirv,
            &Self::initial_descriptor_binding_specs(),
            info.frame_count,
        ) {
            Ok(stage) => stage,
            Err(error) => {
                buffers.destroy(device, allocator);
                return Err(error);
            }
        };
        let temporal_stage = match AreaRestirStage::new(
            device,
            info.temporal_spirv,
            &Self::temporal_descriptor_binding_specs(),
            info.frame_count,
        ) {
            Ok(stage) => stage,
            Err(error) => {
                initial_stage.destroy(device);
                buffers.destroy(device, allocator);
                return Err(error);
            }
        };
        let spatial_stage = match AreaRestirStage::new(
            device,
            info.spatial_spirv,
            &Self::spatial_descriptor_binding_specs(),
            info.frame_count,
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
            initial_reservoirs: buffers.initial_reservoirs,
            temporal_reservoirs: buffers.temporal_reservoirs,
            selected_reservoirs: buffers.selected_reservoirs,
            debug_image: buffers.debug_image,
            width: info.width,
            height: info.height,
            reservoir_count,
        };
        pass.write_descriptor_sets(device);
        pass.write_scene_descriptors(device, info.scene_ubo);
        pass.write_ucvh_descriptors(device, info.ucvh_gpu);
        for slot in 0..info.frame_count {
            pass.update_traversal_stats_descriptors(
                device,
                slot,
                &info.traversal_stats_buffers[slot],
            );
        }
        Ok(pass)
    }

    pub fn update_uniforms(
        &self,
        frame_slot: usize,
        settings: AreaRestirSettings,
        frame_index: u64,
    ) {
        let uniforms = settings.gpu_uniforms(
            frame_index as u32,
            self.reservoir_count,
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
        vpt_surface: &'a VptSurfacePass,
        frame_slot: usize,
        frame_index: u64,
        settings: AreaRestirSettings,
        history_initialized: bool,
        bootstrap_surface_writes: VptCurrentSurfaceResources,
        previous_surface_resources: VptPreviousSurfaceResources,
        traversal_stats_resource: Option<ResourceHandle>,
        profiler: Option<&'a GpuProfiler>,
    ) -> AreaRestirGraphBuffers<'a> {
        let settings = area_restir_effective_settings(settings, history_initialized);
        let temporal_active = settings.temporal_enabled;
        let spatial_active = temporal_active && settings.spatial_enabled;
        self.update_uniforms(frame_slot, settings, frame_index);

        let (uniform_buffer, uniform_size, uniform_usage) = self.uniform_buffer(frame_slot);
        let (initial_buffer, initial_size, initial_usage) = self.initial_buffer();
        let (temporal_buffer, temporal_size, temporal_usage) = self.temporal_buffer();
        let (selected_current_buffer, selected_current_size, selected_current_usage) =
            self.selected_current_buffer(frame_slot);
        let (selected_history_buffer, selected_history_size, selected_history_usage) =
            self.selected_history_buffer(frame_slot);
        self.update_frame_descriptors(
            device,
            frame_slot,
            selected_history_buffer,
            selected_current_buffer,
            temporal_active,
            spatial_active,
        );

        let uniform_resource = graph.import_buffer_with_access(
            uniform_buffer.handle,
            uniform_size,
            uniform_usage,
            AccessKind::ComputeShaderRead,
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
        let debug_resource = graph.import_image_with_access(
            self.debug_image.handle,
            self.debug_image.extent.width,
            self.debug_image.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Undefined,
        );

        let mut traversal_stats_after_initial = traversal_stats_resource;
        let initial_writes = graph.add_pass("area_restir_initial", QueueType::Compute, |builder| {
            builder.read_as(uniform_resource, AccessKind::ComputeShaderRead);
            builder.read_as(
                bootstrap_surface_writes.position_depth,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(
                bootstrap_surface_writes.normal_roughness,
                AccessKind::ComputeShaderRead,
            );
            builder.read_as(
                bootstrap_surface_writes.albedo_material,
                AccessKind::ComputeShaderRead,
            );
            let initial_output_resource = if temporal_active {
                initial_resource
            } else {
                selected_current_resource
            };
            builder.write_as(initial_output_resource, AccessKind::ComputeShaderWrite);
            builder.write_as(debug_resource, AccessKind::ComputeShaderWrite);
            if let Some(traversal_stats_resource) = traversal_stats_resource {
                traversal_stats_after_initial = Some(
                    builder.write_as(traversal_stats_resource, AccessKind::ComputeShaderReadWrite),
                );
            }
            Box::new(move |ctx| {
                if let Some(profiler) = profiler {
                    profiler.begin_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::AreaRestirInitial,
                    );
                }
                self.record_initial(ctx.device, ctx.command_buffer, frame_slot);
                if let Some(profiler) = profiler {
                    profiler.end_scope(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        GpuProfileScope::AreaRestirInitial,
                    );
                }
            })
        });
        let initial_dep = initial_writes[0];
        let debug_dep = initial_writes[1];
        let temporal_dep = if temporal_active {
            let temporal_writes =
                graph.add_pass("area_restir_temporal", QueueType::Compute, |builder| {
                    builder.read_as(uniform_resource, AccessKind::ComputeShaderRead);
                    builder.read_as(initial_dep, AccessKind::ComputeShaderRead);
                    builder.read_as(selected_history_resource, AccessKind::ComputeShaderRead);
                    bootstrap_surface_writes.for_each(|surface_write| {
                        builder.read_as(surface_write, AccessKind::ComputeShaderRead);
                    });
                    previous_surface_resources.for_each(|previous_surface_resource| {
                        builder.read_as(previous_surface_resource, AccessKind::ComputeShaderRead);
                    });
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
                                GpuProfileScope::AreaRestirTemporal,
                            );
                        }
                        self.record_temporal(ctx.device, ctx.command_buffer, frame_slot);
                        if let Some(profiler) = profiler {
                            profiler.end_scope(
                                ctx.device,
                                ctx.command_buffer,
                                frame_slot,
                                GpuProfileScope::AreaRestirTemporal,
                            );
                        }
                    })
                });
            temporal_writes[0]
        } else {
            initial_dep
        };
        let (selected_resource, final_debug_dep) = if spatial_active {
            let spatial_writes =
                graph.add_pass("area_restir_spatial", QueueType::Compute, |builder| {
                    builder.read_as(uniform_resource, AccessKind::ComputeShaderRead);
                    builder.read_as(temporal_dep, AccessKind::ComputeShaderRead);
                    builder.read_as(
                        bootstrap_surface_writes.position_depth,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        bootstrap_surface_writes.normal_roughness,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.read_as(
                        bootstrap_surface_writes.albedo_material,
                        AccessKind::ComputeShaderRead,
                    );
                    builder.write_as(selected_current_resource, AccessKind::ComputeShaderWrite);
                    builder.write_as(debug_dep, AccessKind::ComputeShaderWrite);
                    Box::new(move |ctx| {
                        if let Some(profiler) = profiler {
                            profiler.begin_scope(
                                ctx.device,
                                ctx.command_buffer,
                                frame_slot,
                                GpuProfileScope::AreaRestirSpatial,
                            );
                        }
                        self.record_spatial(ctx.device, ctx.command_buffer, frame_slot);
                        if let Some(profiler) = profiler {
                            profiler.end_scope(
                                ctx.device,
                                ctx.command_buffer,
                                frame_slot,
                                GpuProfileScope::AreaRestirSpatial,
                            );
                        }
                    })
                });
            (spatial_writes[0], spatial_writes[1])
        } else if temporal_active {
            (temporal_dep, debug_dep)
        } else {
            (initial_dep, debug_dep)
        };
        let _ = final_debug_dep;

        vpt.update_area_restir_descriptors(
            device,
            frame_slot,
            uniform_buffer,
            selected_current_buffer,
        );
        vpt_surface.update_area_restir_descriptors(
            device,
            frame_slot,
            uniform_buffer,
            selected_current_buffer,
        );
        let mut traversal_stats_after_selected_surface = traversal_stats_after_initial;
        let selected_surface_writes =
            graph.add_pass("vpt_surface_selected", QueueType::Compute, |builder| {
                builder.read_as(uniform_resource, AccessKind::ComputeShaderRead);
                builder.read_as(selected_resource, AccessKind::ComputeShaderRead);
                bootstrap_surface_writes.for_each(|surface_write| {
                    builder.write_as(surface_write, AccessKind::ComputeShaderWrite);
                });
                if let Some(traversal_stats_resource) = traversal_stats_after_initial {
                    traversal_stats_after_selected_surface = Some(
                        builder
                            .write_as(traversal_stats_resource, AccessKind::ComputeShaderReadWrite),
                    );
                }
                Box::new(move |ctx| {
                    if let Some(profiler) = profiler {
                        profiler.begin_scope(
                            ctx.device,
                            ctx.command_buffer,
                            frame_slot,
                            GpuProfileScope::VptSurfaceSelected,
                        );
                    }
                    vpt_surface.record_selected(ctx.device, ctx.command_buffer, frame_slot);
                    if let Some(profiler) = profiler {
                        profiler.end_scope(
                            ctx.device,
                            ctx.command_buffer,
                            frame_slot,
                            GpuProfileScope::VptSurfaceSelected,
                        );
                    }
                })
            });

        let final_surface_writes = VptCurrentSurfaceResources::from_graph_writes(
            selected_surface_writes.iter().copied().take(9).collect(),
        );

        AreaRestirGraphBuffers {
            uniform_buffer,
            uniform_resource,
            selected_current_buffer,
            selected_current_resource: selected_resource,
            final_surface_writes,
            traversal_stats_resource: traversal_stats_after_selected_surface,
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
            .write_image_descriptors(device, 2, &current_surface_images);
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
        let resized = AreaRestirResizeResources::new(
            device,
            allocator,
            width,
            height,
            self.selected_reservoirs.len(),
        )?;

        std::mem::replace(&mut self.initial_reservoirs, resized.initial_reservoirs)
            .destroy(device, allocator);
        std::mem::replace(&mut self.temporal_reservoirs, resized.temporal_reservoirs)
            .destroy(device, allocator);
        destroy_buffers(
            std::mem::replace(&mut self.selected_reservoirs, resized.selected_reservoirs),
            device,
            allocator,
        );
        std::mem::replace(&mut self.debug_image, resized.debug_image).destroy(device, allocator);

        self.width = width;
        self.height = height;
        self.reservoir_count = reservoir_count;
        self.write_descriptor_sets(device);
        Ok(())
    }

    pub fn update_scene_descriptors(&self, device: &ash::Device, scene_ubo: &SceneUniformBuffer) {
        self.write_scene_descriptors(device, scene_ubo);
    }

    pub fn update_ucvh_descriptors(&self, device: &ash::Device, ucvh_gpu: &UcvhGpuResources) {
        self.write_ucvh_descriptors(device, ucvh_gpu);
    }

    pub fn uniform_buffer(
        &self,
        frame_slot: usize,
    ) -> (&GpuBuffer, vk::DeviceSize, vk::BufferUsageFlags) {
        let buffer = &self.uniform_buffers[frame_slot];
        (buffer, buffer.size, buffer.usage)
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
            1,
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

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.initial_stage.destroy(device);
        self.temporal_stage.destroy(device);
        self.spatial_stage.destroy(device);
        destroy_buffers(self.uniform_buffers, device, allocator);
        self.initial_reservoirs.destroy(device, allocator);
        self.temporal_reservoirs.destroy(device, allocator);
        destroy_buffers(self.selected_reservoirs, device, allocator);
        self.debug_image.destroy(device, allocator);
    }

    fn write_descriptor_sets(&self, device: &ash::Device) {
        self.initial_stage.write_buffer_descriptors(
            device,
            &self.uniform_buffers,
            &[&self.initial_reservoirs],
        );
        self.initial_stage
            .write_image_descriptors(device, 5, &[&self.debug_image]);
        self.temporal_stage.write_buffer_descriptors(
            device,
            &self.uniform_buffers,
            &[
                &self.initial_reservoirs,
                &self.selected_reservoirs[self.selected_history_slot(0)],
                &self.temporal_reservoirs,
            ],
        );
        self.spatial_stage.write_buffer_descriptors(
            device,
            &self.uniform_buffers,
            &[
                &self.temporal_reservoirs,
                &self.selected_reservoirs[self.selected_current_slot(0)],
            ],
        );
        self.spatial_stage
            .write_image_descriptors(device, 6, &[&self.debug_image]);
    }

    fn selected_current_slot(&self, frame_slot: usize) -> usize {
        frame_slot % self.selected_reservoirs.len()
    }

    fn selected_history_slot(&self, frame_slot: usize) -> usize {
        (self.selected_current_slot(frame_slot) + self.selected_reservoirs.len() - 1)
            % self.selected_reservoirs.len()
    }

    fn write_scene_descriptors(&self, device: &ash::Device, scene_ubo: &SceneUniformBuffer) {
        self.initial_stage.write_scene_uniform_descriptors(
            device,
            6,
            scene_ubo,
            std::mem::size_of::<GpuSceneUniforms>() as u64,
        );
    }

    fn write_ucvh_descriptors(&self, device: &ash::Device, ucvh_gpu: &UcvhGpuResources) {
        self.initial_stage
            .write_uniform_buffer_descriptors(device, 7, &[&ucvh_gpu.config_buffer]);
        self.initial_stage.write_storage_buffer_descriptors(
            device,
            8,
            &[
                &ucvh_gpu.hierarchy_l0_buffer,
                &ucvh_gpu.hierarchy_ln_buffers[0],
                &ucvh_gpu.hierarchy_ln_buffers[1],
                &ucvh_gpu.hierarchy_ln_buffers[2],
                &ucvh_gpu.hierarchy_ln_buffers[3],
                &ucvh_gpu.occupancy_buffer,
                &ucvh_gpu.material_buffer,
            ],
        );
    }

    pub fn update_traversal_stats_descriptors(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        traversal_stats: &VptTraversalStatsBuffer,
    ) {
        let Some(&descriptor_set) = self.initial_stage.descriptor_sets.get(frame_slot) else {
            return;
        };
        let stats_info = vk::DescriptorBufferInfo::default()
            .buffer(traversal_stats.handle())
            .offset(0)
            .range(traversal_stats.size());
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(15)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&stats_info));
        unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
    }
}

impl AreaRestirBuffers {
    fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        frame_count: usize,
        reservoir_count: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let uniform_buffers = create_uniform_buffers(device, allocator, frame_count)?;
        let initial_reservoirs = match create_reservoir_buffer(
            device,
            allocator,
            reservoir_count,
            "area_restir_initial",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                destroy_buffers(uniform_buffers, device, allocator);
                return Err(error);
            }
        };
        let temporal_reservoirs = match create_reservoir_buffer(
            device,
            allocator,
            reservoir_count,
            "area_restir_temporal",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                initial_reservoirs.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                return Err(error);
            }
        };
        let selected_reservoirs = match create_selected_reservoir_buffers(
            device,
            allocator,
            frame_count,
            reservoir_count,
            "area_restir_selected",
        ) {
            Ok(buffers) => buffers,
            Err(error) => {
                temporal_reservoirs.destroy(device, allocator);
                initial_reservoirs.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                return Err(error);
            }
        };
        let debug_image = match create_debug_image(device, allocator, width, height) {
            Ok(image) => image,
            Err(error) => {
                destroy_buffers(selected_reservoirs, device, allocator);
                temporal_reservoirs.destroy(device, allocator);
                initial_reservoirs.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                return Err(error);
            }
        };

        Ok(Self {
            uniform_buffers,
            initial_reservoirs,
            temporal_reservoirs,
            selected_reservoirs,
            debug_image,
        })
    }

    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        destroy_buffers(self.uniform_buffers, device, allocator);
        self.initial_reservoirs.destroy(device, allocator);
        self.temporal_reservoirs.destroy(device, allocator);
        destroy_buffers(self.selected_reservoirs, device, allocator);
        self.debug_image.destroy(device, allocator);
    }
}

impl AreaRestirResizeResources {
    fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        selected_slot_count: usize,
    ) -> Result<Self> {
        let reservoir_count = width.saturating_mul(height);
        let initial_reservoirs =
            create_reservoir_buffer(device, allocator, reservoir_count, "area_restir_initial")?;
        let temporal_reservoirs =
            create_reservoir_buffer(device, allocator, reservoir_count, "area_restir_temporal")?;
        let selected_reservoirs = create_selected_reservoir_buffers(
            device,
            allocator,
            selected_slot_count,
            reservoir_count,
            "area_restir_selected",
        )?;
        let debug_image = match create_debug_image(device, allocator, width, height) {
            Ok(image) => image,
            Err(error) => {
                destroy_buffers(selected_reservoirs, device, allocator);
                temporal_reservoirs.destroy(device, allocator);
                initial_reservoirs.destroy(device, allocator);
                return Err(error);
            }
        };

        Ok(Self {
            initial_reservoirs,
            temporal_reservoirs,
            selected_reservoirs,
            debug_image,
        })
    }
}

impl AreaRestirStage {
    fn new(
        device: &ash::Device,
        spirv_bytes: &[u8],
        bindings: &[DescriptorBindingSpec],
        frame_count: usize,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding_specs(bindings)
            .build(device)?;
        let mut uniform_count = 0u32;
        let mut storage_buffer_count = 0u32;
        let mut storage_image_count = 0u32;
        for spec in bindings {
            match spec.descriptor_type {
                vk::DescriptorType::UNIFORM_BUFFER => uniform_count += frame_count as u32,
                vk::DescriptorType::STORAGE_BUFFER => storage_buffer_count += frame_count as u32,
                vk::DescriptorType::STORAGE_IMAGE => storage_image_count += frame_count as u32,
                _ => {}
            }
        }
        let mut pool_sizes = Vec::new();
        if uniform_count > 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: uniform_count,
            });
        }
        if storage_buffer_count > 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: storage_buffer_count,
            });
        }
        if storage_image_count > 0 {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: storage_image_count,
            });
        }
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

    fn write_buffer_descriptors(
        &self,
        device: &ash::Device,
        uniform_buffers: &[GpuBuffer],
        storage_buffers: &[&GpuBuffer],
    ) {
        for (set_idx, &ds) in self.descriptor_sets.iter().enumerate() {
            let ubo_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[set_idx].handle)
                .offset(0)
                .range(std::mem::size_of::<GpuAreaRestirUniforms>() as u64);
            let storage_infos: Vec<_> = storage_buffers
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
        let image_infos: Vec<_> = images
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

    fn write_scene_uniform_descriptors(
        &self,
        device: &ash::Device,
        binding: u32,
        scene_ubo: &SceneUniformBuffer,
        range: vk::DeviceSize,
    ) {
        for (set_idx, &ds) in self.descriptor_sets.iter().enumerate() {
            let info = vk::DescriptorBufferInfo::default()
                .buffer(scene_ubo.buffer_handle(set_idx))
                .offset(0)
                .range(range);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&info));
            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
        }
    }

    fn write_uniform_buffer_descriptors(
        &self,
        device: &ash::Device,
        first_binding: u32,
        buffers: &[&GpuBuffer],
    ) {
        let buffer_infos: Vec<_> = buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        for &ds in &self.descriptor_sets {
            let writes: Vec<_> = buffer_infos
                .iter()
                .enumerate()
                .map(|(idx, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(first_binding + idx as u32)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
    }

    fn write_storage_buffer_descriptors(
        &self,
        device: &ash::Device,
        first_binding: u32,
        buffers: &[&GpuBuffer],
    ) {
        let buffer_infos: Vec<_> = buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        for &ds in &self.descriptor_sets {
            let writes: Vec<_> = buffer_infos
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
            std::mem::size_of::<GpuAreaRestirUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("area_restir_uniforms_{slot}"),
        )?);
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
        (count * std::mem::size_of::<GpuAreaRestirReservoir>()) as u64,
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

fn create_debug_image(
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
            usage: vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "area_restir_debug",
        },
    )
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
mod shader_source_tests;
