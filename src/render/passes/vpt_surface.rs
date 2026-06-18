use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::area_restir::{GpuAreaRestirReservoir, GpuAreaRestirUniforms};
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorBindingSpec, DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
use crate::render::traversal_stats::VptTraversalStatsBuffer;
use crate::render::vpt_history::GpuVptHistoryUniforms;
use crate::voxel::gpu_upload::UcvhGpuResources;

pub struct VptSurfacePass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    bootstrap_descriptor_sets: Vec<vk::DescriptorSet>,
    selected_descriptor_sets: Vec<vk::DescriptorSet>,
    history_uniform_buffers: Vec<GpuBuffer>,
    disabled_area_restir_uniform_buffers: Vec<GpuBuffer>,
    disabled_area_restir_reservoir_buffer: GpuBuffer,
    pub surface_position_depth: GpuImage,
    pub surface_normal_roughness: GpuImage,
    pub surface_albedo_material: GpuImage,
    pub surface_material_roughness: GpuImage,
    pub surface_view_z: GpuImage,
    pub surface_motion_id: GpuImage,
    pub previous_surface_position_depth: GpuImage,
    pub previous_surface_normal_roughness: GpuImage,
    pub previous_surface_albedo_material: GpuImage,
    pub previous_surface_material_roughness: GpuImage,
    pub previous_surface_view_z: GpuImage,
    pub previous_surface_motion_id: GpuImage,
    pub motion_history: GpuImage,
    pub motion_flags: GpuImage,
    pub surface_brick_generation: GpuImage,
    pub previous_surface_brick_generation: GpuImage,
    motion_event_counts: Vec<u32>,
}

#[derive(Clone, Copy)]
pub struct VptSurfaceBootstrapGraph {
    pub surface_writes: VptCurrentSurfaceResources,
    pub previous_surface_resources: VptPreviousSurfaceResources,
    pub traversal_stats_resource: Option<ResourceHandle>,
}

#[derive(Clone, Copy)]
pub struct VptCurrentSurfaceResources {
    pub position_depth: ResourceHandle,
    pub normal_roughness: ResourceHandle,
    pub albedo_material: ResourceHandle,
    pub material_roughness: ResourceHandle,
    pub view_z: ResourceHandle,
    pub motion_id: ResourceHandle,
    pub motion_history: ResourceHandle,
    pub motion_flags: ResourceHandle,
    pub brick_generation: ResourceHandle,
}

impl VptCurrentSurfaceResources {
    pub fn from_graph_writes(writes: Vec<ResourceHandle>) -> Self {
        let mut writes = writes.into_iter();
        let resources = Self {
            position_depth: next_surface_write(&mut writes, "position_depth"),
            normal_roughness: next_surface_write(&mut writes, "normal_roughness"),
            albedo_material: next_surface_write(&mut writes, "albedo_material"),
            material_roughness: next_surface_write(&mut writes, "material_roughness"),
            view_z: next_surface_write(&mut writes, "view_z"),
            motion_id: next_surface_write(&mut writes, "motion_id"),
            motion_history: next_surface_write(&mut writes, "motion_history"),
            motion_flags: next_surface_write(&mut writes, "motion_flags"),
            brick_generation: next_surface_write(&mut writes, "brick_generation"),
        };
        assert!(
            writes.next().is_none(),
            "unexpected extra surface graph write"
        );
        resources
    }

    pub fn for_each(self, mut visit: impl FnMut(ResourceHandle)) {
        visit(self.position_depth);
        visit(self.normal_roughness);
        visit(self.albedo_material);
        visit(self.material_roughness);
        visit(self.view_z);
        visit(self.motion_id);
        visit(self.motion_history);
        visit(self.motion_flags);
        visit(self.brick_generation);
    }
}

fn next_surface_write(
    writes: &mut impl Iterator<Item = ResourceHandle>,
    name: &'static str,
) -> ResourceHandle {
    writes
        .next()
        .unwrap_or_else(|| panic!("missing {name} surface graph write"))
}

#[derive(Clone, Copy)]
pub struct VptPreviousSurfaceResources {
    pub position_depth: ResourceHandle,
    pub normal_roughness: ResourceHandle,
    pub albedo_material: ResourceHandle,
    pub material_roughness: ResourceHandle,
    pub view_z: ResourceHandle,
    pub motion_id: ResourceHandle,
    pub brick_generation: ResourceHandle,
}

pub struct VptSurfacePassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub spirv_bytes: &'a [u8],
    pub ucvh_gpu: &'a UcvhGpuResources,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub traversal_stats_buffers: &'a [VptTraversalStatsBuffer],
}

pub struct VptSurfacePassResizeInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub ucvh_gpu: &'a UcvhGpuResources,
    pub traversal_stats_buffers: &'a [VptTraversalStatsBuffer],
}

impl VptPreviousSurfaceResources {
    pub fn for_each(self, mut visit: impl FnMut(ResourceHandle)) {
        visit(self.position_depth);
        visit(self.normal_roughness);
        visit(self.albedo_material);
        visit(self.material_roughness);
        visit(self.view_z);
        visit(self.motion_id);
        visit(self.brick_generation);
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct VptSurfacePushConstants {
    motion_event_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl VptSurfacePass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 24] {
        [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(2, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(3, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(4, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(5, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(6, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(7, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(8, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(9, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(10, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(11, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(12, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(13, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(14, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(15, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(16, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(17, vk::DescriptorType::UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(18, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(19, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(20, vk::DescriptorType::STORAGE_IMAGE),
            DescriptorBindingSpec::compute(21, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(22, vk::DescriptorType::STORAGE_BUFFER),
            DescriptorBindingSpec::compute(23, vk::DescriptorType::STORAGE_BUFFER),
        ]
    }

    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptSurfacePassCreateInfo<'_>,
    ) -> Result<Self> {
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding_specs(&Self::descriptor_binding_specs())
            .build(device)?;

        let frame_count = info.scene_ubo.frame_count();
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 8 * frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 18 * frame_count as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 22 * frame_count as u32,
            },
        ];
        let descriptor_pool = match DescriptorPool::new(device, 2 * frame_count as u32, &pool_sizes)
        {
            Ok(pool) => pool,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let layouts: Vec<_> = (0..2 * frame_count)
            .map(|_| descriptor_set_layout)
            .collect();
        let mut descriptor_sets = match descriptor_pool.allocate(device, &layouts) {
            Ok(sets) => sets,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let selected_descriptor_sets = descriptor_sets.split_off(frame_count);
        let bootstrap_descriptor_sets = descriptor_sets;

        let images = match create_surface_images(device, allocator, info.width, info.height) {
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
                    images.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let disabled_area_restir_uniform_buffers =
            match create_disabled_area_restir_uniform_buffers(device, allocator, frame_count) {
                Ok(buffers) => buffers,
                Err(error) => {
                    destroy_buffers(history_uniform_buffers, device, allocator);
                    images.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        let disabled_area_restir_reservoir_buffer =
            match create_disabled_area_restir_reservoir_buffer(device, allocator) {
                Ok(buffer) => buffer,
                Err(error) => {
                    destroy_buffers(disabled_area_restir_uniform_buffers, device, allocator);
                    destroy_buffers(history_uniform_buffers, device, allocator);
                    images.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };

        write_descriptor_sets(
            device,
            &bootstrap_descriptor_sets,
            &images,
            VptSurfaceDescriptorRefs {
                scene_ubo: info.scene_ubo,
                ucvh_gpu: info.ucvh_gpu,
                history_uniform_buffers: &history_uniform_buffers,
                traversal_stats_buffers: info.traversal_stats_buffers,
                disabled_area_restir: DisabledAreaRestirRefs {
                    uniform_buffers: &disabled_area_restir_uniform_buffers,
                    reservoir_buffer: &disabled_area_restir_reservoir_buffer,
                },
            },
        );
        write_descriptor_sets(
            device,
            &selected_descriptor_sets,
            &images,
            VptSurfaceDescriptorRefs {
                scene_ubo: info.scene_ubo,
                ucvh_gpu: info.ucvh_gpu,
                history_uniform_buffers: &history_uniform_buffers,
                traversal_stats_buffers: info.traversal_stats_buffers,
                disabled_area_restir: DisabledAreaRestirRefs {
                    uniform_buffers: &disabled_area_restir_uniform_buffers,
                    reservoir_buffer: &disabled_area_restir_reservoir_buffer,
                },
            },
        );

        let shader_module = match create_shader_module(device, info.spirv_bytes) {
            Ok(module) => module,
            Err(error) => {
                disabled_area_restir_reservoir_buffer.destroy(device, allocator);
                destroy_buffers(disabled_area_restir_uniform_buffers, device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                images.destroy(device, allocator);
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
            &[vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: std::mem::size_of::<VptSurfacePushConstants>() as u32,
            }],
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe { device.destroy_shader_module(shader_module, None) };
                disabled_area_restir_reservoir_buffer.destroy(device, allocator);
                destroy_buffers(disabled_area_restir_uniform_buffers, device, allocator);
                destroy_buffers(history_uniform_buffers, device, allocator);
                images.destroy(device, allocator);
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
            bootstrap_descriptor_sets,
            selected_descriptor_sets,
            history_uniform_buffers,
            disabled_area_restir_uniform_buffers,
            disabled_area_restir_reservoir_buffer,
            surface_position_depth: images.surface_position_depth,
            surface_normal_roughness: images.surface_normal_roughness,
            surface_albedo_material: images.surface_albedo_material,
            surface_material_roughness: images.surface_material_roughness,
            surface_view_z: images.surface_view_z,
            surface_motion_id: images.surface_motion_id,
            previous_surface_position_depth: images.previous_surface_position_depth,
            previous_surface_normal_roughness: images.previous_surface_normal_roughness,
            previous_surface_albedo_material: images.previous_surface_albedo_material,
            previous_surface_material_roughness: images.previous_surface_material_roughness,
            previous_surface_view_z: images.previous_surface_view_z,
            previous_surface_motion_id: images.previous_surface_motion_id,
            motion_history: images.motion_history,
            motion_flags: images.motion_flags,
            surface_brick_generation: images.surface_brick_generation,
            previous_surface_brick_generation: images.previous_surface_brick_generation,
            motion_event_counts: vec![0; frame_count],
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptSurfacePassResizeInfo<'_>,
    ) -> Result<()> {
        let new_images = create_surface_images(device, allocator, info.width, info.height)?;
        let old_images = VptSurfaceImages {
            surface_position_depth: std::mem::replace(
                &mut self.surface_position_depth,
                new_images.surface_position_depth,
            ),
            surface_normal_roughness: std::mem::replace(
                &mut self.surface_normal_roughness,
                new_images.surface_normal_roughness,
            ),
            surface_albedo_material: std::mem::replace(
                &mut self.surface_albedo_material,
                new_images.surface_albedo_material,
            ),
            surface_material_roughness: std::mem::replace(
                &mut self.surface_material_roughness,
                new_images.surface_material_roughness,
            ),
            surface_view_z: std::mem::replace(&mut self.surface_view_z, new_images.surface_view_z),
            surface_motion_id: std::mem::replace(
                &mut self.surface_motion_id,
                new_images.surface_motion_id,
            ),
            previous_surface_position_depth: std::mem::replace(
                &mut self.previous_surface_position_depth,
                new_images.previous_surface_position_depth,
            ),
            previous_surface_normal_roughness: std::mem::replace(
                &mut self.previous_surface_normal_roughness,
                new_images.previous_surface_normal_roughness,
            ),
            previous_surface_albedo_material: std::mem::replace(
                &mut self.previous_surface_albedo_material,
                new_images.previous_surface_albedo_material,
            ),
            previous_surface_material_roughness: std::mem::replace(
                &mut self.previous_surface_material_roughness,
                new_images.previous_surface_material_roughness,
            ),
            previous_surface_view_z: std::mem::replace(
                &mut self.previous_surface_view_z,
                new_images.previous_surface_view_z,
            ),
            previous_surface_motion_id: std::mem::replace(
                &mut self.previous_surface_motion_id,
                new_images.previous_surface_motion_id,
            ),
            motion_history: std::mem::replace(&mut self.motion_history, new_images.motion_history),
            motion_flags: std::mem::replace(&mut self.motion_flags, new_images.motion_flags),
            surface_brick_generation: std::mem::replace(
                &mut self.surface_brick_generation,
                new_images.surface_brick_generation,
            ),
            previous_surface_brick_generation: std::mem::replace(
                &mut self.previous_surface_brick_generation,
                new_images.previous_surface_brick_generation,
            ),
        };
        old_images.destroy(device, allocator);

        let current_images = VptSurfaceImageRefs {
            surface_position_depth: &self.surface_position_depth,
            surface_normal_roughness: &self.surface_normal_roughness,
            surface_albedo_material: &self.surface_albedo_material,
            surface_material_roughness: &self.surface_material_roughness,
            surface_view_z: &self.surface_view_z,
            surface_motion_id: &self.surface_motion_id,
            motion_history: &self.motion_history,
            motion_flags: &self.motion_flags,
            surface_brick_generation: &self.surface_brick_generation,
        };
        write_descriptor_sets_from_refs(
            device,
            &self.bootstrap_descriptor_sets,
            current_images,
            VptSurfaceDescriptorRefs {
                scene_ubo: info.scene_ubo,
                ucvh_gpu: info.ucvh_gpu,
                history_uniform_buffers: &self.history_uniform_buffers,
                traversal_stats_buffers: info.traversal_stats_buffers,
                disabled_area_restir: DisabledAreaRestirRefs {
                    uniform_buffers: &self.disabled_area_restir_uniform_buffers,
                    reservoir_buffer: &self.disabled_area_restir_reservoir_buffer,
                },
            },
        );
        write_descriptor_sets_from_refs(
            device,
            &self.selected_descriptor_sets,
            current_images,
            VptSurfaceDescriptorRefs {
                scene_ubo: info.scene_ubo,
                ucvh_gpu: info.ucvh_gpu,
                history_uniform_buffers: &self.history_uniform_buffers,
                traversal_stats_buffers: info.traversal_stats_buffers,
                disabled_area_restir: DisabledAreaRestirRefs {
                    uniform_buffers: &self.disabled_area_restir_uniform_buffers,
                    reservoir_buffer: &self.disabled_area_restir_reservoir_buffer,
                },
            },
        );
        Ok(())
    }

    pub fn update_history_uniforms(&self, frame_slot: usize, uniforms: &GpuVptHistoryUniforms) {
        write_mapped(
            self.history_uniform_buffers[frame_slot].mapped_ptr(),
            uniforms,
        );
    }

    pub fn update_motion_guide_state(&mut self, frame_slot: usize, motion_event_count: u32) {
        self.motion_event_counts[frame_slot] = motion_event_count;
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        self.record_bootstrap(device, cmd, frame_slot);
    }

    pub fn update_area_restir_descriptors(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        area_uniform_buffer: &GpuBuffer,
        area_reservoir_buffer: &GpuBuffer,
    ) {
        write_area_restir_descriptor_set(
            device,
            self.selected_descriptor_sets[frame_slot],
            area_uniform_buffer,
            area_reservoir_buffer,
        );
    }

    pub fn update_traversal_stats_descriptor(
        &self,
        device: &ash::Device,
        frame_slot: usize,
        traversal_stats: &VptTraversalStatsBuffer,
    ) {
        let stats_info = vk::DescriptorBufferInfo::default()
            .buffer(traversal_stats.handle())
            .offset(0)
            .range(traversal_stats.size());
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.bootstrap_descriptor_sets[frame_slot])
                .dst_binding(23)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&stats_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(self.selected_descriptor_sets[frame_slot])
                .dst_binding(23)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&stats_info)),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    pub fn record_bootstrap(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
    ) {
        self.record_with_descriptor_set(
            device,
            cmd,
            self.bootstrap_descriptor_sets[frame_slot],
            frame_slot,
        );
    }

    pub fn record_selected(&self, device: &ash::Device, cmd: vk::CommandBuffer, frame_slot: usize) {
        self.record_with_descriptor_set(
            device,
            cmd,
            self.selected_descriptor_sets[frame_slot],
            frame_slot,
        );
    }

    pub fn register_bootstrap_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        history_initialized: bool,
        frame_slot: usize,
        traversal_stats_resource: Option<ResourceHandle>,
        profiler: Option<&'a GpuProfiler>,
    ) -> VptSurfaceBootstrapGraph {
        let surface_position_resource = graph.import_image_with_access(
            self.surface_position_depth.handle,
            self.surface_position_depth.extent.width,
            self.surface_position_depth.extent.height,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );
        let surface_normal_resource = graph.import_image_with_access(
            self.surface_normal_roughness.handle,
            self.surface_normal_roughness.extent.width,
            self.surface_normal_roughness.extent.height,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );
        let surface_albedo_resource = graph.import_image_with_access(
            self.surface_albedo_material.handle,
            self.surface_albedo_material.extent.width,
            self.surface_albedo_material.extent.height,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );
        let surface_material_roughness_resource = graph.import_image_with_access(
            self.surface_material_roughness.handle,
            self.surface_material_roughness.extent.width,
            self.surface_material_roughness.extent.height,
            vk::Format::R16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );
        let surface_view_z_resource = graph.import_image_with_access(
            self.surface_view_z.handle,
            self.surface_view_z.extent.width,
            self.surface_view_z.extent.height,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );
        let surface_motion_id_resource = graph.import_image_with_access(
            self.surface_motion_id.handle,
            self.surface_motion_id.extent.width,
            self.surface_motion_id.extent.height,
            vk::Format::R32_UINT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );
        let motion_history_resource = graph.import_image_with_access(
            self.motion_history.handle,
            self.motion_history.extent.width,
            self.motion_history.extent.height,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );
        let motion_flags_resource = graph.import_image_with_access(
            self.motion_flags.handle,
            self.motion_flags.extent.width,
            self.motion_flags.extent.height,
            vk::Format::R32_UINT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );
        let surface_brick_generation_resource = graph.import_image_with_access(
            self.surface_brick_generation.handle,
            self.surface_brick_generation.extent.width,
            self.surface_brick_generation.extent.height,
            vk::Format::R32_UINT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );
        let previous_surface_access = if history_initialized {
            AccessKind::TransferWrite
        } else {
            AccessKind::Undefined
        };
        let previous_surface_position_resource = graph.import_image_with_access(
            self.previous_surface_position_depth.handle,
            self.previous_surface_position_depth.extent.width,
            self.previous_surface_position_depth.extent.height,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            previous_surface_access,
        );
        let previous_surface_normal_resource = graph.import_image_with_access(
            self.previous_surface_normal_roughness.handle,
            self.previous_surface_normal_roughness.extent.width,
            self.previous_surface_normal_roughness.extent.height,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            previous_surface_access,
        );
        let previous_surface_albedo_resource = graph.import_image_with_access(
            self.previous_surface_albedo_material.handle,
            self.previous_surface_albedo_material.extent.width,
            self.previous_surface_albedo_material.extent.height,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            previous_surface_access,
        );
        let previous_surface_material_roughness_resource = graph.import_image_with_access(
            self.previous_surface_material_roughness.handle,
            self.previous_surface_material_roughness.extent.width,
            self.previous_surface_material_roughness.extent.height,
            vk::Format::R16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            previous_surface_access,
        );
        let previous_surface_view_z_resource = graph.import_image_with_access(
            self.previous_surface_view_z.handle,
            self.previous_surface_view_z.extent.width,
            self.previous_surface_view_z.extent.height,
            vk::Format::R32_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            previous_surface_access,
        );
        let previous_surface_motion_id_resource = graph.import_image_with_access(
            self.previous_surface_motion_id.handle,
            self.previous_surface_motion_id.extent.width,
            self.previous_surface_motion_id.extent.height,
            vk::Format::R32_UINT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            previous_surface_access,
        );
        let previous_surface_brick_generation_resource = graph.import_image_with_access(
            self.previous_surface_brick_generation.handle,
            self.previous_surface_brick_generation.extent.width,
            self.previous_surface_brick_generation.extent.height,
            vk::Format::R32_UINT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            previous_surface_access,
        );

        let mut traversal_stats_after_bootstrap = traversal_stats_resource;
        let bootstrap_writes =
            graph.add_pass("vpt_surface_bootstrap", QueueType::Compute, |builder| {
                builder.write_as(surface_position_resource, AccessKind::ComputeShaderWrite);
                builder.write_as(surface_normal_resource, AccessKind::ComputeShaderWrite);
                builder.write_as(surface_albedo_resource, AccessKind::ComputeShaderWrite);
                builder.write_as(
                    surface_material_roughness_resource,
                    AccessKind::ComputeShaderWrite,
                );
                builder.write_as(surface_view_z_resource, AccessKind::ComputeShaderWrite);
                builder.write_as(surface_motion_id_resource, AccessKind::ComputeShaderWrite);
                builder.write_as(motion_history_resource, AccessKind::ComputeShaderWrite);
                builder.write_as(motion_flags_resource, AccessKind::ComputeShaderWrite);
                builder.write_as(
                    surface_brick_generation_resource,
                    AccessKind::ComputeShaderWrite,
                );
                if let Some(traversal_stats_resource) = traversal_stats_resource {
                    traversal_stats_after_bootstrap = Some(
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
                            GpuProfileScope::VptSurfaceBootstrap,
                        );
                    }
                    self.record_bootstrap(ctx.device, ctx.command_buffer, frame_slot);
                    if let Some(profiler) = profiler {
                        profiler.end_scope(
                            ctx.device,
                            ctx.command_buffer,
                            frame_slot,
                            GpuProfileScope::VptSurfaceBootstrap,
                        );
                    }
                })
            });
        let surface_writes = VptCurrentSurfaceResources::from_graph_writes(
            bootstrap_writes.iter().copied().take(9).collect(),
        );
        graph.bind_image(
            surface_writes.position_depth,
            self.surface_position_depth.handle,
        );
        graph.bind_image(
            surface_writes.normal_roughness,
            self.surface_normal_roughness.handle,
        );
        graph.bind_image(
            surface_writes.albedo_material,
            self.surface_albedo_material.handle,
        );
        graph.bind_image(
            surface_writes.material_roughness,
            self.surface_material_roughness.handle,
        );
        graph.bind_image(surface_writes.view_z, self.surface_view_z.handle);
        graph.bind_image(surface_writes.motion_id, self.surface_motion_id.handle);
        graph.bind_image(surface_writes.motion_history, self.motion_history.handle);
        graph.bind_image(surface_writes.motion_flags, self.motion_flags.handle);
        graph.bind_image(
            surface_writes.brick_generation,
            self.surface_brick_generation.handle,
        );

        VptSurfaceBootstrapGraph {
            surface_writes,
            previous_surface_resources: VptPreviousSurfaceResources {
                position_depth: previous_surface_position_resource,
                normal_roughness: previous_surface_normal_resource,
                albedo_material: previous_surface_albedo_resource,
                material_roughness: previous_surface_material_roughness_resource,
                view_z: previous_surface_view_z_resource,
                motion_id: previous_surface_motion_id_resource,
                brick_generation: previous_surface_brick_generation_resource,
            },
            traversal_stats_resource: traversal_stats_after_bootstrap,
        }
    }

    fn record_with_descriptor_set(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        descriptor_set: vk::DescriptorSet,
        frame_slot: usize,
    ) {
        let extent = self.surface_position_depth.extent;
        let push_constants = VptSurfacePushConstants {
            motion_event_count: self.motion_event_counts[frame_slot],
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push_constants),
            );
            device.cmd_dispatch(cmd, extent.width.div_ceil(8), extent.height.div_ceil(8), 1);
        }
    }

    pub fn record_history_update(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        copy_surface_image(
            device,
            cmd,
            &self.surface_position_depth,
            &self.previous_surface_position_depth,
        );
        copy_surface_image(
            device,
            cmd,
            &self.surface_normal_roughness,
            &self.previous_surface_normal_roughness,
        );
        copy_surface_image(
            device,
            cmd,
            &self.surface_albedo_material,
            &self.previous_surface_albedo_material,
        );
        copy_surface_image(
            device,
            cmd,
            &self.surface_material_roughness,
            &self.previous_surface_material_roughness,
        );
        copy_surface_image(
            device,
            cmd,
            &self.surface_view_z,
            &self.previous_surface_view_z,
        );
        copy_surface_image(
            device,
            cmd,
            &self.surface_motion_id,
            &self.previous_surface_motion_id,
        );
        copy_surface_image(
            device,
            cmd,
            &self.surface_brick_generation,
            &self.previous_surface_brick_generation,
        );
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        destroy_buffers(self.history_uniform_buffers, device, allocator);
        destroy_buffers(self.disabled_area_restir_uniform_buffers, device, allocator);
        self.disabled_area_restir_reservoir_buffer
            .destroy(device, allocator);
        self.surface_position_depth.destroy(device, allocator);
        self.surface_normal_roughness.destroy(device, allocator);
        self.surface_albedo_material.destroy(device, allocator);
        self.surface_material_roughness.destroy(device, allocator);
        self.surface_view_z.destroy(device, allocator);
        self.surface_motion_id.destroy(device, allocator);
        self.previous_surface_position_depth
            .destroy(device, allocator);
        self.previous_surface_normal_roughness
            .destroy(device, allocator);
        self.previous_surface_albedo_material
            .destroy(device, allocator);
        self.previous_surface_material_roughness
            .destroy(device, allocator);
        self.previous_surface_view_z.destroy(device, allocator);
        self.previous_surface_motion_id.destroy(device, allocator);
        self.motion_history.destroy(device, allocator);
        self.motion_flags.destroy(device, allocator);
        self.surface_brick_generation.destroy(device, allocator);
        self.previous_surface_brick_generation
            .destroy(device, allocator);
    }
}

fn copy_surface_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src: &GpuImage,
    dst: &GpuImage,
) {
    let extent = src.extent;
    let region = vk::ImageCopy::default()
        .src_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .dst_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1),
        )
        .extent(extent);

    unsafe {
        device.cmd_copy_image(
            cmd,
            src.handle,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst.handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }
}

struct VptSurfaceImages {
    surface_position_depth: GpuImage,
    surface_normal_roughness: GpuImage,
    surface_albedo_material: GpuImage,
    surface_material_roughness: GpuImage,
    surface_view_z: GpuImage,
    surface_motion_id: GpuImage,
    previous_surface_position_depth: GpuImage,
    previous_surface_normal_roughness: GpuImage,
    previous_surface_albedo_material: GpuImage,
    previous_surface_material_roughness: GpuImage,
    previous_surface_view_z: GpuImage,
    previous_surface_motion_id: GpuImage,
    motion_history: GpuImage,
    motion_flags: GpuImage,
    surface_brick_generation: GpuImage,
    previous_surface_brick_generation: GpuImage,
}

#[derive(Clone, Copy)]
struct VptSurfaceImageRefs<'a> {
    surface_position_depth: &'a GpuImage,
    surface_normal_roughness: &'a GpuImage,
    surface_albedo_material: &'a GpuImage,
    surface_material_roughness: &'a GpuImage,
    surface_view_z: &'a GpuImage,
    surface_motion_id: &'a GpuImage,
    motion_history: &'a GpuImage,
    motion_flags: &'a GpuImage,
    surface_brick_generation: &'a GpuImage,
}

#[derive(Clone, Copy)]
struct DisabledAreaRestirRefs<'a> {
    uniform_buffers: &'a [GpuBuffer],
    reservoir_buffer: &'a GpuBuffer,
}

#[derive(Clone, Copy)]
struct VptSurfaceDescriptorRefs<'a> {
    scene_ubo: &'a SceneUniformBuffer,
    ucvh_gpu: &'a UcvhGpuResources,
    history_uniform_buffers: &'a [GpuBuffer],
    traversal_stats_buffers: &'a [VptTraversalStatsBuffer],
    disabled_area_restir: DisabledAreaRestirRefs<'a>,
}

impl VptSurfaceImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.surface_position_depth.destroy(device, allocator);
        self.surface_normal_roughness.destroy(device, allocator);
        self.surface_albedo_material.destroy(device, allocator);
        self.surface_material_roughness.destroy(device, allocator);
        self.surface_view_z.destroy(device, allocator);
        self.surface_motion_id.destroy(device, allocator);
        self.previous_surface_position_depth
            .destroy(device, allocator);
        self.previous_surface_normal_roughness
            .destroy(device, allocator);
        self.previous_surface_albedo_material
            .destroy(device, allocator);
        self.previous_surface_material_roughness
            .destroy(device, allocator);
        self.previous_surface_view_z.destroy(device, allocator);
        self.previous_surface_motion_id.destroy(device, allocator);
        self.motion_history.destroy(device, allocator);
        self.motion_flags.destroy(device, allocator);
        self.surface_brick_generation.destroy(device, allocator);
        self.previous_surface_brick_generation
            .destroy(device, allocator);
    }
}

fn create_surface_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptSurfaceImages> {
    let surface_position_depth = create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_surface_position_depth",
    )?;
    let surface_normal_roughness = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_surface_normal_roughness",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let surface_albedo_material = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_surface_albedo_material",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let surface_material_roughness = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R16_SFLOAT,
        "vpt_surface_material_roughness",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let surface_view_z = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32_SFLOAT,
        "vpt_surface_view_z",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let surface_motion_id = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32_UINT,
        "vpt_surface_motion_id",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let motion_history = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_motion_history",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let motion_flags = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32_UINT,
        "vpt_motion_flags",
    ) {
        Ok(image) => image,
        Err(error) => {
            motion_history.destroy(device, allocator);
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let surface_brick_generation = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32_UINT,
        "vpt_surface_brick_generation",
    ) {
        Ok(image) => image,
        Err(error) => {
            motion_flags.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_position_depth = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_previous_surface_position_depth",
    ) {
        Ok(image) => image,
        Err(error) => {
            surface_brick_generation.destroy(device, allocator);
            motion_flags.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_normal_roughness = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_previous_surface_normal_roughness",
    ) {
        Ok(image) => image,
        Err(error) => {
            previous_surface_position_depth.destroy(device, allocator);
            surface_brick_generation.destroy(device, allocator);
            motion_flags.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_albedo_material = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32G32B32A32_SFLOAT,
        "vpt_previous_surface_albedo_material",
    ) {
        Ok(image) => image,
        Err(error) => {
            previous_surface_normal_roughness.destroy(device, allocator);
            previous_surface_position_depth.destroy(device, allocator);
            surface_brick_generation.destroy(device, allocator);
            motion_flags.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_material_roughness = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R16_SFLOAT,
        "vpt_previous_surface_material_roughness",
    ) {
        Ok(image) => image,
        Err(error) => {
            previous_surface_albedo_material.destroy(device, allocator);
            previous_surface_normal_roughness.destroy(device, allocator);
            previous_surface_position_depth.destroy(device, allocator);
            surface_brick_generation.destroy(device, allocator);
            motion_flags.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_view_z = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32_SFLOAT,
        "vpt_previous_surface_view_z",
    ) {
        Ok(image) => image,
        Err(error) => {
            previous_surface_material_roughness.destroy(device, allocator);
            previous_surface_albedo_material.destroy(device, allocator);
            previous_surface_normal_roughness.destroy(device, allocator);
            previous_surface_position_depth.destroy(device, allocator);
            surface_brick_generation.destroy(device, allocator);
            motion_flags.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_motion_id = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32_UINT,
        "vpt_previous_surface_motion_id",
    ) {
        Ok(image) => image,
        Err(error) => {
            previous_surface_view_z.destroy(device, allocator);
            previous_surface_material_roughness.destroy(device, allocator);
            previous_surface_albedo_material.destroy(device, allocator);
            previous_surface_normal_roughness.destroy(device, allocator);
            previous_surface_position_depth.destroy(device, allocator);
            surface_brick_generation.destroy(device, allocator);
            motion_flags.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };
    let previous_surface_brick_generation = match create_surface_image(
        device,
        allocator,
        width,
        height,
        vk::Format::R32_UINT,
        "vpt_previous_surface_brick_generation",
    ) {
        Ok(image) => image,
        Err(error) => {
            previous_surface_motion_id.destroy(device, allocator);
            previous_surface_view_z.destroy(device, allocator);
            previous_surface_material_roughness.destroy(device, allocator);
            previous_surface_albedo_material.destroy(device, allocator);
            previous_surface_normal_roughness.destroy(device, allocator);
            previous_surface_position_depth.destroy(device, allocator);
            surface_brick_generation.destroy(device, allocator);
            motion_flags.destroy(device, allocator);
            motion_history.destroy(device, allocator);
            surface_motion_id.destroy(device, allocator);
            surface_view_z.destroy(device, allocator);
            surface_material_roughness.destroy(device, allocator);
            surface_albedo_material.destroy(device, allocator);
            surface_normal_roughness.destroy(device, allocator);
            surface_position_depth.destroy(device, allocator);
            return Err(error);
        }
    };

    Ok(VptSurfaceImages {
        surface_position_depth,
        surface_normal_roughness,
        surface_albedo_material,
        surface_material_roughness,
        surface_view_z,
        surface_motion_id,
        previous_surface_position_depth,
        previous_surface_normal_roughness,
        previous_surface_albedo_material,
        previous_surface_material_roughness,
        previous_surface_view_z,
        previous_surface_motion_id,
        motion_history,
        motion_flags,
        surface_brick_generation,
        previous_surface_brick_generation,
    })
}

fn create_surface_image(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
    format: vk::Format,
    name: &'static str,
) -> Result<GpuImage> {
    GpuImage::new(
        device,
        allocator,
        &GpuImageDesc {
            width,
            height,
            depth: 1,
            format,
            usage: surface_image_usage(),
            aspect: vk::ImageAspectFlags::COLOR,
            name,
        },
    )
}

fn surface_image_usage() -> vk::ImageUsageFlags {
    vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
}

fn write_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    images: &VptSurfaceImages,
    refs: VptSurfaceDescriptorRefs<'_>,
) {
    write_descriptor_sets_from_refs(
        device,
        descriptor_sets,
        VptSurfaceImageRefs {
            surface_position_depth: &images.surface_position_depth,
            surface_normal_roughness: &images.surface_normal_roughness,
            surface_albedo_material: &images.surface_albedo_material,
            surface_material_roughness: &images.surface_material_roughness,
            surface_view_z: &images.surface_view_z,
            surface_motion_id: &images.surface_motion_id,
            motion_history: &images.motion_history,
            motion_flags: &images.motion_flags,
            surface_brick_generation: &images.surface_brick_generation,
        },
        VptSurfaceDescriptorRefs { ..refs },
    );
}

fn write_descriptor_sets_from_refs(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    images: VptSurfaceImageRefs<'_>,
    refs: VptSurfaceDescriptorRefs<'_>,
) {
    let output_images = [
        images.surface_position_depth,
        images.surface_normal_roughness,
        images.surface_albedo_material,
        images.surface_material_roughness,
        images.surface_view_z,
        images.surface_motion_id,
        images.motion_history,
    ];
    let motion_guide_images = [images.motion_flags, images.surface_brick_generation];
    let ucvh_buffers = [
        &refs.ucvh_gpu.hierarchy_l0_buffer,
        &refs.ucvh_gpu.hierarchy_ln_buffers[0],
        &refs.ucvh_gpu.hierarchy_ln_buffers[1],
        &refs.ucvh_gpu.hierarchy_ln_buffers[2],
        &refs.ucvh_gpu.hierarchy_ln_buffers[3],
        &refs.ucvh_gpu.occupancy_buffer,
        &refs.ucvh_gpu.material_buffer,
    ];
    let ucvh_config_info = vk::DescriptorBufferInfo::default()
        .buffer(refs.ucvh_gpu.config_buffer.handle)
        .offset(0)
        .range(vk::WHOLE_SIZE);
    let motion_guide_buffers = [
        &refs.ucvh_gpu.brick_generation_buffer,
        &refs.ucvh_gpu.motion_event_buffer,
    ];

    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        let ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(refs.scene_ubo.buffer_handle(set_idx))
            .offset(0)
            .range(std::mem::size_of::<GpuSceneUniforms>() as u64);
        let history_ubo_info = vk::DescriptorBufferInfo::default()
            .buffer(refs.history_uniform_buffers[set_idx].handle)
            .offset(0)
            .range(std::mem::size_of::<GpuVptHistoryUniforms>() as u64);
        let image_infos: Vec<vk::DescriptorImageInfo> = output_images
            .iter()
            .map(|image| {
                vk::DescriptorImageInfo::default()
                    .image_view(image.view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect();
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = ucvh_buffers
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
        writes.extend(image_infos.iter().enumerate().map(|(idx, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding((idx + 1) as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(std::slice::from_ref(info))
        }));
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&ucvh_config_info)),
        );
        writes.extend(buffer_infos.iter().enumerate().map(|(idx, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding((idx + 9) as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        }));
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(16)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&history_ubo_info)),
        );
        let motion_guide_image_infos: Vec<_> = motion_guide_images
            .iter()
            .map(|image| {
                vk::DescriptorImageInfo::default()
                    .image_view(image.view)
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect();
        writes.extend(
            motion_guide_image_infos
                .iter()
                .enumerate()
                .map(|(idx, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(19 + idx as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(std::slice::from_ref(info))
                }),
        );
        let motion_guide_buffer_infos: Vec<_> = motion_guide_buffers
            .iter()
            .map(|buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();
        writes.extend(
            motion_guide_buffer_infos
                .iter()
                .enumerate()
                .map(|(idx, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(21 + idx as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                }),
        );
        let stats_buffer = &refs.traversal_stats_buffers[set_idx];
        let stats_info = vk::DescriptorBufferInfo::default()
            .buffer(stats_buffer.handle())
            .offset(0)
            .range(stats_buffer.size());
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(23)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&stats_info)),
        );
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
    write_area_restir_descriptor_sets(
        device,
        descriptor_sets,
        refs.disabled_area_restir.uniform_buffers,
        refs.disabled_area_restir.reservoir_buffer,
    );
}

fn write_area_restir_descriptor_set(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    area_uniform_buffer: &GpuBuffer,
    area_reservoir_buffer: &GpuBuffer,
) {
    let area_uniform_info = vk::DescriptorBufferInfo::default()
        .buffer(area_uniform_buffer.handle)
        .offset(0)
        .range(std::mem::size_of::<GpuAreaRestirUniforms>() as u64);
    let area_reservoir_info = vk::DescriptorBufferInfo::default()
        .buffer(area_reservoir_buffer.handle)
        .offset(0)
        .range(vk::WHOLE_SIZE);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(17)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&area_uniform_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(18)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&area_reservoir_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn write_area_restir_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    disabled_area_restir_uniform_buffers: &[GpuBuffer],
    disabled_area_restir_reservoir_buffer: &GpuBuffer,
) {
    for (set_idx, &ds) in descriptor_sets.iter().enumerate() {
        write_area_restir_descriptor_set(
            device,
            ds,
            &disabled_area_restir_uniform_buffers[set_idx],
            disabled_area_restir_reservoir_buffer,
        );
    }
}

fn create_disabled_area_restir_uniform_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count);
    for slot in 0..frame_count {
        let buffer = match GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GpuAreaRestirUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("vpt_surface_disabled_area_restir_uniforms_{slot}"),
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                destroy_buffers(buffers, device, allocator);
                return Err(error);
            }
        };
        let uniforms = GpuAreaRestirUniforms::zeroed();
        write_mapped(buffer.mapped_ptr(), &uniforms);
        buffers.push(buffer);
    }
    Ok(buffers)
}

fn create_disabled_area_restir_reservoir_buffer(
    device: &ash::Device,
    allocator: &GpuAllocator,
) -> Result<GpuBuffer> {
    let buffer = GpuBuffer::new(
        device,
        allocator,
        std::mem::size_of::<GpuAreaRestirReservoir>() as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        "vpt_surface_disabled_area_restir_reservoir",
    )?;
    let mut invalid_reservoir = GpuAreaRestirReservoir::zeroed();
    invalid_reservoir.sample_state.subpixel_uv = [-1.0, -1.0];
    invalid_reservoir.sample_state.lens_uv = [-1.0, -1.0];
    invalid_reservoir.sample_state.path_sample = u32::MAX;
    write_mapped(buffer.mapped_ptr(), &invalid_reservoir);
    Ok(buffer)
}

fn create_history_uniform_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count);
    for slot in 0..frame_count {
        buffers.push(GpuBuffer::new(
            device,
            allocator,
            std::mem::size_of::<GpuVptHistoryUniforms>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            &format!("vpt_history_uniforms_{slot}"),
        )?);
    }
    Ok(buffers)
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
    use crate::assets::shader_reflect::ShaderReflection;

    use super::*;

    fn dummy_surface_graph_writes(count: u32) -> Vec<ResourceHandle> {
        (0..count)
            .map(|id| ResourceHandle { id, version: 0 })
            .collect()
    }

    #[test]
    fn current_surface_resources_reject_extra_graph_writes() {
        let result = std::panic::catch_unwind(|| {
            VptCurrentSurfaceResources::from_graph_writes(dummy_surface_graph_writes(10));
        });

        assert!(
            result.is_err(),
            "extra surface graph writes must be rejected in release builds"
        );
    }

    #[test]
    fn current_surface_resources_reject_missing_graph_writes() {
        let result = std::panic::catch_unwind(|| {
            VptCurrentSurfaceResources::from_graph_writes(dummy_surface_graph_writes(8));
        });

        assert!(
            result.is_err(),
            "missing surface graph writes must be rejected"
        );
    }

    #[test]
    fn surface_images_are_sampleable_for_nrd_adapter_descriptors() {
        let usage = surface_image_usage();

        assert!(usage.contains(vk::ImageUsageFlags::STORAGE));
        assert!(usage.contains(vk::ImageUsageFlags::SAMPLED));
        assert!(usage.contains(vk::ImageUsageFlags::TRANSFER_SRC));
        assert!(usage.contains(vk::ImageUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn vpt_surface_descriptor_specs_match_shader_manifest() {
        let source_path = "assets/shaders/passes/vpt_surface.slang";
        let source = crate::render::source_checks::read_source(source_path);
        let reflection =
            ShaderReflection::from_slang_compiled_or_source("main", source_path, &source)
                .expect("shader reflection should parse");
        crate::render::descriptor::assert_specs_match_shader_bindings(
            "VPT surface",
            &VptSurfacePass::descriptor_binding_specs(),
            &reflection,
        );
    }
}
