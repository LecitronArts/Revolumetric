use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::area_restir::{
    AreaRestirSettings, GpuAreaRestirReservoir, GpuAreaRestirUniforms,
};
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::vpt_surface::VptSurfacePass;
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::scene_ubo::{GpuSceneUniforms, SceneUniformBuffer};
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

impl AreaRestirPass {
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
            &[
                (0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, vk::DescriptorType::STORAGE_BUFFER),
                (2, vk::DescriptorType::STORAGE_IMAGE),
                (3, vk::DescriptorType::STORAGE_IMAGE),
                (4, vk::DescriptorType::STORAGE_IMAGE),
                (5, vk::DescriptorType::STORAGE_IMAGE),
                (6, vk::DescriptorType::UNIFORM_BUFFER),
                (7, vk::DescriptorType::STORAGE_BUFFER),
                (8, vk::DescriptorType::STORAGE_BUFFER),
                (9, vk::DescriptorType::STORAGE_BUFFER),
                (10, vk::DescriptorType::STORAGE_BUFFER),
            ],
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
            &[
                (0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, vk::DescriptorType::STORAGE_BUFFER),
                (2, vk::DescriptorType::STORAGE_BUFFER),
                (3, vk::DescriptorType::STORAGE_BUFFER),
                (4, vk::DescriptorType::STORAGE_IMAGE),
                (5, vk::DescriptorType::STORAGE_IMAGE),
                (6, vk::DescriptorType::STORAGE_IMAGE),
                (7, vk::DescriptorType::STORAGE_IMAGE),
                (8, vk::DescriptorType::STORAGE_IMAGE),
                (9, vk::DescriptorType::STORAGE_IMAGE),
                (10, vk::DescriptorType::STORAGE_IMAGE),
            ],
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
            &[
                (0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, vk::DescriptorType::STORAGE_BUFFER),
                (2, vk::DescriptorType::STORAGE_BUFFER),
                (3, vk::DescriptorType::STORAGE_IMAGE),
                (4, vk::DescriptorType::STORAGE_IMAGE),
                (5, vk::DescriptorType::STORAGE_IMAGE),
                (6, vk::DescriptorType::STORAGE_IMAGE),
            ],
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
        self.initial_stage.write_storage_buffer_descriptors(
            device,
            7,
            &[
                &ucvh_gpu.config_buffer,
                &ucvh_gpu.hierarchy_l0_buffer,
                &ucvh_gpu.occupancy_buffer,
                &ucvh_gpu.material_buffer,
            ],
        );
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
        bindings: &[(u32, vk::DescriptorType)],
        frame_count: usize,
    ) -> Result<Self> {
        let mut builder = DescriptorLayoutBuilder::new();
        for &(binding, ty) in bindings {
            builder = builder.add_binding(binding, ty, vk::ShaderStageFlags::COMPUTE, 1);
        }
        let descriptor_set_layout = builder.build(device)?;
        let mut uniform_count = 0u32;
        let mut storage_buffer_count = 0u32;
        let mut storage_image_count = 0u32;
        for &(_, ty) in bindings {
            match ty {
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
mod shader_source_tests {
    fn source(path: &str) -> String {
        std::fs::read_to_string(path).expect("source should be readable")
    }

    #[test]
    fn area_restir_pass_declares_independent_resources_without_local_barriers() {
        let implementation = source("src/render/passes/area_restir.rs");
        let implementation = implementation
            .split("#[cfg(test)]")
            .next()
            .expect("implementation section should exist");

        for token in [
            "pub struct AreaRestirPass",
            "initial_reservoirs",
            "temporal_reservoirs",
            "selected_reservoirs",
            "debug_image",
            "MemoryLocation::GpuOnly",
            "resize_buffers",
            "update_surface_descriptors",
            "selected_current_buffer",
            "selected_history_buffer",
            "update_frame_descriptors",
            "record_initial",
            "record_temporal",
            "record_spatial",
            "update_ucvh_descriptors",
            "write_ucvh_descriptors",
        ] {
            assert!(
                implementation.contains(token),
                "Area ReSTIR pass missing token {token}"
            );
        }

        assert!(!implementation.contains("cmd_pipeline_barrier"));
        assert!(!implementation.contains("cmd_copy_buffer"));
        assert!(!implementation.contains("ImageMemoryBarrier"));
        assert!(!implementation.contains("BufferMemoryBarrier"));
        assert!(!implementation.contains("GpuRestirDiReservoir"));
        assert!(!implementation.contains("spatial_reservoirs"));
        assert!(!implementation.contains("history_reservoirs"));
        assert!(!implementation.contains("AreaRestirHistorySource"));
    }

    #[test]
    fn area_restir_resize_cleans_up_failed_resource_creation() {
        let implementation = source("src/render/passes/area_restir.rs");
        let implementation = implementation
            .split("#[cfg(test)]")
            .next()
            .expect("implementation section should exist");

        for token in [
            "struct AreaRestirResizeResources",
            "impl AreaRestirResizeResources",
            "fn new(",
            "fn destroy(self, device: &ash::Device, allocator: &GpuAllocator)",
            "let resized = AreaRestirResizeResources::new(",
            "std::mem::replace(&mut self.initial_reservoirs, resized.initial_reservoirs)",
            "std::mem::replace(&mut self.selected_reservoirs, resized.selected_reservoirs)",
            "std::mem::replace(&mut self.debug_image, resized.debug_image)",
        ] {
            assert!(
                implementation.contains(token),
                "Area ReSTIR resize path missing cleanup token {token}"
            );
        }
    }

    #[test]
    fn app_area_restir_dispatches_only_enabled_reuse_stages() {
        let app = source("src/app.rs");
        let compact = app.split_whitespace().collect::<String>();

        for token in [
            "letarea_temporal_active=area_restir_settings.temporal_enabled;",
            "letarea_spatial_active=area_temporal_active&&area_restir_settings.spatial_enabled;",
            "letarea_initial_output_resource=ifarea_temporal_active{area_initial_resource}else{area_selected_current_resource};",
            "letarea_temporal_dep=ifarea_temporal_active{",
            "graph.add_pass(\"area_restir_temporal\"",
            "}else{area_initial_dep};",
            "letarea_temporal_output_resource=ifarea_spatial_active{area_temporal_resource}else{area_selected_current_resource};",
            ")=ifarea_spatial_active{",
            "graph.add_pass(\"area_restir_spatial\"",
            "vpt.update_area_restir_descriptors(",
        ] {
            assert!(
                compact.contains(token),
                "app Area ReSTIR graph missing conditional-dispatch token {token}"
            );
        }
        assert!(
            compact.contains("if!history_initialized{settings.temporal_enabled=false;}")
                && compact.contains(
                    "settings.spatial_enabled=settings.temporal_enabled&&settings.spatial_enabled;"
                )
                && compact.contains(
                    "area_restir_effective_settings(self.area_restir_settings,self.area_restir_history_initialized"
                )
                && compact.contains(
                    "letarea_spatial_active=area_temporal_active&&area_restir_settings.spatial_enabled;"
                ),
            "effective settings must disable spatial reuse until temporal history is usable"
        );
    }

    #[test]
    fn app_uses_area_restir_selected_frame_ring_for_history_and_vpt_reads() {
        let app = source("src/app.rs");
        let compact = app.split_whitespace().collect::<String>();

        for token in [
            "fnarea_restir_effective_settings(settings:AreaRestirSettings,history_initialized:bool,)->AreaRestirSettings{",
            "area_restir.selected_current_buffer(",
            "area_restir.selected_history_buffer(",
            "area_selected_current_resource",
            "area_selected_history_resource",
            "area_restir_effective_settings(self.area_restir_settings,self.area_restir_history_initialized",
            "builder.read_as(area_selected_history_resource,",
            "builder.write_as(area_selected_current_resource,",
            "vpt_area_restir_reads=Some((area_uniform_resource,area_selected_reservoir_resource));",
        ] {
            assert!(
                compact.contains(token),
                "app Area ReSTIR selected frame-ring policy missing token {token}"
            );
        }
        assert!(
            !compact.contains("area_restir_selected_reservoir("),
            "Area ReSTIR final reservoir selection must not point VPT/history at intermediate buffers"
        );
        assert!(
            !app.contains("area_restir_history_update"),
            "Area ReSTIR graph must not add a transfer history update pass"
        );
    }

    #[test]
    fn area_restir_shaders_declare_expected_entry_points_and_resources() {
        let initial = source("assets/shaders/passes/area_restir_initial.slang");
        let temporal = source("assets/shaders/passes/area_restir_temporal.slang");
        let spatial = source("assets/shaders/passes/area_restir_spatial.slang");

        for shader in [&initial, &temporal, &spatial] {
            assert!(shader.contains("#include \"area_restir_common.slang\""));
            assert!(shader.contains("#include \"scene_common.slang\""));
            assert!(shader.contains("[shader(\"compute\")]"));
            assert!(shader.contains("AreaRestirUniforms"));
            assert!(shader.contains("AreaRestirReservoir"));
            assert!(!shader.contains("RestirDiReservoir"));
        }

        for token in [
            "RWStructuredBuffer<AreaRestirReservoir> output_reservoirs",
            "ConstantBuffer<SceneUniforms> scene_ubo",
            "StructuredBuffer<UcvhConfig> ucvh_config",
            "StructuredBuffer<NodeL0> hierarchy_l0",
            "StructuredBuffer<BrickOccupancy> brick_occupancy",
            "StructuredBuffer<VoxelCell> brick_materials",
            "area_restir_invalid_reservoir",
        ] {
            assert!(initial.contains(token), "initial shader missing {token}");
        }

        assert!(
            !initial.contains("scene_primary_ray_from_area_sample((SceneUniforms)0"),
            "initial shader must use real SceneUniforms instead of a zeroed placeholder"
        );

        for token in [
            "StructuredBuffer<AreaRestirReservoir> history_reservoirs",
            "motion_history",
            "area_restir_temporal_surface_compatible",
            "history_length",
        ] {
            assert!(temporal.contains(token), "temporal shader missing {token}");
        }

        for token in [
            "StructuredBuffer<AreaRestirReservoir> temporal_reservoirs",
            "spatial_sample_count",
            "spatial_radius",
            "neighbor_offsets",
            "area_restir_surface_compatible",
        ] {
            assert!(spatial.contains(token), "spatial shader missing {token}");
        }
    }

    #[test]
    fn area_restir_surface_inputs_use_storage_images_with_cached_reads() {
        let pass = source("src/render/passes/area_restir.rs");
        let pass = pass
            .split("#[cfg(test)]")
            .next()
            .expect("implementation section should exist");
        let initial = source("assets/shaders/passes/area_restir_initial.slang");
        let temporal = source("assets/shaders/passes/area_restir_temporal.slang");
        let spatial = source("assets/shaders/passes/area_restir_spatial.slang");

        for token in [
            "vk::DescriptorType::STORAGE_IMAGE",
            "storage_image_count",
            "write_image_descriptors",
        ] {
            assert!(
                pass.contains(token),
                "Area ReSTIR pass missing storage-image descriptor token {token}"
            );
        }
        assert!(
            !pass.contains("vk::DescriptorType::SAMPLED_IMAGE"),
            "Area ReSTIR sampled-image path regressed profiling and must not be reintroduced without new evidence"
        );

        for (name, shader) in [
            ("initial", initial.as_str()),
            ("temporal", temporal.as_str()),
            ("spatial", spatial.as_str()),
        ] {
            assert!(
                shader.contains("RWTexture2D<float4> surface_position_depth"),
                "{name} shader must bind surface_position_depth as storage image"
            );
            assert!(
                !shader.contains("\nTexture2D<float4> surface_position_depth"),
                "{name} shader must not use sampled surface inputs after profiling regression"
            );
            assert!(
                !shader.contains(".Load(int3("),
                "{name} shader must not use sampled texture Load for surface inputs"
            );
        }
        assert!(
            initial.contains("RWTexture2D<float4> area_restir_debug")
                && spatial.contains("RWTexture2D<float4> area_restir_debug"),
            "Area ReSTIR debug output remains a storage image"
        );
    }

    #[test]
    fn area_restir_shaders_cache_per_pixel_surface_reads() {
        let initial = source("assets/shaders/passes/area_restir_initial.slang");
        let temporal = source("assets/shaders/passes/area_restir_temporal.slang");

        for token in [
            "float4 center_position_depth = surface_position_depth[pixel];",
            "AreaRestirCandidateSurface center_surface = read_center_surface(pixel);",
            "AreaRestirCandidateSurface candidate_surface = evaluate_area_restir_candidate_surface(",
            "uint2 pixel,\n    AreaRestirSampleState sample_state",
            "evaluate_area_restir_candidate_surface(scene_ubo, pixel, sample_state)",
            "ScenePrimaryRay primary_ray = scene_primary_ray_from_area_sample(",
            "HitResult hit = trace_primary_ray(",
            "make_ray(primary_ray.origin, primary_ray.direction)",
            "float target_pdf = area_restir_candidate_target_pdf(center_surface, candidate_surface);",
            "float2 pixel_sample = area_restir_pixel_sample(pixel, sample_state);",
            "surface.position_depth = float4(hit.position, hit.t);",
        ] {
            assert!(
                initial.contains(token),
                "initial shader missing ray-evaluated candidate token {token}"
            );
        }
        assert!(
            initial.contains("#include \"voxel_traverse.slang\"")
                && initial.contains("#include \"material_common.slang\"")
                && initial.contains("StructuredBuffer<UcvhConfig> ucvh_config")
                && initial.contains("StructuredBuffer<NodeL0> hierarchy_l0")
                && initial.contains("StructuredBuffer<BrickOccupancy> brick_occupancy")
                && initial.contains("StructuredBuffer<VoxelCell> brick_materials"),
            "initial shader must bind UCVH resources to evaluate each area candidate ray"
        );
        assert!(
            !initial.contains("float target_pdf = target_luma;")
                && !initial.contains("float target_luma = surface_target_luma("),
            "initial shader must not assign every candidate the same center-surface target"
        );
        assert!(
            !initial.contains("float2 pixel_sample = float2(pixel) + sample_state.subpixel_uv;"),
            "Area ReSTIR subpixel_uv is stored in [0,1) and must be converted to a pixel-center-relative sample before tracing"
        );
        assert!(
            !initial.contains("sample_state.pixel_sample")
                && !initial.contains("reservoir.selected_radiance")
                && !initial.contains("distance(primary_ray.origin, hit.position)"),
            "initial shader must not preserve unused reservoir payload or recompute hit depth"
        );
        assert!(
            temporal.contains("float4 motion = center_context.motion_history;")
                && !temporal.contains("float4 motion = motion_history.Load(int3(pixel, 0));"),
            "temporal shader must reuse motion already loaded into center_context"
        );
    }

    #[test]
    fn area_restir_debug_writes_are_gated_by_debug_view() {
        let initial = source("assets/shaders/passes/area_restir_initial.slang");
        let spatial = source("assets/shaders/passes/area_restir_spatial.slang");

        for (name, shader) in [("initial", initial.as_str()), ("spatial", spatial.as_str())] {
            assert!(
                shader.contains("if (area_restir.debug_view != 0u)"),
                "{name} shader must not write the debug image on the default debug-off path"
            );
        }
    }

    #[test]
    fn area_restir_common_declares_replay_target_and_weighted_reservoir_update() {
        let common = source("assets/shaders/shared/area_restir_common.slang");

        for token in [
            "float area_restir_replay_target_pdf",
            "bool area_restir_candidate_finite",
            "float area_restir_reservoir_stream_weight",
            "float area_restir_reservoir_reuse_weight",
            "void area_restir_finalize_reservoir",
            "void area_restir_reservoir_update",
            "candidate_target_pdf",
            "candidate_stream_weight",
            "candidate_weight_sum",
            "float keep_ratio = capped_m / original_m;",
            "reservoir.weight_sum *= keep_ratio;",
        ] {
            assert!(
                common.contains(token),
                "Area ReSTIR common missing robust resampling token {token}"
            );
        }
    }

    #[test]
    fn area_restir_temporal_reuses_history_in_current_pixel_measure() {
        let temporal = source("assets/shaders/passes/area_restir_temporal.slang");
        let common = source("assets/shaders/shared/area_restir_common.slang");

        for token in [
            "float2 history_sample = motion.xy - 0.5;",
            "float2 history_fraction",
            "static const int2 area_restir_history_tap_offsets[4]",
            "static const float AREA_RESTIR_TEMPORAL_MIN_TAP_WEIGHT",
            "float area_restir_history_tap_weight",
            "bool area_restir_history_tap_inside",
            "for (uint tap = 0u; tap < 4u; tap++)",
            "int2 tap_pixel_i = previous_base_pixel + area_restir_history_tap_offsets[tap]",
            "float tap_weight = area_restir_history_tap_weight(tap, history_fraction)",
            "tap_weight < AREA_RESTIR_TEMPORAL_MIN_TAP_WEIGHT",
            "float center_target_luma = area_restir_context_target_luma(center_context);",
            "float current_target_pdf = center_target_luma;",
            "float history_target_pdf = center_target_luma;",
            "float history_candidate_weight_sum",
            "area_restir_reservoir_update(",
            "area_restir_temporal_surface_compatible(center_context, previous_pixel)",
            "AREA_RESTIR_SAMPLE_HISTORY_VALID",
            "history.rejection_reason = 0u",
            "history.debug_flags",
        ] {
            assert!(
                temporal.contains(token),
                "temporal shader missing current-measure reuse token {token}"
            );
        }

        assert!(
            !temporal.contains("float history_weight = history.weight_sum;"),
            "temporal reuse must not copy previous-frame raw weight_sum into the current pixel measure"
        );
        assert!(
            temporal.contains("area_restir_reservoir_reuse_weight(history)")
                && common.contains("float area_restir_reservoir_reuse_weight")
                && common.contains("reservoir.selected_weight")
                && !temporal.contains("area_restir_reservoir_stream_weight(history)"),
            "temporal reuse must derive bounded reuse weight from selected_weight, not weight_sum/M"
        );
        assert!(
            temporal.contains("area_restir_reservoir_reuse_weight(current)")
                && !temporal
                    .contains("current_target_pdf * float(max(current.sample_count_m, 1u))"),
            "temporal current reservoir must use the same bounded reuse-weight path as reused history"
        );
        assert!(
            !temporal.contains("int2 previous_pixel_i = int2(floor(history_sample));")
                && !temporal.contains("uint2 previous_pixel = uint2(previous_pixel_i);"),
            "temporal reuse must not collapse fractional reprojection to a single previous-pixel reservoir"
        );
        assert!(
            !temporal.contains(
                "area_restir_replay_target_pdf(current_context(pixel), history.sample_state"
            ),
            "temporal history taps must reuse the already-loaded center context instead of rereading current surface textures"
        );
        assert!(
            !temporal
                .contains("area_restir_replay_target_pdf(center_context, history.sample_state")
                && !temporal
                    .contains("area_restir_replay_target_pdf(center_context, current.sample_state"),
            "valid current/history reservoirs can reuse cached center target luma instead of repeating replay-domain checks"
        );
        assert!(
            !temporal.contains("if (combined > 0.0 && history_weight >= current_weight)"),
            "temporal reuse must use weighted reservoir update instead of max-weight replacement"
        );
    }

    #[test]
    fn area_restir_temporal_rejects_history_with_staged_previous_surface_reads() {
        let temporal = source("assets/shaders/passes/area_restir_temporal.slang");

        for token in [
            "bool area_restir_temporal_surface_compatible(",
            "float4 previous_position = previous_surface_position_depth[previous_pixel];",
            "if (center.position_depth.w < 0.0 || previous_position.w < 0.0)",
            "float position_delta = distance(center.position_depth.xyz, previous_position.xyz);",
            "float4 previous_albedo = previous_surface_albedo_material[previous_pixel];",
            "float3 previous_normal = normalize(previous_surface_normal_roughness[previous_pixel].xyz);",
        ] {
            assert!(
                temporal.contains(token),
                "temporal shader missing staged previous-surface token {token}"
            );
        }
        assert!(
            !temporal.contains("previous_context(previous_pixel)"),
            "temporal reuse should not construct a full previous context before cheap rejection tests"
        );
    }

    #[test]
    fn area_restir_spatial_reuses_neighbors_in_current_pixel_measure() {
        let spatial = source("assets/shaders/passes/area_restir_spatial.slang");
        let common = source("assets/shaders/shared/area_restir_common.slang");

        for token in [
            "area_restir_spatial_hash",
            "uint rotated_tap",
            "if (area_restir.enabled == 0u || area_restir.spatial_enabled == 0u || area_restir.spatial_sample_count == 0u",
            "float center_target_pdf = area_restir_context_target_luma(center);",
            "float neighbor_target_pdf = center_target_pdf;",
            "float neighbor_candidate_weight_sum",
            "area_restir_reservoir_update(",
            "reservoir.jacobian",
            "reservoir.debug_flags",
        ] {
            assert!(
                spatial.contains(token),
                "spatial shader missing current-measure reuse token {token}"
            );
        }

        assert!(
            !spatial.contains("reservoir.weight_sum += neighbor.weight_sum;"),
            "spatial reuse must not add raw neighbor weight_sum without current-domain conversion"
        );
        assert!(
            spatial.contains("area_restir_reservoir_reuse_weight(neighbor)")
                && common.contains("float area_restir_reservoir_reuse_weight")
                && common.contains("reservoir.selected_weight")
                && !spatial.contains("area_restir_reservoir_stream_weight(neighbor)"),
            "spatial reuse must derive bounded reuse weight from selected_weight, not weight_sum/M"
        );
        assert!(
            spatial.contains("area_restir_reservoir_reuse_weight(center_reservoir)")
                && !spatial.contains(
                    "center_target_pdf * float(max(center_reservoir.sample_count_m, 1u))"
                ),
            "spatial center reservoir must use the same bounded reuse-weight path as reused neighbors"
        );
        assert!(
            !spatial.contains("if (neighbor.weight_sum > reservoir.weight_sum)"),
            "spatial reuse must not select neighbors by stale raw weight_sum"
        );
        assert!(
            !spatial.contains("area_restir_replay_target_pdf(center, neighbor.sample_state"),
            "spatial reuse should not recompute a neighbor replay target when the current-pixel target is already known"
        );
        assert!(
            !spatial
                .contains("area_restir_replay_target_pdf(center, center_reservoir.sample_state"),
            "valid center reservoirs can reuse cached center target luma instead of repeating replay-domain checks"
        );
    }
}
