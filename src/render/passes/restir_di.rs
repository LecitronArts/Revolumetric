use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::GpuImage;
use crate::render::passes::vpt_surface::VptSurfacePass;
use crate::render::pipeline::{ComputePipeline, create_shader_module};
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

impl RestirDiPass {
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
            &[
                (0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, vk::DescriptorType::STORAGE_BUFFER),
                (2, vk::DescriptorType::STORAGE_BUFFER),
                (3, vk::DescriptorType::STORAGE_IMAGE),
                (4, vk::DescriptorType::STORAGE_IMAGE),
                (5, vk::DescriptorType::STORAGE_IMAGE),
            ],
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
                    descriptor_count: 8 * info.frame_count as u32,
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
            &[
                (0, vk::DescriptorType::UNIFORM_BUFFER),
                (1, vk::DescriptorType::STORAGE_BUFFER),
                (2, vk::DescriptorType::STORAGE_BUFFER),
                (3, vk::DescriptorType::STORAGE_IMAGE),
                (4, vk::DescriptorType::STORAGE_IMAGE),
                (5, vk::DescriptorType::STORAGE_IMAGE),
            ],
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
        bindings: &[(u32, vk::DescriptorType)],
        frame_count: usize,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self> {
        let mut builder = DescriptorLayoutBuilder::new();
        for &(binding, ty) in bindings {
            builder = builder.add_binding(binding, ty, vk::ShaderStageFlags::COMPUTE, 1);
        }
        let descriptor_set_layout = builder.build(device)?;
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
mod shader_source_tests {
    fn source(path: &str) -> String {
        std::fs::read_to_string(path).expect("shader source should be readable")
    }

    #[test]
    fn restir_di_shaders_declare_expected_entry_points_and_resources() {
        let initial = source("assets/shaders/passes/restir_di_initial.slang");
        let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");
        for shader in [&initial, &temporal, &spatial] {
            assert!(shader.contains("#include \"restir_di_common.slang\""));
            assert!(shader.contains("[shader(\"compute\")]"));
            assert!(shader.contains("RestirDiUniforms"));
            assert!(shader.contains("RestirDiReservoir"));
        }
        assert!(initial.contains("StructuredBuffer<DirectLight>"));
        assert!(temporal.contains("history_reservoirs"));
        assert!(spatial.contains("temporal_reservoirs"));
    }

    #[test]
    fn restir_di_initial_writes_valid_candidate_reservoirs() {
        let initial = source("assets/shaders/passes/restir_di_initial.slang");

        assert!(initial.contains("restir.light_count == 0u"));
        assert!(initial.contains("DirectLight light = direct_lights[light_id];"));
        assert!(initial.contains("sample_direct_light_id(rand01(rng_state))"));
        assert!(initial.contains("reservoir.sample_light_id = light_id;"));
        assert!(initial.contains("reservoir.sample_flags ="));
        assert!(initial.contains("reservoir.sample_count_m += 1u;"));
        assert!(initial.contains("reservoir.weight_sum ="));
        assert!(initial.contains("reservoir.selected_weight ="));
        assert!(initial.contains("reservoir.sample_position_pdf ="));
        assert!(initial.contains("reservoir.sample_radiance ="));
    }

    #[test]
    fn restir_di_initial_computes_selected_weight_from_total_candidate_weight() {
        let initial = source("assets/shaders/passes/restir_di_initial.slang");

        assert!(
            initial.contains("reservoir.selected_weight = min(")
                && initial.contains("reservoir.weight_sum /"),
            "direct lighting resolve consumes selected_weight, so initial RIS must compute it from the total candidate weight"
        );
        assert!(initial.contains("reservoir.target_pdf * float(reservoir.sample_count_m)"));
        assert!(
            initial.contains("reservoir.confidence = reservoir.weight_sum;"),
            "initial debug confidence should reflect the bounded reservoir weight, not the unclamped pre-firefly-clamp stream sum"
        );
    }

    #[test]
    fn restir_di_target_pdf_matches_vpt_direct_light_measure() {
        let common = source("assets/shaders/shared/restir_di_common.slang");

        for token in [
            "restir_di_emissive_distance_attenuation",
            "restir_di_sun_target_pdf",
            "restir_di_emissive_target_pdf",
            "distance_sq",
            "return light_power * albedo_luma * light_term * attenuation;",
        ] {
            assert!(
                common.contains(token),
                "ReSTIR target PDF must include the same cosine and attenuation measure as VPT direct resolve: {token}"
            );
        }
        assert!(
            !common.contains("max(dot(surface_normal, light_dir), 0.05)"),
            "target PDF must not assign positive probability to back-facing direct-light samples that resolve to zero"
        );
    }

    #[test]
    fn restir_di_initial_evaluates_sun_as_direction_not_world_position() {
        let initial = source("assets/shaders/passes/restir_di_initial.slang");

        assert!(
            initial.contains("restir_di_target_pdf_for_light_sample("),
            "initial sampling should use the shared light target PDF evaluator"
        );
        assert!(
            !initial
                .contains("light.position_radius.xyz - current_surface_position_depth[pixel].xyz"),
            "sun lights store a direction in position_radius.xyz, not a world-space point"
        );
    }

    #[test]
    fn restir_di_initial_samples_emissive_area_points_instead_of_centroids() {
        let initial = source("assets/shaders/passes/restir_di_initial.slang");
        let common = source("assets/shaders/shared/restir_di_common.slang");

        for token in [
            "restir_di_sample_emissive_area_point",
            "restir_di_emissive_area_pdf",
            "light.position_radius.w",
            "reservoir.sample_position_pdf = float4(sampled_position, sample_pdf)",
            "target_pdf = restir_di_target_pdf_for_light_sample(",
        ] {
            assert!(
                initial.contains(token) || common.contains(token),
                "Area ReSTIR direct-light candidate generation missing token {token}"
            );
        }

        assert!(
            !initial.contains("reservoir.sample_position_pdf = float4(light.position_radius.xyz, 1.0 / float(restir.light_count))"),
            "emissive reservoirs must store a sampled area point and its PDF, not the centroid"
        );
    }

    #[test]
    fn restir_di_initial_weights_candidates_by_light_selection_pdf() {
        let initial = source("assets/shaders/passes/restir_di_initial.slang");

        for token in [
            "uint sample_direct_light_id(float random01)",
            "while (lo < hi)",
            "direct_lights[mid].sampling.x",
            "float light_selection_pdf = max(light.sampling.y, 1.0e-8);",
            "float target_pdf = restir_di_target_pdf_for_light_sample(",
            "float candidate_weight = target_pdf / max(light_selection_pdf, 1.0e-6);",
            "float next_weight_sum = weight_sum + candidate_weight;",
            "reservoir.target_pdf = target_pdf;",
        ] {
            assert!(
                initial.contains(token),
                "ReSTIR-DI initial RIS must compensate the discrete light-selection proposal PDF: {token}"
            );
        }
        assert!(
            !initial.contains("float candidate_weight = restir_di_target_pdf_for_light_sample("),
            "target PDF alone underestimates direct lighting by the light proposal probability"
        );
        assert!(
            !initial.contains("hash_u32(rng_state ^ candidate * 747796405u) % restir.light_count"),
            "uniform light-id sampling causes high-variance direct lighting when the light table is large"
        );
        assert!(
            !initial.contains("while (lo + 1u < hi)"),
            "CDF lower_bound must compare the final one-element interval instead of returning light 0 early"
        );
        assert!(
            !initial.contains("candidate_weight = target_pdf / max(sample_pdf"),
            "the current emissive cluster target is not an area-density measure; area PDF amplification reintroduces bright fireflies"
        );
    }

    #[test]
    fn restir_di_shaders_are_surface_aware() {
        let initial = source("assets/shaders/passes/restir_di_initial.slang");
        let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");
        let pass = source("src/render/passes/restir_di.rs");
        let app = source("src/app.rs");
        let compact_app = app.split_whitespace().collect::<String>();

        for shader in [&initial, &temporal, &spatial] {
            assert!(shader.contains("current_surface_position_depth"));
            assert!(shader.contains("current_surface_normal_roughness"));
            assert!(shader.contains("current_surface_albedo_material"));
        }

        assert!(initial.contains("output_reservoirs[index] = invalid_reservoir();"));
        assert!(initial.contains("surface_is_valid(index)"));
        assert!(
            temporal
                .contains("float4 surface_position_depth = current_surface_position_depth[pixel];")
                && spatial.contains(
                    "float4 surface_position_depth = current_surface_position_depth[pixel];"
                ),
            "temporal and spatial reuse should validate with the cached center surface load"
        );
        assert!(initial.contains("current_surface_albedo_material"));
        assert!(temporal.contains("compatible_temporal_surface"));
        assert!(temporal.contains("uint previous_index"));
        assert!(temporal.contains("RestirDiReservoir history_reservoir"));
        assert!(temporal.contains("uint capped_history_m"));
        assert!(temporal.contains("history_target_pdf"));
        assert!(spatial.contains("compatible_spatial_surface"));
        assert!(spatial.contains("normal_dot"));
        assert!(spatial.contains("position_delta"));

        assert!(pass.contains("update_surface_descriptors"));
        assert!(pass.contains("VptSurfacePass"));
        assert!(pass.contains("DescriptorType::STORAGE_IMAGE"));
        assert!(app.contains("restir_di.update_surface_descriptors"));
        assert!(
            compact_app
                .contains("builder.read_as(final_surface_writes[0],AccessKind::ComputeShaderRead)")
                || compact_app.contains(
                    "builder.read_as(final_surface_writes[0],AccessKind::ComputeShaderRead,)"
                )
        );
    }

    #[test]
    fn restir_di_temporal_uses_explicit_history_surface_and_selected_frame_ring() {
        let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
        let pass = source("src/render/passes/restir_di.rs");
        let pass_impl = pass
            .split("#[cfg(test)]")
            .next()
            .expect("implementation section should exist");
        let app = source("src/app.rs");
        let compact_app = app.split_whitespace().collect::<String>();

        assert!(temporal.contains("previous_surface_position_depth"));
        assert!(temporal.contains("previous_surface_normal_roughness"));
        assert!(temporal.contains("previous_surface_albedo_material"));
        assert!(temporal.contains("motion_history"));
        assert!(temporal.contains("previous_pixel"));
        assert!(temporal.contains("position_delta"));
        assert!(
            !temporal
                .contains("dot(normalize(normal_roughness.xyz), normalize(normal_roughness.xyz))")
        );

        assert!(pass_impl.contains("selected_reservoirs"));
        assert!(pass_impl.contains("selected_current_buffer"));
        assert!(pass_impl.contains("selected_history_buffer"));
        assert!(pass_impl.contains("update_frame_descriptors"));
        assert!(pass_impl.contains("update_surface_descriptors"));
        assert!(
            !pass_impl.contains("record_history_update"),
            "ReSTIR-DI selected history must be maintained by a selected reservoir frame ring, not a fullscreen copy"
        );
        assert!(
            !pass_impl.contains("cmd_copy_buffer"),
            "ReSTIR-DI pass must not issue a per-frame history copy"
        );

        assert!(
            !app.contains("restir_di_history_update"),
            "ReSTIR-DI graph must not add a transfer history update pass"
        );
        assert!(
            compact_app.contains("selected_current_resource")
                && compact_app.contains("selected_history_resource")
                && compact_app.contains("builder.write_as(selected_current_resource,")
                && compact_app.contains("AccessKind::ComputeShaderWrite,);")
                && compact_app
                    .contains("vpt_restir_reads=Some((uniform_resource,selected_current_dep))"),
            "ReSTIR-DI graph must write the current selected slot and feed that exact resource to VPT"
        );
        assert!(
            !compact_app.contains("builder.read_as(selected_current_dep,AccessKind::TransferRead)"),
            "ReSTIR-DI current selected resource must not be copied through the transfer queue"
        );
    }

    #[test]
    fn app_restir_di_dispatches_only_enabled_reuse_stages() {
        let app = source("src/app.rs");
        let compact = app.split_whitespace().collect::<String>();
        let pass = source("src/render/passes/restir_di.rs");

        for token in [
            "letrestir_di_temporal_active=restir_di_settings.temporal_enabled;",
            "letrestir_di_spatial_active=restir_di_temporal_active&&restir_di_settings.spatial_enabled;",
            "restir_di.update_frame_descriptors(renderer.device(),frame.frame_slot,selected_history_buffer,selected_current_buffer,restir_di_temporal_active,restir_di_spatial_active,);",
            "letinitial_output_resource=ifrestir_di_temporal_active{initial_resource}else{selected_current_resource};",
            "lettemporal_dep=ifrestir_di_temporal_active{",
            "letselected_current_dep=ifrestir_di_spatial_active",
        ] {
            assert!(
                compact.contains(token),
                "ReSTIR-DI app graph must gate disabled reuse stages and keep selected-current descriptors in sync: {token}"
            );
        }
        assert!(
            pass.contains("temporal_enabled: bool")
                && pass.contains("let initial_output = if temporal_enabled")
                && pass.contains("self.initial_stage.write_storage_descriptors_for_frame(\n            device,\n            frame_slot,\n            2,\n            &[initial_output],\n        );"),
            "when temporal reuse is disabled, ReSTIR-DI initial must write the selected current slot that VPT reads"
        );
        assert!(
            !compact.contains(
                "letinitial_dep=initial_writes[0];lettemporal_writes=graph.add_pass(\"restir_di_temporal\""
            ),
            "ReSTIR-DI must not run temporal or graph-read selected history when temporal reuse is disabled"
        );
    }

    #[test]
    fn restir_di_surface_descriptors_are_refreshed_only_on_resize_not_every_frame() {
        let app = source("src/app.rs");
        let compact = app.split_whitespace().collect::<String>();
        let needle = "restir_di.update_surface_descriptors(&device,vpt_surface);";
        assert_eq!(
            compact.matches(needle).count(),
            1,
            "surface descriptor rewrites must stay out of the per-frame render path"
        );
        assert!(compact.contains("vpt_surface.resize_images("));
    }

    #[test]
    fn restir_di_temporal_combines_history_in_current_surface_measure_without_unbounded_weight_sum()
    {
        let temporal = source("assets/shaders/passes/restir_di_temporal.slang");

        assert!(temporal.contains("current_target_pdf"));
        assert!(temporal.contains("history_target_pdf"));
        assert!(
            temporal.contains("restir_di_reservoir_stream_weight"),
            "temporal reuse should convert current/history reservoirs into the current pixel's stream measure"
        );
        assert!(
            temporal.contains("restir_di_finalize_reservoir_on_surface"),
            "temporal reuse should rebuild selected_weight from capped combined M and current-surface weight"
        );
        assert!(
            !temporal
                .contains("float weight_sum = reservoir.weight_sum + history_reservoir.weight_sum"),
            "temporal reuse must not keep adding historical weight_sum after sample_count_m is capped"
        );
        assert!(
            !temporal.contains("reservoir.selected_weight = reservoir.weight_sum /"),
            "temporal reuse should use the shared finalizer so weight_sum and selected_weight stay normalized together"
        );
    }

    #[test]
    fn restir_di_reuse_recomputes_selected_target_pdf_on_current_surface() {
        let common = source("assets/shaders/shared/restir_di_common.slang");
        let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

        assert!(common.contains("restir_di_target_pdf_for_reservoir"));
        assert!(common.contains("restir_di_finalize_reservoir_on_surface_with_target"));
        for shader in [temporal, spatial] {
            assert!(
                shader.contains("selected_target_pdf")
                    || shader.contains("restir_di_finalize_reservoir_on_surface("),
                "reused reservoirs must be renormalized against the current pixel surface"
            );
        }
    }

    #[test]
    fn restir_di_temporal_reuses_selected_target_pdf_without_finalizer_recompute() {
        let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
        let common = source("assets/shaders/shared/restir_di_common.slang");

        for token in [
            "float selected_target_pdf = current_target_pdf;",
            "selected_target_pdf = history_target_pdf;",
            "restir_di_finalize_reservoir_on_surface_with_target(",
            "selected_target_pdf,",
            "if (reservoir.sample_light_id != 0xffffffffu && reservoir.sample_count_m > 0u && reservoir.target_pdf <= 0.0)",
        ] {
            assert!(
                temporal.contains(token),
                "temporal shader missing selected-target reuse token {token}"
            );
        }
        assert!(
            common.contains("void restir_di_finalize_reservoir_on_surface_with_target("),
            "shared ReSTIR-DI common must expose finalizer that accepts a precomputed selected target"
        );
        assert!(
            !temporal
                .contains("restir_di_finalize_reservoir_on_surface(\n                reservoir,"),
            "temporal pass must not call the recomputing finalizer after it already computed current/history target PDFs"
        );
    }

    #[test]
    fn restir_di_spatial_reuses_selected_target_pdf_without_finalizer_recompute() {
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

        for token in [
            "float selected_target_pdf = center_target_pdf;",
            "selected_target_pdf = neighbor_target_pdf;",
            "restir_di_finalize_reservoir_on_surface_with_target(",
            "selected_target_pdf,",
        ] {
            assert!(
                spatial.contains(token),
                "spatial shader missing selected-target reuse token {token}"
            );
        }
        assert!(
            !spatial.contains("restir_di_finalize_reservoir_on_surface(\n        reservoir,"),
            "spatial pass must not recompute the selected target PDF after it already evaluated the chosen candidate"
        );
    }

    #[test]
    fn restir_di_reuse_shaders_validate_with_cached_center_position() {
        let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

        for (name, shader) in [
            ("temporal", temporal.as_str()),
            ("spatial", spatial.as_str()),
        ] {
            let compact = shader.split_whitespace().collect::<String>();
            assert!(
                compact.contains(
                    "float4surface_position_depth=current_surface_position_depth[pixel];if(surface_position_depth.w<0.0){"
                ),
                "{name} shader should validate the center surface using the cached position/depth load"
            );
            assert!(
                !shader.contains("if (!surface_is_valid(index))"),
                "{name} shader must not read current_surface_position_depth once for surface_is_valid and again for the center surface"
            );
        }
    }

    #[test]
    fn restir_di_reuse_shaders_cache_center_surface_reads() {
        let temporal = source("assets/shaders/passes/restir_di_temporal.slang");
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

        assert!(
            temporal.contains(
                "compatible_temporal_surface(surface_position_depth, surface_normal_roughness, surface_albedo_material, previous_pixel)"
            ),
            "temporal reuse should pass cached center surface values into compatibility checks"
        );
        assert!(
            spatial.contains(
                "compatible_spatial_surface(surface_position_depth, surface_normal_roughness, surface_albedo_material, neighbor_index)"
            ),
            "spatial reuse should pass cached center surface values into compatibility checks"
        );
        assert!(
            !temporal.contains("compatible_temporal_surface(index, previous_pixel)")
                && !spatial.contains("compatible_spatial_surface(index, neighbor_index)"),
            "reuse shaders must not reload the center surface in each compatibility check"
        );
    }

    #[test]
    fn restir_di_spatial_combines_neighbors_with_current_surface_stream_weights() {
        let common = source("assets/shaders/shared/restir_di_common.slang");
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

        assert!(common.contains("restir_di_reservoir_stream_weight"));
        assert!(common.contains("restir_di_finalize_reservoir_on_surface"));
        assert!(spatial.contains("neighbor_target_pdf"));
        assert!(spatial.contains("accepted_neighbor_m"));
        assert!(
            !spatial.contains("float neighbor_weight = max(neighbor.weight_sum"),
            "spatial reuse must not sample a neighbor using the neighbor surface's weight sum"
        );
        assert!(
            !spatial.contains("float weight_sum = reservoir.weight_sum + neighbor.weight_sum"),
            "spatial reuse must rebuild the combined stream weight in the current pixel's measure"
        );
    }

    #[test]
    fn vpt_restir_direct_resolve_tests_visibility_before_applying_selected_weight() {
        let vpt = source("assets/shaders/passes/vpt.slang");

        assert!(vpt.contains("restir_di_light_visible_from_hit"));
        assert!(
            vpt.contains("trace_any_hit_ray(") && vpt.contains("return !shadow_occluded;"),
            "ReSTIR direct resolve must reject occluded reservoirs with an any-hit visibility query before selected_weight creates bright leaks"
        );
        assert!(
            !vpt.contains("HitResult occluder = trace_primary_ray(shadow_ray"),
            "ReSTIR visibility should not use full material-returning traversal for shadow rays"
        );
        assert!(
            vpt.find("restir_di_light_visible_from_hit")
                .expect("VPT ReSTIR resolve should test visibility")
                < vpt
                    .find("sample.radiance = albedo * reservoir.sample_radiance.rgb")
                    .expect("VPT ReSTIR resolve should assign radiance"),
            "visibility must be tested before applying reservoir.selected_weight"
        );
    }

    #[test]
    fn vpt_restir_direct_resolve_keeps_occluded_reservoirs_unselected_for_fallback() {
        let vpt = source("assets/shaders/passes/vpt.slang");

        assert!(
            vpt.find("if (!restir_di_light_visible_from_hit(hit, reservoir, scene))")
                .expect("VPT ReSTIR resolve should reject occluded light samples")
                < vpt
                    .find("sample.selected_weight = selected_weight;")
                    .expect("VPT ReSTIR resolve should only mark usable samples as selected"),
            "occluded ReSTIR-DI reservoirs must keep selected_weight at zero so VPT falls back to analytic direct light instead of writing black direct-light blocks"
        );
    }

    #[test]
    fn restir_di_light_table_excludes_analytic_sun_and_preserves_emissive_sampling_power() {
        let app = source("src/app.rs");
        let restir = source("src/render/restir_di.rs");
        let vpt = source("assets/shaders/passes/vpt.slang");

        assert!(
            app.contains("build_direct_lights_from_ucvh(ucvh"),
            "ReSTIR-DI direct-light setup should build a finite-emissive light table; the analytic sun is evaluated separately in VPT"
        );
        assert!(
            !app.contains("light.intensity.max_element().max(0.0)")
                && !app.contains("let (sun_direction, sun_intensity)"),
            "collapsing the directional light to max_element before ReSTIR-DI lets one reservoir sample replace the deterministic RGB sun and creates over-bright direct-light blocks"
        );
        assert!(
            restir.contains("pub fn build_direct_lights_from_ucvh")
                && restir.contains("ucvh: &Ucvh")
                && restir.contains("max_lights: usize"),
            "ReSTIR-DI direct-light builders must not include the analytic sun in the stochastic finite-light reservoir table"
        );
        assert!(
            vpt.contains("float3 analytic_direct = analytic_sun_direct(hit, scene, rng_state);")
                && vpt.contains("float3 direct_radiance = analytic_direct + direct.radiance;"),
            "VPT must always keep deterministic sun direct light and add finite-light ReSTIR-DI only when a reservoir is usable"
        );
    }

    #[test]
    fn restir_di_shaders_reject_nonfinite_reservoir_weights_before_vpt_resolve() {
        let common = source("assets/shaders/shared/restir_di_common.slang");
        let vpt = source("assets/shaders/passes/vpt.slang");

        for token in [
            "static const float RESTIR_DI_MAX_SELECTED_WEIGHT",
            "bool restir_di_candidate_finite(float value)",
            "restir_di_bounded_selected_weight",
            "restir_di_is_valid_reservoir(RestirDiReservoir reservoir)",
            "restir_di_candidate_finite(reservoir.target_pdf)",
            "restir_di_candidate_finite(reservoir.weight_sum)",
            "restir_di_candidate_finite(reservoir.selected_weight)",
        ] {
            assert!(
                common.contains(token),
                "ReSTIR-DI common shader must bound invalid/overlarge reservoir state before reuse: {token}"
            );
        }
        assert!(
            vpt.contains(
                "float selected_weight = restir_di_bounded_selected_weight(hit_reservoir);"
            ) && vpt.contains("if (selected_weight <= 0.0)")
                && vpt.contains("sample.selected_weight = selected_weight;")
                && vpt.contains("* selected_weight"),
            "VPT direct resolve must consume a finite bounded ReSTIR-DI weight evaluated on the actual VPT hit, not raw reservoir.selected_weight"
        );
        assert!(
            !vpt.contains("* reservoir.selected_weight"),
            "raw selected_weight can turn bad reservoirs into persistent white fireflies"
        );
    }

    #[test]
    fn vpt_restir_di_falls_back_to_analytic_direct_when_reservoir_unusable() {
        let vpt = source("assets/shaders/passes/vpt.slang");

        assert!(
            vpt.contains(
                "float3 analytic_sun_direct(HitResult hit, SceneUniforms scene, inout uint rng_state)"
            )
                && vpt.contains("float3 analytic_direct = analytic_sun_direct(hit, scene, rng_state);")
                && vpt.contains("float3 direct_radiance = analytic_direct + direct.radiance;")
                && vpt.contains(
                    "float3 contribution = throughput * analytic_sun_direct(hit, scene, rng_state);"
                ),
            "invalid or incompatible ReSTIR-DI reservoirs should still keep analytic sun direct instead of turning first-bounce direct light into black blocks"
        );
    }

    #[test]
    fn restir_di_spatial_disabled_is_exact_temporal_passthrough() {
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

        assert!(spatial.contains("if (restir.spatial_enabled == 0u)"));
        assert!(spatial.contains("output_reservoirs[index] = reservoir;"));
        assert!(
            spatial.find("if (restir.spatial_enabled == 0u)")
                < spatial.find("restir_di_reservoir_stream_weight")
        );
    }

    #[test]
    fn restir_di_hot_reservoir_buffers_are_gpu_only() {
        let source = source("src/render/passes/restir_di.rs");
        let reservoir_fn = source
            .split("fn create_reservoir_buffer")
            .nth(1)
            .expect("create_reservoir_buffer should exist")
            .split("fn write_mapped")
            .next()
            .expect("create_reservoir_buffer body should end before mapped writes");

        assert!(reservoir_fn.contains("MemoryLocation::GpuOnly"));
        assert!(
            !reservoir_fn.contains("TRANSFER_SRC") && !reservoir_fn.contains("TRANSFER_DST"),
            "ReSTIR-DI reservoirs are no longer copied on the transfer queue"
        );
        assert!(
            !reservoir_fn.contains("write_mapped_slice"),
            "fullscreen reservoir buffers are GPU hot resources and should not require host-visible initialization"
        );
    }

    #[test]
    fn app_profiles_each_restir_di_compute_stage_separately() {
        let app = source("src/app.rs");

        for scope in [
            "GpuProfileScope::RestirDiInitial",
            "GpuProfileScope::RestirDiTemporal",
            "GpuProfileScope::RestirDiSpatial",
        ] {
            assert!(
                app.contains(scope),
                "{scope} should be emitted around the matching ReSTIR graph pass"
            );
        }
        assert!(
            app.find("GpuProfileScope::RestirDiInitial") < app.find("restir_di.record_initial(")
        );
        assert!(
            app.find("GpuProfileScope::RestirDiTemporal") < app.find("restir_di.record_temporal(")
        );
        assert!(
            app.find("GpuProfileScope::RestirDiSpatial") < app.find("restir_di.record_spatial(")
        );
    }

    #[test]
    fn app_skips_spatial_passthrough_when_spatial_reuse_is_disabled() {
        let app = source("src/app.rs");
        let compact = app.split_whitespace().collect::<String>();

        assert!(
            compact.contains("ifrestir_di_spatial_active{"),
            "the spatial graph pass should be created only when spatial reuse is enabled"
        );
        assert!(
            compact.contains("lettemporal_output_resource=ifrestir_di_spatial_active{temporal_resource}else{selected_current_resource};")
                && compact.contains("letselected_current_dep=ifrestir_di_spatial_active{")
                && compact.contains("}else{temporal_dep};")
                && compact.contains("vpt_restir_reads=Some((uniform_resource,selected_current_dep))"),
            "spatial-disabled ReSTIR should write the temporal output directly into the current selected slot"
        );
    }

    #[test]
    fn restir_di_spatial_samples_multiple_neighbor_offsets_instead_of_one_right_neighbor() {
        let spatial = source("assets/shaders/passes/restir_di_spatial.slang");

        assert!(spatial.contains("spatial_sample_count"));
        assert!(spatial.contains("int2 spatial_offsets"));
        assert!(spatial.contains("sample < min(restir.spatial_sample_count, 8u)"));
        assert!(spatial.contains("neighbor_offset"));
        assert!(
            !spatial.contains("uint neighbor_index = index + 1u;"),
            "spatial reuse should not propagate only to the right-hand neighbor"
        );
    }

    #[test]
    fn restir_di_pass_does_not_issue_pass_local_barriers() {
        let implementation = std::fs::read_to_string("src/render/passes/restir_di.rs")
            .expect("restir pass source should be readable");
        let implementation = implementation
            .split("#[cfg(test)]")
            .next()
            .expect("implementation section should exist");
        assert!(!implementation.contains("cmd_pipeline_barrier"));
        assert!(!implementation.contains("ImageMemoryBarrier"));
        assert!(!implementation.contains("BufferMemoryBarrier"));
    }

    #[test]
    fn restir_di_pass_cleans_up_failed_construction_paths() {
        let implementation = std::fs::read_to_string("src/render/passes/restir_di.rs")
            .expect("restir pass source should be readable");
        let implementation = implementation
            .split("#[cfg(test)]")
            .next()
            .expect("implementation section should exist");

        assert!(implementation.contains("buffers.destroy(device, allocator);"));
        assert!(implementation.contains("initial_stage.destroy(device);"));
        assert!(implementation.contains("temporal_stage.destroy(device);"));
        assert!(implementation.contains("descriptor_pool.destroy(device);"));
        assert!(
            implementation.contains("device.destroy_descriptor_set_layout(descriptor_set_layout")
        );
    }
}
