use anyhow::Result;
use ash::vk;
use bytemuck::Zeroable;
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
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
    spatial_reservoirs: GpuBuffer,
    history_reservoirs: GpuBuffer,
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
    spatial_reservoirs: GpuBuffer,
    history_reservoirs: GpuBuffer,
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
            spatial_reservoirs: buffers.spatial_reservoirs,
            history_reservoirs: buffers.history_reservoirs,
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
        let spatial_reservoirs =
            create_reservoir_buffer(device, allocator, reservoir_count, "restir_di_spatial")?;
        let history_reservoirs =
            create_reservoir_buffer(device, allocator, reservoir_count, "restir_di_history")?;

        std::mem::replace(&mut self.initial_reservoirs, initial_reservoirs)
            .destroy(device, allocator);
        std::mem::replace(&mut self.temporal_reservoirs, temporal_reservoirs)
            .destroy(device, allocator);
        std::mem::replace(&mut self.spatial_reservoirs, spatial_reservoirs)
            .destroy(device, allocator);
        std::mem::replace(&mut self.history_reservoirs, history_reservoirs)
            .destroy(device, allocator);

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

    pub fn spatial_buffer(&self) -> (&GpuBuffer, vk::DeviceSize, vk::BufferUsageFlags) {
        (
            &self.spatial_reservoirs,
            self.spatial_reservoirs.size,
            self.spatial_reservoirs.usage,
        )
    }

    pub fn history_buffer(&self) -> (&GpuBuffer, vk::DeviceSize, vk::BufferUsageFlags) {
        (
            &self.history_reservoirs,
            self.history_reservoirs.size,
            self.history_reservoirs.usage,
        )
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
        self.spatial_reservoirs.destroy(device, allocator);
        self.history_reservoirs.destroy(device, allocator);
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
                &self.history_reservoirs,
                &self.temporal_reservoirs,
            ],
        );
        self.spatial_stage.write_descriptors(
            device,
            &self.uniform_buffers,
            &[&self.temporal_reservoirs, &self.spatial_reservoirs],
        );
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
        let spatial_reservoirs = match create_reservoir_buffer(
            device,
            allocator,
            reservoir_count,
            "restir_di_spatial",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                temporal_reservoirs.destroy(device, allocator);
                initial_reservoirs.destroy(device, allocator);
                direct_lights.destroy(device, allocator);
                destroy_buffers(uniform_buffers, device, allocator);
                return Err(error);
            }
        };
        let history_reservoirs = match create_reservoir_buffer(
            device,
            allocator,
            reservoir_count,
            "restir_di_history",
        ) {
            Ok(buffer) => buffer,
            Err(error) => {
                spatial_reservoirs.destroy(device, allocator);
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
            spatial_reservoirs,
            history_reservoirs,
        })
    }

    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        destroy_buffers(self.uniform_buffers, device, allocator);
        self.direct_lights.destroy(device, allocator);
        self.initial_reservoirs.destroy(device, allocator);
        self.temporal_reservoirs.destroy(device, allocator);
        self.spatial_reservoirs.destroy(device, allocator);
        self.history_reservoirs.destroy(device, allocator);
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
    let buffer = GpuBuffer::new(
        device,
        allocator,
        (count * std::mem::size_of::<GpuRestirDiReservoir>()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        name,
    )?;
    let invalid_reservoir = GpuRestirDiReservoir {
        sample_light_id: u32::MAX,
        ..GpuRestirDiReservoir::zeroed()
    };
    let initial_data = vec![invalid_reservoir; count];
    write_mapped_slice(buffer.mapped_ptr(), &initial_data);
    Ok(buffer)
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
