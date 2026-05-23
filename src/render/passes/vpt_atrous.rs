use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::gpu_profiler::{GpuProfileScope, GpuProfiler};
use crate::render::graph::RenderGraph;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::passes::vpt_surface::VptSurfacePass;
use crate::render::passes::vpt_temporal::VptTemporalPass;
use crate::render::pipeline::{ComputePipeline, create_shader_module};
use crate::render::resource::{AccessKind, QueueType, ResourceHandle};
use crate::render::scene_ubo::{
    GpuSceneUniforms, LightingSettings, SceneUniformBuffer, VptDebugView,
};

const MAX_ATROUS_ITERATIONS: u32 = 5;
const CHAIN_SET_COUNT_PER_FRAME: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVptAtrousUniforms {
    pub iteration_index: u32,
    pub atrous_step_width: u32,
    pub pass_count: u32,
    pub _pad0: u32,
}

pub struct VptAtrousPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    atrous_uniform_buffers: Vec<GpuBuffer>,
    pub filtered_radiance: GpuImage,
    pub ping_radiance: GpuImage,
    pub pong_radiance: GpuImage,
}

#[derive(Clone, Copy)]
pub struct VptAtrousGraphOutputs {
    pub filtered_radiance: ResourceHandle,
}

#[derive(Clone, Copy)]
pub struct VptAtrousGraphInputs<'a> {
    pub frame_slot: usize,
    pub lighting_settings: LightingSettings,
    pub temporal_radiance: ResourceHandle,
    pub temporal_moments: ResourceHandle,
    pub surface_inputs: [ResourceHandle; 7],
    pub profiler: Option<&'a GpuProfiler>,
}

pub struct VptAtrousPassCreateInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub spirv_bytes: &'a [u8],
    pub scene_ubo: &'a SceneUniformBuffer,
    pub temporal: &'a VptTemporalPass,
    pub vpt_surface: &'a VptSurfacePass,
}

pub struct VptAtrousPassResizeInfo<'a> {
    pub width: u32,
    pub height: u32,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub temporal: &'a VptTemporalPass,
    pub vpt_surface: &'a VptSurfacePass,
}

impl VptAtrousPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptAtrousPassCreateInfo<'_>,
    ) -> Result<Self> {
        let descriptor_set_layout = create_descriptor_set_layout(device)?;
        let frame_count = info.scene_ubo.frame_count();
        let set_count = frame_count * CHAIN_SET_COUNT_PER_FRAME;
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: (2 * set_count) as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: (7 * set_count) as u32,
            },
        ];
        let descriptor_pool = match DescriptorPool::new(device, set_count as u32, &pool_sizes) {
            Ok(pool) => pool,
            Err(error) => {
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let layouts = vec![descriptor_set_layout; set_count];
        let descriptor_sets = match descriptor_pool.allocate(device, &layouts) {
            Ok(sets) => sets,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };

        let images = match create_atrous_images(device, allocator, info.width, info.height) {
            Ok(images) => images,
            Err(error) => {
                descriptor_pool.destroy(device);
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                return Err(error);
            }
        };
        let atrous_uniform_buffers =
            match create_atrous_uniform_buffers(device, allocator, frame_count) {
                Ok(buffers) => buffers,
                Err(error) => {
                    images.destroy(device, allocator);
                    descriptor_pool.destroy(device);
                    unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
                    return Err(error);
                }
            };
        write_descriptor_sets(
            device,
            &descriptor_sets,
            &atrous_uniform_buffers,
            info.scene_ubo,
            info.temporal,
            info.vpt_surface,
            &VptAtrousImageRefs {
                filtered_radiance: &images.filtered_radiance,
                ping_radiance: &images.ping_radiance,
                pong_radiance: &images.pong_radiance,
            },
        );

        let shader_module = match create_shader_module(device, info.spirv_bytes) {
            Ok(module) => module,
            Err(error) => {
                destroy_buffers(atrous_uniform_buffers, device, allocator);
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
            &[],
        ) {
            Ok(pipeline) => pipeline,
            Err(error) => {
                unsafe { device.destroy_shader_module(shader_module, None) };
                destroy_buffers(atrous_uniform_buffers, device, allocator);
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
            descriptor_sets,
            atrous_uniform_buffers,
            filtered_radiance: images.filtered_radiance,
            ping_radiance: images.ping_radiance,
            pong_radiance: images.pong_radiance,
        })
    }

    pub fn resize_images(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        info: VptAtrousPassResizeInfo<'_>,
    ) -> Result<()> {
        let new_images = create_atrous_images(device, allocator, info.width, info.height)?;
        let old_images = VptAtrousImages {
            filtered_radiance: std::mem::replace(
                &mut self.filtered_radiance,
                new_images.filtered_radiance,
            ),
            ping_radiance: std::mem::replace(&mut self.ping_radiance, new_images.ping_radiance),
            pong_radiance: std::mem::replace(&mut self.pong_radiance, new_images.pong_radiance),
        };
        old_images.destroy(device, allocator);
        write_descriptor_sets(
            device,
            &self.descriptor_sets,
            &self.atrous_uniform_buffers,
            info.scene_ubo,
            info.temporal,
            info.vpt_surface,
            &VptAtrousImageRefs {
                filtered_radiance: &self.filtered_radiance,
                ping_radiance: &self.ping_radiance,
                pong_radiance: &self.pong_radiance,
            },
        );
        Ok(())
    }

    pub fn output_image(&self) -> &GpuImage {
        &self.filtered_radiance
    }

    pub fn active_iteration_count(settings: LightingSettings) -> u32 {
        if settings.denoiser_enabled() && settings.vpt_debug_view == VptDebugView::Final {
            settings
                .denoiser_atrous_iterations
                .min(MAX_ATROUS_ITERATIONS)
        } else {
            0
        }
    }

    pub fn pass_count_for_iterations(iterations: u32) -> u32 {
        iterations.clamp(1, MAX_ATROUS_ITERATIONS)
    }

    pub fn record(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frame_slot: usize,
        active_iterations: u32,
        iteration_index: u32,
    ) {
        let set_index = descriptor_set_index(frame_slot, active_iterations, iteration_index);
        let extent = self.filtered_radiance.extent;

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.descriptor_sets[set_index]],
                &[],
            );
            device.cmd_dispatch(cmd, extent.width.div_ceil(8), extent.height.div_ceil(8), 1);
        }
    }

    pub fn register_graph<'a>(
        &'a self,
        graph: &mut RenderGraph<'a>,
        inputs: VptAtrousGraphInputs<'a>,
    ) -> VptAtrousGraphOutputs {
        let VptAtrousGraphInputs {
            frame_slot,
            lighting_settings,
            temporal_radiance,
            temporal_moments,
            surface_inputs,
            profiler,
        } = inputs;
        let atrous_iterations = Self::active_iteration_count(lighting_settings);
        let atrous_pass_count = Self::pass_count_for_iterations(atrous_iterations);
        let atrous_filtered_resource = graph.import_image_with_access(
            self.filtered_radiance.handle,
            self.filtered_radiance.extent.width,
            self.filtered_radiance.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Undefined,
        );
        let atrous_ping_resource = graph.import_image_with_access(
            self.ping_radiance.handle,
            self.ping_radiance.extent.width,
            self.ping_radiance.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Undefined,
        );
        let atrous_pong_resource = graph.import_image_with_access(
            self.pong_radiance.handle,
            self.pong_radiance.extent.width,
            self.pong_radiance.extent.height,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Undefined,
        );
        let mut atrous_input_dep = temporal_radiance;
        let mut atrous_filtered_dep = temporal_radiance;
        let mut atrous_ping_dep = atrous_ping_resource;
        let mut atrous_pong_dep = atrous_pong_resource;
        for iteration_index in 0..atrous_pass_count {
            let output_is_final = iteration_index + 1 == atrous_pass_count;
            let output_is_ping = !output_is_final && iteration_index.is_multiple_of(2);
            let atrous_output_resource = if output_is_final {
                atrous_filtered_resource
            } else if output_is_ping {
                atrous_ping_dep
            } else {
                atrous_pong_dep
            };
            let begin_atrous_scope = iteration_index == 0;
            let end_atrous_scope = iteration_index + 1 == atrous_pass_count;
            let atrous_writes = graph.add_pass("vpt_atrous", QueueType::Compute, |builder| {
                builder.read_as(atrous_input_dep, AccessKind::ComputeShaderRead);
                builder.read_as(temporal_moments, AccessKind::ComputeShaderRead);
                builder.read_as(surface_inputs[0], AccessKind::ComputeShaderRead);
                builder.read_as(surface_inputs[1], AccessKind::ComputeShaderRead);
                builder.read_as(surface_inputs[2], AccessKind::ComputeShaderRead);
                builder.read_as(surface_inputs[3], AccessKind::ComputeShaderRead);
                builder.write_as(atrous_output_resource, AccessKind::ComputeShaderWrite);
                Box::new(move |ctx| {
                    if begin_atrous_scope && let Some(profiler) = profiler {
                        profiler.begin_scope(
                            ctx.device,
                            ctx.command_buffer,
                            frame_slot,
                            GpuProfileScope::VptAtrous,
                        );
                    }
                    self.record(
                        ctx.device,
                        ctx.command_buffer,
                        frame_slot,
                        atrous_iterations,
                        iteration_index,
                    );
                    if end_atrous_scope && let Some(profiler) = profiler {
                        profiler.end_scope(
                            ctx.device,
                            ctx.command_buffer,
                            frame_slot,
                            GpuProfileScope::VptAtrous,
                        );
                    }
                })
            });
            atrous_input_dep = atrous_writes[0];
            atrous_filtered_dep = atrous_writes[0];
            if output_is_ping {
                atrous_ping_dep = atrous_writes[0];
            } else if !output_is_final {
                atrous_pong_dep = atrous_writes[0];
            }
        }

        VptAtrousGraphOutputs {
            filtered_radiance: atrous_filtered_dep,
        }
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        destroy_buffers(self.atrous_uniform_buffers, device, allocator);
        self.filtered_radiance.destroy(device, allocator);
        self.ping_radiance.destroy(device, allocator);
        self.pong_radiance.destroy(device, allocator);
    }
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    DescriptorLayoutBuilder::new()
        .add_binding(
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::COMPUTE,
            1,
        )
        .add_binding(
            1,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
            1,
        )
        .add_binding(
            2,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
            1,
        )
        .add_binding(
            3,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
            1,
        )
        .add_binding(
            4,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
            1,
        )
        .add_binding(
            5,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
            1,
        )
        .add_binding(
            6,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
            1,
        )
        .add_binding(
            7,
            vk::DescriptorType::STORAGE_IMAGE,
            vk::ShaderStageFlags::COMPUTE,
            1,
        )
        .add_binding(
            8,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::COMPUTE,
            1,
        )
        .build(device)
}

struct VptAtrousImages {
    filtered_radiance: GpuImage,
    ping_radiance: GpuImage,
    pong_radiance: GpuImage,
}

struct VptAtrousImageRefs<'a> {
    filtered_radiance: &'a GpuImage,
    ping_radiance: &'a GpuImage,
    pong_radiance: &'a GpuImage,
}

struct VptAtrousDescriptorWriteInfo<'a> {
    descriptor_set: vk::DescriptorSet,
    scene_ubo: &'a SceneUniformBuffer,
    frame_slot: usize,
    atrous_uniform_buffer: &'a GpuBuffer,
    input_radiance: &'a GpuImage,
    output_radiance: &'a GpuImage,
    temporal: &'a VptTemporalPass,
    vpt_surface: &'a VptSurfacePass,
}

impl VptAtrousImages {
    fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.filtered_radiance.destroy(device, allocator);
        self.ping_radiance.destroy(device, allocator);
        self.pong_radiance.destroy(device, allocator);
    }
}

fn create_atrous_images(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
) -> Result<VptAtrousImages> {
    let filtered_radiance =
        create_atrous_image(device, allocator, width, height, "vpt_atrous_filtered")?;
    let ping_radiance =
        match create_atrous_image(device, allocator, width, height, "vpt_atrous_ping") {
            Ok(image) => image,
            Err(error) => {
                filtered_radiance.destroy(device, allocator);
                return Err(error);
            }
        };
    let pong_radiance =
        match create_atrous_image(device, allocator, width, height, "vpt_atrous_pong") {
            Ok(image) => image,
            Err(error) => {
                ping_radiance.destroy(device, allocator);
                filtered_radiance.destroy(device, allocator);
                return Err(error);
            }
        };

    Ok(VptAtrousImages {
        filtered_radiance,
        ping_radiance,
        pong_radiance,
    })
}

fn create_atrous_image(
    device: &ash::Device,
    allocator: &GpuAllocator,
    width: u32,
    height: u32,
    name: &'static str,
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
            name,
        },
    )
}

fn create_atrous_uniform_buffers(
    device: &ash::Device,
    allocator: &GpuAllocator,
    frame_count: usize,
) -> Result<Vec<GpuBuffer>> {
    let mut buffers = Vec::with_capacity(frame_count * CHAIN_SET_COUNT_PER_FRAME);
    for frame_slot in 0..frame_count {
        for active_iterations in 0..=MAX_ATROUS_ITERATIONS {
            let pass_count = VptAtrousPass::pass_count_for_iterations(active_iterations);
            for iteration_index in 0..pass_count {
                let buffer = match GpuBuffer::new(
                    device,
                    allocator,
                    std::mem::size_of::<GpuVptAtrousUniforms>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                    &format!(
                        "vpt_atrous_uniforms_f{frame_slot}_i{active_iterations}_{iteration_index}"
                    ),
                ) {
                    Ok(buffer) => buffer,
                    Err(error) => {
                        destroy_buffers(buffers, device, allocator);
                        return Err(error);
                    }
                };
                let uniforms = GpuVptAtrousUniforms {
                    iteration_index,
                    atrous_step_width: 1u32 << iteration_index,
                    pass_count,
                    _pad0: 0,
                };
                write_mapped(buffer.mapped_ptr(), &uniforms);
                buffers.push(buffer);
            }
        }
    }
    Ok(buffers)
}

fn write_descriptor_sets(
    device: &ash::Device,
    descriptor_sets: &[vk::DescriptorSet],
    atrous_uniform_buffers: &[GpuBuffer],
    scene_ubo: &SceneUniformBuffer,
    temporal: &VptTemporalPass,
    vpt_surface: &VptSurfacePass,
    images: &VptAtrousImageRefs<'_>,
) {
    for frame_slot in 0..scene_ubo.frame_count() {
        for active_iterations in 0..=MAX_ATROUS_ITERATIONS {
            let pass_count = VptAtrousPass::pass_count_for_iterations(active_iterations);
            for iteration_index in 0..pass_count {
                let set_index =
                    descriptor_set_index(frame_slot, active_iterations, iteration_index);
                let input_image =
                    input_image_for_chain(active_iterations, iteration_index, temporal, images);
                let output_image =
                    output_image_for_chain(active_iterations, iteration_index, images);
                write_descriptor_set(
                    device,
                    VptAtrousDescriptorWriteInfo {
                        descriptor_set: descriptor_sets[set_index],
                        scene_ubo,
                        frame_slot,
                        atrous_uniform_buffer: &atrous_uniform_buffers[set_index],
                        input_radiance: input_image,
                        output_radiance: output_image,
                        temporal,
                        vpt_surface,
                    },
                );
            }
        }
    }
}

fn write_descriptor_set(device: &ash::Device, info: VptAtrousDescriptorWriteInfo<'_>) {
    let scene_info = vk::DescriptorBufferInfo::default()
        .buffer(info.scene_ubo.buffer_handle(info.frame_slot))
        .offset(0)
        .range(std::mem::size_of::<GpuSceneUniforms>() as u64);
    let atrous_info = vk::DescriptorBufferInfo::default()
        .buffer(info.atrous_uniform_buffer.handle)
        .offset(0)
        .range(std::mem::size_of::<GpuVptAtrousUniforms>() as u64);
    let image_refs = [
        info.input_radiance,
        &info.temporal.accumulated_moments_history,
        &info.vpt_surface.surface_position_depth,
        &info.vpt_surface.surface_normal_roughness,
        &info.vpt_surface.surface_albedo_material,
        &info.vpt_surface.surface_material_roughness,
        info.output_radiance,
    ];
    let image_infos: Vec<_> = image_refs
        .iter()
        .map(|image| {
            vk::DescriptorImageInfo::default()
                .image_view(image.view)
                .image_layout(vk::ImageLayout::GENERAL)
        })
        .collect();
    let mut writes = vec![
        vk::WriteDescriptorSet::default()
            .dst_set(info.descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&scene_info)),
    ];
    writes.extend(image_infos.iter().enumerate().map(|(idx, image_info)| {
        vk::WriteDescriptorSet::default()
            .dst_set(info.descriptor_set)
            .dst_binding((idx + 1) as u32)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(image_info))
    }));
    writes.push(
        vk::WriteDescriptorSet::default()
            .dst_set(info.descriptor_set)
            .dst_binding(8)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&atrous_info)),
    );
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn input_image_for_chain<'a>(
    _active_iterations: u32,
    iteration_index: u32,
    temporal: &'a VptTemporalPass,
    images: &'a VptAtrousImageRefs<'_>,
) -> &'a GpuImage {
    if iteration_index == 0 {
        &temporal.accumulated_radiance
    } else if (iteration_index - 1).is_multiple_of(2) {
        images.ping_radiance
    } else {
        images.pong_radiance
    }
}

fn output_image_for_chain<'a>(
    active_iterations: u32,
    iteration_index: u32,
    images: &'a VptAtrousImageRefs<'_>,
) -> &'a GpuImage {
    let pass_count = VptAtrousPass::pass_count_for_iterations(active_iterations);
    if iteration_index + 1 == pass_count {
        images.filtered_radiance
    } else if iteration_index.is_multiple_of(2) {
        images.ping_radiance
    } else {
        images.pong_radiance
    }
}

pub fn descriptor_set_index(
    frame_slot: usize,
    active_iterations: u32,
    iteration_index: u32,
) -> usize {
    let active_iterations = active_iterations.min(MAX_ATROUS_ITERATIONS);
    let base = chain_base_index(active_iterations);
    frame_slot * CHAIN_SET_COUNT_PER_FRAME + base + iteration_index as usize
}

fn chain_base_index(active_iterations: u32) -> usize {
    if active_iterations == 0 {
        0
    } else {
        1 + ((active_iterations - 1) * active_iterations / 2) as usize
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
mod tests {
    use super::*;

    #[test]
    fn descriptor_set_indices_cover_all_chain_lengths_without_overlap() {
        let mut indices = Vec::new();
        for active_iterations in 0..=MAX_ATROUS_ITERATIONS {
            let pass_count = VptAtrousPass::pass_count_for_iterations(active_iterations);
            for iteration_index in 0..pass_count {
                indices.push(descriptor_set_index(0, active_iterations, iteration_index));
            }
        }
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();

        assert_eq!(indices.len(), CHAIN_SET_COUNT_PER_FRAME);
        assert_eq!(sorted.len(), CHAIN_SET_COUNT_PER_FRAME);
        assert_eq!(sorted[0], 0);
        assert_eq!(
            sorted[CHAIN_SET_COUNT_PER_FRAME - 1],
            CHAIN_SET_COUNT_PER_FRAME - 1
        );
    }

    #[test]
    fn active_iteration_count_preserves_debug_and_disabled_passthrough() {
        let mut settings = LightingSettings::default();
        assert_eq!(VptAtrousPass::active_iteration_count(settings), 4);

        settings.denoiser_mode = crate::render::scene_ubo::VptDenoiserMode::Off;
        assert_eq!(VptAtrousPass::active_iteration_count(settings), 0);

        settings.denoiser_mode = crate::render::scene_ubo::VptDenoiserMode::Svgf;
        settings.vpt_debug_view = VptDebugView::Raw;
        assert_eq!(VptAtrousPass::active_iteration_count(settings), 0);

        settings.vpt_debug_view = VptDebugView::Final;
        settings.denoiser_atrous_iterations = 9;
        assert_eq!(
            VptAtrousPass::active_iteration_count(settings),
            MAX_ATROUS_ITERATIONS
        );
    }

    #[test]
    fn active_iteration_count_uses_svgf_fallback_for_nrd_modes() {
        for denoiser_mode in [
            crate::render::scene_ubo::VptDenoiserMode::Relax,
            crate::render::scene_ubo::VptDenoiserMode::Reblur,
        ] {
            let settings = LightingSettings {
                denoiser_mode,
                ..LightingSettings::default()
            };

            assert_eq!(VptAtrousPass::active_iteration_count(settings), 4);
        }
    }
}
