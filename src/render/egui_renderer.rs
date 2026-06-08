use anyhow::{Context, Result, anyhow};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use egui::epaint::{ImageData, Primitive, Vertex};
use egui::{ClippedPrimitive, TextureId, TexturesDelta};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::render::device::RenderDevice;
use crate::render::frame::FrameContext;
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::create_shader_module;

const EGUI_VERTEX_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/egui.vert.spv"));
const EGUI_FRAGMENT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/egui.frag.spv"));

#[derive(Debug, Clone)]
pub struct EguiFrame {
    pub clipped_primitives: Vec<ClippedPrimitive>,
    pub textures_delta: TexturesDelta,
    pub pixels_per_point: f32,
}

impl EguiFrame {
    pub fn is_empty(&self) -> bool {
        self.clipped_primitives.is_empty() && self.textures_delta.is_empty()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EguiPushConstants {
    screen_size_points: [f32; 2],
    _pad0: [f32; 2],
}

struct EguiTexture {
    image: GpuImage,
}

struct EguiTextureUpload<'a> {
    command_buffer: vk::CommandBuffer,
    frame_slot: usize,
    offset: [u32; 2],
    extent: [u32; 2],
    rgba: &'a [u8],
    initialize_whole_texture: bool,
}

pub struct EguiRenderer {
    pub sampler: vk::Sampler,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub vertex_buffers: Vec<Option<GpuBuffer>>,
    pub index_buffers: Vec<Option<GpuBuffer>>,
    pending_upload_buffers: Vec<Vec<GpuBuffer>>,
    font_texture: Option<EguiTexture>,
}

impl EguiRenderer {
    pub fn new(renderer: &RenderDevice) -> Result<Self> {
        let device = renderer.device();
        let sampler = create_sampler(device)?;
        let descriptor_set_layout = match create_descriptor_set_layout(device) {
            Ok(layout) => layout,
            Err(error) => {
                unsafe { device.destroy_sampler(sampler, None) };
                return Err(error);
            }
        };
        let descriptor_pool = match create_descriptor_pool(device) {
            Ok(pool) => pool,
            Err(error) => {
                unsafe {
                    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                    device.destroy_sampler(sampler, None);
                }
                return Err(error);
            }
        };
        let descriptor_set =
            match allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout) {
                Ok(set) => set,
                Err(error) => {
                    unsafe {
                        device.destroy_descriptor_pool(descriptor_pool, None);
                        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                        device.destroy_sampler(sampler, None);
                    }
                    return Err(error);
                }
            };
        let (pipeline_layout, pipeline) =
            match create_pipeline(device, descriptor_set_layout, renderer.swapchain_format()) {
                Ok(pipeline) => pipeline,
                Err(error) => {
                    unsafe {
                        device.destroy_descriptor_pool(descriptor_pool, None);
                        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                        device.destroy_sampler(sampler, None);
                    }
                    return Err(error);
                }
            };

        let frame_count = renderer.frame_slot_count();
        Ok(Self {
            sampler,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            pipeline_layout,
            pipeline,
            vertex_buffers: (0..frame_count).map(|_| None).collect(),
            index_buffers: (0..frame_count).map(|_| None).collect(),
            pending_upload_buffers: (0..frame_count).map(|_| Vec::new()).collect(),
            font_texture: None,
        })
    }

    pub fn record(
        &mut self,
        renderer: &RenderDevice,
        frame: &FrameContext,
        egui_frame: &EguiFrame,
    ) -> Result<()> {
        if egui_frame.is_empty() {
            return Ok(());
        }
        let device = renderer.device();
        let allocator = renderer.allocator();
        self.clear_pending_uploads_for_slot(device, allocator, frame.frame_slot);
        self.update_textures(
            device,
            allocator,
            frame.command_buffer,
            frame.frame_slot,
            &egui_frame.textures_delta,
        )?;

        let Some(_font_texture) = &self.font_texture else {
            tracing::warn!("skipping egui draw until font texture is uploaded");
            return Ok(());
        };

        let (vertices, indices, draw_commands) = collect_meshes(
            &egui_frame.clipped_primitives,
            egui_frame.pixels_per_point,
            frame.swapchain_extent,
        );
        if indices.is_empty() {
            return Ok(());
        }

        self.upload_mesh_buffers(device, allocator, frame.frame_slot, &vertices, &indices)?;
        let vertex_buffer = self.vertex_buffers[frame.frame_slot]
            .as_ref()
            .ok_or_else(|| anyhow!("egui vertex buffer missing after upload"))?;
        let index_buffer = self.index_buffers[frame.frame_slot]
            .as_ref()
            .ok_or_else(|| anyhow!("egui index buffer missing after upload"))?;

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(frame.swapchain_image_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let color_attachments = [color_attachment];
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: frame.swapchain_extent,
        };
        let rendering_info = vk::RenderingInfo::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments);

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: frame.swapchain_extent.width as f32,
            height: frame.swapchain_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let screen_size_points = [
            frame.swapchain_extent.width as f32 / egui_frame.pixels_per_point,
            frame.swapchain_extent.height as f32 / egui_frame.pixels_per_point,
        ];
        let push_constants = EguiPushConstants {
            screen_size_points,
            _pad0: [0.0; 2],
        };

        unsafe {
            device.cmd_begin_rendering(frame.command_buffer, &rendering_info);
            device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );
            device.cmd_set_viewport(frame.command_buffer, 0, &[viewport]);
            device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                frame.command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push_constants),
            );
            device.cmd_bind_vertex_buffers(frame.command_buffer, 0, &[vertex_buffer.handle], &[0]);
            device.cmd_bind_index_buffer(
                frame.command_buffer,
                index_buffer.handle,
                0,
                vk::IndexType::UINT32,
            );
            for command in draw_commands {
                device.cmd_set_scissor(frame.command_buffer, 0, &[command.scissor]);
                device.cmd_draw_indexed(
                    frame.command_buffer,
                    command.index_count,
                    1,
                    command.first_index,
                    command.vertex_offset,
                    0,
                );
            }
            device.cmd_end_rendering(frame.command_buffer);
        }

        Ok(())
    }

    pub fn destroy(mut self, device: &ash::Device, allocator: &GpuAllocator) {
        if let Some(texture) = self.font_texture.take() {
            texture.image.destroy(device, allocator);
        }
        for slot_uploads in self.pending_upload_buffers.drain(..) {
            for buffer in slot_uploads {
                buffer.destroy(device, allocator);
            }
        }
        for buffer in self.vertex_buffers.drain(..).flatten() {
            buffer.destroy(device, allocator);
        }
        for buffer in self.index_buffers.drain(..).flatten() {
            buffer.destroy(device, allocator);
        }
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
    }

    fn clear_pending_uploads_for_slot(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        frame_slot: usize,
    ) {
        for buffer in self.pending_upload_buffers[frame_slot].drain(..) {
            buffer.destroy(device, allocator);
        }
    }

    fn update_textures(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        command_buffer: vk::CommandBuffer,
        frame_slot: usize,
        textures_delta: &TexturesDelta,
    ) -> Result<()> {
        for texture_id in &textures_delta.free {
            if *texture_id == TextureId::Managed(0)
                && let Some(texture) = self.font_texture.take()
            {
                texture.image.destroy(device, allocator);
            }
        }

        for (texture_id, delta) in &textures_delta.set {
            if *texture_id != TextureId::Managed(0) {
                tracing::warn!(?texture_id, "skipping unsupported egui user texture");
                continue;
            }
            let size = delta.image.size();
            let width = u32::try_from(size[0]).context("egui texture width exceeds u32")?;
            let height = u32::try_from(size[1]).context("egui texture height exceeds u32")?;
            let rgba = image_data_to_rgba(&delta.image);
            let needs_full_upload = delta.pos.is_none() || self.font_texture.is_none();
            if needs_full_upload {
                if let Some(texture) = self.font_texture.take() {
                    texture.image.destroy(device, allocator);
                }
                self.font_texture = Some(EguiTexture {
                    image: create_font_image(device, allocator, width, height)?,
                });
                self.write_font_descriptor(device);
                self.upload_texture_region(
                    device,
                    allocator,
                    EguiTextureUpload {
                        command_buffer,
                        frame_slot,
                        offset: [0, 0],
                        extent: [width, height],
                        rgba: &rgba,
                        initialize_whole_texture: true,
                    },
                )?;
            } else if let Some(pos) = delta.pos {
                let x = u32::try_from(pos[0]).context("egui texture x exceeds u32")?;
                let y = u32::try_from(pos[1]).context("egui texture y exceeds u32")?;
                self.upload_texture_region(
                    device,
                    allocator,
                    EguiTextureUpload {
                        command_buffer,
                        frame_slot,
                        offset: [x, y],
                        extent: [width, height],
                        rgba: &rgba,
                        initialize_whole_texture: false,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn upload_texture_region(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        upload: EguiTextureUpload<'_>,
    ) -> Result<()> {
        let EguiTextureUpload {
            command_buffer,
            frame_slot,
            offset,
            extent,
            rgba,
            initialize_whole_texture,
        } = upload;
        let expected_len = extent[0] as usize * extent[1] as usize * 4;
        if rgba.len() != expected_len {
            return Err(anyhow!(
                "egui texture upload size mismatch: expected {} bytes, got {}",
                expected_len,
                rgba.len()
            ));
        }
        let staging = GpuBuffer::new(
            device,
            allocator,
            rgba.len() as vk::DeviceSize,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "egui texture staging",
        )?;
        let ptr = staging
            .mapped_ptr()
            .ok_or_else(|| anyhow!("egui texture staging buffer is not host visible"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(rgba.as_ptr(), ptr, rgba.len());
        }

        let texture = self
            .font_texture
            .as_ref()
            .ok_or_else(|| anyhow!("egui font texture missing during upload"))?;
        let old_layout = if initialize_whole_texture {
            vk::ImageLayout::UNDEFINED
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        transition_image(
            device,
            command_buffer,
            texture.image.handle,
            old_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D {
                x: offset[0] as i32,
                y: offset[1] as i32,
                z: 0,
            })
            .image_extent(vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: 1,
            });
        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                staging.handle,
                texture.image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        transition_image(
            device,
            command_buffer,
            texture.image.handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        self.pending_upload_buffers[frame_slot].push(staging);
        Ok(())
    }

    fn upload_mesh_buffers(
        &mut self,
        device: &ash::Device,
        allocator: &GpuAllocator,
        frame_slot: usize,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Result<()> {
        let vertex_bytes = bytemuck::cast_slice(vertices);
        let index_bytes = bytemuck::cast_slice(indices);
        ensure_host_buffer(
            &mut self.vertex_buffers[frame_slot],
            device,
            allocator,
            vertex_bytes.len() as vk::DeviceSize,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "egui vertex buffer",
        )?;
        ensure_host_buffer(
            &mut self.index_buffers[frame_slot],
            device,
            allocator,
            index_bytes.len() as vk::DeviceSize,
            vk::BufferUsageFlags::INDEX_BUFFER,
            "egui index buffer",
        )?;
        write_buffer_bytes(
            self.vertex_buffers[frame_slot]
                .as_ref()
                .expect("vertex buffer should exist"),
            vertex_bytes,
        )?;
        write_buffer_bytes(
            self.index_buffers[frame_slot]
                .as_ref()
                .expect("index buffer should exist"),
            index_bytes,
        )?;
        Ok(())
    }

    fn write_font_descriptor(&self, device: &ash::Device) {
        let Some(texture) = &self.font_texture else {
            return;
        };
        let image_info = [vk::DescriptorImageInfo::default()
            .image_view(texture.image.view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let sampler_info = [vk::DescriptorImageInfo::default().sampler(self.sampler)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(&image_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(&sampler_info),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }
}

struct EguiDrawCommand {
    scissor: vk::Rect2D,
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
}

fn collect_meshes(
    primitives: &[ClippedPrimitive],
    pixels_per_point: f32,
    extent: vk::Extent2D,
) -> (Vec<Vertex>, Vec<u32>, Vec<EguiDrawCommand>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut commands = Vec::new();
    for primitive in primitives {
        let Primitive::Mesh(mesh) = &primitive.primitive else {
            tracing::warn!("skipping unsupported egui paint callback primitive");
            continue;
        };
        if mesh.indices.is_empty() || mesh.vertices.is_empty() {
            continue;
        }
        let Some(scissor) = clip_rect_to_scissor(primitive.clip_rect, pixels_per_point, extent)
        else {
            continue;
        };
        let vertex_offset = vertices.len() as i32;
        let first_index = indices.len() as u32;
        let index_count = mesh.indices.len() as u32;
        vertices.extend_from_slice(&mesh.vertices);
        indices.extend_from_slice(&mesh.indices);
        commands.push(EguiDrawCommand {
            scissor,
            first_index,
            index_count,
            vertex_offset,
        });
    }
    (vertices, indices, commands)
}

fn clip_rect_to_scissor(
    clip_rect: egui::Rect,
    pixels_per_point: f32,
    extent: vk::Extent2D,
) -> Option<vk::Rect2D> {
    let min_x = (clip_rect.min.x * pixels_per_point).floor().max(0.0) as u32;
    let min_y = (clip_rect.min.y * pixels_per_point).floor().max(0.0) as u32;
    let max_x = (clip_rect.max.x * pixels_per_point)
        .ceil()
        .min(extent.width as f32) as u32;
    let max_y = (clip_rect.max.y * pixels_per_point)
        .ceil()
        .min(extent.height as f32) as u32;
    if max_x <= min_x || max_y <= min_y {
        return None;
    }
    Some(vk::Rect2D {
        offset: vk::Offset2D {
            x: min_x as i32,
            y: min_y as i32,
        },
        extent: vk::Extent2D {
            width: max_x - min_x,
            height: max_y - min_y,
        },
    })
}

fn create_sampler(device: &ash::Device) -> Result<vk::Sampler> {
    let create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .max_lod(0.0);
    unsafe { device.create_sampler(&create_info, None) }.context("failed to create egui sampler")
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .context("failed to create egui descriptor set layout")
}

fn create_descriptor_pool(device: &ash::Device) -> Result<vk::DescriptorPool> {
    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::SAMPLER,
            descriptor_count: 1,
        },
    ];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    unsafe { device.create_descriptor_pool(&create_info, None) }
        .context("failed to create egui descriptor pool")
}

fn allocate_descriptor_set(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet> {
    let layouts = [descriptor_set_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);
    let mut sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .context("failed to allocate egui descriptor set")?;
    sets.pop()
        .ok_or_else(|| anyhow!("Vulkan returned no egui descriptor sets"))
}

fn create_pipeline(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
    swapchain_format: vk::Format,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vert_module = create_shader_module(device, EGUI_VERTEX_SPV)
        .context("failed to create egui vertex shader module")?;
    let frag_module = match create_shader_module(device, EGUI_FRAGMENT_SPV) {
        Ok(module) => module,
        Err(error) => {
            unsafe { device.destroy_shader_module(vert_module, None) };
            return Err(error).context("failed to create egui fragment shader module");
        }
    };

    let push_constant_ranges = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::VERTEX,
        offset: 0,
        size: std::mem::size_of::<EguiPushConstants>() as u32,
    }];
    let set_layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = match unsafe { device.create_pipeline_layout(&layout_info, None) } {
        Ok(layout) => layout,
        Err(error) => {
            unsafe {
                device.destroy_shader_module(frag_module, None);
                device.destroy_shader_module(vert_module, None);
            }
            return Err(error).context("failed to create egui pipeline layout");
        }
    };

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];
    let binding_descriptions = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let attribute_descriptions = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: std::mem::offset_of!(Vertex, pos) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: std::mem::offset_of!(Vertex, uv) as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R8G8B8A8_UNORM,
            offset: std::mem::offset_of!(Vertex, color) as u32,
        },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::ONE)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(
            vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        );
    let color_blend_attachments = [color_blend_attachment];
    let color_blend =
        vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [swapchain_format];
    let mut rendering_info =
        vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(pipeline_layout)
        .push_next(&mut rendering_info);

    let pipeline = match unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    } {
        Ok(mut pipelines) => pipelines
            .pop()
            .ok_or_else(|| anyhow!("Vulkan returned no egui graphics pipelines"))?,
        Err((pipelines, error)) => {
            unsafe {
                for pipeline in pipelines {
                    device.destroy_pipeline(pipeline, None);
                }
                device.destroy_pipeline_layout(pipeline_layout, None);
                device.destroy_shader_module(frag_module, None);
                device.destroy_shader_module(vert_module, None);
            }
            return Err(error).context("failed to create egui graphics pipeline");
        }
    };
    unsafe {
        device.destroy_shader_module(frag_module, None);
        device.destroy_shader_module(vert_module, None);
    }
    Ok((pipeline_layout, pipeline))
}

fn create_font_image(
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
            format: vk::Format::R8G8B8A8_UNORM,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect: vk::ImageAspectFlags::COLOR,
            name: "egui font atlas",
        },
    )
}

fn transition_image(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_stage, src_access) = match old_layout {
        vk::ImageLayout::UNDEFINED => (
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
        ),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    };
    let (dst_stage, dst_access) = match new_layout {
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::AccessFlags::SHADER_READ,
        ),
        _ => (
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
        ),
    };
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

fn image_data_to_rgba(image: &ImageData) -> Vec<u8> {
    match image {
        ImageData::Color(color) => color.as_raw().to_vec(),
        ImageData::Font(font) => font
            .srgba_pixels(None)
            .flat_map(|pixel| pixel.to_array())
            .collect(),
    }
}

fn ensure_host_buffer(
    buffer: &mut Option<GpuBuffer>,
    device: &ash::Device,
    allocator: &GpuAllocator,
    required_size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    name: &'static str,
) -> Result<()> {
    let required_size = required_size.max(4);
    let recreate = buffer
        .as_ref()
        .is_none_or(|buffer| buffer.size < required_size || !buffer.usage.contains(usage));
    if recreate {
        if let Some(old) = buffer.take() {
            old.destroy(device, allocator);
        }
        *buffer = Some(GpuBuffer::new(
            device,
            allocator,
            required_size.next_power_of_two(),
            usage,
            MemoryLocation::CpuToGpu,
            name,
        )?);
    }
    Ok(())
}

fn write_buffer_bytes(buffer: &GpuBuffer, bytes: &[u8]) -> Result<()> {
    let ptr = buffer
        .mapped_ptr()
        .ok_or_else(|| anyhow!("egui host buffer is not mapped"))?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_egui_frame_reports_empty() {
        let frame = EguiFrame {
            clipped_primitives: Vec::new(),
            textures_delta: TexturesDelta::default(),
            pixels_per_point: 1.0,
        };

        assert!(frame.is_empty());
    }

    #[test]
    fn clip_rect_to_scissor_clamps_to_swapchain_extent() {
        let scissor = clip_rect_to_scissor(
            egui::Rect::from_min_max(egui::pos2(-10.0, 4.0), egui::pos2(50.5, 80.25)),
            2.0,
            vk::Extent2D {
                width: 96,
                height: 120,
            },
        )
        .expect("clip should intersect extent");

        assert_eq!(scissor.offset.x, 0);
        assert_eq!(scissor.offset.y, 8);
        assert_eq!(scissor.extent.width, 96);
        assert_eq!(scissor.extent.height, 112);
    }

    #[test]
    fn egui_vertex_shader_maps_top_left_ui_space_to_vulkan_top_left_ndc() {
        let source = crate::render::source_checks::read_source("assets/shaders/ui/egui.vert.slang");
        let compact = crate::render::source_checks::compact(&source);

        assert!(
            compact.contains("input.pos.y/egui_push.screen_size_points.y*2.0-1.0"),
            "egui y=0 must map to Vulkan NDC y=-1 so the UI is not vertically flipped"
        );
        assert!(
            !compact.contains("1.0-input.pos.y/egui_push.screen_size_points.y*2.0"),
            "OpenGL-style y inversion flips egui under Vulkan's positive-height viewport"
        );
    }

    #[test]
    fn egui_renderer_source_owns_required_vulkan_resources() {
        let source = crate::render::source_checks::read_source("src/render/egui_renderer.rs");
        let renderer_struct = source
            .split("pub struct EguiRenderer")
            .nth(1)
            .expect("EguiRenderer struct should exist")
            .split("impl EguiRenderer")
            .next()
            .expect("EguiRenderer struct should end before impl");

        for token in [
            "sampler: vk::Sampler",
            "descriptor_set_layout: vk::DescriptorSetLayout",
            "descriptor_pool: vk::DescriptorPool",
            "pipeline_layout: vk::PipelineLayout",
            "pipeline: vk::Pipeline",
            "vertex_buffers: Vec<Option<GpuBuffer>>",
            "index_buffers: Vec<Option<GpuBuffer>>",
        ] {
            assert!(
                renderer_struct.contains(token),
                "EguiRenderer must own Vulkan resource {token}"
            );
        }
    }
}
