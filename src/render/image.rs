use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;

pub struct GpuImage {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub allocation: Option<Allocation>,
    pub current_layout: vk::ImageLayout,
}

#[derive(Clone)]
pub struct GpuImageDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
    pub name: &'static str,
}

impl GpuImage {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        desc: &GpuImageDesc,
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width: desc.width,
            height: desc.height,
            depth: desc.depth,
        };

        let image_type = if desc.depth > 1 {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(image_type)
            .format(desc.format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let handle = unsafe { device.create_image(&image_info, None) }
            .context("failed to create image")?;

        let requirements = unsafe { device.get_image_memory_requirements(handle) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: desc.name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe { device.bind_image_memory(handle, allocation.memory(), allocation.offset()) }
            .context("failed to bind image memory")?;

        let view_type = if desc.depth > 1 {
            vk::ImageViewType::TYPE_3D
        } else {
            vk::ImageViewType::TYPE_2D
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(handle)
            .view_type(view_type)
            .format(desc.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(desc.aspect)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let view = unsafe { device.create_image_view(&view_info, None) }
            .context("failed to create image view")?;

        Ok(Self {
            handle,
            view,
            extent,
            format: desc.format,
            allocation: Some(allocation),
            current_layout: vk::ImageLayout::UNDEFINED,
        })
    }

    pub fn destroy(mut self, device: &ash::Device, allocator: &GpuAllocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.handle, None);
        }
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
    }
}
