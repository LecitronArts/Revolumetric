use anyhow::{Context, Result, anyhow};
use ash::{Device, vk};

#[derive(Debug, Clone)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct SwapchainManager {
    pub handle: vk::SwapchainKHR,
    pub format: vk::Format,
    pub color_space: vk::ColorSpaceKHR,
    pub extent: vk::Extent2D,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub image_layouts: Vec<vk::ImageLayout>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub width: u32,
    pub height: u32,
}

impl Default for SwapchainManager {
    fn default() -> Self {
        Self {
            handle: vk::SwapchainKHR::null(),
            format: vk::Format::UNDEFINED,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            extent: vk::Extent2D {
                width: 1280,
                height: 720,
            },
            images: Vec::new(),
            image_views: Vec::new(),
            image_layouts: Vec::new(),
            in_flight_fences: Vec::new(),
            width: 1280,
            height: 720,
        }
    }
}

impl SwapchainManager {
    pub fn new(
        device: &Device,
        swapchain_loader: &ash::khr::swapchain::Device,
        surface: vk::SurfaceKHR,
        support: &SwapchainSupport,
        graphics_queue_family_index: u32,
        present_queue_family_index: u32,
        desired_width: u32,
        desired_height: u32,
    ) -> Result<Self> {
        let surface_format = choose_surface_format(&support.formats)
            .ok_or_else(|| anyhow!("Vulkan surface reports no supported surface formats"))?;
        let present_mode = choose_present_mode(&support.present_modes);
        let extent = choose_extent(&support.capabilities, desired_width, desired_height);
        let image_count = choose_image_count(&support.capabilities);

        let queue_family_indices = [graphics_queue_family_index, present_queue_family_index];
        let (image_sharing_mode, queue_family_indices) =
            if graphics_queue_family_index == present_queue_family_index {
                (vk::SharingMode::EXCLUSIVE, &queue_family_indices[..1])
            } else {
                (vk::SharingMode::CONCURRENT, &queue_family_indices[..])
            };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(queue_family_indices)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let handle = unsafe { swapchain_loader.create_swapchain(&create_info, None) }
            .context("failed to create Vulkan swapchain")?;
        let images = unsafe { swapchain_loader.get_swapchain_images(handle) }
            .context("failed to fetch Vulkan swapchain images")?;
        let image_views = images
            .iter()
            .copied()
            .map(|image| create_image_view(device, image, surface_format.format))
            .collect::<Result<Vec<_>>>()?;
        let image_layouts = vec![vk::ImageLayout::UNDEFINED; images.len()];
        let in_flight_fences = vec![vk::Fence::null(); images.len()];

        Ok(Self {
            handle,
            format: surface_format.format,
            color_space: surface_format.color_space,
            extent,
            images,
            image_views,
            image_layouts,
            in_flight_fences,
            width: extent.width,
            height: extent.height,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    pub fn destroy(
        &mut self,
        device: &Device,
        swapchain_loader: &ash::khr::swapchain::Device,
    ) {
        unsafe {
            for image_view in self.image_views.drain(..) {
                device.destroy_image_view(image_view, None);
            }
            if self.handle != vk::SwapchainKHR::null() {
                swapchain_loader.destroy_swapchain(self.handle, None);
                self.handle = vk::SwapchainKHR::null();
            }
        }
        self.images.clear();
        self.image_layouts.clear();
        self.in_flight_fences.clear();
    }
}

fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> Option<vk::SurfaceFormatKHR> {
    formats
        .iter()
        .copied()
        .find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first().copied())
}

fn choose_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .copied()
        .find(|mode| *mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn choose_extent(
    capabilities: &vk::SurfaceCapabilitiesKHR,
    desired_width: u32,
    desired_height: u32,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }

    vk::Extent2D {
        width: desired_width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: desired_height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

fn choose_image_count(capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
    let desired = capabilities.min_image_count.saturating_add(1);
    if capabilities.max_image_count > 0 {
        desired.min(capabilities.max_image_count)
    } else {
        desired
    }
}

fn create_image_view(device: &Device, image: vk::Image, format: vk::Format) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range);

    unsafe { device.create_image_view(&create_info, None) }
        .context("failed to create Vulkan swapchain image view")
}
