use anyhow::{Context, Result, anyhow};
use ash::{Device, Entry, Instance, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};
use winit::window::Window;

use crate::render::allocator::GpuAllocator;
use crate::render::frame::FrameContext;
use crate::render::swapchain::{SwapchainManager, SwapchainSupport};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct FrameResources {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

pub struct RenderDevice {
    entry: Entry,
    instance: Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: Device,
    allocator: Option<GpuAllocator>,
    swapchain_loader: ash::khr::swapchain::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    graphics_queue_family_index: u32,
    present_queue_family_index: u32,
    physical_device_name: String,
    backend_name: &'static str,
    frame_index: u64,
    current_frame: usize,
    frames: Vec<FrameResources>,
    swapchain: SwapchainManager,
}

impl RenderDevice {
    pub fn new(window: &Window) -> Result<Self> {
        let entry = unsafe { Entry::load() }.context("failed to load Vulkan entry")?;
        let app_name = CString::new("Revolumetric")?;
        let engine_name = CString::new("Revolumetric")?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let display_handle = window
            .display_handle()
            .context("failed to acquire raw display handle")?;
        let window_handle = window
            .window_handle()
            .context("failed to acquire raw window handle")?;

        let extension_names = ash_window::enumerate_required_extensions(display_handle.as_raw())
            .context("failed to enumerate required Vulkan surface extensions")?;

        let layer_name = CString::new("VK_LAYER_KHRONOS_validation")?;
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .context("failed to enumerate Vulkan instance layers")?;
        let enabled_layers = if has_layer(&available_layers, layer_name.as_c_str()) {
            vec![layer_name.as_ptr()]
        } else {
            Vec::new()
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(extension_names)
            .enabled_layer_names(&enabled_layers);

        let instance = unsafe { entry.create_instance(&create_info, None) }
            .context("failed to create Vulkan instance")?;

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                display_handle.as_raw(),
                window_handle.as_raw(),
                None,
            )
        }
        .context("failed to create Vulkan surface")?;

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let size = window.inner_size();

        let device_extension_names = [ash::khr::swapchain::NAME.as_ptr()];
        let selection = pick_physical_device(
            &instance,
            &surface_loader,
            surface,
            &device_extension_names,
        )?;

        let queue_family_indices = if selection.graphics_queue_family_index
            == selection.present_queue_family_index
        {
            vec![selection.graphics_queue_family_index]
        } else {
            vec![
                selection.graphics_queue_family_index,
                selection.present_queue_family_index,
            ]
        };

        let queue_priorities = [1.0_f32];
        let queue_create_infos = queue_family_indices
            .iter()
            .map(|&queue_family_index| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&queue_priorities)
            })
            .collect::<Vec<_>>();

        let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
            .buffer_device_address(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extension_names)
            .push_next(&mut bda_features);

        let device = unsafe {
            instance.create_device(selection.physical_device, &device_create_info, None)
        }
        .context("failed to create logical Vulkan device")?;

        let allocator = GpuAllocator::new(&instance, &device, selection.physical_device)?;

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let graphics_queue = unsafe {
            device.get_device_queue(selection.graphics_queue_family_index, 0)
        };
        let present_queue = unsafe {
            device.get_device_queue(selection.present_queue_family_index, 0)
        };
        let swapchain_support = query_swapchain_support(
            &surface_loader,
            surface,
            selection.physical_device,
        )?;
        let swapchain = SwapchainManager::new(
            &device,
            &swapchain_loader,
            surface,
            &swapchain_support,
            selection.graphics_queue_family_index,
            selection.present_queue_family_index,
            size.width.max(1),
            size.height.max(1),
        )?;

        let frames = create_frame_resources(&device, selection.graphics_queue_family_index)?;

        Ok(Self {
            entry,
            instance,
            surface_loader,
            surface,
            physical_device: selection.physical_device,
            device,
            allocator: Some(allocator),
            swapchain_loader,
            graphics_queue,
            present_queue,
            graphics_queue_family_index: selection.graphics_queue_family_index,
            present_queue_family_index: selection.present_queue_family_index,
            physical_device_name: selection.device_name,
            backend_name: "vulkan-bootstrap",
            frame_index: 0,
            current_frame: 0,
            frames,
            swapchain,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend_name
    }

    pub fn physical_device_name(&self) -> &str {
        &self.physical_device_name
    }

    pub fn graphics_queue_family_index(&self) -> u32 {
        self.graphics_queue_family_index
    }

    pub fn present_queue_family_index(&self) -> u32 {
        self.present_queue_family_index
    }

    pub fn swapchain_format(&self) -> vk::Format {
        self.swapchain.format
    }

    pub fn swapchain_image_count(&self) -> usize {
        self.swapchain.images.len()
    }

    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    pub fn handle_resize(&mut self, width: u32, height: u32) -> Result<()> {
        self.swapchain.resize(width, height);
        self.recreate_swapchain()
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        unsafe {
            self.device
                .device_wait_idle()
                .context("failed to idle Vulkan device before swapchain recreation")?;
        }

        self.swapchain.destroy(&self.device, &self.swapchain_loader);

        let support = query_swapchain_support(
            &self.surface_loader,
            self.surface,
            self.physical_device,
        )?;
        self.swapchain = SwapchainManager::new(
            &self.device,
            &self.swapchain_loader,
            self.surface,
            &support,
            self.graphics_queue_family_index,
            self.present_queue_family_index,
            self.swapchain.width,
            self.swapchain.height,
        )?;

        Ok(())
    }

    pub fn begin_frame(&mut self) -> Result<FrameContext> {
        let frame_slot = self.current_frame;
        let frame_resources = &self.frames[frame_slot];
        let command_pool = frame_resources.command_pool;
        let command_buffer = frame_resources.command_buffer;
        let image_available_semaphore = frame_resources.image_available_semaphore;
        let render_finished_semaphore = frame_resources.render_finished_semaphore;
        let in_flight_fence = frame_resources.in_flight_fence;

        unsafe {
            self.device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)
                .context("failed to wait for Vulkan in-flight fence")?;
            self.device
                .reset_fences(&[in_flight_fence])
                .context("failed to reset Vulkan in-flight fence")?;
            self.device
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                .context("failed to reset Vulkan command pool")?;
        }

        let (image_index, suboptimal) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                image_available_semaphore,
                vk::Fence::null(),
            )
        } {
            Ok(result) => result,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain()?;
                return Ok(FrameContext::skip(self.frame_index));
            }
            Err(error) => {
                return Err(anyhow!("failed to acquire Vulkan swapchain image: {error:?}"));
            }
        };

        let image_index = image_index as usize;
        let image_fence = self.swapchain.in_flight_fences[image_index];
        if image_fence != vk::Fence::null() && image_fence != in_flight_fence {
            unsafe {
                self.device
                    .wait_for_fences(&[image_fence], true, u64::MAX)
                    .context("failed to wait for Vulkan swapchain image fence")?;
            }
        }
        self.swapchain.in_flight_fences[image_index] = in_flight_fence;

        unsafe {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .context("failed to begin Vulkan command buffer")?;
        }

        self.frame_index += 1;
        self.current_frame = (self.current_frame + 1) % self.frames.len();

        Ok(FrameContext {
            frame_index: self.frame_index,
            should_render: true,
            command_buffer,
            swapchain_image: self.swapchain.images[image_index],
            swapchain_image_index: image_index,
            swapchain_extent: self.swapchain.extent,
            swapchain_format: self.swapchain.format,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            suboptimal,
        })
    }

    pub fn end_frame(&mut self, ctx: FrameContext) -> Result<()> {
        if !ctx.should_render {
            return Ok(());
        }

        unsafe {
            self.device
                .end_command_buffer(ctx.command_buffer)
                .context("failed to end Vulkan command buffer")?;

            let wait_semaphores = [ctx.image_available_semaphore];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::TRANSFER];
            let command_buffers = [ctx.command_buffer];
            let signal_semaphores = [ctx.render_finished_semaphore];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);
            self.device
                .queue_submit(self.graphics_queue, &[submit_info], ctx.in_flight_fence)
                .context("failed to submit Vulkan command buffer")?;

            let present_wait_semaphores = [ctx.render_finished_semaphore];
            let swapchains = [self.swapchain.handle];
            let image_indices = [ctx.swapchain_image_index as u32];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&present_wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            match self
                .swapchain_loader
                .queue_present(self.present_queue, &present_info)
            {
                Ok(is_suboptimal) => {
                    if is_suboptimal || ctx.suboptimal {
                        self.recreate_swapchain()?;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain()?;
                }
                Err(error) => {
                    return Err(anyhow!("failed to present Vulkan swapchain image: {error:?}"));
                }
            }
        }

        self.swapchain.image_layouts[ctx.swapchain_image_index] = vk::ImageLayout::PRESENT_SRC_KHR;

        tracing::trace!(
            frame_index = ctx.frame_index,
            image_index = ctx.swapchain_image_index,
            "completed frame"
        );

        Ok(())
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn instance(&self) -> &Instance {
        &self.instance
    }

    pub fn allocator(&self) -> &GpuAllocator {
        self.allocator.as_ref().expect("allocator already dropped")
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    pub fn present_queue(&self) -> vk::Queue {
        self.present_queue
    }
}

impl Drop for RenderDevice {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // Allocator must be dropped before the device is destroyed
            drop(self.allocator.take());
            destroy_frame_resources(&self.device, &mut self.frames);
            self.swapchain.destroy(&self.device, &self.swapchain_loader);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

fn create_frame_resources(device: &Device, queue_family_index: u32) -> Result<Vec<FrameResources>> {
    (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| create_single_frame_resources(device, queue_family_index))
        .collect()
}

fn create_single_frame_resources(device: &Device, queue_family_index: u32) -> Result<FrameResources> {
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }
        .context("failed to create Vulkan command pool")?;

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
        .context("failed to allocate Vulkan command buffer")?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("Vulkan returned no command buffers"))?;

    let semaphore_info = vk::SemaphoreCreateInfo::default();
    let image_available_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }
        .context("failed to create Vulkan image-available semaphore")?;
    let render_finished_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }
        .context("failed to create Vulkan render-finished semaphore")?;

    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight_fence = unsafe { device.create_fence(&fence_info, None) }
        .context("failed to create Vulkan in-flight fence")?;

    Ok(FrameResources {
        command_pool,
        command_buffer,
        image_available_semaphore,
        render_finished_semaphore,
        in_flight_fence,
    })
}

fn destroy_frame_resources(device: &Device, frames: &mut Vec<FrameResources>) {
    unsafe {
        for frame in frames.drain(..) {
            device.destroy_fence(frame.in_flight_fence, None);
            device.destroy_semaphore(frame.render_finished_semaphore, None);
            device.destroy_semaphore(frame.image_available_semaphore, None);
            device.destroy_command_pool(frame.command_pool, None);
        }
    }
}

fn transition_swapchain_image(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    src_access_mask: vk::AccessFlags,
    dst_access_mask: vk::AccessFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        )
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}

struct PhysicalDeviceSelection {
    physical_device: vk::PhysicalDevice,
    graphics_queue_family_index: u32,
    present_queue_family_index: u32,
    device_name: String,
}

fn pick_physical_device(
    instance: &Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    required_extensions: &[*const i8],
) -> Result<PhysicalDeviceSelection> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .context("failed to enumerate Vulkan physical devices")?;

    physical_devices
        .into_iter()
        .find_map(|physical_device| {
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            let device_name = vk_cstr_to_string(&properties.device_name);

            match query_queue_families(instance, surface_loader, surface, physical_device)
                .and_then(|queue_families| {
                    ensure_required_device_extensions(instance, physical_device, required_extensions)?;
                    Ok(PhysicalDeviceSelection {
                        physical_device,
                        graphics_queue_family_index: queue_families.graphics_queue_family_index,
                        present_queue_family_index: queue_families.present_queue_family_index,
                        device_name,
                    })
                }) {
                Ok(selection) => Some(selection),
                Err(error) => {
                    tracing::debug!(%error, "skipping unsupported Vulkan physical device");
                    None
                }
            }
        })
        .ok_or_else(|| anyhow!("failed to find a Vulkan physical device with graphics+present support"))
}

struct QueueFamilySelection {
    graphics_queue_family_index: u32,
    present_queue_family_index: u32,
}

fn query_queue_families(
    instance: &Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<QueueFamilySelection> {
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let mut graphics_queue_family_index = None;
    let mut present_queue_family_index = None;

    for (index, queue_family) in queue_families.iter().enumerate() {
        let index = index as u32;

        if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics_queue_family_index.get_or_insert(index);
        }

        let supports_present = unsafe {
            surface_loader.get_physical_device_surface_support(physical_device, index, surface)
        }
        .context("failed to query present support for queue family")?;

        if supports_present {
            present_queue_family_index.get_or_insert(index);
        }

        if graphics_queue_family_index.is_some() && present_queue_family_index.is_some() {
            break;
        }
    }

    match (graphics_queue_family_index, present_queue_family_index) {
        (Some(graphics_queue_family_index), Some(present_queue_family_index)) => {
            Ok(QueueFamilySelection {
                graphics_queue_family_index,
                present_queue_family_index,
            })
        }
        _ => Err(anyhow!(
            "physical device is missing required graphics/present queue families"
        )),
    }
}

fn query_swapchain_support(
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<SwapchainSupport> {
    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }
    .context("failed to query Vulkan surface capabilities")?;
    let formats = unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
        .context("failed to query Vulkan surface formats")?;
    let present_modes = unsafe {
        surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
    }
    .context("failed to query Vulkan present modes")?;

    Ok(SwapchainSupport {
        capabilities,
        formats,
        present_modes,
    })
}

fn ensure_required_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    required_extensions: &[*const i8],
) -> Result<()> {
    let available_extensions = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .context("failed to enumerate Vulkan device extensions")?;

    let available_extension_names = available_extensions
        .iter()
        .map(|extension| unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) })
        .collect::<BTreeSet<_>>();

    for &required_extension in required_extensions {
        let required_extension = unsafe { CStr::from_ptr(required_extension) };
        if !available_extension_names.contains(required_extension) {
            return Err(anyhow!(
                "missing required Vulkan device extension: {}",
                required_extension.to_string_lossy()
            ));
        }
    }

    Ok(())
}

fn has_layer(available_layers: &[vk::LayerProperties], target: &CStr) -> bool {
    available_layers.iter().any(|layer| {
        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        name == target
    })
}

fn vk_cstr_to_string(raw_name: &[i8]) -> String {
    let name = unsafe { CStr::from_ptr(raw_name.as_ptr()) };
    name.to_string_lossy().into_owned()
}
