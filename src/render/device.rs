use anyhow::{Context, Result, anyhow};
use ash::{Device, Entry, Instance, vk};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::BTreeSet;
use std::ffi::{CStr, CString, c_char};
use winit::window::Window;

use crate::render::allocator::GpuAllocator;
use crate::render::frame::{FrameCompletion, FrameContext};
use crate::render::rt_capabilities::{RtCapabilities, probe_rt_capabilities};
use crate::render::swapchain::{SwapchainManager, SwapchainSupport};

// ash 0.38 is generated from Vulkan 1.3.281 and only exposes the older NV
// Rust names, but current Vulkan headers alias the NV feature struct/sType to
// VK_KHR_compute_shader_derivatives. Request the vendor-neutral extension name.
const KHR_COMPUTE_SHADER_DERIVATIVES_NAME: &CStr = c"VK_KHR_compute_shader_derivatives";

struct FrameResources {
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

struct RenderDeviceConstructionCleanup {
    instance: Option<Instance>,
    surface_loader: Option<ash::khr::surface::Instance>,
    surface: Option<vk::SurfaceKHR>,
    device: Option<Device>,
    swapchain_loader: Option<ash::khr::swapchain::Device>,
    swapchain: Option<SwapchainManager>,
}

impl RenderDeviceConstructionCleanup {
    fn new(
        instance: Instance,
        surface_loader: ash::khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Self {
        Self {
            instance: Some(instance),
            surface_loader: Some(surface_loader),
            surface: Some(surface),
            device: None,
            swapchain_loader: None,
            swapchain: None,
        }
    }

    fn instance(&self) -> &Instance {
        self.instance
            .as_ref()
            .expect("RenderDevice construction instance should exist")
    }

    fn surface_loader(&self) -> &ash::khr::surface::Instance {
        self.surface_loader
            .as_ref()
            .expect("RenderDevice construction surface loader should exist")
    }

    fn surface(&self) -> vk::SurfaceKHR {
        self.surface
            .expect("RenderDevice construction surface should exist")
    }

    fn device(&self) -> &Device {
        self.device
            .as_ref()
            .expect("RenderDevice construction device should exist")
    }

    fn swapchain_loader(&self) -> &ash::khr::swapchain::Device {
        self.swapchain_loader
            .as_ref()
            .expect("RenderDevice construction swapchain loader should exist")
    }

    fn swapchain(&self) -> &SwapchainManager {
        self.swapchain
            .as_ref()
            .expect("RenderDevice construction swapchain should exist")
    }

    fn set_device(&mut self, device: Device) {
        self.device = Some(device);
    }

    fn set_swapchain_loader(&mut self, swapchain_loader: ash::khr::swapchain::Device) {
        self.swapchain_loader = Some(swapchain_loader);
    }

    fn set_swapchain(&mut self, swapchain: SwapchainManager) {
        self.swapchain = Some(swapchain);
    }

    fn finish(
        mut self,
    ) -> (
        Instance,
        ash::khr::surface::Instance,
        vk::SurfaceKHR,
        Device,
        ash::khr::swapchain::Device,
        SwapchainManager,
    ) {
        (
            self.instance.take().expect("RenderDevice instance missing"),
            self.surface_loader
                .take()
                .expect("RenderDevice surface loader missing"),
            self.surface.take().expect("RenderDevice surface missing"),
            self.device.take().expect("RenderDevice device missing"),
            self.swapchain_loader
                .take()
                .expect("RenderDevice swapchain loader missing"),
            self.swapchain
                .take()
                .expect("RenderDevice swapchain missing"),
        )
    }
}

impl Drop for RenderDeviceConstructionCleanup {
    fn drop(&mut self) {
        unsafe {
            if let (Some(swapchain), Some(device), Some(swapchain_loader)) = (
                self.swapchain.as_mut(),
                self.device.as_ref(),
                self.swapchain_loader.as_ref(),
            ) {
                swapchain.destroy(device, swapchain_loader);
            }
            if let Some(device) = self.device.take() {
                device.destroy_device(None);
            }
            if let (Some(surface), Some(surface_loader)) =
                (self.surface.take(), self.surface_loader.as_ref())
            {
                surface_loader.destroy_surface(surface, None);
            }
            if let Some(instance) = self.instance.take() {
                instance.destroy_instance(None);
            }
        }
    }
}

struct FrameResourcesCreationCleanup<'a> {
    device: &'a Device,
    command_pool: Option<vk::CommandPool>,
    image_available_semaphore: Option<vk::Semaphore>,
    render_finished_semaphore: Option<vk::Semaphore>,
    in_flight_fence: Option<vk::Fence>,
}

impl<'a> FrameResourcesCreationCleanup<'a> {
    fn new(device: &'a Device) -> Self {
        Self {
            device,
            command_pool: None,
            image_available_semaphore: None,
            render_finished_semaphore: None,
            in_flight_fence: None,
        }
    }

    fn set_command_pool(&mut self, command_pool: vk::CommandPool) {
        self.command_pool = Some(command_pool);
    }

    fn set_image_available_semaphore(&mut self, semaphore: vk::Semaphore) {
        self.image_available_semaphore = Some(semaphore);
    }

    fn set_render_finished_semaphore(&mut self, semaphore: vk::Semaphore) {
        self.render_finished_semaphore = Some(semaphore);
    }

    fn set_in_flight_fence(&mut self, fence: vk::Fence) {
        self.in_flight_fence = Some(fence);
    }

    fn disarm(&mut self) {
        self.command_pool = None;
        self.image_available_semaphore = None;
        self.render_finished_semaphore = None;
        self.in_flight_fence = None;
    }
}

impl Drop for FrameResourcesCreationCleanup<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Some(fence) = self.in_flight_fence.take() {
                self.device.destroy_fence(fence, None);
            }
            if let Some(semaphore) = self.render_finished_semaphore.take() {
                self.device.destroy_semaphore(semaphore, None);
            }
            if let Some(semaphore) = self.image_available_semaphore.take() {
                self.device.destroy_semaphore(semaphore, None);
            }
            if let Some(command_pool) = self.command_pool.take() {
                self.device.destroy_command_pool(command_pool, None);
            }
        }
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
enum FramePreparationStep {
    WaitFence,
    AcquireImage,
    ResetFence,
    ResetCommandPool,
}

#[cfg(test)]
fn frame_preparation_order(acquire_succeeds: bool) -> &'static [FramePreparationStep] {
    if acquire_succeeds {
        &[
            FramePreparationStep::WaitFence,
            FramePreparationStep::AcquireImage,
            FramePreparationStep::ResetFence,
            FramePreparationStep::ResetCommandPool,
        ]
    } else {
        &[
            FramePreparationStep::WaitFence,
            FramePreparationStep::AcquireImage,
        ]
    }
}

pub struct RenderDevice {
    // Keeps the dynamically-loaded Vulkan loader alive for all instance/device calls.
    _entry: Entry,
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
    rt_capabilities: RtCapabilities,
    acceleration_structure_loader: Option<ash::khr::acceleration_structure::Device>,
    ray_tracing_pipeline_loader: Option<ash::khr::ray_tracing_pipeline::Device>,
    rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    acceleration_structure_properties:
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
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

        // Determine desired swapchain size. On Android, default to a 720p
        // render target to save performance unless explicitly disabled via
        // REVOLUMETRIC_ANDROID_FORCE_720P=0 in the environment.
        let (desired_width, desired_height) = {
            let w = size.width.max(1);
            let h = size.height.max(1);
            #[cfg(target_os = "android")]
            {
                let force_720p = std::env::var("REVOLUMETRIC_ANDROID_FORCE_720P")
                    .ok()
                    .as_deref()
                    != Some("0");
                if !force_720p {
                    (w, h)
                } else {
                    const MAX_W: u32 = 1280;
                    const MAX_H: u32 = 720;
                    if w <= MAX_W && h <= MAX_H {
                        (w, h)
                    } else {
                        let aspect = w as f32 / h as f32;
                        let target_aspect = MAX_W as f32 / MAX_H as f32;
                        if aspect >= target_aspect {
                            let hh = ((MAX_W as f32) / aspect).round() as u32;
                            (MAX_W, hh.max(1))
                        } else {
                            let ww = ((MAX_H as f32) * aspect).round() as u32;
                            (ww.max(1), MAX_H)
                        }
                    }
                }
            }
            #[cfg(not(target_os = "android"))]
            {
                (w, h)
            }
        };

        let mut cleanup = RenderDeviceConstructionCleanup::new(instance, surface_loader, surface);

        let device_extension_names = base_device_extension_names();
        let selection = pick_physical_device(
            cleanup.instance(),
            cleanup.surface_loader(),
            cleanup.surface(),
            &device_extension_names,
        )?;
        let rt_capabilities = probe_rt_capabilities(cleanup.instance(), selection.physical_device);
        let rt_pipeline_properties =
            query_rt_pipeline_properties(cleanup.instance(), selection.physical_device);
        let acceleration_structure_properties =
            query_acceleration_structure_properties(cleanup.instance(), selection.physical_device);
        let device_extension_names = device_extension_names_for_capabilities(rt_capabilities);

        let queue_family_indices =
            if selection.graphics_queue_family_index == selection.present_queue_family_index {
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

        let mut compute_derivatives_features =
            vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
                .compute_derivative_group_quads(true);
        let mut dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_float16(true);
        let mut acceleration_structure_features =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
                .acceleration_structure(rt_capabilities.supported());
        let mut ray_tracing_pipeline_features =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
                .ray_tracing_pipeline(rt_capabilities.supported());

        let physical_features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_extended_formats(true)
            .shader_int16(true);

        let mut device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&device_extension_names)
            .enabled_features(&physical_features)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut compute_derivatives_features)
            .push_next(&mut vulkan12_features);
        if rt_capabilities.supported() {
            device_create_info = device_create_info
                .push_next(&mut acceleration_structure_features)
                .push_next(&mut ray_tracing_pipeline_features);
        }

        let device = unsafe {
            cleanup
                .instance()
                .create_device(selection.physical_device, &device_create_info, None)
        }
        .context("failed to create logical Vulkan device")?;
        cleanup.set_device(device);

        let allocator = GpuAllocator::new(
            cleanup.instance(),
            cleanup.device(),
            selection.physical_device,
        )?;

        let swapchain_loader =
            ash::khr::swapchain::Device::new(cleanup.instance(), cleanup.device());
        cleanup.set_swapchain_loader(swapchain_loader);
        let graphics_queue = unsafe {
            cleanup
                .device()
                .get_device_queue(selection.graphics_queue_family_index, 0)
        };
        let present_queue = unsafe {
            cleanup
                .device()
                .get_device_queue(selection.present_queue_family_index, 0)
        };
        let swapchain_support = query_swapchain_support(
            cleanup.surface_loader(),
            cleanup.surface(),
            selection.physical_device,
        )?;
        let swapchain = SwapchainManager::new(
            cleanup.device(),
            cleanup.swapchain_loader(),
            cleanup.surface(),
            &swapchain_support,
            selection.graphics_queue_family_index,
            selection.present_queue_family_index,
            desired_width,
            desired_height,
        )?;
        cleanup.set_swapchain(swapchain);

        let frames = create_frame_resources(
            cleanup.device(),
            selection.graphics_queue_family_index,
            cleanup.swapchain().images.len(),
        )?;

        let (instance, surface_loader, surface, device, swapchain_loader, swapchain) =
            cleanup.finish();
        let acceleration_structure_loader = if rt_capabilities.supported() {
            Some(ash::khr::acceleration_structure::Device::new(
                &instance, &device,
            ))
        } else {
            None
        };
        let ray_tracing_pipeline_loader = if rt_capabilities.supported() {
            Some(ash::khr::ray_tracing_pipeline::Device::new(
                &instance, &device,
            ))
        } else {
            None
        };

        Ok(Self {
            _entry: entry,
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
            rt_capabilities,
            acceleration_structure_loader,
            ray_tracing_pipeline_loader,
            rt_pipeline_properties,
            acceleration_structure_properties,
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

        let support =
            query_swapchain_support(&self.surface_loader, self.surface, self.physical_device)?;
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
                return Ok(FrameContext::skip_after_swapchain_recreate(
                    self.frame_index,
                ));
            }
            Err(error) => {
                return Err(anyhow!(
                    "failed to acquire Vulkan swapchain image: {error:?}"
                ));
            }
        };

        unsafe {
            self.device
                .reset_fences(&[in_flight_fence])
                .context("failed to reset Vulkan in-flight fence")?;
            self.device
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                .context("failed to reset Vulkan command pool")?;
        }

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
            frame_slot,
            should_render: true,
            command_buffer,
            swapchain_image: self.swapchain.images[image_index],
            swapchain_image_view: self.swapchain.image_views[image_index],
            swapchain_image_index: image_index,
            swapchain_image_layout: self.swapchain.image_layouts[image_index],
            swapchain_extent: self.swapchain.extent,
            swapchain_format: self.swapchain.format,
            image_available_semaphore,
            render_finished_semaphore,
            in_flight_fence,
            suboptimal,
            swapchain_recreated: false,
        })
    }

    pub fn end_frame(&mut self, ctx: FrameContext) -> Result<FrameCompletion> {
        if !ctx.should_render {
            return Ok(FrameCompletion::default());
        }
        let mut completion = FrameCompletion::default();

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
                        completion.swapchain_recreated = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain()?;
                    completion.swapchain_recreated = true;
                }
                Err(error) => {
                    return Err(anyhow!(
                        "failed to present Vulkan swapchain image: {error:?}"
                    ));
                }
            }
        }

        if !completion.swapchain_recreated {
            self.swapchain.image_layouts[ctx.swapchain_image_index] =
                vk::ImageLayout::PRESENT_SRC_KHR;
        }

        tracing::trace!(
            frame_index = ctx.frame_index,
            image_index = ctx.swapchain_image_index,
            "completed frame"
        );

        Ok(completion)
    }

    pub fn wait_for_fence(&self, fence: vk::Fence) -> Result<()> {
        unsafe {
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .context("failed to wait for Vulkan fence")
        }
    }

    pub fn wait_for_other_frame_fences(&self, excluded_fence: vk::Fence) -> Result<()> {
        let fences = self
            .frames
            .iter()
            .map(|frame| frame.in_flight_fence)
            .filter(|&fence| fence != excluded_fence && fence != vk::Fence::null())
            .collect::<Vec<_>>();
        if fences.is_empty() {
            return Ok(());
        }
        unsafe {
            self.device
                .wait_for_fences(&fences, true, u64::MAX)
                .context("failed to wait for other Vulkan frame fences")
        }
    }

    pub fn wait_idle(&self) -> Result<()> {
        unsafe {
            self.device
                .device_wait_idle()
                .context("failed to idle Vulkan device")
        }
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

    pub fn frame_slot_count(&self) -> usize {
        self.frames.len()
    }

    pub fn physical_device_properties(&self) -> vk::PhysicalDeviceProperties {
        unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        }
    }

    pub fn rt_capabilities(&self) -> RtCapabilities {
        self.rt_capabilities
    }

    pub fn supports_rt(&self) -> bool {
        self.rt_capabilities.supported()
    }

    pub fn acceleration_structure_loader(
        &self,
    ) -> Option<&ash::khr::acceleration_structure::Device> {
        self.acceleration_structure_loader.as_ref()
    }

    pub fn ray_tracing_pipeline_loader(&self) -> Option<&ash::khr::ray_tracing_pipeline::Device> {
        self.ray_tracing_pipeline_loader.as_ref()
    }

    pub fn rt_pipeline_properties(
        &self,
    ) -> vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static> {
        self.rt_pipeline_properties
    }

    pub fn acceleration_structure_properties(
        &self,
    ) -> vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static> {
        self.acceleration_structure_properties
    }

    pub fn graphics_queue_timestamp_valid_bits(&self) -> u32 {
        unsafe {
            self.instance
                .get_physical_device_queue_family_properties(self.physical_device)
        }
        .get(self.graphics_queue_family_index as usize)
        .map_or(0, |properties| properties.timestamp_valid_bits)
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

fn create_frame_resources(
    device: &Device,
    queue_family_index: u32,
    count: usize,
) -> Result<Vec<FrameResources>> {
    let mut frames = Vec::with_capacity(count);
    for _ in 0..count {
        match create_single_frame_resources(device, queue_family_index) {
            Ok(frame) => frames.push(frame),
            Err(error) => {
                destroy_frame_resources(device, &mut frames);
                return Err(error);
            }
        }
    }
    Ok(frames)
}

fn create_single_frame_resources(
    device: &Device,
    queue_family_index: u32,
) -> Result<FrameResources> {
    let mut cleanup = FrameResourcesCreationCleanup::new(device);
    let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }
        .context("failed to create Vulkan command pool")?;
    cleanup.set_command_pool(command_pool);

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
    cleanup.set_image_available_semaphore(image_available_semaphore);
    let render_finished_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }
        .context("failed to create Vulkan render-finished semaphore")?;
    cleanup.set_render_finished_semaphore(render_finished_semaphore);

    let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    let in_flight_fence = unsafe { device.create_fence(&fence_info, None) }
        .context("failed to create Vulkan in-flight fence")?;
    cleanup.set_in_flight_fence(in_flight_fence);
    cleanup.disarm();

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
    required_extensions: &[*const c_char],
) -> Result<PhysicalDeviceSelection> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .context("failed to enumerate Vulkan physical devices")?;

    physical_devices
        .into_iter()
        .find_map(|physical_device| {
            let properties = unsafe { instance.get_physical_device_properties(physical_device) };
            let device_name = vk_cstr_to_string(&properties.device_name);

            match query_queue_families(instance, surface_loader, surface, physical_device).and_then(
                |queue_families| {
                    ensure_required_device_extensions(
                        instance,
                        physical_device,
                        required_extensions,
                    )?;
                    let (
                        features,
                        compute_derivatives_features,
                        dynamic_rendering_features,
                        vulkan12_features,
                    ) = query_required_device_features(instance, physical_device);
                    ensure_required_device_features(
                        &features,
                        &compute_derivatives_features,
                        &dynamic_rendering_features,
                        &vulkan12_features,
                    )?;
                    Ok(PhysicalDeviceSelection {
                        physical_device,
                        graphics_queue_family_index: queue_families.graphics_queue_family_index,
                        present_queue_family_index: queue_families.present_queue_family_index,
                        device_name,
                    })
                },
            ) {
                Ok(selection) => Some(selection),
                Err(error) => {
                    tracing::debug!(%error, "skipping unsupported Vulkan physical device");
                    None
                }
            }
        })
        .ok_or_else(|| {
            anyhow!("failed to find a Vulkan physical device with graphics+present support")
        })
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
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

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
    let formats =
        unsafe { surface_loader.get_physical_device_surface_formats(physical_device, surface) }
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
    required_extensions: &[*const c_char],
) -> Result<()> {
    let available_extensions =
        unsafe { instance.enumerate_device_extension_properties(physical_device) }
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

fn base_device_extension_names() -> Vec<*const c_char> {
    vec![
        ash::khr::swapchain::NAME.as_ptr(),
        KHR_COMPUTE_SHADER_DERIVATIVES_NAME.as_ptr(),
    ]
}

fn device_extension_names_for_capabilities(rt_capabilities: RtCapabilities) -> Vec<*const c_char> {
    let mut extension_names = base_device_extension_names();
    if rt_capabilities.supported() {
        extension_names.extend([
            ash::khr::deferred_host_operations::NAME.as_ptr(),
            ash::khr::acceleration_structure::NAME.as_ptr(),
            ash::khr::ray_tracing_pipeline::NAME.as_ptr(),
        ]);
    }
    extension_names
}

fn query_required_device_features(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> (
    vk::PhysicalDeviceFeatures,
    vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV<'static>,
    vk::PhysicalDeviceDynamicRenderingFeatures<'static>,
    vk::PhysicalDeviceVulkan12Features<'static>,
) {
    let mut compute_derivatives_features =
        vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default();
    let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default();
    let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut compute_derivatives_features)
        .push_next(&mut vulkan12_features);
    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features2);
    }
    (
        features2.features,
        compute_derivatives_features,
        dynamic_rendering_features,
        vulkan12_features,
    )
}

fn query_rt_pipeline_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static> {
    let mut rt_pipeline_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
    let mut properties2 =
        vk::PhysicalDeviceProperties2::default().push_next(&mut rt_pipeline_properties);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut properties2);
    }
    rt_pipeline_properties
}

fn query_acceleration_structure_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static> {
    let mut acceleration_structure_properties =
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut properties2 =
        vk::PhysicalDeviceProperties2::default().push_next(&mut acceleration_structure_properties);
    unsafe {
        instance.get_physical_device_properties2(physical_device, &mut properties2);
    }
    acceleration_structure_properties
}

fn ensure_required_device_features(
    features: &vk::PhysicalDeviceFeatures,
    compute_derivatives_features: &vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV<'_>,
    dynamic_rendering_features: &vk::PhysicalDeviceDynamicRenderingFeatures<'_>,
    vulkan12_features: &vk::PhysicalDeviceVulkan12Features<'_>,
) -> Result<()> {
    if features.shader_int16 == vk::FALSE {
        return Err(anyhow!("missing required Vulkan feature: shaderInt16"));
    }
    if features.shader_storage_image_extended_formats == vk::FALSE {
        return Err(anyhow!(
            "missing required Vulkan feature: shaderStorageImageExtendedFormats"
        ));
    }
    if compute_derivatives_features.compute_derivative_group_quads == vk::FALSE {
        return Err(anyhow!(
            "missing required Vulkan feature: computeDerivativeGroupQuads"
        ));
    }
    if dynamic_rendering_features.dynamic_rendering == vk::FALSE {
        return Err(anyhow!("missing required Vulkan feature: dynamicRendering"));
    }
    if vulkan12_features.buffer_device_address == vk::FALSE {
        return Err(anyhow!(
            "missing required Vulkan feature: bufferDeviceAddress"
        ));
    }
    if vulkan12_features.shader_float16 == vk::FALSE {
        return Err(anyhow!("missing required Vulkan feature: shaderFloat16"));
    }
    Ok(())
}

fn has_layer(available_layers: &[vk::LayerProperties], target: &CStr) -> bool {
    available_layers.iter().any(|layer| {
        let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
        name == target
    })
}

fn vk_cstr_to_string(raw_name: &[c_char]) -> String {
    let name = unsafe { CStr::from_ptr(raw_name.as_ptr()) };
    name.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_resources_reset_only_after_successful_acquire() {
        assert_eq!(
            frame_preparation_order(true),
            &[
                FramePreparationStep::WaitFence,
                FramePreparationStep::AcquireImage,
                FramePreparationStep::ResetFence,
                FramePreparationStep::ResetCommandPool,
            ]
        );
    }

    #[test]
    fn out_of_date_acquire_leaves_current_frame_fence_signaled() {
        assert_eq!(
            frame_preparation_order(false),
            &[
                FramePreparationStep::WaitFence,
                FramePreparationStep::AcquireImage,
            ]
        );
    }

    #[test]
    fn frame_resource_creation_cleans_up_partial_success_on_error() {
        let source = crate::render::source_checks::read_source("src/render/device.rs");

        let frame_resources = source
            .split("fn create_frame_resources(")
            .nth(1)
            .expect("create_frame_resources should exist")
            .split("fn create_single_frame_resources(")
            .next()
            .expect("create_frame_resources should end before create_single_frame_resources");
        let compact_frames = crate::render::source_checks::compact(frame_resources);
        assert!(compact_frames.contains("destroy_frame_resources(device,&mutframes);"));
        assert!(compact_frames.contains("frames.push(frame)"));

        let single_frame_resources = source
            .split("fn create_single_frame_resources(")
            .nth(1)
            .expect("create_single_frame_resources should exist")
            .split("fn destroy_frame_resources(")
            .next()
            .expect("create_single_frame_resources should end before destroy_frame_resources");
        let compact_single = crate::render::source_checks::compact(single_frame_resources);
        assert!(compact_single.contains("FrameResourcesCreationCleanup::new(device)"));
        assert!(compact_single.contains("cleanup.set_command_pool(command_pool)"));
        assert!(compact_single.contains("cleanup.set_image_available_semaphore"));
        assert!(compact_single.contains("cleanup.set_render_finished_semaphore"));
        assert!(compact_single.contains("cleanup.set_in_flight_fence"));
        assert!(compact_single.contains("cleanup.disarm()"));
    }

    #[test]
    fn render_device_constructor_uses_cleanup_guard_for_partially_initialized_resources() {
        let source = crate::render::source_checks::read_source("src/render/device.rs");
        let constructor = source
            .split("pub fn new(window: &Window) -> Result<Self>")
            .nth(1)
            .expect("RenderDevice::new should exist")
            .split("pub fn backend_name")
            .next()
            .expect("RenderDevice::new should end before backend_name");
        let compact = crate::render::source_checks::compact(constructor);

        assert!(compact.contains(
            "letmutcleanup=RenderDeviceConstructionCleanup::new(instance,surface_loader,surface);"
        ));
        assert!(compact.contains("cleanup.set_device(device);"));
        assert!(compact.contains("cleanup.set_swapchain_loader(swapchain_loader);"));
        assert!(compact.contains("cleanup.set_swapchain(swapchain);"));
        assert!(compact.contains(
            "let(instance,surface_loader,surface,device,swapchain_loader,swapchain)=cleanup.finish();"
        ));
    }

    #[test]
    fn device_reports_swapchain_recreation_from_frame_completion() {
        let source = crate::render::source_checks::read_source("src/render/device.rs");
        let end_frame = source
            .split("pub fn end_frame")
            .nth(1)
            .expect("RenderDevice::end_frame should exist")
            .split("pub fn wait_for_fence")
            .next()
            .expect("end_frame should end before wait_for_fence");

        assert!(end_frame.contains("Result<FrameCompletion>"));
        assert!(end_frame.contains("let mut completion = FrameCompletion::default();"));
        assert!(end_frame.contains("completion.swapchain_recreated = true;"));
        assert!(end_frame.contains("Ok(completion)"));
    }

    #[test]
    fn end_frame_does_not_write_old_image_layout_after_present_recreate() {
        let source = crate::render::source_checks::read_source("src/render/device.rs");
        let end_frame = source
            .split("pub fn end_frame")
            .nth(1)
            .expect("RenderDevice::end_frame should exist")
            .split("pub fn wait_for_fence")
            .next()
            .expect("end_frame should end before wait_for_fence");

        let compact = crate::render::source_checks::compact(end_frame);
        assert!(compact.contains(
            "if!completion.swapchain_recreated{self.swapchain.image_layouts[ctx.swapchain_image_index]=vk::ImageLayout::PRESENT_SRC_KHR;"
        ));
    }

    #[test]
    fn render_device_can_wait_for_other_frame_fences_before_shared_resource_mutation() {
        let source = crate::render::source_checks::read_source("src/render/device.rs");
        let compact = crate::render::source_checks::compact(&source);

        assert!(compact.contains(
            "pubfnwait_for_other_frame_fences(&self,excluded_fence:vk::Fence)->Result<()>"
        ));
        assert!(
            compact.contains(".filter(|&fence|fence!=excluded_fence&&fence!=vk::Fence::null())")
        );
        assert!(compact.contains(".wait_for_fences(&fences,true,u64::MAX)"));
    }

    #[test]
    fn required_device_features_accept_supported_features() {
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_extended_formats(true)
            .shader_int16(true);
        let compute_derivatives_features =
            vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
                .compute_derivative_group_quads(true);
        let dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_float16(true);

        ensure_required_device_features(
            &features,
            &compute_derivatives_features,
            &dynamic_rendering_features,
            &vulkan12_features,
        )
        .unwrap();
    }

    #[test]
    fn required_device_features_report_missing_bda() {
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_extended_formats(true)
            .shader_int16(true);
        let compute_derivatives_features =
            vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
                .compute_derivative_group_quads(true);
        let dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(false)
            .shader_float16(true);

        let error = ensure_required_device_features(
            &features,
            &compute_derivatives_features,
            &dynamic_rendering_features,
            &vulkan12_features,
        )
        .unwrap_err();

        assert!(error.to_string().contains("bufferDeviceAddress"));
    }

    #[test]
    fn required_device_features_report_missing_storage_image_extended_formats() {
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_extended_formats(false)
            .shader_int16(true);
        let compute_derivatives_features =
            vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
                .compute_derivative_group_quads(true);
        let dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_float16(true);

        let error = ensure_required_device_features(
            &features,
            &compute_derivatives_features,
            &dynamic_rendering_features,
            &vulkan12_features,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("shaderStorageImageExtendedFormats")
        );
    }

    #[test]
    fn required_device_features_report_missing_shader_int16() {
        let features =
            vk::PhysicalDeviceFeatures::default().shader_storage_image_extended_formats(true);
        let compute_derivatives_features =
            vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
                .compute_derivative_group_quads(true);
        let dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_float16(true);

        let error = ensure_required_device_features(
            &features,
            &compute_derivatives_features,
            &dynamic_rendering_features,
            &vulkan12_features,
        )
        .unwrap_err();

        assert!(error.to_string().contains("shaderInt16"));
    }

    #[test]
    fn required_device_features_report_missing_compute_derivative_quads() {
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_extended_formats(true)
            .shader_int16(true);
        let compute_derivatives_features =
            vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
                .compute_derivative_group_quads(false);
        let dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_float16(true);

        let error = ensure_required_device_features(
            &features,
            &compute_derivatives_features,
            &dynamic_rendering_features,
            &vulkan12_features,
        )
        .unwrap_err();

        assert!(error.to_string().contains("computeDerivativeGroupQuads"));
    }

    #[test]
    fn required_device_features_report_missing_dynamic_rendering() {
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_extended_formats(true)
            .shader_int16(true);
        let compute_derivatives_features =
            vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
                .compute_derivative_group_quads(true);
        let dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(false);
        let vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_float16(true);

        let error = ensure_required_device_features(
            &features,
            &compute_derivatives_features,
            &dynamic_rendering_features,
            &vulkan12_features,
        )
        .unwrap_err();

        assert!(error.to_string().contains("dynamicRendering"));
    }

    #[test]
    fn required_device_features_report_missing_shader_float16() {
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_storage_image_extended_formats(true)
            .shader_int16(true);
        let compute_derivatives_features =
            vk::PhysicalDeviceComputeShaderDerivativesFeaturesNV::default()
                .compute_derivative_group_quads(true);
        let dynamic_rendering_features =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_float16(false);

        let error = ensure_required_device_features(
            &features,
            &compute_derivatives_features,
            &dynamic_rendering_features,
            &vulkan12_features,
        )
        .unwrap_err();

        assert!(error.to_string().contains("shaderFloat16"));
    }

    #[test]
    fn device_extensions_append_rt_dependencies_only_when_fully_supported() {
        let supported = RtCapabilities {
            acceleration_structure: true,
            ray_tracing_pipeline: true,
            deferred_host_operations: true,
            buffer_device_address: true,
        };
        let unsupported = RtCapabilities {
            acceleration_structure: false,
            ..supported
        };

        let supported_names =
            extension_name_strings(&device_extension_names_for_capabilities(supported));
        let unsupported_names =
            extension_name_strings(&device_extension_names_for_capabilities(unsupported));

        assert_eq!(
            unsupported_names,
            extension_name_strings(&base_device_extension_names())
        );
        for required in [
            ash::khr::acceleration_structure::NAME,
            ash::khr::ray_tracing_pipeline::NAME,
            ash::khr::deferred_host_operations::NAME,
        ] {
            assert!(
                supported_names.contains(&required.to_string_lossy().into_owned()),
                "RT-capable devices must enable required extension {}",
                required.to_string_lossy()
            );
            assert!(
                !unsupported_names.contains(&required.to_string_lossy().into_owned()),
                "VPT-only devices must not request optional RT extension {}",
                required.to_string_lossy()
            );
        }
    }

    #[test]
    fn device_exposes_acceleration_structure_loader_for_rt_backend() {
        let source = crate::render::source_checks::read_source("src/render/device.rs");

        assert!(source.contains(
            "acceleration_structure_loader: Option<ash::khr::acceleration_structure::Device>"
        ));
        assert!(source.contains(
            "pub fn acceleration_structure_loader(&self) -> Option<&ash::khr::acceleration_structure::Device>"
        ));
        assert!(source.contains("ash::khr::acceleration_structure::Device::new"));
    }

    #[test]
    fn device_creation_enables_16_bit_shader_features() {
        let source = crate::render::source_checks::read_source("src/render/device.rs");
        let constructor = source
            .split("pub fn new(window: &Window) -> Result<Self>")
            .nth(1)
            .expect("RenderDevice::new should exist")
            .split("let allocator = GpuAllocator::new")
            .next()
            .expect("device creation should precede allocator creation");
        let compact = crate::render::source_checks::compact(constructor);

        assert!(compact.contains("PhysicalDeviceVulkan12Features"));
        assert!(compact.contains(".buffer_device_address(true)"));
        assert!(compact.contains(".shader_float16(true)"));
        assert!(compact.contains(".shader_int16(true)"));
        assert!(compact.contains(".push_next(&mutvulkan12_features)"));
        assert!(!compact.contains("PhysicalDeviceBufferDeviceAddressFeatures::default()"));
    }

    #[test]
    fn required_device_extensions_include_khr_compute_shader_derivatives_for_nrd_spirv() {
        let source = crate::render::source_checks::read_source("src/render/device.rs");
        let device_extensions = source
            .split("fn base_device_extension_names()")
            .nth(1)
            .expect("base device extension list should exist")
            .split("fn device_extension_names_for_capabilities")
            .next()
            .expect("base device extension list should end before RT extension helper");

        assert!(device_extensions.contains("KHR_COMPUTE_SHADER_DERIVATIVES_NAME"));
        assert!(source.contains("VK_KHR_compute_shader_derivatives"));
    }

    #[test]
    fn device_creation_enables_dynamic_rendering_for_graphics_overlay_passes() {
        let source = crate::render::source_checks::read_source("src/render/device.rs");
        let constructor = source
            .split("pub fn new(window: &Window) -> Result<Self>")
            .nth(1)
            .expect("RenderDevice::new should exist")
            .split("let allocator = GpuAllocator::new")
            .next()
            .expect("device creation should precede allocator creation");

        assert!(constructor.contains("PhysicalDeviceDynamicRenderingFeatures"));
        assert!(constructor.contains(".dynamic_rendering(true)"));
    }

    fn extension_name_strings(names: &[*const c_char]) -> Vec<String> {
        names
            .iter()
            .map(|&name| {
                unsafe { CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }
}
