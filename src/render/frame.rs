use ash::vk;

pub struct FrameContext {
    pub frame_index: u64,
    pub should_render: bool,
    pub command_buffer: vk::CommandBuffer,
    pub swapchain_image: vk::Image,
    pub swapchain_image_index: usize,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_format: vk::Format,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
    /// Whether the swapchain was suboptimal at acquire time
    pub(crate) suboptimal: bool,
}

impl FrameContext {
    pub fn skip(frame_index: u64) -> Self {
        Self {
            frame_index,
            should_render: false,
            command_buffer: vk::CommandBuffer::null(),
            swapchain_image: vk::Image::null(),
            swapchain_image_index: 0,
            swapchain_extent: vk::Extent2D::default(),
            swapchain_format: vk::Format::UNDEFINED,
            image_available_semaphore: vk::Semaphore::null(),
            render_finished_semaphore: vk::Semaphore::null(),
            in_flight_fence: vk::Fence::null(),
            suboptimal: false,
        }
    }
}
