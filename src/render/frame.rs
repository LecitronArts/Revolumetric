use ash::vk;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameCompletion {
    pub swapchain_recreated: bool,
}

pub struct FrameContext {
    pub frame_index: u64,
    pub frame_slot: usize,
    pub should_render: bool,
    pub command_buffer: vk::CommandBuffer,
    pub swapchain_image: vk::Image,
    pub swapchain_image_view: vk::ImageView,
    pub swapchain_image_index: usize,
    pub swapchain_image_layout: vk::ImageLayout,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_format: vk::Format,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
    /// Whether the swapchain was suboptimal at acquire time
    pub(crate) suboptimal: bool,
    pub swapchain_recreated: bool,
}

impl FrameContext {
    pub fn skip(frame_index: u64) -> Self {
        Self {
            frame_index,
            frame_slot: 0,
            should_render: false,
            command_buffer: vk::CommandBuffer::null(),
            swapchain_image: vk::Image::null(),
            swapchain_image_view: vk::ImageView::null(),
            swapchain_image_index: 0,
            swapchain_image_layout: vk::ImageLayout::UNDEFINED,
            swapchain_extent: vk::Extent2D::default(),
            swapchain_format: vk::Format::UNDEFINED,
            image_available_semaphore: vk::Semaphore::null(),
            render_finished_semaphore: vk::Semaphore::null(),
            in_flight_fence: vk::Fence::null(),
            suboptimal: false,
            swapchain_recreated: false,
        }
    }

    pub fn skip_after_swapchain_recreate(frame_index: u64) -> Self {
        Self {
            swapchain_recreated: true,
            ..Self::skip(frame_index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skipped_frame_can_report_swapchain_recreation() {
        let frame = FrameContext::skip_after_swapchain_recreate(42);

        assert_eq!(frame.frame_index, 42);
        assert!(!frame.should_render);
        assert!(frame.swapchain_recreated);
    }

    #[test]
    fn frame_completion_defaults_to_no_swapchain_recreation() {
        assert!(!FrameCompletion::default().swapchain_recreated);
    }
}
