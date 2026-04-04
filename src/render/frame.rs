#[derive(Debug, Clone, Copy, Default)]
pub struct FrameContext {
    pub frame_index: u64,
    pub should_render: bool,
}

impl FrameContext {
    pub fn begin(frame_index: u64) -> Self {
        Self {
            frame_index,
            should_render: true,
        }
    }
}
