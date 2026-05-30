#[derive(Debug, Default, Clone, Copy)]
pub struct InputState {
    /// Movement axes (-1.0 to 1.0, accumulated from key press/release)
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    /// Mouse delta in pixels this frame (accumulated, cleared each frame)
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    /// Right mouse button currently held
    pub right_mouse_held: bool,
    /// Scroll wheel delta this frame (accumulated, cleared each frame)
    pub scroll_delta: f32,
}

impl InputState {
    /// Clear per-frame accumulators. Called at the end of each frame.
    pub fn clear_per_frame(&mut self) {
        self.mouse_dx = 0.0;
        self.mouse_dy = 0.0;
        self.scroll_delta = 0.0;
    }

    /// Reset all axes to zero. Called on focus loss.
    pub fn reset_axes(&mut self) {
        self.move_forward = 0.0;
        self.move_right = 0.0;
        self.move_up = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_state_default_is_zeroed() {
        let state = InputState::default();
        assert_eq!(state.move_forward, 0.0);
        assert_eq!(state.move_right, 0.0);
        assert_eq!(state.move_up, 0.0);
        assert_eq!(state.mouse_dx, 0.0);
        assert_eq!(state.mouse_dy, 0.0);
        assert!(!state.right_mouse_held);
        assert_eq!(state.scroll_delta, 0.0);
    }

    #[test]
    fn clear_per_frame_preserves_axes() {
        let mut state = InputState {
            move_forward: 1.0,
            move_right: -1.0,
            move_up: 1.0,
            mouse_dx: 10.0,
            mouse_dy: -5.0,
            right_mouse_held: true,
            scroll_delta: 2.0,
        };
        state.clear_per_frame();
        assert_eq!(state.move_forward, 1.0);
        assert_eq!(state.move_right, -1.0);
        assert_eq!(state.mouse_dx, 0.0);
        assert_eq!(state.mouse_dy, 0.0);
        assert_eq!(state.scroll_delta, 0.0);
        assert!(state.right_mouse_held); // not cleared
    }

    #[test]
    fn reset_axes_clears_movement() {
        let mut state = InputState {
            move_forward: 1.0,
            move_right: -1.0,
            move_up: 1.0,
            mouse_dx: 10.0,
            ..InputState::default()
        };
        state.reset_axes();
        assert_eq!(state.move_forward, 0.0);
        assert_eq!(state.move_right, 0.0);
        assert_eq!(state.move_up, 0.0);
        assert_eq!(state.mouse_dx, 10.0); // not cleared
    }
}
