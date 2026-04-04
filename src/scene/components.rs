use glam::Vec3;

use crate::scene::camera::Camera;

#[derive(Debug, Clone)]
pub struct Transform {
    pub translation: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FlyCameraController {
    pub move_speed: f32,
    pub min_speed: f32,
    pub max_speed: f32,
    pub scroll_multiplier: f32,
    pub mouse_sensitivity: f32,
    pub pitch: f32,
    pub yaw: f32,
}

impl Default for FlyCameraController {
    fn default() -> Self {
        Self {
            move_speed: 50.0,
            min_speed: 5.0,
            max_speed: 500.0,
            scroll_multiplier: 1.2,
            mouse_sensitivity: 0.3,
            pitch: -0.153, // ≈ -8.7°, derived from hardcoded camera looking at sphere
            yaw: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CameraRig {
    pub camera: Camera,
    pub controller: FlyCameraController,
    pub transform: Transform,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_camera_controller_defaults() {
        let ctrl = FlyCameraController::default();
        assert_eq!(ctrl.move_speed, 50.0);
        assert_eq!(ctrl.min_speed, 5.0);
        assert_eq!(ctrl.max_speed, 500.0);
        assert!((ctrl.scroll_multiplier - 1.2).abs() < 1e-5);
        assert!((ctrl.mouse_sensitivity - 0.3).abs() < 1e-5);
        assert!((ctrl.pitch - (-0.153)).abs() < 0.01);
        assert_eq!(ctrl.yaw, 0.0);
    }
}
