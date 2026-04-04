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
}

impl Default for FlyCameraController {
    fn default() -> Self {
        Self { move_speed: 5.0 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CameraRig {
    pub camera: Camera,
    pub controller: FlyCameraController,
    pub transform: Transform,
}
