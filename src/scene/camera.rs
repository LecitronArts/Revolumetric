use glam::{Mat4, Vec3};

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(30.0, 8.0, 33.0),
            forward: Vec3::new(0.0, 0.0, -1.0),
            up: Vec3::Y,
            fov_y_radians: 60.0_f32.to_radians(),
        }
    }
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward, self.up)
    }
}
