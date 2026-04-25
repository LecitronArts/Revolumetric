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
            position: Vec3::new(64.0, 80.0, -40.0),
            forward: Vec3::new(0.0, -0.152, 0.988).normalize(),
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_4, // 45°
        }
    }
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward, self.up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_default_matches_spec() {
        let cam = Camera::default();
        assert!((cam.position - Vec3::new(64.0, 80.0, -40.0)).length() < 1e-3);
        assert!((cam.fov_y_radians - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
        assert!(cam.forward.z > 0.9, "should look along +Z");
        assert!(cam.forward.y < 0.0, "should look slightly down");
    }
}
