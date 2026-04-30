use glam::{Mat4, Vec3};

use crate::platform::input::InputState;
use crate::scene::components::CameraRig;

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

pub fn update_fly_camera(rig: &mut CameraRig, input: InputState, dt: f32) {
    let ctrl = &mut rig.controller;
    let cam = &mut rig.camera;

    if input.scroll_delta != 0.0 {
        ctrl.move_speed *= ctrl.scroll_multiplier.powf(input.scroll_delta);
        ctrl.move_speed = ctrl.move_speed.clamp(ctrl.min_speed, ctrl.max_speed);
    }

    let sens_rad = ctrl.mouse_sensitivity * std::f32::consts::PI / 180.0;
    ctrl.yaw += input.mouse_dx * sens_rad;
    ctrl.pitch -= input.mouse_dy * sens_rad;
    ctrl.pitch = ctrl.pitch.clamp(-1.553, 1.553);

    cam.forward = Vec3::new(
        ctrl.pitch.cos() * ctrl.yaw.sin(),
        ctrl.pitch.sin(),
        ctrl.pitch.cos() * ctrl.yaw.cos(),
    );

    let hz_forward = Vec3::new(ctrl.yaw.sin(), 0.0, ctrl.yaw.cos());
    let hz_right = Vec3::Y.cross(hz_forward);

    let mut velocity =
        hz_forward * input.move_forward + hz_right * input.move_right + Vec3::Y * input.move_up;
    if velocity.length_squared() > 0.0 {
        velocity = velocity.normalize();
    }

    cam.position += velocity * ctrl.move_speed * dt;
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

    #[test]
    fn fly_camera_scroll_scales_and_clamps_speed() {
        let mut rig = CameraRig::default();
        let input = InputState {
            scroll_delta: 100.0,
            ..InputState::default()
        };

        update_fly_camera(&mut rig, input, 0.0);

        assert_eq!(rig.controller.move_speed, rig.controller.max_speed);
    }

    #[test]
    fn fly_camera_mouse_delta_updates_orientation() {
        let mut rig = CameraRig::default();
        let input = InputState {
            mouse_dx: 10.0,
            mouse_dy: -5.0,
            ..InputState::default()
        };

        update_fly_camera(&mut rig, input, 0.0);

        assert!(rig.controller.yaw > 0.0);
        assert!(rig.controller.pitch > -0.153);
        assert!(rig.camera.forward.x > 0.0);
    }

    #[test]
    fn fly_camera_movement_axes_use_normalized_velocity() {
        let mut rig = CameraRig::default();
        let start = rig.camera.position;
        let input = InputState {
            move_forward: 1.0,
            move_right: 1.0,
            ..InputState::default()
        };

        update_fly_camera(&mut rig, input, 1.0);

        assert!((rig.camera.position - start).length() - rig.controller.move_speed < 1e-3);
    }
}
