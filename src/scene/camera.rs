use glam::{Mat4, Vec3};

use crate::platform::input::InputState;
use crate::scene::components::CameraRig;

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub aperture_radius: f32,
    pub focal_distance: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(64.0, 32.0, -40.0),
            forward: Vec3::new(0.0, -0.03, 0.99955).normalize(),
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_4, // 45°
            aperture_radius: 0.0,
            focal_distance: 128.0,
        }
    }
}

impl Camera {
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_to_rh(self.position, self.forward, self.up)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraPathKind {
    Orbit,
    Gallery,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPathConfig {
    pub kind: CameraPathKind,
    pub center: Vec3,
    pub radius: f32,
    pub height: f32,
    pub period_frames: u64,
}

impl CameraPathConfig {
    pub const DEFAULT_CENTER: Vec3 = Vec3::new(64.0, 32.0, 64.0);
    pub const DEFAULT_RADIUS: f32 = 40.0;
    pub const DEFAULT_HEIGHT: f32 = 36.0;
    pub const DEFAULT_PERIOD_FRAMES: u64 = 240;

    pub fn from_env() -> Option<Self> {
        Self::from_values(
            std::env::var("REVOLUMETRIC_CAMERA_PATH").ok().as_deref(),
            std::env::var("REVOLUMETRIC_CAMERA_PATH_CENTER")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_CAMERA_PATH_RADIUS")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_CAMERA_PATH_HEIGHT")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_CAMERA_PATH_PERIOD_FRAMES")
                .ok()
                .as_deref(),
        )
    }

    pub fn from_values(
        path: Option<&str>,
        center: Option<&str>,
        radius: Option<&str>,
        height: Option<&str>,
        period_frames: Option<&str>,
    ) -> Option<Self> {
        let kind = match path.map(str::trim).filter(|value| !value.is_empty()) {
            Some(value) if value.eq_ignore_ascii_case("orbit") => CameraPathKind::Orbit,
            Some(value)
                if value.eq_ignore_ascii_case("gallery")
                    || value.eq_ignore_ascii_case("flythrough") =>
            {
                CameraPathKind::Gallery
            }
            _ => return None,
        };

        Some(Self {
            kind,
            center: center
                .and_then(parse_vec3_csv)
                .unwrap_or(Self::DEFAULT_CENTER),
            radius: radius
                .and_then(parse_positive_f32)
                .unwrap_or(Self::DEFAULT_RADIUS),
            height: height
                .and_then(parse_finite_f32)
                .unwrap_or(Self::DEFAULT_HEIGHT),
            period_frames: period_frames
                .and_then(parse_positive_u64)
                .unwrap_or(Self::DEFAULT_PERIOD_FRAMES),
        })
    }
}

pub fn apply_camera_path(rig: &mut CameraRig, config: CameraPathConfig, frame_index: u64) {
    match config.kind {
        CameraPathKind::Orbit => apply_orbit_camera_path(rig, config, frame_index),
        CameraPathKind::Gallery => apply_gallery_camera_path(rig, config, frame_index),
    }
}

fn apply_orbit_camera_path(rig: &mut CameraRig, config: CameraPathConfig, frame_index: u64) {
    let period = config.period_frames.max(1);
    let phase = (frame_index % period) as f32 / period as f32;
    let angle = phase * std::f32::consts::TAU;
    let offset = Vec3::new(
        angle.cos() * config.radius,
        0.0,
        angle.sin() * config.radius,
    );
    let position = Vec3::new(
        config.center.x + offset.x,
        config.height,
        config.center.z + offset.z,
    );

    rig.camera.position = position;
    rig.camera.forward = (config.center - position).normalize_or_zero();
    if rig.camera.forward.length_squared() == 0.0 {
        rig.camera.forward = Vec3::Z;
    }
    rig.camera.up = Vec3::Y;
}

fn apply_gallery_camera_path(rig: &mut CameraRig, config: CameraPathConfig, frame_index: u64) {
    let period = config.period_frames.max(1);
    let phase = (frame_index % period) as f32 / period as f32;
    let position = gallery_camera_position_at(config, phase);
    let mut target = gallery_camera_position_at(config, phase + 0.125);
    target.y = config.center.y;

    rig.camera.position = position;
    rig.camera.forward = (target - position).normalize_or_zero();
    if rig.camera.forward.length_squared() == 0.0 {
        rig.camera.forward = Vec3::Z;
    }
    rig.camera.up = Vec3::Y;
}

fn gallery_camera_position_at(config: CameraPathConfig, phase: f32) -> Vec3 {
    let angle = phase * std::f32::consts::TAU;
    let lateral_sway = (config.radius * 0.075).min(3.0);
    Vec3::new(
        config.center.x + (angle * 2.0).sin() * lateral_sway,
        config.height,
        config.center.z - angle.cos() * config.radius,
    )
}

fn parse_vec3_csv(value: &str) -> Option<Vec3> {
    let mut parts = value.split(',').map(str::trim);
    let x = parse_finite_f32(parts.next()?)?;
    let y = parse_finite_f32(parts.next()?)?;
    let z = parse_finite_f32(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some(Vec3::new(x, y, z))
}

fn parse_finite_f32(value: &str) -> Option<f32> {
    let parsed = value.trim().parse::<f32>().ok()?;
    parsed.is_finite().then_some(parsed)
}

fn parse_positive_f32(value: &str) -> Option<f32> {
    let parsed = parse_finite_f32(value)?;
    (parsed > 0.0).then_some(parsed)
}

fn parse_positive_u64(value: &str) -> Option<u64> {
    let parsed = value.trim().parse::<u64>().ok()?;
    (parsed > 0).then_some(parsed)
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

    let move_forward = input.move_forward.clamp(-1.0, 1.0);
    let move_right = input.move_right.clamp(-1.0, 1.0);
    let move_up = input.move_up.clamp(-1.0, 1.0);

    let mut velocity = hz_forward * move_forward + hz_right * move_right + Vec3::Y * move_up;
    if velocity.length_squared() > 0.0 {
        velocity = velocity.normalize();
    }

    cam.position += velocity * ctrl.move_speed * dt;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_path_config_parses_orbit_values_with_defaults() {
        let config = CameraPathConfig::from_values(
            Some("orbit"),
            Some("10,20,30"),
            Some("48"),
            Some("40"),
            Some("120"),
        )
        .expect("orbit path should be enabled");

        assert_eq!(config.kind, CameraPathKind::Orbit);
        assert_eq!(config.center, Vec3::new(10.0, 20.0, 30.0));
        assert_eq!(config.radius, 48.0);
        assert_eq!(config.height, 40.0);
        assert_eq!(config.period_frames, 120);
    }

    #[test]
    fn camera_path_config_parses_gallery_with_defaults() {
        let config = CameraPathConfig::from_values(Some("gallery"), None, None, None, None)
            .expect("gallery path should be enabled");

        assert_eq!(config.kind, CameraPathKind::Gallery);
        assert_eq!(config.center, CameraPathConfig::DEFAULT_CENTER);
        assert_eq!(config.radius, CameraPathConfig::DEFAULT_RADIUS);
        assert_eq!(config.height, CameraPathConfig::DEFAULT_HEIGHT);
        assert_eq!(
            config.period_frames,
            CameraPathConfig::DEFAULT_PERIOD_FRAMES
        );
    }

    #[test]
    fn camera_path_config_disables_unknown_paths() {
        assert!(CameraPathConfig::from_values(Some("manual"), None, None, None, None).is_none());
    }

    #[test]
    fn camera_path_config_falls_back_for_invalid_optional_values() {
        let config = CameraPathConfig::from_values(
            Some("orbit"),
            Some("bad"),
            Some("-1"),
            Some("nan"),
            Some("0"),
        )
        .expect("orbit path should stay enabled when optional values are invalid");

        assert_eq!(config.center, CameraPathConfig::DEFAULT_CENTER);
        assert_eq!(config.radius, CameraPathConfig::DEFAULT_RADIUS);
        assert_eq!(config.height, CameraPathConfig::DEFAULT_HEIGHT);
        assert_eq!(
            config.period_frames,
            CameraPathConfig::DEFAULT_PERIOD_FRAMES
        );
    }

    #[test]
    fn apply_camera_path_orbits_around_center_and_points_inward() {
        let config = CameraPathConfig {
            kind: CameraPathKind::Orbit,
            center: Vec3::new(64.0, 32.0, 64.0),
            radius: 32.0,
            height: 48.0,
            period_frames: 4,
        };
        let mut rig = CameraRig::default();

        apply_camera_path(&mut rig, config, 1);

        assert!((rig.camera.position - Vec3::new(64.0, 48.0, 96.0)).length() < 1e-4);
        let expected_forward = (config.center - rig.camera.position).normalize();
        assert!((rig.camera.forward - expected_forward).length() < 1e-5);
        assert_eq!(rig.camera.up, Vec3::Y);
    }

    #[test]
    fn apply_camera_path_gallery_flies_the_central_aisle_and_looks_ahead() {
        let config = CameraPathConfig {
            kind: CameraPathKind::Gallery,
            center: Vec3::new(64.0, 32.0, 64.0),
            radius: 40.0,
            height: 36.0,
            period_frames: 8,
        };
        let mut rig = CameraRig::default();

        apply_camera_path(&mut rig, config, 0);

        assert!((rig.camera.position - Vec3::new(64.0, 36.0, 24.0)).length() < 1e-4);
        assert!(
            rig.camera.forward.z > 0.85,
            "front of the gallery path should look down the aisle, got {:?}",
            rig.camera.forward
        );
        assert!(
            rig.camera.forward.y < 0.0,
            "gallery path should look slightly down toward the scene center"
        );

        apply_camera_path(&mut rig, config, 4);

        assert!((rig.camera.position - Vec3::new(64.0, 36.0, 104.0)).length() < 1e-4);
        assert!(
            rig.camera.forward.z < -0.85,
            "back of the gallery path should look back down the aisle, got {:?}",
            rig.camera.forward
        );
        assert_eq!(rig.camera.up, Vec3::Y);
    }

    #[test]
    fn camera_default_matches_spec() {
        let cam = Camera::default();
        assert!((cam.position - Vec3::new(64.0, 32.0, -40.0)).length() < 1e-3);
        assert!((cam.fov_y_radians - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
        assert!(cam.forward.z > 0.9, "should look along +Z");
        assert!(cam.forward.y < 0.0, "should look slightly down");
    }

    #[test]
    fn camera_defaults_keep_pinhole_lens_disabled() {
        let cam = Camera::default();

        assert_eq!(cam.aperture_radius, 0.0);
        assert!(cam.focal_distance > 0.0);
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
