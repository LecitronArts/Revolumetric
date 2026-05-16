// src/render/camera.rs
use glam::{Mat4, Vec2, Vec3, Vec4};

fn halton(mut index: u64, base: u64) -> f32 {
    let mut result = 0.0;
    let mut factor = 1.0;
    let base_f32 = base as f32;

    while index > 0 {
        factor /= base_f32;
        result += factor * (index % base) as f32;
        index /= base;
    }

    result
}

pub fn taa_frame_jitter(frame_index: u64) -> Vec2 {
    let sample_index = frame_index.saturating_add(1);
    Vec2::new(halton(sample_index, 2) - 0.5, halton(sample_index, 3) - 0.5)
}

/// Compute the pixel_to_ray matrix for a pinhole camera.
///
/// Convention: Y-up, camera looks along its `forward` direction.
/// The shader normalizes the direction, so the 3x3 part need not produce unit vectors.
pub fn compute_pixel_to_ray(
    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_up: Vec3,
    fov_y_rad: f32,
    width: u32,
    height: u32,
) -> Mat4 {
    compute_pixel_to_ray_with_jitter(
        camera_pos,
        camera_forward,
        camera_up,
        fov_y_rad,
        width,
        height,
        Vec2::ZERO,
    )
}

pub fn compute_pixel_to_ray_with_jitter(
    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_up: Vec3,
    fov_y_rad: f32,
    width: u32,
    height: u32,
    jitter: Vec2,
) -> Mat4 {
    let w = width as f32;
    let h = height as f32;
    let aspect = w / h;
    let t = (fov_y_rad * 0.5).tan();

    // Build orthonormal camera basis (right-handed, forward=+Z, up=+Y -> right=+X)
    // right = up x forward  (Y x Z = +X)
    // up    = forward x right (Z x X = +Y)
    let forward = camera_forward.normalize();
    let right = camera_up.cross(forward).normalize();
    let up = forward.cross(right);

    // For pixel (px, py), the view-space direction is:
    //   vx = aspect*t * ((2*(px+0.5)/w) - 1)
    //   vy = t * (1 - (2*(py+0.5)/h))
    //   vz = 1.0
    // direction = right*vx + up*vy + forward*vz
    // This maps to: direction = mat3_cols * (px, py, 1)
    let sx = 2.0 * aspect * t / w;
    let sy = -2.0 * t / h;
    let ox = aspect * t * (1.0 / w - 1.0) + sx * jitter.x;
    let oy = t * (1.0 - 1.0 / h) + sy * jitter.y;

    let col0 = right * sx;
    let col1 = up * sy;
    let col2 = right * ox + up * oy + forward;
    let col3 = camera_pos;

    Mat4::from_cols(
        Vec4::new(col0.x, col0.y, col0.z, 0.0),
        Vec4::new(col1.x, col1.y, col1.z, 0.0),
        Vec4::new(col2.x, col2.y, col2.z, 0.0),
        Vec4::new(col3.x, col3.y, col3.z, 1.0),
    )
}

pub fn compute_view_proj(
    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_up: Vec3,
    fov_y: f32,
    width: u32,
    height: u32,
) -> Mat4 {
    compute_view_proj_with_jitter(
        camera_pos,
        camera_forward,
        camera_up,
        fov_y,
        width,
        height,
        Vec2::ZERO,
    )
}

pub fn compute_view_proj_with_jitter(
    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_up: Vec3,
    fov_y: f32,
    width: u32,
    height: u32,
    jitter: Vec2,
) -> Mat4 {
    let forward = camera_forward.normalize();
    let right = camera_up.cross(forward).normalize();
    let up = forward.cross(right);
    let view = Mat4::from_cols(
        Vec4::new(right.x, up.x, -forward.x, 0.0),
        Vec4::new(right.y, up.y, -forward.y, 0.0),
        Vec4::new(right.z, up.z, -forward.z, 0.0),
        Vec4::new(
            -right.dot(camera_pos),
            -up.dot(camera_pos),
            forward.dot(camera_pos),
            1.0,
        ),
    );
    let projection = Mat4::perspective_rh(fov_y, width as f32 / height as f32, 0.01, 10_000.0);
    let jitter_ndc = Vec2::new(
        -2.0 * jitter.x / width as f32,
        2.0 * jitter.y / height as f32,
    );
    let jitter_matrix = Mat4::from_cols(
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(jitter_ndc.x, jitter_ndc.y, 0.0, 1.0),
    );
    jitter_matrix * projection * view
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_pixel_looks_along_forward() {
        let m = compute_pixel_to_ray(
            Vec3::ZERO,
            Vec3::Z,
            Vec3::Y,
            std::f32::consts::FRAC_PI_2,
            800,
            600,
        );
        let origin = Vec3::new(m.col(3).x, m.col(3).y, m.col(3).z);
        assert!((origin - Vec3::ZERO).length() < 1e-5);
        let mat3 = glam::Mat3::from_cols(
            m.col(0).truncate(),
            m.col(1).truncate(),
            m.col(2).truncate(),
        );
        let dir = (mat3 * Vec3::new(400.0, 300.0, 1.0)).normalize();
        assert!(dir.z > 0.5, "center ray should point along +Z, got {dir}");
    }

    #[test]
    fn origin_matches_camera_position() {
        let pos = Vec3::new(10.0, 20.0, 30.0);
        let m = compute_pixel_to_ray(pos, Vec3::Z, Vec3::Y, 1.0, 1920, 1080);
        let origin = Vec3::new(m.col(3).x, m.col(3).y, m.col(3).z);
        assert!((origin - pos).length() < 1e-5);
    }

    #[test]
    fn horizontal_ray_divergence() {
        let m = compute_pixel_to_ray(Vec3::ZERO, Vec3::Z, Vec3::Y, 1.0, 800, 600);
        let mat3 = glam::Mat3::from_cols(
            m.col(0).truncate(),
            m.col(1).truncate(),
            m.col(2).truncate(),
        );
        let left = (mat3 * Vec3::new(0.0, 300.0, 1.0)).normalize();
        let right = (mat3 * Vec3::new(799.0, 300.0, 1.0)).normalize();
        assert!(left.x < right.x, "left.x={} < right.x={}", left.x, right.x);
    }

    #[test]
    fn taa_frame_jitter_uses_halton_2_3() {
        let jitter0 = taa_frame_jitter(0);
        let jitter1 = taa_frame_jitter(1);

        assert!((jitter0 - glam::Vec2::new(0.0, -1.0 / 6.0)).length() < 1.0e-6);
        assert!((jitter1 - glam::Vec2::new(-0.25, 1.0 / 6.0)).length() < 1.0e-6);
    }

    #[test]
    fn jittered_view_projection_round_trips_current_pixel_center() {
        let camera_pos = Vec3::new(64.0, 32.0, -40.0);
        let camera_forward = Vec3::new(0.0, -0.03, 0.99955).normalize();
        let camera_up = Vec3::Y;
        let width = 800;
        let height = 600;
        let jitter = glam::Vec2::new(0.25, -0.125);
        let pixel = glam::Vec2::new(337.0, 214.0);

        let pixel_to_ray = compute_pixel_to_ray_with_jitter(
            camera_pos,
            camera_forward,
            camera_up,
            std::f32::consts::FRAC_PI_4,
            width,
            height,
            jitter,
        );
        let ray_basis = glam::Mat3::from_cols(
            pixel_to_ray.col(0).truncate(),
            pixel_to_ray.col(1).truncate(),
            pixel_to_ray.col(2).truncate(),
        );
        let world_position =
            camera_pos + (ray_basis * glam::Vec3::new(pixel.x, pixel.y, 1.0)).normalize() * 100.0;

        let clip = compute_view_proj_with_jitter(
            camera_pos,
            camera_forward,
            camera_up,
            std::f32::consts::FRAC_PI_4,
            width,
            height,
            jitter,
        ) * world_position.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let reprojected = glam::Vec2::new(
            (ndc.x * 0.5 + 0.5) * width as f32,
            (0.5 - ndc.y * 0.5) * height as f32,
        );

        let expected = pixel + glam::Vec2::new(0.5, 0.5);
        assert!(
            (reprojected - expected).length() < 1.0e-3,
            "expected {expected}, got {reprojected}"
        );
    }
}
