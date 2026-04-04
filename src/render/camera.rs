// src/render/camera.rs
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

/// Push constants for the primary ray pass.
/// Sent to GPU each frame — must match the Slang `PrimaryRayPC` layout exactly.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PrimaryRayPushConstants {
    /// Column-major 4×4 matrix.
    /// - Columns 0-2 (3×3): maps (pixel_x, pixel_y, 1) → unnormalized world-space ray direction
    /// - Column 3 (xyz):   camera world position (ray origin)
    pub pixel_to_ray: [[f32; 4]; 4],
    pub resolution: [u32; 2],
    pub _pad: [u32; 2],
}

/// Compute the pixel_to_ray matrix for a pinhole camera.
///
/// Convention: Y-up, camera looks along its `forward` direction.
/// The shader normalizes the direction, so the 3×3 part need not produce unit vectors.
pub fn compute_pixel_to_ray(
    camera_pos: Vec3,
    camera_forward: Vec3,
    camera_up: Vec3,
    fov_y_rad: f32,
    width: u32,
    height: u32,
) -> Mat4 {
    let w = width as f32;
    let h = height as f32;
    let aspect = w / h;
    let t = (fov_y_rad * 0.5).tan();

    // Build orthonormal camera basis (right-handed, forward=+Z, up=+Y → right=+X)
    // right = up × forward  (Y × Z = +X)
    // up    = forward × right (Z × X = +Y)
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
    let ox = aspect * t * (1.0 / w - 1.0);
    let oy = t * (1.0 - 1.0 / h);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_pixel_looks_along_forward() {
        let m = compute_pixel_to_ray(
            Vec3::ZERO, Vec3::Z, Vec3::Y,
            std::f32::consts::FRAC_PI_2, 800, 600,
        );
        let origin = Vec3::new(m.col(3).x, m.col(3).y, m.col(3).z);
        assert!((origin - Vec3::ZERO).length() < 1e-5);
        let mat3 = glam::Mat3::from_cols(
            m.col(0).truncate(), m.col(1).truncate(), m.col(2).truncate(),
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
            m.col(0).truncate(), m.col(1).truncate(), m.col(2).truncate(),
        );
        let left = (mat3 * Vec3::new(0.0, 300.0, 1.0)).normalize();
        let right = (mat3 * Vec3::new(799.0, 300.0, 1.0)).normalize();
        assert!(left.x < right.x, "left.x={} < right.x={}", left.x, right.x);
    }

    #[test]
    fn push_constants_size() {
        assert_eq!(std::mem::size_of::<PrimaryRayPushConstants>(), 80);
    }
}
