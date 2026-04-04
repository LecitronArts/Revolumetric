use glam::Vec3;

#[derive(Debug, Clone)]
pub struct MaterialDesc {
    pub albedo: Vec3,
    pub emissive: Vec3,
}

impl Default for MaterialDesc {
    fn default() -> Self {
        Self {
            albedo: Vec3::splat(1.0),
            emissive: Vec3::ZERO,
        }
    }
}
