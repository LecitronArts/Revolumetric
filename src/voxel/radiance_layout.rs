use glam::UVec3;

#[derive(Debug, Clone)]
pub struct CascadeLayout {
    pub lod_count: u32,
    pub base_grid: UVec3,
    pub hemisphere_count: u32,
}

impl Default for CascadeLayout {
    fn default() -> Self {
        Self {
            lod_count: 5,
            base_grid: UVec3::new(32, 32, 48),
            hemisphere_count: 6,
        }
    }
}
