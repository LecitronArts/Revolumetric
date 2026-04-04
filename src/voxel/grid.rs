use glam::UVec3;

#[derive(Debug, Clone, Copy)]
pub struct VoxelCell {
    pub material: u16,
    pub flags: u16,
    pub emissive: [u16; 3],
    pub _pad: u16,
}

impl Default for VoxelCell {
    fn default() -> Self {
        Self {
            material: 0,
            flags: 0,
            emissive: [0; 3],
            _pad: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VoxelGrid {
    pub extent: UVec3,
    pub cells: Vec<VoxelCell>,
}

impl VoxelGrid {
    pub fn new(extent: UVec3) -> Self {
        let len = (extent.x * extent.y * extent.z) as usize;
        Self {
            extent,
            cells: vec![VoxelCell::default(); len],
        }
    }
}
