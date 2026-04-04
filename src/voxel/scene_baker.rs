use glam::UVec3;

use crate::voxel::grid::VoxelGrid;

pub fn bake_demo_scene() -> VoxelGrid {
    VoxelGrid::new(UVec3::new(32, 32, 48))
}
