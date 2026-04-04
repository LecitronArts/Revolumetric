// src/voxel/ucvh.rs
use glam::UVec3;
use crate::voxel::brick::{BrickData, VoxelCell, BRICK_EDGE};
use crate::voxel::brick_pool::{BrickId, BrickPool};
use crate::voxel::morton;
use crate::voxel::occupancy::CascadedOccupancy;

pub struct UcvhConfig {
    pub world_size: UVec3,
    pub brick_grid_size: UVec3,
    pub brick_capacity: u32,
}

impl UcvhConfig {
    pub fn new(world_size: UVec3) -> Self {
        let brick_grid_size = world_size / BRICK_EDGE;
        // Estimate capacity at 40% fill + headroom
        let total_bricks = brick_grid_size.x * brick_grid_size.y * brick_grid_size.z;
        let capacity = (total_bricks * 2 / 5).max(64);
        Self { world_size, brick_grid_size, brick_capacity: capacity }
    }
}

/// Unified Cascaded Volume Hierarchy — the single entry point for voxel data.
pub struct Ucvh {
    pub config: UcvhConfig,
    pub pool: BrickPool,
    pub hierarchy: CascadedOccupancy,
    /// Sparse map: L0 flat index -> BrickId (None = no brick allocated)
    brick_map: Vec<Option<BrickId>>,
    /// Brick IDs that need GPU re-upload
    dirty_bricks: Vec<BrickId>,
    /// Whether the hierarchy needs rebuild
    hierarchy_dirty: bool,
}

impl Ucvh {
    pub fn new(config: UcvhConfig) -> Self {
        let l0_count = (config.brick_grid_size.x
            * config.brick_grid_size.y
            * config.brick_grid_size.z) as usize;
        Self {
            pool: BrickPool::new(config.brick_capacity),
            hierarchy: CascadedOccupancy::new(config.brick_grid_size),
            brick_map: vec![None; l0_count],
            dirty_bricks: Vec::new(),
            hierarchy_dirty: false,
            config,
        }
    }

    /// Convert world voxel position to (brick_grid_pos, local_pos).
    fn decompose(pos: UVec3) -> (UVec3, UVec3) {
        (pos / BRICK_EDGE, pos % BRICK_EDGE)
    }

    fn l0_index(&self, brick_pos: UVec3) -> usize {
        CascadedOccupancy::flat_index(brick_pos, self.config.brick_grid_size)
    }

    /// Ensure a brick exists at `brick_pos`, allocating if needed.
    fn ensure_brick(&mut self, brick_pos: UVec3) -> Option<BrickId> {
        let idx = self.l0_index(brick_pos);
        if let Some(id) = self.brick_map[idx] {
            return Some(id);
        }
        let id = self.pool.allocate()?;
        self.brick_map[idx] = Some(id);
        Some(id)
    }

    pub fn set_voxel(&mut self, pos: UVec3, cell: VoxelCell) -> bool {
        let (bp, lp) = Self::decompose(pos);
        let Some(id) = self.ensure_brick(bp) else { return false };
        let m = morton::encode(lp.x, lp.y, lp.z);
        self.pool.set_material(id, m, cell);
        if cell.is_air() {
            self.pool.occupancy_mut(id).clear(lp.x, lp.y, lp.z);
        } else {
            self.pool.occupancy_mut(id).set(lp.x, lp.y, lp.z);
        }
        if !self.dirty_bricks.contains(&id) {
            self.dirty_bricks.push(id);
        }
        self.hierarchy_dirty = true;
        true
    }

    pub fn get_voxel(&self, pos: UVec3) -> VoxelCell {
        let (bp, lp) = Self::decompose(pos);
        let idx = self.l0_index(bp);
        match self.brick_map[idx] {
            Some(id) => self.pool.get_material(id, morton::encode(lp.x, lp.y, lp.z)),
            None => VoxelCell::AIR,
        }
    }

    /// Write a full BrickData at a brick grid position.
    pub fn write_brick(&mut self, brick_pos: UVec3, data: &BrickData) -> bool {
        let Some(id) = self.ensure_brick(brick_pos) else { return false };
        self.pool.write_brick(id, data);
        if !self.dirty_bricks.contains(&id) {
            self.dirty_bricks.push(id);
        }
        self.hierarchy_dirty = true;
        true
    }

    /// Rebuild occupancy hierarchy from current pool data.
    pub fn rebuild_hierarchy(&mut self) {
        // Update L0 from brick_map + pool
        let bgs = self.config.brick_grid_size;
        for bz in 0..bgs.z {
            for by in 0..bgs.y {
                for bx in 0..bgs.x {
                    let bp = UVec3::new(bx, by, bz);
                    let idx = self.l0_index(bp);
                    match self.brick_map[idx] {
                        Some(id) => {
                            let has_solid = !self.pool.occupancy(id).is_empty();
                            self.hierarchy.set_l0(bp, id, has_solid);
                        }
                        None => {
                            self.hierarchy.set_l0(bp, u32::MAX, false);
                        }
                    }
                }
            }
        }
        self.hierarchy.rebuild();
        self.hierarchy_dirty = false;
    }

    pub fn take_dirty_bricks(&mut self) -> Vec<BrickId> {
        std::mem::take(&mut self.dirty_bricks)
    }

    pub fn is_hierarchy_dirty(&self) -> bool {
        self.hierarchy_dirty
    }

    pub fn allocated_brick_count(&self) -> u32 {
        self.pool.allocated_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_ucvh() -> Ucvh {
        Ucvh::new(UcvhConfig::new(UVec3::splat(128)))
    }

    #[test]
    fn config_computes_grid_size() {
        let c = UcvhConfig::new(UVec3::splat(128));
        assert_eq!(c.brick_grid_size, UVec3::splat(16));
    }

    #[test]
    fn set_and_get_voxel() {
        let mut u = test_ucvh();
        let cell = VoxelCell { material: 5, flags: 0, emissive: [0; 3], _pad: 0 };
        assert!(u.set_voxel(UVec3::new(10, 20, 30), cell));
        assert_eq!(u.get_voxel(UVec3::new(10, 20, 30)).material, 5);
        assert_eq!(u.get_voxel(UVec3::new(0, 0, 0)).material, 0); // air
    }

    #[test]
    fn dirty_tracking() {
        let mut u = test_ucvh();
        let cell = VoxelCell { material: 1, flags: 0, emissive: [0; 3], _pad: 0 };
        u.set_voxel(UVec3::new(0, 0, 0), cell);
        u.set_voxel(UVec3::new(1, 0, 0), cell); // same brick
        let dirty = u.take_dirty_bricks();
        assert_eq!(dirty.len(), 1); // one brick, not two
    }

    #[test]
    fn hierarchy_rebuild_propagates() {
        let mut u = test_ucvh();
        let cell = VoxelCell { material: 1, flags: 0, emissive: [0; 3], _pad: 0 };
        u.set_voxel(UVec3::new(0, 0, 0), cell);
        u.rebuild_hierarchy();

        // L0 at (0,0,0) should have a valid brick_id
        let node = u.hierarchy.get_l0(UVec3::ZERO);
        assert_ne!(node.brick_id, u32::MAX);
        assert_eq!(node.flags & 1, 1);

        // Root of hierarchy should be non-empty
        assert_ne!(u.hierarchy.levels[3][0].child_mask, 0);
    }

    #[test]
    fn write_brick_bulk() {
        let mut u = test_ucvh();
        let mut data = BrickData::new();
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    data.set_voxel(x, y, z, VoxelCell { material: 1, flags: 0, emissive: [0; 3], _pad: 0 });
                }
            }
        }
        assert!(u.write_brick(UVec3::ZERO, &data));
        assert_eq!(u.pool.occupancy(0).count, 512);
    }
}
