// src/voxel/occupancy.rs
use bytemuck::{Pod, Zeroable};
use glam::UVec3;

/// Level 0 node: points to a brick in the BrickPool.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct NodeL0 {
    pub brick_id: u32, // BrickId or u32::MAX if empty
    pub flags: u16,    // bit 0: has_solid
    pub _pad: u16,
}

/// Level 1-4 node: child occupancy mask.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct NodeLN {
    pub child_mask: u8, // which of 8 children (2^3) are occupied
    pub flags: u8,      // bit 0: any_solid
    pub _pad: [u8; 2],
}

pub const EMPTY_L0: NodeL0 = NodeL0 {
    brick_id: u32::MAX,
    flags: 0,
    _pad: 0,
};

/// 5-level cascaded occupancy hierarchy.
/// L0 = brick_grid_size, each subsequent level halves dimensions.
pub struct CascadedOccupancy {
    pub level0: Vec<NodeL0>,
    pub levels: [Vec<NodeLN>; 4], // levels 1-4
    pub dims: [UVec3; 5],
}

impl CascadedOccupancy {
    pub fn new(brick_grid_size: UVec3) -> Self {
        let dims = std::array::from_fn(|i| brick_grid_size >> i as u32);
        let vol = |d: UVec3| (d.x * d.y * d.z) as usize;
        Self {
            level0: vec![EMPTY_L0; vol(dims[0])],
            levels: [
                vec![NodeLN::zeroed(); vol(dims[1])],
                vec![NodeLN::zeroed(); vol(dims[2])],
                vec![NodeLN::zeroed(); vol(dims[3])],
                vec![NodeLN::zeroed(); vol(dims[4])],
            ],
            dims,
        }
    }

    pub fn flat_index(pos: UVec3, dim: UVec3) -> usize {
        (pos.x + pos.y * dim.x + pos.z * dim.x * dim.y) as usize
    }

    pub fn set_l0(&mut self, brick_pos: UVec3, brick_id: u32, has_solid: bool) {
        let idx = Self::flat_index(brick_pos, self.dims[0]);
        self.level0[idx].brick_id = brick_id;
        self.level0[idx].flags = if has_solid { 1 } else { 0 };
    }

    pub fn get_l0(&self, brick_pos: UVec3) -> &NodeL0 {
        &self.level0[Self::flat_index(brick_pos, self.dims[0])]
    }

    /// Rebuild all hierarchy levels bottom-up from level 0.
    pub fn rebuild(&mut self) {
        // L1 from L0
        self.rebuild_l1_from_l0();
        // L2-L4 from previous level
        for level in 1..4 {
            self.rebuild_ln_from_prev(level);
        }
    }

    fn rebuild_l1_from_l0(&mut self) {
        let child_dim = self.dims[0];
        let parent_dim = self.dims[1];
        for idx in 0..(parent_dim.x * parent_dim.y * parent_dim.z) {
            let pz = idx / (parent_dim.x * parent_dim.y);
            let py = (idx / parent_dim.x) % parent_dim.y;
            let px = idx % parent_dim.x;
            let mut mask = 0u8;
            for d in 0..8u32 {
                let (dx, dy, dz) = (d & 1, (d >> 1) & 1, (d >> 2) & 1);
                let cp = UVec3::new(px * 2 + dx, py * 2 + dy, pz * 2 + dz);
                if cp.x < child_dim.x && cp.y < child_dim.y && cp.z < child_dim.z {
                    let ci = Self::flat_index(cp, child_dim);
                    if self.level0[ci].flags & 1 != 0 {
                        mask |= 1 << d;
                    }
                }
            }
            let pi = idx as usize;
            self.levels[0][pi].child_mask = mask;
            self.levels[0][pi].flags = if mask != 0 { 1 } else { 0 };
        }
    }

    fn rebuild_ln_from_prev(&mut self, target_idx: usize) {
        // target_idx: 1=L2, 2=L3, 3=L4 (index into self.levels)
        let child_dim = self.dims[target_idx]; // dims of the source level
        let parent_dim = self.dims[target_idx + 1];
        // Clone source to avoid borrow conflict
        let source = self.levels[target_idx - 1].clone();
        for idx in 0..(parent_dim.x * parent_dim.y * parent_dim.z) {
            let pz = idx / (parent_dim.x * parent_dim.y);
            let py = (idx / parent_dim.x) % parent_dim.y;
            let px = idx % parent_dim.x;
            let mut mask = 0u8;
            for d in 0..8u32 {
                let (dx, dy, dz) = (d & 1, (d >> 1) & 1, (d >> 2) & 1);
                let cp = UVec3::new(px * 2 + dx, py * 2 + dy, pz * 2 + dz);
                if cp.x < child_dim.x && cp.y < child_dim.y && cp.z < child_dim.z {
                    let ci = Self::flat_index(cp, child_dim);
                    if source[ci].flags & 1 != 0 {
                        mask |= 1 << d;
                    }
                }
            }
            let pi = idx as usize;
            self.levels[target_idx][pi].child_mask = mask;
            self.levels[target_idx][pi].flags = if mask != 0 { 1 } else { 0 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_assertions() {
        assert_eq!(std::mem::size_of::<NodeL0>(), 8);
        assert_eq!(std::mem::size_of::<NodeLN>(), 4);
    }

    #[test]
    fn hierarchy_dims() {
        let h = CascadedOccupancy::new(UVec3::splat(16));
        assert_eq!(h.dims[0], UVec3::splat(16)); // L0
        assert_eq!(h.dims[1], UVec3::splat(8)); // L1
        assert_eq!(h.dims[2], UVec3::splat(4)); // L2
        assert_eq!(h.dims[3], UVec3::splat(2)); // L3
        assert_eq!(h.dims[4], UVec3::splat(1)); // L4
    }

    #[test]
    fn empty_hierarchy_rebuild() {
        let mut h = CascadedOccupancy::new(UVec3::splat(16));
        h.rebuild();
        // All child_masks should be 0
        assert!(h.levels[0].iter().all(|n| n.child_mask == 0));
        assert!(h.levels[3].iter().all(|n| n.child_mask == 0));
    }

    #[test]
    fn single_brick_propagates_to_root() {
        let mut h = CascadedOccupancy::new(UVec3::splat(16));
        // Place one occupied brick at (0,0,0)
        h.set_l0(UVec3::ZERO, 0, true);
        h.rebuild();
        // L1: (0,0,0) should have child_mask bit 0 set
        assert_ne!(h.levels[0][0].child_mask, 0);
        // L4 root should be non-empty
        assert_ne!(h.levels[3][0].child_mask, 0);
    }

    #[test]
    fn corner_brick_propagates() {
        let mut h = CascadedOccupancy::new(UVec3::splat(16));
        // Place brick at far corner (15,15,15)
        h.set_l0(UVec3::splat(15), 0, true);
        h.rebuild();
        // L4 root should detect it
        assert_ne!(h.levels[3][0].child_mask, 0);
        // L1 at (7,7,7) should have the bit for child (1,1,1) = bit 7
        let l1_idx = CascadedOccupancy::flat_index(UVec3::splat(7), UVec3::splat(8));
        assert_eq!(h.levels[0][l1_idx].child_mask & (1 << 7), 1 << 7);
    }
}
