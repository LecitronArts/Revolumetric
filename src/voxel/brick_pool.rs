use crate::voxel::brick::{BrickData, BrickOccupancy, VoxelCell, BRICK_VOLUME};
use bytemuck::Zeroable;

pub type BrickId = u32;

/// CPU-side pool storing occupancy + material data for all allocated bricks.
/// Free-list allocator: allocate() pops, free() pushes.
pub struct BrickPool {
    occupancy: Vec<BrickOccupancy>,
    materials: Vec<VoxelCell>, // flat: brick_id * 512 + morton_index
    free_list: Vec<BrickId>,
    capacity: u32,
    allocated_count: u32,
}

impl BrickPool {
    pub fn new(capacity: u32) -> Self {
        let total_voxels = capacity as usize * BRICK_VOLUME;
        Self {
            occupancy: vec![BrickOccupancy::zeroed(); capacity as usize],
            materials: vec![VoxelCell::AIR; total_voxels],
            free_list: (0..capacity).rev().collect(),
            capacity,
            allocated_count: 0,
        }
    }

    pub fn allocate(&mut self) -> Option<BrickId> {
        self.free_list.pop().map(|id| {
            self.allocated_count += 1;
            id
        })
    }

    pub fn free(&mut self, id: BrickId) {
        debug_assert!((id as usize) < self.occupancy.len());
        self.occupancy[id as usize] = BrickOccupancy::zeroed();
        let base = id as usize * BRICK_VOLUME;
        self.materials[base..base + BRICK_VOLUME].fill(VoxelCell::AIR);
        self.free_list.push(id);
        self.allocated_count -= 1;
    }

    pub fn write_brick(&mut self, id: BrickId, data: &BrickData) {
        self.occupancy[id as usize] = data.occupancy;
        let base = id as usize * BRICK_VOLUME;
        self.materials[base..base + BRICK_VOLUME].copy_from_slice(&data.materials[..]);
    }

    pub fn occupancy(&self, id: BrickId) -> &BrickOccupancy {
        &self.occupancy[id as usize]
    }

    pub fn occupancy_mut(&mut self, id: BrickId) -> &mut BrickOccupancy {
        &mut self.occupancy[id as usize]
    }

    pub fn set_material(&mut self, id: BrickId, morton: u32, cell: VoxelCell) {
        self.materials[id as usize * BRICK_VOLUME + morton as usize] = cell;
    }

    pub fn get_material(&self, id: BrickId, morton: u32) -> VoxelCell {
        self.materials[id as usize * BRICK_VOLUME + morton as usize]
    }

    pub fn occupancy_pool(&self) -> &[BrickOccupancy] { &self.occupancy }
    pub fn material_pool(&self) -> &[VoxelCell] { &self.materials }
    pub fn capacity(&self) -> u32 { self.capacity }
    pub fn allocated_count(&self) -> u32 { self.allocated_count }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::morton;

    #[test]
    fn allocate_returns_unique_ids() {
        let mut pool = BrickPool::new(4);
        let ids: Vec<_> = (0..4).filter_map(|_| pool.allocate()).collect();
        assert_eq!(ids.len(), 4);
        let set: std::collections::HashSet<_> = ids.into_iter().collect();
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn pool_exhaustion() {
        let mut pool = BrickPool::new(2);
        assert!(pool.allocate().is_some());
        assert!(pool.allocate().is_some());
        assert!(pool.allocate().is_none());
    }

    #[test]
    fn free_reuses_slot() {
        let mut pool = BrickPool::new(2);
        let id0 = pool.allocate().unwrap();
        let _id1 = pool.allocate().unwrap();
        assert_eq!(pool.allocated_count(), 2);
        pool.free(id0);
        assert_eq!(pool.allocated_count(), 1);
        let id2 = pool.allocate().unwrap();
        assert_eq!(id2, id0);
    }

    #[test]
    fn write_and_read() {
        let mut pool = BrickPool::new(4);
        let id = pool.allocate().unwrap();
        let mut data = BrickData::new();
        let cell = VoxelCell { material: 7, flags: 0, emissive: [0; 3], _pad: 0 };
        data.set_voxel(2, 3, 4, cell);
        pool.write_brick(id, &data);

        assert!(pool.occupancy(id).get(2, 3, 4));
        let m = morton::encode(2, 3, 4);
        assert_eq!(pool.get_material(id, m).material, 7);
    }

    #[test]
    fn free_clears_data() {
        let mut pool = BrickPool::new(4);
        let id = pool.allocate().unwrap();
        let mut data = BrickData::new();
        data.set_voxel(0, 0, 0, VoxelCell { material: 1, flags: 0, emissive: [0; 3], _pad: 0 });
        pool.write_brick(id, &data);
        pool.free(id);

        let id2 = pool.allocate().unwrap();
        assert_eq!(id2, id);
        assert!(pool.occupancy(id2).is_empty());
        assert_eq!(pool.get_material(id2, 0).material, 0);
    }
}
