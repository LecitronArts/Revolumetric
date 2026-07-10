use crate::voxel::brick::{BRICK_VOLUME, BrickData, BrickOccupancy, VoxelCell};
use bytemuck::Zeroable;
use std::mem::MaybeUninit;

pub type BrickId = u32;

/// CPU-side pool storing occupancy + material data for all allocated bricks.
/// Free-list allocator: allocate() pops, free() pushes.
pub struct BrickPool {
    occupancy: Vec<BrickOccupancy>,
    materials: Vec<VoxelCell>, // flat: brick_id * 512 + morton_index
    free_list: Vec<BrickId>,
    allocated: Vec<bool>,
    capacity: u32,
    allocated_count: u32,
}

impl BrickPool {
    pub fn new(capacity: u32) -> Self {
        Self {
            occupancy: Vec::new(),
            materials: Vec::new(),
            free_list: (0..capacity).rev().collect(),
            allocated: vec![false; capacity as usize],
            capacity,
            allocated_count: 0,
        }
    }

    pub fn allocate(&mut self) -> Option<BrickId> {
        let id = self.free_list.pop()?;
        self.ensure_storage_for(id);
        self.allocated[id as usize] = true;
        self.allocated_count += 1;
        Some(id)
    }

    pub fn allocate_with_data(&mut self, data: &BrickData) -> Option<BrickId> {
        let id = self.free_list.pop()?;
        let index = id as usize;
        if index == self.occupancy.len() {
            self.occupancy.push(data.occupancy);
            self.materials.extend_from_slice(&data.materials[..]);
        } else {
            self.ensure_storage_for(id);
            self.occupancy[index] = data.occupancy;
            let base = index * BRICK_VOLUME;
            self.materials[base..base + BRICK_VOLUME].copy_from_slice(&data.materials[..]);
        }
        self.allocated[index] = true;
        self.allocated_count += 1;
        Some(id)
    }

    pub fn allocate_many_with_data_into_empty(
        &mut self,
        bricks: &[&BrickData],
    ) -> Option<Vec<BrickId>> {
        if self.allocated_count != 0
            || !self.occupancy.is_empty()
            || !self.materials.is_empty()
            || bricks.len() > self.free_list.len()
        {
            return None;
        }

        let mut ids = Vec::with_capacity(bricks.len());
        for _ in bricks {
            let id = self.free_list.pop()?;
            self.allocated[id as usize] = true;
            ids.push(id);
        }
        self.allocated_count = ids.len() as u32;

        self.occupancy
            .extend(bricks.iter().map(|brick| brick.occupancy));

        let material_len = bricks.len().saturating_mul(BRICK_VOLUME);
        self.materials = copy_brick_materials_parallel(material_len, bricks);

        Some(ids)
    }

    pub fn reserve_storage_for_allocations(&mut self, additional: usize) {
        self.occupancy.reserve(additional);
        self.materials
            .reserve(additional.saturating_mul(BRICK_VOLUME));
    }

    fn ensure_storage_for(&mut self, id: BrickId) {
        let len = id as usize + 1;
        if self.occupancy.len() < len {
            self.occupancy.resize(len, BrickOccupancy::zeroed());
        }
        let material_len = len * BRICK_VOLUME;
        if self.materials.len() < material_len {
            self.materials.resize(material_len, VoxelCell::AIR);
        }
    }

    pub fn free(&mut self, id: BrickId) -> bool {
        if id >= self.capacity || !self.allocated[id as usize] {
            return false;
        }
        self.allocated[id as usize] = false;
        if let Some(occupancy) = self.occupancy.get_mut(id as usize) {
            *occupancy = BrickOccupancy::zeroed();
        }
        let base = id as usize * BRICK_VOLUME;
        if let Some(materials) = self.materials.get_mut(base..base + BRICK_VOLUME) {
            materials.fill(VoxelCell::AIR);
        }
        self.free_list.push(id);
        self.allocated_count -= 1;
        true
    }

    pub fn write_brick(&mut self, id: BrickId, data: &BrickData) {
        self.ensure_storage_for(id);
        self.occupancy[id as usize] = data.occupancy;
        let base = id as usize * BRICK_VOLUME;
        self.materials[base..base + BRICK_VOLUME].copy_from_slice(&data.materials[..]);
    }

    pub fn occupancy(&self, id: BrickId) -> &BrickOccupancy {
        &self.occupancy[id as usize]
    }

    pub fn occupancy_mut(&mut self, id: BrickId) -> &mut BrickOccupancy {
        self.ensure_storage_for(id);
        &mut self.occupancy[id as usize]
    }

    pub fn set_material(&mut self, id: BrickId, morton: u32, cell: VoxelCell) {
        self.ensure_storage_for(id);
        self.materials[id as usize * BRICK_VOLUME + morton as usize] = cell;
    }

    pub fn get_material(&self, id: BrickId, morton: u32) -> VoxelCell {
        self.materials[id as usize * BRICK_VOLUME + morton as usize]
    }

    pub fn occupancy_pool(&self) -> &[BrickOccupancy] {
        &self.occupancy
    }
    pub fn material_pool(&self) -> &[VoxelCell] {
        &self.materials
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn allocated_count(&self) -> u32 {
        self.allocated_count
    }
}

fn copy_brick_materials_parallel(material_len: usize, bricks: &[&BrickData]) -> Vec<VoxelCell> {
    let mut materials = Vec::<MaybeUninit<VoxelCell>>::with_capacity(material_len);
    // VoxelCell is Copy with no Drop. Every slot is initialized by the scoped copy below before
    // converting the buffer into Vec<VoxelCell>.
    unsafe {
        materials.set_len(material_len);
    }
    if bricks.is_empty() {
        return Vec::new();
    }
    let worker_count = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(bricks.len());
    let chunk_bricks = bricks.len().div_ceil(worker_count);
    let chunk_cells = chunk_bricks * BRICK_VOLUME;

    std::thread::scope(|scope| {
        for (dst_chunk, brick_chunk) in materials
            .chunks_mut(chunk_cells)
            .zip(bricks.chunks(chunk_bricks))
        {
            scope.spawn(move || {
                for (dst, brick) in dst_chunk.chunks_mut(BRICK_VOLUME).zip(brick_chunk.iter()) {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            brick.materials.as_ptr(),
                            dst.as_mut_ptr() as *mut VoxelCell,
                            BRICK_VOLUME,
                        );
                    }
                }
            });
        }
    });

    let ptr = materials.as_mut_ptr() as *mut VoxelCell;
    let len = materials.len();
    let capacity = materials.capacity();
    std::mem::forget(materials);
    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
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
    fn pool_storage_grows_with_allocated_bricks_not_capacity() {
        let mut pool = BrickPool::new(1_000_000);

        assert_eq!(pool.occupancy_pool().len(), 0);
        assert_eq!(pool.material_pool().len(), 0);

        let id = pool.allocate().expect("first brick should allocate");

        assert_eq!(id, 0);
        assert_eq!(pool.occupancy_pool().len(), 1);
        assert_eq!(pool.material_pool().len(), BRICK_VOLUME);
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
    fn double_free_is_rejected_without_corrupting_free_list() {
        let mut pool = BrickPool::new(2);
        let id = pool.allocate().unwrap();

        assert!(pool.free(id));
        assert!(!pool.free(id));
        assert_eq!(pool.allocated_count(), 0);

        let a = pool.allocate().unwrap();
        let b = pool.allocate().unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn free_rejects_never_allocated_id() {
        let mut pool = BrickPool::new(2);

        assert!(!pool.free(1));
        assert_eq!(pool.allocated_count(), 0);
        assert_eq!(pool.allocate(), Some(0));
        assert_eq!(pool.allocate(), Some(1));
    }

    #[test]
    fn free_rejects_out_of_bounds_id() {
        let mut pool = BrickPool::new(2);

        assert!(!pool.free(2));
        assert!(!pool.free(u32::MAX));
        assert_eq!(pool.allocated_count(), 0);
        assert_eq!(pool.allocate(), Some(0));
        assert_eq!(pool.allocate(), Some(1));
    }

    #[test]
    fn write_and_read() {
        let mut pool = BrickPool::new(4);
        let id = pool.allocate().unwrap();
        let mut data = BrickData::new();
        let cell = VoxelCell {
            material: 7,
            flags: 0,
            emissive: [0; 3],
            _pad: 0,
        };
        data.set_voxel(2, 3, 4, cell);
        pool.write_brick(id, &data);

        assert!(pool.occupancy(id).get(2, 3, 4));
        let m = morton::encode(2, 3, 4);
        assert_eq!(pool.get_material(id, m).material, 7);
    }

    #[test]
    fn allocate_with_data_initializes_new_brick_without_prior_air_storage() {
        let mut pool = BrickPool::new(4);
        let mut data = BrickData::new();
        data.set_voxel(2, 3, 4, VoxelCell::new(7, 1, [0; 3]));

        let id = pool
            .allocate_with_data(&data)
            .expect("brick should allocate with initial data");

        assert_eq!(id, 0);
        assert_eq!(pool.allocated_count(), 1);
        assert_eq!(pool.occupancy_pool().len(), 1);
        assert_eq!(pool.material_pool().len(), BRICK_VOLUME);
        assert!(pool.occupancy(id).get(2, 3, 4));
        assert_eq!(pool.get_material(id, morton::encode(2, 3, 4)).material, 7);
    }

    #[test]
    fn reserve_storage_for_allocations_preserves_subsequent_allocations() {
        let mut pool = BrickPool::new(4);
        pool.reserve_storage_for_allocations(3);
        let data = BrickData::new();

        let first = pool.allocate_with_data(&data).expect("first brick");
        let second = pool.allocate_with_data(&data).expect("second brick");

        assert_eq!(first, 0);
        assert_eq!(second, 1);
        assert_eq!(pool.allocated_count(), 2);
        assert_eq!(pool.material_pool().len(), BRICK_VOLUME * 2);
    }

    #[test]
    fn bulk_allocate_into_empty_pool_copies_bricks_in_one_initial_layout() {
        let mut pool = BrickPool::new(4);
        let mut first = BrickData::new();
        first.set_voxel(1, 2, 3, VoxelCell::new(7, 1, [0; 3]));
        let mut second = BrickData::new();
        second.set_voxel(4, 5, 6, VoxelCell::new(9, 1, [1, 2, 3]));

        let ids = pool
            .allocate_many_with_data_into_empty(&[&first, &second])
            .expect("empty pool bulk allocation should fit");

        assert_eq!(ids, vec![0, 1]);
        assert_eq!(pool.allocated_count(), 2);
        assert_eq!(pool.occupancy_pool().len(), 2);
        assert_eq!(pool.material_pool().len(), BRICK_VOLUME * 2);
        assert_eq!(
            pool.get_material(ids[0], morton::encode(1, 2, 3)).material,
            7
        );
        assert_eq!(
            pool.get_material(ids[1], morton::encode(4, 5, 6)).emissive,
            [1, 2, 3]
        );
    }

    #[test]
    fn free_clears_data() {
        let mut pool = BrickPool::new(4);
        let id = pool.allocate().unwrap();
        let mut data = BrickData::new();
        data.set_voxel(
            0,
            0,
            0,
            VoxelCell {
                material: 1,
                flags: 0,
                emissive: [0; 3],
                _pad: 0,
            },
        );
        pool.write_brick(id, &data);
        pool.free(id);

        let id2 = pool.allocate().unwrap();
        assert_eq!(id2, id);
        assert!(pool.occupancy(id2).is_empty());
        assert_eq!(pool.get_material(id2, 0).material, 0);
    }
}
