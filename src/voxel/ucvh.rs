// src/voxel/ucvh.rs
use crate::voxel::brick::{BRICK_EDGE, BRICK_VOLUME, BrickData, VoxelCell};
use crate::voxel::brick_pool::{BrickId, BrickPool};
use crate::voxel::morton;
use crate::voxel::occupancy::CascadedOccupancy;
use glam::{IVec3, UVec3};
use std::collections::HashMap;

pub const UCVH_NO_BRICK_GENERATION: u32 = u32::MAX;

fn div_ceil_uvec3(value: UVec3, divisor: u32) -> UVec3 {
    UVec3::new(
        value.x.div_ceil(divisor),
        value.y.div_ceil(divisor),
        value.z.div_ceil(divisor),
    )
}

pub struct UcvhConfig {
    pub world_size: UVec3,
    pub brick_grid_size: UVec3,
    pub brick_capacity: u32,
}

impl UcvhConfig {
    pub fn new(world_size: UVec3) -> Self {
        let brick_grid_size = div_ceil_uvec3(world_size, BRICK_EDGE);
        // Estimate capacity at 40% fill + headroom
        let total_bricks = brick_grid_size.x * brick_grid_size.y * brick_grid_size.z;
        let capacity = (total_bricks * 2 / 5).max(64);
        Self {
            world_size,
            brick_grid_size,
            brick_capacity: capacity,
        }
    }

    pub fn with_brick_capacity(world_size: UVec3, brick_capacity: u32) -> Self {
        let brick_grid_size = div_ceil_uvec3(world_size, BRICK_EDGE);
        Self {
            world_size,
            brick_grid_size,
            brick_capacity: brick_capacity.max(64),
        }
    }
}

/// Brick-space region invalidated by ordinary UCVH content edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UcvhInvalidationRegion {
    pub brick_min: UVec3,
    pub brick_max_exclusive: UVec3,
    pub generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UcvhMotionEvent {
    pub region_min: UVec3,
    pub region_max_exclusive: UVec3,
    pub world_delta_current_from_previous: IVec3,
    pub generation: u32,
}

impl UcvhMotionEvent {
    fn has_valid_bounds(&self) -> bool {
        self.region_min.x < self.region_max_exclusive.x
            && self.region_min.y < self.region_max_exclusive.y
            && self.region_min.z < self.region_max_exclusive.z
    }

    fn overlaps(self, other: Self) -> bool {
        self.region_min.x < other.region_max_exclusive.x
            && self.region_max_exclusive.x > other.region_min.x
            && self.region_min.y < other.region_max_exclusive.y
            && self.region_max_exclusive.y > other.region_min.y
            && self.region_min.z < other.region_max_exclusive.z
            && self.region_max_exclusive.z > other.region_min.z
    }
}

/// Unified Cascaded Volume Hierarchy — the single entry point for voxel data.
pub struct Ucvh {
    pub config: UcvhConfig,
    pub pool: BrickPool,
    pub hierarchy: CascadedOccupancy,
    /// Sparse map: L0 flat index -> BrickId.
    brick_map: HashMap<u64, BrickId>,
    allocated_brick_positions: Vec<UVec3>,
    /// Brick IDs that need GPU re-upload
    dirty_bricks: Vec<BrickId>,
    dirty_brick_flags: Vec<bool>,
    brick_generations: Vec<u32>,
    /// Brick-space regions whose content edits should invalidate temporal history.
    invalidation_regions: Vec<UcvhInvalidationRegion>,
    invalidation_region_indices: HashMap<[u32; 6], usize>,
    /// Explicit semantic content motion. Ordinary edits never synthesize these.
    motion_events: Vec<UcvhMotionEvent>,
    content_generation: u32,
    /// Whether the hierarchy needs rebuild
    hierarchy_dirty: bool,
}

impl Ucvh {
    pub fn new(config: UcvhConfig) -> Self {
        Self {
            pool: BrickPool::new(config.brick_capacity),
            hierarchy: CascadedOccupancy::new(config.brick_grid_size),
            brick_map: HashMap::new(),
            allocated_brick_positions: Vec::new(),
            dirty_bricks: Vec::new(),
            dirty_brick_flags: vec![false; config.brick_capacity as usize],
            brick_generations: vec![UCVH_NO_BRICK_GENERATION; config.brick_capacity as usize],
            invalidation_regions: Vec::new(),
            invalidation_region_indices: HashMap::new(),
            motion_events: Vec::new(),
            content_generation: 0,
            hierarchy_dirty: false,
            config,
        }
    }

    /// Convert world voxel position to (brick_grid_pos, local_pos).
    fn decompose(pos: UVec3) -> (UVec3, UVec3) {
        (pos / BRICK_EDGE, pos % BRICK_EDGE)
    }

    fn contains_world_pos(&self, pos: UVec3) -> bool {
        pos.x < self.config.world_size.x
            && pos.y < self.config.world_size.y
            && pos.z < self.config.world_size.z
    }

    fn contains_brick_pos(&self, brick_pos: UVec3) -> bool {
        brick_pos.x < self.config.brick_grid_size.x
            && brick_pos.y < self.config.brick_grid_size.y
            && brick_pos.z < self.config.brick_grid_size.z
    }

    fn l0_key(&self, brick_pos: UVec3) -> u64 {
        CascadedOccupancy::flat_index(brick_pos, self.config.brick_grid_size) as u64
    }

    fn next_content_generation(&mut self) -> u32 {
        if self.content_generation == UCVH_NO_BRICK_GENERATION {
            self.content_generation = 0;
        }
        let generation = self.content_generation;
        self.content_generation = self.content_generation.wrapping_add(1);
        if self.content_generation == UCVH_NO_BRICK_GENERATION {
            self.content_generation = 0;
        }
        generation
    }

    fn set_brick_generation(&mut self, id: BrickId, generation: u32) {
        if let Some(slot) = self.brick_generations.get_mut(id as usize) {
            *slot = generation;
        }
    }

    fn record_invalidation_region(&mut self, brick_pos: UVec3, generation: u32) {
        let region = UcvhInvalidationRegion {
            brick_min: brick_pos,
            brick_max_exclusive: brick_pos + UVec3::ONE,
            generation,
        };
        let key = invalidation_region_key(region.brick_min, region.brick_max_exclusive);
        if let Some(index) = self.invalidation_region_indices.get(&key).copied() {
            if let Some(existing) = self.invalidation_regions.get_mut(index) {
                *existing = region;
            }
        } else {
            self.invalidation_region_indices
                .insert(key, self.invalidation_regions.len());
            self.invalidation_regions.push(region);
        }
    }

    fn mark_brick_dirty(&mut self, id: BrickId) {
        let index = id as usize;
        let Some(is_dirty) = self.dirty_brick_flags.get_mut(index) else {
            return;
        };
        if !*is_dirty {
            *is_dirty = true;
            self.dirty_bricks.push(id);
        }
    }

    fn motion_event_inside_world(&self, event: UcvhMotionEvent) -> bool {
        event.has_valid_bounds()
            && event.region_max_exclusive.x <= self.config.world_size.x
            && event.region_max_exclusive.y <= self.config.world_size.y
            && event.region_max_exclusive.z <= self.config.world_size.z
    }

    /// Ensure a brick exists at `brick_pos`, allocating if needed.
    fn ensure_brick(&mut self, brick_pos: UVec3) -> Option<BrickId> {
        let key = self.l0_key(brick_pos);
        if let Some(&id) = self.brick_map.get(&key) {
            return Some(id);
        }
        let id = self.pool.allocate()?;
        self.brick_map.insert(key, id);
        self.allocated_brick_positions.push(brick_pos);
        Some(id)
    }

    fn allocate_brick_with_data(&mut self, brick_pos: UVec3, data: &BrickData) -> Option<BrickId> {
        let key = self.l0_key(brick_pos);
        let id = self.pool.allocate_with_data(data)?;
        self.brick_map.insert(key, id);
        self.allocated_brick_positions.push(brick_pos);
        Some(id)
    }

    pub fn set_voxel(&mut self, pos: UVec3, cell: VoxelCell) -> bool {
        if !self.contains_world_pos(pos) {
            return false;
        }
        let (bp, lp) = Self::decompose(pos);
        let id = match self.brick_map.get(&self.l0_key(bp)).copied() {
            Some(id) => id,
            None if cell.is_air() => return true,
            None => {
                let Some(id) = self.ensure_brick(bp) else {
                    return false;
                };
                id
            }
        };
        let m = morton::encode(lp.x, lp.y, lp.z);
        let previous = self.pool.get_material(id, m);
        if voxel_cell_eq(previous, cell) {
            return true;
        }
        self.pool.set_material(id, m, cell);
        if cell.is_air() {
            self.pool.occupancy_mut(id).clear(lp.x, lp.y, lp.z);
        } else {
            self.pool.occupancy_mut(id).set(lp.x, lp.y, lp.z);
        }
        self.mark_brick_dirty(id);
        let generation = self.next_content_generation();
        self.set_brick_generation(id, generation);
        self.record_invalidation_region(bp, generation);
        self.hierarchy_dirty = true;
        true
    }

    pub fn get_voxel(&self, pos: UVec3) -> VoxelCell {
        if !self.contains_world_pos(pos) {
            return VoxelCell::AIR;
        }
        let (bp, lp) = Self::decompose(pos);
        match self.brick_map.get(&self.l0_key(bp)).copied() {
            Some(id) => self.pool.get_material(id, morton::encode(lp.x, lp.y, lp.z)),
            None => VoxelCell::AIR,
        }
    }

    /// Write a full BrickData at a brick grid position.
    pub fn write_brick(&mut self, brick_pos: UVec3, data: &BrickData) -> bool {
        if !self.contains_brick_pos(brick_pos) {
            return false;
        }
        let (id, new_with_data) = match self.brick_map.get(&self.l0_key(brick_pos)).copied() {
            Some(id) => (id, false),
            None if brick_data_is_air(data) => return true,
            None => {
                let Some(id) = self.allocate_brick_with_data(brick_pos, data) else {
                    return false;
                };
                (id, true)
            }
        };
        let base = id as usize * BRICK_VOLUME;
        if !new_with_data
            && self.pool.occupancy(id) == &data.occupancy
            && self.pool.material_pool()[base..base + BRICK_VOLUME] == data.materials[..]
        {
            return true;
        }
        if !new_with_data {
            self.pool.write_brick(id, data);
        }
        self.mark_brick_dirty(id);
        let generation = self.next_content_generation();
        self.set_brick_generation(id, generation);
        self.record_invalidation_region(brick_pos, generation);
        self.hierarchy_dirty = true;
        true
    }

    pub fn write_bricks_bulk<'a, I>(&mut self, bricks: I) -> u32
    where
        I: IntoIterator<Item = (UVec3, &'a BrickData)>,
    {
        let iter = bricks.into_iter();
        let (lower_bound, _) = iter.size_hint();
        self.brick_map.reserve(lower_bound);
        self.allocated_brick_positions.reserve(lower_bound);
        self.dirty_bricks.reserve(lower_bound);
        self.invalidation_regions.reserve(lower_bound);
        self.invalidation_region_indices.reserve(lower_bound);
        self.pool.reserve_storage_for_allocations(lower_bound);

        let mut failed = 0;
        for (brick_pos, data) in iter {
            if !self.contains_brick_pos(brick_pos) {
                failed += 1;
                continue;
            }
            let key = self.l0_key(brick_pos);
            let (id, new_with_data) = match self.brick_map.get(&key).copied() {
                Some(id) => (id, false),
                None if brick_data_is_air(data) => continue,
                None => {
                    let Some(id) = self.allocate_brick_with_data(brick_pos, data) else {
                        failed += 1;
                        continue;
                    };
                    (id, true)
                }
            };
            let base = id as usize * BRICK_VOLUME;
            if !new_with_data
                && self.pool.occupancy(id) == &data.occupancy
                && self.pool.material_pool()[base..base + BRICK_VOLUME] == data.materials[..]
            {
                continue;
            }
            if !new_with_data {
                self.pool.write_brick(id, data);
            }
            self.mark_brick_dirty(id);
            let generation = self.next_content_generation();
            self.set_brick_generation(id, generation);
            self.record_invalidation_region(brick_pos, generation);
            self.hierarchy_dirty = true;
        }
        failed
    }

    pub fn write_static_bricks_bulk<'a, I>(&mut self, bricks: I) -> u32
    where
        I: IntoIterator<Item = (UVec3, &'a BrickData)>,
    {
        let iter = bricks.into_iter();
        if self.pool.allocated_count() == 0 && self.brick_map.is_empty() {
            return self.try_write_initial_static_bricks_bulk(iter);
        }
        let (lower_bound, _) = iter.size_hint();
        self.brick_map.reserve(lower_bound);
        self.allocated_brick_positions.reserve(lower_bound);
        self.dirty_bricks.reserve(lower_bound);
        self.pool.reserve_storage_for_allocations(lower_bound);

        let mut failed = 0;
        for (brick_pos, data) in iter {
            if !self.contains_brick_pos(brick_pos) {
                failed += 1;
                continue;
            }
            let key = self.l0_key(brick_pos);
            let (id, new_with_data) = match self.brick_map.get(&key).copied() {
                Some(id) => (id, false),
                None if brick_data_is_air(data) => continue,
                None => {
                    let Some(id) = self.allocate_brick_with_data(brick_pos, data) else {
                        failed += 1;
                        continue;
                    };
                    (id, true)
                }
            };
            let base = id as usize * BRICK_VOLUME;
            if !new_with_data
                && self.pool.occupancy(id) == &data.occupancy
                && self.pool.material_pool()[base..base + BRICK_VOLUME] == data.materials[..]
            {
                continue;
            }
            if !new_with_data {
                self.pool.write_brick(id, data);
            }
            self.mark_brick_dirty(id);
            let generation = self.next_content_generation();
            self.set_brick_generation(id, generation);
            self.hierarchy_dirty = true;
        }
        failed
    }

    fn try_write_initial_static_bricks_bulk<'a, I>(&mut self, bricks: I) -> u32
    where
        I: IntoIterator<Item = (UVec3, &'a BrickData)>,
    {
        let mut failed = 0;
        let mut entries = Vec::new();
        for (brick_pos, data) in bricks {
            if !self.contains_brick_pos(brick_pos) {
                failed += 1;
                continue;
            }
            if brick_data_is_air(data) {
                continue;
            }
            entries.push((brick_pos, data));
        }
        if entries.is_empty() {
            return failed;
        }

        let capacity = self.pool.capacity() as usize;
        let alloc_count = entries.len().min(capacity);
        failed += (entries.len() - alloc_count) as u32;
        entries.truncate(alloc_count);

        let data: Vec<_> = entries.iter().map(|(_, data)| *data).collect();
        let Some(ids) = self.pool.allocate_many_with_data_into_empty(&data) else {
            return failed + entries.len() as u32;
        };
        self.brick_map.reserve(entries.len());
        self.allocated_brick_positions.reserve(entries.len());
        self.dirty_bricks.reserve(entries.len());
        for ((brick_pos, _data), id) in entries.into_iter().zip(ids) {
            self.brick_map.insert(self.l0_key(brick_pos), id);
            self.allocated_brick_positions.push(brick_pos);
            self.mark_brick_dirty(id);
            let generation = self.next_content_generation();
            self.set_brick_generation(id, generation);
            self.hierarchy_dirty = true;
        }
        failed
    }

    /// Rebuild occupancy hierarchy from current pool data.
    pub fn rebuild_hierarchy(&mut self) {
        let allocated_positions = self.allocated_brick_positions().collect::<Vec<_>>();
        let mut occupied_positions = Vec::new();
        for bp in allocated_positions {
            let Some(id) = self.brick_id_at(bp) else {
                continue;
            };
            let has_solid = !self.pool.occupancy(id).is_empty();
            self.hierarchy.set_l0(bp, id, has_solid);
            if has_solid {
                occupied_positions.push(bp);
            }
        }
        self.hierarchy
            .rebuild_from_occupied_l0_positions(occupied_positions);
        self.hierarchy_dirty = false;
    }

    pub fn take_dirty_bricks(&mut self) -> Vec<BrickId> {
        for id in &self.dirty_bricks {
            if let Some(is_dirty) = self.dirty_brick_flags.get_mut(*id as usize) {
                *is_dirty = false;
            }
        }
        std::mem::take(&mut self.dirty_bricks)
    }

    pub fn take_invalidation_regions(&mut self) -> Vec<UcvhInvalidationRegion> {
        self.invalidation_region_indices.clear();
        std::mem::take(&mut self.invalidation_regions)
    }

    pub fn invalidation_regions(&self) -> &[UcvhInvalidationRegion] {
        &self.invalidation_regions
    }

    pub fn brick_id_at(&self, brick_pos: UVec3) -> Option<BrickId> {
        if !self.contains_brick_pos(brick_pos) {
            return None;
        }
        self.brick_map.get(&self.l0_key(brick_pos)).copied()
    }

    pub fn brick_generation(&self, brick_id: BrickId) -> Option<u32> {
        self.brick_generations.get(brick_id as usize).copied()
    }

    pub fn brick_generations(&self) -> &[u32] {
        &self.brick_generations
    }

    pub fn push_motion_event(&mut self, event: UcvhMotionEvent) -> bool {
        if !self.motion_event_inside_world(event) {
            return false;
        }
        if self
            .motion_events
            .iter()
            .any(|existing| existing.overlaps(event))
        {
            return false;
        }
        self.motion_events.push(event);
        true
    }

    pub fn motion_events(&self) -> &[UcvhMotionEvent] {
        &self.motion_events
    }

    pub fn take_motion_events(&mut self) -> Vec<UcvhMotionEvent> {
        std::mem::take(&mut self.motion_events)
    }

    pub fn is_hierarchy_dirty(&self) -> bool {
        self.hierarchy_dirty
    }

    pub fn allocated_brick_count(&self) -> u32 {
        self.pool.allocated_count()
    }

    pub fn allocated_brick_positions(&self) -> impl Iterator<Item = UVec3> + '_ {
        self.allocated_brick_positions.iter().copied()
    }
}

fn voxel_cell_eq(a: VoxelCell, b: VoxelCell) -> bool {
    a.material == b.material && a.flags == b.flags && a.emissive == b.emissive && a._pad == b._pad
}

fn invalidation_region_key(brick_min: UVec3, brick_max_exclusive: UVec3) -> [u32; 6] {
    [
        brick_min.x,
        brick_min.y,
        brick_min.z,
        brick_max_exclusive.x,
        brick_max_exclusive.y,
        brick_max_exclusive.z,
    ]
}

fn brick_data_is_air(data: &BrickData) -> bool {
    data.occupancy.is_empty() && data.materials.iter().all(|cell| *cell == VoxelCell::AIR)
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
    fn config_can_use_explicit_capacity_for_sparse_large_worlds() {
        let c = UcvhConfig::with_brick_capacity(UVec3::splat(512), 16_384);

        assert_eq!(c.world_size, UVec3::splat(512));
        assert_eq!(c.brick_grid_size, UVec3::splat(64));
        assert_eq!(c.brick_capacity, 16_384);
        assert!(c.brick_capacity < UcvhConfig::new(UVec3::splat(512)).brick_capacity);
    }

    #[test]
    fn non_aligned_world_size_can_write_last_valid_voxel() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(129, 9, 8)));
        let cell = VoxelCell::new(1, 0, [0; 3]);

        assert!(u.set_voxel(UVec3::new(128, 8, 7), cell));
        assert_eq!(u.get_voxel(UVec3::new(128, 8, 7)).material, 1);
    }

    #[test]
    fn tiny_world_size_can_write_valid_voxel() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(1, 1, 1)));
        let cell = VoxelCell::new(1, 0, [0; 3]);

        assert!(u.set_voxel(UVec3::ZERO, cell));
        assert_eq!(u.get_voxel(UVec3::ZERO).material, 1);
    }

    #[test]
    fn setting_air_in_missing_brick_does_not_allocate() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(9, 9, 9)));

        assert!(u.set_voxel(UVec3::new(8, 8, 8), VoxelCell::AIR));
        assert_eq!(u.allocated_brick_count(), 0);
        assert_eq!(u.get_voxel(UVec3::new(8, 8, 8)).material, 0);
        assert!(!u.is_hierarchy_dirty());
        assert!(u.take_dirty_bricks().is_empty());
    }

    #[test]
    fn set_and_get_voxel() {
        let mut u = test_ucvh();
        let cell = VoxelCell {
            material: 5,
            flags: 0,
            emissive: [0; 3],
            _pad: 0,
        };
        assert!(u.set_voxel(UVec3::new(10, 20, 30), cell));
        assert_eq!(u.get_voxel(UVec3::new(10, 20, 30)).material, 5);
        assert_eq!(u.get_voxel(UVec3::new(0, 0, 0)).material, 0); // air
    }

    #[test]
    fn dirty_tracking() {
        let mut u = test_ucvh();
        let cell = VoxelCell {
            material: 1,
            flags: 0,
            emissive: [0; 3],
            _pad: 0,
        };
        u.set_voxel(UVec3::new(0, 0, 0), cell);
        u.set_voxel(UVec3::new(1, 0, 0), cell); // same brick
        let dirty = u.take_dirty_bricks();
        assert_eq!(dirty.len(), 1); // one brick, not two
    }

    #[test]
    fn dirty_tracking_marks_brick_again_after_drain() {
        let mut u = test_ucvh();
        let cell = VoxelCell::new(1, 0, [0; 3]);

        assert!(u.set_voxel(UVec3::new(1, 1, 1), cell));
        assert_eq!(u.take_dirty_bricks().len(), 1);
        assert!(u.set_voxel(UVec3::new(2, 1, 1), cell));

        let dirty = u.take_dirty_bricks();
        assert_eq!(dirty.len(), 1);
    }

    #[test]
    fn hierarchy_rebuild_propagates() {
        let mut u = test_ucvh();
        let cell = VoxelCell {
            material: 1,
            flags: 0,
            emissive: [0; 3],
            _pad: 0,
        };
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
                    data.set_voxel(
                        x,
                        y,
                        z,
                        VoxelCell {
                            material: 1,
                            flags: 0,
                            emissive: [0; 3],
                            _pad: 0,
                        },
                    );
                }
            }
        }
        assert!(u.write_brick(UVec3::ZERO, &data));
        assert_eq!(u.pool.occupancy(0).count, 512);
    }

    #[test]
    fn write_bricks_bulk_imports_multiple_staged_bricks() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(16, 8, 8)));
        let mut first = BrickData::new();
        first.set_voxel(0, 0, 0, VoxelCell::new(3, 0, [0; 3]));
        let mut second = BrickData::new();
        second.set_voxel(1, 0, 0, VoxelCell::new(4, 0, [0; 3]));
        let entries = [
            (UVec3::ZERO, &first),
            (UVec3::new(1, 0, 0), &second),
            (UVec3::new(2, 0, 0), &first),
        ];

        let failed = u.write_bricks_bulk(entries);

        assert_eq!(failed, 1);
        assert_eq!(u.allocated_brick_count(), 2);
        assert_eq!(u.get_voxel(UVec3::ZERO).material, 3);
        assert_eq!(u.get_voxel(UVec3::new(BRICK_EDGE + 1, 0, 0)).material, 4);
        assert!(u.is_hierarchy_dirty());
    }

    #[test]
    fn write_static_bricks_bulk_marks_dirty_without_per_brick_invalidation() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(16, 8, 8)));
        let mut first = BrickData::new();
        first.set_voxel(0, 0, 0, VoxelCell::new(3, 0, [0; 3]));
        let mut second = BrickData::new();
        second.set_voxel(1, 0, 0, VoxelCell::new(4, 0, [0; 3]));
        let entries = [(UVec3::ZERO, &first), (UVec3::new(1, 0, 0), &second)];

        let failed = u.write_static_bricks_bulk(entries);

        assert_eq!(failed, 0);
        assert_eq!(u.allocated_brick_count(), 2);
        assert_eq!(u.take_dirty_bricks().len(), 2);
        assert!(u.invalidation_regions().is_empty());
        assert!(u.is_hierarchy_dirty());
    }

    #[test]
    fn write_static_bricks_bulk_uses_initial_empty_pool_fast_path() {
        let source = crate::render::source_checks::read_source("src/voxel/ucvh.rs");
        let body = source
            .split("pub fn write_static_bricks_bulk")
            .nth(1)
            .expect("write_static_bricks_bulk should exist")
            .split("/// Rebuild occupancy hierarchy")
            .next()
            .expect("write_static_bricks_bulk should end before hierarchy rebuild");

        assert!(
            body.contains("try_write_initial_static_bricks_bulk"),
            "Vintessa initial import should route empty UCVH static bricks through a contiguous bulk allocation fast path"
        );
        assert!(
            source.contains("allocate_many_with_data_into_empty"),
            "initial static bulk import should avoid per-brick pool growth/copy overhead"
        );
    }

    #[test]
    fn write_brick_repeated_identical_data_does_not_duplicate_invalidation_region() {
        let mut u = test_ucvh();
        let mut data = BrickData::new();
        data.set_voxel(0, 0, 0, VoxelCell::new(2, 0, [1, 2, 3]));

        assert!(u.write_brick(UVec3::new(2, 3, 4), &data));
        assert_eq!(u.take_invalidation_regions().len(), 1);

        assert!(u.write_brick(UVec3::new(2, 3, 4), &data));
        assert!(u.take_invalidation_regions().is_empty());
    }

    #[test]
    fn adjacent_bricks_preserve_continuous_wall_across_boundary() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(16, 8, 8)));

        for y in 0..8 {
            for z in 0..8 {
                assert!(u.set_voxel(UVec3::new(7, y, z), VoxelCell::new(3, 1, [0; 3])));
                assert!(u.set_voxel(UVec3::new(8, y, z), VoxelCell::new(3, 1, [0; 3])));
            }
        }
        u.rebuild_hierarchy();

        assert_eq!(u.config.brick_grid_size, UVec3::new(2, 1, 1));
        assert_eq!(u.allocated_brick_count(), 2);

        for y in 0..8 {
            for z in 0..8 {
                assert_eq!(
                    u.get_voxel(UVec3::new(7, y, z)).material,
                    3,
                    "left brick boundary voxel should remain solid at y={y} z={z}"
                );
                assert_eq!(
                    u.get_voxel(UVec3::new(8, y, z)).material,
                    3,
                    "right brick boundary voxel should remain solid at y={y} z={z}"
                );
            }
        }

        assert_ne!(u.hierarchy.get_l0(UVec3::new(0, 0, 0)).brick_id, u32::MAX);
        assert_ne!(u.hierarchy.get_l0(UVec3::new(1, 0, 0)).brick_id, u32::MAX);
        assert_eq!(u.hierarchy.get_l0(UVec3::new(0, 0, 0)).flags & 1, 1);
        assert_eq!(u.hierarchy.get_l0(UVec3::new(1, 0, 0)).flags & 1, 1);
    }

    #[test]
    fn out_of_bounds_set_voxel_returns_false_without_allocating() {
        let mut u = test_ucvh();
        let cell = VoxelCell {
            material: 1,
            flags: 0,
            emissive: [0; 3],
            _pad: 0,
        };

        assert!(!u.set_voxel(UVec3::new(128, 0, 0), cell));
        assert_eq!(u.allocated_brick_count(), 0);
    }

    #[test]
    fn out_of_bounds_get_voxel_returns_air() {
        let u = test_ucvh();

        assert_eq!(
            u.get_voxel(UVec3::new(0, 128, 0)).material,
            VoxelCell::AIR.material
        );
    }

    #[test]
    fn out_of_bounds_write_brick_returns_false_without_allocating() {
        let mut u = test_ucvh();
        let data = BrickData::new();

        assert!(!u.write_brick(UVec3::new(16, 0, 0), &data));
        assert_eq!(u.allocated_brick_count(), 0);
    }

    #[test]
    fn set_voxel_records_a_single_invalidated_brick_region() {
        let mut u = test_ucvh();
        let cell = VoxelCell::new(1, 0, [0; 3]);

        assert!(u.set_voxel(UVec3::new(2, 3, 4), cell));

        let invalidations = u.take_invalidation_regions();
        assert_eq!(invalidations.len(), 1);
        assert_eq!(invalidations[0].brick_min, UVec3::ZERO);
        assert_eq!(invalidations[0].brick_max_exclusive, UVec3::new(1, 1, 1));
    }

    #[test]
    fn write_brick_records_the_full_brick_region_and_generation() {
        let mut u = test_ucvh();
        let mut data = BrickData::new();
        data.set_voxel(0, 0, 0, VoxelCell::new(1, 0, [0; 3]));

        assert!(u.write_brick(UVec3::new(1, 2, 3), &data));

        let invalidations = u.take_invalidation_regions();
        assert_eq!(invalidations.len(), 1);
        assert_eq!(invalidations[0].brick_min, UVec3::new(1, 2, 3));
        assert_eq!(invalidations[0].brick_max_exclusive, UVec3::new(2, 3, 4));
        assert_eq!(invalidations[0].generation, 0);
    }

    #[test]
    fn write_brick_with_empty_data_on_missing_brick_is_a_no_op() {
        let mut u = test_ucvh();
        let data = BrickData::new();

        assert!(u.write_brick(UVec3::new(1, 2, 3), &data));
        assert_eq!(u.allocated_brick_count(), 0);
        assert!(u.take_invalidation_regions().is_empty());
    }

    #[test]
    fn semantic_move_rejects_overlapping_regions() {
        let mut u = test_ucvh();
        let event = UcvhMotionEvent {
            region_min: UVec3::new(4, 4, 4),
            region_max_exclusive: UVec3::new(8, 8, 8),
            world_delta_current_from_previous: glam::IVec3::new(1, 0, 0),
            generation: 42,
        };

        assert!(u.push_motion_event(event));
        assert!(!u.push_motion_event(UcvhMotionEvent {
            region_min: UVec3::new(6, 6, 6),
            region_max_exclusive: UVec3::new(9, 9, 9),
            world_delta_current_from_previous: glam::IVec3::new(0, 1, 0),
            generation: 43,
        }));
    }

    #[test]
    fn next_content_generation_skips_sentinel() {
        let mut u = test_ucvh();
        u.content_generation = UCVH_NO_BRICK_GENERATION - 1;

        assert_eq!(u.next_content_generation(), UCVH_NO_BRICK_GENERATION - 1);
        let wrapped = u.next_content_generation();

        assert_ne!(wrapped, UCVH_NO_BRICK_GENERATION);
        assert_eq!(wrapped, 0);
    }

    #[test]
    fn set_voxel_updates_generation_for_allocated_brick_slot() {
        let mut u = test_ucvh();

        assert!(u.set_voxel(UVec3::new(2, 3, 4), VoxelCell::new(8, 0, [0; 3])));

        let invalidation = u.invalidation_regions()[0];
        let brick_id = u
            .brick_id_at(UVec3::ZERO)
            .expect("brick should be allocated");
        assert_eq!(u.brick_generation(brick_id), Some(invalidation.generation));
    }

    #[test]
    fn unallocated_brick_generation_is_sentinel() {
        let u = test_ucvh();

        assert_eq!(u.brick_generation(0), Some(UCVH_NO_BRICK_GENERATION));
        assert_eq!(u.brick_id_at(UVec3::new(1, 2, 3)), None);
    }

    #[test]
    fn allocated_brick_positions_reports_sparse_grid_coordinates() {
        let mut u = Ucvh::new(UcvhConfig::with_brick_capacity(
            UVec3::new(2048, 768, 2048),
            64,
        ));

        assert!(u.set_voxel(UVec3::new(0, 0, 0), VoxelCell::new(2, 0, [0; 3])));
        assert!(u.set_voxel(UVec3::new(2040, 760, 2040), VoxelCell::new(3, 0, [0; 3])));

        let positions = u.allocated_brick_positions().collect::<Vec<_>>();

        assert_eq!(positions, vec![UVec3::ZERO, UVec3::new(255, 95, 255)]);
    }

    #[test]
    fn rebuild_hierarchy_uses_sparse_allocated_brick_positions() {
        let source = crate::render::source_checks::read_source("src/voxel/ucvh.rs");
        let body = source
            .split("pub fn rebuild_hierarchy")
            .nth(1)
            .expect("rebuild_hierarchy should exist")
            .split("pub fn take_dirty_bricks")
            .next()
            .expect("rebuild_hierarchy should end before dirty draining");

        assert!(
            body.contains("allocated_brick_positions"),
            "Vintessa-scale startup must rebuild UCVH hierarchy from sparse allocated bricks"
        );
        assert!(
            !body.contains("0..bgs.z"),
            "rebuild_hierarchy must not scan every brick-grid coordinate during startup"
        );
    }

    #[test]
    fn allocated_brick_positions_uses_tracked_sparse_positions_not_full_brick_map_scan() {
        let source = crate::render::source_checks::read_source("src/voxel/ucvh.rs");
        let fields = source
            .split("pub struct Ucvh {")
            .nth(1)
            .expect("Ucvh struct should exist")
            .split("impl Ucvh")
            .next()
            .expect("Ucvh fields should end before impl");
        let method = source
            .split("pub fn allocated_brick_positions")
            .nth(1)
            .expect("allocated_brick_positions should exist")
            .split("}")
            .next()
            .expect("allocated_brick_positions should have a body");

        assert!(
            fields.contains("allocated_brick_positions: Vec<UVec3>"),
            "Ucvh must track sparse allocated brick coordinates as bricks are allocated"
        );
        assert!(
            !method.contains("brick_map") && !method.contains("enumerate"),
            "allocated_brick_positions must not scan the full Vintessa-scale brick map"
        );
    }

    #[test]
    fn ucvh_uses_sparse_brick_map_not_dense_world_sized_index() {
        let source = crate::render::source_checks::read_source("src/voxel/ucvh.rs");
        let fields = source
            .split("pub struct Ucvh {")
            .nth(1)
            .expect("Ucvh struct should exist")
            .split("impl Ucvh")
            .next()
            .expect("Ucvh fields should end before impl");
        let constructor = source
            .split("pub fn new(config: UcvhConfig) -> Self")
            .nth(1)
            .expect("Ucvh::new should exist")
            .split("/// Convert world voxel position")
            .next()
            .expect("constructor should end before helpers");

        assert!(
            fields.contains("brick_map: HashMap<u64, BrickId>"),
            "Vintessa-scale UCVH must use a sparse brick coordinate map"
        );
        assert!(
            !constructor.contains("vec![None; l0_count]") && !constructor.contains("l0_count"),
            "Ucvh::new must not allocate one brick-map slot per empty brick-grid coordinate"
        );
    }
}
