// src/voxel/ucvh.rs
use crate::voxel::brick::{BRICK_EDGE, BRICK_VOLUME, BrickData, BrickOccupancy, VoxelCell};
use crate::voxel::brick_pool::{BrickId, BrickPool};
use crate::voxel::morton;
use crate::voxel::occupancy::{CascadedOccupancy, CascadedOccupancyChanges};
use glam::{IVec3, UVec3};
use std::collections::{BTreeSet, HashMap};

pub const UCVH_NO_BRICK_GENERATION: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct UcvhPageBoundaryMask(u8);

impl UcvhPageBoundaryMask {
    pub const NONE: Self = Self(0);
    pub const NEG_X: Self = Self(1 << 0);
    pub const POS_X: Self = Self(1 << 1);
    pub const NEG_Y: Self = Self(1 << 2);
    pub const POS_Y: Self = Self(1 << 3);
    pub const NEG_Z: Self = Self(1 << 4);
    pub const POS_Z: Self = Self(1 << 5);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    fn from_local_voxel(local: UVec3) -> Self {
        let mut mask = Self::NONE;
        if local.x == 0 {
            mask.insert(Self::NEG_X);
        }
        if local.x + 1 == BRICK_EDGE {
            mask.insert(Self::POS_X);
        }
        if local.y == 0 {
            mask.insert(Self::NEG_Y);
        }
        if local.y + 1 == BRICK_EDGE {
            mask.insert(Self::POS_Y);
        }
        if local.z == 0 {
            mask.insert(Self::NEG_Z);
        }
        if local.z + 1 == BRICK_EDGE {
            mask.insert(Self::POS_Z);
        }
        mask
    }
}

impl std::ops::BitOr for UcvhPageBoundaryMask {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for UcvhPageBoundaryMask {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

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

/// A single authoritative brick change that must reach render-side consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UcvhChangedBrick {
    pub brick_id: BrickId,
    pub brick_coord: UVec3,
    pub generation: u32,
    pub revision: u64,
    pub occupancy_changed: bool,
    pub material_changed: bool,
    pub touched_boundaries: UcvhPageBoundaryMask,
}

/// A stable snapshot of render-relevant UCVH changes.
///
/// The batch remains pending until acknowledged. Acknowledgement clears only
/// the generations represented by this snapshot, so a newer edit to the same
/// brick cannot be lost while an upload or derived render operation is pending.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UcvhRenderChangeBatch {
    pub id: u64,
    pub bricks: Vec<UcvhChangedBrick>,
    /// Changed render pages plus their in-bounds face neighbors.
    ///
    /// A render page is exactly one UCVH `8^3` brick. This is the precise
    /// invalidation set for derived exposed-face page geometry.
    pub invalidated_pages: Vec<UVec3>,
    /// Deprecated aggregate-cache invalidation retained until all existing
    /// callers move from `16^3` render cells to page ownership.
    pub invalidated_render_cells: Vec<UVec3>,
}

impl UcvhRenderChangeBatch {
    pub fn is_empty(&self) -> bool {
        self.bricks.is_empty()
    }
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
    brick_coords_by_id: Vec<Option<UVec3>>,
    /// Brick IDs that need GPU re-upload
    dirty_bricks: Vec<BrickId>,
    dirty_brick_flags: Vec<bool>,
    brick_generations: Vec<u32>,
    brick_topology_revisions: Vec<u64>,
    pending_render_change_generations: Vec<u32>,
    pending_render_change_revisions: Vec<u64>,
    pending_render_occupancy_changes: Vec<bool>,
    pending_render_material_changes: Vec<bool>,
    pending_render_touched_boundaries: Vec<UcvhPageBoundaryMask>,
    pending_render_change_ids: BTreeSet<BrickId>,
    next_render_change_batch_id: u64,
    next_render_change_revision: u64,
    pending_render_change_snapshot: Option<UcvhRenderChangeBatch>,
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
            brick_coords_by_id: vec![None; config.brick_capacity as usize],
            dirty_bricks: Vec::new(),
            dirty_brick_flags: vec![false; config.brick_capacity as usize],
            brick_generations: vec![UCVH_NO_BRICK_GENERATION; config.brick_capacity as usize],
            brick_topology_revisions: vec![0; config.brick_capacity as usize],
            pending_render_change_generations: vec![
                UCVH_NO_BRICK_GENERATION;
                config.brick_capacity as usize
            ],
            pending_render_change_revisions: vec![0; config.brick_capacity as usize],
            pending_render_occupancy_changes: vec![false; config.brick_capacity as usize],
            pending_render_material_changes: vec![false; config.brick_capacity as usize],
            pending_render_touched_boundaries: vec![
                UcvhPageBoundaryMask::NONE;
                config.brick_capacity as usize
            ],
            pending_render_change_ids: BTreeSet::new(),
            next_render_change_batch_id: 1,
            next_render_change_revision: 1,
            pending_render_change_snapshot: None,
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

    fn mark_render_change(
        &mut self,
        id: BrickId,
        generation: u32,
        occupancy_changed: bool,
        material_changed: bool,
        touched_boundaries: UcvhPageBoundaryMask,
    ) {
        let index = id as usize;
        let revision = self.next_render_change_revision;
        if let (
            Some(generation_slot),
            Some(revision_slot),
            Some(occupancy_slot),
            Some(material_slot),
            Some(boundary_slot),
        ) = (
            self.pending_render_change_generations.get_mut(index),
            self.pending_render_change_revisions.get_mut(index),
            self.pending_render_occupancy_changes.get_mut(index),
            self.pending_render_material_changes.get_mut(index),
            self.pending_render_touched_boundaries.get_mut(index),
        ) {
            *generation_slot = generation;
            *revision_slot = revision;
            *occupancy_slot |= occupancy_changed;
            *material_slot |= material_changed;
            *boundary_slot |= touched_boundaries;
            if occupancy_changed
                && let Some(topology_revision) = self.brick_topology_revisions.get_mut(index)
            {
                *topology_revision = revision;
            }
            self.next_render_change_revision = self
                .next_render_change_revision
                .checked_add(1)
                .expect("UCVH render change revision space exhausted");
            self.pending_render_change_ids.insert(id);
        }
    }

    fn record_content_change(
        &mut self,
        id: BrickId,
        brick_pos: UVec3,
        generation: u32,
        occupancy_changed: bool,
        material_changed: bool,
        touched_boundaries: UcvhPageBoundaryMask,
    ) {
        self.set_brick_generation(id, generation);
        self.mark_render_change(
            id,
            generation,
            occupancy_changed,
            material_changed,
            touched_boundaries,
        );
        self.record_invalidation_region(brick_pos, generation);
    }

    fn record_static_content_change(
        &mut self,
        id: BrickId,
        generation: u32,
        occupancy_changed: bool,
        material_changed: bool,
        touched_boundaries: UcvhPageBoundaryMask,
    ) {
        self.set_brick_generation(id, generation);
        self.mark_render_change(
            id,
            generation,
            occupancy_changed,
            material_changed,
            touched_boundaries,
        );
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
        self.set_brick_coord(id, brick_pos);
        Some(id)
    }

    fn allocate_brick_with_data(&mut self, brick_pos: UVec3, data: &BrickData) -> Option<BrickId> {
        let key = self.l0_key(brick_pos);
        let id = self.pool.allocate_with_data(data)?;
        self.brick_map.insert(key, id);
        self.allocated_brick_positions.push(brick_pos);
        self.set_brick_coord(id, brick_pos);
        Some(id)
    }

    fn set_brick_coord(&mut self, id: BrickId, brick_pos: UVec3) {
        if let Some(slot) = self.brick_coords_by_id.get_mut(id as usize) {
            *slot = Some(brick_pos);
        }
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
        let occupancy_changed = previous.is_air() != cell.is_air();
        let touched_boundaries = if occupancy_changed {
            UcvhPageBoundaryMask::from_local_voxel(lp)
        } else {
            UcvhPageBoundaryMask::NONE
        };
        self.record_content_change(
            id,
            bp,
            generation,
            occupancy_changed,
            true,
            touched_boundaries,
        );
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
        let (occupancy_changed, material_changed, touched_boundaries) = if new_with_data {
            classify_brick_replacement(None, None, data)
        } else {
            classify_brick_replacement(
                Some(self.pool.occupancy(id)),
                Some(&self.pool.material_pool()[base..base + BRICK_VOLUME]),
                data,
            )
        };
        if !new_with_data {
            self.pool.write_brick(id, data);
        }
        self.mark_brick_dirty(id);
        let generation = self.next_content_generation();
        self.record_content_change(
            id,
            brick_pos,
            generation,
            occupancy_changed,
            material_changed,
            touched_boundaries,
        );
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
            let (occupancy_changed, material_changed, touched_boundaries) = if new_with_data {
                classify_brick_replacement(None, None, data)
            } else {
                classify_brick_replacement(
                    Some(self.pool.occupancy(id)),
                    Some(&self.pool.material_pool()[base..base + BRICK_VOLUME]),
                    data,
                )
            };
            if !new_with_data {
                self.pool.write_brick(id, data);
            }
            self.mark_brick_dirty(id);
            let generation = self.next_content_generation();
            self.record_static_content_change(
                id,
                generation,
                occupancy_changed,
                material_changed,
                touched_boundaries,
            );
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
            let (occupancy_changed, material_changed, touched_boundaries) = if new_with_data {
                classify_brick_replacement(None, None, data)
            } else {
                classify_brick_replacement(
                    Some(self.pool.occupancy(id)),
                    Some(&self.pool.material_pool()[base..base + BRICK_VOLUME]),
                    data,
                )
            };
            if !new_with_data {
                self.pool.write_brick(id, data);
            }
            self.mark_brick_dirty(id);
            let generation = self.next_content_generation();
            self.record_content_change(
                id,
                brick_pos,
                generation,
                occupancy_changed,
                material_changed,
                touched_boundaries,
            );
            self.hierarchy_dirty = true;
        }
        failed
    }

    fn try_write_initial_static_bricks_bulk<'a, I>(&mut self, bricks: I) -> u32
    where
        I: IntoIterator<Item = (UVec3, &'a BrickData)>,
    {
        let mut failed = 0;
        let mut entries = HashMap::new();
        for (brick_pos, data) in bricks {
            if !self.contains_brick_pos(brick_pos) {
                failed += 1;
                continue;
            }
            if brick_data_is_air(data) {
                continue;
            }
            entries.insert(brick_pos, data);
        }
        if entries.is_empty() {
            return failed;
        }

        let mut entries = entries.into_iter().collect::<Vec<_>>();
        entries.sort_by_key(|(brick_pos, _)| (brick_pos.z, brick_pos.y, brick_pos.x));

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
        for ((brick_pos, data), id) in entries.into_iter().zip(ids) {
            self.brick_map.insert(self.l0_key(brick_pos), id);
            self.allocated_brick_positions.push(brick_pos);
            self.set_brick_coord(id, brick_pos);
            self.mark_brick_dirty(id);
            let generation = self.next_content_generation();
            let (occupancy_changed, material_changed, touched_boundaries) =
                classify_brick_replacement(None, None, data);
            self.record_static_content_change(
                id,
                generation,
                occupancy_changed,
                material_changed,
                touched_boundaries,
            );
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

    /// Rebuilds only hierarchy parent nodes affected by a render change batch.
    pub fn update_hierarchy_for_render_change_batch(
        &mut self,
        batch: &UcvhRenderChangeBatch,
    ) -> CascadedOccupancyChanges {
        for brick in &batch.bricks {
            let has_solid = !self.pool.occupancy(brick.brick_id).is_empty();
            self.hierarchy
                .set_l0(brick.brick_coord, brick.brick_id, has_solid);
        }
        let changes = self
            .hierarchy
            .update_from_l0_positions(batch.bricks.iter().map(|brick| brick.brick_coord));
        self.hierarchy_dirty = false;
        changes
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

    /// Acknowledges exactly the invalidation snapshot that was uploaded.
    ///
    /// A later edit to the same region replaces its generation in place. In
    /// that case this returns false and retains the newer region for retry.
    pub fn ack_invalidation_regions(&mut self, snapshot: &[UcvhInvalidationRegion]) -> bool {
        if snapshot.len() > self.invalidation_regions.len()
            || self.invalidation_regions[..snapshot.len()] != *snapshot
        {
            return false;
        }
        self.invalidation_regions.drain(..snapshot.len());
        self.invalidation_region_indices.clear();
        for (index, region) in self.invalidation_regions.iter().enumerate() {
            self.invalidation_region_indices.insert(
                invalidation_region_key(region.brick_min, region.brick_max_exclusive),
                index,
            );
        }
        true
    }

    /// Returns the pending render changes without consuming them.
    pub fn snapshot_render_change_batch(&mut self) -> UcvhRenderChangeBatch {
        if self.pending_render_change_ids.is_empty() {
            return UcvhRenderChangeBatch {
                id: 0,
                bricks: Vec::new(),
                invalidated_pages: Vec::new(),
                invalidated_render_cells: Vec::new(),
            };
        }
        if let Some(snapshot) = &self.pending_render_change_snapshot {
            return snapshot.clone();
        }

        let bricks = self
            .pending_render_change_ids
            .iter()
            .filter_map(|&brick_id| {
                let generation = *self
                    .pending_render_change_generations
                    .get(brick_id as usize)?;
                let revision = *self
                    .pending_render_change_revisions
                    .get(brick_id as usize)?;
                let occupancy_changed = *self
                    .pending_render_occupancy_changes
                    .get(brick_id as usize)?;
                let material_changed = *self
                    .pending_render_material_changes
                    .get(brick_id as usize)?;
                let touched_boundaries = *self
                    .pending_render_touched_boundaries
                    .get(brick_id as usize)?;
                let brick_coord = self.brick_coord_for_id(brick_id)?;
                (generation != UCVH_NO_BRICK_GENERATION && revision != 0).then_some(
                    UcvhChangedBrick {
                        brick_id,
                        brick_coord,
                        generation,
                        revision,
                        occupancy_changed,
                        material_changed,
                        touched_boundaries,
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut invalidated_pages = Vec::with_capacity(bricks.len().saturating_mul(7));
        let mut invalidated_render_cells = Vec::with_capacity(bricks.len().saturating_mul(7));
        for brick in &bricks {
            if brick.occupancy_changed {
                invalidated_pages.extend(pages_affected_by_brick_boundaries(
                    brick.brick_coord,
                    brick.touched_boundaries,
                    self.config.brick_grid_size,
                ));
            }
            invalidated_render_cells.extend(render_cells_affected_by_brick(
                brick.brick_coord,
                self.config.brick_grid_size,
            ));
        }
        invalidated_pages.sort_by_key(|page| (page.z, page.y, page.x));
        invalidated_pages.dedup();
        invalidated_render_cells.sort_by_key(|cell| (cell.z, cell.y, cell.x));
        invalidated_render_cells.dedup();
        let snapshot = UcvhRenderChangeBatch {
            id: self.next_render_change_batch_id,
            bricks,
            invalidated_pages,
            invalidated_render_cells,
        };
        self.next_render_change_batch_id = self
            .next_render_change_batch_id
            .checked_add(1)
            .expect("UCVH render change batch ID space exhausted");
        self.pending_render_change_snapshot = Some(snapshot.clone());
        snapshot
    }

    /// Acknowledges exactly one previously returned snapshot.
    pub fn ack_render_change_batch(&mut self, batch_id: u64) -> bool {
        let Some(snapshot) = self.pending_render_change_snapshot.take() else {
            return false;
        };
        if snapshot.id != batch_id {
            self.pending_render_change_snapshot = Some(snapshot);
            return false;
        }
        for changed in snapshot.bricks {
            let (
                Some(current_generation),
                Some(current_revision),
                Some(occupancy_changed),
                Some(material_changed),
                Some(touched_boundaries),
            ) = (
                self.pending_render_change_generations
                    .get_mut(changed.brick_id as usize),
                self.pending_render_change_revisions
                    .get_mut(changed.brick_id as usize),
                self.pending_render_occupancy_changes
                    .get_mut(changed.brick_id as usize),
                self.pending_render_material_changes
                    .get_mut(changed.brick_id as usize),
                self.pending_render_touched_boundaries
                    .get_mut(changed.brick_id as usize),
            )
            else {
                continue;
            };
            if *current_generation == changed.generation && *current_revision == changed.revision {
                *current_generation = UCVH_NO_BRICK_GENERATION;
                *current_revision = 0;
                *occupancy_changed = false;
                *material_changed = false;
                *touched_boundaries = UcvhPageBoundaryMask::NONE;
                self.pending_render_change_ids.remove(&changed.brick_id);
            }
        }
        true
    }

    pub fn brick_id_at(&self, brick_pos: UVec3) -> Option<BrickId> {
        if !self.contains_brick_pos(brick_pos) {
            return None;
        }
        self.brick_map.get(&self.l0_key(brick_pos)).copied()
    }

    pub fn brick_coord_for_id(&self, brick_id: BrickId) -> Option<UVec3> {
        self.brick_coords_by_id
            .get(brick_id as usize)
            .copied()
            .flatten()
    }

    pub fn brick_generation(&self, brick_id: BrickId) -> Option<u32> {
        self.brick_generations.get(brick_id as usize).copied()
    }

    pub fn brick_generations(&self) -> &[u32] {
        &self.brick_generations
    }

    pub fn brick_topology_revision(&self, brick_id: BrickId) -> Option<u64> {
        self.brick_topology_revisions
            .get(brick_id as usize)
            .copied()
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

    /// Acknowledges the immutable prefix successfully copied to the motion
    /// buffer. Events appended after the snapshot remain pending.
    pub fn ack_motion_events(&mut self, snapshot: &[UcvhMotionEvent]) -> bool {
        if snapshot.len() > self.motion_events.len()
            || self.motion_events[..snapshot.len()] != *snapshot
        {
            return false;
        }
        self.motion_events.drain(..snapshot.len());
        true
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

const RENDER_CELL_BRICK_EDGE: u32 = 2;

fn pages_affected_by_brick_boundaries(
    brick_coord: UVec3,
    touched_boundaries: UcvhPageBoundaryMask,
    brick_grid_size: UVec3,
) -> Vec<UVec3> {
    let mut pages = Vec::with_capacity(7);
    pages.push(brick_coord);
    for (boundary, offset) in [
        (UcvhPageBoundaryMask::NEG_X, glam::IVec3::NEG_X),
        (UcvhPageBoundaryMask::POS_X, glam::IVec3::X),
        (UcvhPageBoundaryMask::NEG_Y, glam::IVec3::NEG_Y),
        (UcvhPageBoundaryMask::POS_Y, glam::IVec3::Y),
        (UcvhPageBoundaryMask::NEG_Z, glam::IVec3::NEG_Z),
        (UcvhPageBoundaryMask::POS_Z, glam::IVec3::Z),
    ] {
        if !touched_boundaries.contains(boundary) {
            continue;
        }
        let candidate = brick_coord.as_ivec3() + offset;
        if candidate.x < 0
            || candidate.y < 0
            || candidate.z < 0
            || candidate.x >= brick_grid_size.x as i32
            || candidate.y >= brick_grid_size.y as i32
            || candidate.z >= brick_grid_size.z as i32
        {
            continue;
        }
        pages.push(candidate.as_uvec3());
    }
    pages
}

fn classify_brick_replacement(
    old_occupancy: Option<&BrickOccupancy>,
    old_materials: Option<&[VoxelCell]>,
    new_data: &BrickData,
) -> (bool, bool, UcvhPageBoundaryMask) {
    let occupancy_changed = old_occupancy
        .map(|old| old != &new_data.occupancy)
        .unwrap_or_else(|| !new_data.occupancy.is_empty());
    let material_changed = old_materials
        .map(|old| old != &new_data.materials[..])
        .unwrap_or_else(|| {
            new_data
                .materials
                .iter()
                .any(|cell| *cell != VoxelCell::AIR)
        });
    if !occupancy_changed {
        return (false, material_changed, UcvhPageBoundaryMask::NONE);
    }

    let mut touched_boundaries = UcvhPageBoundaryMask::NONE;
    for z in 0..BRICK_EDGE {
        for y in 0..BRICK_EDGE {
            for x in 0..BRICK_EDGE {
                let old_solid = old_occupancy.is_some_and(|old| old.get(x, y, z));
                if old_solid != new_data.occupancy.get(x, y, z) {
                    touched_boundaries |=
                        UcvhPageBoundaryMask::from_local_voxel(UVec3::new(x, y, z));
                }
            }
        }
    }
    (occupancy_changed, material_changed, touched_boundaries)
}

fn render_cells_affected_by_brick(brick_coord: UVec3, brick_grid_size: UVec3) -> Vec<UVec3> {
    let render_cell_grid_size = div_ceil_uvec3(brick_grid_size, RENDER_CELL_BRICK_EDGE);
    let owner = brick_coord / RENDER_CELL_BRICK_EDGE;
    let mut cells = Vec::with_capacity(7);
    for offset in [
        glam::IVec3::ZERO,
        glam::IVec3::NEG_X,
        glam::IVec3::X,
        glam::IVec3::NEG_Y,
        glam::IVec3::Y,
        glam::IVec3::NEG_Z,
        glam::IVec3::Z,
    ] {
        let candidate = owner.as_ivec3() + offset;
        if candidate.x < 0
            || candidate.y < 0
            || candidate.z < 0
            || candidate.x >= render_cell_grid_size.x as i32
            || candidate.y >= render_cell_grid_size.y as i32
            || candidate.z >= render_cell_grid_size.z as i32
        {
            continue;
        }
        cells.push(candidate.as_uvec3());
    }
    cells
}

fn brick_data_is_air(data: &BrickData) -> bool {
    data.occupancy.is_empty() && data.materials.iter().all(|cell| *cell == VoxelCell::AIR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_change_batch_classifies_interior_material_replacement() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(24)));
        let pos = UVec3::new(9, 9, 9);
        assert!(u.set_voxel(pos, VoxelCell::new(1, 0, [0; 3])));
        let initial = u.snapshot_render_change_batch();
        assert!(u.ack_render_change_batch(initial.id));

        assert!(u.set_voxel(pos, VoxelCell::new(2, 1, [3, 4, 5])));
        let batch = u.snapshot_render_change_batch();

        assert_eq!(batch.bricks.len(), 1);
        assert!(!batch.bricks[0].occupancy_changed);
        assert!(batch.bricks[0].material_changed);
        assert_eq!(
            batch.bricks[0].touched_boundaries,
            UcvhPageBoundaryMask::NONE
        );
        assert!(batch.invalidated_pages.is_empty());
    }

    #[test]
    fn render_change_batch_classifies_air_solid_transition_and_owner_page() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(24)));
        assert!(u.set_voxel(UVec3::new(9, 9, 9), VoxelCell::new(1, 0, [0; 3])));

        let batch = u.snapshot_render_change_batch();

        assert!(batch.bricks[0].occupancy_changed);
        assert!(batch.bricks[0].material_changed);
        assert_eq!(batch.invalidated_pages, vec![UVec3::ONE]);
    }

    #[test]
    fn render_change_batch_classifies_negative_x_boundary_only() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(24)));
        assert!(u.set_voxel(UVec3::new(8, 9, 9), VoxelCell::new(1, 0, [0; 3])));

        let batch = u.snapshot_render_change_batch();

        assert_eq!(
            batch.bricks[0].touched_boundaries,
            UcvhPageBoundaryMask::NEG_X
        );
        assert_eq!(
            batch.invalidated_pages,
            vec![UVec3::new(0, 1, 1), UVec3::ONE]
        );
    }

    #[test]
    fn render_change_batch_classifies_positive_xyz_boundaries() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(24)));
        assert!(u.set_voxel(UVec3::new(15, 15, 15), VoxelCell::new(1, 0, [0; 3])));

        let batch = u.snapshot_render_change_batch();

        assert_eq!(
            batch.bricks[0].touched_boundaries,
            UcvhPageBoundaryMask::POS_X | UcvhPageBoundaryMask::POS_Y | UcvhPageBoundaryMask::POS_Z
        );
    }

    #[test]
    fn render_change_batch_classification_preserves_newer_merged_flags() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(24)));
        let pos = UVec3::new(8, 9, 9);
        assert!(u.set_voxel(pos, VoxelCell::new(1, 0, [0; 3])));
        let old = u.snapshot_render_change_batch();

        assert!(u.set_voxel(pos, VoxelCell::new(2, 0, [0; 3])));
        assert!(u.set_voxel(UVec3::new(15, 9, 9), VoxelCell::new(3, 0, [0; 3])));
        assert!(u.ack_render_change_batch(old.id));
        let current = u.snapshot_render_change_batch();

        assert_ne!(current.bricks[0].revision, old.bricks[0].revision);
        assert!(current.bricks[0].occupancy_changed);
        assert!(current.bricks[0].material_changed);
        assert!(
            current.bricks[0]
                .touched_boundaries
                .contains(UcvhPageBoundaryMask::NEG_X | UcvhPageBoundaryMask::POS_X)
        );
    }

    #[test]
    fn render_change_batch_classifies_full_brick_boundary_delta() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(24)));
        let mut initial = BrickData::new();
        initial.set_voxel(1, 1, 1, VoxelCell::new(1, 0, [0; 3]));
        assert!(u.write_brick(UVec3::ONE, &initial));
        let initial_batch = u.snapshot_render_change_batch();
        assert!(u.ack_render_change_batch(initial_batch.id));

        let mut replacement = BrickData::new();
        replacement.set_voxel(1, 1, 1, VoxelCell::new(2, 0, [0; 3]));
        replacement.set_voxel(7, 2, 3, VoxelCell::new(3, 0, [0; 3]));
        assert!(u.write_brick(UVec3::ONE, &replacement));
        let batch = u.snapshot_render_change_batch();

        assert!(batch.bricks[0].occupancy_changed);
        assert!(batch.bricks[0].material_changed);
        assert_eq!(
            batch.bricks[0].touched_boundaries,
            UcvhPageBoundaryMask::POS_X
        );
        assert_eq!(
            batch.invalidated_pages,
            vec![UVec3::ONE, UVec3::new(2, 1, 1)]
        );
    }

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
    fn render_change_batch_is_retained_until_the_matching_snapshot_is_acknowledged() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        assert!(u.set_voxel(UVec3::new(1, 2, 3), VoxelCell::new(1, 0, [0; 3])));

        let first = u.snapshot_render_change_batch();
        assert_eq!(first.bricks.len(), 1);
        assert_eq!(u.snapshot_render_change_batch().id, first.id);
        assert!(u.ack_render_change_batch(first.id));
        assert!(u.snapshot_render_change_batch().is_empty());
    }

    #[test]
    fn acknowledging_an_old_render_change_snapshot_keeps_a_newer_edit_to_the_same_brick() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        assert!(u.set_voxel(UVec3::new(1, 2, 3), VoxelCell::new(1, 0, [0; 3])));
        let old = u.snapshot_render_change_batch();

        assert!(u.set_voxel(UVec3::new(2, 2, 3), VoxelCell::new(2, 0, [0; 3])));

        assert!(u.ack_render_change_batch(old.id));
        let current = u.snapshot_render_change_batch();
        assert_eq!(current.bricks.len(), 1);
        assert_ne!(current.bricks[0].generation, old.bricks[0].generation);
    }

    #[test]
    fn render_change_batch_invalidates_its_cell_and_face_neighbors() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(48, 48, 48)));
        assert!(u.set_voxel(UVec3::new(17, 17, 17), VoxelCell::new(1, 0, [0; 3])));

        let batch = u.snapshot_render_change_batch();

        assert_eq!(batch.bricks[0].brick_coord, UVec3::new(2, 2, 2));
        assert_eq!(
            batch.invalidated_render_cells,
            vec![
                UVec3::new(1, 1, 0),
                UVec3::new(1, 0, 1),
                UVec3::new(0, 1, 1),
                UVec3::new(1, 1, 1),
                UVec3::new(2, 1, 1),
                UVec3::new(1, 2, 1),
                UVec3::new(1, 1, 2),
            ]
        );
    }

    #[test]
    fn render_change_batch_invalidates_only_owner_for_interior_occupancy_change() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::new(32, 32, 32)));
        assert!(u.set_voxel(UVec3::new(9, 9, 9), VoxelCell::new(1, 0, [0; 3])));

        let batch = u.snapshot_render_change_batch();

        assert_eq!(batch.invalidated_pages, vec![UVec3::new(1, 1, 1)]);
    }

    #[test]
    fn acknowledging_snapshotted_motion_events_keeps_events_appended_after_snapshot() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        let first = UcvhMotionEvent {
            region_min: UVec3::new(0, 0, 0),
            region_max_exclusive: UVec3::new(8, 8, 8),
            world_delta_current_from_previous: IVec3::X,
            generation: 1,
        };
        let second = UcvhMotionEvent {
            region_min: UVec3::new(8, 0, 0),
            region_max_exclusive: UVec3::new(16, 8, 8),
            world_delta_current_from_previous: IVec3::Y,
            generation: 2,
        };
        assert!(u.push_motion_event(first));
        let snapshot = u.motion_events().to_vec();
        assert!(u.push_motion_event(second));

        assert!(u.ack_motion_events(&snapshot));

        assert_eq!(u.motion_events(), &[second]);
    }

    #[test]
    fn acknowledging_stale_invalidation_snapshot_preserves_rewritten_region() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        assert!(u.set_voxel(UVec3::new(1, 1, 1), VoxelCell::new(1, 0, [0; 3])));
        let snapshot = u.invalidation_regions().to_vec();
        assert!(u.set_voxel(UVec3::new(2, 1, 1), VoxelCell::new(2, 0, [0; 3])));

        assert!(!u.ack_invalidation_regions(&snapshot));
        assert_eq!(u.invalidation_regions().len(), 1);
        assert_ne!(
            u.invalidation_regions()[0].generation,
            snapshot[0].generation
        );
    }

    #[test]
    fn render_change_ack_uses_non_wrapping_revision_not_only_content_generation() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        assert!(u.set_voxel(UVec3::new(1, 2, 3), VoxelCell::new(1, 0, [0; 3])));
        let old = u.snapshot_render_change_batch();

        let brick_id = old.bricks[0].brick_id;
        u.content_generation = old.bricks[0].generation;
        assert!(u.set_voxel(UVec3::new(2, 2, 3), VoxelCell::new(2, 0, [0; 3])));
        assert!(u.ack_render_change_batch(old.id));
        let pending = u.snapshot_render_change_batch();
        assert_eq!(pending.bricks[0].brick_id, brick_id);
        assert_eq!(pending.bricks[0].generation, old.bricks[0].generation);
        assert_ne!(pending.bricks[0].revision, old.bricks[0].revision);
        assert_ne!(pending.id, old.id);
    }

    #[test]
    fn initial_static_bulk_overwrites_duplicate_coordinates_without_leaking_brick_slots() {
        let mut u = Ucvh::new(UcvhConfig::with_brick_capacity(UVec3::splat(16), 4));
        let mut first = BrickData::new();
        first.set_voxel(0, 0, 0, VoxelCell::new(1, 0, [0; 3]));
        let mut replacement = BrickData::new();
        replacement.set_voxel(1, 0, 0, VoxelCell::new(2, 0, [0; 3]));
        let mut second = BrickData::new();
        second.set_voxel(2, 0, 0, VoxelCell::new(3, 0, [0; 3]));

        assert_eq!(
            u.write_static_bricks_bulk([
                (UVec3::ZERO, &first),
                (UVec3::ZERO, &replacement),
                (UVec3::new(1, 0, 0), &second),
            ]),
            0
        );

        assert_eq!(u.allocated_brick_count(), 2);
        assert_eq!(u.get_voxel(UVec3::new(1, 0, 0)).material, 2);
        assert_eq!(u.get_voxel(UVec3::new(10, 0, 0)).material, 3);
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
    fn render_change_batch_updates_only_its_hierarchy_ancestor_chain() {
        let mut u = Ucvh::new(UcvhConfig::new(UVec3::splat(32)));
        assert!(u.set_voxel(UVec3::new(9, 1, 1), VoxelCell::new(1, 0, [0; 3])));
        let batch = u.snapshot_render_change_batch();

        let changes = u.update_hierarchy_for_render_change_batch(&batch);

        assert_eq!(changes.l0, vec![UVec3::new(1, 0, 0)]);
        assert_eq!(changes.levels[0], vec![UVec3::ZERO]);
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
