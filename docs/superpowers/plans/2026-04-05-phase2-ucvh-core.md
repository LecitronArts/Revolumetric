# Phase 2: UCVH Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the CPU-side brick pool with Morton-order storage, 5-level cascaded occupancy hierarchy, GPU upload pipeline, and a procedural sphere demo scene — producing SSBO data ready for Phase 3's ray tracer.

**Architecture:** Voxel data lives in a two-tier brick pool (occupancy bitmask + material data), indexed by Morton curve for cache locality. A 5-level occupancy hierarchy enables hierarchical DDA skip. CPU generates the scene and uploads to device-local SSBOs via staging buffers. The test pattern pass remains as visual output; Phase 3 replaces it with ray tracing.

**Tech Stack:** Rust, ash (Vulkan), gpu-allocator, bytemuck, glam

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/voxel/morton.rs` | Morton (Z-curve) encode/decode for 8^3 bricks |
| `src/voxel/brick.rs` | `BrickOccupancy`, `VoxelCell`, `BrickData` data types |
| `src/voxel/brick_pool.rs` | `BrickPool` with free-list allocation |
| `src/voxel/occupancy.rs` | `CascadedOccupancy` — 5-level hierarchy with CPU rebuild |
| `src/voxel/ucvh.rs` | `Ucvh` facade — unified API for set/get voxel + dirty tracking |
| `src/voxel/generator.rs` | `VoxelGenerator` trait + `SphereGenerator` |
| `src/voxel/gpu_upload.rs` | `UcvhGpuResources` — GPU SSBOs + staging + upload logic |

### Modified Files

| File | Change |
|------|--------|
| `src/voxel/mod.rs` | Add new modules, remove old stubs |
| `src/app.rs` | Create Ucvh, generate scene, upload to GPU |
| `Cargo.toml` | No new deps needed (bytemuck already present) |

### Removed (stubs replaced)

| File | Replaced By |
|------|-------------|
| `src/voxel/grid.rs` | `brick.rs` + `brick_pool.rs` |
| `src/voxel/chunk.rs` | `ucvh.rs` |
| `src/voxel/scene_baker.rs` | `generator.rs` |
| `src/voxel/radiance_layout.rs` | `occupancy.rs` (hierarchy dims) |

---

## Task 1: Morton Indexing Module

**Files:**
- Create: `src/voxel/morton.rs`
- Modify: `src/voxel/mod.rs`

- [ ] **Step 1: Write failing tests for Morton encode/decode**

```rust
// src/voxel/morton.rs

/// Encodes a 3D position within an 8^3 brick to a Morton (Z-curve) index.
/// Each coordinate must be in [0, 7]. Returns index in [0, 511].
pub fn encode(x: u32, y: u32, z: u32) -> u32 {
    todo!()
}

/// Decodes a Morton index back to (x, y, z) within an 8^3 brick.
pub fn decode(index: u32) -> (u32, u32, u32) {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_is_zero() {
        assert_eq!(encode(0, 0, 0), 0);
    }

    #[test]
    fn unit_axes() {
        assert_eq!(encode(1, 0, 0), 0b001);
        assert_eq!(encode(0, 1, 0), 0b010);
        assert_eq!(encode(0, 0, 1), 0b100);
    }

    #[test]
    fn max_corner() {
        assert_eq!(encode(7, 7, 7), 511);
    }

    #[test]
    fn roundtrip_all_512() {
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let m = encode(x, y, z);
                    assert_eq!(decode(m), (x, y, z));
                }
            }
        }
    }

    #[test]
    fn unique_indices() {
        let mut seen = std::collections::HashSet::new();
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    assert!(seen.insert(encode(x, y, z)));
                }
            }
        }
        assert_eq!(seen.len(), 512);
    }
}
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cargo test --lib voxel::morton -- --nocapture`
Expected: FAIL (todo panics)

- [ ] **Step 3: Implement encode and decode**

```rust
pub fn encode(x: u32, y: u32, z: u32) -> u32 {
    debug_assert!(x < 8 && y < 8 && z < 8, "coordinates must be in [0, 7]");
    let mut m = 0u32;
    for bit in 0..3u32 {
        m |= ((x >> bit) & 1) << (3 * bit);
        m |= ((y >> bit) & 1) << (3 * bit + 1);
        m |= ((z >> bit) & 1) << (3 * bit + 2);
    }
    m
}

pub fn decode(index: u32) -> (u32, u32, u32) {
    debug_assert!(index < 512, "index must be in [0, 511]");
    let (mut x, mut y, mut z) = (0u32, 0u32, 0u32);
    for bit in 0..3u32 {
        x |= ((index >> (3 * bit)) & 1) << bit;
        y |= ((index >> (3 * bit + 1)) & 1) << bit;
        z |= ((index >> (3 * bit + 2)) & 1) << bit;
    }
    (x, y, z)
}
```

- [ ] **Step 4: Run tests, verify all 5 pass**

Run: `cargo test --lib voxel::morton -- --nocapture`
Expected: 5 tests PASS

- [ ] **Step 5: Add module to mod.rs and commit**

Add `pub mod morton;` to `src/voxel/mod.rs`.

```bash
git add src/voxel/morton.rs src/voxel/mod.rs
git commit -m "feat(voxel): add Morton Z-curve encode/decode for 8^3 bricks"
```

---

## Task 2: Brick Data Types

**Files:**
- Create: `src/voxel/brick.rs`
- Modify: `src/voxel/mod.rs`

- [ ] **Step 1: Write brick.rs with types and failing tests**

```rust
// src/voxel/brick.rs
use bytemuck::{Pod, Zeroable};
use crate::voxel::morton;

pub const BRICK_EDGE: u32 = 8;
pub const BRICK_VOLUME: usize = 512; // 8^3

/// Occupancy bitmask for an 8^3 brick (hot-path, fits GPU L2).
/// Bits are indexed in Morton order matching material storage.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BrickOccupancy {
    pub bits: [u32; 16],  // 512 bits
    pub count: u32,       // number of solid voxels (0 = empty, skip instantly)
    pub _pad: [u32; 3],   // pad to 80 bytes (16-byte GPU alignment)
}

impl BrickOccupancy {
    pub fn set(&mut self, x: u32, y: u32, z: u32) {
        let m = morton::encode(x, y, z);
        let (word, bit) = (m as usize / 32, m % 32);
        if self.bits[word] & (1 << bit) == 0 {
            self.bits[word] |= 1 << bit;
            self.count += 1;
        }
    }

    pub fn clear(&mut self, x: u32, y: u32, z: u32) {
        let m = morton::encode(x, y, z);
        let (word, bit) = (m as usize / 32, m % 32);
        if self.bits[word] & (1 << bit) != 0 {
            self.bits[word] &= !(1 << bit);
            self.count -= 1;
        }
    }

    pub fn get(&self, x: u32, y: u32, z: u32) -> bool {
        let m = morton::encode(x, y, z);
        let (word, bit) = (m as usize / 32, m % 32);
        self.bits[word] & (1 << bit) != 0
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Single voxel material data (cold-path, read only on ray hit).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct VoxelCell {
    pub material: u16,
    pub flags: u16,
    pub emissive: [u16; 3],
    pub _pad: u16,
}

impl VoxelCell {
    pub const AIR: Self = Self { material: 0, flags: 0, emissive: [0; 3], _pad: 0 };
    pub fn is_air(&self) -> bool { self.material == 0 }
}

/// Complete brick data: occupancy + 512 materials in Morton order.
pub struct BrickData {
    pub occupancy: BrickOccupancy,
    pub materials: Box<[VoxelCell; BRICK_VOLUME]>,
}

impl BrickData {
    pub fn new() -> Self {
        Self {
            occupancy: BrickOccupancy::zeroed(),
            materials: Box::new([VoxelCell::AIR; BRICK_VOLUME]),
        }
    }

    pub fn set_voxel(&mut self, x: u32, y: u32, z: u32, cell: VoxelCell) {
        let m = morton::encode(x, y, z) as usize;
        self.materials[m] = cell;
        if cell.is_air() {
            self.occupancy.clear(x, y, z);
        } else {
            self.occupancy.set(x, y, z);
        }
    }

    pub fn get_voxel(&self, x: u32, y: u32, z: u32) -> VoxelCell {
        self.materials[morton::encode(x, y, z) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_assertions() {
        assert_eq!(std::mem::size_of::<BrickOccupancy>(), 80);
        assert_eq!(std::mem::size_of::<VoxelCell>(), 8);
    }

    #[test]
    fn empty_brick_occupancy() {
        let occ = BrickOccupancy::zeroed();
        assert!(occ.is_empty());
        assert_eq!(occ.count, 0);
    }

    #[test]
    fn set_get_clear() {
        let mut occ = BrickOccupancy::zeroed();
        occ.set(3, 4, 5);
        assert!(occ.get(3, 4, 5));
        assert!(!occ.get(0, 0, 0));
        assert_eq!(occ.count, 1);
        occ.clear(3, 4, 5);
        assert!(!occ.get(3, 4, 5));
        assert_eq!(occ.count, 0);
    }

    #[test]
    fn set_idempotent() {
        let mut occ = BrickOccupancy::zeroed();
        occ.set(1, 2, 3);
        occ.set(1, 2, 3);
        assert_eq!(occ.count, 1);
    }

    #[test]
    fn brick_data_set_and_read() {
        let mut b = BrickData::new();
        let cell = VoxelCell { material: 42, flags: 1, emissive: [0; 3], _pad: 0 };
        b.set_voxel(7, 7, 7, cell);
        assert!(b.occupancy.get(7, 7, 7));
        assert_eq!(b.get_voxel(7, 7, 7).material, 42);
        assert_eq!(b.occupancy.count, 1);
    }

    #[test]
    fn brick_data_air_clears() {
        let mut b = BrickData::new();
        b.set_voxel(0, 0, 0, VoxelCell { material: 1, ..VoxelCell::AIR });
        assert!(b.occupancy.get(0, 0, 0));
        b.set_voxel(0, 0, 0, VoxelCell::AIR);
        assert!(!b.occupancy.get(0, 0, 0));
        assert_eq!(b.occupancy.count, 0);
    }
}
```

- [ ] **Step 2: Run tests, verify all pass**

Run: `cargo test --lib voxel::brick -- --nocapture`
Expected: 6 tests PASS

- [ ] **Step 3: Add module to mod.rs and commit**

Add `pub mod brick;` to `src/voxel/mod.rs`.

```bash
git add src/voxel/brick.rs src/voxel/mod.rs
git commit -m "feat(voxel): add BrickOccupancy, VoxelCell, BrickData with Morton-order storage"
```

---

## Task 3: BrickPool with Free-List Allocation

**Files:**
- Create: `src/voxel/brick_pool.rs`
- Modify: `src/voxel/mod.rs`

- [ ] **Step 1: Write brick_pool.rs with tests**

```rust
// src/voxel/brick_pool.rs
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
        data.set_voxel(0, 0, 0, VoxelCell { material: 1, ..VoxelCell::AIR });
        pool.write_brick(id, &data);
        pool.free(id);

        // Re-allocate same slot — should be clean
        let id2 = pool.allocate().unwrap();
        assert_eq!(id2, id);
        assert!(pool.occupancy(id2).is_empty());
        assert_eq!(pool.get_material(id2, 0).material, 0);
    }
}
```

- [ ] **Step 2: Run tests, verify all pass**

Run: `cargo test --lib voxel::brick_pool -- --nocapture`
Expected: 5 tests PASS

- [ ] **Step 3: Add module and commit**

Add `pub mod brick_pool;` to `src/voxel/mod.rs`.

```bash
git add src/voxel/brick_pool.rs src/voxel/mod.rs
git commit -m "feat(voxel): add BrickPool with free-list allocation"
```

---

## Task 4: Cascaded Occupancy Hierarchy

**Files:**
- Create: `src/voxel/occupancy.rs` (replace existing `radiance_layout.rs`)
- Modify: `src/voxel/mod.rs`

- [ ] **Step 1: Write occupancy.rs with hierarchy types and rebuild logic**

```rust
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

pub const EMPTY_L0: NodeL0 = NodeL0 { brick_id: u32::MAX, flags: 0, _pad: 0 };

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
        assert_eq!(h.dims[1], UVec3::splat(8));  // L1
        assert_eq!(h.dims[2], UVec3::splat(4));  // L2
        assert_eq!(h.dims[3], UVec3::splat(2));  // L3
        assert_eq!(h.dims[4], UVec3::splat(1));  // L4
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
```

- [ ] **Step 2: Run tests, verify all pass**

Run: `cargo test --lib voxel::occupancy -- --nocapture`
Expected: 5 tests PASS

- [ ] **Step 3: Update mod.rs — add occupancy, remove radiance_layout**

Replace `pub mod radiance_layout;` with `pub mod occupancy;` in `src/voxel/mod.rs`.
Delete `src/voxel/radiance_layout.rs`.

```bash
git add src/voxel/occupancy.rs src/voxel/mod.rs
git rm src/voxel/radiance_layout.rs
git commit -m "feat(voxel): add 5-level CascadedOccupancy hierarchy with CPU rebuild"
```

---

## Task 5: Ucvh Facade

**Files:**
- Create: `src/voxel/ucvh.rs`
- Modify: `src/voxel/mod.rs`
- Remove: `src/voxel/grid.rs`, `src/voxel/chunk.rs`, `src/voxel/scene_baker.rs`

The `Ucvh` struct is the unified entry point for all voxel operations. It owns the BrickPool, CascadedOccupancy, and a sparse brick map (L0 grid index -> BrickId).

- [ ] **Step 1: Write ucvh.rs**

```rust
// src/voxel/ucvh.rs
use glam::UVec3;
use crate::voxel::brick::{BrickData, BrickOccupancy, VoxelCell, BRICK_EDGE, BRICK_VOLUME};
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
        let cell = VoxelCell { material: 1, ..VoxelCell::AIR };
        u.set_voxel(UVec3::new(0, 0, 0), cell);
        u.set_voxel(UVec3::new(1, 0, 0), cell); // same brick
        let dirty = u.take_dirty_bricks();
        assert_eq!(dirty.len(), 1); // one brick, not two
    }

    #[test]
    fn hierarchy_rebuild_propagates() {
        let mut u = test_ucvh();
        let cell = VoxelCell { material: 1, ..VoxelCell::AIR };
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
                    data.set_voxel(x, y, z, VoxelCell { material: 1, ..VoxelCell::AIR });
                }
            }
        }
        assert!(u.write_brick(UVec3::ZERO, &data));
        assert_eq!(u.pool.occupancy(0).count, 512);
    }
}
```

- [ ] **Step 2: Run tests, verify all pass**

Run: `cargo test --lib voxel::ucvh -- --nocapture`
Expected: 5 tests PASS

- [ ] **Step 3: Update mod.rs — add ucvh, remove stubs**

Update `src/voxel/mod.rs`:
```rust
pub mod brick;
pub mod brick_pool;
pub mod generator;  // will be added in Task 6
pub mod material;
pub mod morton;
pub mod occupancy;
pub mod ucvh;
pub mod gpu_upload;  // will be added in Task 7
```

Delete stubs: `src/voxel/grid.rs`, `src/voxel/chunk.rs`, `src/voxel/scene_baker.rs`.

Note: temporarily comment out `pub mod generator;` and `pub mod gpu_upload;` since those files don't exist yet. Uncomment when their tasks are done.

```bash
git add src/voxel/ucvh.rs src/voxel/mod.rs
git rm src/voxel/grid.rs src/voxel/chunk.rs src/voxel/scene_baker.rs
git commit -m "feat(voxel): add Ucvh facade with set/get voxel and dirty tracking"
```

---

## Task 6: Procedural Generator

**Files:**
- Create: `src/voxel/generator.rs`
- Modify: `src/voxel/mod.rs`

- [ ] **Step 1: Write generator.rs with SphereGenerator**

```rust
// src/voxel/generator.rs
use glam::{UVec3, Vec3};
use crate::voxel::brick::{BrickData, VoxelCell, BRICK_EDGE};
use crate::voxel::ucvh::{Ucvh, UcvhConfig};

pub trait VoxelGenerator {
    /// Generate brick data at the given brick grid coordinate.
    /// Returns None if the brick would be entirely empty.
    fn generate_brick(&self, brick_pos: UVec3, config: &UcvhConfig) -> Option<BrickData>;
}

pub struct SphereGenerator {
    pub center: Vec3,
    pub radius: f32,
    pub material: u16,
}

impl VoxelGenerator for SphereGenerator {
    fn generate_brick(&self, brick_pos: UVec3, _config: &UcvhConfig) -> Option<BrickData> {
        let base = brick_pos * BRICK_EDGE;
        let mut data = BrickData::new();
        let mut any_solid = false;

        for lz in 0..BRICK_EDGE {
            for ly in 0..BRICK_EDGE {
                for lx in 0..BRICK_EDGE {
                    let world = Vec3::new(
                        (base.x + lx) as f32 + 0.5,
                        (base.y + ly) as f32 + 0.5,
                        (base.z + lz) as f32 + 0.5,
                    );
                    if world.distance(self.center) <= self.radius {
                        data.set_voxel(lx, ly, lz, VoxelCell {
                            material: self.material,
                            flags: 1, // solid
                            emissive: [0; 3],
                            _pad: 0,
                        });
                        any_solid = true;
                    }
                }
            }
        }

        if any_solid { Some(data) } else { None }
    }
}

/// Generate a demo scene: solid sphere in center of world.
pub fn generate_demo_scene(ucvh: &mut Ucvh) -> u32 {
    let world = ucvh.config.world_size.as_vec3();
    let gen = SphereGenerator {
        center: world * 0.5,
        radius: world.x.min(world.y).min(world.z) * 0.35,
        material: 1,
    };

    let bgs = ucvh.config.brick_grid_size;
    let mut count = 0u32;
    for bz in 0..bgs.z {
        for by in 0..bgs.y {
            for bx in 0..bgs.x {
                let bp = UVec3::new(bx, by, bz);
                if let Some(data) = gen.generate_brick(bp, &ucvh.config) {
                    if ucvh.write_brick(bp, &data) {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_generates_bricks() {
        let config = UcvhConfig::new(UVec3::splat(64)); // 8^3 brick grid
        let gen = SphereGenerator {
            center: Vec3::splat(32.0),
            radius: 20.0,
            material: 1,
        };
        // Center brick at (4,4,4) in brick coords should be fully inside
        let data = gen.generate_brick(UVec3::splat(4), &config);
        assert!(data.is_some());
        let data = data.unwrap();
        assert!(data.occupancy.count > 0);
    }

    #[test]
    fn sphere_empty_outside() {
        let config = UcvhConfig::new(UVec3::splat(64));
        let gen = SphereGenerator {
            center: Vec3::splat(32.0),
            radius: 10.0,
            material: 1,
        };
        // Corner brick at (0,0,0) should be empty (far from sphere center)
        let data = gen.generate_brick(UVec3::ZERO, &config);
        assert!(data.is_none());
    }

    #[test]
    fn demo_scene_populates_ucvh() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));
        let count = generate_demo_scene(&mut ucvh);
        assert!(count > 0, "should have allocated some bricks");
        ucvh.rebuild_hierarchy();
        // Root should be non-empty
        assert_ne!(ucvh.hierarchy.levels[3][0].child_mask, 0);
    }
}
```

- [ ] **Step 2: Run tests, verify all pass**

Run: `cargo test --lib voxel::generator -- --nocapture`
Expected: 3 tests PASS

- [ ] **Step 3: Uncomment module in mod.rs and commit**

Uncomment `pub mod generator;` in `src/voxel/mod.rs`.

```bash
git add src/voxel/generator.rs src/voxel/mod.rs
git commit -m "feat(voxel): add VoxelGenerator trait and sphere demo scene generator"
```

---

## Task 7: GPU Upload Resources

**Files:**
- Create: `src/voxel/gpu_upload.rs`
- Modify: `src/voxel/mod.rs`

This creates the GPU SSBO buffers and staging buffers for UCVH data, plus the upload logic that copies dirty bricks from CPU to GPU via staging + vkCmdCopyBuffer.

**GPU buffer layout** (matches future shader bindings):
- SSBO binding 0: `UcvhGpuConfig` uniform (world_size, brick_grid_size, etc.)
- SSBO binding 1: `BrickOccupancy[]` — occupancy pool
- SSBO binding 2: `VoxelCell[]` — material pool (flat, 512 per brick)
- SSBO binding 3: `NodeL0[]` — hierarchy level 0
- SSBO binding 4: `NodeLN[]` — hierarchy level 1
- SSBO binding 5-7: `NodeLN[]` — hierarchy levels 2-4

- [ ] **Step 1: Write gpu_upload.rs**

```rust
// src/voxel/gpu_upload.rs
use anyhow::{Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable, bytes_of, cast_slice};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;
use crate::render::buffer::GpuBuffer;
use crate::voxel::brick::{BrickOccupancy, VoxelCell, BRICK_VOLUME};
use crate::voxel::occupancy::{NodeL0, NodeLN};
use crate::voxel::ucvh::Ucvh;

/// GPU-side config matching the shader UBO.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UcvhGpuConfig {
    pub world_size: [u32; 4],       // xyz + pad
    pub brick_grid_size: [u32; 4],  // xyz + pad
    pub brick_capacity: u32,
    pub allocated_bricks: u32,
    pub _pad: [u32; 2],
}

/// All GPU buffers for UCVH data.
pub struct UcvhGpuResources {
    pub config_buffer: GpuBuffer,
    pub occupancy_buffer: GpuBuffer,
    pub material_buffer: GpuBuffer,
    pub hierarchy_l0_buffer: GpuBuffer,
    pub hierarchy_ln_buffers: [GpuBuffer; 4], // L1-L4
    // Staging buffers (host-visible, used for transfer)
    staging_occupancy: GpuBuffer,
    staging_material: GpuBuffer,
    staging_hierarchy: GpuBuffer,
    staging_config: GpuBuffer,
}

impl UcvhGpuResources {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        ucvh: &Ucvh,
    ) -> Result<Self> {
        let cap = ucvh.pool.capacity() as usize;
        let occ_size = cap * std::mem::size_of::<BrickOccupancy>();
        let mat_size = cap * BRICK_VOLUME * std::mem::size_of::<VoxelCell>();

        let ssbo_usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        let staging_usage = vk::BufferUsageFlags::TRANSFER_SRC;

        // Device-local SSBOs
        let config_buffer = GpuBuffer::new(
            device, allocator,
            std::mem::size_of::<UcvhGpuConfig>() as u64,
            ssbo_usage, MemoryLocation::GpuOnly, "ucvh_config",
        )?;
        let occupancy_buffer = GpuBuffer::new(
            device, allocator, occ_size as u64,
            ssbo_usage, MemoryLocation::GpuOnly, "ucvh_occupancy",
        )?;
        let material_buffer = GpuBuffer::new(
            device, allocator, mat_size as u64,
            ssbo_usage, MemoryLocation::GpuOnly, "ucvh_materials",
        )?;

        // Hierarchy buffers
        let h = &ucvh.hierarchy;
        let l0_size = h.level0.len() * std::mem::size_of::<NodeL0>();
        let ln_sizes: [usize; 4] = std::array::from_fn(|i| {
            h.levels[i].len() * std::mem::size_of::<NodeLN>()
        });

        let hierarchy_l0_buffer = GpuBuffer::new(
            device, allocator, l0_size.max(16) as u64,
            ssbo_usage, MemoryLocation::GpuOnly, "ucvh_hierarchy_l0",
        )?;
        let hierarchy_ln_buffers = [
            GpuBuffer::new(device, allocator, ln_sizes[0].max(16) as u64, ssbo_usage, MemoryLocation::GpuOnly, "ucvh_hierarchy_l1")?,
            GpuBuffer::new(device, allocator, ln_sizes[1].max(16) as u64, ssbo_usage, MemoryLocation::GpuOnly, "ucvh_hierarchy_l2")?,
            GpuBuffer::new(device, allocator, ln_sizes[2].max(16) as u64, ssbo_usage, MemoryLocation::GpuOnly, "ucvh_hierarchy_l3")?,
            GpuBuffer::new(device, allocator, ln_sizes[3].max(16) as u64, ssbo_usage, MemoryLocation::GpuOnly, "ucvh_hierarchy_l4")?,
        ];

        // Staging buffers (host-visible)
        let total_hierarchy = l0_size + ln_sizes.iter().sum::<usize>();
        let staging_occupancy = GpuBuffer::new(
            device, allocator, occ_size as u64,
            staging_usage, MemoryLocation::CpuToGpu, "staging_occupancy",
        )?;
        let staging_material = GpuBuffer::new(
            device, allocator, mat_size as u64,
            staging_usage, MemoryLocation::CpuToGpu, "staging_materials",
        )?;
        let staging_hierarchy = GpuBuffer::new(
            device, allocator, total_hierarchy.max(16) as u64,
            staging_usage, MemoryLocation::CpuToGpu, "staging_hierarchy",
        )?;
        let staging_config = GpuBuffer::new(
            device, allocator,
            std::mem::size_of::<UcvhGpuConfig>() as u64,
            staging_usage, MemoryLocation::CpuToGpu, "staging_config",
        )?;

        Ok(Self {
            config_buffer,
            occupancy_buffer,
            material_buffer,
            hierarchy_l0_buffer,
            hierarchy_ln_buffers,
            staging_occupancy,
            staging_material,
            staging_hierarchy,
            staging_config,
        })
    }

    /// Upload all UCVH data to GPU. Call once after scene generation.
    /// Records copy commands into `cmd` — must be called between begin/end command buffer.
    pub fn upload_all(&self, device: &ash::Device, cmd: vk::CommandBuffer, ucvh: &Ucvh) {
        // Upload config
        let gpu_config = UcvhGpuConfig {
            world_size: [ucvh.config.world_size.x, ucvh.config.world_size.y, ucvh.config.world_size.z, 0],
            brick_grid_size: [ucvh.config.brick_grid_size.x, ucvh.config.brick_grid_size.y, ucvh.config.brick_grid_size.z, 0],
            brick_capacity: ucvh.pool.capacity(),
            allocated_bricks: ucvh.pool.allocated_count(),
            _pad: [0; 2],
        };
        Self::copy_to_staging(&self.staging_config, bytes_of(&gpu_config));
        Self::record_copy(device, cmd, &self.staging_config, &self.config_buffer, std::mem::size_of::<UcvhGpuConfig>() as u64);

        // Upload occupancy pool
        let occ_bytes = cast_slice::<BrickOccupancy, u8>(ucvh.pool.occupancy_pool());
        Self::copy_to_staging(&self.staging_occupancy, occ_bytes);
        Self::record_copy(device, cmd, &self.staging_occupancy, &self.occupancy_buffer, occ_bytes.len() as u64);

        // Upload material pool
        let mat_bytes = cast_slice::<VoxelCell, u8>(ucvh.pool.material_pool());
        Self::copy_to_staging(&self.staging_material, mat_bytes);
        Self::record_copy(device, cmd, &self.staging_material, &self.material_buffer, mat_bytes.len() as u64);

        // Upload hierarchy
        let mut offset = 0u64;
        let l0_bytes = cast_slice::<NodeL0, u8>(&ucvh.hierarchy.level0);
        Self::copy_to_staging_offset(&self.staging_hierarchy, l0_bytes, offset as usize);
        offset += l0_bytes.len() as u64;

        let mut ln_offsets = [0u64; 4];
        for i in 0..4 {
            ln_offsets[i] = offset;
            let ln_bytes = cast_slice::<NodeLN, u8>(&ucvh.hierarchy.levels[i]);
            Self::copy_to_staging_offset(&self.staging_hierarchy, ln_bytes, offset as usize);
            offset += ln_bytes.len() as u64;
        }

        // Record copies: staging_hierarchy -> individual device-local buffers
        let l0_size = l0_bytes.len() as u64;
        Self::record_copy_region(device, cmd, &self.staging_hierarchy, &self.hierarchy_l0_buffer, 0, 0, l0_size);
        for i in 0..4 {
            let ln_size = (ucvh.hierarchy.levels[i].len() * std::mem::size_of::<NodeLN>()) as u64;
            Self::record_copy_region(device, cmd, &self.staging_hierarchy, &self.hierarchy_ln_buffers[i], ln_offsets[i], 0, ln_size);
        }

        // Buffer memory barrier: ensure transfers complete before shader reads
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[barrier], &[], &[],
            );
        }
    }

    fn copy_to_staging(buffer: &GpuBuffer, data: &[u8]) {
        if let Some(ptr) = buffer.mapped_ptr() {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len()) };
        }
    }

    fn copy_to_staging_offset(buffer: &GpuBuffer, data: &[u8], offset: usize) {
        if let Some(ptr) = buffer.mapped_ptr() {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len()) };
        }
    }

    fn record_copy(device: &ash::Device, cmd: vk::CommandBuffer, src: &GpuBuffer, dst: &GpuBuffer, size: u64) {
        Self::record_copy_region(device, cmd, src, dst, 0, 0, size);
    }

    fn record_copy_region(device: &ash::Device, cmd: vk::CommandBuffer, src: &GpuBuffer, dst: &GpuBuffer, src_offset: u64, dst_offset: u64, size: u64) {
        if size == 0 { return; }
        let region = vk::BufferCopy {
            src_offset,
            dst_offset,
            size,
        };
        unsafe { device.cmd_copy_buffer(cmd, src.handle, dst.handle, &[region]) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.config_buffer.destroy(device, allocator);
        self.occupancy_buffer.destroy(device, allocator);
        self.material_buffer.destroy(device, allocator);
        self.hierarchy_l0_buffer.destroy(device, allocator);
        for buf in self.hierarchy_ln_buffers {
            buf.destroy(device, allocator);
        }
        self.staging_occupancy.destroy(device, allocator);
        self.staging_material.destroy(device, allocator);
        self.staging_hierarchy.destroy(device, allocator);
        self.staging_config.destroy(device, allocator);
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build`
Expected: compiles (no unit tests for GPU code — tested via integration in Task 8)

- [ ] **Step 3: Uncomment module in mod.rs and commit**

Uncomment `pub mod gpu_upload;` in `src/voxel/mod.rs`.

```bash
git add src/voxel/gpu_upload.rs src/voxel/mod.rs
git commit -m "feat(voxel): add UcvhGpuResources with staging upload pipeline"
```

---

## Task 8: Integration — Wire UCVH into App

**Files:**
- Modify: `src/app.rs`

This wires everything together: create the UCVH, generate the demo sphere scene, upload data to GPU on the first frame, and log the result. The test pattern pass remains as visual output until Phase 3 adds ray tracing.

- [ ] **Step 1: Add UCVH fields to RevolumetricApp**

Add to `RevolumetricApp` struct fields:
```rust
ucvh: Option<Ucvh>,
ucvh_gpu: Option<UcvhGpuResources>,
ucvh_uploaded: bool,
```

Add imports at top of `src/app.rs`:
```rust
use crate::voxel::ucvh::{Ucvh, UcvhConfig};
use crate::voxel::generator;
use crate::voxel::gpu_upload::UcvhGpuResources;
```

Initialize in `RevolumetricApp::new()`:
```rust
ucvh: None,
ucvh_gpu: None,
ucvh_uploaded: false,
```

- [ ] **Step 2: Generate scene in `resumed()` after renderer init**

After `test_pattern_pass` initialization, add:
```rust
// Generate UCVH demo scene
if self.ucvh.is_none() {
    let config = UcvhConfig::new(glam::UVec3::splat(128));
    let mut ucvh = Ucvh::new(config);
    let brick_count = generator::generate_demo_scene(&mut ucvh);
    ucvh.rebuild_hierarchy();
    tracing::info!(
        bricks = brick_count,
        total_voxels = ucvh.pool.allocated_count() as u64 * 512,
        "generated demo sphere scene"
    );

    let renderer = self.renderer.as_ref().unwrap();
    match UcvhGpuResources::new(renderer.device(), renderer.allocator(), &ucvh) {
        Ok(gpu) => {
            tracing::info!("created UCVH GPU resources");
            self.ucvh_gpu = Some(gpu);
        }
        Err(e) => tracing::error!(%e, "failed to create UCVH GPU resources"),
    }
    self.ucvh = Some(ucvh);
}
```

- [ ] **Step 3: Upload UCVH data on first frame in `tick_frame()`**

In `tick_frame()`, after `begin_frame()` succeeds and before the render graph, add:
```rust
// Upload UCVH data to GPU (first frame only)
if !self.ucvh_uploaded {
    if let (Some(ucvh), Some(gpu)) = (&self.ucvh, &self.ucvh_gpu) {
        gpu.upload_all(renderer.device(), frame.command_buffer, ucvh);
        self.ucvh_uploaded = true;
        tracing::info!("uploaded UCVH data to GPU");
    }
}
```

- [ ] **Step 4: Destroy UCVH GPU resources in Drop impl**

Update the `Drop` impl for `RevolumetricApp`:
```rust
impl Drop for RevolumetricApp {
    fn drop(&mut self) {
        if let Some(renderer) = &self.renderer {
            unsafe { renderer.device().device_wait_idle().ok() };
            if let Some(pass) = self.test_pattern_pass.take() {
                pass.destroy(renderer.device(), renderer.allocator());
            }
            if let Some(gpu) = self.ucvh_gpu.take() {
                gpu.destroy(renderer.device(), renderer.allocator());
            }
        }
    }
}
```

- [ ] **Step 5: Build and run**

Run: `cargo build && cargo run`
Expected:
- No compilation errors
- Log output shows: `generated demo sphere scene` with brick count > 0
- Log output shows: `uploaded UCVH data to GPU`
- Animated rainbow gradient still displays (test pattern pass unchanged)
- No Vulkan validation errors on startup or shutdown

- [ ] **Step 6: Run all tests**

Run: `cargo test --lib`
Expected: all tests pass (morton, brick, brick_pool, occupancy, ucvh, generator)

- [ ] **Step 7: Commit**

```bash
git add src/app.rs
git commit -m "feat(app): integrate UCVH demo scene generation and GPU upload"
```

---

## Task 9: Cleanup and Final Verification

**Files:**
- Modify: `src/voxel/mod.rs` (final cleanup)
- Modify: `src/voxel/material.rs` (keep as-is for now, used by future palette system)

- [ ] **Step 1: Verify final voxel/mod.rs is clean**

`src/voxel/mod.rs` should contain:
```rust
pub mod brick;
pub mod brick_pool;
pub mod generator;
pub mod gpu_upload;
pub mod material;
pub mod morton;
pub mod occupancy;
pub mod ucvh;
```

Ensure no references to deleted files (`grid`, `chunk`, `scene_baker`, `radiance_layout`).

- [ ] **Step 2: Run full test suite**

Run: `cargo test --lib`
Expected: all ~25 tests pass across morton, brick, brick_pool, occupancy, ucvh, generator, render::graph

- [ ] **Step 3: Run the application and verify no validation errors**

Run: `RUST_LOG=info cargo run`
Expected output includes:
```
INFO initialized renderer bootstrap ...
INFO initialized test pattern pass ...
INFO generated demo sphere scene bricks=NNN total_voxels=NNN
INFO created UCVH GPU resources
INFO uploaded UCVH data to GPU
```
No Vulkan validation errors. Window shows animated rainbow gradient. Clean shutdown.

- [ ] **Step 4: Commit final cleanup**

```bash
git add -A
git commit -m "chore(voxel): finalize Phase 2 module structure"
```

---

## Summary

| Task | Module | Tests | Commit |
|------|--------|-------|--------|
| 1 | `morton.rs` | 5 unit tests | Morton encode/decode |
| 2 | `brick.rs` | 6 unit tests | BrickOccupancy + VoxelCell |
| 3 | `brick_pool.rs` | 5 unit tests | Free-list allocation |
| 4 | `occupancy.rs` | 5 unit tests | 5-level hierarchy + rebuild |
| 5 | `ucvh.rs` | 5 unit tests | Facade + dirty tracking |
| 6 | `generator.rs` | 3 unit tests | Sphere demo scene |
| 7 | `gpu_upload.rs` | compile check | GPU SSBOs + staging |
| 8 | `app.rs` | integration run | Wire everything together |
| 9 | cleanup | full suite | Final verification |

**Total: ~29 unit tests, 9 commits, ~7 new files**

Phase 3 (Voxel Ray Tracing) will add a compute shader that binds these SSBOs and traces rays through the hierarchical DDA — the data will already be on the GPU waiting.
