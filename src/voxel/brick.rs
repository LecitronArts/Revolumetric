use crate::voxel::morton;
use bytemuck::{Pod, Zeroable};

pub const BRICK_EDGE: u32 = 8;
pub const BRICK_VOLUME: usize = 512; // 8^3

/// Occupancy bitmask for an 8^3 brick (hot-path, fits GPU L2).
/// Bits are indexed in Morton order matching material storage.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BrickOccupancy {
    pub bits: [u32; 16], // 512 bits
    pub count: u32,      // number of solid voxels (0 = empty, skip instantly)
    pub _pad: [u32; 3],  // pad to 80 bytes (16-byte GPU alignment)
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
    pub emissive: [u8; 3],
    pub _pad: u8,
}

impl VoxelCell {
    pub const AIR: Self = Self {
        material: 0,
        flags: 0,
        emissive: [0; 3],
        _pad: 0,
    };
    pub fn is_air(&self) -> bool {
        self.material == 0
    }

    pub fn new(material: u16, flags: u16, emissive: [u8; 3]) -> Self {
        Self {
            material,
            flags,
            emissive,
            _pad: 0,
        }
    }
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

impl Default for BrickData {
    fn default() -> Self {
        Self::new()
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
        let cell = VoxelCell {
            material: 42,
            flags: 1,
            emissive: [0; 3],
            _pad: 0,
        };
        b.set_voxel(7, 7, 7, cell);
        assert!(b.occupancy.get(7, 7, 7));
        assert_eq!(b.get_voxel(7, 7, 7).material, 42);
        assert_eq!(b.occupancy.count, 1);
    }

    #[test]
    fn brick_data_air_clears() {
        let mut b = BrickData::new();
        b.set_voxel(
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
        assert!(b.occupancy.get(0, 0, 0));
        b.set_voxel(0, 0, 0, VoxelCell::AIR);
        assert!(!b.occupancy.get(0, 0, 0));
        assert_eq!(b.occupancy.count, 0);
    }
}
