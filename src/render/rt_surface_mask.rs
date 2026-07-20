use crate::voxel::brick::BRICK_EDGE;
use crate::voxel::morton;
use crate::voxel::ucvh::{UCVH_NO_BRICK_GENERATION, Ucvh};
use glam::{IVec3, UVec3};

pub const SURFACE_MASK_DIRECTION_COUNT: usize = 6;
pub const SURFACE_MASK_WORDS_PER_DIRECTION: usize = 8;
pub const SURFACE_MASK_PAYLOAD_BYTES: usize = 384;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FaceDirection {
    NegativeX = 0,
    PositiveX = 1,
    NegativeY = 2,
    PositiveY = 3,
    NegativeZ = 4,
    PositiveZ = 5,
}

impl FaceDirection {
    pub const ALL: [Self; SURFACE_MASK_DIRECTION_COUNT] = [
        Self::NegativeX,
        Self::PositiveX,
        Self::NegativeY,
        Self::PositiveY,
        Self::NegativeZ,
        Self::PositiveZ,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn opposite(self) -> Self {
        match self {
            Self::NegativeX => Self::PositiveX,
            Self::PositiveX => Self::NegativeX,
            Self::NegativeY => Self::PositiveY,
            Self::PositiveY => Self::NegativeY,
            Self::NegativeZ => Self::PositiveZ,
            Self::PositiveZ => Self::NegativeZ,
        }
    }

    pub const fn offset(self) -> IVec3 {
        match self {
            Self::NegativeX => IVec3::NEG_X,
            Self::PositiveX => IVec3::X,
            Self::NegativeY => IVec3::NEG_Y,
            Self::PositiveY => IVec3::Y,
            Self::NegativeZ => IVec3::NEG_Z,
            Self::PositiveZ => IVec3::Z,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceMaskDependencyStamp {
    pub brick_id: u32,
    pub generation: u32,
    pub topology_revision: u64,
}

impl SurfaceMaskDependencyStamp {
    pub const ABSENT: Self = Self {
        brick_id: u32::MAX,
        generation: UCVH_NO_BRICK_GENERATION,
        topology_revision: 0,
    };

    fn topology_matches(self, current: Self) -> bool {
        self.brick_id == current.brick_id && self.topology_revision == current.topology_revision
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceMaskSourceStamp {
    pub page: UVec3,
    /// Owner first, followed by neighbors in `FaceDirection::ALL` order.
    pub dependencies: [SurfaceMaskDependencyStamp; SURFACE_MASK_DIRECTION_COUNT + 1],
}

impl SurfaceMaskSourceStamp {
    pub fn from_ucvh(ucvh: &Ucvh, page: UVec3) -> Self {
        let mut dependencies =
            [SurfaceMaskDependencyStamp::ABSENT; SURFACE_MASK_DIRECTION_COUNT + 1];
        dependencies[0] = dependency_stamp(ucvh, page.as_ivec3());
        for direction in FaceDirection::ALL {
            dependencies[direction.index() + 1] =
                dependency_stamp(ucvh, page.as_ivec3() + direction.offset());
        }
        Self { page, dependencies }
    }

    pub fn matches_ucvh(&self, ucvh: &Ucvh) -> bool {
        let current = Self::from_ucvh(ucvh, self.page);
        self.dependencies
            .iter()
            .copied()
            .zip(current.dependencies)
            .all(|(recorded, current)| recorded.topology_matches(current))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceMaskPage {
    directions: [[u64; SURFACE_MASK_WORDS_PER_DIRECTION]; SURFACE_MASK_DIRECTION_COUNT],
    source_stamp: SurfaceMaskSourceStamp,
}

impl SurfaceMaskPage {
    pub fn from_ucvh(ucvh: &Ucvh, page: UVec3) -> Self {
        let source_stamp = SurfaceMaskSourceStamp::from_ucvh(ucvh, page);
        let mut result = Self {
            directions: [[0; SURFACE_MASK_WORDS_PER_DIRECTION]; SURFACE_MASK_DIRECTION_COUNT],
            source_stamp,
        };
        if page.x >= ucvh.config.brick_grid_size.x
            || page.y >= ucvh.config.brick_grid_size.y
            || page.z >= ucvh.config.brick_grid_size.z
        {
            return result;
        }

        let origin = (page * BRICK_EDGE).as_ivec3();
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let local = UVec3::new(x, y, z);
                    let owner = origin + local.as_ivec3();
                    if !voxel_is_solid(ucvh, owner) {
                        continue;
                    }
                    for direction in FaceDirection::ALL {
                        if !voxel_is_solid(ucvh, owner + direction.offset()) {
                            result.set_exposed(local, direction);
                        }
                    }
                }
            }
        }
        result
    }

    pub fn directions(
        &self,
    ) -> &[[u64; SURFACE_MASK_WORDS_PER_DIRECTION]; SURFACE_MASK_DIRECTION_COUNT] {
        &self.directions
    }

    pub fn source_stamp(&self) -> SurfaceMaskSourceStamp {
        self.source_stamp
    }

    pub fn is_exposed(&self, local: UVec3, direction: FaceDirection) -> bool {
        if local.x >= BRICK_EDGE || local.y >= BRICK_EDGE || local.z >= BRICK_EDGE {
            return false;
        }
        let bit_index = morton::encode(local.x, local.y, local.z) as usize;
        self.directions[direction.index()][bit_index / 64] & (1u64 << (bit_index % 64)) != 0
    }

    pub fn exposed_face_count(&self) -> u32 {
        self.directions
            .iter()
            .flatten()
            .map(|word| word.count_ones())
            .sum()
    }

    pub fn interface_cell_count(&self) -> u32 {
        self.exposed_face_count()
    }

    fn set_exposed(&mut self, local: UVec3, direction: FaceDirection) {
        let bit_index = morton::encode(local.x, local.y, local.z) as usize;
        self.directions[direction.index()][bit_index / 64] |= 1u64 << (bit_index % 64);
    }
}

fn dependency_stamp(ucvh: &Ucvh, brick_coord: IVec3) -> SurfaceMaskDependencyStamp {
    if brick_coord.x < 0
        || brick_coord.y < 0
        || brick_coord.z < 0
        || brick_coord.x >= ucvh.config.brick_grid_size.x as i32
        || brick_coord.y >= ucvh.config.brick_grid_size.y as i32
        || brick_coord.z >= ucvh.config.brick_grid_size.z as i32
    {
        return SurfaceMaskDependencyStamp::ABSENT;
    }
    let Some(brick_id) = ucvh.brick_id_at(brick_coord.as_uvec3()) else {
        return SurfaceMaskDependencyStamp::ABSENT;
    };
    SurfaceMaskDependencyStamp {
        brick_id,
        generation: ucvh
            .brick_generation(brick_id)
            .unwrap_or(UCVH_NO_BRICK_GENERATION),
        topology_revision: ucvh.brick_topology_revision(brick_id).unwrap_or(0),
    }
}

fn voxel_is_solid(ucvh: &Ucvh, position: IVec3) -> bool {
    position.x >= 0
        && position.y >= 0
        && position.z >= 0
        && !ucvh.get_voxel(position.as_uvec3()).is_air()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::brick::VoxelCell;
    use crate::voxel::ucvh::{Ucvh, UcvhConfig};
    use glam::UVec3;

    const SOLID: VoxelCell = VoxelCell {
        material: 1,
        flags: 0,
        emissive: [0; 3],
        _pad: 0,
    };

    fn ucvh(world_size: UVec3) -> Ucvh {
        Ucvh::new(UcvhConfig::new(world_size))
    }

    fn fill_page(ucvh: &mut Ucvh, page: UVec3, cell: VoxelCell) {
        let origin = page * 8;
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    assert!(ucvh.set_voxel(origin + UVec3::new(x, y, z), cell));
                }
            }
        }
    }

    #[test]
    fn rt_surface_mask_empty_page_has_no_owned_faces() {
        let ucvh = ucvh(UVec3::splat(8));
        let mask = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::ZERO);

        assert_eq!(mask.exposed_face_count(), 0);
        assert_eq!(mask.interface_cell_count(), 0);
        assert_eq!(std::mem::size_of_val(mask.directions()), 384);
        assert!(!mask.is_exposed(UVec3::ZERO, FaceDirection::NegativeX));
    }

    #[test]
    fn rt_surface_mask_single_voxel_owns_six_faces() {
        let mut ucvh = ucvh(UVec3::splat(8));
        assert!(ucvh.set_voxel(UVec3::new(3, 4, 5), SOLID));

        let mask = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::ZERO);

        assert_eq!(mask.exposed_face_count(), 6);
        for direction in FaceDirection::ALL {
            assert!(mask.is_exposed(UVec3::new(3, 4, 5), direction));
        }
    }

    #[test]
    fn rt_surface_mask_full_isolated_page_contains_only_outer_shell() {
        let mut ucvh = ucvh(UVec3::splat(8));
        fill_page(&mut ucvh, UVec3::ZERO, SOLID);

        let mask = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::ZERO);

        assert_eq!(mask.exposed_face_count(), 6 * 8 * 8);
        assert!(mask.is_exposed(UVec3::new(0, 3, 4), FaceDirection::NegativeX));
        assert!(!mask.is_exposed(UVec3::new(3, 3, 3), FaceDirection::PositiveX));
    }

    #[test]
    fn rt_surface_mask_cavity_adds_six_inward_faces() {
        let mut ucvh = ucvh(UVec3::splat(8));
        fill_page(&mut ucvh, UVec3::ZERO, SOLID);
        assert!(ucvh.set_voxel(UVec3::new(3, 3, 3), VoxelCell::AIR));

        let mask = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::ZERO);

        assert_eq!(mask.exposed_face_count(), 6 * 8 * 8 + 6);
        assert!(mask.is_exposed(UVec3::new(2, 3, 3), FaceDirection::PositiveX));
        assert!(!mask.is_exposed(UVec3::new(3, 3, 3), FaceDirection::NegativeX));
    }

    #[test]
    fn rt_surface_mask_checkerboard_has_six_faces_per_solid_owner() {
        let mut ucvh = ucvh(UVec3::splat(8));
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    if (x + y + z) % 2 == 0 {
                        assert!(ucvh.set_voxel(UVec3::new(x, y, z), SOLID));
                    }
                }
            }
        }

        let mask = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::ZERO);

        assert_eq!(mask.exposed_face_count(), 256 * 6);
        assert_eq!(mask.interface_cell_count(), mask.exposed_face_count());
    }

    #[test]
    fn rt_surface_mask_world_edge_treats_outside_as_air() {
        let mut ucvh = ucvh(UVec3::splat(8));
        fill_page(&mut ucvh, UVec3::ZERO, SOLID);

        let mask = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::ZERO);
        let outside = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::X);

        assert!(mask.is_exposed(UVec3::new(7, 2, 3), FaceDirection::PositiveX));
        assert_eq!(outside.exposed_face_count(), 0);
    }

    #[test]
    fn rt_surface_mask_cross_page_solid_solid_interface_is_hidden_on_both_sides() {
        let mut ucvh = ucvh(UVec3::new(16, 8, 8));
        assert!(ucvh.set_voxel(UVec3::new(7, 2, 3), SOLID));
        assert!(ucvh.set_voxel(UVec3::new(8, 2, 3), SOLID));

        let left = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::ZERO);
        let right = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::X);

        assert!(!left.is_exposed(UVec3::new(7, 2, 3), FaceDirection::PositiveX));
        assert!(!right.is_exposed(UVec3::new(0, 2, 3), FaceDirection::NegativeX));
        assert_eq!(left.exposed_face_count() + right.exposed_face_count(), 10);
    }

    #[test]
    fn rt_surface_mask_cross_page_solid_air_interface_has_exactly_one_owner() {
        let mut ucvh = ucvh(UVec3::new(16, 8, 8));
        assert!(ucvh.set_voxel(UVec3::new(7, 2, 3), SOLID));

        let left = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::ZERO);
        let right = SurfaceMaskPage::from_ucvh(&ucvh, UVec3::X);

        assert!(left.is_exposed(UVec3::new(7, 2, 3), FaceDirection::PositiveX));
        assert!(!right.is_exposed(UVec3::new(0, 2, 3), FaceDirection::NegativeX));
        assert_eq!(left.exposed_face_count(), 6);
        assert_eq!(right.exposed_face_count(), 0);
    }

    #[test]
    fn rt_surface_mask_source_stamp_tracks_owner_and_six_neighbor_topology() {
        let mut ucvh = ucvh(UVec3::new(24, 24, 24));
        assert!(ucvh.set_voxel(UVec3::new(9, 9, 9), SOLID));
        let page = UVec3::ONE;
        let stamp = SurfaceMaskSourceStamp::from_ucvh(&ucvh, page);
        assert!(stamp.matches_ucvh(&ucvh));

        assert!(ucvh.set_voxel(UVec3::new(8, 9, 9), SOLID));
        assert!(!stamp.matches_ucvh(&ucvh));

        let stamp = SurfaceMaskSourceStamp::from_ucvh(&ucvh, page);
        assert!(ucvh.set_voxel(UVec3::new(7, 9, 9), SOLID));
        assert!(!stamp.matches_ucvh(&ucvh));
    }

    #[test]
    fn rt_surface_mask_source_stamp_ignores_material_only_edits() {
        let mut ucvh = ucvh(UVec3::new(16, 8, 8));
        let owner = UVec3::new(7, 2, 3);
        let neighbor = UVec3::new(8, 2, 3);
        assert!(ucvh.set_voxel(owner, SOLID));
        assert!(ucvh.set_voxel(neighbor, SOLID));
        let stamp = SurfaceMaskSourceStamp::from_ucvh(&ucvh, UVec3::ZERO);

        assert!(ucvh.set_voxel(owner, VoxelCell::new(2, 1, [1, 2, 3])));
        assert!(ucvh.set_voxel(neighbor, VoxelCell::new(3, 1, [3, 2, 1])));

        assert!(stamp.matches_ucvh(&ucvh));
    }
}
