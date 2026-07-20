use crate::render::rt_surface_mask::FaceDirection;
use bytemuck::{Pod, Zeroable};
use glam::UVec3;
use thiserror::Error;

pub const RT_REFERENCE_HIT_GROUP_ID: u32 = 0;
pub const RT_COMPACT_EXACT_HIT_GROUP_ID: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RtPageRepresentationId {
    Reference = 0,
    CompactExact = 1,
    HotOmm = 2,
    HotInterface = 3,
    CompactGreedy = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuRtSurfaceKey {
    pub page_coord: [u32; 3],
    pub page_slot: u32,
    pub brick_id: u32,
    pub owner_local_and_face: u32,
    pub material_generation: u32,
    pub reserved: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtSurfaceOwner {
    local_voxel: UVec3,
    face: FaceDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum RtSurfaceOwnerError {
    #[error("RT surface owner local voxel is outside an 8x8x8 brick: {0:?}")]
    LocalVoxelOutOfRange(UVec3),
    #[error("RT surface owner contains reserved packed bits: {0:#x}")]
    ReservedBits(u32),
    #[error("RT surface owner face direction is invalid: {0}")]
    InvalidFace(u32),
}

impl RtSurfaceOwner {
    const COORD_MASK: u32 = 0x7;
    const FACE_SHIFT: u32 = 9;
    const USED_MASK: u32 = 0x0fff;

    pub fn new(local_voxel: UVec3, face: FaceDirection) -> Result<Self, RtSurfaceOwnerError> {
        if local_voxel.x > 7 || local_voxel.y > 7 || local_voxel.z > 7 {
            return Err(RtSurfaceOwnerError::LocalVoxelOutOfRange(local_voxel));
        }
        Ok(Self { local_voxel, face })
    }

    pub fn from_packed(packed: u32) -> Result<Self, RtSurfaceOwnerError> {
        if packed & !Self::USED_MASK != 0 {
            return Err(RtSurfaceOwnerError::ReservedBits(packed));
        }
        let face_index = (packed >> Self::FACE_SHIFT) & 0x7;
        let face = FaceDirection::ALL
            .get(face_index as usize)
            .copied()
            .ok_or(RtSurfaceOwnerError::InvalidFace(face_index))?;
        Self::new(
            UVec3::new(
                packed & Self::COORD_MASK,
                (packed >> 3) & Self::COORD_MASK,
                (packed >> 6) & Self::COORD_MASK,
            ),
            face,
        )
    }

    pub const fn packed(self) -> u32 {
        self.local_voxel.x
            | (self.local_voxel.y << 3)
            | (self.local_voxel.z << 6)
            | ((self.face as u32) << Self::FACE_SHIFT)
    }

    pub const fn local_voxel(self) -> UVec3 {
        self.local_voxel
    }

    pub const fn face(self) -> FaceDirection {
        self.face
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::rt_surface_mask::FaceDirection;
    use glam::UVec3;

    #[test]
    fn gpu_surface_key_layout_is_stable_and_32_bytes() {
        assert_eq!(std::mem::size_of::<GpuRtSurfaceKey>(), 32);
        assert_eq!(std::mem::offset_of!(GpuRtSurfaceKey, page_coord), 0);
        assert_eq!(std::mem::offset_of!(GpuRtSurfaceKey, page_slot), 12);
        assert_eq!(std::mem::offset_of!(GpuRtSurfaceKey, brick_id), 16);
        assert_eq!(
            std::mem::offset_of!(GpuRtSurfaceKey, owner_local_and_face),
            20
        );
        assert_eq!(
            std::mem::offset_of!(GpuRtSurfaceKey, material_generation),
            24
        );
    }

    #[test]
    fn surface_owner_pack_round_trips_local_voxel_and_all_faces() {
        for direction in FaceDirection::ALL {
            let packed = RtSurfaceOwner::new(UVec3::new(7, 3, 5), direction)
                .unwrap()
                .packed();
            let decoded = RtSurfaceOwner::from_packed(packed).unwrap();
            assert_eq!(decoded.local_voxel(), UVec3::new(7, 3, 5));
            assert_eq!(decoded.face(), direction);
        }
    }

    #[test]
    fn surface_owner_rejects_out_of_brick_coordinates_and_reserved_bits() {
        assert!(RtSurfaceOwner::new(UVec3::new(8, 0, 0), FaceDirection::PositiveX).is_err());
        assert!(RtSurfaceOwner::from_packed(1 << 16).is_err());
    }

    #[test]
    fn representation_and_hit_group_ids_match_tlas_offsets() {
        use crate::render::rt_page_tlas::{
            RT_PAGE_COMPACT_EXACT_HIT_GROUP_OFFSET, RT_PAGE_REFERENCE_HIT_GROUP_OFFSET,
        };

        assert_eq!(RtPageRepresentationId::Reference as u32, 0);
        assert_eq!(RtPageRepresentationId::CompactExact as u32, 1);
        assert_eq!(RT_REFERENCE_HIT_GROUP_ID, 0);
        assert_eq!(RT_COMPACT_EXACT_HIT_GROUP_ID, 1);
        assert_eq!(
            RT_REFERENCE_HIT_GROUP_ID,
            RT_PAGE_REFERENCE_HIT_GROUP_OFFSET
        );
        assert_eq!(
            RT_COMPACT_EXACT_HIT_GROUP_ID,
            RT_PAGE_COMPACT_EXACT_HIT_GROUP_OFFSET
        );
    }

    #[test]
    fn slang_page_common_mirrors_surface_key_and_owner_packing() {
        let source =
            crate::render::source_checks::read_source("assets/shaders/shared/rt_page_common.slang");
        for token in [
            "struct SurfaceKey",
            "uint page_slot",
            "uint3 page_coord",
            "uint brick_id",
            "uint owner_local_and_face",
            "uint material_generation",
            "RT_REFERENCE_HIT_GROUP_ID",
            "RT_COMPACT_EXACT_HIT_GROUP_ID",
            "decode_surface_owner",
        ] {
            assert!(source.contains(token), "shared RT page ABI missing {token}");
        }
        let compact = crate::render::source_checks::compact(&source);
        assert!(compact.contains(
            "staticconstuintRT_REFERENCE_HIT_GROUP_ID=0;staticconstuintRT_COMPACT_EXACT_HIT_GROUP_ID=1;"
        ));
        assert!(compact.contains(
            "structSurfaceKey{uint3page_coord;uintpage_slot;uintbrick_id;uintowner_local_and_face;uintmaterial_generation;uintreserved;}"
        ));
        for packing in [
            "packed_owner&7u",
            "(packed_owner>>3u)&7u",
            "(packed_owner>>6u)&7u",
            "(packed_owner>>9u)&7u",
        ] {
            assert!(
                compact.contains(packing),
                "Slang owner packing drifted: {packing}"
            );
        }
    }
}
