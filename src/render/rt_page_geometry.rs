use crate::render::rt_hit_abi::RtSurfaceOwner;
use crate::render::rt_surface_mask::{FaceDirection, SurfaceMaskPage};
use crate::voxel::morton;
use bytemuck::{Pod, Zeroable};
use glam::UVec3;

pub const RT_PAGE_LATTICE_EDGE: usize = 9;
pub const RT_PAGE_LATTICE_VERTEX_COUNT: usize =
    RT_PAGE_LATTICE_EDGE * RT_PAGE_LATTICE_EDGE * RT_PAGE_LATTICE_EDGE;
pub const RT_INTERNAL_INTERFACE_QUAD_COUNT: usize = 3 * 7 * 8 * 8;
pub const RT_OWNER_BOUNDARY_QUAD_COUNT: usize = 6 * 8 * 8;
pub const RT_INTERFACE_QUAD_COUNT: usize =
    RT_INTERNAL_INTERFACE_QUAD_COUNT + RT_OWNER_BOUNDARY_QUAD_COUNT;
pub const RT_INTERFACE_TRIANGLE_COUNT: usize = RT_INTERFACE_QUAD_COUNT * 2;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct RtCompactFaceRecord {
    pub packed_owner_direction: u32,
}

impl RtCompactFaceRecord {
    /// Pack `owner_local` and `direction` into the linear layout used by the GPU shader.
    ///
    /// Layout: `x[2:0] | y[5:3] | z[8:6] | face[11:9]`
    ///
    /// This matches `decode_surface_owner()` in `rt_page_common.slang` exactly, so the
    /// bytes written to the GPU face buffer are directly readable by the closest-hit shader
    /// without any conversion.  The previous morton layout (`morton(x,y,z) << 3 | dir`) was
    /// incompatible with the shader decoder and is intentionally replaced here.
    pub fn new(owner_local: UVec3, direction: FaceDirection) -> Self {
        Self {
            packed_owner_direction: RtSurfaceOwner::new(owner_local, direction)
                .expect("compact face record owner must be within an 8x8x8 brick")
                .packed(),
        }
    }

    pub fn owner_local(self) -> UVec3 {
        RtSurfaceOwner::from_packed(self.packed_owner_direction)
            .expect("compact face record must contain a valid linear-packed owner")
            .local_voxel()
    }

    pub fn direction(self) -> FaceDirection {
        RtSurfaceOwner::from_packed(self.packed_owner_direction)
            .expect("compact face record must contain a valid linear-packed owner")
            .face()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RtCompactPageGeometry {
    pub indices: Vec<u16>,
    pub faces: Vec<RtCompactFaceRecord>,
}

impl RtCompactPageGeometry {
    pub fn from_surface_mask(mask: &SurfaceMaskPage) -> Self {
        let face_count = mask.exposed_face_count() as usize;
        let mut geometry = Self {
            indices: Vec::with_capacity(face_count * 6),
            faces: Vec::with_capacity(face_count),
        };

        for morton_index in 0..512 {
            let (x, y, z) = morton::decode(morton_index);
            let owner = UVec3::new(x, y, z);
            for direction in FaceDirection::ALL {
                if !mask.is_exposed(owner, direction) {
                    continue;
                }
                let corners = face_corners(owner, direction);
                let lattice = corners.map(lattice_vertex_index);
                geometry.indices.extend_from_slice(&[
                    lattice[0], lattice[1], lattice[2], lattice[0], lattice[2], lattice[3],
                ]);
                geometry
                    .faces
                    .push(RtCompactFaceRecord::new(owner, direction));
            }
        }
        geometry
    }
}

pub fn lattice_vertex(index: u16) -> UVec3 {
    let index = usize::from(index);
    assert!(
        index < RT_PAGE_LATTICE_VERTEX_COUNT,
        "page lattice vertex index out of range: {index}"
    );
    let z = index / (RT_PAGE_LATTICE_EDGE * RT_PAGE_LATTICE_EDGE);
    let remainder = index % (RT_PAGE_LATTICE_EDGE * RT_PAGE_LATTICE_EDGE);
    let y = remainder / RT_PAGE_LATTICE_EDGE;
    let x = remainder % RT_PAGE_LATTICE_EDGE;
    UVec3::new(x as u32, y as u32, z as u32)
}

fn lattice_vertex_index(vertex: UVec3) -> u16 {
    debug_assert!(vertex.x <= 8 && vertex.y <= 8 && vertex.z <= 8);
    (vertex.x + vertex.y * 9 + vertex.z * 81) as u16
}

fn face_corners(owner: UVec3, direction: FaceDirection) -> [UVec3; 4] {
    let x = owner.x;
    let y = owner.y;
    let z = owner.z;
    match direction {
        FaceDirection::NegativeX => [
            UVec3::new(x, y, z),
            UVec3::new(x, y, z + 1),
            UVec3::new(x, y + 1, z + 1),
            UVec3::new(x, y + 1, z),
        ],
        FaceDirection::PositiveX => [
            UVec3::new(x + 1, y, z),
            UVec3::new(x + 1, y + 1, z),
            UVec3::new(x + 1, y + 1, z + 1),
            UVec3::new(x + 1, y, z + 1),
        ],
        FaceDirection::NegativeY => [
            UVec3::new(x, y, z),
            UVec3::new(x + 1, y, z),
            UVec3::new(x + 1, y, z + 1),
            UVec3::new(x, y, z + 1),
        ],
        FaceDirection::PositiveY => [
            UVec3::new(x, y + 1, z),
            UVec3::new(x, y + 1, z + 1),
            UVec3::new(x + 1, y + 1, z + 1),
            UVec3::new(x + 1, y + 1, z),
        ],
        FaceDirection::NegativeZ => [
            UVec3::new(x, y, z),
            UVec3::new(x, y + 1, z),
            UVec3::new(x + 1, y + 1, z),
            UVec3::new(x + 1, y, z),
        ],
        FaceDirection::PositiveZ => [
            UVec3::new(x, y, z + 1),
            UVec3::new(x + 1, y, z + 1),
            UVec3::new(x + 1, y + 1, z + 1),
            UVec3::new(x, y + 1, z + 1),
        ],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RtInterfaceAxis {
    X = 0,
    Y = 1,
    Z = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RtInterfaceSlot {
    pub axis: RtInterfaceAxis,
    /// Lattice plane 0 and 8 are owner boundaries; 1 through 7 are internal.
    pub plane: u8,
    pub u: u8,
    pub v: u8,
}

impl RtInterfaceSlot {
    pub fn from_index(index: usize) -> Option<Self> {
        if index >= RT_INTERFACE_QUAD_COUNT {
            return None;
        }
        let slots_per_axis = 9 * 8 * 8;
        let axis_index = index / slots_per_axis;
        let axis = match axis_index {
            0 => RtInterfaceAxis::X,
            1 => RtInterfaceAxis::Y,
            2 => RtInterfaceAxis::Z,
            _ => unreachable!(),
        };
        let axis_local = index % slots_per_axis;
        let plane = axis_local / 64;
        let cell = axis_local % 64;
        Some(Self {
            axis,
            plane: plane as u8,
            u: (cell % 8) as u8,
            v: (cell / 8) as u8,
        })
    }

    pub fn index(self) -> usize {
        self.axis as usize * 9 * 8 * 8
            + self.plane as usize * 8 * 8
            + self.v as usize * 8
            + self.u as usize
    }

    pub fn is_boundary(self) -> bool {
        self.plane == 0 || self.plane == 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::rt_surface_mask::{FaceDirection, SurfaceMaskPage};
    use crate::voxel::brick::VoxelCell;
    use crate::voxel::ucvh::{Ucvh, UcvhConfig};
    use glam::{UVec3, Vec3};

    const SOLID: VoxelCell = VoxelCell {
        material: 1,
        flags: 0,
        emissive: [0; 3],
        _pad: 0,
    };

    fn one_voxel_geometry(local: UVec3) -> RtCompactPageGeometry {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(8)));
        assert!(ucvh.set_voxel(local, SOLID));
        RtCompactPageGeometry::from_surface_mask(&SurfaceMaskPage::from_ucvh(&ucvh, UVec3::ZERO))
    }

    #[test]
    fn rt_page_geometry_uses_shared_729_vertex_lattice_and_compact_face_records() {
        let geometry = one_voxel_geometry(UVec3::new(3, 4, 5));

        assert_eq!(RT_PAGE_LATTICE_VERTEX_COUNT, 729);
        assert_eq!(std::mem::size_of::<RtCompactFaceRecord>(), 4);
        assert_eq!(geometry.faces.len(), 6);
        assert_eq!(geometry.indices.len(), 6 * 6);
        assert!(
            geometry
                .indices
                .iter()
                .all(|&index| usize::from(index) < RT_PAGE_LATTICE_VERTEX_COUNT)
        );
    }

    #[test]
    fn rt_page_geometry_face_records_round_trip_owner_and_direction() {
        let owner = UVec3::new(3, 4, 5);
        let geometry = one_voxel_geometry(owner);

        for direction in FaceDirection::ALL {
            let face = geometry
                .faces
                .iter()
                .find(|face| face.direction() == direction)
                .expect("single voxel should emit every direction");
            assert_eq!(face.owner_local(), owner);
        }
    }

    #[test]
    fn rt_page_geometry_emits_two_consecutive_triangles_per_face() {
        let geometry = one_voxel_geometry(UVec3::new(3, 4, 5));

        for face_index in 0..geometry.faces.len() {
            let indices = &geometry.indices[face_index * 6..face_index * 6 + 6];
            assert_eq!(indices[0], indices[3]);
            assert_eq!(indices[2], indices[4]);
            assert_ne!(indices[0], indices[1]);
            assert_ne!(indices[1], indices[2]);
        }
    }

    #[test]
    fn rt_page_geometry_winding_points_outward_for_all_directions() {
        let geometry = one_voxel_geometry(UVec3::new(3, 4, 5));

        for (face_index, face) in geometry.faces.iter().enumerate() {
            let indices = &geometry.indices[face_index * 6..face_index * 6 + 3];
            let a = lattice_vertex(indices[0]).as_vec3();
            let b = lattice_vertex(indices[1]).as_vec3();
            let c = lattice_vertex(indices[2]).as_vec3();
            let normal = (b - a).cross(c - a);
            let expected = face.direction().offset().as_vec3();
            assert!(
                normal.dot(expected) > 0.0,
                "face {:?} has inward winding: normal={normal:?}",
                face.direction()
            );
            assert_eq!(normal.length_squared(), Vec3::ONE.x);
        }
    }

    #[test]
    fn rt_page_geometry_omits_hidden_surfaces() {
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(8)));
        assert!(ucvh.set_voxel(UVec3::new(3, 3, 3), SOLID));
        assert!(ucvh.set_voxel(UVec3::new(4, 3, 3), SOLID));
        let geometry = RtCompactPageGeometry::from_surface_mask(&SurfaceMaskPage::from_ucvh(
            &ucvh,
            UVec3::ZERO,
        ));

        assert_eq!(geometry.faces.len(), 10);
        assert_eq!(geometry.indices.len(), 60);
        assert!(!geometry.faces.iter().any(|face| {
            (face.owner_local() == UVec3::new(3, 3, 3)
                && face.direction() == FaceDirection::PositiveX)
                || (face.owner_local() == UVec3::new(4, 3, 3)
                    && face.direction() == FaceDirection::NegativeX)
        }));
    }

    #[test]
    fn rt_page_geometry_empty_mask_has_no_indices_or_faces() {
        let ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(8)));
        let geometry = RtCompactPageGeometry::from_surface_mask(&SurfaceMaskPage::from_ucvh(
            &ucvh,
            UVec3::ZERO,
        ));

        assert!(geometry.indices.is_empty());
        assert!(geometry.faces.is_empty());
    }

    #[test]
    fn rt_page_geometry_interface_topology_constants_are_corrected() {
        assert_eq!(RT_INTERNAL_INTERFACE_QUAD_COUNT, 3 * 7 * 8 * 8);
        assert_eq!(RT_OWNER_BOUNDARY_QUAD_COUNT, 6 * 8 * 8);
        assert_eq!(RT_INTERFACE_QUAD_COUNT, 1_728);
        assert_eq!(RT_INTERFACE_TRIANGLE_COUNT, 3_456);
    }

    #[test]
    fn rt_page_geometry_interface_slot_mapping_round_trips_all_1728_quads() {
        let mut internal = 0;
        let mut boundary = 0;
        for index in 0..RT_INTERFACE_QUAD_COUNT {
            let slot = RtInterfaceSlot::from_index(index).expect("valid fixed interface slot");
            assert_eq!(slot.index(), index);
            assert!(slot.plane <= 8 && slot.u < 8 && slot.v < 8);
            if slot.is_boundary() {
                boundary += 1;
            } else {
                internal += 1;
            }
        }
        assert_eq!(internal, RT_INTERNAL_INTERFACE_QUAD_COUNT);
        assert_eq!(boundary, RT_OWNER_BOUNDARY_QUAD_COUNT);
        assert!(RtInterfaceSlot::from_index(RT_INTERFACE_QUAD_COUNT).is_none());
    }

    /// Cross-language boundary test: the bytes `RtCompactFaceRecord` writes to the GPU face
    /// buffer must be decoded identically by the Slang `decode_surface_owner()` function in
    /// `rt_page_common.slang`.
    ///
    /// Slang implementation (verbatim):
    /// ```
    /// SurfaceOwner decode_surface_owner(uint packed_owner) {
    ///     owner.local_voxel = uint3(packed_owner & 7u,
    ///                               (packed_owner >> 3u) & 7u,
    ///                               (packed_owner >> 6u) & 7u);
    ///     owner.face_direction = (packed_owner >> 9u) & 7u;
    /// }
    /// ```
    ///
    /// If the packing ever drifts this test catches it before GPU integration.
    #[test]
    fn compact_face_record_packing_matches_shader_decode_surface_owner() {
        // Mirror decode_surface_owner() from rt_page_common.slang in pure Rust.
        fn shader_decode(packed: u32) -> (UVec3, u32) {
            let local = UVec3::new(packed & 7, (packed >> 3) & 7, (packed >> 6) & 7);
            let face_index = (packed >> 9) & 7;
            (local, face_index)
        }

        // Exhaustively verify all 512 voxel positions × 6 face directions (3072 cases).
        for direction in FaceDirection::ALL {
            for z in 0u32..8 {
                for y in 0u32..8 {
                    for x in 0u32..8 {
                        let owner = UVec3::new(x, y, z);
                        let record = RtCompactFaceRecord::new(owner, direction);
                        let (decoded_local, decoded_face) =
                            shader_decode(record.packed_owner_direction);
                        assert_eq!(
                            decoded_local, owner,
                            "shader decode_surface_owner local mismatch at ({x},{y},{z}) dir={direction:?}"
                        );
                        assert_eq!(
                            decoded_face,
                            direction.index() as u32,
                            "shader decode_surface_owner face mismatch at ({x},{y},{z}) dir={direction:?}"
                        );
                    }
                }
            }
        }
    }
}
