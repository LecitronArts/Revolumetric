static const uint RT_REFERENCE_HIT_GROUP_ID = 0;
static const uint RT_COMPACT_EXACT_HIT_GROUP_ID = 1;

struct SurfaceKey {
    uint3 page_coord;
    uint page_slot;
    uint brick_id;
    uint owner_local_and_face;
    uint material_generation;
    uint reserved;
};

struct SurfaceOwner {
    uint3 local_voxel;
    uint face_direction;
};

SurfaceOwner decode_surface_owner(uint packed_owner) {
    SurfaceOwner owner;
    owner.local_voxel = uint3(
        packed_owner & 7u,
        (packed_owner >> 3u) & 7u,
        (packed_owner >> 6u) & 7u
    );
    owner.face_direction = (packed_owner >> 9u) & 7u;
    return owner;
}
