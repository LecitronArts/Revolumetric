/// CompactExact triangle closest-hit shader for primary surface rays.
///
/// This shader is the triangle hit group (offset 1) counterpart of the procedural
/// reference DDA shader (offset 0).  It decodes the compact face record written by
/// RtCompactFaceRecord::new() — linear layout x|y<<3|z<<6|face<<9 — exactly as the
/// Slang decode_surface_owner() helper specifies.  Both encodings were verified to
/// agree by the cross-language unit test compact_face_record_packing_matches_shader_decode_surface_owner.
///
/// Descriptor bindings match rt_surface.rchit.slang (binding 11 = brick_materials),
/// with two additions:
///   14 = face_records  (RtCompactFaceRecord as uint, one per face quad pair)
///   15 = page_records  (GpuRtPageRecord: page_coord + brick_id)

#include "rt_page_common.slang"
#include "rt_surface_common.slang"
#include "material_common.slang"
#include "voxel_common.slang"

struct GpuRtPageRecord {
    uint3 page_coord;
    uint  brick_id;
};

[[vk::binding(11, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

/// Packed RtCompactFaceRecord values (one uint per face).  Two consecutive triangles
/// share a face record: face_index = PrimitiveIndex() / 2.
[[vk::binding(14, 0)]]
StructuredBuffer<uint> face_records;

/// GpuRtPageRecord table indexed by InstanceCustomIndex() (= RtPageSlot).
[[vk::binding(15, 0)]]
StructuredBuffer<GpuRtPageRecord> page_records;

float3 face_direction_to_normal(uint face_direction) {
    if (face_direction == 0u) return float3(-1.0,  0.0,  0.0); // NegativeX
    if (face_direction == 1u) return float3( 1.0,  0.0,  0.0); // PositiveX
    if (face_direction == 2u) return float3( 0.0, -1.0,  0.0); // NegativeY
    if (face_direction == 3u) return float3( 0.0,  1.0,  0.0); // PositiveY
    if (face_direction == 4u) return float3( 0.0,  0.0, -1.0); // NegativeZ
    return                          float3( 0.0,  0.0,  1.0);  // PositiveZ
}

[shader("closesthit")]
void main(inout RtSurfacePayload payload,
          in BuiltInTriangleIntersectionAttributes attr)
{
    // Two triangles represent one quad; they share the same face record.
    uint face_index = PrimitiveIndex() / 2u;
    uint packed_owner = face_records[face_index];
    SurfaceOwner owner = decode_surface_owner(packed_owner);

    GpuRtPageRecord page = page_records[InstanceID()];
    uint voxel_index = morton_encode(owner.local_voxel);
    VoxelCell cell = brick_materials[page.brick_id * 512u + voxel_index];

    // Transform the object-space outward normal to world space.
    // The instance transform is a pure translation (page-to-world), so the
    // rotation component is identity and the normal passes through unchanged.
    float3 obj_normal = face_direction_to_normal(owner.face_direction);
    float3x4 obj_to_world = ObjectToWorld3x4();
    float3 world_normal = normalize(mul((float3x3)obj_to_world, obj_normal));

    float3 world_origin    = WorldRayOrigin();
    float3 world_direction = WorldRayDirection();
    float  hit_t           = RayTCurrent();

    payload.hit_kind          = RT_SURFACE_HIT_KIND_VOXEL;
    payload.hit_t             = hit_t;
    payload.primitive_index   = PrimitiveIndex();
    payload.instance_id       = InstanceID();
    payload.ray_direction     = world_direction;
    payload.position          = world_origin + world_direction * hit_t;
    payload.normal            = world_normal;
    payload.albedo            = material_cell_albedo(cell);
    payload.roughness         = material_cell_roughness(cell);
    payload.material_id       = voxel_material(cell);
    payload.brick_id          = page.brick_id;
    payload.local             = owner.local_voxel;
    payload.emissive_radiance = material_emissive(cell) * 3.0;
}
