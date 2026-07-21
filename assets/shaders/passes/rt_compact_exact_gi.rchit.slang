/// CompactExact triangle closest-hit shader for GI / secondary surface rays.
///
/// Mirrors rt_compact_exact_surface.rchit.slang but uses the rt_restir_gi descriptor
/// layout. brick_materials is at binding 16 (same as existing rt_restir_gi.rchit.slang).
/// Bindings 17 is already occupied in rt_restir_gi, so CompactExact page buffers go at:
///   18 = face_records  (RtCompactFaceRecord as uint)
///   19 = page_records  (GpuRtPageRecord)

#include "rt_page_common.slang"
#include "rt_surface_common.slang"
#include "material_common.slang"
#include "voxel_common.slang"

struct GpuRtPageRecord {
    uint3 page_coord;
    uint  brick_id;
};

[[vk::binding(16, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

[[vk::binding(18, 0)]]
StructuredBuffer<uint> face_records;

[[vk::binding(19, 0)]]
StructuredBuffer<GpuRtPageRecord> page_records;

float3 face_direction_to_normal(uint face_direction) {
    if (face_direction == 0u) return float3(-1.0,  0.0,  0.0);
    if (face_direction == 1u) return float3( 1.0,  0.0,  0.0);
    if (face_direction == 2u) return float3( 0.0, -1.0,  0.0);
    if (face_direction == 3u) return float3( 0.0,  1.0,  0.0);
    if (face_direction == 4u) return float3( 0.0,  0.0, -1.0);
    return                          float3( 0.0,  0.0,  1.0);
}

[shader("closesthit")]
void main(inout RtSurfacePayload payload,
          in BuiltInTriangleIntersectionAttributes attr)
{
    uint face_index  = PrimitiveIndex() / 2u;
    uint packed_owner = face_records[face_index];
    SurfaceOwner owner = decode_surface_owner(packed_owner);

    GpuRtPageRecord page = page_records[InstanceID()];
    uint voxel_index = morton_encode(owner.local_voxel);
    VoxelCell cell = brick_materials[page.brick_id * 512u + voxel_index];

    float3 obj_normal  = face_direction_to_normal(owner.face_direction);
    float3x4 obj_to_world = ObjectToWorld3x4();
    float3 world_normal = normalize(mul((float3x3)obj_to_world, obj_normal));

    float3 world_origin    = WorldRayOrigin();
    float3 world_direction = WorldRayDirection();
    float  hit_t           = RayTCurrent();

    payload.hit_kind          = RT_SURFACE_HIT_KIND_VOXEL;
    payload.hit_t             = hit_t;
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
