// Radiance Cascade probe ray tracing.
// One invocation per (probe, face, direction). Traces a ray through UCVH,
// writes float4(weighted_radiance, ray_distance) to the probe buffer.

#include "voxel_traverse.slang"
#include "lighting_common.slang"
#include "radiance_cascade.slang"

// Push constants
struct RcTracePush {
    uint cascade_level;
    uint probe_grid_dim;
    uint probe_size;       // directions per face edge (3, 6, or 12)
    uint buffer_offset;
};
[[vk::push_constant]] RcTracePush push;

// Binding 0: Scene UBO
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

// Binding 1-4: UCVH buffers
[[vk::binding(1, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

[[vk::binding(2, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

[[vk::binding(3, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

[[vk::binding(4, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

// Binding 5: Probe buffer READ (previous frame)
[[vk::binding(5, 0)]]
StructuredBuffer<float4> probe_read;

// Binding 6: Probe buffer WRITE (current frame)
[[vk::binding(6, 0)]]
RWStructuredBuffer<float4> probe_write;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    uint grid_dim = push.probe_grid_dim;
    uint probe_size = push.probe_size;
    uint dirs_per_face = probe_size * probe_size;
    uint grid_dim_l0 = ucvh_config[0].brick_grid_size.x;

    uint total = grid_dim * grid_dim * grid_dim * 6u * dirs_per_face;
    uint tid = dtid.x;
    if (tid >= total) return;

    uint dir_index = tid % dirs_per_face;
    uint remainder = tid / dirs_per_face;
    uint face = remainder % 6u;
    uint probe_linear = remainder / 6u;
    uint3 probe_coord = uint3(
        probe_linear % grid_dim,
        (probe_linear / grid_dim) % grid_dim,
        probe_linear / (grid_dim * grid_dim)
    );

    float lod_factor = float(1u << push.cascade_level);
    float probe_spacing = 8.0 * lod_factor;
    float3 probe_pos = (float3(probe_coord) + 0.5) * probe_spacing;

    if (push.cascade_level == 0u) {
        float3 relocated_probe_pos;
        if (rc_find_nearest_empty_probe_position(probe_pos, 2.0, float3(19.0, 2.0, 7.0), float3(109.0, 87.0, 121.0),
                                                 hierarchy_l0, brick_occupancy, grid_dim_l0, relocated_probe_pos)) {
            probe_pos = relocated_probe_pos;
        }
    } else {
        probe_pos += rc_geo_offset(probe_pos, hierarchy_l0, brick_occupancy, grid_dim_l0);
    }

    // Flag C0 probes outside the building interior as inside-geo.
    // Only for C0 — higher cascades use GeoOffset and their larger spacing
    // means centers can be outside bounds while still covering interior regions.
    if (push.cascade_level == 0u) {
        if (probe_pos.x < 19.0 || probe_pos.x > 109.0 ||
            probe_pos.y < 2.0  || probe_pos.y > 87.0  ||
            probe_pos.z < 7.0  || probe_pos.z > 121.0) {
            uint idx = rc_probe_index(probe_coord, grid_dim, dirs_per_face, face, dir_index, push.buffer_offset);
            probe_write[idx] = float4(0.0, 0.0, 0.0, -1.0);
            return;
        }
    }

    float3 face_normal, face_tangent, face_bitangent;
    get_face_tbn(face, face_normal, face_tangent, face_bitangent);

    if (!rc_outside_geo(probe_pos, hierarchy_l0, brick_occupancy, grid_dim_l0)) {
        uint idx = rc_probe_index(probe_coord, grid_dim, dirs_per_face, face, dir_index, push.buffer_offset);
        probe_write[idx] = float4(0.0, 0.0, 0.0, -1.0);
        return;
    }

    float2 dir_uv = float2(float(dir_index % probe_size) + 0.5, float(dir_index / probe_size) + 0.5);
    float3 local_dir = ComputeDir(dir_uv, float(probe_size));
    float3 world_dir = face_tangent * local_dir.x + face_bitangent * local_dir.y + face_normal * local_dir.z;

    Ray ray = make_ray(probe_pos, world_dir);
    HitResult hit = trace_primary_ray(ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);

    float4 result;
    if (hit.hit) {
        result.w = hit.t;

        float3 emissive = voxel_emissive(hit.cell);
        if (emissive.x + emissive.y + emissive.z > 0.001) {
            result.xyz = emissive * 8.0;  // HDR boost for probe radiance
        } else {
            uint mat_id = voxel_material(hit.cell);
            float3 albedo = float3(0.7, 0.7, 0.7);
            if (mat_id == 1u) albedo = float3(0.65, 0.63, 0.60);
            else if (mat_id == 2u) albedo = float3(0.70, 0.20, 0.20);
            else if (mat_id == 3u) albedo = float3(0.20, 0.70, 0.20);
            else if (mat_id == 4u) albedo = float3(0.35, 0.42, 0.65);
            else if (mat_id == 5u) albedo = float3(0.55, 0.40, 0.30);
            else if (mat_id == 6u) albedo = float3(0.40, 0.37, 0.34);
            else if (mat_id == 7u) albedo = float3(0.45, 0.30, 0.18);

            float3 hit_normal = hit.normal;
            float ndotl = max(dot(hit_normal, scene.sun_direction), 0.0);
            float3 direct = float3(0.0);
            if (ndotl > 0.0) {
                float3 shadow_origin = hit.position + hit_normal * 0.51;
                Ray shadow_ray = make_ray(shadow_origin, scene.sun_direction);
                HitResult shadow_hit = trace_primary_ray(shadow_ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);
                float shadow = shadow_hit.hit ? 0.0 : 1.0;
                direct = scene.sun_intensity * ndotl * shadow;
            }

            float3 indirect = float3(0.0);
            if (scene.rc_enabled > 0u) {
                indirect = integrate_probe(hit.position, hit_normal, probe_read, scene.rc_c0_offset,
                                           uint3(scene.rc_c0_grid_x, scene.rc_c0_grid_y, scene.rc_c0_grid_z));
            }

            result.xyz = (direct + indirect) * albedo;
        }
    } else {
        result.w = 1e7;
        result.xyz = sky_color_for_dir(world_dir, scene);
    }

    float cos_theta = dot(world_dir, face_normal);
    if (push.cascade_level == 0u) {
        if (length(dir_uv - float2(1.5, 1.5)) < 0.75) {
            result.xyz *= (1.0 - cos(0.25 * RC_PI));
        } else {
            result.xyz *= cos(0.25 * RC_PI) / 8.0;
        }
    } else {
        float2 probeRel = dir_uv - float(probe_size) * 0.5;
        float probeThetai = max(abs(probeRel.x), abs(probeRel.y));
        float probeTheta = acos(cos_theta);
        result.xyz *= (cos(probeTheta - 0.5 * RC_PI / float(probe_size)) -
                       cos(probeTheta + 0.5 * RC_PI / float(probe_size))) /
                      (4.0 + 8.0 * floor(probeThetai));
    }
    result.xyz *= cos_theta;

    uint idx = rc_probe_index(probe_coord, grid_dim, dirs_per_face, face, dir_index, push.buffer_offset);
    probe_write[idx] = result;
}
