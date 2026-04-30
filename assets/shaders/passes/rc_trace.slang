// Radiance Cascade probe ray tracing.
// One invocation per (probe, face, direction). Traces a ray through UCVH,
// writes float4(weighted_radiance, ray_distance) to the probe buffer.

#include "voxel_traverse.slang"
#include "material_common.slang"
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

static const float RC_TEMPORAL_BLEND = 0.18;

bool rc_probe_history_is_uninitialized(float4 previous) {
    return abs(previous.w) < 0.00001 && dot(abs(previous.xyz), float3(1.0)) < 0.00001;
}

float4 rc_temporal_blend(float4 current, float4 previous) {
    if (current.w < -0.5) return current;
    if (previous.w < -0.5) return current;
    if (rc_probe_history_is_uninitialized(previous)) return current;
    return float4(lerp(previous.xyz, current.xyz, RC_TEMPORAL_BLEND), current.w);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    SceneUniforms scene = scene_ubo;
    uint grid_dim = push.probe_grid_dim;
    uint probe_size = push.probe_size;
    uint dirs_per_face = probe_size * probe_size;

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

    uint3 brick_grid_size = ucvh_config[0].brick_grid_size.xyz;
    uint3 world_size = ucvh_config[0].world_size.xyz;
    float3 probe_pos = rc_probe_world_position(probe_coord, push.cascade_level);

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

    if (!rc_outside_geo(probe_pos, hierarchy_l0, brick_occupancy, brick_grid_size, world_size)) {
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

        float3 emissive = material_emissive(hit.cell);
        if (material_has_emissive(emissive)) {
            result.xyz = emissive * 8.0;  // HDR boost for probe radiance
        } else {
            float3 albedo = material_cell_albedo(hit.cell);

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
    probe_write[idx] = rc_temporal_blend(result, probe_read[idx]);
}
