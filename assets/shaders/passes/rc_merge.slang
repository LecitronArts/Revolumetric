// Radiance Cascade merge pass.
// Merges higher cascade data into lower cascade using visibility-weighted trilinear interpolation.
// Distance-based blend: near hits use own data, far hits blend with higher cascade.

#include "radiance_cascade.slang"

struct RcMergePush {
    uint cascade_level;      // level being merged INTO (0 or 1)
    uint own_grid_dim;       // 16 or 8
    uint own_probe_size;     // 3 or 6
    uint own_offset;         // buffer offset for this cascade
    uint higher_grid_dim;    // 8 or 4
    uint higher_probe_size;  // 6 or 12
    uint higher_offset;      // buffer offset for higher cascade
    uint _pad;
};
[[vk::push_constant]] RcMergePush push;

// Single probe buffer (read-write). Merge operates in-place.
[[vk::binding(0, 0)]]
RWStructuredBuffer<float4> probe_buffer;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint own_dirs = push.own_probe_size * push.own_probe_size;
    uint total = push.own_grid_dim * push.own_grid_dim * push.own_grid_dim * 6u * own_dirs;
    uint tid = dtid.x;
    if (tid >= total) return;

    uint dir_index = tid % own_dirs;
    uint remainder = tid / own_dirs;
    uint face = remainder % 6u;
    uint probe_linear = remainder / 6u;
    uint grid_dim = push.own_grid_dim;
    uint3 probe_coord = uint3(
        probe_linear % grid_dim,
        (probe_linear / grid_dim) % grid_dim,
        probe_linear / (grid_dim * grid_dim)
    );

    uint own_idx = rc_probe_index(probe_coord, grid_dim, own_dirs, face, dir_index, push.own_offset);
    float4 own_data = probe_buffer[own_idx];

    if (own_data.w < -0.5) return;

    float lod_factor = float(1u << push.cascade_level);
    float higher_lod_factor = lod_factor * 2.0;
    float probe_spacing = 8.0 * lod_factor;
    float3 voxel_pos = (float3(probe_coord) + 0.5) * probe_spacing;

    uint h_grid = push.higher_grid_dim;
    uint h_dirs = push.higher_probe_size * push.higher_probe_size;
    float3 h_pos = clamp(voxel_pos / (8.0 * higher_lod_factor), float3(0.5), float3(float(h_grid)) - 0.5);

    float2 own_dir_uv = float2(float(dir_index % push.own_probe_size), float(dir_index / push.own_probe_size));
    float h_dir_index0 = floor(own_dir_uv.x) * 2.0 + floor(own_dir_uv.y) * float(push.higher_probe_size) * 2.0 + 0.5;

    float3 fPos = clamp(floor(h_pos - 0.5), float3(0.0), float3(float(h_grid) - 2.0));
    float3 frPos = min(float3(1.0), h_pos - 0.5 - fPos);

    float h_pow4 = pow(4.0, float(push.cascade_level + 1u));
    float h_probe_size_f = float(push.higher_probe_size);
    float3 h_size = float3(float(h_grid));

    float4 lDIUV0 = float4(
        floor(fmod(h_dir_index0, h_pow4)) * h_size.x * h_size.y,
        floor(h_dir_index0 / h_pow4) * h_size.z,
        floor(fmod(h_dir_index0 + 1.0, h_pow4)) * h_size.x * h_size.y,
        floor((h_dir_index0 + 1.0) / h_pow4) * h_size.z
    );
    float4 lDIUV1 = float4(
        floor(fmod(h_dir_index0 + h_probe_size_f, h_pow4)) * h_size.x * h_size.y,
        floor((h_dir_index0 + h_probe_size_f) / h_pow4) * h_size.z,
        floor(fmod(h_dir_index0 + h_probe_size_f + 1.0, h_pow4)) * h_size.x * h_size.y,
        floor((h_dir_index0 + h_probe_size_f + 1.0) / h_pow4) * h_size.z
    );

    float3 merged_radiance = float3(0.0);
    float merged_weight = 0.0;

    for (uint i = 0; i < 8; i++) {
        int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        uint3 neighbor = uint3(int3(fPos) + offset);
        neighbor = clamp(neighbor, uint3(0), uint3(h_grid - 1));

        uint vis_dir_idx = uint(h_dir_index0);
        uint vis_idx = rc_probe_index(neighbor, h_grid, h_dirs, face, min(vis_dir_idx, h_dirs - 1u), push.higher_offset);
        float vis_ray_dist = probe_buffer[vis_idx].w;
        float3 neighbor_world = (float3(neighbor) + 0.5) * 8.0 * higher_lod_factor;
        float dist_to_us = length(voxel_pos - neighbor_world);
        float vis_w = (vis_ray_dist < dist_to_us - 0.5) ? 0.00001 : 1.0;

        float4 sum = float4(0.0);
        uint base = rc_probe_index(neighbor, h_grid, h_dirs, face, 0u, push.higher_offset);
        uint d0 = min(uint(h_dir_index0), h_dirs - 1u);
        uint d1 = min(uint(h_dir_index0 + 1.0), h_dirs - 1u);
        uint d2 = min(uint(h_dir_index0 + h_probe_size_f), h_dirs - 1u);
        uint d3 = min(uint(h_dir_index0 + h_probe_size_f + 1.0), h_dirs - 1u);
        sum += probe_buffer[base + d0];
        sum += probe_buffer[base + d1];
        sum += probe_buffer[base + d2];
        sum += probe_buffer[base + d3];

        float3 w = lerp(1.0 - frPos, frPos, float3(offset));
        float tri_w = w.x * w.y * w.z;
        float weight = tri_w * vis_w;

        if (sum.w < -0.5) {
            merged_radiance += sum.xyz * 0.001 * weight;
        } else {
            merged_radiance += sum.xyz * weight;
        }
        merged_weight += weight;
    }

    if (merged_weight > 0.0001) {
        merged_radiance /= merged_weight;
    }

    float inv_lod_factor = 1.0 / lod_factor;
    float dist_interp = clamp((own_data.w - lod_factor) * inv_lod_factor * 0.5, 0.0, 1.0);
    float3 final_radiance = lerp(own_data.xyz, merged_radiance, dist_interp);

    probe_buffer[own_idx] = float4(final_radiance, own_data.w);
}
