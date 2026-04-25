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

// Compute visibility weight for a higher-cascade neighbor probe.
// Reprojects direction from neighbor to current position onto the neighbor's
// dominant face, finds the nearest traced direction, and checks ray distance.
float neighbor_visibility(uint3 neighbor, float3 voxel_pos, float higher_lod_factor,
                          uint h_grid, uint h_dirs, float h_probe_size_f) {
    float3 neighbor_world = (float3(neighbor) + 0.5) * 8.0 * higher_lod_factor;
    float3 rPos = voxel_pos - neighbor_world;
    float dist = length(rPos);
    if (dist < 0.001) return 1.0;

    float3 aPos = abs(rPos);
    uint vis_face;
    float3 local_dir;

    // Determine dominant face and transform to local TBN space
    if (aPos.x > max(aPos.y, aPos.z)) {
        vis_face = (rPos.x > 0.0) ? 0u : 1u;   // +X or -X
        local_dir = float3(rPos.z, rPos.y, aPos.x);
    } else if (aPos.y > aPos.z) {
        vis_face = (rPos.y > 0.0) ? 2u : 3u;   // +Y or -Y
        local_dir = float3(rPos.x, rPos.z, aPos.y);
    } else {
        vis_face = (rPos.z > 0.0) ? 4u : 5u;   // +Z or -Z
        local_dir = float3(rPos.x, rPos.y, aPos.z);
    }

    // Project onto the face hemisphere to find the nearest direction
    float2 projUV = ProjectDir(local_dir, h_probe_size_f);
    if (projUV.x < 0.0) return 1.0; // dir behind hemisphere — assume visible

    uint dir_idx = min(uint(projUV.x + floor(projUV.y) * h_probe_size_f), h_dirs - 1u);
    uint vis_idx = rc_probe_index(neighbor, h_grid, h_dirs, vis_face, dir_idx, push.higher_offset);
    float ray_dist = probe_buffer[vis_idx].w;

    return (ray_dist < dist - 0.5) ? 0.00001 : 1.0;
}

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
    float h_probe_size_f = float(push.higher_probe_size);
    float3 h_pos = clamp(voxel_pos / (8.0 * higher_lod_factor), float3(0.5), float3(float(h_grid)) - 0.5);

    // Map own direction to the 2x2 block of higher-cascade directions
    float2 own_dir_uv = float2(float(dir_index % push.own_probe_size), float(dir_index / push.own_probe_size));
    float h_dir_index0 = floor(own_dir_uv.x) * 2.0 + floor(own_dir_uv.y) * h_probe_size_f * 2.0 + 0.5;

    float3 fPos = clamp(floor(h_pos - 0.5), float3(0.0), float3(float(h_grid) - 2.0));
    float3 frPos = min(float3(1.0), h_pos - 0.5 - fPos);

    // 4 direction offsets in the higher cascade (2x2 block)
    uint d0 = min(uint(h_dir_index0), h_dirs - 1u);
    uint d1 = min(uint(h_dir_index0 + 1.0), h_dirs - 1u);
    uint d2 = min(uint(h_dir_index0 + h_probe_size_f), h_dirs - 1u);
    uint d3 = min(uint(h_dir_index0 + h_probe_size_f + 1.0), h_dirs - 1u);

    float3 merged_radiance = float3(0.0);
    float merged_weight = 0.0;

    for (uint i = 0; i < 8; i++) {
        int3 offset = int3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        uint3 neighbor = uint3(int3(fPos) + offset);
        neighbor = clamp(neighbor, uint3(0), uint3(h_grid - 1));

        // Visibility: reproject face + direction from neighbor toward us
        float vis_w = neighbor_visibility(neighbor, voxel_pos, higher_lod_factor,
                                          h_grid, h_dirs, h_probe_size_f);

        // Sum the 4 matching directions from this neighbor
        uint base = rc_probe_index(neighbor, h_grid, h_dirs, face, 0u, push.higher_offset);
        float4 sum = probe_buffer[base + d0]
                   + probe_buffer[base + d1]
                   + probe_buffer[base + d2]
                   + probe_buffer[base + d3];

        float3 w = lerp(1.0 - frPos, frPos, float3(offset));
        float tri_w = w.x * w.y * w.z;
        float weight = tri_w * vis_w;

        // Inside-geo probes (w=-1) contribute zero radiance.
        // Reduce their weight so they don't dilute the result.
        // Matches reference NearestCS which returns weight=0.001 for inside-geo.
        float inside_w = (sum.w < -0.5) ? 0.001 : 1.0;
        merged_radiance += sum.xyz * inside_w * weight;
        merged_weight += inside_w * weight;
    }

    if (merged_weight > 0.0001) {
        merged_radiance /= merged_weight;
    }

    // Blend own (near-field) vs merged (far-field) based on ray distance.
    // Reference: (t - lod) / (lod * 2). Our probes are 8x sparser, but the
    // merge formula operates on ray distance in voxels, not probe spacings.
    // Match reference scaling: start at lod_factor, fully at 3 * lod_factor.
    // C0: transition 1-3 voxels. C1: transition 2-6 voxels.
    float dist_interp = clamp((own_data.w - lod_factor) / (lod_factor * 2.0), 0.0, 1.0);
    float3 final_radiance = lerp(own_data.xyz, merged_radiance, dist_interp);

    probe_buffer[own_idx] = float4(final_radiance, own_data.w);
}
