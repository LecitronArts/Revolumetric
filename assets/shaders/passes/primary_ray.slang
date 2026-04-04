// Primary ray compute shader — screen-space voxel ray tracing.
// Traces one ray per pixel through the UCVH brick grid.
// Output: flat-shaded normal color on hit, sky gradient on miss.

#include "voxel_traverse.slang"

struct PrimaryRayPC {
    float4x4 pixel_to_ray; // col 0-2: direction matrix, col 3: camera origin
    uint2 resolution;
    uint2 _pad;
};

[[vk::push_constant]]
PrimaryRayPC pc;

// Binding 0: output image
[[vk::binding(0, 0)]]
RWTexture2D<float4> output_image;

// Binding 1: UCVH config (single element)
[[vk::binding(1, 0)]]
StructuredBuffer<UcvhConfig> ucvh_config;

// Binding 2: brick occupancy pool
[[vk::binding(2, 0)]]
StructuredBuffer<BrickOccupancy> brick_occupancy;

// Binding 3: brick material pool
[[vk::binding(3, 0)]]
StructuredBuffer<VoxelCell> brick_materials;

// Binding 4: hierarchy L0
[[vk::binding(4, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;

// Bindings 5-8: hierarchy L1-L4 (reserved for future hierarchy skip optimization)
// Not used in the basic two-level DDA, but bound for forward compatibility.
[[vk::binding(5, 0)]] StructuredBuffer<NodeLN> hierarchy_l1;
[[vk::binding(6, 0)]] StructuredBuffer<NodeLN> hierarchy_l2;
[[vk::binding(7, 0)]] StructuredBuffer<NodeLN> hierarchy_l3;
[[vk::binding(8, 0)]] StructuredBuffer<NodeLN> hierarchy_l4;

// Sky gradient (simple procedural sky)
float3 sky_color(float3 dir) {
    float t = dir.y * 0.5 + 0.5; // -1..1 → 0..1
    float3 horizon = float3(0.7, 0.8, 1.0);
    float3 zenith  = float3(0.2, 0.4, 0.9);
    return lerp(horizon, zenith, saturate(t));
}

// Normal to flat color: maps face normal to visible RGB
float3 normal_color(float3 n, float3 position) {
    // Base color from normal direction
    float3 base = abs(n) * 0.5 + 0.3;
    // Simple directional lighting (sun from upper-right)
    float3 sun_dir = normalize(float3(0.5, 0.8, 0.3));
    float ndotl = max(dot(n, sun_dir), 0.0);
    float ambient = 0.3;
    return base * (ambient + (1.0 - ambient) * ndotl);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.resolution.x || tid.y >= pc.resolution.y) return;

    // Generate ray from pixel_to_ray matrix.
    // Rust side stores column-major via glam Mat4::from_cols + to_cols_array_2d.
    // In memory: [col0, col1, col2, col3]. Slang float4x4 reads m[i] as i-th
    // group of 4 floats = Rust column i.
    // origin = m[3].xyz = Rust col 3 = camera position.
    // direction = col0*px + col1*py + col2*1 = mul(vec, mat) in Slang
    //   because mul(v, M) computes v[0]*row0 + v[1]*row1 + v[2]*row2
    //   and our "rows" are actually Rust columns.
    float2 pixel = float2(tid.xy);
    float3 origin = pc.pixel_to_ray[3].xyz;
    float3x3 dir_mat = float3x3(
        pc.pixel_to_ray[0].xyz,
        pc.pixel_to_ray[1].xyz,
        pc.pixel_to_ray[2].xyz
    );
    float3 dir = normalize(mul(float3(pixel, 1.0), dir_mat));

    Ray ray = make_ray(origin, dir);
    HitResult hit = trace_primary_ray(ray, ucvh_config, hierarchy_l0, brick_occupancy, brick_materials);

    float3 color;
    if (hit.hit) {
        color = normal_color(hit.normal, hit.position);
    } else {
        color = sky_color(dir);
    }

    output_image[tid.xy] = float4(color, 1.0);
}
