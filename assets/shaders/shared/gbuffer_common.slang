#pragma once
// Shared G-buffer packing helpers.

static const float GBUFFER_SURFACE_FLAG_NONE = 0.0;
static const float GBUFFER_SURFACE_FLAG_VOXEL_BACKED = 1.0;

float3 gbuffer_encode_normal(float3 normal) {
    float len2 = dot(normal, normal);
    float3 n = (len2 > 0.0001) ? normal * rsqrt(len2) : float3(0.0, 1.0, 0.0);
    return saturate(n * 0.5 + 0.5);
}

float3 gbuffer_decode_normal(float3 encoded_normal) {
    float3 normal = encoded_normal * 2.0 - 1.0;
    return (dot(normal, normal) > 0.0001) ? normalize(normal) : float3(0.0, 1.0, 0.0);
}
