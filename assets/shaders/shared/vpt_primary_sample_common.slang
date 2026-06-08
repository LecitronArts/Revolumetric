#pragma once

#include "scene_common.slang"
#include "area_restir_common.slang"

uint vpt_primary_hash_u32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float vpt_primary_rand01(inout uint state) {
    state = vpt_primary_hash_u32(state);
    return float(state & 0x00ffffffu) / 16777216.0;
}

uint vpt_primary_rng_seed(uint2 pixel, SceneUniforms scene) {
    // Keep sampling deterministic across wall-clock time; temporal state should be driven by the
    // VPT sample index and explicit history reset logic, not elapsed seconds.
    return vpt_primary_hash_u32(
        pixel.x * 1973u ^
        pixel.y * 9277u ^
        scene.vpt_sample_index * 26699u
    );
}

ScenePrimaryRay vpt_center_primary_ray(uint2 pixel, SceneUniforms scene) {
    return scene_primary_ray_from_area_sample(scene, float2(pixel), float2(0.5, 0.5));
}

ScenePrimaryRay vpt_fallback_primary_ray(uint2 pixel, SceneUniforms scene, inout uint rng_state) {
    float2 fallback_jitter = float2(vpt_primary_rand01(rng_state), vpt_primary_rand01(rng_state)) - 0.5;
    return scene_primary_ray_from_area_sample(scene, float2(pixel) + fallback_jitter, float2(0.5, 0.5));
}

ScenePrimaryRay vpt_primary_ray_from_area_reservoir(uint2 pixel, SceneUniforms scene, AreaRestirReservoir reservoir) {
    return scene_primary_ray_from_area_sample(
        scene,
        area_restir_pixel_sample(pixel, reservoir.sample_state),
        reservoir.sample_state.lens_uv
    );
}

ScenePrimaryRay vpt_resolve_area_restir_primary_ray(
    uint2 pixel,
    SceneUniforms scene,
    AreaRestirUniforms area,
    AreaRestirReservoir reservoir,
    uint index,
    inout uint rng_state
) {
    if (area.enabled != 0u && index < area.reservoir_count && area_restir_is_valid_reservoir(reservoir)) {
        return vpt_primary_ray_from_area_reservoir(pixel, scene, reservoir);
    }

    // fallback jitter: preserve stochastic VPT rays when Area ReSTIR is disabled or invalid.
    return vpt_fallback_primary_ray(pixel, scene, rng_state);
}

ScenePrimaryRay vpt_resolve_surface_primary_ray(
    uint2 pixel,
    SceneUniforms scene,
    AreaRestirUniforms area,
    AreaRestirReservoir reservoir,
    uint index
) {
    if (area.enabled != 0u && index < area.reservoir_count && area_restir_is_valid_reservoir(reservoir)) {
        return vpt_primary_ray_from_area_reservoir(pixel, scene, reservoir);
    }

    // Surface guide fallback stays on the pixel center so history guides do not shift every frame.
    return vpt_center_primary_ray(pixel, scene);
}
