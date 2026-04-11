#pragma once
// Lighting helper functions shared across passes.
// Includes: ACES tonemap, sky gradient, hemisphere ambient.

#include "scene_common.slang"

// ACES Filmic Tonemapping (Narkowicz 2015 approximation).
// Input: linear HDR color. Output: [0, 1] LDR color.
float3 aces_tonemap(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

// Sky color gradient based on ray direction.
// Interpolates between ground_color (horizon-down) and sky_color (zenith).
float3 sky_color_for_dir(float3 dir, SceneUniforms scene) {
    float t = dir.y * 0.5 + 0.5;
    return lerp(scene.ground_color, scene.sky_color, saturate(t));
}

// Hemisphere ambient: blend sky/ground based on surface normal Y component.
float3 hemisphere_ambient(float3 normal, float3 sky, float3 ground) {
    float t = normal.y * 0.5 + 0.5;
    return lerp(ground, sky, t);
}

// Normal ID lookup table — reconstructs float3 normal from 0-5 ID.
static const float3 NORMAL_TABLE[6] = {
    float3(1,0,0), float3(-1,0,0),
    float3(0,1,0), float3(0,-1,0),
    float3(0,0,1), float3(0,0,-1),
};
