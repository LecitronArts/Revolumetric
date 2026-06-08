#pragma once
// Lighting helper functions shared across passes.
// Includes: ACES tonemap, sky gradient, hemisphere ambient.

#include "scene_common.slang"

static const float LIGHTING_INV_PI = 0.31830988618;
static const float LIGHTING_TWO_PI = 6.28318530718;

float sun_disk_solid_angle(SceneUniforms scene) {
    float sun_radius = scene.sun_angular_radius;
    if (sun_radius <= 0.0) {
        return 0.0;
    }

    return max(LIGHTING_TWO_PI * (1.0 - cos(sun_radius)), 1.0e-8);
}

float sun_direction_pdf(SceneUniforms scene) {
    float solid_angle = sun_disk_solid_angle(scene);
    return solid_angle > 0.0 ? 1.0 / solid_angle : 0.0;
}

// ACES Filmic Tonemapping (Narkowicz 2015 approximation).
// Input: linear HDR color. Output: [0, 1] LDR color.
float3 aces_tonemap(float3 x) {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

// Sky color gradient based on ray direction.
// Interpolates between sunlit ground radiance (horizon-down) and sky_color (zenith).
// Models outdoor ground illuminated by sun — provides warm indirect through windows.
// Uses finite solar-disk irradiance; do not treat sun_intensity as directional irradiance.
float3 sky_color_for_dir(float3 dir, SceneUniforms scene) {
    float t = saturate(dir.y * 0.5 + 0.5);
    float ground_ndotl = max(scene.sun_direction.y, 0.0);
    float3 finite_sun_irradiance = scene.sun_intensity * ground_ndotl * sun_disk_solid_angle(scene);
    float3 sunlit_ground = scene.ground_color * (1.0 + LIGHTING_INV_PI * finite_sun_irradiance);
    return lerp(sunlit_ground, scene.sky_color, t);
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
