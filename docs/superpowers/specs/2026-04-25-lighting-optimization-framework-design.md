# Lighting Optimization Framework - Design Specification

## Goal

Build a sustainable lighting optimization framework that keeps high-quality lighting as the default while making expensive lighting decisions explicit, configurable, testable, and measurable through the P1 GPU profiler.

## Baseline

The P1 profiler run recorded `4502` frames in `target/gpu-profile.csv`:

- Total GPU average: `2.8147ms`
- Lighting average: `0.9301ms` (`33.0%` of total)
- RC Trace C0/C1 average: `1.0308ms` combined (`36.7%` of total)
- Primary Ray average: `0.4653ms` (`16.5%` of total)

Lighting is the largest single pass and currently performs avoidable work per hit pixel.

## Current Hotspots

`assets/shaders/passes/lighting.slang` currently does this for every non-emissive hit pixel:

1. Trace a shadow ray before computing whether the surface faces the sun.
2. Compute smooth RC normals through six occupancy-neighbor queries.
3. Integrate RC probes through the full C0 trilinear multi-face path.

The first item is an unconditional waste on back-facing pixels. The second and third items are quality-sensitive and should become explicit strategy choices rather than hard-coded shader behavior.

## Requirements

- Do not reduce default visual quality.
- Do not increase `SceneUniforms` size in this phase.
- Keep Rust and Slang UBO layouts byte-for-byte synchronized.
- Add named Rust settings instead of scattering magic `u32` values through `app.rs`.
- Add shader constants for flags and strategy values.
- Use profiler CSV before/after data to validate performance impact.
- Keep future extension points for runtime toggles and quality tiers.

## Runtime Controls

The framework exposes environment variables for reproducible profiling runs:

- `REVOLUMETRIC_LIGHTING_SHADOWS=on|off|1|0|true|false`
- `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=on|off|1|0|true|false`
- `REVOLUMETRIC_RC_NORMAL_STRATEGY=occupancy-gradient|gradient|axis|axis-normal`
- `REVOLUMETRIC_RC_PROBE_QUALITY=full`

Defaults preserve the previous high-quality path: shadows on, backface shadow skip off, occupancy-gradient RC normals, full probe quality. Backface skip remains available for profiling, but the A/B runs in this phase were too noisy to justify enabling it by default.

## UBO Design

Reuse the existing 12 bytes of padding after `rc_enabled` in `GpuSceneUniforms`:

```text
rc_enabled: u32
lighting_flags: u32
rc_normal_strategy: u32
rc_probe_quality: u32
```

The struct remains `176` bytes. Descriptor ranges that currently use `176` remain valid.

### Lighting Flags

- `LIGHTING_FLAG_SHADOWS_ENABLED = 1 << 0`
- `LIGHTING_FLAG_SKIP_BACKFACE_SHADOWS = 1 << 1`

Default settings enable shadows but leave backface shadow skipping disabled to preserve the previous shader behavior. The skip flag remains available for explicit profiling runs.

### RC Normal Strategy

- `0 = AxisNormal`: use the DDA face normal from the G-buffer.
- `1 = OccupancyGradient`: compute the current six-neighbor smooth normal.

Default is `OccupancyGradient` to preserve existing visual quality.

### RC Probe Quality

Add the field now but keep behavior unchanged in this phase:

- `0 = Full`: current full C0 trilinear multi-face integration.

Future phases can add lower-cost modes without changing the UBO ABI again.

## Shader Behavior

Direct lighting computes `ndotl` before shadow tracing:

```text
ndotl = max(dot(normal, sun_direction), 0)
if shadows enabled and (ndotl > 0 or backface-skip disabled): trace shadow ray
else: shadow = 1
diffuse = base_color * sun_intensity * ndotl * shadow
```

This preserves output because pixels with `ndotl == 0` contribute no direct diffuse light regardless of shadow state.

RC normal selection becomes a helper:

```text
normal_for_rc = normal
if rc_normal_strategy == OccupancyGradient:
    normal_for_rc = occupancy_gradient_or_axis_normal(...)
```

Default output should match the prior quality path exactly for shadow tracing: backface shadow skipping is opt-in until same-binary profiler data shows a stable win.

## Testing Strategy

Automated tests:

- `GpuSceneUniforms` remains exactly 176 bytes.
- Default `LightingSettings` enables shadows and leaves backface skip disabled.
- Default `LightingSettings` uses `OccupancyGradient`.
- Uniform conversion populates `lighting_flags`, `rc_normal_strategy`, and `rc_probe_quality` deterministically.

Verification commands:

- `cargo test`
- `cargo build`
- `REVOLUMETRIC_GPU_PROFILE_CSV=target/gpu-profile-lighting-opt.csv cargo run`

Performance comparison:

- Compare `target/gpu-profile.csv` baseline with `target/gpu-profile-lighting-opt.csv`.
- Report average and max for `lighting_ms` and `total_ms`.
- For same-binary A/B, compare default `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=off` behavior against an explicit `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=on` run.

## Non-Goals

- No in-app UI toggle in this phase.
- No runtime config file in this phase.
- No change to RC trace pass quality or probe layout.
- No visual regression automation yet.
