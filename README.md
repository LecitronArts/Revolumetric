# Revolumetric

Revolumetric is a Rust + Vulkan voxel rendering prototype. It builds a custom rendering stack around `ash`, `winit`, Slang compute shaders, a CPU/GPU Unified Cascaded Volume Hierarchy (UCVH), voxel path tracing, post-processing, and GPU timing instrumentation.

The project is currently an engine prototype, not a packaged application. The core code lives in `src/` and `assets/shaders/`; `reference/` is for external research material.

## Requirements

- Rust toolchain with edition 2024 support.
- Vulkan 1.3 capable driver.
- `slangc` on `PATH` for real shader compilation, or set
  `REVOLUMETRIC_SLANGC` to the absolute `slangc.exe` path.
- Optional native NRD validation requires an accepted NVIDIA NRD SDK checkout
  with headers and `NRD.lib`/`libNRD.a`. Static official NRD builds may also
  need `ShaderMakeBlob.lib` beside `NRD.lib`. Set `REVOLUMETRIC_NRD_ROOT`, or
  place the SDK under `run/nrd`. The native bridge now lives in
  `crates/revolumetric-nrd-sys`.

By default, if `slangc` is missing, the build script writes empty placeholder `.spv` files so Rust compilation can still proceed. Runtime render passes that require non-empty shaders will log warnings and skip initialization.

## Common Commands

```powershell
.\run\validate-local.ps1
.\run\validate-local.ps1 -StrictShaders
.\run\validate-local.ps1 -StrictShaders -Nrd
.\run\validate-visual-baseline.ps1
.\run\validate-visual-baseline.ps1 -Nrd
cargo run --features desktop --bin revolumetric
.\run\validate-nrd.ps1 -Denoiser reblur -Frames 3
```

## Runtime And Build Config

Build-time shader compilation is controlled with `REVOLUMETRIC_SHADER_COMPILE`:

- `auto` or unset: compile shaders with `slangc` when available. If `slangc` is not found, write empty placeholder `.spv` files and emit a Cargo warning.
- `strict`: require `slangc`. Missing `slangc` or a shader compiler failure fails the build. Use this mode for CI and release validation so shader ABI and compiler errors cannot be hidden by placeholder SPIR-V.
- `skip`: do not invoke `slangc`; write empty placeholder `.spv` files. Use this only for CPU-only test environments.
- `REVOLUMETRIC_SLANGC=<absolute path to slangc.exe>`: optional explicit
  compiler path for launch environments, such as IDEs, whose `PATH` does not
  include the Vulkan SDK `Bin` directory.

Invalid values fail the build instead of silently falling back to a default.

Rendering settings can be overridden through environment variables:

- `REVOLUMETRIC_RENDER_MODE=auto|vpt|rt`: selects automatic backend routing, the VPT renderer, or the hardware RT backend. `auto` is the default and uses RT on RT-capable devices, otherwise VPT. Explicit RT requests fall back to VPT when the device does not expose the required ray tracing support.
- `REVOLUMETRIC_VPT_MAX_BOUNCES=1..8`: bounds VPT path length. Default is `2`.
- `REVOLUMETRIC_EXPOSURE=<finite non-negative float>`: postprocess exposure multiplier before tonemap. Default is `1.0`.
- `REVOLUMETRIC_LIGHTING_SHADOWS=on|off|1|0|true|false`: enables direct-light shadow rays.
- `REVOLUMETRIC_SUN_ANGULAR_RADIUS=<finite positive float in (0.0, 0.25]>`: analytic sun disk radius in radians for VPT soft shadow edges. Default is `0.02`.
  The default sun intensity is interpreted as solar-disk radiance and the VPT direct-light estimator evaluates Lambertian `f * Li * cos / pdf`; changing the angular radius changes the sampled disk solid angle rather than applying a legacy directional-light brightness compensation.
- `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=on|off|1|0|true|false`: skips backface shadow hits when enabled.
- `REVOLUMETRIC_LIGHTING_DEBUG_VIEW=final|off|diffuse|direct|normal`: selects runtime lighting debug output.
- `REVOLUMETRIC_DENOISER=off|svgf|relax|reblur`: selects VPT denoising. `relax` and `reblur` use the native NRD path when the `nrd` Cargo feature and NRD SDK are available; otherwise they fall back to the existing SVGF path.
- `REVOLUMETRIC_DENOISER_ATROUS_ITERATIONS=0..5`: controls the fallback SVGF/A-trous iteration budget and related denoiser settings.
- `REVOLUMETRIC_VPT_DEBUG_VIEW=final|raw|temporal|variance|history_valid|motion|normal|depth|reservoir_weight|direct|indirect|area_subpixel|area_lens|area_weight|area_history_valid|area_rejection|area_jacobian|voxel_brick|voxel_local|voxel_hit|nrd_normal_roughness|nrd_viewz|nrd_motion|nrd_motion_z|nrd_validation`: selects VPT diagnostics. Area ReSTIR, voxel traversal, and NRD guide/debug views are written through the final postprocess path without temporal smoothing.
- `REVOLUMETRIC_VPT_TRAVERSAL_STATS=on|off|1|0|true|false`: enables per-frame VPT traversal counter readback. When enabled, the runtime waits for the submitted frame fence and logs a `TraversalStats` line with primary/shadow ray counts, hierarchy skip counts, and brick DDA step counts.

Invalid rendering environment values emit parse warnings and keep the default for the invalid setting.

Hardware RT settings are opt-in while the backend is being brought up:

- `REVOLUMETRIC_RT_RESTIR_DI=on|off|1|0|true|false`: enables RT ReSTIR-DI direct-light reservoirs. Default is `off`.
- `REVOLUMETRIC_RT_RESTIR_DI_SPATIAL=on|off|1|0|true|false`: enables RT ReSTIR-DI spatial reservoir reuse after temporal history is valid. Default is `off`.
- `REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES=0..8`: spatial neighbor sample count for RT ReSTIR-DI. Default is `4`.
- `REVOLUMETRIC_RT_RESTIR_GI=on|off|1|0|true|false`: enables RT ReSTIR-GI after RT surface generation. It traces a one-bounce hardware RT indirect sample, stores an indirect path-vertex reservoir with temporal reuse, and feeds the reservoir into the RT final lighting resolve. The RT path still does not enable NRD, SER, or path guiding. Default is `off`.
- `REVOLUMETRIC_RT_TEMPORAL_DENOISE=on|off|1|0|true|false`: enables the RT temporal-only accumulation path. Default is `on`.
- `REVOLUMETRIC_RT_TEMPORAL_HISTORY_LENGTH=1..64`: RT temporal and reservoir history budget. Default is `20`.
- `REVOLUMETRIC_RT_TEMPORAL_NORMAL_THRESHOLD=0.0..1.0`: normal compatibility threshold for RT temporal/spatial reuse. Default is `0.85`.
- `REVOLUMETRIC_RT_TEMPORAL_DEPTH_THRESHOLD=0.0..1.0`: depth/position compatibility threshold for RT temporal/spatial reuse. Default is `0.02`.
- `REVOLUMETRIC_RT_DEBUG_VIEW=off|final|surface|hit_distance|history_valid|direct_reservoir|indirect_reservoir|temporal`: selects RT diagnostics.

ReSTIR-DI is an experimental direct-light reuse layer and is disabled by default:

- `REVOLUMETRIC_VPT_RESTIR_DI=on|off|1|0|true|false`: enables the ReSTIR-DI pass chain. Default is `off`.
- `REVOLUMETRIC_RESTIR_DI_TEMPORAL=on|off|1|0|true|false`: enables temporal reservoir reuse when ReSTIR-DI is active. Default is `on`.
- `REVOLUMETRIC_RESTIR_DI_SPATIAL=on|off|1|0|true|false`: enables spatial reservoir reuse when ReSTIR-DI is active. Default is `off` while the spatial reuse stage is still being stabilized.
- `REVOLUMETRIC_RESTIR_DI_INITIAL_CANDIDATES=1..16`: candidate count for initial direct-light sampling. Default is `1`.
- `REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES=0..8`: spatial neighbor sample count. Default is `4`.
- `REVOLUMETRIC_RESTIR_DI_HISTORY_LENGTH=1..64`: temporal history length budget. Default is `20`.
- `REVOLUMETRIC_RESTIR_DI_DEBUG=off|reservoir_weight|light_id|visibility|temporal_valid|spatial_neighbors`: selects a future ReSTIR-DI debug view. Default is `off`.

Area ReSTIR is an experimental VPT sample-area reuse layer and is disabled by default. It is separate from ReSTIR-DI: Area ReSTIR chooses primary-ray film/lens sample state, while ReSTIR-DI chooses direct-light samples.

- `REVOLUMETRIC_AREA_RESTIR=on|off|1|0|true|false`: enables the Area ReSTIR pass chain. Default is `off`.
- `REVOLUMETRIC_AREA_RESTIR_TEMPORAL=on|off|1|0|true|false`: enables temporal sample-area reservoir reuse. Default is `on`.
- `REVOLUMETRIC_AREA_RESTIR_SPATIAL=on|off|1|0|true|false`: enables spatial sample-area reservoir reuse. Default is `on`.
- `REVOLUMETRIC_AREA_RESTIR_SUBPIXEL=on|off|1|0|true|false`: enables film/subpixel sample reuse. Default is `on`.
- `REVOLUMETRIC_AREA_RESTIR_LENS=on|off|1|0|true|false`: enables lens sample reuse. Default is `on`, but the default camera aperture is zero so pinhole behavior is preserved unless aperture is explicitly changed in code.
- `REVOLUMETRIC_AREA_RESTIR_INITIAL_CANDIDATES=1..16`: initial sample-area candidate count. Default is `4`.
- `REVOLUMETRIC_AREA_RESTIR_SPATIAL_SAMPLES=0..8`: spatial neighbor sample count. Default is `4`.
- `REVOLUMETRIC_AREA_RESTIR_HISTORY_LENGTH=1..64`: sample-area history length budget. Default is `20`.
- `REVOLUMETRIC_AREA_RESTIR_DEBUG=off|subpixel|lens|weight|history_valid|rejection|jacobian`: selects an Area ReSTIR debug view and bridges it into the VPT final display path.

GPU profiler behavior is configured in `src/render/gpu_profiler.rs`; CSV output is intended for profiling runs under `target/`. For comparable ReSTIR performance runs, prefer the checked-in profiling wrapper instead of hand-setting environment variables:

```powershell
.\tools\profile_restir_area.ps1 -Frames 120 -WarmupFrames 20 -Csv target\profile-restir-area.csv
```

The wrapper enables strict shader compilation, GPU CSV timing, ReSTIR-DI, ReSTIR-DI spatial reuse, and Area ReSTIR together. This matters because `REVOLUMETRIC_VPT_RESTIR_DI=on` does not enable `REVOLUMETRIC_RESTIR_DI_SPATIAL=on` by itself.

## Current Rendering Path

The default renderer selection is automatic: RT-capable devices use the hardware RT backend, and unsupported devices use VPT. `REVOLUMETRIC_RENDER_MODE=vpt` forces the VPT fallback path, while `REVOLUMETRIC_RENDER_MODE=rt` requires RT support and falls back to VPT with a warning when unavailable. The former Radiance Cascades and voxel cone tracing paths have been removed from active runtime code and shaders; older planning documents may mention them only as migration history.

Default VPT runtime flow:

1. UCVH upload/update.
2. VPT surface state pass writes current/previous surface attributes and motion history.
3. Optional ReSTIR-DI initial, temporal, and spatial passes build direct-light reservoirs.
4. Optional Area ReSTIR initial, temporal, and spatial passes build primary-ray sample-area reservoirs.
5. VPT traces bounded stochastic camera paths, optionally resolving ReSTIR-DI direct lighting and Area ReSTIR-selected primary-ray samples, then writes noisy HDR radiance and moments.
6. VPT temporal denoise reprojects and clamps radiance history, while raw/direct/Area ReSTIR debug views bypass temporal smoothing.
7. Postprocess applies exposure, ACES tonemap, and gamma, then writes LDR `rgba8`.
8. Blit copies postprocess output to the swapchain.

Opt-in hardware RT flow:

1. UCVH and RT acceleration structures are updated.
2. RT surface generation traces primary rays and writes surface attributes plus motion guides.
3. Optional RT ReSTIR-DI builds direct-light reservoirs with temporal and optional spatial reuse.
4. Optional RT ReSTIR-GI traces one-bounce indirect samples, stores path-vertex reservoirs, and applies temporal reuse.
5. RT direct lighting resolves albedo, ReSTIR-DI direct light, and ReSTIR-GI indirect reservoirs into HDR radiance.
6. RT temporal-only accumulation denoises the HDR radiance without using NRD.
7. RT resolve writes the display image, then the render graph blits it to the swapchain.

The current VPT path is still noisy and progressive. The active implementation plan is to replace simple progressive averaging with explicit VPT surface state, temporal reprojection, moments, and edge-aware denoising.

RenderGraph currently supports imported resources, explicit access declarations, dependency validation, graph-owned image and buffer barrier emission for the active pass chains, and graph-owned transient image and buffer allocation for graph-created resources. It does not yet own descriptor automation, resource aliasing, or async compute scheduling.

Validation matrix for this MVP:

```powershell
.\run\validate-local.ps1
.\run\validate-local.ps1 -StrictShaders
.\run\validate-local.ps1 -StrictShaders -Nrd
.\run\validate-visual-baseline.ps1
.\run\validate-visual-baseline.ps1 -Nrd
.\run\validate-visual-baseline.ps1 -Rt
.\run\validate-nrd.ps1 -Denoiser reblur -Frames 3
$env:REVOLUMETRIC_RENDER_MODE='rt'
$env:REVOLUMETRIC_RT_RESTIR_DI='on'
$env:REVOLUMETRIC_RT_RESTIR_DI_SPATIAL='on'
$env:REVOLUMETRIC_RT_RESTIR_GI='on'
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='2'
cargo run --features desktop --bin revolumetric
Remove-Item Env:\REVOLUMETRIC_RENDER_MODE
Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_DI
Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_DI_SPATIAL
Remove-Item Env:\REVOLUMETRIC_RT_RESTIR_GI
Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='2'
cargo run --features desktop --bin revolumetric # default auto backend smoke
Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
rg -n "REVOLUMETRIC_RENDER_MODE|REVOLUMETRIC_RT_RESTIR_DI|REVOLUMETRIC_RT_RESTIR_GI|REVOLUMETRIC_RT_TEMPORAL|REVOLUMETRIC_RT_DEBUG_VIEW|REVOLUMETRIC_VPT_RESTIR_DI" README.md docs/superpowers
```

The visual regression baseline wrapper captures deterministic smoke frames and
checks the PPM/metadata contract. It is a baseline health gate, not a final image
quality judgment. The default visual baseline run pins cases to VPT for
cross-machine stability; `-Nrd` adds NRD-backed VPT captures, and `-Rt` runs the
hardware RT capture case on RT-capable hardware. Capture metadata records
`render_backend`, `render_mode`, and RT controls such as `rt_debug_view` so
fallback output and hardware RT resolve output can be distinguished.

## Current Shape

Implemented pieces include:

- Vulkan device, swapchain, descriptors, buffers, images, and compute pipeline helpers.
- A lightweight render graph for pass ordering.
- Render-graph access declarations and a single-queue barrier planning model.
- UCVH brick storage, occupancy hierarchy, dirty tracking, and GPU upload.
- Procedural demo scene generation.
- VPT rendering, surface state, temporal denoise, HDR output, explicit post-processing, and swapchain presentation.
- Hardware RT scene AABB caching with dirty-region resampling, RT surface generation, RT ReSTIR-DI, RT ReSTIR-GI, RT final lighting resolve, temporal-only accumulation, and swapchain presentation.
- VPT ReSTIR-DI settings, direct-light table construction, shader skeletons, reservoir resources, and graph-gated pass wiring.
- VPT-owned Area ReSTIR settings, sample-area reservoir resources, temporal/spatial reuse shaders, primary-ray integration, and debug visualization routing.
- Unit tests for the CPU-side data structures and ABI-sensitive uniform layout.

Known prototype limits:

- `app.rs` still owns too much runtime orchestration.
- RT acceleration structures resample only dirty UCVH brick regions after the initial scan, update/refit existing BLAS/TLAS resources when the AABB primitive and instance counts still match, and fall back to full resource rebuilds when the AS shape is no longer update-compatible.
- The render graph owns image access transitions and non-aliased graph-owned transient image and buffer allocation. Descriptor automation and resource aliasing are not implemented yet.
- VPT temporal denoising and Area ReSTIR are still experimental and need representative scene captures before visual quality can be considered validated.
- Postprocess owns exposure/ACES/gamma, but bloom and richer display controls are not implemented.
- External asset import is not implemented.
- Some repository reference material is intentionally separate from the main engine code.
