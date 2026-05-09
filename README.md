# Revolumetric

Revolumetric is a Rust + Vulkan voxel rendering prototype. It builds a custom rendering stack around `ash`, `winit`, Slang compute shaders, a CPU/GPU Unified Cascaded Volume Hierarchy (UCVH), voxel path tracing, post-processing, and GPU timing instrumentation.

The project is currently an engine prototype, not a packaged application. The core code lives in `src/` and `assets/shaders/`; `reference/` is for external research material.

## Requirements

- Rust toolchain with edition 2024 support.
- Vulkan 1.3 capable driver.
- `slangc` on `PATH` for real shader compilation.

By default, if `slangc` is missing, the build script writes empty placeholder `.spv` files so Rust compilation can still proceed. Runtime render passes that require non-empty shaders will log warnings and skip initialization.

## Common Commands

```powershell
cargo test
$env:REVOLUMETRIC_SHADER_COMPILE = "skip"; cargo test; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE = "strict"; cargo test --lib; cargo build --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
cargo clippy --all-targets -- -D warnings
cargo build
cargo run
```

## Runtime And Build Config

Build-time shader compilation is controlled with `REVOLUMETRIC_SHADER_COMPILE`:

- `auto` or unset: compile shaders with `slangc` when available. If `slangc` is not found, write empty placeholder `.spv` files and emit a Cargo warning.
- `strict`: require `slangc`. Missing `slangc` or a shader compiler failure fails the build. Use this mode for CI and release validation so shader ABI and compiler errors cannot be hidden by placeholder SPIR-V.
- `skip`: do not invoke `slangc`; write empty placeholder `.spv` files. Use this only for CPU-only test environments.

Invalid values fail the build instead of silently falling back to a default.

Rendering settings can be overridden through environment variables:

- `REVOLUMETRIC_RENDER_MODE=vpt`: accepted for compatibility; VPT is the only active renderer. Other values emit a parse warning and keep the VPT default.
- `REVOLUMETRIC_VPT_MAX_BOUNCES=1..8`: bounds VPT path length. Default is `2`.
- `REVOLUMETRIC_EXPOSURE=<finite non-negative float>`: postprocess exposure multiplier before tonemap. Default is `1.0`.
- `REVOLUMETRIC_LIGHTING_SHADOWS=on|off|1|0|true|false`: enables direct-light shadow rays.
- `REVOLUMETRIC_SUN_ANGULAR_RADIUS=<finite float in 0.0..=0.25>`: analytic sun disk radius in radians for VPT soft shadow edges. Default is `0.02`.
- `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=on|off|1|0|true|false`: skips backface shadow hits when enabled.
- `REVOLUMETRIC_LIGHTING_DEBUG_VIEW=final|off|diffuse|direct|normal`: selects runtime lighting debug output.
- `REVOLUMETRIC_VPT_DEBUG_VIEW=final|raw|temporal|variance|history_valid|motion|normal|depth|reservoir_weight|direct|indirect|area_subpixel|area_lens|area_weight|area_history_valid|area_rejection|area_jacobian|voxel_brick|voxel_local|voxel_hit`: selects VPT diagnostics. Area ReSTIR and voxel traversal debug views are written through the final postprocess path without temporal smoothing.

Invalid rendering environment values emit parse warnings and keep the default for the invalid setting.

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

The active renderer is VPT-only. The former Radiance Cascades and voxel cone tracing paths have been removed from active runtime code and shaders; older planning documents may mention them only as migration history.

Default runtime flow:

1. UCVH upload/update.
2. VPT surface state pass writes current/previous surface attributes and motion history.
3. Optional ReSTIR-DI initial, temporal, and spatial passes build direct-light reservoirs.
4. Optional Area ReSTIR initial, temporal, and spatial passes build primary-ray sample-area reservoirs.
5. VPT traces bounded stochastic camera paths, optionally resolving ReSTIR-DI direct lighting and Area ReSTIR-selected primary-ray samples, then writes noisy HDR radiance and moments.
6. VPT temporal denoise reprojects and clamps radiance history, while raw/direct/Area ReSTIR debug views bypass temporal smoothing.
7. Postprocess applies exposure, ACES tonemap, and gamma, then writes LDR `rgba8`.
8. Blit copies postprocess output to the swapchain.

The current VPT path is still noisy and progressive. The active implementation plan is to replace simple progressive averaging with explicit VPT surface state, temporal reprojection, moments, and edge-aware denoising.

RenderGraph currently supports imported resources, explicit access declarations, dependency validation, graph-owned image and buffer barrier emission for the active pass chains. It does not yet own full transient allocation, descriptor automation, or async compute scheduling.

Validation matrix for this MVP:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
rg -n "REVOLUMETRIC_RENDER_MODE|REVOLUMETRIC_VPT_MAX_BOUNCES|REVOLUMETRIC_EXPOSURE|REVOLUMETRIC_VPT_RESTIR_DI" README.md docs/superpowers
```

## Current Shape

Implemented pieces include:

- Vulkan device, swapchain, descriptors, buffers, images, and compute pipeline helpers.
- A lightweight render graph for pass ordering.
- Render-graph access declarations and a single-queue barrier planning model.
- UCVH brick storage, occupancy hierarchy, dirty tracking, and GPU upload.
- Procedural demo scene generation.
- VPT rendering, surface state, temporal denoise, HDR output, explicit post-processing, and swapchain presentation.
- VPT-only ReSTIR-DI settings, direct-light table construction, shader skeletons, reservoir resources, and graph-gated pass wiring.
- VPT-owned Area ReSTIR settings, sample-area reservoir resources, temporal/spatial reuse shaders, primary-ray integration, and debug visualization routing.
- Unit tests for the CPU-side data structures and ABI-sensitive uniform layout.

Known prototype limits:

- `app.rs` still owns too much runtime orchestration.
- The render graph owns image access transitions, but it does not yet own real transient GPU resource allocation or descriptor automation.
- VPT temporal denoising and Area ReSTIR are still experimental and need representative scene captures before visual quality can be considered validated.
- Postprocess owns exposure/ACES/gamma, but bloom and richer display controls are not implemented.
- External asset import is not implemented.
- Some repository reference material is intentionally separate from the main engine code.
