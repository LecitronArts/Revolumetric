# Revolumetric

Revolumetric is a Rust + Vulkan voxel rendering prototype. It builds a custom rendering stack around `ash`, `winit`, Slang compute shaders, a CPU/GPU Unified Cascaded Volume Hierarchy (UCVH), voxel ray tracing, VCT/VPT rendering experiments, post-processing, and GPU timing instrumentation.

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

Lighting settings can be overridden through environment variables:

- `REVOLUMETRIC_RENDER_MODE=vct|vpt`: selects the default VCT lighting path or optional VPT reference path. Default is `vct`.
- `REVOLUMETRIC_VCT=on|off|1|0|true|false`: enables or disables VCT indirect contribution. Default is enabled.
- `REVOLUMETRIC_VPT_MAX_BOUNCES=1..8`: bounds VPT path length. Default is `2`.
- `REVOLUMETRIC_EXPOSURE=<finite non-negative float>`: postprocess exposure multiplier before tonemap. Default is `1.0`.
- `REVOLUMETRIC_LIGHTING_SHADOWS=on|off|1|0|true|false`: enables direct-light shadow rays.
- `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=on|off|1|0|true|false`: skips backface shadow hits when enabled.
- `REVOLUMETRIC_LIGHTING_DEBUG_VIEW=final|off|diffuse|direct|normal`: selects runtime lighting debug output.

Invalid lighting environment values emit parse warnings and keep the default for the invalid setting.

ReSTIR-DI is an experimental VPT-only path and is disabled by default:

- `REVOLUMETRIC_VPT_RESTIR_DI=on|off|1|0|true|false`: enables the ReSTIR-DI pass chain only when `REVOLUMETRIC_RENDER_MODE=vpt`. Default is `off`.
- `REVOLUMETRIC_RESTIR_DI_TEMPORAL=on|off|1|0|true|false`: enables temporal reservoir reuse when ReSTIR-DI is active. Default is `on`.
- `REVOLUMETRIC_RESTIR_DI_SPATIAL=on|off|1|0|true|false`: enables spatial reservoir reuse when ReSTIR-DI is active. Default is `on`.
- `REVOLUMETRIC_RESTIR_DI_INITIAL_CANDIDATES=1..16`: candidate count for initial direct-light sampling. Default is `1`.
- `REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES=0..8`: spatial neighbor sample count. Default is `4`.
- `REVOLUMETRIC_RESTIR_DI_HISTORY_LENGTH=1..64`: temporal history length budget. Default is `20`.
- `REVOLUMETRIC_RESTIR_DI_DEBUG=off|reservoir_weight|light_id|visibility|temporal_valid|spatial_neighbors`: selects a future ReSTIR-DI debug view. Default is `off`.

GPU profiler behavior is configured in `src/render/gpu_profiler.rs`; CSV output is intended for profiling runs under `target/`.

## Current Rendering MVP

The active renderer is VCT-first. The former Radiance Cascades path has been removed from runtime code and shaders; new documentation may mention it only as migration history or deletion boundary.

Default runtime flow:

1. UCVH upload/update.
2. Primary voxel ray pass writes G-buffer images.
3. VCT lighting reads the G-buffer and UCVH resources, then writes linear HDR `rgba16f`.
4. Postprocess applies exposure, ACES tonemap, and gamma, then writes LDR `rgba8`.
5. Blit copies postprocess output to the swapchain.

Optional VPT mode uses `REVOLUMETRIC_RENDER_MODE=vpt` to run a bounded, stochastic, progressive reference pass directly from camera rays. Its HDR accumulation image feeds the same postprocess pass. VPT is intended for reference/debug use and is not denoised. ReSTIR-DI can be enabled only inside this VPT mode; its current pass chain builds direct-light/reservoir buffers but VPT reservoir resolve is still future work.

RenderGraph currently supports imported resources, explicit access declarations, dependency validation, graph-owned image and buffer barrier emission for the active pass chains. It does not yet own full transient allocation, descriptor automation, or async compute scheduling.

Validation matrix for this MVP:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
rg -n "REVOLUMETRIC_RENDER_MODE|REVOLUMETRIC_VCT|REVOLUMETRIC_VPT_MAX_BOUNCES|REVOLUMETRIC_EXPOSURE|REVOLUMETRIC_VPT_RESTIR_DI" README.md docs/superpowers
```

## Current Shape

Implemented pieces include:

- Vulkan device, swapchain, descriptors, buffers, images, and compute pipeline helpers.
- A lightweight render graph for pass ordering.
- Render-graph access declarations and a single-queue barrier planning model.
- UCVH brick storage, occupancy hierarchy, dirty tracking, and GPU upload.
- Procedural demo scene generation.
- Primary voxel ray tracing, VCT-first deferred lighting, optional VPT reference mode, HDR lighting output, and explicit post-processing.
- VPT-only ReSTIR-DI settings, direct-light table construction, shader skeletons, reservoir resources, and graph-gated pass wiring.
- Unit tests for the CPU-side data structures and ABI-sensitive uniform layout.

Known prototype limits:

- `app.rs` still owns too much runtime orchestration.
- The render graph owns image access transitions, but it does not yet own real transient GPU resource allocation or descriptor automation.
- VCT is an MVP shader path over existing UCVH traversal, not a production radiance-cache implementation.
- VPT is a noisy progressive reference/debug path and currently has no denoiser.
- Postprocess owns exposure/ACES/gamma, but bloom and richer display controls are not implemented.
- External asset import is not implemented.
- Some repository reference material is intentionally separate from the main engine code.
