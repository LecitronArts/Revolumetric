# Revolumetric

Revolumetric is a Rust + Vulkan voxel rendering prototype. It builds a custom rendering stack around `ash`, `winit`, Slang compute shaders, a CPU/GPU Unified Cascaded Volume Hierarchy (UCVH), deferred lighting, Radiance Cascades experiments, and GPU timing instrumentation.

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

- `REVOLUMETRIC_LIGHTING_SHADOWS=on|off|1|0|true|false`
- `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=on|off|1|0|true|false`
- `REVOLUMETRIC_RC_NORMAL_STRATEGY=occupancy-gradient|gradient|axis|axis-normal`
- `REVOLUMETRIC_RC_PROBE_QUALITY=full`

GPU profiler behavior is configured in `src/render/gpu_profiler.rs`; CSV output is intended for profiling runs under `target/`.

## Current Shape

Implemented pieces include:

- Vulkan device, swapchain, descriptors, buffers, images, and compute pipeline helpers.
- A lightweight render graph for pass ordering.
- UCVH brick storage, occupancy hierarchy, dirty tracking, and GPU upload.
- Procedural demo scene generation.
- Primary voxel ray tracing, deferred lighting, and Radiance Cascades trace/merge passes.
- Unit tests for the CPU-side data structures and ABI-sensitive uniform layout.

Known prototype limits:

- `app.rs` still owns too much runtime orchestration.
- The render graph orders passes but does not yet own real transient GPU resource allocation.
- External asset import is not implemented.
- Some repository reference material is intentionally separate from the main engine code.
