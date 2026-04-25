# Lighting Optimization Framework Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sustainable, high-quality-default lighting optimization controls and make avoidable backface shadow tracing explicitly measurable before enabling it by default.

**Architecture:** Reuse existing Scene UBO padding for explicit lighting flags and strategies, expose typed Rust settings, and route shader behavior through named constants/helpers. Validate with unit tests and P1 GPU profiler CSV before/after comparison.

**Tech Stack:** Rust 2024, Slang compute shaders, Vulkan UBO layout, existing P1 GPU profiler CSV output.

---

## File Map

- Modify: `src/render/scene_ubo.rs` - add lighting settings fields, typed settings, and layout tests.
- Modify: `assets/shaders/shared/scene_common.slang` - mirror UBO fields and constants.
- Modify: `assets/shaders/passes/lighting.slang` - use flags/strategy helpers and skip backface shadow traces.
- Modify: `src/app.rs` - populate default high-quality lighting settings.
- Modify: `src/render/passes/lighting.rs`, `src/render/passes/radiance_cascade_trace.rs`, `src/render/passes/primary_ray.rs` only if descriptor range constants are needed.
- Add/update: docs if implementation differs from spec.

## Task 1: Scene UBO Settings API

**Files:**
- Modify: `src/render/scene_ubo.rs`

- [ ] Add RED tests for `GpuSceneUniforms` remaining 176 bytes.
- [ ] Add RED tests for default `LightingSettings` flags and strategy values.
- [ ] Add RED tests for `GpuSceneUniforms::with_lighting_settings` or equivalent conversion.
- [ ] Implement `LightingSettings`, `RcNormalStrategy`, `RcProbeQuality`, and constants.
- [ ] Replace `_pad4` with named fields while preserving total size.
- [ ] Run `cargo test render::scene_ubo` and verify green.

## Task 2: Shader UBO and Constants

**Files:**
- Modify: `assets/shaders/shared/scene_common.slang`
- Modify: `src/render/scene_ubo.rs`

- [ ] Mirror Rust field names in Slang: `lighting_flags`, `rc_normal_strategy`, `rc_probe_quality`.
- [ ] Add Slang constants for lighting flags and strategy values.
- [ ] Update comments to say total size remains 176 bytes.
- [ ] Run `cargo build` to verify Slang compilation succeeds.

## Task 3: Lighting Shader Strategy Helpers

**Files:**
- Modify: `assets/shaders/passes/lighting.slang`

- [ ] Extract `compute_occupancy_gradient_normal` helper from current inline code.
- [ ] Add `select_rc_normal` helper that chooses DDA normal or occupancy-gradient normal by strategy.
- [ ] Compute `ndotl` before shadow tracing.
- [ ] Skip shadow tracing when shadows are disabled or when `ndotl <= 0` and backface skip is explicitly enabled.
- [ ] Preserve current default visual path by setting defaults to shadows enabled, backface skip disabled, occupancy-gradient normals, full probe quality.
- [ ] Run `cargo build` to verify shader compilation.

## Task 4: App Integration

**Files:**
- Modify: `src/app.rs`

- [ ] Import `LightingSettings`.
- [ ] Populate scene UBO with cached `LightingSettings::from_env_report()` so profiling runs can switch strategies without recompiling while invalid values warn once.
- [ ] Keep RC enabled behavior unchanged.
- [ ] Run `cargo test`.

## Task 5: Performance Verification

**Files:**
- No code changes expected.

- [ ] Run `cargo test`.
- [ ] Run `cargo build`.
- [ ] Run `REVOLUMETRIC_GPU_PROFILE_CSV=target/gpu-profile-lighting-opt.csv cargo run`.
- [ ] Run `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=on REVOLUMETRIC_GPU_PROFILE_CSV=target/gpu-profile-lighting-skip-on.csv cargo run`.
- [ ] Compare `target/gpu-profile.csv` with `target/gpu-profile-lighting-opt.csv` for `lighting_ms` and `total_ms`.
- [ ] Compare default skip-off vs explicit skip-on output from the same binary for a cleaner A/B measurement.
- [ ] Report whether the optimization improved, regressed, or stayed neutral.

Commits are intentionally omitted from this plan because this session has not been given permission to commit.
