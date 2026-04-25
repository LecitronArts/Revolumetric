# GPU Profiling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add robust Vulkan GPU pass timing with debug-default logging and optional CSV export.

**Architecture:** Add a focused `GpuProfiler` module that owns timestamp query pools, environment configuration, readback aggregation, and CSV/log formatting. Integrate it into the existing frame lifecycle after `begin_frame` fence waits and before command submission, while exposing fine-grained RC trace/merge recording hooks for per-cascade timings.

**Tech Stack:** Rust 2024, ash Vulkan timestamp queries, `tracing`, existing render graph and compute pass wrappers.

---

## File Map

- Create: `src/render/gpu_profiler.rs` — profiler config, scope enum, query-pool owner, aggregation, CSV formatting, tests.
- Modify: `src/render/mod.rs` — export the new profiler module.
- Modify: `src/render/device.rs` — expose physical-device properties needed for timestamp period and queue timestamp support.
- Modify: `src/render/passes/radiance_cascade_trace.rs` — add per-cascade recording API while preserving existing `record` behavior.
- Modify: `src/render/passes/radiance_cascade_merge.rs` — add per-step recording API while preserving existing `record` behavior.
- Modify: `src/app.rs` — construct/destroy profiler, read/reset frame slots, wrap GPU command regions.
- Update: `docs/superpowers/specs/2026-04-25-gpu-profiling-design.md` if implementation reveals necessary design adjustments.

## Task 1: Profiler Pure Logic

**Files:**
- Create: `src/render/gpu_profiler.rs`
- Modify: `src/render/mod.rs`

- [ ] Write failing tests for config parsing from raw optional values.
- [ ] Write failing tests for scope count, log names, CSV header order, timestamp conversion, and rolling summary aggregation.
- [ ] Run `cargo test render::gpu_profiler` and verify the new tests fail because the module does not exist.
- [ ] Implement the non-Vulkan pure logic: `GpuProfilerConfig`, `GpuProfileScope`, frame timing structs, aggregation, CSV formatting.
- [ ] Run `cargo test render::gpu_profiler` and verify the pure logic tests pass.

## Task 2: Vulkan Query Owner

**Files:**
- Modify: `src/render/gpu_profiler.rs`
- Modify: `src/render/device.rs`

- [ ] Add `RenderDevice` accessors for `physical_device_properties()` and frame count, or equivalent data needed by profiler construction.
- [ ] Implement `GpuProfiler::new`, query-pool creation, safe unsupported-device disable path, frame-slot reset, begin/end timestamp writes, delayed readback, and `destroy`.
- [ ] Keep disabled profiler state out of the hot path by storing `Option<GpuProfiler>` in the app.
- [ ] Run focused tests and `cargo test`.

## Task 3: Fine-Grained RC Recording Hooks

**Files:**
- Modify: `src/render/passes/radiance_cascade_trace.rs`
- Modify: `src/render/passes/radiance_cascade_merge.rs`

- [ ] Add APIs that bind once and record individual cascade/merge dispatches.
- [ ] Preserve existing `record` methods by delegating to the new APIs.
- [ ] Add lightweight unit tests only if pure logic becomes testable without Vulkan; otherwise rely on compile tests.
- [ ] Run `cargo test` to verify existing behavior still compiles and tests pass.

## Task 4: App Integration

**Files:**
- Modify: `src/app.rs`

- [ ] Add a `gpu_profiler: Option<GpuProfiler>` field.
- [ ] Construct the profiler after `RenderDevice` exists.
- [ ] After `begin_frame`, read back and reset the current frame slot before recording pass commands.
- [ ] Wrap Primary Ray, RC Clear, RC Trace C0/C1/C2, RC Merge C2→C1/C1→C0, Lighting, and Blit scopes.
- [ ] Destroy profiler before dropping `RenderDevice`.
- [ ] Run `cargo test`.

## Task 5: Final Verification

**Files:**
- Potentially modify any touched files for warnings or formatting.

- [ ] Run `cargo test` and verify all tests pass.
- [ ] Run `cargo build` and check for new warnings introduced by the profiler work.
- [ ] If feasible, run `cargo run` manually to confirm `[GPU]` output and optional CSV generation.
- [ ] Summarize changes and note any manual Vulkan/window verification not performed.

Commits are intentionally omitted from this plan because this session has not been given permission to commit.
