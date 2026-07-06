# RT Backend Switch Resize Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent stale-size RT or VPT frame resources when switching renderer backend after a window or swapchain resize.

**Architecture:** Keep backend selection in `RenderRuntime`, keep GPU resource replacement in the resize path, and add small pipeline resource-presence helpers so inactive-but-existing pipelines can be resized without eagerly creating inactive passes.

**Tech Stack:** Rust 2024, existing render runtime source-contract tests, `cargo test --lib`, `cargo fmt`, `cargo clippy`.

---

## File Structure

- Modify `src/render/rt_pipeline.rs`: add `has_frame_resources()` and a source/behavior test.
- Modify `src/render/vpt_pipeline.rs`: add `has_frame_resources()` and a source/behavior test.
- Modify `src/render/runtime.rs`: route resize through RT/VPT helper methods and add source-contract tests.

## Task 1: Pipeline Resource Presence Helpers

- [ ] Add failing tests proving RT and VPT pipelines expose `has_frame_resources()`.
- [ ] Run the focused tests and confirm they fail.
- [ ] Implement `RtRuntimePipeline::has_frame_resources()` and `VptRuntimePipeline::has_frame_resources()`.
- [ ] Re-run the focused tests and confirm they pass.

## Task 2: Runtime Resize Routing

- [ ] Add failing runtime source-contract tests proving resize calls both backend-specific helpers and that each helper resizes when selected or already allocated.
- [ ] Add a test proving `render_frame` does not resize selected resources just because backend changed.
- [ ] Run focused runtime tests and confirm they fail.
- [ ] Implement `resize_rt_pipeline_to_swapchain()` and `resize_vpt_pipeline_to_swapchain()`.
- [ ] Update `resize_pipeline_to_swapchain()` to call both helpers after `ensure_passes`.
- [ ] Re-run focused runtime tests and confirm they pass.

## Task 3: Verification And Commit

- [ ] Run `cargo fmt --check`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings`.
- [ ] Run `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib`.
- [ ] Run `git diff --check` and inspect `git diff --stat`.
- [ ] Commit docs and implementation.
