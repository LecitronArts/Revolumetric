# VPT Motion Guide Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire motion-class flags and brick-generation invalidation into VPT so temporal consumers can reject stale history with O(1) checks.

**Architecture:** Keep the existing `motion_history` delta ABI, add a shared motion-flag header, project UCVH brick generations into a GPU buffer indexed by `BrickId`, and extend the surface and temporal passes to write and consume the new signals. Use the current `BrickId` as the atlas slot because the codebase does not expose a separate slot lookup abstraction.

**Tech Stack:** Rust 2024, Slang shaders, Vulkan buffers/images, existing render graph and UCVH code.

---

### Task 1: Add ABI and behavior tests for motion flags and UCVH generation tracking

**Files:**
- Modify: `src/render/vpt_motion.rs`
- Modify: `src/voxel/ucvh.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`
- Modify: `src/voxel/gpu_upload.rs`

- [ ] Add failing tests that pin the new motion flag bit layout, legal combinations, generation sentinel skip, brick-generation snapshot upload, and shader binding names.
- [ ] Run the targeted tests and confirm they fail for the expected missing symbols and bindings.

### Task 2: Implement brick generation projection and motion-event upload

**Files:**
- Create: `src/render/brick_generation_atlas.rs`
- Modify: `src/voxel/ucvh.rs`
- Modify: `src/voxel/gpu_upload.rs`

- [ ] Add UCVH-side brick-generation storage, lookup helpers, and sentinel-safe generation minting.
- [ ] Add a CPU shadow + sparse upload path for the brick-generation buffer and capped motion-event buffer.
- [ ] Verify the new tests pass for generation initialization and sparse updates.

### Task 3: Extend VPT surface and temporal passes with motion flags

**Files:**
- Modify: `assets/shaders/shared/vpt_history_common.slang`
- Create: `assets/shaders/shared/vpt_motion_common.slang`
- Modify: `assets/shaders/passes/vpt_surface.slang`
- Modify: `assets/shaders/passes/vpt_temporal.slang`
- Modify: `assets/shaders/passes/restir_di_temporal.slang`
- Modify: `assets/shaders/passes/area_restir_temporal.slang`
- Modify: `src/render/vpt_history.rs`
- Modify: `src/render/passes/vpt_surface.rs`
- Modify: `src/render/passes/vpt_temporal.rs`
- Modify: `src/render/passes/restir_di.rs`
- Modify: `src/render/passes/area_restir.rs`
- Modify: `src/render/vpt_pipeline.rs`

- [ ] Add the new `motion_flags` and `surface_brick_generation` images and bind them through the surface pass.
- [ ] Make temporal consumers reject history on reset/disocclusion and generation mismatch.
- [ ] Re-run shader-source tests and the Rust pass-layout tests.

### Task 4: Full verification

**Files:**
- None

- [ ] Run `cargo fmt`.
- [ ] Run focused library tests, then the full library test suite, then the build/clippy checks required by the spec.
- [ ] Inspect `git diff --check` and final touched files.
