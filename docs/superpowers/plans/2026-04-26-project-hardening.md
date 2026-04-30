# Revolumetric Project Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the concrete engineering risks found in the project review without changing the renderer's feature scope.

**Architecture:** Keep fixes narrow and local. Add regression tests for behavior changes in CPU-side code, use command verification for build-script and documentation changes, and avoid broad refactors of the Vulkan runtime in this pass.

**Tech Stack:** Rust 2024, Cargo, Vulkan via `ash`, Slang shader compilation through `slangc`.

---

## Chunk 1: Safety And Build Hygiene

### Task 1: UCVH Bounds Safety

**Files:**
- Modify: `src/voxel/ucvh.rs`

- [x] **Step 1: Write failing tests**

Add tests that `set_voxel()` returns `false` for positions outside `world_size`, `get_voxel()` returns `VoxelCell::AIR`, and the allocated brick count does not change.

- [x] **Step 2: Verify red**

Run: `cargo test voxel::ucvh::tests::out_of_bounds`

Expected: tests fail or panic on current unchecked indexing.

- [x] **Step 3: Implement minimal bounds checks**

Add `contains_world_pos()` and use it at the start of `set_voxel()` and `get_voxel()`.

- [x] **Step 4: Verify green**

Run: `cargo test voxel::ucvh::tests::out_of_bounds`

Expected: tests pass.

### Task 2: RenderGraph Cycle Detection

**Files:**
- Modify: `src/render/graph.rs`

- [x] **Step 1: Write failing test**

Add a test that manually creates a dependency cycle in `PassDecl` data and verifies `compile()` reports it instead of silently producing a partial order.

- [x] **Step 2: Verify red**

Run: `cargo test render::graph::tests::compile_reports_cycles`

Expected: test fails because `compile()` currently has no error path.

- [x] **Step 3: Implement minimal error path**

Change `compile()` to return `anyhow::Result<()>` and error when `order.len() != pass_count`. Update call sites and existing tests.

- [x] **Step 4: Verify green**

Run: `cargo test render::graph::tests`

Expected: render graph tests pass.

### Task 3: Shader Build Script Robustness

**Files:**
- Modify: `build.rs`

- [x] **Step 1: Fix build-script lint issues**

Replace needless borrows and simplify extension checks.

- [x] **Step 2: Fix missing `slangc` fallback**

When `slangc` is missing, write placeholder `.spv` files for every pass shader, not only the first one before `break`.

- [x] **Step 3: Verify**

Run: `cargo clippy --all-targets -- -D warnings`

Expected: no clippy errors.

## Chunk 2: Project Usability

### Task 4: README And Repo Hygiene Notes

**Files:**
- Modify: `.gitignore`
- Create: `README.md`

- [x] **Step 1: Extend ignore rules**

Add common local IDE/editor and generated profiler output patterns without deleting already tracked files.

- [x] **Step 2: Add README**

Document purpose, prerequisites, build/test commands, runtime controls, and known prototype limits.

- [x] **Step 3: Verify docs and project commands**

Run: `cargo test`

Expected: all tests pass.

Run: `cargo clippy --all-targets -- -D warnings`

Expected: no clippy errors.

Run: `cargo build`

Expected: build completes and shader passes compile when `slangc` is available.
