# Vox Default Map Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MagicaVoxel `.vox` scene loading and make the app start from `run/Vintessa_Hills_static.vox`, falling back to a black/white checkerboard platform when that file is unavailable.

**Architecture:** Add a focused `src/voxel/vox_loader.rs` parser/UCVH writer, keep default scene selection in `src/voxel/generator.rs`, and change `src/app.rs` to call the new default scene path instead of the old Sponza generator. The old Sponza generator stays in the tree but is no longer used for default startup.

**Tech Stack:** Rust 2024, existing `glam` math types, existing `Ucvh`/`BrickData`/`VoxelCell` CPU voxel storage, standard-library file IO.

---

### Task 1: VOX Parser Tests

**Files:**
- Create: `src/voxel/vox_loader.rs`
- Modify: `src/voxel/mod.rs`

- [ ] **Step 1: Write failing tests** for parsing a minimal MagicaVoxel file with `SIZE`, `XYZI`, and `RGBA`, rejecting non-VOX input, and writing parsed voxels into a small UCVH.
- [ ] **Step 2: Run** `cargo test vox_loader --lib` and confirm failure because `vox_loader` does not exist.
- [ ] **Step 3: Implement** a small parser and writer into UCVH.
- [ ] **Step 4: Re-run** `cargo test vox_loader --lib` and confirm pass.

### Task 2: Default Scene Selection Tests

**Files:**
- Modify: `src/voxel/generator.rs`

- [ ] **Step 1: Write failing tests** proving missing `.vox` path returns checkerboard fallback and creates alternating black/white platform cells.
- [ ] **Step 2: Run** `cargo test default_scene --lib` and confirm failure.
- [ ] **Step 3: Implement** `generate_default_scene`, `generate_checkerboard_platform_scene`, and result metadata.
- [ ] **Step 4: Re-run** `cargo test default_scene --lib` and confirm pass.

### Task 3: Startup Integration

**Files:**
- Modify: `src/app.rs`

- [ ] Replace the startup Sponza call with `generator::generate_default_scene`.
- [ ] Log whether startup used the `.vox` file or the checkerboard fallback.
- [ ] Run `cargo test --lib` or a narrower verified subset if unrelated dirty-tree failures block full tests.
