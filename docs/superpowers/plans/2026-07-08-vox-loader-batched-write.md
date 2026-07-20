# VOX Loader Batched Write Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default VOX-to-UCVH write path's per-voxel `set_voxel` loop with brick-batched writes that preserve existing behavior and reduce load time.

**Architecture:** Keep VOX parsing and scene graph traversal unchanged. Convert transformed source voxels into target `BrickData` batches keyed by brick coordinate, then submit each completed brick through `Ucvh::write_brick`; when a target brick already exists, seed the batch from existing UCVH cells so this remains an additive write path instead of clearing unrelated cells.

**Tech Stack:** Rust 2024, existing `glam`, `HashMap`, `Ucvh`, `BrickData`, and cargo tests.

---

### Task 1: Behavior Tests

**Files:**
- Modify: `src/voxel/vox_loader.rs`

- [ ] Add a regression test proving `write_scene_to_ucvh` preserves pre-existing cells in a brick that also receives VOX cells.
- [ ] Add a regression test proving duplicate target voxels still count one unique voxel and keep the last material according to scene traversal order.
- [ ] Run `cargo test vox_loader --lib`.

### Task 2: Batched Writer

**Files:**
- Modify: `src/voxel/vox_loader.rs`

- [ ] Add helpers to compute target cells once, group them by brick position, seed touched bricks from existing UCVH cells, and write each touched brick with `Ucvh::write_brick`.
- [ ] Preserve `VoxWriteStats` semantics: `input_voxels`, `written_voxels`, `out_of_bounds_voxels`, `unique_written_voxels`, `source_bounds`, and `target_bounds`.
- [ ] Run `cargo test vox_loader --lib`.

### Task 3: Verification

**Files:**
- Modify only files touched by Tasks 1 and 2.

- [ ] Run `cargo test vox_loader --lib`.
- [ ] Run ignored local load test with `cargo test local_default_vintessa_vox_file_loads_when_present --lib -- --ignored --nocapture`.
- [ ] Compare elapsed time against the current hot baseline of about 13.1 seconds measured in this workspace.
