# Teardown Zip Static Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load a Teardown workshop zip directly into UCVH as static geometry.

**Architecture:** Add a focused Rust loader for zip/XML resource traversal and reuse the existing MagicaVoxel parser for referenced `.vox` files. Integrate it into default scene selection ahead of the intermediate `.vox` fallback.

**Tech Stack:** Rust 2024, `zip`, `roxmltree`, existing `glam`, `Ucvh`, `BrickData`, and `vox_loader`.

---

### Task 1: Loader RED Tests

**Files:**
- Create: `src/voxel/teardown_zip_loader.rs`
- Modify: `src/voxel/mod.rs`

- [ ] Add tests for loading a MOD `.vox` from a workshop zip into UCVH, expanding a BUILT-IN prefab instance, and rasterizing a `voxbox`.
- [ ] Run `cargo test teardown_zip_loader --lib` and confirm the tests fail before implementation.

### Task 2: Loader Implementation

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/voxel/vox_loader.rs`
- Modify: `src/voxel/teardown_zip_loader.rs`

- [ ] Add `zip` and `roxmltree`.
- [ ] Expose safe visible-voxel visitation and color-to-material helpers from `vox_loader`.
- [ ] Implement zip root detection, `MOD/...` and `BUILT-IN/...` resource reads, XML transform traversal, instance recursion guards, static geometry bounds pass, and brick-batched write pass.
- [ ] Run `cargo test teardown_zip_loader --lib`.

### Task 3: Default Scene Integration

**Files:**
- Modify: `src/voxel/generator.rs`
- Modify: `src/app.rs` if logging metadata needs a new scene kind label.

- [ ] Prefer `C:/Users/mc897/Downloads/Vintessa Hills.zip` when present.
- [ ] Fall back to `run/Vintessa_Hills_static.vox`, then checkerboard.
- [ ] Add unit tests for fallback ordering.
- [ ] Run targeted cargo tests and report any unrelated compile blockers.
