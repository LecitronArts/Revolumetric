# Render Graph Resource Declarations Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RenderGraph` retain declared transient resource metadata and reject reads of resources that no pass creates or writes.

**Architecture:** Keep this as a pure graph-layer improvement. `PassBuilder` records resource descriptors during setup, `RenderGraph` stores them by resource id, and `compile()` validates graph references before topological sorting.

**Tech Stack:** Rust 2024, Cargo tests, existing `ash::vk` resource flag types.

---

## Chunk 1: Resource Metadata

### Task 1: Record Resource Descriptions

**Files:**
- Modify: `src/render/resource.rs`
- Modify: `src/render/pass_context.rs`
- Modify: `src/render/graph.rs`

- [x] **Step 1: Write failing tests**

Add tests in `src/render/graph.rs` that `create_image()` and `create_buffer()` descriptors can be queried from `RenderGraph`.

- [x] **Step 2: Verify red**

Run: `cargo test render::graph::tests::resource_descriptions`

Expected: tests fail because descriptors are not stored and `create_buffer()` does not exist.

- [x] **Step 3: Implement descriptor storage**

Add descriptor collection to `PassBuilder`, store descriptors in `RenderGraph`, and expose `resource_desc(handle)`.

- [x] **Step 4: Verify green**

Run: `cargo test render::graph::tests::resource_descriptions`

Expected: tests pass.

### Task 2: Validate Unknown Resource References

**Files:**
- Modify: `src/render/graph.rs`

- [x] **Step 1: Write failing tests**

Add tests that `compile()` returns an error when a pass reads an unknown resource or writes a new version of an unknown resource id.

- [x] **Step 2: Verify red**

Run: `cargo test render::graph::tests::compile_rejects_unknown`

Expected: tests fail because compile currently accepts unknown handles.

- [x] **Step 3: Implement validation**

Before dependency sorting, validate that every read id and versioned write id has a known resource declaration.

- [x] **Step 4: Verify green**

Run: `cargo test render::graph::tests`

Expected: all render graph tests pass.

## Chunk 2: Verification

### Task 3: Final Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-04-26-render-graph-resource-declarations.md`

- [x] **Step 1: Format**

Run: `cargo fmt`

Expected: no errors.

- [x] **Step 2: Full tests**

Run: `cargo test`

Expected: all tests pass.

- [x] **Step 3: Lint**

Run: `cargo clippy --all-targets -- -D warnings`

Expected: no clippy errors.

- [x] **Step 4: Build**

Run: `cargo build`

Expected: build completes.
