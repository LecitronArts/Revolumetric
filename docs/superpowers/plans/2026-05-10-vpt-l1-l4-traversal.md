# VPT L1-L4 Traversal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make active VPT voxel traversal consume the uploaded UCVH L1-L4 hierarchy for empty-space skipping.

**Architecture:** Keep the existing L0 brick-grid DDA and brick-local occupancy DDA as the correctness base. Add a hierarchy-assisted skip helper that tests aligned empty L4/L3/L2/L1 blocks at the current brick coordinate and advances the DDA across the largest empty block before falling back to L0/brick testing. Bind L1-L4 buffers into VPT, VPT surface, and Area ReSTIR initial descriptors so all shader callers share the same traversal contract.

**Tech Stack:** Rust, Vulkan descriptors via ash, Slang compute shaders, existing Rust source tests.

---

### Task 1: Add Failing Tests For L1-L4 Traversal Contract

**Files:**
- Modify: `src/render/passes/vpt/shader_source_tests.rs`
- Modify: `src/render/passes/area_restir/shader_source_tests.rs`

- [ ] **Step 1: Update expected descriptor manifests**

Require `vpt.slang` to expose `hierarchy_l1` through `hierarchy_l4` between `hierarchy_l0` and brick data, and move ReSTIR/Area ReSTIR bindings after those buffers.

- [ ] **Step 2: Add traversal behavior source tests**

Add assertions that `voxel_traverse.slang` contains `try_skip_empty_hierarchy_block`, uses `StructuredBuffer<NodeLN>` for L1-L4, calls `node_ln_child_mask`, and calls the helper from both primary and any-hit traversal loops.

- [ ] **Step 3: Verify red**

Run: `cargo test render::passes::vpt::shader_source_tests --lib`

Expected: FAIL because shaders and Rust descriptor specs still only bind L0 and traversal lacks the helper.

---

### Task 2: Bind L1-L4 Buffers Through Rust And Slang ABI

**Files:**
- Modify: `src/render/passes/vpt.rs`
- Modify: `src/render/passes/vpt_surface.rs`
- Modify: `src/render/passes/area_restir.rs`
- Modify: `assets/shaders/passes/vpt.slang`
- Modify: `assets/shaders/passes/vpt_surface.slang`
- Modify: `assets/shaders/passes/area_restir_initial.slang`

- [ ] **Step 1: Extend descriptor specs and pools**

Increase VPT, surface, and Area ReSTIR initial descriptor arrays by four storage-buffer bindings. Increase storage-buffer pool counts accordingly.

- [ ] **Step 2: Write descriptor buffers**

Insert `&ucvh_gpu.hierarchy_ln_buffers[0..4]` after `hierarchy_l0_buffer` in all UCVH descriptor writes.

- [ ] **Step 3: Update Slang bindings**

Declare `StructuredBuffer<NodeLN> hierarchy_l1` through `hierarchy_l4` in all traversal-calling pass shaders. Shift later binding numbers consistently.

- [ ] **Step 4: Verify descriptor tests**

Run: `cargo test render::passes::vpt::shader_source_tests --lib`

Expected: descriptor manifest failures are resolved; traversal helper test still fails until Task 3.

---

### Task 3: Implement Hierarchy-Assisted Empty Block Skipping

**Files:**
- Modify: `assets/shaders/shared/voxel_traverse.slang`
- Modify call sites in `assets/shaders/passes/vpt.slang`, `assets/shaders/passes/vpt_surface.slang`, `assets/shaders/passes/area_restir_initial.slang`

- [ ] **Step 1: Add hierarchy helper functions**

Add `ucvh_flat_index`, `node_ln_has_child`, `hierarchy_level_empty_at_brick`, `try_skip_empty_hierarchy_block`, and `advance_brick_dda_to_t`.

- [ ] **Step 2: Update traversal signatures**

Extend `trace_primary_ray`, `trace_any_hit_ray_skip_voxel`, and `trace_any_hit_ray` with L1-L4 `StructuredBuffer<NodeLN>` parameters.

- [ ] **Step 3: Use helper in loops**

At the start of primary and any-hit brick loops, attempt to skip the largest aligned empty hierarchy block. Only fall back to L0 brick lookup if no empty block skip was possible.

- [ ] **Step 4: Verify green**

Run: `cargo test render::passes::vpt::shader_source_tests --lib`

Expected: PASS.

Run: `cargo test render::passes::area_restir::shader_source_tests --lib`

Expected: PASS.

---

### Task 4: Full Verification And Review

**Files:**
- No intentional extra files.

- [ ] **Step 1: Run focused tests**

Run:
- `cargo test render::passes::vpt::shader_source_tests --lib`
- `cargo test render::passes::area_restir::shader_source_tests --lib`
- `cargo test voxel::occupancy --lib`

- [ ] **Step 2: Run full library tests**

Run: `cargo test --lib`

Expected: PASS.

- [ ] **Step 3: Review diff**

Run: `git diff -- assets/shaders src/render docs/superpowers/plans/2026-05-10-vpt-l1-l4-traversal.md`

Check only intended ABI/shader/test/plan changes are present.
