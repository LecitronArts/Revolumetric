# Superseded: Fixed-Face Dynamic Page Phase 0-1 Plan

> **Status:** Superseded on 2026-07-10 by
> `2026-07-10-surface-mask-rt-core-phase0-2.md`.

The upload-safety work in Tasks 1-3 remains relevant and is carried into the
replacement plan. Do not implement the 3,072-quad/6,144-triangle hot-page
topology from this historical plan. The selected design now uses a directional
surface-mask contract, compact exact BLAS builds as the KHR baseline, and a
benchmark-gated 27-plane OMM hot backend.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make UCVH edits safely visible to a future page-triangle RT backend,
then prove the exact `8^3` hot-page topology and exposed-face rules without
building an RT scene yet.

**Architecture:** UCVH owns data. A frame-slot staging ring uploads only changed
occupancy/material/generation/hierarchy ranges and acknowledges a durable batch
only after recording successful upload work. A pure CPU page oracle maps each
of 3,072 owner faces to a deterministic quad and marks it exposed only when its
solid owner sees air across the correct local or cross-page neighbor.

**Tech Stack:** Rust 2024, ash Vulkan, glam, bytemuck, existing lib/source
contract tests.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/voxel/ucvh.rs` | Durable change batches and page-neighbor invalidation. |
| `src/voxel/gpu_upload.rs` | Per-frame staging resources and incremental authority upload. |
| `src/render/runtime.rs` | Snapshot/upload/ack orchestration and safe resource recreation. |
| `src/render/rt_page.rs` | Pure page topology, exposed-face oracle, source stamps, and tests. |
| `src/render/mod.rs` | Exposes the page module to later RT scene code. |

## Task 1: Correct Batch Page Invalidation

**Files:**
- Modify: `src/voxel/ucvh.rs`
- Test: `src/voxel/ucvh.rs`

- [ ] **Step 1: Write the failing page-neighbor test**

```rust
#[test]
fn render_change_batch_invalidates_changed_page_and_six_face_neighbors() {
    let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::new(32, 32, 32)));
    assert!(ucvh.set_voxel(UVec3::new(9, 9, 9), VoxelCell::new(1, 0, [0; 3])));

    let batch = ucvh.snapshot_render_change_batch();

    assert_eq!(batch.invalidated_pages, vec![
        UVec3::new(1, 1, 0), UVec3::new(1, 0, 1), UVec3::new(0, 1, 1),
        UVec3::new(1, 1, 1), UVec3::new(2, 1, 1), UVec3::new(1, 2, 1),
        UVec3::new(1, 1, 2),
    ]);
}
```

- [ ] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render_change_batch_invalidates_changed_page_and_six_face_neighbors -- --nocapture
```

Expected: compile failure because `invalidated_pages` is not part of the batch.

- [ ] **Step 3: Add page invalidation without removing existing temporal data**

Add `invalidated_pages: Vec<UVec3>` to `UcvhRenderChangeBatch`. Derive it from
each changed brick coordinate and the six in-bounds face offsets. Keep existing
temporal invalidation regions and the temporary `invalidated_render_cells` data
until their callers have been migrated; do not reinterpret a 16-cube cell as a
page.

- [ ] **Step 4: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib ucvh::tests -- --nocapture
```

Expected: all UCVH tests pass.

## Task 2: Frame-Slot UCVH Staging Ring

**Files:**
- Modify: `src/voxel/gpu_upload.rs`
- Modify: `src/render/runtime.rs`
- Test: `src/voxel/gpu_upload.rs`
- Test: `src/render/runtime.rs`

- [ ] **Step 1: Write failing source and pure-layout tests**

Add a test that requires the uploader to own a `Vec<UcvhStagingResources>` and
that `staging_for_frame_slot(1)` returns a different resource set from slot zero.
Add a source contract that rejects `if frame_slot != 0` in
`upload_incremental_changes`.

- [ ] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib incremental_upload -- --nocapture
```

Expected: the old uploader still rejects all slots other than zero.

- [ ] **Step 3: Implement `UcvhStagingResources`**

Move the six host-visible staging buffers into:

```rust
struct UcvhStagingResources {
    occupancy: GpuBuffer,
    material: GpuBuffer,
    hierarchy: GpuBuffer,
    config: GpuBuffer,
    brick_generations: GpuBuffer,
    motion_events: GpuBuffer,
}
```

Change `UcvhGpuResources::new` to receive `frame_slot_count: usize`, reject
zero, allocate one complete staging set per slot, and select it through a checked
`staging_for_frame_slot(frame_slot)` helper. Update both runtime creation sites
to pass `renderer.frame_slot_count()`.

- [ ] **Step 4: Make all upload paths slot-aware**

Pass `frame_slot` to incremental, initial, and motion-guide uploads. Use only
that slot's buffers for CPU writes and transfer copies. Do not write the full
generation prefix from `upload_motion_guide`; incremental authority upload owns
generation freshness. Preserve the existing transfer-to-compute/RT/AS barrier.

- [ ] **Step 5: Integrate durable incremental acknowledgement**

After initial upload completes, call `snapshot_render_change_batch`, record
`upload_incremental_changes`, and call `ack_render_change_batch(batch.id)` only
when it returns `Ok`. An error logs and preserves the snapshot. Temporal motion
events remain separately consumed only after their upload succeeds.

- [ ] **Step 6: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib incremental_upload -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib runtime::tests -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo check --lib
```

Expected: no slot-zero restriction, runtime source contracts pass, and the
library compiles.

## Task 3: Capacity Growth Recovery Contract

**Files:**
- Modify: `src/voxel/gpu_upload.rs`
- Modify: `src/render/runtime.rs`
- Test: `src/render/runtime.rs`

- [ ] **Step 1: Write a failing runtime source contract**

Require an incremental-upload capacity error to select a recovery path rather
than merely log the error and clear frame changes. The contract must contain
`recreate_ucvh_gpu_resources` and reset initial upload progress.

- [ ] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib runtime::tests::incremental -- --nocapture
```

Expected: source contract fails because there is no explicit recovery path.

- [ ] **Step 3: Implement safe recreation**

When capacity is exceeded, wait all frame fences that can read current UCVH
descriptors, destroy the old `UcvhGpuResources`, recreate it with the current
frame-slot count, reset `UcvhInitialUploadProgress`, set `ucvh_uploaded = false`,
and leave the durable change batch pending. Descriptor-owning passes must be
rebound through their existing per-frame update methods before tracing resumes.

- [ ] **Step 4: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib runtime::tests -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo check --lib
```

Expected: recovery is observable in source tests and code compiles.

## Task 4: Pure Hot Page Topology

**Files:**
- Create: `src/render/rt_page.rs`
- Modify: `src/render/mod.rs`
- Test: `src/render/rt_page.rs`

- [ ] **Step 1: Write failing topology tests**

```rust
#[test]
fn hot_page_reserves_one_quad_for_each_voxel_direction() {
    assert_eq!(HOT_PAGE_QUAD_COUNT, 3_072);
    assert_eq!(HOT_PAGE_TRIANGLE_COUNT, 6_144);
}

#[test]
fn candidate_round_trip_covers_every_voxel_direction() {
    for quad in 0..HOT_PAGE_QUAD_COUNT {
        let face = HotPageFace::from_quad_index(quad).unwrap();
        assert_eq!(face.quad_index(), quad);
    }
}
```

- [ ] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_page -- --nocapture
```

Expected: module and constants do not exist.

- [ ] **Step 3: Implement deterministic decode and finite degenerate geometry**

Define `HotPageFace { local: UVec3, direction: HotPageDirection }`, six outward
directions, quad/triangle constants, `from_quad_index`, `quad_index`, and a
four-vertex geometry helper. `degenerate_quad` returns four identical finite
positions at the voxel center. `exposed_quad` returns outward-wound unit-face
positions. Never represent a hidden face with NaN.

- [ ] **Step 4: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_page -- --nocapture
```

Expected: every candidate round-trips and degenerate geometry is finite.

## Task 5: Cross-Page Exposed-Face Oracle

**Files:**
- Modify: `src/render/rt_page.rs`
- Test: `src/render/rt_page.rs`

- [ ] **Step 1: Write failing owner/neighbor tests**

Add tests for a full page adjacent to a full page, a full page adjacent to an
absent page, and a page containing a carved cavity. Assert that shared solid
faces are absent, each solid-to-air face is emitted exactly once by its solid
owner, and an absent neighbor has no required page instance.

- [ ] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_page::tests::cross_page -- --nocapture
```

Expected: the oracle does not yet have cross-page lookup.

- [ ] **Step 3: Implement an injected occupancy lookup**

Make `page_exposed_faces` accept a closure
`Fn(UVec3) -> bool` over world voxel coordinates. It returns active owner faces
only for occupied local voxels whose neighboring coordinate is air. The closure
lets tests model pages without coupling the pure topology module to `Ucvh`.

- [ ] **Step 4: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_page -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo fmt --check
git diff --check
```

Expected: topology/oracle tests pass with no formatting or whitespace error.

## Follow-On Plans

The next plan adds a GPU page-position generator, one `ALLOW_UPDATE |
PREFER_FAST_BUILD` triangle BLAS, a fixed TLAS page slot, primary ray agreement
against the procedural path, then extends the same hit identity to shadow and GI
rays. It must not reintroduce a shared candidate lattice or normal-path
any-hit filtering.
