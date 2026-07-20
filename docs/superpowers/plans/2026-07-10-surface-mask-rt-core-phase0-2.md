# Surface-Mask RT Core Phase 0-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish submission-safe UCVH authority upload, introduce the typed
surface-mask contract, and build the correctness/measurement foundation used by
the compact triangle and OMM page backends.

**Architecture:** UCVH remains authoritative. A durable typed batch reaches GPU
authority only after successful submission. `SurfaceMaskPage` derives the six
solid-owned exposed-face masks for one `8^3` brick with precise cross-page
dependencies. Representation-neutral topology and hit records then feed a
compact triangle baseline and a later 27-plane OMM backend without changing
voxel truth.

**Tech Stack:** Rust 2024, ash/Vulkan, glam, bytemuck, Slang, existing source
contract tests, `REVOLUMETRIC_SHADER_COMPILE=skip` CPU gates.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/render/runtime.rs` | Initial/catch-up upload commit and GPU resource recovery. |
| `src/voxel/gpu_upload.rs` | Frame-slot uploads and buffer capacity contract. |
| `src/voxel/ucvh.rs` | Typed occupancy/material changes and precise boundary invalidation. |
| `src/render/rt_surface_mask.rs` | Pure surface-mask topology, source stamps, and page oracle. |
| `src/render/rt_page_geometry.rs` | Representation-neutral compact face records and shared lattice indices. |
| `src/render/mod.rs` | Exposes focused RT page modules. |
| `src/render/gpu_profiler.rs` | Separate geometry, BLAS, TLAS, and trace timings. |

## Task 1: Commit Initial Upload Only After Submission

**Files:**
- Modify: `src/render/runtime.rs`
- Test: `src/render/runtime.rs`

- [x] **Step 1: Keep the existing RED lifecycle contract**

The existing test
`initial_ucvh_upload_waits_for_submission_and_incremental_catch_up_before_ready`
requires these runtime states:

```rust
ucvh_initial_upload_batch_id: Option<u64>,
ucvh_initial_upload_snapshot_taken: bool,
ucvh_initial_upload_committed: bool,
```

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib initial_ucvh_upload_waits_for_submission_and_incremental_catch_up_before_ready -- --nocapture
```

Expected: FAIL because the runtime currently marks `ucvh_uploaded` when upload
commands are recorded, before `end_frame` accepts submission.

- [x] **Step 2: Snapshot the initial authority revision once**

Before the first initial-upload chunk, call `snapshot_render_change_batch()` and
retain its non-empty batch ID. Do not acknowledge it while chunks are merely
recorded. Initialize/reset all three lifecycle fields in both runtime creation
and UCVH GPU recreation.

- [x] **Step 3: Separate recorded, committed, and ready states**

Use a frame-local:

```rust
let mut initial_upload_completed_this_frame = false;
```

`upload.completed` only sets this flag. After a successful
`renderer.end_frame(frame)?`:

```rust
self.ucvh_initial_upload_committed = true;
```

Then acknowledge the retained initial batch. Newer revisions must survive the
old acknowledgement through UCVH's generation/revision comparison.

- [x] **Step 4: Upload post-snapshot edits before readiness**

Allow `upload_incremental_changes` when initial upload is committed even if
`ucvh_uploaded` is still false. A submitted catch-up batch is acknowledged only
after `end_frame`. Set `ucvh_uploaded = true` and
`outcome.uploaded_ucvh = true` only when initial upload is committed and a fresh
render-change snapshot is empty after acknowledgement.

Factor post-submit acknowledgement so the scene-UBO fallback path cannot return
after `end_frame` without committing initial/catch-up state.

- [x] **Step 5: Run GREEN and neighboring runtime tests**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib initial_ucvh_upload_waits_for_submission_and_incremental_catch_up_before_ready -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render_runtime_uploads_and_acknowledges_durable_render_change_batches -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo check --lib
```

Expected: PASS.

## Task 2: Recover From UCVH GPU Capacity Growth

**Files:**
- Modify: `src/render/runtime.rs`
- Modify: `src/voxel/gpu_upload.rs`
- Test: `src/render/runtime.rs`

- [x] **Step 1: Add a failing pure capacity decision test**

Extract and test:

```rust
fn ucvh_gpu_requires_recreation(gpu_capacity: usize, cpu_storage_len: usize) -> bool {
    gpu_capacity < cpu_storage_len
}
```

The test covers equal, smaller, and larger capacities and a source contract
requires capacity checking before `begin_frame`.

- [x] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib ucvh_gpu_capacity -- --nocapture
```

Expected: FAIL because no pre-frame recovery path exists.

- [x] **Step 3: Implement explicit safe recreation**

Before `begin_frame`, compare `gpu.brick_capacity()` with
`ucvh.pool.occupancy_pool().len()`. On growth:

1. wait for current frame resources to become idle;
2. destroy and recreate RT/VPT pipelines that own old UCVH descriptors;
3. destroy old `UcvhGpuResources`;
4. recreate with current UCVH storage and frame-slot count;
5. reset initial-upload lifecycle and RT/VPT history;
6. retain every durable authority batch for full upload plus catch-up.

Ordinary same-capacity edits must not take this path.

- [x] **Step 4: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib ucvh_gpu_capacity -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib runtime::tests -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo check --lib
```

## Task 3: Add Typed Brick Change Metadata

**Files:**
- Modify: `src/voxel/ucvh.rs`
- Test: `src/voxel/ucvh.rs`

- [x] **Step 1: Write failing occupancy/material classification tests**

Require `UcvhChangedBrick` to include:

```rust
pub occupancy_changed: bool,
pub material_changed: bool,
pub touched_boundaries: UcvhPageBoundaryMask,
```

Tests must show:

- interior material replacement sets only `material_changed`;
- air/solid transition sets `occupancy_changed`;
- a voxel at local `x == 0` sets only negative-X boundary;
- a voxel at local `(7, 7, 7)` sets positive X/Y/Z;
- multiple edits OR flags without losing a newer revision.

- [x] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render_change_batch_classifies -- --nocapture
```

- [x] **Step 3: Track typed pending metadata**

Add per-brick pending occupancy/material flags and a six-bit boundary mask beside
the existing generation/revision arrays. Update them in the authoritative edit
path based on old and new occupancy/material values. Old-batch acknowledgement
clears only metadata whose revision still matches.

- [x] **Step 4: Narrow invalidated pages**

The owner page is always invalidated for occupancy changes. Add only the
in-bounds neighbor pages named by `touched_boundaries`. Material-only edits keep
the authority batch but do not enter `invalidated_pages`.

- [x] **Step 5: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib ucvh::tests -- --nocapture
```

## Task 4: Implement `SurfaceMaskPage`

**Files:**
- Create: `src/render/rt_surface_mask.rs`
- Modify: `src/render/mod.rs`
- Test: `src/render/rt_surface_mask.rs`

- [x] **Step 1: Write the failing public-contract tests**

Define the wished-for API:

```rust
pub struct SurfaceMaskPage {
    directions: [[u64; 8]; 6],
}

impl SurfaceMaskPage {
    pub fn from_ucvh(ucvh: &Ucvh, page: UVec3) -> Self;
    pub fn is_exposed(&self, local: UVec3, direction: FaceDirection) -> bool;
    pub fn exposed_face_count(&self) -> u32;
    pub fn interface_cell_count(&self) -> u32;
}
```

Tests cover empty, one voxel, full isolated page, cavity, checkerboard,
world-edge, cross-page solid/solid, and cross-page solid/air ownership.

- [x] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_surface_mask -- --nocapture
```

Expected: compile failure because the module and types do not exist.

- [x] **Step 3: Implement deterministic owner-face masks**

Use the existing Morton mapping for local voxel bits and direct UCVH neighbor
queries. Keep `FaceDirection` order explicit and stable. Missing/out-of-world
neighbors are air. Do not add greedy merging or geometry generation here.

- [x] **Step 4: Add source stamps**

Add a `SurfaceMaskSourceStamp` containing owner and six-neighbor generation or
absent sentinels. Test that changing either side of a page boundary invalidates
the old stamp.

- [x] **Step 5: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_surface_mask -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo check --lib
```

## Task 5: Add Compact Face And Interface Topology Records

**Files:**
- Create: `src/render/rt_page_geometry.rs`
- Modify: `src/render/mod.rs`
- Test: `src/render/rt_page_geometry.rs`

- [x] **Step 1: Write failing compact geometry tests**

Require:

```rust
#[repr(C)]
pub struct RtCompactFaceRecord {
    pub packed_owner_direction: u32,
}

pub struct RtCompactPageGeometry {
    pub indices: Vec<u16>,
    pub faces: Vec<RtCompactFaceRecord>,
}
```

Tests assert `729` shared lattice vertices, six indices per exposed face,
two consecutive triangles per face record, valid winding for all directions,
and no indices/faces for hidden surfaces.

- [x] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_page_geometry -- --nocapture
```

- [x] **Step 3: Build exact geometry from `SurfaceMaskPage`**

Generate only index and face-record streams. Vertices are immutable local
lattice coordinates and are shared by all page BLAS builds. Decode functions
must round-trip every local owner and direction.

- [x] **Step 4: Add corrected interface topology constants**

Expose tested constants/mapping for 1,344 internal plus 384 owner-boundary
quads. This is benchmark input for `HotInterface`, not the selected production
backend yet.

- [x] **Step 5: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_page_geometry -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo check --lib
```

## Task 6: Establish The Representation Bake-Off Contract

**Files:**
- Modify: `src/render/gpu_profiler.rs`
- Create: `src/render/rt_representation_metrics.rs`
- Modify: `src/render/mod.rs`
- Test: corresponding module tests

- [x] **Step 1: Write failing metric/schema tests**

The metrics record representation kind, page count, exposed faces, carrier or
candidate primitives, transparent/degenerate count, geometry/OMM time, BLAS
BUILD/UPDATE time, TLAS time, trace time, and persistent/scratch bytes.

- [x] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_representation_metrics -- --nocapture
```

- [x] **Step 3: Add profiler scopes and stable logging schema**

Add distinct scopes for surface generation, OMM build, BLAS work, TLAS work,
primary trace, shadow trace, and GI trace. Keep absent stages explicit rather
than folding them into zero without a representation label.

- [x] **Step 4: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_representation_metrics -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib gpu_profiler -- --nocapture
```

## Task 7: Phase Verification And Phase 3 Handoff

**Files:**
- Modify: this plan's checkboxes/execution notes
- Create after evidence: `docs/superpowers/plans/2026-07-10-surface-mask-rt-core-phase3-5.md`

- [x] **Step 1: Run focused and library gates**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib ucvh::tests -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_surface_mask -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rt_page_geometry -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib runtime::tests -- --nocapture
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo check --lib
cargo fmt --check
```

- [x] **Step 2: Inspect the scoped diff**

Confirm no normal RT shader or runtime has been falsely labeled triangle-native
before a triangle BLAS exists, no pending batch can be lost, and historical
plans remain explicitly superseded.

- [x] **Step 3: Write the Phase 3-5 plan from measured APIs**

The next plan must cover real GPU compact-page buffers, versioned BLAS arena,
fixed TLAS slots, triangle hit groups, primary agreement, OMM resource/build
integration, shadow/GI migration, and RT visual capture. It must use the actual
types and measurements produced by this phase rather than guessed Vulkan sizes.

## Execution Record (2026-07-10)

- The initial upload lifecycle now distinguishes recorded, submitted, catch-up,
  and ready states. Authority batches are acknowledged only after `end_frame`
  accepts submission, including the scene-UBO fallback path.
- UCVH GPU storage growth is detected before `begin_frame`; dependent RT/VPT
  pipelines and old descriptors are retired only after `device_wait_idle`.
- Typed changes carry occupancy/material flags and precise six-way boundary
  masks. Persistent topology revisions let source stamps ignore material-only
  edits while rejecting stale owner or neighbor topology.
- `SurfaceMaskPage` passed empty, single-voxel, shell, cavity, checkerboard,
  world-edge, cross-page ownership, and source-stamp tests. Its directional mask
  payload is 384 bytes.
- `RtCompactPageGeometry` uses the shared 729-vertex lattice and emits only two
  triangles per exposed face. The fixed interface mapping round-trips all 1,728
  candidate quads.
- Representation metrics keep absent stages as `na`. The legacy timestamp API
  cannot name the synchronization2 micromap stage, so `RtOmmBuild` temporarily
  uses `ALL_COMMANDS` until the profiler moves to `vkCmdWriteTimestamp2`.
- `run/validate-local.ps1 -StrictShaders` passed: formatting, 989 library tests
  (`987` passed, `2` environment-dependent tests ignored), clippy with warnings
  denied, real Slang compilation, and strict library build.
- This phase does **not** claim triangle-native RT. The active runtime still uses
  the procedural AABB + brick-DDA reference until the Phase 3-5 plan is executed.
