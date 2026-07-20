# Surface-Mask RT Core Phase 3-5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active monolithic procedural-AABB RT scene with a
frame-safe mixed page registry whose normal path is CompactExact triangle BLAS,
then migrate every ray class and admit OMM only through measured gates.

**Architecture:** UCVH remains authoritative. Every allocated surface page owns
one fixed TLAS slot which selects exactly one of a shared procedural reference
BLAS, a versioned CompactExact triangle BLAS, a later optimized BLAS, or a
mask-zero dummy. Dirty pages enter Reference immediately, build replacements
out of place from `SurfaceMaskSourceStamp`, and retire old versions only after
all frame-slot TLAS users have completed.

**Tech Stack:** Rust 2024, ash 0.38, Vulkan 1.3,
`VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_pipeline`,
`VK_KHR_buffer_device_address`, Slang, optional
`VK_EXT_opacity_micromap`, existing render graph/profiler/capture tooling.

---

## Verified Starting Point

- `UcvhRenderChangeBatch` is durable through authority upload and contains
  typed occupancy/material changes plus precise page invalidations.
- `SurfaceMaskPage` is the CPU topology oracle and carries topology-only source
  stamps for owner plus six neighbors.
- `RtCompactPageGeometry` emits six `u16` indices and one 4-byte face record per
  exposed quad against a shared 729-vertex lattice.
- `RtRepresentationMetrics` and separate geometry/OMM/BLAS/TLAS/trace profiler
  scopes exist.
- `RtScene` still owns one procedural AABB BLAS and one TLAS instance.
- `RayTracingPipeline::new_surface_pipeline` still hardcodes one procedural hit
  group.
- The current intersection shaders still execute brick-local DDA. This remains
  the bounded Reference path, not the normal completed backend.

## File Structure

| Path | Responsibility |
| --- | --- |
| `src/render/rt_page_registry.rs` | Sparse page-to-slot map, ownership state machine, source generations, dirty/build queues. |
| `src/render/rt_page_gpu.rs` | Shared lattice, page/index/face arenas, page records, upload ranges. |
| `src/render/rt_page_blas.rs` | Triangle BLAS size queries, BUILD recording, scratch slices, version ownership. |
| `src/render/rt_page_tlas.rs` | Dummy/reference BLAS, frame-slot instance stores, TLAS BUILD/UPDATE, deferred retirement. |
| `src/render/rt_scene.rs` | Facade integrating registry, build scheduling, reference fallback, and descriptors. |
| `src/render/pipeline.rs` | Generalized mixed procedural/triangle hit-group pipeline and SBT layout. |
| `src/render/rt_hit_abi.rs` | CPU mirror of page records, representation IDs, hit-group IDs, packing contracts. |
| `assets/shaders/shared/rt_page_common.slang` | Shared page record and logical `SurfaceKey` decoding. |
| `assets/shaders/passes/rt_surface_triangle.rchit.slang` | CompactExact primary closest hit. |
| `assets/shaders/passes/rt_reference.rint.slang` | Shared-unit-AABB page reference DDA. |
| `assets/shaders/passes/rt_*_triangle.rchit.slang` | Shadow/GI triangle hit groups using the same page ABI. |
| `src/render/rt_omm.rs` | OMM encoder, capability contract, GPU build resources, candidate A/B metrics. |
| `src/render/rt_representation_router.rs` | Measured backend selection, hysteresis, queue and memory budgets. |

## Phase 3: CompactExact End To End

### Task 1: Lock The Page Registry State Machine

**Files:**
- Create: `src/render/rt_page_registry.rs`
- Modify: `src/render/mod.rs`
- Test: `src/render/rt_page_registry.rs`

- [x] **Step 1: Write failing ownership and stale-result tests**

Require the following public contract:

```rust
pub type RtPageSlot = u32;
pub type RtPageBuildGeneration = u64;

pub enum RtPageRepresentation {
    Missing,
    Reference,
    CompactExact,
    HotOmm,
    HotInterface,
    CompactGreedy,
}

pub enum RtPageState {
    Missing,
    Reference,
    Building {
        target: RtPageRepresentation,
        generation: RtPageBuildGeneration,
        source: SurfaceMaskSourceStamp,
    },
    Resident {
        representation: RtPageRepresentation,
        generation: RtPageBuildGeneration,
        source: SurfaceMaskSourceStamp,
        resource_version: u64,
    },
    Failed {
        generation: RtPageBuildGeneration,
    },
}
```

Tests must prove:

- one page maps to one stable slot;
- an occupancy invalidation changes Resident to Reference before scheduling;
- material-only changes do not schedule topology work;
- a result with an old build generation or stale source stamp is rejected;
- successful install selects exactly one representation;
- failure leaves Reference traceable;
- queue capacity reports overflow without dropping the page identity.

- [x] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_page_registry -- --nocapture
```

Expected: compile failure because the registry types do not exist.

- [x] **Step 3: Implement sparse slots and bounded queues**

Use a `HashMap<UVec3, RtPageSlot>` plus dense records. Do not allocate one slot
per world-grid coordinate. Slot zero is reserved for the dummy record. Queue
insertion is idempotent by page and preserves the oldest enqueue frame for age
telemetry.

- [x] **Step 4: Connect durable invalidations before authority acknowledgement**

In `RenderRuntime::render_frame`, copy `batch.invalidated_pages` into the
registry's durable dirty queue before `ack_render_change_batch` can clear the
UCVH snapshot. Initial bootstrap enumerates allocated brick coordinates and
creates Reference slots even when no edit batch exists.

- [x] **Step 5: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_page_registry -- --nocapture
cargo test --lib render_runtime_uploads_and_acknowledges_durable_render_change_batches -- --nocapture
```

**Execution record (2026-07-10):** `RtPageRegistry` now owns stable sparse
slots, monotonic build tickets, Reference fallback, stale-result rejection,
and a bounded ready queue. Overflow identities remain durable in page records
and an age-ordered `BTreeSet`, so refilling does not scan every page. Runtime
bootstrap enumerates sparse allocated brick coordinates, and both initial and
incremental authority snapshots enter the registry before acknowledgement.
Focused tests, all 998 library tests, all-target clippy, strict shader
compilation, and strict library build passed; two asset-dependent tests were
ignored by their existing guards.

### Task 2: Add The Shared Lattice And Versioned Geometry Arena

**Files:**
- Create: `src/render/rt_page_gpu.rs`
- Modify: `src/render/buffer.rs` only if a checked device-address helper is needed
- Test: `src/render/rt_page_gpu.rs`

- [x] **Step 1: Write failing layout and allocator tests**

Require:

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRtPageRecord {
    pub brick_id: u32,
    pub face_record_offset: u32,
    pub face_count: u32,
    pub representation: u32,
    pub page_coord: [u32; 4],
    pub topology_revision: u64,
    pub resource_version: u64,
}

pub struct RtPageGeometryAllocation {
    pub index_offset_bytes: u64,
    pub index_size_bytes: u64,
    pub face_offset_records: u32,
    pub face_count: u32,
    pub allocation_id: u64,
}
```

Tests cover 16-bit index alignment, face-record alignment, exact-fit
allocation, fragmentation reuse, double-free rejection, generation-safe free,
and deterministic shared lattice coordinates `[0, 8]^3`.

- [x] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_page_gpu -- --nocapture
```

- [x] **Step 3: Create immutable GPU lattice**

Upload 729 `[f32; 3]` vertices once with
`VERTEX_BUFFER | SHADER_DEVICE_ADDRESS | ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR`.
Assert vertex zero, vertex 728, stride 12, and nonzero aligned device address.

- [x] **Step 4: Implement bounded index/face/page-record arenas**

Use explicit free ranges and checked arithmetic. Index arena usage includes
`INDEX_BUFFER` and AS build-input flags; face/page-record arenas include
`STORAGE_BUFFER`. A replacement allocation is distinct from the resident
allocation until install succeeds.

- [x] **Step 5: Record frame-slot-owned uploads**

Use one host-visible staging region per frame slot. Emit
TRANSFER_WRITE-to-AS_BUILD_READ and TRANSFER_WRITE-to-RAY_TRACING_SHADER_READ
barriers. Do not overwrite a staging slice before its frame fence completes.

- [x] **Step 6: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_page_gpu -- --nocapture
cargo check --lib
```

**Execution record (2026-07-10):** Added the 48-byte GPU page ABI, immutable
729-vertex lattice, checked first-fit index/face arenas, stable page-record
storage, generation-safe frees, and out-of-place replacement allocations.
`RtPageGpuResources` owns device-local lattice/index/face/page-record buffers
plus one host-visible staging buffer per frame slot. Explicit frame state
prevents reuse before completion, while a page-record journal commits CPU
shadow state only after submission and discards cancelled recordings. Uploads
record exact transfer-to-AS-build and transfer-to-RT-shader barriers. Nineteen
focused tests, all 1,017 library tests, all-target clippy, strict shader
compilation, and strict library build passed; two asset-dependent tests were
ignored by their existing guards. Real command execution remains part of the
Task 3 BLAS integration smoke, where these resources first become active.

### Task 3: Build Real CompactExact Triangle BLAS Versions

**Files:**
- Create: `src/render/rt_page_blas.rs`
- Modify: `src/render/rt_scene.rs`
- Test: `src/render/rt_page_blas.rs`

- [x] **Step 1: Write failing pure build-plan tests**

`RtCompactBlasBuildPlan` must produce:

```rust
vk::AccelerationStructureGeometryTrianglesDataKHR {
    vertex_format: vk::Format::R32G32B32_SFLOAT,
    vertex_stride: 12,
    max_vertex: 728,
    index_type: vk::IndexType::UINT16,
    ..
}
```

The primitive count is `indices.len() / 3` and exactly
`2 * faces.len()`. Empty geometry returns `None` rather than an invalid BLAS
build.

- [x] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_page_blas -- --nocapture
```

- [x] **Step 3: Query sizes on the actual device**

Call `get_acceleration_structure_build_sizes` with the exact primitive count.
Allocate BLAS storage with
`ACCELERATION_STRUCTURE_STORAGE_KHR | SHADER_DEVICE_ADDRESS` and allocate scratch
at `min_acceleration_structure_scratch_offset_alignment`. Store queried
persistent and scratch bytes in `RtRepresentationMetrics`.

- [x] **Step 4: Record out-of-place BUILD**

Use `PREFER_FAST_BUILD` and `BUILD`, never UPDATE for CompactExact topology
changes. Record geometry-upload-to-BLAS and BLAS-write-to-TLAS-read barriers.
Wrap only the BLAS command with `GpuProfileScope::RtBlasWork`.

- [x] **Step 5: Keep failed/stale builds nonresident**

The build result owns all new buffers and AS handles until registry install.
On stale stamp, allocation failure, or recording failure, enqueue those
resources for safe retirement and keep the page in Reference.

- [x] **Step 6: Run GREEN and strict shader-independent gates**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_page_blas -- --nocapture
cargo test --lib rt_page_geometry -- --nocapture
cargo check --lib
```

**Execution record (2026-07-10):** `RtCompactBlasBuildResources` now queries
exact device build sizes, owns out-of-place CompactExact BLAS and aligned
GPU-only scratch allocations, records BUILD-only work under the dedicated
profiler scope, and publishes BLAS writes for TLAS reads. Its consuming
install API atomically returns an installed owned version or moves rejected,
failed, stale, and crossed-ticket resources into the retirement queue while
the registry remains Reference-traceable. Twelve focused BLAS tests, the
CompactExact geometry tests, library check, all-target clippy, and the strict
local gate passed. Fence-backed retirement and frame-slot TLAS references
remain intentionally owned by Tasks 4 and 8.

### Task 4: Replace The Single Instance With Fixed Frame-Slot TLAS Slots

**Files:**
- Create: `src/render/rt_page_tlas.rs`
- Modify: `src/render/rt_scene.rs`
- Modify: `src/render/device.rs` only for a read-only completed-frame epoch API
- Test: `src/render/rt_page_tlas.rs`

- [x] **Step 1: Write failing slot and lifetime tests**

Require one `vk::AccelerationStructureInstanceKHR` per dense page slot.
Tests cover:

- mask-zero dummy slots;
- page translation transform;
- 24-bit `instance_custom_index_and_mask` bounds;
- `instance_shader_binding_table_record_offset_and_flags` selecting Reference
  versus CompactExact;
- TLAS capacity bounded by device `max_instance_count` and `0x00ff_ffff`;
- a BLAS version cannot retire while any frame-slot TLAS generation references
  it;
- capacity growth requires a quiescent rebuild, ordinary edits do not.

- [x] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_page_tlas -- --nocapture
```

- [x] **Step 3: Build dummy and shared reference BLAS**

The dummy BLAS contains valid finite triangle geometry and is always used with
instance mask zero. The shared Reference BLAS contains one procedural local
`[0, 8]^3` AABB. Every fallback page instances that BLAS with its page
translation; the reference intersection shader reads page identity through
`InstanceCustomIndex` and performs bounded brick-local DDA.

- [x] **Step 4: Create one instance buffer and TLAS per frame slot**

Frame slot reuse is protected by the existing slot fence. A slot's TLAS UPDATE
uses `ALLOW_UPDATE | PREFER_FAST_BUILD` only after an initial BUILD with fixed
capacity. Instance input storage is never shared with an in-flight frame.

- [x] **Step 5: Record ownership transitions before trace**

For each current frame slot, write Reference, resident BLAS, or dummy into every
changed instance record, record TLAS UPDATE under `RtTlasWork`, then emit the
AS-write-to-ray-tracing-read barrier. Never leave the previous resident BLAS in
the slot after authority invalidation.

- [x] **Step 6: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_page_tlas -- --nocapture
cargo test --lib rt_scene -- --nocapture
```

**Execution record (2026-07-10):** `RtPageTlasGpuResources` now owns finite
dummy and shared local-page Reference BLASes plus independent fixed-capacity
instance buffers, TLASes, and aligned scratch for every frame slot. Registry
invalidation synchronizes the current slot to Reference before RT pass
recording; submitted resource-version references clear only after a fence-
confirmed completed epoch. Capacity growth waits all other slot fences,
proves quiescence, and replaces resources out of place. Initial BUILD and
ordinary UPDATE paths use distinct AS-build and trace barriers, including the
cross-submission dependency required before another slot's first UPDATE.
Eighteen focused page-TLAS/runtime tests, all 1,048 library tests, all-target
clippy, strict shader compilation, and strict library build passed; two local
asset tests remained ignored by their existing guards. The new page TLAS is
recorded but does not become the trace target until Task 5 supplies mixed hit
groups and stable SBT offsets.

### Task 5: Generalize RT Pipelines To Mixed Hit Groups

**Files:**
- Modify: `src/render/pipeline.rs`
- Create: `src/render/rt_hit_abi.rs`
- Create: `assets/shaders/shared/rt_page_common.slang`
- Test: `src/render/pipeline.rs` and `src/render/rt_hit_abi.rs`

- [ ] **Step 1: Write failing SBT/group tests**

Replace the hardcoded constructor with:

```rust
pub enum RtHitGroupKind {
    Procedural {
        closest_hit_stage: u32,
        intersection_stage: u32,
    },
    Triangles {
        closest_hit_stage: u32,
    },
}

pub fn new_mixed_surface_pipeline(
    /* existing device/layout arguments */,
    stages: &[RtShaderStageSpec],
    hit_groups: &[RtHitGroupKind],
) -> Result<Self>;
```

Tests require group 0 = Reference procedural, group 1 = CompactExact triangles,
stable SBT ordering, no normal-path any-hit stage, and checked stage indices.

- [ ] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib mixed_surface_pipeline -- --nocapture
cargo test --lib rt_hit_abi -- --nocapture
```

- [ ] **Step 3: Define the shared logical hit ABI**

CPU and Slang mirrors must include:

```text
SurfaceKey
  page record / page coordinate
  BrickId
  owner local voxel
  FaceDirection
  material generation
```

Add size/offset tests and shader source-contract tests. Reference and triangle
decoders must produce the same fields.

- [ ] **Step 4: Build the generalized pipeline and SBT**

Keep existing raygen and miss records. Add both hit records with stride and
base alignment from device properties. Instance SBT offsets, not separate
TLASes, choose the representation.

- [ ] **Step 5: Run GREEN**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib mixed_surface_pipeline -- --nocapture
cargo test --lib rt_hit_abi -- --nocapture
```

### Task 6: Implement CompactExact Primary Closest Hit

**Files:**
- Create: `assets/shaders/passes/rt_surface_triangle.rchit.slang`
- Modify: `assets/shaders/passes/rt_surface.rgen.slang`
- Modify: `src/render/passes/rt_surface.rs`
- Modify: `src/render/rt_pipeline.rs`
- Test: source tests in `src/render/passes/rt_surface.rs`

- [ ] **Step 1: Write failing shader ABI tests**

Require the triangle closest hit to:

- use `PrimitiveIndex() / 2` to load one `RtCompactFaceRecord`;
- use `InstanceCustomIndex()` to load `GpuRtPageRecord`;
- decode owner local and direction;
- derive world position from `WorldRayOrigin() + WorldRayDirection() * RayTCurrent()`;
- fetch material from authoritative `brick_materials`;
- write the existing `RtSurfacePayload` without reading AABB/intersection
  attributes;
- contain no DDA and no any-hit shader.

- [ ] **Step 2: Run RED**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_surface_triangle -- --nocapture
```

- [ ] **Step 3: Add page-record and face-record descriptors**

Bindings are frame-slot scoped. Descriptor updates happen only after the
corresponding arena version exists. Retain the Reference descriptors for the
procedural hit group.

- [ ] **Step 4: Wire mixed primary tracing**

`RtSurfacePass` uses the frame-slot TLAS and mixed SBT. Add
`GpuProfileScope::RtPrimaryTrace` around `cmd_trace_rays`. Keep
`RtSurface` as a compatibility aggregate only until profiling consumers move.

- [ ] **Step 5: Run strict shader compilation**

```powershell
.\run\validate-local.ps1 -StrictShaders
```

Expected: all Slang jobs compile and descriptor reflection matches Rust specs.

### Task 7: Prove Primary-Ray Agreement Before Making CompactExact Normal

**Files:**
- Create: `src/render/rt_agreement.rs`
- Modify: `src/render/capture.rs`
- Modify: `src/render/rt_settings.rs`
- Modify: `run/visual-baselines.json`
- Test: corresponding Rust/source tests

- [ ] **Step 1: Write failing deterministic agreement tests**

Generate empty, shell, cavity, checkerboard, cross-page, and randomized pages.
For deterministic rays compare CompactExact against the CPU/reference oracle
for hit/miss, `t` tolerance, owner voxel, normal, and material.

- [ ] **Step 2: Add runtime dual-trace diagnostic mode**

The mode records mismatch counters and first-failure `SurfaceKey` without
changing the displayed result. It is bounded to selected frames/pixels and is
off by default.

- [ ] **Step 3: Capture primary output**

Run:

```powershell
.\run\validate-visual-baseline.ps1 -Rt
.\tools\rt_flythrough_capture.ps1 -Mode rt -Frames 32 -CaptureFrames "2,16,31"
```

Acceptance:

- zero agreement mismatches in deterministic cases;
- nonblank PPM signal checks;
- no holes, duplicate cross-page faces, inverted normals, or material drift;
- capture metadata identifies `compact_exact`, page count, Reference count,
  BLAS/TLAS timings, and memory.

- [ ] **Step 4: Change the normal primary owner**

Only after the agreement and visual gates pass, make CompactExact the default
resident target. Reference remains explicit for dirty, failed, and overflow
pages.

## Phase 4: Incremental Destruction And Every Ray Class

### Task 8: Add Submission-Safe Page Replacement And Retirement

**Files:**
- Modify: `src/render/rt_page_registry.rs`
- Modify: `src/render/rt_page_blas.rs`
- Modify: `src/render/rt_page_tlas.rs`
- Modify: `src/render/runtime.rs`
- Test: all three modules and runtime source contracts

- [ ] **Step 1: Write failing multi-frame lifecycle tests**

Simulate at least three frame slots. Prove:

- edit N invalidates resident version A before trace;
- replacement B recorded but not submitted cannot install;
- submission installs B only in the current frame-slot TLAS;
- A remains alive until every frame slot that referenced A completes;
- an edit during B build rejects B by source stamp and retains a newer task;
- allocation/build failure remains Reference and retries with backoff;
- no ordinary edit path calls `wait_idle`.

- [ ] **Step 2: Implement submitted work receipts**

Create a frame-local receipt containing built versions, TLAS transitions, and
retire candidates. Commit it only after `RenderDevice::end_frame` succeeds,
matching the authority upload lifecycle.

- [ ] **Step 3: Implement fence-backed retirement**

Track the frame-slot generations that can reference each resource version.
Destroy index/face allocations, BLAS handles/storage, and optional micromaps
only after all bits clear. Do not infer safety from CPU age alone.

- [ ] **Step 4: Run GREEN and destruction stress**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_page_registry -- --nocapture
cargo test --lib rt_page_blas -- --nocapture
cargo test --lib rt_page_tlas -- --nocapture
cargo test --lib runtime::tests -- --nocapture
```

### Task 9: Migrate Shadow And GI To The Shared TLAS And Hit ABI

**Files:**
- Create: `assets/shaders/passes/rt_direct_lighting_triangle.rchit.slang`
- Create: `assets/shaders/passes/rt_restir_gi_triangle.rchit.slang`
- Modify: `src/render/passes/rt_direct_lighting.rs`
- Modify: `src/render/passes/rt_restir_gi.rs`
- Modify: `src/render/rt_pipeline.rs`
- Test: pass-local shader/source tests

- [ ] **Step 1: Write failing shared-TLAS tests**

Require primary, direct-light shadow, and ReSTIR-GI to receive the same
frame-slot TLAS handle and page-record descriptors. Their SBTs must use the same
Reference/CompactExact group offsets.

- [ ] **Step 2: Implement triangle shadow closest hit**

The normal shadow hit only sets occluded. Ray origin bias and `TMin` handle
self-intersection; no voxel-skip any-hit and no software retrace are allowed.
Wrap shadow tracing with `RtShadowTrace`.

- [ ] **Step 3: Implement triangle GI closest hit**

Decode the same `SurfaceKey` and material as primary. Preserve current
`RtRestirGiPayload` semantics and wrap GI rays with `RtGiTrace`.

- [ ] **Step 4: Keep Reference procedural groups**

Reference intersection shaders derive page coordinate from the page record,
not from a global AABB array. Their closest-hit output must match triangle hit
output field for field.

- [ ] **Step 5: Run strict and visual gates**

```powershell
.\run\validate-local.ps1 -StrictShaders
.\run\validate-visual-baseline.ps1 -Rt
```

### Task 10: Enforce Bounded Scheduling, Memory, And Diagnostics

**Files:**
- Create: `src/render/rt_page_scheduler.rs`
- Modify: `src/render/rt_representation_metrics.rs`
- Modify: `src/render/gpu_profiler.rs`
- Modify: `src/editor/ui.rs`
- Test: scheduler and source-contract tests

- [ ] **Step 1: Write failing priority/backpressure tests**

The scheduler order is authority, visible masks, replacement geometry/BLAS,
TLAS transition, visible recovery, then promotions. Tests cover per-frame page,
byte, scratch, and command budgets; starvation age; queue deduplication; and
overflow fallback.

- [ ] **Step 2: Implement measured budget accounting**

Use queried sizes and recorded timings, not constants guessed from another GPU.
Scratch slices cannot overlap in one command buffer unless builds are ordered
and the Vulkan scratch reuse barrier is present.

- [ ] **Step 3: Expose operational telemetry**

Report dirty pages, Reference pages, resident kinds, oldest age, builds,
stale discards, failures, arena bytes, scratch high-water mark, BLAS/TLAS times,
and primary/shadow/GI trace times. UI is diagnostic only and does not own state.

- [ ] **Step 4: Run a high-churn capture**

Add a deterministic edit script which changes interior and cross-page voxels
for 300 frames. Acceptance:

- no stale geometry;
- no use-after-free validation errors;
- bounded queue/memory growth;
- ordinary frames contain no device-idle wait;
- Reference backlog recovers after edits stop.

## Phase 5: Measured Optimized Representations

### Task 11: Add A Verified OMM Encoder And Capability Boundary

**Files:**
- Create: `src/render/rt_omm.rs`
- Modify: `src/render/rt_capabilities.rs`
- Modify: `src/render/device.rs`
- Test: `src/render/rt_omm.rs` and capability tests

- [ ] **Step 1: Write failing 27-sheet mapping tests**

For all 27 sheets and 54 carrier triangles, compare a reference interface grid
with Vulkan triangular space-filling-order bits. Cover empty, full, one face,
cavity, checkerboard, and all boundary directions. Assert 3,456 logical
microtriangles and double-sided carrier flags.

- [ ] **Step 2: Gate device creation correctly**

Query and enable `VkPhysicalDeviceOpacityMicromapFeaturesEXT` only when the
extension, feature, required subdivision level, and function loader exist.
Capability logging includes the actual limits. Extension presence alone does
not select HotOmm.

- [ ] **Step 3: Run CPU encoder and strict gates**

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test --lib rt_omm -- --nocapture
.\run\validate-local.ps1 -StrictShaders
```

### Task 12: Implement Both Legal OMM Replacement Candidates

**Files:**
- Modify: `src/render/rt_omm.rs`
- Modify: `src/render/rt_page_blas.rs`
- Modify: `src/render/rt_page_registry.rs`
- Create: `assets/shaders/passes/rt_surface_omm.rchit.slang`
- Test: OMM source/layout tests

- [ ] **Step 1: Implement candidate A**

Build replacement micromap data with `vkCmdBuildMicromapsEXT`, build a
replacement 54-triangle carrier BLAS, and install it through the same source
stamp/retirement receipt as CompactExact.

- [ ] **Step 2: Implement candidate B**

After an initial BLAS BUILD with `ALLOW_UPDATE` and
`ALLOW_OPACITY_MICROMAP_DATA_UPDATE_EXT`, build a replacement micromap and
record structurally identical BLAS UPDATE. Do not claim micromap UPDATE; the EXT
API rebuilds the micromap.

- [ ] **Step 3: Add required barriers and lifetime ownership**

OMM input upload, micromap build, BLAS read, TLAS read, and trace barriers must
use extension-valid stage/access masks. Micromap storage remains alive as long
as any BLAS version references it.

- [ ] **Step 4: Implement bounded OMM hit decoding**

Decode carrier ID plus standard barycentrics/object position, inspect the two
adjacent occupancy cells, and produce the shared `SurfaceKey`. Carrier winding
is never used as voxel normal.

- [ ] **Step 5: Run strict and agreement gates**

```powershell
.\run\validate-local.ps1 -StrictShaders
.\run\validate-visual-baseline.ps1 -Rt
```

### Task 13: Benchmark And Gate CompactExact, OMM, And HotInterface

**Files:**
- Create: `src/render/rt_representation_router.rs`
- Modify: `src/render/rt_representation_metrics.rs`
- Modify: `src/render/rt_page_scheduler.rs`
- Test: router tests

- [ ] **Step 1: Write failing deterministic router tests**

The router consumes measured update-plus-trace time, persistent/scratch bytes,
page churn, occupancy density, and backlog. It uses hysteresis and a minimum
sample count. Unsupported or unmeasured backends cannot win.

- [ ] **Step 2: Implement HotInterface only as a measured candidate**

Use the tested 1,728-slot map, finite degenerate inactive quads,
`ALLOW_UPDATE | PREFER_FAST_BUILD`, and explicit owner-side metadata. NaNs,
duplicate boundary ownership, and empty-neighbor instances are forbidden.

- [ ] **Step 3: Run the bake-off on the target RTX device**

Record separate rows for sparse, shell, cavity, checkerboard, and destruction
traces. Include fresh CompactExact BUILD, OMM candidate A, OMM candidate B,
HotInterface UPDATE, TLAS, all trace classes, and bytes.

- [ ] **Step 4: Apply the selection gate**

HotOmm becomes selectable only with a repeatable combined update-plus-trace win
and acceptable memory/backlog. Otherwise CompactExact remains normal and
HotInterface remains the bounded portable fallback.

### Task 14: Add Stable Greedy Promotion And Final Product Gates

**Files:**
- Create: `src/render/rt_greedy.rs`
- Modify: `src/render/rt_representation_router.rs`
- Modify: `src/render/rt_page_registry.rs`
- Modify: `docs/superpowers/specs/2026-07-08-rtx-native-dynamic-voxel-surface-as-design.md`
- Test: greedy, router, stress, and visual gates

- [ ] **Step 1: Write same-material greedy tests**

Only coplanar faces with identical material identity merge. Cross-page merges
remain disabled in this phase. Decode metadata must still identify the logical
owner voxel/face for all hit classes.

- [ ] **Step 2: Promote only stable pages**

After a measured quiet period, build `PREFER_FAST_TRACE` geometry out of place.
Any occupancy/material edit invalidates the promotion through the page state
machine and returns the page to Reference until exact recovery.

- [ ] **Step 3: Run final verification**

```powershell
.\run\validate-local.ps1 -StrictShaders
.\run\validate-visual-baseline.ps1 -Rt
.\tools\rt_flythrough_capture.ps1 -Mode rt -Frames 64 -CaptureFrames "2,16,31,63"
```

Additionally run the 300-frame destruction stress under Vulkan validation and
record bake-off CSVs for the target RTX device.

- [ ] **Step 4: Audit completion against the product goal**

Completion requires evidence that:

- normal primary, shadow, and GI rays use hardware triangle traversal for
  resident pages;
- dirty/failed pages use bounded Reference without stale overlap;
- edits do not wait device idle;
- resource retirement is fence-backed;
- CompactExact agreement is clean;
- optimized representations are capability- and benchmark-gated;
- all strict, stress, and visual gates pass.

Mega Geometry remains a separately planned NVIDIA-only backend which reuses
this registry and source truth. It is not required to claim the portable KHR
architecture complete.
