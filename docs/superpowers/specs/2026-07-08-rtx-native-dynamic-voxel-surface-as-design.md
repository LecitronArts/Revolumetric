# Surface-Mask Driven RT Core Voxel Rendering Design

## Status

Selected target architecture. This revision supersedes both earlier revisions
of this document:

- the shared interface-lattice BLAS with normal-path any-hit validation; and
- the `512 voxels * 6 faces * 2 triangles = 6,144 triangles` fixed hot-page
  BLAS as the only selected dynamic representation.

The durable UCVH upload, page invalidation, source-version, frame-slot, and
deferred-retirement work already started in the repository remains valid. The
geometry choice changes; the authority and synchronization requirements do not.

The implementation is complete only when the normal primary, direct-shadow,
reflection, and GI paths trace triangle or opacity-micromap page geometry,
committed edits become visible without stale geometry, bounded procedural
fallback is observable, and the required stress and visual gates pass.

## Product Goal

Render small, editable, opaque voxels with frequent local destruction while
using RT hardware for normal surface traversal and built-in triangle
intersection. Physics, connectivity, and rigid fragments are intentionally
deferred, but rendering must be suitable for adding those systems later.

"Almost full RT Core" means:

- TLAS/BLAS owns normal scene traversal;
- normal surface candidates are triangles or OMM microtriangles;
- no brick-local DDA or custom intersection shader runs on the normal path;
- fully opaque/transparent OMM states do not invoke any-hit;
- closest-hit performs bounded metadata and direct voxel lookups only.

Voxel edits, mask generation, uploads, AS builds, scheduling, shading,
temporal reuse, and denoising remain CPU/GPU work. RT Cores do not edit voxels
or build acceleration structures.

## Current Repository Facts

- UCVH stores sparse `8^3` bricks and remains the sole authority for occupancy,
  material, and generation data.
- The current RT scene builds one procedural AABB BLAS containing occupied
  bricks and one TLAS instance.
- Current RT intersection shaders run an `8^3` brick DDA before `ReportHit`.
- Durable render-change snapshots and frame-slot UCVH staging are partially
  implemented, but initial-upload submission commit/catch-up and runtime
  capacity recovery are not complete.
- The local RTX 4060 Laptop reports KHR AS/RT pipeline support,
  `VK_EXT_opacity_micromap`, OMM subdivision level 12, and advertises the
  Cluster AS and PTLAS extensions. KHR indirect AS build is not supported.
- Current `ash 0.38` contains EXT OMM bindings but not the current Cluster
  AS/PTLAS API surface required by the advertised extension revisions.

These are implementation inputs, not performance claims.

## Selected Architecture

```text
UCVH authority
  -> durable typed edit batch
  -> incremental authority upload
  -> SurfaceMask page cache
  -> representation router
       CompactExact: real exposed faces, KHR triangle BLAS BUILD
       HotOmm:       27 carrier sheets, OMM BUILD + tiny triangle BLAS
       CompactGreedy:stable merged faces, FAST_TRACE triangle BLAS
       HotInterface: corrected 1,728-quad fixed UPDATE fallback
       Reference:    bounded procedural AABB + DDA recovery
  -> versioned page registry and frame-safe TLAS slots
  -> primary / shadow / reflection / GI
```

The architecture separates four concerns:

1. UCVH says which voxels exist.
2. `SurfaceMaskPage` says which solid-owned faces are exposed.
3. A representation backend turns that mask into traceable hardware geometry.
4. The page registry atomically selects exactly one traceable representation.

No ray pass may infer a different ownership model.

## Surface Mask Contract

### Directional Owner Masks

Each `8^3` page caches six 512-bit masks:

```text
SurfaceMaskPage
  negative_x[512]
  positive_x[512]
  negative_y[512]
  positive_y[512]
  negative_z[512]
  positive_z[512]
```

The packed payload is 3,072 bits, or 384 bytes before alignment.

A bit is set exactly when:

```text
owner voxel is solid && face-neighbor voxel is air
```

Air does not own or emit a face. A missing brick and a position outside the
world are air. Solid/solid material boundaries do not emit geometry in this
phase.

The mask is object/world-space derived data. It is not screen-space visibility,
occlusion culling, or greedy meshing.

### Incremental Changes

One occupancy edit can alter the six faces of the edited voxel and the six
opposite faces of its neighbors. Implementations may recompute an entire `8^3`
mask with bit operations when a batch is dense, but the public contract remains
face-local and deterministic.

`UcvhChangedBrick` must distinguish:

- occupancy changed;
- material-only changed;
- generation/lifetime changed; and
- which of the six brick boundaries an occupancy edit touched.

Material-only changes upload material data but do not rebuild surface geometry.
An adjacent page is invalidated only when an occupancy change touches the
shared brick boundary. Bulk replacement without precise locality conservatively
sets all six boundary bits.

### Source Stamp

Every derived page result records:

- owner brick coordinate, ID, generation, and non-wrapping revision;
- six neighbor generations or absent sentinels;
- surface-mask revision; and
- representation build generation.

A result is installed only when every dependency still matches authority.

## Portable KHR Baseline: Compact Exact Pages

`CompactExact` is the first fully usable triangle backend and the correctness
baseline for every optimized backend.

It contains only currently exposed owner faces. There are no hidden,
transparent, NaN, or finite-degenerate face slots.

### Geometry Layout

All pages share one immutable local `9 * 9 * 9 = 729` vertex lattice covering
coordinates `[0, 8]^3`. Each page version owns:

- a packed `u16` triangle index stream referencing the shared lattice;
- one compact face record per emitted quad;
- a page-local triangle BLAS; and
- a TLAS instance transform translating local page coordinates to world space.

Two consecutive triangles represent one face record. The face record stores
the solid owner local coordinate and direction. Material remains authoritative
in UCVH and is fetched on hit.

Occupancy topology changes use a full BLAS `BUILD` with
`PREFER_FAST_BUILD`. Primitive count is known on the CPU from the surface mask,
so the local device's missing KHR indirect-build feature is not a blocker.

The first implementation emits one quad per exposed face. Same-material greedy
merging is a separate stable representation and must not obscure exact-page
correctness or benchmark results.

### Versioned Replacement

A dirty page builds a replacement index/face/BLAS version in a free slot. The
old version remains traceable by already submitted frames. The current frame's
TLAS switches to the new version only after its build dependency is recorded.
The old version enters a fence-backed deferred-retire queue.

This avoids mutating a BLAS still read by another frame and avoids global
device-idle waits during ordinary edits.

## RTX Hot Backend: 27-Plane Opacity Micromaps

`HotOmm` is a capability-gated, benchmark-gated representation for pages that
change frequently. It is semantically valid but is not presumed faster before
measurement.

### Carrier Geometry

An `8^3` page has:

```text
3 axes * (7 internal planes + 2 owner boundary planes) = 27 sheets
27 sheets * 2 base triangles                         = 54 BLAS primitives
```

Each base triangle carries one level-3 two-state opacity micromap containing 64
microtriangles. The page therefore represents:

```text
54 * 64 = 3,456 logical microtriangles
```

Those microtriangles cover the same 1,728 interface cells as the corrected
interface topology. Opaque bits represent exposed surface halves; transparent
bits are ignored by traversal.

Internal sheets set opacity when the two adjacent voxels differ. The closest
hit reads the two direct neighbors to choose the solid owner and outward normal.
At a page boundary, only the local-solid page marks its outward face opaque.
This avoids an empty-page dependency and duplicate cross-page hits.

Carrier triangles are double-sided. Ray flags must not cull front or back
faces. Carrier winding is not the voxel normal.

### OMM Encoding And Update

OMM input is encoded in Vulkan's triangular space-filling order, not row-major
order. A tested lookup table/reference encoder maps page `(sheet, u, v,
triangle-half)` to the required bit.

All carrier triangles use explicit OMM entries from their first build. A page
must not switch between a fully-transparent special index and an explicit OMM
entry during an AS data update.

Two legal update candidates must be benchmarked:

1. build a replacement micromap and a replacement 54-triangle BLAS;
2. build a structurally identical replacement micromap and perform BLAS
   `UPDATE` after an initial build with `ALLOW_UPDATE` and
   `ALLOW_OPACITY_MICROMAP_DATA_UPDATE_EXT`.

EXT micromaps have BUILD but no micromap UPDATE mode. Both candidates rebuild
micromap data. BLAS data-update saves only the 54-triangle full build and may
not win once command, barrier, and frame-versioning costs are included.

### OMM Selection Gate

OMM becomes the normal hot backend only if the target workload shows a clear
combined update-plus-trace win over `CompactExact` and `HotInterface` without
exceeding memory or backlog limits. Extension presence alone is insufficient.

## Corrected Fixed Interface Backend

`HotInterface` is the portable bounded-update alternative if compact rebuilds
miss the destruction budget and OMM is unsupported or slower.

It reserves one slot per page-local interface rather than one slot per oriented
voxel face:

```text
3 * (8 - 1) * 8 * 8 internal interfaces = 1,344 quads
6 * 8 * 8 owner boundary interfaces     =   384 quads
total                                    = 1,728 quads
                                           3,456 triangles
```

For an internal interface, `00` and `11` collapse to a finite degenerate quad;
`10` and `01` activate the same slot with an explicit owner-side bit and
appropriate vertex winding. Boundary slots activate only for
`local solid && remote air`.

This design does not require an empty neighbor page or normal-path any-hit.
The previous document's rejection applied to a shared-boundary XOR variant
that activated duplicate copies, not to this corrected ownership model.

The backend uses fixed indices and vertex count, finite degenerate positions,
`ALLOW_UPDATE`, and `PREFER_FAST_BUILD`. NaN inactive primitives are forbidden.

## Stable Greedy And Cluster Representations

After a configurable quiet period, `CompactGreedy` may merge coplanar,
same-material faces within an owned page or cold aggregate. It uses
`PREFER_FAST_TRACE` and a full build. It is installed through the same source
stamp and versioned replacement protocol.

Portable aggregation is optional until per-page correctness and instance costs
are measured. Disabling an aggregate and enabling its children must occur in
one ownership transition; stale aggregate triangles must never coexist with a
replacement page.

NVIDIA Mega Geometry is a later backend:

- CLAS/templates accelerate construction of frequently changing triangle
  clusters but do not update CLAS in place;
- Cluster BLAS references CLAS objects and is rebuilt from their bounds;
- PTLAS partitions a large instance hierarchy so local instance edits do not
  rebuild the whole TLAS;
- official samples show faster construction can trade against slower tracing,
  so static cold content may remain traditional triangle BLAS.

Mega Geometry reuses `SurfaceMaskPage`, page records, source stamps, and
ownership transitions. It must not introduce a separate voxel truth model.

## Page Ownership State Machine

At a traceable boundary, exactly one representation owns a page:

```text
Missing
Reference
CompactExactBuilding -> CompactExact
HotOmmBuilding       -> HotOmm
HotInterfaceBuilding -> HotInterface
CompactGreedyBuilding -> CompactGreedy
Failed -> Reference
```

Rules:

- a committed edit invalidates stale derived ownership before it can trace;
- authority upload and replacement geometry are acknowledged only after the
  submission that makes both visible succeeds;
- a build failure leaves the durable batch retryable and routes the page to an
  explicit reference or missing diagnostic state;
- a stale asynchronous result is discarded, never patched into current state;
- no normal representation overlaps another representation for the same owner
  face;
- resource exhaustion has a bounded, reported fallback rather than stale data.

## TLAS And Frame Safety

The TLAS uses a fixed-capacity page-slot store for in-place updates. An unused
slot references a valid dummy BLAS with mask zero. A live slot carries:

- representation kind and hit-group selector;
- page-record index through `InstanceCustomIndex`;
- page transform;
- current BLAS address; and
- generation/debug identity.

TLAS instance inputs and output are frame-versioned or protected by completed
frame fences. Page buffers, OMMs, BLAS storage, and records are retired only
after the final referencing frame completes.

Capacity growth is explicit: finish frames that can reference the old arena,
allocate a larger version, rebuild descriptors/registries, re-upload current
authority, and retain pending durable batches until catch-up completes.

Ordinary page edits must not call `wait_idle` or wait all other frame fences.

## Shader And Hit ABI

Normal triangle hit groups contain closest-hit only. The procedural reference
group retains intersection plus closest-hit and uses a distinct SBT record.

Every normal backend produces the same logical hit identity:

```text
SurfaceKey
  page coordinate / page record
  solid owner BrickId
  solid owner local voxel
  face direction
  material generation
```

`CompactExact` and `CompactGreedy` decode a compact face record.
`HotInterface` decodes `(axis, plane, u, v)` plus owner-side metadata.
`HotOmm` decodes carrier triangle plus standard barycentrics/object hit position
and performs bounded adjacent-occupancy lookup.

Primary, shadow, reflection, and GI must share this ABI and the same TLAS. Ray
origin offsets and `TMin` handle self-intersection; the normal path must not
reintroduce a voxel-skipping any-hit shader.

## Scheduling And Backpressure

Priority order:

```text
1. durable authority upload and initial catch-up
2. visible dirty SurfaceMask pages
3. replacement page geometry/OMM and BLAS
4. current-frame TLAS ownership transition
5. visible exact-page recovery
6. stable greedy promotion
7. cold aggregation and Mega Geometry work
```

The renderer owns bounded queues and arenas. When the high-priority queue cannot
finish a page before trace, that page uses the reference backend or an explicit
missing diagnostic. It never traces the previous stale proxy.

Required telemetry:

- typed edits and boundary-neighbor invalidations;
- dirty mask pages and pending age;
- compact face counts and OMM opaque density;
- geometry/micromap generation time;
- BLAS BUILD/UPDATE and TLAS time;
- primary, shadow, and GI trace time;
- page state counts and ownership transitions;
- arena use, deferred-retire bytes, and capacity growth;
- fallback count, cause, duration, and oldest pending age;
- p50/p95/p99 burst-drain latency.

## Required Representation Bake-Off

The production backend choice must be based on the same page registry and ray
workload. Benchmark:

1. current procedural AABB plus brick DDA;
2. `CompactExact` triangle BLAS BUILD;
3. corrected `HotInterface` BLAS UPDATE;
4. `HotOmm` fresh BLAS build;
5. `HotOmm` opacity-data BLAS update.

Use `4^3` and `8^3` hot granularity only in the benchmark. The product baseline
remains `8^3` until evidence justifies introducing render subpages.

Workloads:

- isolated voxel, full page, checkerboard, thin walls, cavities, and tunnels;
- solid/solid and solid/air cross-page boundaries;
- sparse shattered and dense architectural scenes;
- repeated one-page edits;
- moving destruction fronts;
- bursts affecting 1, 8, 32, and 128 pages;
- primary-only, primary plus shadow, and primary plus GI ray mixes.

Report build/update, trace, CPU submission, memory, active/transparent/degenerate
density, and p50/p95/p99. No FPS multiplier is accepted without this evidence.

## Verification Gates

### CPU And Contract Tests

- empty/full/checkerboard/cavity masks are exact;
- every set mask bit has a solid owner and air neighbor;
- cross-page interfaces emit exactly one owner face;
- material-only edits do not schedule mask or AS work;
- interior edits do not invalidate unrelated neighbor pages;
- stale source stamps are rejected;
- initial upload becomes ready only after submission and incremental catch-up;
- capacity recovery preserves pending batches;
- page ownership never enables two normal representations.

### GPU Agreement

For deterministic and random rays, compare each normal representation with the
procedural reference for:

- hit/miss;
- first-hit distance within an explicit tolerance;
- owner voxel and face direction;
- material and emissive identity;
- primary, shadow, and GI first-hit agreement.

Integer-grid edge/corner ties may select a different geometrically equivalent
face, but the allowed rule must be explicit and temporal identity stable.

### Runtime And Visual Gates

- edit during upload, mask generation, page build, TLAS update, and promotion;
- sustained destruction does not show stale, duplicate, or missing surfaces
  after the bounded fallback policy;
- no normal shader contains brick DDA, `ReportHit`, or `IgnoreHit`;
- no ordinary edit waits the entire device;
- RT-off and unsupported-device paths remain functional;
- desktop captures cover static, active destruction, fallback, OMM/compact
  transition, shadow, and GI views;
- required local, strict-shader, and RT visual validation scripts pass.

## Delivery Phases

1. Finish submission-safe UCVH initial upload, catch-up, and capacity recovery.
2. Implement typed edits, precise boundary invalidation, and CPU/GPU
   `SurfaceMaskPage` agreement.
3. Build the representation bake-off harness.
4. Make `CompactExact` the first end-to-end triangle RT path for primary rays.
5. Add the page registry, frame-safe TLAS slots, resource retirement, and all
   ray classes.
6. Implement and gate `HotOmm`; retain `HotInterface` only if measurements need
   it.
7. Add stable greedy promotion, stress telemetry, and visual verification.
8. Add NVIDIA CLAS/PTLAS behind a separate capability/backend boundary.

Each phase must leave a usable, retry-safe path. The full product goal is not
complete until phases 1 through 7 and their verification gates are satisfied.

## Research Basis

- Vulkan AS update and degenerate primitive rules:
  https://docs.vulkan.org/spec/latest/chapters/accelstructures.html
- Vulkan opacity micromaps and traversal semantics:
  https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_opacity_micromap.html
- NVIDIA OMM sample implementation:
  https://github.com/NVIDIA-RTX/OMM-Samples
- NVIDIA RTX Mega Geometry SDK:
  https://github.com/NVIDIA-RTX/RTXMG
- NVIDIA animated Cluster AS sample and build/trace tradeoffs:
  https://github.com/nvpro-samples/vk_animated_clusters
- NVIDIA PTLAS sample:
  https://github.com/nvpro-samples/vk_partitioned_tlas
