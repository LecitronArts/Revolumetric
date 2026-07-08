# RTX-Native Dynamic Voxel Surface AS Design

## Goal

Define a rendering-first architecture for Revolumetric that keeps voxel data as
the edit authority while moving primary visibility, shadows, reflections, and GI
as close as practical to RT Core-native triangle traversal.

The recommended target is:

```text
Voxel Authority
  -> Dirty RT Pages
  -> Volatile Degenerate-Triangle Surface Lattice BLAS
  -> Stable Compact Surface BLAS
  -> TLAS
  -> RT Primary / Shadow / Reflection / GI
```

This supersedes the procedural occupied-brick AABB RT path as the desired
long-term primary RT renderer, but the existing AABB RT and VPT paths should
remain available as debug/reference paths while the triangle AS path is proven.

## Definition Of "Almost Full RT Core"

This design does not claim that RT Cores directly trace voxel occupancy bits,
3D textures, SVO nodes, or brick-local voxel DDA.

The practical definition is:

- ray traversal should use TLAS/BLAS;
- visible voxel surfaces should be represented as triangle geometry;
- hit testing should use built-in ray-triangle intersection;
- hit shaders should fetch material and face metadata only;
- shader-side voxel traversal should not be part of the normal hit path.

Work that remains outside RT Core includes voxel edits, dirty tracking, face
activation, vertex/material buffer updates, BLAS build/update, TLAS build/update,
denoising, and shading.

## Current Facts

- Revolumetric already has an 8^3 voxel storage brick model through UCVH.
- The current hardware RT scene path builds BLAS geometry from occupied brick
  AABBs and then performs brick-local voxel/material evaluation in shader code.
- The current `src/render/rt_scene.rs` GPU update path already respects a key
  acceleration-structure constraint: BLAS and TLAS primitive counts must match
  for in-place update; otherwise it falls back to full rebuild or resource clear.
- The previous Versioned Render Proxy document recommended rasterized mesh
  proxies as the conventional first renderer. This document chooses a different
  target because the user explicitly wants an RTX-first, almost fully RT
  Core-native renderer.
- Physics, connectivity splitting, rigid fragments, structural simulation, and
  Mega Geometry integration are out of scope for the first implementation phase.

## External Constraints

Vulkan ray tracing acceleration structures are strongest when the scene is
represented as triangles. Procedural/custom primitives require shader
intersection work, which would move fine voxel hit testing back onto shader
cores.

Important API constraints:

- `VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR` must be selected on
  the initial build if the AS will later be updated.
- AS update cannot change primitive count, geometry count, geometry flags,
  vertex format, index format, or active/inactive primitive state.
- `PREFER_FAST_TRACE` prioritizes ray traversal performance over build time.
- `PREFER_FAST_BUILD` prioritizes build/update time over trace performance.
- `ALLOW_COMPACTION` enables a compacted copy after build; it can increase build
  cost and should be used only where the memory/tracing win is worth it.
- Vulkan inactive primitives, such as NaN-deactivated triangles, are not a safe
  face on/off mechanism for AS update because active/inactive transitions require
  a full rebuild.
- Degenerate triangles are different from inactive primitives. The design uses
  degenerate triangles as active placeholder slots, then updates their vertex
  positions into real triangles when a voxel face becomes visible.

The degenerate-slot behavior should be validated on target drivers before it is
used as a correctness-critical optimization. If a vendor behaves poorly with
large numbers of degenerate triangles, the fallback is smaller volatile pages
with fast rebuild instead of update.

## Recommended Architecture

Use two classes of triangle BLAS:

```text
Hot / Volatile Page
  8^3 voxels
  fixed potential face slots
  inactive face = degenerate triangles
  active face = real voxel face triangles
  flags = ALLOW_UPDATE | PREFER_FAST_BUILD

Cold / Stable Chunk
  16^3 or 32^3 voxels
  compact exposed faces / greedy quads
  no inactive face slots
  flags = PREFER_FAST_TRACE | ALLOW_COMPACTION
```

Hot pages absorb frequent destruction without changing primitive counts. Stable
chunks recover trace quality and memory efficiency after the area stops changing.

The renderer traces the same TLAS for primary visibility, shadow rays,
reflection rays, and GI rays. Current VPT and procedural AABB RT outputs remain
reference/debug outputs, not the main composition layer.

## Why 8^3 Hot Pages

Use 8^3 hot pages for the first version.

For a page with `N^3` voxels, potential voxel boundary quads are:

```text
3 * (N + 1) * N * N
```

Triangle counts are:

```text
8^3 page:
  3 * 9 * 8 * 8 = 1,728 quads = 3,456 triangles

16^3 page:
  3 * 17 * 16 * 16 = 13,056 quads = 26,112 triangles
```

A 16^3 hot lattice has 7.56x the potential triangle count of an 8^3 page. In a
heavy destruction scene, many hot pages may exist at once, so update granularity
and BVH quality matter more than reducing page object count. 8^3 also matches
the existing storage brick size, which reduces first-phase mapping risk.

Use 16^3 or 32^3 only for stable compact chunks after cooking removes inactive
face slots.

## Surface Lattice Model

Each 8^3 hot page owns all potential axis-aligned voxel boundary faces:

```text
X faces: (N + 1) * N * N
Y faces: N * (N + 1) * N
Z faces: N * N * (N + 1)
```

Each face slot owns two triangles. The first implementation should use a fixed
index buffer and six distinct vertex indices per face slot, even when inactive.
Do not rely on duplicate vertex indices to represent degenerate triangles.

Face activation rule:

```text
left_solid XOR right_solid -> active visible face
left_solid == right_solid  -> degenerate inactive face
```

For world boundaries:

```text
outside world = air
unallocated neighbor brick/page = air
```

For material boundaries:

```text
solid material A next to solid material B = no interior face in phase 1
```

Transparent, cutout, glass, and volumetric materials are out of scope for the
first version.

## Geometry Encoding

The hot-page vertex buffer should be update-friendly:

```text
HotSurfaceVertex
  position: vec3
  normal_or_face_axis: packed
  material_id: u32
  face_slot_id: u32
```

The fixed index buffer is created once per page template or shared across pages
when addressing permits.

For inactive faces:

- write all triangle vertices to the same position or to a zero-area line;
- keep finite coordinates;
- do not write NaNs;
- keep the primitive active from the API point of view.

For active faces:

- write the real voxel face quad as two triangles;
- orient winding consistently by face side;
- store material from the solid side;
- store face slot metadata so hit shaders can decode page-local voxel position,
  axis, side, and material lookup index.

The closest-hit shader must not perform brick-local voxel traversal. It may
decode:

```text
InstanceCustomIndex -> page or stable chunk record
PrimitiveID         -> face slot or compact face record
```

## Stable Compact Surface BLAS

When a hot page has been stable for a configured number of frames or edit
generations, schedule it for stable cooking.

Stable cooking should produce compact triangles:

1. exposed faces;
2. optional greedy quads for same-normal and same-material surfaces;
3. triangle buffer;
4. material/face metadata buffer;
5. BLAS with `PREFER_FAST_TRACE`;
6. optional compaction copy.

Stable chunks may aggregate multiple 8^3 storage bricks. Recommended rollout:

- first stable unit: 8^3, to prove identical output against hot pages;
- second stable unit: 16^3, to reduce TLAS instance count;
- later stable unit: 32^3 only if instrumentation shows instance/draw count is
  a real bottleneck.

## Dirty Tracking And State Machine

Each RT page has an explicit state:

```text
Missing
QueuedHotBuild
HotBuilding
HotValid
HotDirty
Cooling
QueuedStableCook
StableBuilding
StableValid
Failed
```

Rules:

- editing a voxel dirties the containing 8^3 page;
- editing a boundary voxel dirties the adjacent face-neighbor page;
- conservative first implementation may dirty all six face-neighbor pages;
- hot page updates replace geometry only after source generation validation;
- stable build output is discarded if source generation changed during cooking;
- old AS resources remain live until no in-flight frame can reference them.

Recommended hot/cold transition:

```text
edit -> HotDirty
HotDirty -> HotValid after vertex update + BLAS update or fast rebuild
HotValid -> Cooling after no edits for K frames
Cooling -> QueuedStableCook after K stable frames
StableBuilding -> StableValid after generation-matched BLAS build
new edit in StableValid -> replace or overlay with HotDirty page
```

Use K = 8 frames for the first runtime experiment. Tune by instrumentation, not
by preference.

## AS Update Strategy

Hot pages:

- build once with fixed triangle count;
- use `ALLOW_UPDATE | PREFER_FAST_BUILD`;
- update vertex buffer and record BLAS `UPDATE`;
- rebuild instead of update if driver validation or runtime telemetry shows poor
  update behavior;
- keep per-frame update budgets to avoid AS-build spikes.

Stable chunks:

- build from compact geometry;
- use `PREFER_FAST_TRACE`;
- use `ALLOW_COMPACTION` when the chunk is expected to remain stable long enough
  to repay build/copy cost;
- avoid `ALLOW_UPDATE` unless a stable chunk is intentionally maintained as an
  updatable dynamic asset.

TLAS:

- one TLAS instance per active hot page or stable chunk in phase 1;
- update TLAS when page/chunk instances or BLAS addresses change;
- consider batching adjacent stable chunks into fewer instances only after
  instrumentation shows TLAS instance count dominates.

## Scheduling And Budgets

The scheduler must explicitly budget:

- hot vertex-buffer update bytes;
- hot BLAS update count;
- hot fast-rebuild count;
- stable cook jobs;
- stable BLAS builds;
- compaction copies;
- TLAS updates;
- deferred AS resource frees.

First runtime policy:

```text
Per frame:
  1. process visible hot pages first;
  2. process near-camera hot pages second;
  3. defer far hot pages if over budget;
  4. run stable cooking only with leftover budget;
  5. never block presentation on optional stable compaction.
```

If a hot update cannot complete in budget, keep the previous generation visible
and expose the page in a debug overlay. Do not partially install a page.

## Expected Performance Shape

Compared with the current procedural AABB RT path:

- primary and secondary rays should avoid brick-local voxel DDA;
- triangle hit testing should move to built-in ray-triangle hardware;
- many shadow/reflection/GI rays should benefit from the same TLAS;
- AS update/build work becomes the main edit-time cost.

Compared with VPT:

- stable pixels should be less traversal-bound and less noisy;
- secondary ray cost should scale with triangle AS traversal instead of repeated
  voxel hierarchy traversal;
- image quality still depends on sampling, ReSTIR, temporal history, and
  denoising, but surface hit cost is moved closer to RT hardware.

Worst cases:

- large explosions that touch many pages in one frame;
- dense shattered geometry with high exposed surface area;
- many hot pages near the camera;
- scenes where degenerate slots greatly degrade hot-page BVH quality;
- repeated hot-to-stable churn before stable cooking can repay its cost.

## Metrics

Add metrics before optimizing:

- dirty voxel edits;
- dirty hot pages;
- hot pages updated, rebuilt, skipped, and failed;
- stable chunks cooked, built, compacted, skipped, and failed;
- active versus degenerate face count per hot page;
- hot BLAS update GPU time;
- stable BLAS build GPU time;
- TLAS update GPU time;
- trace GPU time per pass;
- memory used by hot vertex/index buffers;
- memory used by BLAS/TLAS resources;
- stale page count and oldest stale generation;
- hot-to-stable transition count;
- stable-to-hot invalidation count.

## Debug Views

Required debug views:

- page state;
- source generation;
- hot versus stable;
- active face density;
- BLAS build/update time heatmap;
- stale or over-budget pages;
- PrimitiveID-to-face decoding;
- material id;
- normal;
- current AABB RT or VPT comparison for simple scenes.

## Failure Handling

Failure behavior must be explicit:

- AS update failure: keep old page AS active, mark page `Failed`, and requeue a
  fast rebuild with a bounded retry count.
- Stable cook stale result: discard output and requeue if the page is still
  stable enough.
- Budget overflow: keep previous generation visible and show debug state.
- Empty world: keep a valid empty/fallback TLAS or route through the existing
  fallback path without using stale AS handles.
- Device without required ray tracing support: use VPT fallback as today.

The renderer must not install partially updated page geometry or partially built
AS resources.

## Alternatives Considered

### Procedural Brick AABB RT

Pros:

- existing code path;
- simple mapping from current UCVH bricks;
- direct voxel authority;
- fewer generated triangles.

Cons:

- fine voxel hit testing stays in shader;
- primary, shadow, reflection, and GI rays repeatedly pay voxel traversal;
- does not satisfy the almost-full RT Core goal.

### Pure Stable Triangle BLAS

Pros:

- cleanest RT Core hit path;
- compact and fast to trace after cooking;
- compatible with stable GI/reflection workloads.

Cons:

- frequent destruction changes primitive counts and forces rebuilds;
- edit latency can spike;
- not robust enough alone for heavy destruction.

### 16^3 Volatile Lattice

Pros:

- fewer pages and TLAS instances;
- matches common chunk sizes.

Cons:

- 26,112 potential triangles per page;
- too expensive for many simultaneous hot regions;
- worse first implementation risk than 8^3 pages.

### Opacity Micromap Hot Mask

Pros:

- can move alpha/opacity decisions away from any-hit shaders on supported GPUs;
- may reduce shader overhead for dense potential-face representations.

Cons:

- still carries potential geometry in AS;
- introduces another hardware feature and update path;
- should be measured after the triangle lattice path works.

### RTX Mega Geometry

Pros:

- designed for massive dynamic triangle geometry;
- cluster AS may reduce dynamic AS build pressure;
- good long-term fit for dense voxel surface clusters.

Cons:

- NVIDIA-specific;
- adds SDK/extension complexity;
- should be a separate mid/late optimization path, not the baseline design.

## Phased Delivery Recommendation

### Phase 1: Triangle AS Proof

- Build one 8^3 hot page as fixed degenerate-triangle lattice.
- Generate active voxel faces into the page vertex buffer.
- Build a triangle BLAS and trace primary rays.
- Validate output against current AABB RT/VPT on small scenes.

### Phase 2: Hot Page Updates

- Add dirty page mapping from voxel edits.
- Add generation validation.
- Add BLAS update for fixed-count hot pages.
- Add fallback fast rebuild when update is not possible or fails.

### Phase 3: Multi-Page TLAS

- Add TLAS instances for multiple hot pages.
- Add page state/debug views.
- Add scheduling budgets and deferred resource free.

### Phase 4: Stable Compact BLAS

- Add exposed-face and greedy stable cooker.
- Add hot-to-stable transition.
- Add stable BLAS compaction where worthwhile.
- Keep current hot page output as reference for stable chunks.

### Phase 5: RT Lighting Integration

- Route RT primary, direct-light visibility, reflections, and GI through the new
  triangle TLAS.
- Keep ReSTIR/temporal history contracts explicit.
- Use AABB RT/VPT only for reference and fallback.

### Phase 6: Optional Modern RTX Features

- Evaluate opacity micromaps for hot masks.
- Evaluate SER only after shaders have enough divergence to justify it.
- Evaluate RTX Mega Geometry as a separate NVIDIA-specific cluster AS backend.

## Validation

Required unit tests:

- 8^3 full solid page emits only outer active faces;
- empty page produces only degenerate faces;
- two adjacent full pages suppress the interior boundary;
- carved hole activates inward-facing faces;
- boundary edit dirties adjacent page;
- generation mismatch rejects a hot update result;
- stable cook result is rejected after source generation changes;
- PrimitiveID decodes to the expected axis, side, and voxel coordinate.

Required source/API tests:

- hot BLAS uses triangle geometry, not AABB geometry;
- hot BLAS uses `ALLOW_UPDATE` and `PREFER_FAST_BUILD`;
- stable BLAS uses `PREFER_FAST_TRACE`;
- stable compaction is only used on stable resources;
- AS update path preserves primitive count, geometry count, flags, vertex format,
  and index format;
- inactive NaN primitives are not used as face toggles.

Required runtime checks:

- hot page output matches current AABB RT/VPT on small reference scenes;
- continuous edits do not crash, install partial AS resources, or leave stale
  pages after budgets drain;
- per-frame AS update time is visible in GPU timings;
- stable conversion reduces triangle count and/or trace time in static scenes;
- fallback path remains selectable.

## Non-Goals

- Physics or structural destruction.
- Connectivity splitting.
- Rigid-body fragments.
- Transparent voxel material handling.
- SVO or sparse voxel DAG ray traversal as the primary renderer.
- Procedural AABB RT as the primary long-term renderer.
- Mega Geometry in the first implementation phase.
- Vendor-specific code paths before the standard KHR triangle BLAS path works.

## Research Basis

- Vulkan acceleration structure update rules and degenerate primitive behavior:
  https://docs.vulkan.org/spec/latest/chapters/accelstructures.html
- Vulkan build flags for update, compaction, fast trace, and fast build:
  https://docs.vulkan.org/refpages/latest/refpages/source/VkBuildAccelerationStructureFlagBitsKHR.html
- Vulkan ray tracing overview:
  https://docs.vulkan.org/guide/latest/extensions/ray_tracing.html
- NVIDIA Vulkan ray tracing overview:
  https://developer.nvidia.com/blog/vulkan-raytracing/
- Khronos Vulkan ray tracing best practices:
  https://www.khronos.org/blog/vulkan-ray-tracing-best-practices-for-hybrid-rendering
- NVIDIA dynamic BLAS compaction case study:
  https://developer.nvidia.com/blog/path-tracing-optimizations-in-indiana-jones-opacity-micromaps-and-compaction-of-dynamic-blass/
- NVIDIA RTX Mega Geometry Vulkan samples:
  https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples/

## Suggested Next Thread Prompt

Use this document to plan Phase 1 and Phase 2:

> Plan the implementation of the RTX-Native Dynamic Voxel Surface AS path for
> Revolumetric. Start with an 8^3 hot page fixed degenerate-triangle surface
> lattice, triangle BLAS build/update, generation validation, dirty-neighbor
> invalidation, minimal RT primary tracing, and comparison against the existing
> AABB RT/VPT reference paths. Do not implement physics, stable compact BLAS,
> opacity micromaps, SER, or RTX Mega Geometry yet.
