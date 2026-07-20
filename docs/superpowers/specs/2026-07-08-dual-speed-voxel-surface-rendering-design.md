# Dual-Speed Voxel Surface Rendering Design

## Goal

Design a rendering-first path for Revolumetric that supports small editable voxels with immediate visual feedback, while deferring physics, rigid fragments, connectivity splitting, and full destruction simulation.

The preferred long-term direction is a dual-speed renderer:

- Recently edited or otherwise unstable regions are rendered from live voxel data so visual feedback is immediate and authoritative.
- Stable regions are rendered from an incremental surface cache so primary visibility is fast, clean, and compatible with conventional raster and triangle ray tracing pipelines.

This document is intended as handoff context for a later implementation-planning thread. It is not an implementation plan.

## Current Facts

- Revolumetric is a Rust + Vulkan voxel rendering prototype with UCVH, brick storage, VPT, and a hardware RT backend.
- The voxel core uses 8^3 bricks. Each brick stores occupancy and material data, with dirty brick tracking and generation/invalidation metadata.
- GPU upload already has a dirty-data concept for voxel resources.
- The current hardware RT path builds acceleration structures from occupied brick AABBs.
- The current RT surface path traces into those AABBs, then validates real voxel hits in shader through brick-local traversal before reporting a hit.
- This makes the current RT path authoritative with respect to voxel data, but it pays traversal cost during rendering.
- Physics, falling fragments, rigid bodies, and connectivity splitting are intentionally out of scope for the next phase.

## Problem Statement

The current RT path is a good voxel-accurate renderer and reference path, but it is not the best primary path for small editable voxel surfaces if the near-term goal is stable real-time visual feedback.

The main tension:

- Pure voxel RT keeps one source of truth and naturally handles edits, but primary rays, shadow rays, and GI rays can repeatedly pay voxel traversal cost.
- Pure mesh surface cache can make primary rendering fast and stable, but edit-to-cache latency, stale mesh, cracks, and dirty-neighbor synchronization become correctness risks.

The desired design should keep voxel data authoritative while avoiding a hard choice between live voxel rendering and cached surface rendering.

## Recommendation

Use a dual-speed architecture:

```text
UCVH / Voxel Authority
  -> Dirty Region Tracker
      -> Live Voxel Layer
      -> Async Surface Compiler
          -> Surface Cache
              -> Raster / Visibility Buffer
              -> Optional Triangle BLAS
```

The live voxel layer handles regions that have changed recently or whose surface cache is stale. The surface cache handles stable regions. The renderer composites both layers with explicit ownership rules so the same surface is not drawn twice.

This is better than replacing the current RT path outright because the current voxel renderer remains valuable as:

- an authoritative live layer for dirty regions;
- a reference path for mesh-cache correctness;
- a debug path for voxel hit, brick, material, and traversal issues;
- a fallback when surface cache generation fails or is over budget.

## Alternatives Considered

### Option A: Keep Current RT As Primary

Current shape:

```text
occupied brick AABB AS -> intersection shader -> brick-local voxel traversal -> RT surface payload
```

Pros:

- Single authoritative voxel representation.
- Easy to reason about voxel correctness.
- Natural for voxel path tracing, shadow rays, GI rays, and debug views.
- Existing code already has a working path.

Cons:

- Primary visibility pays ray tracing and voxel traversal cost every frame.
- Secondary rays multiply traversal cost.
- Dynamic small-voxel scenes can become traversal-bound.
- Image stability depends on temporal behavior and ray tracing output quality.

### Option B: Pure Incremental Mesh Surface Cache

Current proposed shape:

```text
dirty voxel bricks -> exposed faces / greedy quads -> mesh cache -> raster
```

Pros:

- Fast and stable primary visibility.
- No primary-ray noise.
- Compatible with classic deferred rendering and later triangle RT.
- Edited surfaces can be rendered as ordinary geometry once compiled.

Cons:

- Mesh cache can become stale.
- Dirty brick updates require neighbor rebuilds at brick boundaries.
- Small fractured surfaces can create many faces.
- CPU mesh generation and buffer upload can bottleneck if implemented naively.
- It loses the directness of voxel traversal for live edits.

### Option C: GPU-Driven Meshlet Surface Cache

Current proposed shape:

```text
dirty brick queue -> compute face generation -> compaction -> meshlet cache -> indirect raster
```

Pros:

- More scalable than CPU surface generation.
- Meshlets fit GPU culling and indirect draw.
- Good foundation for later cluster LOD and triangle RT.

Cons:

- Requires GPU allocation, compaction, free-list or arena management, and indirect draw plumbing.
- Harder to debug than CPU-generated cache.
- Still needs a solution for immediate edit feedback before meshlets are ready.

### Option D: Dual-Speed Live Voxel + Stable Surface Cache

Recommended shape:

```text
dirty / stale regions -> live voxel renderer
stable regions -> surface cache renderer
background work -> compile dirty regions into surface cache
```

Pros:

- Immediate edits are visible from authoritative voxel data.
- Stable regions get fast rasterized primary visibility.
- Cache compilation can be budgeted over multiple frames.
- Existing RT/VPT work remains useful.
- Later GPU meshlet generation and virtualized surface cache can be added without changing the authority model.

Cons:

- Two rendering layers must be composited consistently.
- Cache generation state must be tracked precisely.
- The transition from live voxel to cached surface must avoid popping, double draw, and holes.
- More debug tooling is required.

## Core Concepts

### Voxel Authority

The UCVH remains the source of truth for voxel occupancy, material, brick generation, and invalidation state.

Surface caches are derived data. They must never be treated as authoritative for editing or material ownership.

### Dirty Region Tracker

The dirty tracker converts voxel edits into rendering work:

- source dirty brick set;
- expanded surface dirty set including the 6 face-neighbor bricks;
- generation numbers for cache validation;
- live-region masks for regions whose cache is missing, stale, or under compilation.

The 6-neighbor expansion is required because an exposed face in one brick can disappear or appear when an adjacent brick changes.

### Live Voxel Layer

The live layer renders authoritative voxel data for regions where the surface cache is not valid.

Acceptable initial implementation:

- use current RT/VPT surface path for the full frame while the cache path is being developed;
- then restrict live rendering to dirty/stale regions after masking support exists.

The live layer should be treated as correctness-first, not performance-first.

### Surface Cache

The first surface cache should be simple and brick-local:

- generate faces for solid voxels whose neighbor is air;
- do not generate interior faces between solid voxels;
- support cross-brick neighbor lookup;
- treat world-outside as air;
- store per-brick or per-chunk mesh entries with source generations.

Greedy meshing, meshlets, and LOD are follow-up optimizations. They should not be required for the first correctness pass.

### Surface Compiler

The surface compiler converts dirty/stale voxel regions into mesh cache entries.

Recommended rollout:

1. CPU exposed-face compiler for correctness and tests.
2. Raster path consuming CPU-generated cache.
3. Budgeted rebuild queue with dirty-neighbor expansion.
4. GPU exposed-face compiler.
5. Meshlet packing and indirect draw.
6. Optional cluster LOD and virtualized cache.

### Stable Surface Renderer

The stable surface renderer draws valid cache entries. The first version can use a simple raster pass that writes a G-buffer or display target.

Later versions can move to:

- deferred G-buffer;
- visibility buffer with cluster/primitive IDs;
- meshlet indirect draw;
- cluster LOD;
- stable triangle BLAS for RT shadows, reflections, and GI.

### RT Core Role

The surface cache is not automatically an RT Core scheme.

RT Core becomes useful after generated surfaces are represented as triangles and built into BLAS/TLAS:

- stable cached surfaces can use triangle BLAS;
- dirty live regions can remain voxel-rendered until their mesh cache is ready;
- very frequently changing regions can skip BLAS updates to avoid acceleration-structure spikes.

The current procedural AABB RT path should remain available for voxel-accurate live rendering and validation.

## Rendering Composition

The renderer needs a clear ownership rule for each screen pixel or region:

- If a cache entry is valid for the current voxel generation, the stable surface layer may render it.
- If a cache entry is missing or stale, the live voxel layer owns that region.
- If both layers can render the same region during transition, stable cache should only replace live output after the replacement cache entry is complete and generation-matched.

Possible composition strategies:

### Region Mask

Build a screen-space or world-region mask for live dirty regions. Render stable mesh normally, then render live voxel regions with depth testing and region restriction.

Pros: simple to reason about.

Cons: region masks can be conservative and overdraw more pixels than necessary.

### Cache Ownership Per Brick

Each brick or chunk has a cache state:

- `valid`
- `dirty`
- `building`
- `failed`
- `disabled`

The draw list includes only `valid` cache entries. The live renderer covers all non-valid entries.

Pros: maps directly to voxel generations.

Cons: live renderer must be able to restrict itself to non-valid entries to avoid full-frame cost.

### Visibility Buffer Transition

Longer term, render cached surfaces into a visibility buffer and render live voxel surfaces into the same buffer under explicit depth and ownership rules.

Pros: clean long-term model for shading and debug.

Cons: too much for the first implementation phase.

Recommended first implementation: cache ownership per brick/chunk, with full-frame live fallback until region-restricted live rendering is available.

## Data Model Sketch

The exact Rust names can change, but the implementation plan should preserve these concepts.

```text
SurfaceCache
  entries: map BrickCoord -> SurfaceCacheEntry
  rebuild_queue: Vec<BrickCoord>
  stale_regions: Vec<BrickCoord>
  vertex_buffer
  index_buffer
  draw_ranges
  stats
```

```text
SurfaceCacheEntry
  brick_coord
  brick_id
  source_generation
  state
  vertex_offset
  vertex_count
  index_offset
  index_count
  bounds
```

```text
SurfaceVertex
  position
  normal
  material_id
  optional_light_or_ao
```

The first version may use CPU vectors and whole-buffer upload. Later versions should move to an arena or free-list so dirty entries can be replaced without rebuilding the entire cache.

## Initial Correctness Rules

- A face is emitted when the current voxel is solid and the neighbor voxel is air.
- A neighbor outside the world is air.
- A neighbor in an unallocated brick is air.
- A neighbor solid voxel suppresses the face, even if material differs.
- Material-boundary interior faces are not emitted in the first version.
- Transparent, cutout, glass, and volume materials are out of scope for the first version.
- Editing a brick marks that brick and its 6 face-neighbor bricks stale for surface cache purposes.
- A cache entry is valid only when its stored generation matches the current relevant voxel generation.

## Phased Delivery Recommendation

### Phase 1: CPU Exposed-Face Cache

Purpose: prove correctness.

Deliverables:

- CPU surface compiler for one brick/chunk.
- Cross-brick neighbor lookup.
- Unit tests for full brick, empty brick, adjacent bricks, and carved holes.
- Debug stats for generated faces and dirty cache entries.

### Phase 2: Raster Surface Path

Purpose: display cache output.

Deliverables:

- Vertex/index buffers for surface cache.
- Basic raster pass with material color and normal.
- Draw list from valid cache entries.
- Debug view showing cached vs live/stale regions.

### Phase 3: Edit-Driven Dirty Rebuild

Purpose: make it incremental.

Deliverables:

- Dirty brick expansion to 6-neighbors.
- Per-entry source generation checks.
- Rebuild queue.
- Whole-buffer upload is still acceptable if correctness is proven.

### Phase 4: Dual-Speed Composition

Purpose: combine live voxel and cached surface rendering.

Deliverables:

- Stable cache draws valid entries.
- Live voxel renderer covers missing/stale entries.
- Clear transition rules to avoid double draw and holes.
- Debug views for cache state and generation mismatch.

### Phase 5: GPU Surface Generation

Purpose: scale beyond CPU meshing.

Deliverables:

- Compute shader face generation for dirty bricks.
- Append or prefix-sum compaction.
- GPU-visible draw ranges.
- CPU path remains as reference.

### Phase 6: Meshlet / Cluster Cache

Purpose: support larger scenes.

Deliverables:

- Pack generated faces into bounded clusters or meshlets.
- GPU culling.
- Indirect draw.
- Optional cluster LOD/error metrics.

### Phase 7: Triangle RT Integration

Purpose: use RT Core where it is strongest.

Deliverables:

- Build BLAS for stable cached surfaces.
- Use triangle RT for shadows, reflections, or GI.
- Keep live voxel RT for dirty regions and validation.
- Budget BLAS updates to avoid edit-time spikes.

## Validation Strategy

Required early tests:

- Empty brick emits no faces.
- Full 8^3 brick emits only the outer surface.
- Two adjacent full bricks emit no interior boundary faces.
- A carved hole emits inward-facing surfaces correctly.
- Editing one brick invalidates the brick and its 6 face-neighbors.
- Cache entry generation mismatch prevents stale rendering.
- Surface cache output and voxel RT output agree on simple scenes.

Required runtime checks:

- Continuous edits do not leave stale surfaces.
- Dirty rebuild count is bounded and visible in debug UI.
- Full-frame fallback still works if cache compilation fails.
- Cache and live layer do not double draw the same stable region.
- Switching from live to cached output does not visibly pop for unchanged geometry.

Useful debug views:

- cache state per brick;
- dirty/stale/building regions;
- generated face count;
- material id;
- normal;
- source generation;
- live-vs-cached layer ownership.

## Risks

- The dual-speed architecture is more complex than either pure RT or pure mesh cache.
- Region ownership bugs can cause double draw, holes, or one-frame popping.
- CPU meshing can bottleneck if used too long beyond the correctness phase.
- GPU meshing adds allocator, compaction, and synchronization complexity.
- Surface caches can hide voxel authority bugs if the RT/live reference path is removed too early.
- Triangle BLAS updates can spike after large edits unless they are budgeted or restricted to stable regions.

## Non-Goals

- Rigid-body fragments.
- Connectivity splitting.
- Structural simulation.
- Full Teardown-style game mechanics.
- Full global path tracing as the first surface-cache milestone.
- Transparent or semi-transparent voxel material handling.
- Cross-brick greedy meshing in the first implementation.

## Suggested Starting Point For The Next Thread

Start with a focused implementation plan for Phase 1 and Phase 2:

1. Add a CPU exposed-face surface compiler beside the voxel module or render module.
2. Define minimal surface vertex/index structs and cache-entry metadata.
3. Add unit tests for face emission and cross-brick neighbor rules.
4. Add a simple raster pass or debug path to draw generated cache output.
5. Keep current RT/VPT paths untouched as reference renderers.

The first implementation should deliberately avoid GPU generation, meshlets, LOD, triangle BLAS, and dual-layer composition. Those belong after the surface cache is correct and visible.

