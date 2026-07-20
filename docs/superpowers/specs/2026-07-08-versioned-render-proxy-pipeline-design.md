# Versioned Render Proxy Pipeline Design

## Goal

Define a more intuitive and industrial rendering architecture for small editable voxel worlds in Revolumetric.

The recommended direction is:

```text
Voxel Authority
  -> Dirty Render Chunks
  -> Mesh Cook Queue
  -> Versioned Render Proxies
  -> Raster / Deferred Renderer
  -> Optional Stable-Proxy Triangle RT
```

This replaces the earlier dual-speed live-voxel plus surface-cache composition as the primary recommendation. The live voxel RT/VPT paths should remain as references and debug tools, but the first implementation should make rasterized render proxies the normal rendering product.

## Current Facts

- Revolumetric already has a voxel authority model through UCVH, brick occupancy, brick material data, dirty bricks, and generation/invalidation metadata.
- The current voxel storage unit is an 8^3 brick. That is a good edit/storage granularity, but it is likely too small as the final render-proxy granularity.
- The current hardware RT path builds acceleration structures from occupied brick AABBs and validates real voxel hits in shader through brick-local traversal.
- That RT path is voxel-accurate and useful as a reference, but it is not the most conventional primary rendering path for editable block surfaces.
- Physics, rigid fragments, connectivity splitting, and structural simulation remain out of scope for this rendering phase.

## External Evidence

Open-source voxel projects point toward chunked mesh/proxy rendering rather than primary per-pixel voxel traversal as the normal real-time rendering path.

- Luanti/Minetest splits the world into 16x16x16 MapBlocks and uses client-side mesh objects for those blocks. This supports the idea of a chunk-level render proxy rather than drawing directly from every voxel every frame.
- Godot Voxel Tools describes a realtime editable voxel terrain approach where voxel data is converted into polygon meshes, with heavy work moved off the main thread where possible.
- `block-mesh-rs` exposes both simple visible-face generation and greedy quad meshing, and its examples use chunk data with boundary padding. This directly matches the need for cross-chunk neighbor queries during mesh cooking.
- Vercidium's Sector's Edge voxel meshing work uses 32x32x32 chunks and separates mesh generation from buffering. It also prioritizes generation throughput over producing the absolute minimum triangle count, which is a useful production tradeoff.

These projects do not prove one exact implementation for Revolumetric, but they do support the broad architecture: authoritative voxel data, dirty chunks, mesh cooking, and renderable proxy assets.

## Problem Statement

The earlier dual-speed proposal solved edit immediacy by rendering dirty regions directly from live voxel data while stable regions used cached surfaces. That is powerful, but it introduces a complicated compositing problem:

- two rendering layers;
- ownership masks;
- duplicate-depth risks;
- live-to-cached transition artifacts;
- more debug states before the basic surface renderer is proven.

A more industrial architecture is to separate responsibility:

- voxel data answers what exists;
- the cooker answers how it becomes renderable;
- the render proxy store answers which derived asset is current;
- the renderer draws only valid render proxies.

This is a cleaner mental model and a better first implementation target.

## Recommendation

Build a Versioned Render Proxy Pipeline.

Voxel edits should not directly change rendering buffers. Instead, edits mark render chunks dirty. A cooker rebuilds the affected render chunks, produces new render proxies, and atomically swaps them into the render proxy store when complete.

The renderer consumes only valid proxies. It does not need to understand voxel edit details.

### Primary Flow

```text
edit voxel data
  -> record dirty storage bricks
  -> map dirty storage bricks to render chunks
  -> expand dirty render chunks by neighbor dependency
  -> cook new render proxies
  -> validate source generation
  -> atomically replace old proxy
  -> draw valid proxies
```

### Role Of Current RT/VPT

The current RT/VPT paths should remain valuable, but not as the first proxy renderer's composition layer.

Recommended roles:

- visual reference for proxy correctness;
- debug path for voxel hit, material, brick, and traversal issues;
- optional fallback if proxy cooking fails;
- later RT path for shadow/reflection/GI once stable triangle proxies exist.

## Render Granularity

Keep the existing 8^3 brick as a storage and edit unit.

Introduce a larger render chunk as the cooking unit. Candidate sizes:

- 16^3 voxels: closer to Luanti/Minetest style, smaller edit latency, more proxy objects.
- 32^3 voxels: closer to Sector's Edge style, fewer proxy objects, higher rebuild cost per edit.

Recommended first implementation: 16^3 render chunks.

Reasons:

- it maps cleanly to eight existing 8^3 storage bricks;
- it keeps first rebuilds small and easier to debug;
- it provides room to move to 32^3 later if draw/proxy counts dominate.

## Neighbor And Halo Rules

Surface cooking needs neighbor access.

Each render chunk should cook with a one-voxel halo around its owned volume. The halo can be fetched from neighboring storage bricks or render chunks.

Rules:

- Owned voxels generate faces.
- Halo voxels only answer neighbor occupancy.
- A face is emitted when an owned voxel is solid and the adjacent voxel is air.
- A neighbor outside the world is air.
- A neighbor in an unallocated brick is air.
- Interior faces between solid voxels are suppressed.
- Material-boundary interior faces are not emitted in the first version.

Dirty expansion:

- Editing a storage brick dirties the render chunk containing it.
- If the edit touches a render-chunk boundary, the adjacent render chunk is also dirtied.
- A conservative first implementation may always dirty the 6 face-neighbor render chunks.

The conservative rule is simpler and safer. It can be narrowed after instrumentation shows unnecessary rebuilds matter.

## Render Proxy Model

Each render chunk owns at most one active proxy.

```text
ChunkRenderProxy
  chunk_coord
  source_generation
  state
  bounds
  vertex_range
  index_range
  material_range
  triangle_count
  face_count
  last_used_frame
```

States:

- `Missing`: no proxy exists yet.
- `Queued`: dirty and scheduled for cooking.
- `Building`: cooker is producing a replacement.
- `Valid`: proxy matches current voxel generation.
- `Failed`: last cook failed; renderer should use fallback behavior or skip the chunk.

Atomic replacement:

- The old proxy remains active while a replacement is being built.
- The replacement is installed only after source generation validation.
- If source generation changed during cooking, discard the result and requeue.

This prevents partially updated meshes and makes failure behavior explicit.

## Surface Cooking

### First Cooker: CPU Exposed Faces

The first cooker should be CPU-side and simple:

- iterate owned voxels in a render chunk;
- inspect six neighbors through a voxel query interface;
- emit quads for exposed faces;
- split quads into triangles;
- store material id and normal.

This is not the final performance path. It is the correctness path.

### Second Cooker: CPU Greedy Meshing

Once exposed-face output is correct, add chunk-local greedy meshing:

- merge only same-normal, same-material faces;
- do not merge across render chunks;
- keep the CPU exposed-face cooker as a test/reference path.

Greedy meshing is the first meaningful geometry optimization because it attacks triangle count without changing the renderer.

### Later Cooker: GPU Or Job-System Cooker

After correctness and raster display are stable:

- move CPU cooking to a job queue;
- budget work per frame;
- add GPU generation only after CPU cook/upload is measured as a real bottleneck.

Do not begin with GPU cooking. It is harder to validate and introduces allocator/compaction complexity before the proxy contract is proven.

## Buffer Strategy

Use simple whole-proxy uploads first.

Recommended rollout:

1. CPU vector output per render chunk.
2. Pack valid proxies into one or more GPU vertex/index buffers.
3. Rebuild and upload the whole proxy buffer while the feature is small.
4. Add a GPU arena/free-list only after proxy churn and upload size are measured.

The first implementation should optimize for correctness and observability, not perfect buffer reuse.

Later buffer allocator:

- allocate vertex/index ranges per proxy;
- keep old ranges alive until no frame references them;
- use a deferred-free queue tied to frame fences;
- compact only when fragmentation becomes measurable.

## Rendering Path

First renderer:

- rasterize proxy triangles;
- write color, normal, material id, and depth;
- use simple direct lighting or existing postprocess where practical;
- add debug views for chunk state and proxy generation.

Later renderer:

- deferred G-buffer;
- material table indirection;
- indirect draw;
- frustum and occlusion culling;
- visibility buffer;
- meshlet/cluster path.

Do not start with visibility buffer or meshlets. They are strong long-term tools but poor first milestones.

## RT Core Role

This design is not initially an RT Core acceleration scheme.

RT Core becomes useful after proxies are stable triangle geometry:

- build triangle BLAS for stable valid proxies;
- skip BLAS for chunks that are changing frequently;
- use RT for shadows, reflections, AO, or GI;
- budget BLAS updates to avoid edit-time spikes.

The current procedural AABB RT path can remain as a voxel-accurate reference path. It does not need to be removed.

## Optimization Priorities

Optimize in this order:

1. Correct dirty chunk mapping and neighbor rules.
2. Correct proxy versioning and stale rejection.
3. Reduce triangle count with greedy meshing.
4. Limit cook time with a job queue and frame budget.
5. Reduce upload cost with proxy buffer ranges.
6. Reduce draw cost with indirect draw and culling.
7. Move cooking to GPU only when measured CPU cooking or upload becomes the bottleneck.
8. Add stable-proxy triangle BLAS only after raster proxies are reliable.

Key metrics:

- dirty storage bricks;
- dirty render chunks;
- queued/building/valid/failed proxy counts;
- cook time;
- generated quads/triangles;
- upload bytes;
- draw count;
- raster GPU time;
- proxy cache hit rate;
- rejected stale cook results;
- optional BLAS build/update time.

## Comparison With Dual-Speed Design

The dual-speed design keeps live voxel rendering in the main composition path. This maximizes immediate correctness but adds render-layer complexity.

The versioned proxy design keeps the main render path simpler:

- no live/cached layer ownership mask in the first phase;
- no mixed depth source in the first phase;
- no live-to-cached per-pixel transition in the first phase;
- current RT/VPT stays outside the main composition path.

Tradeoff:

- Small edits should be cooked synchronously or near-synchronously to preserve responsiveness.
- Large edits may show the previous proxy for a few frames or be handled by explicit visual feedback.

This is a practical trade: clearer architecture first, perfect edit immediacy later if it is still needed.

## First Implementation Scope

Implement only:

- render chunk coordinate mapping over existing 8^3 bricks;
- CPU exposed-face cooker;
- proxy state and source generation;
- basic vertex/index data;
- raster debug draw or simple material draw;
- dirty chunk + neighbor invalidation;
- proxy replacement after successful cook;
- tests for face generation and stale rejection.

Do not implement yet:

- physics;
- connectivity;
- live voxel composition;
- GPU cooking;
- meshlets;
- visibility buffer;
- triangle BLAS;
- global illumination;
- transparent voxels.

## Validation

Required unit tests:

- empty chunk emits no faces;
- full 16^3 render chunk emits only outer faces;
- two adjacent solid chunks emit no interior boundary faces;
- one missing neighbor emits boundary faces;
- carved hole emits inward-facing faces;
- editing boundary voxels dirties adjacent render chunks;
- stale cook output is rejected if source generation changed;
- proxy replacement does not mutate the active proxy before success.

Required runtime checks:

- render proxy debug overlay shows chunk state;
- continuous edits do not leave stale geometry after the cook queue drains;
- proxy triangle counts change when expected after edits;
- current RT/VPT and proxy raster agree on simple reference scenes;
- large dirty edits are budgeted and observable.

## Risks

- If proxy update latency is too visible, the design may need a small live overlay for only the affected chunks. That should be a later fix, not the initial architecture.
- If 16^3 chunks create too many draw/proxy objects, move the render chunk size to 32^3 after correctness is proven.
- If 32^3 chunks are chosen too early, small edits may rebuild too much geometry.
- If greedy meshing is introduced before exposed-face tests are strong, bugs will be harder to isolate.
- If GPU cooking is introduced before CPU cooking is measured, implementation complexity may outrun evidence.

## Research Basis

- Luanti/Minetest documents world partitioning into 16x16x16 MapBlocks and client-side mesh concepts: https://docs.luanti.org/for-engine-devs/basic-data-structures/
- Godot Voxel Tools describes realtime editable voxel terrain and polygon-based rendering: https://github.com/Zylann/godot_voxel and https://voxel-tools.readthedocs.io/en/latest/
- `block-mesh-rs` provides visible-face and greedy-quad chunk meshing utilities: https://github.com/bonsairobo/block-mesh-rs and https://docs.rs/block-mesh/latest/block_mesh/
- Vercidium's voxel mesh generation work uses 32x32x32 chunks and separates mesh generation from buffering: https://github.com/Vercidium/voxel-mesh-generation

## Suggested Next Thread Prompt

Use this document to plan Phase 1:

> Plan the implementation of the Versioned Render Proxy Pipeline Phase 1 for Revolumetric. Start with CPU exposed-face cooking for 16^3 render chunks over existing 8^3 UCVH bricks, versioned proxy metadata, dirty-neighbor invalidation, unit tests, and a minimal raster/debug draw path. Do not implement physics, GPU cooking, meshlets, visibility buffer, or triangle BLAS yet.

