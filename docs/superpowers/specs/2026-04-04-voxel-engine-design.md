# Revolumetric Voxel Engine — Design Specification

## 1. Overview

Revolumetric is an industrial-grade real-time voxel engine built in Rust + Vulkan 1.3, featuring a Unified Cascaded Volume Hierarchy (UCVH) that structurally merges voxel spatial storage with 3D Radiance Cascades global illumination into a single hierarchical data structure.

### 1.1 Goals

- Real-time interactive voxel engine at 512^3 scale (~134M voxels)
- State-of-the-art GI via improved 3D Radiance Cascades (multi-bounce, specular, emissive, AO)
- Full data pipeline: procedural generation, external import (.vox/.vdb), runtime editing
- Data-driven render graph with automatic resource management
- Full Slang shader pipeline

### 1.2 Reference

Based on Shadertoy M3ycWt — a 6-pass volumetric Radiance Cascades implementation with:
- 32x32x48 voxel grid, 6 hemispheres per voxel
- 5 LOD cascade levels, (3x3) * 4^N rays per hemisphere
- Visibility-weighted trilinear interpolation merging
- DDA voxel ray tracing, directional shadow map

### 1.3 Key Insight: UCVH

Traditional approach: separate voxel storage + separate RC probe storage + mapping layer.

UCVH approach: the voxel spatial hierarchy IS the cascade hierarchy. One tree traversal yields both geometry hits and all cascade-level probe data. No redundant mapping, no duplicate traversals.

---

## 2. UCVH Data Structure

### 2.1 Brick Pool

The atomic storage unit is an **8^3 Brick** (512 voxels).

```rust
#[repr(C)]
struct VoxelCell {
    material: u16,    // material palette index (0 = air)
    flags: u16,       // solid, emissive, transparent, etc.
    emissive: [u16; 3], // HDR emissive color (half-float)
    _pad: u16,
}
// sizeof(VoxelCell) = 8 bytes
// sizeof(Brick) = 512 * 8 = 4096 bytes
```

**GPU storage**: Single `VkBuffer` with buffer device address. Free-list allocator (GPU-side atomic ring buffer) for dynamic allocation/deallocation.

```
Capacity: 81,920 bricks (512^3 * 30% / 512 ≈ 78K, rounded up with ~5% headroom)
Size: 81,920 * 4,096 = 320 MB
Note: If fill rate exceeds 30%, the pool can be resized via reallocation + copy.
```

### 2.2 Cascaded Occupancy Hierarchy

Five levels, each a flat 3D grid stored as an SSBO. Index: `x + y * dim_x + z * dim_x * dim_y`.

| Level | Grid Dim | Covers        | Node Data                          | Size    |
|-------|----------|---------------|------------------------------------|---------|
| 0     | 64^3     | 8^3 brick     | `brick_id: u32`, `flags: u16`      | 1.5 MB  |
| 1     | 32^3     | 2^3 L0 nodes  | `child_mask: u8`, `flags: u8`      | 64 KB   |
| 2     | 16^3     | 2^3 L1 nodes  | `child_mask: u8`, `flags: u8`      | 8 KB    |
| 3     | 8^3      | 2^3 L2 nodes  | `child_mask: u8`, `flags: u8`      | 1 KB    |
| 4     | 4^3      | 2^3 L3 nodes  | `child_mask: u8`, `flags: u8`      | 128 B   |

`child_mask` is a bitmask of which of the 8 children (2^3) contain any solid voxels. Used for hierarchical DDA ray skip.

### 2.3 RC Probe Storage

Each cascade level has a per-level storage buffer. Probes exist only at occupied nodes.

```rust
#[repr(C)]
struct RayResult {
    radiance: [f32; 3], // RGB irradiance
    distance: f32,       // hit distance (for visibility weight)
}
// 16 bytes per ray
```

Per hemisphere ray counts follow the Shadertoy pattern: `(3x3) * 4^N`:

| Level | Probe Size | Rays/Hemisphere | Bytes/Probe (6 hemispheres) | Grid  | Fill Est. | Total        |
|-------|------------|-----------------|----------------------------|-------|-----------|--------------|
| 0     | 3          | 9               | 864 B                      | 64^3  | 30%       | 68 MB        |
| 1     | 6          | 36              | 3,456 B                    | 32^3  | 60%       | 68 MB        |
| 2     | 12         | 144             | 13,824 B                   | 16^3  | 90%       | 51 MB        |
| 3     | 24         | 576             | 55,296 B                   | 8^3   | 100%      | 28 MB        |
| 4     | 48         | 2304            | 221,184 B                  | 4^3   | 100%      | 14 MB        |
| **Total** |       |                 |                            |       |           | **~229 MB**  |

Note: Fill rate propagates upward — a parent node is occupied if ANY child is.
L0 fill of 30% yields L1 ~60%, L2 ~90%, L3-L4 ~100% in worst case.
Budget uses these conservative per-level estimates.

**Sparse allocation**: Probes for empty nodes are not allocated. A parallel prefix-sum compaction pass builds an indirect dispatch table mapping only occupied nodes to probe buffer offsets.

### 2.4 Probe Placement

Improvement over Shadertoy's "first empty voxel near center":

- **Level 0**: offset probe position to nearest empty cell within the 8^3 brick using a precomputed 3D distance field (per-brick, 512 bytes, computed on upload)
- **Level 1+**: use occupancy bitmask to find the centroid of empty space within the node's coverage, then snap to nearest empty cell via the Level-0 distance fields

Store as `probe_offset: [i8; 3]` per node (3 bytes).

### 2.5 Memory Budget Summary

| Component            | Size     |
|----------------------|----------|
| Brick Pool           | ~320 MB  |
| Occupancy Hierarchy  | ~1.6 MB  |
| RC Probe Buffers     | ~229 MB  |
| Shadow Map (2048^2)  | ~16 MB   |
| Staging / Scratch    | ~64 MB   |
| Swapchain + Targets  | ~32 MB   |
| **Total**            | **~663 MB** |

Note: Probe distance fields (for probe placement) are computed transiently
during VoxelUploadPass and discarded after writing the 3-byte probe_offset
per node. They live in the Staging/Scratch budget, not as persistent storage.

Target GPU: 8 GB+ VRAM recommended (RTX 3060 class). 6 GB GPUs can work with
reduced probe resolution or smaller world size.

---

## 3. Render Graph

### 3.1 Architecture

A data-driven frame graph that manages resource lifetimes, pipeline barriers, and pass scheduling automatically.

```rust
pub struct RenderGraph {
    passes: Vec<RenderPassNode>,
    resources: ResourceRegistry,
    barriers: BarrierBatcher,
}

pub struct RenderPassNode {
    name: &'static str,
    reads: Vec<ResourceHandle>,
    writes: Vec<ResourceHandle>,
    execute: Box<dyn FnOnce(&mut PassContext)>,
    queue_type: QueueType, // Graphics | Compute | Transfer
}

pub struct ResourceHandle {
    id: u32,
    version: u32, // tracks read-after-write dependencies
}
```

**Key behaviors**:
- Topological sort of passes based on read/write dependencies
- Automatic `VkImageMemoryBarrier` / `VkBufferMemoryBarrier` insertion
- Transient resource aliasing (resources that don't overlap lifetimes share memory)
- Dead-pass elimination (passes whose outputs are unused get culled)

### 3.2 Frame Pass Order

```
Frame N:
  ┌─ Transfer ─────────────────────────────────┐
  │  1. VoxelUploadPass     (dirty bricks → GPU) │
  │  2. OccupancyUpdatePass (rebuild hierarchy)  │
  └──────────────────────────────────────────────┘
           │
  ┌─ Compute (prepare) ───────────────────────┐
  │  3. ProbeCompactPass    (prefix-sum compact) │  ← builds indirect dispatch table
  │  4. ShadowTracePass     (directional SM)    │     for CascadeTracePass
  └──────────────────────────────────────────────┘
           │
  ┌─ Compute (GI) ────────────────────────────┐
  │  5. CascadeTracePass    (RC ray tracing)    │  ← most expensive, uses dispatch table
  │  6. CascadeMergePass    (inter-level merge) │
  └──────────────────────────────────────────────┘
           │
  ┌─ Graphics ─────────────────────────────────┐
  │  7. PrimaryRayPass      (screen-space DDA)  │
  │  8. CompositePass       (GI + shadow + sky) │
  │  9. TonemapPass         (ACES + gamma)      │
  │ 10. DebugOverlayPass    (optional)          │
  └──────────────────────────────────────────────┘
           │
        Present

Bootstrap (Frame 0): ProbeCompactPass runs on the initial occupancy from
OccupancyUpdatePass. No double-buffering needed — compact always runs
before trace within the same frame.
```

### 3.3 Temporal Strategy

- **CascadeTracePass** processes a fraction of probes per frame (round-robin across bricks), spreading the cost. Configurable: 1/4 to 1/16 of probes per frame.
- **CascadeMergePass** uses exponential moving average to blend new radiance with previous frame.
- **Temporal reprojection**: camera motion vectors reproject previous frame's screen-space probe lookups, reducing shimmering.

---

## 4. Radiance Cascades — Improved Algorithm

### 4.1 Hemisphere Direction Mapping

Retain the Shadertoy's square-to-hemisphere mapping (`ComputeDir` / `ComputeDirEven`) but extend for larger probe sizes with better uniformity at high LODs.

### 4.2 Cascade Tracing (Per Level)

For each occupied node at level N:
1. Determine 6 hemisphere normals (±X, ±Y, ±Z)
2. For each hemisphere, trace `(3x3) * 4^N` rays from probe position
3. DDA through the UCVH occupancy hierarchy (skip empty nodes at coarser levels first)
4. On hit: record `(radiance, distance)` per ray
5. On miss: sample sky light
6. Weight by `cos(theta) * solid_angle` (hemisphere-aware normalization)

### 4.3 Cascade Merging (Inter-Level)

For each probe at level N, merge with level N+1 using visibility-weighted trilinear interpolation:

1. Map the probe position to level N+1 grid coordinates
2. Find the 8 surrounding level N+1 probes (trilinear)
3. For each neighbor: project the ray direction onto the neighbor's hemisphere, read the stored distance
4. **Visibility test**: if the neighbor's stored distance is shorter than the actual distance to this probe, the neighbor is occluded → weight = 0
5. Weighted blend of 4 neighboring rays (2x2 subgrid of the parent probe)
6. Distance-based interpolation: `mix(local, merged, clamp((hit_dist - voxel_size) / voxel_size * 0.5, 0, 1))`

### 4.4 Improvements Over Reference

**a) Hemisphere overlap reduction**:
- Store up to 3 weights per ray for the 3 closest hemispheres (max overlap)
- When integrating, each hemisphere uses only its weighted share of shared rays
- Reduces redundant ray tracing by ~30%

**b) Multi-bounce GI**:
- Cascade tracing reads the *previous frame's* merged probe data when hitting a surface
- This provides one extra bounce per frame
- After N frames of temporal accumulation, effectively N-bounce GI (converges to ~3 bounces in practice)

**c) Specular reflection (GGX)**:
- In the PrimaryRayPass, for glossy surfaces: importance-sample GGX distribution
- Look up the nearest RC probe and filter the stored ray results by the GGX lobe direction
- Uses the `BRDF_GGX` function from the Shadertoy common pass, extended to filter cascade data

**d) Emissive volume light**:
- Emissive voxels (`flags & EMISSIVE`) write directly to Level-0 probe radiance
- Propagates through cascade merging naturally — no special code path needed

**e) Ambient occlusion**:
- Derived from Level-0 probe ray hit distances: `AO = 1 - avg(clamp(hit_dist / ao_radius, 0, 1))`
- Nearly free — data already exists in probe storage

---

## 5. Voxel Ray Tracing

### 5.1 Hierarchical DDA

```
trace(origin, direction):
    t = entry_distance(world_bounds)
    level = 4 (coarsest)

    while level >= 0:
        node = occupancy[level][position_at_level(origin + direction * t, level)]

        if node is empty:
            // Skip entire node extent via AABB exit
            t = exit_distance(node_bounds)
            advance to next sibling at this level
        else if level == 0:
            // DDA through 8^3 brick
            brick = brick_pool[node.brick_id]
            result = brick_dda(brick, origin, direction, t)
            if hit: return result
            t = exit_distance(brick_bounds)
            level = ascend_to_next_sibling()
        else:
            // Descend to finer level
            level -= 1
```

### 5.2 Acceleration

- **Occupancy skip**: at each level, `child_mask` tells us which octants are empty. DDA only visits non-empty octants.
- **Brick-internal DDA**: standard 3D-DDA within the 8^3 grid (same algorithm as Shadertoy's `ABoxfarNormal` traversal).
- **Early termination**: for shadow rays, return on first hit (no shading needed).

---

## 6. Data Pipeline

### 6.1 Procedural Generation

```rust
pub trait VoxelGenerator: Send + Sync {
    fn generate(&self, brick_coord: UVec3, time: f32) -> Option<BrickData>;
}
```

Generators produce `BrickData` (8^3 cells) on demand. The engine calls generators for all occupied brick coordinates during scene initialization or when streaming new regions.

### 6.2 External Import

**MagicaVoxel (.vox)**:
- Parse palette + voxel data → map to engine material palette
- Chunk into 8^3 bricks, upload via VoxelUploadPass
- Typical .vox files are 256^3 max; multiple files can be composed into the 512^3 world

**OpenVDB (.vdb)**:
- Read via `openvdb` crate (or FFI to C++ lib)
- VDB's internal tiling maps naturally to brick allocation
- Density field → threshold to solid/empty, gradient → surface normal hint

### 6.3 Runtime Editing

```rust
pub struct EditOperation {
    pub region: BrickCoord,
    pub op: EditOp,
}

pub enum EditOp {
    SetVoxel { local: UVec3, cell: VoxelCell },
    FillBrick { cell: VoxelCell },
    ClearBrick,
    CSG { shape: Shape, material: u16 },
}
```

Edit pipeline:
1. CPU records `EditOperation` into a per-frame edit queue
2. `VoxelUploadPass` applies edits to staging buffer, uploads dirty bricks
3. `OccupancyUpdatePass` rebuilds affected occupancy nodes (bottom-up)
4. Cascade probes in affected regions are invalidated → re-traced over next N frames

---

## 7. Shader Architecture (Slang)

### 7.1 Module Layout

```
assets/shaders/
├── shared/
│   ├── math.slang          — vec math, AABB, TBN, smin
│   ├── voxel_common.slang  — VoxelCell, BrickData, occupancy access
│   ├── ray.slang           — hierarchical DDA, brick DDA
│   ├── hemisphere.slang    — ComputeDir, ProjectDir, solid angle weights
│   ├── brdf.slang          — GGX, Lambert, Fresnel
│   └── lighting.slang      — sky, sun, shadow sampling
├── passes/
│   ├── voxel_upload.slang
│   ├── occupancy_update.slang
│   ├── shadow_trace.slang
│   ├── cascade_trace.slang      — core RC ray tracing
│   ├── cascade_merge.slang      — inter-level merging
│   ├── probe_compact.slang      — prefix-sum for sparse dispatch
│   ├── primary_ray.slang        — screen-space voxel trace
│   ├── composite.slang          — GI integration + shadow + sky
│   └── tonemap.slang            — ACES + gamma
└── fullscreen.vert.slang        — fullscreen triangle
```

### 7.2 Binding Model

Slang parameter blocks map to Vulkan descriptor sets:

```slang
// Set 0: per-frame global data
struct FrameUniforms {
    float4x4 view_proj;
    float4x4 inv_view_proj;
    float3 camera_pos;
    float3 sun_dir;
    float3 sun_color;
    float time;
    uint frame_index;
    uint2 resolution;
};

// Set 1: UCVH data (persistent, rarely changes layout)
struct VolumeData {
    StructuredBuffer<VoxelCell> brick_pool;  // buffer device address
    StructuredBuffer<NodeL0> occupancy_l0;
    StructuredBuffer<NodeL1> occupancy_l1;
    // ... l2, l3, l4
    RWStructuredBuffer<RayResult> probes_l0;
    RWStructuredBuffer<RayResult> probes_l1;
    // ... l2, l3, l4
};

// Set 2: per-pass scratch/output
// (varies per pass)
```

---

## 8. Rust Module Architecture

### 8.1 Crate Structure (single crate, module hierarchy)

```
src/
├── main.rs
├── app.rs                      — event loop, init
├── platform/
│   ├── window.rs, input.rs, time.rs
├── ecs/
│   ├── world.rs, entity.rs, query.rs, ...
├── render/
│   ├── device.rs               — Vulkan device, queues
│   ├── graph.rs                — RenderGraph, PassNode, ResourceRegistry
│   ├── buffer.rs               — GPU buffer wrapper + allocator
│   ├── image.rs                — GPU image wrapper
│   ├── descriptor.rs           — descriptor set layout/pool
│   ├── pipeline.rs             — compute/graphics pipeline cache
│   ├── shader.rs               — Slang compilation, reflection
│   ├── sampler.rs
│   ├── swapchain.rs
│   ├── frame.rs
│   └── passes/
│       ├── voxel_upload.rs
│       ├── occupancy_update.rs
│       ├── shadow_trace.rs
│       ├── cascade_trace.rs
│       ├── cascade_merge.rs
│       ├── probe_compact.rs
│       ├── primary_ray.rs
│       ├── composite.rs
│       ├── tonemap.rs
│       └── debug_views.rs
├── voxel/
│   ├── brick.rs                — BrickData, BrickPool, free-list
│   ├── occupancy.rs            — CascadedOccupancy (5-level hierarchy)
│   ├── probes.rs               — ProbeStorage (per-level buffers)
│   ├── ucvh.rs                 — UCVH facade (unified access)
│   ├── material.rs             — MaterialPalette
│   ├── edit.rs                 — EditOperation, EditQueue
│   ├── generator.rs            — VoxelGenerator trait + procedural impls
│   └── import/
│       ├── vox.rs              — MagicaVoxel loader
│       └── vdb.rs              — OpenVDB loader (optional, feature-gated)
├── scene/
│   ├── camera.rs               — FPS camera, orbit camera
│   ├── light.rs                — DirectionalLight, PointLight
│   ├── components.rs           — Transform, CameraRig
│   └── systems.rs              — tick_time, camera_update, input_dispatch
└── assets/
    ├── shader_compiler.rs      — Slang → SPIR-V compilation
    ├── shader_reflect.rs       — SPIR-V reflection for auto descriptor layout
    └── hot_reload.rs           — file watcher, shader hot reload
```

### 8.2 Key Types

```rust
// voxel/ucvh.rs — the unified facade
pub struct Ucvh {
    pub bricks: BrickPool,
    pub occupancy: CascadedOccupancy,
    pub probes: ProbeStorage,
    pub materials: MaterialPalette,
}

impl Ucvh {
    pub fn allocate_brick(&mut self, coord: UVec3) -> BrickId;
    pub fn free_brick(&mut self, coord: UVec3);
    pub fn edit(&mut self, ops: &[EditOperation]);
    pub fn rebuild_occupancy(&mut self);  // bottom-up from dirty bricks
    pub fn dirty_bricks(&self) -> &[BrickCoord];
    pub fn upload_to_gpu(&mut self, ctx: &mut TransferContext);
}

// render/graph.rs
pub struct RenderGraph { /* as described in section 3.1 */ }

impl RenderGraph {
    pub fn add_pass(&mut self, name: &'static str, setup: impl FnOnce(&mut PassBuilder));
    pub fn compile(&mut self);     // topological sort, barrier insertion
    pub fn execute(&self, device: &RenderDevice, frame: &FrameContext);
}
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

- `voxel::brick` — allocation, free, reuse, boundary indexing
- `voxel::occupancy` — hierarchy rebuild correctness, child_mask accuracy
- `voxel::edit` — CSG operations, brick splitting
- `render::graph` — dependency sort, barrier correctness, dead-pass elimination

### 9.2 Integration Tests

- Load a .vox file → verify brick count and occupancy match expected scene
- Generate procedural scene → ray trace known directions → verify hit positions
- Full frame render → screenshot comparison against reference (golden image)

### 9.3 Visual Regression

- Headless Vulkan rendering (using `VK_EXT_headless_surface`)
- Capture framebuffer → compare against golden images with perceptual diff (FLIP metric)
- Run in CI via Playwright + screenshot diff (existing test infrastructure)

---

## 10. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Frame time | < 16.6 ms (60 FPS) | at 1080p, 512^3, RTX 3060 class (8 GB) |
| Cascade trace | < 8 ms | with 1/4 probe update per frame |
| Primary ray | < 2 ms | hierarchical DDA |
| Shadow trace | < 1 ms | 2048^2 shadow map |
| Edit latency | < 3 frames | from CPU edit to visual update |
| Streaming | 64 bricks/frame | ~2 MB/frame upload budget |
| Startup | < 2 s | procedural scene generation |

---

## 11. Non-Goals (Explicit Exclusions)

- Mesh rendering (this is a pure voxel engine)
- Networking / multiplayer
- Audio
- UI framework (debug overlay only, via immediate-mode)
- Mobile / WebGPU (Vulkan desktop only)

---

## 12. Dependencies

| Crate | Purpose | Existing? |
|-------|---------|-----------|
| ash | Vulkan bindings | Yes |
| ash-window | Surface creation | Yes |
| gpu-allocator | VMA-style memory allocation | Yes |
| glam | Linear algebra | Yes |
| winit | Windowing | Yes |
| bytemuck | Safe transmute for GPU uploads | Yes |
| slang (FFI) | Shader compilation | New |
| dot_vox | MagicaVoxel .vox parser | New |
| notify | File watcher for hot reload | New |
| image | Screenshot / golden image | New (test only) |
