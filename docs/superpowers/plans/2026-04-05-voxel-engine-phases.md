# Revolumetric — Implementation Phase Overview

Each phase produces working, testable software. Later phases build on earlier ones.

| Phase | Name | Deliverable | Depends On |
|-------|------|-------------|------------|
| 1 | Render Foundation | Render graph + GPU resource management + Slang pipeline + fullscreen compute | — |
| 2 | UCVH Core | Brick pool (Morton order, occupancy count) + hierarchy + upload pass + procedural demo | Phase 1 |
| 3 | Voxel Ray Tracing | Two-level hierarchical DDA (brick-grid → brick-internal) + primary ray pass + flat-color | Phase 2 |
| 4 | Shadow & Lighting | Shadow map pass + sun/sky lighting + composite + tonemap | Phase 3 |
| 5 | Radiance Cascades | Probe storage + cascade trace + cascade merge + GI visualization | Phase 3 |
| 6 | Full Composite | Integrate RC GI + shadow + specular + emissive + AO into final image | Phase 4, 5 |
| 7 | Data Pipeline | .vox import + runtime editing + procedural generators | Phase 2 |
| 8 | Advanced GI | Multi-bounce + hemisphere overlap reduction + temporal reprojection | Phase 5 |
| 9 | Camera & Input | FPS camera controller + WASD + mouse look | Phase 3 |
| 10 | Polish | Hot reload + debug views + perf tuning + visual regression tests | Phase 6 |
| 11 | Voxel Physics & Automata | GPU cellular automata: liquid, sand, gravity simulation | Phase 2 |
| 12 | Advanced AO & Shading | Per-face vertex AO with UV interpolation + improved material shading | Phase 3 |

Phase 1 plan: `2026-04-05-phase1-render-foundation.md`

---

## Phase 11: Voxel Physics & Automata (Future)

GPU compute-based cellular automata for dynamic voxel simulation.

### Techniques (from reference projects)

**Double-buffered atomicCompSwap liquid physics** (from GDVoxelPlayground):
- Two material buffers, ping-pong each frame (`frame % 2` selects read/write)
- Liquid voxels attempt to move (gravity → lateral) using `atomicCompSwap` on destination
- Lock-free: if CAS fails (another thread moved there first), try next direction or stay
- Water, lava, sand each have distinct movement rules (gravity priority, lateral spread, diagonal slide)
- Direction persistence: liquid remembers last flow direction (stored in low bits) for coherent flow

**Incremental neighbor counting** (from 3d_celluar_automata):
- Persist per-cell neighbor count instead of recomputing each frame
- On cell state change, update only the 6/26 neighbors' counts (Von Neumann / Moore)
- ~20x faster than full recount for sparse changes (typical: <1% cells change per frame)
- Enables complex cellular automata rules (birth/death thresholds) at interactive rates

**Chunk boundary atomics** (from 3d_celluar_automata):
- Divide world into 32^3 chunks; interior cells use normal writes
- Only cells at chunk boundaries use `AtomicU8` for thread-safe neighbor access
- Avoids global synchronization overhead — interior cells (87.5% of volume) are lock-free

**Sparse change lists**:
- Each chunk maintains a list of cells that changed this frame
- Only propagate neighbor updates for changed cells
- Reduces per-frame work from O(N) to O(changed_cells * neighbors)

### Implementation Notes

- New compute passes: `LiquidSimPass`, `SandSimPass` inserted after `VoxelUploadPass`
- Dirty brick tracking feeds into `OccupancyUpdatePass` (only rebuild affected bricks)
- Physics rate can be decoupled from render rate (e.g., 30 Hz physics, 60 Hz render)

---

## Phase 12: Advanced AO & Shading (Future)

Per-face vertex ambient occlusion and enhanced material shading.

### Techniques (from GDVoxelPlayground)

**Per-face vertex AO with UV interpolation**:
- For each hit face, sample 8 neighboring voxels (4 edge + 4 corner)
- Compute per-vertex AO using: `ao = (side1 + side2 + max(corner, side1 * side2)) / 3`
- Bilinear interpolate across the face using hit UV coordinates
- Produces smooth, artifact-free ambient occlusion without any ray tracing
- Extremely cheap: 8 voxel lookups + 4 interpolations per pixel

**Randomized voxel colors**:
- Per-voxel color variation using deterministic hash of position
- Slight HSV jitter (hue +/-2.5%, saturation/value +/-10%) prevents flat look
- Hash-based: same position always produces same color (no flickering)

### Integration

- AO computation lives in `PrimaryRayPass` or a dedicated `AOPass`
- Can replace or augment the RC-derived AO from Phase 5 (faster, lower quality)
- Useful as fallback when RC probes haven't converged yet (first few frames)
