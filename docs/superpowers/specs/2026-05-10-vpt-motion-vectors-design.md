# VPT Motion Vectors Design

## Decision

Introduce a unified VPT motion guide instead of continuing to overload
`motion_history` as an unnamed previous-pixel lookup.

The current implementation already has per-pixel reprojection data, but it is
not a complete motion-vector system:

- `assets/shaders/passes/vpt_surface.slang` writes `motion_history[pixel] =
  project_previous_pixel(hit.position)`.
- `assets/shaders/passes/vpt_temporal.slang`,
  `assets/shaders/passes/restir_di_temporal.slang`, and
  `assets/shaders/passes/area_restir_temporal.slang` consume `motion_history.xy`
  as previous-frame pixel coordinates and `motion_history.w` as validity.
- `src/render/vpt_history.rs` and
  `assets/shaders/shared/vpt_history_common.slang` provide current/previous
  view-projection matrices, resolution, jitter fields, reset generation, and
  reset flags.
- `assets/shaders/passes/vpt_surface.slang` has already been updated to use
  the Area ReSTIR primary-sample contract when a selected reservoir is valid.
- `src/scene/components.rs` only has `Transform { translation }`, with no
  entity motion identity or previous transform.
- `src/voxel/ucvh.rs` tracks dirty bricks and hierarchy rebuild state, but it
  does not record semantic voxel movement.

Therefore the first implementation should turn the existing guide into an
explicit, testable motion-vector contract, then add rigid/entity and UCVH
content motion sources behind that contract.

## Reference Basis

Use these references for architecture and validation, not for source copying:

- [NVIDIA NRD](https://github.com/NVIDIA-RTX/NRD): denoisers require stable
  guide buffers such as normal, roughness, viewZ, and motion vectors, with
  history reset behavior kept explicit.
- [AMD FidelityFX FSR2](https://github.com/GPUOpen-Effects/FidelityFX-FSR2):
  upscaling/temporal reconstruction expects motion vectors in a documented
  screen-space convention, plus depth, jitter, reactive/lock masks, and reset
  handling.
- [GPU Gems 3, Chapter 27](https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-27-motion-blur-post-processing-effect):
  camera motion can be derived from current and previous view-projection data;
  moving geometry needs object-level previous transforms, not only camera
  reprojection.
- [RTXDI](https://github.com/NVIDIA-RTX/RTXDI): temporal/spatial ReSTIR reuse
  requires application-owned G-buffer/surface data, previous-frame state,
  motion/reprojection, visibility, and explicit compatibility tests.
- [Area ReSTIR](https://github.com/guiqi134/Area-ReSTIR): temporal reuse in
  the 4D primary-ray domain uses subpixel/lens sample history and previous
  scene data; this supports keeping VPT's Area ReSTIR primary sample and
  surface guide in one coherent domain.
- [SVGF paper page](https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced):
  temporal accumulation and variance-guided filtering depend on reprojected
  history plus geometry-aware rejection.

Locally inspected references that remain relevant:

- `target/research/RTXDI/Doc/Integration.md`
- `target/research/RTXDI/Samples/FullSample/Shaders/LightingPasses/DI/TemporalResampling.hlsl`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/TemporalResampling_FloatMotion.cs.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/PixelAreaSampleData.slang`
- `docs/superpowers/specs/2026-05-02-vpt-only-temporal-denoise-design.md`
- `docs/superpowers/specs/2026-05-02-area-restir-reference-design.md`
- `docs/superpowers/plans/2026-05-03-vpt-primary-sample-domain-unification.md`

## Approaches Considered

### Approach A: Camera-Only Previous Pixel

Keep the current `motion_history.xy = previous_pixel` behavior and only fix
small validity bugs.

This is too narrow. It handles static-world camera motion, but it cannot
describe rigid object movement or semantic voxel content movement. It also
keeps the ABI ambiguous: a texture named "motion" contains absolute previous
coordinates rather than a motion vector.

### Approach B: Derive Everything From World Position

Always project the current world hit point into the previous camera.

This is valid only for static geometry. If an object moved, the current world
point may not have existed at that previous position. If voxel content was
edited, there may be no physically meaningful previous point. This approach
would create ghosting by pretending topology changes are motion.

### Approach C: Unified Motion Guide With Source-Specific Previous Position

Recommended. The surface pass resolves a previous world position per hit using
the best available source:

- static world / camera motion: previous position equals current world position;
- rigid entity motion: previous position comes from current local hit point
  transformed by that entity's previous transform;
- semantic UCVH content motion: previous position comes from explicit moved
  region metadata;
- arbitrary UCVH edits: history is invalidated, not reprojected.

The guide then emits a single screen-space motion-vector contract consumed by
VPT temporal, ReSTIR-DI temporal, Area ReSTIR temporal, debug views, and later
denoisers.

This preserves the current pipeline shape while making motion semantics
explicit and extensible.

## Motion Guide Contract

Create a named VPT motion guide ABI. The implementation may initially keep the
existing image allocation name for migration, but tests and shader helpers must
use the new semantics.

Recommended pixel payload:

```text
float4 vpt_motion_guide
  xy: screen-space motion vector in pixels, current pixel center -> previous pixel center
  z : reserved for future extension, keep 0.0 in phase 1
  w : confidence, 0.0 invalid, 1.0 fully valid
```

The convention is:

```text
previous_pixel_center = current_pixel_center + motion.xy
motion.xy = previous_pixel_center - current_pixel_center
```

This should replace the current absolute previous-pixel consumer contract.
Consumers that need the old coordinate can reconstruct it with the formula
above. Debug motion view should visualize this delta directly.

A compact class/flag signal is also needed, but should not be forced into the
floating-point confidence channel long term. The planned layout is:

- phase 1: encode only validity/confidence in `.w`;
- phase 2: add `vpt_motion_flags` as `r32ui` or move motion guide to a
  structured per-pixel buffer if RenderGraph and descriptors make that cleaner;
- required flags: `CAMERA_STATIC`, `RIGID_ENTITY`, `UCVH_REGION_MOVE`,
  `CONTENT_INVALIDATED`, `DISOCCLUDED`, `HISTORY_RESET`.

The `.xy` convention must be covered by Rust and shader source tests because a
sign flip silently breaks all temporal reuse.

Depth and surface compatibility stay on the surface/history guides that already
exist in the renderer. Motion vectors should not try to replace those signals.

## Camera Motion

For static world hits, compute motion from current and previous camera data:

1. Trace the current primary surface using the authoritative VPT primary sample
   contract.
2. Use the current hit world position as the previous world position.
3. Project it by `previous_view_proj`.
4. Reject if `previous_clip.w <= epsilon`, not only when `abs(w)` is small.
5. Reject if the projected pixel is outside the previous resolution.
6. Emit `motion.xy = previous_pixel_center - current_pixel_center`.
7. Set confidence to `0.0` on camera cut, resize, scene reset, light-table reset,
   or missing previous history.

Current code already covers most of this, but it should fix the behind-camera
case in `project_previous_pixel`: `previous_clip.w <= 1.0e-5` must invalidate
history. The current `abs(previous_clip.w) < 1.0e-5` accepts negative `w`.

`current_jitter` and `previous_jitter` currently exist in `VptHistoryUniforms`
but are written as zero. If jitter is later enabled, motion vectors must use the
same jitter convention as temporal consumers. Until then, tests should assert
that jitter is zero or explicitly ignored.

## Rigid Entity Motion

Rigid motion cannot be inferred from the current world hit alone. Add a CPU/GPU
motion source contract:

```rust
pub struct MotionSource {
    pub motion_id: u32,
    pub current_world_from_local: glam::Mat4,
    pub previous_world_from_local: glam::Mat4,
    pub generation: u32,
    pub flags: u32,
}
```

Renderer-facing rules:

- Every dynamic rigid renderable gets a stable `motion_id`.
- A surface hit must carry `motion_id` and enough local-space information to
  reconstruct the previous world position.
- For rigid hits:

```text
current_local_position = inverse(current_world_from_local) * current_world_position
previous_world_position = previous_world_from_local * current_local_position
```

- If `motion_id` is missing, reused, or generation-mismatched, the guide must
  invalidate history for that hit.
- If only translation exists at first, still store the API as matrices so
  rotation and scale do not require another ABI break.

Current Revolumetric state has no renderable entity hit path beyond UCVH world
voxels, so the first code phase can add the data model/tests without claiming
full dynamic object coverage until a hit source writes `motion_id`.

## UCVH Content Motion

UCVH edits need conservative handling. Do not invent motion for arbitrary voxel
topology changes.

Required policy:

- `set_voxel` and `write_brick` are content edits. They invalidate history in
  affected bricks/regions unless the caller provides a semantic move.
- Dirty bricks should become a motion invalidation source, not a fake velocity
  source.
- A new semantic move API may later declare a region translation:

```rust
pub struct UcvhMotionEvent {
    pub region_min: glam::UVec3,
    pub region_max_exclusive: glam::UVec3,
    pub world_delta_current_from_previous: glam::IVec3,
    pub generation: u32,
}
```

For a hit inside a moved region:

```text
previous_world_position = current_world_position - world_delta_current_from_previous
```

If multiple events overlap, pick the most recent exact region match or
invalidate. Overlapping semantic motion should be rejected in debug builds
until a clear composition rule exists.

This mirrors the principle from temporal denoisers and ReSTIR integrations:
history reuse is only valid when the previous sample describes the same
surface/sample domain.

## Consumer Updates

Update all current consumers to reconstruct previous pixel coordinates from the
new delta contract:

- `assets/shaders/passes/vpt_temporal.slang`
- `assets/shaders/passes/restir_di_temporal.slang`
- `assets/shaders/passes/area_restir_temporal.slang`

Shared helper:

```hlsl
float2 vpt_previous_pixel_center_from_motion(uint2 pixel, float4 motion) {
    return float2(pixel) + 0.5 + motion.xy;
}
```

Temporal 4-tap sampling then uses:

```hlsl
float2 history_sample = previous_pixel_center - 0.5;
int2 previous_base_pixel = int2(floor(history_sample));
float2 history_fraction = saturate(history_sample - float2(previous_base_pixel));
```

This keeps existing bilinear history behavior while making the guide a real
motion vector.

## Reset And Invalidation

The existing reset flags remain necessary:

- `VPT_HISTORY_FLAG_CAMERA_CUT`
- `VPT_HISTORY_FLAG_RESIZE`
- `VPT_HISTORY_FLAG_SCENE_INVALIDATED`
- `VPT_HISTORY_FLAG_LIGHTS_INVALIDATED`

Add region-level invalidation before trying to preserve history through UCVH
edits. Full-frame invalidation is acceptable when no per-brick invalidation mask
exists, but the design target is a per-brick or per-region invalidation guide so
small edits do not flush the whole frame.

History must be invalid when:

- previous camera data is absent;
- previous resolution differs;
- previous surface ping-pong images are not initialized;
- current or previous surface is a miss;
- normal, material, or world-position compatibility fails;
- the motion source generation changed;
- a UCVH dirty brick overlaps the hit and has no semantic move event.

## Debug And Diagnostics

Keep `REVOLUMETRIC_VPT_DEBUG_VIEW=motion`, but define it as motion-vector
delta visualization, not previous absolute pixel visualization.

Add or preserve debug views for:

- motion validity/confidence;
- motion source class once flags exist;
- UCVH invalidated regions;
- rigid `motion_id` missing/generation mismatch;
- rejected previous clip `w <= 0`;
- disocclusion / out-of-bounds previous pixel.

These views are required before visual tuning because temporal ghosting and
valid noise look similar without guide diagnostics.

## Testing Strategy

Required RED tests before implementation:

- ABI tests for `GpuVptMotionGuide` or updated `GpuVptSurfacePixel` layout.
- Shader source tests asserting the motion convention:
  `previous_pixel_center = float2(pixel) + 0.5 + motion.xy`.
- Shader source tests proving consumers no longer treat `motion.xy` as absolute
  previous pixel coordinates.
- Unit/source tests for behind-camera rejection:
  `previous_clip.w <= 1.0e-5`.
- Frame-state tests proving missing previous history sets invalid motion.
- Rust tests for stable `motion_id` and previous transform retention.
- UCVH tests proving ordinary `set_voxel` / `write_brick` invalidates affected
  regions and semantic moves emit a region motion event.
- Existing ReSTIR-DI, Area ReSTIR, and VPT temporal source tests updated to the
  new motion contract.

Verification commands for the implementation phase:

```powershell
cargo fmt
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
git diff --check
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; cargo run --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; $env:REVOLUMETRIC_VPT_DEBUG_VIEW='motion'; cargo run --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES; Remove-Item Env:\REVOLUMETRIC_VPT_DEBUG_VIEW
```

## Implementation Phasing

Phase 1: Rename and lock the motion guide convention.

- Add shared shader helpers for delta motion.
- Convert consumers from absolute previous pixel to delta motion.
- Fix previous-clip `w <= epsilon` rejection.
- Preserve existing behavior for static scenes except for the sign/contract
  change.

Phase 2: Add motion source ABI and CPU history.

- Add `MotionSource` and previous transform retention.
- Extend scene/entity transform data beyond translation-only API boundaries.
- Add tests for missing/generation-mismatched motion sources.

Phase 3: Add rigid hit support.

- Carry `motion_id` and local hit position through the relevant surface-hit
  path.
- Project previous rigid position into previous frame.
- Reject history when source identity is invalid.

Phase 4: Add UCVH invalidation and semantic region motion.

- Track dirty/invalidation regions from UCVH edits.
- Invalidate affected hits by default.
- Add explicit semantic move events for translated voxel regions.

Phase 5: Expand debug views and runtime captures.

- Verify slow camera pan, rigid object movement, UCVH edit, UCVH semantic move,
  and disocclusion scenes.
- Capture `motion`, `history_valid`, and final output before visual tuning.

## Non-Goals

- Do not implement optical flow.
- Do not implement final motion blur in this phase.
- Do not copy source code from NRD, RTXDI, FSR2, or Area ReSTIR.
- Do not claim deformable/skinned motion support until hit data can reconstruct
  a previous position for those surfaces.
- Do not fabricate motion for arbitrary voxel creation/deletion.

## Risks

- Sign convention mistakes cause subtle ghosting across all temporal passes.
- Negative previous clip `w` currently risks accepting invalid behind-camera
  history.
- UCVH edits are content changes, not motion. Treating them as velocity will
  smear newly created/deleted voxels.
- Adding flags in a float texture is fragile. A separate `r32ui` image or
  structured buffer is cleaner for long-term source-class diagnostics.
- Rigid motion needs stable identity. Reusing `motion_id` without generation
  tracking can reproject unrelated objects.

## Review Gate

This document is the convergence point for the motion-vector scope. No
production code should change until the implementation plan is written from
this spec and reviewed.
