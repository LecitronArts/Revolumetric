# VCT/VPT Rendering MVP Design

## Goal

Revolumetric now targets a Voxel Cone Tracing first renderer with a small progressive Voxel Path Tracing reference mode, an explicit postprocess stage, and a stricter render-pipeline boundary.

The MVP target is not a full production engine. It is a stable, verifiable Rust/Vulkan prototype where:

- `cargo run` initializes one primary voxel scene path.
- Former Radiance Cascades runtime code, shaders, and docs have been removed from the active path.
- VCT is the default indirect-lighting path.
- VPT is an optional progressive/reference path that uses the same voxel traversal contract and does not require hardware ray tracing.
- Lighting outputs linear HDR and postprocess owns exposure, tonemap, and gamma.
- RenderGraph work has moved from dependency-only sorting toward explicit resource access declarations and barrier planning.

## External Basis

Voxel Cone Tracing follows the Crassin et al. 2011 direction: hierarchical voxel data plus approximate cone marching to estimate visibility and incoming energy interactively. The current MVP uses this idea in a simpler form than the paper: it samples through the existing dense UCVH-backed traversal and shader helpers rather than introducing a sparse voxel octree or a separate production radiance cache.

The render graph direction follows Vulkan synchronization guidance and established renderer practice: resources declare read/write access and image layout intent so pass-boundary barriers can be planned. For this project, current MVP scope is single-queue compute/transfer/graphics ordering with tests for barrier planning before async compute, aliasing, or full transient allocation.

Postprocess follows the common HDR pipeline split: lighting writes linear HDR, postprocess applies exposure, ACES tonemapping, gamma encoding, and writes an LDR storage image before blitting/presenting.

## Migration Boundary

RC is no longer a current renderer feature. Historical references in this document are limited to explaining the migration/deletion boundary:

- The old RC pass, probe buffer, shared shader include, pass shaders, runtime fields, profiler scopes, traversal stats slots, and old RC docs were deleted.
- New docs should not describe RC as selectable, compiled, or maintained current functionality.
- RC behavior is not preserved behind feature flags in the MVP.

## Current Runtime Pipeline

Default VCT mode:

1. UCVH upload/update.
2. Primary ray pass writes G-buffer images: position, material/albedo, normal/depth, and hit metadata.
3. Lighting pass reads G-buffer and UCVH resources, computes direct light plus VCT indirect/fallback lighting, and writes linear HDR `rgba16f`.
4. Postprocess pass reads HDR, applies exposure, ACES tonemap, gamma, and writes LDR `rgba8`.
5. Blit/present copies postprocess output to the swapchain.

Optional VPT mode:

1. UCVH upload/update.
2. Primary ray pass writes G-buffer images.
3. VPT pass traces stochastic voxel paths with bounded bounces and writes/updates an HDR accumulation image.
4. Postprocess reads VPT accumulation instead of lighting output.
5. Blit/present copies postprocess output to the swapchain.

## Runtime Configuration

- `REVOLUMETRIC_RENDER_MODE=vct|vpt`: selects default VCT path or optional VPT reference path. Default: `vct`.
- `REVOLUMETRIC_VCT=on|off|1|0|true|false`: enables VCT indirect lighting. Default: enabled.
- `REVOLUMETRIC_VPT_MAX_BOUNCES=1..8`: bounds VPT path length. Default: `2`.
- `REVOLUMETRIC_EXPOSURE=<finite non-negative float>`: exposure multiplier before tonemap. Default: `1.0`.
- `REVOLUMETRIC_LIGHTING_SHADOWS=on|off|1|0|true|false`: enables direct-light shadow rays.
- `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=on|off|1|0|true|false`: skips backface shadow hits when enabled.
- `REVOLUMETRIC_LIGHTING_DEBUG_VIEW=final|off|diffuse|direct|normal`: selects runtime lighting debug output.

Invalid values produce parse warnings and retain defaults for the invalid setting.

## VCT MVP

VCT MVP is a compute-shader path inside lighting.

Inputs:

- Scene UBO.
- Primary G-buffer images.
- UCVH config, hierarchy, brick occupancy, and material buffers.
- Shader-side VCT helper code in `vct_common.slang`.

Current behavior:

- VCT is default-on and can be disabled with `REVOLUMETRIC_VCT`.
- Direct lighting and shadow behavior remain available independently of VCT.
- Indirect lighting has conservative fallbacks so missing/zero VCT contribution does not force black output.
- Shader source tests verify the VCT include/helper path and absence of RC dependencies.

Remaining limits:

- No production radiance cache or radiance mip resource is implemented yet.
- Cone quality is MVP-level and should not be treated as final GI quality.
- Descriptor ABI validation is still test/source based rather than generated reflection.

## VPT MVP

VPT is a progressive voxel path tracing reference mode.

Inputs:

- Same scene UBO, G-buffer, and UCVH traversal resources used by the main path.
- Per-frame accumulation image.
- Sample index / stochastic seed state.
- `REVOLUMETRIC_VPT_MAX_BOUNCES`.

Current behavior:

- Selected with `REVOLUMETRIC_RENDER_MODE=vpt`.
- Traces bounded stochastic paths using voxel traversal.
- Accumulates into HDR history and feeds postprocess.
- Source tests cover stochastic sampling, sample index, bounce count, traversal entry, and accumulation.

Remaining limits:

- No denoiser is implemented.
- Accumulation reset/invalidation should be live-validated across resize, camera motion, and scene changes.
- It is a reference/debug mode, not the default production path.

## Postprocess MVP

Postprocess is an explicit compute pass.

Current behavior:

- Reads linear HDR `rgba16f`.
- Applies `scene.exposure`.
- Applies ACES tonemap.
- Applies gamma correction.
- Writes LDR `rgba8` for blit/present.
- Source tests verify HDR input, LDR output, exposure, ACES, gamma, and app ordering.

Remaining limits:

- Bloom is not implemented.
- Display controls are limited to exposure and fixed tonemap/gamma behavior.

## RenderGraph MVP

Current behavior:

- Passes declare imported and created image/buffer resources.
- `AccessKind` maps access intent to Vulkan stage, access mask, and image layout.
- Graph compile validates unknown resources and cycles.
- Barrier planning tracks initial writes, write-to-read transitions, persistent VPT read/write accumulation, transfer blit, and present transitions.
- App graph declarations use explicit compute and transfer access kinds around the MVP pipeline.
- Active image barriers are emitted by RenderGraph; primary, lighting, VPT, postprocess, and blit passes only bind/dispatch/copy.

Remaining limits:

- No transient GPU allocation, resource aliasing, descriptor automation, async compute scheduling, or full frame graph lifetime management.
- Swapchain presentation is graph-declared for the current single-queue blit/clear paths, but not yet a full presentation abstraction.

## Testing Strategy

Validation matrix:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
rg -n "REVOLUMETRIC_RENDER_MODE|REVOLUMETRIC_VCT|REVOLUMETRIC_VPT_MAX_BOUNCES|REVOLUMETRIC_EXPOSURE" README.md docs/superpowers
```

Required coverage areas:

- RC deletion boundary: no active runtime module exports, shader includes, or env configuration for RC.
- Scene UBO ABI tests for VCT/VPT/postprocess controls.
- Shader source tests for VCT, VPT, postprocess, and tonemap ownership.
- RenderGraph access mapping and barrier planner tests.
- GpuProfiler scope stability after RC removal.

## Risks

- VCT quality depends on radiance data that is not yet a first-class resource.
- VPT without denoising is noisy and should be used as a reference/debug path.
- RenderGraph barrier automation can be overbuilt; current scope must stay single-queue and testable.
- Future transient allocation, aliasing, and async compute need fresh synchronization validation rather than assuming the single-queue MVP generalizes.
- Strict shader compilation depends on `slangc` being available and may fail in CPU-only environments.
