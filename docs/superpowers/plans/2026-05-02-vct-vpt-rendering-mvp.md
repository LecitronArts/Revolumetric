# VCT/VPT Rendering MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the renderer on a VCT-first path with optional VPT reference mode, explicit postprocess, and a stricter render-pipeline boundary.

**Architecture:** The current MVP has migrated away from the former Radiance Cascades experiment. Runtime rendering now flows through primary voxel rays, VCT lighting or VPT accumulation, postprocess, and blit; RenderGraph declares resource accesses and owns single-queue image barrier emission for the active pipeline.

**Tech Stack:** Rust 2024, ash/Vulkan 1.3, Slang compute shaders, UCVH SSBO traversal, bytemuck ABI tests.

---

## Current Implementation Snapshot

- [x] RC runtime modules, shaders, buffer type, profiler scopes, traversal slots, scene UBO fields, and old RC docs have been removed from the active runtime tree.
- [x] `LightingSettings` exposes VCT/VPT/postprocess controls through `REVOLUMETRIC_RENDER_MODE`, `REVOLUMETRIC_VCT`, `REVOLUMETRIC_VPT_MAX_BOUNCES`, and `REVOLUMETRIC_EXPOSURE`.
- [x] Lighting writes linear HDR to `R16G16B16A16_SFLOAT` / shader `rgba16f`.
- [x] `PostprocessPass` reads HDR, applies exposure, ACES tonemap, and gamma, then writes LDR `rgba8`.
- [x] VCT is the default indirect-lighting path and uses `vct_common.slang` from the lighting shader.
- [x] VPT exists as an optional progressive reference mode with stochastic sampling, bounded bounce count, and an accumulation image.
- [x] `GpuProfileScope` has primary ray, lighting, VPT, postprocess, blit, and total CSV columns.
- [x] RenderGraph supports imported images/buffers, access declarations, compile-time dependency validation, graph-owned image barrier emission, and barrier-plan tests for lighting/VPT -> postprocess -> blit.
- [x] Old RC documents were deleted. New docs may mention RC only as migration history or deletion boundary, not as current functionality.

## Runtime Pipeline

Default mode:

1. UCVH upload/update.
2. Primary ray pass writes G-buffer images.
3. Lighting pass reads G-buffer plus UCVH resources and writes linear HDR.
4. Postprocess pass maps HDR to LDR.
5. Blit copies postprocess output to the swapchain.

VPT mode:

1. UCVH upload/update.
2. Primary ray pass writes G-buffer images.
3. VPT pass traces stochastic voxel paths and updates an HDR accumulation image.
4. Postprocess reads the VPT accumulation image.
5. Blit copies postprocess output to the swapchain.

## Runtime Configuration

- `REVOLUMETRIC_RENDER_MODE=vct|vpt`: selects the default VCT lighting path or optional VPT reference path. Default is `vct`.
- `REVOLUMETRIC_VCT=on|off|1|0|true|false`: enables or disables VCT indirect contribution while keeping direct lighting/fallbacks available. Default is enabled.
- `REVOLUMETRIC_VPT_MAX_BOUNCES=1..8`: clamps VPT path length. Default is `2`.
- `REVOLUMETRIC_EXPOSURE=<finite non-negative float>`: postprocess exposure multiplier before tonemap. Default is `1.0`.
- `REVOLUMETRIC_LIGHTING_SHADOWS=on|off|1|0|true|false`: controls direct shadow rays.
- `REVOLUMETRIC_LIGHTING_SKIP_BACKFACE_SHADOWS=on|off|1|0|true|false`: controls backface shadow filtering.
- `REVOLUMETRIC_LIGHTING_DEBUG_VIEW=final|off|diffuse|direct|normal`: selects final, direct diffuse, or normal debug output.

Invalid values are reported as parse warnings and the default value for that setting is retained.

## Completed Checklist

### Task 1: Remove RC Runtime and ABI

- [x] Scene settings no longer expose RC debug, quality, normal strategy, or probe controls.
- [x] RC fields were removed from CPU and shader scene uniforms.
- [x] RC profiler scopes and traversal stats slots were removed.
- [x] RC app fields, initialization, dispatch, descriptor updates, pass barriers, and buffer swapping were removed.
- [x] RC pass/buffer/shader files and module exports were deleted.
- [x] New docs keep RC references historical only.

### Task 2: Split HDR Lighting and Postprocess

- [x] Lighting no longer owns final tonemap/gamma writes.
- [x] Lighting output is linear HDR.
- [x] Scene exposure/postprocess fields have ABI coverage.
- [x] Postprocess shader tests cover HDR input, LDR output, exposure, ACES, and gamma.
- [x] `PostprocessPass` reads an HDR storage image and writes an LDR output image.
- [x] Frame order is primary ray -> lighting/VPT -> postprocess -> blit.

### Task 3: Add VCT MVP

- [x] Lighting includes `vct_common.slang` and has no RC include/binding dependency.
- [x] VCT scene setting constants and env parsing exist.
- [x] VCT indirect lighting is gated by `REVOLUMETRIC_VCT`.
- [x] Direct lighting and ambient fallback remain available when VCT is disabled or returns no contribution.
- [ ] Remaining quality work: first-class radiance cache/mips and higher-quality cone sampling are not complete.

### Task 4: Add VPT Reference Mode

- [x] `REVOLUMETRIC_RENDER_MODE=vct|vpt` settings tests exist.
- [x] VPT shader source tests cover stochastic seed/sample index, bounce count, UCVH traversal entry, and accumulation image.
- [x] `VptPass` exists as an optional compute pass after primary rays.
- [x] Postprocess can consume VPT accumulation when VPT mode is active.
- [ ] Remaining runtime work: accumulation invalidation on all camera/scene changes needs live-run validation and may need broader dirty-state tracking.
- [ ] Remaining quality work: no denoiser is implemented; VPT is expected to be noisy.

### Task 5: Render Graph Barrier MVP

- [x] `AccessKind` maps declared accesses to Vulkan stage, access, and layout fields.
- [x] Imported image and buffer resource declarations exist.
- [x] Barrier plan tests cover lighting -> postprocess -> blit.
- [x] App pass declarations now use `read_as`/`write_as` for compute and transfer dependencies.
- [x] Active image layout/access transitions are emitted by RenderGraph rather than pass-local barriers.
- [ ] Remaining architecture work: no transient allocation, aliasing, descriptor automation, async compute, or full graph-owned GPU resource lifetime yet.

### Task 6: Documentation and Validation

- [x] README describes the VCT/VPT/Postprocess/RenderGraph shape instead of treating RC as current.
- [x] Runtime environment variables are documented.
- [x] Validation command matrix is documented.
- [ ] Full matrix should be rerun after any implementation change:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
rg -n "REVOLUMETRIC_RENDER_MODE|REVOLUMETRIC_VCT|REVOLUMETRIC_VPT_MAX_BOUNCES|REVOLUMETRIC_EXPOSURE" README.md docs/superpowers
```

## Current Limits

- VCT is an MVP shader path over existing UCVH traversal, not a production radiance-cache implementation.
- VPT is a reference/debug path with bounded bounces and progressive accumulation, not a denoised production renderer.
- Postprocess owns exposure/ACES/gamma but bloom and richer display controls are not implemented.
- RenderGraph owns active image barrier emission, but real GPU resource ownership and descriptor automation have not moved fully into the graph.
- Descriptor ABI reflection between Rust and Slang is still source/test based rather than generated reflection.
- This plan documents current worktree state; it does not prove the full validation matrix passed unless the commands are run fresh.
