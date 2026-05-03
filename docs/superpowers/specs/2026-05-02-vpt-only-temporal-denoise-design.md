# VPT-Only Temporal Denoise Design

## Goal

Make VPT the only active renderer and remove VCT from runtime, shaders, settings, documentation, and validation expectations. The target renderer is an industrial-quality voxel path tracer with explicit surface history, temporal reprojection, ReSTIR-DI direct-light reuse, and an edge-aware denoising chain.

This is not a cleanup-only task. Removing VCT is a forcing function: the VPT path must become the product path, not a noisy debug fallback.

## Non-Goals

- Do not preserve `REVOLUMETRIC_RENDER_MODE=vct`.
- Do not keep `LightingPass`, `PrimaryRayPass`, `vct_common.slang`, or `lighting.slang` as active runtime paths.
- Do not restore Radiance Cascades, RC probes, or VCT indirect lighting under new names.
- Do not rely on pass-local barriers. RenderGraph remains the synchronization owner.
- Do not treat ReSTIR-DI alone as denoising. It is direct-light sample reuse, not a final image denoiser.
- Do not copy RTXDI or other licensed implementation code. Public references are for architecture and validation only.

## Current Code Facts

Snapshot after the active VPT-only migration work in this worktree:

- `src/render/scene_ubo.rs` has a VPT-only render mode and no active `RenderMode::Vct` default path.
- `src/app.rs` registers `vpt_surface` before optional ReSTIR-DI passes and the `vpt` pass.
- `src/render/passes/vpt_surface.rs` and `assets/shaders/passes/vpt_surface.slang` own the VPT surface-state resources.
- `src/render/vpt_history.rs` and `assets/shaders/shared/vpt_history_common.slang` define the VPT history/reprojection ABI.
- The old active VCT files are removed from this dirty tree: `src/render/passes/lighting.rs`, `src/render/passes/primary_ray.rs`, `assets/shaders/passes/lighting.slang`, `assets/shaders/passes/primary_ray.slang`, and `assets/shaders/shared/vct_common.slang`.
- `assets/shaders/passes/vpt.slang` now writes current-frame noisy HDR radiance and luminance moments; `vpt_temporal` performs the first temporal accumulation/reprojection baseline.
- ReSTIR-DI has surface-aware wiring, but the final product path still needs the Phase 5 atrous/debug-view denoiser work from the implementation plan.
- Phase 5 still needs the atrous spatial filter and debug-view output routing. Settings/ABI for `REVOLUMETRIC_DENOISER`, `REVOLUMETRIC_DENOISER_ATROUS_ITERATIONS`, and `REVOLUMETRIC_VPT_DEBUG_VIEW` have been added.

## Research Basis

The implementation should follow the existing research and open-source patterns below:

- ReSTIR direct lighting: candidate generation -> temporal reuse -> spatial reuse -> integration. The 2023 course notes emphasize starting with candidate samples, then validating spatial reuse before temporal reuse, and capping confidence/history to avoid runaway reuse.
- Rearchitecting Spatiotemporal Resampling for Production: decouple shading from sample forwarding when useful, use more cache-friendly candidate organization, and treat visibility reuse as a deliberate quality/performance trade-off.
- RTXDI reference implementation: the repo provides a minimal single-pass ReSTIR DI sample and a multi-pass sample integrated into a larger pipeline. The split is a useful engineering template for this repo as well.
- NRD: denoising wants per-pixel guides, especially normal, roughness, viewZ, and motion vectors, plus explicit history confidence and reset behavior.
- SVGF: temporal accumulation + luminance variance + edge-aware spatial filtering remain the right baseline for low-spp path tracing; the open-source reference explicitly uses normals, depth, variance, and albedo demodulation.
- BMFR and related reconstruction samples: the pre-process stage reprojects current pixels into the previous camera space, matches by world position and normal, and accumulates with a conservative moving average before blockwise or spatial reconstruction.
- Area ReSTIR is a later extension for subpixel/lens-area reuse, not the immediate fix for global VPT accumulation reset. See `docs/superpowers/specs/2026-05-02-area-restir-reference-design.md`.

Engineering rule derived from the above:

- VPT must own its surface/history contract; it cannot treat old VCT G-buffer state as an acceptable proxy.
- Temporal accumulation and ReSTIR temporal reuse should both consume explicit motion/history/reprojection data.
- Denoising inputs should stay demodulated and guided by world-space geometry, not by ad hoc color-only smoothing.

## Target Runtime Pipeline

The only normal frame pipeline is:

1. UCVH upload/update.
2. `vpt_surface` traces the primary visible voxel surface and writes compact surface state.
3. `restir_di_initial` samples direct-light candidates using `vpt_surface`.
4. `restir_di_temporal` reprojects previous reservoirs and validates history against current surface state.
5. `restir_di_spatial` combines compatible neighbor reservoirs.
6. `vpt` traces path radiance, consumes final ReSTIR-DI reservoirs for primary-bounce direct light, and writes noisy HDR radiance plus moments.
7. `vpt_temporal_accumulation` reprojects previous denoiser history and accumulates radiance/moments with rejection.
8. `vpt_atrous_denoise` runs edge-aware spatial filtering using surface normal/depth/material.
9. `postprocess` tonemaps filtered HDR into LDR.
10. `blit_to_swapchain` presents.

If a pass is disabled for debugging, it must be disabled by an explicit VPT-specific debug setting and still keep descriptor/resource safety intact.

## VCT Removal Boundary

Remove from active code:

- `RenderMode::Vct`, `RENDER_MODE_VCT`, and `REVOLUMETRIC_RENDER_MODE=vct`.
- `LightingSettings::vct_enabled` and any `LIGHTING_FLAG_VCT_ENABLED` semantics.
- `LightingPass` construction, resize, graph registration, profiling scope, tests, and shader compilation.
- `PrimaryRayPass` as a product rendering pass. If its logic is useful, create a VPT-owned `VptSurfacePass`; do not keep the old pass name or VCT-oriented tests.
- `assets/shaders/passes/lighting.slang`, `assets/shaders/shared/vct_common.slang`, and active build references to them.
- README statements that describe VCT as default, current, or supported.

Allowed historical references:

- Older `docs/superpowers/plans/2026-04-*` and migration notes may mention VCT as historical context if clearly marked as superseded.
- Tests may grep for absence of active VCT references in `README.md`, `src`, and `assets/shaders`.

## Surface State Contract

Create a VPT-owned surface state pass and ABI. It should be independent of the old primary-ray G-buffer.

Recommended GPU resources:

- `vpt_surface_position_depth`: `rgba32f`
  - `xyz`: world position.
  - `w`: linear camera-ray distance, `-1.0` for miss.
- `vpt_surface_normal_roughness`: `rgba16f` or `rgba32f`
  - `xyz`: world normal.
  - `w`: roughness or reserved material class value.
- `vpt_surface_albedo_material`: `rgba16f` or packed storage buffer
  - `rgb`: material albedo.
  - `a`: material id/class, `0` for miss.
- `vpt_motion_history`: `rgba32f`
  - `xy`: previous-frame pixel coordinate in normalized or pixel space.
  - `z`: history confidence.
  - `w`: validity flag.

The pass must share voxel traversal helpers with VPT but not the legacy primary-ray pass. Miss pixels must be explicit so temporal rejection and denoising do not reinterpret zero-filled buffers as valid geometry.

## Camera And Reprojection Data

The renderer needs a dedicated VPT history uniform or expanded scene uniform with:

- Current frame index.
- Current and previous camera-to-world or view-projection data.
- Current and previous resolution.
- Jitter used for current and previous frame.
- History reset generation.
- Flags for camera-cut, resize, and scene/light-table invalidation.

Minimum reprojection behavior:

- For each current hit, project its world position into the previous frame.
- Reject if previous pixel is outside the previous viewport.
- Reject if previous history surface is miss while current is hit, or the reverse.
- Reject if normal dot is below threshold.
- Reject if world-position delta exceeds a voxel-scale threshold.
- Reject if material id/class differs.
- Reject all history on resize, UCVH rebuild generation change, or light-table generation change.

The current `last_vpt_camera_key` sample-reset mechanism is not enough for industrial temporal reuse. It can remain temporarily as a reset signal during migration, but the final path must use explicit reprojection and rejection.

## ReSTIR-DI Upgrade

ReSTIR-DI must become surface-aware:

- `restir_di_initial` reads `vpt_surface_*` and only generates reservoirs for valid primary hits.
- Candidate weights include material albedo, cosine term, light power, distance/solid-angle approximation, and visibility.
- `restir_di_temporal` reads current surface state and previous surface/history state before accepting prior reservoirs.
- `restir_di_spatial` rejects neighbors with incompatible miss state, normal, position, and material.
- Final VPT resolve reads the selected reservoir and re-evaluates or validates visibility for the selected light.
- ReSTIR confidence should be explicit and capped. The first usable ceiling should be conservative, not unbounded accumulation.
- Neighbor choice for spatial reuse must not depend on the reused sample itself; choose the neighbor set independently, then evaluate reuse quality.
- If the initial pass can only afford one sample, it still needs a valid target PDF and confidence so later passes can reject or downweight it consistently.

Reservoir buffers should keep ping-pong ownership explicit:

- current initial
- current temporal
- current spatial
- previous history reservoir
- previous history surface

History update must be an explicit graph pass or an explicit end-of-frame copy with declared resources. No hidden CPU/GPU ordering assumptions.

## Denoising Chain

Use a staged SVGF-like design adapted to the voxel renderer:

1. VPT radiance pass writes noisy HDR radiance and first/second luminance moments.
2. Temporal accumulation reprojects previous filtered/accumulated radiance and moments.
3. Variance estimate is updated from luminance moments and history length.
4. Atrous spatial filter runs 3-5 iterations with edge stops:
   - normal angle
   - depth/world-position delta
   - material id/class
   - variance
5. Postprocess reads the filtered HDR output, not the raw VPT accumulation.

Implementation detail:

- Keep noisy radiance and moments separate. Filtering should not operate on display-tonemapped output.
- Use history confidence / accumulation length as an explicit signal, not an implicit result of loop count.
- Start with normal/depth/material edge stops; add variance and motion rejection before any more complex edge heuristics.
- If a frame is a camera cut, resize, or scene/light-table change, force a history restart rather than trying to salvage stale buffers.

The denoiser must support debug views before visual tuning:

- raw VPT radiance
- temporal accumulated radiance
- filtered final radiance
- history valid mask
- history length
- motion vector
- luminance variance
- normal rejection
- depth/position rejection
- ReSTIR reservoir weight
- direct-only and indirect-only contribution if available

## Accumulation Policy

Remove the old assumption that VPT progressive accumulation is the primary quality mechanism. In an interactive renderer:

- Temporal accumulation should be history/reprojection based, not simple `sample_index` averaging.
- `vpt_sample_index` may remain as a stochastic seed, but not as the core accumulation validity mechanism.
- Camera movement should reduce or reject history per pixel, not globally erase all useful stable pixels.
- Debug camera cuts and resize events must explicitly reset history generations.

## Resource And Lifetime Rules

- Every image and buffer transition must be declared through RenderGraph.
- Every descriptor set must have valid fallback resources even when a debug stage is disabled.
- Resize recreates all resolution-dependent VPT, ReSTIR, and denoiser resources.
- Scene/UCVH/light-table generation changes invalidate affected histories.
- No pass may read and write the same history resource in-place unless the shader and graph prove non-overlap; default to ping-pong.
- Resource names should include `vpt_`, `restir_di_`, or `denoise_` prefixes for capture/debug readability.

## Profiling

Replace VCT/primary/lighting profiling with VPT-stage scopes:

- `vpt_surface_ms`
- `restir_di_initial_ms`
- `restir_di_temporal_ms`
- `restir_di_spatial_ms`
- `vpt_trace_ms`
- `vpt_temporal_ms`
- `vpt_atrous_0_ms` or aggregate `vpt_atrous_ms`
- `postprocess_ms`
- `blit_to_swapchain_ms`
- `total_ms`

CSV column order must be tested. Profiling writes must keep the existing batched flush behavior.

## Settings

Remove:

- `REVOLUMETRIC_RENDER_MODE=vct`
- `REVOLUMETRIC_VCT_ENABLED`
- VCT debug aliases and VCT-specific flags.

Keep or add:

- `REVOLUMETRIC_RENDER_MODE=vpt`
- `REVOLUMETRIC_VPT_MAX_BOUNCES`
- `REVOLUMETRIC_VPT_RESTIR_DI`
- `REVOLUMETRIC_RESTIR_DI_TEMPORAL`
- `REVOLUMETRIC_RESTIR_DI_SPATIAL`
- `REVOLUMETRIC_RESTIR_DI_INITIAL_CANDIDATES`
- `REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES`
- `REVOLUMETRIC_RESTIR_DI_HISTORY_LENGTH`
- `REVOLUMETRIC_DENOISER=on|off`
- `REVOLUMETRIC_DENOISER_ATROUS_ITERATIONS=0..5`
- `REVOLUMETRIC_VPT_DEBUG_VIEW=final|raw|temporal|variance|history_valid|motion|normal|depth|reservoir_weight|direct|indirect`

Invalid values must warn and keep defaults.

## Testing Strategy

Required test groups:

- Source tests proving active code has no `RenderMode::Vct`, `LIGHTING_FLAG_VCT_ENABLED`, `LightingPass`, active `vct_common.slang`, or active `lighting.slang` dependency.
- Settings tests proving default runtime is VPT-only and `REVOLUMETRIC_RENDER_MODE=vct` is rejected or mapped to the VPT runtime.
- ABI size/offset tests for VPT surface, history, and denoiser uniforms.
- Shader source tests for VPT surface outputs, temporal reprojection, history rejection, atrous edge stops, and debug views.
- RenderGraph tests proving the target pass order and buffer/image barriers:
  `vpt_surface -> restir_di_initial -> restir_di_temporal -> restir_di_spatial -> vpt -> vpt_temporal_accumulation -> vpt_atrous_denoise -> postprocess -> blit`.
- Resize/history invalidation tests.
- Strict shader compilation.
- Runtime smoke test:
  `REVOLUMETRIC_EXIT_AFTER_FRAMES=3 cargo run`
- Debug runtime smoke for at least:
  `REVOLUMETRIC_VPT_DEBUG_VIEW=history_valid`
  `REVOLUMETRIC_VPT_DEBUG_VIEW=variance`
  `REVOLUMETRIC_DENOISER=off`

Full verification command set:

```powershell
cargo fmt
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --lib; cargo build --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
git diff --check
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; cargo run; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; $env:REVOLUMETRIC_VPT_DEBUG_VIEW='history_valid'; cargo run; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES; Remove-Item Env:\REVOLUMETRIC_VPT_DEBUG_VIEW
```

## Implementation Phasing

Phase 1 removes active VCT runtime and makes VPT the only boot/render path.

Phase 2 adds VPT surface state and graph-owned surface resources.

Phase 3 upgrades ReSTIR-DI temporal/spatial reuse to use surface reprojection and compatibility checks.

Phase 4 replaces progressive averaging with temporal accumulation and moments.

Phase 5 adds atrous denoising and debug views.

Phase 6 deletes remaining VCT files and stale current-path docs after active code no longer references them.

Suggested phase 1 order:

1. Switch settings and startup defaults to VPT-only.
2. Remove VCT graph creation and shader bindings from `src/app.rs`.
3. Delete or quarantine VCT-specific active files and tests.
4. Add VPT surface/history ABI and shader scaffolding.
5. Verify the app still launches and exits cleanly in a short smoke run.

Each phase must be TDD-driven and independently verifiable. Do not combine VCT deletion, reprojection, and denoising into one unreviewable diff.

## Risks

- Removing VCT before VPT startup is stable can leave no working renderer. Phase 1 must include runtime smoke before proceeding.
- Reprojection without surface-state validation will smear history during camera movement. Surface contracts must precede temporal accumulation.
- Denoising raw path-traced radiance without variance and edge stops will blur geometry and emissive detail.
- ReSTIR temporal reuse without generation invalidation can reuse reservoirs from incompatible light tables.
- Exact visual quality cannot be proven by unit tests alone. Debug views and capture/readback comparisons are required for final tuning.
