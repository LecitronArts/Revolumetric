# Area ReSTIR Reference Design

## Decision

Area ReSTIR is worth keeping as a later VPT research target, but it should not replace the current VPT-only temporal denoise plan.

For this project, the immediate quality problem is that camera motion still behaves like a global progressive-accumulation reset: stable parts of the image lose useful history instead of preserving per-pixel reprojected radiance. Area ReSTIR does not solve that first-order problem. The current priority remains:

```text
vpt_surface -> restir_di_initial -> restir_di_temporal -> restir_di_spatial
-> vpt -> vpt_temporal -> vpt_atrous -> postprocess
```

Area ReSTIR becomes useful after that chain is stable, especially for subpixel voxel silhouettes, thin emissive details, high-frequency normal/material variation, depth of field, antialiasing, foliage-like structures, and path samples whose primary ray lives inside a pixel or lens area instead of only at the pixel center.

## Terminology

Do not use "Area ReSTIR" to mean "ReSTIR for area lights".

- ReSTIR-DI is direct-light reservoir reuse. It chooses and reuses light samples for a visible surface, usually in screen space.
- RTXDI is NVIDIA's SDK implementation family for ReSTIR-DI, ReGIR, ReSTIR-GI, and ReSTIR-PT.
- Area ReSTIR, from "Area ReSTIR: Resampling for Real-Time Defocus and Antialiasing", extends reservoir resampling into each pixel's 4D primary-ray area: 2D film/subpixel area plus 2D lens area.
- Area ReSTIR still contains direct-light sampling in the public implementation, but the important contribution for this project is sample-area reuse, not just direct-light reservoir math.

## External References

Primary Area ReSTIR references:

- Paper/project page: <https://dqlin.xyz/pubs/2024-sig-AREA/>
- NVIDIA project page: <https://research.nvidia.com/labs/rtr/publication/zhang2024area/>
- Public code: <https://github.com/guiqi134/Area-ReSTIR>
- Local inspection path: `target/research/Area-ReSTIR`

Related ReSTIR references:

- ReSTIR-DI paper: <https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/>
- Generalized RIS / ReSTIR foundations: <https://research.nvidia.com/publication/2022-07_generalized-resampled-importance-sampling-foundations-restir>
- RTXDI public SDK: <https://github.com/NVIDIA-RTX/RTXDI>
- Local inspection path: `target/research/RTXDI`

Use these references for algorithms, pipeline shapes, validation ideas, and terminology. Do not copy source code into this repository without an explicit license review.

Local clone note:

- `target/research/Area-ReSTIR` was inspected at `master`, commit `c705a82`.
- `target/research/RTXDI` was inspected at `main`, commit `1b55517`.
- `target/research/RTXDI/Libraries/Rtxdi` exists but is empty in this local clone. Do not document or depend on local RTXDI runtime library files unless the submodule is initialized later. Current local RTXDI evidence is from docs and samples.

## Local Research Findings

### Area ReSTIR Repository

The repository is a Falcor 7.0 fork with Area ReSTIR integrated into the Falcor `PathTracer` render pass.

Important local files:

- `target/research/Area-ReSTIR/README.md`
- `target/research/Area-ReSTIR/scripts/PathTracerAreaReSTIR.py`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/AreaReSTIR.h`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/AreaReSTIR.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/Reservoir.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/Resampling.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/PixelAreaSampleData.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/TemporalResampling.cs.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/TemporalResampling_FloatMotion.cs.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/TemporalMSAATracePrimaryRays.cs.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/SpatialResampling.cs.slang`
- `target/research/Area-ReSTIR/Source/Modules/AreaReSTIR/EvaluateFinalSamples.cs.slang`
- `target/research/Area-ReSTIR/Source/RenderPasses/PathTracer/PathTracer.slang`
- `target/research/Area-ReSTIR/Source/RenderPasses/PathTracer/GeneratePaths.cs.slang`

The README states that Area ReSTIR extends reservoirs to 4D ray space, including film and lens areas, and that copying only `Source/Modules/AreaReSTIR` into another project is insufficient because Falcor source changes are also required to store previous-frame scene data. This is a strong signal that this is not a drop-in module for Revolumetric.

The inspected source confirms several design constraints:

- `Reservoir.slang` stores selected light sample, `M`, weight, target PDF, subpixel sample UV, lens sample UV, and path sample.
- `PixelAreaSampleData.slang` stores current and previous subpixel/lens UV textures and can retrace primary rays with previous camera data.
- `Resampling.slang` implements random-replay and primary-hit reconnection shift mappings, pairwise MIS weights, Jacobian evaluation, lens validity checks, and visibility checks.
- `AreaReSTIR.h` owns explicit options for temporal reuse, spatial reuse, normal/depth thresholds, history length, subpixel reuse, lens reuse, previous-frame scene data, unbiased mode, and debug output.
- `PathTracerAreaReSTIR.py` wires a render graph that emits `subPixelUV`, `lensUV`, depth, motion vectors, and VBuffer outputs before the path tracer.

Project implication: if we implement Area ReSTIR later, it must be a VPT-owned sample-area subsystem with explicit history resources. It cannot be bolted onto the current `vpt.slang` as a few reservoir fields.

### RTXDI Repository

Important local files:

- `target/research/RTXDI/README.md`
- `target/research/RTXDI/Doc/Integration.md`
- `target/research/RTXDI/Doc/NoiseAndBias.md`
- `target/research/RTXDI/Doc/ShaderAPI.md`
- `target/research/RTXDI/Samples/MinimalSample`
- `target/research/RTXDI/Samples/FullSample`
- `target/research/RTXDI/Samples/FullSample/Shaders/LightingPasses/DI`
- `target/research/RTXDI/Samples/FullSample/Shaders/LightingPasses/PT`
- `target/research/RTXDI/Samples/FullSample/Source/RenderPasses/LightingPasses/ReSTIRDIRenderPasses.cpp`
- `target/research/RTXDI/Samples/FullSample/Shaders/LightingPasses/DI/GenerateInitialSamples.hlsl`
- `target/research/RTXDI/Samples/FullSample/Shaders/LightingPasses/DI/TemporalResampling.hlsl`
- `target/research/RTXDI/Samples/FullSample/Shaders/LightingPasses/DI/SpatialResampling.hlsl`
- `target/research/RTXDI/Samples/FullSample/Shaders/LightingPasses/DI/FusedResampling.hlsl`
- `target/research/RTXDI/Samples/FullSample/Shaders/LightingPasses/DI/ShadeSamples.hlsl`
- `target/research/RTXDI/Samples/MinimalSample/Shaders/Render.hlsl`

RTXDI is more directly useful for the current project than Area ReSTIR because the current implementation already has ReSTIR-DI-shaped passes and buffers.

The local clone exposes sample-level ReSTIR-DI code. The integrable RTXDI runtime library is advertised by the README as `Libraries/Rtxdi`, but that directory is empty locally because the submodule is not populated. Treat RTXDI runtime API details as official-repository or documentation references, not as locally inspected source.

Useful RTXDI lessons for Revolumetric:

- Keep the application bridge explicit. Revolumetric owns UCVH traversal, materials, light table, visibility, and RenderGraph barriers.
- Direct-light reuse needs a correct target PDF. A bad target PDF can remain unbiased if nonzero where needed, but it will be noisy and may boil.
- Temporal and spatial reuse are separate quality/performance levers. Spatial reuse after temporal reuse helps break overly stable blotchy noise patterns.
- Bias correction depends on correct previous-frame surface and light data. Current-frame-only temporal reuse is not an industrial-grade solution.
- Reservoir buffers often need several screen-sized arrays. Do not design in-place reuse unless the graph and shader prove it safe.
- Disocclusion needs explicit handling, often with more spatial samples or shorter history near newly visible pixels.

## Fit With Current Revolumetric State

Current project facts from the active worktree:

- VCT files are already deleted from the dirty tree or scheduled for deletion.
- `src/app.rs` currently wires `vpt_surface`, optional ReSTIR-DI passes, `vpt`, and `postprocess`.
- `src/render/vpt_history.rs` and `assets/shaders/shared/vpt_history_common.slang` exist as the VPT history ABI.
- Existing docs define the target temporal-denoise path in `docs/superpowers/specs/2026-05-02-vpt-only-temporal-denoise-design.md`.
- Existing ReSTIR-DI docs define direct-light-only scope in `docs/superpowers/specs/2026-05-02-restir-di-vpt-design.md`.

The renderer still needs VPT temporal radiance accumulation and atrous denoising before Area ReSTIR can pay off. Without stable radiance history, Area ReSTIR would reduce some primary sample noise while the whole image still refreshes during camera motion.

## Comparison To Current Options

### Option A: Continue VPT Temporal Denoise First

This is the recommended current path.

Benefits:

- Directly addresses the visible camera-motion problem.
- Uses surface/history resources that are already being introduced.
- Builds the prerequisites Area ReSTIR needs later: previous surface, motion, camera history, rejection, debug views.
- Keeps ReSTIR-DI focused on direct-light reuse instead of mixing three research topics at once.

Cost:

- Does not improve lens/subpixel sample reuse immediately.
- Thin silhouettes and high-frequency voxel edges still depend on normal temporal accumulation and atrous filtering for now.

### Option B: Add Area ReSTIR After VPT Temporal Is Stable

This is the recommended later path.

Benefits:

- Better handles defocus, antialiasing, subpixel detail, and thin features.
- Reuses the VPT surface/history infrastructure from the current plan.
- Can be built as an opt-in VPT feature and compared against baseline ReSTIR-DI plus temporal denoise.

Cost:

- Requires sample-area state, previous subpixel/lens data, retracing or reconnection, pairwise MIS, Jacobian logic, and more debug views.
- Increases memory and pass cost.
- Has higher bias/debugging risk than basic VPT temporal denoising.

### Option C: Replace The Current Plan With Area ReSTIR Now

This is rejected.

Reasons:

- It does not directly solve global accumulation reset.
- It would require more prerequisites than the current renderer has.
- It would delay the denoiser and make visual debugging harder.
- It would mix direct-light reuse, temporal denoise, sample-area reuse, and anti-aliasing into one unreviewable implementation.

## Proposed Later Architecture

Area ReSTIR should be a separate VPT-owned subsystem. Suggested future pipeline:

```text
vpt_surface
-> restir_di_initial
-> restir_di_temporal
-> restir_di_spatial
-> area_restir_initial_area_samples
-> area_restir_temporal
-> area_restir_spatial
-> area_restir_resolve
-> vpt
-> vpt_temporal
-> vpt_atrous
-> postprocess
```

This exact pass count may be reduced after profiling, but the resource boundaries should stay explicit while the implementation is being validated.

Important boundary:

- ReSTIR-DI reservoirs choose direct-light samples.
- Area ReSTIR reservoirs choose primary-ray sample-area state: subpixel UV, lens UV, path sample, and selected light/sample contribution context.
- VPT temporal history accumulates radiance and moments.

Do not store all three meanings in one reservoir struct.

## Future Data Model

Suggested resources:

- `area_restir_subpixel_uv`: current selected subpixel sample per pixel/pass.
- `area_restir_lens_uv`: current selected lens sample per pixel/pass.
- `area_restir_prev_subpixel_uv`: previous selected subpixel sample.
- `area_restir_prev_lens_uv`: previous selected lens sample.
- `area_restir_reservoirs`: current area reservoirs.
- `area_restir_prev_reservoirs`: previous area reservoirs.
- `area_restir_eval_context`: compact surface/material/normal/depth context for selected samples.
- `area_restir_prev_eval_context`: previous-frame selected-sample context.
- `area_restir_mis_jacobian`: optional scratch buffer for reconnection MIS validation.
- `area_restir_debug`: debug output texture or buffer.

Suggested reservoir fields:

```rust
#[repr(C)]
pub struct GpuAreaRestirReservoir {
    pub light_id: u32,
    pub path_sample: u32,
    pub sample_count_m: u32,
    pub flags: u32,
    pub weight: f32,
    pub target_pdf: f32,
    pub confidence: f32,
    pub jacobian: f32,
    pub subpixel_uv: [f32; 2],
    pub lens_uv: [f32; 2],
    pub sample_radiance: [f32; 4],
}
```

This is only a future ABI sketch. It must be redesigned with tests before implementation.

## Required Prerequisites Before Implementation

Do not start Area ReSTIR implementation until these are true:

- VPT is the only active renderer.
- VPT surface state exists and has current/previous ping-pong ownership.
- VPT temporal accumulation is per-pixel reprojected, not global `sample_index` averaging.
- Atrous or equivalent spatial denoising exists with debug views.
- ReSTIR-DI has surface-aware temporal/spatial rejection.
- Debug views can show history validity, motion vector, normal/depth rejection, reservoir weight, and raw/temporal/filtered radiance.
- Strict shader compilation and short runtime smoke are green.

## Implementation Strategy When Activated

Phase 0: write a dedicated spec and implementation plan. Use TDD and keep it on a separate worktree.

Phase 1: add sample-area ABI only.

- Rust/Slang ABI structs for subpixel/lens sample state.
- Tests for struct size, fields, and shader bindings.
- No visual behavior change.

Phase 2: add subpixel AA-only reuse.

- Thin-lens disabled.
- Reuse only film/subpixel sample state.
- No reconnection shift initially; use conservative random replay and strict rejection.
- Compare against baseline VPT temporal denoise.

Phase 3: add lens/DOF reuse.

- Store current and previous lens UV.
- Add lens validity and aperture checks.
- Add debug views for selected lens sample and history acceptance.

Phase 4: add reconnection shift and pairwise MIS.

- Implement only after random replay is correct.
- Add Jacobian diagnostics.
- Add bias-risk toggles and visual tests.

Phase 5: integrate with final VPT denoiser.

- Decide whether area-resampled output feeds noisy radiance, direct-only radiance, or a separate guide signal.
- Re-tune temporal/atrous rejection for noisy G-buffer risk.

## Validation Requirements

Required tests:

- ABI tests for Rust and Slang sample-area structs.
- Source tests proving Area ReSTIR resources are separate from ReSTIR-DI resources.
- RenderGraph ordering tests for area passes.
- Resize/history invalidation tests.
- Camera-cut invalidation tests.
- Debug-view source tests.
- Strict shader compilation.
- Runtime smoke with Area ReSTIR off and on.

Required visual/debug captures:

- Static camera convergence comparison.
- Slow camera pan with stable background and noisy disocclusion only.
- Subpixel thin geometry/voxel silhouette.
- Thin emissive details.
- DOF scene with foreground/background focus transitions if lens reuse is enabled.
- Debug views for subpixel UV, lens UV, reservoir weight, history valid, and rejection reason.

Success criteria:

- Moving the camera must not refresh the whole frame's noise.
- Newly revealed regions may be noisy, but stable reprojected regions should preserve history.
- Area ReSTIR must improve subpixel/DOF/high-frequency cases over VPT temporal denoise alone.
- It must not create persistent bias, ghosting, darkening around discontinuities, or denoiser-incompatible blotches.

## Risks

- Bias from reusing samples across incompatible surfaces or lens/subpixel domains.
- Ghosting if previous-frame scene data is incomplete or wrong.
- Darkening/brightening from incorrect pairwise MIS or Jacobian terms.
- Denoiser conflict because Area ReSTIR can intentionally vary primary sample locations, making G-buffer guides noisier.
- Large memory cost from multiple screen-sized reservoirs and scratch buffers.
- Debugging burden: failures look like normal path-tracing noise unless debug views exist first.
- License ambiguity: the Area ReSTIR root license is permissive, but inspected source headers include restrictive NVIDIA wording. Treat the code as reference only until reviewed.
- RTXDI license is an NVIDIA RTX SDK license with distribution and usage restrictions. Treat RTXDI as an architecture reference, not source material for copying.

## Current Implementation Status

Area ReSTIR is now an opt-in VPT-owned sample-area subsystem rather than a deferred design note. It remains separate from ReSTIR-DI:

- ReSTIR-DI reservoirs choose direct-light samples.
- Area ReSTIR reservoirs choose primary-ray sample-area state: film/subpixel UV, lens UV, pixel sample, path sample, and current-surface validation state.
- VPT temporal denoise still owns radiance accumulation after tracing.

The current graph shape is:

```text
vpt_surface
-> optional restir_di_initial / restir_di_temporal / restir_di_spatial
-> optional area_restir_initial / area_restir_temporal / area_restir_spatial
-> area_restir_history_update
-> vpt
-> vpt_temporal
-> postprocess
-> blit
```

Runtime controls:

- `REVOLUMETRIC_AREA_RESTIR=on|off|1|0|true|false`
- `REVOLUMETRIC_AREA_RESTIR_TEMPORAL=on|off|1|0|true|false`
- `REVOLUMETRIC_AREA_RESTIR_SPATIAL=on|off|1|0|true|false`
- `REVOLUMETRIC_AREA_RESTIR_SUBPIXEL=on|off|1|0|true|false`
- `REVOLUMETRIC_AREA_RESTIR_LENS=on|off|1|0|true|false`
- `REVOLUMETRIC_AREA_RESTIR_INITIAL_CANDIDATES=1..16`
- `REVOLUMETRIC_AREA_RESTIR_SPATIAL_SAMPLES=0..8`
- `REVOLUMETRIC_AREA_RESTIR_HISTORY_LENGTH=1..64`
- `REVOLUMETRIC_AREA_RESTIR_DEBUG=off|subpixel|lens|weight|history_valid|rejection|jacobian`

Debug visualization can also be selected directly with:

- `REVOLUMETRIC_VPT_DEBUG_VIEW=area_subpixel|area_lens|area_weight|area_history_valid|area_rejection|area_jacobian`

Area debug views are emitted by `vpt.slang` into noisy radiance and bypass `vpt_temporal.slang` smoothing/clamping, so the final displayed frame shows the actual selected Area ReSTIR reservoir state.

Remaining high-risk items:

- Random replay is conservative; full reconnection shift and pairwise MIS are not yet implemented.
- Lens reuse exists in the ABI, but default aperture is zero, so DOF behavior still needs a dedicated scene and capture pass.
- Visual quality still needs runtime captures for static convergence, slow camera pan, disocclusion, subpixel thin geometry, and rejection/debug overlays.
