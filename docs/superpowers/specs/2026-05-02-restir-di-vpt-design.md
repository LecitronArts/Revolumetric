# ReSTIR-DI VPT Development Reference

## Goal

Add ReSTIR-DI as the direct-light estimator used by the VPT reference path.

This document is a research-backed development reference, not an implementation plan. It fixes the algorithm boundary, project constraints, resource model, and verification targets for the later implementation work.

The target is:

- Keep `REVOLUMETRIC_RENDER_MODE=vpt` as the opt-in reference/debug mode.
- Replace the current one-off direct sun term inside VPT with a reservoir-resampled direct lighting estimate.
- Use ReSTIR-DI for direct illumination only. It does not replace VCT indirect lighting, does not make VPT denoised, and does not implement ReSTIR GI/PT.
- Stay compatible with the current UCVH shader contract: Slang `StructuredBuffer<T>` resources and ordinary indexing, not shader-side BDA dereference.

## External Basis

Primary paper:

- Benedikt Bitterli, Chris Wyman, Matt Pharr, Peter Shirley, Aaron Lefohn, and Wojciech Jarosz, "Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting", SIGGRAPH 2020. Project page: <https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/>. PDF: <https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf>.

Industrial and open-source references:

- NVIDIA RTXDI: <https://github.com/NVIDIA-RTX/RTXDI>. Useful as an architecture and parameter reference for ReSTIR-DI/RTX Direct Illumination. Do not copy code directly into this repository without a license review; the SDK has its own NVIDIA license file at <https://github.com/NVIDIA-RTX/RTXDI/blob/main/LICENSE.txt>.
- NVIDIA technical overview: <https://developer.nvidia.com/blog/?p=21722>. Useful for high-level product framing: RTXDI uses reservoir-based spatiotemporal importance resampling to reduce direct-light shadow ray count for many dynamic lights.
- Kajiya renderer: <https://github.com/EmbarkStudios/kajiya>. Useful as a Rust-oriented real-time GI renderer reference, but it is experimental and not a drop-in design for this project.
- ReSTIR course notes by Chris Wyman: <https://intro-to-restir.cwyman.org/presentations/2023ReSTIR_Course_Notes.pdf>. Useful for implementation math and terminology checks.
- ReSTIR GI and later ReSTIR PT papers are related but out of current scope. Use them only when planning indirect/path-space reuse after ReSTIR-DI is stable.

Project-local basis:

- `docs/superpowers/specs/2026-05-02-vct-vpt-rendering-mvp-design.md` defines the current VCT/VPT/postprocess/RenderGraph boundary.
- `docs/superpowers/plans/2026-05-02-vct-vpt-rendering-mvp.md` records the current implementation state and validation matrix.
- `src/render/passes/vpt.rs` owns the VPT pass, descriptor layout, accumulation image, and dispatch.
- `assets/shaders/passes/vpt.slang` owns the current stochastic path tracing shader.
- `src/render/scene_ubo.rs` owns render mode, VPT bounce count, sample index, and ABI tests.
- `src/render/graph.rs` owns active image access transitions and must remain the owner of pass-boundary image barriers.

## Integration Boundary

ReSTIR-DI must not:

- Restore Radiance Cascades runtime code, shader includes, probe buffers, scene UBO fields, or docs as current functionality.
- Require hardware ray tracing. Visibility checks should use the existing voxel traversal path unless a separate hardware RT architecture decision is made later.
- Replace the VCT default renderer.
- Pretend to solve indirect lighting. Direct lighting at a VPT path vertex is in scope; path-space reuse across bounces is ReSTIR GI/PT territory and is out of scope for this stage.
- Add pass-local `cmd_pipeline_barrier` transitions. RenderGraph owns active image barriers.
- Copy RTXDI implementation code. The acceptable use is algorithm study, API shape comparison, parameter naming inspiration, and validation against public papers/docs.

## Current Runtime Pipeline

Current VPT mode:

1. UCVH upload/update.
2. VPT pass traces bounded stochastic voxel paths directly from the camera ray and writes an HDR accumulation image.
3. Postprocess reads VPT accumulation.
4. Blit/present copies postprocess output to the swapchain.

Important code-backed correction:

- Current VPT mode does not register the primary-ray graph pass before VPT.
- Existing `2026-05-02-vct-vpt-rendering-mvp` docs still describe VPT as `primary_ray -> VPT -> postprocess`; that is stale relative to `src/app.rs` and the source test in `src/render/passes/vpt.rs`.
- Any ReSTIR-DI design that needs G-buffer inputs must explicitly add a VPT-mode surface-state pass or reintroduce primary-ray output into the VPT branch as a deliberate pipeline change.

Current VPT shader behavior:

- `assets/shaders/passes/vpt.slang` includes `voxel_traverse.slang`, `material_common.slang`, and `lighting_common.slang`.
- Each pixel builds a stochastic camera ray using `scene.vpt_sample_index`.
- `trace_path()` loops up to `scene.vpt_max_bounces`.
- Each hit adds emissive material if hit directly.
- Direct sunlight is currently a local heuristic: `material_cell_albedo(hit.cell) * scene.sun_intensity * sun_term * 0.2`.
- The pass progressively averages into `accumulation_image`.

This is the exact point where ReSTIR-DI should replace the ad hoc direct-light term.

## Proposed ReSTIR-DI Pipeline Placement

Recommended MVP placement:

1. Keep ReSTIR-DI VPT-only.
2. Add explicit primary-surface state for VPT mode before temporal/spatial reuse is enabled.
3. Evaluate ReSTIR-DI at the primary visible surface first.
4. After primary-surface ReSTIR-DI is stable, allow the same direct-light sampling function at secondary diffuse hits without temporal/spatial reuse for those secondary vertices.

Reasoning:

- Screen-space temporal and spatial reuse is well-defined for primary visible surfaces because they have stable pixel coordinates, depth/normal/material, and history reprojection candidates.
- Secondary path vertices do not map cleanly to a persistent screen-space reservoir. Treating them as if they do would introduce biased or unstable reuse unless this becomes a full ReSTIR GI/PT design.
- This project currently has VPT as a reference/debug path, so the first high-value step is variance reduction in the directly visible direct-light estimate.

Two implementation shapes are possible:

### Option A: Self-Contained VPT Surface And Reservoir

VPT-owned passes trace the primary surface, write compact surface state, own reservoir resources, and perform candidate generation, temporal reuse, spatial reuse, shading resolve, and path tracing without using the primary-ray G-buffer images.

Pros:

- Avoids changing the default VCT primary-ray pipeline.
- Keeps feature scoped to VPT mode.
- Makes the current code fact explicit: VPT does not depend on `PrimaryRayPass`.

Cons:

- Duplicates some primary-surface data already produced by `PrimaryRayPass` in VCT mode.
- Spatial reuse needs reading neighbor reservoirs and writing a new reservoir, which is awkward in a single read/write resource.
- Harder to profile each ReSTIR stage.

### Option B: Add Primary-Ray Surface State To VPT Mode

Add explicit compute passes:

1. `primary_ray` or `vpt_surface`
2. `restir_di_initial_candidates`
3. `restir_di_temporal`
4. `restir_di_spatial`
5. `vpt` shading/path resolve reads final reservoirs

Pros:

- Reuses or mirrors existing G-buffer/surface-state contracts.
- Clear resource hazards and stage-level profiling.
- Easier tests for RenderGraph ordering.
- Better long-term fit for debug views and reservoir visualization.

Cons:

- More Rust pass/resource code.
- Requires ping-pong reservoir images/buffers.
- Changes the current VPT graph shape and must update `src/render/passes/vpt.rs` source tests that currently assert VPT does not register `primary_ray`.

Recommendation: implement Option B for MVP-quality engineering, but name the first pass `vpt_surface` unless the implementation intentionally reuses `PrimaryRayPass`. This keeps the current VPT independence explicit while still producing the stable surface data ReSTIR-DI needs. If a quick experiment is needed, prototype Option A in a branch, then promote to the staged pipeline before merging.

## Runtime Configuration

Add VPT-only environment settings through `LightingSettings` / `GpuSceneUniforms` or a focused ReSTIR config UBO:

- `REVOLUMETRIC_VPT_RESTIR_DI=on|off|1|0|true|false`
  - Default: `off` until the feature is validated, then consider `on` only for `REVOLUMETRIC_RENDER_MODE=vpt`.
- `REVOLUMETRIC_RESTIR_DI_TEMPORAL=on|off|1|0|true|false`
  - Default: `on` when ReSTIR-DI is enabled.
- `REVOLUMETRIC_RESTIR_DI_SPATIAL=on|off|1|0|true|false`
  - Default: `on` when ReSTIR-DI is enabled.
- `REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES=0..8`
  - Default: `4`.
- `REVOLUMETRIC_RESTIR_DI_HISTORY_LENGTH=1..64`
  - Default: `20`.
- `REVOLUMETRIC_RESTIR_DI_INITIAL_CANDIDATES=1..16`
  - Default: `1` for first implementation, `4` after light sampling is stable.
- `REVOLUMETRIC_RESTIR_DI_DEBUG=off|reservoir_weight|light_id|visibility|temporal_valid|spatial_neighbors`
  - Default: `off`.

Invalid values should produce parse warnings and retain defaults, matching existing `LightingSettings` behavior.

## Data And Resource Model

ReSTIR-DI is only useful if the renderer has a sampleable direct-light set. The current scene has sun lighting and emissive voxel materials, but no explicit many-light table. Add that first.

### Scene Light Table

Add a GPU buffer of direct-light candidates:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDirectLight {
    pub position_radius: [f32; 4],
    pub normal_type: [f32; 4],
    pub color_power: [f32; 4],
}
```

Semantics:

- `position_radius.xyz`: light center in world space.
- `position_radius.w`: conservative sphere or voxel proxy radius.
- `normal_type.xyz`: optional emitter normal for one-sided emitters; zero means omnidirectional.
- `normal_type.w`: `0 = sun`, `1 = emissive voxel/cluster`, future values reserved.
- `color_power.rgb`: linear radiance/color.
- `color_power.w`: importance scalar used by light sampling.

For the current voxel scene, emissive voxels should be clustered before upload. A one-light-per-emissive-cell table can explode memory and add noise from tiny emitters. MVP clustering can be coarse:

- Group emissive voxels by brick.
- Use brick-local emissive centroid and summed power.
- Clamp the light count with deterministic top-N by power if necessary.

### Reservoir Storage

Use per-pixel reservoir buffers or storage images at render resolution. Prefer buffers for explicit struct layout and easier ping-pong:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRestirDiReservoir {
    pub sample_light_id: u32,
    pub sample_flags: u32,
    pub sample_count_m: u32,
    pub pad0: u32,
    pub target_pdf: f32,
    pub weight_sum: f32,
    pub selected_weight: f32,
    pub confidence: f32,
    pub sample_position_pdf: [f32; 4],
    pub sample_radiance: [f32; 4],
}
```

Minimum fields:

- Selected light ID.
- Reservoir `M`.
- Weight sum.
- Target PDF for the selected sample.
- Final reservoir weight.
- Optional selected light position/radiance cache to reduce buffer reads.

Keep the struct 16-byte aligned and covered by Rust offset tests plus Slang ABI source tests.

### Ping-Pong Resources

Required resources:

- `restir_di_initial_reservoirs`
- `restir_di_temporal_reservoirs`
- `restir_di_spatial_reservoirs`
- `restir_di_history_reservoirs`
- `restir_di_history_gbuffer` or compact history surface state for reprojection validation

The final VPT shading pass reads `restir_di_spatial_reservoirs` when spatial reuse is enabled, otherwise temporal or initial reservoirs depending on enabled stages.

## Scene UBO And ABI Contract

Add fields only after deciding whether they belong in `GpuSceneUniforms` or a dedicated `RestirDiUniforms`.

Recommended: use a dedicated `RestirDiUniforms` buffer for feature-specific controls to avoid growing `GpuSceneUniforms` for every experimental renderer feature.

Suggested fields:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRestirDiUniforms {
    pub enabled: u32,
    pub temporal_enabled: u32,
    pub spatial_enabled: u32,
    pub debug_view: u32,
    pub initial_candidate_count: u32,
    pub spatial_sample_count: u32,
    pub history_length: u32,
    pub frame_index: u32,
    pub reservoir_count: u32,
    pub light_count: u32,
    pub width: u32,
    pub height: u32,
}
```

ABI rules:

- Rust and Slang structs must use the same field order.
- Add offset/size tests in Rust.
- Add shader source tests that check expected field names and resource bindings.
- Do not rely on implicit bool layout; use `u32`.

## G-buffer Inputs

ReSTIR-DI candidate evaluation needs:

- World position.
- Normal.
- Material albedo / rough material class.
- Hit/miss flag.
- Depth, camera key, or compact surface ID for temporal rejection.

Current VPT traces directly and does not consume the primary G-buffer. ReSTIR-DI temporal/spatial reuse should therefore use either:

- A new VPT-owned compact surface-state resource.
- Or a deliberate graph change that runs `PrimaryRayPass` in VPT mode and exposes its G-buffer outputs to ReSTIR-DI.

Do not write a ReSTIR-DI pass that silently assumes `gbuffer_pos`, `gbuffer0`, or `gbuffer1` exist in VPT mode.

Reprojection MVP:

- Start with same-pixel temporal reuse only when camera key is unchanged.
- Then add previous-view reprojection once previous `view_proj` or equivalent camera matrices are available.
- Reject history if normal dot is below threshold, position delta exceeds voxel-scale threshold, material ID differs, or previous reservoir light ID is invalid.

## Candidate Light Sampling

Initial candidate generation:

1. For each valid primary hit pixel, draw `N` candidate lights from `GpuDirectLight`.
2. Compute the candidate target contribution:
   - Geometry term.
   - Emitter intensity.
   - Surface cosine term.
   - Material albedo/BRDF approximation.
   - Visibility from hit point to light proxy.
3. Convert the candidate into a reservoir sample with `target_pdf / proposal_pdf`.
4. Update the reservoir using weighted reservoir sampling.

Proposal distribution:

- MVP: power-weighted alias table or prefix-sum CDF over `GpuDirectLight.color_power.w`.
- Fallback: uniform light ID sampling only for early tests; mark it as lower quality in debug output.

Visibility:

- Use the current UCVH `trace_primary_ray`-style traversal for shadow rays.
- For sun, test along `scene.sun_direction` with a long ray.
- For voxel emitters, test toward the selected light proxy center with a max distance.
- Offset the origin by surface normal to avoid self-shadowing, using the same scale as existing shadow/VPT rays.

## Reservoir Update Rules

Use the standard weighted reservoir pattern:

- Track selected sample `y`.
- Track `w_sum`.
- Track candidate count `M`.
- When considering candidate `x` with weight `w`, set `w_sum += w`, `M += 1`, and select `x` with probability `w / w_sum`.
- After all candidates, compute final contribution weight from `w_sum`, `M`, and the selected sample target PDF using the chosen bias correction mode.

Bias handling:

- First implementation may use a conservative biased mode if documented and debugged.
- The development target should include an unbiased or pairwise-MIS-compatible mode after basic images are stable.
- Every bias mode must be explicit in shader constants/config. Do not hide it as a magic formula inside the shader.

## Temporal Reuse

Temporal reuse combines the current initial reservoir with a previous-frame reservoir if history is valid.

MVP validity checks:

- Same render resolution.
- ReSTIR-DI enabled in both frames.
- Same `REVOLUMETRIC_RENDER_MODE=vpt`.
- Same or reprojected pixel maps inside viewport.
- Hit state matches.
- Position difference below a voxel-scale threshold.
- Normal dot above threshold.
- Material/light table version compatible.
- Camera movement invalidates same-pixel reuse until reprojection is implemented.

History resource lifecycle:

- Resize destroys and recreates all reservoir/history resources.
- Light table rebuild increments a light-table version and invalidates history.
- Scene/UCVH changes invalidate history unless a stable per-light ID scheme is implemented.

## Spatial Reuse

Spatial reuse samples neighboring pixels and combines compatible reservoirs.

MVP rules:

- Use a fixed small radius, initially 1-2 pixels.
- Use `REVOLUMETRIC_RESTIR_DI_SPATIAL_SAMPLES` random neighbor picks.
- Reject neighbors with miss/hit mismatch, large normal difference, large position delta, or invalid reservoir.
- Read from temporal reservoirs and write into a separate spatial reservoir buffer.
- Do not read and write the same reservoir buffer in-place.

Neighbor selection:

- Start with random offsets in a small square.
- Later add blue-noise or frame-rotated patterns if visible structure appears.

## Shading Resolve In VPT

At the primary surface:

1. Read the final reservoir.
2. Re-evaluate selected light contribution and visibility unless the chosen stage explicitly stores trusted visibility.
3. Add weighted direct-light contribution to `radiance`.
4. Continue the VPT path for indirect/emissive/sky contributions.

At secondary bounces:

- MVP should keep local one-sample direct lighting or disable direct-light next-event estimation at secondary vertices.
- Do not reuse screen-space reservoirs for secondary vertices.
- If direct lighting at secondary vertices is required, implement a local RIS-only estimator first, without temporal/spatial reuse.

## RenderGraph And Synchronization

ReSTIR-DI must declare all resources through RenderGraph.

Expected pass chain in VPT mode:

1. `vpt_surface` or an intentionally enabled `primary_ray`
2. `restir_di_initial`
3. `restir_di_temporal`
4. `restir_di_spatial`
5. `vpt`
6. `postprocess`
7. `blit_to_swapchain`

Access patterns:

- Initial pass writes initial reservoir buffer.
- Temporal pass reads initial reservoir and previous history, writes temporal reservoir.
- Spatial pass reads temporal reservoir, VPT surface state or G-buffer surface state, and writes spatial reservoir.
- VPT pass reads final reservoir and writes VPT HDR accumulation.
- A history update pass or end-of-frame copy writes next-frame history.

Current RenderGraph boundary:

- Image and buffer barriers are implemented for declared single-queue accesses. ReSTIR-DI reservoir buffers must continue to be declared through RenderGraph imports/accesses rather than relying on pass-local barriers.
- Do not add manual pass-local buffer/image barriers as a workaround.
- Future async compute, transient aliasing, or descriptor automation must revalidate the buffer-barrier model instead of assuming the current single-queue plan generalizes.

## VCT/VPT Interaction

VCT remains the default renderer. ReSTIR-DI belongs to VPT mode first.

Later reuse into the VCT lighting path is possible only after:

- The direct-light table is stable.
- Reservoir resources and history invalidation are robust.
- Debug views show stable temporal/spatial reuse.
- The lighting pass can consume a direct-light reservoir without coupling VCT indirect lighting to VPT-only path state.

## Debugging And Profiling

Add debug views before relying on visual quality:

- Selected light ID hash.
- Reservoir `M`.
- Reservoir weight sum.
- Final contribution weight.
- Temporal history accepted/rejected.
- Spatial neighbor count accepted.
- Visibility result.
- Direct contribution only.

Add profiler scopes:

- `restir_di_initial_ms`
- `restir_di_temporal_ms`
- `restir_di_spatial_ms`
- `restir_di_resolve_ms` if resolve is separate from VPT

Profiler CSV column order must have tests, following existing `GpuProfiler` tests.

## Testing Strategy

Required local validation:

```powershell
cargo fmt
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; cargo build --lib; cargo build --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_RENDER_MODE='vpt'; $env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'; .\target\debug\revolumetric.exe; Remove-Item Env:\REVOLUMETRIC_RENDER_MODE; Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
```

Required test coverage:

- Env parsing tests for ReSTIR-DI settings.
- Rust ABI size/offset tests for `GpuRestirDiUniforms`, `GpuDirectLight`, and `GpuRestirDiReservoir`.
- Shader source tests for ReSTIR-DI resource bindings and no pass-local barriers.
- RenderGraph compile tests for ReSTIR-DI pass ordering and resource accesses.
- History invalidation tests for resize, render mode switch, light table version change, and camera key change.
- Source grep proving RC is not restored.
- Strict shader compile with all ReSTIR-DI shaders.

Useful grep checks:

```powershell
rg -n -i "radiance cascade|radiance_cascade|rc_trace|rc_merge|rc_probe|REVOLUMETRIC_RC|LIGHTING_DEBUG_VIEW_RC|\bRC\b" README.md docs src assets/shaders reference
rg -n "cmd_pipeline_barrier|ImageMemoryBarrier|BufferMemoryBarrier" src/render/passes assets/shaders
rg -n "RESTIR|restir|reservoir|REVOLUMETRIC_VPT_RESTIR" README.md docs src assets/shaders
```

## Current Limits

- Current UCVH traversal consumes dense SSBO resources and `hierarchy_l0`; ReSTIR-DI visibility must work within that traversal limit until deeper hierarchy traversal is implemented.
- Current VPT has no denoiser.
- Current renderer now has a CPU-built direct-light table for ReSTIR-DI, including sun and brick-clustered emissive voxels. It is not yet consumed by VPT reservoir resolve.
- Current RenderGraph emits buffer barriers for declared single-queue buffer accesses, but does not yet own transient buffer lifetime or descriptor automation.
- Current descriptor ABI validation is source/test based, not generated from Slang reflection.
- Same-pixel temporal reuse is only valid when camera and scene state are unchanged. Real camera motion needs reprojection data.

## Risks

- Without a stable light table, ReSTIR-DI degenerates into a complicated wrapper around sun lighting and provides little value.
- Without robust history invalidation, temporal reuse will smear stale lights across camera moves, scene edits, or light-table rebuilds.
- If future ReSTIR-DI passes bypass RenderGraph-declared buffer accesses, separate reservoir stages can still introduce synchronization bugs.
- Without debug views, reservoir math failures look like ordinary path tracing noise and are hard to diagnose.
- Applying screen-space reservoirs to secondary path vertices would be a conceptual bug unless the work is reframed as ReSTIR GI/PT.
- Copying RTXDI code directly creates license and integration risk. Treat it as a reference, not a source drop.
