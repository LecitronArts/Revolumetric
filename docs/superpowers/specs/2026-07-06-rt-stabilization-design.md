# RT Stabilization Design

## Goal

Stabilize the hardware RT path so it is a trustworthy default renderer on RT-capable machines before adding more ReSTIR, GI, denoising, or scheduling complexity.

This phase turns current RT behavior from "passes compile and produce output" into "final RT output, fallback output, editor overlay composition, and visual baseline coverage have explicit contracts." It intentionally allows breaking internal RT ABI changes if they make the pipeline easier to reason about and verify.

## Current Facts

- The runtime supports `auto`, `vpt`, and `rt` render modes. `auto` selects RT on RT-capable devices and VPT otherwise.
- The hardware RT frame path currently performs AS update, surface generation, optional RT ReSTIR-DI, optional RT ReSTIR-GI, direct lighting, temporal accumulation, resolve, capture, and swapchain blit.
- The current working tree already contains uncommitted RT changes that add a sky/ground background to RT direct lighting and add a desktop egui overlay pass after successful RT swapchain blit.
- Focused source tests for the current RT direct-lighting background and RT egui overlay changes pass.
- `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib` passed with 722 tests.
- `REVOLUMETRIC_SHADER_COMPILE=strict cargo build --features desktop --bin revolumetric` passed.
- `.\run\validate-visual-baseline.ps1 -Rt` passed on this machine and produced an RT capture with `render_backend = rt`, `rt_debug_view = surface`, `rt_frame_rendered = true`, `rt_restir_di_rendered = true`, `rt_restir_gi_rendered = true`, and `rt_resolve_ready = true`.
- The existing RT visual baseline case is `rt_surface_debug`. It verifies RT surface debug output and active RT pass metadata, but it does not prove final RT direct-lighting output with `REVOLUMETRIC_RT_DEBUG_VIEW=off`.
- `cargo fmt --check` currently fails because one uncommitted RT Rust function signature is not rustfmt-formatted.

## Problem Statement

The current RT path has enough infrastructure to run, but three stability gaps make it weak as a foundation for deeper algorithm work:

1. RT miss/background direction is implicit. Current background shading derives a miss direction from `RtSurfacePixel.normal_roughness`. True primary misses write `normal = -ray_direction`, but the procedural AABB closest-hit path can produce an invalid surface after brick-local voxel traversal misses and write an AABB normal instead. That can shade final background with a primitive normal rather than the camera ray.
2. Desktop editor overlay composition is only covered on the successful RT resolve/blit path. If RT falls back to clear/present because UCVH, AS, descriptors, or required passes are not ready, the editor can still disappear even though that is the path where status UI is most useful.
3. RT visual baseline coverage verifies `surface` debug output, not final direct-lighting output. A final-output regression such as black miss pixels or incorrect background direction can pass the current RT baseline.

## Scope

### In Scope

- Make RT miss/background direction explicit and stable.
- Keep CPU and Slang ABI layout tests for any changed RT surface/direct-lighting fields.
- Ensure desktop egui overlay composition works after both successful RT resolve blit and fallback clear/present paths.
- Replace source-token-only graph confidence with a concrete RenderGraph contract test for swapchain blit/clear followed by color attachment overlay and present.
- Add an RT final/default visual baseline case that runs with `REVOLUMETRIC_RT_DEBUG_VIEW` unset or `off`.
- Keep existing RT surface debug baseline coverage.
- Fix the rustfmt failure caused by current RT edits.
- Preserve unrelated user/worktree changes, including IDE files and untracked folders, unless explicitly asked otherwise.

### Out of Scope

- New ReSTIR-DI or ReSTIR-GI sampling algorithms.
- NRD integration for the RT path.
- Shader Execution Reordering.
- Path guiding.
- Async compute scheduling.
- RenderGraph descriptor automation.
- Broad `RtRuntimePipeline` decomposition beyond small helper extraction required for this stabilization.

## Recommended Approach

Use a narrow stabilization pass, not a broad RT rewrite.

### 1. Explicit RT Background Direction

Preferred design: extend the RT surface ABI with an explicit background or primary-ray direction field that is valid for miss-like surfaces. The field must be written by the surface raygen/miss flow and preserved for invalid surfaces that direct lighting resolves as background.

Acceptable implementation shapes:

- Add a `view_direction_background` or equivalent vector to `RtSurfacePixel`, with a validity convention documented in `rt_history_common.slang`.
- Or repurpose existing unused padding only if the ABI remains clear and tests name the semantic field.
- Or reconstruct the primary ray in RT direct lighting from camera uniforms if that keeps ABI smaller and matches the existing scene uniform conventions.

The recommended option is a named RT surface field because it makes the contract local to the data being consumed by direct lighting and avoids duplicating camera ray reconstruction in a later lighting pass.

Required behavior:

- True primary misses shade sky/ground using the primary ray direction.
- AABB closest-hit followed by brick-local voxel traversal miss also shades sky/ground using the original primary ray direction, not the AABB normal.
- Valid voxel hits continue to use geometric normals for direct, indirect, temporal, and debug behavior.
- RT temporal compatibility must not treat miss/background pixels as voxel history.

### 2. Unified RT Swapchain Presentation Helper

Introduce a focused helper in `rt_pipeline.rs` for desktop overlay composition over any RT swapchain-producing path:

- Successful RT resolve path: resolve output is blitted to the swapchain, then egui records as a color attachment, then the swapchain is finished as present.
- Fallback clear path: clear writes the swapchain, then egui records as a color attachment, then the swapchain is finished as present.
- Android remains unchanged with overlay disabled by `cfg`.

The helper should avoid duplicating present-finalization logic across success and fallback paths. It should keep RenderGraph ownership of resource access transitions rather than recording ad hoc layout barriers in passes.

### 3. RT Final Visual Baseline

Add a new manifest case such as `rt_final`:

- `renderMode`: `rt`
- `rtDebugView`: absent or `off`
- RT ReSTIR-DI enabled
- RT ReSTIR-DI spatial enabled
- RT ReSTIR-GI enabled
- RT temporal denoise enabled
- expected backend: `rt`
- expected active RT pass metadata: true for frame, DI, GI, resolve
- signal thresholds at least as strong as current RT surface debug unless final output proves it needs separate thresholds

This case must verify final RT direct-lighting output, not only surface debug output. The existing `rt_surface_debug` case stays because it remains useful for geometry/surface diagnosis.

### 4. Tests And Contracts

Add or update tests in these categories:

- ABI layout tests for any RT surface or direct-lighting uniform changes.
- Shader source tests proving the direct-lighting background uses explicit primary/background direction rather than `normal_roughness` as a miss-direction proxy.
- Shader source tests proving the AABB local traversal miss path preserves primary/background direction.
- RenderGraph unit test for blit/clear followed by egui color attachment write and final present.
- Source check or manifest test proving visual baselines include both RT surface debug and RT final cases.
- Focused tests for RT fallback overlay helper behavior.

## Data Flow

1. `rt_surface.rgen.slang` builds the primary ray.
2. Surface payload carries both hit data and the primary/background direction semantic.
3. `rt_surface.rmiss.slang` and the AABB closest-hit local-miss path write an invalid/miss `RtSurfacePixel` whose background direction still refers to the camera ray.
4. `rt_direct_lighting.rgen.slang` resolves invalid/miss surfaces with sky/ground shading from the explicit background direction.
5. Valid voxel surfaces continue through ReSTIR-DI, ReSTIR-GI, direct lighting, temporal, and resolve.
6. The RT pipeline writes either resolved output or fallback clear output to the swapchain.
7. Desktop egui overlay writes after the RT swapchain image has been produced and owns the final present access.

## Error Handling

- If RT required passes are missing, UCVH upload is pending, AS resources are unavailable, or descriptors are missing, the fallback clear path should still present and still compose egui on desktop when an egui frame is present.
- If RT final visual baseline cannot run because RT hardware is unavailable, the existing `requiresRt` behavior should skip it rather than fail non-RT machines.
- Strict shader compilation remains the gate for shader ABI and syntax errors.

## Validation

Required verification after implementation:

```powershell
cargo fmt --check
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo build --features desktop --bin revolumetric; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
.\run\validate-visual-baseline.ps1 -Rt
git diff --check
```

Expected results:

- Formatting passes.
- Unit tests pass in both skip and strict shader modes.
- Clippy has no warnings.
- Desktop binary builds with strict shader compilation.
- RT visual baseline runs `rt_surface_debug` and the new RT final/default case on RT-capable hardware.
- Metadata for both RT cases records `render_backend = rt`, RT pass activity, and resolve readiness.
- PPM signal checks pass for the new final RT case.

## Risks

- Growing `RtSurfacePixel` increases per-pixel buffer bandwidth. This is acceptable for stabilization if the field has a clear semantic and avoids incorrect final output. If bandwidth becomes a measured bottleneck, a later optimization can pack the direction more tightly.
- Adding another RT visual baseline increases local validation time. The value is high because it closes a gap in final-output coverage.
- Refactoring swapchain presentation helpers can break fallback paths if final present access is not consistently declared. The RenderGraph test should catch this before runtime.

## Research Basis

- RTXDI keeps ReSTIR integration behind clear runtime/sample boundaries instead of folding all lighting, history, and presentation behavior into one implicit path: https://github.com/NVIDIAGameWorks/RTXDI
- ReSTIR-DI depends on stable direct-light reservoir and visibility semantics, so final-output contracts should be stabilized before expanding algorithm complexity: https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/
- ReSTIR-GI extends resampling to path vertices and increases history complexity, making explicit RT surface/history contracts more important before further GI work: https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
- Vulkan ray tracing relies on explicit acceleration structure, shader binding table, and synchronization contracts; the project should keep using RenderGraph-visible access declarations where possible: https://docs.vulkan.org/spec/latest/chapters/raytracing.html
