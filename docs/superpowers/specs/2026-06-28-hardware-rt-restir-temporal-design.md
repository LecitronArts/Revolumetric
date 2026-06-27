# Hardware RT + ReSTIR-DI/GI + Temporal History Design

## Goal

Introduce a hardware ray tracing backend for Revolumetric, then layer ReSTIR-DI and ReSTIR-GI on top of it, while keeping the image post-process denoising stage temporal-only for this phase.

The target end state is:

- A hardware RT-backed primary renderer is available on RT-capable devices.
- ReSTIR-DI is available first for direct lighting.
- ReSTIR-GI is added on the same RT backend for indirect lighting.
- Final image stability uses an in-house temporal history pass only.
- NRD, SER, and path guiding are deferred to later work.

## Current Facts

- The repository is currently VPT-only in active runtime docs and code.
- `src/render/device.rs` enables swapchain, buffer device address, compute shader derivatives, and dynamic rendering, but not ray tracing pipeline or acceleration structure extensions.
- Existing code already has render graph, pass wrappers, settings parsing, history state, and ReSTIR-related research scaffolding.
- Current ReSTIR-DI, Area ReSTIR, and VPT temporal docs are useful references for settings and history conventions, but they are not a hardware RT backend.

## Scope Boundary

### In Scope

- Add a new RT renderer path and make it the primary supported path on RT-capable hardware.
- Keep VPT as a fallback/reference path.
- Build RT scene acceleration from the current voxel/brick world representation.
- Add ReSTIR-DI first, then ReSTIR-GI.
- Add a temporal-only history pass for final image stabilization.
- Add explicit history reset on camera cut, resize, scene change, and acceleration-structure rebuild.

### Out of Scope

- NRD integration.
- Shader Execution Reordering.
- Path guiding.
- Area ReSTIR.
- Async compute overhaul.
- Descriptor automation or resource aliasing work unrelated to this feature.
- Replacing the voxel world with triangles.

## Recommendation

Use a staged rollout, not a single leap:

1. RT backend + surface/hit buffers + temporal history.
2. ReSTIR-DI on the RT backend.
3. ReSTIR-GI on the same RT backend.
4. Optional spatial ReSTIR reuse only after temporal history and lighting are stable.

Reasoning:

- The backend risk and algorithm risk should not be mixed in the same release step.
- RT validation is easiest when the first output is a simple surface/hit path.
- ReSTIR-DI gives a direct-light win quickly and is easier to debug than GI.
- ReSTIR-GI should reuse the same scene, history, and reservoir conventions once DI is stable.
- Temporal-only denoising keeps the image chain understandable while NRD is handled elsewhere.

## Architecture

### 1. RT Scene Backend

Inference from the current codebase: because the scene is voxel/brick based, the lowest-risk hardware RT mapping is a procedural primitive AS over occupied brick AABBs, with brick-local voxel/material evaluation preserved in shader code.

Recommended shape:

- Build acceleration structures from occupied voxel bricks.
- Rebuild only dirty bricks or dirty regions when UCVH changes.
- Use ray tracing pipeline shaders for primary visibility, direct-light visibility, and indirect path evaluation.
- Keep brick-local material lookup and any fine voxel logic in shader code.

### 2. Render Modes

Add a new render mode for the RT path.

- `rt` is the primary path on supported hardware.
- `vpt` remains as a reference/fallback path.
- If RT support is unavailable, the app may fall back to VPT with a warning instead of failing hard.

### 3. Pass Structure

Recommended first-phase pipeline:

1. UCVH upload/update.
2. Acceleration-structure rebuild for dirty voxel/brick regions.
3. RT primary surface pass writes hit state, material state, depth, and motion guides.
4. ReSTIR-DI candidate generation and temporal reuse.
5. ReSTIR-GI candidate generation and temporal reuse.
6. RT lighting resolve uses the selected reservoirs.
7. Temporal history pass reprojects and clamps final HDR.
8. Postprocess tonemaps and writes LDR.
9. Blit presents.

### 4. Temporal History

The temporal pass is not NRD. It should be a conservative reprojection and accumulation stage only.

Recommended rejection rules:

- camera cut or resize invalidates history
- acceleration-structure rebuild generation invalidates history
- previous pixel outside the viewport invalidates history
- hit/miss mismatch invalidates history
- normal threshold failure invalidates history
- depth or world-position threshold failure invalidates history
- material-class mismatch invalidates history
- invalid motion guide invalidates history

Recommended accumulation behavior:

- maintain explicit history length and confidence
- clamp history growth
- prefer stable pixels over global frame resets
- keep raw and temporally filtered outputs separate

## ReSTIR-DI and ReSTIR-GI

### ReSTIR-DI

ReSTIR-DI should be the first algorithmic layer on the RT backend.

Minimum viable behavior:

- sample direct-light candidates
- evaluate visibility on the RT backend
- store reservoir state explicitly
- reuse reservoirs temporally
- resolve direct lighting in the final lighting pass

Spatial reuse is intentionally deferred until temporal behavior is stable.

### ReSTIR-GI

ReSTIR-GI should reuse the same RT backend and history conventions, but operate on indirect path vertices.

Minimum viable behavior:

- trace indirect path samples on the RT backend
- store path-vertex reservoir state explicitly
- reuse reservoirs temporally
- resolve indirect lighting in the final lighting pass

The first GI version should stay conservative. Do not couple it to path guiding in this phase.

## Data Model

Keep the feature-specific state explicit instead of growing unrelated shared buffers.

Recommended resources:

- RT surface state images or buffers
- direct-light reservoir buffers
- indirect-light reservoir buffers
- previous-frame history buffers
- motion / reprojection guide buffers
- light table or emissive cluster buffer
- RT acceleration-structure scratch and SBT resources

Recommended settings:

- `REVOLUMETRIC_RENDER_MODE=rt|vpt`
- `REVOLUMETRIC_RT_RESTIR_DI=on|off`
- `REVOLUMETRIC_RT_RESTIR_GI=on|off`
- `REVOLUMETRIC_RT_TEMPORAL_DENOISE=on|off`
- `REVOLUMETRIC_RT_TEMPORAL_HISTORY_LENGTH=1..64`
- `REVOLUMETRIC_RT_TEMPORAL_NORMAL_THRESHOLD`
- `REVOLUMETRIC_RT_TEMPORAL_DEPTH_THRESHOLD`

Use a dedicated RT debug view enum rather than overloading VPT debug names.

## Testing Strategy

Required checks:

- source tests for `RenderMode::Rt` and env parsing
- source tests for RT extension / pipeline setup
- source tests for AS and SBT usage in the RT backend
- ABI tests for surface, reservoir, and temporal history data
- render graph ordering tests for RT, ReSTIR, temporal, postprocess, and blit
- runtime smoke on RT-capable hardware
- fallback smoke when RT is unavailable
- history reset tests for camera cut, resize, and AS rebuild

Additional verification:

- confirm temporal-only history does not claim NRD behavior
- confirm no SER or path-guiding settings are introduced in phase 1
- confirm VPT reference path still compiles and can be selected

## Risks

- A procedural brick-AABB AS may be slower than a triangle-style scene, but it matches the current voxel world better than a forced geometry conversion.
- ReSTIR-GI history state can grow quickly if it is not kept narrow.
- Temporal-only denoising may be insufficient for very low-sample scenes, but that is acceptable for this phase.
- RT hardware availability is not universal, so fallback behavior must remain explicit and testable.

## Research Basis

- Vulkan ray tracing pipeline: https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_ray_tracing_pipeline.html
- Vulkan ray tracing overview: https://docs.vulkan.org/spec/latest/chapters/raytracing.html
- ReSTIR-DI paper: https://research.nvidia.com/labs/rtr/publication/bitterli2020spatiotemporal/
- ReSTIR-GI paper: https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing
- Shader Execution Reordering overview: https://developer.nvidia.com/rtx-kit

