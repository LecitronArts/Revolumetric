# VPT Traversal Acceleration Design

## Goal

Reduce the cost of VPT voxel traversal without changing rendered semantics or making hardware ray tracing the default path. The first goal is to make traversal cost measurable. The second goal is to remove obvious wasted work in the current UCVH traversal. Only after that should the project consider larger structural changes such as bricked SVO or RT Core backed experiments.

## Current Facts

- The active renderer is VPT-only, and VPT uses the shared UCVH traversal contract.
- `trace_primary_ray` is used by VPT surface, VPT path tracing, and Area ReSTIR initial candidate evaluation.
- `trace_any_hit_ray_skip_voxel` is used for VPT shadow visibility.
- The current traversal already has a dense brick grid L0 plus L1-L4 occupancy hierarchy, and brick-local 8^3 DDA.
- The current GPU profiler reports pass-level times only.
- The profiling wrapper still expects an older `vpt_surface_ms` CSV column, while the profiler now emits `vpt_surface_bootstrap_ms` and `vpt_surface_selected_ms`.
- The device setup currently enables Vulkan swapchain and buffer device address, but not ray tracing pipeline or acceleration structure extensions.

## Non-Goals

- Do not switch the main renderer to Vulkan ray tracing.
- Do not replace the current UCVH path with a full SVO rewrite in this phase.
- Do not change shading, denoising, reservoir logic, or bounce semantics.
- Do not make traversal stats affect output.
- Do not remove the existing compute-based fallback path.

## Research Basis

- [A Fast Voxel Traversal Algorithm for Ray Tracing](https://diglib.eg.org/items/60c72224-00f3-416d-9952-ee41e8c408da): the current brick-local DDA should stay close to Amanatides-Woo style stepping.
- [Efficient Sparse Voxel Octrees](https://research.nvidia.com/publication/2010-02_efficient-sparse-voxel-octrees): a useful reference if the project later moves from fixed hierarchy to a more sparse brick hierarchy.
- [GigaVoxels](https://www.icare3d.org/research-cat/publications/gigavoxels-ray-guided-streaming-for-efficient-and-detailed-voxel-rendering.html): relevant for brick streaming and hierarchy-driven empty-space skipping at larger scales.
- [VK_KHR_ray_tracing_pipeline](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_ray_tracing_pipeline.html): required reading if the project later evaluates an RT Core backed backend.

## Options Considered

### Option A: Instrument current UCVH, then optimize the existing traversal

Add traversal counters, profile representative scenes, and then remove repeated work in the current hierarchy skip path.

Pros:
- Lowest risk.
- Preserves current data structures and shader contracts.
- Gives hard evidence before larger architecture work.

Cons:
- Does not unlock RT Core.
- Gains are limited by the current brick-grid design.

### Option B: Move to bricked sparse voxel octree

Replace the fixed hierarchy with a more traditional sparse octree where bricks live at the leaves.

Pros:
- Better fit for very large sparse worlds.
- Can reduce empty-space cost more aggressively.

Cons:
- Larger rewrite.
- More complex GPU updates and traversal state.
- Harder to validate against current behavior.

### Option C: Build a Vulkan ray tracing backend for occupied bricks

Use acceleration structures for coarse brick/AABB traversal and keep brick-local traversal in shader code.

Pros:
- Can use RT Core on supported GPUs.
- Good for experiments on high-end desktop hardware.

Cons:
- Platform-specific.
- Requires a new AS/SBT pipeline.
- Still needs the current brick-local traversal logic.

## Recommendation

Choose Option A.

The current code already has the important structural pieces: brick storage, occupancy hierarchy, and shared traversal helpers. The missing piece is observability. Without per-traversal counters, it is not possible to tell whether the next win is hierarchy skipping, shadow rays, repeated primary traversal, candidate count, or a deeper data-structure change.

## Success Criteria

- A profiling run can attribute traversal cost to primary rays, shadow rays, hierarchy skips, and brick-local voxel stepping separately.
- The profiling wrapper and profiler CSV names agree on the same VPT surface scope columns.
- At least one representative traversal metric improves on the current demo scene without changing the rendered output when stats are disabled.
- Existing shader source and Rust unit tests continue to pass after the measurement and optimization work.

## Architecture

Add a traversal stats path that is explicitly separate from rendering output:

- A small GPU stats buffer for per-frame traversal counters.
- Optional shader increments inside `trace_primary_ray`, `trace_any_hit_ray_skip_voxel`, and the hierarchy skip helper.
- A CPU readback / summary path that reports the counters alongside existing GPU pass timings.
- A profiling wrapper update so CSV output matches current profiler scope names.

After measurement, keep the current compute traversal but target the highest-value low-risk reduction:

- avoid repeated hierarchy loads in the skip helper where possible;
- avoid duplicate primary traversal when a prior pass already produced the same hit data;
- keep brick-local DDA unchanged unless a specific count shows it is the dominant cost.

If representative measurements still show traversal dominating frame time after those steps, then branch into a separate architecture decision for either bricked SVO or RT Core.

## Data Flow

1. CPU builds and uploads UCVH as it does now.
2. VPT surface, VPT path, and Area ReSTIR initial all call the shared traversal helpers.
3. Traversal helpers increment counters when stats are enabled.
4. GPU pass timing and traversal stats are collected for the same frame window.
5. The profiling wrapper aggregates the results and identifies which traversal stage dominates.

## Validation

- Keep existing shader source tests green.
- Add tests for the profiler column names used by the profiling wrapper.
- Add tests for traversal stats layout and counter presence.
- Run a representative profiling session and confirm the stats identify primary traversal, shadow traversal, and skip effectiveness separately.
- Confirm the rendered output does not change when traversal stats are disabled.

## Risks

- The current profiler already has a naming mismatch in the profiling wrapper, so measurements must be fixed before any performance claim is trusted.
- If the scene stays small and mostly dense, the hierarchy may already be near the practical limit of this design.
- A bricked SVO or RT Core path is a separate architecture decision and should not be smuggled into the current compute path.
