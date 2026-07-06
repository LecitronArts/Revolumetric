# RT Scene GPU Rebuild State Design

## Goal

Keep `RtSceneBackend` CPU scene state consistent with GPU acceleration structure resources when RT scene GPU rebuilds or updates fail.

## Current Facts

- `RtSceneBackend::rebuild` updates `build_generation`, `dirty_generation`, `brick_bounds`, `bounds_initialized`, and `last_rebuild_sampled_bricks`.
- `RtSceneBackend::rebuild_gpu` currently calls `self.rebuild(ucvh)` before building or updating GPU resources.
- Full GPU rebuild can fail after CPU state has already advanced.
- If CPU state advances while old GPU resources remain, a later frame can treat the scene as unchanged and keep tracing stale acceleration structures.
- `RtSceneBackend::rebuild` has existing CPU-only tests that should keep passing.

## Design

Introduce an internal `RtSceneRebuildState` value that represents the next CPU scene state without mutating `RtSceneBackend`.

Add a private helper on `RtSceneBackend`:

- `plan_rebuild(&self, ucvh: &Ucvh) -> RtSceneRebuildState`
- It computes the same dirty-region and brick-bound logic currently inside `rebuild`.
- It includes whether acceleration-structure geometry changed.

Add a private commit helper:

- `commit_rebuild_state(&mut self, state: RtSceneRebuildState)`
- It updates sampled count, dirty generation, bounds, initialization flag, and increments `build_generation` only when geometry changed.

Update `rebuild` to call `plan_rebuild`, commit immediately, and return `state.scene_changed`. This preserves the CPU-only API.

Update `rebuild_gpu` to:

1. Plan the CPU state without mutating `self`.
2. If the planned scene is unchanged and GPU resources exist, commit metadata-only state and return.
3. Build AS inputs from the planned brick bounds, not from `self.brick_bounds`.
4. For an empty planned scene, clear GPU resources first, then commit the planned empty CPU state.
5. For in-place updates, update mapped inputs and record update first, then commit planned CPU state.
6. For full rebuilds, create and record new GPU resources first, then destroy old resources, install new resources, and commit planned CPU state.

## Non-Goals

- Do not change shader code.
- Do not change RT graph pass order.
- Do not add GPU-dependent tests that require constructing Vulkan devices.
- Do not change public render settings or UI in this phase.

## Testing Strategy

Use focused source-contract tests for the GPU path because GPU allocation failure injection is not currently available without adding a test-only Vulkan abstraction.

Add tests proving:

- `rebuild_gpu` plans state before GPU build/update work and does not call `self.rebuild(ucvh)`.
- full GPU rebuild commits CPU state only after new resources are created, recorded, and installed.
- empty-scene GPU rebuild clears resources before committing the planned empty CPU state.
- the CPU-only `rebuild` path still commits through the same helper.

Run:

- `REVOLUMETRIC_SHADER_COMPILE=skip cargo test render::rt_scene::tests::... --lib`
- full lib tests, clippy, strict shader lib tests before committing.

## Risks

The tests are source-contract tests rather than a true GPU failure injection test. This is acceptable for this phase because the project already uses source contracts for Vulkan lifecycle invariants, and adding an injectable AS builder would be a larger refactor.
