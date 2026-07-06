# RT Scene GPU Rebuild State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep RT scene CPU state from advancing until GPU acceleration structure rebuild/update work has succeeded.

**Architecture:** Add a private planned rebuild state in `src/render/rt_scene.rs`. CPU-only `rebuild` commits the planned state immediately, while `rebuild_gpu` commits only after GPU resources have been updated, cleared, or replaced successfully.

**Tech Stack:** Rust, Vulkan via `ash`, existing source-contract tests, existing `REVOLUMETRIC_SHADER_COMPILE` validation gates.

---

### Task 1: Add Planned RT Scene Rebuild State

**Files:**
- Modify: `src/render/rt_scene.rs`

- [ ] **Step 1: Write failing source-contract tests**

Add tests in `src/render/rt_scene.rs`:

```rust
#[test]
fn rt_scene_rebuild_gpu_plans_state_without_committing_before_gpu_success() {
    let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
    let rebuild_gpu = source
        .split("pub fn rebuild_gpu")
        .nth(1)
        .expect("rebuild_gpu should exist")
        .split("pub fn tlas_handle")
        .next()
        .expect("rebuild_gpu should end before tlas_handle");
    let compact = crate::render::source_checks::compact(rebuild_gpu);

    assert!(
        !compact.contains("self.rebuild(ucvh)"),
        "GPU rebuild must not commit CPU scene state before GPU work succeeds"
    );
    let plan = compact
        .find("letrebuild_state=self.plan_rebuild(ucvh);")
        .expect("GPU rebuild must plan CPU state without mutating backend");
    let inputs = compact
        .find("RtSceneAsBuildInputs::from_brick_bounds(&rebuild_state.brick_bounds)")
        .expect("GPU inputs must use planned brick bounds");
    let commit = compact
        .rfind("self.commit_rebuild_state(rebuild_state);")
        .expect("GPU rebuild must commit planned state after GPU work succeeds");

    assert!(plan < inputs);
    assert!(inputs < commit);
}

#[test]
fn rt_scene_full_gpu_rebuild_commits_state_after_new_resources_are_installed() {
    let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
    let rebuild_gpu = source
        .split("pub fn rebuild_gpu")
        .nth(1)
        .expect("rebuild_gpu should exist")
        .split("pub fn tlas_handle")
        .next()
        .expect("rebuild_gpu should end before tlas_handle");
    let compact = crate::render::source_checks::compact(rebuild_gpu);

    let create = compact
        .find("letnew_resources=RtSceneGpuBuildResources::new(")
        .expect("full GPU rebuild must create replacement resources");
    let record = compact
        .find("new_resources.record_build(")
        .expect("full GPU rebuild must record the AS build");
    let install = compact
        .find("self.gpu_resources=Some(new_resources);")
        .expect("full GPU rebuild must install replacement resources");
    let commit = compact
        .find("self.commit_rebuild_state(rebuild_state);")
        .expect("full GPU rebuild must commit CPU state after resource installation");

    assert!(create < record);
    assert!(record < install);
    assert!(install < commit);
}

#[test]
fn rt_scene_empty_gpu_rebuild_clears_resources_before_committing_empty_state() {
    let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
    let rebuild_gpu = source
        .split("pub fn rebuild_gpu")
        .nth(1)
        .expect("rebuild_gpu should exist")
        .split("pub fn tlas_handle")
        .next()
        .expect("rebuild_gpu should end before tlas_handle");
    let compact = crate::render::source_checks::compact(rebuild_gpu);

    let empty = compact
        .find("ifinputs.aabbs.is_empty(){")
        .expect("empty planned scenes must be handled explicitly");
    let clear = compact[empty..]
        .find("self.clear_gpu_resources(")
        .map(|offset| empty + offset)
        .expect("empty planned scenes must clear stale GPU resources");
    let commit = compact[empty..]
        .find("self.commit_rebuild_state(rebuild_state);")
        .map(|offset| empty + offset)
        .expect("empty planned scenes must commit empty CPU state after clear");

    assert!(empty < clear);
    assert!(clear < commit);
}

#[test]
fn rt_scene_cpu_rebuild_commits_planned_state() {
    let source = crate::render::source_checks::read_source("src/render/rt_scene.rs");
    let rebuild = source
        .split("pub fn rebuild(&mut self, ucvh: &Ucvh) -> bool")
        .nth(1)
        .expect("CPU rebuild should exist")
        .split("pub fn rebuild_gpu")
        .next()
        .expect("CPU rebuild should end before GPU rebuild");
    let compact = crate::render::source_checks::compact(rebuild);

    assert!(compact.contains("letrebuild_state=self.plan_rebuild(ucvh);"));
    assert!(compact.contains("letscene_changed=rebuild_state.scene_changed;"));
    assert!(compact.contains("self.commit_rebuild_state(rebuild_state);"));
    assert!(compact.contains("scene_changed"));
}
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test render::rt_scene::tests::rt_scene_rebuild_gpu_plans_state_without_committing_before_gpu_success --lib
```

Expected: fail because `rebuild_gpu` still calls `self.rebuild(ucvh)`.

- [ ] **Step 3: Add private planned state and helpers**

Add near `RtSceneBackend`:

```rust
struct RtSceneRebuildState {
    dirty_generation: u32,
    brick_bounds: Vec<RtBrickBounds>,
    sampled_bricks: u32,
    scene_changed: bool,
}
```

Move the body of `rebuild` into:

```rust
fn plan_rebuild(&self, ucvh: &Ucvh) -> RtSceneRebuildState
fn commit_rebuild_state(&mut self, state: RtSceneRebuildState)
```

`commit_rebuild_state` increments `build_generation` only when `state.scene_changed` is true.

- [ ] **Step 4: Update `rebuild`**

Replace `rebuild` with:

```rust
pub fn rebuild(&mut self, ucvh: &Ucvh) -> bool {
    let rebuild_state = self.plan_rebuild(ucvh);
    let scene_changed = rebuild_state.scene_changed;
    self.commit_rebuild_state(rebuild_state);
    scene_changed
}
```

- [ ] **Step 5: Update `rebuild_gpu` commit ordering**

Use planned state and commit only after GPU success:

```rust
let rebuild_state = self.plan_rebuild(ucvh);
if !rebuild_state.scene_changed && self.gpu_resources.is_some() {
    self.commit_rebuild_state(rebuild_state);
    return Ok(());
}

let inputs = RtSceneAsBuildInputs::from_brick_bounds(&rebuild_state.brick_bounds);
...
self.commit_rebuild_state(rebuild_state);
```

- [ ] **Step 6: Verify focused RT scene tests pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'
cargo test render::rt_scene::tests:: --lib
```

Expected: all `rt_scene` tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/render/rt_scene.rs docs/superpowers/specs/2026-07-06-rt-scene-gpu-rebuild-state-design.md docs/superpowers/plans/2026-07-06-rt-scene-gpu-rebuild-state.md
git commit -m "fix: defer RT scene state commits until GPU rebuild succeeds"
```

---

## Self-Review

- Spec coverage: The plan covers planned state, CPU rebuild compatibility, GPU success-only commit, empty scene clear ordering, and validation.
- Placeholder scan: No TODO/TBD/deferred implementation placeholders remain.
- Type consistency: `RtSceneRebuildState`, `plan_rebuild`, and `commit_rebuild_state` are consistently named across tests and implementation steps.
