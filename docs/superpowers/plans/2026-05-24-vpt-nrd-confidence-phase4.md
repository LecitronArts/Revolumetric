# VPT NRD Confidence Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative `vpt_nrd_confidence` pass that produces NRD-shaped diffuse/specular confidence resources from existing VPT motion guide data without changing the SVGF product path.

**Architecture:** Keep the confidence pass local and optional: it reads named surface motion guide resources after `vpt_surface`, writes single-channel confidence textures, and is recorded only when `REVOLUMETRIC_DENOISER=relax|reblur` is requested. Official NRD dispatch and resolve remain out of scope for this phase, so the pass output is prepared for later adapter mapping but does not feed postprocess yet.

**Tech Stack:** Rust/ash render passes, RenderGraph image resources, Slang compute shader, existing shader reflection/source contract tests.

---

### Task 1: Source Contract Test

**Files:**
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Write the failing test**

Add `vpt_nrd_confidence_pass_declares_history_confidence_contract` that checks:
- `assets/shaders/passes/vpt_nrd_confidence.slang` exists.
- `src/render/passes/vpt_nrd_confidence.rs` exists.
- Shader bindings are:
  - 0 `scene_ubo`
  - 1 `motion_history`
  - 2 `motion_flags`
  - 3 `surface_brick_generation`
  - 4 `previous_surface_brick_generation`
  - 5 `diff_confidence`
  - 6 `spec_confidence`
- Rust descriptor specs match shader reflection.
- Shader includes `vpt_motion_common.slang`, calls `vpt_motion_flags_reject_history`, reconstructs `previous_pixel` from `motion_history[pixel].xy`, checks bounds, calls `vpt_surface_generation_rejects_history`, writes diffuse confidence as `0.0` or `1.0`, and writes specular confidence as `0.0`.
- Rust pass exposes named `VptNrdConfidenceGraphInputs` and `VptNrdConfidenceResources`, uses `R16_SFLOAT` images, and does not use magic `[ResourceHandle; N]` arrays.
- Pipeline creates/resizes/destroys the pass and records it only for `Relax | Reblur`, after surface guide generation and before NRD frontend.

- [ ] **Step 2: Run red test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt::shader_source_tests::vpt_nrd_confidence_pass_declares_history_confidence_contract -- --exact
```

Expected: FAIL because the shader/pass and wiring are not present yet.

### Task 2: Implement Shader And Rust Pass

**Files:**
- Create: `assets/shaders/passes/vpt_nrd_confidence.slang`
- Create: `src/render/passes/vpt_nrd_confidence.rs`
- Modify: `src/render/passes/mod.rs`

- [ ] **Step 1: Add shader**

Create `vpt_nrd_confidence.slang` with:
- `scene_ubo` and five storage image bindings described in Task 1.
- `motion_history` as `rgba32f`.
- motion flags and generation images as `r32ui`.
- confidence outputs as `r16f`.
- A helper that rounds `motion_history[pixel].xy` to the previous pixel, rejects out-of-bounds coordinates, rejects motion flags through `vpt_motion_flags_reject_history`, rejects generation mismatch through `vpt_surface_generation_rejects_history`, returns `1.0` only for valid history, and otherwise returns `0.0`.
- `spec_confidence[tid.xy] = 0.0;` until real specular denoising exists.

- [ ] **Step 2: Add Rust pass**

Create `VptNrdConfidencePass` following `vpt_nrd_frontend.rs` style:
- `descriptor_binding_specs() -> [DescriptorBindingSpec; 7]`.
- Own `diff_confidence` and `spec_confidence` `GpuImage`s.
- Output image format `vk::Format::R16_SFLOAT`.
- Usage `STORAGE | SAMPLED | TRANSFER_SRC`.
- `VptNrdConfidenceGraphInputs` carries `frame_slot`, `surface_inputs: VptCurrentSurfaceResources`, `previous_surface_inputs: VptPreviousSurfaceResources`, and optional profiler.
- `VptNrdConfidenceResources` carries named `diff_confidence` and `spec_confidence`.
- `register_graph` reads `surface_inputs.motion_history`, `surface_inputs.motion_flags`, `surface_inputs.brick_generation`, and `previous_surface_inputs.brick_generation`, then writes the two confidence resources.

- [ ] **Step 3: Export module**

Add:

```rust
pub mod vpt_nrd_confidence;
```

to `src/render/passes/mod.rs`.

- [ ] **Step 4: Run targeted test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt::shader_source_tests::vpt_nrd_confidence_pass_declares_history_confidence_contract -- --exact
```

Expected: PASS.

### Task 3: Pipeline Wiring And Profiler Scope

**Files:**
- Modify: `src/render/vpt_pipeline.rs`
- Modify: `src/render/gpu_profiler.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Wire creation and graph recording**

Add `vpt_nrd_confidence_pass: Option<VptNrdConfidencePass>` to `VptRuntimePipeline`.
Create it after `VptSurfacePass` is available, using:

```rust
include_bytes!(concat!(env!("OUT_DIR"), "/shaders/vpt_nrd_confidence.spv"))
```

Record the confidence graph only when:

```rust
matches!(inputs.lighting_settings.denoiser_mode, VptDenoiserMode::Relax | VptDenoiserMode::Reblur)
```

Use `final_surface_writes` and `previous_surface_resources` as inputs. Keep `PostprocessGraphInputs.input_radiance` as `atrous_filtered_dep`.

- [ ] **Step 2: Add profiler scope**

Add `GpuProfileScope::VptNrdConfidence` before `VptNrdFrontend`, then update:
- `COUNT`
- `ALL`
- `log_name`
- `csv_column`
- `timestamp_stage`
- `scope_names_and_csv_columns_are_stable`
- `timestamp_stages_match_queue_usage`
- `csv_header_and_rows_match_scope_order`

- [ ] **Step 3: Run targeted test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt::shader_source_tests::vpt_nrd_confidence_pass_declares_history_confidence_contract -- --exact
```

Expected: PASS.

### Task 4: Verification And Commit

**Files:**
- All files above.

- [ ] **Step 1: Format**

Run:

```powershell
cargo fmt
```

- [ ] **Step 2: Run full library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib
```

Expected: all library tests pass.

- [ ] **Step 3: Run clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
```

Expected: no warnings.

- [ ] **Step 4: Run shader strict smoke for the new pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib render::passes::vpt::shader_source_tests::vpt_nrd_confidence_pass_declares_history_confidence_contract -- --exact
```

Expected: PASS and `slangc` compiles `vpt_nrd_confidence.slang`.

- [ ] **Step 5: Check diff whitespace and scope**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; unrelated pre-existing dirty files remain unstaged.

- [ ] **Step 6: Commit relevant files only**

Run:

```powershell
git add docs/superpowers/plans/2026-05-24-vpt-nrd-confidence-phase4.md assets/shaders/passes/vpt_nrd_confidence.slang src/render/passes/vpt_nrd_confidence.rs src/render/passes/mod.rs src/render/vpt_pipeline.rs src/render/gpu_profiler.rs src/render/passes/vpt/shader_source_tests.rs
git commit -m "feat: add VPT NRD confidence pass"
```

Expected: commit contains only Phase 4 confidence work.
