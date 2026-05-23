# VPT NRD ReLAX Frontend Pack Phase 3b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local VPT NRD frontend pass that converts raw VPT NRD noisy outputs into ReLAX-ready packed graph resources without changing the existing SVGF product path.

**Architecture:** Keep VPT as the raw noisy producer, add `vpt_nrd_frontend` as a compute pass that reads named `VptNrdNoisyResources` and writes named packed resources. The pass is prepared and recorded for NRD-requested modes, but postprocess continues to consume the existing `vpt_temporal -> vpt_atrous` fallback until the official NRD backend and resolve pass are validated.

**Tech Stack:** Rust/ash render passes, RenderGraph image resources, Slang compute shader, existing shader source reflection tests.

---

### Task 1: Source Tests For Frontend Contract

**Files:**
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Write the failing test**

Add a test named `vpt_nrd_frontend_pass_declares_relax_packing_contract` that checks:
- `assets/shaders/passes/vpt_nrd_frontend.slang` exists and exposes bindings 0-8.
- `src/render/passes/vpt_nrd_frontend.rs` exists.
- Rust pass exposes `VptNrdFrontendGraphInputs`, `VptNrdFrontendGraphOutputs`, `VptNrdPackedResources`, and `VptNrdFrontendPass::descriptor_binding_specs`.
- The pass uses named `VptNrdNoisyResources` rather than `[ResourceHandle; N]`.
- The shader has local `pack_relax_*` helpers, uses invalid fp16 hit distance `65504.0`, sanitizes non-finite/negative radiance, and does not include or copy NVIDIA SDK helper files.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt::shader_source_tests::vpt_nrd_frontend_pass_declares_relax_packing_contract -- --exact
```

Expected: FAIL because the shader/pass do not exist yet.

### Task 2: Implement Shader And Rust Pass

**Files:**
- Create: `assets/shaders/passes/vpt_nrd_frontend.slang`
- Create: `src/render/passes/vpt_nrd_frontend.rs`
- Modify: `src/render/passes/mod.rs`

- [ ] **Step 1: Add shader**

Create a Slang compute shader with:
- `scene_ubo` at binding 0.
- raw inputs at bindings 1-4:
  `input_diff_radiance_hitdist`, `input_spec_radiance_hitdist`, `input_residual_radiance`, `input_material_factors`.
- packed outputs at bindings 5-8:
  `packed_diff_radiance_hitdist`, `packed_spec_radiance_hitdist`, `residual_radiance`, `material_factors`.
- `pack_relax_radiance_hitdist` that clamps negative/non-finite radiance to zero and clamps invalid hit distance to `65504.0`.
- Pass-through residual/material output, with alpha normalized to `1.0`.

- [ ] **Step 2: Add Rust pass**

Create `VptNrdFrontendPass` following existing compute pass patterns:
- descriptor specs mirror shader bindings.
- owned output images use `R16G16B16A16_SFLOAT` and `STORAGE | SAMPLED | TRANSFER_SRC`.
- `VptNrdFrontendGraphInputs` carries `frame_slot`, `raw_noisy: VptNrdNoisyResources`, and optional profiler.
- `VptNrdFrontendGraphOutputs` carries `packed: VptNrdPackedResources`.
- `register_graph` reads all raw noisy resources and writes all four packed resources.

- [ ] **Step 3: Export module**

Add `pub mod vpt_nrd_frontend;` to `src/render/passes/mod.rs`.

- [ ] **Step 4: Run targeted green test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt::shader_source_tests::vpt_nrd_frontend_pass_declares_relax_packing_contract -- --exact
```

Expected: PASS.

### Task 3: Pipeline Preparation Without Product Routing Change

**Files:**
- Modify: `src/render/vpt_pipeline.rs`
- Modify: `src/render/gpu_profiler.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Add failing routing test**

Extend the source test to require:
- `pub vpt_nrd_frontend_pass: Option<VptNrdFrontendPass>`.
- `ensure_vpt_nrd_frontend_pass`.
- `include_bytes!(... "vpt_nrd_frontend.spv")`.
- `vpt_nrd_frontend.register_graph(` after `vpt.register_graph(`.
- `PostprocessGraphInputs` still uses `atrous_filtered_dep`.

- [ ] **Step 2: Run test to verify it fails**

Run the same targeted test. Expected: FAIL because pipeline is not wired yet.

- [ ] **Step 3: Wire pass creation and graph recording**

Add the pass to `VptRuntimePipeline`, ensure it after `VptPass`, resize/destroy it, and record `vpt_nrd_frontend.register_graph` only when requested denoiser mode is `Relax` or `Reblur`. Keep postprocess input as `atrous_filtered_dep`.

- [ ] **Step 4: Add profiler scope**

Add `GpuProfileScope::VptNrdFrontend`, update `COUNT`, `ALL`, `log_name`, `csv_column`, and `timestamp_stage`.

- [ ] **Step 5: Run targeted green test**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::vpt::shader_source_tests::vpt_nrd_frontend_pass_declares_relax_packing_contract -- --exact
```

Expected: PASS.

### Task 4: Full Verification And Commit

**Files:**
- All files modified above.

- [ ] **Step 1: Format**

Run:

```powershell
cargo fmt
```

- [ ] **Step 2: Run tests**

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

- [ ] **Step 4: Check diff whitespace**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors except any pre-existing line-ending warnings.

- [ ] **Step 5: Commit only relevant files**

Run:

```powershell
git status --short
git add docs/superpowers/plans/2026-05-24-vpt-nrd-relax-frontend-pack-phase3b.md assets/shaders/passes/vpt_nrd_frontend.slang src/render/passes/vpt_nrd_frontend.rs src/render/passes/mod.rs src/render/vpt_pipeline.rs src/render/gpu_profiler.rs src/render/passes/vpt/shader_source_tests.rs
git commit -m "feat: add VPT NRD frontend packing pass"
```

Expected: unrelated dirty files stay unstaged.
