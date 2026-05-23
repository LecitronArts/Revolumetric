# VPT NRD Noisy Frontend Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NRD-shaped noisy outputs to VPT while preserving the existing SVGF fallback path.

**Architecture:** Keep `noisy_radiance/noisy_moments` intact for the current temporal/a-trous path, and add four new storage images for NRD-ready diffuse/specular/residual/material factors. The VPT trace shader writes both the legacy fallback outputs and the new NRD contract so later phases can switch consumers without rewriting the path tracer again.

**Tech Stack:** Rust render pass/resource graph code, Slang compute shaders, shader source contract tests.

---

### Task 1: Red Tests

**Files:**
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [x] **Step 1: Add noisy frontend contract assertions**

Add a test that requires the trace shader to declare and write:

```rust
"RWTexture2D<float4> nrd_diff_radiance_hitdist"
"RWTexture2D<float4> nrd_spec_radiance_hitdist"
"RWTexture2D<float4> nrd_residual_radiance"
"RWTexture2D<float4> nrd_material_factors"
"float first_indirect_hit_distance"
"sample.first_indirect_hit_distance"
"nrd_diff_radiance_hitdist[tid.xy]"
"nrd_spec_radiance_hitdist[tid.xy]"
"nrd_residual_radiance[tid.xy]"
"nrd_material_factors[tid.xy]"
```

Add a Rust source assertion that requires:

```rust
"pub struct VptNrdNoisyResources"
"pub diff_radiance_hitdist: ResourceHandle"
"pub spec_radiance_hitdist: ResourceHandle"
"pub residual_radiance: ResourceHandle"
"pub material_factors: ResourceHandle"
"pub nrd_diff_radiance_hitdist: GpuImage"
"pub nrd_spec_radiance_hitdist: GpuImage"
"pub nrd_residual_radiance: GpuImage"
"pub nrd_material_factors: GpuImage"
"descriptor_count: 6 * frame_count as u32"
```

- [x] **Step 2: Run tests to verify failure**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_nrd_noisy_frontend_contract -- --exact
```

Expected: FAIL because the new NRD noisy contract does not exist yet.

### Task 2: Noisy Contract

**Files:**
- Modify: `assets/shaders/passes/vpt.slang`
- Modify: `src/render/passes/vpt.rs`

- [x] **Step 1: Add NRD noisy outputs**

Extend `VptTraceSample` with a first-indirect hit-distance field. Add four new storage images and bind/write them alongside the existing noisy outputs:

```slang
RWTexture2D<float4> nrd_diff_radiance_hitdist;
RWTexture2D<float4> nrd_spec_radiance_hitdist;
RWTexture2D<float4> nrd_residual_radiance;
RWTexture2D<float4> nrd_material_factors;
```

Write demodulated indirect diffuse into `nrd_diff_radiance_hitdist.rgb`, use the first post-primary hit distance in `.a`, keep specular zero-filled, put primary-hit direct/emissive/sky/debug in residual, and store primary albedo plus roughness in `nrd_material_factors`.

- [x] **Step 2: Verify focused tests pass**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_nrd_noisy_frontend_contract
```

Expected: PASS.

### Task 3: Regression Sweep

**Files:**
- Modify only files from Tasks 1-2.

- [x] **Step 1: Run full verification**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
git diff --check
```

Expected: all exit 0.

- [ ] **Step 2: Commit relevant files only**

Stage:

```powershell
git add docs/superpowers/plans/2026-05-23-vpt-nrd-noisy-frontend-phase3.md assets/shaders/passes/vpt.slang src/render/passes/vpt.rs src/render/passes/vpt/shader_source_tests.rs
git commit -m "feat: add VPT NRD noisy frontend contract"
```
