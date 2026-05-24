# VPT NRD Adapter Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an official NRD adapter path for `RELAX_DIFFUSE` behind an opt-in `nrd` feature and external SDK root, with Vulkan resource creation, dispatch recording, and validation hooks, while preserving the SVGF fallback path.

**Architecture:** Keep the NVIDIA NRD SDK behind a small native C ABI wrapper and a Cargo `nrd` feature. Rust owns the RenderGraph, Vulkan resource lifetime, resize handling, and pass ordering; the wrapper only exposes the NRD library/instance/dispatch descriptors plus opaque settings upload blocks. Phase 5 stops at the adapter and validation layer: `RELAX_DIFFUSE` can run and be inspected, but resolve/remodulation into the product path remains Phase 6.

**Tech Stack:** Rust, `ash`, RenderGraph, `build.rs`, `cc` or equivalent native build glue, external NRD SDK checkout via `REVOLUMETRIC_NRD_ROOT`, existing VPT guide/noisy/confidence passes.

---

### Task 1: Feature Gate And SDK Discovery

**Files:**
- Modify: `Cargo.toml`
- Modify: `build.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Add the `nrd` feature and native build dependency**

Add a new Cargo feature:

```toml
[features]
desktop = []
nrd = []
default = []
```

Add a build dependency for the native wrapper path:

```toml
[build-dependencies]
cc = "1"
walkdir = "2"
```

- [ ] **Step 2: Fail early when `nrd` is enabled without a valid SDK root**

Extend `build.rs` so it:
- checks `CARGO_FEATURE_NRD`
- requires `REVOLUMETRIC_NRD_ROOT`
- verifies the root contains at least `Include/NRD.h`, `Include/NRDDescs.h`, and `Include/NRDSettings.h`
- emits a clear panic that names `REVOLUMETRIC_NRD_ROOT` and the NVIDIA RTX SDK license requirement if the root is missing or incomplete

Suggested shape:

```rust
fn nrd_root() -> PathBuf {
    let root = env::var("REVOLUMETRIC_NRD_ROOT").unwrap_or_else(|_| {
        panic!(
            "REVOLUMETRIC_NRD_ROOT is required when the nrd feature is enabled; \
             point it at an accepted local NVIDIA RTX SDK checkout before building with --features nrd"
        )
    });
    let root = PathBuf::from(root);
    assert!(root.join("Include/NRD.h").exists());
    assert!(root.join("Include/NRDDescs.h").exists());
    assert!(root.join("Include/NRDSettings.h").exists());
    root
}
```

- [ ] **Step 3: Keep the fallback build unchanged**

Do not change the default non-`nrd` build path. `cargo test --lib` and `cargo run --features desktop` must continue to use SVGF and the existing passes.

- [ ] **Step 4: Validate the gate**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib
```

Expected: pass.

Run:

```powershell
cargo build --features nrd
```

Expected: fail early with the new `REVOLUMETRIC_NRD_ROOT` message when the SDK root is absent.

### Task 2: Native Wrapper And Rust FFI Types

**Files:**
- Create: `native/nrd_adapter.h`
- Create: `native/nrd_adapter.cpp`
- Create: `src/render/nrd_adapter.rs`
- Modify: `src/render/mod.rs`

- [ ] **Step 1: Define a POD-only C ABI surface**

Create C-compatible structs for the data Rust needs to read or upload:
- `NrdLibraryDesc`
- `NrdInstanceDesc`
- `NrdTextureDesc`
- `NrdSamplerDesc`
- `NrdPipelineDesc`
- `NrdDispatchDesc`
- `NrdCommonSettings`
- `NrdRelaxDiffuseSettings`

Keep these as plain `#[repr(C)]` / C structs. Do not expose C++ references, `std::vector`, or NRD-owned pointers across the Rust boundary.

Example layout:

```cpp
extern "C" {
struct NrdLibraryDesc {
    uint32_t textureOffset;
    uint32_t samplerOffset;
    uint32_t constantBufferOffset;
    uint32_t storageTextureAndBufferOffset;
};
}
```

- [ ] **Step 2: Wrap the NRD API in C++**

Implement native functions that:
- create and destroy an NRD instance for `RELAX_DIFFUSE`
- copy the NRD library/instance descriptors into caller-provided buffers
- copy pipeline and dispatch descriptors into caller-provided arrays
- accept common settings and method settings blocks from Rust
- expose the SPIR-V bytecode and resource ranges required to build Vulkan pipelines in Rust

The wrapper should be thin: it may use NRD C++ internally, but it should only export C ABI data and opaque handles.

- [ ] **Step 3: Add Rust bindings and a stub fallback**

In `src/render/nrd_adapter.rs`:
- add `#[cfg(feature = "nrd")]` extern declarations for the native wrapper
- add `#[cfg(not(feature = "nrd"))]` stubs that return a typed `NrdUnavailableError`
- provide mirror enums/structs for the descriptor data Rust needs to inspect
- add `size_of` / `offset_of` tests for the Rust mirror types

- [ ] **Step 4: Export the module**

Add:

```rust
pub mod nrd_adapter;
```

to `src/render/mod.rs`.

### Task 3: Vulkan Adapter Pass And Runtime Integration

**Files:**
- Create: `src/render/passes/vpt_nrd_adapter.rs`
- Modify: `src/render/passes/mod.rs`
- Modify: `src/render/vpt_pipeline.rs`
- Modify: `src/render/gpu_profiler.rs`
- Modify: `src/render/passes/vpt/shader_source_tests.rs`

- [ ] **Step 1: Implement the adapter pass skeleton**

Create `VptNrdAdapterPass` that owns:
- permanent and transient NRD pool textures
- descriptor sets and descriptor set layouts
- compute pipelines created from NRD-provided SPIR-V bytecode
- per-frame constant buffers / settings upload blocks

The pass should consume the named VPT NRD guide/noisy resources that already exist:
- `VptNrdPackedResources`
- `VptNrdConfidenceResources`

Initial target: `RELAX_DIFFUSE` only.

- [ ] **Step 2: Wire the pass into the runtime pipeline**

Extend `VptRuntimePipeline` to create, resize, and destroy the adapter pass when:
- the `nrd` feature is enabled
- `REVOLUMETRIC_NRD_ROOT` resolves successfully
- `REVOLUMETRIC_DENOISER` requests `relax`

Keep the SVGF path untouched. The adapter must not replace the existing postprocess input yet; that remains the Phase 6 resolve step.

Suggested control flow:

```rust
let nrd_outputs = if matches!(inputs.lighting_settings.denoiser_mode, VptDenoiserMode::Relax) {
    self.vpt_nrd_adapter_pass.as_ref().map(|pass| pass.register_graph(...))
} else {
    None
};
```

- [ ] **Step 3: Add a profiler scope and debug routing**

Add a dedicated profiler scope for the NRD adapter dispatches and update:
- `GpuProfileScope`
- CSV header / rows
- log names
- timestamp stage coverage

Route `nrd_validation` and the initial `RELAX_DIFFUSE` debug output through explicit graph resources so captures can inspect them before Phase 6 resolve work lands.

- [ ] **Step 4: Update source-contract tests**

Extend `src/render/passes/vpt/shader_source_tests.rs` to assert:
- the new adapter module exists
- the runtime pipeline wires the adapter behind `RELAX`
- the adapter consumes the packed noisy + confidence resources
- the final SVGF fallback path still feeds postprocess
- no anonymous `[ResourceHandle; N]` arrays are introduced for the adapter contract

### Task 4: Verification And Commit

**Files:**
- All files above.

- [ ] **Step 1: Format**

Run:

```powershell
cargo fmt
```

- [ ] **Step 2: Run fallback library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib
```

Expected: pass with the default non-`nrd` build.

- [ ] **Step 3: Run clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings
```

Expected: no warnings.

- [ ] **Step 4: Check diff hygiene**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; only the files from this phase should be staged later.

- [ ] **Step 5: Validate the feature gate**

Run:

```powershell
cargo build --features nrd
```

Expected: fail early until `REVOLUMETRIC_NRD_ROOT` points at a valid accepted SDK checkout. Once a valid SDK root exists, replace that with a real `cargo build --features 'desktop nrd' --bin revolumetric` smoke.

- [ ] **Step 6: Commit the phase**

Stage only the phase-5 files and commit once the fallback build is clean.
