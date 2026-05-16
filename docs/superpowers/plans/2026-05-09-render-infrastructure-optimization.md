# Render Infrastructure Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace brittle render-infrastructure contracts with stable tests, shader interface validation, pass-owned graph registration, and a smaller app orchestration boundary.

**Architecture:** Stabilize the current red test suite first, then replace source-snippet assertions with normalized source checks and shader binding manifests. After the gate is green, move descriptor updates and RenderGraph resource declarations into pass-owned registration APIs, then extract VPT runtime ownership out of `src/app.rs` without changing rendered output.

**Tech Stack:** Rust 2024, Vulkan through `ash`, existing `RenderGraph`, Slang shader sources, Cargo unit tests, strict shader compilation through `REVOLUMETRIC_SHADER_COMPILE=strict`.

---

## Review Inputs

This plan addresses these concrete findings:

- `src/render/passes/*` source tests fail on Windows line endings and exact indentation.
- `src/app.rs` manually duplicates RenderGraph resources, descriptor updates, and runtime pass ordering.
- `src/assets/shader_reflect.rs` is a stub and does not validate shader descriptor ABI.
- `src/app.rs` owns window/input, pass lifecycle, render graph assembly, history state, capture, and teardown.

Current baseline command:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Current observed baseline on 2026-05-09:

```text
277 tests, 272 passed, 5 failed
```

The plan must preserve the active VPT-only runtime. Do not restore RC/VCT runtime paths.

## File Structure

- Create `src/render/source_checks.rs`: shared test-only helpers for normalized source reading and whitespace-insensitive assertions.
- Modify `src/render/mod.rs`: expose `source_checks` only under `#[cfg(test)]`.
- Modify `src/render/passes/area_restir.rs`: use normalized source helpers and keep Area ReSTIR shader contract tests stable across CRLF/LF.
- Modify `src/render/passes/restir_di.rs`: use normalized source helpers for app/pass source contract tests.
- Modify `src/render/passes/vpt.rs`: use normalized source helpers and shader binding manifest checks instead of indentation-sensitive descriptor snippets.
- Modify `src/assets/shader_reflect.rs`: replace stub with a Slang source binding manifest parser.
- Modify `src/assets/mod.rs`: keep `shader_reflect` exported for tests and build-time helpers.
- Modify `src/render/descriptor.rs`: add descriptor binding spec utilities that can be compared with shader reflection.
- Modify `src/render/passes/vpt_surface.rs`, `src/render/passes/vpt.rs`, `src/render/passes/restir_di.rs`, `src/render/passes/area_restir.rs`: expose expected descriptor binding specs and pass-owned graph registration entry points.
- Create `src/render/vpt_pipeline.rs`: own VPT-family pass lifecycle, resize, frame graph registration, and history state.
- Modify `src/render/mod.rs`: export `vpt_pipeline`.
- Modify `src/app.rs`: reduce to window/input/world orchestration and delegate render pipeline work to `VptRuntimePipeline`.
- Modify `README.md`: document the infrastructure verification ladder and the new shader-interface validation policy.

## Task 1: Stabilize Existing Red Source Tests

**Files:**
- Create: `src/render/source_checks.rs`
- Modify: `src/render/mod.rs`
- Modify: `src/render/passes/area_restir.rs`
- Modify: `src/render/passes/restir_di.rs`
- Modify: `src/render/passes/vpt.rs`

- [ ] **Step 1: Add normalized source helper tests**

Create `src/render/source_checks.rs` with test helpers and tests:

```rust
#[cfg(test)]
pub(crate) fn read_source(path: &str) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("failed to read source {path}: {error}"))
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

#[cfg(test)]
pub(crate) fn compact(source: &str) -> String {
    source.split_whitespace().collect::<String>()
}

#[cfg(test)]
pub(crate) fn assert_contains_all(source: &str, tokens: &[&str], context: &str) {
    for token in tokens {
        assert!(
            source.contains(token),
            "{context} missing token {token}"
        );
    }
}

#[cfg(test)]
pub(crate) fn assert_compact_contains_all(source: &str, tokens: &[&str], context: &str) {
    let compact_source = compact(source);
    for token in tokens {
        assert!(
            compact_source.contains(token),
            "{context} missing compact token {token}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_ignores_line_endings_and_spacing() {
        let source = "builder\r\n    .add_binding(\r\n        6,\r\n    )";
        assert_eq!(compact(source), "builder.add_binding(6,)");
    }

    #[test]
    fn read_source_normalizes_crlf_to_lf() {
        let normalized = "a\r\nb\rc".replace("\r\n", "\n").replace('\r', "\n");
        assert_eq!(normalized, "a\nb\nc");
    }
}
```

- [ ] **Step 2: Export helper under test configuration**

Modify `src/render/mod.rs`:

```rust
pub mod allocator;
pub mod area_restir;
pub mod buffer;
pub mod camera;
pub mod capture;
pub mod descriptor;
pub mod device;
pub mod frame;
pub mod gpu_profiler;
pub mod graph;
pub mod image;
pub mod pass_context;
pub mod passes;
pub mod pipeline;
pub mod resource;
pub mod restir_di;
pub mod sampler;
pub mod scene_ubo;
pub mod shader;
pub mod swapchain;
pub mod vpt_history;

#[cfg(test)]
pub(crate) mod source_checks;
```

Keep existing module entries that are already present in the file; add only the `source_checks` line if the rest differs.

- [ ] **Step 3: Replace local source readers**

In `src/render/passes/area_restir.rs`, replace the local source helper:

```rust
fn source(path: &str) -> String {
    crate::render::source_checks::read_source(path)
}
```

In `src/render/passes/restir_di.rs`, replace the local source helper:

```rust
fn source(path: &str) -> String {
    crate::render::source_checks::read_source(path)
}
```

In `src/render/passes/vpt.rs`, replace direct `std::fs::read_to_string(...)` usages inside `shader_source_tests` with `crate::render::source_checks::read_source(...)` when the assertion is about source contents.

- [ ] **Step 4: Replace indentation-sensitive descriptor checks**

In `src/render/passes/vpt.rs`, replace exact snippets like:

```rust
assert!(implementation.contains(".add_binding(\n                6,"));
```

with compact checks:

```rust
let compact = crate::render::source_checks::compact(implementation);
for token in [".add_binding(6,", ".add_binding(7,"] {
    assert!(
        compact.contains(token),
        "VPT pass missing descriptor binding token {token}"
    );
}
```

Apply the same pattern to VPT Area ReSTIR bindings `9` and `10`, and VPT surface bindings `10` and `11`.

- [ ] **Step 5: Preserve semantic shader checks**

In `src/render/passes/area_restir.rs`, keep the Area ReSTIR candidate-ray checks, but source-normalize first. The check should pass on both CRLF and LF:

```rust
let initial = source("assets/shaders/passes/area_restir_initial.slang");
crate::render::source_checks::assert_contains_all(
    &initial,
    &[
        "float4 center_position_depth = surface_position_depth[pixel];",
        "AreaRestirCandidateSurface center_surface = read_center_surface(pixel);",
        "AreaRestirCandidateSurface candidate_surface = evaluate_area_restir_candidate_surface(",
        "uint2 pixel,\n    AreaRestirSampleState sample_state",
        "evaluate_area_restir_candidate_surface(scene_ubo, pixel, sample_state)",
        "ScenePrimaryRay primary_ray = scene_primary_ray_from_area_sample(",
        "HitResult hit = trace_primary_ray(",
        "make_ray(primary_ray.origin, primary_ray.direction)",
        "float target_pdf = area_restir_candidate_target_pdf(center_surface, candidate_surface);",
        "float2 pixel_sample = area_restir_pixel_sample(pixel, sample_state);",
        "surface.position_depth = float4(hit.position, hit.t);",
    ],
    "Area ReSTIR initial shader",
);
```

- [ ] **Step 6: Verify targeted red tests are green**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib render::passes::area_restir::shader_source_tests::area_restir_shaders_cache_per_pixel_surface_reads render::passes::restir_di::shader_source_tests::app_restir_di_dispatches_only_enabled_reuse_stages render::passes::vpt::shader_source_tests::vpt_pass_binds_restir_di_uniform_and_reservoir_resources render::passes::vpt::shader_source_tests::vpt_pass_binds_area_restir_as_independent_sample_area_resources render::passes::vpt::shader_source_tests::vpt_surface_pass_binds_area_restir_selected_primary_sample; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all five tests pass. If Cargo treats multiple filters as one malformed filter, run each test filter separately.

- [ ] **Step 7: Verify full library test gate**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all library tests pass.

- [ ] **Step 8: Commit**

```powershell
git -c safe.directory=E:/.Codes/Revolumetric add src/render/source_checks.rs src/render/mod.rs src/render/passes/area_restir.rs src/render/passes/restir_di.rs src/render/passes/vpt.rs
git -c safe.directory=E:/.Codes/Revolumetric commit -m "test: stabilize render source contract tests"
```

## Task 2: Replace Shader Reflection Stub With Binding Manifest Parsing

**Files:**
- Modify: `src/assets/shader_reflect.rs`
- Modify: `src/render/passes/vpt.rs`
- Modify: `src/render/passes/vpt_surface.rs`
- Modify: `src/render/passes/restir_di.rs`
- Modify: `src/render/passes/area_restir.rs`

- [ ] **Step 1: Write failing reflection parser tests**

Replace `src/assets/shader_reflect.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_slang_descriptor_bindings() {
        let source = r#"
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
RWTexture2D<float4> output_image;

[[vk::binding(2, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;
"#;

        let reflection = ShaderReflection::from_slang_source("main", source).unwrap();

        assert_eq!(reflection.entry_point, "main");
        assert_eq!(
            reflection.bindings,
            vec![
                DescriptorBinding {
                    set: 0,
                    binding: 0,
                    kind: DescriptorKind::UniformBuffer,
                    name: "scene_ubo".to_string(),
                },
                DescriptorBinding {
                    set: 0,
                    binding: 1,
                    kind: DescriptorKind::StorageImage,
                    name: "output_image".to_string(),
                },
                DescriptorBinding {
                    set: 0,
                    binding: 2,
                    kind: DescriptorKind::StorageBuffer,
                    name: "hierarchy_l0".to_string(),
                },
            ]
        );
    }

    #[test]
    fn rejects_unknown_descriptor_declaration() {
        let source = r#"
[[vk::binding(3, 0)]]
SamplerState sampler_state;
"#;

        let error = ShaderReflection::from_slang_source("main", source).unwrap_err();
        assert!(error.to_string().contains("unsupported descriptor declaration"));
    }
}
```

Expected: tests fail because `DescriptorBinding`, `DescriptorKind`, and `from_slang_source` do not exist.

- [ ] **Step 2: Implement binding manifest structs**

In `src/assets/shader_reflect.rs`, define:

```rust
use anyhow::{Result, anyhow};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderReflection {
    pub entry_point: String,
    pub bindings: Vec<DescriptorBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorBinding {
    pub set: u32,
    pub binding: u32,
    pub kind: DescriptorKind,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    UniformBuffer,
    StorageBuffer,
    StorageImage,
}
```

- [ ] **Step 3: Implement Slang binding parser**

Add parser functions:

```rust
impl ShaderReflection {
    pub fn from_slang_source(entry_point: impl Into<String>, source: &str) -> Result<Self> {
        let mut bindings = Vec::new();
        let mut pending_binding = None;

        for (line_index, raw_line) in source.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }

            if let Some((binding, set)) = parse_vk_binding(line) {
                pending_binding = Some((binding, set, line_index + 1));
                continue;
            }

            if let Some((binding, set, binding_line)) = pending_binding.take() {
                let (kind, name) = parse_descriptor_declaration(line).ok_or_else(|| {
                    anyhow!(
                        "unsupported descriptor declaration after vk::binding at line {}: {}",
                        binding_line,
                        line
                    )
                })?;
                bindings.push(DescriptorBinding {
                    set,
                    binding,
                    kind,
                    name,
                });
            }
        }

        if let Some((_binding, _set, binding_line)) = pending_binding {
            return Err(anyhow!(
                "vk::binding at line {} has no following descriptor declaration",
                binding_line
            ));
        }

        bindings.sort_by_key(|binding| (binding.set, binding.binding));
        Ok(Self {
            entry_point: entry_point.into(),
            bindings,
        })
    }
}

fn parse_vk_binding(line: &str) -> Option<(u32, u32)> {
    let prefix = "[[vk::binding(";
    let start = line.find(prefix)? + prefix.len();
    let end = line[start..].find(")]]")? + start;
    let mut parts = line[start..end].split(',').map(str::trim);
    let binding = parts.next()?.parse().ok()?;
    let set = parts.next()?.parse().ok()?;
    Some((binding, set))
}

fn parse_descriptor_declaration(line: &str) -> Option<(DescriptorKind, String)> {
    let kind = if line.starts_with("ConstantBuffer<") {
        DescriptorKind::UniformBuffer
    } else if line.starts_with("StructuredBuffer<") || line.starts_with("RWStructuredBuffer<") {
        DescriptorKind::StorageBuffer
    } else if line.starts_with("RWTexture2D<") || line.starts_with("RWTexture3D<") {
        DescriptorKind::StorageImage
    } else {
        return None;
    };
    let name = line
        .trim_end_matches(';')
        .split_whitespace()
        .last()?
        .trim()
        .to_string();
    Some((kind, name))
}
```

- [ ] **Step 4: Add shader-specific binding tests**

In `src/render/passes/vpt.rs`, add a test that validates the active VPT shader interface:

```rust
#[test]
fn vpt_shader_binding_manifest_matches_expected_resources() {
    use crate::assets::shader_reflect::{DescriptorBinding, DescriptorKind, ShaderReflection};

    let source = crate::render::source_checks::read_source("assets/shaders/passes/vpt.slang");
    let reflection = ShaderReflection::from_slang_source("main", &source).unwrap();

    for expected in [
        DescriptorBinding { set: 0, binding: 0, kind: DescriptorKind::UniformBuffer, name: "scene_ubo".to_string() },
        DescriptorBinding { set: 0, binding: 1, kind: DescriptorKind::StorageImage, name: "noisy_radiance_image".to_string() },
        DescriptorBinding { set: 0, binding: 6, kind: DescriptorKind::UniformBuffer, name: "restir_di".to_string() },
        DescriptorBinding { set: 0, binding: 7, kind: DescriptorKind::StorageBuffer, name: "restir_di_reservoirs".to_string() },
        DescriptorBinding { set: 0, binding: 9, kind: DescriptorKind::UniformBuffer, name: "area_restir".to_string() },
        DescriptorBinding { set: 0, binding: 10, kind: DescriptorKind::StorageBuffer, name: "area_restir_reservoirs".to_string() },
    ] {
        assert!(
            reflection.bindings.contains(&expected),
            "missing VPT shader binding {expected:?}"
        );
    }
}
```

Add equivalent focused tests for:

- `assets/shaders/passes/vpt_surface.slang` bindings `0`, `9`, `10`, `11`.
- `assets/shaders/passes/restir_di_initial.slang` bindings `0..=5`.
- `assets/shaders/passes/area_restir_initial.slang` bindings `0..=10`.

- [ ] **Step 5: Remove direct descriptor snippet assertions**

Delete tests that only assert formatting-sensitive Rust snippets such as:

```rust
implementation.contains(".add_binding(\n                6,")
```

Keep semantic tests that assert runtime graph order, history behavior, and pass-local barrier absence.

- [ ] **Step 6: Verify shader reflection tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib shader_reflect vpt_shader_binding_manifest vpt_surface_pass_binds_area_restir_selected_primary_sample; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: parser tests and shader binding manifest tests pass.

- [ ] **Step 7: Commit**

```powershell
git -c safe.directory=E:/.Codes/Revolumetric add src/assets/shader_reflect.rs src/render/passes/vpt.rs src/render/passes/vpt_surface.rs src/render/passes/restir_di.rs src/render/passes/area_restir.rs
git -c safe.directory=E:/.Codes/Revolumetric commit -m "test: validate shader descriptor manifests"
```

## Task 3: Add Descriptor Binding Specs To Pass Wrappers

**Files:**
- Modify: `src/render/descriptor.rs`
- Modify: `src/render/passes/vpt.rs`
- Modify: `src/render/passes/vpt_surface.rs`
- Modify: `src/render/passes/restir_di.rs`
- Modify: `src/render/passes/area_restir.rs`

- [ ] **Step 1: Add descriptor spec type and conversion tests**

In `src/render/descriptor.rs`, add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBindingSpec {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub stage_flags: vk::ShaderStageFlags,
    pub count: u32,
}

impl DescriptorBindingSpec {
    pub const fn compute(
        binding: u32,
        descriptor_type: vk::DescriptorType,
    ) -> Self {
        Self {
            binding,
            descriptor_type,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            count: 1,
        }
    }
}
```

Add tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_descriptor_spec_uses_compute_stage_and_count_one() {
        let spec = DescriptorBindingSpec::compute(6, vk::DescriptorType::UNIFORM_BUFFER);
        assert_eq!(spec.binding, 6);
        assert_eq!(spec.descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
        assert_eq!(spec.stage_flags, vk::ShaderStageFlags::COMPUTE);
        assert_eq!(spec.count, 1);
    }
}
```

- [ ] **Step 2: Teach DescriptorLayoutBuilder to consume specs**

Add:

```rust
impl DescriptorLayoutBuilder {
    pub fn add_binding_spec(self, spec: DescriptorBindingSpec) -> Self {
        self.add_binding(
            spec.binding,
            spec.descriptor_type,
            spec.stage_flags,
            spec.count,
        )
    }

    pub fn add_binding_specs(mut self, specs: &[DescriptorBindingSpec]) -> Self {
        for spec in specs {
            self = self.add_binding_spec(*spec);
        }
        self
    }
}
```

- [ ] **Step 3: Add VPT descriptor specs**

In `src/render/passes/vpt.rs`, add:

```rust
impl VptPass {
    pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 11] {
        use vk::DescriptorType::{STORAGE_BUFFER, STORAGE_IMAGE, UNIFORM_BUFFER};
        [
            DescriptorBindingSpec::compute(0, UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(1, STORAGE_IMAGE),
            DescriptorBindingSpec::compute(2, STORAGE_BUFFER),
            DescriptorBindingSpec::compute(3, STORAGE_BUFFER),
            DescriptorBindingSpec::compute(4, STORAGE_BUFFER),
            DescriptorBindingSpec::compute(5, STORAGE_BUFFER),
            DescriptorBindingSpec::compute(6, UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(7, STORAGE_BUFFER),
            DescriptorBindingSpec::compute(8, STORAGE_IMAGE),
            DescriptorBindingSpec::compute(9, UNIFORM_BUFFER),
            DescriptorBindingSpec::compute(10, STORAGE_BUFFER),
        ]
    }
}
```

Change descriptor layout creation to:

```rust
let descriptor_set_layout = DescriptorLayoutBuilder::new()
    .add_binding_specs(&Self::descriptor_binding_specs())
    .build(device)?;
```

- [ ] **Step 4: Add specs for VPT surface, ReSTIR-DI, and Area ReSTIR**

Use the same pattern:

- `VptSurfacePass::descriptor_binding_specs() -> [DescriptorBindingSpec; 12]`.
- `RestirDiPass::initial_descriptor_binding_specs() -> [DescriptorBindingSpec; 6]`.
- `RestirDiPass::temporal_descriptor_binding_specs() -> [DescriptorBindingSpec; 11]`.
- `RestirDiPass::spatial_descriptor_binding_specs() -> [DescriptorBindingSpec; 6]`.
- `AreaRestirPass::initial_descriptor_binding_specs() -> [DescriptorBindingSpec; 11]`.
- `AreaRestirPass::temporal_descriptor_binding_specs() -> [DescriptorBindingSpec; 11]`.
- `AreaRestirPass::spatial_descriptor_binding_specs() -> [DescriptorBindingSpec; 7]`.

Keep existing binding numbers exactly as current shaders declare them.

- [ ] **Step 5: Compare Rust descriptor specs to shader manifests**

Add test helper in `src/render/descriptor.rs`:

```rust
#[cfg(test)]
pub(crate) fn assert_specs_match_shader_bindings(
    pass_name: &str,
    specs: &[DescriptorBindingSpec],
    reflection: &crate::assets::shader_reflect::ShaderReflection,
) {
    use crate::assets::shader_reflect::DescriptorKind;

    for spec in specs {
        let expected_kind = match spec.descriptor_type {
            vk::DescriptorType::UNIFORM_BUFFER => DescriptorKind::UniformBuffer,
            vk::DescriptorType::STORAGE_BUFFER => DescriptorKind::StorageBuffer,
            vk::DescriptorType::STORAGE_IMAGE => DescriptorKind::StorageImage,
            other => panic!("{pass_name} uses unsupported descriptor type {other:?}"),
        };
        assert!(
            reflection
                .bindings
                .iter()
                .any(|binding| binding.set == 0 && binding.binding == spec.binding && binding.kind == expected_kind),
            "{pass_name} descriptor binding {} missing from shader reflection",
            spec.binding
        );
    }
}
```

Use this helper in VPT, VPT surface, ReSTIR-DI, and Area ReSTIR pass tests.

- [ ] **Step 6: Verify descriptor spec tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib descriptor_binding_specs shader_binding_manifest; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: descriptor specs match shader sources.

- [ ] **Step 7: Commit**

```powershell
git -c safe.directory=E:/.Codes/Revolumetric add src/render/descriptor.rs src/render/passes/vpt.rs src/render/passes/vpt_surface.rs src/render/passes/restir_di.rs src/render/passes/area_restir.rs
git -c safe.directory=E:/.Codes/Revolumetric commit -m "refactor: centralize pass descriptor binding specs"
```

## Task 4: Introduce Pass-Owned Graph Registration For ReSTIR-DI And Area ReSTIR

**Files:**
- Modify: `src/render/passes/restir_di.rs`
- Modify: `src/render/passes/area_restir.rs`
- Modify: `src/render/passes/vpt_surface.rs`
- Modify: `src/render/passes/vpt.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Add graph output structs**

In `src/render/passes/restir_di.rs`, add:

```rust
pub struct RestirDiGraphBuffers<'a> {
    pub uniform_buffer: &'a GpuBuffer,
    pub selected_current_buffer: &'a GpuBuffer,
    pub selected_current_resource: ResourceHandle,
}
```

In `src/render/passes/area_restir.rs`, add:

```rust
pub struct AreaRestirGraphBuffers<'a> {
    pub uniform_buffer: &'a GpuBuffer,
    pub selected_current_buffer: &'a GpuBuffer,
    pub selected_current_resource: ResourceHandle,
}
```

Import `ResourceHandle` from `crate::render::resource`.

- [ ] **Step 2: Add failing app-source contract tests**

In `src/render/passes/restir_di.rs` tests, add:

```rust
#[test]
fn restir_di_pass_owns_graph_registration_contract() {
    let pass = source("src/render/passes/restir_di.rs");
    assert!(pass.contains("pub fn register_graph"));
    assert!(pass.contains("builder.read_as(final_surface_writes[0], AccessKind::ComputeShaderRead)"));
    assert!(pass.contains("vpt.update_restir_di_descriptors"));
}
```

In `src/render/passes/area_restir.rs` tests, add:

```rust
#[test]
fn area_restir_pass_owns_graph_registration_contract() {
    let pass = source("src/render/passes/area_restir.rs");
    assert!(pass.contains("pub fn register_graph"));
    assert!(pass.contains("vpt_surface.update_area_restir_descriptors"));
    assert!(pass.contains("vpt.update_area_restir_descriptors"));
}
```

Expected: fail until registration functions exist.

- [ ] **Step 3: Move ReSTIR-DI graph registration into pass wrapper**

Add method to `RestirDiPass`:

```rust
#[allow(clippy::too_many_arguments)]
pub fn register_graph<'a>(
    &'a self,
    graph: &mut RenderGraph<'a>,
    device: &ash::Device,
    vpt: &'a VptPass,
    frame_slot: usize,
    frame_index: u64,
    settings: RestirDiSettings,
    history_initialized: bool,
    final_surface_writes: [ResourceHandle; 4],
    previous_surface_resources: [ResourceHandle; 3],
    profiler: Option<&'a GpuProfiler>,
) -> RestirDiGraphBuffers<'a> {
    let effective = restir_di_effective_settings(settings, history_initialized);
    self.update_uniforms(frame_slot, effective, frame_index);

    let (uniform_buffer, uniform_size, uniform_usage) = self.uniform_buffer(frame_slot);
    let (direct_light_buffer, direct_light_size, direct_light_usage) = self.direct_light_buffer();
    let (initial_buffer, initial_size, initial_usage) = self.initial_buffer();
    let (temporal_buffer, temporal_size, temporal_usage) = self.temporal_buffer();
    let (selected_current_buffer, selected_current_size, selected_current_usage) =
        self.selected_current_buffer(frame_slot);
    let (selected_history_buffer, selected_history_size, selected_history_usage) =
        self.selected_history_buffer(frame_slot);

    let temporal_active = effective.temporal_enabled;
    let spatial_active = temporal_active && effective.spatial_enabled;
    self.update_frame_descriptors(
        device,
        frame_slot,
        selected_history_buffer,
        selected_current_buffer,
        temporal_active,
        spatial_active,
    );

    // Port the current src/app.rs ReSTIR-DI sequence into this method:
    // 1. Import uniform, direct-light, initial, temporal, selected-current,
    //    and selected-history buffers with the same AccessKind values.
    // 2. Add restir_di_initial. It reads final_surface_writes[0..=2],
    //    uniform, and direct lights; it writes initial_resource when
    //    temporal_active, otherwise selected_current_resource.
    // 3. Add restir_di_temporal only when temporal_active. It reads
    //    final_surface_writes[0..=3], previous_surface_resources[0..=2],
    //    initial_dep, selected_history_resource, and uniform.
    // 4. Add restir_di_spatial only when spatial_active. It reads
    //    final_surface_writes[0..=2], temporal_dep, and uniform.
    // 5. Preserve GpuProfileScope::RestirDiInitial,
    //    GpuProfileScope::RestirDiTemporal, and GpuProfileScope::RestirDiSpatial.

    vpt.update_restir_di_descriptors(device, frame_slot, uniform_buffer, selected_current_buffer);

    RestirDiGraphBuffers {
        uniform_buffer,
        selected_current_buffer,
        selected_current_resource: selected_current_dep,
    }
}
```

When implementing, use the exact current app logic from `src/app.rs` lines around `restir_di_initial`, `restir_di_temporal`, and `restir_di_spatial`. Do not reorder these passes.

- [ ] **Step 4: Move Area ReSTIR graph registration into pass wrapper**

Add method to `AreaRestirPass`:

```rust
#[allow(clippy::too_many_arguments)]
pub fn register_graph<'a>(
    &'a self,
    graph: &mut RenderGraph<'a>,
    device: &ash::Device,
    vpt: &'a VptPass,
    vpt_surface: &'a VptSurfacePass,
    frame_slot: usize,
    frame_index: u64,
    settings: AreaRestirSettings,
    history_initialized: bool,
    bootstrap_surface_writes: [ResourceHandle; 4],
    previous_surface_resources: [ResourceHandle; 3],
    profiler: Option<&'a GpuProfiler>,
) -> AreaRestirGraphBuffers<'a> {
    let effective = area_restir_effective_settings(settings, history_initialized);
    self.update_uniforms(frame_slot, effective, frame_index);

    // Port the current src/app.rs Area ReSTIR sequence into this method:
    // 1. Import area uniform, initial, temporal, selected-current,
    //    selected-history, and debug resources with the same AccessKind values.
    // 2. Add area_restir_initial. It reads bootstrap_surface_writes[0..=2]
    //    and writes initial_resource when temporal is active, otherwise
    //    selected-current.
    // 3. Add area_restir_temporal only when temporal is active. It reads
    //    bootstrap_surface_writes[0..=3], previous_surface_resources[0..=2],
    //    selected history, initial, and uniform.
    // 4. Add area_restir_spatial only when spatial is active. It reads
    //    bootstrap_surface_writes[0..=2], temporal, and uniform.
    // 5. Add vpt_surface_selected after the final Area ReSTIR reservoir
    //    resource is written, and return that selected reservoir resource.

    vpt.update_area_restir_descriptors(device, frame_slot, area_uniform_buffer, area_selected_current_buffer);
    vpt_surface.update_area_restir_descriptors(device, frame_slot, area_uniform_buffer, area_selected_current_buffer);

    AreaRestirGraphBuffers {
        uniform_buffer: area_uniform_buffer,
        selected_current_buffer: area_selected_current_buffer,
        selected_current_resource: area_selected_reservoir_resource,
    }
}
```

When implementing, preserve the current selected surface rule: ReSTIR-DI, VPT, VPT temporal, and VPT atrous consume `vpt_surface_selected` when Area ReSTIR is enabled.

- [ ] **Step 5: Shrink app graph block**

In `src/app.rs`, replace inline ReSTIR-DI and Area ReSTIR graph assembly with calls:

```rust
let area_graph = if area_restir_enabled {
    self.area_restir_pass.as_ref().map(|area_restir| {
        area_restir.register_graph(
            &mut graph,
            renderer.device(),
            vpt,
            vpt_surface,
            frame.frame_slot,
            frame.frame_index,
            self.area_restir_settings,
            self.area_restir_history_initialized,
            bootstrap_surface_writes,
            previous_surface_resources,
            profiler,
        )
    })
} else {
    None
};

let restir_graph = if restir_di_enabled {
    self.restir_di_pass.as_ref().map(|restir_di| {
        restir_di.register_graph(
            &mut graph,
            renderer.device(),
            vpt,
            frame.frame_slot,
            frame.frame_index,
            self.restir_di_settings,
            self.restir_di_history_initialized,
            final_surface_writes,
            previous_surface_resources,
            profiler,
        )
    })
} else {
    None
};
```

Keep `area_restir_selected_written` and `restir_di_selected_written` updates in `app.rs` until Task 5 moves history state into the pipeline.

- [ ] **Step 6: Verify targeted graph tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib app_uses_area_restir_selected_frame_ring app_restir_di_dispatches_only_enabled_reuse_stages app_uses_selected_vpt_surface_after_area_restir_for_di_trace_and_temporal; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: graph ordering and enabled-stage tests pass.

- [ ] **Step 7: Commit**

```powershell
git -c safe.directory=E:/.Codes/Revolumetric add src/render/passes/restir_di.rs src/render/passes/area_restir.rs src/render/passes/vpt_surface.rs src/render/passes/vpt.rs src/app.rs
git -c safe.directory=E:/.Codes/Revolumetric commit -m "refactor: move reuse pass graph registration into pass owners"
```

## Task 5: Extract VPT Runtime Pipeline From App

**Files:**
- Create: `src/render/vpt_pipeline.rs`
- Modify: `src/render/mod.rs`
- Modify: `src/app.rs`

- [ ] **Step 1: Add pipeline state struct tests**

Create `src/render/vpt_pipeline.rs` with tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_state_reset_clears_history_and_accumulation() {
        let mut state = VptPipelineFrameState {
            vpt_sample_index: 7,
            last_vpt_camera_key: Some([1; 15]),
            vpt_accumulation_needs_init: false,
            vpt_temporal_history_initialized: true,
            postprocess_output_initialized: true,
            area_restir_history_initialized: true,
            restir_di_history_initialized: true,
            previous_vpt_view_proj: Some(glam::Mat4::IDENTITY),
            previous_vpt_resolution: Some([1280, 720]),
        };

        state.reset_for_resize_or_camera_cut();

        assert_eq!(state.vpt_sample_index, 0);
        assert_eq!(state.last_vpt_camera_key, None);
        assert!(state.vpt_accumulation_needs_init);
        assert!(!state.vpt_temporal_history_initialized);
        assert!(!state.postprocess_output_initialized);
        assert!(!state.area_restir_history_initialized);
        assert!(!state.restir_di_history_initialized);
        assert_eq!(state.previous_vpt_view_proj, None);
        assert_eq!(state.previous_vpt_resolution, None);
    }
}
```

Expected: fail until `VptPipelineFrameState` exists.

- [ ] **Step 2: Implement frame state**

Add:

```rust
pub struct VptPipelineFrameState {
    pub vpt_sample_index: u32,
    pub last_vpt_camera_key: Option<[u32; 15]>,
    pub vpt_accumulation_needs_init: bool,
    pub vpt_temporal_history_initialized: bool,
    pub postprocess_output_initialized: bool,
    pub area_restir_history_initialized: bool,
    pub restir_di_history_initialized: bool,
    pub previous_vpt_view_proj: Option<glam::Mat4>,
    pub previous_vpt_resolution: Option<[u32; 2]>,
}

impl Default for VptPipelineFrameState {
    fn default() -> Self {
        Self {
            vpt_sample_index: 0,
            last_vpt_camera_key: None,
            vpt_accumulation_needs_init: true,
            vpt_temporal_history_initialized: false,
            postprocess_output_initialized: false,
            area_restir_history_initialized: false,
            restir_di_history_initialized: false,
            previous_vpt_view_proj: None,
            previous_vpt_resolution: None,
        }
    }
}

impl VptPipelineFrameState {
    pub fn reset_for_resize_or_camera_cut(&mut self) {
        self.vpt_sample_index = 0;
        self.last_vpt_camera_key = None;
        self.vpt_accumulation_needs_init = true;
        self.vpt_temporal_history_initialized = false;
        self.postprocess_output_initialized = false;
        self.area_restir_history_initialized = false;
        self.restir_di_history_initialized = false;
        self.previous_vpt_view_proj = None;
        self.previous_vpt_resolution = None;
    }
}
```

- [ ] **Step 3: Move pass fields into VptRuntimePipeline**

Add:

```rust
pub struct VptRuntimePipeline {
    pub postprocess_pass: Option<PostprocessPass>,
    pub vpt_surface_pass: Option<VptSurfacePass>,
    pub vpt_pass: Option<VptPass>,
    pub vpt_temporal_pass: Option<VptTemporalPass>,
    pub vpt_atrous_pass: Option<VptAtrousPass>,
    pub area_restir_pass: Option<AreaRestirPass>,
    pub restir_di_pass: Option<RestirDiPass>,
    pub frame_state: VptPipelineFrameState,
}

impl VptRuntimePipeline {
    pub fn new() -> Self {
        Self {
            postprocess_pass: None,
            vpt_surface_pass: None,
            vpt_pass: None,
            vpt_temporal_pass: None,
            vpt_atrous_pass: None,
            area_restir_pass: None,
            restir_di_pass: None,
            frame_state: VptPipelineFrameState::default(),
        }
    }
}
```

- [ ] **Step 4: Move resize_render_passes into VptRuntimePipeline**

Move `RevolumetricApp::resize_render_passes` to:

```rust
impl VptRuntimePipeline {
    pub fn resize(
        &mut self,
        renderer: &RenderDevice,
        scene_ubo: &SceneUniformBuffer,
        ucvh_gpu: &UcvhGpuResources,
        width: u32,
        height: u32,
        restir_di_enabled: bool,
        area_restir_enabled: bool,
    ) -> Result<()> {
        // Port src/app.rs resize_render_passes into this method exactly in
        // the order listed below this code block. Keep descriptor refreshes
        // immediately after the resources they depend on are resized.
        self.frame_state.reset_for_resize_or_camera_cut();
        Ok(())
    }
}
```

The implementation must preserve current ordering:

1. VPT images.
2. VPT surface images.
3. ReSTIR-DI buffers and surface descriptors.
4. Area ReSTIR buffers, scene descriptors, UCVH descriptors, VPT/VPT surface descriptors.
5. VPT temporal images.
6. VPT atrous images.
7. Postprocess images.

- [ ] **Step 5: Move pass destruction into VptRuntimePipeline**

Add:

```rust
impl VptRuntimePipeline {
    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        if let Some(pass) = self.postprocess_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_atrous_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_temporal_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.area_restir_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_surface_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.vpt_pass {
            pass.destroy(device, allocator);
        }
        if let Some(pass) = self.restir_di_pass {
            pass.destroy(device, allocator);
        }
    }
}
```

- [ ] **Step 6: Replace app fields**

In `src/app.rs`, replace individual pass fields and history flags with:

```rust
vpt_pipeline: VptRuntimePipeline,
```

Keep these fields in `RevolumetricApp`:

- `world`
- `schedule`
- `renderer`
- `gpu_profiler`
- `capture`
- `ucvh`
- `ucvh_gpu`
- `ucvh_uploaded`
- `scene_ubo`
- `lighting_settings`
- `area_restir_settings`
- `restir_di_settings`
- window/input/event-loop fields

- [ ] **Step 7: Verify app no longer owns pass fields**

Add a source test in `src/app.rs` tests:

```rust
#[test]
fn app_delegates_vpt_pass_ownership_to_runtime_pipeline() {
    let source = std::fs::read_to_string("src/app.rs").unwrap();
    let app_struct = source
        .split("struct RevolumetricApp")
        .nth(1)
        .unwrap()
        .split("impl RevolumetricApp")
        .next()
        .unwrap();

    assert!(app_struct.contains("vpt_pipeline: VptRuntimePipeline"));
    assert!(!app_struct.contains("vpt_pass: Option<VptPass>"));
    assert!(!app_struct.contains("vpt_surface_pass: Option<VptSurfacePass>"));
    assert!(!app_struct.contains("postprocess_pass: Option<PostprocessPass>"));
}
```

- [ ] **Step 8: Verify pipeline extraction**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib vpt_pipeline app_delegates_vpt_pass_ownership_to_runtime_pipeline; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: pipeline state tests and app delegation test pass.

- [ ] **Step 9: Commit**

```powershell
git -c safe.directory=E:/.Codes/Revolumetric add src/render/vpt_pipeline.rs src/render/mod.rs src/app.rs
git -c safe.directory=E:/.Codes/Revolumetric commit -m "refactor: extract VPT runtime pipeline from app"
```

## Task 6: Move Frame Graph Assembly Into VptRuntimePipeline

**Files:**
- Modify: `src/render/vpt_pipeline.rs`
- Modify: `src/app.rs`
- Modify: `src/render/passes/vpt.rs`
- Modify: `src/render/passes/vpt_temporal.rs`
- Modify: `src/render/passes/vpt_atrous.rs`
- Modify: `src/render/passes/postprocess.rs`

- [ ] **Step 1: Add frame input and output structs**

In `src/render/vpt_pipeline.rs`, add:

```rust
pub struct VptPipelineFrameInput<'a> {
    pub renderer: &'a RenderDevice,
    pub frame: &'a FrameContext,
    pub scene_ubo: &'a SceneUniformBuffer,
    pub ucvh_gpu: &'a UcvhGpuResources,
    pub lighting_settings: LightingSettings,
    pub restir_di_settings: RestirDiSettings,
    pub area_restir_settings: AreaRestirSettings,
    pub profiler: Option<&'a GpuProfiler>,
    pub capture: Option<&'a mut RenderCapture>,
}

pub struct VptPipelineFrameOutput {
    pub submitted_capture: Option<CaptureMetadata>,
    pub rendered_vpt_frame: bool,
    pub restir_di_selected_written: bool,
    pub area_restir_selected_written: bool,
}
```

- [ ] **Step 2: Add pipeline frame recording method**

Add:

```rust
impl VptRuntimePipeline {
    pub fn record_frame<'a>(
        &'a mut self,
        graph: &mut RenderGraph<'a>,
        input: VptPipelineFrameInput<'a>,
    ) -> Result<VptPipelineFrameOutput> {
        // Port the current src/app.rs VPT graph assembly block into this
        // method. The moved block starts at the required-pass availability
        // check and ends after blit_to_swapchain registration. Keep fallback
        // clear/present handling in app until this method is green.
        Ok(VptPipelineFrameOutput {
            submitted_capture: None,
            rendered_vpt_frame: false,
            restir_di_selected_written: false,
            area_restir_selected_written: false,
        })
    }
}
```

Expected initial compile failure until imports and moved logic are completed.

- [ ] **Step 3: Move surface, VPT, temporal, atrous, history, postprocess, capture, and blit graph setup**

Move these graph passes from `src/app.rs` into `VptRuntimePipeline::record_frame`:

- `vpt_surface_bootstrap`
- Area ReSTIR registration call from Task 4
- `vpt_surface_selected`
- ReSTIR-DI registration call from Task 4
- `vpt`
- `vpt_temporal`
- `vpt_atrous`
- `vpt_surface_history_update`
- `postprocess`
- `capture_postprocess`
- `blit_to_swapchain`

Keep `add_swapchain_clear_present_pass` in `src/app.rs` for the fallback path until the pipeline method is verified.

- [ ] **Step 4: Reduce app tick_frame**

In `src/app.rs`, replace the VPT graph block with:

```rust
let output = self.vpt_pipeline.record_frame(
    &mut graph,
    VptPipelineFrameInput {
        renderer,
        frame: &frame,
        scene_ubo,
        ucvh_gpu,
        lighting_settings: self.lighting_settings,
        restir_di_settings: self.restir_di_settings,
        area_restir_settings: self.area_restir_settings,
        profiler: self.gpu_profiler.as_ref(),
        capture: self.capture.as_mut(),
    },
)?;

pending_capture = output.submitted_capture;
```

After `graph.execute(...)`, update pipeline state through output flags:

```rust
self.vpt_pipeline
    .frame_state
    .commit_frame_output(&output, current_vpt_view_proj, frame.swapchain_extent);
```

Add `commit_frame_output` in `VptPipelineFrameState`.

- [ ] **Step 5: Add app source boundary test**

In `src/app.rs` tests:

```rust
#[test]
fn app_no_longer_registers_vpt_family_graph_passes_inline() {
    let source = std::fs::read_to_string("src/app.rs").unwrap();
    let implementation = source.split("#[cfg(test)]").next().unwrap();

    for forbidden in [
        "graph.add_pass(\"vpt_surface_bootstrap\"",
        "graph.add_pass(\"area_restir_initial\"",
        "graph.add_pass(\"restir_di_initial\"",
        "graph.add_pass(\"vpt_temporal\"",
        "graph.add_pass(\"vpt_atrous\"",
        "graph.add_pass(\"postprocess\"",
        "graph.add_pass(\"blit_to_swapchain\"",
    ] {
        assert!(
            !implementation.contains(forbidden),
            "app.rs should delegate VPT graph registration, found {forbidden}"
        );
    }
}
```

- [ ] **Step 6: Verify graph ordering still matches current behavior**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib app_no_longer_registers_vpt_family_graph_passes_inline app_wires_area_restir_between_surface_and_vpt_with_history_and_vpt_reads app_routes_temporal_radiance_through_vpt_atrous_before_postprocess; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: tests pass with graph registration now owned by `VptRuntimePipeline`.

- [ ] **Step 7: Commit**

```powershell
git -c safe.directory=E:/.Codes/Revolumetric add src/render/vpt_pipeline.rs src/app.rs src/render/passes/vpt.rs src/render/passes/vpt_temporal.rs src/render/passes/vpt_atrous.rs src/render/passes/postprocess.rs
git -c safe.directory=E:/.Codes/Revolumetric commit -m "refactor: delegate VPT frame graph assembly to runtime pipeline"
```

## Task 7: Verification Matrix And Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-05-09-render-infrastructure-optimization.md`

- [ ] **Step 1: Document infrastructure gates**

In `README.md`, add this under the validation matrix:

```markdown
Infrastructure verification order:

1. `REVOLUMETRIC_SHADER_COMPILE=skip cargo test --lib`
2. `REVOLUMETRIC_SHADER_COMPILE=strict cargo test --lib`
3. `REVOLUMETRIC_SHADER_COMPILE=skip cargo clippy --all-targets -- -D warnings`
4. `cargo build --bin revolumetric`
5. Runtime smoke with representative VPT feature flags

Descriptor ABI checks are source-manifest based. Rust pass descriptor specs must match Slang `[[vk::binding(...)] ]` declarations before strict shader compilation is considered meaningful.
```

Fix the extra space in the markdown binding token while editing so it reads exactly:

```markdown
`[[vk::binding(...)]]`
```

- [ ] **Step 2: Run formatting**

Run:

```powershell
cargo fmt
```

Expected: no formatting errors.

- [ ] **Step 3: Run skip-mode library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all library tests pass.

- [ ] **Step 4: Run strict shader library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all library tests pass. If `slangc` is missing, stop and report that strict shader validation is blocked by local toolchain.

- [ ] **Step 5: Run clippy**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo clippy --all-targets -- -D warnings; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: no warnings.

- [ ] **Step 6: Build runtime binary**

Run:

```powershell
cargo build --bin revolumetric
```

Expected: build completes.

- [ ] **Step 7: Runtime smoke**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='strict'
$env:REVOLUMETRIC_RENDER_MODE='vpt'
$env:REVOLUMETRIC_VPT_RESTIR_DI='on'
$env:REVOLUMETRIC_RESTIR_DI_SPATIAL='on'
$env:REVOLUMETRIC_AREA_RESTIR='on'
$env:REVOLUMETRIC_EXIT_AFTER_FRAMES='3'
cargo run --bin revolumetric
Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
Remove-Item Env:\REVOLUMETRIC_RENDER_MODE
Remove-Item Env:\REVOLUMETRIC_VPT_RESTIR_DI
Remove-Item Env:\REVOLUMETRIC_RESTIR_DI_SPATIAL
Remove-Item Env:\REVOLUMETRIC_AREA_RESTIR
Remove-Item Env:\REVOLUMETRIC_EXIT_AFTER_FRAMES
```

Expected: app exits after 3 frames without Vulkan validation errors, panics, or missing shader pass warnings.

- [ ] **Step 8: Final commit**

```powershell
git -c safe.directory=E:/.Codes/Revolumetric add README.md docs/superpowers/plans/2026-05-09-render-infrastructure-optimization.md
git -c safe.directory=E:/.Codes/Revolumetric commit -m "docs: document render infrastructure verification"
```

## Execution Notes

- Keep commits narrow. Do not stage `CONOUT$` or unrelated IDE files.
- Do not broaden the renderer feature set while executing this plan.
- If a task exposes a real shader/runtime bug, stop after the smallest reproducer is committed and create a follow-up fix plan.
- If strict shader compilation fails because `slangc` is unavailable, keep skip-mode tests green and report the toolchain blocker explicitly.
- Runtime visual correctness still needs representative capture review after infrastructure is green. Green unit tests are not enough for final renderer-quality claims.
