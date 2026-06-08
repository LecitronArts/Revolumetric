# RenderGraph Resource Lifecycle Design

## Goal

Make RenderGraph explicitly model resource origin, execution lifetime, and transient reuse opportunities. This phase creates a verifiable planning layer for future graph-owned allocation without changing existing Vulkan image or buffer ownership in passes.

## Current Facts

- `src/render/graph.rs` records `ResourceDesc` for imported resources and resources declared through `PassBuilder::create_image` / `create_buffer`.
- `RenderGraph::compile` sorts passes, validates accesses, and plans barriers.
- Runtime passes still create, resize, bind, and destroy their own `GpuImage` / `GpuBuffer` instances.
- `RenderGraph::execute` records barriers against bound raw `vk::Image` and `vk::Buffer` handles.
- Existing production pass code mostly imports pass-owned resources into the graph rather than asking the graph to allocate them.

## Problem

The graph can describe resources, but it cannot answer lifecycle questions that an industrial render backend needs:

- Was a resource imported from a pass/runtime owner, or was it graph-created?
- Which sorted execution steps first and last touch a resource?
- Which graph-created resources have non-overlapping lifetimes and compatible descriptors, making them candidates for future memory aliasing?
- Can tests prove the above without requiring a Vulkan device?

Without these answers, moving allocation into RenderGraph would be unsafe. The allocator would need to infer ownership and lifetime from scattered pass code, which is exactly the coupling this architecture work is removing.

## Design

### Resource Origin

Add `ResourceOrigin` in `src/render/resource.rs`:

- `Imported`: resource was provided by a caller through `RenderGraph::import_image` or `import_buffer`.
- `Transient`: resource was declared by a pass through `PassBuilder::create_image` or `create_buffer`.

The graph stores `resource_origins: BTreeMap<u32, ResourceOrigin>` keyed by resource id.

### Resource Lifetime

Add `ResourceLifetime` in `src/render/resource.rs`:

- `resource_id: u32`
- `origin: ResourceOrigin`
- `first_step: usize`
- `last_step: usize`

`first_step` and `last_step` use sorted execution order, not pass insertion index. This makes lifetime data match actual graph execution after dependency sorting.

A resource is considered touched by:

- reads
- writes
- explicit accesses
- final accesses

Dependency-only reads from `PassBuilder::depend_on` count as lifetime touches because they constrain execution order even without barriers.

### Transient Alias Candidates

Add `TransientAliasCandidate` in `src/render/resource.rs`:

- `first_resource_id: u32`
- `second_resource_id: u32`

`RenderGraph::compile` fills `transient_alias_candidates` after lifetimes are computed.

A candidate is emitted only when:

- both resources are `ResourceOrigin::Transient`
- their `ResourceDesc` values are equal
- their lifetimes do not overlap in sorted execution steps

Imported resources are never alias candidates in this phase. Descriptor-compatible but differently sized or differently used resources are also not candidates.

### Public Inspection API

Add read-only APIs on `RenderGraph`:

- `resource_origin(handle: ResourceHandle) -> Option<ResourceOrigin>`
- `resource_lifetimes() -> &[ResourceLifetime]`
- `transient_alias_candidates() -> &[TransientAliasCandidate]`

These APIs are intentionally CPU-only and testable with fake Vulkan handles.

Compile products are invalidated when the graph is mutated through import, pass registration, or resource binding APIs. This prevents stale barrier, lifetime, or alias plans from being exposed after a graph has changed.

## Non-Goals

- Do not allocate `GpuImage` or `GpuBuffer` from RenderGraph in this phase.
- Do not remove pass-owned resize or descriptor update logic.
- Do not introduce memory aliasing at runtime.
- Do not change barrier semantics or queue scheduling.
- Do not relax existing binding validation for executable graph resources.

## Failure And Edge Cases

- Recompiling after graph mutation must clear stale lifetime and alias plans.
- Mutating a compiled graph must clear old compile products before the next compile.
- Imported resources with matching descriptors must not be reported as transient alias candidates.
- Overlapping transient lifetimes must not be reported as alias candidates.
- Lifetimes must be based on sorted execution order so future graph scheduling changes do not invalidate allocator assumptions.

## Verification

Use unit tests in `src/render/graph.rs` and `src/render/resource.rs` with fake `vk::Image` / `vk::Buffer` handles:

- origins are recorded for imported and transient resources
- compile records sorted execution lifetimes
- non-overlapping compatible transients are alias candidates
- imported and overlapping resources are not alias candidates

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_resource_lifecycle; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

## Follow-Up Phases

This phase enables, but does not implement:

- graph-owned transient image and buffer allocation
- descriptor binding through graph-resolved resources
- resource aliasing with Vulkan memory compatibility checks
- per-frame transient arenas
- pass migration away from manual resize/destroy code
