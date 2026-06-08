# RenderGraph Transient Slot Plan Design

## Goal

Turn RenderGraph transient alias information into a deterministic CPU-side slot plan. A slot represents one future physical transient resource allocation that may be reused by multiple graph-created resources with compatible descriptors and non-overlapping lifetimes.

## Current Facts

- RenderGraph now records resource origin, sorted execution lifetimes, and transient alias candidates.
- Existing passes still own Vulkan allocation, resize, descriptor updates, and destruction.
- The graph can identify compatible non-overlapping transient pairs, but it does not yet produce a final grouping suitable for a future allocator.

## Design

Add `TransientResourceSlot` in `src/render/resource.rs`:

- `slot_index: usize`
- `desc: ResourceDesc`
- `resource_ids: Vec<u32>`

Add `transient_resource_slots: Vec<TransientResourceSlot>` to `RenderGraph`.

During `RenderGraph::compile`, after lifetimes are computed, build slots with a deterministic greedy algorithm:

1. Consider only `ResourceOrigin::Transient` lifetimes with a `ResourceDesc`.
2. Sort candidates by `(first_step, last_step, resource_id)`.
3. Place a resource into the first existing slot whose `desc` matches and whose last resource lifetime ends before this resource starts.
4. If no slot is compatible, create a new slot with the next `slot_index`.

Expose:

```rust
pub fn transient_resource_slots(&self) -> &[TransientResourceSlot]
```

Mutating the graph clears stale slots through the same compile-product invalidation path that clears lifetimes, alias candidates, and barrier plans.

## Non-Goals

- Do not allocate Vulkan memory.
- Do not create `GpuImage` or `GpuBuffer` from the slot plan.
- Do not change pass-owned resources, descriptor writes, or resize behavior.
- Do not implement Vulkan memory compatibility checks beyond exact `ResourceDesc` equality.

## Verification

Add CPU-only tests with fake Vulkan handles:

- non-overlapping compatible transient resources are packed into one slot
- overlapping compatible transient resources remain in separate slots
- different descriptors remain in separate slots
- mutating the graph clears stale slot plans

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_transient_slot; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

## Follow-Up

The next phase can use this slot plan to introduce a real transient arena that owns Vulkan image and buffer allocations for graph-created resources.
