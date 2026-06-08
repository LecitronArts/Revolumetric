# RenderGraph Transient Slot Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a deterministic CPU-side transient resource slot plan from RenderGraph lifetimes without changing Vulkan allocation ownership.

**Architecture:** `TransientResourceSlot` lives with resource metadata in `src/render/resource.rs`; `RenderGraph` computes slots during `compile` after lifetimes are available and invalidates them with other compile products. Slot grouping is descriptor-exact and lifetime-non-overlapping.

**Tech Stack:** Rust, ash Vulkan descriptor types, existing RenderGraph unit tests, skip-mode Cargo library tests.

---

## File Structure

- Modify `src/render/resource.rs`: add `TransientResourceSlot`.
- Modify `src/render/graph.rs`: store, invalidate, compute, and expose transient resource slots.
- Create `docs/superpowers/specs/2026-05-31-rendergraph-transient-slot-plan-design.md`.
- Create `docs/superpowers/plans/2026-05-31-rendergraph-transient-slot-plan.md`.

---

### Task 1: Slot Type

**Files:**
- Modify: `src/render/resource.rs`

- [x] **Step 1: Write failing type test**

Add to `src/render/resource.rs` tests:

```rust
#[test]
fn rendergraph_transient_slot_plan_struct_exposes_slot_desc_and_resources() {
    let slot = TransientResourceSlot {
        slot_index: 2,
        desc: ResourceDesc::Buffer {
            size: 256,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        },
        resource_ids: vec![4, 9],
    };

    assert_eq!(slot.slot_index, 2);
    assert_eq!(slot.resource_ids, vec![4, 9]);
    assert_eq!(
        slot.desc,
        ResourceDesc::Buffer {
            size: 256,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        }
    );
}
```

- [x] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_transient_slot_plan_struct_exposes_slot_desc_and_resources; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile failure because `TransientResourceSlot` does not exist.

- [x] **Step 3: Implement type**

Add to `src/render/resource.rs`:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransientResourceSlot {
    pub slot_index: usize,
    pub desc: ResourceDesc,
    pub resource_ids: Vec<u32>,
}
```

- [x] **Step 4: Verify GREEN**

Run the focused type test. Expected: pass.

---

### Task 2: Slot Planning API

**Files:**
- Modify: `src/render/graph.rs`

- [x] **Step 1: Write failing packing test**

Add to `src/render/graph.rs` tests:

```rust
#[test]
fn rendergraph_transient_slot_plan_packs_non_overlapping_transients() {
    let mut graph = RenderGraph::new();
    let first = graph.add_pass("first_write", QueueType::Compute, |builder| {
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(first, fake_image(601));
    let marker = graph.add_pass("first_read", QueueType::Compute, |builder| {
        builder.read_as(first, AccessKind::ComputeShaderRead);
        builder.create_buffer(4, vk::BufferUsageFlags::STORAGE_BUFFER);
        Box::new(|_ctx| {})
    })[0];
    graph.bind_buffer(marker, fake_buffer(602));
    let second = graph.add_pass("second_write", QueueType::Compute, |builder| {
        builder.depend_on(marker);
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(second, fake_image(603));

    graph.compile().unwrap();

    let image_slots: Vec<_> = graph
        .transient_resource_slots()
        .iter()
        .filter(|slot| matches!(slot.desc, ResourceDesc::Image { .. }))
        .collect();
    assert_eq!(image_slots.len(), 1);
    assert_eq!(image_slots[0].resource_ids, vec![first.id, second.id]);
}
```

- [x] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_transient_slot_plan_packs_non_overlapping_transients; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile failure because `transient_resource_slots` does not exist.

- [x] **Step 3: Implement slot storage and API**

Add `transient_resource_slots: Vec<TransientResourceSlot>` to `RenderGraph`, initialize and invalidate it, and expose:

```rust
pub fn transient_resource_slots(&self) -> &[TransientResourceSlot] {
    &self.transient_resource_slots
}
```

- [x] **Step 4: Implement slot planner**

Add `compute_transient_resource_slots`. It uses transient lifetimes sorted by `(first_step, last_step, resource_id)` and greedily places compatible non-overlapping resources into existing slots before creating a new slot.

Call it during `compile` after lifetimes are computed.

- [x] **Step 5: Verify GREEN**

Run the focused packing test. Expected: pass.

---

### Task 3: Slot Planning Boundaries

**Files:**
- Modify: `src/render/graph.rs`

- [x] **Step 1: Write failing boundary tests**

Add tests:

```rust
#[test]
fn rendergraph_transient_slot_plan_separates_overlapping_transients() {
    let mut graph = RenderGraph::new();
    let first = graph.add_pass("first_write", QueueType::Compute, |builder| {
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(first, fake_image(701));
    let second = graph.add_pass("second_write", QueueType::Compute, |builder| {
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(second, fake_image(702));
    graph.add_pass("read_both", QueueType::Compute, |builder| {
        builder.read_as(first, AccessKind::ComputeShaderRead);
        builder.read_as(second, AccessKind::ComputeShaderRead);
        Box::new(|_ctx| {})
    });

    graph.compile().unwrap();

    let image_slots: Vec<_> = graph
        .transient_resource_slots()
        .iter()
        .filter(|slot| matches!(slot.desc, ResourceDesc::Image { .. }))
        .collect();
    assert_eq!(image_slots.len(), 2);
}

#[test]
fn rendergraph_transient_slot_plan_separates_different_descriptors() {
    let mut graph = RenderGraph::new();
    let small = graph.add_pass("small_write", QueueType::Compute, |builder| {
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(small, fake_image(801));
    let marker = graph.add_pass("small_read", QueueType::Compute, |builder| {
        builder.read_as(small, AccessKind::ComputeShaderRead);
        builder.create_buffer(4, vk::BufferUsageFlags::STORAGE_BUFFER);
        Box::new(|_ctx| {})
    })[0];
    graph.bind_buffer(marker, fake_buffer(802));
    let large = graph.add_pass("large_write", QueueType::Compute, |builder| {
        builder.depend_on(marker);
        builder.create_image(
            128,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(large, fake_image(803));

    graph.compile().unwrap();

    let image_slots: Vec<_> = graph
        .transient_resource_slots()
        .iter()
        .filter(|slot| matches!(slot.desc, ResourceDesc::Image { .. }))
        .collect();
    assert_eq!(image_slots.len(), 2);
}
```

- [x] **Step 2: Verify RED/GREEN**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_transient_slot_plan_separates_overlapping_transients; cargo test --lib rendergraph_transient_slot_plan_separates_different_descriptors; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected after Task 2 implementation: both pass. If either fails, fix the planner before continuing.

- [x] **Step 3: Add invalidation assertion**

Extend `rendergraph_resource_lifecycle_clears_compile_products_after_graph_mutation` to also assert:

```rust
assert!(graph.transient_resource_slots().is_empty());
```

- [x] **Step 4: Verify invalidation**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_resource_lifecycle_clears_compile_products_after_graph_mutation; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: pass.

---

### Task 4: Full Verification

**Files:**
- All modified files above.

- [x] **Step 1: Format**

Run:

```powershell
cargo fmt
```

- [x] **Step 2: Focused slot tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_transient_slot; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all slot-plan tests pass.

- [x] **Step 3: Full library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all library tests pass.

- [x] **Step 4: Whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. Existing LF/CRLF warnings on Windows are acceptable.

---

## Execution Results

- Added `TransientResourceSlot` in `src/render/resource.rs` with deterministic slot metadata: `slot_index`, exact `ResourceDesc`, and grouped `resource_ids`.
- Added `RenderGraph::transient_resource_slots()` and compile-time slot planning in `src/render/graph.rs`.
- Slot planner currently runs after resource lifetimes are computed and before alias candidates/barrier validation are finalized. It only packs transient resources with equal descriptors and non-overlapping sorted execution lifetimes.
- Graph mutation now clears stale transient slot plans through `invalidate_compile_products`.
- Focused verification:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_transient_slot; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Result: `4 passed, 0 failed`.

- Full library verification:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Result: `457 passed, 0 failed`.

- Whitespace verification:

```powershell
git diff --check
```

Result: exit code `0`; Windows LF-to-CRLF warnings were emitted for existing modified files, with no whitespace errors.
