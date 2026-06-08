# RenderGraph Resource Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CPU-verifiable RenderGraph resource origin, lifetime, and transient alias candidate planning without changing runtime Vulkan allocation ownership.

**Architecture:** `ResourceOrigin`, `ResourceLifetime`, and `TransientAliasCandidate` live in `src/render/resource.rs`; `RenderGraph` records origins as resources are imported or graph-created, computes lifetimes after pass sorting, and exposes read-only planning APIs. Existing pass-owned GPU resources and barrier execution stay unchanged.

**Tech Stack:** Rust, ash Vulkan handle types, existing RenderGraph unit tests, skip-mode library tests.

---

## File Structure

- Modify `src/render/resource.rs`: add origin and lifecycle planning structs.
- Modify `src/render/graph.rs`: store origins, compute lifetimes, compute transient alias candidates, and expose inspection APIs.
- Modify `docs/superpowers/specs/2026-05-31-rendergraph-resource-lifecycle-design.md`: written design for this phase.
- Modify `docs/superpowers/plans/2026-05-31-rendergraph-resource-lifecycle.md`: implementation plan and execution results.

---

### Task 1: Resource Lifecycle Types

**Files:**
- Modify: `src/render/resource.rs`

- [ ] **Step 1: Write type-level tests**

Add tests in `src/render/resource.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rendergraph_resource_lifecycle_origin_values_are_copyable() {
        let imported = ResourceOrigin::Imported;
        let transient = ResourceOrigin::Transient;

        assert_eq!(imported, ResourceOrigin::Imported);
        assert_eq!(transient, ResourceOrigin::Transient);
    }

    #[test]
    fn rendergraph_resource_lifecycle_structs_expose_ids_and_steps() {
        let lifetime = ResourceLifetime {
            resource_id: 7,
            origin: ResourceOrigin::Transient,
            first_step: 1,
            last_step: 3,
        };
        let candidate = TransientAliasCandidate {
            first_resource_id: 7,
            second_resource_id: 9,
        };

        assert_eq!(lifetime.resource_id, 7);
        assert_eq!(lifetime.origin, ResourceOrigin::Transient);
        assert_eq!(lifetime.first_step, 1);
        assert_eq!(lifetime.last_step, 3);
        assert_eq!(candidate.first_resource_id, 7);
        assert_eq!(candidate.second_resource_id, 9);
    }
}
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_resource_lifecycle_origin_values_are_copyable; cargo test --lib rendergraph_resource_lifecycle_structs_expose_ids_and_steps; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile failure because `ResourceOrigin`, `ResourceLifetime`, and `TransientAliasCandidate` do not exist.

- [ ] **Step 3: Implement lifecycle types**

Add to `src/render/resource.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceOrigin {
    Imported,
    Transient,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceLifetime {
    pub resource_id: u32,
    pub origin: ResourceOrigin,
    pub first_step: usize,
    pub last_step: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransientAliasCandidate {
    pub first_resource_id: u32,
    pub second_resource_id: u32,
}
```

- [ ] **Step 4: Verify GREEN**

Run the same focused tests. Expected: both tests pass.

---

### Task 2: Record Resource Origins

**Files:**
- Modify: `src/render/graph.rs`

- [ ] **Step 1: Write failing origin test**

Add to `src/render/graph.rs` tests:

```rust
#[test]
fn rendergraph_resource_lifecycle_records_imported_and_transient_origins() {
    let mut graph = RenderGraph::new();
    let imported = graph.import_image(
        64,
        64,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::STORAGE,
    );
    let transient = graph.add_pass("producer", QueueType::Compute, |builder| {
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];

    assert_eq!(graph.resource_origin(imported), Some(ResourceOrigin::Imported));
    assert_eq!(graph.resource_origin(transient), Some(ResourceOrigin::Transient));
}
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_resource_lifecycle_records_imported_and_transient_origins; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile failure because `resource_origin` and origin storage do not exist.

- [ ] **Step 3: Store origins**

In `RenderGraph`, add:

```rust
resource_origins: BTreeMap<u32, ResourceOrigin>,
resource_lifetimes: Vec<ResourceLifetime>,
transient_alias_candidates: Vec<TransientAliasCandidate>,
```

Set imported origin in `import_image` and `import_buffer`. Set transient origin for each `builder.resource_descs` entry in `add_pass`.

Add:

```rust
pub fn resource_origin(&self, handle: ResourceHandle) -> Option<ResourceOrigin> {
    self.resource_origins.get(&handle.id).copied()
}
```

- [ ] **Step 4: Verify GREEN**

Run the focused origin test. Expected: pass.

---

### Task 3: Compute Sorted Resource Lifetimes

**Files:**
- Modify: `src/render/graph.rs`

- [ ] **Step 1: Write failing lifetime test**

Add to `src/render/graph.rs` tests:

```rust
#[test]
fn rendergraph_resource_lifecycle_records_sorted_execution_lifetimes() {
    let mut graph = RenderGraph::new();
    let imported = graph.import_image_with_access(
        fake_image(201),
        64,
        64,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::STORAGE,
        AccessKind::ComputeShaderRead,
    );
    let transient = graph.add_pass("producer", QueueType::Compute, |builder| {
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(transient, fake_image(202));

    graph.add_pass("consumer", QueueType::Compute, |builder| {
        builder.read_as(transient, AccessKind::ComputeShaderRead);
        builder.read_as(imported, AccessKind::ComputeShaderRead);
        Box::new(|_ctx| {})
    });

    graph.compile().unwrap();

    let imported_lifetime = graph
        .resource_lifetimes()
        .iter()
        .find(|lifetime| lifetime.resource_id == imported.id)
        .unwrap();
    let transient_lifetime = graph
        .resource_lifetimes()
        .iter()
        .find(|lifetime| lifetime.resource_id == transient.id)
        .unwrap();

    assert_eq!(imported_lifetime.origin, ResourceOrigin::Imported);
    assert_eq!(imported_lifetime.first_step, 1);
    assert_eq!(imported_lifetime.last_step, 1);
    assert_eq!(transient_lifetime.origin, ResourceOrigin::Transient);
    assert_eq!(transient_lifetime.first_step, 0);
    assert_eq!(transient_lifetime.last_step, 1);
}
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_resource_lifecycle_records_sorted_execution_lifetimes; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile failure because `resource_lifetimes` does not exist.

- [ ] **Step 3: Implement lifetime planning**

Add `compute_resource_lifetimes` in `RenderGraph`. It walks `self.sorted_order`, collects touched resource ids from each pass, records first and last sorted step, and returns lifetimes sorted by `resource_id`.

Set `self.resource_lifetimes = self.compute_resource_lifetimes();` during `compile` after `self.sorted_order = order;`.

Add:

```rust
pub fn resource_lifetimes(&self) -> &[ResourceLifetime] {
    &self.resource_lifetimes
}
```

- [ ] **Step 4: Verify GREEN**

Run the focused lifetime test. Expected: pass.

---

### Task 4: Compute Transient Alias Candidates

**Files:**
- Modify: `src/render/graph.rs`

- [ ] **Step 1: Write failing alias tests**

Add to `src/render/graph.rs` tests:

```rust
#[test]
fn rendergraph_resource_lifecycle_plans_alias_candidates_for_non_overlapping_transients() {
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
    graph.bind_image(first, fake_image(301));
    let first_read_marker = graph.add_pass("first_read", QueueType::Compute, |builder| {
        builder.read_as(first, AccessKind::ComputeShaderRead);
        builder.create_buffer(4, vk::BufferUsageFlags::STORAGE_BUFFER);
        Box::new(|_ctx| {})
    })[0];
    graph.bind_buffer(first_read_marker, fake_buffer(303));
    let second = graph.add_pass("second_write", QueueType::Compute, |builder| {
        builder.depend_on(first_read_marker);
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(second, fake_image(302));

    graph.compile().unwrap();

    assert!(
        graph.transient_alias_candidates().contains(&TransientAliasCandidate {
            first_resource_id: first.id,
            second_resource_id: second.id,
        }),
        "compatible non-overlapping transient images should be alias candidates"
    );
}

#[test]
fn rendergraph_resource_lifecycle_excludes_imported_and_overlapping_alias_candidates() {
    let mut graph = RenderGraph::new();
    let imported = graph.import_image_with_access(
        fake_image(401),
        64,
        64,
        vk::Format::R8G8B8A8_UNORM,
        vk::ImageUsageFlags::STORAGE,
        AccessKind::ComputeShaderRead,
    );
    let first = graph.add_pass("first_write", QueueType::Compute, |builder| {
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(first, fake_image(402));
    let overlapping = graph.add_pass("overlapping_write", QueueType::Compute, |builder| {
        builder.create_image(
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
        );
        Box::new(|_ctx| {})
    })[0];
    graph.bind_image(overlapping, fake_image(403));
    graph.add_pass("read_both", QueueType::Compute, |builder| {
        builder.read_as(first, AccessKind::ComputeShaderRead);
        builder.read_as(overlapping, AccessKind::ComputeShaderRead);
        builder.read_as(imported, AccessKind::ComputeShaderRead);
        Box::new(|_ctx| {})
    });

    graph.compile().unwrap();

    assert!(
        graph
            .transient_alias_candidates()
            .iter()
            .all(|candidate| candidate.first_resource_id != imported.id
                && candidate.second_resource_id != imported.id),
        "imported resources must not be transient alias candidates"
    );
    assert!(
        !graph.transient_alias_candidates().contains(&TransientAliasCandidate {
            first_resource_id: first.id,
            second_resource_id: overlapping.id,
        }),
        "overlapping transient lifetimes must not alias"
    );
}
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_resource_lifecycle_plans_alias_candidates_for_non_overlapping_transients; cargo test --lib rendergraph_resource_lifecycle_excludes_imported_and_overlapping_alias_candidates; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: compile failure because `transient_alias_candidates` does not exist.

- [ ] **Step 3: Implement alias planning**

Add `compute_transient_alias_candidates` in `RenderGraph`. It compares lifetime pairs and emits candidates only for `ResourceOrigin::Transient`, equal `ResourceDesc`, and non-overlapping sorted-step lifetimes.

Set `self.transient_alias_candidates = self.compute_transient_alias_candidates();` during `compile` after lifetimes are computed.

Add:

```rust
pub fn transient_alias_candidates(&self) -> &[TransientAliasCandidate] {
    &self.transient_alias_candidates
}
```

- [ ] **Step 4: Verify GREEN**

Run the focused alias tests. Expected: both tests pass.

---

### Task 5: Full Verification

**Files:**
- All modified files above.

- [ ] **Step 1: Format**

Run:

```powershell
cargo fmt
```

- [ ] **Step 2: Focused lifecycle tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_resource_lifecycle; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all lifecycle-focused tests pass.

- [ ] **Step 3: Full library tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: all library tests pass.

- [ ] **Step 4: Whitespace check**

Run:

```powershell
git diff --check
```

Expected: no whitespace errors. Existing LF/CRLF warnings on Windows are acceptable.

---

## Execution Results

- Added `ResourceOrigin`, `ResourceLifetime`, and `TransientAliasCandidate` in `src/render/resource.rs`.
- `RenderGraph` now records imported vs transient resource origins, computes sorted execution lifetimes during `compile`, and exposes `resource_origin`, `resource_lifetimes`, and `transient_alias_candidates`.
- Alias candidates are limited to graph-transient resources with equal descriptors and non-overlapping sorted-step lifetimes.
- Graph mutations now clear stale compile products, including barrier plans, lifetimes, and alias candidates.
- Focused verification:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib rendergraph_resource_lifecycle; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

- Result observed during execution: 7 passed, 0 failed.
- Full verification:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

- Result observed during execution: 453 passed, 0 failed.
