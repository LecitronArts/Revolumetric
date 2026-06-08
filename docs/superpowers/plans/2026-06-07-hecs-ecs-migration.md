# hecs ECS Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom ECS storage layer with `hecs` while preserving app resources, schedule execution, and scene bootstrap behavior.

**Architecture:** `src/ecs/world.rs` becomes the only entity/component store and wraps `hecs::World` directly. `Resources` stays custom for app-wide typed state. `src/ecs/mod.rs` stops exporting the retired slotmap/query/commands scaffolding, and `src/scene/systems.rs` uses hecs spawn semantics while still inserting the same resources. Tests cover behavior at the world boundary and source-level guardrails so we can prove the old ECS surface is gone rather than only relying on compilation.

**Tech Stack:** Rust 2024, `hecs 0.11`, existing typed resource table, `cargo test --lib`, `REVOLUMETRIC_SHADER_COMPILE=skip`.

---

## File Structure

- Modify: `Cargo.toml` - add `hecs` and drop the now-unused `slotmap` dependency.
- Modify: `Cargo.lock` - refresh dependency resolution.
- Modify: `src/ecs/world.rs` - wrap `hecs::World`, keep `Resources`, add hecs-backed spawn/query/despawn helpers, and add behavior tests.
- Modify: `src/ecs/mod.rs` - export only the live ECS modules and add source guard tests for the removed public surface.
- Delete: `src/ecs/archetype.rs`
- Delete: `src/ecs/column.rs`
- Delete: `src/ecs/commands.rs`
- Delete: `src/ecs/entity.rs`
- Delete: `src/ecs/query.rs`
- Modify: `src/scene/systems.rs` - switch bootstrap spawn to hecs bundle syntax and add integration tests.

---

### Task 1: Rebuild The World Boundary Around hecs

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `src/ecs/world.rs`

- [ ] **Step 1: Write the failing tests**

Add these tests to `src/ecs/world.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Marker(u32);

#[test]
fn world_spawns_queries_and_despawns_hecs_entities() {
    let mut world = World::new();
    let entity = world.spawn((Marker(7),));
    assert_eq!(world.entity_count(), 1);

    let values: Vec<u32> = world
        .query_mut::<&mut Marker>()
        .map(|marker| {
            marker.0 += 1;
            marker.0
        })
        .collect();

    assert_eq!(values, vec![8]);
    assert!(world.despawn(entity));
    assert_eq!(world.entity_count(), 0);
    assert!(!world.despawn(entity));
}

#[test]
fn world_resource_table_still_round_trips_values() {
    let mut world = World::new();
    world.insert_resource(String::from("hello"));
    assert_eq!(world.resource::<String>().map(String::as_str), Some("hello"));

    world.resource_mut::<String>().unwrap().push('!');
    assert_eq!(world.resource::<String>().map(String::as_str), Some("hello!"));
}
```

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib world_spawns_queries_and_despawns_hecs_entities; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib world_resource_table_still_round_trips_values; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: fail to compile because `hecs` is not wired in and `World` still uses the old slotmap-backed shape.

- [ ] **Step 2: Implement the hecs-backed world**

Replace `src/ecs/world.rs` with:

```rust
use hecs::{Bundle, Query, QueryBorrow, QueryMut, World as HecsWorld};

use crate::ecs::resource::Resources;

#[derive(Default)]
pub struct World {
    entities: HecsWorld,
    resources: Resources,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> hecs::Entity {
        self.entities.spawn(bundle)
    }

    pub fn despawn(&mut self, entity: hecs::Entity) -> bool {
        self.entities.despawn(entity).is_ok()
    }

    pub fn query<Q: Query>(&self) -> QueryBorrow<'_, Q> {
        self.entities.query::<Q>()
    }

    pub fn query_mut<Q: Query>(&mut self) -> QueryMut<'_, Q> {
        self.entities.query_mut::<Q>()
    }

    pub fn insert_resource<T>(&mut self, value: T)
    where
        T: Send + Sync + 'static,
    {
        self.resources.insert(value);
    }

    pub fn resource<T>(&self) -> Option<&T>
    where
        T: Send + Sync + 'static,
    {
        self.resources.get::<T>()
    }

    pub fn resource_mut<T>(&mut self) -> Option<&mut T>
    where
        T: Send + Sync + 'static,
    {
        self.resources.get_mut::<T>()
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

#[cfg(test)]
mod tests {
    use super::World;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct Marker(u32);

    #[test]
    fn world_spawns_queries_and_despawns_hecs_entities() {
        let mut world = World::new();
        let entity = world.spawn((Marker(7),));
        assert_eq!(world.entity_count(), 1);

        let values: Vec<u32> = world
            .query_mut::<&mut Marker>()
            .map(|marker| {
                marker.0 += 1;
                marker.0
            })
            .collect();

        assert_eq!(values, vec![8]);
        assert!(world.despawn(entity));
        assert_eq!(world.entity_count(), 0);
        assert!(!world.despawn(entity));
    }

    #[test]
    fn world_resource_table_still_round_trips_values() {
        let mut world = World::new();
        world.insert_resource(String::from("hello"));
        assert_eq!(world.resource::<String>().map(String::as_str), Some("hello"));

        world.resource_mut::<String>().unwrap().push('!');
        assert_eq!(world.resource::<String>().map(String::as_str), Some("hello!"));
    }
}
```

- [ ] **Step 3: Run the focused tests again**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib world_spawns_queries_and_despawns_hecs_entities; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib world_resource_table_still_round_trips_values; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: pass.

---

### Task 2: Remove The Legacy ECS Modules And Export Surface

**Files:**
- Modify: `src/ecs/mod.rs`
- Delete: `src/ecs/archetype.rs`
- Delete: `src/ecs/column.rs`
- Delete: `src/ecs/commands.rs`
- Delete: `src/ecs/entity.rs`
- Delete: `src/ecs/query.rs`

- [ ] **Step 1: Write the failing source guard tests**

Add these tests to `src/ecs/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn ecs_module_stops_exporting_legacy_helpers() {
        let source = crate::render::source_checks::read_source("src/ecs/mod.rs");
        for module in ["archetype", "column", "commands", "entity", "query"] {
            assert!(
                !source.contains(&format!("pub mod {module};")),
                "src/ecs/mod.rs must not export {module}"
            );
        }
    }
}
```

Add this source guard to `src/ecs/world.rs`:

```rust
#[test]
fn world_is_backed_by_hecs_instead_of_slotmap() {
    let source = crate::render::source_checks::read_source("src/ecs/world.rs");
    assert!(source.contains("hecs::World"));
    assert!(!source.contains("slotmap::SlotMap"));
    assert!(!source.contains("entity_cache"));
}
```

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib ecs_module_stops_exporting_legacy_helpers; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib world_is_backed_by_hecs_instead_of_slotmap; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: fail because the old exports and slotmap-backed world are still present.

- [ ] **Step 2: Remove the legacy modules**

Update `src/ecs/mod.rs` to export only the live modules:

```rust
pub mod resource;
pub mod schedule;
pub mod system;
pub mod world;
```

Delete the legacy files listed above. Do not replace them with stub implementations.

- [ ] **Step 3: Re-run the source guard tests**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib ecs_module_stops_exporting_legacy_helpers; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib world_is_backed_by_hecs_instead_of_slotmap; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: pass.

---

### Task 3: Update Scene Bootstrapping And Verify The Full Library

**Files:**
- Modify: `src/scene/systems.rs`

- [ ] **Step 1: Write the failing integration and source guard tests**

Add these tests to `src/scene/systems.rs`:

```rust
#[test]
fn bootstrap_scene_spawns_one_entity_and_inserts_resources() {
    let mut world = World::new();

    bootstrap_scene(&mut world).unwrap();

    assert_eq!(world.entity_count(), 1);
    assert!(world.resource::<CameraRig>().is_some());
    assert!(world.resource::<DirectionalLight>().is_some());
    assert!(world.resource::<InputState>().is_some());
}

#[test]
fn bootstrap_scene_uses_hecs_spawn_semantics() {
    let source = crate::render::source_checks::read_source("src/scene/systems.rs");
    assert!(source.contains("let _ = world.spawn(())"));
    assert!(!source.contains("world.spawn();"));
}
```

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib bootstrap_scene_spawns_one_entity_and_inserts_resources; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo test --lib bootstrap_scene_uses_hecs_spawn_semantics; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Expected: fail because `bootstrap_scene` still calls the old zero-argument spawn path.

- [ ] **Step 2: Implement hecs bundle spawning in the scene bootstrap**

Replace `src/scene/systems.rs` with:

```rust
use anyhow::Result;

use crate::ecs::world::World;
use crate::platform::input::InputState;
use crate::scene::components::CameraRig;
use crate::scene::light::DirectionalLight;

pub fn bootstrap_scene(world: &mut World) -> Result<()> {
    let _ = world.spawn(());
    world.insert_resource(CameraRig::default());
    world.insert_resource(DirectionalLight::default());
    world.insert_resource(InputState::default());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_scene_spawns_one_entity_and_inserts_resources() {
        let mut world = World::new();

        bootstrap_scene(&mut world).unwrap();

        assert_eq!(world.entity_count(), 1);
        assert!(world.resource::<CameraRig>().is_some());
        assert!(world.resource::<DirectionalLight>().is_some());
        assert!(world.resource::<InputState>().is_some());
    }

    #[test]
    fn bootstrap_scene_uses_hecs_spawn_semantics() {
        let source = crate::render::source_checks::read_source("src/scene/systems.rs");
        assert!(source.contains("let _ = world.spawn(())"));
        assert!(!source.contains("world.spawn();"));
    }
}
```

- [ ] **Step 3: Run the full library verification**

Run:

```powershell
$env:REVOLUMETRIC_SHADER_COMPILE='skip'; cargo fmt; cargo test --lib; Remove-Item Env:\REVOLUMETRIC_SHADER_COMPILE
```

Then run:

```powershell
git diff --check
git status --short --branch
```

Expected: all library tests pass, and the diff only contains the hecs migration work.

---

## Self-Review

- Spec coverage: The plan covers the hecs dependency, world wrapper, resource preservation, query access, legacy module removal, bootstrap scene migration, and final verification.
- Placeholder scan: No TODO/TBD placeholders remain in the plan.
- Type consistency: `World`, `Resources`, `hecs::Entity`, `QueryBorrow`, `QueryMut`, and `bootstrap_scene` are named consistently across tasks.
