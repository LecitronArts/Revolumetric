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
        let legacy_spawn_call = ["world", ".spawn();"].concat();
        assert!(source.contains("let _ = world.spawn(())"));
        assert!(!source.contains(&legacy_spawn_call));
    }
}
