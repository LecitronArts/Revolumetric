use anyhow::Result;

use crate::ecs::world::World;
use crate::platform::input::InputState;
use crate::scene::components::CameraRig;
use crate::scene::light::DirectionalLight;

pub fn bootstrap_scene(world: &mut World) -> Result<()> {
    world.spawn();
    world.insert_resource(CameraRig::default());
    world.insert_resource(DirectionalLight::default());
    world.insert_resource(InputState::default());
    Ok(())
}
