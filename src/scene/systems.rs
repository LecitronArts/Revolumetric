use anyhow::Result;

use crate::ecs::world::World;
use crate::platform::time::Time;
use crate::scene::components::CameraRig;
use crate::scene::light::DirectionalLight;

pub fn bootstrap_scene(world: &mut World) -> Result<()> {
    world.spawn();
    world.insert_resource(CameraRig::default());
    world.insert_resource(DirectionalLight::default());
    Ok(())
}

pub fn tick_time(world: &mut World) -> Result<()> {
    if let Some(time) = world.resource_mut::<Time>() {
        time.advance(1.0 / 60.0);
    }
    Ok(())
}
