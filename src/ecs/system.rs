use anyhow::Result;

use crate::ecs::world::World;

pub type SystemFn = fn(&mut World) -> Result<()>;
