use std::collections::BTreeMap;

use anyhow::Result;

use crate::ecs::system::SystemFn;
use crate::ecs::world::World;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    Startup,
    PreUpdate,
    Update,
    PostUpdate,
    ExtractRender,
    PrepareRender,
    ExecuteRender,
}

#[derive(Default)]
pub struct Schedule {
    stages: BTreeMap<Stage, Vec<SystemFn>>,
}

impl Schedule {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_stage(&mut self, stage: Stage) {
        self.stages.entry(stage).or_default();
    }

    pub fn add_system(&mut self, stage: Stage, system: SystemFn) {
        self.stages.entry(stage).or_default().push(system);
    }

    pub fn run(&self, world: &mut World) -> Result<()> {
        for stage in self.stages.keys().copied() {
            self.run_stage(stage, world)?;
        }
        Ok(())
    }

    pub fn run_stage(&self, stage: Stage, world: &mut World) -> Result<()> {
        if let Some(systems) = self.stages.get(&stage) {
            for system in systems {
                system(world)?;
            }
        }
        Ok(())
    }
}
