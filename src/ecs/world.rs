use slotmap::SlotMap;

use crate::ecs::entity::{Entity, EntityId};
use crate::ecs::query::Query;
use crate::ecs::resource::Resources;

pub struct World {
    entities: SlotMap<EntityId, ()>,
    resources: Resources,
    entity_cache: Vec<Entity>,
}

impl Default for World {
    fn default() -> Self {
        Self {
            entities: SlotMap::with_key(),
            resources: Resources::default(),
            entity_cache: Vec::new(),
        }
    }
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn(&mut self) -> Entity {
        let id = self.entities.insert(());
        let entity = Entity::new(id);
        self.entity_cache.push(entity);
        entity
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

    pub fn query(&self) -> Query<'_> {
        Query::new(&self.entity_cache)
    }

    pub fn entity_count(&self) -> usize {
        self.entity_cache.len()
    }
}
