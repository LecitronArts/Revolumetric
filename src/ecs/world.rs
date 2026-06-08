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
        self.entities.len() as usize
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
            .into_iter()
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
        assert_eq!(
            world.resource::<String>().map(String::as_str),
            Some("hello")
        );

        world.resource_mut::<String>().unwrap().push('!');
        assert_eq!(
            world.resource::<String>().map(String::as_str),
            Some("hello!")
        );
    }

    #[test]
    fn world_is_backed_by_hecs_instead_of_slotmap() {
        let source = crate::render::source_checks::read_source("src/ecs/world.rs");
        let slotmap_import = ["slotmap", "::SlotMap"].concat();
        let legacy_cache_token = ["entity", "_cache"].concat();
        assert!(source.contains("hecs::World"));
        assert!(!source.contains(&slotmap_import));
        assert!(!source.contains(&legacy_cache_token));
    }
}
