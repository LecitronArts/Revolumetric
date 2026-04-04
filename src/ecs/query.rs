use crate::ecs::entity::Entity;

#[derive(Debug, Clone, Copy)]
pub struct Query<'a> {
    entities: &'a [Entity],
}

impl<'a> Query<'a> {
    pub fn new(entities: &'a [Entity]) -> Self {
        Self { entities }
    }

    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.entities.iter().copied()
    }
}
