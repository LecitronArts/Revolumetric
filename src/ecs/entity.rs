use slotmap::new_key_type;

new_key_type! {
    pub struct EntityId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    id: EntityId,
}

impl Entity {
    pub fn new(id: EntityId) -> Self {
        Self { id }
    }

    pub fn id(self) -> EntityId {
        self.id
    }
}
