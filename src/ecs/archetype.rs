use std::any::TypeId;

#[derive(Debug, Clone, Default)]
pub struct Archetype {
    component_types: Vec<TypeId>,
}

impl Archetype {
    pub fn new(mut component_types: Vec<TypeId>) -> Self {
        component_types.sort();
        component_types.dedup();
        Self { component_types }
    }

    pub fn component_types(&self) -> &[TypeId] {
        &self.component_types
    }
}
