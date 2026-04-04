use std::any::{Any, TypeId};
use std::collections::HashMap;

#[derive(Default)]
pub struct ColumnStorage {
    columns: HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>,
}

impl ColumnStorage {
    pub fn insert<T>(&mut self, value: T)
    where
        T: Any + Send + Sync,
    {
        self.columns
            .entry(TypeId::of::<T>())
            .or_default()
            .push(Box::new(value));
    }

    pub fn len_for<T>(&self) -> usize
    where
        T: Any + Send + Sync,
    {
        self.columns
            .get(&TypeId::of::<T>())
            .map(Vec::len)
            .unwrap_or_default()
    }
}
