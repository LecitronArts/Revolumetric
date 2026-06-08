pub mod resource;
pub mod schedule;
pub mod system;
pub mod world;

#[cfg(test)]
mod tests {
    #[test]
    fn ecs_module_stops_exporting_legacy_helpers() {
        let source = crate::render::source_checks::read_source("src/ecs/mod.rs");
        for module in ["archetype", "column", "commands", "entity", "query"] {
            assert!(
                !source.contains(&format!("pub mod {module};")),
                "src/ecs/mod.rs must not export {module}"
            );
        }
    }
}
