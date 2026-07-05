pub mod allocator;
pub mod area_restir;
pub mod brick_generation_atlas;
pub mod buffer;
pub mod camera;
pub mod capture;
pub mod descriptor;
pub mod device;
#[cfg(not(target_os = "android"))]
pub mod egui_renderer;
pub mod frame;
pub mod gpu_profiler;
pub mod graph;
pub mod image;
pub mod pass_context;
pub mod passes;
pub mod pipeline;
pub mod resource;
pub mod restir_di;
pub mod restir_gi;
pub mod rt_capabilities;
pub mod rt_history;
pub mod rt_pipeline;
pub mod rt_scene;
pub mod rt_settings;
pub mod runtime;
pub mod sampler;
pub mod scene_ubo;
pub mod shader;
pub mod swapchain;
pub mod traversal_stats;
pub mod vpt_history;
pub mod vpt_motion;
pub mod vpt_pipeline;

#[cfg(test)]
pub(crate) mod source_checks;

#[cfg(test)]
mod tests {
    #[test]
    fn render_exports_egui_renderer_module() {
        let source = crate::render::source_checks::read_source("src/render/mod.rs");
        let has_export = source
            .lines()
            .any(|line| line.trim() == "pub mod egui_renderer;");

        assert!(
            has_export,
            "render module must export the custom Vulkan egui renderer"
        );
    }
}
