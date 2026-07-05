#![allow(clippy::collapsible_if, clippy::collapsible_match)]

#[cfg(target_os = "android")]
use winit::platform::android::activity::AndroidApp;

pub mod app;
pub mod assets;
pub mod build_support;
pub mod ecs;
#[cfg(not(target_os = "android"))]
pub mod editor;
pub mod platform;
pub mod render;
pub mod scene;
pub mod voxel;

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub extern "C" fn android_main(app: AndroidApp) {
    if let Err(error) = app::run_android(app) {
        eprintln!("{error}");
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn editor_module_is_exported() {
        let source = crate::render::source_checks::read_source("src/lib.rs");

        let has_editor_export = source.lines().any(|line| line.trim() == "pub mod editor;");

        assert!(
            has_editor_export,
            "src/lib.rs must export the desktop editor module"
        );
    }
}
