#![allow(clippy::collapsible_if, clippy::collapsible_match)]

#[cfg(target_os = "android")]
use winit::platform::android::activity::AndroidApp;

pub mod app;
pub mod assets;
pub mod ecs;
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
