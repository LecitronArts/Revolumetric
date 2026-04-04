mod app;
mod assets;
mod ecs;
mod platform;
mod render;
mod scene;
mod voxel;

fn main() -> anyhow::Result<()> {
    app::run()
}
