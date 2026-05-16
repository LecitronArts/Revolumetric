#[cfg(not(target_os = "android"))]
fn main() -> anyhow::Result<()> {
    revolumetric::app::run()
}
