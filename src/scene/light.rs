use glam::Vec3;

#[derive(Debug, Clone)]
pub struct DirectionalLight {
    pub direction: Vec3,
    /// Solar-disk radiance used by the VPT finite sun estimator.
    ///
    /// This is not legacy directional irradiance. VPT samples the solar disk
    /// and evaluates Lambertian direct lighting as f * Li * cos / pdf.
    pub intensity: Vec3,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.5, 1.0, 0.25).normalize(),
            intensity: Vec3::new(2.0, 1.5, 1.25),
        }
    }
}
