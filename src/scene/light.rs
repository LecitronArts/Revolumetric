use glam::Vec3;

pub const DEFAULT_SUN_IRRADIANCE: Vec3 = Vec3::new(2.0, 1.5, 1.25);

#[derive(Debug, Clone)]
pub struct DirectionalLight {
    pub direction: Vec3,
    /// Total solar irradiance used by VPT/RT finite sun estimators.
    ///
    /// This matches the legacy directional-light strength. Finite sun shaders
    /// derive the per-direction disk radiance internally, so changing angular
    /// radius changes penumbra size rather than total brightness.
    pub intensity: Vec3,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: Vec3::new(0.5, 1.0, 0.25).normalize(),
            intensity: DEFAULT_SUN_IRRADIANCE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_sun_intensity_is_legacy_directional_irradiance() {
        let light = DirectionalLight::default();

        assert!(
            (light.intensity - DEFAULT_SUN_IRRADIANCE).length() < 1.0e-6,
            "DirectionalLight::intensity should be total sun irradiance/legacy directional strength, got {:?}",
            light.intensity
        );
        assert!(
            light.intensity.max_element() < 10.0,
            "default sun intensity must not be pre-scaled solar-disk radiance"
        );
    }
}
