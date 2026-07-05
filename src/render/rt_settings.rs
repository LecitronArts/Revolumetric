use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtDebugView {
    Off,
    Surface,
    HitDistance,
    HistoryValid,
    DirectReservoir,
    IndirectReservoir,
    Temporal,
}

impl RtDebugView {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Surface => 1,
            Self::HitDistance => 2,
            Self::HistoryValid => 3,
            Self::DirectReservoir => 4,
            Self::IndirectReservoir => 5,
            Self::Temporal => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RtSettings {
    pub restir_di_enabled: bool,
    pub restir_gi_enabled: bool,
    pub temporal_denoise_enabled: bool,
    pub restir_di_spatial_enabled: bool,
    pub restir_di_spatial_sample_count: u32,
    pub history_length: u32,
    pub normal_threshold: f32,
    pub depth_threshold: f32,
    pub debug_view: RtDebugView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtSettingsParseWarning {
    pub variable: &'static str,
    pub expected: &'static str,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RtSettingsParse {
    pub settings: RtSettings,
    pub warnings: Vec<RtSettingsParseWarning>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRtSettings {
    pub restir_di_enabled: u32,
    pub restir_gi_enabled: u32,
    pub temporal_denoise_enabled: u32,
    pub debug_view: u32,
    pub history_length: u32,
    pub normal_threshold: f32,
    pub depth_threshold: f32,
    pub restir_di_spatial_enabled: u32,
    pub restir_di_spatial_sample_count: u32,
    pub pad0: [u32; 3],
}

impl Default for RtSettings {
    fn default() -> Self {
        Self {
            restir_di_enabled: false,
            restir_gi_enabled: false,
            temporal_denoise_enabled: true,
            restir_di_spatial_enabled: false,
            restir_di_spatial_sample_count: 4,
            history_length: 20,
            normal_threshold: 0.85,
            depth_threshold: 0.02,
            debug_view: RtDebugView::Off,
        }
    }
}

impl RtSettings {
    pub fn from_env() -> RtSettingsParse {
        Self::from_values(
            std::env::var("REVOLUMETRIC_RT_RESTIR_DI").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RT_RESTIR_GI").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RT_TEMPORAL_DENOISE")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RT_TEMPORAL_HISTORY_LENGTH")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RT_TEMPORAL_NORMAL_THRESHOLD")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RT_TEMPORAL_DEPTH_THRESHOLD")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RT_DEBUG_VIEW").ok().as_deref(),
            std::env::var("REVOLUMETRIC_RT_RESTIR_DI_SPATIAL")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES")
                .ok()
                .as_deref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_values(
        restir_di_enabled: Option<&str>,
        restir_gi_enabled: Option<&str>,
        temporal_denoise_enabled: Option<&str>,
        history_length: Option<&str>,
        normal_threshold: Option<&str>,
        depth_threshold: Option<&str>,
        debug_view: Option<&str>,
        restir_di_spatial_enabled: Option<&str>,
        restir_di_spatial_samples: Option<&str>,
    ) -> RtSettingsParse {
        let mut settings = Self::default();
        let mut warnings = Vec::new();

        parse_bool(
            "REVOLUMETRIC_RT_RESTIR_DI",
            restir_di_enabled,
            &mut settings.restir_di_enabled,
            &mut warnings,
        );
        parse_bool(
            "REVOLUMETRIC_RT_RESTIR_GI",
            restir_gi_enabled,
            &mut settings.restir_gi_enabled,
            &mut warnings,
        );
        parse_bool(
            "REVOLUMETRIC_RT_TEMPORAL_DENOISE",
            temporal_denoise_enabled,
            &mut settings.temporal_denoise_enabled,
            &mut warnings,
        );
        parse_u32_range(
            "REVOLUMETRIC_RT_TEMPORAL_HISTORY_LENGTH",
            history_length,
            1,
            64,
            &mut settings.history_length,
            &mut warnings,
        );
        parse_f32_range(
            "REVOLUMETRIC_RT_TEMPORAL_NORMAL_THRESHOLD",
            normal_threshold,
            0.0,
            1.0,
            &mut settings.normal_threshold,
            &mut warnings,
        );
        parse_f32_range(
            "REVOLUMETRIC_RT_TEMPORAL_DEPTH_THRESHOLD",
            depth_threshold,
            0.0,
            1.0,
            &mut settings.depth_threshold,
            &mut warnings,
        );
        parse_debug_view(debug_view, &mut settings.debug_view, &mut warnings);
        parse_bool(
            "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL",
            restir_di_spatial_enabled,
            &mut settings.restir_di_spatial_enabled,
            &mut warnings,
        );
        parse_u32_range(
            "REVOLUMETRIC_RT_RESTIR_DI_SPATIAL_SAMPLES",
            restir_di_spatial_samples,
            0,
            8,
            &mut settings.restir_di_spatial_sample_count,
            &mut warnings,
        );

        RtSettingsParse { settings, warnings }
    }

    pub fn gpu_uniforms(self) -> GpuRtSettings {
        GpuRtSettings {
            restir_di_enabled: self.restir_di_enabled as u32,
            restir_gi_enabled: self.restir_gi_enabled as u32,
            temporal_denoise_enabled: self.temporal_denoise_enabled as u32,
            debug_view: self.debug_view.as_gpu_value(),
            history_length: self.history_length,
            normal_threshold: self.normal_threshold,
            depth_threshold: self.depth_threshold,
            restir_di_spatial_enabled: self.restir_di_spatial_enabled as u32,
            restir_di_spatial_sample_count: self.restir_di_spatial_sample_count,
            pad0: [0; 3],
        }
    }
}

fn parse_bool(
    variable: &'static str,
    value: Option<&str>,
    target: &mut bool,
    warnings: &mut Vec<RtSettingsParseWarning>,
) {
    let Some(value) = value else {
        return;
    };

    let trimmed = value.trim();
    if matches!(trimmed, "1")
        || trimmed.eq_ignore_ascii_case("on")
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
    {
        *target = true;
    } else if matches!(trimmed, "0")
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
    {
        *target = false;
    } else {
        warnings.push(RtSettingsParseWarning {
            variable,
            expected: "on|off|1|0|true|false|yes|no",
            value: value.to_owned(),
        });
    }
}

fn parse_u32_range(
    variable: &'static str,
    value: Option<&str>,
    min: u32,
    max: u32,
    target: &mut u32,
    warnings: &mut Vec<RtSettingsParseWarning>,
) {
    let Some(value) = value else {
        return;
    };

    match value
        .trim()
        .parse::<u32>()
        .ok()
        .filter(|parsed| (min..=max).contains(parsed))
    {
        Some(parsed) => *target = parsed,
        None => warnings.push(RtSettingsParseWarning {
            variable,
            expected: "integer in configured range",
            value: value.to_owned(),
        }),
    }
}

fn parse_f32_range(
    variable: &'static str,
    value: Option<&str>,
    min: f32,
    max: f32,
    target: &mut f32,
    warnings: &mut Vec<RtSettingsParseWarning>,
) {
    let Some(value) = value else {
        return;
    };

    match value
        .trim()
        .parse::<f32>()
        .ok()
        .filter(|parsed| parsed.is_finite() && *parsed >= min && *parsed <= max)
    {
        Some(parsed) => *target = parsed,
        None => warnings.push(RtSettingsParseWarning {
            variable,
            expected: "finite float in configured range",
            value: value.to_owned(),
        }),
    }
}

fn parse_debug_view(
    value: Option<&str>,
    target: &mut RtDebugView,
    warnings: &mut Vec<RtSettingsParseWarning>,
) {
    let Some(value) = value else {
        return;
    };

    let trimmed = value.trim();
    *target = if trimmed.eq_ignore_ascii_case("off") || trimmed.eq_ignore_ascii_case("final") {
        RtDebugView::Off
    } else if trimmed.eq_ignore_ascii_case("surface") {
        RtDebugView::Surface
    } else if trimmed.eq_ignore_ascii_case("hit_distance") {
        RtDebugView::HitDistance
    } else if trimmed.eq_ignore_ascii_case("history_valid") {
        RtDebugView::HistoryValid
    } else if trimmed.eq_ignore_ascii_case("direct_reservoir") {
        RtDebugView::DirectReservoir
    } else if trimmed.eq_ignore_ascii_case("indirect_reservoir") {
        RtDebugView::IndirectReservoir
    } else if trimmed.eq_ignore_ascii_case("temporal") {
        RtDebugView::Temporal
    } else {
        warnings.push(RtSettingsParseWarning {
            variable: "REVOLUMETRIC_RT_DEBUG_VIEW",
            expected: "off|final|surface|hit_distance|history_valid|direct_reservoir|indirect_reservoir|temporal",
            value: value.to_owned(),
        });
        *target
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_settings_defaults_keep_temporal_denoise_enabled_and_restir_disabled() {
        let settings = RtSettings::default();

        assert!(!settings.restir_di_enabled);
        assert!(!settings.restir_gi_enabled);
        assert!(settings.temporal_denoise_enabled);
        assert!(!settings.restir_di_spatial_enabled);
        assert_eq!(settings.restir_di_spatial_sample_count, 4);
        assert_eq!(settings.history_length, 20);
        assert_eq!(settings.normal_threshold, 0.85);
        assert_eq!(settings.depth_threshold, 0.02);
        assert_eq!(settings.debug_view, RtDebugView::Off);
    }

    #[test]
    fn rt_settings_parse_valid_overrides() {
        let parsed = RtSettings::from_values(
            Some("on"),
            Some("off"),
            Some("true"),
            Some("32"),
            Some("0.85"),
            Some("0.02"),
            Some("surface"),
            Some("off"),
            Some("4"),
        );

        assert!(parsed.settings.restir_di_enabled);
        assert!(!parsed.settings.restir_gi_enabled);
        assert!(parsed.settings.temporal_denoise_enabled);
        assert!(!parsed.settings.restir_di_spatial_enabled);
        assert_eq!(parsed.settings.restir_di_spatial_sample_count, 4);
        assert_eq!(parsed.settings.history_length, 32);
        assert_eq!(parsed.settings.normal_threshold, 0.85);
        assert_eq!(parsed.settings.depth_threshold, 0.02);
        assert_eq!(parsed.settings.debug_view, RtDebugView::Surface);
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn rt_settings_parse_rt_restir_di_spatial_overrides() {
        let parsed = RtSettings::from_values(
            Some("on"),
            Some("off"),
            Some("true"),
            Some("20"),
            Some("0.85"),
            Some("0.02"),
            Some("direct_reservoir"),
            Some("on"),
            Some("8"),
        );

        assert!(parsed.settings.restir_di_spatial_enabled);
        assert_eq!(parsed.settings.restir_di_spatial_sample_count, 8);
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn rt_settings_reject_invalid_values_without_changing_defaults() {
        let parsed = RtSettings::from_values(
            Some("maybe"),
            Some("later"),
            Some("perhaps"),
            Some("0"),
            Some("1.5"),
            Some("-0.01"),
            Some("heatmap"),
            Some("maybe"),
            Some("9"),
        );

        assert_eq!(parsed.settings, RtSettings::default());
        assert_eq!(parsed.warnings.len(), 9);
    }

    #[test]
    fn rt_gpu_uniforms_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuRtSettings>(), 48);
        assert_eq!(std::mem::offset_of!(GpuRtSettings, restir_di_enabled), 0);
        assert_eq!(std::mem::offset_of!(GpuRtSettings, debug_view), 12);
        assert_eq!(std::mem::offset_of!(GpuRtSettings, history_length), 16);
        assert_eq!(std::mem::offset_of!(GpuRtSettings, depth_threshold), 24);
        assert_eq!(
            std::mem::offset_of!(GpuRtSettings, restir_di_spatial_enabled),
            28
        );
        assert_eq!(
            std::mem::offset_of!(GpuRtSettings, restir_di_spatial_sample_count),
            32
        );

        let uniforms = RtSettings::default().gpu_uniforms();
        assert_eq!(uniforms.restir_di_enabled, 0);
        assert_eq!(uniforms.temporal_denoise_enabled, 1);
        assert_eq!(uniforms.restir_di_spatial_enabled, 0);
        assert_eq!(uniforms.restir_di_spatial_sample_count, 4);
        assert_eq!(uniforms.debug_view, RtDebugView::Off.as_gpu_value());
    }
}
