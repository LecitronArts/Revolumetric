use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AreaRestirDebugView {
    Off,
    Subpixel,
    Lens,
    Weight,
    HistoryValid,
    Rejection,
    Jacobian,
}

impl AreaRestirDebugView {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Subpixel => 1,
            Self::Lens => 2,
            Self::Weight => 3,
            Self::HistoryValid => 4,
            Self::Rejection => 5,
            Self::Jacobian => 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaRestirSettings {
    pub enabled: bool,
    pub temporal_enabled: bool,
    pub spatial_enabled: bool,
    pub subpixel_enabled: bool,
    pub lens_enabled: bool,
    pub initial_candidate_count: u32,
    pub spatial_sample_count: u32,
    pub history_length: u32,
    pub normal_threshold: f32,
    pub depth_threshold: f32,
    pub spatial_radius: f32,
    pub debug_view: AreaRestirDebugView,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AreaRestirParseWarning {
    pub variable: &'static str,
    pub expected: &'static str,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AreaRestirSettingsParse {
    pub settings: AreaRestirSettings,
    pub warnings: Vec<AreaRestirParseWarning>,
}

impl Default for AreaRestirSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            temporal_enabled: true,
            spatial_enabled: true,
            subpixel_enabled: true,
            lens_enabled: true,
            initial_candidate_count: 1,
            spatial_sample_count: 1,
            history_length: 20,
            normal_threshold: 0.85,
            depth_threshold: 0.02,
            spatial_radius: 24.0,
            debug_view: AreaRestirDebugView::Off,
        }
    }
}

impl AreaRestirSettings {
    pub fn from_env() -> AreaRestirSettingsParse {
        Self::from_values(
            std::env::var("REVOLUMETRIC_AREA_RESTIR").ok().as_deref(),
            std::env::var("REVOLUMETRIC_AREA_RESTIR_TEMPORAL")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_AREA_RESTIR_SPATIAL")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_AREA_RESTIR_SUBPIXEL")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_AREA_RESTIR_LENS")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_AREA_RESTIR_INITIAL_CANDIDATES")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_AREA_RESTIR_SPATIAL_SAMPLES")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_AREA_RESTIR_HISTORY_LENGTH")
                .ok()
                .as_deref(),
            std::env::var("REVOLUMETRIC_AREA_RESTIR_DEBUG")
                .ok()
                .as_deref(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_values(
        enabled: Option<&str>,
        temporal: Option<&str>,
        spatial: Option<&str>,
        subpixel: Option<&str>,
        lens: Option<&str>,
        initial_candidates: Option<&str>,
        spatial_samples: Option<&str>,
        history_length: Option<&str>,
        debug_view: Option<&str>,
    ) -> AreaRestirSettingsParse {
        let mut settings = Self::default();
        let mut warnings = Vec::new();

        parse_bool(
            "REVOLUMETRIC_AREA_RESTIR",
            enabled,
            &mut settings.enabled,
            &mut warnings,
        );
        parse_bool(
            "REVOLUMETRIC_AREA_RESTIR_TEMPORAL",
            temporal,
            &mut settings.temporal_enabled,
            &mut warnings,
        );
        parse_bool(
            "REVOLUMETRIC_AREA_RESTIR_SPATIAL",
            spatial,
            &mut settings.spatial_enabled,
            &mut warnings,
        );
        parse_bool(
            "REVOLUMETRIC_AREA_RESTIR_SUBPIXEL",
            subpixel,
            &mut settings.subpixel_enabled,
            &mut warnings,
        );
        parse_bool(
            "REVOLUMETRIC_AREA_RESTIR_LENS",
            lens,
            &mut settings.lens_enabled,
            &mut warnings,
        );
        parse_u32_range(
            "REVOLUMETRIC_AREA_RESTIR_INITIAL_CANDIDATES",
            initial_candidates,
            1,
            16,
            &mut settings.initial_candidate_count,
            &mut warnings,
        );
        parse_u32_range(
            "REVOLUMETRIC_AREA_RESTIR_SPATIAL_SAMPLES",
            spatial_samples,
            0,
            16,
            &mut settings.spatial_sample_count,
            &mut warnings,
        );
        parse_u32_range(
            "REVOLUMETRIC_AREA_RESTIR_HISTORY_LENGTH",
            history_length,
            1,
            64,
            &mut settings.history_length,
            &mut warnings,
        );
        parse_debug_view(debug_view, &mut settings.debug_view, &mut warnings);

        AreaRestirSettingsParse { settings, warnings }
    }

    pub fn gpu_uniforms(
        self,
        frame_index: u32,
        reservoir_count: u32,
        width: u32,
        height: u32,
    ) -> GpuAreaRestirUniforms {
        GpuAreaRestirUniforms {
            enabled: self.enabled as u32,
            temporal_enabled: self.temporal_enabled as u32,
            spatial_enabled: self.spatial_enabled as u32,
            subpixel_enabled: self.subpixel_enabled as u32,
            lens_enabled: self.lens_enabled as u32,
            initial_candidate_count: self.initial_candidate_count,
            spatial_sample_count: self.spatial_sample_count,
            history_length: self.history_length,
            frame_index,
            reservoir_count,
            width,
            height,
            normal_threshold: self.normal_threshold,
            depth_threshold: self.depth_threshold,
            spatial_radius: self.spatial_radius,
            debug_view: self.debug_view.as_gpu_value(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuAreaRestirUniforms {
    pub enabled: u32,
    pub temporal_enabled: u32,
    pub spatial_enabled: u32,
    pub subpixel_enabled: u32,
    pub lens_enabled: u32,
    pub initial_candidate_count: u32,
    pub spatial_sample_count: u32,
    pub history_length: u32,
    pub frame_index: u32,
    pub reservoir_count: u32,
    pub width: u32,
    pub height: u32,
    pub normal_threshold: f32,
    pub depth_threshold: f32,
    pub spatial_radius: f32,
    pub debug_view: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuAreaRestirSampleState {
    pub subpixel_uv: [f32; 2],
    pub lens_uv: [f32; 2],
    pub pixel_sample: [f32; 2],
    pub path_sample: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuAreaRestirEvalContext {
    pub position_depth: [f32; 4],
    pub normal_roughness: [f32; 4],
    pub albedo_material: [f32; 4],
    pub motion_history: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuAreaRestirReservoir {
    pub sample_state: GpuAreaRestirSampleState,
    pub sample_count_m: u32,
    pub pad0: u32,
    pub weight_sum: f32,
    pub target_pdf: f32,
    pub selected_weight: f32,
    pub confidence: f32,
    pub jacobian: f32,
    pub contribution_luma: f32,
    pub rejection_reason: u32,
    pub debug_flags: u32,
    pub pad1: [u32; 2],
    pub selected_radiance: [f32; 4],
}

fn parse_bool(
    variable: &'static str,
    value: Option<&str>,
    target: &mut bool,
    warnings: &mut Vec<AreaRestirParseWarning>,
) {
    let Some(value) = value else {
        return;
    };
    if matches!(value.trim(), "1" | "on" | "true" | "yes") {
        *target = true;
    } else if matches!(value.trim(), "0" | "off" | "false" | "no") {
        *target = false;
    } else {
        warnings.push(AreaRestirParseWarning {
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
    warnings: &mut Vec<AreaRestirParseWarning>,
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
        None => warnings.push(AreaRestirParseWarning {
            variable,
            expected: "integer in configured range",
            value: value.to_owned(),
        }),
    }
}

fn parse_debug_view(
    value: Option<&str>,
    target: &mut AreaRestirDebugView,
    warnings: &mut Vec<AreaRestirParseWarning>,
) {
    let Some(value) = value else {
        return;
    };
    let trimmed = value.trim();
    *target = if trimmed.eq_ignore_ascii_case("off") {
        AreaRestirDebugView::Off
    } else if trimmed.eq_ignore_ascii_case("subpixel") {
        AreaRestirDebugView::Subpixel
    } else if trimmed.eq_ignore_ascii_case("lens") {
        AreaRestirDebugView::Lens
    } else if trimmed.eq_ignore_ascii_case("weight") {
        AreaRestirDebugView::Weight
    } else if trimmed.eq_ignore_ascii_case("history_valid") {
        AreaRestirDebugView::HistoryValid
    } else if trimmed.eq_ignore_ascii_case("rejection") {
        AreaRestirDebugView::Rejection
    } else if trimmed.eq_ignore_ascii_case("jacobian") {
        AreaRestirDebugView::Jacobian
    } else {
        warnings.push(AreaRestirParseWarning {
            variable: "REVOLUMETRIC_AREA_RESTIR_DEBUG",
            expected: "off|subpixel|lens|weight|history_valid|rejection|jacobian",
            value: value.to_owned(),
        });
        *target
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_defaults_keep_area_restir_disabled() {
        let settings = AreaRestirSettings::default();

        assert!(!settings.enabled);
        assert!(settings.temporal_enabled);
        assert!(settings.spatial_enabled);
        assert!(settings.subpixel_enabled);
        assert!(settings.lens_enabled);
        assert_eq!(settings.initial_candidate_count, 1);
        assert_eq!(settings.spatial_sample_count, 1);
        assert_eq!(settings.history_length, 20);
        assert_eq!(settings.debug_view, AreaRestirDebugView::Off);
    }

    #[test]
    fn settings_parse_valid_overrides() {
        let parsed = AreaRestirSettings::from_values(
            Some("on"),
            Some("off"),
            Some("true"),
            Some("yes"),
            Some("0"),
            Some("8"),
            Some("6"),
            Some("32"),
            Some("history_valid"),
        );

        assert!(parsed.settings.enabled);
        assert!(!parsed.settings.temporal_enabled);
        assert!(parsed.settings.spatial_enabled);
        assert!(parsed.settings.subpixel_enabled);
        assert!(!parsed.settings.lens_enabled);
        assert_eq!(parsed.settings.initial_candidate_count, 8);
        assert_eq!(parsed.settings.spatial_sample_count, 6);
        assert_eq!(parsed.settings.history_length, 32);
        assert_eq!(
            parsed.settings.debug_view,
            AreaRestirDebugView::HistoryValid
        );
        assert!(parsed.warnings.is_empty());
    }

    #[test]
    fn settings_reject_invalid_values_without_changing_defaults() {
        let parsed = AreaRestirSettings::from_values(
            Some("maybe"),
            Some("later"),
            Some("sometimes"),
            Some("subpixel"),
            Some("lens"),
            Some("0"),
            Some("99"),
            Some("0"),
            Some("beauty"),
        );

        assert_eq!(parsed.settings, AreaRestirSettings::default());
        assert_eq!(parsed.warnings.len(), 9);
    }

    #[test]
    fn gpu_area_restir_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuAreaRestirUniforms>(), 64);
        assert_eq!(std::mem::size_of::<GpuAreaRestirSampleState>(), 32);
        assert_eq!(std::mem::size_of::<GpuAreaRestirEvalContext>(), 64);
        assert_eq!(std::mem::size_of::<GpuAreaRestirReservoir>(), 96);
        assert_eq!(std::mem::offset_of!(GpuAreaRestirUniforms, enabled), 0);
        assert_eq!(
            std::mem::offset_of!(GpuAreaRestirUniforms, spatial_radius),
            56
        );
        assert_eq!(
            std::mem::offset_of!(GpuAreaRestirSampleState, subpixel_uv),
            0
        );
        assert_eq!(std::mem::offset_of!(GpuAreaRestirSampleState, lens_uv), 8);
        assert_eq!(
            std::mem::offset_of!(GpuAreaRestirReservoir, selected_radiance),
            80
        );
    }

    #[test]
    fn slang_area_restir_common_declares_matching_abi_without_restir_di_coupling() {
        let source = std::fs::read_to_string("assets/shaders/shared/area_restir_common.slang")
            .expect("area_restir_common.slang should be readable");

        for token in [
            "struct AreaRestirUniforms",
            "uint subpixel_enabled",
            "uint lens_enabled",
            "struct AreaRestirSampleState",
            "float2 subpixel_uv",
            "float2 lens_uv",
            "struct AreaRestirEvalContext",
            "struct AreaRestirReservoir",
            "area_restir_invalid_reservoir",
            "area_restir_is_valid_reservoir",
            "area_restir_surface_compatible",
            "area_restir_finalize_reservoir",
        ] {
            assert!(
                source.contains(token),
                "area_restir_common.slang missing token {token}"
            );
        }
        assert!(!source.contains("restir_di_common.slang"));
        assert!(!source.contains("RestirDiReservoir"));
    }
}
