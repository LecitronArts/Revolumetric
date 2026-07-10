use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestirGiDebugView {
    Off,
    ReservoirWeight,
    TemporalValid,
    SpatialValid,
}

impl RestirGiDebugView {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::ReservoirWeight => 1,
            Self::TemporalValid => 2,
            Self::SpatialValid => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestirGiSettings {
    pub enabled: bool,
    pub temporal_enabled: bool,
    pub spatial_enabled: bool,
    pub spatial_sample_count: u32,
    pub initial_candidate_count: u32,
    pub history_length: u32,
    pub max_bounces: u32,
    pub debug_view: RestirGiDebugView,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RestirGiLightingUniformInputs {
    pub sun_direction: glam::Vec3,
    pub sun_intensity: glam::Vec3,
    pub sun_angular_radius: f32,
}

impl Default for RestirGiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            temporal_enabled: false,
            spatial_enabled: false,
            spatial_sample_count: 0,
            initial_candidate_count: 1,
            history_length: 20,
            max_bounces: 1,
            debug_view: RestirGiDebugView::Off,
        }
    }
}

impl RestirGiSettings {
    pub fn gpu_uniforms(
        self,
        frame_index: u32,
        reservoir_count: u32,
        width: u32,
        height: u32,
        lighting: RestirGiLightingUniformInputs,
    ) -> GpuRestirGiUniforms {
        GpuRestirGiUniforms {
            enabled: self.enabled as u32,
            temporal_enabled: self.temporal_enabled as u32,
            max_bounces: self.max_bounces,
            debug_view: self.debug_view.as_gpu_value(),
            initial_candidate_count: self.initial_candidate_count,
            history_length: self.history_length,
            frame_index,
            reservoir_count,
            width,
            height,
            spatial_enabled: self.spatial_enabled as u32,
            spatial_sample_count: self.spatial_sample_count,
            sky_color_sun_angular_radius: [0.4, 0.5, 0.7, lighting.sun_angular_radius],
            ground_color_pad: [0.15, 0.1, 0.08, 0.0],
            sun_direction_pad: [
                lighting.sun_direction.x,
                lighting.sun_direction.y,
                lighting.sun_direction.z,
                0.0,
            ],
            sun_intensity_pad: [
                lighting.sun_intensity.x,
                lighting.sun_intensity.y,
                lighting.sun_intensity.z,
                0.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRestirGiUniforms {
    pub enabled: u32,
    pub temporal_enabled: u32,
    pub max_bounces: u32,
    pub debug_view: u32,
    pub initial_candidate_count: u32,
    pub history_length: u32,
    pub frame_index: u32,
    pub reservoir_count: u32,
    pub width: u32,
    pub height: u32,
    pub spatial_enabled: u32,
    pub spatial_sample_count: u32,
    pub sky_color_sun_angular_radius: [f32; 4],
    pub ground_color_pad: [f32; 4],
    pub sun_direction_pad: [f32; 4],
    pub sun_intensity_pad: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRestirGiReservoir {
    pub sample_position_depth: [f32; 4],
    pub sample_normal_roughness: [f32; 4],
    pub sample_radiance_pdf: [f32; 4],
    pub target_pdf: f32,
    pub weight_sum: f32,
    pub selected_weight: f32,
    pub sample_count_m: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restir_gi_settings_default_to_conservative_values() {
        let settings = RestirGiSettings::default();

        assert!(!settings.enabled);
        assert!(!settings.temporal_enabled);
        assert!(!settings.spatial_enabled);
        assert_eq!(settings.spatial_sample_count, 0);
        assert_eq!(settings.initial_candidate_count, 1);
        assert_eq!(settings.history_length, 20);
        assert_eq!(settings.max_bounces, 1);
        assert_eq!(settings.debug_view, RestirGiDebugView::Off);
    }

    #[test]
    fn restir_gi_gpu_uniforms_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuRestirGiUniforms>(), 112);
        assert_eq!(std::mem::offset_of!(GpuRestirGiUniforms, enabled), 0);
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiUniforms, initial_candidate_count),
            16
        );
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiUniforms, reservoir_count),
            28
        );
        assert_eq!(std::mem::offset_of!(GpuRestirGiUniforms, height), 36);
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiUniforms, spatial_enabled),
            40
        );
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiUniforms, spatial_sample_count),
            44
        );
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiUniforms, sky_color_sun_angular_radius),
            48
        );
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiUniforms, ground_color_pad),
            64
        );
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiUniforms, sun_direction_pad),
            80
        );
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiUniforms, sun_intensity_pad),
            96
        );
    }

    #[test]
    fn restir_gi_reservoir_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuRestirGiReservoir>(), 64);
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiReservoir, sample_position_depth),
            0
        );
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiReservoir, sample_normal_roughness),
            16
        );
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiReservoir, sample_radiance_pdf),
            32
        );
        assert_eq!(
            std::mem::offset_of!(GpuRestirGiReservoir, sample_count_m),
            60
        );
    }

    #[test]
    fn slang_restir_gi_common_declares_matching_abi() {
        let source = std::fs::read_to_string("assets/shaders/shared/restir_gi_common.slang")
            .expect("restir_gi_common.slang should be readable");

        for token in [
            "struct RestirGiUniforms",
            "uint enabled;",
            "uint temporal_enabled;",
            "uint max_bounces;",
            "struct RestirGiReservoir",
            "float4 sample_position_depth;",
            "float4 sample_normal_roughness;",
            "float4 sample_radiance_pdf;",
            "uint sample_count_m;",
            "float4 sky_color_sun_angular_radius;",
            "float4 ground_color_pad;",
            "float4 sun_direction_pad;",
            "float4 sun_intensity_pad;",
        ] {
            assert!(source.contains(token), "GI common shader missing {token}");
        }
    }

    #[test]
    fn restir_gi_common_exposes_spatial_reuse_abi() {
        let rust_source =
            std::fs::read_to_string("src/render/restir_gi.rs").expect("restir_gi.rs should exist");
        let shader_source = std::fs::read_to_string("assets/shaders/shared/restir_gi_common.slang")
            .expect("restir_gi_common.slang should be readable");

        for token in [
            "pub spatial_enabled: bool",
            "pub spatial_sample_count: u32",
            "spatial_enabled: false",
            "spatial_sample_count: 0",
            "pub spatial_enabled: u32",
            "pub spatial_sample_count: u32",
        ] {
            assert!(
                rust_source.contains(token),
                "RT ReSTIR-GI Rust ABI missing {token}"
            );
        }

        for token in ["uint spatial_enabled;", "uint spatial_sample_count;"] {
            assert!(
                shader_source.contains(token),
                "RT ReSTIR-GI shader ABI missing {token}"
            );
        }
    }
}
