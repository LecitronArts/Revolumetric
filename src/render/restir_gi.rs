use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestirGiDebugView {
    Off,
    ReservoirWeight,
    TemporalValid,
}

impl RestirGiDebugView {
    pub fn as_gpu_value(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::ReservoirWeight => 1,
            Self::TemporalValid => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestirGiSettings {
    pub enabled: bool,
    pub temporal_enabled: bool,
    pub initial_candidate_count: u32,
    pub history_length: u32,
    pub max_bounces: u32,
    pub debug_view: RestirGiDebugView,
}

impl Default for RestirGiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            temporal_enabled: false,
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
            pad0: [0; 2],
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
    pub pad0: [u32; 2],
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
        assert_eq!(settings.initial_candidate_count, 1);
        assert_eq!(settings.history_length, 20);
        assert_eq!(settings.max_bounces, 1);
        assert_eq!(settings.debug_view, RestirGiDebugView::Off);
    }

    #[test]
    fn restir_gi_gpu_uniforms_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuRestirGiUniforms>(), 48);
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
        ] {
            assert!(source.contains(token), "GI common shader missing {token}");
        }
    }
}
