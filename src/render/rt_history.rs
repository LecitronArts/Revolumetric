use bytemuck::{Pod, Zeroable};

pub const RT_HISTORY_FLAG_CAMERA_CUT: u32 = 1 << 0;
pub const RT_HISTORY_FLAG_RESIZE: u32 = 1 << 1;
pub const RT_HISTORY_FLAG_SCENE_INVALIDATED: u32 = 1 << 2;
pub const RT_HISTORY_FLAG_AS_REBUILT: u32 = 1 << 3;
pub const RT_HISTORY_FLAG_LIGHTS_INVALIDATED: u32 = 1 << 4;
pub const RT_HISTORY_RESET_FLAGS: u32 = RT_HISTORY_FLAG_CAMERA_CUT
    | RT_HISTORY_FLAG_RESIZE
    | RT_HISTORY_FLAG_SCENE_INVALIDATED
    | RT_HISTORY_FLAG_AS_REBUILT
    | RT_HISTORY_FLAG_LIGHTS_INVALIDATED;

pub fn should_reset_rt_history(flags: u32) -> bool {
    flags & RT_HISTORY_RESET_FLAGS != 0
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRtHistoryUniforms {
    pub current_view_proj: [[f32; 4]; 4],
    pub previous_view_proj: [[f32; 4]; 4],
    pub current_resolution: [u32; 2],
    pub previous_resolution: [u32; 2],
    pub current_jitter: [f32; 2],
    pub previous_jitter: [f32; 2],
    pub frame_index: u32,
    pub history_reset_generation: u32,
    pub as_rebuild_generation: u32,
    pub scene_generation: u32,
    pub lights_generation: u32,
    pub temporal_denoise_enabled: u32,
    pub flags: u32,
    pub debug_view: u32,
    pub history_length: u32,
    pub normal_threshold: f32,
    pub depth_threshold: f32,
    pub _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuRtSurfacePixel {
    pub position_depth: [f32; 4],
    pub normal_roughness: [f32; 4],
    pub albedo_material: [f32; 4],
    pub motion_history: [f32; 4],
    pub view_direction_background: [f32; 4],
    pub linear_depth: f32,
    pub material_id: u32,
    pub history_confidence: f32,
    pub hit_kind: u32,
    pub brick_id: u32,
    pub local: [u32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_history_uniforms_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuRtHistoryUniforms>(), 208);
        assert_eq!(
            std::mem::offset_of!(GpuRtHistoryUniforms, current_view_proj),
            0
        );
        assert_eq!(std::mem::offset_of!(GpuRtHistoryUniforms, frame_index), 160);
        assert_eq!(std::mem::offset_of!(GpuRtHistoryUniforms, flags), 184);
        assert_eq!(
            std::mem::offset_of!(GpuRtHistoryUniforms, temporal_denoise_enabled),
            180
        );
        assert_eq!(std::mem::offset_of!(GpuRtHistoryUniforms, debug_view), 188);
        assert_eq!(
            std::mem::offset_of!(GpuRtHistoryUniforms, history_length),
            192
        );
        assert_eq!(
            std::mem::offset_of!(GpuRtHistoryUniforms, depth_threshold),
            200
        );
    }

    #[test]
    fn rt_history_flags_are_non_overlapping() {
        let all = RT_HISTORY_FLAG_CAMERA_CUT
            | RT_HISTORY_FLAG_RESIZE
            | RT_HISTORY_FLAG_SCENE_INVALIDATED
            | RT_HISTORY_FLAG_AS_REBUILT
            | RT_HISTORY_FLAG_LIGHTS_INVALIDATED;

        assert_eq!(all.count_ones(), 5);
    }

    #[test]
    fn rt_temporal_invalidates_on_camera_cut_resize_scene_change_and_as_rebuild() {
        for flag in [
            RT_HISTORY_FLAG_CAMERA_CUT,
            RT_HISTORY_FLAG_RESIZE,
            RT_HISTORY_FLAG_SCENE_INVALIDATED,
            RT_HISTORY_FLAG_AS_REBUILT,
            RT_HISTORY_FLAG_LIGHTS_INVALIDATED,
        ] {
            assert!(
                should_reset_rt_history(flag),
                "RT history flag {flag:#x} must reset temporal accumulation"
            );
        }

        assert!(!should_reset_rt_history(0));
    }

    #[test]
    fn rt_surface_pixel_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuRtSurfacePixel>(), 112);
        assert_eq!(std::mem::offset_of!(GpuRtSurfacePixel, position_depth), 0);
        assert_eq!(
            std::mem::offset_of!(GpuRtSurfacePixel, view_direction_background),
            64
        );
        assert_eq!(
            std::mem::offset_of!(GpuRtSurfacePixel, history_confidence),
            88
        );
        assert_eq!(std::mem::offset_of!(GpuRtSurfacePixel, hit_kind), 92);
        assert_eq!(std::mem::offset_of!(GpuRtSurfacePixel, brick_id), 96);
        assert_eq!(std::mem::offset_of!(GpuRtSurfacePixel, local), 100);
    }

    #[test]
    fn rt_history_common_shader_declares_matching_abi() {
        let source = std::fs::read_to_string("assets/shaders/shared/rt_history_common.slang")
            .expect("RT history common shader should be readable");

        for token in [
            "struct RtHistoryUniforms",
            "float4x4 current_view_proj",
            "float4x4 previous_view_proj",
            "uint2 current_resolution",
            "uint2 previous_resolution",
            "float2 current_jitter",
            "float2 previous_jitter",
            "uint frame_index",
            "uint history_reset_generation",
            "uint as_rebuild_generation",
            "uint temporal_denoise_enabled",
            "uint flags",
            "uint debug_view",
            "uint history_length",
            "float normal_threshold",
            "float depth_threshold",
            "struct RtSurfacePixel",
            "float4 view_direction_background",
            "float linear_depth",
            "uint material_id",
            "float history_confidence",
            "uint hit_kind",
            "uint brick_id",
            "uint3 local",
        ] {
            assert!(
                source.contains(token),
                "RT history shader missing token {token}"
            );
        }
    }
}
