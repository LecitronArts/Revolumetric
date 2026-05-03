use bytemuck::{Pod, Zeroable};

pub const VPT_HISTORY_FLAG_CAMERA_CUT: u32 = 1 << 0;
pub const VPT_HISTORY_FLAG_RESIZE: u32 = 1 << 1;
pub const VPT_HISTORY_FLAG_SCENE_INVALIDATED: u32 = 1 << 2;
pub const VPT_HISTORY_FLAG_LIGHTS_INVALIDATED: u32 = 1 << 3;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVptHistoryUniforms {
    pub current_view_proj: [[f32; 4]; 4],
    pub previous_view_proj: [[f32; 4]; 4],
    pub current_resolution: [u32; 2],
    pub previous_resolution: [u32; 2],
    pub current_jitter: [f32; 2],
    pub previous_jitter: [f32; 2],
    pub frame_index: u32,
    pub history_reset_generation: u32,
    pub flags: u32,
    pub _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuVptSurfacePixel {
    pub position_depth: [f32; 4],
    pub normal_roughness: [f32; 4],
    pub albedo_material: [f32; 4],
    pub motion_history: [f32; 4],
    pub linear_depth: f32,
    pub material_id: u32,
    pub history_confidence: f32,
    pub _pad0: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vpt_history_uniforms_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuVptHistoryUniforms>(), 176);
        assert_eq!(
            std::mem::offset_of!(GpuVptHistoryUniforms, current_view_proj),
            0
        );
        assert_eq!(
            std::mem::offset_of!(GpuVptHistoryUniforms, previous_view_proj),
            64
        );
        assert_eq!(
            std::mem::offset_of!(GpuVptHistoryUniforms, current_resolution),
            128
        );
        assert_eq!(
            std::mem::offset_of!(GpuVptHistoryUniforms, previous_resolution),
            136
        );
        assert_eq!(
            std::mem::offset_of!(GpuVptHistoryUniforms, current_jitter),
            144
        );
        assert_eq!(
            std::mem::offset_of!(GpuVptHistoryUniforms, previous_jitter),
            152
        );
        assert_eq!(
            std::mem::offset_of!(GpuVptHistoryUniforms, frame_index),
            160
        );
        assert_eq!(
            std::mem::offset_of!(GpuVptHistoryUniforms, history_reset_generation),
            164
        );
        assert_eq!(std::mem::offset_of!(GpuVptHistoryUniforms, flags), 168);
    }

    #[test]
    fn vpt_surface_pixel_layout_is_stable() {
        assert_eq!(std::mem::size_of::<GpuVptSurfacePixel>(), 80);
        assert_eq!(std::mem::offset_of!(GpuVptSurfacePixel, position_depth), 0);
        assert_eq!(
            std::mem::offset_of!(GpuVptSurfacePixel, normal_roughness),
            16
        );
        assert_eq!(
            std::mem::offset_of!(GpuVptSurfacePixel, albedo_material),
            32
        );
        assert_eq!(std::mem::offset_of!(GpuVptSurfacePixel, motion_history), 48);
        assert_eq!(std::mem::offset_of!(GpuVptSurfacePixel, linear_depth), 64);
        assert_eq!(std::mem::offset_of!(GpuVptSurfacePixel, material_id), 68);
        assert_eq!(
            std::mem::offset_of!(GpuVptSurfacePixel, history_confidence),
            72
        );
    }

    #[test]
    fn vpt_history_flags_are_non_overlapping() {
        let all = VPT_HISTORY_FLAG_CAMERA_CUT
            | VPT_HISTORY_FLAG_RESIZE
            | VPT_HISTORY_FLAG_SCENE_INVALIDATED
            | VPT_HISTORY_FLAG_LIGHTS_INVALIDATED;

        assert_eq!(all.count_ones(), 4);
    }

    #[test]
    fn slang_vpt_history_common_declares_matching_abi() {
        let source = std::fs::read_to_string("assets/shaders/shared/vpt_history_common.slang")
            .expect("VPT history common shader should be readable");

        assert!(source.contains("struct VptHistoryUniforms"));
        assert!(source.contains("float4x4 current_view_proj"));
        assert!(source.contains("float4x4 previous_view_proj"));
        assert!(source.contains("uint2 current_resolution"));
        assert!(source.contains("uint2 previous_resolution"));
        assert!(source.contains("float2 current_jitter"));
        assert!(source.contains("float2 previous_jitter"));
        assert!(source.contains("uint history_reset_generation"));
        assert!(source.contains("struct VptSurfacePixel"));
        assert!(source.contains("float linear_depth"));
        assert!(source.contains("uint material_id"));
        assert!(source.contains("float history_confidence"));
    }
}
