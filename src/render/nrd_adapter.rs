use std::fmt;
use std::os::raw::{c_char, c_void};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NrdLibraryDesc {
    pub texture_offset: u32,
    pub sampler_offset: u32,
    pub constant_buffer_offset: u32,
    pub storage_texture_and_buffer_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NrdTextureDesc {
    pub format: u32,
    pub downsample_factor: u16,
    pub reserved0: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NrdResourceDesc {
    pub descriptor_type: u32,
    pub resource_type: u32,
    pub index_in_pool: u16,
    pub reserved0: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NrdResourceRangeDesc {
    pub descriptor_type: u32,
    pub descriptors_num: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NrdSamplerDesc {
    pub mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NrdPipelineDesc {
    pub spirv_bytecode: *const u32,
    pub spirv_bytecode_size: u64,
    pub resource_ranges: *const NrdResourceRangeDesc,
    pub resource_ranges_num: u32,
    pub has_constant_data: u32,
    pub shader_identifier: [c_char; 256],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NrdInstanceDesc {
    pub constant_buffer_and_samplers_space_index: u32,
    pub resources_space_index: u32,
    pub constant_buffer_register_index: u32,
    pub samplers_base_register_index: u32,
    pub resources_base_register_index: u32,
    pub constant_buffer_max_data_size: u32,
    pub samplers: *const NrdSamplerDesc,
    pub samplers_num: u32,
    pub pipelines: *const NrdPipelineDesc,
    pub pipelines_num: u32,
    pub permanent_pool: *const NrdTextureDesc,
    pub permanent_pool_size: u32,
    pub transient_pool: *const NrdTextureDesc,
    pub transient_pool_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NrdDispatchDesc {
    pub name: *const c_char,
    pub identifier: u32,
    pub resources: *const NrdResourceDesc,
    pub resources_num: u32,
    pub constant_buffer_data: *const u8,
    pub constant_buffer_data_size: u32,
    pub constant_buffer_data_matches_previous_dispatch: u32,
    pub pipeline_index: u16,
    pub grid_width: u16,
    pub grid_height: u16,
    pub reserved0: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NrdCommonSettings {
    pub view_to_clip_matrix: [f32; 16],
    pub view_to_clip_matrix_prev: [f32; 16],
    pub world_to_view_matrix: [f32; 16],
    pub world_to_view_matrix_prev: [f32; 16],
    pub camera_jitter: [f32; 2],
    pub camera_jitter_prev: [f32; 2],
    pub motion_vector_scale: [f32; 3],
    pub resource_size: [u16; 2],
    pub resource_size_prev: [u16; 2],
    pub rect_size: [u16; 2],
    pub rect_size_prev: [u16; 2],
    pub denoising_range: f32,
    pub disocclusion_threshold: f32,
    pub disocclusion_threshold_alternate: f32,
    pub split_screen: f32,
    pub time_delta_between_frames: f32,
    pub view_z_scale: f32,
    pub frame_index: u32,
    pub accumulation_mode: u32,
    pub is_motion_vector_in_world_space: u32,
    pub is_history_confidence_available: u32,
    pub is_disocclusion_threshold_mix_available: u32,
    pub enable_validation: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NrdRelaxDiffuseSettings {
    pub antilag_acceleration_amount: f32,
    pub antilag_spatial_sigma_scale: f32,
    pub antilag_temporal_sigma_scale: f32,
    pub antilag_reset_amount: f32,
    pub diffuse_max_accumulated_frame_num: u32,
    pub diffuse_max_fast_accumulated_frame_num: u32,
    pub history_fix_frame_num: u32,
    pub history_fix_base_pixel_stride: u32,
    pub history_fix_alternate_pixel_stride: u32,
    pub history_fix_edge_stopping_normal_power: f32,
    pub fast_history_clamping_sigma_scale: f32,
    pub diffuse_prepass_blur_radius: f32,
    pub min_hit_distance_weight: f32,
    pub spatial_variance_estimation_history_threshold: u32,
    pub diffuse_phi_luminance: f32,
    pub atrous_iteration_num: u32,
    pub diffuse_min_luminance_weight: f32,
    pub depth_threshold: f32,
    pub confidence_driven_relaxation_multiplier: f32,
    pub confidence_driven_luminance_edge_stopping_relaxation: f32,
    pub confidence_driven_normal_edge_stopping_relaxation: f32,
    pub luminance_edge_stopping_relaxation: f32,
    pub normal_edge_stopping_relaxation: f32,
    pub roughness_edge_stopping_relaxation: f32,
    pub checkerboard_mode: u32,
    pub hit_distance_reconstruction_mode: u32,
    pub min_material_for_diffuse: f32,
    pub enable_anti_firefly: u32,
    pub enable_roughness_edge_stopping: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NrdUnavailableError {
    reason: &'static str,
}

impl NrdUnavailableError {
    pub const fn new(reason: &'static str) -> Self {
        Self { reason }
    }
}

impl fmt::Display for NrdUnavailableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.reason)
    }
}

impl std::error::Error for NrdUnavailableError {}

pub type NrdResult<T> = Result<T, NrdUnavailableError>;

#[derive(Debug)]
pub struct NrdInstance {
    #[cfg(feature = "nrd")]
    ptr: std::ptr::NonNull<RevolumetricNrdInstance>,
}

impl NrdInstance {
    #[cfg(not(feature = "nrd"))]
    pub fn relax_diffuse(_width: u32, _height: u32) -> NrdResult<Self> {
        Err(NrdUnavailableError::new(
            "NRD is unavailable because the nrd Cargo feature is disabled",
        ))
    }

    #[cfg(feature = "nrd")]
    pub fn relax_diffuse(width: u32, height: u32) -> NrdResult<Self> {
        let mut raw = std::ptr::null_mut();
        let status = unsafe { revolumetric_nrd_create_relax_diffuse(width, height, &mut raw) };
        if status != REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to create NRD RELAX_DIFFUSE instance",
            ));
        }
        let ptr = std::ptr::NonNull::new(raw).ok_or_else(|| {
            NrdUnavailableError::new("NRD returned a null RELAX_DIFFUSE instance")
        })?;
        Ok(Self { ptr })
    }
}

#[cfg(feature = "nrd")]
impl Drop for NrdInstance {
    fn drop(&mut self) {
        unsafe { revolumetric_nrd_destroy(self.ptr.as_ptr()) };
    }
}

#[cfg(feature = "nrd")]
#[repr(C)]
pub struct RevolumetricNrdInstance {
    _private: [u8; 0],
}

#[cfg(feature = "nrd")]
const REVOLUMETRIC_NRD_STATUS_OK: u32 = 0;

#[cfg(feature = "nrd")]
unsafe extern "C" {
    fn revolumetric_nrd_create_relax_diffuse(
        width: u32,
        height: u32,
        out_instance: *mut *mut RevolumetricNrdInstance,
    ) -> u32;
    fn revolumetric_nrd_destroy(instance: *mut RevolumetricNrdInstance);
    fn revolumetric_nrd_get_library_desc(out_desc: *mut NrdLibraryDesc) -> u32;
    fn revolumetric_nrd_get_instance_desc(
        instance: *const RevolumetricNrdInstance,
        out_desc: *mut NrdInstanceDesc,
    ) -> u32;
    fn revolumetric_nrd_set_common_settings(
        instance: *mut RevolumetricNrdInstance,
        settings: *const NrdCommonSettings,
    ) -> u32;
    fn revolumetric_nrd_set_relax_diffuse_settings(
        instance: *mut RevolumetricNrdInstance,
        settings: *const NrdRelaxDiffuseSettings,
    ) -> u32;
    fn revolumetric_nrd_get_dispatches(
        instance: *mut RevolumetricNrdInstance,
        out_dispatches: *mut *const NrdDispatchDesc,
        out_dispatches_num: *mut u32,
    ) -> u32;
}

#[allow(dead_code)]
type OpaqueVoid = c_void;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_descriptor_layouts_are_pod_c_abi_shapes() {
        assert_eq!(std::mem::size_of::<NrdLibraryDesc>(), 16);
        assert_eq!(std::mem::size_of::<NrdTextureDesc>(), 8);
        assert_eq!(std::mem::size_of::<NrdResourceDesc>(), 12);
        assert_eq!(std::mem::size_of::<NrdResourceRangeDesc>(), 8);
        assert_eq!(std::mem::size_of::<NrdSamplerDesc>(), 4);
        assert_eq!(std::mem::offset_of!(NrdPipelineDesc, shader_identifier), 32);
        assert_eq!(std::mem::offset_of!(NrdInstanceDesc, samplers), 24);
        assert_eq!(std::mem::offset_of!(NrdDispatchDesc, pipeline_index), 48);
        assert_eq!(std::mem::size_of::<NrdDispatchDesc>(), 56);
    }

    #[test]
    fn default_build_reports_nrd_unavailable() {
        #[cfg(not(feature = "nrd"))]
        {
            let error = NrdInstance::relax_diffuse(1, 1).unwrap_err();
            assert!(error.to_string().contains("nrd Cargo feature is disabled"));
        }
    }
}
