use std::os::raw::c_char;

pub type RevolumetricNrdStatus = u32;

pub const REVOLUMETRIC_NRD_STATUS_OK: RevolumetricNrdStatus = 0;
pub const REVOLUMETRIC_NRD_STATUS_INVALID_ARGUMENT: RevolumetricNrdStatus = 1;
pub const REVOLUMETRIC_NRD_STATUS_SDK_ERROR: RevolumetricNrdStatus = 2;
pub const REVOLUMETRIC_NRD_STATUS_INSUFFICIENT_CAPACITY: RevolumetricNrdStatus = 3;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdTextureFormat {
    Unknown = 0,
    R8Unorm = 1,
    R8Snorm = 2,
    R8Uint = 3,
    R8Sint = 4,
    Rg8Unorm = 5,
    Rg8Snorm = 6,
    Rg8Uint = 7,
    Rg8Sint = 8,
    Rgba8Unorm = 9,
    Rgba8Snorm = 10,
    Rgba8Uint = 11,
    Rgba8Sint = 12,
    Rgba8Srgb = 13,
    R16Unorm = 14,
    R16Snorm = 15,
    R16Uint = 16,
    R16Sint = 17,
    R16Sfloat = 18,
    Rg16Unorm = 19,
    Rg16Snorm = 20,
    Rg16Uint = 21,
    Rg16Sint = 22,
    Rg16Sfloat = 23,
    Rgba16Unorm = 24,
    Rgba16Snorm = 25,
    Rgba16Uint = 26,
    Rgba16Sint = 27,
    Rgba16Sfloat = 28,
    R32Uint = 29,
    R32Sint = 30,
    R32Sfloat = 31,
    Rg32Uint = 32,
    Rg32Sint = 33,
    Rg32Sfloat = 34,
    Rgb32Uint = 35,
    Rgb32Sint = 36,
    Rgb32Sfloat = 37,
    Rgba32Uint = 38,
    Rgba32Sint = 39,
    Rgba32Sfloat = 40,
    R10G10B10A2Unorm = 41,
    R11G11B10Ufloat = 42,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdDescriptorType {
    Unsupported = 0,
    Texture = 1,
    StorageTexture = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdResourceType {
    Unsupported = 0,
    InMv = 1,
    InNormalRoughness = 2,
    InViewZ = 3,
    InDiffConfidence = 4,
    InSpecConfidence = 5,
    InDisocclusionThresholdMix = 6,
    InDiffRadianceHitdist = 7,
    InSpecRadianceHitdist = 8,
    InDiffHitdist = 9,
    InSpecHitdist = 10,
    InDiffDirectionHitdist = 11,
    InDiffSh0 = 12,
    InDiffSh1 = 13,
    InSpecSh0 = 14,
    InSpecSh1 = 15,
    InPenumbra = 16,
    InTranslucency = 17,
    InSignal = 18,
    OutDiffRadianceHitdist = 19,
    OutSpecRadianceHitdist = 20,
    OutDiffSh0 = 21,
    OutDiffSh1 = 22,
    OutSpecSh0 = 23,
    OutSpecSh1 = 24,
    OutDiffHitdist = 25,
    OutSpecHitdist = 26,
    OutDiffDirectionHitdist = 27,
    OutShadowTranslucency = 28,
    OutSignal = 29,
    OutValidation = 30,
    TransientPool = 31,
    PermanentPool = 32,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdNormalEncoding {
    Rgba8Unorm = 0,
    Rgba8Snorm = 1,
    R10G10B10A2Unorm = 2,
    Rgba16Unorm = 3,
    Rgba16Snorm = 4,
    MaxNum = 5,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdRoughnessEncoding {
    SqLinear = 0,
    Linear = 1,
    SqrtLinear = 2,
    MaxNum = 3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NrdLibraryDesc {
    pub texture_offset: u32,
    pub sampler_offset: u32,
    pub constant_buffer_offset: u32,
    pub storage_texture_and_buffer_offset: u32,
    pub normal_encoding: u8,
    pub roughness_encoding: u8,
    pub reserved0: u16,
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NrdReblurHitDistanceParameters {
    pub a: f32,
    pub b: f32,
    pub c: f32,
}

impl Default for NrdReblurHitDistanceParameters {
    fn default() -> Self {
        Self {
            a: 3.0,
            b: 0.1,
            c: 20.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NrdReblurDiffuseSettings {
    pub hit_distance_parameters: NrdReblurHitDistanceParameters,
    pub antilag_luminance_sigma_scale: f32,
    pub antilag_luminance_sensitivity: f32,
    pub responsive_accumulation_roughness_threshold: f32,
    pub responsive_accumulation_min_accumulated_frame_num: u32,
    pub convergence_s: f32,
    pub convergence_b: f32,
    pub convergence_p: f32,
    pub max_accumulated_frame_num: u32,
    pub max_fast_accumulated_frame_num: u32,
    pub max_stabilized_frame_num: u32,
    pub history_fix_frame_num: u32,
    pub history_fix_base_pixel_stride: u32,
    pub history_fix_alternate_pixel_stride: u32,
    pub fast_history_clamping_sigma_scale: f32,
    pub diffuse_prepass_blur_radius: f32,
    pub specular_prepass_blur_radius: f32,
    pub min_hit_distance_weight: f32,
    pub min_blur_radius: f32,
    pub max_blur_radius: f32,
    pub lobe_angle_fraction: f32,
    pub roughness_fraction: f32,
    pub plane_distance_sensitivity: f32,
    pub specular_probability_thresholds_for_mv_modification: [f32; 2],
    pub firefly_suppressor_min_relative_scale: f32,
    pub min_material_for_diffuse: f32,
    pub min_material_for_specular: f32,
    pub checkerboard_mode: u32,
    pub hit_distance_reconstruction_mode: u32,
    pub enable_anti_firefly: u32,
    pub use_prepass_only_for_specular_motion_estimation: u32,
    pub return_history_length_instead_of_occlusion: u32,
}

impl Default for NrdReblurDiffuseSettings {
    fn default() -> Self {
        Self {
            hit_distance_parameters: NrdReblurHitDistanceParameters::default(),
            antilag_luminance_sigma_scale: 2.0,
            antilag_luminance_sensitivity: 3.0,
            responsive_accumulation_roughness_threshold: 0.0,
            responsive_accumulation_min_accumulated_frame_num: 3,
            convergence_s: 1.0,
            convergence_b: 0.2,
            convergence_p: 0.8,
            max_accumulated_frame_num: 30,
            max_fast_accumulated_frame_num: 6,
            max_stabilized_frame_num: 63,
            history_fix_frame_num: 3,
            history_fix_base_pixel_stride: 14,
            history_fix_alternate_pixel_stride: 14,
            fast_history_clamping_sigma_scale: 2.0,
            diffuse_prepass_blur_radius: 30.0,
            specular_prepass_blur_radius: 50.0,
            min_hit_distance_weight: 0.1,
            min_blur_radius: 1.0,
            max_blur_radius: 30.0,
            lobe_angle_fraction: 0.15,
            roughness_fraction: 0.15,
            plane_distance_sensitivity: 0.02,
            specular_probability_thresholds_for_mv_modification: [0.5, 0.9],
            firefly_suppressor_min_relative_scale: 2.0,
            min_material_for_diffuse: 4.0,
            min_material_for_specular: 4.0,
            checkerboard_mode: 0,
            hit_distance_reconstruction_mode: 0,
            enable_anti_firefly: 1,
            use_prepass_only_for_specular_motion_estimation: 0,
            return_history_length_instead_of_occlusion: 0,
        }
    }
}

#[repr(C)]
pub struct RevolumetricNrdInstance {
    _private: [u8; 0],
}

#[cfg(feature = "nrd")]
unsafe extern "C" {
    pub fn revolumetric_nrd_create_relax_diffuse(
        width: u32,
        height: u32,
        out_instance: *mut *mut RevolumetricNrdInstance,
    ) -> RevolumetricNrdStatus;
    pub fn revolumetric_nrd_create_reblur_diffuse(
        width: u32,
        height: u32,
        out_instance: *mut *mut RevolumetricNrdInstance,
    ) -> RevolumetricNrdStatus;
    pub fn revolumetric_nrd_destroy(instance: *mut RevolumetricNrdInstance);
    pub fn revolumetric_nrd_get_library_desc(
        out_desc: *mut NrdLibraryDesc,
    ) -> RevolumetricNrdStatus;
    pub fn revolumetric_nrd_get_instance_desc(
        instance: *const RevolumetricNrdInstance,
        out_desc: *mut NrdInstanceDesc,
    ) -> RevolumetricNrdStatus;
    pub fn revolumetric_nrd_set_common_settings(
        instance: *mut RevolumetricNrdInstance,
        settings: *const NrdCommonSettings,
    ) -> RevolumetricNrdStatus;
    pub fn revolumetric_nrd_set_relax_diffuse_settings(
        instance: *mut RevolumetricNrdInstance,
        settings: *const NrdRelaxDiffuseSettings,
    ) -> RevolumetricNrdStatus;
    pub fn revolumetric_nrd_set_reblur_diffuse_settings(
        instance: *mut RevolumetricNrdInstance,
        settings: *const NrdReblurDiffuseSettings,
    ) -> RevolumetricNrdStatus;
    pub fn revolumetric_nrd_get_dispatches(
        instance: *mut RevolumetricNrdInstance,
        out_dispatches: *mut *const NrdDispatchDesc,
        out_dispatches_num: *mut u32,
    ) -> RevolumetricNrdStatus;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sys_descriptor_layouts_are_pod_c_abi_shapes() {
        assert_eq!(std::mem::size_of::<NrdLibraryDesc>(), 20);
        assert_eq!(std::mem::size_of::<NrdTextureDesc>(), 8);
        assert_eq!(std::mem::size_of::<NrdResourceDesc>(), 12);
        assert_eq!(std::mem::size_of::<NrdResourceRangeDesc>(), 8);
        assert_eq!(std::mem::size_of::<NrdSamplerDesc>(), 4);
        assert_eq!(std::mem::offset_of!(NrdPipelineDesc, shader_identifier), 32);
        assert_eq!(std::mem::offset_of!(NrdInstanceDesc, samplers), 24);
        assert_eq!(std::mem::offset_of!(NrdDispatchDesc, pipeline_index), 48);
        assert_eq!(std::mem::size_of::<NrdDispatchDesc>(), 56);
    }
}
