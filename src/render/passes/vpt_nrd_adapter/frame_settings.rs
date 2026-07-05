use anyhow::{Context, Result};

use revolumetric_nrd::{
    NrdCommonSettings, NrdLibraryDesc, NrdNormalEncoding, NrdReblurDiffuseSettings,
    NrdRelaxDiffuseSettings, NrdRoughnessEncoding,
};

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdAccumulationMode {
    Continue = 0,
    Restart = 1,
    ClearAndRestart = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdCheckerboardMode {
    Off = 0,
    Black = 1,
    White = 2,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NrdHitDistanceReconstructionMode {
    Off = 0,
    Area3x3 = 1,
    Area5x5 = 2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VptNrdFrameSettingsInputs {
    pub current_world_to_view: glam::Mat4,
    pub previous_world_to_view: glam::Mat4,
    pub current_view_to_clip: glam::Mat4,
    pub previous_view_to_clip: glam::Mat4,
    pub current_resolution: [u32; 2],
    pub previous_resolution: [u32; 2],
    pub frame_index: u32,
    pub time_delta_seconds: f32,
    pub reset_history: bool,
    pub history_confidence_available: bool,
    pub relax_atrous_iteration_num: u32,
    pub enable_validation: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VptNrdFrameSettings {
    pub common: NrdCommonSettings,
    pub relax_diffuse: NrdRelaxDiffuseSettings,
    pub reblur_diffuse: NrdReblurDiffuseSettings,
}

impl VptNrdFrameSettings {
    pub fn from_inputs(inputs: VptNrdFrameSettingsInputs) -> Result<Self> {
        Ok(Self {
            common: build_nrd_common_settings(inputs)?,
            relax_diffuse: build_nrd_relax_diffuse_settings(inputs.relax_atrous_iteration_num),
            reblur_diffuse: build_nrd_reblur_diffuse_settings(),
        })
    }
}

fn build_nrd_common_settings(inputs: VptNrdFrameSettingsInputs) -> Result<NrdCommonSettings> {
    let current_resolution = nrd_resolution_to_u16(inputs.current_resolution)?;
    let previous_resolution = nrd_resolution_to_u16(inputs.previous_resolution)?;
    let current_width = inputs.current_resolution[0].max(1) as f32;
    let current_height = inputs.current_resolution[1].max(1) as f32;
    let time_delta_ms = inputs.time_delta_seconds.max(0.0) * 1000.0;

    Ok(NrdCommonSettings {
        view_to_clip_matrix: inputs.current_view_to_clip.to_cols_array(),
        view_to_clip_matrix_prev: inputs.previous_view_to_clip.to_cols_array(),
        world_to_view_matrix: inputs.current_world_to_view.to_cols_array(),
        world_to_view_matrix_prev: inputs.previous_world_to_view.to_cols_array(),
        camera_jitter: [0.0, 0.0],
        camera_jitter_prev: [0.0, 0.0],
        motion_vector_scale: [1.0 / current_width, 1.0 / current_height, 1.0],
        resource_size: current_resolution,
        resource_size_prev: previous_resolution,
        rect_size: current_resolution,
        rect_size_prev: previous_resolution,
        denoising_range: 10_000.0,
        disocclusion_threshold: 0.01,
        disocclusion_threshold_alternate: 0.05,
        split_screen: 0.0,
        time_delta_between_frames: time_delta_ms,
        view_z_scale: 1.0,
        frame_index: inputs.frame_index,
        accumulation_mode: if inputs.reset_history {
            NrdAccumulationMode::Restart as u32
        } else {
            NrdAccumulationMode::Continue as u32
        },
        is_motion_vector_in_world_space: 0,
        is_history_confidence_available: inputs.history_confidence_available as u32,
        is_disocclusion_threshold_mix_available: 0,
        enable_validation: inputs.enable_validation as u32,
    })
}

pub(crate) fn build_initial_nrd_frame_settings(
    width: u32,
    height: u32,
    relax_atrous_iteration_num: u32,
) -> Result<VptNrdFrameSettings> {
    VptNrdFrameSettings::from_inputs(VptNrdFrameSettingsInputs {
        current_world_to_view: glam::Mat4::IDENTITY,
        previous_world_to_view: glam::Mat4::IDENTITY,
        current_view_to_clip: glam::Mat4::IDENTITY,
        previous_view_to_clip: glam::Mat4::IDENTITY,
        current_resolution: [width, height],
        previous_resolution: [width, height],
        frame_index: 0,
        time_delta_seconds: 0.0,
        reset_history: true,
        history_confidence_available: true,
        relax_atrous_iteration_num,
        enable_validation: false,
    })
}

fn build_nrd_relax_diffuse_settings(atrous_iteration_num: u32) -> NrdRelaxDiffuseSettings {
    NrdRelaxDiffuseSettings {
        antilag_acceleration_amount: 0.0,
        antilag_spatial_sigma_scale: 0.0,
        antilag_temporal_sigma_scale: 0.0,
        antilag_reset_amount: 0.5,
        diffuse_max_accumulated_frame_num: 30,
        diffuse_max_fast_accumulated_frame_num: 6,
        history_fix_frame_num: 3,
        history_fix_base_pixel_stride: 14,
        history_fix_alternate_pixel_stride: 14,
        history_fix_edge_stopping_normal_power: 8.0,
        fast_history_clamping_sigma_scale: 2.0,
        diffuse_prepass_blur_radius: 0.0,
        min_hit_distance_weight: 0.1,
        spatial_variance_estimation_history_threshold: 3,
        diffuse_phi_luminance: 2.0,
        atrous_iteration_num: atrous_iteration_num.clamp(2, 8),
        diffuse_min_luminance_weight: 0.0,
        depth_threshold: 0.003,
        confidence_driven_relaxation_multiplier: 0.0,
        confidence_driven_luminance_edge_stopping_relaxation: 0.0,
        confidence_driven_normal_edge_stopping_relaxation: 0.0,
        luminance_edge_stopping_relaxation: 0.5,
        normal_edge_stopping_relaxation: 0.3,
        roughness_edge_stopping_relaxation: 1.0,
        checkerboard_mode: NrdCheckerboardMode::Off as u32,
        hit_distance_reconstruction_mode: NrdHitDistanceReconstructionMode::Off as u32,
        min_material_for_diffuse: 4.0,
        enable_anti_firefly: 1,
        enable_roughness_edge_stopping: 1,
    }
}

fn build_nrd_reblur_diffuse_settings() -> NrdReblurDiffuseSettings {
    NrdReblurDiffuseSettings {
        diffuse_prepass_blur_radius: 0.0,
        specular_prepass_blur_radius: 0.0,
        firefly_suppressor_min_relative_scale: 2.0,
        enable_anti_firefly: 1,
        use_prepass_only_for_specular_motion_estimation: 1,
        ..NrdReblurDiffuseSettings::default()
    }
}

pub(crate) fn validate_nrd_library_desc(library_desc: NrdLibraryDesc) -> Result<()> {
    anyhow::ensure!(
        library_desc.normal_encoding == NrdNormalEncoding::R10G10B10A2Unorm as u8,
        "unsupported NRD normal encoding {}; expected R10_G10_B10_A2_UNORM",
        library_desc.normal_encoding
    );
    anyhow::ensure!(
        library_desc.roughness_encoding == NrdRoughnessEncoding::Linear as u8,
        "unsupported NRD roughness encoding {}; expected LINEAR",
        library_desc.roughness_encoding
    );
    Ok(())
}

fn nrd_resolution_to_u16(resolution: [u32; 2]) -> Result<[u16; 2]> {
    let width = u16::try_from(resolution[0]).context("NRD frame resolution exceeds u16")?;
    let height = u16::try_from(resolution[1]).context("NRD frame resolution exceeds u16")?;
    Ok([width.max(1), height.max(1)])
}
