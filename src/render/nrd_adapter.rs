use std::ffi::CStr;
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NrdPipelineSnapshot {
    pub spirv_bytecode: Vec<u32>,
    pub resource_ranges: Vec<NrdResourceRangeDesc>,
    pub has_constant_data: bool,
    pub shader_identifier: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NrdInstanceSnapshot {
    pub constant_buffer_and_samplers_space_index: u32,
    pub resources_space_index: u32,
    pub constant_buffer_register_index: u32,
    pub samplers_base_register_index: u32,
    pub resources_base_register_index: u32,
    pub constant_buffer_max_data_size: u32,
    pub samplers: Vec<NrdSamplerDesc>,
    pub pipelines: Vec<NrdPipelineSnapshot>,
    pub permanent_pool: Vec<NrdTextureDesc>,
    pub transient_pool: Vec<NrdTextureDesc>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NrdDispatchSnapshot {
    pub name: String,
    pub identifier: u32,
    pub resources: Vec<NrdResourceDesc>,
    pub constant_buffer_data: Vec<u8>,
    pub constant_buffer_data_matches_previous_dispatch: bool,
    pub pipeline_index: u16,
    pub grid_width: u16,
    pub grid_height: u16,
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

impl NrdInstanceSnapshot {
    pub fn from_ffi(desc: &NrdInstanceDesc) -> NrdResult<Self> {
        let samplers = copy_ffi_slice(desc.samplers, desc.samplers_num, "NRD samplers")?;
        let permanent_pool = copy_ffi_slice(
            desc.permanent_pool,
            desc.permanent_pool_size,
            "NRD permanent texture pool",
        )?;
        let transient_pool = copy_ffi_slice(
            desc.transient_pool,
            desc.transient_pool_size,
            "NRD transient texture pool",
        )?;
        let pipelines = copy_ffi_slice(desc.pipelines, desc.pipelines_num, "NRD pipelines")?
            .into_iter()
            .map(|pipeline| {
                let spirv_bytecode =
                    copy_spirv_bytecode(pipeline.spirv_bytecode, pipeline.spirv_bytecode_size)?;
                let resource_ranges = copy_ffi_slice(
                    pipeline.resource_ranges,
                    pipeline.resource_ranges_num,
                    "NRD pipeline resource ranges",
                )?;
                Ok(NrdPipelineSnapshot {
                    spirv_bytecode,
                    resource_ranges,
                    has_constant_data: pipeline.has_constant_data != 0,
                    shader_identifier: copy_shader_identifier(&pipeline.shader_identifier),
                })
            })
            .collect::<NrdResult<Vec<_>>>()?;

        Ok(Self {
            constant_buffer_and_samplers_space_index: desc.constant_buffer_and_samplers_space_index,
            resources_space_index: desc.resources_space_index,
            constant_buffer_register_index: desc.constant_buffer_register_index,
            samplers_base_register_index: desc.samplers_base_register_index,
            resources_base_register_index: desc.resources_base_register_index,
            constant_buffer_max_data_size: desc.constant_buffer_max_data_size,
            samplers,
            pipelines,
            permanent_pool,
            transient_pool,
        })
    }
}

impl NrdDispatchSnapshot {
    pub fn copy_slice(
        dispatches: *const NrdDispatchDesc,
        dispatches_num: u32,
    ) -> NrdResult<Vec<Self>> {
        copy_ffi_slice(dispatches, dispatches_num, "NRD dispatches")?
            .into_iter()
            .map(Self::from_ffi)
            .collect()
    }

    fn from_ffi(desc: NrdDispatchDesc) -> NrdResult<Self> {
        Ok(Self {
            name: copy_c_string(desc.name, "NRD dispatch name")?,
            identifier: desc.identifier,
            resources: copy_ffi_slice(
                desc.resources,
                desc.resources_num,
                "NRD dispatch resources",
            )?,
            constant_buffer_data: copy_ffi_slice(
                desc.constant_buffer_data,
                desc.constant_buffer_data_size,
                "NRD dispatch constant buffer data",
            )?,
            constant_buffer_data_matches_previous_dispatch: desc
                .constant_buffer_data_matches_previous_dispatch
                != 0,
            pipeline_index: desc.pipeline_index,
            grid_width: desc.grid_width,
            grid_height: desc.grid_height,
        })
    }
}

fn copy_ffi_slice<T: Copy>(ptr: *const T, len: u32, context: &'static str) -> NrdResult<Vec<T>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(NrdUnavailableError::new(context));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec())
}

fn copy_spirv_bytecode(ptr: *const u32, byte_size: u64) -> NrdResult<Vec<u32>> {
    if byte_size == 0 {
        return Ok(Vec::new());
    }
    if !byte_size.is_multiple_of(std::mem::size_of::<u32>() as u64) {
        return Err(NrdUnavailableError::new(
            "NRD SPIR-V bytecode size is not u32 aligned",
        ));
    }
    let word_count = byte_size / std::mem::size_of::<u32>() as u64;
    if word_count > usize::MAX as u64 {
        return Err(NrdUnavailableError::new(
            "NRD SPIR-V bytecode is too large to copy",
        ));
    }
    copy_ffi_slice(ptr, word_count as u32, "NRD SPIR-V bytecode")
}

fn copy_shader_identifier(identifier: &[c_char; 256]) -> String {
    let bytes: Vec<u8> = identifier
        .iter()
        .take_while(|&&byte| byte != 0)
        .map(|&byte| byte as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn copy_c_string(ptr: *const c_char, context: &'static str) -> NrdResult<String> {
    if ptr.is_null() {
        return Err(NrdUnavailableError::new(context));
    }
    let string = unsafe { CStr::from_ptr(ptr) };
    Ok(string.to_string_lossy().into_owned())
}

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

    #[cfg(not(feature = "nrd"))]
    pub fn library_desc() -> NrdResult<NrdLibraryDesc> {
        Err(NrdUnavailableError::new(
            "NRD is unavailable because the nrd Cargo feature is disabled",
        ))
    }

    #[cfg(feature = "nrd")]
    pub fn library_desc() -> NrdResult<NrdLibraryDesc> {
        let mut desc = NrdLibraryDesc::default();
        let status = unsafe { revolumetric_nrd_get_library_desc(&mut desc) };
        if status != REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to query NRD library descriptor",
            ));
        }
        Ok(desc)
    }

    #[cfg(not(feature = "nrd"))]
    pub fn instance_snapshot(&self) -> NrdResult<NrdInstanceSnapshot> {
        Err(NrdUnavailableError::new(
            "NRD is unavailable because the nrd Cargo feature is disabled",
        ))
    }

    #[cfg(feature = "nrd")]
    pub fn instance_snapshot(&self) -> NrdResult<NrdInstanceSnapshot> {
        let mut desc = NrdInstanceDesc {
            constant_buffer_and_samplers_space_index: 0,
            resources_space_index: 0,
            constant_buffer_register_index: 0,
            samplers_base_register_index: 0,
            resources_base_register_index: 0,
            constant_buffer_max_data_size: 0,
            samplers: std::ptr::null(),
            samplers_num: 0,
            pipelines: std::ptr::null(),
            pipelines_num: 0,
            permanent_pool: std::ptr::null(),
            permanent_pool_size: 0,
            transient_pool: std::ptr::null(),
            transient_pool_size: 0,
        };
        let status = unsafe { revolumetric_nrd_get_instance_desc(self.ptr.as_ptr(), &mut desc) };
        if status != REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to query NRD instance descriptor",
            ));
        }
        NrdInstanceSnapshot::from_ffi(&desc)
    }

    #[cfg(not(feature = "nrd"))]
    pub fn set_common_settings(&mut self, _settings: &NrdCommonSettings) -> NrdResult<()> {
        Err(NrdUnavailableError::new(
            "NRD is unavailable because the nrd Cargo feature is disabled",
        ))
    }

    #[cfg(feature = "nrd")]
    pub fn set_common_settings(&mut self, settings: &NrdCommonSettings) -> NrdResult<()> {
        let status = unsafe { revolumetric_nrd_set_common_settings(self.ptr.as_ptr(), settings) };
        if status != REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to upload NRD common settings",
            ));
        }
        Ok(())
    }

    #[cfg(not(feature = "nrd"))]
    pub fn set_relax_diffuse_settings(
        &mut self,
        _settings: &NrdRelaxDiffuseSettings,
    ) -> NrdResult<()> {
        Err(NrdUnavailableError::new(
            "NRD is unavailable because the nrd Cargo feature is disabled",
        ))
    }

    #[cfg(feature = "nrd")]
    pub fn set_relax_diffuse_settings(
        &mut self,
        settings: &NrdRelaxDiffuseSettings,
    ) -> NrdResult<()> {
        let status =
            unsafe { revolumetric_nrd_set_relax_diffuse_settings(self.ptr.as_ptr(), settings) };
        if status != REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to upload NRD RELAX_DIFFUSE settings",
            ));
        }
        Ok(())
    }

    #[cfg(not(feature = "nrd"))]
    pub fn dispatch_snapshot(&mut self) -> NrdResult<Vec<NrdDispatchSnapshot>> {
        Err(NrdUnavailableError::new(
            "NRD is unavailable because the nrd Cargo feature is disabled",
        ))
    }

    #[cfg(feature = "nrd")]
    pub fn dispatch_snapshot(&mut self) -> NrdResult<Vec<NrdDispatchSnapshot>> {
        let mut dispatches = std::ptr::null();
        let mut dispatches_num = 0;
        let status = unsafe {
            revolumetric_nrd_get_dispatches(self.ptr.as_ptr(), &mut dispatches, &mut dispatches_num)
        };
        if status != REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to query NRD compute dispatches",
            ));
        }
        NrdDispatchSnapshot::copy_slice(dispatches, dispatches_num)
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
    use std::ffi::CString;

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

    #[test]
    fn owned_instance_snapshot_deep_copies_ffi_arrays() {
        let mut samplers = [NrdSamplerDesc { mode: 7 }];
        let mut permanent_pool = [NrdTextureDesc {
            format: 1,
            downsample_factor: 2,
            reserved0: 0,
        }];
        let mut transient_pool = [NrdTextureDesc {
            format: 3,
            downsample_factor: 4,
            reserved0: 0,
        }];
        let mut resource_ranges = [NrdResourceRangeDesc {
            descriptor_type: 5,
            descriptors_num: 6,
        }];
        let mut spirv = [0x0723_0203, 1, 2, 3];
        let mut shader_identifier = [0 as std::os::raw::c_char; 256];
        for (dst, src) in shader_identifier.iter_mut().zip(b"relax") {
            *dst = *src as std::os::raw::c_char;
        }
        let mut pipelines = [NrdPipelineDesc {
            spirv_bytecode: spirv.as_ptr(),
            spirv_bytecode_size: (spirv.len() * std::mem::size_of::<u32>()) as u64,
            resource_ranges: resource_ranges.as_ptr(),
            resource_ranges_num: resource_ranges.len() as u32,
            has_constant_data: 1,
            shader_identifier,
        }];
        let desc = NrdInstanceDesc {
            constant_buffer_and_samplers_space_index: 10,
            resources_space_index: 11,
            constant_buffer_register_index: 12,
            samplers_base_register_index: 13,
            resources_base_register_index: 14,
            constant_buffer_max_data_size: 512,
            samplers: samplers.as_ptr(),
            samplers_num: samplers.len() as u32,
            pipelines: pipelines.as_ptr(),
            pipelines_num: pipelines.len() as u32,
            permanent_pool: permanent_pool.as_ptr(),
            permanent_pool_size: permanent_pool.len() as u32,
            transient_pool: transient_pool.as_ptr(),
            transient_pool_size: transient_pool.len() as u32,
        };

        let snapshot = NrdInstanceSnapshot::from_ffi(&desc).unwrap();

        samplers[0].mode = 99;
        permanent_pool[0].format = 99;
        transient_pool[0].format = 99;
        resource_ranges[0].descriptor_type = 99;
        spirv[0] = 99;
        pipelines[0].has_constant_data = 0;
        assert_eq!(samplers[0].mode, 99);
        assert_eq!(permanent_pool[0].format, 99);
        assert_eq!(transient_pool[0].format, 99);
        assert_eq!(resource_ranges[0].descriptor_type, 99);
        assert_eq!(spirv[0], 99);
        assert_eq!(pipelines[0].has_constant_data, 0);

        assert_eq!(snapshot.samplers[0].mode, 7);
        assert_eq!(snapshot.permanent_pool[0].format, 1);
        assert_eq!(snapshot.transient_pool[0].format, 3);
        assert_eq!(snapshot.pipelines[0].resource_ranges[0].descriptor_type, 5);
        assert_eq!(snapshot.pipelines[0].spirv_bytecode[0], 0x0723_0203);
        assert!(snapshot.pipelines[0].has_constant_data);
        assert_eq!(snapshot.pipelines[0].shader_identifier, "relax");
    }

    #[test]
    fn owned_dispatch_snapshot_deep_copies_ffi_arrays() {
        let name = CString::new("Relax Diffuse Prepass").unwrap();
        let mut resources = [NrdResourceDesc {
            descriptor_type: 1,
            resource_type: 2,
            index_in_pool: 3,
            reserved0: 0,
        }];
        let mut constant_data = [4_u8, 5, 6, 7];
        let dispatches = [NrdDispatchDesc {
            name: name.as_ptr(),
            identifier: 8,
            resources: resources.as_ptr(),
            resources_num: resources.len() as u32,
            constant_buffer_data: constant_data.as_ptr(),
            constant_buffer_data_size: constant_data.len() as u32,
            constant_buffer_data_matches_previous_dispatch: 1,
            pipeline_index: 9,
            grid_width: 10,
            grid_height: 11,
            reserved0: 0,
        }];

        let snapshot =
            NrdDispatchSnapshot::copy_slice(dispatches.as_ptr(), dispatches.len() as u32).unwrap();

        resources[0].index_in_pool = 99;
        constant_data[0] = 99;
        assert_eq!(resources[0].index_in_pool, 99);
        assert_eq!(constant_data[0], 99);

        assert_eq!(snapshot[0].name, "Relax Diffuse Prepass");
        assert_eq!(snapshot[0].identifier, 8);
        assert_eq!(snapshot[0].resources[0].index_in_pool, 3);
        assert_eq!(snapshot[0].constant_buffer_data, vec![4, 5, 6, 7]);
        assert!(snapshot[0].constant_buffer_data_matches_previous_dispatch);
        assert_eq!(snapshot[0].pipeline_index, 9);
        assert_eq!(snapshot[0].grid_width, 10);
        assert_eq!(snapshot[0].grid_height, 11);
    }
}
