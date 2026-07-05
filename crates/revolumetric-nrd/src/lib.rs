use std::ffi::CStr;
use std::fmt;
use std::os::raw::c_char;

#[cfg(feature = "nrd")]
use std::ptr::NonNull;

#[cfg(feature = "nrd")]
use revolumetric_nrd_sys as nrd_sys;

pub use revolumetric_nrd_sys::{
    NrdCommonSettings, NrdDescriptorType, NrdDispatchDesc, NrdInstanceDesc, NrdLibraryDesc,
    NrdNormalEncoding, NrdPipelineDesc, NrdReblurDiffuseSettings, NrdReblurHitDistanceParameters,
    NrdRelaxDiffuseSettings, NrdResourceDesc, NrdResourceRangeDesc, NrdResourceType,
    NrdRoughnessEncoding, NrdSamplerDesc, NrdTextureDesc, NrdTextureFormat,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NrdTextureImageDesc {
    pub format: ash::vk::Format,
    pub downsample_factor: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NrdResourceBindingDesc {
    pub descriptor_type: NrdDescriptorType,
    pub resource_type: NrdResourceType,
    pub index_in_pool: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NrdResourceRangeBindingDesc {
    pub descriptor_type: NrdDescriptorType,
    pub descriptors_num: u32,
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

pub trait NrdTextureDescExt {
    fn image_desc(self) -> NrdResult<NrdTextureImageDesc>;
}

impl NrdTextureDescExt for NrdTextureDesc {
    fn image_desc(self) -> NrdResult<NrdTextureImageDesc> {
        Ok(NrdTextureImageDesc {
            format: nrd_texture_format_to_vk(self.format)?,
            downsample_factor: self.downsample_factor,
        })
    }
}

pub trait NrdResourceDescExt {
    fn binding_desc(self) -> NrdResult<NrdResourceBindingDesc>;
}

impl NrdResourceDescExt for NrdResourceDesc {
    fn binding_desc(self) -> NrdResult<NrdResourceBindingDesc> {
        Ok(NrdResourceBindingDesc {
            descriptor_type: nrd_descriptor_type_from_abi(self.descriptor_type)?,
            resource_type: nrd_resource_type_from_abi(self.resource_type)?,
            index_in_pool: self.index_in_pool,
        })
    }
}

pub trait NrdResourceRangeDescExt {
    fn binding_desc(self) -> NrdResult<NrdResourceRangeBindingDesc>;
}

impl NrdResourceRangeDescExt for NrdResourceRangeDesc {
    fn binding_desc(self) -> NrdResult<NrdResourceRangeBindingDesc> {
        Ok(NrdResourceRangeBindingDesc {
            descriptor_type: nrd_descriptor_type_from_abi(self.descriptor_type)?,
            descriptors_num: self.descriptors_num,
        })
    }
}

fn nrd_descriptor_type_from_abi(value: u32) -> NrdResult<NrdDescriptorType> {
    let descriptor_type = match value {
        value if value == NrdDescriptorType::Texture as u32 => NrdDescriptorType::Texture,
        value if value == NrdDescriptorType::StorageTexture as u32 => {
            NrdDescriptorType::StorageTexture
        }
        _ => {
            return Err(NrdUnavailableError::new(
                "unsupported NRD descriptor type from sys ABI",
            ));
        }
    };
    Ok(descriptor_type)
}

fn nrd_resource_type_from_abi(value: u32) -> NrdResult<NrdResourceType> {
    let resource_type = match value {
        value if value == NrdResourceType::InMv as u32 => NrdResourceType::InMv,
        value if value == NrdResourceType::InNormalRoughness as u32 => {
            NrdResourceType::InNormalRoughness
        }
        value if value == NrdResourceType::InViewZ as u32 => NrdResourceType::InViewZ,
        value if value == NrdResourceType::InDiffConfidence as u32 => {
            NrdResourceType::InDiffConfidence
        }
        value if value == NrdResourceType::InSpecConfidence as u32 => {
            NrdResourceType::InSpecConfidence
        }
        value if value == NrdResourceType::InDisocclusionThresholdMix as u32 => {
            NrdResourceType::InDisocclusionThresholdMix
        }
        value if value == NrdResourceType::InDiffRadianceHitdist as u32 => {
            NrdResourceType::InDiffRadianceHitdist
        }
        value if value == NrdResourceType::InSpecRadianceHitdist as u32 => {
            NrdResourceType::InSpecRadianceHitdist
        }
        value if value == NrdResourceType::InDiffHitdist as u32 => NrdResourceType::InDiffHitdist,
        value if value == NrdResourceType::InSpecHitdist as u32 => NrdResourceType::InSpecHitdist,
        value if value == NrdResourceType::InDiffDirectionHitdist as u32 => {
            NrdResourceType::InDiffDirectionHitdist
        }
        value if value == NrdResourceType::InDiffSh0 as u32 => NrdResourceType::InDiffSh0,
        value if value == NrdResourceType::InDiffSh1 as u32 => NrdResourceType::InDiffSh1,
        value if value == NrdResourceType::InSpecSh0 as u32 => NrdResourceType::InSpecSh0,
        value if value == NrdResourceType::InSpecSh1 as u32 => NrdResourceType::InSpecSh1,
        value if value == NrdResourceType::InPenumbra as u32 => NrdResourceType::InPenumbra,
        value if value == NrdResourceType::InTranslucency as u32 => NrdResourceType::InTranslucency,
        value if value == NrdResourceType::InSignal as u32 => NrdResourceType::InSignal,
        value if value == NrdResourceType::OutDiffRadianceHitdist as u32 => {
            NrdResourceType::OutDiffRadianceHitdist
        }
        value if value == NrdResourceType::OutSpecRadianceHitdist as u32 => {
            NrdResourceType::OutSpecRadianceHitdist
        }
        value if value == NrdResourceType::OutDiffSh0 as u32 => NrdResourceType::OutDiffSh0,
        value if value == NrdResourceType::OutDiffSh1 as u32 => NrdResourceType::OutDiffSh1,
        value if value == NrdResourceType::OutSpecSh0 as u32 => NrdResourceType::OutSpecSh0,
        value if value == NrdResourceType::OutSpecSh1 as u32 => NrdResourceType::OutSpecSh1,
        value if value == NrdResourceType::OutDiffHitdist as u32 => NrdResourceType::OutDiffHitdist,
        value if value == NrdResourceType::OutSpecHitdist as u32 => NrdResourceType::OutSpecHitdist,
        value if value == NrdResourceType::OutDiffDirectionHitdist as u32 => {
            NrdResourceType::OutDiffDirectionHitdist
        }
        value if value == NrdResourceType::OutShadowTranslucency as u32 => {
            NrdResourceType::OutShadowTranslucency
        }
        value if value == NrdResourceType::OutSignal as u32 => NrdResourceType::OutSignal,
        value if value == NrdResourceType::OutValidation as u32 => NrdResourceType::OutValidation,
        value if value == NrdResourceType::TransientPool as u32 => NrdResourceType::TransientPool,
        value if value == NrdResourceType::PermanentPool as u32 => NrdResourceType::PermanentPool,
        _ => {
            return Err(NrdUnavailableError::new(
                "unsupported NRD resource type from sys ABI",
            ));
        }
    };
    Ok(resource_type)
}

fn nrd_texture_format_to_vk(format: u32) -> NrdResult<ash::vk::Format> {
    let format = match format {
        value if value == NrdTextureFormat::R8Unorm as u32 => ash::vk::Format::R8_UNORM,
        value if value == NrdTextureFormat::R8Snorm as u32 => ash::vk::Format::R8_SNORM,
        value if value == NrdTextureFormat::R8Uint as u32 => ash::vk::Format::R8_UINT,
        value if value == NrdTextureFormat::R8Sint as u32 => ash::vk::Format::R8_SINT,
        value if value == NrdTextureFormat::Rg8Unorm as u32 => ash::vk::Format::R8G8_UNORM,
        value if value == NrdTextureFormat::Rg8Snorm as u32 => ash::vk::Format::R8G8_SNORM,
        value if value == NrdTextureFormat::Rg8Uint as u32 => ash::vk::Format::R8G8_UINT,
        value if value == NrdTextureFormat::Rg8Sint as u32 => ash::vk::Format::R8G8_SINT,
        value if value == NrdTextureFormat::Rgba8Unorm as u32 => ash::vk::Format::R8G8B8A8_UNORM,
        value if value == NrdTextureFormat::Rgba8Snorm as u32 => ash::vk::Format::R8G8B8A8_SNORM,
        value if value == NrdTextureFormat::Rgba8Uint as u32 => ash::vk::Format::R8G8B8A8_UINT,
        value if value == NrdTextureFormat::Rgba8Sint as u32 => ash::vk::Format::R8G8B8A8_SINT,
        value if value == NrdTextureFormat::Rgba8Srgb as u32 => ash::vk::Format::R8G8B8A8_SRGB,
        value if value == NrdTextureFormat::R16Unorm as u32 => ash::vk::Format::R16_UNORM,
        value if value == NrdTextureFormat::R16Snorm as u32 => ash::vk::Format::R16_SNORM,
        value if value == NrdTextureFormat::R16Uint as u32 => ash::vk::Format::R16_UINT,
        value if value == NrdTextureFormat::R16Sint as u32 => ash::vk::Format::R16_SINT,
        value if value == NrdTextureFormat::R16Sfloat as u32 => ash::vk::Format::R16_SFLOAT,
        value if value == NrdTextureFormat::Rg16Unorm as u32 => ash::vk::Format::R16G16_UNORM,
        value if value == NrdTextureFormat::Rg16Snorm as u32 => ash::vk::Format::R16G16_SNORM,
        value if value == NrdTextureFormat::Rg16Uint as u32 => ash::vk::Format::R16G16_UINT,
        value if value == NrdTextureFormat::Rg16Sint as u32 => ash::vk::Format::R16G16_SINT,
        value if value == NrdTextureFormat::Rg16Sfloat as u32 => ash::vk::Format::R16G16_SFLOAT,
        value if value == NrdTextureFormat::Rgba16Unorm as u32 => {
            ash::vk::Format::R16G16B16A16_UNORM
        }
        value if value == NrdTextureFormat::Rgba16Snorm as u32 => {
            ash::vk::Format::R16G16B16A16_SNORM
        }
        value if value == NrdTextureFormat::Rgba16Uint as u32 => ash::vk::Format::R16G16B16A16_UINT,
        value if value == NrdTextureFormat::Rgba16Sint as u32 => ash::vk::Format::R16G16B16A16_SINT,
        value if value == NrdTextureFormat::Rgba16Sfloat as u32 => {
            ash::vk::Format::R16G16B16A16_SFLOAT
        }
        value if value == NrdTextureFormat::R32Uint as u32 => ash::vk::Format::R32_UINT,
        value if value == NrdTextureFormat::R32Sint as u32 => ash::vk::Format::R32_SINT,
        value if value == NrdTextureFormat::R32Sfloat as u32 => ash::vk::Format::R32_SFLOAT,
        value if value == NrdTextureFormat::Rg32Uint as u32 => ash::vk::Format::R32G32_UINT,
        value if value == NrdTextureFormat::Rg32Sint as u32 => ash::vk::Format::R32G32_SINT,
        value if value == NrdTextureFormat::Rg32Sfloat as u32 => ash::vk::Format::R32G32_SFLOAT,
        value if value == NrdTextureFormat::Rgb32Uint as u32 => ash::vk::Format::R32G32B32_UINT,
        value if value == NrdTextureFormat::Rgb32Sint as u32 => ash::vk::Format::R32G32B32_SINT,
        value if value == NrdTextureFormat::Rgb32Sfloat as u32 => ash::vk::Format::R32G32B32_SFLOAT,
        value if value == NrdTextureFormat::Rgba32Uint as u32 => ash::vk::Format::R32G32B32A32_UINT,
        value if value == NrdTextureFormat::Rgba32Sint as u32 => ash::vk::Format::R32G32B32A32_SINT,
        value if value == NrdTextureFormat::Rgba32Sfloat as u32 => {
            ash::vk::Format::R32G32B32A32_SFLOAT
        }
        value if value == NrdTextureFormat::R10G10B10A2Unorm as u32 => {
            ash::vk::Format::A2B10G10R10_UNORM_PACK32
        }
        value if value == NrdTextureFormat::R11G11B10Ufloat as u32 => {
            ash::vk::Format::B10G11R11_UFLOAT_PACK32
        }
        _ => {
            return Err(NrdUnavailableError::new(
                "unsupported NRD texture format from sys ABI",
            ));
        }
    };
    Ok(format)
}

impl NrdInstanceSnapshot {
    pub fn from_sys(desc: &NrdInstanceDesc) -> NrdResult<Self> {
        let samplers = copy_sys_slice(desc.samplers, desc.samplers_num, "NRD samplers")?;
        let permanent_pool = copy_sys_slice(
            desc.permanent_pool,
            desc.permanent_pool_size,
            "NRD permanent texture pool",
        )?;
        let transient_pool = copy_sys_slice(
            desc.transient_pool,
            desc.transient_pool_size,
            "NRD transient texture pool",
        )?;
        let pipelines = copy_sys_slice(desc.pipelines, desc.pipelines_num, "NRD pipelines")?
            .into_iter()
            .map(|pipeline| {
                let spirv_bytecode =
                    copy_spirv_bytecode(pipeline.spirv_bytecode, pipeline.spirv_bytecode_size)?;
                let resource_ranges = copy_sys_slice(
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

    pub fn from_ffi(desc: &NrdInstanceDesc) -> NrdResult<Self> {
        Self::from_sys(desc)
    }
}

impl NrdDispatchSnapshot {
    pub fn copy_slice(
        dispatches: *const NrdDispatchDesc,
        dispatches_num: u32,
    ) -> NrdResult<Vec<Self>> {
        copy_sys_slice(dispatches, dispatches_num, "NRD dispatches")?
            .into_iter()
            .map(Self::from_sys)
            .collect()
    }

    fn from_sys(desc: NrdDispatchDesc) -> NrdResult<Self> {
        Ok(Self {
            name: copy_c_string(desc.name, "NRD dispatch name")?,
            identifier: desc.identifier,
            resources: copy_sys_slice(
                desc.resources,
                desc.resources_num,
                "NRD dispatch resources",
            )?,
            constant_buffer_data: copy_sys_slice(
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

fn copy_sys_slice<T: Copy>(ptr: *const T, len: u32, context: &'static str) -> NrdResult<Vec<T>> {
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
    let bytes = copy_sys_bytes(ptr.cast::<u8>(), byte_size, "NRD SPIR-V bytecode bytes")?;
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u32>())
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn copy_sys_bytes(ptr: *const u8, len: u64, context: &'static str) -> NrdResult<Vec<u8>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(NrdUnavailableError::new(context));
    }
    if len > isize::MAX as u64 || len > usize::MAX as u64 {
        return Err(NrdUnavailableError::new(
            "NRD byte buffer is too large to copy",
        ));
    }
    let len = len as usize;
    let mut bytes = vec![0; len];
    unsafe {
        std::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), len);
    }
    Ok(bytes)
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
    ptr: NonNull<nrd_sys::RevolumetricNrdInstance>,
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
        let status =
            unsafe { nrd_sys::revolumetric_nrd_create_relax_diffuse(width, height, &mut raw) };
        if status != nrd_sys::REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to create NRD RELAX_DIFFUSE instance",
            ));
        }
        let ptr = NonNull::new(raw).ok_or_else(|| {
            NrdUnavailableError::new("NRD returned a null RELAX_DIFFUSE instance")
        })?;
        Ok(Self { ptr })
    }

    #[cfg(not(feature = "nrd"))]
    pub fn reblur_diffuse(_width: u32, _height: u32) -> NrdResult<Self> {
        Err(NrdUnavailableError::new(
            "NRD is unavailable because the nrd Cargo feature is disabled",
        ))
    }

    #[cfg(feature = "nrd")]
    pub fn reblur_diffuse(width: u32, height: u32) -> NrdResult<Self> {
        let mut raw = std::ptr::null_mut();
        let status =
            unsafe { nrd_sys::revolumetric_nrd_create_reblur_diffuse(width, height, &mut raw) };
        if status != nrd_sys::REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to create NRD REBLUR_DIFFUSE instance",
            ));
        }
        let ptr = NonNull::new(raw).ok_or_else(|| {
            NrdUnavailableError::new("NRD returned a null REBLUR_DIFFUSE instance")
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
        let status = unsafe { nrd_sys::revolumetric_nrd_get_library_desc(&mut desc) };
        if status != nrd_sys::REVOLUMETRIC_NRD_STATUS_OK {
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
        let status =
            unsafe { nrd_sys::revolumetric_nrd_get_instance_desc(self.ptr.as_ptr(), &mut desc) };
        if status != nrd_sys::REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to query NRD instance descriptor",
            ));
        }
        NrdInstanceSnapshot::from_sys(&desc)
    }

    #[cfg(not(feature = "nrd"))]
    pub fn set_common_settings(&mut self, _settings: &NrdCommonSettings) -> NrdResult<()> {
        Err(NrdUnavailableError::new(
            "NRD is unavailable because the nrd Cargo feature is disabled",
        ))
    }

    #[cfg(feature = "nrd")]
    pub fn set_common_settings(&mut self, settings: &NrdCommonSettings) -> NrdResult<()> {
        let status =
            unsafe { nrd_sys::revolumetric_nrd_set_common_settings(self.ptr.as_ptr(), settings) };
        if status != nrd_sys::REVOLUMETRIC_NRD_STATUS_OK {
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
        let status = unsafe {
            nrd_sys::revolumetric_nrd_set_relax_diffuse_settings(self.ptr.as_ptr(), settings)
        };
        if status != nrd_sys::REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to upload NRD RELAX_DIFFUSE settings",
            ));
        }
        Ok(())
    }

    #[cfg(not(feature = "nrd"))]
    pub fn set_reblur_diffuse_settings(
        &mut self,
        _settings: &NrdReblurDiffuseSettings,
    ) -> NrdResult<()> {
        Err(NrdUnavailableError::new(
            "NRD is unavailable because the nrd Cargo feature is disabled",
        ))
    }

    #[cfg(feature = "nrd")]
    pub fn set_reblur_diffuse_settings(
        &mut self,
        settings: &NrdReblurDiffuseSettings,
    ) -> NrdResult<()> {
        let status = unsafe {
            nrd_sys::revolumetric_nrd_set_reblur_diffuse_settings(self.ptr.as_ptr(), settings)
        };
        if status != nrd_sys::REVOLUMETRIC_NRD_STATUS_OK {
            return Err(NrdUnavailableError::new(
                "failed to upload NRD REBLUR_DIFFUSE settings",
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
            nrd_sys::revolumetric_nrd_get_dispatches(
                self.ptr.as_ptr(),
                &mut dispatches,
                &mut dispatches_num,
            )
        };
        if status != nrd_sys::REVOLUMETRIC_NRD_STATUS_OK {
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
        unsafe { nrd_sys::revolumetric_nrd_destroy(self.ptr.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn public_descriptor_layouts_reexport_sys_abi_shapes() {
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

    #[test]
    fn public_library_desc_exposes_normal_and_roughness_encoding_contract() {
        assert_eq!(std::mem::size_of::<NrdLibraryDesc>(), 20);
        assert_eq!(std::mem::offset_of!(NrdLibraryDesc, normal_encoding), 16);
        assert_eq!(std::mem::offset_of!(NrdLibraryDesc, roughness_encoding), 17);
        assert_eq!(NrdNormalEncoding::R10G10B10A2Unorm as u8, 2);
        assert_eq!(NrdRoughnessEncoding::Linear as u8, 1);
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
    fn owned_instance_snapshot_deep_copies_sys_arrays() {
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

        let snapshot = NrdInstanceSnapshot::from_sys(&desc).unwrap();

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
    fn owned_instance_snapshot_copies_unaligned_spirv_bytecode() {
        let mut unaligned_spirv_bytes: [u8; 9] =
            [0xFF, 0x03, 0x02, 0x23, 0x07, 0x01, 0x00, 0x00, 0x00];
        let mut resource_ranges = [NrdResourceRangeDesc {
            descriptor_type: 1,
            descriptors_num: 1,
        }];
        let mut shader_identifier = [0 as std::os::raw::c_char; 256];
        for (dst, src) in shader_identifier.iter_mut().zip(b"unaligned") {
            *dst = *src as std::os::raw::c_char;
        }
        let pipeline = NrdPipelineDesc {
            spirv_bytecode: unsafe { unaligned_spirv_bytes.as_ptr().add(1) }.cast::<u32>(),
            spirv_bytecode_size: 8,
            resource_ranges: resource_ranges.as_ptr(),
            resource_ranges_num: resource_ranges.len() as u32,
            has_constant_data: 0,
            shader_identifier,
        };
        let desc = NrdInstanceDesc {
            constant_buffer_and_samplers_space_index: 0,
            resources_space_index: 0,
            constant_buffer_register_index: 0,
            samplers_base_register_index: 0,
            resources_base_register_index: 0,
            constant_buffer_max_data_size: 0,
            samplers: std::ptr::null(),
            samplers_num: 0,
            pipelines: &pipeline,
            pipelines_num: 1,
            permanent_pool: std::ptr::null(),
            permanent_pool_size: 0,
            transient_pool: std::ptr::null(),
            transient_pool_size: 0,
        };

        let snapshot = NrdInstanceSnapshot::from_sys(&desc).unwrap();

        unaligned_spirv_bytes[1] = 0;
        resource_ranges[0].descriptors_num = 99;
        assert_eq!(unaligned_spirv_bytes[1], 0);
        assert_eq!(resource_ranges[0].descriptors_num, 99);
        assert_eq!(snapshot.pipelines[0].spirv_bytecode, vec![0x0723_0203, 1]);
        assert_eq!(snapshot.pipelines[0].resource_ranges[0].descriptors_num, 1);
        assert_eq!(snapshot.pipelines[0].shader_identifier, "unaligned");
    }

    #[test]
    fn owned_dispatch_snapshot_deep_copies_sys_arrays() {
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

    #[test]
    fn nrd_resource_desc_maps_stable_abi_resource_semantics() {
        let desc = NrdResourceDesc {
            descriptor_type: NrdDescriptorType::Texture as u32,
            resource_type: NrdResourceType::InDiffRadianceHitdist as u32,
            index_in_pool: 7,
            reserved0: 0,
        };

        let image_desc = desc.binding_desc().unwrap();

        assert_eq!(image_desc.descriptor_type, NrdDescriptorType::Texture);
        assert_eq!(
            image_desc.resource_type,
            NrdResourceType::InDiffRadianceHitdist
        );
        assert_eq!(image_desc.index_in_pool, 7);
    }

    #[test]
    fn nrd_resource_desc_rejects_unknown_stable_abi_values() {
        let unknown_descriptor = NrdResourceDesc {
            descriptor_type: NrdDescriptorType::Unsupported as u32,
            resource_type: NrdResourceType::InMv as u32,
            index_in_pool: 0,
            reserved0: 0,
        };
        let unknown_resource = NrdResourceDesc {
            descriptor_type: NrdDescriptorType::Texture as u32,
            resource_type: NrdResourceType::Unsupported as u32,
            index_in_pool: 0,
            reserved0: 0,
        };

        assert!(
            unknown_descriptor
                .binding_desc()
                .unwrap_err()
                .to_string()
                .contains("unsupported NRD descriptor type")
        );
        assert!(
            unknown_resource
                .binding_desc()
                .unwrap_err()
                .to_string()
                .contains("unsupported NRD resource type")
        );
    }

    #[test]
    fn nrd_resource_range_desc_maps_stable_abi_descriptor_type() {
        let range = NrdResourceRangeDesc {
            descriptor_type: NrdDescriptorType::StorageTexture as u32,
            descriptors_num: 5,
        };

        let binding = range.binding_desc().unwrap();

        assert_eq!(binding.descriptor_type, NrdDescriptorType::StorageTexture);
        assert_eq!(binding.descriptors_num, 5);
    }

    #[test]
    fn nrd_resource_range_desc_rejects_unknown_descriptor_type() {
        let range = NrdResourceRangeDesc {
            descriptor_type: NrdDescriptorType::Unsupported as u32,
            descriptors_num: 1,
        };

        let error = range.binding_desc().unwrap_err();

        assert!(
            error
                .to_string()
                .contains("unsupported NRD descriptor type")
        );
    }

    #[test]
    fn nrd_texture_desc_maps_stable_abi_formats_to_vulkan_formats() {
        for (format, expected_vk_format) in [
            (NrdTextureFormat::R16Sfloat, ash::vk::Format::R16_SFLOAT),
            (NrdTextureFormat::Rg16Sfloat, ash::vk::Format::R16G16_SFLOAT),
            (
                NrdTextureFormat::Rgba16Sfloat,
                ash::vk::Format::R16G16B16A16_SFLOAT,
            ),
            (NrdTextureFormat::R32Sfloat, ash::vk::Format::R32_SFLOAT),
            (NrdTextureFormat::Rg32Sfloat, ash::vk::Format::R32G32_SFLOAT),
            (
                NrdTextureFormat::Rgba32Sfloat,
                ash::vk::Format::R32G32B32A32_SFLOAT,
            ),
            (
                NrdTextureFormat::R11G11B10Ufloat,
                ash::vk::Format::B10G11R11_UFLOAT_PACK32,
            ),
        ] {
            let desc = NrdTextureDesc {
                format: format as u32,
                downsample_factor: 2,
                reserved0: 0,
            };

            let image_desc = desc.image_desc().unwrap();

            assert_eq!(image_desc.format, expected_vk_format);
            assert_eq!(image_desc.downsample_factor, 2);
        }
    }

    #[test]
    fn nrd_texture_desc_rejects_unknown_format() {
        let desc = NrdTextureDesc {
            format: NrdTextureFormat::Unknown as u32,
            downsample_factor: 1,
            reserved0: 0,
        };

        let error = desc.image_desc().unwrap_err();

        assert!(error.to_string().contains("unsupported NRD texture format"));
    }
}
