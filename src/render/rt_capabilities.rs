use ash::{Instance, vk};
use std::collections::BTreeSet;
use std::ffi::CStr;

use crate::render::scene_ubo::RenderMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBackend {
    Vpt,
    Rt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RtCapabilities {
    pub acceleration_structure: bool,
    pub ray_tracing_pipeline: bool,
    pub deferred_host_operations: bool,
    pub buffer_device_address: bool,
}

impl RtCapabilities {
    pub fn supported(self) -> bool {
        self.acceleration_structure
            && self.ray_tracing_pipeline
            && self.deferred_host_operations
            && self.buffer_device_address
    }
}

pub fn resolve_render_backend(requested: RenderMode, rt_supported: bool) -> RenderBackend {
    match (requested, rt_supported) {
        (RenderMode::Auto, true) => RenderBackend::Rt,
        (RenderMode::Rt, true) => RenderBackend::Rt,
        _ => RenderBackend::Vpt,
    }
}

pub fn probe_rt_capabilities(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> RtCapabilities {
    let available_extensions =
        match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
            Ok(extensions) => extensions
                .into_iter()
                .map(|extension| unsafe {
                    CStr::from_ptr(extension.extension_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                })
                .collect::<BTreeSet<_>>(),
            Err(error) => {
                tracing::debug!(
                    %error,
                    "failed to enumerate Vulkan device extensions while probing RT support"
                );
                return RtCapabilities::default();
            }
        };

    let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default();
    let mut acceleration_structure_features =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut ray_tracing_pipeline_features =
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut vulkan12_features)
        .push_next(&mut acceleration_structure_features)
        .push_next(&mut ray_tracing_pipeline_features);

    unsafe {
        instance.get_physical_device_features2(physical_device, &mut features2);
    }

    RtCapabilities {
        acceleration_structure: has_extension(
            &available_extensions,
            ash::khr::acceleration_structure::NAME,
        ) && acceleration_structure_features.acceleration_structure
            == vk::TRUE,
        ray_tracing_pipeline: has_extension(
            &available_extensions,
            ash::khr::ray_tracing_pipeline::NAME,
        ) && ray_tracing_pipeline_features.ray_tracing_pipeline == vk::TRUE,
        deferred_host_operations: has_extension(
            &available_extensions,
            ash::khr::deferred_host_operations::NAME,
        ),
        buffer_device_address: vulkan12_features.buffer_device_address == vk::TRUE,
    }
}

fn has_extension(available_extensions: &BTreeSet<String>, target: &CStr) -> bool {
    available_extensions.contains(
        target
            .to_str()
            .expect("Vulkan extension names must be valid UTF-8"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_render_backend_falls_back_without_rt_support() {
        assert_eq!(
            resolve_render_backend(RenderMode::Auto, true),
            RenderBackend::Rt
        );
        assert_eq!(
            resolve_render_backend(RenderMode::Auto, false),
            RenderBackend::Vpt
        );
        assert_eq!(
            resolve_render_backend(RenderMode::Rt, false),
            RenderBackend::Vpt
        );
        assert_eq!(
            resolve_render_backend(RenderMode::Vpt, true),
            RenderBackend::Vpt
        );
        assert_eq!(
            resolve_render_backend(RenderMode::Rt, true),
            RenderBackend::Rt
        );
    }

    #[test]
    fn rt_capabilities_supported_requires_all_probe_components() {
        let supported = RtCapabilities {
            acceleration_structure: true,
            ray_tracing_pipeline: true,
            deferred_host_operations: true,
            buffer_device_address: true,
        };
        let unsupported = RtCapabilities {
            buffer_device_address: false,
            ..supported
        };

        assert!(supported.supported());
        assert!(!unsupported.supported());
    }
}
