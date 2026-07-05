use anyhow::{Context, Result};
use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DescriptorBindingSpec {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub stage_flags: vk::ShaderStageFlags,
    pub count: u32,
}

impl DescriptorBindingSpec {
    pub const fn compute(binding: u32, descriptor_type: vk::DescriptorType) -> Self {
        Self {
            binding,
            descriptor_type,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            count: 1,
        }
    }

    pub fn ray_tracing(binding: u32, descriptor_type: vk::DescriptorType) -> Self {
        Self {
            binding,
            descriptor_type,
            stage_flags: ray_tracing_shader_stage_flags(),
            count: 1,
        }
    }
}

pub(crate) fn ray_tracing_shader_stage_flags() -> vk::ShaderStageFlags {
    vk::ShaderStageFlags::RAYGEN_KHR
        | vk::ShaderStageFlags::ANY_HIT_KHR
        | vk::ShaderStageFlags::CLOSEST_HIT_KHR
        | vk::ShaderStageFlags::MISS_KHR
        | vk::ShaderStageFlags::INTERSECTION_KHR
        | vk::ShaderStageFlags::CALLABLE_KHR
}

pub struct DescriptorLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
}

impl DescriptorLayoutBuilder {
    pub fn new() -> Self {
        Self {
            bindings: Vec::new(),
        }
    }

    pub fn add_binding(
        mut self,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
        count: u32,
    ) -> Self {
        self.bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(descriptor_type)
                .descriptor_count(count)
                .stage_flags(stage_flags),
        );
        self
    }

    pub fn add_binding_spec(self, spec: DescriptorBindingSpec) -> Self {
        self.add_binding(
            spec.binding,
            spec.descriptor_type,
            spec.stage_flags,
            spec.count,
        )
    }

    pub fn add_binding_specs(mut self, specs: &[DescriptorBindingSpec]) -> Self {
        for spec in specs {
            self = self.add_binding_spec(*spec);
        }
        self
    }

    pub fn build(&self, device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&self.bindings);
        unsafe { device.create_descriptor_set_layout(&create_info, None) }
            .context("failed to create descriptor set layout")
    }
}

impl Default for DescriptorLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DescriptorPool {
    pub handle: vk::DescriptorPool,
}

impl DescriptorPool {
    pub fn new(
        device: &ash::Device,
        max_sets: u32,
        pool_sizes: &[vk::DescriptorPoolSize],
    ) -> Result<Self> {
        let create_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(max_sets)
            .pool_sizes(pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let handle = unsafe { device.create_descriptor_pool(&create_info, None) }
            .context("failed to create descriptor pool")?;
        Ok(Self { handle })
    }

    pub fn allocate(
        &self,
        device: &ash::Device,
        layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Vec<vk::DescriptorSet>> {
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.handle)
            .set_layouts(layouts);
        unsafe { device.allocate_descriptor_sets(&alloc_info) }
            .context("failed to allocate descriptor sets")
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_pool(self.handle, None) };
    }
}

#[cfg(test)]
pub(crate) fn assert_specs_match_shader_bindings(
    pass_name: &str,
    specs: &[DescriptorBindingSpec],
    reflection: &crate::assets::shader_reflect::ShaderReflection,
) {
    use crate::assets::shader_reflect::DescriptorKind;

    assert_eq!(
        specs.len(),
        reflection.bindings.len(),
        "{pass_name} descriptor spec count must match shader reflection"
    );
    for spec in specs {
        assert_eq!(
            spec.count, 1,
            "{pass_name} descriptor binding {} must describe exactly one descriptor",
            spec.binding
        );
        assert!(
            spec.stage_flags
                .intersects(vk::ShaderStageFlags::COMPUTE | ray_tracing_shader_stage_flags()),
            "{pass_name} descriptor binding {} must target compute or ray-tracing stages",
            spec.binding
        );
        let expected_kind = match spec.descriptor_type {
            vk::DescriptorType::UNIFORM_BUFFER => DescriptorKind::UniformBuffer,
            vk::DescriptorType::STORAGE_BUFFER => DescriptorKind::StorageBuffer,
            vk::DescriptorType::STORAGE_IMAGE => DescriptorKind::StorageImage,
            vk::DescriptorType::SAMPLED_IMAGE => DescriptorKind::SampledImage,
            vk::DescriptorType::SAMPLER => DescriptorKind::Sampler,
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => DescriptorKind::AccelerationStructure,
            other => panic!("{pass_name} uses unsupported descriptor type {other:?}"),
        };
        assert!(
            reflection.bindings.iter().any(|binding| {
                binding.set == 0 && binding.binding == spec.binding && binding.kind == expected_kind
            }),
            "{pass_name} descriptor binding {} missing from shader reflection",
            spec.binding
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::shader_reflect::{DescriptorBinding, DescriptorKind, ShaderReflection};

    #[test]
    fn compute_descriptor_spec_uses_compute_stage_and_count_one() {
        let spec = DescriptorBindingSpec::compute(6, vk::DescriptorType::UNIFORM_BUFFER);
        assert_eq!(spec.binding, 6);
        assert_eq!(spec.descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
        assert_eq!(spec.stage_flags, vk::ShaderStageFlags::COMPUTE);
        assert_eq!(spec.count, 1);
    }

    #[test]
    fn assert_specs_match_shader_bindings_rejects_wrong_stage_flags() {
        let reflection = ShaderReflection {
            entry_point: "main".to_string(),
            bindings: vec![DescriptorBinding {
                set: 0,
                binding: 0,
                kind: DescriptorKind::UniformBuffer,
                name: "scene_ubo".to_string(),
            }],
        };
        let specs = [DescriptorBindingSpec {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            count: 1,
        }];

        let result = std::panic::catch_unwind(|| {
            assert_specs_match_shader_bindings("test pass", &specs, &reflection);
        });

        assert!(
            result.is_err(),
            "descriptor binding validation must reject wrong stage flags"
        );
    }

    #[test]
    fn assert_specs_match_shader_bindings_rejects_wrong_descriptor_count() {
        let reflection = ShaderReflection {
            entry_point: "main".to_string(),
            bindings: vec![DescriptorBinding {
                set: 0,
                binding: 0,
                kind: DescriptorKind::UniformBuffer,
                name: "scene_ubo".to_string(),
            }],
        };
        let specs = [DescriptorBindingSpec {
            binding: 0,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            count: 4,
        }];

        let result = std::panic::catch_unwind(|| {
            assert_specs_match_shader_bindings("test pass", &specs, &reflection);
        });

        assert!(
            result.is_err(),
            "descriptor binding validation must reject wrong descriptor counts"
        );
    }

    #[test]
    fn assert_specs_match_shader_bindings_accepts_sampled_image_and_sampler_types() {
        let reflection = ShaderReflection {
            entry_point: "main".to_string(),
            bindings: vec![
                DescriptorBinding {
                    set: 0,
                    binding: 0,
                    kind: DescriptorKind::SampledImage,
                    name: "egui_texture".to_string(),
                },
                DescriptorBinding {
                    set: 0,
                    binding: 1,
                    kind: DescriptorKind::Sampler,
                    name: "egui_sampler".to_string(),
                },
            ],
        };
        let specs = [
            DescriptorBindingSpec::compute(0, vk::DescriptorType::SAMPLED_IMAGE),
            DescriptorBindingSpec::compute(1, vk::DescriptorType::SAMPLER),
        ];

        assert_specs_match_shader_bindings("ui pass", &specs, &reflection);
    }

    #[test]
    fn assert_specs_match_shader_bindings_accepts_ray_tracing_and_acceleration_structure_types() {
        let reflection = ShaderReflection {
            entry_point: "main".to_string(),
            bindings: vec![DescriptorBinding {
                set: 0,
                binding: 0,
                kind: DescriptorKind::AccelerationStructure,
                name: "scene_tlas".to_string(),
            }],
        };
        let specs = [DescriptorBindingSpec::ray_tracing(
            0,
            vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        )];

        assert_specs_match_shader_bindings("rt pass", &specs, &reflection);
    }
}
