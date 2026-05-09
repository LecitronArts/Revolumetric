use anyhow::{Result, anyhow};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderReflection {
    pub entry_point: String,
    pub bindings: Vec<DescriptorBinding>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DescriptorBinding {
    pub set: u32,
    pub binding: u32,
    pub kind: DescriptorKind,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    UniformBuffer,
    StorageBuffer,
    StorageImage,
}

impl ShaderReflection {
    pub fn from_slang_source(entry_point: impl Into<String>, source: &str) -> Result<Self> {
        let mut bindings = Vec::new();
        let mut pending_binding = None;

        for (line_index, raw_line) in source.lines().enumerate() {
            let line = raw_line.split("//").next().unwrap_or_default().trim();
            if line.is_empty() {
                continue;
            }

            if let Some((binding, set)) = parse_vk_binding(line) {
                pending_binding = Some((binding, set, line_index + 1));
                continue;
            }

            if line.starts_with("[[") {
                continue;
            }

            if let Some((binding, set, binding_line)) = pending_binding.take() {
                let (kind, name) = parse_descriptor_declaration(line).ok_or_else(|| {
                    anyhow!(
                        "unsupported descriptor declaration after vk::binding at line {}: {}",
                        binding_line,
                        line
                    )
                })?;
                bindings.push(DescriptorBinding {
                    set,
                    binding,
                    kind,
                    name,
                });
            }
        }

        if let Some((_binding, _set, binding_line)) = pending_binding {
            return Err(anyhow!(
                "vk::binding at line {} has no following descriptor declaration",
                binding_line
            ));
        }

        bindings.sort_by_key(|binding| (binding.set, binding.binding));
        Ok(Self {
            entry_point: entry_point.into(),
            bindings,
        })
    }
}

fn parse_vk_binding(line: &str) -> Option<(u32, u32)> {
    let prefix = "[[vk::binding(";
    let start = line.find(prefix)? + prefix.len();
    let end = line[start..].find(")]]")? + start;
    let mut parts = line[start..end].split(',').map(str::trim);
    let binding = parts.next()?.parse().ok()?;
    let set = parts.next()?.parse().ok()?;
    Some((binding, set))
}

fn parse_descriptor_declaration(line: &str) -> Option<(DescriptorKind, String)> {
    let kind = if line.starts_with("ConstantBuffer<") {
        DescriptorKind::UniformBuffer
    } else if line.starts_with("StructuredBuffer<") || line.starts_with("RWStructuredBuffer<") {
        DescriptorKind::StorageBuffer
    } else if line.starts_with("RWTexture2D<") || line.starts_with("RWTexture3D<") {
        DescriptorKind::StorageImage
    } else {
        return None;
    };
    let name = line
        .trim_end_matches(';')
        .split_whitespace()
        .last()?
        .trim()
        .to_string();
    Some((kind, name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_slang_descriptor_bindings() {
        let source = r#"
[[vk::binding(0, 0)]]
ConstantBuffer<SceneUniforms> scene_ubo;

[[vk::binding(1, 0)]]
RWTexture2D<float4> output_image;

[[vk::binding(2, 0)]]
StructuredBuffer<NodeL0> hierarchy_l0;
"#;

        let reflection = ShaderReflection::from_slang_source("main", source).unwrap();

        assert_eq!(reflection.entry_point, "main");
        assert_eq!(
            reflection.bindings,
            vec![
                DescriptorBinding {
                    set: 0,
                    binding: 0,
                    kind: DescriptorKind::UniformBuffer,
                    name: "scene_ubo".to_string(),
                },
                DescriptorBinding {
                    set: 0,
                    binding: 1,
                    kind: DescriptorKind::StorageImage,
                    name: "output_image".to_string(),
                },
                DescriptorBinding {
                    set: 0,
                    binding: 2,
                    kind: DescriptorKind::StorageBuffer,
                    name: "hierarchy_l0".to_string(),
                },
            ]
        );
    }

    #[test]
    fn rejects_unknown_descriptor_declaration() {
        let source = r#"
[[vk::binding(3, 0)]]
SamplerState sampler_state;
"#;

        let error = ShaderReflection::from_slang_source("main", source).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported descriptor declaration")
        );
    }
}
