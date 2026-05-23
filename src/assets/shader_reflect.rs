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
    pub fn from_slang_compiled_or_source(
        entry_point: impl Into<String>,
        source_path: &str,
        source: &str,
    ) -> Result<Self> {
        if let Some(reflection_json_path) = reflection_json_path_for_source(source_path) {
            if reflection_json_path.exists() {
                let reflection_json =
                    std::fs::read_to_string(&reflection_json_path).map_err(|error| {
                        anyhow!(
                            "failed to read Slang reflection JSON {}: {error}",
                            reflection_json_path.display()
                        )
                    })?;
                return Self::from_slang_reflection_json(entry_point, &reflection_json).map_err(
                    |error| {
                        anyhow!(
                            "failed to parse Slang reflection JSON {}: {error}",
                            reflection_json_path.display()
                        )
                    },
                );
            }
        }
        Self::from_slang_source(entry_point, source)
    }

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

    pub fn from_slang_reflection_json(
        entry_point: impl Into<String>,
        reflection_json: &str,
    ) -> Result<Self> {
        let reflection_json = root_json_object(reflection_json)?;
        let parameters = top_level_field(reflection_json, "parameters")
            .ok_or_else(|| anyhow!("Slang reflection JSON missing parameters array"))?;
        let mut bindings = Vec::new();
        for parameter in array_objects(parameters)? {
            let name = string_field(parameter, "name")
                .ok_or_else(|| anyhow!("Slang reflection parameter missing name"))?
                .to_string();
            let binding_object = top_level_field(parameter, "binding")
                .ok_or_else(|| anyhow!("Slang reflection parameter {name} missing binding"))?;
            let binding_kind = string_field(binding_object, "kind")
                .ok_or_else(|| anyhow!("Slang reflection parameter {name} missing binding kind"))?;
            if binding_kind == "pushConstantBuffer" {
                continue;
            }
            if binding_kind != "descriptorTableSlot" {
                return Err(anyhow!(
                    "Slang reflection parameter {name} uses unsupported binding kind {binding_kind:?}"
                ));
            }
            let binding = number_field(binding_object, "index").ok_or_else(|| {
                anyhow!("Slang reflection parameter {name} missing descriptor binding index")
            })? as u32;
            let set = number_field(binding_object, "space")
                .or_else(|| number_field(binding_object, "set"))
                .unwrap_or(0) as u32;
            let kind = descriptor_kind_from_reflection_parameter(parameter).map_err(|error| {
                anyhow!("unsupported Slang reflection parameter {name}: {error}")
            })?;
            bindings.push(DescriptorBinding {
                set,
                binding,
                kind,
                name,
            });
        }

        bindings.sort_by_key(|binding| (binding.set, binding.binding));
        Ok(Self {
            entry_point: entry_point.into(),
            bindings,
        })
    }
}

fn root_json_object(source: &str) -> Result<&str> {
    let trimmed = source.trim();
    if !trimmed.starts_with('{') {
        return Err(anyhow!("Slang reflection JSON must be an object"));
    }
    let end = find_balanced_json_end(trimmed, 0, b'{', b'}')
        .ok_or_else(|| anyhow!("unterminated Slang reflection JSON object"))?;
    if !trimmed[end..].trim().is_empty() {
        return Err(anyhow!("Slang reflection JSON has trailing content"));
    }
    Ok(trimmed)
}

fn descriptor_kind_from_reflection_parameter(parameter: &str) -> Result<DescriptorKind> {
    let ty = top_level_field(parameter, "type").ok_or_else(|| anyhow!("missing type object"))?;
    let type_kind = string_field(ty, "kind").ok_or_else(|| anyhow!("missing type kind"))?;

    match type_kind {
        "constantBuffer" => Ok(DescriptorKind::UniformBuffer),
        "resource" => {
            let base_shape = string_field(ty, "baseShape")
                .ok_or_else(|| anyhow!("resource missing baseShape"))?;
            match base_shape {
                "structuredBuffer" => Ok(DescriptorKind::StorageBuffer),
                "texture2D" | "texture3D" => match string_field(ty, "access") {
                    Some("readWrite") => Ok(DescriptorKind::StorageImage),
                    Some(access) => Err(anyhow!(
                        "texture resource access {access:?} is not supported by descriptor specs"
                    )),
                    None => Err(anyhow!(
                        "texture resource missing access; sampled images are not supported"
                    )),
                },
                other => Err(anyhow!("unsupported resource baseShape {other:?}")),
            }
        }
        other => Err(anyhow!("unsupported type kind {other:?}")),
    }
}

fn top_level_field<'a>(object: &'a str, field: &str) -> Option<&'a str> {
    let mut index = object.find('{')? + 1;
    let end = object.rfind('}')?;
    while index < end {
        index = skip_json_ws_and_commas(object, index);
        if index >= end {
            break;
        }
        if object.as_bytes().get(index) != Some(&b'"') {
            return None;
        }
        let key_end = find_json_string_end(object, index)?;
        let key = &object[index + 1..key_end];
        index = skip_json_ws(object, key_end + 1);
        if object.as_bytes().get(index) != Some(&b':') {
            return None;
        }
        index = skip_json_ws(object, index + 1);
        let value_end = find_json_value_end(object, index)?;
        if key == field {
            return Some(object[index..value_end].trim());
        }
        index = value_end;
    }
    None
}

fn string_field<'a>(object: &'a str, field: &str) -> Option<&'a str> {
    let value = top_level_field(object, field)?.trim();
    if !value.starts_with('"') {
        return None;
    }
    let end = find_json_string_end(value, 0)?;
    Some(&value[1..end])
}

fn number_field(object: &str, field: &str) -> Option<u64> {
    top_level_field(object, field)?.trim().parse().ok()
}

fn array_objects(array: &str) -> Result<Vec<&str>> {
    let array = array.trim();
    if !array.starts_with('[') || !array.ends_with(']') {
        return Err(anyhow!("expected JSON array"));
    }
    let mut objects = Vec::new();
    let mut index = 1usize;
    let end = array.len() - 1;
    while index < end {
        index = skip_json_ws_and_commas(array, index);
        if index >= end {
            break;
        }
        if array.as_bytes().get(index) != Some(&b'{') {
            return Err(anyhow!("expected object in JSON array"));
        }
        let value_end = find_json_value_end(array, index)
            .ok_or_else(|| anyhow!("unterminated object in JSON array"))?;
        objects.push(&array[index..value_end]);
        index = value_end;
    }
    Ok(objects)
}

fn skip_json_ws_and_commas(source: &str, mut index: usize) -> usize {
    while let Some(byte) = source.as_bytes().get(index) {
        if byte.is_ascii_whitespace() || *byte == b',' {
            index += 1;
        } else {
            break;
        }
    }
    index
}

fn skip_json_ws(source: &str, mut index: usize) -> usize {
    while source
        .as_bytes()
        .get(index)
        .is_some_and(|byte| byte.is_ascii_whitespace())
    {
        index += 1;
    }
    index
}

fn find_json_string_end(source: &str, start: usize) -> Option<usize> {
    if source.as_bytes().get(start) != Some(&b'"') {
        return None;
    }
    let mut escaped = false;
    for index in start + 1..source.len() {
        let byte = source.as_bytes()[index];
        if escaped {
            escaped = false;
            continue;
        }
        match byte {
            b'\\' => escaped = true,
            b'"' => return Some(index),
            _ => {}
        }
    }
    None
}

fn find_json_value_end(source: &str, start: usize) -> Option<usize> {
    match source.as_bytes().get(start).copied()? {
        b'"' => find_json_string_end(source, start).map(|index| index + 1),
        b'{' => find_balanced_json_end(source, start, b'{', b'}'),
        b'[' => find_balanced_json_end(source, start, b'[', b']'),
        _ => {
            let mut index = start;
            while let Some(byte) = source.as_bytes().get(index) {
                if *byte == b',' || *byte == b'}' || *byte == b']' {
                    break;
                }
                index += 1;
            }
            Some(index)
        }
    }
}

fn find_balanced_json_end(source: &str, start: usize, open: u8, close: u8) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for index in start..source.len() {
        let byte = source.as_bytes()[index];
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        if byte == b'"' {
            in_string = true;
        } else if byte == open {
            depth += 1;
        } else if byte == close {
            depth = depth.checked_sub(1)?;
            if depth == 0 {
                return Some(index + 1);
            }
        }
    }
    None
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

fn reflection_json_path_for_source(source_path: &str) -> Option<std::path::PathBuf> {
    let path = std::path::Path::new(source_path);
    let parent = path.parent()?;
    if parent.file_name()?.to_str()? != "passes" {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    Some(
        std::path::Path::new(env!("OUT_DIR"))
            .join("shaders")
            .join(format!("{stem}.reflection.json")),
    )
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

    #[test]
    fn parses_slang_reflection_json_bindings() {
        let json = r#"
{
  "parameters": [
    {
      "name": "scene_ubo",
      "binding": { "kind": "descriptorTableSlot", "index": 0 },
      "type": { "kind": "constantBuffer" }
    },
    {
      "name": "output_image",
      "binding": { "kind": "descriptorTableSlot", "index": 1 },
      "type": { "kind": "resource", "baseShape": "texture2D", "access": "readWrite" }
    },
    {
      "name": "reservoirs",
      "binding": { "kind": "descriptorTableSlot", "index": 2 },
      "type": { "kind": "resource", "baseShape": "structuredBuffer" }
    }
  ]
}
"#;

        let reflection = ShaderReflection::from_slang_reflection_json("main", json).unwrap();

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
                    name: "reservoirs".to_string(),
                },
            ]
        );
    }

    #[test]
    fn ignores_push_constant_buffers_in_reflection_json_bindings() {
        let json = r#"
{
  "parameters": [
    {
      "name": "scene_ubo",
      "binding": { "kind": "descriptorTableSlot", "index": 0 },
      "type": { "kind": "constantBuffer" }
    },
    {
      "name": "vpt_motion_guide",
      "binding": { "kind": "pushConstantBuffer", "index": 0 },
      "type": { "kind": "constantBuffer" }
    }
  ]
}
"#;

        let reflection = ShaderReflection::from_slang_reflection_json("main", json).unwrap();

        assert_eq!(
            reflection.bindings,
            vec![DescriptorBinding {
                set: 0,
                binding: 0,
                kind: DescriptorKind::UniformBuffer,
                name: "scene_ubo".to_string(),
            }]
        );
    }

    #[test]
    fn rejects_reflection_json_trailing_content() {
        let json = r#"{ "parameters": [] } trailing"#;

        let error = ShaderReflection::from_slang_reflection_json("main", json).unwrap_err();
        assert!(error.to_string().contains("trailing content"));
    }

    #[test]
    fn rejects_sampled_texture_reflection_json() {
        let json = r#"
{
  "parameters": [
    {
      "name": "sampled_image",
      "binding": { "kind": "descriptorTableSlot", "index": 1 },
      "type": { "kind": "resource", "baseShape": "texture2D" }
    }
  ]
}
"#;

        let error = ShaderReflection::from_slang_reflection_json("main", json).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported Slang reflection parameter sampled_image")
        );
    }
}
