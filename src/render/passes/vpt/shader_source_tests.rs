use crate::assets::shader_reflect::{DescriptorBinding, DescriptorKind, ShaderReflection};
use crate::render::passes::vpt_atrous::VptAtrousPass;
use crate::render::passes::vpt_nrd_confidence::VptNrdConfidencePass;
use crate::render::passes::vpt_nrd_frontend::VptNrdFrontendPass;
use crate::render::passes::vpt_nrd_resolve::VptNrdResolvePass;
use crate::render::passes::vpt_surface::VptSurfacePass;
use crate::render::passes::vpt_temporal::VptTemporalPass;

use super::VptPass;

fn normalized_source(path_source: &str) -> String {
    crate::render::source_checks::normalize(path_source)
}

fn source(path: &str) -> String {
    crate::render::source_checks::read_source(path)
}

fn binding(binding: u32, kind: DescriptorKind, name: &str) -> DescriptorBinding {
    DescriptorBinding {
        set: 0,
        binding,
        kind,
        name: name.to_string(),
    }
}

fn shader_reflection(path: &str) -> ShaderReflection {
    ShaderReflection::from_slang_compiled_or_source("main", path, &source(path))
        .expect("shader reflection should parse")
}

fn shader_bindings(path: &str) -> Vec<DescriptorBinding> {
    shader_reflection(path).bindings
}

#[test]
fn vpt_descriptor_specs_match_shader_manifest() {
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "VPT trace",
        &VptPass::descriptor_binding_specs(),
        &shader_reflection("assets/shaders/passes/vpt.slang"),
    );
}

#[test]
fn vpt_svgf_descriptor_specs_match_shader_manifests() {
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "VPT surface",
        &VptSurfacePass::descriptor_binding_specs(),
        &shader_reflection("assets/shaders/passes/vpt_surface.slang"),
    );
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "VPT temporal",
        &VptTemporalPass::descriptor_binding_specs(),
        &shader_reflection("assets/shaders/passes/vpt_temporal.slang"),
    );
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "VPT A-trous",
        &VptAtrousPass::descriptor_binding_specs(),
        &shader_reflection("assets/shaders/passes/vpt_atrous.slang"),
    );
    crate::render::descriptor::assert_specs_match_shader_bindings(
        "VPT NRD resolve",
        &VptNrdResolvePass::descriptor_binding_specs(),
        &shader_reflection("assets/shaders/passes/vpt_nrd_resolve.slang"),
    );
}

#[test]
fn vpt_shader_binding_manifest_matches_expected_resources() {
    assert_eq!(
        shader_bindings("assets/shaders/passes/vpt.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "scene_ubo"),
            binding(1, DescriptorKind::StorageImage, "noisy_radiance_image"),
            binding(2, DescriptorKind::UniformBuffer, "ucvh_config"),
            binding(3, DescriptorKind::StorageBuffer, "hierarchy_l0"),
            binding(4, DescriptorKind::StorageBuffer, "hierarchy_l1"),
            binding(5, DescriptorKind::StorageBuffer, "hierarchy_l2"),
            binding(6, DescriptorKind::StorageBuffer, "hierarchy_l3"),
            binding(7, DescriptorKind::StorageBuffer, "hierarchy_l4"),
            binding(8, DescriptorKind::StorageBuffer, "brick_occupancy"),
            binding(9, DescriptorKind::StorageBuffer, "brick_materials"),
            binding(10, DescriptorKind::UniformBuffer, "restir"),
            binding(11, DescriptorKind::StorageBuffer, "restir_reservoirs"),
            binding(12, DescriptorKind::StorageImage, "noisy_moments_image"),
            binding(13, DescriptorKind::UniformBuffer, "area_restir"),
            binding(14, DescriptorKind::StorageBuffer, "area_restir_reservoirs"),
            binding(
                15,
                DescriptorKind::StorageImage,
                "nrd_diff_radiance_hitdist"
            ),
            binding(
                16,
                DescriptorKind::StorageImage,
                "nrd_spec_radiance_hitdist"
            ),
            binding(17, DescriptorKind::StorageImage, "nrd_residual_radiance"),
            binding(18, DescriptorKind::StorageImage, "nrd_material_factors"),
            binding(19, DescriptorKind::StorageBuffer, "traversal_stats"),
        ]
    );
}

#[test]
fn material_common_declares_deterministic_roughness_helpers() {
    let material = source("assets/shaders/shared/material_common.slang");

    for token in [
        "static const float MATERIAL_ROUGHNESS[8]",
        "float material_roughness(uint material_id)",
        "float material_cell_roughness(VoxelCell cell)",
        "float material_emissive_luminance(VoxelCell cell)",
        "return MATERIAL_ROUGHNESS[min(material_id, 7u)]",
        "return material_roughness(voxel_material(cell));",
        "return dot(material_emissive(cell), float3(0.2126, 0.7152, 0.0722));",
    ] {
        assert!(material.contains(token), "material common missing {token}");
    }
}

#[test]
fn vpt_surface_writes_explicit_material_roughness_guide() {
    let shader = source("assets/shaders/passes/vpt_surface.slang");
    let rust = source("src/render/passes/vpt_surface.rs");
    let compact_rust = rust.split_whitespace().collect::<String>();
    let temporal = source("src/render/passes/vpt_temporal.rs");
    let compact_temporal = temporal.split_whitespace().collect::<String>();

    for token in [
        "RWTexture2D<float> surface_material_roughness",
        "RWTexture2D<float> surface_view_z",
        "surface_material_roughness[pixel] = material_cell_roughness(hit.cell);",
        "surface_material_roughness[pixel] = 1.0;",
        "material_emissive_luminance(hit.cell)",
    ] {
        assert!(shader.contains(token), "VPT surface shader missing {token}");
    }

    for token in [
        "pub surface_material_roughness: GpuImage",
        "pub previous_surface_material_roughness: GpuImage",
        "pub surface_writes: VptCurrentSurfaceResources",
        "pub previous_surface_resources: VptPreviousSurfaceResources",
        "vpt_surface_material_roughness",
        "vpt_previous_surface_material_roughness",
        "vpt_surface_view_z",
        "vpt_previous_surface_view_z",
    ] {
        assert!(rust.contains(token), "VPT surface pass missing {token}");
    }

    for token in [
        "letoutput_images=[images.surface_position_depth,images.surface_normal_roughness,images.surface_albedo_material,images.surface_material_roughness,images.surface_view_z,images.surface_motion_id,images.motion_history,];",
        "letsurface_writes=VptCurrentSurfaceResources::from_graph_writes(bootstrap_writes.iter().copied().take(9).collect(),);",
        "surface_writes.position_depth,self.surface_position_depth.handle,",
        "graph.bind_image(surface_writes.material_roughness,self.surface_material_roughness.handle,);",
        "graph.bind_image(surface_writes.view_z,self.surface_view_z.handle);",
        "graph.bind_image(surface_writes.motion_id,self.surface_motion_id.handle);",
        "graph.bind_image(surface_writes.brick_generation,self.surface_brick_generation.handle,);",
        "surface_writes,previous_surface_resources:VptPreviousSurfaceResources",
        "previous_surface_resources:VptPreviousSurfaceResources{position_depth:previous_surface_position_resource,normal_roughness:previous_surface_normal_resource,albedo_material:previous_surface_albedo_resource,material_roughness:previous_surface_material_roughness_resource,view_z:previous_surface_view_z_resource,motion_id:previous_surface_motion_id_resource,brick_generation:previous_surface_brick_generation_resource,}",
        "copy_surface_image(device,cmd,&self.surface_material_roughness,&self.previous_surface_material_roughness,);copy_surface_image(device,cmd,&self.surface_view_z,&self.previous_surface_view_z,);copy_surface_image(device,cmd,&self.surface_motion_id,&self.previous_surface_motion_id,);copy_surface_image(device,cmd,&self.surface_brick_generation,&self.previous_surface_brick_generation,);",
    ] {
        assert!(
            compact_rust.contains(token),
            "VPT surface pass order contract missing {token}"
        );
    }

    for token in [
        "builder.read_as(surface_inputs.position_depth,AccessKind::TransferRead);",
        "builder.read_as(surface_inputs.material_roughness,AccessKind::TransferRead);",
        "builder.read_as(surface_inputs.view_z,AccessKind::TransferRead);",
        "builder.read_as(surface_inputs.motion_id,AccessKind::TransferRead);",
        "builder.read_as(surface_inputs.brick_generation,AccessKind::TransferRead);",
        "previous_surface_inputs.position_depth,AccessKind::TransferWrite,",
        "previous_surface_inputs.material_roughness,AccessKind::TransferWrite,",
        "builder.write_as(previous_surface_inputs.view_z,AccessKind::TransferWrite);",
        "builder.write_as(previous_surface_inputs.motion_id,AccessKind::TransferWrite);",
        "previous_surface_inputs.brick_generation,AccessKind::TransferWrite,",
    ] {
        assert!(
            compact_temporal.contains(token),
            "VPT surface history update order contract missing {token}"
        );
    }
}

#[test]
fn vpt_surface_writes_independent_view_z_guide() {
    let scene_common = source("assets/shaders/shared/scene_common.slang");
    let surface = source("assets/shaders/passes/vpt_surface.slang");

    for token in [
        "static const float VPT_VIEW_Z_MISS_SENTINEL = 1.0e20;",
        "float scene_view_z(float3 world_position, SceneUniforms scene)",
        "return max(dot(world_position - scene.pixel_to_ray[3].xyz, normalize(scene.camera_forward)), 0.0);",
    ] {
        assert!(scene_common.contains(token), "scene common missing {token}");
    }

    for token in [
        "RWTexture2D<float> surface_view_z",
        "surface_view_z[pixel] = VPT_VIEW_Z_MISS_SENTINEL;",
        "float view_z = scene_view_z(hit.position, scene);",
        "surface_view_z[pixel] = view_z;",
        "surface_position_depth[pixel] = float4(hit.position, hit.t);",
        "motion.z = previous_view_z - view_z;",
    ] {
        assert!(
            surface.contains(token),
            "VPT surface shader missing {token}"
        );
    }

    assert!(
        !surface.contains("surface_position_depth[pixel] = float4(hit.position, view_z);"),
        "viewZ must not replace the legacy hit.t lane in surface_position_depth"
    );
}

#[test]
fn vpt_material_roughness_guide_uses_single_channel_storage() {
    let surface = source("assets/shaders/passes/vpt_surface.slang");
    let temporal = source("assets/shaders/passes/vpt_temporal.slang");
    let atrous = source("assets/shaders/passes/vpt_atrous.slang");
    let surface_rs = source("src/render/passes/vpt_surface.rs");

    for token in [
        "[[vk::image_format(\"r16f\")]]",
        "RWTexture2D<float> surface_material_roughness",
        "surface_material_roughness[pixel] = material_cell_roughness(hit.cell);",
        "surface_material_roughness[pixel] = 1.0;",
    ] {
        assert!(
            surface.contains(token),
            "surface roughness guide missing {token}"
        );
    }

    for token in [
        "RWTexture2D<float> surface_material_roughness",
        "RWTexture2D<float> previous_surface_material_roughness",
        "float roughness_delta = abs(surface_material_roughness[pixel] - previous_surface_material_roughness[previous_pixel]);",
    ] {
        assert!(
            temporal.contains(token),
            "temporal roughness guide missing {token}"
        );
    }

    for token in [
        "RWTexture2D<float> surface_material_roughness",
        "float center_material_roughness = surface_material_roughness[pixel];",
        "float neighbor_material_roughness = surface_material_roughness[neighbor_pixel];",
    ] {
        assert!(
            atrous.contains(token),
            "A-trous roughness guide missing {token}"
        );
    }

    assert!(surface_rs.contains("vk::Format::R16_SFLOAT"));
    assert!(
        !surface.contains("RWTexture2D<float4> surface_material_roughness"),
        "roughness guide must not allocate unused RGBA lanes"
    );
}

#[test]
fn vpt_surface_view_z_resource_graph_contract_is_ordered() {
    let surface_rs = source("src/render/passes/vpt_surface.rs");
    let temporal_rs = source("src/render/passes/vpt_temporal.rs");
    let atrous_rs = source("src/render/passes/vpt_atrous.rs");
    let area_rs = source("src/render/passes/area_restir.rs");
    let restir_rs = source("src/render/passes/restir_di.rs");
    let compact_surface = surface_rs.split_whitespace().collect::<String>();
    let compact_temporal = temporal_rs.split_whitespace().collect::<String>();

    for token in [
        "pub surface_view_z: GpuImage",
        "pub previous_surface_view_z: GpuImage",
        "pub surface_motion_id: GpuImage",
        "pub previous_surface_motion_id: GpuImage",
        "pub surface_writes: VptCurrentSurfaceResources",
        "pub previous_surface_resources: VptPreviousSurfaceResources",
        "vpt_surface_view_z",
        "vpt_previous_surface_view_z",
        "vpt_surface_motion_id",
        "vpt_previous_surface_motion_id",
        "descriptor_count: 18 * frame_count as u32",
        "vk::Format::R32_SFLOAT",
        "vk::Format::R32_UINT",
    ] {
        assert!(
            surface_rs.contains(token),
            "VPT surface pass missing {token}"
        );
    }

    for token in [
        "surface_view_z_resource",
        "surface_motion_id_resource",
        "previous_surface_view_z_resource",
        "previous_surface_motion_id_resource",
        "letsurface_writes=VptCurrentSurfaceResources::from_graph_writes(bootstrap_writes.iter().copied().take(9).collect(),);",
        "surface_writes.position_depth,self.surface_position_depth.handle,",
        "graph.bind_image(surface_writes.view_z,self.surface_view_z.handle);",
        "graph.bind_image(surface_writes.motion_id,self.surface_motion_id.handle);",
        "graph.bind_image(surface_writes.brick_generation,self.surface_brick_generation.handle,);",
        "previous_surface_resources:VptPreviousSurfaceResources{position_depth:previous_surface_position_resource,normal_roughness:previous_surface_normal_resource,albedo_material:previous_surface_albedo_resource,material_roughness:previous_surface_material_roughness_resource,view_z:previous_surface_view_z_resource,motion_id:previous_surface_motion_id_resource,brick_generation:previous_surface_brick_generation_resource,}",
        "copy_surface_image(device,cmd,&self.surface_material_roughness,&self.previous_surface_material_roughness,);copy_surface_image(device,cmd,&self.surface_view_z,&self.previous_surface_view_z,);copy_surface_image(device,cmd,&self.surface_motion_id,&self.previous_surface_motion_id,);copy_surface_image(device,cmd,&self.surface_brick_generation,&self.previous_surface_brick_generation,);",
    ] {
        assert!(
            compact_surface.contains(token),
            "surface graph order missing {token}"
        );
    }

    for token in [
        "builder.read_as(surface_inputs.position_depth,AccessKind::TransferRead);",
        "builder.read_as(surface_inputs.view_z,AccessKind::TransferRead);",
        "builder.read_as(surface_inputs.motion_id,AccessKind::TransferRead);",
        "builder.read_as(surface_inputs.brick_generation,AccessKind::TransferRead);",
        "previous_surface_inputs.position_depth,AccessKind::TransferWrite,",
        "builder.write_as(previous_surface_inputs.view_z,AccessKind::TransferWrite);",
        "builder.write_as(previous_surface_inputs.motion_id,AccessKind::TransferWrite);",
        "previous_surface_inputs.brick_generation,AccessKind::TransferWrite,",
    ] {
        assert!(
            compact_temporal.contains(token),
            "history copy order missing {token}"
        );
    }

    for token in [
        "surface_inputs: VptCurrentSurfaceResources",
        "previous_surface_inputs: VptPreviousSurfaceResources",
        "surface_inputs.view_z",
        "previous_surface_inputs.view_z",
        "surface_inputs.motion_id",
        "previous_surface_inputs.motion_id",
    ] {
        assert!(
            temporal_rs.contains(token),
            "temporal graph missing {token}"
        );
    }
    assert!(atrous_rs.contains("surface_inputs: VptCurrentSurfaceResources"));
    assert!(area_rs.contains("bootstrap_surface_writes: VptCurrentSurfaceResources"));
    assert!(area_rs.contains("previous_surface_resources: VptPreviousSurfaceResources"));
    assert!(restir_rs.contains("final_surface_writes: VptCurrentSurfaceResources"));
    assert!(restir_rs.contains("previous_surface_resources: VptPreviousSurfaceResources"));
}

#[test]
fn vpt_surface_graph_uses_named_resources_instead_of_magic_arrays() {
    let surface_rs = source("src/render/passes/vpt_surface.rs");
    let temporal_rs = source("src/render/passes/vpt_temporal.rs");
    let atrous_rs = source("src/render/passes/vpt_atrous.rs");
    let area_rs = source("src/render/passes/area_restir.rs");
    let restir_rs = source("src/render/passes/restir_di.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");

    for token in [
        "pub struct VptCurrentSurfaceResources",
        "pub struct VptPreviousSurfaceResources",
        "pub position_depth: ResourceHandle",
        "pub material_roughness: ResourceHandle",
        "pub view_z: ResourceHandle",
        "pub motion_id: ResourceHandle",
        "pub brick_generation: ResourceHandle",
    ] {
        assert!(
            surface_rs.contains(token),
            "surface resources missing {token}"
        );
    }

    for source in [
        surface_rs.as_str(),
        temporal_rs.as_str(),
        atrous_rs.as_str(),
        area_rs.as_str(),
        restir_rs.as_str(),
    ] {
        for forbidden in [
            "[ResourceHandle; 8]",
            "[ResourceHandle; 9]",
            "[ResourceHandle; 6]",
            "[ResourceHandle; 7]",
            "surface_inputs[",
            "previous_surface_inputs[",
            "final_surface_writes[",
            "bootstrap_surface_writes[",
            "bootstrap_writes[",
            "selected_surface_writes[",
        ] {
            assert!(
                !source.contains(forbidden),
                "surface graph code must not use magic resource array token {forbidden}"
            );
        }
    }

    for token in [
        "surface_inputs.position_depth",
        "surface_inputs.material_roughness",
        "surface_inputs.view_z",
        "surface_inputs.motion_id",
        "surface_inputs.brick_generation",
        "previous_surface_inputs.view_z",
        "previous_surface_inputs.motion_id",
        "final_surface_writes.material_roughness",
        "bootstrap_surface_writes.position_depth",
        "area_graph.final_surface_writes",
    ] {
        assert!(
            temporal_rs.contains(token)
                || atrous_rs.contains(token)
                || area_rs.contains(token)
                || restir_rs.contains(token)
                || pipeline.contains(token),
            "named surface resource token missing {token}"
        );
    }
}

#[test]
fn vpt_surface_shader_binding_manifest_matches_expected_resources() {
    assert_eq!(
        shader_bindings("assets/shaders/passes/vpt_surface.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "scene_ubo"),
            binding(1, DescriptorKind::StorageImage, "surface_position_depth"),
            binding(2, DescriptorKind::StorageImage, "surface_normal_roughness"),
            binding(3, DescriptorKind::StorageImage, "surface_albedo_material"),
            binding(
                4,
                DescriptorKind::StorageImage,
                "surface_material_roughness"
            ),
            binding(5, DescriptorKind::StorageImage, "surface_view_z"),
            binding(6, DescriptorKind::StorageImage, "surface_motion_id"),
            binding(7, DescriptorKind::StorageImage, "motion_history"),
            binding(8, DescriptorKind::UniformBuffer, "ucvh_config"),
            binding(9, DescriptorKind::StorageBuffer, "hierarchy_l0"),
            binding(10, DescriptorKind::StorageBuffer, "hierarchy_l1"),
            binding(11, DescriptorKind::StorageBuffer, "hierarchy_l2"),
            binding(12, DescriptorKind::StorageBuffer, "hierarchy_l3"),
            binding(13, DescriptorKind::StorageBuffer, "hierarchy_l4"),
            binding(14, DescriptorKind::StorageBuffer, "brick_occupancy"),
            binding(15, DescriptorKind::StorageBuffer, "brick_materials"),
            binding(16, DescriptorKind::UniformBuffer, "vpt_history"),
            binding(17, DescriptorKind::UniformBuffer, "area_restir"),
            binding(18, DescriptorKind::StorageBuffer, "area_restir_reservoirs"),
            binding(19, DescriptorKind::StorageImage, "motion_flags"),
            binding(20, DescriptorKind::StorageImage, "surface_brick_generation"),
            binding(21, DescriptorKind::StorageBuffer, "brick_generations"),
            binding(22, DescriptorKind::StorageBuffer, "ucvh_motion_events"),
            binding(23, DescriptorKind::StorageBuffer, "traversal_stats"),
        ]
    );
}

#[test]
fn vpt_temporal_shader_binding_manifest_matches_motion_guide_resources() {
    assert_eq!(
        shader_bindings("assets/shaders/passes/vpt_temporal.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "scene_ubo"),
            binding(1, DescriptorKind::StorageImage, "noisy_radiance_image"),
            binding(2, DescriptorKind::StorageImage, "noisy_moments_image"),
            binding(3, DescriptorKind::StorageImage, "surface_position_depth"),
            binding(4, DescriptorKind::StorageImage, "surface_normal_roughness"),
            binding(5, DescriptorKind::StorageImage, "surface_albedo_material"),
            binding(
                6,
                DescriptorKind::StorageImage,
                "surface_material_roughness"
            ),
            binding(7, DescriptorKind::StorageImage, "surface_view_z"),
            binding(
                8,
                DescriptorKind::StorageImage,
                "previous_surface_position_depth"
            ),
            binding(
                9,
                DescriptorKind::StorageImage,
                "previous_surface_normal_roughness"
            ),
            binding(
                10,
                DescriptorKind::StorageImage,
                "previous_surface_albedo_material"
            ),
            binding(
                11,
                DescriptorKind::StorageImage,
                "previous_surface_material_roughness"
            ),
            binding(12, DescriptorKind::StorageImage, "motion_history"),
            binding(
                13,
                DescriptorKind::StorageImage,
                "accumulated_radiance_image"
            ),
            binding(
                14,
                DescriptorKind::StorageImage,
                "accumulated_moments_history_image",
            ),
            binding(
                15,
                DescriptorKind::StorageImage,
                "previous_accumulated_radiance_image"
            ),
            binding(
                16,
                DescriptorKind::StorageImage,
                "previous_accumulated_moments_history_image",
            ),
            binding(17, DescriptorKind::StorageImage, "motion_flags"),
            binding(18, DescriptorKind::StorageImage, "surface_brick_generation"),
            binding(
                19,
                DescriptorKind::StorageImage,
                "previous_surface_brick_generation",
            ),
        ]
    );
}

#[test]
fn vpt_temporal_and_atrous_consume_material_roughness_guide() {
    let temporal = source("assets/shaders/passes/vpt_temporal.slang");
    let atrous = source("assets/shaders/passes/vpt_atrous.slang");
    let temporal_rs = source("src/render/passes/vpt_temporal.rs");
    let atrous_rs = source("src/render/passes/vpt_atrous.rs");
    let compact_temporal_rs = temporal_rs.split_whitespace().collect::<String>();
    let compact_atrous_rs = atrous_rs.split_whitespace().collect::<String>();

    for token in [
        "RWTexture2D<float> surface_material_roughness",
        "RWTexture2D<float> previous_surface_material_roughness",
        "roughness_delta",
        "surface_material_roughness[pixel]",
        "previous_surface_material_roughness[previous_pixel]",
    ] {
        assert!(temporal.contains(token), "temporal shader missing {token}");
    }

    for token in [
        "RWTexture2D<float> surface_material_roughness",
        "float roughness_weight(",
        "surface_material_roughness[neighbor_pixel]",
        "center_material_roughness",
    ] {
        assert!(atrous.contains(token), "A-trous shader missing {token}");
    }

    assert!(temporal_rs.contains("surface_inputs: VptCurrentSurfaceResources"));
    assert!(temporal_rs.contains("previous_surface_inputs: VptPreviousSurfaceResources"));
    assert!(atrous_rs.contains("surface_inputs: VptCurrentSurfaceResources"));

    for token in [
        "DescriptorBindingSpec::compute(19,vk::DescriptorType::STORAGE_IMAGE),",
        "letimage_refs=[&vpt.noisy_radiance_image,&vpt.noisy_moments_image,&vpt_surface.surface_position_depth,&vpt_surface.surface_normal_roughness,&vpt_surface.surface_albedo_material,&vpt_surface.surface_material_roughness,&vpt_surface.surface_view_z,&vpt_surface.previous_surface_position_depth,&vpt_surface.previous_surface_normal_roughness,&vpt_surface.previous_surface_albedo_material,&vpt_surface.previous_surface_material_roughness,&vpt_surface.motion_history,temporal.accumulated_radiance,temporal.accumulated_moments_history,temporal.previous_accumulated_radiance,temporal.previous_accumulated_moments_history,&vpt_surface.motion_flags,&vpt_surface.surface_brick_generation,&vpt_surface.previous_surface_brick_generation,];",
    ] {
        assert!(
            compact_temporal_rs.contains(token),
            "temporal Rust roughness descriptor order missing {token}"
        );
    }

    for token in [
        "surface_inputs.position_depth",
        "surface_inputs.normal_roughness",
        "surface_inputs.albedo_material",
        "surface_inputs.material_roughness",
        "letimage_refs=[info.input_radiance,&info.temporal.accumulated_moments_history,&info.vpt_surface.surface_position_depth,&info.vpt_surface.surface_normal_roughness,&info.vpt_surface.surface_albedo_material,&info.vpt_surface.surface_material_roughness,info.output_radiance,];",
        ".dst_binding(8).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)",
    ] {
        assert!(
            compact_atrous_rs.contains(token),
            "A-trous Rust roughness descriptor order missing {token}"
        );
    }
}

#[test]
fn vpt_atrous_shader_binding_manifest_matches_roughness_guide_resources() {
    assert_eq!(
        shader_bindings("assets/shaders/passes/vpt_atrous.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "scene_ubo"),
            binding(1, DescriptorKind::StorageImage, "input_radiance_image"),
            binding(2, DescriptorKind::StorageImage, "moments_history_image"),
            binding(3, DescriptorKind::StorageImage, "surface_position_depth"),
            binding(4, DescriptorKind::StorageImage, "surface_normal_roughness"),
            binding(5, DescriptorKind::StorageImage, "surface_albedo_material"),
            binding(
                6,
                DescriptorKind::StorageImage,
                "surface_material_roughness"
            ),
            binding(7, DescriptorKind::StorageImage, "filtered_radiance_image"),
            binding(8, DescriptorKind::UniformBuffer, "vpt_atrous"),
        ]
    );
}

#[test]
fn phase4_motion_guide_shader_contract_is_declared() {
    let surface = source("assets/shaders/passes/vpt_surface.slang");
    let temporal = source("assets/shaders/passes/vpt_temporal.slang");
    let restir = source("assets/shaders/passes/restir_di_temporal.slang");
    let area = source("assets/shaders/passes/area_restir_temporal.slang");
    let motion_common = source("assets/shaders/shared/vpt_motion_common.slang");

    for token in [
        "VPT_MOTION_FLAG_HISTORY_VALID",
        "VPT_MOTION_FLAG_CAMERA_STATIC",
        "VPT_MOTION_FLAG_UCVH_REGION_MOVE",
        "VPT_MOTION_FLAG_DISOCCLUDED",
        "VPT_MOTION_FLAG_HISTORY_RESET",
        "VPT_MOTION_FLAG_BEHIND_CAMERA",
        "VPT_NO_BRICK_GENERATION",
    ] {
        assert!(
            motion_common.contains(token),
            "motion common missing {token}"
        );
    }

    for token in [
        "#include \"vpt_motion_common.slang\"",
        "RWTexture2D<uint> motion_flags",
        "RWTexture2D<uint> surface_brick_generation",
        "StructuredBuffer<uint> brick_generations",
        "StructuredBuffer<UcvhMotionEvent> ucvh_motion_events",
        "write_surface_motion_outputs",
        "motion_flags[pixel]",
        "surface_brick_generation[pixel]",
    ] {
        assert!(surface.contains(token), "surface shader missing {token}");
    }

    for (name, shader) in [
        ("VPT temporal", temporal.as_str()),
        ("ReSTIR-DI temporal", restir.as_str()),
        ("Area ReSTIR temporal", area.as_str()),
    ] {
        for token in [
            "#include \"vpt_motion_common.slang\"",
            "RWTexture2D<uint> motion_flags",
            "RWTexture2D<uint> surface_brick_generation",
            "RWTexture2D<uint> previous_surface_brick_generation",
            "vpt_motion_flags_reject_history",
            "vpt_surface_generation_rejects_history",
        ] {
            assert!(shader.contains(token), "{name} missing {token}");
        }
    }
}

#[test]
fn vpt_shader_declares_stochastic_accumulating_reference_path() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let primary_sample_common = normalized_source(include_str!(
        "../../../../assets/shaders/shared/vpt_primary_sample_common.slang"
    ));

    assert!(source.contains("RWTexture2D<float4> noisy_radiance_image;"));
    assert!(source.contains("RWTexture2D<float4> noisy_moments_image;"));
    assert!(source.contains("hash_u32("));
    assert!(source.contains("#include \"vpt_primary_sample_common.slang\""));
    assert!(primary_sample_common.contains("scene.vpt_sample_index"));
    assert!(source.contains("scene.vpt_max_bounces"));
    assert!(source.contains("trace_primary_ray("));
    assert!(source.contains("VptTraceSample sample = trace_path("));
    assert!(source.contains("float3 sample_radiance = sample.radiance;"));
    assert!(source.contains("float luminance = dot(sample_radiance"));
    assert!(source.contains("noisy_radiance_image[tid.xy] = float4(sample_radiance, 1.0);"));
}

#[test]
fn vpt_primary_seed_does_not_depend_on_wall_clock_time() {
    let primary_sample_common = normalized_source(include_str!(
        "../../../../assets/shaders/shared/vpt_primary_sample_common.slang"
    ));

    assert!(primary_sample_common.contains("uint vpt_primary_rng_seed"));
    assert!(primary_sample_common.contains("scene.vpt_sample_index"));
    assert!(
        !primary_sample_common.contains("scene.time"),
        "primary RNG seed must not depend on wall-clock time"
    );
}

#[test]
fn vpt_nrd_noisy_frontend_contract() {
    let shader = source("assets/shaders/passes/vpt.slang");
    let rust = source("src/render/passes/vpt.rs");
    let compact_shader = shader.split_whitespace().collect::<String>();

    for token in [
        "struct VptTraceSample",
        "float first_indirect_hit_distance",
        "RWTexture2D<float4> nrd_diff_radiance_hitdist",
        "RWTexture2D<float4> nrd_spec_radiance_hitdist",
        "RWTexture2D<float4> nrd_residual_radiance",
        "RWTexture2D<float4> nrd_material_factors",
        "sample.first_indirect_hit_distance = VPT_NRD_INVALID_HIT_DISTANCE;",
        "if (hit.hit && bounce > 0u && sample.first_indirect_hit_distance == VPT_NRD_INVALID_HIT_DISTANCE)",
        "float3 nrd_residual = sample.direct_radiance;",
        "float3 nrd_diffuse_signal = demodulate_by_primary_albedo(sample.indirect_radiance, sample.first_hit);",
        "float nrd_diffuse_hitdist = sample.first_indirect_hit_distance;",
        "float nrd_material_id = sample.first_hit.hit ? float(voxel_material(sample.first_hit.cell)) : 0.0;",
        "nrd_diff_radiance_hitdist[tid.xy]",
        "nrd_spec_radiance_hitdist[tid.xy]",
        "nrd_residual_radiance[tid.xy]",
        "nrd_material_factors[tid.xy]",
    ] {
        assert!(
            shader.contains(token),
            "VPT shader missing NRD noisy token {token}"
        );
    }

    assert!(
        !shader.contains("demodulate_by_primary_albedo(sample_radiance, sample.first_hit)"),
        "RELAX diffuse input must not fold direct, sky, primary emissive, or debug radiance into the indirect diffuse denoising signal"
    );
    assert!(
        !shader.contains("nrd_diffuse_radiance"),
        "VPT should write the dominant final radiance signal to NRD diffuse instead of a partial side channel"
    );
    assert!(
        !shader.contains("max(sample.first_hit.t, 0.0)"),
        "RELAX diffuse hit distance must exclude the primary hit distance"
    );
    assert!(
        compact_shader.contains(
            "float3nrd_residual=sample.direct_radiance;float3nrd_diffuse_signal=demodulate_by_primary_albedo(sample.indirect_radiance,sample.first_hit);"
        ),
        "RELAX should denoise only indirect diffuse and keep primary direct/emissive radiance in the residual channel"
    );
    assert!(
        compact_shader.contains(
            "if(!sample.first_hit.hit){nrd_residual=sample_radiance;nrd_diffuse_signal=float3(0.0);nrd_diffuse_hitdist=VPT_NRD_INVALID_HIT_DISTANCE;}elseif(scene.vpt_debug_view!=VPT_DEBUG_VIEW_FINAL){nrd_residual=sample_radiance;nrd_diffuse_signal=float3(0.0);nrd_diffuse_hitdist=VPT_NRD_INVALID_HIT_DISTANCE;}"
        ),
        "miss/debug paths should bypass RELAX through residual radiance"
    );

    for token in [
        "pub struct VptNrdNoisyResources",
        "pub diff_radiance_hitdist: ResourceHandle",
        "pub spec_radiance_hitdist: ResourceHandle",
        "pub residual_radiance: ResourceHandle",
        "pub material_factors: ResourceHandle",
        "pub nrd_diff_radiance_hitdist: GpuImage",
        "pub nrd_spec_radiance_hitdist: GpuImage",
        "pub nrd_residual_radiance: GpuImage",
        "pub nrd_material_factors: GpuImage",
        "descriptor_count: 6 * frame_count as u32",
        "vpt_nrd_diff_radiance_hitdist",
        "vpt_nrd_spec_radiance_hitdist",
        "vpt_nrd_residual_radiance",
        "vpt_nrd_material_factors",
        "let noisy_image_usage = vk::ImageUsageFlags::STORAGE",
        "| vk::ImageUsageFlags::SAMPLED",
    ] {
        assert!(
            rust.contains(token),
            "VPT Rust missing NRD noisy contract token {token}"
        );
    }
}

#[test]
fn vpt_nrd_frontend_pass_declares_relax_packing_contract() {
    let shader = source("assets/shaders/passes/vpt_nrd_frontend.slang");
    let pass = source("src/render/passes/vpt_nrd_frontend.rs");
    let pass_mod = source("src/render/passes/mod.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact_pass = pass.split_whitespace().collect::<String>();
    let compact_shader = shader.split_whitespace().collect::<String>();
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();

    crate::render::descriptor::assert_specs_match_shader_bindings(
        "VPT NRD frontend",
        &VptNrdFrontendPass::descriptor_binding_specs(),
        &shader_reflection("assets/shaders/passes/vpt_nrd_frontend.slang"),
    );
    assert_eq!(
        shader_bindings("assets/shaders/passes/vpt_nrd_frontend.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "scene_ubo"),
            binding(
                1,
                DescriptorKind::StorageImage,
                "input_diff_radiance_hitdist"
            ),
            binding(
                2,
                DescriptorKind::StorageImage,
                "input_spec_radiance_hitdist"
            ),
            binding(3, DescriptorKind::StorageImage, "input_residual_radiance"),
            binding(4, DescriptorKind::StorageImage, "input_material_factors"),
            binding(5, DescriptorKind::StorageImage, "surface_normal_roughness"),
            binding(
                6,
                DescriptorKind::StorageImage,
                "surface_material_roughness"
            ),
            binding(7, DescriptorKind::StorageImage, "surface_view_z"),
            binding(
                8,
                DescriptorKind::StorageImage,
                "packed_diff_radiance_hitdist"
            ),
            binding(
                9,
                DescriptorKind::StorageImage,
                "packed_spec_radiance_hitdist"
            ),
            binding(10, DescriptorKind::StorageImage, "residual_radiance"),
            binding(11, DescriptorKind::StorageImage, "material_factors"),
            binding(12, DescriptorKind::StorageImage, "packed_normal_roughness"),
        ]
    );

    for token in [
        "#include \"scene_common.slang\"",
        "static const float VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE = 65504.0;",
        "if (value != value)",
        "return min(max(value, 0.0), VPT_NRD_FRONTEND_INVALID_HIT_DISTANCE);",
        "if (hit_distance != hit_distance || hit_distance < 0.0)",
        "float3 sanitize_relax_radiance(float3 radiance)",
        "float sanitize_relax_hit_distance(float hit_distance)",
        "float4 pack_relax_radiance_hitdist(float4 radiance_hitdist)",
        "float _REBLUR_GetHitDistanceNormalization(float view_z, float roughness, float3 hit_distance_parameters)",
        "float REBLUR_FrontEnd_GetNormHitDist(float hit_distance, float view_z, float roughness, float3 hit_distance_parameters)",
        "float4 REBLUR_FrontEnd_PackRadianceAndNormHitDist(float4 radiance_hitdist, float view_z, float roughness)",
        "float3 sanitize_nrd_normal(float3 normal)",
        "float2 encode_nrd_unit_vector_oct(float3 normal)",
        "float4 pack_nrd_normal_roughness(float3 normal, float roughness, float material_id)",
        "RWTexture2D<float> surface_view_z",
    ] {
        assert!(
            shader.contains(token),
            "NRD frontend shader missing {token}"
        );
    }
    for token in [
        "returnfloat4(sanitize_relax_radiance(radiance_hitdist.rgb),sanitize_relax_hit_distance(radiance_hitdist.a));",
        "packed_spec_radiance_hitdist[tid.xy]=pack_relax_radiance_hitdist(input_spec_radiance_hitdist[tid.xy]);",
        "booluse_reblur=(scene.denoiser_flags&DENOISER_MODE_MASK)==DENOISER_MODE_REBLUR;",
        "floatview_z=surface_view_z[tid.xy];",
        "floatroughness=surface_material_roughness[tid.xy];",
        "packed_diff_radiance_hitdist[tid.xy]=use_reblur?REBLUR_FrontEnd_PackRadianceAndNormHitDist(input_diff_radiance_hitdist[tid.xy],view_z,roughness):pack_relax_radiance_hitdist(input_diff_radiance_hitdist[tid.xy]);",
        "residual_radiance[tid.xy]=float4(sanitize_relax_radiance(input_residual_radiance[tid.xy].rgb),1.0);",
        "material_factors[tid.xy]=float4(saturate(input_material_factors[tid.xy].rgb),input_material_factors[tid.xy].a);",
        "normal/=max(abs(normal.x)+abs(normal.y)+abs(normal.z),1.0e-8);",
        "float2encoded_normal=encode_nrd_unit_vector_oct(sanitize_nrd_normal(normal));",
        "returnfloat4(encoded_normal,saturate(roughness),saturate(material_id/3.0));",
        "packed_normal_roughness[tid.xy]=pack_nrd_normal_roughness(surface_normal_roughness[tid.xy].xyz,surface_material_roughness[tid.xy],input_material_factors[tid.xy].a);",
    ] {
        assert!(
            compact_shader.contains(token),
            "NRD frontend shader missing compact token {token}"
        );
    }
    for forbidden in [
        "NRD.hlsli",
        "RELAX_FrontEnd",
        "PackRadiance.cs.slang",
        "encode_nrd_normal_roughness_101010",
        "return float4(encoded, 0.0);",
    ] {
        assert!(
            !shader.contains(forbidden),
            "local frontend must not copy SDK helper token {forbidden}"
        );
    }
    assert!(
        !compact_shader.contains(
            "REBLUR_FrontEnd_PackRadianceAndNormHitDist(input_diff_radiance_hitdist[tid.xy],view_z,1.0)"
        ),
        "ReBLUR hit-distance normalization must use the surface roughness guide, not a hard-coded fully rough material"
    );

    for token in [
        "pub struct VptNrdFrontendPass",
        "pub struct VptNrdFrontendGraphInputs",
        "pub struct VptNrdFrontendGraphOutputs",
        "pub struct VptNrdPackedResources",
        "pub raw_noisy: VptNrdNoisyResources",
        "pub surface_inputs: VptCurrentSurfaceResources",
        "pub packed: VptNrdPackedResources",
        "pub packed_diff_radiance_hitdist: GpuImage",
        "pub packed_spec_radiance_hitdist: GpuImage",
        "pub residual_radiance: GpuImage",
        "pub material_factors: GpuImage",
        "pub packed_normal_roughness: GpuImage",
        "pub normal_roughness: ResourceHandle",
        "pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 13]",
        "descriptor_count: 12 * frame_count as u32",
        "vpt_nrd_packed_diff_radiance_hitdist",
        "vpt_nrd_packed_spec_radiance_hitdist",
        "vpt_nrd_packed_normal_roughness",
        "vpt_nrd_frontend_residual_radiance",
        "vpt_nrd_frontend_material_factors",
        "graph.add_pass(\"vpt_nrd_frontend\"",
        "GpuProfileScope::VptNrdFrontend",
    ] {
        assert!(pass.contains(token), "NRD frontend pass missing {token}");
    }
    for token in [
        "builder.read_as(raw_noisy.diff_radiance_hitdist,AccessKind::ComputeShaderRead,);",
        "builder.read_as(raw_noisy.spec_radiance_hitdist,AccessKind::ComputeShaderRead,);",
        "builder.read_as(raw_noisy.residual_radiance,AccessKind::ComputeShaderRead);",
        "builder.read_as(raw_noisy.material_factors,AccessKind::ComputeShaderRead);",
        "builder.read_as(surface_inputs.normal_roughness,AccessKind::ComputeShaderRead,);",
        "builder.read_as(surface_inputs.material_roughness,AccessKind::ComputeShaderRead,);",
        "builder.read_as(surface_inputs.view_z,AccessKind::ComputeShaderRead);",
        "builder.write_as(packed_normal_resource,AccessKind::ComputeShaderWrite);",
        "&vpt_surface.surface_view_z",
    ] {
        assert!(
            compact_pass.contains(token),
            "NRD frontend pass missing compact token {token}"
        );
    }
    for forbidden in [
        "[ResourceHandle; 4]",
        "[ResourceHandle; 8]",
        "raw_noisy[",
        "packed_outputs[",
    ] {
        assert!(
            !pass.contains(forbidden),
            "NRD frontend pass must use named resources, found {forbidden}"
        );
    }
    assert!(compact_pass.contains(
        "letpacked_writes=graph.add_pass(\"vpt_nrd_frontend\",QueueType::Compute,|builder|{"
    ));
    assert!(pass_mod.contains("pub mod vpt_nrd_frontend;"));

    for token in [
        "use crate::render::passes::vpt_nrd_frontend::{",
        "pub vpt_nrd_frontend_pass: Option<VptNrdFrontendPass>",
        "vpt_nrd_frontend_pass: None",
        "self.ensure_vpt_nrd_frontend_pass(renderer, scene_ubo);",
        "fn ensure_vpt_nrd_frontend_pass(",
        "include_bytes!(concat!(env!(\"OUT_DIR\"), \"/shaders/vpt_nrd_frontend.spv\"))",
        "VptNrdFrontendPass::new(",
        ".resize_images(",
        "failed to resize VPT NRD frontend images",
        "pass.destroy(device, allocator);",
    ] {
        assert!(pipeline.contains(token), "pipeline missing {token}");
    }
    assert!(
        compact_pipeline.contains(
            "letnrd_frontend_outputs=ifmatches!(inputs.lighting_settings.denoiser_mode,VptDenoiserMode::Relax|VptDenoiserMode::Reblur){"
        ),
        "pipeline must record NRD frontend only for requested NRD modes"
    );
    assert!(
        compact_pipeline
            .contains("vpt_nrd_frontend.register_graph(&mutgraph,VptNrdFrontendGraphInputs{"),
        "pipeline must register frontend graph"
    );
    assert!(
        compact_pipeline.contains("letnrd_frontend_outputs="),
        "pipeline must keep the NRD frontend output explicit for adapter wiring"
    );
    assert!(
        compact_pipeline.contains("surface_inputs:final_surface_writes,"),
        "pipeline must feed selected surface guides into the NRD frontend"
    );
    assert!(
        compact_pipeline.contains("atrous_filtered_dep,vpt_atrous.output_image()"),
        "SVGF fallback output must remain available for postprocess"
    );
    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT graph registration should exist");
    let nrd_idx = pipeline
        .find("vpt_nrd_frontend.register_graph(")
        .expect("NRD frontend graph registration should exist");
    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("SVGF temporal graph registration should exist");
    assert!(vpt_idx < nrd_idx);
    assert!(vpt_idx < temporal_idx);
}

#[test]
fn vpt_nrd_reblur_hit_distance_parameters_match_native_settings() {
    let shader = source("assets/shaders/passes/vpt_nrd_frontend.slang");
    let sys = source("crates/revolumetric-nrd-sys/src/lib.rs");
    let compact_shader = shader.split_whitespace().collect::<String>();
    let compact_sys = sys.split_whitespace().collect::<String>();

    assert!(
        shader.contains(
            "static const float3 VPT_NRD_REBLUR_HIT_DISTANCE_PARAMETERS = float3(3.0, 0.1, 20.0);"
        ),
        "ReBLUR frontend shader must expose the same hit-distance parameters as the native settings default"
    );
    assert!(
        compact_shader.contains(
            "REBLUR_FrontEnd_GetNormHitDist(radiance_hitdist.a,view_z,roughness,VPT_NRD_REBLUR_HIT_DISTANCE_PARAMETERS)"
        ),
        "ReBLUR frontend packing must use the shared named hit-distance parameter constant"
    );
    assert!(
        compact_sys.contains("Self{a:3.0,b:0.1,c:20.0,}"),
        "Rust NRD ReBLUR hit-distance default must stay aligned with the shader frontend parameters"
    );
}

#[test]
fn vpt_nrd_confidence_pass_declares_history_confidence_contract() {
    let shader = source("assets/shaders/passes/vpt_nrd_confidence.slang");
    let pass = source("src/render/passes/vpt_nrd_confidence.rs");
    let pass_mod = source("src/render/passes/mod.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact_shader = shader.split_whitespace().collect::<String>();
    let compact_pass = pass.split_whitespace().collect::<String>();
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();

    crate::render::descriptor::assert_specs_match_shader_bindings(
        "VPT NRD confidence",
        &VptNrdConfidencePass::descriptor_binding_specs(),
        &shader_reflection("assets/shaders/passes/vpt_nrd_confidence.slang"),
    );
    assert_eq!(
        shader_bindings("assets/shaders/passes/vpt_nrd_confidence.slang"),
        vec![
            binding(0, DescriptorKind::UniformBuffer, "scene_ubo"),
            binding(1, DescriptorKind::StorageImage, "motion_history"),
            binding(2, DescriptorKind::StorageImage, "motion_flags"),
            binding(3, DescriptorKind::StorageImage, "surface_brick_generation"),
            binding(
                4,
                DescriptorKind::StorageImage,
                "previous_surface_brick_generation"
            ),
            binding(5, DescriptorKind::StorageImage, "diff_confidence"),
            binding(6, DescriptorKind::StorageImage, "spec_confidence"),
        ]
    );

    for token in [
        "#include \"scene_common.slang\"",
        "#include \"vpt_motion_common.slang\"",
        "[[vk::image_format(\"rgba32f\")]]",
        "[[vk::image_format(\"r32ui\")]]",
        "[[vk::image_format(\"r16f\")]]",
        "RWTexture2D<float4> motion_history",
        "RWTexture2D<uint> motion_flags",
        "RWTexture2D<uint> surface_brick_generation",
        "RWTexture2D<uint> previous_surface_brick_generation",
        "RWTexture2D<float> diff_confidence",
        "RWTexture2D<float> spec_confidence",
        "float vpt_nrd_diffuse_history_confidence(uint2 pixel, SceneUniforms scene)",
        "vpt_motion_flags_reject_history(flags)",
        "vpt_surface_generation_rejects_history(previous_generation, current_generation)",
        "spec_confidence[tid.xy] = 0.0;",
    ] {
        assert!(
            shader.contains(token),
            "NRD confidence shader missing {token}"
        );
    }
    for token in [
        "float2previous_pixel_f=motion_history[pixel].xy+float2(pixel)+0.5;",
        "int2previous_pixel=int2(floor(previous_pixel_f));",
        "if(previous_pixel.x<0||previous_pixel.y<0||previous_pixel.x>=int(scene.resolution.x)||previous_pixel.y>=int(scene.resolution.y)){return0.0;}",
        "uintprevious_generation=previous_surface_brick_generation[uint2(previous_pixel)];",
        "uintcurrent_generation=surface_brick_generation[pixel];",
        "return1.0;",
        "diff_confidence[tid.xy]=vpt_nrd_diffuse_history_confidence(tid.xy,scene);",
        "spec_confidence[tid.xy]=0.0;",
    ] {
        assert!(
            compact_shader.contains(token),
            "NRD confidence shader missing compact token {token}"
        );
    }

    for token in [
        "pub struct VptNrdConfidencePass",
        "pub struct VptNrdConfidenceGraphInputs",
        "pub struct VptNrdConfidenceGraphOutputs",
        "pub struct VptNrdConfidenceResources",
        "pub surface_inputs: VptCurrentSurfaceResources",
        "pub previous_surface_inputs: VptPreviousSurfaceResources",
        "pub confidence: VptNrdConfidenceResources",
        "pub diff_confidence: GpuImage",
        "pub spec_confidence: GpuImage",
        "pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 7]",
        "descriptor_count: 6 * frame_count as u32",
        "vk::Format::R16_SFLOAT",
        "vpt_nrd_diff_confidence",
        "vpt_nrd_spec_confidence",
        "graph.add_pass(\"vpt_nrd_confidence\"",
        "GpuProfileScope::VptNrdConfidence",
    ] {
        assert!(pass.contains(token), "NRD confidence pass missing {token}");
    }
    for token in [
        "builder.read_as(surface_inputs.motion_history,AccessKind::ComputeShaderRead);",
        "builder.read_as(surface_inputs.motion_flags,AccessKind::ComputeShaderRead);",
        "builder.read_as(surface_inputs.brick_generation,AccessKind::ComputeShaderRead,);",
        "builder.read_as(previous_surface_inputs.brick_generation,AccessKind::ComputeShaderRead,);",
        "builder.write_as(diff_resource,AccessKind::ComputeShaderWrite);",
        "builder.write_as(spec_resource,AccessKind::ComputeShaderWrite);",
    ] {
        assert!(
            compact_pass.contains(token),
            "NRD confidence pass missing compact token {token}"
        );
    }
    for forbidden in [
        "[ResourceHandle; 4]",
        "[ResourceHandle; 6]",
        "surface_inputs[",
        "previous_surface_inputs[",
        "confidence_outputs[",
    ] {
        assert!(
            !pass.contains(forbidden),
            "NRD confidence pass must use named resources, found {forbidden}"
        );
    }
    assert!(pass_mod.contains("pub mod vpt_nrd_confidence;"));

    for token in [
        "use crate::render::passes::vpt_nrd_confidence::{",
        "pub vpt_nrd_confidence_pass: Option<VptNrdConfidencePass>",
        "vpt_nrd_confidence_pass: None",
        "self.ensure_vpt_nrd_confidence_pass(renderer, scene_ubo);",
        "fn ensure_vpt_nrd_confidence_pass(",
        "include_bytes!(concat!(env!(\"OUT_DIR\"), \"/shaders/vpt_nrd_confidence.spv\"))",
        "VptNrdConfidencePass::new(",
        "failed to resize VPT NRD confidence images",
        "vpt_nrd_confidence_ready = self.vpt_nrd_confidence_pass.is_some()",
        "pass.destroy(device, allocator);",
    ] {
        assert!(pipeline.contains(token), "pipeline missing {token}");
    }
    assert!(
        compact_pipeline.contains(
            "letnrd_confidence_outputs=ifmatches!(inputs.lighting_settings.denoiser_mode,VptDenoiserMode::Relax|VptDenoiserMode::Reblur){"
        ),
        "pipeline must record NRD confidence only for requested NRD modes"
    );
    assert!(
        compact_pipeline
            .contains("vpt_nrd_confidence.register_graph(&mutgraph,VptNrdConfidenceGraphInputs{"),
        "pipeline must register confidence graph"
    );
    assert!(
        compact_pipeline.contains("surface_inputs:final_surface_writes,"),
        "confidence pass must consume final selected surface writes"
    );
    assert!(
        compact_pipeline.contains("previous_surface_inputs:previous_surface_resources,"),
        "confidence pass must compare against previous surface resources"
    );
    assert!(
        compact_pipeline.contains("letnrd_confidence_outputs="),
        "pipeline must keep the confidence output explicit for NRD adapter wiring"
    );
    assert!(
        compact_pipeline.contains("atrous_filtered_dep,vpt_atrous.output_image()"),
        "SVGF fallback output must remain available for postprocess"
    );
    let surface_idx = pipeline
        .find("vpt_surface.register_bootstrap_graph(")
        .expect("surface graph registration should exist");
    let confidence_idx = pipeline
        .find("vpt_nrd_confidence.register_graph(")
        .expect("confidence graph registration should exist");
    let frontend_idx = pipeline
        .find("vpt_nrd_frontend.register_graph(")
        .expect("NRD frontend graph registration should exist");
    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("SVGF temporal graph registration should exist");
    assert!(surface_idx < confidence_idx);
    assert!(confidence_idx < frontend_idx);
    assert!(confidence_idx < temporal_idx);
}

#[test]
fn vpt_nrd_adapter_declares_relax_integration_contract() {
    let adapter = source("src/render/passes/vpt_nrd_adapter.rs");
    let frame_settings = source("src/render/passes/vpt_nrd_adapter/frame_settings.rs");
    let rust_api = source("crates/revolumetric-nrd/src/lib.rs");
    let sys = source("crates/revolumetric-nrd-sys/src/lib.rs");
    let native_header = source("crates/revolumetric-nrd-sys/native/nrd_adapter.h");
    let native_cpp = source("crates/revolumetric-nrd-sys/native/nrd_adapter.cpp");
    let build_rs = source("crates/revolumetric-nrd-sys/build.rs");
    let render_mod = source("src/render/mod.rs");
    let pass_mod = source("src/render/passes/mod.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let profiler = source("src/render/gpu_profiler.rs");
    let compact_adapter = adapter.split_whitespace().collect::<String>();
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();

    for token in [
        "pub struct NrdLibraryDesc",
        "pub struct NrdInstanceDesc",
        "pub struct NrdTextureDesc",
        "pub struct NrdResourceDesc",
        "pub struct NrdResourceRangeDesc",
        "pub struct NrdPipelineDesc",
        "pub struct NrdDispatchDesc",
        "pub struct NrdCommonSettings",
        "pub struct NrdRelaxDiffuseSettings",
        "pub struct NrdReblurHitDistanceParameters",
        "pub struct NrdReblurDiffuseSettings",
        "pub struct RevolumetricNrdInstance",
        "pub type RevolumetricNrdStatus = u32",
        "REVOLUMETRIC_NRD_STATUS_OK",
        "#[cfg(feature = \"nrd\")]",
    ] {
        assert!(sys.contains(token), "NRD sys layer missing {token}");
    }
    for token in [
        "pub use revolumetric_nrd_sys::{",
        "pub struct NrdTextureImageDesc",
        "pub struct NrdResourceBindingDesc",
        "pub struct NrdPipelineSnapshot",
        "pub struct NrdInstanceSnapshot",
        "pub struct NrdDispatchSnapshot",
        "pub struct NrdUnavailableError",
        "pub type NrdResult<T> = Result<T, NrdUnavailableError>",
        "pub trait NrdTextureDescExt",
        "pub trait NrdResourceDescExt",
        "pub trait NrdResourceRangeDescExt",
        "pub struct NrdInstance",
        "#[cfg(feature = \"nrd\")]",
        "#[cfg(not(feature = \"nrd\"))]",
        "NrdInstanceSnapshot::from_sys(&desc)",
        "nrd_sys::revolumetric_nrd_create_relax_diffuse",
        "nrd_sys::revolumetric_nrd_create_reblur_diffuse",
        "nrd_sys::revolumetric_nrd_destroy",
        "relax_diffuse",
        "reblur_diffuse",
        "set_reblur_diffuse_settings",
    ] {
        assert!(rust_api.contains(token), "NRD Rust API missing {token}");
    }
    for token in [
        "println!(\"cargo:rerun-if-env-changed=REVOLUMETRIC_NRD_ROOT\");",
        "if !nrd_feature_enabled()",
        "fn validate_nrd_sdk_root() -> PathBuf",
        "println!(\"cargo:rerun-if-changed=native/nrd_adapter.h\");",
        "fn nrd_library_dir(root: &Path) -> PathBuf",
        "include(\"native\")",
        "cargo:rustc-link-lib=static=NRD",
    ] {
        assert!(build_rs.contains(token), "NRD build gate missing {token}");
    }
    for token in [
        "typedef enum RevolumetricNrdTextureFormat",
        "typedef enum RevolumetricNrdDescriptorType",
        "typedef enum RevolumetricNrdSamplerMode",
        "typedef enum RevolumetricNrdAccumulationMode",
        "typedef enum RevolumetricNrdCheckerboardMode",
        "typedef enum RevolumetricNrdHitDistanceReconstructionMode",
        "typedef enum RevolumetricNrdResourceType",
        "REVOLUMETRIC_NRD_TEXTURE_FORMAT_R16_SFLOAT",
        "REVOLUMETRIC_NRD_TEXTURE_FORMAT_RGBA16_SFLOAT",
        "REVOLUMETRIC_NRD_DESCRIPTOR_TYPE_TEXTURE",
        "REVOLUMETRIC_NRD_DESCRIPTOR_TYPE_STORAGE_TEXTURE",
        "REVOLUMETRIC_NRD_SAMPLER_MODE_NEAREST_CLAMP",
        "REVOLUMETRIC_NRD_SAMPLER_MODE_LINEAR_CLAMP",
        "REVOLUMETRIC_NRD_ACCUMULATION_MODE_CONTINUE",
        "REVOLUMETRIC_NRD_ACCUMULATION_MODE_RESTART",
        "REVOLUMETRIC_NRD_CHECKERBOARD_MODE_OFF",
        "REVOLUMETRIC_NRD_HIT_DISTANCE_RECONSTRUCTION_MODE_OFF",
        "REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_MV",
        "REVOLUMETRIC_NRD_RESOURCE_TYPE_IN_DIFF_RADIANCE_HITDIST",
        "REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_DIFF_RADIANCE_HITDIST",
        "REVOLUMETRIC_NRD_RESOURCE_TYPE_OUT_VALIDATION",
        "REVOLUMETRIC_NRD_RESOURCE_TYPE_TRANSIENT_POOL",
        "REVOLUMETRIC_NRD_RESOURCE_TYPE_PERMANENT_POOL",
        "struct NrdLibraryDesc",
        "struct NrdInstanceDesc",
        "struct NrdTextureDesc",
        "struct NrdResourceDesc",
        "struct NrdResourceRangeDesc",
        "struct NrdPipelineDesc",
        "struct NrdDispatchDesc",
        "struct NrdCommonSettings",
        "struct NrdRelaxDiffuseSettings",
        "struct NrdReblurHitDistanceParameters",
        "struct NrdReblurDiffuseSettings",
        "revolumetric_nrd_create_relax_diffuse",
        "revolumetric_nrd_create_reblur_diffuse",
        "revolumetric_nrd_destroy",
        "revolumetric_nrd_get_library_desc",
        "revolumetric_nrd_get_instance_desc",
        "revolumetric_nrd_get_dispatches",
        "revolumetric_nrd_set_common_settings",
        "revolumetric_nrd_set_relax_diffuse_settings",
        "revolumetric_nrd_set_reblur_diffuse_settings",
    ] {
        assert!(
            native_header.contains(token) || native_cpp.contains(token),
            "native NRD wrapper missing {token}"
        );
    }
    for token in [
        "static uint32_t to_texture_format(nrd::Format value)",
        "static uint32_t to_descriptor_type(nrd::DescriptorType value)",
        "static uint32_t to_sampler_mode(nrd::Sampler value)",
        "static bool to_accumulation_mode(uint32_t value, nrd::AccumulationMode& out)",
        "static bool to_checkerboard_mode(uint32_t value, nrd::CheckerboardMode& out)",
        "static bool to_hit_distance_reconstruction_mode(uint32_t value, nrd::HitDistanceReconstructionMode& out)",
        "static void copy_hit_distance_parameters",
        "static RevolumetricNrdStatus copy_reblur_diffuse_settings",
        "static uint32_t to_resource_type(nrd::ResourceType value)",
        "case nrd::Format::R16_SFLOAT:",
        "case nrd::Format::RGBA16_SFLOAT:",
        "REVOLUMETRIC_NRD_TEXTURE_FORMAT_UNSUPPORTED",
        "case nrd::DescriptorType::TEXTURE:",
        "case nrd::DescriptorType::STORAGE_TEXTURE:",
        "REVOLUMETRIC_NRD_DESCRIPTOR_TYPE_UNSUPPORTED",
        "case nrd::Sampler::NEAREST_CLAMP:",
        "case nrd::Sampler::LINEAR_CLAMP:",
        "REVOLUMETRIC_NRD_SAMPLER_MODE_UNSUPPORTED",
        "case REVOLUMETRIC_NRD_ACCUMULATION_MODE_CONTINUE:",
        "case REVOLUMETRIC_NRD_ACCUMULATION_MODE_RESTART:",
        "case REVOLUMETRIC_NRD_ACCUMULATION_MODE_CLEAR_AND_RESTART:",
        "case REVOLUMETRIC_NRD_CHECKERBOARD_MODE_OFF:",
        "case REVOLUMETRIC_NRD_CHECKERBOARD_MODE_BLACK:",
        "case REVOLUMETRIC_NRD_CHECKERBOARD_MODE_WHITE:",
        "case REVOLUMETRIC_NRD_HIT_DISTANCE_RECONSTRUCTION_MODE_OFF:",
        "case REVOLUMETRIC_NRD_HIT_DISTANCE_RECONSTRUCTION_MODE_AREA_3X3:",
        "case REVOLUMETRIC_NRD_HIT_DISTANCE_RECONSTRUCTION_MODE_AREA_5X5:",
        "case nrd::ResourceType::IN_MV:",
        "case nrd::ResourceType::OUT_VALIDATION:",
        "case nrd::ResourceType::PERMANENT_POOL:",
        "REVOLUMETRIC_NRD_RESOURCE_TYPE_UNSUPPORTED",
        "out.resourceRanges.reserve(",
        "out.pipelines.reserve(desc.pipelinesNum)",
        "instance->dispatchResources.reserve(",
        "instance->dispatches.reserve(dispatches_num)",
    ] {
        assert!(
            native_cpp.contains(token),
            "native NRD wrapper must reserve vector storage before exposing internal pointers: missing {token}"
        );
    }
    let reblur_sys_struct = sys
        .split("pub struct NrdReblurDiffuseSettings")
        .nth(1)
        .and_then(|tail| {
            tail.split("impl Default for NrdReblurDiffuseSettings")
                .next()
        })
        .expect("Rust sys ReBLUR settings struct should be present");
    let reblur_native_struct = native_header
        .split("typedef struct NrdReblurDiffuseSettings")
        .nth(1)
        .and_then(|tail| tail.split("} NrdReblurDiffuseSettings;").next())
        .expect("native ReBLUR settings struct should be present");
    for forbidden in [
        "historyFixEdgeStoppingNormalPower",
        "history_fix_edge_stopping_normal_power",
        "enableRoughnessEdgeStopping",
        "enable_roughness_edge_stopping",
    ] {
        assert!(
            !reblur_sys_struct.contains(forbidden) && !reblur_native_struct.contains(forbidden),
            "ReBLUR ABI must not include SDK-absent RELAX field {forbidden}"
        );
    }
    assert!(
        native_cpp.contains("nrd::ReblurHitDistanceParameters& dst"),
        "native ReBLUR ABI must copy into the SDK ReblurHitDistanceParameters type"
    );
    assert!(
        !native_cpp.contains("static uint32_t to_u32(nrd::DescriptorType value)")
            && !native_cpp.contains("static uint32_t to_u32(nrd::ResourceType value)"),
        "native NRD wrapper must not expose SDK enum ordinals as stable ABI values"
    );

    for token in [
        "pub struct VptNrdAdapterPass",
        "pub struct VptNrdAdapterGraphInputs",
        "pub struct VptNrdAdapterGraphOutputs",
        "pub struct VptNrdAdapterResources",
        "pub packed: VptNrdPackedResources",
        "pub confidence: VptNrdConfidenceResources",
        "pub surface_inputs: VptCurrentSurfaceResources",
        "pub diff_radiance_hitdist: ResourceHandle",
        "pub validation: ResourceHandle",
        "backend: VptNrdAdapterBackend",
        "pub enum VptNrdAdapterBackend",
        "Ready(Box<VptNrdReadyBackend>)",
        "Unavailable(String)",
        "pub struct VptNrdReadyBackend",
        "mod frame_settings;",
        "pub use frame_settings::{",
        "VptNrdFrameSettings",
        "VptNrdFrameSettingsInputs",
        "pub struct VptNrdTexturePoolPlan",
        "pub struct VptNrdTexturePoolImagePlan",
        "pub struct VptNrdDispatchResourcePlan",
        "pub struct VptNrdDispatchResourceBindingPlan",
        "pub enum VptNrdDispatchResource",
        "texture_pool_plan: VptNrdTexturePoolPlan",
        "dispatch_resource_plans: Vec<VptNrdDispatchResourcePlan>",
        "VptNrdDispatchResourcePlan::from_dispatches(&dispatches, &texture_pool_plan)",
        "dispatch_resource_plan_count: self.dispatch_resource_plans.len()",
        "library_desc: NrdLibraryDesc",
        "instance_snapshot: NrdInstanceSnapshot",
        "VptNrdTexturePoolPlan::from_instance_snapshot(width, height, &instance_snapshot)",
        "pub fn texture_pool_plan(&self) -> Option<&VptNrdTexturePoolPlan>",
        "NrdInstance::relax_diffuse(width, height)",
        "NrdInstance::reblur_diffuse(width, height)",
        "instance.instance_snapshot()",
        "set_common_settings",
        "set_relax_diffuse_settings",
        "set_reblur_diffuse_settings",
        "fn update_frame_settings(",
        "refresh_dispatches",
        "refresh_constant_upload_plan",
        "denoiser_mode: VptDenoiserMode",
        "pub fn is_ready(&self) -> bool",
        "pub fn dispatch_count(&self) -> usize",
        "pub fn unavailable_reason(&self) -> Option<&str>",
        "pub nrd_diff_radiance_hitdist: GpuImage",
        "pub nrd_validation: GpuImage",
        "vpt_nrd_adapter_diff_radiance_hitdist",
        "vpt_nrd_adapter_validation",
        "graph.add_pass(\"vpt_nrd_adapter\"",
        "GpuProfileScope::VptNrdAdapter",
    ] {
        assert!(adapter.contains(token), "NRD adapter pass missing {token}");
    }
    for token in [
        "pub struct VptNrdFrameSettingsInputs",
        "pub struct VptNrdFrameSettings",
        "pub enum NrdAccumulationMode",
        "pub enum NrdCheckerboardMode",
        "pub enum NrdHitDistanceReconstructionMode",
        "reblur_diffuse: NrdReblurDiffuseSettings",
    ] {
        assert!(
            frame_settings.contains(token),
            "NRD adapter frame settings module missing {token}"
        );
    }
    for token in [
        "builder.read_as(packed.diff_radiance_hitdist,AccessKind::ComputeShaderRead);",
        "builder.read_as(packed.spec_radiance_hitdist,AccessKind::ComputeShaderRead);",
        "builder.read_as(packed.residual_radiance,AccessKind::ComputeShaderRead);",
        "builder.read_as(packed.material_factors,AccessKind::ComputeShaderRead);",
        "builder.read_as(packed.normal_roughness,AccessKind::ComputeShaderRead);",
        "builder.read_as(confidence.diff_confidence,AccessKind::ComputeShaderRead);",
        "builder.read_as(confidence.spec_confidence,AccessKind::ComputeShaderRead);",
        "builder.read_as(surface_inputs.view_z,AccessKind::ComputeShaderRead);",
        "builder.read_as(surface_inputs.motion_history,AccessKind::ComputeShaderRead);",
        "device.update_descriptor_sets(&writes,&[])",
        "device.cmd_bind_pipeline(cmd,vk::PipelineBindPoint::COMPUTE,pipeline.pipeline.handle,);",
        "device.cmd_bind_descriptor_sets(cmd,vk::PipelineBindPoint::COMPUTE,pipeline.pipeline.layout,pipeline.shared_set_index,&[shared_descriptor_set],&[],);",
        "device.cmd_bind_descriptor_sets(cmd,vk::PipelineBindPoint::COMPUTE,pipeline.pipeline.layout,pipeline.resource_set_index,&[update.descriptor_set],&[],);",
        "device.cmd_dispatch(cmd,u32::from(dispatch.grid_width),u32::from(dispatch.grid_height),1,);",
    ] {
        assert!(
            compact_adapter.contains(token),
            "NRD adapter graph missing compact token {token}"
        );
    }
    for forbidden in [
        "[ResourceHandle; 4]",
        "[ResourceHandle; 6]",
        "[ResourceHandle; 8]",
        "packed[",
        "confidence[",
        "surface_inputs[",
    ] {
        assert!(
            !adapter.contains(forbidden),
            "NRD adapter pass must use named resources, found {forbidden}"
        );
    }

    assert!(!render_mod.contains("pub mod nrd_adapter;"));
    assert!(!render_mod.contains("pub mod nrd_sys;"));
    assert!(pass_mod.contains("pub mod vpt_nrd_adapter;"));
    assert!(profiler.contains("VptNrdAdapter"));
    assert!(profiler.contains("vpt_nrd_adapter_ms"));

    for token in [
        "use crate::render::passes::vpt_nrd_adapter::{",
        "pub vpt_nrd_adapter_pass: Option<VptNrdAdapterPass>",
        "vpt_nrd_adapter_pass: None",
        "self.ensure_vpt_nrd_adapter_pass(renderer, scene_ubo, lighting_settings);",
        "fn ensure_vpt_nrd_adapter_pass(",
        "VptNrdAdapterPass::new(",
        "VptNrdFrameSettings::from_inputs",
        "VptNrdFrameSettingsInputs",
        "compute_nrd_world_to_view",
        "compute_nrd_view_to_clip",
        "previous_nrd_world_to_view",
        "previous_nrd_view_to_clip",
        "previous_nrd_elapsed_seconds",
        "failed to resize VPT NRD adapter images",
        "vpt_nrd_adapter_ready = self.vpt_nrd_adapter_pass.is_some()",
        "pass.destroy(device, allocator);",
    ] {
        assert!(pipeline.contains(token), "pipeline missing {token}");
    }
    assert!(
        compact_pipeline.contains(
            "letnrd_adapter_outputs=ifmatches!(inputs.lighting_settings.denoiser_mode,VptDenoiserMode::Relax|VptDenoiserMode::Reblur){"
        ),
        "pipeline must record NRD adapter for requested RELAX or REBLUR mode"
    );
    assert!(
        compact_pipeline
            .contains("vpt_nrd_adapter.register_graph(&mutgraph,VptNrdAdapterGraphInputs{"),
        "pipeline must register adapter graph"
    );
    assert!(
        compact_pipeline.contains("packed:nrd_frontend_outputs.packed,"),
        "adapter must consume packed NRD frontend resources"
    );
    assert!(
        compact_pipeline.contains("confidence:nrd_confidence_outputs.confidence,"),
        "adapter must consume history confidence resources"
    );
    assert!(
        compact_pipeline.contains("vpt_nrd_adapter.update_frame_settings("),
        "pipeline must refresh NRD settings and dispatches before registering the adapter graph"
    );
    assert!(
        compact_pipeline.contains("surface_inputs:final_surface_writes,"),
        "adapter must consume the selected current surface guides"
    );
    assert!(
        compact_pipeline.contains("atrous_filtered_dep,vpt_atrous.output_image()"),
        "SVGF fallback must keep A-trous output available for postprocess"
    );

    let frontend_idx = pipeline
        .find("vpt_nrd_frontend.register_graph(")
        .expect("NRD frontend graph registration should exist");
    let adapter_idx = pipeline
        .find("vpt_nrd_adapter.register_graph(")
        .expect("NRD adapter graph registration should exist");
    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("SVGF temporal graph registration should exist");
    assert!(frontend_idx < adapter_idx);
    assert!(adapter_idx < temporal_idx);
}

#[test]
fn vpt_nrd_resolve_remodulates_relax_output_before_postprocess() {
    let shader = source("assets/shaders/passes/vpt_nrd_resolve.slang");
    let pass = source("src/render/passes/vpt_nrd_resolve.rs");
    let pass_mod = source("src/render/passes/mod.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let profiler = source("src/render/gpu_profiler.rs");
    let compact_shader = shader.split_whitespace().collect::<String>();
    let compact_pass = pass.split_whitespace().collect::<String>();
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();

    for token in [
        "RWTexture2D<float4> denoised_diff_radiance_hitdist",
        "RWTexture2D<float4> residual_radiance",
        "RWTexture2D<float4> material_factors",
        "RWTexture2D<float4> resolved_radiance",
        "float3 sanitize_nrd_resolve_radiance(float3 radiance)",
        "float3 nrd_ycocg_to_linear(float3 color)",
    ] {
        assert!(shader.contains(token), "NRD resolve shader missing {token}");
    }
    for token in [
        "booluse_reblur=(scene.denoiser_flags&DENOISER_MODE_MASK)==DENOISER_MODE_REBLUR;",
        "float3denoised_diffuse=sanitize_nrd_resolve_radiance(unpack_nrd_diffuse_radiance(denoised_diff_radiance_hitdist[tid.xy].rgb,use_reblur));",
        "float3residual=sanitize_nrd_resolve_radiance(residual_radiance[tid.xy].rgb);",
        "float3albedo=saturate(material_factors[tid.xy].rgb);",
        "resolved_radiance[tid.xy]=float4(sanitize_nrd_resolve_radiance(residual+denoised_diffuse*albedo),1.0);",
    ] {
        assert!(
            compact_shader.contains(token),
            "NRD resolve shader missing compact token {token}"
        );
    }

    for token in [
        "pub struct VptNrdResolvePass",
        "pub struct VptNrdResolveGraphInputs",
        "pub struct VptNrdResolveGraphOutputs",
        "pub struct VptNrdResolvePassCreateInfo",
        "pub struct VptNrdResolvePassResizeInfo",
        "pub(crate) fn descriptor_binding_specs() -> [DescriptorBindingSpec; 5]",
        "descriptor_count: 4 * frame_count as u32",
        "pub resolved_radiance: GpuImage",
        "vpt_nrd_resolved_radiance",
        "graph.add_pass(\"vpt_nrd_resolve\"",
        "GpuProfileScope::VptNrdResolve",
    ] {
        assert!(pass.contains(token), "NRD resolve pass missing {token}");
    }
    for token in [
        "builder.read_as(denoised_diff_radiance_hitdist,AccessKind::ComputeShaderRead,);",
        "builder.read_as(packed.residual_radiance,AccessKind::ComputeShaderRead);",
        "builder.read_as(packed.material_factors,AccessKind::ComputeShaderRead);",
        "builder.write_as(resolved_resource,AccessKind::ComputeShaderWrite);",
    ] {
        assert!(
            compact_pass.contains(token),
            "NRD resolve graph missing compact token {token}"
        );
    }

    assert!(pass_mod.contains("pub mod vpt_nrd_resolve;"));
    assert!(profiler.contains("VptNrdResolve"));
    assert!(profiler.contains("vpt_nrd_resolve_ms"));

    for token in [
        "use crate::render::passes::vpt_nrd_resolve::{",
        "pub vpt_nrd_resolve_pass: Option<VptNrdResolvePass>",
        "vpt_nrd_resolve_pass: None",
        "self.ensure_vpt_nrd_resolve_pass(renderer, scene_ubo);",
        "fn ensure_vpt_nrd_resolve_pass(",
        "include_bytes!(concat!(env!(\"OUT_DIR\"), \"/shaders/vpt_nrd_resolve.spv\"))",
        "VptNrdResolvePass::new(",
        "failed to resize VPT NRD resolve images",
        "vpt_nrd_resolve_ready = self.vpt_nrd_resolve_pass.is_some()",
        "pass.destroy(device, allocator);",
    ] {
        assert!(pipeline.contains(token), "pipeline missing {token}");
    }
    assert!(
        compact_pipeline.contains("vpt_nrd_adapter.is_ready()"),
        "pipeline must only consume adapter output when the native NRD backend is ready"
    );
    assert!(
        compact_pipeline.contains(
            "letnrd_resolve_outputs=ifmatches!(inputs.lighting_settings.denoiser_mode,VptDenoiserMode::Relax|VptDenoiserMode::Reblur){"
        ),
        "pipeline must resolve NRD output for requested RELAX or REBLUR mode"
    );
    assert!(
        compact_pipeline.contains(
            "denoised_diff_radiance_hitdist:nrd_adapter_outputs.resources.diff_radiance_hitdist,"
        ),
        "resolve pass must consume denoised RELAX diffuse output"
    );
    assert!(
        compact_pipeline.contains("packed:nrd_frontend_outputs.packed,"),
        "resolve pass must consume frontend residual and material factors"
    );
    assert!(
        compact_pipeline.contains("ifletSome(nrd_resolve_outputs)=nrd_resolve_outputs"),
        "postprocess input must branch between NRD resolve and SVGF fallback"
    );
    assert!(
        compact_pipeline.contains(
            "elseifinputs.lighting_settings.vpt_debug_view==VptDebugView::Final{ifletSome(nrd_resolve_outputs)=nrd_resolve_outputs"
        ),
        "NRD resolve must only override the postprocess input for the final VPT view so guide debug views remain visible"
    );
    assert!(
        compact_pipeline.contains("input_radiance:postprocess_input_radiance,"),
        "postprocess must use the selected denoiser output dependency"
    );
    assert!(
        compact_pipeline.contains("hdr_image:postprocess_hdr_image,"),
        "postprocess descriptors must bind the selected HDR source image"
    );
    assert!(
        compact_pipeline.contains(
            "letactual_effective_denoiser_mode_name=capture_effective_denoiser_mode_name(inputs.lighting_settings,nrd_resolve_available,);"
        ),
        "capture metadata must report the requested NRD mode whenever the native NRD path is available"
    );
    assert!(
        compact_pipeline.contains("effective_denoiser_mode:actual_effective_denoiser_mode_name,"),
        "capture metadata must use the actual selected denoiser route"
    );

    let adapter_idx = pipeline
        .find("vpt_nrd_adapter.register_graph(")
        .expect("NRD adapter graph registration should exist");
    let resolve_idx = pipeline
        .find("vpt_nrd_resolve.register_graph(")
        .expect("NRD resolve graph registration should exist");
    let postprocess_idx = pipeline
        .find("postprocess.register_graph(")
        .expect("postprocess graph registration should exist");
    assert!(adapter_idx < resolve_idx);
    assert!(resolve_idx < postprocess_idx);
}

#[test]
fn vpt_nrd_validation_debug_view_routes_native_overlay_to_postprocess() {
    let scene_ubo = source("src/render/scene_ubo.rs");
    let scene_common = source("assets/shaders/shared/scene_common.slang");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let compact_scene_ubo = scene_ubo.split_whitespace().collect::<String>();

    for token in [
        "VPT_DEBUG_VIEW_NRD_VALIDATION",
        "NrdValidation",
        "nrd_validation",
    ] {
        assert!(
            scene_ubo.contains(token) || scene_common.contains(token),
            "NRD validation debug view plumbing missing {token}"
        );
    }
    assert!(
        compact_scene_ubo.contains("(\"nrd_validation\",24)"),
        "NRD validation debug view should parse to GPU value 24"
    );
    assert!(
        compact_pipeline.contains(
            "enable_validation:matches!(inputs.lighting_settings.vpt_debug_view,VptDebugView::NrdValidation),"
        ),
        "NRD frame settings must enable native validation only for the validation debug view"
    );
    assert!(
        compact_pipeline.contains(
            "ifmatches!(inputs.lighting_settings.vpt_debug_view,VptDebugView::NrdValidation)"
        ) && compact_pipeline.contains("nrd_adapter_outputs.resources.validation")
            && compact_pipeline.contains("nrd_adapter_outputs.validation_image"),
        "pipeline must route native NRD validation output into postprocess when requested"
    );
}

#[test]
fn vpt_nrd_adapter_owns_backend_texture_pools() {
    let adapter = source("src/render/passes/vpt_nrd_adapter.rs");
    let compact_adapter = adapter.split_whitespace().collect::<String>();

    for token in [
        "texture_pools: Option<VptNrdTexturePools>",
        "struct VptNrdTexturePools",
        "permanent_pool: Vec<GpuImage>",
        "transient_pool: Vec<GpuImage>",
        "fn create(",
        "plan: &VptNrdTexturePoolPlan",
        ") -> Result<Self>",
        "fn create_texture_pools(",
        "fn create_texture_pool_images(",
        "fn create_texture_pool_image(",
        "\"vpt_nrd_permanent_texture_pool\"",
        "\"vpt_nrd_transient_texture_pool\"",
    ] {
        assert!(
            adapter.contains(token),
            "NRD adapter pool owner missing {token}"
        );
    }

    for token in [
        "lettexture_pools=create_texture_pools(device,allocator,&backend)?;",
        "letbackend=VptNrdAdapterBackend::initialize(info.denoiser_mode,info.width,info.height,info.relax_atrous_iteration_num,);",
        "letnew_texture_pools=matchcreate_texture_pools(device,allocator,&new_backend)",
        "new_images.destroy(device,allocator);",
        "std::mem::replace(&mutself.texture_pools,new_texture_pools)",
        "old_texture_pools.destroy(device,allocator);",
        "texture_pools.destroy(device,allocator);",
        "pool.destroy(device,allocator);",
        "usage:vk::ImageUsageFlags::STORAGE|vk::ImageUsageFlags::SAMPLED,",
    ] {
        assert!(
            compact_adapter.contains(token),
            "NRD adapter pool lifecycle missing compact token {token}"
        );
    }
}

#[test]
fn vpt_phase4_writes_noisy_radiance_and_moments_not_progressive_accumulation() {
    let shader = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let rust = std::fs::read_to_string("src/render/passes/vpt.rs").expect("vpt source is readable");

    assert!(
        shader.contains("RWTexture2D<float4> noisy_radiance_image;"),
        "VPT shader must write current-frame noisy HDR radiance"
    );
    assert!(
        shader.contains("RWTexture2D<float4> noisy_moments_image;"),
        "VPT shader must write current-frame luminance moments"
    );
    assert!(
        shader.contains("float luminance = dot(sample_radiance"),
        "VPT shader must compute luminance moments from the noisy sample"
    );
    assert!(
        !shader.contains("lerp(previous, sample_radiance, 1.0 / sample_count)"),
        "VPT must not use progressive accumulation as the interactive quality mechanism"
    );
    assert!(
        rust.contains("pub noisy_radiance_image: GpuImage")
            && rust.contains("pub noisy_moments_image: GpuImage"),
        "VptPass must own separate noisy radiance and moments images"
    );
    assert!(
        rust.contains("vpt_noisy_radiance") && rust.contains("vpt_noisy_moments"),
        "VPT image names must be explicit for captures and graph debugging"
    );
}

#[test]
fn vpt_temporal_pass_declares_reprojection_and_history_contract() {
    let rust = std::fs::read_to_string("src/render/passes/vpt_temporal.rs")
        .expect("VPT temporal pass source should exist");
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");
    let pipeline = source("src/render/vpt_pipeline.rs");

    for token in [
        "pub struct VptTemporalPass",
        "accumulated_radiance",
        "accumulated_moments_history",
        "history_length",
        "previous_accumulated_radiance",
        "previous_accumulated_moments_history",
        "pub fn record",
        "pub fn record_history_update",
    ] {
        assert!(rust.contains(token), "VPT temporal Rust missing {token}");
    }

    for token in [
        "RWTexture2D<float4> noisy_radiance_image",
        "RWTexture2D<float4> noisy_moments_image",
        "RWTexture2D<float4> surface_position_depth",
        "RWTexture2D<float4> previous_surface_position_depth",
        "RWTexture2D<float4> motion_history",
        "RWTexture2D<float4> accumulated_radiance_image",
        "RWTexture2D<float4> accumulated_moments_history_image",
        "compatible_history",
        "history_length",
        "bounded_alpha",
    ] {
        assert!(
            shader.contains(token),
            "VPT temporal shader missing {token}"
        );
    }

    assert!(pipeline.contains("vpt.register_graph("));
    assert!(pipeline.contains("vpt_temporal.register_graph("));
    assert!(pipeline.contains("vpt_temporal.register_history_update_graph("));
    assert!(pipeline.contains("postprocess.register_graph("));

    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT graph registration should exist");
    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("VPT temporal graph registration should exist");
    let history_update_idx = pipeline
        .find("vpt_temporal.register_history_update_graph(")
        .expect("VPT temporal history update graph registration should exist");
    let postprocess_idx = pipeline
        .find("postprocess.register_graph(")
        .expect("postprocess graph registration should exist");

    assert!(vpt_idx < temporal_idx);
    assert!(temporal_idx < history_update_idx);
    assert!(history_update_idx < postprocess_idx);
}

#[test]
fn vpt_atrous_pass_declares_svgf_edge_aware_filter_contract() {
    let rust = std::fs::read_to_string("src/render/passes/vpt_atrous.rs")
        .expect("VPT atrous pass source should exist");
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_atrous.slang")
        .expect("VPT atrous shader should exist");

    for token in [
        "pub struct VptAtrousPass",
        "pub filtered_radiance",
        "ping_radiance",
        "pong_radiance",
        "pub fn record",
        "pub fn output_image",
        "vpt_atrous_filtered",
        "vpt_atrous_ping",
        "vpt_atrous_pong",
    ] {
        assert!(rust.contains(token), "VPT atrous Rust missing {token}");
    }

    for token in [
        "RWTexture2D<float4> input_radiance_image",
        "RWTexture2D<float4> moments_history_image",
        "RWTexture2D<float4> surface_position_depth",
        "RWTexture2D<float4> surface_normal_roughness",
        "RWTexture2D<float4> surface_albedo_material",
        "RWTexture2D<float4> filtered_radiance_image",
        "atrous_step_width",
        "normal_weight",
        "depth_weight",
        "safe_normalize",
        "albedo_weight",
        "center_albedo_material.rgb",
        "material_weight",
        "variance_weight",
        "spatial_luminance_variance",
        "effective_luminance_variance",
        "history_length",
        "center_moments_history.z < 4.0",
        "vpt_atrous.iteration_index == 0u",
        "neighbor_moments_history",
        "clamp_radiance_to_luminance_interval",
        "struct AtrousLuminanceStats",
        "gather_atrous_luminance_stats",
        "atrous_luminance_limit",
        "clamp_center_radiance_to_spatial_interval",
        "float3 center_filtered_radiance",
        "scene.denoiser_atrous_iterations",
        "DENOISER_FLAG_ENABLED",
        "scene.vpt_debug_view != VPT_DEBUG_VIEW_FINAL",
    ] {
        assert!(shader.contains(token), "VPT atrous shader missing {token}");
    }
}

#[test]
fn app_routes_temporal_radiance_through_vpt_atrous_before_postprocess() {
    let runtime = source("src/render/runtime.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let atrous_pass = source("src/render/passes/vpt_atrous.rs");
    let postprocess_pass = source("src/render/passes/postprocess.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let compact_atrous = atrous_pass.split_whitespace().collect::<String>();
    let compact_postprocess = postprocess_pass.split_whitespace().collect::<String>();

    assert!(pipeline.contains("pub vpt_atrous_pass: Option<VptAtrousPass>"));
    assert!(pipeline.contains("pub postprocess_pass: Option<PostprocessPass>"));
    assert!(pipeline.contains("VptAtrousPass::new"));
    assert!(pipeline.contains("VptAtrousPassCreateInfo"));
    assert!(pipeline.contains("PostprocessPass::new"));
    assert!(runtime.contains("self.vpt_pipeline.ensure_passes("));
    assert!(runtime.contains("self.vpt_pipeline.record_and_execute_frame("));
    assert!(pipeline.contains("vpt_temporal.register_graph("));
    assert!(pipeline.contains("vpt_atrous.register_graph("));
    assert!(pipeline.contains("postprocess.register_graph("));
    assert!(pipeline.contains("vpt_atrous.output_image()"));
    assert!(atrous_pass.contains("graph.add_pass(\"vpt_atrous\""));
    assert!(atrous_pass.contains("GpuProfileScope::VptAtrous"));
    assert!(postprocess_pass.contains("graph.add_pass(\"postprocess\""));
    assert!(postprocess_pass.contains("GpuProfileScope::Postprocess"));

    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("VPT temporal graph registration should exist");
    let atrous_idx = pipeline
        .find("vpt_atrous.register_graph(")
        .expect("VPT atrous graph registration should exist");
    let postprocess_idx = pipeline
        .find("postprocess.register_graph(")
        .expect("postprocess graph registration should exist");

    assert!(temporal_idx < atrous_idx);
    assert!(atrous_idx < postprocess_idx);
    assert!(compact_atrous.contains("letmutatrous_input_dep=temporal_radiance;"));
    assert!(compact_atrous.contains("letmutatrous_ping_dep=atrous_ping_resource;"));
    assert!(compact_atrous.contains("letmutatrous_pong_dep=atrous_pong_resource;"));
    assert!(compact_atrous.contains("letoutput_is_final=iteration_index+1==atrous_pass_count;"));
    assert!(
        compact_atrous
            .contains("letoutput_is_ping=!output_is_final&&iteration_index.is_multiple_of(2);")
    );
    assert!(
        compact_atrous.contains("builder.read_as(atrous_input_dep,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_atrous.contains("builder.read_as(temporal_moments,AccessKind::ComputeShaderRead")
    );
    assert!(compact_atrous.contains("letatrous_filtered_resource=graph.import_image_with_access("));
    assert!(
        compact_atrous
            .contains("builder.write_as(atrous_output_resource,AccessKind::ComputeShaderWrite")
    );
    assert!(compact_pipeline.contains("letatrous_filtered_dep=atrous_outputs.filtered_radiance;"));
    assert!(compact_atrous.contains("atrous_ping_dep=atrous_writes[0];"));
    assert!(compact_atrous.contains("atrous_pong_dep=atrous_writes[0];"));
    assert!(compact_postprocess.contains("self.update_input_image(device,hdr_image,frame_slot);"));
    assert!(
        compact_postprocess
            .contains("builder.read_as(input_radiance,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_postprocess.contains(
            "builder.write_as(postprocess_output_resource,AccessKind::ComputeShaderWrite"
        )
    );
    assert!(compact_pipeline.contains("vpt_atrous.output_image()"));
}

#[test]
fn vpt_pipeline_uses_pass_owned_graph_registration_for_vpt_stages() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let pipeline_impl = pipeline
        .split("#[cfg(test)]")
        .next()
        .expect("pipeline implementation section should exist");
    let vpt_surface = source("src/render/passes/vpt_surface.rs");
    let vpt_trace = source("src/render/passes/vpt.rs");
    let vpt_temporal = source("src/render/passes/vpt_temporal.rs");
    let vpt_atrous = source("src/render/passes/vpt_atrous.rs");
    let postprocess = source("src/render/passes/postprocess.rs");

    for (name, implementation, api) in [
        (
            "VPT surface",
            vpt_surface.as_str(),
            "pub fn register_bootstrap_graph",
        ),
        ("VPT trace", vpt_trace.as_str(), "pub fn register_graph"),
        (
            "VPT temporal",
            vpt_temporal.as_str(),
            "pub fn register_graph",
        ),
        (
            "VPT temporal history",
            vpt_temporal.as_str(),
            "pub fn register_history_update_graph",
        ),
        ("VPT atrous", vpt_atrous.as_str(), "pub fn register_graph"),
        ("postprocess", postprocess.as_str(), "pub fn register_graph"),
    ] {
        assert!(
            implementation.contains(api),
            "{name} pass must expose pass-owned graph registration API {api}"
        );
    }

    for call in [
        "vpt_surface.register_bootstrap_graph(",
        "vpt.register_graph(",
        "vpt_temporal.register_graph(",
        "vpt_atrous.register_graph(",
        "vpt_temporal.register_history_update_graph(",
        "postprocess.register_graph(",
    ] {
        assert!(
            pipeline_impl.contains(call),
            "VPT pipeline must use pass-owned graph registration call {call}"
        );
    }

    for forbidden in [
        "graph.add_pass(\"vpt_surface_bootstrap\"",
        "graph.add_pass(\"vpt\"",
        "graph.add_pass(\"vpt_temporal\"",
        "graph.add_pass(\"vpt_atrous\"",
        "graph.add_pass(\"vpt_surface_history_update\"",
        "graph.add_pass(\"postprocess\"",
    ] {
        assert!(
            !pipeline_impl.contains(forbidden),
            "VPT pipeline must not manually register pass graph node {forbidden}"
        );
    }
}

#[test]
fn vpt_temporal_seeds_valid_history_when_no_previous_history_is_accepted() {
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    assert!(
        shader.contains("bool current_surface_valid = surface_position_depth[pixel].w >= 0.0;")
            && shader.contains("float history_valid = current_surface_valid ? 1.0 : 0.0;"),
        "VPT temporal must seed valid history from valid current surface samples, otherwise reuse can never start"
    );
    assert!(
        shader.contains("float history_length = current_surface_valid ? 1.0 : 0.0;"),
        "VPT temporal must seed history length only for valid current surface samples"
    );
    assert!(
        shader.contains("history_valid = 1.0;") && shader.contains("history.history_length > 0.0"),
        "accepted previous history should keep validity while increasing history length"
    );
    assert!(
        shader.contains("scene.vpt_debug_view == VPT_DEBUG_VIEW_HISTORY_VALID")
            && shader.contains("accumulated = float3(history_reused);"),
        "history_valid debug view must expose where previous-frame temporal reuse was accepted"
    );
}

#[test]
fn vpt_temporal_encodes_history_reset_generation_in_radiance_alpha_and_gates_reuse() {
    let scene_common = source("assets/shaders/shared/scene_common.slang");
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");
    let compact_shader = shader.split_whitespace().collect::<String>();

    assert!(
        scene_common.contains("history_reset_generation"),
        "scene ABI must expose history_reset_generation for temporal history gating"
    );

    for token in [
        "uintvpt_history_generation_half_bits(uintgeneration)",
        "floatvpt_history_generation_alpha(uinthistory_generation_bits)",
        "previous_history_generation_bits=f32tof16(previous_accumulated_radiance.a);",
        "previous_history_generation_bits!=current_history_generation_bits",
        "history_generation_alpha=vpt_history_generation_alpha(current_history_generation_bits);",
        "accumulated_radiance_image[pixel]=float4(accumulated,history_generation_alpha);",
    ] {
        assert!(
            compact_shader.contains(token),
            "VPT temporal history generation gate missing {token}"
        );
    }

    let generation_gate_idx = compact_shader
        .find("previous_history_generation_bits!=current_history_generation_bits")
        .expect("generation gate should exist before reusing previous history");
    let compatibility_idx = compact_shader
        .find("!compatible_history(pixel,tap_pixel)")
        .expect("surface compatibility check should exist");
    let generation_helper_idx = compact_shader
        .find("uintvpt_history_generation_half_bits(uintgeneration)")
        .expect("generation helper should exist");
    let previous_radiance_idx = compact_shader
        .find("previous_accumulated_radiance_image[tap_pixel]")
        .expect("previous accumulated radiance access should exist");
    let previous_moments_idx = compact_shader
        .find("previous_accumulated_moments_history_image[tap_pixel]")
        .expect("previous accumulated moments access should exist");

    assert!(generation_helper_idx < generation_gate_idx);
    assert!(generation_gate_idx < compatibility_idx);
    assert!(previous_radiance_idx < previous_moments_idx);
}

#[test]
fn app_does_not_rewrite_all_vpt_temporal_descriptors_per_frame() {
    let runtime = std::fs::read_to_string("src/render/runtime.rs")
        .expect("app source should be readable for VPT temporal descriptor lifetime test");
    let render_loop_start = runtime
        .find("pub fn render_frame(")
        .expect("render loop should begin a Vulkan frame");
    let render_loop_end = runtime[render_loop_start..]
        .find("fn snapshot_ucvh_frame_changes")
        .map(|offset| render_loop_start + offset)
        .expect("render loop should end the Vulkan frame");
    let frame_body = &runtime[render_loop_start..render_loop_end];

    assert!(
        !frame_body.contains("vpt_temporal.update_input_images"),
        "VPT temporal descriptor rebinding must not rewrite all frame-slot descriptor sets from the per-frame render path"
    );
}

#[test]
fn app_declares_persistent_vpt_image_layouts_from_previous_frame_final_access() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact = pipeline.split_whitespace().collect::<String>();
    let temporal = source("src/render/passes/vpt_temporal.rs");
    let postprocess = source("src/render/passes/postprocess.rs");
    let compact_temporal = temporal.split_whitespace().collect::<String>();
    let compact_postprocess = postprocess.split_whitespace().collect::<String>();

    assert!(
        compact_temporal
            .contains("lettemporal_initial_access=ifhistory_initialized{AccessKind::TransferRead"),
        "temporal radiance must be imported from the previous frame's history-copy read layout"
    );
    assert!(
        compact_temporal.contains(
            "letprevious_temporal_access=ifhistory_initialized{AccessKind::TransferWrite"
        ),
        "temporal moments must be imported from the previous frame's history-copy read layout"
    );
    assert!(
        pipeline.contains("pub postprocess_output_initialized: bool")
            && pipeline.contains("self.frame_state.reset_for_resize_or_camera_cut();")
            && compact.contains("vpt_temporal.register_graph(")
            && compact.contains("postprocess.register_graph("),
        "pipeline must hand persistent temporal/postprocess history to pass-owned graph registration"
    );
    assert!(
        compact_postprocess
            .contains("letoutput_initial_access=ifoutput_initialized{AccessKind::TransferRead"),
        "postprocess output must be imported from the previous frame's history-copy read layout"
    );
    assert!(
        compact_postprocess
            .contains("letpostprocess_output_resource=graph.import_image_with_access("),
        "postprocess output is a persistent image and must be imported with its tracked previous-frame layout"
    );
}

#[test]
fn app_resets_vpt_temporal_state_on_resize_and_key_changes() {
    let source = std::fs::read_to_string("src/app.rs")
        .expect("app source should be readable for VPT reset test");
    let runtime = std::fs::read_to_string("src/render/runtime.rs")
        .expect("runtime source should be readable for VPT reset test");
    let pipeline = std::fs::read_to_string("src/render/vpt_pipeline.rs")
        .expect("VPT pipeline source should be readable for VPT reset test");
    let temporal = std::fs::read_to_string("src/render/passes/vpt_temporal.rs")
        .expect("VPT temporal source should be readable for VPT reset test");
    let compact = pipeline.split_whitespace().collect::<String>();

    assert!(runtime.contains("self.vpt_pipeline.resize("));
    assert!(
        source
            .contains("fn resize_render_runtime(&mut self, width: u32, height: u32) -> Result<()>")
    );
    assert!(pipeline.contains("self.frame_state.reset_for_resize_or_camera_cut();"));
    assert!(compact.contains("self.frame_state.vpt_temporal_history_initialized"));
    assert!(pipeline.contains("frame.swapchain_extent.width"));
    assert!(pipeline.contains("lighting_settings.vpt_max_bounces"));
    assert!(pipeline.contains("initialized postprocess pass from VPT output"));
    assert!(pipeline.contains("skipping VPT frame until required passes are initialized"));
    assert!(pipeline.contains("graph.has_final_access(AccessKind::Present)"));
    assert!(pipeline.contains("add_swapchain_clear_present_pass"));
    assert!(!source.contains("graph.add_pass(\"primary_ray\""));
    assert!(!source.contains("primary_ray_writes"));
    assert!(pipeline.contains("graph.import_image_with_access("));
    assert!(pipeline.contains("vpt_temporal.register_graph("));
    assert!(temporal.contains("graph.add_pass(\"vpt_temporal\""));
    assert!(temporal.contains("AccessKind::ComputeShaderWrite"));
}

#[test]
fn app_keeps_vpt_first_use_sample_zero_until_noisy_radiance_is_written() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let compact = pipeline.split_whitespace().collect::<String>();
    let compact_pass = vpt_pass.split_whitespace().collect::<String>();

    assert!(
        compact.contains(
            "letscene_vpt_sample_index=ifself.frame_state.vpt_accumulation_needs_init{0}else{self.frame_state.vpt_sample_index};"
        ) && compact.contains("vpt_sample_index:scene_vpt_sample_index"),
        "scene UBO must see sample 0 while the accumulation image is still first-use"
    );
    assert!(
        compact_pass.contains("pubstructVptGraphInputs")
            && compact_pass.contains("pubaccumulation_needs_init:bool")
            && compact_pass
                .contains("letnoisy_initial_access=ifinputs.accumulation_needs_init{AccessKind::Undefined}else{AccessKind::ComputeShaderRead};"),
        "first-use VPT noisy images must start from Undefined even if internal sample state was advanced"
    );
    assert!(
        pipeline.contains("self.frame_state.last_vpt_camera_key = None;"),
        "skipped VPT frames must not advance reusable accumulation state"
    );
}

#[test]
fn app_keeps_vpt_primary_sampling_sequence_advancing_without_camera_key_resets() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact = pipeline.split_whitespace().collect::<String>();

    assert!(
        compact.contains(
            "letscene_vpt_sample_index=ifself.frame_state.vpt_accumulation_needs_init{0}else{self.frame_state.vpt_sample_index};"
        ),
        "scene UBO sample index must still be gated only by explicit accumulation initialization"
    );
    assert!(
        compact.contains(
            "ifvpt_accumulation_written{self.frame_state.vpt_sample_index=self.frame_state.vpt_sample_index.saturating_add(1);"
        ),
        "VPT sample index must advance when a noisy accumulation frame is written"
    );
    assert!(
        !compact.contains("ifself.frame_state.last_vpt_camera_key==Some(camera_key){"),
        "moving the camera must not reset the primary sampling sequence"
    );
    assert!(
        !compact.contains("self.frame_state.vpt_sample_index=0;self.frame_state.last_vpt_camera_key=Some(camera_key);"),
        "camera-key changes must not zero the primary sampling index"
    );
}

#[test]
fn app_supports_frame_limited_runtime_smoke_validation() {
    let source = std::fs::read_to_string("src/app.rs")
        .expect("app source should be readable for runtime smoke validation test");

    assert!(source.contains("REVOLUMETRIC_EXIT_AFTER_FRAMES"));
    assert!(source.contains("exit_after_frames: parse_exit_after_frames()"));
    assert!(source.contains("if let Some(limit) = self.exit_after_frames"));
    assert!(source.contains("event_loop.exit();"));
}

#[test]
fn desktop_builds_reject_placeholder_shaders_instead_of_opening_black_window() {
    let build = source("build.rs");

    for token in [
        "fn desktop_feature_enabled() -> bool",
        "CARGO_FEATURE_DESKTOP",
        "REVOLUMETRIC_SLANGC",
        "fn slangc_command() -> Command",
        "REVOLUMETRIC_SHADER_COMPILE=skip cannot be used with the desktop feature",
        "desktop builds require real shaders",
    ] {
        assert!(build.contains(token), "build.rs missing {token}");
    }
}

#[test]
fn build_script_tracks_ui_shaders_and_shader_compile_modes() {
    let build = source("build.rs");
    let build_support = source("src/build_support.rs");

    for token in [
        "#[path = \"src/build_support.rs\"]",
        "mod build_support;",
        "for job in ui_shader_jobs(shader_dir) {",
        "shader_jobs.push(job);",
        "if shader_compile_mode == ShaderCompileMode::Skip",
        "ShaderCompileMode::Auto =>",
        "ShaderCompileMode::Strict =>",
        "ShaderCompileMode::Skip =>",
        "write_placeholder_spirv_files(&shader_jobs, &out_dir);",
        "cargo:rerun-if-changed=assets/shaders",
        "cargo:rerun-if-changed=assets/shaders/passes",
    ] {
        assert!(build.contains(token), "build.rs missing {token}");
    }
    for token in [
        "pub fn ui_shader_jobs(shader_dir: &Path) -> Vec<ShaderJobSpec>",
        "shader_dir.join(\"ui\").join(\"egui.vert.slang\")",
        "shader_dir.join(\"ui\").join(\"egui.frag.slang\")",
        "ui_shader_jobs_only_collects_existing_ui_shaders",
    ] {
        assert!(
            build_support.contains(token),
            "src/build_support.rs missing {token}"
        );
    }
}

#[test]
fn app_registers_vpt_graph_without_primary_ray_pass() {
    let app = source("src/app.rs");
    let runtime = source("src/render/runtime.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");

    assert!(runtime.contains("self.vpt_pipeline.record_and_execute_frame("));
    assert!(!app.contains("graph.add_pass(\"vpt\""));
    assert!(pipeline.contains("vpt.register_graph("));
    assert!(pipeline.contains("postprocess.register_graph("));
    assert!(!pipeline.contains("graph.add_pass(\"primary_ray\""));
    assert!(vpt_pass.contains("graph.add_pass(\"vpt\""));
    assert!(!vpt_pass.contains("graph.add_pass(\"primary_ray\""));
    assert!(!app.contains("let use_vpt ="));
}

#[test]
fn active_runtime_has_no_vct_primary_or_lighting_path() {
    let app_source = std::fs::read_to_string("src/app.rs")
        .expect("app source should be readable for VPT-only runtime test");
    let passes_mod = std::fs::read_to_string("src/render/passes/mod.rs")
        .expect("passes module source should be readable for VPT-only runtime test");
    let readme = std::fs::read_to_string("README.md")
        .expect("README should be readable for VPT-only runtime test");
    let active_source = format!(
        "{}\n{}\n{}",
        app_source
            .split("#[cfg(test)]")
            .next()
            .unwrap_or(&app_source),
        passes_mod,
        readme,
    );

    for forbidden in [
        concat!("Lighting", "Pass"),
        concat!("PrimaryRay", "Pass"),
        concat!("graph.add_pass(\"", "lighting", "\""),
        concat!("graph.add_pass(\"", "primary_ray", "\""),
        concat!("lighting", ".spv"),
        concat!("primary_ray", ".spv"),
        "pub mod lighting;",
        "pub mod primary_ray;",
        concat!("REVOLUMETRIC_", "VCT"),
        concat!("REVOLUMETRIC_RENDER_MODE=", "vct"),
        concat!("Default is `", "vct", "`"),
        concat!("VCT", "-first"),
        concat!("vct", "_common", ".slang"),
    ] {
        assert!(
            !active_source.contains(forbidden),
            "forbidden active VCT token remains: {forbidden}"
        );
    }
    for deleted_file in [
        "src/render/passes/lighting.rs",
        "src/render/passes/primary_ray.rs",
        "assets/shaders/passes/lighting.slang",
        "assets/shaders/passes/primary_ray.slang",
        concat!("assets/shaders/shared/", "vct", "_common", ".slang"),
    ] {
        assert!(
            !std::path::Path::new(deleted_file).exists(),
            "deleted VCT artifact still exists: {deleted_file}"
        );
    }
}

#[test]
fn vpt_history_abi_declares_surface_and_reprojection_contract() {
    let rust = std::fs::read_to_string("src/render/vpt_history.rs")
        .expect("VPT history ABI source should exist");
    let slang = std::fs::read_to_string("assets/shaders/shared/vpt_history_common.slang")
        .expect("VPT history Slang ABI should exist");

    for token in [
        "GpuVptHistoryUniforms",
        "current_view_proj",
        "previous_view_proj",
        "current_resolution",
        "previous_resolution",
        "current_jitter",
        "previous_jitter",
        "history_reset_generation",
        "VPT_HISTORY_FLAG_CAMERA_CUT",
        "VPT_HISTORY_FLAG_RESIZE",
        "VPT_HISTORY_FLAG_SCENE_INVALIDATED",
        "VPT_HISTORY_FLAG_LIGHTS_INVALIDATED",
        "GpuVptSurfacePixel",
        "linear_depth",
        "material_id",
        "history_confidence",
    ] {
        assert!(rust.contains(token), "Rust VPT history ABI missing {token}");
    }
    for token in [
        "struct VptHistoryUniforms",
        "float4x4 current_view_proj",
        "float4x4 previous_view_proj",
        "uint2 current_resolution",
        "uint2 previous_resolution",
        "float2 current_jitter",
        "float2 previous_jitter",
        "uint history_reset_generation",
        "static const uint VPT_HISTORY_FLAG_CAMERA_CUT",
        "static const uint VPT_HISTORY_FLAG_LIGHTS_INVALIDATED",
        "float2 vpt_previous_pixel_center_from_motion",
        "float2 vpt_history_sample_from_motion",
        "struct VptSurfacePixel",
    ] {
        assert!(
            slang.contains(token),
            "Slang VPT history ABI missing {token}"
        );
    }
}

#[test]
fn vpt_surface_pass_declares_owned_surface_outputs() {
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let rust = std::fs::read_to_string("src/render/passes/vpt_surface.rs")
        .expect("VPT surface pass source should exist");

    for token in [
        "RWTexture2D<float4> surface_position_depth",
        "RWTexture2D<float4> surface_normal_roughness",
        "RWTexture2D<float4> surface_albedo_material",
        "RWTexture2D<float4> motion_history",
        "trace_primary_ray(",
        "position_depth = float4(0.0, 0.0, 0.0, -1.0)",
        "voxel_material(hit.cell)",
        "surface_position_depth[pixel] = float4(hit.position, hit.t);",
    ] {
        assert!(shader.contains(token), "VPT surface shader missing {token}");
    }
    assert!(
        !shader.contains("distance(origin, hit.position)"),
        "VPT surface pass should use DDA hit.t instead of recomputing hit distance"
    );

    for token in [
        "pub struct VptSurfacePass",
        "surface_position_depth",
        "surface_normal_roughness",
        "surface_albedo_material",
        "motion_history",
        "pub fn resize_images",
        "pub fn record",
        "pub fn destroy",
    ] {
        assert!(rust.contains(token), "VPT surface pass missing {token}");
    }
    assert!(!rust.contains("cmd_pipeline_barrier"));
    assert!(!rust.contains("ImageMemoryBarrier"));
}

#[test]
fn vpt_surface_motion_uses_history_uniform_reprojection() {
    let shader = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let rust = std::fs::read_to_string("src/render/passes/vpt_surface.rs")
        .expect("VPT surface pass source should exist");
    let pipeline = source("src/render/vpt_pipeline.rs");

    assert!(shader.contains("ConstantBuffer<VptHistoryUniforms> vpt_history"));
    assert!(shader.contains("project_previous_pixel"));
    assert!(shader.contains("vpt_history.current_view_proj"));
    assert!(shader.contains("vpt_history.previous_view_proj"));
    assert!(shader.contains("vpt_history.previous_resolution"));
    assert!(shader.contains("VPT_HISTORY_FLAG_CAMERA_CUT"));
    assert!(shader.contains("current_clip.w <= 1.0e-5"));
    assert!(shader.contains("previous_clip.w <= 1.0e-5"));
    assert!(shader.contains("float2 current_pixel = float2(pixel) + 0.5"));
    assert!(shader.contains("float2 motion_delta = previous_pixel - current_pixel"));
    assert!(shader.contains("float4 motion = float4(motion_delta, 0.0, 1.0);"));
    assert!(shader.contains("motion.z = previous_view_z - view_z;"));
    assert!(!shader.contains("motion_history[pixel] = float4(float2(pixel), 0.0, 1.0);"));

    assert!(rust.contains("GpuVptHistoryUniforms"));
    assert!(rust.contains("history_uniform_buffers"));
    assert!(rust.contains("update_history_uniforms"));
    assert!(rust.contains("std::mem::size_of::<GpuVptHistoryUniforms>()"));

    assert!(pipeline.contains("update_history_uniforms"));
    assert!(pipeline.contains("previous_vpt_view_proj"));
    assert!(pipeline.contains("compute_view_proj"));
}

#[test]
fn vpt_motion_id_uses_independent_guide_without_changing_reprojection_consumers() {
    let ray = source("assets/shaders/shared/ray.slang");
    let traversal = source("assets/shaders/shared/voxel_traverse.slang");
    let surface = source("assets/shaders/passes/vpt_surface.slang");
    let temporal = source("assets/shaders/passes/vpt_temporal.slang");
    let restir_di = source("assets/shaders/passes/restir_di_temporal.slang");
    let area_restir = source("assets/shaders/passes/area_restir_temporal.slang");

    for token in [
        "static const uint VPT_MOTION_ID_INVALID = 0u;",
        "static const uint VPT_MOTION_ID_STATIC_UCVH = 1u;",
        "uint  motion_id",
        "VPT_MOTION_ID_INVALID",
    ] {
        assert!(ray.contains(token), "ray hit payload missing {token}");
    }
    assert!(
        traversal.contains("result.motion_id = VPT_MOTION_ID_STATIC_UCVH;"),
        "primary UCVH hits should carry an explicit static motion source id"
    );
    assert!(
        surface.contains("RWTexture2D<uint> surface_motion_id")
            && surface.contains("surface_motion_id[pixel] = hit.motion_id;")
            && surface.contains("surface_motion_id[pixel] = VPT_MOTION_ID_INVALID;"),
        "surface motion guide should preserve the hit motion id in an independent uint guide"
    );
    assert!(
        surface.contains("motion.z = previous_view_z - view_z;")
            && surface.contains("motion_history[pixel] = motion;")
            && !surface.contains("motion.z = float(hit.motion_id);"),
        "motion.z must carry the NRD 2.5D depth delta, not the semantic motion id"
    );
    for consumer in [restir_di, area_restir] {
        assert!(
            consumer.contains("vpt_history_sample_from_motion(pixel, motion);"),
            "temporal consumers should continue using the shared xy/w reprojection contract"
        );
        assert!(
            !consumer.contains("motion.z"),
            "motion id plumbing must not change temporal reprojection acceptance yet"
        );
    }

    assert!(
        surface.contains("motion.z = previous_view_z - view_z;")
            && surface.contains("motion_history[pixel] = motion;")
            && temporal.contains("visualize_nrd_motion_z(motion.z, motion_valid)")
            && temporal.contains("vpt_history_sample_from_motion(pixel, motion);"),
        "NRD z-guide must be written in surface and only consumed as a debug view in temporal"
    );
}

#[test]
fn vpt_pipeline_keeps_camera_projection_unjittered_until_taa_resolve_exists() {
    let pipeline = source("src/render/vpt_pipeline.rs");

    assert!(
        !pipeline.contains("taa_frame_jitter(frame.frame_index)"),
        "VPT must not apply frame-varying camera jitter until temporal resolve is explicitly jitter-stable"
    );
    assert!(
        pipeline.contains("current_jitter: [0.0, 0.0]"),
        "history uniforms should declare zero jitter while the VPT camera path is unjittered"
    );
    assert!(
        pipeline.contains("previous_jitter: [0.0, 0.0]"),
        "history uniforms should declare zero previous jitter while the VPT camera path is unjittered"
    );
}

#[test]
fn vpt_temporal_bilinear_reprojection_converts_pixel_centers_to_texel_corners() {
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");
    let common = std::fs::read_to_string("assets/shaders/shared/vpt_history_common.slang")
        .expect("VPT history common shader should exist");

    assert!(
        surface.contains("float2 current_pixel = float2(")
            && surface.contains("float2 previous_pixel = float2("),
        "surface pass should compute current and previous pixel centers before differencing them"
    );
    assert!(
        common.contains("return float2(pixel) + 0.5 + motion.xy;"),
        "shared helper must define previous pixel centers from motion delta"
    );
    assert!(
        temporal.contains("float2 history_sample = vpt_history_sample_from_motion(pixel, motion);"),
        "4-tap temporal reprojection must reconstruct the previous pixel-center coordinate from the motion delta"
    );
    assert!(
        temporal.contains("int2 previous_base_pixel = int2(floor(history_sample));"),
        "bilinear reprojection must keep a signed base pixel so edge taps can be rejected per tap"
    );
    assert!(
        temporal.contains(
            "float2 history_fraction = saturate(history_sample - float2(previous_base_pixel));"
        ),
        "bilinear weights must be computed from the center-corrected base pixel"
    );
    assert!(
        !temporal.contains("uint2 previous_pixel = uint2(motion.xy);"),
        "motion.xy is now a delta, not an absolute previous pixel coordinate"
    );
    assert!(
        !temporal.contains("float2 history_sample = motion.xy - 0.5;"),
        "subtracting 0.5 from a delta motion vector would bias the history sample"
    );
}

#[test]
fn vpt_temporal_bilinearly_samples_reprojected_history_and_clamps_variance_outliers() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    assert!(temporal.contains("sample_reprojected_history("));
    assert!(temporal.contains("history_tap_offsets"));
    assert!(temporal.contains("history_fraction"));
    assert!(temporal.contains("history_reliability"));
    assert!(temporal.contains("clamp_noisy_radiance_to_history"));
    assert!(temporal.contains("history_variance"));
}

#[test]
fn vpt_temporal_applies_edge_aware_spatial_firefly_clamp_before_history() {
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    for token in [
        "static const int2 firefly_tap_offsets[8]",
        "struct FireflyClampStats",
        "compatible_current_surface",
        "gather_firefly_clamp_stats",
        "clamp_noisy_radiance_to_spatial_firefly_stats",
        "spatial_firefly_luma_limit",
    ] {
        assert!(
            temporal.contains(token),
            "VPT temporal shader missing firefly clamp token {token}"
        );
    }

    assert!(
        temporal.contains("if (surface_normal_roughness[pixel].w > 1.0e-4)"),
        "firefly clamp must skip visible emissive surfaces instead of crushing real light sources"
    );
    assert!(
        temporal.contains("noisy_radiance = float4(clamped_noisy.radiance, noisy_radiance.a);"),
        "spatial firefly clamp must run before temporal history blending consumes noisy_radiance"
    );
    assert!(
        temporal.contains("clamped.moments = float2(clamped_luma, clamped_luma * clamped_luma);"),
        "firefly clamp must keep luminance moments consistent with the clamped radiance"
    );
    assert!(
        surface.contains("float emissive_luma = material_emissive_luminance(hit.cell);")
            && surface.contains(
                "surface_normal_roughness[pixel] = float4(normalize(hit.normal), emissive_luma);"
            ),
        "VPT surface pass must expose emissive luminance for high-quality firefly clamp rejection"
    );
}

#[test]
fn vpt_temporal_firefly_clamp_uses_robust_trimmed_neighbors_not_max_neighbor() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    for token in [
        "float sorted_luma[8]",
        "sort_firefly_luma",
        "stats.median_luma",
        "stats.trimmed_mean_luma",
        "stats.robust_sigma_luma",
        "FIREFLY_CLAMP_MEDIAN_SCALE",
        "FIREFLY_CLAMP_TRIMMED_MEAN_SCALE",
    ] {
        assert!(
            temporal.contains(token),
            "VPT temporal shader missing robust clamp token {token}"
        );
    }
    assert!(
        !temporal.contains("max_neighbor_luma * FIREFLY_CLAMP_NEIGHBOR_SCALE"),
        "firefly clamp must not let one bright neighbor raise the limit for nearby outliers"
    );
}

#[test]
fn vpt_temporal_firefly_clamp_sanitizes_history_blended_output() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    assert!(
        temporal.contains("FireflyClampStats firefly_stats = (FireflyClampStats)0;"),
        "firefly stats must outlive the raw clamp so they can sanitize temporal history output"
    );
    assert!(
        temporal.contains(
            "ClampedRadianceSample output_clamped = clamp_noisy_radiance_to_spatial_firefly_stats("
        ) && temporal.contains("accumulated = output_clamped.radiance;")
            && temporal.contains("accumulated_moments = output_clamped.moments;"),
        "firefly clamp must also be applied after temporal blending, otherwise old history fireflies persist"
    );
}

#[test]
fn vpt_temporal_denoiser_disabled_short_circuits_to_raw_passthrough() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");
    let compact_temporal = temporal.split_whitespace().collect::<String>();

    let off_idx = compact_temporal
        .find("if((scene.denoiser_flags&DENOISER_FLAG_ENABLED)==0u){")
        .expect("VPT temporal shader should short-circuit when denoiser is disabled");
    let clamp_idx = compact_temporal
        .find("firefly_stats=gather_firefly_clamp_stats(pixel,scene.resolution);")
        .expect("VPT temporal shader should still contain the spatial firefly clamp path");
    let history_idx = compact_temporal
        .find("ClampedRadianceSamplehistory_clamped_noisy=clamp_noisy_radiance_to_history(")
        .expect("VPT temporal shader should still contain the history clamp path");

    assert!(
        compact_temporal.contains(
            "if((scene.denoiser_flags&DENOISER_FLAG_ENABLED)==0u){accumulated_radiance_image[pixel]=float4(noisy_radiance.rgb,history_generation_alpha);accumulated_moments_history_image[pixel]=float4(noisy_moments.xy,current_surface_valid?1.0:0.0,current_surface_valid?1.0:0.0);return;}"
        ),
        "denoiser-off temporal path must write raw noisy radiance and return before filtering"
    );
    assert!(off_idx < clamp_idx);
    assert!(off_idx < history_idx);
}

#[test]
fn app_records_egui_overlay_only_after_postprocess_capture_and_swapchain_blit() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();

    for token in [
        "let postprocess_outputs = postprocess.register_graph(",
        "graph.add_pass(\"capture_postprocess\"",
        "graph.add_pass(\"blit_to_swapchain\"",
        "graph.add_pass(\"egui_overlay\"",
        "builder.finish_as(swapchain_after_blit, AccessKind::Present);",
    ] {
        assert!(pipeline.contains(token), "VPT pipeline missing {token}");
    }

    assert!(
        compact_pipeline.contains(
            "builder.read_as(dep_handle,AccessKind::TransferRead);ifletSome(capture_dep)=capture_dependency{builder.depend_on(capture_dep);}builder.write_as(swapchain_dep,AccessKind::TransferWrite);if!has_egui_overlay{builder.finish_as(swapchain_dep,AccessKind::Present);}"
        ),
        "swapchain blit must depend on postprocess capture and present directly only when no UI overlay is recorded"
    );
    assert!(
        compact_pipeline.contains(
            "letswapchain_after_blit=blit_writes[0];iflet(Some(egui_renderer),Some(egui_frame))=(egui_renderer,egui_frame){graph.add_pass(\"egui_overlay\",QueueType::Graphics,|builder|{builder.write_as(swapchain_after_blit,AccessKind::ColorAttachmentWrite);builder.finish_as(swapchain_after_blit,AccessKind::Present);"
        ),
        "egui overlay must draw onto the swapchain after blit and must not feed postprocess or capture"
    );

    let postprocess_idx = pipeline
        .find("let postprocess_outputs = postprocess.register_graph(")
        .expect("postprocess graph should exist");
    let capture_idx = pipeline
        .find("graph.add_pass(\"capture_postprocess\"")
        .expect("capture graph should exist");
    let blit_idx = pipeline
        .find("graph.add_pass(\"blit_to_swapchain\"")
        .expect("blit graph should exist");
    let egui_idx = pipeline
        .find("graph.add_pass(\"egui_overlay\"")
        .expect("egui overlay graph should exist");

    assert!(postprocess_idx < capture_idx);
    assert!(capture_idx < blit_idx);
    assert!(blit_idx < egui_idx);
}

#[test]
fn vpt_temporal_motion_debug_view_encodes_reprojection_delta() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    assert!(
        temporal.contains("VPT_DEBUG_VIEW_MOTION"),
        "temporal shader should expose motion history for reprojection debugging"
    );
    assert!(
        temporal.contains("motion.xy"),
        "motion debug view should show the raw motion delta"
    );
}

#[test]
fn vpt_debug_views_are_actually_routed_to_temporal_output() {
    let temporal = std::fs::read_to_string("assets/shaders/passes/vpt_temporal.slang")
        .expect("VPT temporal shader should exist");

    for token in [
        "VPT_DEBUG_VIEW_RAW",
        "VPT_DEBUG_VIEW_RESERVOIR_WEIGHT",
        "VPT_DEBUG_VIEW_DIRECT",
        "VPT_DEBUG_VIEW_INDIRECT",
        "VPT_DEBUG_VIEW_NORMAL",
        "VPT_DEBUG_VIEW_DEPTH",
        "VPT_DEBUG_VIEW_VARIANCE",
    ] {
        assert!(
            temporal.contains(token),
            "parsed VPT debug view is not consumed by temporal output: {token}"
        );
    }

    assert!(
        temporal.contains("debug_view_bypasses_temporal(scene.vpt_debug_view)")
            && temporal
                .contains(
                    "accumulated_radiance_image[pixel] = float4(noisy_radiance.rgb, history_generation_alpha);"
                )
            && temporal.contains("return;"),
        "raw/ReSTIR/direct/indirect debug views must bypass temporal reuse and firefly clamps"
    );
    assert!(
        temporal.contains("visualize_luminance_variance("),
        "variance debug view should expose temporal moment variance instead of silently showing final color"
    );
}

#[test]
fn vpt_nrd_guide_debug_views_are_routed_to_temporal_output() {
    let scene_common = source("assets/shaders/shared/scene_common.slang");
    let temporal = source("assets/shaders/passes/vpt_temporal.slang");
    let temporal_rs = source("src/render/passes/vpt_temporal.rs");
    let compact_temporal_rs = temporal_rs.split_whitespace().collect::<String>();

    for token in [
        "VPT_DEBUG_VIEW_NRD_NORMAL_ROUGHNESS",
        "VPT_DEBUG_VIEW_NRD_VIEWZ",
        "VPT_DEBUG_VIEW_NRD_MOTION",
        "VPT_DEBUG_VIEW_NRD_MOTION_Z",
    ] {
        assert!(
            scene_common.contains(token),
            "scene common missing NRD guide debug constant {token}"
        );
        assert!(
            temporal.contains(token),
            "VPT temporal shader missing NRD guide debug token {token}"
        );
    }

    for token in [
        "RWTexture2D<float> surface_view_z",
        "visualize_nrd_normal_roughness",
        "visualize_nrd_motion_z",
        "surface_view_z[pixel]",
        "motion.z",
        "accumulated_radiance_image[pixel] = float4(guide_debug, history_generation_alpha);",
    ] {
        assert!(
            temporal.contains(token),
            "VPT temporal shader missing NRD guide debug routing token {token}"
        );
    }

    for token in [
        "descriptor_count: 19 * frame_count as u32",
        "&vpt_surface.surface_view_z",
    ] {
        assert!(
            temporal_rs.contains(token) || compact_temporal_rs.contains(token),
            "VPT temporal Rust/tests missing NRD guide descriptor token {token}"
        );
    }
}

#[test]
fn vpt_trace_shader_emits_isolated_direct_indirect_and_reservoir_debug_samples() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));

    for token in [
        "struct VptTraceSample",
        "sample.direct_radiance",
        "sample.indirect_radiance",
        "sample.reservoir_weight",
        "VPT_DEBUG_VIEW_RESERVOIR_WEIGHT",
        "VPT_DEBUG_VIEW_DIRECT",
        "VPT_DEBUG_VIEW_INDIRECT",
        "visualize_restir_reservoir_weight",
    ] {
        assert!(
            source.contains(token),
            "VPT trace shader missing diagnostic token {token}"
        );
    }
}

#[test]
fn vpt_trace_shader_exposes_voxel_traversal_debug_views() {
    let scene_common = normalized_source(include_str!(
        "../../../../assets/shaders/shared/scene_common.slang"
    ));
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let temporal = normalized_source(include_str!(
        "../../../../assets/shaders/passes/vpt_temporal.slang"
    ));

    for token in [
        "VPT_DEBUG_VIEW_VOXEL_BRICK",
        "VPT_DEBUG_VIEW_VOXEL_LOCAL",
        "VPT_DEBUG_VIEW_VOXEL_HIT",
    ] {
        assert!(
            scene_common.contains(token),
            "scene_common missing voxel debug constant {token}"
        );
        assert!(
            source.contains(token),
            "VPT shader missing voxel debug routing token {token}"
        );
        assert!(
            temporal.contains(token),
            "temporal shader must bypass voxel traversal debug token {token}"
        );
    }

    for token in [
        "visualize_voxel_brick_id",
        "visualize_voxel_local_coord",
        "visualize_voxel_hit",
        "sample.first_hit",
        "first_hit.brick_id",
        "first_hit.local",
        "first_hit.normal",
    ] {
        assert!(
            source.contains(token),
            "VPT shader missing voxel traversal debug token {token}"
        );
    }
}

#[test]
fn app_runs_vpt_surface_before_restir_and_vpt_trace() {
    let runtime = source("src/render/runtime.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let surface_pass = source("src/render/passes/vpt_surface.rs");
    let area_pass = std::fs::read_to_string("src/render/passes/area_restir.rs")
        .expect("Area ReSTIR pass source should be readable");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let bootstrap_surface_idx = pipeline
        .find("vpt_surface.register_bootstrap_graph(")
        .expect("VPT bootstrap surface graph pass should exist");
    let area_register_idx = pipeline
        .find("area_restir.register_graph(")
        .expect("Area ReSTIR graph registration should exist");
    let selected_surface_idx = area_pass
        .find("\"vpt_surface_selected\"")
        .expect("VPT selected surface graph pass should exist");
    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT trace graph pass should exist after surface registration");

    assert!(runtime.contains("vpt_pipeline: VptRuntimePipeline"));
    assert!(runtime.contains("self.vpt_pipeline.ensure_passes("));
    assert!(pipeline.contains("pub vpt_surface_pass: Option<VptSurfacePass>"));
    assert!(pipeline.contains("VptSurfacePass::new"));
    assert!(surface_pass.contains("\"vpt_surface_bootstrap\""));
    assert!(bootstrap_surface_idx < area_register_idx);
    assert!(area_pass.find("\"area_restir_initial\"") < Some(selected_surface_idx));
    assert!(area_register_idx < vpt_idx);
    assert!(vpt_pass.contains("graph.add_pass(\"vpt\""));
    if let Some(restir_idx) = pipeline.find("restir_di.register_graph(") {
        assert!(area_register_idx < restir_idx);
    }
}

#[test]
fn app_profiles_bootstrap_and_selected_vpt_surface_with_distinct_query_scopes() {
    let surface_pass = source("src/render/passes/vpt_surface.rs");
    let area_pass = std::fs::read_to_string("src/render/passes/area_restir.rs")
        .expect("Area ReSTIR pass source should be readable");
    let profiler = std::fs::read_to_string("src/render/gpu_profiler.rs")
        .expect("GPU profiler source should be readable");

    assert!(profiler.contains("VptSurfaceBootstrap"));
    assert!(profiler.contains("VptSurfaceSelected"));
    assert!(surface_pass.contains("GpuProfileScope::VptSurfaceBootstrap"));
    assert!(area_pass.contains("GpuProfileScope::VptSurfaceSelected"));
    assert!(
        !surface_pass.contains("GpuProfileScope::VptSurface,")
            && !area_pass.contains("GpuProfileScope::VptSurface,"),
        "bootstrap and selected surface passes must not reuse one timestamp query scope"
    );
}

#[test]
fn app_keeps_restir_di_behind_vpt_setting() {
    let source = std::fs::read_to_string("src/app.rs")
        .expect("app source should be readable for ReSTIR-DI app wiring test");
    let runtime = std::fs::read_to_string("src/render/runtime.rs")
        .expect("runtime source should be readable for ReSTIR-DI app wiring test");
    let pipeline = std::fs::read_to_string("src/render/vpt_pipeline.rs")
        .expect("VPT pipeline source should be readable for ReSTIR-DI app wiring test");
    let compact_source = source.split_whitespace().collect::<String>();

    assert!(source.contains("RestirDiSettings::from_env"));
    assert!(source.contains("restir_di_settings: RestirDiSettings"));
    assert!(pipeline.contains("pub restir_di_pass: Option<RestirDiPass>"));
    assert!(source.contains("fn restir_di_vpt_enabled(&self) -> bool"));
    assert!(source.contains("let restir_di_enabled = self.restir_di_vpt_enabled();"));
    assert!(runtime.contains("self.vpt_pipeline.ensure_passes("));
    assert!(
        compact_source.contains("self.restir_di_settings.enabled"),
        "ReSTIR-DI must stay disabled unless the explicit setting is enabled"
    );
    assert!(
        pipeline.contains("fn ensure_restir_di_pass")
            && pipeline.contains("if self.restir_di_pass.is_some() || !restir_di_enabled {"),
        "ReSTIR-DI pass creation must be nested behind the explicit setting guard"
    );
}

#[test]
fn vpt_shader_can_resolve_restir_di_direct_light_when_enabled() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));

    assert!(source.contains("#include \"restir_di_common.slang\""));
    assert!(source.contains("ConstantBuffer<RestirDiUniforms> restir;"));
    assert!(source.contains("StructuredBuffer<RestirDiReservoir> restir_reservoirs;"));
    assert!(source.contains("resolve_restir_di_direct_light"));
    assert!(source.contains("if (restir.enabled != 0u && bounce == 0u)"));
}

#[test]
fn vpt_restir_direct_resolve_uses_area_sample_position_without_pdf_amplification() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));

    for token in [
        "restir_di_emissive_geometry_term",
        "reservoir.sample_radiance.rgb * geometry_term",
        "restir_di_light_visible_from_hit(hit, reservoir, scene)",
        "float4 hit_position_depth = float4(hit.position, max(hit.t, 0.0));",
        "float4 hit_normal_roughness = float4(normalize(hit.normal), 0.0);",
        "float4 hit_albedo_material = float4(material_cell_albedo(hit.cell), float(voxel_material(hit.cell)));",
        "restir_di_target_pdf_for_reservoir(",
        "hit_reservoir.target_pdf = hit_target_pdf;",
        "float selected_weight = restir_di_bounded_selected_weight(hit_reservoir);",
    ] {
        assert!(
            source.contains(token),
            "VPT Area ReSTIR resolve missing token {token}"
        );
    }
    assert!(
        !source.contains("reservoir.sample_radiance.rgb / sample_pdf"),
        "cluster color_power already represents total emissive power; dividing final radiance by area PDF causes brightness explosions"
    );
}

#[test]
fn vpt_restir_direct_resolve_applies_lambertian_brdf_factor() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));

    for token in [
        "LIGHTING_INV_PI",
        "sample.radiance = albedo * LIGHTING_INV_PI * reservoir.sample_radiance.rgb * sun_term * selected_weight;",
        "sample.radiance = albedo * LIGHTING_INV_PI * reservoir.sample_radiance.rgb * geometry_term * selected_weight;",
    ] {
        assert!(
            source.contains(token),
            "VPT ReSTIR direct resolve must shade reservoirs with Lambertian f=albedo/pi; missing token {token}"
        );
    }
}

#[test]
fn vpt_restir_di_visibility_uses_any_hit_shadow_traversal() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "bool trace_any_hit_ray(",
        "bool trace_any_hit_ray_skip_voxel(",
        "bool brick_any_hit(",
        "float max_t",
        "StructuredBuffer<BrickOccupancy> occupancy_buf",
        "ray_axis_t_max(ray.origin.x, ray.direction.x, inv_dir.x",
        "ray_axis_t_max(ray.origin.y, ray.direction.y, inv_dir.y",
        "ray_axis_t_max(ray.origin.z, ray.direction.z, inv_dir.z",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing any-hit token {token}"
        );
    }
    assert!(
        !traverse.contains(
            "bool trace_any_hit_ray(\n    Ray ray,\n    float max_t,\n    ConstantBuffer<UcvhConfig> config_buf,\n    StructuredBuffer<NodeL0> hierarchy_l0,\n    StructuredBuffer<BrickOccupancy> occupancy_buf,\n    StructuredBuffer<VoxelCell> material_buf"
        ),
        "any-hit shadow traversal must not bind or read material cells"
    );
    assert!(
        vpt.contains(
            "bool shadow_occluded = voxel_shadow_occluded_from_hit(hit, shadow_dir, max_light_t);"
        ) && vpt.contains("return !shadow_occluded;"),
        "VPT ReSTIR-DI visibility should use source-voxel-skipping boolean any-hit shadow traversal"
    );
    assert!(
        !vpt.contains("HitResult occluder = trace_primary_ray(shadow_ray"),
        "VPT ReSTIR-DI visibility must not run full material-returning primary traversal for shadow rays"
    );
}

#[test]
fn voxel_primary_traversal_reads_occupancy_bits_on_demand() {
    let traverse =
        crate::render::source_checks::read_source("assets/shaders/shared/voxel_traverse.slang");
    let common =
        crate::render::source_checks::read_source("assets/shaders/shared/voxel_common.slang");

    for token in [
        "bool brick_dda(\n    StructuredBuffer<BrickOccupancy> occupancy_buf,\n    uint brick_id,",
        "if (occupancy_buf[node.brick_id].count > 0u) {",
        "if (brick_dda(",
        "occupancy_buf,\n                    node.brick_id,\n                    local_origin,\n                    ray.direction,",
        "bool brick_any_hit(\n    StructuredBuffer<BrickOccupancy> occupancy_buf,\n    uint brick_id,",
        "if (read_occupancy_bit(occupancy_buf, brick_id, uint3(coord))) {",
    ] {
        assert!(
            traverse.contains(token),
            "primary brick traversal should read occupancy bits on demand token {token}"
        );
    }

    for token in [
        "bool read_occupancy_bit(StructuredBuffer<BrickOccupancy> occupancy_buf, uint brick_id, uint3 local)",
        "return (occupancy_buf[brick_id].bits[word] & (1u << bit)) != 0u;",
    ] {
        assert!(
            common.contains(token),
            "occupancy helper should load only the required word token {token}"
        );
    }

    assert!(
        !traverse.contains("BrickOccupancy occ = occupancy_buf[node.brick_id];")
            && !traverse.contains("BrickOccupancy occ = occupancy_buf[brick_id];"),
        "primary brick traversal must not materialize the whole BrickOccupancy struct before the DDA"
    );
}

#[test]
fn vpt_restir_di_visibility_respects_shadow_disable_flag() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "bool restir_di_light_visible_from_hit(HitResult hit, RestirDiReservoir reservoir, SceneUniforms scene)",
        "(scene.lighting_flags & LIGHTING_FLAG_SHADOWS_ENABLED) == 0u",
        "restir_di_light_visible_from_hit(hit, reservoir, scene)",
    ] {
        assert!(
            vpt.contains(token),
            "VPT ReSTIR-DI visibility should follow REVOLUMETRIC_LIGHTING_SHADOWS; missing token {token}"
        );
    }
}

#[test]
fn vpt_analytic_sun_direct_respects_voxel_shadow_visibility() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "bool voxel_shadow_occluded_from_hit(HitResult hit, float3 shadow_dir, float max_t)",
        "bool analytic_sun_visible_from_hit(HitResult hit, SceneUniforms scene, float3 sun_dir)",
        "(scene.lighting_flags & LIGHTING_FLAG_SHADOWS_ENABLED) == 0u",
        "Ray shadow_ray = make_ray(hit.position + hit.normal * VPT_RAY_SURFACE_BIAS, shadow_dir);",
        "bool sun_occluded = voxel_shadow_occluded_from_hit(hit, sun_dir, 1.0e20);",
        "return !sun_occluded;",
        "if (sun_term <= 0.0 || !analytic_sun_visible_from_hit(hit, scene, sun_dir))",
    ] {
        assert!(
            vpt.contains(token),
            "VPT analytic sun must test voxel shadow visibility; missing token {token}"
        );
    }

    assert!(
        !vpt.contains(
            "return material_cell_albedo(hit.cell) * scene.sun_intensity * sun_term * 0.2;"
        ),
        "VPT analytic sun must not unconditionally add direct sunlight through voxel occluders"
    );
}

#[test]
fn vpt_analytic_sun_samples_solar_disk_for_soft_shadow_edges() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "float3 sample_sun_direction(SceneUniforms scene, inout uint rng_state)",
        "scene.sun_angular_radius",
        "float cos_min = cos(sun_radius);",
        "float cos_theta = lerp(cos_min, 1.0, rand01(rng_state));",
        "float phi = 6.28318530718 * rand01(rng_state);",
        "float3 sun_dir = sample_sun_direction(scene, rng_state);",
        "float sun_term = max(dot(hit.normal, sun_dir), 0.0);",
        "analytic_sun_direct(hit, scene, rng_state)",
    ] {
        assert!(
            vpt.contains(token),
            "VPT analytic sun should sample a finite solar disk for soft shadows; missing token {token}"
        );
    }

    assert!(
        !vpt.contains("analytic_sun_direct(hit, scene);"),
        "analytic sun direct lighting must consume rng_state so penumbrae can converge across VPT samples"
    );
}

#[test]
fn vpt_analytic_sun_normalizes_by_solar_disk_solid_angle() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");
    let lighting_common = std::fs::read_to_string("assets/shaders/shared/lighting_common.slang")
        .expect("lighting common shader should be readable");
    let scene_light = std::fs::read_to_string("src/scene/light.rs")
        .expect("scene light source should be readable");
    let scene_ubo =
        std::fs::read_to_string("src/render/scene_ubo.rs").expect("scene UBO should be readable");
    let readme = std::fs::read_to_string("README.md").expect("README should be readable");

    for token in [
        "float sun_pdf = sun_direction_pdf(scene);",
        "if (sun_pdf <= 0.0)",
        "return albedo * LIGHTING_INV_PI * scene.sun_intensity * sun_term / sun_pdf;",
    ] {
        assert!(
            vpt.contains(token),
            "VPT analytic sun must normalize the sampled solar disk with its pdf; missing token {token}"
        );
    }
    for token in [
        "float sun_disk_solid_angle(SceneUniforms scene)",
        "float sun_direction_pdf(SceneUniforms scene)",
        "if (sun_radius <= 0.0)",
        "return 0.0;",
        "return max(LIGHTING_TWO_PI * (1.0 - cos(sun_radius)), 1.0e-8);",
        "return solid_angle > 0.0 ? 1.0 / solid_angle : 0.0;",
    ] {
        assert!(
            lighting_common.contains(token),
            "shared lighting helpers must expose finite solar disk normalization; missing token {token}"
        );
    }
    assert!(
        lighting_common.contains(
            "float3 finite_sun_irradiance = scene.sun_intensity * ground_ndotl * sun_disk_solid_angle(scene);"
        ) && lighting_common.contains(
            "float3 sunlit_ground = scene.ground_color * (1.0 + LIGHTING_INV_PI * finite_sun_irradiance);"
        ),
        "sky miss ground lighting must use the same finite-disk radiance scale as direct sun instead of treating sun_intensity as directional irradiance"
    );
    assert!(
        !lighting_common
            .contains("scene.ground_color * (1.0 + scene.sun_intensity * ground_ndotl)"),
        "sky miss path must not retain the legacy directional-irradiance ground boost"
    );

    assert!(
        scene_light.contains("Solar-disk radiance used by the VPT finite sun estimator."),
        "DirectionalLight::intensity must document that VPT consumes solar-disk radiance, not legacy directional irradiance"
    );
    assert!(
        scene_ubo.contains("solar-disk radiance for VPT finite sun estimator"),
        "GpuSceneUniforms::sun_intensity must document the finite-disk radiance semantic"
    );
    assert!(
        readme.contains("The default sun intensity is interpreted as solar-disk radiance"),
        "README must tell users that finite sun disk direct lighting uses radiance/pdf semantics"
    );

    assert!(
        !vpt.contains("* 0.2"),
        "VPT analytic sun must not use a magic constant in place of the solar-disk sampling pdf"
    );
}

#[test]
fn voxel_ray_traversal_treats_parallel_axes_as_non_stepping_slabs() {
    let ray =
        std::fs::read_to_string("assets/shaders/shared/ray.slang").expect("ray shader exists");
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "static const float RAY_DIRECTION_EPSILON",
        "static const float RAY_PARALLEL_INV_DIR",
        "static const float RAY_T_MAX",
        "float ray_safe_rcp(float direction_component)",
        "float3 ray_safe_inv_dir(float3 direction)",
        "int ray_step_component(float direction_component)",
        "int3 ray_step_dir(float3 direction)",
        "float ray_axis_t_delta(",
        "float ray_axis_t_max(",
        "r.inv_dir = ray_safe_inv_dir(direction);",
    ] {
        assert!(ray.contains(token), "ray helper missing token {token}");
    }

    assert!(
        !ray.contains("sign(direction.x) *")
            && !ray.contains("sign(direction.y) *")
            && !ray.contains("sign(direction.z) *"),
        "parallel ray reciprocals must not use sign(0), which produces a zero reciprocal"
    );

    for token in [
        "float3 inv_dir = ray_safe_inv_dir(ray_dir);",
        "int3 step_dir = ray_step_dir(ray_dir);",
        "ray_axis_t_delta(ray_dir.x, inv_dir.x, 1.0)",
        "ray_axis_t_delta(ray.direction.x, inv_dir.x, 8.0)",
        "ray_axis_t_max(ray_origin.x, ray_dir.x, inv_dir.x",
        "ray_axis_t_max(ray.origin.x, ray.direction.x, inv_dir.x",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal missing robust parallel-axis token {token}"
        );
    }

    assert!(
        !traverse.contains("int3(sign(ray_dir))")
            && !traverse.contains("int3(sign(ray.direction))")
            && !traverse.contains("sign(ray_dir.x) *")
            && !traverse.contains("sign(ray.direction.x) *"),
        "DDA must not step a zero-direction axis or derive reciprocal from sign(0)"
    );
}

#[test]
fn ray_aabb_rejects_parallel_axes_that_start_outside_the_slab() {
    let math = std::fs::read_to_string("assets/shaders/shared/math.slang")
        .expect("math shader should be readable");

    for token in [
        "static const float AABB_PARALLEL_INV_DIR_THRESHOLD",
        "static const float AABB_T_MAX",
        "float2 aabb_axis_interval(",
        "origin_component < min_component || origin_component > max_component",
        "return float2(1.0, 0.0);",
        "return float2(-AABB_T_MAX, AABB_T_MAX);",
        "float2 x = aabb_axis_interval(origin.x, inv_dir.x, box.mn.x, box.mx.x);",
        "float2 y = aabb_axis_interval(origin.y, inv_dir.y, box.mn.y, box.mx.y);",
        "float2 z = aabb_axis_interval(origin.z, inv_dir.z, box.mn.z, box.mx.z);",
    ] {
        assert!(
            math.contains(token),
            "AABB slab intersection missing parallel-axis rejection token {token}"
        );
    }
}

#[test]
fn voxel_traversal_steps_all_tied_dda_axes_to_avoid_edge_seams() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "static const float DDA_TIE_EPSILON",
        "float dda_next_t(float3 t_max)",
        "bool dda_axis_is_tied(float axis_t, float next_t)",
        "float3 dda_step_normal(",
        "void dda_step_voxel(",
        "void dda_step_brick(",
        "dda_step_voxel(coord, t_max, t_delta, step_dir, current_t, hit_normal);",
        "dda_step_brick_with_t(brick_coord, t_max, t_delta, step_dir, current_t);",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing tie-aware DDA token {token}"
        );
    }

    assert!(
        !traverse.contains("if (t_max.x < t_max.y)"),
        "single-axis DDA tie breaking visits edge-touching cells and creates voxel/brick seams"
    );
    assert!(
        !traverse.contains("normal * rsqrt(len2)"),
        "tie-aware traversal must keep exported hit normals axis-aligned for lighting and reprojection"
    );
}

#[test]
fn voxel_traversal_uses_direction_aware_entry_cells_instead_of_fixed_nudge() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "float dda_adjust_boundary_position(",
        "float dda_boundary_ulp(",
        "float dda_next_up(",
        "float dda_next_down(",
        "nextafter(",
        "int dda_start_coord(",
        "int3 dda_start_coord3(",
        "dda_start_coord3(entry_pos",
        "ray.origin.x, ray.direction.x, inv_dir.x",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing direction-aware DDA entry token {token}"
        );
    }

    assert!(
        !traverse.contains("DDA_GRID_BOUNDARY_EPSILON = 1.0e-5"),
        "fixed boundary epsilon loses nudging power once coordinates exceed the literal's scale"
    );
    assert!(
        !traverse.contains("target_t + DDA_GRID_BOUNDARY_EPSILON"),
        "hierarchy skip reset must not rely on a fixed t-space offset"
    );
    assert!(
        !traverse.contains("t_enter + 0.001"),
        "fixed t-space entry nudges perturb cell selection at voxel and brick boundaries"
    );
}

#[test]
fn voxel_traversal_brick_grid_steps_do_not_truncate_large_grids() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "int max_steps = igrid.x + igrid.y + igrid.z;",
        "for (int i = 0; i < max_steps; i++) {",
        "for (int i = 0; i < max_steps && current_t < max_t; i++) {",
    ] {
        assert!(
            traverse.contains(token),
            "brick-grid traversal missing grid-sized loop token {token}"
        );
    }

    assert!(
        !traverse.contains("i < 256"),
        "brick-grid traversal must not truncate larger grids at 256 steps"
    );
}

#[test]
fn voxel_traversal_uses_uploaded_l1_l4_hierarchy_for_empty_space_skipping() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("surface shader should be readable");

    for token in [
        "StructuredBuffer<NodeLN> hierarchy_l1",
        "StructuredBuffer<NodeLN> hierarchy_l2",
        "StructuredBuffer<NodeLN> hierarchy_l3",
        "StructuredBuffer<NodeLN> hierarchy_l4",
        "bool hierarchy_level_empty_at_brick(",
        "bool try_skip_empty_hierarchy_block(",
        "void reset_brick_dda_at_t(",
        "node_ln_child_mask(parent)",
        "try_skip_empty_hierarchy_block(",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing hierarchy skip token {token}"
        );
    }

    assert!(
        traverse.matches("try_skip_empty_hierarchy_block(").count() >= 3,
        "primary and any-hit traversal loops must both use hierarchy empty-space skipping"
    );
    assert!(
        vpt.contains("hierarchy_l4") && surface.contains("hierarchy_l4"),
        "VPT trace and surface passes must pass all L1-L4 buffers into shared traversal"
    );
}

#[test]
fn voxel_hierarchy_skip_reinitializes_dda_state_in_constant_time() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");

    for token in [
        "void reset_brick_dda_at_t(",
        "float reset_t = dda_next_up(target_t);",
        "float3 reset_pos = ray.origin + ray.direction * reset_t;",
        "brick_coord = dda_start_coord3(reset_pos / 8.0, step_dir, igrid - int3(1));",
        "ray_axis_t_max(ray.origin.x, ray.direction.x, ray.inv_dir.x",
        "int3 original_brick_coord = skipped_brick_coord;",
        "bool skipped = any(skipped_brick_coord != original_brick_coord);",
        "return skipped;",
    ] {
        assert!(
            traverse.contains(token),
            "hierarchy skip DDA reset missing constant-time token {token}"
        );
    }

    assert!(
        !traverse.contains("void advance_brick_dda_to_t(")
            && !traverse.contains("while (current_t < target_t)"),
        "hierarchy empty-space skipping must not walk brick DDA one boundary at a time"
    );
}

#[test]
fn vpt_ray_biases_do_not_skip_voxel_faces_or_shadow_endpoints() {
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "static const float VPT_RAY_SURFACE_BIAS",
        "static const float VPT_LIGHT_ENDPOINT_BIAS",
        "hit.position + hit.normal * VPT_RAY_SURFACE_BIAS",
        "max_light_t = max(light_distance - VPT_LIGHT_ENDPOINT_BIAS, 0.0);",
    ] {
        assert!(vpt.contains(token), "VPT shader missing bias token {token}");
    }

    for forbidden in [
        "hit.normal * 0.75",
        "hit.normal * 0.05 + shadow_dir * 0.05",
        "light_distance - 1.0",
    ] {
        assert!(
            !vpt.contains(forbidden),
            "VPT shader uses leak-prone large ray bias token {forbidden}"
        );
    }
}

#[test]
fn vpt_shadow_visibility_skips_originating_voxel_to_avoid_surface_acne() {
    let traverse = std::fs::read_to_string("assets/shaders/shared/voxel_traverse.slang")
        .expect("voxel traversal shader should be readable");
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("vpt shader should be readable");

    for token in [
        "static const uint VOXEL_TRAVERSAL_NO_SKIP_BRICK",
        "bool voxel_traversal_should_skip_shadow_hit(",
        "bool trace_any_hit_ray_skip_voxel(",
        "uint skip_brick_id",
        "uint3 skip_local",
        "node.brick_id",
    ] {
        assert!(
            traverse.contains(token),
            "voxel traversal shader missing source-voxel shadow skip token {token}"
        );
    }

    for token in [
        "bool voxel_shadow_occluded_from_hit(HitResult hit, float3 shadow_dir, float max_t)",
        "trace_any_hit_ray_skip_voxel(",
        "hit.brick_id",
        "hit.local",
        "bool sun_occluded = voxel_shadow_occluded_from_hit(hit, sun_dir, 1.0e20);",
        "bool shadow_occluded = voxel_shadow_occluded_from_hit(hit, shadow_dir, max_light_t);",
    ] {
        assert!(
            vpt.contains(token),
            "VPT shader missing source-voxel shadow skip token {token}"
        );
    }
}

#[test]
fn vpt_pass_binds_restir_di_uniform_and_reservoir_resources() {
    let pass_source = source("src/render/passes/vpt.rs");
    let implementation = pass_source
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    crate::render::source_checks::assert_contains_all(
        implementation,
        &[
            "descriptor_binding_specs",
            "update_restir_di_descriptors",
            "GpuRestirDiUniforms",
            "GpuRestirDiReservoir",
        ],
        "VPT pass ReSTIR-DI descriptors",
    );
}

#[test]
fn vpt_pass_binds_area_restir_as_independent_sample_area_resources() {
    let pass_source = source("src/render/passes/vpt.rs");
    let implementation = pass_source
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    crate::render::source_checks::assert_contains_all(
        implementation,
        &[
            "descriptor_binding_specs",
            "disabled_area_restir_uniform_buffers",
            "disabled_area_restir_reservoir_buffer",
            "update_area_restir_descriptors",
            "GpuAreaRestirUniforms",
            "GpuAreaRestirReservoir",
            "create_disabled_area_restir_uniform_buffers",
            "create_disabled_area_restir_reservoir_buffer",
        ],
        "VPT pass Area ReSTIR descriptors",
    );

    assert!(
        implementation.find("update_restir_di_descriptors")
            < implementation.find("update_area_restir_descriptors"),
        "Area ReSTIR must be bound separately from ReSTIR-DI, not merged into DI descriptors"
    );
}

#[test]
fn vpt_shader_uses_area_restir_reservoir_to_override_primary_ray_when_valid() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let primary_sample_common = normalized_source(include_str!(
        "../../../../assets/shaders/shared/vpt_primary_sample_common.slang"
    ));
    let combined_source = format!("{source}\n{primary_sample_common}");

    for token in [
        "#include \"area_restir_common.slang\"",
        "#include \"vpt_primary_sample_common.slang\"",
        "ConstantBuffer<AreaRestirUniforms> area_restir;",
        "StructuredBuffer<AreaRestirReservoir> area_restir_reservoirs;",
        "resolve_area_restir_primary_ray",
        "area_restir_is_valid_reservoir",
        "scene_primary_ray_from_area_sample",
        "area_restir_pixel_sample(pixel, reservoir.sample_state)",
        "reservoir.sample_state.lens_uv",
        "if (area.enabled != 0u",
        "fallback jitter",
        "float4 hit_position_depth = float4(hit.position, max(hit.t, 0.0));",
        "hit_reservoir.target_pdf = hit_target_pdf;",
    ] {
        assert!(
            combined_source.contains(token),
            "VPT shader missing Area ReSTIR primary-ray token {token}"
        );
    }
    assert!(
        !source.contains("reservoir.sample_state.pixel_sample"),
        "VPT must not replay a history/neighbor reservoir's source pixel; only its subpixel/lens state is reusable"
    );
    assert!(
        !source.contains("float2(pixel) + reservoir.sample_state.subpixel_uv"),
        "VPT must not shift Area ReSTIR samples by half a pixel; use the shared pixel-sample conversion"
    );
    assert!(
        !source.contains("current_surface_position_depth")
            && !source.contains("restir_di_surface_compatible_with_hit"),
        "VPT must not reject Area ReSTIR subpixel/lens hits by comparing them against the center-pixel surface buffer"
    );
    let area_common = include_str!("../../../../assets/shaders/shared/area_restir_common.slang");
    assert!(
        !area_common.contains("float2 pixel_sample")
            && !area_common.contains("float4 selected_radiance"),
        "Area ReSTIR reservoir ABI must not carry unused pixel-sample or radiance payload"
    );
}

#[test]
fn vpt_surface_and_trace_share_area_restir_primary_sample_contract() {
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let vpt = std::fs::read_to_string("assets/shaders/passes/vpt.slang")
        .expect("VPT trace shader should exist");
    let common = std::fs::read_to_string("assets/shaders/shared/vpt_primary_sample_common.slang")
        .expect("shared VPT primary sample contract should exist");

    for (name, source) in [("surface", surface.as_str()), ("vpt", vpt.as_str())] {
        assert!(
            source.contains("#include \"vpt_primary_sample_common.slang\""),
            "{name} shader must include the shared VPT primary-sample contract"
        );
        assert!(
            !source.contains("float3 primary_ray_direction(SceneUniforms scene, uint2 pixel)"),
            "{name} shader must not keep a private pixel-center primary-ray path"
        );
    }
    assert!(
        surface.contains("vpt_resolve_surface_primary_ray"),
        "surface shader must replay selected Area ReSTIR samples through the stable surface resolver"
    );
    assert!(
        vpt.contains("vpt_resolve_area_restir_primary_ray"),
        "trace shader must replay selected Area ReSTIR samples while preserving stochastic VPT fallback"
    );

    for token in [
        "uint vpt_primary_rng_seed",
        "ScenePrimaryRay vpt_center_primary_ray",
        "ScenePrimaryRay vpt_fallback_primary_ray",
        "ScenePrimaryRay vpt_primary_ray_from_area_reservoir",
        "ScenePrimaryRay vpt_resolve_area_restir_primary_ray",
        "ScenePrimaryRay vpt_resolve_surface_primary_ray",
        "area_restir_pixel_sample(pixel, reservoir.sample_state)",
        "scene_primary_ray_from_area_sample",
    ] {
        assert!(
            common.contains(token),
            "shared VPT primary-sample contract missing {token}"
        );
    }
}

#[test]
fn vpt_surface_fallback_keeps_stable_center_ray_for_history_guides() {
    let surface = std::fs::read_to_string("assets/shaders/passes/vpt_surface.slang")
        .expect("VPT surface shader should exist");
    let common = std::fs::read_to_string("assets/shaders/shared/vpt_primary_sample_common.slang")
        .expect("shared VPT primary sample contract should exist");

    assert!(
        common.contains("ScenePrimaryRay vpt_center_primary_ray"),
        "shared contract must expose a deterministic center-ray guide for surface history"
    );
    assert!(
        common.contains("ScenePrimaryRay vpt_resolve_surface_primary_ray"),
        "surface pass must have a resolver that replays valid Area ReSTIR samples but does not jitter invalid fallback"
    );
    assert!(
        common.contains("return vpt_center_primary_ray(pixel, scene);"),
        "surface fallback must return the stable center ray, not a stochastic VPT sample"
    );
    assert!(
        surface.contains("vpt_resolve_surface_primary_ray"),
        "VPT surface shader must use the stable surface resolver"
    );
    assert!(
        !surface.contains("vpt_resolve_area_restir_primary_ray"),
        "VPT surface shader must not use the stochastic VPT trace fallback resolver"
    );
}

#[test]
fn vpt_surface_pass_binds_area_restir_selected_primary_sample() {
    let surface_shader = source("assets/shaders/passes/vpt_surface.slang");
    let pass_source = source("src/render/passes/vpt_surface.rs");
    let implementation = pass_source
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    for token in [
        "#include \"area_restir_common.slang\"",
        "#include \"vpt_primary_sample_common.slang\"",
        "ConstantBuffer<AreaRestirUniforms> area_restir;",
        "StructuredBuffer<AreaRestirReservoir> area_restir_reservoirs;",
        "vpt_resolve_surface_primary_ray",
        "make_ray(primary_ray.origin, primary_ray.direction)",
    ] {
        assert!(
            surface_shader.contains(token),
            "VPT surface shader missing Area ReSTIR selected-primary token {token}"
        );
    }

    crate::render::source_checks::assert_contains_all(
        implementation,
        &[
            "descriptor_binding_specs",
            "bootstrap_descriptor_sets",
            "selected_descriptor_sets",
            "disabled_area_restir_uniform_buffers",
            "disabled_area_restir_reservoir_buffer",
            "update_area_restir_descriptors",
            "record_bootstrap",
            "record_selected",
            "write_area_restir_descriptor_sets",
            "GpuAreaRestirUniforms",
            "GpuAreaRestirReservoir",
        ],
        "VPT surface pass selected-primary descriptors",
    );
}

#[test]
fn app_uses_selected_vpt_surface_after_area_restir_for_di_trace_and_temporal() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let surface_pass = source("src/render/passes/vpt_surface.rs");
    let area_pass = std::fs::read_to_string("src/render/passes/area_restir.rs")
        .expect("Area ReSTIR pass source should be readable");
    let restir_pass = source("src/render/passes/restir_di.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let temporal_pass = source("src/render/passes/vpt_temporal.rs");
    let compact_area_pass = area_pass.split_whitespace().collect::<String>();
    let compact_vpt = vpt_pass.split_whitespace().collect::<String>();
    let compact_temporal = temporal_pass.split_whitespace().collect::<String>();
    let compact_restir = restir_pass.split_whitespace().collect::<String>();

    for token in [
        "vpt_surface.register_bootstrap_graph(",
        "area_restir.register_graph(",
        "vpt_area_restir_reads",
        "vpt.register_graph(",
        "vpt_temporal.register_graph(",
        "vpt_temporal.register_history_update_graph(",
    ] {
        assert!(
            pipeline.contains(token),
            "VPT pipeline graph missing selected-surface token {token}"
        );
    }
    for token in [
        "\"vpt_surface_selected\"",
        "vpt_surface.record_selected",
        "vpt_surface.update_area_restir_descriptors",
    ] {
        assert!(
            area_pass.contains(token),
            "Area ReSTIR pass graph missing selected-surface token {token}"
        );
    }

    let bootstrap_idx = pipeline
        .find("vpt_surface.register_bootstrap_graph(")
        .expect("bootstrap surface pass should exist");
    let area_register_idx = pipeline
        .find("area_restir.register_graph(")
        .expect("Area ReSTIR graph registration should exist");
    let area_initial_idx = area_pass
        .find("\"area_restir_initial\"")
        .expect("Area ReSTIR initial pass should exist");
    let selected_idx = area_pass
        .find("\"vpt_surface_selected\"")
        .expect("selected surface pass should exist");
    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT pass should exist");
    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("VPT temporal pass should exist");
    let history_update_idx = pipeline
        .find("vpt_temporal.register_history_update_graph(")
        .expect("VPT surface history update pass should exist");
    let restir_idx = pipeline
        .find("restir_di.register_graph(")
        .unwrap_or(usize::MAX);

    assert!(bootstrap_idx < area_register_idx);
    assert!(area_initial_idx < selected_idx);
    if restir_idx != usize::MAX {
        assert!(area_register_idx < restir_idx);
    }
    assert!(area_register_idx < vpt_idx);
    assert!(area_register_idx < temporal_idx);
    assert!(area_register_idx < history_update_idx);
    assert!(surface_pass.contains("\"vpt_surface_bootstrap\""));
    assert!(
        compact_vpt
            .contains("builder.read_as(area_uniform_resource,AccessKind::ComputeShaderRead)")
    );
    assert!(compact_vpt.contains(
        "builder.read_as(area_selected_reservoir_resource,AccessKind::ComputeShaderRead"
    ));
    assert!(
        compact_temporal.contains("builder.read_as(surface_input,AccessKind::ComputeShaderRead)")
    );
    assert!(
        compact_temporal
            .contains("builder.read_as(previous_surface_input,AccessKind::ComputeShaderRead)")
    );
    assert!(
        compact_restir
            .contains("final_surface_writes.position_depth,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_restir
            .contains("final_surface_writes.normal_roughness,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_restir
            .contains("final_surface_writes.albedo_material,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_area_pass
            .contains("builder.read_as(uniform_resource,AccessKind::ComputeShaderRead)")
            && compact_area_pass
                .contains("builder.read_as(selected_resource,AccessKind::ComputeShaderRead)"),
        "Area ReSTIR selected-surface pass must read the selected reservoir and uniform resources"
    );
}

#[test]
fn app_declares_spatial_reservoir_as_vpt_read_dependency() {
    let pipeline = source("src/render/vpt_pipeline.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let compact_vpt = vpt_pass.split_whitespace().collect::<String>();
    let restir_pass = std::fs::read_to_string("src/render/passes/restir_di.rs")
        .expect("ReSTIR-DI source should be readable");

    assert!(pipeline.contains("vpt_restir_reads"));
    assert!(pipeline.contains("vpt.register_graph("));
    assert!(
        compact_pipeline.contains(
            "vpt_restir_reads=Some((restir_graph.uniform_resource,restir_graph.selected_current_resource,));"
        )
    );
    assert!(restir_pass.contains("vpt.update_restir_di_descriptors"));
    assert!(
        compact_vpt
            .contains("builder.read_as(restir_uniform_resource,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_vpt
            .contains("builder.read_as(restir_reservoir_resource,AccessKind::ComputeShaderRead")
    );
}

#[test]
fn app_wires_area_restir_between_surface_and_vpt_with_history_and_vpt_reads() {
    let app = source("src/app.rs");
    let runtime = source("src/render/runtime.rs");
    let pipeline = source("src/render/vpt_pipeline.rs");
    let surface_pass = source("src/render/passes/vpt_surface.rs");
    let area_pass = std::fs::read_to_string("src/render/passes/area_restir.rs")
        .expect("Area ReSTIR pass source should be readable");
    let restir_pass = source("src/render/passes/restir_di.rs");
    let vpt_pass = source("src/render/passes/vpt.rs");
    let temporal_pass = source("src/render/passes/vpt_temporal.rs");
    let area_pass_impl = area_pass
        .split("#[cfg(test)]")
        .next()
        .expect("Area ReSTIR implementation section should exist");
    let compact_pipeline = pipeline.split_whitespace().collect::<String>();
    let compact_area_pass = area_pass.split_whitespace().collect::<String>();
    let compact_vpt = vpt_pass.split_whitespace().collect::<String>();

    for app_token in [
        "AreaRestirSettings::from_env",
        "area_restir_settings: AreaRestirSettings",
    ] {
        assert!(
            app.contains(app_token),
            "app missing Area ReSTIR setup token {app_token}"
        );
    }
    for runtime_token in ["ucvh_gpu", "self.vpt_pipeline.ensure_passes("] {
        assert!(
            runtime.contains(runtime_token),
            "runtime missing Area ReSTIR setup token {runtime_token}"
        );
    }

    for pipeline_token in [
        "vpt_surface.register_bootstrap_graph(",
        "area_restir.register_graph(",
        "vpt_area_restir_reads",
        "vpt.register_graph(",
        "vpt_temporal.register_graph(",
        "vpt_temporal.register_history_update_graph(",
        "self.frame_state.area_restir_history_initialized",
    ] {
        assert!(
            if pipeline_token == "self.frame_state.area_restir_history_initialized" {
                compact_pipeline.contains(pipeline_token)
            } else {
                pipeline.contains(pipeline_token)
            },
            "VPT pipeline missing Area ReSTIR graph token {pipeline_token}"
        );
    }

    assert!(pipeline.contains("AreaRestirPass::new"));
    assert!(pipeline.contains("AreaRestirPassCreateInfo"));

    for pass_token in [
        "self.update_uniforms",
        "\"area_restir_initial\"",
        "\"area_restir_temporal\"",
        "\"area_restir_spatial\"",
        "self.selected_current_buffer",
        "self.selected_history_buffer",
        "update_surface_descriptors",
        "update_ucvh_descriptors",
        "vpt.update_area_restir_descriptors",
    ] {
        assert!(
            area_pass.contains(pass_token) || pipeline.contains(pass_token),
            "Area ReSTIR pass missing graph token {pass_token}"
        );
    }

    let bootstrap_surface_idx = pipeline
        .find("vpt_surface.register_bootstrap_graph(")
        .expect("bootstrap surface pass should exist");
    let area_register_idx = pipeline
        .find("area_restir.register_graph(")
        .expect("Area ReSTIR graph registration should exist");
    let area_initial_idx = area_pass
        .find("\"area_restir_initial\"")
        .expect("Area ReSTIR initial pass should exist");
    let selected_surface_idx = area_pass
        .find("\"vpt_surface_selected\"")
        .expect("selected surface pass should exist");
    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT pass should exist");
    let temporal_idx = pipeline
        .find("vpt_temporal.register_graph(")
        .expect("VPT temporal pass should exist");
    let history_update_idx = pipeline
        .find("vpt_temporal.register_history_update_graph(")
        .expect("VPT temporal history update pass should exist");
    assert!(bootstrap_surface_idx < area_register_idx);
    assert!(area_initial_idx < selected_surface_idx);
    assert!(area_register_idx < vpt_idx);
    assert!(vpt_idx < temporal_idx);
    assert!(temporal_idx < history_update_idx);

    assert!(
        compact_area_pass
            .contains("builder.read_as(uniform_resource,AccessKind::ComputeShaderRead")
    );
    assert!(
        compact_area_pass
            .contains("builder.read_as(selected_resource,AccessKind::ComputeShaderRead")
    );
    assert!(surface_pass.contains("\"vpt_surface_bootstrap\""));
    assert!(
        compact_vpt
            .contains("builder.read_as(area_uniform_resource,AccessKind::ComputeShaderRead)")
    );
    assert!(compact_vpt.contains(
        "builder.read_as(area_selected_reservoir_resource,AccessKind::ComputeShaderRead"
    ));
    assert!(
        restir_pass.contains("final_surface_writes.position_depth")
            && restir_pass.contains("AccessKind::ComputeShaderRead")
    );
    assert!(
        temporal_pass.contains("builder.read_as(surface_input, AccessKind::ComputeShaderRead)")
    );
    assert!(
        temporal_pass
            .contains("builder.read_as(previous_surface_input, AccessKind::ComputeShaderRead)")
    );
    assert!(
        !pipeline.contains("\"area_restir_history_update\"")
            && !area_pass_impl.contains("\"area_restir_history_update\""),
        "Area ReSTIR selected reservoirs must not be copied through a transfer history pass"
    );
}

#[test]
fn vpt_does_not_assume_primary_gbuffer_for_restir_di() {
    let design =
        std::fs::read_to_string("docs/superpowers/specs/2026-05-02-restir-di-vpt-design.md")
            .expect("ReSTIR-DI VPT design doc should be readable");

    assert!(
        design.contains("Current VPT mode does not register the primary-ray graph pass before VPT")
    );
    assert!(
        design.contains("must explicitly add a VPT-mode surface-state pass")
            || design.contains("Do not write a ReSTIR-DI pass that silently assumes `gbuffer_pos`")
    );
}

#[test]
fn vpt_temporal_barriers_are_owned_by_render_graph() {
    let vpt_source =
        std::fs::read_to_string("src/render/passes/vpt.rs").expect("vpt source is readable");
    let temporal_source = std::fs::read_to_string("src/render/passes/vpt_temporal.rs")
        .expect("vpt temporal source is readable");
    let postprocess_source = std::fs::read_to_string("src/render/passes/postprocess.rs")
        .expect("postprocess source is readable");
    let pipeline_source = source("src/render/vpt_pipeline.rs");
    let implementation = vpt_source
        .split("#[cfg(test)]")
        .next()
        .expect("implementation section should exist");

    assert!(!implementation.contains("cmd_pipeline_barrier"));
    assert!(!implementation.contains("ImageMemoryBarrier"));
    assert!(temporal_source.contains("let temporal_initial_access = if history_initialized"));
    assert!(temporal_source.contains("let previous_temporal_access = if history_initialized"));
    assert!(postprocess_source.contains("let output_initial_access = if output_initialized"));
    assert!(pipeline_source.contains("vpt_temporal.register_graph("));
    assert!(pipeline_source.contains("postprocess.register_graph("));
}

#[test]
fn vpt_trace_shader_routes_area_restir_debug_views_to_noisy_output() {
    let source = normalized_source(include_str!("../../../../assets/shaders/passes/vpt.slang"));
    let temporal = normalized_source(include_str!(
        "../../../../assets/shaders/passes/vpt_temporal.slang"
    ));

    for token in [
        "VPT_DEBUG_VIEW_AREA_SUBPIXEL",
        "VPT_DEBUG_VIEW_AREA_LENS",
        "VPT_DEBUG_VIEW_AREA_WEIGHT",
        "VPT_DEBUG_VIEW_AREA_HISTORY_VALID",
        "VPT_DEBUG_VIEW_AREA_REJECTION",
        "VPT_DEBUG_VIEW_AREA_JACOBIAN",
        "visualize_area_restir_debug",
        "area_restir_reservoirs[index]",
        "reservoir.sample_state.subpixel_uv",
        "reservoir.sample_state.lens_uv",
        "reservoir.rejection_reason",
        "reservoir.jacobian",
    ] {
        assert!(
            source.contains(token),
            "VPT shader missing Area ReSTIR debug token {token}"
        );
    }

    for token in [
        "VPT_DEBUG_VIEW_AREA_SUBPIXEL",
        "VPT_DEBUG_VIEW_AREA_LENS",
        "VPT_DEBUG_VIEW_AREA_WEIGHT",
        "VPT_DEBUG_VIEW_AREA_HISTORY_VALID",
        "VPT_DEBUG_VIEW_AREA_REJECTION",
        "VPT_DEBUG_VIEW_AREA_JACOBIAN",
    ] {
        assert!(
            temporal.contains(token),
            "VPT temporal shader must consume and bypass Area ReSTIR debug token {token}"
        );
    }
}

#[test]
fn app_maps_area_restir_debug_setting_to_visible_vpt_debug_view() {
    let source = normalized_source(include_str!("../../../app.rs"));

    for token in [
        "fn area_restir_debug_to_vpt_debug_view",
        "AreaRestirDebugView::Subpixel => Some(VptDebugView::AreaSubpixel)",
        "AreaRestirDebugView::Lens => Some(VptDebugView::AreaLens)",
        "AreaRestirDebugView::Weight => Some(VptDebugView::AreaWeight)",
        "AreaRestirDebugView::HistoryValid => Some(VptDebugView::AreaHistoryValid)",
        "AreaRestirDebugView::Rejection => Some(VptDebugView::AreaRejection)",
        "AreaRestirDebugView::Jacobian => Some(VptDebugView::AreaJacobian)",
        "self.lighting_settings.vpt_debug_view = vpt_debug_view",
    ] {
        assert!(
            source.contains(token),
            "app missing Area ReSTIR debug bridge token {token}"
        );
    }
}
