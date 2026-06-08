fn normalized_source(path_source: &str) -> String {
    path_source.replace("\r\n", "\n")
}

#[test]
fn postprocess_shader_declares_hdr_input_and_ldr_output_abi() {
    let source = normalized_source(include_str!(
        "../../../../assets/shaders/passes/postprocess.slang"
    ));

    assert!(
        source.contains("[[vk::image_format(\"rgba16f\")]]\nRWTexture2D<float4> hdr_image;"),
        "postprocess input must be rgba16f HDR storage image"
    );
    assert!(
        source.contains("[[vk::image_format(\"rgba8\")]]\nRWTexture2D<float4> output_image;"),
        "postprocess output must be rgba8 LDR storage image"
    );
}

#[test]
fn postprocess_shader_applies_exposure_aces_and_gamma() {
    let source = normalized_source(include_str!(
        "../../../../assets/shaders/passes/postprocess.slang"
    ));

    assert!(
        source.contains("scene.exposure"),
        "postprocess shader must read exposure from SceneUniforms"
    );
    assert!(
        source.contains("aces_tonemap("),
        "postprocess shader must apply ACES tonemapping"
    );
    assert!(
        source.contains("pow(mapped, float3(1.0 / 2.2))"),
        "postprocess shader must apply gamma correction after tonemapping"
    );
}

#[test]
fn postprocess_input_update_is_frame_slot_scoped() {
    let source = std::fs::read_to_string("src/render/passes/postprocess.rs")
        .expect("postprocess source should be readable");
    let compact = source.split_whitespace().collect::<String>();

    assert!(compact.contains(
        "pubfnupdate_input_image(&self,device:&ash::Device,hdr_image:&GpuImage,frame_slot:usize"
    ));
    assert!(compact.contains("self.descriptor_sets.get(frame_slot)"));
    let start = source
        .find("pub fn update_input_image")
        .expect("update_input_image should exist");
    let end = source[start..]
        .find("pub fn destroy")
        .map(|offset| start + offset)
        .expect("destroy should follow update_input_image");
    let body = &source[start..end];
    assert!(
        !body.contains("for &ds in &self.descriptor_sets"),
        "postprocess input rebinding must not rewrite descriptor sets still in flight"
    );
}

#[test]
fn app_wires_vpt_through_postprocess_before_blit() {
    let runtime = normalized_source(
        &std::fs::read_to_string("src/render/runtime.rs")
            .expect("runtime source should be readable for render-pipeline source test"),
    );
    let pipeline = normalized_source(
        &std::fs::read_to_string("src/render/vpt_pipeline.rs")
            .expect("VPT pipeline source should be readable for render-pipeline source test"),
    );
    let postprocess = normalized_source(
        &std::fs::read_to_string("src/render/passes/postprocess.rs")
            .expect("postprocess source should be readable for render-pipeline source test"),
    );

    assert!(runtime.contains("capture: Option<RenderCapture>"));
    assert!(runtime.contains("self.vpt_pipeline.ensure_passes("));
    assert!(pipeline.contains("pub postprocess_pass: Option<PostprocessPass>"));
    assert!(pipeline.contains("PostprocessPass::new"));
    assert!(pipeline.contains("postprocess.register_graph("));
    assert!(postprocess.contains("graph.add_pass(\"postprocess\""));
    assert!(postprocess.contains("GpuProfileScope::Postprocess"));

    let vpt_idx = pipeline
        .find("vpt.register_graph(")
        .expect("VPT graph registration should exist");
    let postprocess_idx = vpt_idx
        + pipeline[vpt_idx..]
            .find("postprocess.register_graph(")
            .expect("postprocess graph registration should exist after VPT");
    let capture_idx = postprocess_idx
        + pipeline[postprocess_idx..]
            .find("\"capture_postprocess\"")
            .expect("capture graph pass should exist after postprocess");
    let blit_idx = postprocess_idx
        + pipeline[postprocess_idx..]
            .find("graph.add_pass(\"blit_to_swapchain\"")
            .or_else(|| {
                pipeline[postprocess_idx..]
                    .find("graph.add_pass(\n                                        \"blit_to_swapchain\"")
            })
            .expect("blit graph pass should exist");

    assert!(vpt_idx < postprocess_idx);
    assert!(postprocess_idx < blit_idx);
    assert!(postprocess_idx < capture_idx);
    assert!(capture_idx < blit_idx);
    assert!(pipeline.contains("cmd_copy_image_to_buffer"));
    assert!(runtime.contains("self.renderer.wait_for_fence"));
}
