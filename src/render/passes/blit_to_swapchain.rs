use ash::vk;

/// Records only the blit command. The RenderGraph owns all image layout transitions.
pub fn record_blit_core(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src_image: vk::Image,
    src_extent: vk::Extent3D,
    dst_image: vk::Image,
    dst_extent: vk::Extent2D,
) {
    let region = vk::ImageBlit {
        src_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        src_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: src_extent.width as i32,
                y: src_extent.height as i32,
                z: 1,
            },
        ],
        dst_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        dst_offsets: [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: dst_extent.width as i32,
                y: dst_extent.height as i32,
                z: 1,
            },
        ],
    };

    unsafe {
        device.cmd_blit_image(
            cmd,
            src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
            vk::Filter::LINEAR,
        );
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn blit_core_does_not_issue_layout_transitions() {
        let source = std::fs::read_to_string("src/render/passes/blit_to_swapchain.rs")
            .expect("blit source should be readable");
        let implementation = source
            .split("#[cfg(test)]")
            .next()
            .expect("implementation section should exist");

        assert!(!implementation.contains("cmd_pipeline_barrier"));
        assert!(!implementation.contains("ImageMemoryBarrier"));
    }
}
