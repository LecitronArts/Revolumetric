use crate::render::buffer::GpuBuffer;
use crate::voxel::gpu_upload::UcvhGpuResources;

#[derive(Clone, Copy)]
pub enum SceneTraceBackendResources<'a> {
    Voxel(&'a UcvhGpuResources),
}

impl<'a> SceneTraceBackendResources<'a> {
    pub fn voxel(resources: &'a UcvhGpuResources) -> Self {
        Self::Voxel(resources)
    }

    pub fn storage_buffers(self) -> [&'a GpuBuffer; 8] {
        match self {
            Self::Voxel(resources) => [
                &resources.config_buffer,
                &resources.hierarchy_l0_buffer,
                &resources.hierarchy_ln_buffers[0],
                &resources.hierarchy_ln_buffers[1],
                &resources.hierarchy_ln_buffers[2],
                &resources.hierarchy_ln_buffers[3],
                &resources.occupancy_buffer,
                &resources.material_buffer,
            ],
        }
    }
}
