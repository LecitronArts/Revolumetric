use anyhow::{Result, anyhow};
use ash::vk;
use std::collections::{BTreeMap, VecDeque};

use crate::render::pass_context::{PassBuilder, PassContext};
use crate::render::resource::{
    AccessKind, PassDecl, QueueType, ResourceAccess, ResourceDesc, ResourceHandle,
};

type ExecuteFn<'a> = Box<dyn FnOnce(&mut PassContext) + 'a>;

struct PassNode<'a> {
    decl: PassDecl,
    execute: ExecuteFn<'a>,
}

pub struct RenderGraph<'a> {
    passes: Vec<PassNode<'a>>,
    sorted_order: Vec<usize>,
    resources: BTreeMap<u32, ResourceDesc>,
    image_handles: BTreeMap<u32, vk::Image>,
    buffer_handles: BTreeMap<u32, vk::Buffer>,
    imported_accesses: BTreeMap<u32, AccessKind>,
    barrier_plan: Vec<PlannedBarrier>,
    next_resource_id: u32,
    compiled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannedBarrier {
    pub pass_index: usize,
    pub timing: BarrierTiming,
    pub resource: ResourceHandle,
    pub from: AccessKind,
    pub to: AccessKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierTiming {
    BeforePass,
    AfterPass,
}

impl<'a> RenderGraph<'a> {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            sorted_order: Vec::new(),
            resources: BTreeMap::new(),
            image_handles: BTreeMap::new(),
            buffer_handles: BTreeMap::new(),
            imported_accesses: BTreeMap::new(),
            barrier_plan: Vec::new(),
            next_resource_id: 0,
            compiled: false,
        }
    }

    pub fn import_image(
        &mut self,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> ResourceHandle {
        let handle = ResourceHandle {
            id: self.next_resource_id,
            version: 0,
        };
        self.next_resource_id += 1;
        self.resources.insert(
            handle.id,
            ResourceDesc::Image {
                width,
                height,
                format,
                usage,
            },
        );
        self.compiled = false;
        handle
    }

    pub fn import_image_with_access(
        &mut self,
        image: vk::Image,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        access: AccessKind,
    ) -> ResourceHandle {
        let handle = self.import_image(width, height, format, usage);
        self.image_handles.insert(handle.id, image);
        self.imported_accesses.insert(handle.id, access);
        handle
    }

    pub fn import_buffer(
        &mut self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> ResourceHandle {
        let handle = ResourceHandle {
            id: self.next_resource_id,
            version: 0,
        };
        self.next_resource_id += 1;
        self.resources
            .insert(handle.id, ResourceDesc::Buffer { size, usage });
        self.compiled = false;
        handle
    }

    pub fn import_buffer_with_access(
        &mut self,
        buffer: vk::Buffer,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        access: AccessKind,
    ) -> ResourceHandle {
        let handle = self.import_buffer(size, usage);
        self.buffer_handles.insert(handle.id, buffer);
        self.imported_accesses.insert(handle.id, access);
        handle
    }

    /// Add a pass. Returns the list of resource handles the pass writes,
    /// so downstream passes can declare reads on them.
    pub fn add_pass(
        &mut self,
        name: &'static str,
        queue_type: QueueType,
        setup: impl FnOnce(&mut PassBuilder) -> ExecuteFn<'a>,
    ) -> Vec<ResourceHandle> {
        let mut builder = PassBuilder::new(name, queue_type, self.next_resource_id);
        let execute = setup(&mut builder);
        self.next_resource_id = builder.next_resource_id;
        let writes = builder.writes.clone();
        for (handle, desc) in &builder.resource_descs {
            self.resources.insert(handle.id, desc.clone());
        }
        let decl = PassDecl {
            name: builder.name,
            queue_type: builder.queue_type,
            reads: builder.reads,
            writes: builder.writes,
            accesses: builder.accesses,
            final_accesses: builder.final_accesses,
        };
        self.passes.push(PassNode { decl, execute });
        self.compiled = false;
        writes
    }

    pub fn resource_desc(&self, handle: ResourceHandle) -> Option<&ResourceDesc> {
        self.resources.get(&handle.id)
    }

    pub fn bind_image(&mut self, handle: ResourceHandle, image: vk::Image) {
        self.image_handles.insert(handle.id, image);
        self.compiled = false;
    }

    pub fn bind_buffer(&mut self, handle: ResourceHandle, buffer: vk::Buffer) {
        self.buffer_handles.insert(handle.id, buffer);
        self.compiled = false;
    }

    pub fn compile(&mut self) -> Result<()> {
        self.validate_resource_references()?;
        self.validate_resource_timeline()?;
        self.barrier_plan.clear();

        let n = self.passes.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];

        let producers = self.producer_map()?;
        let mut last_access: BTreeMap<u32, usize> = BTreeMap::new();
        for i in 0..n {
            let pass = &self.passes[i].decl;
            for read in &pass.reads {
                if let Some(&producer) = producers.get(&(read.id, read.version)) {
                    Self::add_dependency(&mut adj, &mut in_degree, producer, i);
                }
            }

            let mut touched_ids: Vec<u32> = pass
                .reads
                .iter()
                .chain(pass.writes.iter())
                .chain(pass.final_accesses.iter().map(|access| &access.handle))
                .map(|handle| handle.id)
                .collect();
            touched_ids.sort_unstable();
            touched_ids.dedup();

            for id in &touched_ids {
                if let Some(&previous) = last_access.get(id) {
                    Self::add_dependency(&mut adj, &mut in_degree, previous, i);
                }
            }
            for id in touched_ids {
                last_access.insert(id, i);
            }
        }

        for edges in &mut adj {
            edges.sort_unstable();
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() != n {
            self.sorted_order.clear();
            self.compiled = false;
            return Err(anyhow!("render graph contains a dependency cycle"));
        }

        self.sorted_order = order;
        let barrier_plan = self.plan_barriers();
        self.validate_barrier_resource_bindings(&barrier_plan)?;
        self.barrier_plan = barrier_plan;
        self.compiled = true;
        Ok(())
    }

    fn add_dependency(adj: &mut [Vec<usize>], in_degree: &mut [usize], from: usize, to: usize) {
        if from == to || adj[from].contains(&to) {
            return;
        }
        adj[from].push(to);
        in_degree[to] += 1;
    }

    fn producer_map(&self) -> Result<BTreeMap<(u32, u32), usize>> {
        let mut producers = BTreeMap::new();
        for (pass_idx, pass) in self.passes.iter().enumerate() {
            for write in &pass.decl.writes {
                if let Some(previous) = producers.insert((write.id, write.version), pass_idx) {
                    return Err(anyhow!(
                        "resource id {} version {} is written by both pass '{}' and pass '{}'",
                        write.id,
                        write.version,
                        self.passes[previous].decl.name,
                        pass.decl.name
                    ));
                }
            }
        }
        Ok(producers)
    }

    fn plan_barriers(&self) -> Vec<PlannedBarrier> {
        let mut latest_access: BTreeMap<u32, ResourceAccess> = BTreeMap::new();
        let mut barriers = Vec::new();

        for &pass_idx in &self.sorted_order {
            let pass = &self.passes[pass_idx];
            for access in &pass.decl.accesses {
                let is_write = pass.decl.writes.iter().any(|write| {
                    write.id == access.handle.id && write.version == access.handle.version
                });

                let previous = latest_access.get(&access.handle.id).copied().or_else(|| {
                    self.imported_accesses
                        .get(&access.handle.id)
                        .copied()
                        .map(|kind| ResourceAccess {
                            handle: ResourceHandle {
                                id: access.handle.id,
                                version: 0,
                            },
                            kind,
                        })
                });

                match previous {
                    Some(previous) if previous.kind != access.kind => {
                        barriers.push(PlannedBarrier {
                            pass_index: pass_idx,
                            timing: BarrierTiming::BeforePass,
                            resource: previous.handle,
                            from: previous.kind,
                            to: access.kind,
                        });
                    }
                    None if is_write
                        && self
                            .resource_desc(access.handle)
                            .is_some_and(|desc| matches!(desc, ResourceDesc::Image { .. })) =>
                    {
                        barriers.push(PlannedBarrier {
                            pass_index: pass_idx,
                            timing: BarrierTiming::BeforePass,
                            resource: access.handle,
                            from: AccessKind::Undefined,
                            to: access.kind,
                        });
                    }
                    _ => {}
                }

                latest_access.insert(access.handle.id, *access);
            }

            for final_access in &pass.decl.final_accesses {
                let previous = latest_access
                    .get(&final_access.handle.id)
                    .copied()
                    .or_else(|| {
                        self.imported_accesses
                            .get(&final_access.handle.id)
                            .copied()
                            .map(|kind| ResourceAccess {
                                handle: ResourceHandle {
                                    id: final_access.handle.id,
                                    version: 0,
                                },
                                kind,
                            })
                    });
                if let Some(previous) = previous
                    && previous.kind != final_access.kind
                {
                    barriers.push(PlannedBarrier {
                        pass_index: pass_idx,
                        timing: BarrierTiming::AfterPass,
                        resource: previous.handle,
                        from: previous.kind,
                        to: final_access.kind,
                    });
                }
                latest_access.insert(final_access.handle.id, *final_access);
            }
        }

        barriers
    }

    fn validate_barrier_resource_bindings(&self, barriers: &[PlannedBarrier]) -> Result<()> {
        for barrier in barriers {
            match self.resources.get(&barrier.resource.id) {
                Some(ResourceDesc::Image { .. }) => {
                    match self.image_handles.get(&barrier.resource.id) {
                        Some(&image) if image != vk::Image::null() => {}
                        _ => {
                            return Err(anyhow!(
                                "image resource id {} requires a barrier but has no non-null bound vk::Image",
                                barrier.resource.id
                            ));
                        }
                    }
                }
                Some(ResourceDesc::Buffer { .. }) => {
                    match self.buffer_handles.get(&barrier.resource.id) {
                        Some(&buffer) if buffer != vk::Buffer::null() => {}
                        _ => {
                            return Err(anyhow!(
                                "buffer resource id {} requires a barrier but has no non-null bound vk::Buffer",
                                barrier.resource.id
                            ));
                        }
                    }
                }
                None => {}
            }
        }
        Ok(())
    }

    fn validate_image_binding(&self, pass_name: &str, handle: ResourceHandle) -> Result<()> {
        if matches!(
            self.resources.get(&handle.id),
            Some(ResourceDesc::Image { .. })
        ) {
            match self.image_handles.get(&handle.id) {
                Some(&image) if image != vk::Image::null() => {}
                _ => {
                    return Err(anyhow!(
                        "pass '{}' image resource id {} has no non-null bound vk::Image",
                        pass_name,
                        handle.id
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_buffer_binding(&self, pass_name: &str, handle: ResourceHandle) -> Result<()> {
        if matches!(
            self.resources.get(&handle.id),
            Some(ResourceDesc::Buffer { .. })
        ) {
            match self.buffer_handles.get(&handle.id) {
                Some(&buffer) if buffer != vk::Buffer::null() => {}
                _ => {
                    return Err(anyhow!(
                        "pass '{}' buffer resource id {} has no non-null bound vk::Buffer",
                        pass_name,
                        handle.id
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_image_usage(
        &self,
        pass_name: &str,
        access: ResourceAccess,
        desc: &ResourceDesc,
    ) -> Result<()> {
        let ResourceDesc::Image { usage, .. } = desc else {
            return Ok(());
        };

        let required = match access.kind {
            AccessKind::Undefined | AccessKind::Present => vk::ImageUsageFlags::empty(),
            AccessKind::ComputeShaderRead
            | AccessKind::ComputeShaderReadWrite
            | AccessKind::ComputeShaderWrite => vk::ImageUsageFlags::STORAGE,
            AccessKind::TransferRead => vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::TransferWrite => vk::ImageUsageFlags::TRANSFER_DST,
        };

        if !required.is_empty() && !usage.contains(required) {
            return Err(anyhow!(
                "pass '{}' resource id {} access {:?} missing required image usage {:?}",
                pass_name,
                access.handle.id,
                access.kind,
                required
            ));
        }
        Ok(())
    }

    fn validate_buffer_usage(
        &self,
        pass_name: &str,
        access: ResourceAccess,
        desc: &ResourceDesc,
    ) -> Result<()> {
        let ResourceDesc::Buffer { usage, .. } = desc else {
            return Ok(());
        };

        let required = match access.kind {
            AccessKind::Undefined | AccessKind::Present => vk::BufferUsageFlags::empty(),
            AccessKind::ComputeShaderRead => {
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER
            }
            AccessKind::ComputeShaderReadWrite | AccessKind::ComputeShaderWrite => {
                vk::BufferUsageFlags::STORAGE_BUFFER
            }
            AccessKind::TransferRead => vk::BufferUsageFlags::TRANSFER_SRC,
            AccessKind::TransferWrite => vk::BufferUsageFlags::TRANSFER_DST,
        };

        if !required.is_empty() && !usage.intersects(required) {
            return Err(anyhow!(
                "pass '{}' resource id {} access {:?} requires buffer usage {:?}",
                pass_name,
                access.handle.id,
                access.kind,
                required
            ));
        }
        Ok(())
    }

    fn validate_resource_timeline(&self) -> Result<()> {
        let mut latest_version: BTreeMap<u32, u32> = BTreeMap::new();
        for pass in &self.passes {
            for read in &pass.decl.reads {
                let current = latest_version
                    .get(&read.id)
                    .copied()
                    .unwrap_or(read.version);
                if read.version < current {
                    return Err(anyhow!(
                        "pass '{}' reads stale resource version {} for id {}; latest version is {}",
                        pass.decl.name,
                        read.version,
                        read.id,
                        current
                    ));
                }
            }
            for write in &pass.decl.writes {
                if let Some(current) = latest_version.get(&write.id).copied()
                    && write.version <= current
                {
                    return Err(anyhow!(
                        "pass '{}' writes stale resource version {} for id {}; latest version is {}",
                        pass.decl.name,
                        write.version,
                        write.id,
                        current
                    ));
                }
                latest_version.insert(write.id, write.version);
            }
        }
        Ok(())
    }

    fn validate_resource_references(&self) -> Result<()> {
        for pass in &self.passes {
            for read in &pass.decl.reads {
                if !self.resources.contains_key(&read.id) {
                    return Err(anyhow!(
                        "pass '{}' reads unknown resource id {}",
                        pass.decl.name,
                        read.id
                    ));
                }
            }
            for write in &pass.decl.writes {
                if write.version > 0 && !self.resources.contains_key(&write.id) {
                    return Err(anyhow!(
                        "pass '{}' writes unknown resource id {}",
                        pass.decl.name,
                        write.id
                    ));
                }
            }
            for access in pass
                .decl
                .accesses
                .iter()
                .chain(pass.decl.final_accesses.iter())
            {
                let Some(desc) = self.resources.get(&access.handle.id) else {
                    return Err(anyhow!(
                        "pass '{}' accesses unknown resource id {}",
                        pass.decl.name,
                        access.handle.id
                    ));
                };
                self.validate_image_binding(pass.decl.name, access.handle)?;
                self.validate_image_usage(pass.decl.name, *access, desc)?;
                self.validate_buffer_binding(pass.decl.name, access.handle)?;
                self.validate_buffer_usage(pass.decl.name, *access, desc)?;
                if access.kind == AccessKind::Present && !matches!(desc, ResourceDesc::Image { .. })
                {
                    return Err(anyhow!(
                        "pass '{}' presents non-image resource id {}",
                        pass.decl.name,
                        access.handle.id
                    ));
                }
            }
            for final_access in &pass.decl.final_accesses {
                if !pass
                    .decl
                    .accesses
                    .iter()
                    .any(|access| access.handle.id == final_access.handle.id)
                {
                    return Err(anyhow!(
                        "pass '{}' declares final access without pass access for resource id {}",
                        pass.decl.name,
                        final_access.handle.id
                    ));
                }
                if final_access.kind == AccessKind::Present
                    && !pass.decl.accesses.iter().any(|access| {
                        access.handle.id == final_access.handle.id
                            && access.kind == AccessKind::TransferWrite
                    })
                {
                    return Err(anyhow!(
                        "pass '{}' presents resource id {} without a transfer write access",
                        pass.decl.name,
                        final_access.handle.id
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn execute(
        self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        frame_index: u64,
    ) {
        assert!(
            self.compiled,
            "RenderGraph must be compiled before execution"
        );
        let RenderGraph {
            passes,
            sorted_order,
            image_handles,
            buffer_handles,
            resources,
            barrier_plan,
            ..
        } = self;
        let mut passes: Vec<Option<PassNode>> = passes.into_iter().map(Some).collect();

        for &idx in &sorted_order {
            if let Some(pass) = passes[idx].take() {
                Self::record_barriers(
                    device,
                    command_buffer,
                    &barrier_plan,
                    &image_handles,
                    &buffer_handles,
                    &resources,
                    idx,
                    BarrierTiming::BeforePass,
                );
                let mut ctx = PassContext {
                    device,
                    command_buffer,
                    frame_index,
                };
                tracing::trace!(pass = pass.decl.name, "executing render pass");
                (pass.execute)(&mut ctx);
                Self::record_barriers(
                    device,
                    command_buffer,
                    &barrier_plan,
                    &image_handles,
                    &buffer_handles,
                    &resources,
                    idx,
                    BarrierTiming::AfterPass,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_barriers(
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        barrier_plan: &[PlannedBarrier],
        image_handles: &BTreeMap<u32, vk::Image>,
        buffer_handles: &BTreeMap<u32, vk::Buffer>,
        resources: &BTreeMap<u32, ResourceDesc>,
        pass_index: usize,
        timing: BarrierTiming,
    ) {
        let matching_barriers = barrier_plan
            .iter()
            .filter(|barrier| barrier.pass_index == pass_index && barrier.timing == timing);

        let image_barriers: Vec<_> = barrier_plan
            .iter()
            .filter(|barrier| barrier.pass_index == pass_index && barrier.timing == timing)
            .filter_map(|barrier| {
                let image = image_handles.get(&barrier.resource.id).copied()?;
                if image == vk::Image::null() {
                    return None;
                }
                Some(
                    vk::ImageMemoryBarrier::default()
                        .old_layout(barrier.from.image_layout())
                        .new_layout(barrier.to.image_layout())
                        .src_access_mask(barrier.from.access_flags())
                        .dst_access_mask(barrier.to.access_flags())
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1),
                        ),
                )
            })
            .collect();

        let buffer_barriers: Vec<_> = barrier_plan
            .iter()
            .filter(|barrier| barrier.pass_index == pass_index && barrier.timing == timing)
            .filter_map(|barrier| {
                let ResourceDesc::Buffer { size, .. } = resources.get(&barrier.resource.id)? else {
                    return None;
                };
                let buffer = buffer_handles.get(&barrier.resource.id).copied()?;
                if buffer == vk::Buffer::null() {
                    return None;
                }
                Some(
                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(barrier.from.access_flags())
                        .dst_access_mask(barrier.to.access_flags())
                        .buffer(buffer)
                        .offset(0)
                        .size(*size),
                )
            })
            .collect();

        if image_barriers.is_empty() && buffer_barriers.is_empty() {
            return;
        }

        let src_stage = matching_barriers
            .clone()
            .fold(vk::PipelineStageFlags::empty(), |stages, barrier| {
                stages | barrier.from.stage_flags()
            });
        let dst_stage = matching_barriers
            .fold(vk::PipelineStageFlags::empty(), |stages, barrier| {
                stages | barrier.to.stage_flags()
            });

        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &buffer_barriers,
                &image_barriers,
            );
        }
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty()
    }

    pub fn has_final_access(&self, kind: AccessKind) -> bool {
        self.passes.iter().any(|pass| {
            pass.decl
                .final_accesses
                .iter()
                .any(|access| access.kind == kind)
        })
    }

    pub fn barrier_plan(&self) -> &[PlannedBarrier] {
        &self.barrier_plan
    }

    pub fn planned_barriers(&self) -> &[PlannedBarrier] {
        &self.barrier_plan
    }
}

impl Default for RenderGraph<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::resource::{AccessKind, QueueType, ResourceDesc};
    use ash::vk::Handle;

    fn fake_image(raw: u64) -> vk::Image {
        vk::Image::from_raw(raw)
    }

    fn fake_buffer(raw: u64) -> vk::Buffer {
        vk::Buffer::from_raw(raw)
    }

    #[test]
    fn empty_graph_compiles() {
        let mut graph = RenderGraph::new();
        graph.compile().unwrap();
        assert_eq!(graph.pass_count(), 0);
    }

    #[test]
    fn single_pass_executes() {
        let mut graph = RenderGraph::new();
        graph.add_pass("test", QueueType::Compute, |_builder| {
            Box::new(|_ctx| {
                // no-op — just verify the closure runs
            })
        });
        graph.compile().unwrap();
        assert_eq!(graph.pass_count(), 1);
        // Can't call execute without a real device/command buffer,
        // but compile + pass_count verifies the graph logic.
    }

    #[test]
    fn dependency_ordering() {
        let mut graph = RenderGraph::new();

        let writes = graph.add_pass("producer", QueueType::Compute, |builder| {
            let _img = builder.create_image(
                100,
                100,
                ash::vk::Format::R8G8B8A8_UNORM,
                ash::vk::ImageUsageFlags::STORAGE,
            );
            Box::new(|_ctx| {})
        });

        assert_eq!(writes.len(), 1);
        let dep = writes[0];
        graph.bind_image(dep, fake_image(1));

        graph.add_pass("consumer", QueueType::Graphics, |builder| {
            builder.read(dep);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        assert_eq!(graph.pass_count(), 2);
        // Producer should come before consumer in sorted_order
        assert_eq!(graph.sorted_order[0], 0); // producer
        assert_eq!(graph.sorted_order[1], 1); // consumer
    }

    #[test]
    fn resource_descriptions_are_recorded_for_created_images() {
        let mut graph = RenderGraph::new();
        let writes = graph.add_pass("producer", QueueType::Compute, |builder| {
            builder.create_image(
                320,
                180,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE,
            );
            Box::new(|_ctx| {})
        });

        assert_eq!(
            graph.resource_desc(writes[0]),
            Some(&ResourceDesc::Image {
                width: 320,
                height: 180,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE,
            })
        );
    }

    #[test]
    fn resource_descriptions_are_recorded_for_created_buffers() {
        let mut graph = RenderGraph::new();
        let writes = graph.add_pass("producer", QueueType::Compute, |builder| {
            builder.create_buffer(4096, vk::BufferUsageFlags::STORAGE_BUFFER);
            Box::new(|_ctx| {})
        });

        assert_eq!(
            graph.resource_desc(writes[0]),
            Some(&ResourceDesc::Buffer {
                size: 4096,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            })
        );
    }

    #[test]
    fn compile_rejects_unknown_resource_reads() {
        let mut graph = RenderGraph::new();
        graph.add_pass("consumer", QueueType::Compute, |builder| {
            builder.read(ResourceHandle {
                id: 999,
                version: 0,
            });
            Box::new(|_ctx| {})
        });

        let error = graph.compile().unwrap_err();
        assert!(error.to_string().contains("unknown resource"));
    }

    #[test]
    fn compile_rejects_unknown_versioned_resource_writes() {
        let mut graph = RenderGraph::new();
        graph.add_pass("writer", QueueType::Compute, |builder| {
            builder.write(ResourceHandle {
                id: 999,
                version: 0,
            });
            Box::new(|_ctx| {})
        });

        let error = graph.compile().unwrap_err();
        assert!(error.to_string().contains("unknown resource"));
    }

    #[test]
    fn compile_reports_cycles() {
        let a = ResourceHandle { id: 1, version: 0 };
        let b = ResourceHandle { id: 2, version: 0 };
        let mut graph = RenderGraph::new();

        graph.passes.push(PassNode {
            decl: PassDecl {
                name: "a",
                queue_type: QueueType::Compute,
                reads: vec![b],
                writes: vec![a],
                accesses: Vec::new(),
                final_accesses: Vec::new(),
            },
            execute: Box::new(|_ctx| {}),
        });
        graph.passes.push(PassNode {
            decl: PassDecl {
                name: "b",
                queue_type: QueueType::Compute,
                reads: vec![a],
                writes: vec![b],
                accesses: Vec::new(),
                final_accesses: Vec::new(),
            },
            execute: Box::new(|_ctx| {}),
        });

        assert!(graph.compile().is_err());
        assert!(!graph.compiled);
    }

    #[test]
    fn compile_rejects_unbound_image_barriers() {
        let mut graph = RenderGraph::new();
        let image = graph.add_pass("producer", QueueType::Compute, |builder| {
            builder.create_image(
                64,
                64,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE,
            );
            Box::new(|_ctx| {})
        })[0];

        graph.add_pass("consumer", QueueType::Compute, |builder| {
            builder.read_as(image, AccessKind::ComputeShaderRead);
            Box::new(|_ctx| {})
        });

        let error = graph.compile().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("has no non-null bound vk::Image")
        );
    }

    #[test]
    fn compile_rejects_null_image_barrier_binding() {
        let mut graph = RenderGraph::new();
        let image = graph.add_pass("producer", QueueType::Compute, |builder| {
            builder.create_image(
                64,
                64,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE,
            );
            Box::new(|_ctx| {})
        })[0];
        graph.bind_image(image, vk::Image::null());

        graph.add_pass("consumer", QueueType::Compute, |builder| {
            builder.read_as(image, AccessKind::ComputeShaderRead);
            Box::new(|_ctx| {})
        });

        let error = graph.compile().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("has no non-null bound vk::Image")
        );
    }

    #[test]
    fn compile_plans_buffer_barriers_between_compute_writes_and_reads() {
        let mut graph = RenderGraph::new();
        let writer = graph.add_pass("writer", QueueType::Compute, |builder| {
            builder.create_buffer(4096, vk::BufferUsageFlags::STORAGE_BUFFER);
            Box::new(|_ctx| {})
        })[0];

        let reader = graph.pass_count();
        graph.add_pass("reader", QueueType::Compute, |builder| {
            builder.read_as(writer, AccessKind::ComputeShaderRead);
            Box::new(|_ctx| {})
        });

        graph.bind_buffer(writer, fake_buffer(44));
        graph.compile().unwrap();
        let barriers = graph.planned_barriers();
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].pass_index, reader);
        assert_eq!(barriers[0].resource, writer);
        assert_eq!(barriers[0].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[0].to, AccessKind::ComputeShaderRead);
    }

    #[test]
    fn imported_buffer_with_access_can_transition_to_compute_write() {
        let mut graph = RenderGraph::new();
        let buffer = graph.import_buffer_with_access(
            fake_buffer(45),
            4096,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            AccessKind::ComputeShaderRead,
        );
        graph.add_pass("writer", QueueType::Compute, |builder| {
            builder.write_as(buffer, AccessKind::ComputeShaderWrite);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        let barriers = graph.planned_barriers();
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].resource, buffer);
        assert_eq!(barriers[0].from, AccessKind::ComputeShaderRead);
        assert_eq!(barriers[0].to, AccessKind::ComputeShaderWrite);
    }

    #[test]
    fn compile_plans_restir_di_buffer_chain_with_uniform_and_light_reads() {
        let mut graph = RenderGraph::new();
        let uniform = graph.import_buffer_with_access(
            fake_buffer(120),
            48,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            AccessKind::ComputeShaderRead,
        );
        let direct_lights = graph.import_buffer_with_access(
            fake_buffer(121),
            4096,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            AccessKind::ComputeShaderRead,
        );
        let initial = graph.import_buffer_with_access(
            fake_buffer(122),
            8192,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            AccessKind::Undefined,
        );
        let temporal = graph.import_buffer_with_access(
            fake_buffer(123),
            8192,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            AccessKind::Undefined,
        );
        let spatial = graph.import_buffer_with_access(
            fake_buffer(124),
            8192,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            AccessKind::Undefined,
        );
        let history = graph.import_buffer_with_access(
            fake_buffer(125),
            8192,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            AccessKind::Undefined,
        );

        let initial_writes = graph.add_pass("restir_di_initial", QueueType::Compute, |builder| {
            builder.read_as(uniform, AccessKind::ComputeShaderRead);
            builder.read_as(direct_lights, AccessKind::ComputeShaderRead);
            builder.write_as(initial, AccessKind::ComputeShaderWrite);
            Box::new(|_ctx| {})
        });
        let temporal_writes = graph.add_pass("restir_di_temporal", QueueType::Compute, |builder| {
            builder.read_as(uniform, AccessKind::ComputeShaderRead);
            builder.read_as(initial_writes[0], AccessKind::ComputeShaderRead);
            builder.read_as(history, AccessKind::ComputeShaderRead);
            builder.write_as(temporal, AccessKind::ComputeShaderWrite);
            Box::new(|_ctx| {})
        });
        let spatial_writes = graph.add_pass("restir_di_spatial", QueueType::Compute, |builder| {
            builder.read_as(uniform, AccessKind::ComputeShaderRead);
            builder.read_as(temporal_writes[0], AccessKind::ComputeShaderRead);
            builder.write_as(spatial, AccessKind::ComputeShaderWrite);
            Box::new(|_ctx| {})
        });
        graph.add_pass("vpt", QueueType::Compute, |builder| {
            builder.read_as(spatial_writes[0], AccessKind::ComputeShaderRead);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        assert_eq!(graph.sorted_order, vec![0, 1, 2, 3]);
        let barriers = graph.planned_barriers();
        assert_eq!(barriers.len(), 7);
        assert_eq!(barriers[0].resource.id, initial.id);
        assert_eq!(barriers[0].from, AccessKind::Undefined);
        assert_eq!(barriers[0].to, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[1].resource.id, initial.id);
        assert_eq!(barriers[1].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[1].to, AccessKind::ComputeShaderRead);
        assert_eq!(barriers[2].resource.id, history.id);
        assert_eq!(barriers[2].from, AccessKind::Undefined);
        assert_eq!(barriers[2].to, AccessKind::ComputeShaderRead);
        assert_eq!(barriers[3].resource.id, temporal.id);
        assert_eq!(barriers[3].from, AccessKind::Undefined);
        assert_eq!(barriers[3].to, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[4].resource.id, temporal.id);
        assert_eq!(barriers[4].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[4].to, AccessKind::ComputeShaderRead);
        assert_eq!(barriers[5].resource.id, spatial.id);
        assert_eq!(barriers[5].from, AccessKind::Undefined);
        assert_eq!(barriers[5].to, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[6].resource.id, spatial.id);
        assert_eq!(barriers[6].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[6].to, AccessKind::ComputeShaderRead);
    }

    #[test]
    fn compile_rejects_null_buffer_barrier_binding() {
        let mut graph = RenderGraph::new();
        let writer = graph.add_pass("writer", QueueType::Compute, |builder| {
            builder.create_buffer(4096, vk::BufferUsageFlags::STORAGE_BUFFER);
            Box::new(|_ctx| {})
        })[0];
        graph.add_pass("reader", QueueType::Compute, |builder| {
            builder.read_as(writer, AccessKind::ComputeShaderRead);
            Box::new(|_ctx| {})
        });

        graph.bind_buffer(writer, vk::Buffer::null());
        let err = graph.compile().unwrap_err().to_string();
        assert!(err.contains("buffer resource id"));
        assert!(err.contains("non-null bound vk::Buffer"));
    }

    #[test]
    fn compile_rejects_transfer_read_without_transfer_src_buffer_usage() {
        let mut graph = RenderGraph::new();
        let buffer = graph.import_buffer_with_access(
            fake_buffer(46),
            4096,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            AccessKind::ComputeShaderWrite,
        );
        graph.add_pass("copy", QueueType::Transfer, |builder| {
            builder.read_as(buffer, AccessKind::TransferRead);
            Box::new(|_ctx| {})
        });

        let err = graph.compile().unwrap_err().to_string();
        assert!(err.contains("requires buffer usage"));
    }

    #[test]
    fn imported_images_can_be_read_by_passes() {
        let mut graph = RenderGraph::new();
        let imported = graph.import_image(
            1280,
            720,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_SRC,
        );

        graph.add_pass("blit", QueueType::Graphics, |builder| {
            builder.read_as(imported, AccessKind::TransferRead);
            Box::new(|_ctx| {})
        });
        graph.bind_image(imported, fake_image(2));

        graph.compile().unwrap();
        assert_eq!(
            graph.resource_desc(imported),
            Some(&ResourceDesc::Image {
                width: 1280,
                height: 720,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::TRANSFER_SRC,
            })
        );
    }

    #[test]
    fn access_kind_maps_storage_write_to_vulkan_barrier_fields() {
        let cases = [
            (
                AccessKind::Undefined,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::ImageLayout::UNDEFINED,
            ),
            (
                AccessKind::ComputeShaderRead,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ,
                vk::ImageLayout::GENERAL,
            ),
            (
                AccessKind::ComputeShaderWrite,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
                vk::ImageLayout::GENERAL,
            ),
            (
                AccessKind::TransferRead,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ),
            (
                AccessKind::TransferWrite,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ),
            (
                AccessKind::Present,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::AccessFlags::empty(),
                vk::ImageLayout::PRESENT_SRC_KHR,
            ),
        ];
        for (kind, stage, access, layout) in cases {
            assert_eq!(kind.stage_flags(), stage);
            assert_eq!(kind.access_flags(), access);
            assert_eq!(kind.image_layout(), layout);
        }
        assert_eq!(
            AccessKind::ComputeShaderReadWrite.access_flags(),
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE
        );
        assert_eq!(
            AccessKind::from_swapchain_layout(vk::ImageLayout::PRESENT_SRC_KHR),
            Some(AccessKind::Present)
        );
        assert_eq!(
            AccessKind::from_swapchain_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            None
        );
    }

    #[test]
    fn read_old_version_before_write_new_version_keeps_submission_order() {
        let mut graph = RenderGraph::new();
        let image = graph.import_image_with_access(
            fake_image(110),
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
            AccessKind::ComputeShaderWrite,
        );

        graph.add_pass("read_old", QueueType::Compute, |builder| {
            builder.read_as(image, AccessKind::ComputeShaderRead);
            Box::new(|_ctx| {})
        });
        graph.add_pass("write_new", QueueType::Compute, |builder| {
            builder.write_as(image, AccessKind::ComputeShaderWrite);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        assert_eq!(graph.sorted_order, vec![0, 1]);
        let barriers = graph.barrier_plan();
        assert_eq!(barriers.len(), 2);
        assert_eq!(barriers[0].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[0].to, AccessKind::ComputeShaderRead);
        assert_eq!(barriers[1].from, AccessKind::ComputeShaderRead);
        assert_eq!(barriers[1].to, AccessKind::ComputeShaderWrite);
    }

    #[test]
    fn compile_rejects_stale_version_read_after_newer_write() {
        let mut graph = RenderGraph::new();
        let image = graph.import_image_with_access(
            fake_image(111),
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
            AccessKind::ComputeShaderRead,
        );

        graph.add_pass("write_new", QueueType::Compute, |builder| {
            builder.write_as(image, AccessKind::ComputeShaderWrite);
            Box::new(|_ctx| {})
        });
        graph.add_pass("read_stale", QueueType::Compute, |builder| {
            builder.read_as(image, AccessKind::ComputeShaderRead);
            Box::new(|_ctx| {})
        });

        let error = graph.compile().unwrap_err();
        assert!(error.to_string().contains("stale resource version"));
    }

    #[test]
    fn compile_rejects_final_access_without_matching_pass_access() {
        let mut graph = RenderGraph::new();
        let swapchain = graph.import_image_with_access(
            fake_image(112),
            320,
            180,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Present,
        );

        graph.add_pass("fake_present", QueueType::Graphics, |builder| {
            builder.finish_as(swapchain, AccessKind::Present);
            Box::new(|_ctx| {})
        });

        let error = graph.compile().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("final access without pass access")
        );
    }

    #[test]
    fn compile_rejects_transfer_read_without_transfer_src_usage() {
        let mut graph = RenderGraph::new();
        let image = graph.import_image_with_access(
            fake_image(113),
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
            AccessKind::ComputeShaderWrite,
        );

        graph.add_pass("invalid_blit_src", QueueType::Graphics, |builder| {
            builder.read_as(image, AccessKind::TransferRead);
            Box::new(|_ctx| {})
        });

        let error = graph.compile().unwrap_err();
        assert!(error.to_string().contains("missing required image usage"));
    }

    #[test]
    fn compile_rejects_null_image_even_when_no_barrier_is_needed() {
        let mut graph = RenderGraph::new();
        let image = graph.import_image_with_access(
            vk::Image::null(),
            64,
            64,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::STORAGE,
            AccessKind::ComputeShaderRead,
        );

        graph.add_pass("read_same_state", QueueType::Compute, |builder| {
            builder.read_as(image, AccessKind::ComputeShaderRead);
            Box::new(|_ctx| {})
        });

        let error = graph.compile().unwrap_err();
        assert!(
            error
                .to_string()
                .contains("has no non-null bound vk::Image")
        );
    }

    #[test]
    fn compile_plans_full_primary_lighting_postprocess_blit_chain() {
        let mut graph = RenderGraph::new();
        let primary_writes = graph.add_pass("primary_ray", QueueType::Compute, |builder| {
            builder.create_image(
                320,
                180,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageUsageFlags::STORAGE,
            );
            builder.create_image(
                320,
                180,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            builder.create_image(
                320,
                180,
                vk::Format::R8G8B8A8_UINT,
                vk::ImageUsageFlags::STORAGE,
            );
            Box::new(|_ctx| {})
        });
        for handle in &primary_writes {
            graph.bind_image(*handle, fake_image(handle.id as u64 + 10));
        }

        let lighting_output = graph.add_pass("lighting", QueueType::Compute, |builder| {
            builder.read(primary_writes[0]);
            builder.read(primary_writes[1]);
            builder.read(primary_writes[2]);
            builder.create_image(
                320,
                180,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            Box::new(|_ctx| {})
        })[0];
        graph.bind_image(lighting_output, fake_image(20));

        let postprocess_output = graph.add_pass("postprocess", QueueType::Compute, |builder| {
            builder.read_as(lighting_output, AccessKind::ComputeShaderRead);
            builder.create_image(
                320,
                180,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            Box::new(|_ctx| {})
        })[0];
        graph.bind_image(postprocess_output, fake_image(21));

        let swapchain = graph.import_image_with_access(
            fake_image(22),
            320,
            180,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Present,
        );
        graph.add_pass("blit", QueueType::Graphics, |builder| {
            builder.read_as(postprocess_output, AccessKind::TransferRead);
            builder.write_as(swapchain, AccessKind::TransferWrite);
            builder.finish_as(swapchain, AccessKind::Present);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        let barriers = graph.barrier_plan();
        assert_eq!(barriers.len(), 12);
        assert_eq!(barriers[0].from, AccessKind::Undefined);
        assert_eq!(barriers[0].to, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[3].resource, primary_writes[0]);
        assert_eq!(barriers[3].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[3].to, AccessKind::ComputeShaderRead);
        assert_eq!(barriers[6].resource, lighting_output);
        assert_eq!(barriers[6].from, AccessKind::Undefined);
        assert_eq!(barriers[6].to, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[10].resource.id, swapchain.id);
        assert_eq!(barriers[10].from, AccessKind::Present);
        assert_eq!(barriers[10].to, AccessKind::TransferWrite);
        assert_eq!(barriers[11].resource.id, swapchain.id);
        assert_eq!(barriers[11].timing, BarrierTiming::AfterPass);
        assert_eq!(barriers[11].from, AccessKind::TransferWrite);
        assert_eq!(barriers[11].to, AccessKind::Present);
    }

    #[test]
    fn compile_plans_image_barriers_between_declared_accesses() {
        let mut graph = RenderGraph::new();
        let lighting_output = graph.add_pass("lighting", QueueType::Compute, |builder| {
            builder.create_image(
                320,
                180,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            Box::new(|_ctx| {})
        })[0];
        graph.bind_image(lighting_output, fake_image(30));

        let postprocess_output = graph.add_pass("postprocess", QueueType::Compute, |builder| {
            builder.read_as(lighting_output, AccessKind::ComputeShaderRead);
            builder.create_image(
                320,
                180,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            Box::new(|_ctx| {})
        })[0];
        graph.bind_image(postprocess_output, fake_image(31));

        graph.add_pass("blit", QueueType::Graphics, |builder| {
            builder.read_as(postprocess_output, AccessKind::TransferRead);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        let barriers = graph.barrier_plan();
        assert_eq!(barriers.len(), 4);
        assert_eq!(barriers[0].resource, lighting_output);
        assert_eq!(barriers[0].from, AccessKind::Undefined);
        assert_eq!(barriers[0].to, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[1].resource, lighting_output);
        assert_eq!(barriers[1].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[1].to, AccessKind::ComputeShaderRead);
        assert_eq!(barriers[2].resource, postprocess_output);
        assert_eq!(barriers[2].from, AccessKind::Undefined);
        assert_eq!(barriers[2].to, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[3].resource, postprocess_output);
        assert_eq!(barriers[3].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[3].to, AccessKind::TransferRead);
    }

    #[test]
    fn compile_plans_blit_source_swapchain_and_present_barriers() {
        let mut graph = RenderGraph::new();
        let src = graph.import_image_with_access(
            fake_image(40),
            320,
            180,
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::ComputeShaderWrite,
        );
        let swapchain = graph.import_image_with_access(
            fake_image(41),
            320,
            180,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Undefined,
        );

        graph.add_pass("blit", QueueType::Graphics, |builder| {
            builder.read_as(src, AccessKind::TransferRead);
            builder.write_as(swapchain, AccessKind::TransferWrite);
            builder.finish_as(swapchain, AccessKind::Present);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        let barriers = graph.barrier_plan();
        assert_eq!(barriers.len(), 3);
        assert_eq!(barriers[0].timing, BarrierTiming::BeforePass);
        assert_eq!(barriers[0].resource, src);
        assert_eq!(barriers[0].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[0].to, AccessKind::TransferRead);
        assert_eq!(barriers[1].timing, BarrierTiming::BeforePass);
        assert_eq!(barriers[1].resource.id, swapchain.id);
        assert_eq!(barriers[1].from, AccessKind::Undefined);
        assert_eq!(barriers[1].to, AccessKind::TransferWrite);
        assert_eq!(barriers[2].timing, BarrierTiming::AfterPass);
        assert_eq!(barriers[2].resource.id, swapchain.id);
        assert_eq!(barriers[2].from, AccessKind::TransferWrite);
        assert_eq!(barriers[2].to, AccessKind::Present);
    }

    #[test]
    fn compile_plans_raw_gbuffer_fallback_to_swapchain() {
        let mut graph = RenderGraph::new();
        let primary_writes = graph.add_pass("primary_ray", QueueType::Compute, |builder| {
            builder.create_image(
                320,
                180,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageUsageFlags::STORAGE,
            );
            builder.create_image(
                320,
                180,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            );
            builder.create_image(
                320,
                180,
                vk::Format::R8G8B8A8_UINT,
                vk::ImageUsageFlags::STORAGE,
            );
            Box::new(|_ctx| {})
        });
        for handle in &primary_writes {
            graph.bind_image(*handle, fake_image(handle.id as u64 + 50));
        }
        let swapchain = graph.import_image_with_access(
            fake_image(60),
            320,
            180,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Undefined,
        );

        graph.add_pass("blit", QueueType::Graphics, |builder| {
            builder.read_as(primary_writes[1], AccessKind::TransferRead);
            builder.write_as(swapchain, AccessKind::TransferWrite);
            builder.finish_as(swapchain, AccessKind::Present);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        let barriers = graph.barrier_plan();
        assert_eq!(barriers.len(), 6);
        assert_eq!(barriers[3].resource, primary_writes[1]);
        assert_eq!(barriers[3].from, AccessKind::ComputeShaderWrite);
        assert_eq!(barriers[3].to, AccessKind::TransferRead);
        assert_eq!(barriers[4].resource.id, swapchain.id);
        assert_eq!(barriers[4].from, AccessKind::Undefined);
        assert_eq!(barriers[4].to, AccessKind::TransferWrite);
        assert_eq!(barriers[5].timing, BarrierTiming::AfterPass);
        assert_eq!(barriers[5].from, AccessKind::TransferWrite);
        assert_eq!(barriers[5].to, AccessKind::Present);
    }

    #[test]
    fn compile_plans_clear_fallback_to_present_from_previous_present() {
        let mut graph = RenderGraph::new();
        let swapchain = graph.import_image_with_access(
            fake_image(70),
            320,
            180,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Present,
        );

        graph.add_pass("clear_swapchain", QueueType::Graphics, |builder| {
            builder.write_as(swapchain, AccessKind::TransferWrite);
            builder.finish_as(swapchain, AccessKind::Present);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        let barriers = graph.barrier_plan();
        assert_eq!(barriers.len(), 2);
        assert_eq!(barriers[0].timing, BarrierTiming::BeforePass);
        assert_eq!(barriers[0].from, AccessKind::Present);
        assert_eq!(barriers[0].to, AccessKind::TransferWrite);
        assert_eq!(barriers[1].timing, BarrierTiming::AfterPass);
        assert_eq!(barriers[1].from, AccessKind::TransferWrite);
        assert_eq!(barriers[1].to, AccessKind::Present);
    }

    #[test]
    fn compile_plans_persistent_readwrite_image_first_use() {
        let mut graph = RenderGraph::new();
        let accumulation = graph.import_image_with_access(
            fake_image(80),
            320,
            180,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );

        let writes = graph.add_pass("vpt", QueueType::Compute, |builder| {
            builder.write_as(accumulation, AccessKind::ComputeShaderReadWrite);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        assert_eq!(writes.len(), 1);
        assert_eq!(graph.passes[0].decl.reads, vec![accumulation]);
        let barriers = graph.barrier_plan();
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].from, AccessKind::Undefined);
        assert_eq!(barriers[0].to, AccessKind::ComputeShaderReadWrite);
    }

    #[test]
    fn compile_plans_persistent_write_only_image_first_use() {
        let mut graph = RenderGraph::new();
        let accumulation = graph.import_image_with_access(
            fake_image(85),
            320,
            180,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::Undefined,
        );

        let writes = graph.add_pass("vpt", QueueType::Compute, |builder| {
            builder.write_as(accumulation, AccessKind::ComputeShaderWrite);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        assert_eq!(writes.len(), 1);
        assert!(graph.passes[0].decl.reads.is_empty());
        let barriers = graph.barrier_plan();
        assert_eq!(barriers.len(), 1);
        assert_eq!(barriers[0].from, AccessKind::Undefined);
        assert_eq!(barriers[0].to, AccessKind::ComputeShaderWrite);
    }

    #[test]
    fn compile_plans_persistent_readwrite_image_after_previous_read() {
        let mut graph = RenderGraph::new();
        let accumulation = graph.import_image_with_access(
            fake_image(90),
            320,
            180,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            AccessKind::ComputeShaderRead,
        );

        let vpt_writes = graph.add_pass("vpt", QueueType::Compute, |builder| {
            builder.write_as(accumulation, AccessKind::ComputeShaderReadWrite);
            Box::new(|_ctx| {})
        });
        let vpt_output = vpt_writes[0];
        graph.add_pass("postprocess", QueueType::Compute, |builder| {
            builder.read_as(vpt_output, AccessKind::ComputeShaderRead);
            Box::new(|_ctx| {})
        });

        graph.compile().unwrap();
        let barriers = graph.barrier_plan();
        assert_eq!(barriers.len(), 2);
        assert_eq!(barriers[0].from, AccessKind::ComputeShaderRead);
        assert_eq!(barriers[0].to, AccessKind::ComputeShaderReadWrite);
        assert_eq!(barriers[1].from, AccessKind::ComputeShaderReadWrite);
        assert_eq!(barriers[1].to, AccessKind::ComputeShaderRead);
    }

    #[test]
    fn has_final_access_reports_present_passes() {
        let mut graph = RenderGraph::new();
        assert!(!graph.has_final_access(AccessKind::Present));

        let swapchain = graph.import_image_with_access(
            fake_image(100),
            320,
            180,
            vk::Format::B8G8R8A8_UNORM,
            vk::ImageUsageFlags::TRANSFER_DST,
            AccessKind::Undefined,
        );

        graph.add_pass("clear_swapchain", QueueType::Graphics, |builder| {
            builder.write_as(swapchain, AccessKind::TransferWrite);
            builder.finish_as(swapchain, AccessKind::Present);
            Box::new(|_ctx| {})
        });

        assert!(graph.has_final_access(AccessKind::Present));
    }

    #[test]
    fn compile_rejects_unknown_final_access_resources() {
        let mut graph = RenderGraph::new();
        graph.add_pass("invalid_present", QueueType::Graphics, |builder| {
            builder.finish_as(
                ResourceHandle {
                    id: 999,
                    version: 0,
                },
                AccessKind::Present,
            );
            Box::new(|_ctx| {})
        });

        assert!(graph.has_final_access(AccessKind::Present));
        let error = graph.compile().unwrap_err();
        assert!(error.to_string().contains("unknown resource id 999"));
    }

    #[test]
    fn stable_order_preserves_independent_pass_submission_order() {
        let mut graph = RenderGraph::new();
        graph.add_pass("first", QueueType::Compute, |_builder| Box::new(|_ctx| {}));
        graph.add_pass("second", QueueType::Compute, |_builder| Box::new(|_ctx| {}));
        graph.add_pass("third", QueueType::Compute, |_builder| Box::new(|_ctx| {}));

        graph.compile().unwrap();
        assert_eq!(graph.sorted_order, vec![0, 1, 2]);
    }
}
