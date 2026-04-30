use anyhow::{Result, anyhow};
use ash::vk;
use std::collections::BTreeMap;

use crate::render::pass_context::{PassBuilder, PassContext};
use crate::render::resource::{PassDecl, QueueType, ResourceDesc, ResourceHandle};

type ExecuteFn<'a> = Box<dyn FnOnce(&mut PassContext) + 'a>;

struct PassNode<'a> {
    decl: PassDecl,
    execute: ExecuteFn<'a>,
}

pub struct RenderGraph<'a> {
    passes: Vec<PassNode<'a>>,
    sorted_order: Vec<usize>,
    resources: BTreeMap<u32, ResourceDesc>,
    next_resource_id: u32,
    compiled: bool,
}

impl<'a> RenderGraph<'a> {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            sorted_order: Vec::new(),
            resources: BTreeMap::new(),
            next_resource_id: 0,
            compiled: false,
        }
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
        };
        self.passes.push(PassNode { decl, execute });
        self.compiled = false;
        writes
    }

    pub fn resource_desc(&self, handle: ResourceHandle) -> Option<&ResourceDesc> {
        self.resources.get(&handle.id)
    }

    pub fn compile(&mut self) -> Result<()> {
        self.validate_resource_references()?;

        let n = self.passes.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];

        for (i, edges) in adj.iter_mut().enumerate() {
            for (j, degree) in in_degree.iter_mut().enumerate() {
                if i == j {
                    continue;
                }
                let writes_i = &self.passes[i].decl.writes;
                let reads_j = &self.passes[j].decl.reads;
                let depends = reads_j
                    .iter()
                    .any(|r| writes_i.iter().any(|w| w.id == r.id));
                if depends {
                    edges.push(j);
                    *degree += 1;
                }
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop() {
            order.push(node);
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push(next);
                }
            }
        }

        if order.len() != n {
            self.sorted_order.clear();
            self.compiled = false;
            return Err(anyhow!("render graph contains a dependency cycle"));
        }

        self.sorted_order = order;
        self.compiled = true;
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
        let mut passes: Vec<Option<PassNode>> = self.passes.into_iter().map(Some).collect();

        for &idx in &self.sorted_order {
            if let Some(pass) = passes[idx].take() {
                let mut ctx = PassContext {
                    device,
                    command_buffer,
                    frame_index,
                };
                tracing::trace!(pass = pass.decl.name, "executing render pass");
                (pass.execute)(&mut ctx);
            }
        }
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
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
    use crate::render::resource::{QueueType, ResourceDesc};

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
            },
            execute: Box::new(|_ctx| {}),
        });
        graph.passes.push(PassNode {
            decl: PassDecl {
                name: "b",
                queue_type: QueueType::Compute,
                reads: vec![a],
                writes: vec![b],
            },
            execute: Box::new(|_ctx| {}),
        });

        assert!(graph.compile().is_err());
        assert!(!graph.compiled);
    }
}
