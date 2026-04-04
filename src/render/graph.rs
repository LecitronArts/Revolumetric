use ash::vk;

use crate::render::pass_context::{PassBuilder, PassContext};
use crate::render::resource::{PassDecl, QueueType, ResourceHandle};

type ExecuteFn<'a> = Box<dyn FnOnce(&mut PassContext) + 'a>;

struct PassNode<'a> {
    decl: PassDecl,
    execute: ExecuteFn<'a>,
}

pub struct RenderGraph<'a> {
    passes: Vec<PassNode<'a>>,
    sorted_order: Vec<usize>,
    next_resource_id: u32,
    compiled: bool,
}

impl<'a> RenderGraph<'a> {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            sorted_order: Vec::new(),
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

    pub fn compile(&mut self) {
        let n = self.passes.len();
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];

        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                let writes_i = &self.passes[i].decl.writes;
                let reads_j = &self.passes[j].decl.reads;
                let depends = reads_j.iter().any(|r| {
                    writes_i.iter().any(|w| w.id == r.id)
                });
                if depends {
                    adj[i].push(j);
                    in_degree[j] += 1;
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

        self.sorted_order = order;
        self.compiled = true;
    }

    pub fn execute(self, device: &ash::Device, command_buffer: vk::CommandBuffer, frame_index: u64) {
        assert!(self.compiled, "RenderGraph must be compiled before execution");
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
