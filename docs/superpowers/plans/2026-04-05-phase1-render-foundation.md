# Phase 1: Render Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the GPU resource management layer, render graph, Slang shader compilation pipeline, and a fullscreen compute pass that writes a test pattern to the swapchain — proving the entire pipeline works end-to-end.

**Architecture:** Replace the current hardcoded `render_frame()` clear-color loop with a data-driven render graph. GPU buffers and images are managed through typed wrappers over `ash` + `gpu-allocator`. Slang shaders compile to SPIR-V at build time via FFI. A single fullscreen compute pass writes to a storage image that gets blitted to the swapchain.

**Tech Stack:** Rust, ash (Vulkan 1.3), gpu-allocator, Slang (via slang-sys FFI), glam, winit

---

## File Map

### New Files

| Path | Responsibility |
|------|---------------|
| `src/render/allocator.rs` | GPU memory allocator wrapper over `gpu-allocator` |
| `src/render/graph.rs` | RenderGraph, PassNode, ResourceRegistry, BarrierBatcher (replace stub) |
| `src/render/resource.rs` | ResourceHandle, ResourceDesc, transient resource tracking |
| `src/render/pass_context.rs` | PassContext handed to pass execute closures |
| `src/render/passes/test_pattern.rs` | Fullscreen compute pass: writes gradient to storage image |
| `src/render/passes/blit_to_swapchain.rs` | Copies storage image → swapchain image |
| `assets/shaders/passes/test_pattern.slang` | Compute shader: rainbow gradient based on UV + time |
| `build.rs` | Build script: compile Slang shaders to SPIR-V at build time |
| `tests/render_graph_test.rs` | Unit tests for render graph dependency sort and barriers |

### Modified Files

| Path | Changes |
|------|---------|
| `src/render/device.rs` | Add gpu-allocator instance, expose `Device` + `Allocator` to graph |
| `src/render/frame.rs` | Extend FrameContext with command buffer access, image index |
| `src/render/mod.rs` | Add new modules |
| `src/render/passes/mod.rs` | Add new pass modules |
| `src/app.rs` | Replace hardcoded render_frame with graph.compile() + graph.execute() |
| `src/render/pipeline.rs` | Implement compute pipeline creation + caching |
| `src/render/descriptor.rs` | Implement descriptor set layout/pool/allocation |
| `src/render/image.rs` | Implement GPU image creation with allocator |
| `src/render/buffer.rs` | Implement GPU buffer creation with allocator |
| `Cargo.toml` | Add `gpu-allocator` features, `slang-sys` or build-dep |

---

## Task 1: GPU Memory Allocator Integration

**Files:**
- Modify: `src/render/device.rs`
- Create: `src/render/allocator.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add gpu-allocator to device initialization**

In `Cargo.toml`, ensure `gpu-allocator` has the `vulkan` feature:
```toml
gpu-allocator = { version = "0.28", features = ["vulkan"] }
```

- [ ] **Step 2: Create allocator.rs wrapper**

```rust
// src/render/allocator.rs
use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use parking_lot::Mutex;
use std::sync::Arc;

pub struct GpuAllocator {
    inner: Mutex<Allocator>,
}

impl GpuAllocator {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;
        Ok(Self {
            inner: Mutex::new(allocator),
        })
    }

    pub fn allocate(&self, desc: &AllocationCreateDesc) -> Result<Allocation> {
        Ok(self.inner.lock().allocate(desc)?)
    }

    pub fn free(&self, allocation: Allocation) -> Result<()> {
        Ok(self.inner.lock().free(allocation)?)
    }
}
```

- [ ] **Step 3: Integrate allocator into RenderDevice**

In `src/render/device.rs`:

a) Enable `bufferDeviceAddress` device feature. In the `RenderDevice::new()` method, before creating the logical device, add:
```rust
let mut bda_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
    .buffer_device_address(true);

let device_create_info = vk::DeviceCreateInfo::default()
    .queue_create_infos(&queue_create_infos)
    .enabled_extension_names(&device_extension_names)
    .push_next(&mut bda_features);
```

b) Add `allocator: GpuAllocator` field to `RenderDevice`. Initialize after device creation:
```rust
let allocator = GpuAllocator::new(&instance, &device, selection.physical_device)?;
```

c) Expose via `pub fn allocator(&self) -> &GpuAllocator`.

d) In `Drop for RenderDevice`, drop the allocator *before* destroying the device (move it to a field that implements Drop in the right order, or manually drop it).

- [ ] **Step 4: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`
Expected: successful build (or only warnings)

- [ ] **Step 5: Commit**

```
git add src/render/allocator.rs src/render/device.rs Cargo.toml src/render/mod.rs
git commit -m "feat(render): integrate gpu-allocator into RenderDevice"
```

---

## Task 2: GPU Buffer Wrapper

**Files:**
- Modify: `src/render/buffer.rs`

- [ ] **Step 1: Implement GpuBuffer**

```rust
// src/render/buffer.rs
use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;

pub struct GpuBuffer {
    pub handle: vk::Buffer,
    pub size: vk::DeviceSize,
    pub allocation: Option<Allocation>,
    pub usage: vk::BufferUsageFlags,
}

impl GpuBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { device.create_buffer(&buffer_info, None) }
            .context("failed to create buffer")?;

        let requirements = unsafe { device.get_buffer_memory_requirements(handle) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe {
            device.bind_buffer_memory(handle, allocation.memory(), allocation.offset())
        }
        .context("failed to bind buffer memory")?;

        Ok(Self {
            handle,
            size,
            allocation: Some(allocation),
            usage,
        })
    }

    pub fn destroy(mut self, device: &ash::Device, allocator: &GpuAllocator) {
        unsafe { device.destroy_buffer(self.handle, None) };
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
    }

    /// Returns a mapped pointer if the buffer is host-visible.
    pub fn mapped_ptr(&self) -> Option<*mut u8> {
        self.allocation
            .as_ref()
            .and_then(|a| a.mapped_ptr())
            .map(|p| p.as_ptr() as *mut u8)
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```
git add src/render/buffer.rs
git commit -m "feat(render): implement GpuBuffer with gpu-allocator"
```

---

## Task 3: GPU Image Wrapper

**Files:**
- Modify: `src/render/image.rs`

- [ ] **Step 1: Implement GpuImage**

```rust
// src/render/image.rs
use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

use crate::render::allocator::GpuAllocator;

pub struct GpuImage {
    pub handle: vk::Image,
    pub view: vk::ImageView,
    pub extent: vk::Extent3D,
    pub format: vk::Format,
    pub allocation: Option<Allocation>,
    pub current_layout: vk::ImageLayout,
}

#[derive(Clone)]
pub struct GpuImageDesc {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
    pub name: &'static str,
}

impl GpuImage {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        desc: &GpuImageDesc,
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width: desc.width,
            height: desc.height,
            depth: desc.depth,
        };

        let image_type = if desc.depth > 1 {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(image_type)
            .format(desc.format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let handle = unsafe { device.create_image(&image_info, None) }
            .context("failed to create image")?;

        let requirements = unsafe { device.get_image_memory_requirements(handle) };

        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: desc.name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe { device.bind_image_memory(handle, allocation.memory(), allocation.offset()) }
            .context("failed to bind image memory")?;

        let view_type = if desc.depth > 1 {
            vk::ImageViewType::TYPE_3D
        } else {
            vk::ImageViewType::TYPE_2D
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(handle)
            .view_type(view_type)
            .format(desc.format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(desc.aspect)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let view = unsafe { device.create_image_view(&view_info, None) }
            .context("failed to create image view")?;

        Ok(Self {
            handle,
            view,
            extent,
            format: desc.format,
            allocation: Some(allocation),
            current_layout: vk::ImageLayout::UNDEFINED,
        })
    }

    pub fn destroy(mut self, device: &ash::Device, allocator: &GpuAllocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.handle, None);
        }
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```
git add src/render/image.rs
git commit -m "feat(render): implement GpuImage with allocator and image view"
```

---

## Task 4: Descriptor Set Management

**Files:**
- Modify: `src/render/descriptor.rs`

- [ ] **Step 1: Implement descriptor pool and set layout builder**

```rust
// src/render/descriptor.rs
use anyhow::{Context, Result};
use ash::vk;

pub struct DescriptorLayoutBuilder {
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl DescriptorLayoutBuilder {
    pub fn new() -> Self {
        Self { bindings: Vec::new() }
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

    pub fn build(&self, device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
        let create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&self.bindings);
        unsafe { device.create_descriptor_set_layout(&create_info, None) }
            .context("failed to create descriptor set layout")
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
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```
git add src/render/descriptor.rs
git commit -m "feat(render): implement descriptor set layout builder and pool"
```

---

## Task 5: Compute Pipeline Creation

**Files:**
- Modify: `src/render/pipeline.rs`

- [ ] **Step 1: Implement compute pipeline wrapper**

```rust
// src/render/pipeline.rs
use anyhow::{Context, Result};
use ash::vk;

pub struct ComputePipeline {
    pub handle: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl ComputePipeline {
    pub fn new(
        device: &ash::Device,
        shader_module: vk::ShaderModule,
        entry_point: &std::ffi::CStr,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> Result<Self> {
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(descriptor_set_layouts)
            .push_constant_ranges(push_constant_ranges);
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
            .context("failed to create pipeline layout")?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(entry_point);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let handle = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        }
        .map_err(|(_, err)| err)
        .context("failed to create compute pipeline")?[0];

        Ok(Self { handle, layout })
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.handle, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub fn create_shader_module(device: &ash::Device, spirv: &[u8]) -> Result<vk::ShaderModule> {
    assert!(spirv.len() % 4 == 0, "SPIR-V byte length must be a multiple of 4");
    // Copy into an aligned Vec<u32> to avoid UB from misaligned &[u8] → &[u32] cast.
    // include_bytes!() only guarantees 1-byte alignment.
    let mut code = vec![0u32; spirv.len() / 4];
    unsafe {
        std::ptr::copy_nonoverlapping(spirv.as_ptr(), code.as_mut_ptr() as *mut u8, spirv.len());
    }
    let create_info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&create_info, None) }
        .context("failed to create shader module")
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```
git add src/render/pipeline.rs
git commit -m "feat(render): implement compute pipeline and shader module creation"
```

---

## Task 6: Render Graph Core

**Files:**
- Modify: `src/render/graph.rs`
- Create: `src/render/resource.rs`
- Create: `src/render/pass_context.rs`

- [ ] **Step 1: Define resource types**

```rust
// src/render/resource.rs
use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle {
    pub id: u32,
    pub version: u32,
}

#[derive(Debug, Clone)]
pub enum ResourceDesc {
    Image {
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    },
    Buffer {
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    Graphics,
    Compute,
    Transfer,
}

#[derive(Debug, Clone)]
pub struct PassDecl {
    pub name: &'static str,
    pub queue_type: QueueType,
    pub reads: Vec<ResourceHandle>,
    pub writes: Vec<ResourceHandle>,
}
```

- [ ] **Step 2: Implement PassBuilder**

```rust
// src/render/pass_context.rs
use ash::vk;
use crate::render::resource::{ResourceHandle, ResourceDesc, QueueType};

pub struct PassBuilder {
    pub name: &'static str,
    pub queue_type: QueueType,
    pub reads: Vec<ResourceHandle>,
    pub writes: Vec<ResourceHandle>,
    pub(crate) next_resource_id: u32,
}

impl PassBuilder {
    pub fn new(name: &'static str, queue_type: QueueType, next_id: u32) -> Self {
        Self {
            name,
            queue_type,
            reads: Vec::new(),
            writes: Vec::new(),
            next_resource_id: next_id,
        }
    }

    pub fn read(&mut self, handle: ResourceHandle) {
        self.reads.push(handle);
    }

    pub fn create_image(
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
        self.writes.push(handle);
        handle
    }

    pub fn write(&mut self, handle: ResourceHandle) -> ResourceHandle {
        let new = ResourceHandle {
            id: handle.id,
            version: handle.version + 1,
        };
        self.writes.push(new);
        new
    }
}

pub struct PassContext<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub frame_index: u64,
}
```

- [ ] **Step 3: Implement RenderGraph**

```rust
// src/render/graph.rs
use anyhow::Result;
use ash::vk;

use crate::render::pass_context::{PassBuilder, PassContext};
use crate::render::resource::{PassDecl, QueueType, ResourceHandle};

// Lifetime 'a allows closures to borrow from persistent state (e.g. passes owned by the app).
// A new RenderGraph is built each frame, so 'a is the frame's borrow scope.
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
        // Topological sort based on read/write dependencies.
        // A pass P2 depends on P1 if P2 reads a resource that P1 writes.
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
```

- [ ] **Step 4: Add modules to mod.rs**

In `src/render/mod.rs`, add:
```rust
pub mod allocator;
pub mod resource;
pub mod pass_context;
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 6: Commit**

```
git add src/render/graph.rs src/render/resource.rs src/render/pass_context.rs src/render/mod.rs
git commit -m "feat(render): implement data-driven render graph with topological sort"
```

---

## Task 7: Render Graph Unit Tests

**Files:**
- Create: `tests/render_graph_test.rs`

- [ ] **Step 1: Write tests for dependency sorting**

```rust
// tests/render_graph_test.rs
// NOTE: These test the graph logic without Vulkan — we only test ordering.
// We need to make PassBuilder and graph testable without a real device.
// For now, test the topological sort logic in isolation.

#[test]
fn test_empty_graph_compiles() {
    // Verify an empty graph can compile and has 0 passes
    use revolumetric::render::graph::RenderGraph;
    let mut graph = RenderGraph::new();
    graph.compile();
    assert_eq!(graph.pass_count(), 0);
}
```

Note: Full render graph tests with mock device will be added in Phase 10 (Polish). For now we verify compilation and basic logic.

- [ ] **Step 2: Run test**

Run: `cargo test --test render_graph_test -- --nocapture`
Expected: PASS

- [ ] **Step 3: Commit**

```
git add tests/render_graph_test.rs
git commit -m "test(render): add basic render graph unit test"
```

---

## Task 8: Slang Shader Compilation (Build Script)

**Files:**
- Create: `build.rs`
- Modify: `Cargo.toml`

- [ ] **Step 1: Add build dependencies**

In `Cargo.toml`:
```toml
[build-dependencies]
walkdir = "2"
```

- [ ] **Step 2: Create build.rs that compiles .slang → .spv using slangc CLI**

```rust
// build.rs
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let shader_dir = Path::new("assets/shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("shaders");
    std::fs::create_dir_all(&out_dir).unwrap();

    println!("cargo:rerun-if-changed=assets/shaders");

    // Find all .slang files in passes/
    let passes_dir = shader_dir.join("passes");
    if !passes_dir.exists() {
        return;
    }

    for entry in walkdir::WalkDir::new(&passes_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "slang"))
    {
        let path = entry.path();
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let spv_path = out_dir.join(format!("{stem}.spv"));

        let status = Command::new("slangc")
            .arg(path)
            .arg("-target").arg("spirv")
            .arg("-entry").arg("main")
            .arg("-stage").arg("compute")
            .arg("-o").arg(&spv_path)
            .arg("-I").arg(shader_dir.join("shared"))
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Compiled {}", path.display());
            }
            Ok(s) => {
                panic!("slangc failed for {} with exit code {:?}", path.display(), s.code());
            }
            Err(e) => {
                println!("cargo:warning=slangc not found ({e}), skipping shader compilation");
                // Write a placeholder so the build doesn't fail when slangc isn't installed
                std::fs::write(&spv_path, &[]).unwrap();
                break;
            }
        }
    }
}
```

- [ ] **Step 3: Verify build script runs**

Run: `cargo build 2>&1 | tail -10`
Expected: Either shader compilation warnings or "slangc not found" warning (both OK for now)

- [ ] **Step 4: Commit**

```
git add build.rs Cargo.toml
git commit -m "feat(build): add Slang → SPIR-V compilation build script"
```

---

## Task 9: Test Pattern Compute Shader

**Files:**
- Create: `assets/shaders/passes/test_pattern.slang`

- [ ] **Step 1: Write the test pattern shader**

```slang
// assets/shaders/passes/test_pattern.slang
// A simple compute shader that writes a gradient pattern to a storage image.
// Proves the entire pipeline: Slang compile → SPIR-V → compute dispatch → image write.

struct PushConstants {
    float time;
    uint width;
    uint height;
    uint _pad;
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]]
RWTexture2D<float4> output_image;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.width || tid.y >= pc.height) return;

    float2 uv = float2(tid.xy) / float2(pc.width, pc.height);

    float r = 0.5 + 0.5 * sin(uv.x * 6.28 + pc.time);
    float g = 0.5 + 0.5 * sin(uv.y * 6.28 + pc.time * 1.3);
    float b = 0.5 + 0.5 * sin((uv.x + uv.y) * 3.14 + pc.time * 0.7);

    output_image[tid.xy] = float4(r, g, b, 1.0);
}
```

- [ ] **Step 2: Verify slangc can compile it (if installed)**

Run: `slangc assets/shaders/passes/test_pattern.slang -target spirv -entry main -stage compute -o /tmp/test_pattern.spv 2>&1 || echo "slangc not installed yet — OK"`

- [ ] **Step 3: Commit**

```
git add assets/shaders/passes/test_pattern.slang
git commit -m "feat(shader): add test pattern compute shader in Slang"
```

---

## Task 10a: Refactor RenderDevice begin/end Frame

**Files:**
- Modify: `src/render/device.rs`
- Modify: `src/render/frame.rs`

- [ ] **Step 1: Extend FrameContext**

In `src/render/frame.rs`:
```rust
pub struct FrameContext {
    pub frame_index: u64,
    pub should_render: bool,
    pub command_buffer: vk::CommandBuffer,
    pub swapchain_image: vk::Image,
    pub swapchain_image_index: usize,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_format: vk::Format,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
}
```

- [ ] **Step 2: Split render_frame into begin_frame + end_frame**

In `src/render/device.rs`, replace `render_frame()` with:

`begin_frame() -> Result<FrameContext>`:
- Wait for in-flight fence
- Reset fence and command pool
- Acquire swapchain image
- Begin command buffer
- Return FrameContext with command_buffer, swapchain_image, semaphores, fence
- If swapchain out-of-date, recreate and return `should_render: false`

`end_frame(ctx: FrameContext) -> Result<()>`:
- End command buffer
- Submit to graphics queue with wait/signal semaphores and fence
- Present

Key: move the clear-color and layout transition logic OUT of device.rs — that will be handled by render passes.

- [ ] **Step 3: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 4: Commit**

```
git add src/render/device.rs src/render/frame.rs
git commit -m "refactor(render): split render_frame into begin_frame/end_frame"
```

---

## Task 10b: Test Pattern Compute Pass

**Files:**
- Create: `src/render/passes/test_pattern.rs`

- [ ] **Step 1: Implement the test pattern pass**

```rust
// src/render/passes/test_pattern.rs
use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;

use crate::render::allocator::GpuAllocator;
use crate::render::descriptor::{DescriptorLayoutBuilder, DescriptorPool};
use crate::render::image::{GpuImage, GpuImageDesc};
use crate::render::pipeline::{create_shader_module, ComputePipeline};

#[repr(C)]
#[derive(Clone, Copy)]
struct TestPatternPushConstants {
    time: f32,
    width: u32,
    height: u32,
    _pad: u32,
}

pub struct TestPatternPass {
    pipeline: ComputePipeline,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    pub output_image: GpuImage,
}

impl TestPatternPass {
    pub fn new(
        device: &ash::Device,
        allocator: &GpuAllocator,
        width: u32,
        height: u32,
        spirv_bytes: &[u8],
    ) -> Result<Self> {
        // Descriptor set layout: binding 0 = storage image
        let descriptor_set_layout = DescriptorLayoutBuilder::new()
            .add_binding(
                0,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ShaderStageFlags::COMPUTE,
                1,
            )
            .build(device)?;

        // Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            descriptor_count: 1,
        }];
        let descriptor_pool = DescriptorPool::new(device, 1, &pool_sizes)?;
        let descriptor_set = descriptor_pool.allocate(device, &[descriptor_set_layout])?[0];

        // Output image (RGBA8, storage + transfer src for blit)
        let output_image = GpuImage::new(
            device,
            allocator,
            &GpuImageDesc {
                width,
                height,
                depth: 1,
                format: vk::Format::R8G8B8A8_UNORM,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                aspect: vk::ImageAspectFlags::COLOR,
                name: "test_pattern_output",
            },
        )?;

        // Update descriptor set to point to output image
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(output_image.view)
            .image_layout(vk::ImageLayout::GENERAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(std::slice::from_ref(&image_info));
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        // Pipeline
        let shader_module = create_shader_module(device, spirv_bytes)?;
        let push_constant_ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<TestPatternPushConstants>() as u32,
        }];
        let pipeline = ComputePipeline::new(
            device,
            shader_module,
            // Safety: null-terminated
            unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") },
            &[descriptor_set_layout],
            &push_constant_ranges,
        )?;
        unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self {
            pipeline,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            output_image,
        })
    }

    pub fn record(&self, device: &ash::Device, cmd: vk::CommandBuffer, time: f32) {
        let extent = self.output_image.extent;

        // Transition output image to GENERAL for compute write
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .image(self.output_image.handle)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[], &[], &[barrier],
            );
        }

        // Bind pipeline and descriptor set
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline.handle);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout,
                0,
                &[self.descriptor_set],
                &[],
            );
        }

        // Push constants
        let pc = TestPatternPushConstants {
            time,
            width: extent.width,
            height: extent.height,
            _pad: 0,
        };
        unsafe {
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<TestPatternPushConstants>(),
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );
        }

        // Dispatch workgroups (8x8 threads per group)
        let groups_x = (extent.width + 7) / 8;
        let groups_y = (extent.height + 7) / 8;
        unsafe { device.cmd_dispatch(cmd, groups_x, groups_y, 1) };
    }

    pub fn destroy(self, device: &ash::Device, allocator: &GpuAllocator) {
        self.pipeline.destroy(device);
        self.descriptor_pool.destroy(device);
        unsafe { device.destroy_descriptor_set_layout(self.descriptor_set_layout, None) };
        self.output_image.destroy(device, allocator);
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```
git add src/render/passes/test_pattern.rs src/render/passes/mod.rs
git commit -m "feat(render): implement test pattern compute pass"
```

---

## Task 10c: Blit-to-Swapchain Pass

**Files:**
- Create: `src/render/passes/blit_to_swapchain.rs`

- [ ] **Step 1: Implement blit pass**

```rust
// src/render/passes/blit_to_swapchain.rs
use ash::vk;

/// Records commands to blit a storage image to a swapchain image.
/// Handles all layout transitions.
pub fn record_blit(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src_image: vk::Image,
    src_extent: vk::Extent3D,
    dst_image: vk::Image,
    dst_extent: vk::Extent2D,
) {
    // Transition src: GENERAL → TRANSFER_SRC
    let src_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
        .image(src_image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );

    // Transition dst: UNDEFINED → TRANSFER_DST
    let dst_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .image(dst_image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[], &[], &[src_barrier, dst_barrier],
        );
    }

    // Blit
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

    // Transition dst: TRANSFER_DST → PRESENT_SRC
    let present_barrier = vk::ImageMemoryBarrier::default()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::empty())
        .image(dst_image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );

    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[], &[], &[present_barrier],
        );
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo build 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```
git add src/render/passes/blit_to_swapchain.rs src/render/passes/mod.rs
git commit -m "feat(render): implement blit-to-swapchain pass with layout transitions"
```

---

## Task 10d: Wire Up App Loop

**Files:**
- Modify: `src/app.rs`
- Modify: `src/render/passes/mod.rs`

- [ ] **Step 1: Update passes/mod.rs**

```rust
pub mod blit_to_swapchain;
pub mod composite;
pub mod debug_views;
pub mod radiance_cascade_merge;
pub mod radiance_cascade_trace;
pub mod shadow_trace;
pub mod test_pattern;
pub mod voxel_upload;
```

- [ ] **Step 2: Add TestPatternPass to RevolumetricApp**

In `src/app.rs`, add a `test_pattern_pass: Option<TestPatternPass>` field. Initialize it in `resumed()` after the renderer is created, loading the SPIR-V from the build output:

```rust
let spirv = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/test_pattern.spv"));
let test_pattern = TestPatternPass::new(
    renderer.device(),
    renderer.allocator(),
    extent.width,
    extent.height,
    spirv,
)?;
```

- [ ] **Step 3: Replace tick_frame with graph-based rendering**

Route both passes through the render graph so that `compile()` and `execute()` are validated end-to-end — not just unit-tested in isolation.

```rust
use crate::render::graph::RenderGraph;
use crate::render::resource::QueueType;
use crate::render::passes::blit_to_swapchain;

fn tick_frame(&mut self) -> Result<()> {
    self.schedule.run_stage(Stage::PreUpdate, &mut self.world)?;
    self.schedule.run_stage(Stage::Update, &mut self.world)?;
    self.schedule.run_stage(Stage::PostUpdate, &mut self.world)?;

    if let Some(renderer) = self.renderer.as_mut() {
        let frame = renderer.begin_frame()?;
        if frame.should_render {
            let time = self.world.resource::<Time>()
                .map(|t| t.elapsed())
                .unwrap_or(0.0);

            let mut graph = RenderGraph::new();

            if let Some(pass) = &self.test_pattern_pass {
                // Pass 1: test pattern compute → writes output_image
                // The closure borrows `pass` by reference — RenderGraph<'a>
                // lifetime allows this since the graph lives within this frame.
                let output_extent = pass.output_image.extent;
                let output_img = pass.output_image.handle;

                let test_pattern_writes = graph.add_pass(
                    "test_pattern",
                    QueueType::Compute,
                    |builder| {
                        let _output = builder.create_image(
                            frame.swapchain_extent.width,
                            frame.swapchain_extent.height,
                            vk::Format::R8G8B8A8_UNORM,
                            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                        );
                        Box::new(move |ctx| {
                            // Record directly using captured Vulkan handles
                            pass.record(ctx.device, ctx.command_buffer, time);
                        })
                    },
                );

                // Pass 2: blit to swapchain → reads the test pattern output
                let src_image = output_img;
                let src_extent = output_extent;
                let dst_image = frame.swapchain_image;
                let dst_extent = frame.swapchain_extent;
                // Declare read dependency on the first write handle from test_pattern
                let dep_handle = test_pattern_writes[0];
                graph.add_pass(
                    "blit_to_swapchain",
                    QueueType::Graphics,
                    |builder| {
                        builder.read(dep_handle);
                        Box::new(move |ctx| {
                            blit_to_swapchain::record_blit(
                                ctx.device,
                                ctx.command_buffer,
                                src_image,
                                src_extent,
                                dst_image,
                                dst_extent,
                            );
                        })
                    },
                );
            }

            graph.compile();
            graph.execute(renderer.device(), frame.command_buffer, frame.frame_index);
            renderer.end_frame(frame)?;
        }
    }
    Ok(())
}
```

- [ ] **Step 4: Verify the window shows a rainbow gradient**

Run: `RUST_LOG=info cargo run`
Expected: Window opens with an animated rainbow gradient pattern. No crashes.

- [ ] **Step 5: Commit**

```
git add src/app.rs src/render/passes/mod.rs
git commit -m "feat: end-to-end render graph with test pattern compute shader"
```

---

## Completion Criteria

When all 10 tasks are done, Phase 1 delivers:

1. **GPU resource management** — `GpuBuffer`, `GpuImage`, `GpuAllocator`, descriptors, compute pipelines
2. **Data-driven render graph** — topological sort, pass declaration, execution
3. **Slang build pipeline** — `build.rs` compiles `.slang` → `.spv`
4. **End-to-end proof** — animated test pattern rendered via compute shader → blit → swapchain
5. **Clean foundation** for Phase 2 (UCVH data upload) and Phase 3 (voxel ray tracing)

Next: Phase 2 plan (`2026-04-05-phase2-ucvh-core.md`) — Brick pool, occupancy hierarchy, voxel upload pass, procedural demo scene.
