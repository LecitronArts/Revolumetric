# Project Hardening Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the prototype's build behavior, GPU upload failure handling, and repository hygiene without changing renderer features.

**Architecture:** Keep changes local to the risk points identified in review: build script policy, UCVH GPU upload staging validation, app upload error handling, and ignore rules for large local research checkouts. Do not refactor the render runtime in this pass.

**Tech Stack:** Rust 2024, Cargo, Vulkan via `ash`, Slang shader compilation through `slangc`.

---

### Task 1: Shader Compile Policy

**Files:**
- Modify: `build.rs`
- Modify: `README.md`

- [x] **Step 1: Add compile policy parsing**

Add `REVOLUMETRIC_SHADER_COMPILE=auto|strict|skip`.

- [x] **Step 2: Preserve local development behavior**

`auto` keeps the existing fallback: compile with `slangc` when present and write empty `.spv` placeholders when missing.

- [x] **Step 3: Add strict and skip modes**

`strict` fails the build if `slangc` is missing or a shader fails to compile. `skip` writes placeholders without invoking `slangc`.

- [x] **Step 4: Document the modes**

Update README requirements and command notes so CI/release users know to use strict mode.

### Task 2: UCVH Upload Failure Handling

**Files:**
- Modify: `src/voxel/gpu_upload.rs`
- Modify: `src/app.rs`

- [x] **Step 1: Add CPU-only tests for staging validation**

Cover unmapped staging buffers and offset/length overflow without requiring a Vulkan device.

- [x] **Step 2: Return upload errors**

Change `UcvhGpuResources::upload_all` to return `anyhow::Result<()>`.

- [x] **Step 3: Keep retry behavior on failure**

Only set `ucvh_uploaded = true` after upload succeeds; log failures and retry on later frames.

### Task 3: Repository Hygiene

**Files:**
- Modify: `.gitignore`

- [x] **Step 1: Ignore large local research checkouts**

Ignore untracked large `reference/` subdirectories that are local research material, while leaving curated tracked reference files available.

### Task 4: Verification

**Files:**
- No direct file edits.

- [x] **Step 1: Format**

Run: `cargo fmt`

- [x] **Step 2: Unit tests**

Run: `cargo test`

- [x] **Step 3: Lints**

Run: `cargo clippy --all-targets -- -D warnings`

- [x] **Step 4: Build**

Run: `cargo build`
