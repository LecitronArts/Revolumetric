# Teardown Static Vox Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a one-command best-effort converter that turns a Teardown workshop zip into a static `.vox` using only resources present in the zip.

**Architecture:** Add a standalone Python tool under `tools/` with a tiny MagicaVoxel reader/writer, XML traversal, missing-resource reporting, and chunked scene export. Add unittest coverage for VOX parsing/writing and XML resource classification.

**Tech Stack:** Python standard library only: `argparse`, `zipfile`, `xml.etree.ElementTree`, `struct`, `unittest`.

---

### Task 1: Tests

**Files:**
- Create: `tools/test_teardown_static_vox_export.py`

- [ ] Write failing unittest cases for minimal `.vox` parsing, scene writer roundtrip header/chunks, and Teardown path resolution.
- [ ] Run `python -m unittest tools.test_teardown_static_vox_export -v` and confirm failure because the tool does not exist.

### Task 2: Converter

**Files:**
- Create: `tools/teardown_static_vox_export.py`

- [ ] Implement MagicaVoxel `SIZE`/`XYZI`/`RGBA` parsing.
- [ ] Implement chunked `.vox` writing with `PACK`, model chunks, and a scene graph containing translated chunk nodes.
- [ ] Implement zip extraction lookup for `MOD/...` paths and skip/report `BUILT-IN/...`.
- [ ] Implement XML traversal for `group`, `body`, `vox`, and axis-aligned `voxbox`.
- [ ] Implement CLI arguments for input zip, output path, report path, and voxels-per-unit.

### Task 3: Verification

**Files:**
- Modify only generated outputs under `run/` if requested by command output.

- [ ] Run unit tests.
- [ ] Run converter against `C:\Users\mc897\Downloads\Vintessa Hills.zip`.
- [ ] Confirm output `.vox` and JSON report are created.
- [ ] Inspect report counts and call out missing `BUILT-IN` resources and approximation limits.
