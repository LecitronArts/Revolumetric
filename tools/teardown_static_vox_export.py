#!/usr/bin/env python3
"""Best-effort Teardown workshop zip to static MagicaVoxel .vox exporter.

This uses MOD assets present in the workshop zip and can resolve Teardown
`BUILT-IN/...` assets from a local Teardown install.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path


CHUNK_SIZE = 256
TEARDOWN_VOXELS_PER_UNIT = 10
MAGICAVOXEL_CUBE_LEVELS = (255, 204, 153, 102, 51, 0)
MAGICAVOXEL_RAMP_LEVELS = (238, 221, 187, 170, 136, 119, 85, 68, 34, 17)
TRANSFORM_TAGS = {
    "scene",
    "prefab",
    "group",
    "body",
    "compound",
    "vehicle",
    "wheel",
    "shape",
    "joint",
    "rope",
    "script",
    "screen",
    "trigger",
    "boundary",
    "location",
    "light",
    "spawnpoint",
    "water",
    "vox",
    "voxbox",
    "voxagon",
    "voxscript",
    "instance",
}
DEFAULT_TEARDOWN_DIR_CANDIDATES = (
    r"D:\SteamLibrary\steamapps\common\Teardown",
    r"D:\Steam\steamapps\common\Teardown",
    r"D:\Games\SteamLibrary\steamapps\common\Teardown",
    r"C:\Program Files (x86)\Steam\steamapps\common\Teardown",
)


@dataclass(frozen=True)
class VoxModel:
    size: tuple[int, int, int]
    voxels: list[tuple[int, int, int, int]]
    palette: list[tuple[int, int, int, int]]
    names: tuple[str, ...] = ()
    hidden: bool = False


@dataclass(frozen=True)
class VoxTransform:
    rotation: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    translation: tuple[int, int, int]


IDENTITY_TRANSFORM = VoxTransform(
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    (0, 0, 0),
)


def default_palette() -> list[tuple[int, int, int, int]]:
    palette = [(0, 0, 0, 0)]
    for r in MAGICAVOXEL_CUBE_LEVELS:
        for g in MAGICAVOXEL_CUBE_LEVELS:
            for b in MAGICAVOXEL_CUBE_LEVELS:
                if (r, g, b) != (0, 0, 0):
                    palette.append((r, g, b, 255))
    for v in MAGICAVOXEL_RAMP_LEVELS:
        palette.append((v, 0, 0, 255))
    for v in MAGICAVOXEL_RAMP_LEVELS:
        palette.append((0, v, 0, 255))
    for v in MAGICAVOXEL_RAMP_LEVELS:
        palette.append((0, 0, v, 255))
    for v in MAGICAVOXEL_RAMP_LEVELS:
        palette.append((v, v, v, 255))
    assert len(palette) == 256
    return palette


class PaletteBuilder:
    def __init__(self):
        self.colors: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)]
        self._lookup: dict[tuple[int, int, int, int], int] = {(0, 0, 0, 0): 0}

    def index(self, color: tuple[int, int, int, int]) -> int:
        color = tuple(max(0, min(255, int(channel))) for channel in color)
        if color in self._lookup:
            return self._lookup[color]
        if len(self.colors) < 256:
            index = len(self.colors)
            self.colors.append(color)
            self._lookup[color] = index
            return index
        return self._nearest(color)

    def _nearest(self, color: tuple[int, int, int, int]) -> int:
        r, g, b, _a = color
        best_index = 1
        best_dist = None
        for index, existing in enumerate(self.colors[1:], start=1):
            er, eg, eb, _ea = existing
            dist = (r - er) * (r - er) + (g - eg) * (g - eg) + (b - eb) * (b - eb)
            if best_dist is None or dist < best_dist:
                best_index = index
                best_dist = dist
        return best_index

    def palette(self) -> list[tuple[int, int, int, int]]:
        return self.colors + default_palette()[len(self.colors) :]


def _read_chunks(data: bytes, start: int, end: int):
    off = start
    while off + 12 <= end:
        chunk_id = data[off : off + 4]
        content_size, children_size = struct.unpack_from("<II", data, off + 4)
        content_start = off + 12
        content_end = content_start + content_size
        children_start = content_end
        if chunk_id == b"MAIN" and children_size == 0xFFFFFFFF:
            children_end = end
        else:
            children_end = children_start + children_size
        if children_end > end:
            raise ValueError(f"Malformed VOX chunk {chunk_id!r}")
        yield chunk_id, data[content_start:content_end], children_start, children_end
        off = children_end


def _read_dict(content: bytes, offset: int) -> tuple[dict[str, str], int]:
    if offset + 4 > len(content):
        raise ValueError("Malformed VOX dictionary")
    count = struct.unpack_from("<I", content, offset)[0]
    offset += 4
    values = {}
    for _ in range(count):
        if offset + 4 > len(content):
            raise ValueError("Malformed VOX dictionary")
        key_len = struct.unpack_from("<I", content, offset)[0]
        offset += 4
        if offset + key_len + 4 > len(content):
            raise ValueError("Malformed VOX dictionary")
        key = content[offset : offset + key_len].decode("utf-8")
        offset += key_len
        value_len = struct.unpack_from("<I", content, offset)[0]
        offset += 4
        if offset + value_len > len(content):
            raise ValueError("Malformed VOX dictionary")
        value = content[offset : offset + value_len].decode("utf-8")
        offset += value_len
        values[key] = value
    return values, offset


def _dict_bool(values: dict[str, str], key: str, default: bool = False) -> bool:
    value = values.get(key)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no"}


def _logical_palette_from_raw_rgba(
    raw: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    fixed = (raw + [(0, 0, 0, 255)] * 256)[:256]
    r, g, b, _a = fixed[255]
    return [(r, g, b, 0)] + fixed[:255]


def _raw_rgba_from_logical_palette(
    palette: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    fixed = (palette + default_palette())[:256]
    return fixed[1:256] + [fixed[0]]


def _apply_imap(
    raw_palette: list[tuple[int, int, int, int]],
    pending_models: list[tuple[tuple[int, int, int], list[tuple[int, int, int, int]]]],
    index_map: list[int],
) -> tuple[
    list[tuple[int, int, int, int]],
    list[tuple[tuple[int, int, int], list[tuple[int, int, int, int]]]],
]:
    if len(index_map) < 256:
        return raw_palette, pending_models

    inverse = [0] * 256
    for display_index, actual_index in enumerate(index_map[:256]):
        inverse[actual_index] = display_index

    remapped_palette = [raw_palette[(index_map[i] + 255) & 0xFF] for i in range(256)]
    remapped_models = []
    for size, voxels in pending_models:
        remapped_voxels = []
        for x, y, z, color in voxels:
            remapped = 0 if color == 0 else 1 + inverse[color]
            remapped_voxels.append((x, y, z, remapped & 0xFF))
        remapped_models.append((size, remapped_voxels))
    return remapped_palette, remapped_models


def _rotation_from_packed(value: str | None) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if value is None:
        return IDENTITY_TRANSFORM.rotation
    packed = int(value)
    row0_index = (packed >> 0) & 3
    row1_index = (packed >> 2) & 3
    row2_lookup = {
        (0, 1): 2,
        (1, 0): 2,
        (0, 2): 1,
        (2, 0): 1,
        (1, 2): 0,
        (2, 1): 0,
    }
    row2_index = row2_lookup.get((row0_index, row1_index))
    if row2_index is None:
        return IDENTITY_TRANSFORM.rotation
    rows = [[0, 0, 0] for _ in range(3)]
    for row, index in enumerate((row0_index, row1_index, row2_index)):
        rows[row][index] = -1 if packed & (1 << (4 + row)) else 1
    return tuple(tuple(row) for row in rows)  # type: ignore[return-value]


def _parse_translation(value: str | None) -> tuple[int, int, int]:
    if not value:
        return (0, 0, 0)
    parts = [int(float(part)) for part in value.split()]
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    return (0, 0, 0)


def _mat3_vec(
    matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    vector: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3))  # type: ignore[return-value]


def _mat3_mul(
    a: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
    b: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _swap_yz_point(point: tuple[int, int, int]) -> tuple[int, int, int]:
    return (point[0], point[2], point[1])


def _swap_yz_rotation(
    rotation: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    swap_yz = ((1, 0, 0), (0, 0, 1), (0, 1, 0))
    return _mat3_mul(_mat3_mul(swap_yz, rotation), swap_yz)


def _compose_transform(parent: VoxTransform, child: VoxTransform) -> VoxTransform:
    rotated_translation = _mat3_vec(parent.rotation, child.translation)
    return VoxTransform(
        _mat3_mul(parent.rotation, child.rotation),
        tuple(parent.translation[i] + rotated_translation[i] for i in range(3)),  # type: ignore[arg-type]
    )


def _read_transform_node(content: bytes) -> tuple[int, int, VoxTransform, str | None, int, bool]:
    node_id = struct.unpack_from("<I", content, 0)[0]
    attrs, offset = _read_dict(content, 4)
    if offset + 16 > len(content):
        raise ValueError("Malformed nTRN chunk")
    child_id, _reserved_id, layer_id, frame_count = struct.unpack_from("<iiiI", content, offset)
    offset += 16
    transform = IDENTITY_TRANSFORM
    for frame_index in range(frame_count):
        frame, offset = _read_dict(content, offset)
        if frame_index == 0:
            transform = VoxTransform(
                _swap_yz_rotation(_rotation_from_packed(frame.get("_r"))),
                _swap_yz_point(_parse_translation(frame.get("_t"))),
            )
    return node_id, child_id, transform, attrs.get("_name"), layer_id, _dict_bool(attrs, "_hidden")


def _read_group_node(content: bytes) -> tuple[int, list[int], bool]:
    node_id = struct.unpack_from("<I", content, 0)[0]
    attrs, offset = _read_dict(content, 4)
    child_count = struct.unpack_from("<I", content, offset)[0]
    offset += 4
    children = []
    for _ in range(child_count):
        children.append(struct.unpack_from("<I", content, offset)[0])
        offset += 4
    return node_id, children, _dict_bool(attrs, "_hidden")


def _read_layer_node(content: bytes) -> tuple[int, bool]:
    if len(content) < 12:
        raise ValueError("Malformed LAYR chunk")
    layer_id = struct.unpack_from("<i", content, 0)[0]
    attrs, _offset = _read_dict(content, 4)
    return layer_id, _dict_bool(attrs, "_hidden")


def _read_shape_node(content: bytes) -> tuple[int, list[int]]:
    node_id = struct.unpack_from("<I", content, 0)[0]
    _attrs, offset = _read_dict(content, 4)
    model_count = struct.unpack_from("<I", content, offset)[0]
    offset += 4
    model_ids = []
    for _ in range(model_count):
        model_ids.append(struct.unpack_from("<I", content, offset)[0])
        offset += 4
        _model_attrs, offset = _read_dict(content, offset)
    return node_id, model_ids


def _collect_vox_instances(
    model_count: int,
    transforms: dict[int, tuple[int, VoxTransform, str | None, int, bool]],
    groups: dict[int, tuple[list[int], bool]],
    shapes: dict[int, list[int]],
    referenced_children: set[int],
    hidden_layers: set[int],
) -> list[tuple[int, VoxTransform, tuple[str, ...], bool]]:
    if not shapes:
        return [(model_index, IDENTITY_TRANSFORM, (), False) for model_index in range(model_count)]
    roots = sorted((set(transforms) | set(groups) | set(shapes)) - referenced_children)
    instances: list[tuple[int, VoxTransform, tuple[str, ...], bool]] = []

    def visit(node_id: int, parent: VoxTransform, names: tuple[str, ...], hidden: bool) -> None:
        if node_id in transforms:
            child_id, transform, name, layer_id, node_hidden = transforms[node_id]
            next_names = names + ((name,) if name else ())
            next_hidden = hidden or node_hidden or layer_id in hidden_layers
            visit(child_id, _compose_transform(parent, transform), next_names, next_hidden)
            return
        if node_id in groups:
            children, group_hidden = groups[node_id]
            for child in children:
                visit(child, parent, names, hidden or group_hidden)
            return
        if node_id in shapes:
            for model_index in shapes[node_id]:
                if 0 <= model_index < model_count:
                    instances.append((model_index, parent, names, hidden))

    for root in roots:
        visit(root, IDENTITY_TRANSFORM, (), False)
    return instances


def _transform_model(
    model: VoxModel,
    transform: VoxTransform,
    names: tuple[str, ...] | None = None,
    hidden: bool | None = None,
) -> VoxModel:
    model_hidden = model.hidden if hidden is None else hidden
    if transform == IDENTITY_TRANSFORM:
        if (names is None or model.names == names) and model.hidden == model_hidden:
            return model
        return VoxModel(model.size, model.voxels, model.palette, names or model.names, model_hidden)
    voxels = []
    mins = [0, 0, 0]
    maxs = [0, 0, 0]
    for index, (x, y, z, color) in enumerate(model.voxels):
        rotated = _mat3_vec(transform.rotation, (x, y, z))
        tx, ty, tz = tuple(rotated[i] + transform.translation[i] for i in range(3))
        if index == 0:
            mins = [tx, ty, tz]
            maxs = [tx, ty, tz]
        else:
            mins[0] = min(mins[0], tx)
            mins[1] = min(mins[1], ty)
            mins[2] = min(mins[2], tz)
            maxs[0] = max(maxs[0], tx)
            maxs[1] = max(maxs[1], ty)
            maxs[2] = max(maxs[2], tz)
        voxels.append((tx, ty, tz, color))
    size = (maxs[0] - mins[0] + 1, maxs[1] - mins[1] + 1, maxs[2] - mins[2] + 1) if voxels else model.size
    return VoxModel(size, voxels, model.palette, names or model.names, model_hidden)


def read_vox_models(
    data: bytes,
    object_name: str | None = None,
    include_hidden: bool = False,
) -> list[VoxModel]:
    if len(data) < 20 or data[:4] != b"VOX ":
        raise ValueError("Not a MagicaVoxel VOX file")

    models: list[VoxModel] = []
    palette = default_palette()
    raw_palette: list[tuple[int, int, int, int]] | None = None
    index_map: list[int] | None = None
    current_size: tuple[int, int, int] | None = None
    main_children: tuple[int, int] | None = None
    transforms: dict[int, tuple[int, VoxTransform, str | None, int, bool]] = {}
    groups: dict[int, tuple[list[int], bool]] = {}
    shapes: dict[int, list[int]] = {}
    referenced_children: set[int] = set()
    hidden_layers: set[int] = set()

    for chunk_id, _content, child_start, child_end in _read_chunks(data, 8, len(data)):
        if chunk_id == b"MAIN":
            main_children = (child_start, child_end)
            break
    if main_children is None:
        raise ValueError("VOX MAIN chunk not found")

    pending_models: list[tuple[tuple[int, int, int], list[tuple[int, int, int, int]]]] = []
    for chunk_id, content, _child_start, _child_end in _read_chunks(data, *main_children):
        if chunk_id == b"SIZE":
            current_size = _swap_yz_point(struct.unpack_from("<III", content, 0))
        elif chunk_id == b"XYZI":
            if current_size is None:
                raise ValueError("XYZI chunk appeared before SIZE")
            count = struct.unpack_from("<I", content, 0)[0]
            voxels = []
            offset = 4
            for _ in range(count):
                x, y, z, color = content[offset : offset + 4]
                sx, sy, sz = _swap_yz_point((x, y, z))
                voxels.append((sx, sy, sz, color))
                offset += 4
            pending_models.append((current_size, voxels))
            current_size = None
        elif chunk_id == b"RGBA":
            raw_palette = []
            for i in range(min(len(content) // 4, 256)):
                raw_palette.append(struct.unpack_from("<BBBB", content, i * 4))
            raw_palette = (raw_palette + _raw_rgba_from_logical_palette(default_palette())[len(raw_palette) :])[:256]
        elif chunk_id == b"IMAP":
            index_map = list(content[:256])
        elif chunk_id == b"nTRN":
            node_id, child_id, transform, name, layer_id, hidden = _read_transform_node(content)
            transforms[node_id] = (child_id, transform, name, layer_id, hidden)
            referenced_children.add(child_id)
        elif chunk_id == b"nGRP":
            node_id, children, hidden = _read_group_node(content)
            groups[node_id] = (children, hidden)
            referenced_children.update(children)
        elif chunk_id == b"nSHP":
            node_id, model_ids = _read_shape_node(content)
            shapes[node_id] = model_ids
        elif chunk_id == b"LAYR":
            layer_id, hidden = _read_layer_node(content)
            if hidden:
                hidden_layers.add(layer_id)

    if raw_palette is not None:
        if index_map is not None:
            raw_palette, pending_models = _apply_imap(raw_palette, pending_models, index_map)
        palette = _logical_palette_from_raw_rgba(raw_palette)

    models = [VoxModel(size, voxels, palette) for size, voxels in pending_models]
    instances = _collect_vox_instances(
        len(models),
        transforms,
        groups,
        shapes,
        referenced_children,
        hidden_layers,
    )
    if instances and (shapes or transforms or groups):
        transformed = [
            _transform_model(models[model_index], transform, names, hidden)
            for model_index, transform, names, hidden in instances
            if object_name is None or object_name in names
        ]
        if not include_hidden:
            transformed = [model for model in transformed if not model.hidden]
        return transformed
    if object_name is not None:
        return []
    return models


def _chunk(chunk_id: bytes, content: bytes = b"", children: bytes = b"") -> bytes:
    return chunk_id + struct.pack("<II", len(content), len(children)) + content + children


def _dict(values: dict[str, str] | None = None) -> bytes:
    values = values or {}
    out = struct.pack("<I", len(values))
    for key, value in values.items():
        key_b = key.encode("utf-8")
        value_b = value.encode("utf-8")
        out += struct.pack("<I", len(key_b)) + key_b
        out += struct.pack("<I", len(value_b)) + value_b
    return out


def _pack_content(count: int) -> bytes:
    return struct.pack("<I", count)


def _size_content(size: tuple[int, int, int]) -> bytes:
    return struct.pack("<III", *_swap_yz_point(size))


def _xyzi_content(voxels: list[tuple[int, int, int, int]]) -> bytes:
    out = struct.pack("<I", len(voxels))
    out += bytes(
        value
        for x, y, z, color in voxels
        for value in (*_swap_yz_point((x, y, z)), color)
    )
    return out


def _rgba_content(palette: list[tuple[int, int, int, int]]) -> bytes:
    entries = _raw_rgba_from_logical_palette(palette)
    return bytes([channel for color in entries for channel in color])


def _ntrn(node_id: int, child_id: int, translation: tuple[int, int, int] | None = None) -> bytes:
    attrs = {}
    frame_attrs = {"_f": "0"}
    if translation is not None:
        tx, ty, tz = _swap_yz_point(translation)
        frame_attrs["_t"] = f"{tx} {ty} {tz}"
    content = struct.pack("<I", node_id)
    content += _dict(attrs)
    content += struct.pack("<IiiI", child_id, -1, -1, 1)
    content += _dict(frame_attrs)
    return _chunk(b"nTRN", content)


def _ngrp(node_id: int, children: list[int]) -> bytes:
    content = struct.pack("<I", node_id) + _dict()
    content += struct.pack("<I", len(children))
    for child in children:
        content += struct.pack("<I", child)
    return _chunk(b"nGRP", content)


def _nshp(node_id: int, model_id: int) -> bytes:
    content = struct.pack("<I", node_id) + _dict()
    content += struct.pack("<I", 1)
    content += struct.pack("<I", model_id) + _dict({"_f": "0"})
    return _chunk(b"nSHP", content)


def write_chunked_vox(
    chunks: dict[tuple[int, int, int], list[tuple[int, int, int, int]]],
    palette: list[tuple[int, int, int, int]] | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> bytes:
    non_empty = []
    for key, voxels in sorted(chunks.items()):
        solid_voxels = [voxel for voxel in voxels if voxel[3] != 0]
        if solid_voxels:
            non_empty.append((key, solid_voxels))
    children = _chunk(b"PACK", _pack_content(len(non_empty)))

    for _key, voxels in non_empty:
        max_x = max(v[0] for v in voxels) + 1
        max_y = max(v[1] for v in voxels) + 1
        max_z = max(v[2] for v in voxels) + 1
        size = (min(chunk_size, max_x), min(chunk_size, max_y), min(chunk_size, max_z))
        children += _chunk(b"SIZE", _size_content(size))
        children += _chunk(b"XYZI", _xyzi_content(voxels))

    root_transform = 0
    root_group = 1
    children += _ntrn(root_transform, root_group, (0, 0, 0))
    node_ids = []
    next_node_id = 2
    for model_id, (chunk_key, _voxels) in enumerate(non_empty):
        transform_id = next_node_id
        shape_id = next_node_id + 1
        next_node_id += 2
        node_ids.append(transform_id)
        translation = tuple(axis * chunk_size for axis in chunk_key)
        children += _ntrn(transform_id, shape_id, translation)
        children += _nshp(shape_id, model_id)
    children += _ngrp(root_group, node_ids)
    children += _chunk(b"LAYR", struct.pack("<I", 0) + _dict({"_name": "static"}) + struct.pack("<i", -1))
    children += _chunk(b"RGBA", _rgba_content(palette or default_palette()))

    return b"VOX " + struct.pack("<I", 150) + _chunk(b"MAIN", b"", children)


class ZipMapSource:
    def __init__(self, zip_path: Path, teardown_dir: Path | None = None):
        self.zip_path = Path(zip_path)
        self.teardown_dir = _find_teardown_dir(teardown_dir)
        self.missing_builtin: set[str] = set()
        self.missing_mod: set[str] = set()
        self._zip = zipfile.ZipFile(self.zip_path, "r")
        try:
            names = [name for name in self._zip.namelist() if not name.endswith("/")]
            self.root = self._detect_root(names)
            self._lookup = {name.lower().replace("\\", "/"): name for name in names}
        except Exception:
            self._zip.close()
            raise

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> "ZipMapSource":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    @staticmethod
    def _detect_root(names: list[str]) -> str:
        for name in names:
            if name.lower() == "main.xml":
                return ""
            if name.lower().endswith("/main.xml"):
                return name.rsplit("/", 1)[0]
        raise FileNotFoundError("main.xml not found in workshop zip")

    def read_main_xml(self) -> bytes:
        if not self.root:
            return self._zip.read("main.xml")
        return self._zip.read(f"{self.root}/main.xml")

    def read_mod_file(self, path: str) -> bytes | None:
        normalized = path.replace("\\", "/")
        if normalized.startswith("BUILT-IN/"):
            return self._read_builtin_file(normalized)
        if not normalized.startswith("MOD/"):
            self.missing_mod.add(normalized)
            return None
        rel = normalized[4:]
        zip_key = rel if not self.root else f"{self.root}/{rel}"
        zip_name = self._lookup.get(zip_key.lower())
        if zip_name is None:
            self.missing_mod.add(normalized)
            return None
        return self._zip.read(zip_name)

    def _read_builtin_file(self, normalized: str) -> bytes | None:
        if self.teardown_dir is None:
            self.missing_builtin.add(normalized)
            return None
        rel = normalized[len("BUILT-IN/") :]
        candidate = _safe_join(self.teardown_dir / "data" / "built-in", rel)
        if candidate is not None and candidate.is_file():
            return candidate.read_bytes()
        self.missing_builtin.add(normalized)
        return None


def _find_teardown_dir(explicit: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    env_dir = os.environ.get("TEARDOWN_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(Path(path) for path in DEFAULT_TEARDOWN_DIR_CANDIDATES)

    for candidate in candidates:
        data_dir = candidate / "data" / "built-in"
        if data_dir.is_dir():
            return candidate
    return None


def _safe_join(base: Path, rel: str) -> Path | None:
    base_resolved = base.resolve()
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        return None
    return candidate


def _parse_vec(value: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not value:
        return default
    parts = [float(part) for part in value.split()]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) >= 3:
        return (parts[0], parts[1], parts[2])
    return default


def _mat_mul(a, b):
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4))
        for row in range(4)
    )


def _translation(pos: tuple[float, float, float]):
    return (
        (1.0, 0.0, 0.0, pos[0]),
        (0.0, 1.0, 0.0, pos[1]),
        (0.0, 0.0, 1.0, pos[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


def _rotation_xyz(rot: tuple[float, float, float]):
    rx, ry, rz = [math.radians(v) for v in rot]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = ((1, 0, 0, 0), (0, cx, -sx, 0), (0, sx, cx, 0), (0, 0, 0, 1))
    my = ((cy, 0, sy, 0), (0, 1, 0, 0), (-sy, 0, cy, 0), (0, 0, 0, 1))
    mz = ((cz, -sz, 0, 0), (sz, cz, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    return _mat_mul(_mat_mul(mz, my), mx)


def _transform_point(matrix, point: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def _node_transform(node: ET.Element):
    pos = _parse_vec(node.get("pos"), (0.0, 0.0, 0.0))
    rot = _parse_vec(node.get("rot"), (0.0, 0.0, 0.0))
    return _mat_mul(_translation(pos), _rotation_xyz(rot))


def _parse_color(value: str | None, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if not value:
        return default
    parts = [float(match) for match in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", value)]
    if len(parts) < 3:
        return default
    if max(parts[:3]) <= 1.0:
        return (
            round(parts[0] * 255),
            round(parts[1] * 255),
            round(parts[2] * 255),
            255,
        )
    return (
        round(parts[0]),
        round(parts[1]),
        round(parts[2]),
        255,
    )


def _is_hole_brush(brush: str) -> bool:
    return brush.strip().lower() in {"hole", "hoe"}


def _tint_color(
    color: tuple[int, int, int, int],
    tint: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if tint == (255, 255, 255, 255):
        return color
    return (
        round(color[0] * tint[0] / 255),
        round(color[1] * tint[1] / 255),
        round(color[2] * tint[2] / 255),
        color[3],
    )


def _average_model_color(model: VoxModel) -> tuple[int, int, int, int]:
    if not model.voxels:
        return (160, 160, 160, 255)
    totals = [0, 0, 0]
    count = 0
    for _x, _y, _z, color_index in model.voxels:
        if color_index == 0:
            continue
        color = model.palette[color_index] if color_index < len(model.palette) else (160, 160, 160, 255)
        totals[0] += color[0]
        totals[1] += color[1]
        totals[2] += color[2]
        count += 1
    if count == 0:
        return (160, 160, 160, 255)
    return (round(totals[0] / count), round(totals[1] / count), round(totals[2] / count), 255)


def _average_models_color(models: list[VoxModel]) -> tuple[int, int, int, int]:
    totals = [0, 0, 0]
    count = 0
    for model in models:
        color = _average_model_color(model)
        if color[3] == 0:
            continue
        weight = max(1, len(model.voxels))
        totals[0] += color[0] * weight
        totals[1] += color[1] * weight
        totals[2] += color[2] * weight
        count += weight
    if count == 0:
        return (160, 160, 160, 255)
    return (round(totals[0] / count), round(totals[1] / count), round(totals[2] / count), 255)


def _add_voxel(
    chunks: dict[tuple[int, int, int], dict[tuple[int, int, int], int]],
    point: tuple[float, float, float],
    color: int,
    voxels_per_unit: int,
) -> None:
    wx = _round_voxel_coord(point[0] * voxels_per_unit)
    wy = _round_voxel_coord(point[1] * voxels_per_unit)
    wz = _round_voxel_coord(point[2] * voxels_per_unit)
    chunk_key = (wx // CHUNK_SIZE, wy // CHUNK_SIZE, wz // CHUNK_SIZE)
    local = (wx - chunk_key[0] * CHUNK_SIZE, wy - chunk_key[1] * CHUNK_SIZE, wz - chunk_key[2] * CHUNK_SIZE)
    chunks.setdefault(chunk_key, {})[local] = color


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _round_voxel_coord(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _native_axis_count(size: float, voxels_per_unit: int) -> int:
    scale = voxels_per_unit / TEARDOWN_VOXELS_PER_UNIT
    return max(0, _round_half_up(size * scale))


def _native_voxel_output_range(coord: int, voxels_per_unit: int) -> range:
    scale = voxels_per_unit / TEARDOWN_VOXELS_PER_UNIT
    start = _round_half_up(coord * scale)
    end = _round_half_up((coord + 1) * scale)
    return range(start, max(start + 1, end))


def _is_identity_linear_transform(matrix) -> bool:
    epsilon = 1.0e-9
    for row in range(3):
        for col in range(3):
            expected = 1.0 if row == col else 0.0
            if abs(matrix[row][col] - expected) > epsilon:
                return False
    return True


def _try_add_axis_aligned_box(
    chunks: dict[tuple[int, int, int], dict[tuple[int, int, int], int]],
    matrix,
    size: tuple[float, float, float],
    color: int,
    voxels_per_unit: int,
) -> bool:
    if not _is_identity_linear_transform(matrix):
        return False

    sx, sy, sz = [_native_axis_count(v, voxels_per_unit) for v in size]
    tx = matrix[0][3]
    ty = matrix[1][3]
    tz = matrix[2][3]
    wx_values = [_round_voxel_coord(tx * voxels_per_unit + x) for x in range(sx)]
    wy_values = [_round_voxel_coord(ty * voxels_per_unit + y) for y in range(sy)]
    wz_values = [_round_voxel_coord(tz * voxels_per_unit + z) for z in range(sz)]
    for wx in wx_values:
        chunk_x = wx // CHUNK_SIZE
        local_x = wx - chunk_x * CHUNK_SIZE
        for wy in wy_values:
            chunk_y = wy // CHUNK_SIZE
            local_y = wy - chunk_y * CHUNK_SIZE
            for wz in wz_values:
                chunk_z = wz // CHUNK_SIZE
                chunk_key = (chunk_x, chunk_y, chunk_z)
                local = (local_x, local_y, wz - chunk_z * CHUNK_SIZE)
                chunks.setdefault(chunk_key, {})[local] = color
    return True


def _solid_box_points(size: tuple[float, float, float], voxels_per_unit: int):
    sx, sy, sz = [_native_axis_count(v, voxels_per_unit) for v in size]
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                yield (x / voxels_per_unit, y / voxels_per_unit, z / voxels_per_unit)


def _expanded_native_voxel_points(x: int, y: int, z: int, voxels_per_unit: int):
    for ox in _native_voxel_output_range(x, voxels_per_unit):
        for oy in _native_voxel_output_range(y, voxels_per_unit):
            for oz in _native_voxel_output_range(z, voxels_per_unit):
                yield (ox / voxels_per_unit, oy / voxels_per_unit, oz / voxels_per_unit)


def _point_in_polygon(x: float, z: float, vertices: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(vertices) - 1
    for i, (xi, zi) in enumerate(vertices):
        xj, zj = vertices[j]
        intersects = (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / ((zj - zi) or 1e-12) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


def _voxagon_points(node: ET.Element, voxels_per_unit: int):
    vertices = []
    for child in node:
        if child.tag == "vertex":
            pos = child.get("pos", "")
            parts = [float(part) for part in pos.split()]
            if len(parts) >= 2:
                vertices.append((parts[0], parts[1]))
    if len(vertices) < 3:
        return

    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_z = min(v[1] for v in vertices)
    max_z = max(v[1] for v in vertices)
    extrude = max(1, int(round(float(node.get("extrude", "1")))))
    x0 = math.floor(min_x * voxels_per_unit)
    x1 = math.ceil(max_x * voxels_per_unit)
    z0 = math.floor(min_z * voxels_per_unit)
    z1 = math.ceil(max_z * voxels_per_unit)
    for xi in range(x0, x1):
        x = (xi + 0.5) / voxels_per_unit
        for zi in range(z0, z1):
            z = (zi + 0.5) / voxels_per_unit
            if _point_in_polygon(x, z, vertices):
                for yi in range(extrude):
                    yield (x, yi / voxels_per_unit, z)


def export_static_map(
    zip_path: Path,
    out_path: Path,
    report_path: Path,
    voxels_per_unit: int = 10,
    teardown_dir: Path | None = None,
) -> dict:
    if voxels_per_unit <= 0:
        raise ValueError("voxels_per_unit must be greater than zero")

    resolved_teardown_dir = _find_teardown_dir(teardown_dir)
    stats = {
        "vox_nodes": 0,
        "vox_nodes_exported": 0,
        "vox_object_refs_missing": 0,
        "voxbox_nodes": 0,
        "voxbox_nodes_exported": 0,
        "voxagon_nodes_exported": 0,
        "voxagon_nodes_skipped": 0,
        "instance_nodes": 0,
        "instance_nodes_exported": 0,
        "instance_refs_missing": 0,
        "instance_refs_recursive": 0,
        "instance_refs_malformed": 0,
        "malformed_vox_refs": [],
        "builtin_refs_skipped": 0,
        "mod_refs_missing": 0,
        "output_chunks": 0,
        "output_voxels": 0,
        "teardown_dir": str(resolved_teardown_dir) if resolved_teardown_dir else None,
        "teardown_native_voxels_per_unit": TEARDOWN_VOXELS_PER_UNIT,
        "density_scale": voxels_per_unit / TEARDOWN_VOXELS_PER_UNIT,
        "notes": [
            "Best-effort export: MOD resources inside the zip and local BUILT-IN resources are used.",
            "BUILT-IN Teardown resources are loaded from the local Teardown install when available.",
            "Missing BUILT-IN brush geometry is approximated with XML color or a neutral material.",
            "scripts, lights, joints, water, vehicles, and gameplay entities are not exported.",
        ],
    }
    chunks: dict[tuple[int, int, int], dict[tuple[int, int, int], int]] = {}
    palette_builder = PaletteBuilder()
    model_cache: dict[tuple[str, bool], list[VoxModel] | None] = {}
    brush_color_cache: dict[str, tuple[int, int, int, int]] = {}
    missing_vox_objects: set[str] = set()
    malformed_vox_refs: set[str] = set()

    with ZipMapSource(zip_path, resolved_teardown_dir) as source:
        root = ET.fromstring(source.read_main_xml())

        def read_models_best_effort(
            file_ref: str,
            blob: bytes | None,
            *,
            include_hidden: bool = False,
        ) -> list[VoxModel] | None:
            if blob is None:
                return None
            try:
                return read_vox_models(blob, include_hidden=include_hidden)
            except ValueError:
                malformed_vox_refs.add(file_ref)
                return None

        def walk(node: ET.Element, parent_matrix, instance_stack: tuple[str, ...] = ()) -> None:
            tag = node.tag
            matrix = parent_matrix
            if tag in TRANSFORM_TAGS:
                matrix = _mat_mul(parent_matrix, _node_transform(node))

            if tag == "vox":
                stats["vox_nodes"] += 1
                file_ref = node.get("file")
                if file_ref:
                    object_name = node.get("object")
                    include_hidden = object_name is not None
                    cache_key = (file_ref, include_hidden)
                    if cache_key not in model_cache:
                        blob = source.read_mod_file(file_ref)
                        model_cache[cache_key] = read_models_best_effort(
                            file_ref,
                            blob,
                            include_hidden=include_hidden,
                        )
                    models = model_cache[cache_key]
                    if models:
                        selected_models = [
                            model
                            for model in models
                            if object_name is None or object_name in model.names
                        ]
                        if selected_models:
                            tint = _parse_color(node.get("color"), (255, 255, 255, 255))
                            for model in selected_models:
                                for x, y, z, color in model.voxels:
                                    source_color = model.palette[color] if color < len(model.palette) else (160, 160, 160, 255)
                                    color_index = palette_builder.index(_tint_color(source_color, tint))
                                    for point in _expanded_native_voxel_points(x, y, z, voxels_per_unit):
                                        _add_voxel(chunks, _transform_point(matrix, point), color_index, voxels_per_unit)
                            stats["vox_nodes_exported"] += 1
                        elif object_name:
                            missing_vox_objects.add(f"{file_ref}#{object_name}")
            elif tag == "voxbox":
                stats["voxbox_nodes"] += 1
                brush = node.get("brush", "")
                if _is_hole_brush(brush):
                    pass
                else:
                    size = _parse_vec(node.get("size"), (1.0, 1.0, 1.0))
                    color = _parse_color(node.get("color"), (0, 0, 0, 0))
                    if color == (0, 0, 0, 0):
                        if brush.startswith("MOD/") and brush.endswith(".vox"):
                            if brush not in brush_color_cache:
                                blob = source.read_mod_file(brush)
                                models = read_models_best_effort(brush, blob)
                                brush_color_cache[brush] = _average_models_color(models) if models else (120, 120, 120, 255)
                            color = brush_color_cache[brush]
                        elif brush.startswith("BUILT-IN/") and brush.endswith(".vox"):
                            if brush not in brush_color_cache:
                                blob = source.read_mod_file(brush)
                                models = read_models_best_effort(brush, blob)
                                brush_color_cache[brush] = _average_models_color(models) if models else (155, 155, 150, 255)
                            color = brush_color_cache[brush]
                        elif brush.startswith("BUILT-IN/"):
                            source.read_mod_file(brush)
                            color = (155, 155, 150, 255)
                        else:
                            color = (155, 155, 150, 255)
                    color_index = palette_builder.index(color)
                    if not _try_add_axis_aligned_box(chunks, matrix, size, color_index, voxels_per_unit):
                        for point in _solid_box_points(size, voxels_per_unit):
                            _add_voxel(chunks, _transform_point(matrix, point), color_index, voxels_per_unit)
                    stats["voxbox_nodes_exported"] += 1
            elif tag == "voxagon":
                brush = node.get("brush", "")
                if _is_hole_brush(brush):
                    stats["voxagon_nodes_skipped"] += 1
                else:
                    color = _parse_color(node.get("color"), (0, 0, 0, 0))
                    if color == (0, 0, 0, 0):
                        if brush.startswith("MOD/") and brush.endswith(".vox"):
                            if brush not in brush_color_cache:
                                blob = source.read_mod_file(brush)
                                models = read_models_best_effort(brush, blob)
                                brush_color_cache[brush] = _average_models_color(models) if models else (140, 140, 135, 255)
                            color = brush_color_cache[brush]
                        elif brush.startswith("BUILT-IN/") and brush.endswith(".vox"):
                            if brush not in brush_color_cache:
                                blob = source.read_mod_file(brush)
                                models = read_models_best_effort(brush, blob)
                                brush_color_cache[brush] = _average_models_color(models) if models else (155, 155, 150, 255)
                            color = brush_color_cache[brush]
                        elif brush.startswith("BUILT-IN/"):
                            source.read_mod_file(brush)
                            color = (155, 155, 150, 255)
                        else:
                            color = (155, 155, 150, 255)
                    color_index = palette_builder.index(color)
                    emitted = False
                    for point in _voxagon_points(node, voxels_per_unit):
                        emitted = True
                        _add_voxel(chunks, _transform_point(matrix, point), color_index, voxels_per_unit)
                    if emitted:
                        stats["voxagon_nodes_exported"] += 1
                    else:
                        stats["voxagon_nodes_skipped"] += 1
            elif tag == "instance":
                stats["instance_nodes"] += 1
                file_ref = node.get("file")
                if not file_ref:
                    stats["instance_refs_missing"] += 1
                elif file_ref in instance_stack:
                    stats["instance_refs_recursive"] += 1
                else:
                    blob = source.read_mod_file(file_ref)
                    if blob is None:
                        stats["instance_refs_missing"] += 1
                    else:
                        try:
                            prefab_root = ET.fromstring(blob)
                        except ET.ParseError:
                            stats["instance_refs_malformed"] += 1
                        else:
                            walk(prefab_root, matrix, instance_stack + (file_ref,))
                            stats["instance_nodes_exported"] += 1

            for child in node:
                walk(child, matrix, instance_stack)

        identity = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        walk(root, identity)

        stats["missing_builtin_refs"] = sorted(source.missing_builtin)
        stats["missing_mod_refs"] = sorted(source.missing_mod)
        stats["missing_vox_objects"] = sorted(missing_vox_objects)
        stats["malformed_vox_refs"] = sorted(malformed_vox_refs)
        stats["builtin_refs_skipped"] = len(source.missing_builtin)
        stats["mod_refs_missing"] = len(source.missing_mod)
        stats["vox_object_refs_missing"] = len(missing_vox_objects)

    chunk_lists = {key: [(x, y, z, color) for (x, y, z), color in voxels.items()] for key, voxels in chunks.items()}
    stats["output_chunks"] = len(chunk_lists)
    stats["output_voxels"] = sum(len(voxels) for voxels in chunk_lists.values())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(write_chunked_vox(chunk_lists, palette_builder.palette()))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path, help="Teardown workshop zip")
    parser.add_argument("--out", type=Path, required=True, help="Output .vox path")
    parser.add_argument("--report", type=Path, help="Output JSON report path")
    parser.add_argument(
        "--voxels-per-unit",
        type=int,
        default=10,
        help="Output voxel density per Teardown unit; Teardown native .vox/.voxbox density is 10, so 20 doubles each native axis",
    )
    parser.add_argument(
        "--teardown-dir",
        type=Path,
        help="Teardown install directory used to resolve BUILT-IN/... resources",
    )
    args = parser.parse_args(argv)

    report = args.report or args.out.with_suffix(".report.json")
    stats = export_static_map(args.zip, args.out, report, args.voxels_per_unit, args.teardown_dir)
    print(f"Wrote {args.out}")
    print(f"Wrote {report}")
    print(f"Exported {stats['output_voxels']} voxels in {stats['output_chunks']} chunks")
    print(f"Skipped {stats['builtin_refs_skipped']} BUILT-IN refs and {stats['voxagon_nodes_skipped']} voxagon nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
