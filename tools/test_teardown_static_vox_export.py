import io
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools import teardown_static_vox_export as conv


def chunk(chunk_id, content=b"", children=b""):
    return chunk_id + struct.pack("<II", len(content), len(children)) + content + children


def dict_bytes(values=None):
    values = values or {}
    out = struct.pack("<I", len(values))
    for key, value in values.items():
        key = key.encode("utf-8")
        value = value.encode("utf-8")
        out += struct.pack("<I", len(key)) + key
        out += struct.pack("<I", len(value)) + value
    return out


def minimal_vox_bytes(rgba_len=256 * 4):
    size = chunk(b"SIZE", struct.pack("<III", 2, 3, 4))
    xyzi = chunk(
        b"XYZI",
        struct.pack("<I", 2) + bytes([0, 1, 2, 7, 1, 2, 3, 8]),
    )
    rgba = chunk(b"RGBA", bytes(i % 256 for i in range(rgba_len)))
    main = chunk(b"MAIN", b"", size + xyzi + rgba)
    return b"VOX " + struct.pack("<I", 150) + main


def minimal_vox_without_rgba_bytes():
    size = chunk(b"SIZE", struct.pack("<III", 2, 3, 4))
    xyzi = chunk(
        b"XYZI",
        struct.pack("<I", 2) + bytes([0, 1, 2, 7, 1, 2, 3, 8]),
    )
    main = chunk(b"MAIN", b"", size + xyzi)
    return b"VOX " + struct.pack("<I", 150) + main


def minimal_vox_with_unsized_main_children():
    size = chunk(b"SIZE", struct.pack("<III", 1, 1, 1))
    xyzi = chunk(b"XYZI", struct.pack("<I", 1) + bytes([0, 0, 0, 1]))
    rgba = chunk(b"RGBA", bytes([10, 20, 30, 255]) + bytes([0, 0, 0, 255]) * 255)
    children = size + xyzi + rgba
    main = b"MAIN" + struct.pack("<II", 0, 0xFFFFFFFF) + children
    return b"VOX " + struct.pack("<I", 150) + main


def named_object_vox_bytes():
    model0 = chunk(b"SIZE", struct.pack("<III", 1, 1, 1))
    model0 += chunk(b"XYZI", struct.pack("<I", 1) + bytes([0, 0, 0, 1]))
    model1 = chunk(b"SIZE", struct.pack("<III", 1, 1, 1))
    model1 += chunk(b"XYZI", struct.pack("<I", 1) + bytes([0, 0, 0, 2]))
    red_transform = chunk(
        b"nTRN",
        struct.pack("<I", 10)
        + dict_bytes({"_name": "red_object"})
        + struct.pack("<iiiI", 11, -1, -1, 1)
        + dict_bytes(),
    )
    blue_transform = chunk(
        b"nTRN",
        struct.pack("<I", 20)
        + dict_bytes({"_name": "blue_object"})
        + struct.pack("<iiiI", 21, -1, -1, 1)
        + dict_bytes({"_t": "2 0 0"}),
    )
    red_shape = chunk(
        b"nSHP",
        struct.pack("<I", 11)
        + dict_bytes()
        + struct.pack("<I", 1)
        + struct.pack("<I", 0)
        + dict_bytes(),
    )
    blue_shape = chunk(
        b"nSHP",
        struct.pack("<I", 21)
        + dict_bytes()
        + struct.pack("<I", 1)
        + struct.pack("<I", 1)
        + dict_bytes(),
    )
    rgba = chunk(
        b"RGBA",
        bytes([220, 0, 0, 255])
        + bytes([0, 40, 220, 255])
        + bytes([0, 0, 0, 255]) * 254,
    )
    main = chunk(
        b"MAIN",
        b"",
        chunk(b"PACK", struct.pack("<I", 2))
        + model0
        + model1
        + red_transform
        + blue_transform
        + red_shape
        + blue_shape
        + rgba,
    )
    return b"VOX " + struct.pack("<I", 150) + main


def hidden_object_vox_bytes():
    model0 = chunk(b"SIZE", struct.pack("<III", 1, 1, 1))
    model0 += chunk(b"XYZI", struct.pack("<I", 1) + bytes([0, 0, 0, 1]))
    model1 = chunk(b"SIZE", struct.pack("<III", 1, 1, 1))
    model1 += chunk(b"XYZI", struct.pack("<I", 1) + bytes([0, 0, 0, 2]))
    visible_transform = chunk(
        b"nTRN",
        struct.pack("<I", 10)
        + dict_bytes({"_name": "visible_object"})
        + struct.pack("<iiiI", 11, -1, -1, 1)
        + dict_bytes(),
    )
    hidden_transform = chunk(
        b"nTRN",
        struct.pack("<I", 20)
        + dict_bytes({"_name": "hidden_object", "_hidden": "1"})
        + struct.pack("<iiiI", 21, -1, -1, 1)
        + dict_bytes({"_t": "2 0 0"}),
    )
    visible_shape = chunk(
        b"nSHP",
        struct.pack("<I", 11)
        + dict_bytes()
        + struct.pack("<I", 1)
        + struct.pack("<I", 0)
        + dict_bytes(),
    )
    hidden_shape = chunk(
        b"nSHP",
        struct.pack("<I", 21)
        + dict_bytes()
        + struct.pack("<I", 1)
        + struct.pack("<I", 1)
        + dict_bytes(),
    )
    rgba = chunk(b"RGBA", bytes([255, 255, 255, 255]) * 256)
    main = chunk(
        b"MAIN",
        b"",
        chunk(b"PACK", struct.pack("<I", 2))
        + model0
        + model1
        + visible_transform
        + hidden_transform
        + visible_shape
        + hidden_shape
        + rgba,
    )
    return b"VOX " + struct.pack("<I", 150) + main


class VoxParserTests(unittest.TestCase):
    def test_reads_size_xyzi_and_rgba(self):
        model = conv.read_vox_models(minimal_vox_bytes())[0]

        self.assertEqual(model.size, (2, 4, 3))
        self.assertEqual(model.voxels, [(0, 2, 1, 7), (1, 3, 2, 8)])
        self.assertEqual(len(model.palette), 256)

    def test_pads_short_rgba_chunk(self):
        data = minimal_vox_bytes(rgba_len=1020)

        model = conv.read_vox_models(data)[0]

        self.assertEqual(len(model.palette), 256)

    def test_uses_magicavoxel_default_palette_when_rgba_is_absent(self):
        model = conv.read_vox_models(minimal_vox_without_rgba_bytes())[0]

        self.assertEqual(model.palette[0], (0, 0, 0, 0))
        self.assertEqual(model.palette[1], (255, 255, 255, 255))
        self.assertEqual(model.palette[2], (255, 255, 204, 255))
        self.assertEqual(model.palette[255], (17, 17, 17, 255))

    def test_read_vox_models_accepts_teardown_unsized_main_children(self):
        model = conv.read_vox_models(minimal_vox_with_unsized_main_children())[0]

        self.assertEqual(len(model.voxels), 1)
        self.assertEqual(model.palette[1], (10, 20, 30, 255))

    def test_rgba_chunk_is_mapped_to_one_based_palette_indices(self):
        content = (
            bytes([10, 20, 30, 255])
            + bytes([40, 50, 60, 255])
            + bytes([0, 0, 0, 255]) * 254
        )
        data = minimal_vox_bytes(rgba_len=1024).replace(
            bytes(i % 256 for i in range(256 * 4)),
            content,
            1,
        )

        model = conv.read_vox_models(data)[0]

        self.assertEqual(model.palette[0], (0, 0, 0, 0))
        self.assertEqual(model.palette[1], (10, 20, 30, 255))
        self.assertEqual(model.palette[2], (40, 50, 60, 255))

    def test_imap_reorders_palette_and_voxel_indices_like_magicavoxel(self):
        raw_palette = (
            bytes([10, 0, 0, 255])
            + bytes([20, 0, 0, 255])
            + bytes([30, 0, 0, 255])
            + bytes([0, 0, 0, 255]) * 253
        )
        imap = bytes([2, 1] + list(range(3, 256)) + [0])
        size = chunk(b"SIZE", struct.pack("<III", 2, 1, 1))
        xyzi = chunk(
            b"XYZI",
            struct.pack("<I", 2) + bytes([0, 0, 0, 1, 1, 0, 0, 2]),
        )
        data = b"VOX " + struct.pack("<I", 150) + chunk(
            b"MAIN",
            b"",
            size + xyzi + chunk(b"RGBA", raw_palette) + chunk(b"IMAP", imap),
        )

        model = conv.read_vox_models(data)[0]

        self.assertEqual(model.voxels, [(0, 0, 0, 2), (1, 0, 0, 1)])
        self.assertEqual(model.palette[1], (20, 0, 0, 255))
        self.assertEqual(model.palette[2], (10, 0, 0, 255))

    def test_scene_graph_transform_rotates_and_translates_model_instances(self):
        size = chunk(b"SIZE", struct.pack("<III", 2, 1, 1))
        xyzi = chunk(b"XYZI", struct.pack("<I", 1) + bytes([1, 0, 0, 1]))
        ntrn = chunk(
            b"nTRN",
            struct.pack("<I", 0)
            + dict_bytes()
            + struct.pack("<iiiI", 1, -1, -1, 1)
            + dict_bytes({"_r": "20", "_t": "10 0 0"}),
        )
        nshp = chunk(
            b"nSHP",
            struct.pack("<I", 1)
            + dict_bytes()
            + struct.pack("<I", 1)
            + struct.pack("<I", 0)
            + dict_bytes(),
        )
        rgba = chunk(b"RGBA", bytes([255, 255, 255, 255]) * 256)
        data = b"VOX " + struct.pack("<I", 150) + chunk(b"MAIN", b"", size + xyzi + ntrn + nshp + rgba)

        model = conv.read_vox_models(data)[0]

        self.assertEqual(model.voxels, [(9, 0, 0, 1)])

    def test_read_vox_models_can_filter_named_scene_object(self):
        models = conv.read_vox_models(named_object_vox_bytes(), object_name="blue_object")

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].voxels, [(2, 0, 0, 2)])
        self.assertIn("blue_object", models[0].names)

    def test_read_vox_models_skips_hidden_transform_nodes_by_default(self):
        models = conv.read_vox_models(hidden_object_vox_bytes())

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].names, ("visible_object",))

    def test_read_vox_models_can_include_hidden_transform_nodes_for_teardown_object_refs(self):
        models = conv.read_vox_models(hidden_object_vox_bytes(), include_hidden=True)

        self.assertEqual(len(models), 2)
        self.assertEqual(models[1].names, ("hidden_object",))
        self.assertTrue(models[1].hidden)

    def test_writer_emits_vox_pack_and_chunk_models(self):
        out = conv.write_chunked_vox(
            {
                (0, 0, 0): [(0, 0, 0, 3)],
                (1, 0, 0): [(0, 0, 0, 4)],
            },
            conv.default_palette(),
            chunk_size=256,
        )

        self.assertTrue(out.startswith(b"VOX "))
        self.assertIn(b"PACK", out)
        self.assertEqual(out.count(b"SIZE"), 2)
        self.assertEqual(out.count(b"XYZI"), 2)
        self.assertIn(b"nTRN", out)
        self.assertIn(b"_t", out)

    def test_writer_stores_palette_index_one_as_first_rgba_entry(self):
        palette = conv.default_palette()
        palette[1] = (10, 20, 30, 255)
        palette[2] = (40, 50, 60, 255)

        out = conv.write_chunked_vox({(0, 0, 0): [(0, 0, 0, 1)]}, palette)
        rgba_offset = out.index(b"RGBA")
        content_size = struct.unpack_from("<I", out, rgba_offset + 4)[0]
        content = out[rgba_offset + 12 : rgba_offset + 12 + content_size]

        self.assertEqual(content[:8], bytes([10, 20, 30, 255, 40, 50, 60, 255]))
        self.assertEqual(content_size, 256 * 4)

    def test_writer_stores_internal_y_up_as_magicavoxel_z_up_coordinates(self):
        out = conv.write_chunked_vox({(0, 0, 0): [(1, 2, 3, 5)]}, conv.default_palette())

        size_offset = out.index(b"SIZE")
        xyzi_offset = out.index(b"XYZI")

        self.assertEqual(struct.unpack_from("<III", out, size_offset + 12), (2, 4, 3))
        self.assertEqual(out[xyzi_offset + 16 : xyzi_offset + 20], bytes([1, 3, 2, 5]))

    def test_writer_emits_scene_graph_frame_indices_required_by_magicavoxel(self):
        out = conv.write_chunked_vox({(0, 0, 0): [(0, 0, 0, 1)]}, conv.default_palette())

        self.assertGreaterEqual(out.count(b"_f"), 3)


class TeardownPathTests(unittest.TestCase):
    def test_resolves_mod_path_and_reports_builtin_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "map.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("root/main.xml", "<scene/>")
                zf.writestr("root/vox/tree.vox", minimal_vox_bytes())

            with conv.ZipMapSource(zip_path) as source:
                self.assertIsNotNone(source.read_mod_file("MOD/vox/tree.vox"))
                self.assertIsNone(source.read_mod_file("BUILT-IN/vox/nature/tree.vox"))
                self.assertIsNone(source.read_mod_file("MOD/vox/missing.vox"))
                self.assertIn("BUILT-IN/vox/nature/tree.vox", source.missing_builtin)
                self.assertIn("MOD/vox/missing.vox", source.missing_mod)

    def test_resolves_flat_workshop_zip_with_root_main_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "map.zip"
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("main.xml", "<scene/>")
                zf.writestr("vox/tree.vox", minimal_vox_bytes())

            with conv.ZipMapSource(zip_path) as source:
                self.assertEqual(source.read_main_xml(), b"<scene/>")
                self.assertIsNotNone(source.read_mod_file("MOD/vox/tree.vox"))

    def test_resolves_builtin_path_from_teardown_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "map.zip"
            install = tmp_path / "Teardown"
            builtin = install / "data" / "built-in" / "vox" / "nature" / "tree.vox"
            builtin.parent.mkdir(parents=True)
            builtin.write_bytes(minimal_vox_bytes())
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("root/main.xml", "<scene/>")

            with conv.ZipMapSource(zip_path, teardown_dir=install) as source:
                self.assertEqual(source.read_mod_file("BUILT-IN/vox/nature/tree.vox"), minimal_vox_bytes())
                self.assertNotIn("BUILT-IN/vox/nature/tree.vox", source.missing_builtin)

    def test_resolves_builtin_prefab_xml_from_teardown_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = tmp_path / "map.zip"
            install = tmp_path / "Teardown"
            prefab = install / "data" / "built-in" / "prefab" / "door" / "simple.xml"
            prefab.parent.mkdir(parents=True)
            prefab.write_text('<scene><vox file="BUILT-IN/vox/prop/crate.vox"/></scene>', encoding="utf-8")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("root/main.xml", "<scene/>")

            with conv.ZipMapSource(zip_path, teardown_dir=install) as source:
                self.assertEqual(
                    source.read_mod_file("BUILT-IN/prefab/door/simple.xml"),
                    b'<scene><vox file="BUILT-IN/vox/prop/crate.vox"/></scene>',
                )


class StaticMapExportTests(unittest.TestCase):
    def make_zip(self, files):
        tmp = tempfile.TemporaryDirectory()
        zip_path = Path(tmp.name) / "map.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, content in files.items():
                zf.writestr(f"root/{name}", content)
        self.addCleanup(tmp.cleanup)
        return zip_path

    def export_zip(self, zip_path):
        out = zip_path.with_name("out.vox")
        report = zip_path.with_name("out.report.json")
        stats = conv.export_static_map(zip_path, out, report, voxels_per_unit=10)
        return stats, conv.read_vox_models(out.read_bytes())[0]

    def test_export_preserves_model_palette_color_in_output_palette(self):
        red_vox = minimal_vox_bytes()
        red_vox = red_vox.replace(bytes([0, 1, 2, 7, 1, 2, 3, 8]), bytes([0, 1, 2, 2, 1, 2, 3, 2]), 1)
        red_vox = red_vox.replace(
            bytes(i % 256 for i in range(256 * 4)),
            bytes([40, 40, 40, 255, 220, 20, 20, 255])
            + bytes([0, 0, 0, 255]) * 254,
            1,
        )
        main_xml = '<scene><vox file="MOD/vox/red.vox"/></scene>'
        zip_path = self.make_zip({"main.xml": main_xml, "vox/red.vox": red_vox})

        _stats, model = self.export_zip(zip_path)

        color_index = model.voxels[0][3]
        self.assertEqual(model.palette[color_index], (220, 20, 20, 255))

    def test_export_rasterizes_voxagon_instead_of_skipping_it(self):
        main_xml = """
        <scene>
          <voxagon pos="0 0 0" extrude="2" color="0.8 0.7 0.6">
            <vertex pos="0 0"/>
            <vertex pos="1 0"/>
            <vertex pos="1 1"/>
            <vertex pos="0 1"/>
          </voxagon>
        </scene>
        """
        zip_path = self.make_zip({"main.xml": main_xml})

        stats, model = self.export_zip(zip_path)

        self.assertEqual(stats["voxagon_nodes_skipped"], 0)
        self.assertEqual(stats["voxagon_nodes_exported"], 1)
        self.assertGreater(len(model.voxels), 0)
        color_index = model.voxels[0][3]
        self.assertEqual(model.palette[color_index], (204, 178, 153, 255))

    def test_export_tolerates_malformed_color_suffixes(self):
        main_xml = """
        <scene>
          <voxagon pos="0 0 0" extrude="1" color="0.8 0.21ű 0.6">
            <vertex pos="0 0"/>
            <vertex pos="1 0"/>
            <vertex pos="1 1"/>
            <vertex pos="0 1"/>
          </voxagon>
        </scene>
        """
        zip_path = self.make_zip({"main.xml": main_xml})

        stats, model = self.export_zip(zip_path)

        self.assertEqual(stats["voxagon_nodes_exported"], 1)
        color_index = model.voxels[0][3]
        self.assertEqual(model.palette[color_index], (204, 54, 153, 255))

    def test_export_uses_vox_object_attribute_instead_of_whole_asset_pack(self):
        main_xml = '<scene><vox file="MOD/vox/set.vox" object="blue_object"/></scene>'
        zip_path = self.make_zip({"main.xml": main_xml, "vox/set.vox": named_object_vox_bytes()})

        stats, model = self.export_zip(zip_path)

        self.assertEqual(stats["vox_nodes_exported"], 1)
        self.assertEqual(len(model.voxels), 1)
        color_index = model.voxels[0][3]
        self.assertEqual(model.palette[color_index], (0, 40, 220, 255))

    def test_export_skips_malformed_vox_resource_without_aborting_map(self):
        main_xml = """
        <scene>
          <vox file="MOD/vox/bad.vox"/>
          <voxbox size="1 1 1" color="1 0 0"/>
        </scene>
        """
        zip_path = self.make_zip({"main.xml": main_xml, "vox/bad.vox": b"not a vox"})

        stats, model = self.export_zip(zip_path)

        self.assertEqual(stats["malformed_vox_refs"], ["MOD/vox/bad.vox"])
        self.assertEqual(stats["vox_nodes_exported"], 0)
        self.assertEqual(stats["voxbox_nodes_exported"], 1)
        self.assertGreater(len(model.voxels), 0)

    def test_export_reads_builtin_vox_from_teardown_install(self):
        with tempfile.TemporaryDirectory() as install_tmp:
            install = Path(install_tmp)
            builtin = install / "data" / "built-in" / "vox" / "prop" / "crate.vox"
            builtin.parent.mkdir(parents=True)
            builtin.write_bytes(minimal_vox_bytes())
            main_xml = '<scene><vox file="BUILT-IN/vox/prop/crate.vox"/></scene>'
            zip_path = self.make_zip({"main.xml": main_xml})
            out = zip_path.with_name("out.vox")
            report = zip_path.with_name("out.report.json")

            stats = conv.export_static_map(zip_path, out, report, voxels_per_unit=10, teardown_dir=install)
            model = conv.read_vox_models(out.read_bytes())[0]

        self.assertEqual(stats["builtin_refs_skipped"], 0)
        self.assertEqual(stats["vox_nodes_exported"], 1)
        self.assertEqual(len(model.voxels), 2)

    def test_export_expands_builtin_prefab_instance_from_teardown_install(self):
        with tempfile.TemporaryDirectory() as install_tmp:
            install = Path(install_tmp)
            builtin_vox = install / "data" / "built-in" / "vox" / "prop" / "crate.vox"
            builtin_vox.parent.mkdir(parents=True)
            builtin_vox.write_bytes(minimal_vox_bytes())
            prefab = install / "data" / "built-in" / "prefab" / "prop" / "crate.xml"
            prefab.parent.mkdir(parents=True)
            prefab.write_text('<scene><vox file="BUILT-IN/vox/prop/crate.vox"/></scene>', encoding="utf-8")
            main_xml = '<scene><instance pos="1 0 0" file="BUILT-IN/prefab/prop/crate.xml"/></scene>'
            zip_path = self.make_zip({"main.xml": main_xml})
            out = zip_path.with_name("out.vox")
            report = zip_path.with_name("out.report.json")

            stats = conv.export_static_map(zip_path, out, report, voxels_per_unit=10, teardown_dir=install)
            model = conv.read_vox_models(out.read_bytes())[0]

        self.assertEqual(stats["instance_nodes_exported"], 1)
        self.assertEqual(stats["builtin_refs_skipped"], 0)
        self.assertEqual(len(model.voxels), 2)
        self.assertEqual(min(x for x, _y, _z, _color in model.voxels), 10)

    def test_export_applies_vehicle_transform_to_child_geometry(self):
        main_xml = """
        <scene>
          <vehicle pos="10 0 0">
            <voxbox size="1 1 1" color="1 0 0"/>
          </vehicle>
        </scene>
        """
        zip_path = self.make_zip({"main.xml": main_xml})

        stats, model = self.export_zip(zip_path)

        self.assertEqual(stats["voxbox_nodes_exported"], 1)
        self.assertEqual(min(x for x, _y, _z, _color in model.voxels), 100)

    def test_voxbox_voxel_coordinates_stay_contiguous_when_density_changes(self):
        main_xml = '<scene><voxbox size="3 2 1" color="1 0 0"/></scene>'
        zip_path = self.make_zip({"main.xml": main_xml})
        out = zip_path.with_name("out.vox")
        report = zip_path.with_name("out.report.json")

        conv.export_static_map(zip_path, out, report, voxels_per_unit=20)
        model = conv.read_vox_models(out.read_bytes())[0]

        coords = sorted((x, y, z) for x, y, z, _color in model.voxels)
        self.assertEqual(
            coords,
            [
                (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (0, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
                (1, 2, 0), (1, 2, 1), (1, 3, 0), (1, 3, 1),
                (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1),
                (2, 2, 0), (2, 2, 1), (2, 3, 0), (2, 3, 1),
                (3, 0, 0), (3, 0, 1), (3, 1, 0), (3, 1, 1),
                (3, 2, 0), (3, 2, 1), (3, 3, 0), (3, 3, 1),
                (4, 0, 0), (4, 0, 1), (4, 1, 0), (4, 1, 1),
                (4, 2, 0), (4, 2, 1), (4, 3, 0), (4, 3, 1),
                (5, 0, 0), (5, 0, 1), (5, 1, 0), (5, 1, 1),
                (5, 2, 0), (5, 2, 1), (5, 3, 0), (5, 3, 1),
            ],
        )
        self.assertEqual(model.size, (6, 4, 2))

    def test_export_treats_hoe_voxbox_brush_as_hole(self):
        main_xml = '<scene><voxbox brush="hoe" size="2 2 2"/></scene>'
        zip_path = self.make_zip({"main.xml": main_xml})
        out = zip_path.with_name("out.vox")
        report = zip_path.with_name("out.report.json")

        stats = conv.export_static_map(zip_path, out, report, voxels_per_unit=10)
        models = conv.read_vox_models(out.read_bytes())

        self.assertEqual(stats["voxbox_nodes_exported"], 0)
        self.assertEqual(models, [])

    def test_export_treats_hoe_voxagon_brush_as_hole(self):
        main_xml = """
        <scene>
          <voxagon brush="hoe" extrude="1">
            <vertex pos="0 0"/>
            <vertex pos="1 0"/>
            <vertex pos="1 1"/>
            <vertex pos="0 1"/>
          </voxagon>
        </scene>
        """
        zip_path = self.make_zip({"main.xml": main_xml})
        out = zip_path.with_name("out.vox")
        report = zip_path.with_name("out.report.json")

        stats = conv.export_static_map(zip_path, out, report, voxels_per_unit=10)
        models = conv.read_vox_models(out.read_bytes())

        self.assertEqual(stats["voxagon_nodes_exported"], 0)
        self.assertEqual(stats["voxagon_nodes_skipped"], 1)
        self.assertEqual(models, [])

    def test_vox_asset_density_twenty_expands_each_native_voxel_to_two_by_two_by_two(self):
        main_xml = '<scene><vox file="MOD/vox/red.vox"/></scene>'
        red_vox = minimal_vox_bytes().replace(bytes([0, 1, 2, 7, 1, 2, 3, 8]), bytes([0, 0, 0, 2, 0, 0, 0, 2]), 1)
        zip_path = self.make_zip({"main.xml": main_xml, "vox/red.vox": red_vox})
        out = zip_path.with_name("out.vox")
        report = zip_path.with_name("out.report.json")

        conv.export_static_map(zip_path, out, report, voxels_per_unit=20)
        model = conv.read_vox_models(out.read_bytes())[0]

        coords = sorted((x, y, z) for x, y, z, _color in model.voxels)
        self.assertEqual(
            coords,
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
        )

    def test_axis_aligned_voxbox_fast_path_uses_integer_density_coordinates(self):
        chunks = {}
        matrix = conv._translation((0.25, 0.0, 0.0))

        used_fast_path = conv._try_add_axis_aligned_box(chunks, matrix, (3, 2, 1), 9, voxels_per_unit=20)

        self.assertTrue(used_fast_path)
        coords = sorted(chunks[(0, 0, 0)])
        self.assertEqual(len(coords), 48)
        self.assertEqual(coords[0], (5, 0, 0))
        self.assertEqual(coords[-1], (10, 3, 1))

    def test_axis_aligned_voxbox_fast_path_matches_slow_rounding(self):
        matrix = conv._translation((0.15, 0.0, 0.0))
        slow_chunks = {}
        fast_chunks = {}
        for point in conv._solid_box_points((3, 1, 1), voxels_per_unit=10):
            conv._add_voxel(slow_chunks, conv._transform_point(matrix, point), 4, voxels_per_unit=10)

        used_fast_path = conv._try_add_axis_aligned_box(fast_chunks, matrix, (3, 1, 1), 4, voxels_per_unit=10)

        self.assertTrue(used_fast_path)
        self.assertEqual(fast_chunks, slow_chunks)


if __name__ == "__main__":
    unittest.main()
