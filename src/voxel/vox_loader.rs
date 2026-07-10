use crate::voxel::brick::{BRICK_EDGE, BRICK_VOLUME, BrickData, VoxelCell};
use crate::voxel::morton;
use crate::voxel::ucvh::Ucvh;
use glam::{IVec3, UVec3, Vec3};
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoxLoadError {
    InvalidHeader,
    MissingMainChunk,
    MalformedChunk,
    MalformedDict,
    MissingSizeBeforeVoxels,
    Io(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxVoxel {
    pub position: UVec3,
    pub color_index: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxModel {
    pub size: UVec3,
    pub voxels: Vec<VoxVoxel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxTransform {
    pub rotation: [[i32; 3]; 3],
    pub translation: IVec3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxInstance {
    pub model_index: usize,
    pub rotation: [[i32; 3]; 3],
    pub translation: IVec3,
    pub hidden: bool,
    pub object_name: Option<String>,
    pub object_origin: Option<IVec3>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxBounds {
    pub min: IVec3,
    pub max_exclusive: IVec3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxTargetBounds {
    pub min: UVec3,
    pub max_exclusive: UVec3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxScene {
    pub models: Vec<VoxModel>,
    pub palette: [[u8; 4]; 256],
    pub instances: Vec<VoxInstance>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VoxWriteStats {
    pub input_voxels: u64,
    pub written_voxels: u64,
    pub out_of_bounds_voxels: u64,
    pub unique_written_voxels: u64,
    pub source_bounds: Option<VoxBounds>,
    pub target_bounds: Option<VoxTargetBounds>,
}

#[derive(Debug, Clone, Copy)]
struct Chunk<'a> {
    id: [u8; 4],
    content: &'a [u8],
    children: &'a [u8],
}

#[derive(Debug, Clone)]
struct TransformNode {
    child: u32,
    transform: VoxTransform,
    layer_id: i32,
    hidden: bool,
    name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GroupNode {
    children: Vec<u32>,
    hidden: bool,
    name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShapeModel {
    model_index: usize,
    name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ShapeNode {
    models: Vec<ShapeModel>,
    name: Option<String>,
}

const IDENTITY_ROTATION: [[i32; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
const ENCODED_COLOR_MATERIAL_FLAG: u16 = 0x8000;
const MAGICAVOXEL_CUBE_LEVELS: [u8; 6] = [255, 204, 153, 102, 51, 0];
const MAGICAVOXEL_RAMP_LEVELS: [u8; 10] = [238, 221, 187, 170, 136, 119, 85, 68, 34, 17];

fn default_palette() -> [[u8; 4]; 256] {
    let mut palette = [[0, 0, 0, 0]; 256];
    let mut index = 1;
    for &r in &MAGICAVOXEL_CUBE_LEVELS {
        for &g in &MAGICAVOXEL_CUBE_LEVELS {
            for &b in &MAGICAVOXEL_CUBE_LEVELS {
                if (r, g, b) != (0, 0, 0) {
                    palette[index] = [r, g, b, 255];
                    index += 1;
                }
            }
        }
    }
    for &value in &MAGICAVOXEL_RAMP_LEVELS {
        palette[index] = [value, 0, 0, 255];
        index += 1;
    }
    for &value in &MAGICAVOXEL_RAMP_LEVELS {
        palette[index] = [0, value, 0, 255];
        index += 1;
    }
    for &value in &MAGICAVOXEL_RAMP_LEVELS {
        palette[index] = [0, 0, value, 255];
        index += 1;
    }
    for &value in &MAGICAVOXEL_RAMP_LEVELS {
        palette[index] = [value, value, value, 255];
        index += 1;
    }
    debug_assert_eq!(index, 256);
    palette
}

pub fn load_vox_file_into_ucvh(
    path: impl AsRef<Path>,
    ucvh: &mut Ucvh,
) -> Result<VoxWriteStats, VoxLoadError> {
    let data = std::fs::read(path.as_ref()).map_err(|error| VoxLoadError::Io(error.to_string()))?;
    let scene = parse_vox(&data)?;
    Ok(write_scene_to_ucvh(&scene, ucvh))
}

pub fn parse_vox(data: &[u8]) -> Result<VoxScene, VoxLoadError> {
    if data.len() < 8 || &data[0..4] != b"VOX " {
        return Err(VoxLoadError::InvalidHeader);
    }

    let mut main_children = None;
    for chunk in ChunkIter::new(&data[8..]) {
        let chunk = chunk?;
        if &chunk.id == b"MAIN" {
            main_children = Some(chunk.children);
            break;
        }
    }
    let Some(main_children) = main_children else {
        return Err(VoxLoadError::MissingMainChunk);
    };

    let mut models = Vec::new();
    let mut raw_palette = raw_palette_from_logical_palette(default_palette());
    let mut index_map = None;
    let mut pending_size = None;
    let mut transforms = HashMap::new();
    let mut groups: HashMap<u32, GroupNode> = HashMap::new();
    let mut shapes: HashMap<u32, ShapeNode> = HashMap::new();
    let mut referenced_children = HashSet::new();
    let mut hidden_layers = HashSet::new();

    for chunk in ChunkIter::new(main_children) {
        let chunk = chunk?;
        match &chunk.id {
            b"SIZE" => {
                pending_size = Some(read_size(chunk.content)?);
            }
            b"XYZI" => {
                let Some(size) = pending_size.take() else {
                    return Err(VoxLoadError::MissingSizeBeforeVoxels);
                };
                models.push(VoxModel {
                    size,
                    voxels: read_xyzi(chunk.content)?,
                });
            }
            b"RGBA" => {
                raw_palette = read_raw_palette(chunk.content);
            }
            b"IMAP" => {
                index_map = Some(read_index_map(chunk.content)?);
            }
            b"nTRN" => {
                let (node_id, transform) = read_transform(chunk.content)?;
                referenced_children.insert(transform.child);
                transforms.insert(node_id, transform);
            }
            b"nGRP" => {
                let (node_id, group) = read_group(chunk.content)?;
                referenced_children.extend(group.children.iter().copied());
                groups.insert(node_id, group);
            }
            b"nSHP" => {
                let (node_id, shape) = read_shape(chunk.content)?;
                shapes.insert(node_id, shape);
            }
            b"LAYR" => {
                let (layer_id, hidden) = read_layer(chunk.content)?;
                if hidden {
                    hidden_layers.insert(layer_id);
                }
            }
            _ => {}
        }
    }

    if let Some(index_map) = index_map {
        apply_index_map(&mut raw_palette, &mut models, &index_map);
    }
    let palette = logical_palette_from_raw_rgba(raw_palette);
    let instances = collect_instances(
        &models,
        &transforms,
        &groups,
        &shapes,
        &referenced_children,
        &hidden_layers,
    );
    Ok(VoxScene {
        models,
        palette,
        instances,
    })
}

pub fn write_scene_to_ucvh(scene: &VoxScene, ucvh: &mut Ucvh) -> VoxWriteStats {
    let mut stats = VoxWriteStats::default();
    let Some(bounds) = scene.bounds() else {
        return stats;
    };
    stats.source_bounds = Some(bounds);
    let extent = (bounds.max_exclusive - bounds.min).as_vec3().max(Vec3::ONE);
    let world = ucvh.config.world_size.as_vec3();
    let padding = if world.min_element() > 16.0 { 4.0 } else { 1.0 };
    let usable = (world - Vec3::splat(padding * 2.0)).max(Vec3::ONE);
    let scale = (usable / extent).min_element().min(1.0);
    let mut staged_bricks: HashMap<UVec3, StagedBrick> = HashMap::new();
    let mut target_min = UVec3::splat(u32::MAX);
    let mut target_max_exclusive = UVec3::ZERO;

    for instance in scene.iter_instances() {
        let Some(model) = scene.models.get(instance.model_index) else {
            continue;
        };
        for voxel in &model.voxels {
            stats.input_voxels += 1;
            let source = instance.transform_point(voxel.position.as_ivec3());
            let target = ((source - bounds.min).as_vec3() * scale + Vec3::splat(padding)).floor();
            let target = UVec3::new(target.x as u32, target.y as u32, target.z as u32);
            let cell = VoxelCell::new(
                material_for_color(scene.palette_for(voxel.color_index)),
                1,
                [0; 3],
            );
            if target.x < ucvh.config.world_size.x
                && target.y < ucvh.config.world_size.y
                && target.z < ucvh.config.world_size.z
            {
                stats.written_voxels += 1;
                let brick_pos = target / BRICK_EDGE;
                let local_pos = target - brick_pos * BRICK_EDGE;
                staged_bricks
                    .entry(brick_pos)
                    .or_insert_with(|| StagedBrick::new_seeded(ucvh, brick_pos))
                    .write(local_pos, cell);
                target_min = target_min.min(target);
                target_max_exclusive = target_max_exclusive.max(target + UVec3::ONE);
            } else {
                stats.out_of_bounds_voxels += 1;
            }
        }
    }

    for (brick_pos, brick) in &staged_bricks {
        if !ucvh.write_brick(*brick_pos, &brick.data) {
            stats.out_of_bounds_voxels += u64::from(brick.touched_count);
        }
    }

    stats.unique_written_voxels = staged_bricks
        .values()
        .map(|brick| u64::from(brick.touched_count))
        .sum();
    if stats.unique_written_voxels > 0 {
        stats.target_bounds = Some(VoxTargetBounds {
            min: target_min,
            max_exclusive: target_max_exclusive,
        });
    }
    stats
}

struct StagedBrick {
    data: BrickData,
    touched: [bool; BRICK_VOLUME],
    touched_count: u32,
}

impl StagedBrick {
    fn new_seeded(ucvh: &Ucvh, brick_pos: UVec3) -> Self {
        let mut data = BrickData::new();
        if ucvh.brick_id_at(brick_pos).is_some() {
            let base = brick_pos * BRICK_EDGE;
            for z in 0..BRICK_EDGE {
                for y in 0..BRICK_EDGE {
                    for x in 0..BRICK_EDGE {
                        let world = base + UVec3::new(x, y, z);
                        data.set_voxel(x, y, z, ucvh.get_voxel(world));
                    }
                }
            }
        }
        Self {
            data,
            touched: [false; BRICK_VOLUME],
            touched_count: 0,
        }
    }

    fn write(&mut self, local_pos: UVec3, cell: VoxelCell) {
        let morton = morton::encode(local_pos.x, local_pos.y, local_pos.z) as usize;
        if !self.touched[morton] {
            self.touched[morton] = true;
            self.touched_count += 1;
        }
        self.data
            .set_voxel(local_pos.x, local_pos.y, local_pos.z, cell);
    }
}

impl VoxScene {
    pub fn bounds(&self) -> Option<VoxBounds> {
        let mut min = IVec3::splat(i32::MAX);
        let mut max_exclusive = IVec3::splat(i32::MIN);
        let mut any = false;

        for instance in self.iter_instances() {
            let Some(model) = self.models.get(instance.model_index) else {
                continue;
            };
            for voxel in &model.voxels {
                any = true;
                let pos = instance.transform_point(voxel.position.as_ivec3());
                min = min.min(pos);
                max_exclusive = max_exclusive.max(pos + IVec3::ONE);
            }
        }

        any.then_some(VoxBounds { min, max_exclusive })
    }

    fn iter_instances(&self) -> Box<dyn Iterator<Item = VoxInstance> + '_> {
        if self.instances.is_empty() {
            Box::new(
                (0..self.models.len())
                    .map(|model_index| VoxInstance::new(model_index, VoxTransform::identity())),
            )
        } else {
            Box::new(
                self.instances
                    .iter()
                    .filter(|instance| !instance.hidden)
                    .cloned(),
            )
        }
    }

    fn palette_for(&self, color_index: u8) -> [u8; 4] {
        self.palette
            .get(color_index as usize)
            .copied()
            .unwrap_or([180, 180, 180, 255])
    }

    pub fn visit_visible_voxels(&self, mut visit: impl FnMut(IVec3, [u8; 4])) {
        for instance in self.iter_instances() {
            let Some(model) = self.models.get(instance.model_index) else {
                continue;
            };
            for voxel in &model.voxels {
                visit(
                    instance.transform_point(voxel.position.as_ivec3()),
                    self.palette_for(voxel.color_index),
                );
            }
        }
    }

    pub fn visible_voxel_count(&self) -> u64 {
        self.iter_instances()
            .filter_map(|instance| self.models.get(instance.model_index))
            .map(|model| model.voxels.len() as u64)
            .sum()
    }
}

impl VoxTransform {
    fn identity() -> Self {
        Self {
            rotation: IDENTITY_ROTATION,
            translation: IVec3::ZERO,
        }
    }

    fn compose(self, child: Self) -> Self {
        Self {
            rotation: mat3_mul(self.rotation, child.rotation),
            translation: self.translation + mat3_vec(self.rotation, child.translation),
        }
    }
}

impl VoxInstance {
    fn new(model_index: usize, transform: VoxTransform) -> Self {
        Self::new_with_hidden(model_index, transform, false)
    }

    fn new_with_hidden(model_index: usize, transform: VoxTransform, hidden: bool) -> Self {
        Self::new_with_hidden_and_name(model_index, transform, hidden, None, None)
    }

    fn new_with_hidden_and_name(
        model_index: usize,
        transform: VoxTransform,
        hidden: bool,
        object_name: Option<String>,
        object_origin: Option<IVec3>,
    ) -> Self {
        Self {
            model_index,
            rotation: transform.rotation,
            translation: transform.translation,
            hidden,
            object_name,
            object_origin,
        }
    }

    fn transform_point(&self, point: IVec3) -> IVec3 {
        mat3_vec(self.rotation, point) + self.translation
    }
}

fn collect_instances(
    models: &[VoxModel],
    transforms: &HashMap<u32, TransformNode>,
    groups: &HashMap<u32, GroupNode>,
    shapes: &HashMap<u32, ShapeNode>,
    referenced_children: &HashSet<u32>,
    hidden_layers: &HashSet<i32>,
) -> Vec<VoxInstance> {
    if shapes.is_empty() {
        return (0..models.len())
            .map(|model_index| VoxInstance::new(model_index, VoxTransform::identity()))
            .collect();
    }

    let mut instances = Vec::new();
    let mut roots: Vec<u32> = transforms
        .keys()
        .chain(groups.keys())
        .chain(shapes.keys())
        .copied()
        .filter(|node_id| !referenced_children.contains(node_id))
        .collect();
    roots.sort_unstable();
    roots.dedup();

    for root in roots {
        visit_node(
            root,
            VoxTransform::identity(),
            transforms,
            groups,
            shapes,
            hidden_layers,
            false,
            None,
            None,
            &mut instances,
        );
    }
    instances
}

#[allow(clippy::too_many_arguments)]
fn visit_node(
    node_id: u32,
    parent: VoxTransform,
    transforms: &HashMap<u32, TransformNode>,
    groups: &HashMap<u32, GroupNode>,
    shapes: &HashMap<u32, ShapeNode>,
    hidden_layers: &HashSet<i32>,
    hidden: bool,
    object_name: Option<String>,
    object_origin: Option<IVec3>,
    instances: &mut Vec<VoxInstance>,
) {
    if let Some(node_transform) = transforms.get(&node_id) {
        let next_hidden =
            hidden || node_transform.hidden || hidden_layers.contains(&node_transform.layer_id);
        let next_transform = parent.compose(node_transform.transform);
        let (next_name, next_origin) = match node_transform.name.clone() {
            Some(name) => (Some(name), Some(next_transform.translation)),
            None => (object_name, object_origin),
        };
        visit_node(
            node_transform.child,
            next_transform,
            transforms,
            groups,
            shapes,
            hidden_layers,
            next_hidden,
            next_name,
            next_origin,
            instances,
        );
        return;
    }
    if let Some(group) = groups.get(&node_id) {
        let (next_name, next_origin) = match group.name.clone() {
            Some(name) => (Some(name), Some(parent.translation)),
            None => (object_name, object_origin),
        };
        for child in &group.children {
            visit_node(
                *child,
                parent,
                transforms,
                groups,
                shapes,
                hidden_layers,
                hidden || group.hidden,
                next_name.clone(),
                next_origin,
                instances,
            );
        }
        return;
    }
    if let Some(shape) = shapes.get(&node_id) {
        instances.extend(shape.models.iter().map(|model| {
            let shape_name = shape.name.clone().or_else(|| model.name.clone());
            let (name, origin) = match shape_name {
                Some(name) => (Some(name), Some(parent.translation)),
                None => (object_name.clone(), object_origin),
            };
            VoxInstance::new_with_hidden_and_name(model.model_index, parent, hidden, name, origin)
        }));
    }
}

pub fn material_for_color(color: [u8; 4]) -> u16 {
    let [r, g, b, _a] = color;
    let r = quantize_u8_to_5_bits(r);
    let g = quantize_u8_to_5_bits(g);
    let b = quantize_u8_to_5_bits(b);
    ENCODED_COLOR_MATERIAL_FLAG | (r << 10) | (g << 5) | b
}

fn quantize_u8_to_5_bits(value: u8) -> u16 {
    (u16::from(value) * 31 + 127) / 255
}

fn read_size(content: &[u8]) -> Result<UVec3, VoxLoadError> {
    if content.len() < 12 {
        return Err(VoxLoadError::MalformedChunk);
    }
    Ok(from_magicavoxel_position(UVec3::new(
        u32::from_le_bytes(content[0..4].try_into().unwrap()),
        u32::from_le_bytes(content[4..8].try_into().unwrap()),
        u32::from_le_bytes(content[8..12].try_into().unwrap()),
    )))
}

fn read_xyzi(content: &[u8]) -> Result<Vec<VoxVoxel>, VoxLoadError> {
    if content.len() < 4 {
        return Err(VoxLoadError::MalformedChunk);
    }
    let count = u32::from_le_bytes(content[0..4].try_into().unwrap()) as usize;
    let expected = 4 + count * 4;
    if content.len() < expected {
        return Err(VoxLoadError::MalformedChunk);
    }
    let mut voxels = Vec::with_capacity(count);
    for voxel in content[4..expected].chunks_exact(4) {
        voxels.push(VoxVoxel {
            position: from_magicavoxel_position(UVec3::new(
                voxel[0] as u32,
                voxel[1] as u32,
                voxel[2] as u32,
            )),
            color_index: voxel[3],
        });
    }
    Ok(voxels)
}

fn from_magicavoxel_position(position: UVec3) -> UVec3 {
    UVec3::new(position.x, position.z, position.y)
}

fn from_magicavoxel_translation(translation: IVec3) -> IVec3 {
    IVec3::new(translation.x, translation.z, translation.y)
}

fn from_magicavoxel_rotation(rotation: [[i32; 3]; 3]) -> [[i32; 3]; 3] {
    const SWAP_YZ: [[i32; 3]; 3] = [[1, 0, 0], [0, 0, 1], [0, 1, 0]];
    mat3_mul(mat3_mul(SWAP_YZ, rotation), SWAP_YZ)
}

fn mat3_vec(matrix: [[i32; 3]; 3], vector: IVec3) -> IVec3 {
    let values = [vector.x, vector.y, vector.z];
    IVec3::new(
        matrix[0][0] * values[0] + matrix[0][1] * values[1] + matrix[0][2] * values[2],
        matrix[1][0] * values[0] + matrix[1][1] * values[1] + matrix[1][2] * values[2],
        matrix[2][0] * values[0] + matrix[2][1] * values[1] + matrix[2][2] * values[2],
    )
}

fn mat3_mul(a: [[i32; 3]; 3], b: [[i32; 3]; 3]) -> [[i32; 3]; 3] {
    let mut out = [[0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
        }
    }
    out
}

fn raw_palette_from_logical_palette(palette: [[u8; 4]; 256]) -> [[u8; 4]; 256] {
    let mut raw = [[0, 0, 0, 255]; 256];
    raw[..255].copy_from_slice(&palette[1..256]);
    raw[255] = palette[0];
    raw
}

fn logical_palette_from_raw_rgba(raw: [[u8; 4]; 256]) -> [[u8; 4]; 256] {
    let mut palette = [[0, 0, 0, 0]; 256];
    palette[0] = raw[255];
    palette[0][3] = 0;
    palette[1..256].copy_from_slice(&raw[..255]);
    palette
}

fn read_raw_palette(content: &[u8]) -> [[u8; 4]; 256] {
    let mut raw = raw_palette_from_logical_palette(default_palette());
    for (index, rgba) in content.chunks_exact(4).take(256).enumerate() {
        raw[index] = [rgba[0], rgba[1], rgba[2], rgba[3]];
    }
    raw
}

fn read_index_map(content: &[u8]) -> Result<[u8; 256], VoxLoadError> {
    if content.len() < 256 {
        return Err(VoxLoadError::MalformedChunk);
    }
    let mut index_map = [0; 256];
    index_map.copy_from_slice(&content[..256]);
    Ok(index_map)
}

fn apply_index_map(
    raw_palette: &mut [[u8; 4]; 256],
    models: &mut [VoxModel],
    index_map: &[u8; 256],
) {
    let mut inverse = [0; 256];
    for (display_index, actual_index) in index_map.iter().copied().enumerate() {
        inverse[actual_index as usize] = display_index as u8;
    }

    let old_palette = *raw_palette;
    for index in 0..256 {
        raw_palette[index] = old_palette[(usize::from(index_map[index]) + 255) & 0xFF];
    }

    for model in models {
        for voxel in &mut model.voxels {
            if voxel.color_index != 0 {
                voxel.color_index = (1u16 + u16::from(inverse[voxel.color_index as usize])) as u8;
            }
        }
    }
}

fn read_transform(content: &[u8]) -> Result<(u32, TransformNode), VoxLoadError> {
    let mut reader = Reader::new(content);
    let node_id = reader.u32()?;
    let attrs = reader.dict()?;
    let child = reader.i32()?;
    if child < 0 {
        return Err(VoxLoadError::MalformedChunk);
    }
    let _reserved = reader.i32()?;
    let layer_id = reader.i32()?;
    let frame_count = reader.u32()?;
    let mut transform = VoxTransform::identity();
    for frame_index in 0..frame_count {
        let frame = reader.dict()?;
        if frame_index == 0 {
            let rotation = parse_rotation(frame.get("_r").map(String::as_str))?;
            let translation = frame
                .get("_t")
                .map(|value| parse_translation(value))
                .transpose()?
                .unwrap_or(IVec3::ZERO);
            transform = VoxTransform {
                rotation: from_magicavoxel_rotation(rotation),
                translation: from_magicavoxel_translation(translation),
            };
        }
    }
    Ok((
        node_id,
        TransformNode {
            child: child as u32,
            transform,
            layer_id,
            hidden: dict_bool(&attrs, "_hidden"),
            name: attrs.get("_name").cloned(),
        },
    ))
}

fn read_group(content: &[u8]) -> Result<(u32, GroupNode), VoxLoadError> {
    let mut reader = Reader::new(content);
    let node_id = reader.u32()?;
    let attrs = reader.dict()?;
    let child_count = reader.u32()?;
    let mut children = Vec::with_capacity(child_count as usize);
    for _ in 0..child_count {
        children.push(reader.u32()?);
    }
    Ok((
        node_id,
        GroupNode {
            children,
            hidden: dict_bool(&attrs, "_hidden"),
            name: attrs.get("_name").cloned(),
        },
    ))
}

fn read_shape(content: &[u8]) -> Result<(u32, ShapeNode), VoxLoadError> {
    let mut reader = Reader::new(content);
    let node_id = reader.u32()?;
    let attrs = reader.dict()?;
    let model_count = reader.u32()?;
    let mut models = Vec::with_capacity(model_count as usize);
    for _ in 0..model_count {
        let model_index = reader.u32()? as usize;
        let model_attrs = reader.dict()?;
        models.push(ShapeModel {
            model_index,
            name: model_attrs.get("_name").cloned(),
        });
    }
    Ok((
        node_id,
        ShapeNode {
            models,
            name: attrs.get("_name").cloned(),
        },
    ))
}

fn read_layer(content: &[u8]) -> Result<(i32, bool), VoxLoadError> {
    let mut reader = Reader::new(content);
    let layer_id = reader.i32()?;
    let attrs = reader.dict()?;
    let _reserved = reader.i32()?;
    Ok((layer_id, dict_bool(&attrs, "_hidden")))
}

fn dict_bool(values: &HashMap<String, String>, key: &str) -> bool {
    values.get(key).is_some_and(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no"
        )
    })
}

fn parse_translation(value: &str) -> Result<IVec3, VoxLoadError> {
    let mut parts = value.split_whitespace();
    let x = parts
        .next()
        .ok_or(VoxLoadError::MalformedChunk)?
        .parse()
        .map_err(|_| VoxLoadError::MalformedChunk)?;
    let y = parts
        .next()
        .ok_or(VoxLoadError::MalformedChunk)?
        .parse()
        .map_err(|_| VoxLoadError::MalformedChunk)?;
    let z = parts
        .next()
        .ok_or(VoxLoadError::MalformedChunk)?
        .parse()
        .map_err(|_| VoxLoadError::MalformedChunk)?;
    Ok(IVec3::new(x, y, z))
}

fn parse_rotation(value: Option<&str>) -> Result<[[i32; 3]; 3], VoxLoadError> {
    let Some(value) = value else {
        return Ok(IDENTITY_ROTATION);
    };
    let packed = value
        .parse::<u32>()
        .map_err(|_| VoxLoadError::MalformedChunk)?;
    let row0_index = packed & 3;
    let row1_index = (packed >> 2) & 3;
    let row2_index = match (row0_index, row1_index) {
        (0, 1) | (1, 0) => 2,
        (0, 2) | (2, 0) => 1,
        (1, 2) | (2, 1) => 0,
        _ => return Err(VoxLoadError::MalformedChunk),
    };

    let row_indexes = [row0_index, row1_index, row2_index];
    let mut rotation = [[0; 3]; 3];
    for (row, axis) in row_indexes.iter().copied().enumerate() {
        let sign = if (packed & (1 << (4 + row))) != 0 {
            -1
        } else {
            1
        };
        rotation[row][axis as usize] = sign;
    }
    Ok(rotation)
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn bytes(&mut self, len: usize) -> Result<&'a [u8], VoxLoadError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or(VoxLoadError::MalformedChunk)?;
        if end > self.bytes.len() {
            return Err(VoxLoadError::MalformedChunk);
        }
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, VoxLoadError> {
        Ok(u32::from_le_bytes(
            self.bytes(4)?
                .try_into()
                .map_err(|_| VoxLoadError::MalformedChunk)?,
        ))
    }

    fn i32(&mut self) -> Result<i32, VoxLoadError> {
        Ok(i32::from_le_bytes(
            self.bytes(4)?
                .try_into()
                .map_err(|_| VoxLoadError::MalformedChunk)?,
        ))
    }

    fn string(&mut self) -> Result<String, VoxLoadError> {
        let len = self.u32()? as usize;
        let bytes = self.bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| VoxLoadError::MalformedDict)
    }

    fn dict(&mut self) -> Result<HashMap<String, String>, VoxLoadError> {
        let count = self.u32()?;
        let mut values = HashMap::new();
        for _ in 0..count {
            let key = self.string()?;
            let value = self.string()?;
            values.insert(key, value);
        }
        Ok(values)
    }
}

struct ChunkIter<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ChunkIter<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }
}

impl<'a> Iterator for ChunkIter<'a> {
    type Item = Result<Chunk<'a>, VoxLoadError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.bytes.len() {
            return None;
        }
        if self.offset + 12 > self.bytes.len() {
            return Some(Err(VoxLoadError::MalformedChunk));
        }
        let start = self.offset;
        let id = self.bytes[start..start + 4].try_into().unwrap();
        let content_size =
            u32::from_le_bytes(self.bytes[start + 4..start + 8].try_into().unwrap()) as usize;
        let raw_children_size =
            u32::from_le_bytes(self.bytes[start + 8..start + 12].try_into().unwrap());
        let content_start = start + 12;
        let Some(content_end) = content_start.checked_add(content_size) else {
            return Some(Err(VoxLoadError::MalformedChunk));
        };
        let children_end = if raw_children_size == u32::MAX {
            self.bytes.len()
        } else {
            let children_size = raw_children_size as usize;
            let Some(children_end) = content_end.checked_add(children_size) else {
                return Some(Err(VoxLoadError::MalformedChunk));
            };
            children_end
        };
        if children_end > self.bytes.len() {
            return Some(Err(VoxLoadError::MalformedChunk));
        }
        self.offset = children_end;
        Some(Ok(Chunk {
            id,
            content: &self.bytes[content_start..content_end],
            children: &self.bytes[content_end..children_end],
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::ucvh::{Ucvh, UcvhConfig};
    use glam::UVec3;

    fn chunk(id: &[u8; 4], content: &[u8], children: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(id);
        bytes.extend_from_slice(&(content.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(children.len() as u32).to_le_bytes());
        bytes.extend_from_slice(content);
        bytes.extend_from_slice(children);
        bytes
    }

    fn dict(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (key, value) in entries {
            bytes.extend_from_slice(&(key.len() as u32).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(&(value.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        bytes
    }

    fn minimal_vox() -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0], &[]));
        let xyzi = [2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2];
        children.extend(chunk(b"XYZI", &xyzi, &[]));
        let mut palette = Vec::new();
        for i in 1..=255u16 {
            palette.extend_from_slice(&[i as u8, i as u8, i as u8, 255]);
        }
        palette.extend_from_slice(&[0, 0, 0, 0]);
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    fn minimal_vox_without_rgba() -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0], &[]));
        let xyzi = [2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2];
        children.extend(chunk(b"XYZI", &xyzi, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    fn minimal_vox_with_unsized_main_children() -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        children.extend(chunk(b"XYZI", &[1, 0, 0, 0, 0, 0, 0, 1], &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&200u32.to_le_bytes());
        bytes.extend_from_slice(b"MAIN");
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend(children);
        bytes
    }

    fn translated_two_model_vox() -> Vec<u8> {
        let mut children = Vec::new();
        children.extend(chunk(b"PACK", &2u32.to_le_bytes(), &[]));
        for color in [1u8, 2] {
            children.extend(chunk(b"SIZE", &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
            let xyzi = [1, 0, 0, 0, 0, 0, 0, color];
            children.extend(chunk(b"XYZI", &xyzi, &[]));
        }

        let root_trn = {
            let mut content = Vec::new();
            content.extend_from_slice(&0u32.to_le_bytes());
            content.extend(dict(&[]));
            content.extend_from_slice(&1u32.to_le_bytes());
            content.extend_from_slice(&(-1i32).to_le_bytes());
            content.extend_from_slice(&(-1i32).to_le_bytes());
            content.extend_from_slice(&1u32.to_le_bytes());
            content.extend(dict(&[]));
            chunk(b"nTRN", &content, &[])
        };
        children.extend(root_trn);
        for (node_id, shape_id, tx) in [(2u32, 3u32, 0i32), (4, 5, 20)] {
            let mut content = Vec::new();
            content.extend_from_slice(&node_id.to_le_bytes());
            content.extend(dict(&[]));
            content.extend_from_slice(&shape_id.to_le_bytes());
            content.extend_from_slice(&(-1i32).to_le_bytes());
            content.extend_from_slice(&(-1i32).to_le_bytes());
            content.extend_from_slice(&1u32.to_le_bytes());
            content.extend(dict(&[("_t", &format!("{tx} 0 0"))]));
            children.extend(chunk(b"nTRN", &content, &[]));
        }
        let mut group = Vec::new();
        group.extend_from_slice(&1u32.to_le_bytes());
        group.extend(dict(&[]));
        group.extend_from_slice(&2u32.to_le_bytes());
        group.extend_from_slice(&2u32.to_le_bytes());
        group.extend_from_slice(&4u32.to_le_bytes());
        children.extend(chunk(b"nGRP", &group, &[]));
        for (shape_id, model_id) in [(3u32, 0u32), (5, 1)] {
            let mut shape = Vec::new();
            shape.extend_from_slice(&shape_id.to_le_bytes());
            shape.extend(dict(&[]));
            shape.extend_from_slice(&1u32.to_le_bytes());
            shape.extend_from_slice(&model_id.to_le_bytes());
            shape.extend(dict(&[]));
            children.extend(chunk(b"nSHP", &shape, &[]));
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        bytes
    }

    fn transform_node(node_id: u32, child_id: u32, frame_attrs: &[(&str, &str)]) -> Vec<u8> {
        transform_node_with_attrs(node_id, child_id, &[], -1, frame_attrs)
    }

    fn transform_node_with_attrs(
        node_id: u32,
        child_id: u32,
        node_attrs: &[(&str, &str)],
        layer_id: i32,
        frame_attrs: &[(&str, &str)],
    ) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend_from_slice(&node_id.to_le_bytes());
        content.extend(dict(node_attrs));
        content.extend_from_slice(&child_id.to_le_bytes());
        content.extend_from_slice(&(-1i32).to_le_bytes());
        content.extend_from_slice(&layer_id.to_le_bytes());
        content.extend_from_slice(&1u32.to_le_bytes());
        content.extend(dict(frame_attrs));
        chunk(b"nTRN", &content, &[])
    }

    fn shape_node(node_id: u32, model_id: u32) -> Vec<u8> {
        let mut shape = Vec::new();
        shape.extend_from_slice(&node_id.to_le_bytes());
        shape.extend(dict(&[]));
        shape.extend_from_slice(&1u32.to_le_bytes());
        shape.extend_from_slice(&model_id.to_le_bytes());
        shape.extend(dict(&[]));
        chunk(b"nSHP", &shape, &[])
    }

    #[test]
    fn parses_size_xyzi_and_palette() {
        let scene = parse_vox(&minimal_vox()).expect("minimal vox should parse");

        assert_eq!(scene.models.len(), 1);
        assert_eq!(scene.models[0].size, UVec3::new(2, 2, 2));
        assert_eq!(scene.models[0].voxels.len(), 2);
        assert_eq!(scene.palette[2], [2, 2, 2, 255]);
    }

    #[test]
    fn parses_teardown_vox_with_unsized_main_children() {
        let scene = parse_vox(&minimal_vox_with_unsized_main_children())
            .expect("Teardown VOX MAIN children size sentinel should parse");

        assert_eq!(scene.models.len(), 1);
        assert_eq!(scene.visible_voxel_count(), 1);
    }

    #[test]
    fn maps_rgba_chunk_entries_to_one_based_palette_indices() {
        let mut vox = minimal_vox();
        let rgba_offset = vox
            .windows(4)
            .position(|bytes| bytes == b"RGBA")
            .expect("minimal vox has RGBA chunk");
        let content_offset = rgba_offset + 12;
        vox[content_offset..content_offset + 8]
            .copy_from_slice(&[10, 20, 30, 255, 40, 50, 60, 255]);

        let scene = parse_vox(&vox).expect("minimal vox should parse");

        assert_eq!(scene.palette[0], [0, 0, 0, 0]);
        assert_eq!(scene.palette[1], [10, 20, 30, 255]);
        assert_eq!(scene.palette[2], [40, 50, 60, 255]);
    }

    #[test]
    fn imap_reorders_palette_and_voxel_indices_like_magicavoxel() {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        let xyzi = [2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2];
        children.extend(chunk(b"XYZI", &xyzi, &[]));
        let mut palette = Vec::new();
        palette.extend_from_slice(&[10, 0, 0, 255]);
        palette.extend_from_slice(&[20, 0, 0, 255]);
        palette.extend_from_slice(&[30, 0, 0, 255]);
        for _ in 3..256 {
            palette.extend_from_slice(&[0, 0, 0, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));
        let mut imap = vec![2u8, 1];
        imap.extend(3u8..=255);
        imap.push(0);
        children.extend(chunk(b"IMAP", &imap, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));

        let scene = parse_vox(&bytes).expect("imap vox should parse");

        assert_eq!(scene.models[0].voxels[0].color_index, 2);
        assert_eq!(scene.models[0].voxels[1].color_index, 1);
        assert_eq!(scene.palette[1], [20, 0, 0, 255]);
        assert_eq!(scene.palette[2], [10, 0, 0, 255]);
    }

    #[test]
    fn uses_magicavoxel_default_palette_when_rgba_is_absent() {
        let scene = parse_vox(&minimal_vox_without_rgba()).expect("minimal vox should parse");

        assert_eq!(scene.palette[0], [0, 0, 0, 0]);
        assert_eq!(scene.palette[1], [255, 255, 255, 255]);
        assert_eq!(scene.palette[2], [255, 255, 204, 255]);
        assert_eq!(scene.palette[255], [17, 17, 17, 255]);
    }

    #[test]
    fn converts_magicavoxel_z_up_coordinates_to_renderer_y_up() {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0], &[]));
        let xyzi = [1, 0, 0, 0, 1, 2, 3, 7];
        children.extend(chunk(b"XYZI", &xyzi, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));

        let scene = parse_vox(&bytes).expect("axis test vox should parse");

        assert_eq!(scene.models[0].size, UVec3::new(2, 4, 3));
        assert_eq!(scene.models[0].voxels[0].position, UVec3::new(1, 3, 2));
    }

    #[test]
    fn rejects_non_vox_data() {
        let err = parse_vox(b"not a vox").expect_err("invalid data should fail");

        assert!(matches!(err, VoxLoadError::InvalidHeader));
    }

    #[test]
    fn writes_voxels_into_ucvh_with_fit_to_world() {
        let scene = parse_vox(&minimal_vox()).expect("minimal vox should parse");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));

        let stats = write_scene_to_ucvh(&scene, &mut ucvh);

        assert_eq!(stats.input_voxels, 2);
        assert_eq!(stats.written_voxels, 2);
        assert!(ucvh.allocated_brick_count() > 0);
    }

    #[test]
    fn writes_voxels_reports_source_and_target_bounds() {
        let scene = parse_vox(&translated_two_model_vox()).expect("translated scene should parse");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(64)));

        let stats = write_scene_to_ucvh(&scene, &mut ucvh);

        assert_eq!(
            stats.source_bounds,
            Some(VoxBounds {
                min: IVec3::new(0, 0, 0),
                max_exclusive: IVec3::new(21, 1, 1),
            })
        );
        assert_eq!(
            stats.target_bounds,
            Some(VoxTargetBounds {
                min: UVec3::new(4, 4, 4),
                max_exclusive: UVec3::new(25, 5, 5),
            })
        );
    }

    #[test]
    fn writes_voxels_preserves_existing_cells_in_touched_bricks() {
        let scene = parse_vox(&minimal_vox()).expect("minimal vox should parse");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));
        let existing_pos = UVec3::new(2, 2, 2);
        let existing = VoxelCell::new(77, 1, [9, 8, 7]);
        assert!(ucvh.set_voxel(existing_pos, existing));

        let stats = write_scene_to_ucvh(&scene, &mut ucvh);

        assert_eq!(stats.unique_written_voxels, 2);
        let preserved = ucvh.get_voxel(existing_pos);
        assert_eq!(preserved.material, existing.material);
        assert_eq!(preserved.flags, existing.flags);
        assert_eq!(preserved.emissive, existing.emissive);
    }

    #[test]
    fn duplicate_target_voxels_count_once_and_last_material_wins() {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        let xyzi = [2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2];
        children.extend(chunk(b"XYZI", &xyzi, &[]));
        let mut palette = Vec::new();
        palette.extend_from_slice(&[10, 20, 30, 255]);
        palette.extend_from_slice(&[200, 210, 220, 255]);
        for _ in 2..256 {
            palette.extend_from_slice(&[0, 0, 0, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        let scene = parse_vox(&bytes).expect("duplicate target vox should parse");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));

        let stats = write_scene_to_ucvh(&scene, &mut ucvh);

        assert_eq!(stats.input_voxels, 2);
        assert_eq!(stats.written_voxels, 2);
        assert_eq!(stats.unique_written_voxels, 1);
        assert_eq!(
            ucvh.get_voxel(UVec3::new(1, 1, 1)).material,
            material_for_color([200, 210, 220, 255])
        );
    }

    #[test]
    fn writes_vox_palette_colors_as_encoded_material_colors() {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        let xyzi = [1, 0, 0, 0, 0, 0, 0, 1];
        children.extend(chunk(b"XYZI", &xyzi, &[]));
        let mut palette = Vec::new();
        palette.extend_from_slice(&[10, 20, 30, 255]);
        for _ in 1..256 {
            palette.extend_from_slice(&[0, 0, 0, 255]);
        }
        children.extend(chunk(b"RGBA", &palette, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));
        let scene = parse_vox(&bytes).expect("colored vox should parse");
        let mut ucvh = Ucvh::new(UcvhConfig::new(UVec3::splat(16)));

        let stats = write_scene_to_ucvh(&scene, &mut ucvh);

        assert_eq!(stats.written_voxels, 1);
        let cell = ucvh.get_voxel(UVec3::splat(1));
        assert_ne!(cell.material, 0);
        assert_ne!(cell.material, 1);
        assert_ne!(cell.material, 6);
        assert_eq!(cell.material & 0x8000, 0x8000);
    }

    #[test]
    fn scene_graph_translations_affect_bounds() {
        let scene = parse_vox(&translated_two_model_vox()).expect("translated scene should parse");

        let bounds = scene.bounds().expect("scene should have bounds");

        assert_eq!(bounds.min, glam::IVec3::new(0, 0, 0));
        assert_eq!(bounds.max_exclusive, glam::IVec3::new(21, 1, 1));
    }

    #[test]
    fn scene_graph_rotation_affects_bounds() {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        let xyzi = [1, 0, 0, 0, 1, 0, 0, 1];
        children.extend(chunk(b"XYZI", &xyzi, &[]));
        children.extend(transform_node(0, 1, &[("_r", "20"), ("_t", "10 0 0")]));
        children.extend(shape_node(1, 0));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));

        let scene = parse_vox(&bytes).expect("rotated scene should parse");
        let bounds = scene.bounds().expect("scene should have bounds");

        assert_eq!(bounds.min, glam::IVec3::new(9, 0, 0));
        assert_eq!(bounds.max_exclusive, glam::IVec3::new(10, 1, 1));
    }

    #[test]
    fn scene_graph_translation_is_converted_from_magicavoxel_z_up() {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        let xyzi = [1, 0, 0, 0, 0, 0, 0, 1];
        children.extend(chunk(b"XYZI", &xyzi, &[]));
        children.extend(transform_node(0, 1, &[("_t", "0 2 3")]));
        children.extend(shape_node(1, 0));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));

        let scene = parse_vox(&bytes).expect("translated scene should parse");
        let bounds = scene.bounds().expect("scene should have bounds");

        assert_eq!(bounds.min, glam::IVec3::new(0, 3, 2));
        assert_eq!(bounds.max_exclusive, glam::IVec3::new(1, 4, 3));
    }

    #[test]
    fn hidden_transform_nodes_are_excluded_from_scene_bounds() {
        let mut children = Vec::new();
        children.extend(chunk(b"PACK", &2u32.to_le_bytes(), &[]));
        for color in [1u8, 2] {
            children.extend(chunk(b"SIZE", &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
            let xyzi = [1, 0, 0, 0, 0, 0, 0, color];
            children.extend(chunk(b"XYZI", &xyzi, &[]));
        }
        children.extend(transform_node(0, 1, &[]));
        children.extend(transform_node_with_attrs(
            2,
            3,
            &[("_hidden", "1")],
            -1,
            &[("_t", "20 0 0")],
        ));
        children.extend(shape_node(1, 0));
        children.extend(shape_node(3, 1));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));

        let scene = parse_vox(&bytes).expect("hidden scene should parse");
        let bounds = scene.bounds().expect("visible instance should remain");

        assert_eq!(bounds.min, glam::IVec3::new(0, 0, 0));
        assert_eq!(bounds.max_exclusive, glam::IVec3::new(1, 1, 1));
    }

    #[test]
    fn hidden_layer_instances_are_excluded_from_scene_bounds() {
        let mut children = Vec::new();
        children.extend(chunk(b"SIZE", &[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], &[]));
        let xyzi = [1, 0, 0, 0, 0, 0, 0, 1];
        children.extend(chunk(b"XYZI", &xyzi, &[]));
        children.extend(transform_node_with_attrs(0, 1, &[], 7, &[]));
        children.extend(shape_node(1, 0));
        let mut layer = Vec::new();
        layer.extend_from_slice(&7u32.to_le_bytes());
        layer.extend(dict(&[("_hidden", "1")]));
        layer.extend_from_slice(&(-1i32).to_le_bytes());
        children.extend(chunk(b"LAYR", &layer, &[]));

        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"VOX ");
        bytes.extend_from_slice(&150u32.to_le_bytes());
        bytes.extend(chunk(b"MAIN", &[], &children));

        let scene = parse_vox(&bytes).expect("hidden layer scene should parse");

        assert_eq!(scene.bounds(), None);
    }

    #[test]
    fn vox_palette_colors_are_encoded_in_rgb555_material_range() {
        let olive = material_for_color([101, 119, 9, 255]);
        let dark_olive = material_for_color([69, 82, 7, 255]);

        assert_eq!(
            olive & ENCODED_COLOR_MATERIAL_FLAG,
            ENCODED_COLOR_MATERIAL_FLAG
        );
        assert_eq!(
            dark_olive & ENCODED_COLOR_MATERIAL_FLAG,
            ENCODED_COLOR_MATERIAL_FLAG
        );
        assert_ne!(olive, dark_olive);
    }
}
