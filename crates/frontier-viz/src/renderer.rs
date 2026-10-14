// SPDX-License-Identifier: Apache-2.0
//! wgpu pipeline for [`SurfaceMesh`]: lit triangles coloured by a scalar field through a
//! colormap, with an optional wireframe overlay. Renders into any `TextureView` of the
//! format it was created with (canvas surface, offscreen texture, ...).
//!
//! Geometry is three vertex buffers per slot: positions (`xyz`), normals (`xyz`, only
//! when smooth shading asks for them; flat shading takes the face normal from screen-space
//! derivatives and needs none) and the colouring scalar. A time step of a static mesh is
//! therefore one scalar write; a step of a moving mesh is one positions write and one
//! scalar write, nothing is recomputed on the CPU. Steps can be kept on the GPU in a
//! per-slot *frame ring* ([`SurfaceRenderer::stage_frame`]): showing a resident step is
//! a change of which buffers the draw binds, no upload at all, so a looping playback
//! whose steps fit in the ring costs only its draws.

use crate::camera::OrbitCamera;
use crate::colormap::{ColorScale, Colormap};
use crate::lut::{ColormapAtlas, ScaleUniform};
use crate::surface::SurfaceMesh;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderSettings {
    /// Colormap, bands, log scale and the colours of out-of-range and NaN values.
    pub scale: ColorScale,
    pub wireframe: bool,
    /// Vertex (smooth) normals instead of per-face normals computed in the fragment
    /// shader. Off by default: decimated LODs have sliver triangles and the Ahmed body
    /// has 90-degree creases, both of which smooth normals smear into star artifacts.
    pub smooth: bool,
    /// Scalar range mapped to the colormap.
    pub range: (f32, f32),
    pub background: [f64; 4],
    pub light_dir: [f32; 3],
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            scale: ColorScale::from(Colormap::Viridis),
            wireframe: false,
            smooth: false,
            range: (0.0, 1.0),
            background: [0.055, 0.067, 0.086, 1.0],
            light_dir: [0.4, 0.9, 0.35],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    range: [f32; 2],
    colormap: u32,
    flags: u32,
    wire_offset: f32,
    steps: u32,
    scale_flags: u32,
    _pad: f32,
    below: [f32; 4],
    above: [f32; 4],
    nan: [f32; 4],
}

impl Uniforms {
    /// The colour-scale fields from the settings.
    fn with_scale(mut self, settings: &RenderSettings) -> Self {
        let scale = ScaleUniform::new(&settings.scale, settings.range);
        self.colormap = scale.colormap;
        self.steps = scale.steps;
        self.scale_flags = scale.flags;
        self.below = scale.below;
        self.above = scale.above;
        self.nan = scale.nan;
        self
    }
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// One time step resident on the GPU.
struct RingEntry {
    key: u64,
    positions: Option<wgpu::Buffer>,
    scalars: wgpu::Buffer,
    bytes: usize,
    used: u64,
}

/// GPU buffers of one uploaded mesh. A renderer holds several slots so split views and
/// small multiples draw many meshes into one target with shared depth / MSAA attachments.
/// Each slot has its own uniform buffer, so several slots can be drawn in one render pass.
struct Slot {
    positions: Option<wgpu::Buffer>,
    normals: Option<wgpu::Buffer>,
    scalars: Option<wgpu::Buffer>,
    vertex_count: u32,
    indices: Option<wgpu::Buffer>,
    index_count: u32,
    /// CPU copy of the indices, kept to build the wireframe on first use.
    cpu_indices: Vec<u32>,
    wire_indices: Option<wgpu::Buffer>,
    wire_count: u32,
    extent: f32,
    uniforms: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    ring: Vec<RingEntry>,
    ring_capacity: usize,
    /// Ring entry bound for drawing (`None`: the slot's own position / scalar buffers).
    current: Option<usize>,
    tick: u64,
}

impl Slot {
    fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, atlas: &ColormapAtlas) -> Self {
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface uniforms"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(atlas.view()),
                },
            ],
        });
        Self {
            positions: None,
            normals: None,
            scalars: None,
            vertex_count: 0,
            indices: None,
            index_count: 0,
            cpu_indices: Vec::new(),
            wire_indices: None,
            wire_count: 0,
            extent: 1.0,
            uniforms,
            bind_group,
            ring: Vec::new(),
            ring_capacity: 16,
            current: None,
            tick: 0,
        }
    }

    fn clear(&mut self) {
        self.positions = None;
        self.normals = None;
        self.scalars = None;
        self.indices = None;
        self.cpu_indices = Vec::new();
        self.wire_indices = None;
        self.index_count = 0;
        self.wire_count = 0;
        self.vertex_count = 0;
        self.ring.clear();
        self.current = None;
    }

    /// The buffers a draw binds: positions, normals (positions when none, the shader
    /// ignores them in flat mode) and scalars.
    fn bound(&self) -> Option<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer)> {
        let (pos, sc) = match self.current.and_then(|i| self.ring.get(i)) {
            Some(e) => (
                e.positions.as_ref().or(self.positions.as_ref())?,
                &e.scalars,
            ),
            None => (self.positions.as_ref()?, self.scalars.as_ref()?),
        };
        Some((pos, self.normals.as_ref().unwrap_or(pos), sc))
    }

    fn ring_bytes(&self) -> usize {
        self.ring.iter().map(|e| e.bytes).sum()
    }
}

/// One cell of a small-multiples frame: mesh slot `slot` drawn into `viewport`
/// (`x, y, w, h` in target pixels) with its own camera and colour settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tile {
    pub slot: usize,
    pub viewport: (f32, f32, f32, f32),
    pub camera: OrbitCamera,
    pub settings: RenderSettings,
}

pub struct SurfaceRenderer {
    format: wgpu::TextureFormat,
    sample_count: u32,
    fill: wgpu::RenderPipeline,
    wire: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    slots: Vec<Slot>,
    atlas: ColormapAtlas,
    depth: Option<(wgpu::TextureView, u32, u32)>,
    msaa: Option<(wgpu::TextureView, u32, u32)>,
}

fn buffer_f32(
    device: &wgpu::Device,
    label: &str,
    data: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

impl SurfaceRenderer {
    /// `sample_count` 1 or 4 (MSAA).
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, sample_count: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frontier-viz surface"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", include_str!("shader.wgsl"), crate::lut::WGSL).into(),
            ),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface uniforms"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                ColormapAtlas::layout_entry(1),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("surface"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let xyz = |location: u32| wgpu::VertexBufferLayout {
            array_stride: 3 * 4,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: match location {
                0 => &wgpu::vertex_attr_array![0 => Float32x3],
                _ => &wgpu::vertex_attr_array![1 => Float32x3],
            },
        };
        let scalar_layout = wgpu::VertexBufferLayout {
            array_stride: 4,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![2 => Float32],
        };
        let make = |topology: wgpu::PrimitiveTopology, entry: &str, blend: bool, label: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(xyz(0)), Some(xyz(1)), Some(scalar_layout.clone())],
                    compilation_options: Default::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(!blend),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    // A depth bias is not allowed on a line list; the wire pass is pulled
                    // towards the camera in the vertex shader instead (flag bit 1).
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: if blend {
                            Some(wgpu::BlendState::ALPHA_BLENDING)
                        } else {
                            Some(wgpu::BlendState::REPLACE)
                        },
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let fill = make(
            wgpu::PrimitiveTopology::TriangleList,
            "fs_main",
            false,
            "surface fill",
        );
        let wire = make(
            wgpu::PrimitiveTopology::LineList,
            "fs_wire",
            true,
            "surface wire",
        );
        let atlas = ColormapAtlas::new(device);
        let slots = vec![Slot::new(device, &bgl, &atlas)];
        Self {
            format,
            sample_count,
            fill,
            wire,
            layout: bgl,
            slots,
            atlas,
            depth: None,
            msaa: None,
        }
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn has_mesh(&self) -> bool {
        self.slots[0].index_count > 0
    }

    pub fn triangle_count(&self) -> u32 {
        self.slots[0].index_count / 3
    }

    /// Upload a mesh with the per-point scalar used for colouring (slot 0).
    pub fn upload(&mut self, device: &wgpu::Device, mesh: &SurfaceMesh, scalar: &[f32]) {
        self.upload_slot(device, 0, mesh, scalar);
    }

    /// Upload into mesh slot `slot` (created on demand); see [`Self::render_region`].
    /// Vertex normals are computed and uploaded (they are needed for smooth shading);
    /// the winding is normalised so `front_facing` means "outside".
    pub fn upload_slot(
        &mut self,
        device: &wgpu::Device,
        slot: usize,
        mesh: &SurfaceMesh,
        scalar: &[f32],
    ) {
        let flip = !mesh.winding_is_outward();
        let sign = if flip { -1.0 } else { 1.0 };
        let normals: Vec<f32> = mesh
            .vertex_normals()
            .iter()
            .flat_map(|n| [n[0] * sign, n[1] * sign, n[2] * sign])
            .collect();
        let positions: Vec<f32> = mesh.points.iter().flat_map(|p| *p).collect();
        let idx: Vec<u32> = mesh
            .faces
            .iter()
            .flat_map(|f| if flip { [f[0], f[2], f[1]] } else { *f })
            .collect();
        self.upload_topology(device, slot, &positions, &idx, Some(&normals));
        self.set_scalars(device, slot, scalar);
    }

    /// Upload GPU-ready geometry: `positions` (`xyz` per vertex), triangle `indices` and
    /// optional vertex `normals` (without them the surface is flat shaded, which is the
    /// default anyway, and moving meshes need no per-step normal computation). Clears the
    /// slot's frame ring; the scalars start at zero until [`Self::set_scalars`] or a
    /// staged frame.
    pub fn upload_topology(
        &mut self,
        device: &wgpu::Device,
        slot: usize,
        positions: &[f32],
        indices: &[u32],
        normals: Option<&[f32]>,
    ) {
        self.ensure_slot(device, slot);
        let n = positions.len() / 3;
        let (lo, hi) = bounds(positions);
        let s = &mut self.slots[slot];
        s.clear();
        s.extent = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2))
            .sqrt()
            .max(1e-6);
        s.vertex_count = n as u32;
        s.positions = Some(buffer_f32(
            device,
            "surface positions",
            positions,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        ));
        s.normals = normals
            .filter(|nn| nn.len() == positions.len())
            .map(|nn| buffer_f32(device, "surface normals", nn, wgpu::BufferUsages::VERTEX));
        s.scalars = Some(buffer_f32(
            device,
            "surface scalars",
            &vec![0.0f32; n],
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        ));
        s.indices = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("surface indices"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
        s.index_count = indices.len() as u32;
        s.cpu_indices = indices.to_vec();
    }

    /// Write the slot's own scalars (creates the buffer when the length changed).
    pub fn set_scalars(&mut self, device: &wgpu::Device, slot: usize, scalar: &[f32]) {
        let Some(s) = self.slots.get_mut(slot) else {
            return;
        };
        let mut v: Vec<f32> = scalar.to_vec();
        v.resize(s.vertex_count as usize, 0.0);
        s.scalars = Some(buffer_f32(
            device,
            "surface scalars",
            &v,
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        ));
        s.current = None;
    }

    fn ensure_slot(&mut self, device: &wgpu::Device, slot: usize) {
        while self.slots.len() <= slot {
            self.slots
                .push(Slot::new(device, &self.layout, &self.atlas));
        }
    }

    /// Number of mesh slots that currently hold a mesh.
    pub fn loaded_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.index_count > 0).count()
    }

    /// Drop the buffers of a slot (small multiples scrolling out of view).
    pub fn clear_slot(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.clear();
        }
    }

    /// Replace only the colouring scalar of an uploaded mesh (a new time step of the same
    /// geometry): one `write_buffer` of 4 bytes per vertex. Returns `false` when the slot
    /// holds no mesh or the length does not match, in which case nothing is written and
    /// the caller should [`Self::upload_slot`] instead.
    pub fn update_scalar(&mut self, queue: &wgpu::Queue, slot: usize, scalar: &[f32]) -> bool {
        match self.slots.get_mut(slot) {
            Some(Slot {
                scalars: Some(buf),
                vertex_count,
                current,
                ..
            }) if *vertex_count as usize == scalar.len() && !scalar.is_empty() => {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(scalar));
                *current = None;
                true
            }
            _ => false,
        }
    }

    /// Replace the slot's own positions (a step of a moving mesh outside the ring).
    pub fn update_positions(
        &mut self,
        queue: &wgpu::Queue,
        slot: usize,
        positions: &[f32],
    ) -> bool {
        match self.slots.get_mut(slot) {
            Some(Slot {
                positions: Some(buf),
                vertex_count,
                current,
                ..
            }) if *vertex_count as usize * 3 == positions.len() && !positions.is_empty() => {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(positions));
                *current = None;
                true
            }
            _ => false,
        }
    }

    // ---- frame ring ---------------------------------------------------------------------

    /// How many steps slot `slot` keeps resident (default 16). Shrinking drops the least
    /// recently shown entries.
    pub fn set_ring_capacity(&mut self, slot: usize, n: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.ring_capacity = n;
            while s.ring.len() > n {
                let victim = Self::lru(s);
                Self::drop_entry(s, victim);
            }
        }
    }

    fn lru(s: &Slot) -> usize {
        let cur = s.current;
        s.ring
            .iter()
            .enumerate()
            .filter(|(i, _)| Some(*i) != cur)
            .min_by_key(|(_, e)| e.used)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn drop_entry(s: &mut Slot, i: usize) {
        if i >= s.ring.len() {
            return;
        }
        s.ring.remove(i);
        s.current = match s.current {
            Some(c) if c == i => None,
            Some(c) if c > i => Some(c - 1),
            other => other,
        };
    }

    /// Is step `key` resident in the ring of `slot`?
    pub fn has_frame(&self, slot: usize, key: u64) -> bool {
        self.slots
            .get(slot)
            .is_some_and(|s| s.ring.iter().any(|e| e.key == key))
    }

    /// Bytes of resident steps in the ring of `slot`.
    pub fn ring_bytes(&self, slot: usize) -> usize {
        self.slots.get(slot).map_or(0, Slot::ring_bytes)
    }

    /// Resident step keys of `slot`.
    pub fn ring_keys(&self, slot: usize) -> Vec<u64> {
        self.slots
            .get(slot)
            .map(|s| s.ring.iter().map(|e| e.key).collect())
            .unwrap_or_default()
    }

    /// Drop resident steps of `slot` not in `keep` until the ring holds at most `max_bytes`.
    /// Returns the bytes freed.
    pub fn trim_ring(&mut self, slot: usize, max_bytes: usize, keep: &[u64]) -> usize {
        let Some(s) = self.slots.get_mut(slot) else {
            return 0;
        };
        let mut freed = 0;
        while s.ring_bytes() > max_bytes {
            let victim = s
                .ring
                .iter()
                .enumerate()
                .filter(|(i, e)| Some(*i) != s.current && !keep.contains(&e.key))
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i);
            let Some(i) = victim else { break };
            freed += s.ring[i].bytes;
            Self::drop_entry(s, i);
        }
        freed
    }

    /// Put step `key` (positions when the mesh moves, one scalar per vertex) into the ring
    /// of `slot`, reusing the entry that already holds `key`, a free entry, or the least
    /// recently shown one. Returns the bytes written to the GPU (0 when the step was
    /// already resident, or when the slot holds no mesh / the lengths do not match).
    pub fn stage_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: usize,
        key: u64,
        positions: Option<&[f32]>,
        scalars: &[f32],
    ) -> usize {
        let Some(s) = self.slots.get_mut(slot) else {
            return 0;
        };
        let n = s.vertex_count as usize;
        if n == 0 || scalars.len() != n || positions.is_some_and(|p| p.len() != n * 3) {
            return 0;
        }
        s.tick += 1;
        if let Some(e) = s.ring.iter_mut().find(|e| e.key == key) {
            e.used = s.tick;
            return 0;
        }
        let bytes = scalars.len() * 4 + positions.map_or(0, |p| p.len() * 4);
        let entry = if s.ring.len() < s.ring_capacity.max(1) {
            None
        } else {
            Some(Self::lru(s))
        };
        match entry {
            Some(i) => {
                // Reuse the victim's buffers when the layout matches; else recreate.
                let e = &mut s.ring[i];
                let reuse = e.positions.is_some() == positions.is_some();
                if reuse {
                    queue.write_buffer(&e.scalars, 0, bytemuck::cast_slice(scalars));
                    if let (Some(buf), Some(p)) = (&e.positions, positions) {
                        queue.write_buffer(buf, 0, bytemuck::cast_slice(p));
                    }
                    e.key = key;
                    e.used = s.tick;
                    e.bytes = bytes;
                } else {
                    s.ring[i] = Self::new_entry(device, key, positions, scalars, s.tick, bytes);
                }
                if s.current == Some(i) {
                    s.current = None;
                }
            }
            None => s.ring.push(Self::new_entry(
                device, key, positions, scalars, s.tick, bytes,
            )),
        }
        bytes
    }

    fn new_entry(
        device: &wgpu::Device,
        key: u64,
        positions: Option<&[f32]>,
        scalars: &[f32],
        used: u64,
        bytes: usize,
    ) -> RingEntry {
        let usage = wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        RingEntry {
            key,
            positions: positions.map(|p| buffer_f32(device, "frame positions", p, usage)),
            scalars: buffer_f32(device, "frame scalars", scalars, usage),
            bytes,
            used,
        }
    }

    /// Bind resident step `key` of `slot` for the next draws. `false` when it is not
    /// resident (the slot keeps showing what it shows).
    pub fn show_frame(&mut self, slot: usize, key: u64) -> bool {
        let Some(s) = self.slots.get_mut(slot) else {
            return false;
        };
        s.tick += 1;
        match s.ring.iter().position(|e| e.key == key) {
            Some(i) => {
                s.ring[i].used = s.tick;
                s.current = Some(i);
                true
            }
            None => false,
        }
    }

    /// Draw from the slot's own buffers again (after [`Self::show_frame`]).
    pub fn show_base(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.current = None;
        }
    }

    fn ensure_wire(&mut self, device: &wgpu::Device, slot: usize) {
        let s = &mut self.slots[slot];
        if s.wire_indices.is_some() || s.cpu_indices.is_empty() {
            return;
        }
        let wire = edge_indices(&s.cpu_indices);
        s.wire_count = wire.len() as u32;
        s.wire_indices = Some(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("surface wire indices"),
                contents: bytemuck::cast_slice(&wire),
                usage: wgpu::BufferUsages::INDEX,
            }),
        );
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let fresh = |usage_label: &str, format: wgpu::TextureFormat, samples: u32| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(usage_label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        if !matches!(&self.depth, Some((_, w, h)) if *w == width && *h == height) {
            self.depth = Some((
                fresh("surface depth", DEPTH_FORMAT, self.sample_count),
                width,
                height,
            ));
        }
        if self.sample_count > 1
            && !matches!(&self.msaa, Some((_, w, h)) if *w == width && *h == height)
        {
            self.msaa = Some((
                fresh("surface msaa", self.format, self.sample_count),
                width,
                height,
            ));
        }
    }

    /// Render one frame into `target` (`width` x `height` px).
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        camera: &OrbitCamera,
        settings: &RenderSettings,
    ) {
        self.render_region(
            device, queue, target, width, height, camera, settings, 0, None,
        );
    }

    /// Render mesh slot `slot` into a sub-rectangle `(x, y, w, h, clear)` of the target
    /// (split views): the projection keeps the full target's aspect so both halves share
    /// one camera; `clear` decides whether the colour/depth attachments are cleared first
    /// (true for the first region of a frame, false for the others). `None` = full target.
    #[allow(clippy::too_many_arguments)]
    pub fn render_region(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        camera: &OrbitCamera,
        settings: &RenderSettings,
        slot: usize,
        region: Option<(u32, u32, u32, u32, bool)>,
    ) {
        let (width, height) = (width.max(1), height.max(1));
        if slot >= self.slots.len() {
            return;
        }
        self.atlas.sync(queue);
        self.ensure_targets(device, width, height);
        if settings.wireframe {
            self.ensure_wire(device, slot);
        }
        let mesh = &self.slots[slot];
        let (scissor, clear) = match region {
            Some((x, y, w, h, clear)) => (
                (
                    x.min(width - 1),
                    y.min(height - 1),
                    w.clamp(1, width - x.min(width - 1)),
                    h.clamp(1, height - y.min(height - 1)),
                ),
                clear,
            ),
            None => ((0, 0, width, height), true),
        };
        let aspect = width as f32 / height as f32;
        let eye = camera.eye();
        let mut flags = 0u32;
        if !self.format.is_srgb() {
            flags |= 1;
        }
        if settings.smooth && mesh.normals.is_some() {
            flags |= 4;
        }
        let base = Uniforms {
            view_proj: camera.view_proj(aspect).to_cols_array_2d(),
            camera_pos: [eye.x, eye.y, eye.z, 1.0],
            light_dir: [
                settings.light_dir[0],
                settings.light_dir[1],
                settings.light_dir[2],
                0.0,
            ],
            range: [settings.range.0, settings.range.1],
            flags,
            // The normal offset of the wire pass only makes sense with real normals.
            wire_offset: if mesh.normals.is_some() {
                mesh.extent * 0.0015
            } else {
                0.0
            },
            ..Zeroable::zeroed()
        }
        .with_scale(settings);
        queue.write_buffer(&mesh.uniforms, 0, bytemuck::bytes_of(&base));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("surface frame"),
        });
        let depth_view = &self.depth.as_ref().expect("depth target").0;
        let (view, resolve) = match &self.msaa {
            Some((msaa, _, _)) => (msaa, Some(target)),
            None => (target, None),
        };
        let bg = settings.background;
        let (color_load, depth_load) = if clear {
            (
                wgpu::LoadOp::Clear(wgpu::Color {
                    r: bg[0],
                    g: bg[1],
                    b: bg[2],
                    a: bg[3],
                }),
                wgpu::LoadOp::Clear(1.0),
            )
        } else {
            (wgpu::LoadOp::Load, wgpu::LoadOp::Load)
        };
        let bound = mesh.bound();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: color_load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: depth_load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_scissor_rect(scissor.0, scissor.1, scissor.2, scissor.3);
            if let (Some((pb, nb, sb)), Some(ib)) = (bound, &mesh.indices)
                && mesh.index_count > 0
            {
                pass.set_pipeline(&self.fill);
                pass.set_bind_group(0, &mesh.bind_group, &[]);
                pass.set_vertex_buffer(0, pb.slice(..));
                pass.set_vertex_buffer(1, nb.slice(..));
                pass.set_vertex_buffer(2, sb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
        if settings.wireframe
            && let (Some((pb, nb, sb)), Some(wb)) = (bound, &mesh.wire_indices)
            && mesh.wire_count > 0
        {
            // Second pass with the wire flag set; the vertex shader offsets along normals.
            let wire_u = Uniforms {
                flags: flags | 2,
                ..base
            };
            // A second uniform buffer keeps both passes in one submission.
            let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wire uniforms"),
                contents: bytemuck::bytes_of(&wire_u),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
            encoder.copy_buffer_to_buffer(
                &ub,
                0,
                &mesh.uniforms,
                0,
                std::mem::size_of::<Uniforms>() as u64,
            );
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface wire"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_scissor_rect(scissor.0, scissor.1, scissor.2, scissor.3);
            pass.set_pipeline(&self.wire);
            pass.set_bind_group(0, &mesh.bind_group, &[]);
            pass.set_vertex_buffer(0, pb.slice(..));
            pass.set_vertex_buffer(1, nb.slice(..));
            pass.set_vertex_buffer(2, sb.slice(..));
            pass.set_index_buffer(wb.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.wire_count, 0, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Small multiples: draw many mesh slots into one target in a single render pass and
    /// a single submission. Each tile has its own viewport, camera and colour settings;
    /// the target is cleared once to `background`. Tiles whose slot holds no mesh are
    /// skipped (their cell stays background), tiles outside the target are clipped.
    /// Returns the number of triangles drawn.
    #[allow(clippy::too_many_arguments)]
    pub fn render_tiles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        tiles: &[Tile],
        background: [f64; 4],
    ) -> u64 {
        let (width, height) = (width.max(1), height.max(1));
        self.atlas.sync(queue);
        self.ensure_targets(device, width, height);
        let srgb_flag = if self.format.is_srgb() { 0 } else { 1 };
        // Clip every tile to the target; remember the clipped rect for viewport + scissor.
        let mut drawn: Vec<(usize, (f32, f32, f32, f32))> = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let Some(mesh) = self.slots.get(tile.slot) else {
                continue;
            };
            let (x, y, w, h) = tile.viewport;
            if mesh.index_count == 0 || w < 1.0 || h < 1.0 {
                continue;
            }
            if x >= width as f32 || y >= height as f32 || x + w <= 0.0 || y + h <= 0.0 {
                continue;
            }
            let eye = tile.camera.eye();
            let mut flags = srgb_flag;
            if tile.settings.smooth && mesh.normals.is_some() {
                flags |= 4;
            }
            let u = Uniforms {
                view_proj: tile.camera.view_proj(w / h).to_cols_array_2d(),
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                light_dir: [
                    tile.settings.light_dir[0],
                    tile.settings.light_dir[1],
                    tile.settings.light_dir[2],
                    0.0,
                ],
                range: [tile.settings.range.0, tile.settings.range.1],
                flags,
                ..Zeroable::zeroed()
            }
            .with_scale(&tile.settings);
            queue.write_buffer(&mesh.uniforms, 0, bytemuck::bytes_of(&u));
            drawn.push((tile.slot, (x, y, w, h)));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("surface tiles"),
        });
        let depth_view = &self.depth.as_ref().expect("depth target").0;
        let (view, resolve) = match &self.msaa {
            Some((msaa, _, _)) => (msaa, Some(target)),
            None => (target, None),
        };
        let mut triangles = 0u64;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("surface tiles"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: background[0],
                            g: background[1],
                            b: background[2],
                            a: background[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.fill);
            for (slot, (x, y, w, h)) in drawn {
                let mesh = &self.slots[slot];
                let (Some((pb, nb, sb)), Some(ib)) = (mesh.bound(), &mesh.indices) else {
                    continue;
                };
                // The viewport may hang over the target edge (the projection must keep
                // the full tile's aspect); the scissor is what must stay inside.
                let sx = x.max(0.0).floor();
                let sy = y.max(0.0).floor();
                let sw = ((x + w).min(width as f32) - sx).floor();
                let sh = ((y + h).min(height as f32) - sy).floor();
                if sw < 1.0 || sh < 1.0 {
                    continue;
                }
                pass.set_viewport(x, y, w, h, 0.0, 1.0);
                pass.set_scissor_rect(sx as u32, sy as u32, sw as u32, sh as u32);
                pass.set_bind_group(0, &mesh.bind_group, &[]);
                pass.set_vertex_buffer(0, pb.slice(..));
                pass.set_vertex_buffer(1, nb.slice(..));
                pass.set_vertex_buffer(2, sb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
                triangles += (mesh.index_count / 3) as u64;
            }
        }
        queue.submit(Some(encoder.finish()));
        triangles
    }
}

fn bounds(positions: &[f32]) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for p in positions.as_chunks::<3>().0 {
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    if positions.is_empty() {
        ([0.0; 3], [0.0; 3])
    } else {
        (lo, hi)
    }
}

/// Unique edges of a triangle index list as a line-list index buffer.
pub fn edge_indices(indices: &[u32]) -> Vec<u32> {
    let mut keys: Vec<u64> = Vec::with_capacity(indices.len());
    for f in indices.as_chunks::<3>().0 {
        for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            keys.push(((lo as u64) << 32) | hi as u64);
        }
    }
    keys.sort_unstable();
    keys.dedup();
    let mut out = Vec::with_capacity(keys.len() * 2);
    for k in keys {
        out.push((k >> 32) as u32);
        out.push(k as u32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges_are_unique_and_ordered() {
        let e = edge_indices(&[0, 1, 2, 0, 2, 3]);
        assert_eq!(e, vec![0, 1, 0, 2, 0, 3, 1, 2, 2, 3]);
        assert_eq!(bounds(&[]), ([0.0; 3], [0.0; 3]));
        assert_eq!(
            bounds(&[1.0, 2.0, 3.0, -1.0, 5.0, 0.0]),
            ([-1.0, 2.0, 0.0], [1.0, 5.0, 3.0])
        );
    }
}
