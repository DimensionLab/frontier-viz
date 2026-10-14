// SPDX-License-Identifier: Apache-2.0
//! Regular-grid fields as images: each cell is one `R32Float` texel, the fragment shader
//! maps it through the colormap and a [`ImageView`] (centre + zoom) pans and zooms inside
//! the viewport without touching the texture. A new time step is one texture write.
//!
//! One renderer holds several texture slots and draws any number of [`ImageLayer`]s into
//! one target in a single pass: a single field, a split view (two layers with the same
//! viewport and complementary scissors), a triptych (three viewports) or a grid of small
//! multiples.

use crate::colormap::{ColorScale, Colormap};
use crate::lut::{ColormapAtlas, ScaleUniform};
use crate::offscreen::RgbaFrame;
use bytemuck::{Pod, Zeroable};

/// Pan / zoom state of an image view. `center` is the image-space point (0..1, y down)
/// shown at the middle of the viewport; `zoom` 1 fits the whole image inside it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageView {
    pub center: [f32; 2],
    pub zoom: f32,
}

impl Default for ImageView {
    fn default() -> Self {
        Self {
            center: [0.5, 0.5],
            zoom: 1.0,
        }
    }
}

impl ImageView {
    pub const MIN_ZOOM: f32 = 0.5;
    pub const MAX_ZOOM: f32 = 256.0;

    /// Size of the image in units of the viewport height, `(w, h)`.
    fn extent(&self, viewport_aspect: f32, image_aspect: f32) -> (f32, f32) {
        let ia = image_aspect.max(1e-6);
        let h = self.zoom * (viewport_aspect / ia).min(1.0);
        (h * ia, h)
    }

    /// Image uv (0..1, y down) under the viewport position `(u, v)` (0..1, y down). The
    /// same mapping as `image_uv` in `image.wgsl`.
    pub fn image_uv(&self, u: f32, v: f32, viewport_aspect: f32, image_aspect: f32) -> [f32; 2] {
        let (w, h) = self.extent(viewport_aspect, image_aspect);
        [
            (u - 0.5) * viewport_aspect / w + self.center[0],
            (v - 0.5) / h + self.center[1],
        ]
    }

    /// Cell under a viewport pixel, `None` outside the image. `(nx, ny)` is the grid size;
    /// the returned `y` counts from the bottom (the grid's own y axis points up).
    pub fn cell_at(
        &self,
        px: f32,
        py: f32,
        viewport: (f32, f32),
        image_aspect: f32,
        (nx, ny): (usize, usize),
    ) -> Option<(usize, usize)> {
        let (vw, vh) = (viewport.0.max(1.0), viewport.1.max(1.0));
        let [u, v] = self.image_uv(px / vw, py / vh, vw / vh, image_aspect);
        if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) || nx == 0 || ny == 0 {
            return None;
        }
        let x = ((u * nx as f32) as usize).min(nx - 1);
        let row = ((v * ny as f32) as usize).min(ny - 1);
        Some((x, ny - 1 - row))
    }

    /// Drag by `(dx, dy)` viewport pixels: the image follows the pointer.
    pub fn pan(&mut self, dx: f32, dy: f32, viewport: (f32, f32), image_aspect: f32) {
        let (vw, vh) = (viewport.0.max(1.0), viewport.1.max(1.0));
        let (w, h) = self.extent(vw / vh, image_aspect);
        self.center[0] -= dx / vh / w;
        self.center[1] -= dy / vh / h;
        self.clamp_center();
    }

    /// Zoom by `factor` keeping the image point under the viewport pixel `(px, py)` fixed.
    pub fn zoom_at(
        &mut self,
        factor: f32,
        px: f32,
        py: f32,
        viewport: (f32, f32),
        image_aspect: f32,
    ) {
        let (vw, vh) = (viewport.0.max(1.0), viewport.1.max(1.0));
        let (u, v) = (px / vw, py / vh);
        let before = self.image_uv(u, v, vw / vh, image_aspect);
        self.zoom = (self.zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        let after = self.image_uv(u, v, vw / vh, image_aspect);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
        self.clamp_center();
    }

    /// Keep at least the image centre line in view so the image cannot be lost.
    fn clamp_center(&mut self) {
        self.center[0] = self.center[0].clamp(0.0, 1.0);
        self.center[1] = self.center[1].clamp(0.0, 1.0);
    }
}

/// Reorder a grid frame `(X, Y)` stored row-major with x first (the dataset layout:
/// `values[x * ny + y]`) into texture rows from the top (`out[row * nx + x]`, row 0 = the
/// largest y), which is what [`FieldImageRenderer::upload`] expects.
pub fn grid_to_texels(values: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let mut out = vec![f32::NAN; nx * ny];
    if values.len() < nx * ny {
        return out;
    }
    for row in 0..ny {
        let y = ny - 1 - row;
        let dst = &mut out[row * nx..(row + 1) * nx];
        for (x, d) in dst.iter_mut().enumerate() {
            *d = values[x * ny + y];
        }
    }
    out
}

/// CPU reference of the shader's colour mapping for texel-ordered values (nearest, no
/// lighting): legends, tests and non-GPU fallbacks. NaN becomes transparent.
pub fn colorize(
    texels: &[f32],
    width: u32,
    height: u32,
    scale: impl Into<ColorScale>,
    range: (f32, f32),
) -> RgbaFrame {
    let scale = scale.into();
    let byte = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut data = Vec::with_capacity(texels.len() * 4);
    for &v in texels.iter().take((width * height) as usize) {
        let c = scale.color(v, range);
        if c[3] == 0.0 {
            data.extend_from_slice(&[0, 0, 0, 0]);
        } else {
            data.extend_from_slice(&[byte(c[0]), byte(c[1]), byte(c[2]), 255]);
        }
    }
    data.resize((width * height * 4) as usize, 0);
    RgbaFrame {
        width,
        height,
        data,
    }
}

/// One image to draw: texture slot `slot` into `viewport` (`x, y, w, h` in target pixels),
/// optionally clipped to `scissor` (split views share a viewport and split the scissor).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageLayer {
    pub slot: usize,
    pub viewport: (f32, f32, f32, f32),
    pub scissor: Option<(u32, u32, u32, u32)>,
    pub view: ImageView,
    /// Colormap, bands, log scale and the colours of out-of-range and NaN values.
    pub scale: ColorScale,
    pub range: (f32, f32),
    /// Bilinear interpolation between cell centres instead of flat cells.
    pub smooth: bool,
    /// Outline cells once they are magnified enough to be told apart.
    pub cell_grid: bool,
    /// Physical aspect of the image (`nx * dx / (ny * dy)`); `None` = texel aspect.
    pub aspect: Option<f32>,
}

impl ImageLayer {
    pub fn new(slot: usize, viewport: (f32, f32, f32, f32)) -> Self {
        Self {
            slot,
            viewport,
            scissor: None,
            view: ImageView::default(),
            scale: ColorScale::from(Colormap::Viridis),
            range: (0.0, 1.0),
            smooth: false,
            cell_grid: true,
            aspect: None,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view: [f32; 4],
    size: [f32; 4],
    range: [f32; 2],
    colormap: u32,
    flags: u32,
    background: [f32; 4],
    below: [f32; 4],
    above: [f32; 4],
    nan: [f32; 4],
    steps: u32,
    scale_flags: u32,
    _pad: [u32; 2],
}

struct Slot {
    texture: Option<wgpu::Texture>,
    size: (u32, u32),
    uniforms: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
}

pub struct FieldImageRenderer {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    slots: Vec<Slot>,
    atlas: ColormapAtlas,
    /// Texture uploads and bytes since creation (the viewer reports these in the HUD).
    pub uploads: u64,
    pub upload_bytes: u64,
}

impl FieldImageRenderer {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frontier-viz field image"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", include_str!("image.wgsl"), crate::lut::WGSL).into(),
            ),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("field image"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    // R32Float is not filterable without an optional feature: the shader
                    // uses textureLoad and interpolates itself.
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                ColormapAtlas::layout_entry(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("field image"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("field image"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            format,
            pipeline,
            layout,
            slots: Vec::new(),
            atlas: ColormapAtlas::new(device),
            uploads: 0,
            upload_bytes: 0,
        }
    }

    fn ensure_slot(&mut self, device: &wgpu::Device, slot: usize) {
        while self.slots.len() <= slot {
            self.slots.push(Slot {
                texture: None,
                size: (0, 0),
                uniforms: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("field image uniforms"),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                bind_group: None,
            });
        }
    }

    /// Size of the texture in a slot, `None` when nothing was uploaded.
    pub fn slot_size(&self, slot: usize) -> Option<(u32, u32)> {
        self.slots
            .get(slot)
            .filter(|s| s.texture.is_some())
            .map(|s| s.size)
    }

    pub fn clear_slot(&mut self, slot: usize) {
        if let Some(s) = self.slots.get_mut(slot) {
            s.texture = None;
            s.bind_group = None;
            s.size = (0, 0);
        }
    }

    /// Write `texels` (`width * height` values, rows from the top; see
    /// [`grid_to_texels`]) into a slot. The texture is reused while the size stays the
    /// same, so stepping through time is a single `write_texture`.
    pub fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: usize,
        width: u32,
        height: u32,
        texels: &[f32],
    ) {
        if width == 0 || height == 0 || texels.len() < (width * height) as usize {
            return;
        }
        self.ensure_slot(device, slot);
        let s = &mut self.slots[slot];
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        if s.texture.is_none() || s.size != (width, height) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("field image"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            s.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("field image"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: s.uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(self.atlas.view()),
                    },
                ],
            }));
            s.texture = Some(texture);
            s.size = (width, height);
        }
        let Some(texture) = &s.texture else { return };
        let bytes: &[u8] = bytemuck::cast_slice(&texels[..(width * height) as usize]);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            extent,
        );
        self.uploads += 1;
        self.upload_bytes += bytes.len() as u64;
    }

    #[allow(clippy::too_many_arguments)]
    /// Draw `layers` into `target` (`width` x `height` px) in one pass, cleared to
    /// `background` (a display colour, like the colormap) first. Layers whose slot is
    /// empty are skipped.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        layers: &[ImageLayer],
        background: [f64; 4],
    ) {
        self.render_over(
            device, queue, target, width, height, layers, background, true,
        );
    }

    /// Like [`Self::render`], but with `clear = false` the target keeps what is already
    /// in it (images drawn next to meshes in one frame). `background` still fills the
    /// part of each layer's viewport the image does not cover.
    #[allow(clippy::too_many_arguments)]
    pub fn render_over(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        width: u32,
        height: u32,
        layers: &[ImageLayer],
        background: [f64; 4],
        clear: bool,
    ) {
        let (width, height) = (width.max(1), height.max(1));
        self.atlas.sync(queue);
        let srgb = self.format.is_srgb();
        let srgb_flag = u32::from(srgb);
        // `background` is a display colour too.
        let lin = |c: f64| {
            if srgb {
                if c <= 0.04045 {
                    c / 12.92
                } else {
                    ((c + 0.055) / 1.055).powf(2.4)
                }
            } else {
                c
            }
        };
        // Two layers may share a slot (split views of one texture are not a thing, but
        // small multiples of the same sample are): uniforms are per slot, so the last
        // layer of a slot would win. Callers give each layer its own slot.
        for layer in layers {
            let Some(s) = self.slots.get(layer.slot) else {
                continue;
            };
            if s.texture.is_none() {
                continue;
            }
            let (_, _, w, h) = layer.viewport;
            let texel_aspect = s.size.0 as f32 / s.size.1.max(1) as f32;
            let mut flags = srgb_flag;
            if layer.smooth {
                flags |= 2;
            }
            if layer.cell_grid {
                flags |= 4;
            }
            let scale = ScaleUniform::new(&layer.scale, layer.range);
            let u = Uniforms {
                view: [
                    layer.view.center[0],
                    layer.view.center[1],
                    layer.view.zoom,
                    w / h.max(1.0),
                ],
                size: [
                    s.size.0 as f32,
                    s.size.1 as f32,
                    layer.aspect.unwrap_or(texel_aspect),
                    0.0,
                ],
                range: [layer.range.0, layer.range.1],
                colormap: scale.colormap,
                flags,
                background: [
                    background[0] as f32,
                    background[1] as f32,
                    background[2] as f32,
                    background[3] as f32,
                ],
                below: scale.below,
                above: scale.above,
                nan: scale.nan,
                steps: scale.steps,
                scale_flags: scale.flags,
                _pad: [0; 2],
            };
            queue.write_buffer(&s.uniforms, 0, bytemuck::bytes_of(&u));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("field image frame"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("field image"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if clear {
                            wgpu::LoadOp::Clear(wgpu::Color {
                                r: lin(background[0]),
                                g: lin(background[1]),
                                b: lin(background[2]),
                                a: background[3],
                            })
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            for layer in layers {
                let Some(Slot {
                    bind_group: Some(bind_group),
                    ..
                }) = self.slots.get(layer.slot)
                else {
                    continue;
                };
                let (x, y, w, h) = layer.viewport;
                if w < 1.0 || h < 1.0 || x >= width as f32 || y >= height as f32 {
                    continue;
                }
                // Scissor = the requested clip (or the viewport) intersected with the target.
                let (cx, cy, cw, ch) = match layer.scissor {
                    Some((sx, sy, sw, sh)) => (sx as f32, sy as f32, sw as f32, sh as f32),
                    None => (x, y, w, h),
                };
                let sx = cx.max(0.0).floor();
                let sy = cy.max(0.0).floor();
                let sw = ((cx + cw).min(width as f32) - sx).floor();
                let sh = ((cy + ch).min(height as f32) - sy).floor();
                if sw < 1.0 || sh < 1.0 {
                    continue;
                }
                pass.set_viewport(x, y, w, h, 0.0, 1.0);
                pass.set_scissor_rect(sx as u32, sy as u32, sw as u32, sh as u32);
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texel_order_puts_the_largest_y_in_the_first_row() {
        // 3 x 2 grid, values[x * ny + y] = 10 * x + y
        let v: Vec<f32> = (0..3)
            .flat_map(|x| (0..2).map(move |y| (10 * x + y) as f32))
            .collect();
        let t = grid_to_texels(&v, 3, 2);
        assert_eq!(t, vec![1.0, 11.0, 21.0, 0.0, 10.0, 20.0]);
        assert!(grid_to_texels(&v[..4], 3, 2).iter().all(|x| x.is_nan()));
    }

    #[test]
    fn fit_view_maps_viewport_corners_to_the_letterboxed_image() {
        let v = ImageView::default();
        // Image twice as wide as tall in a square viewport: full width, half the height.
        let [u0, v0] = v.image_uv(0.0, 0.25, 1.0, 2.0);
        let [u1, v1] = v.image_uv(1.0, 0.75, 1.0, 2.0);
        assert!((u0 - 0.0).abs() < 1e-6 && (v0 - 0.0).abs() < 1e-6);
        assert!((u1 - 1.0).abs() < 1e-6 && (v1 - 1.0).abs() < 1e-6);
        // Above the letterbox there is no cell; inside, y counts from the bottom.
        assert_eq!(v.cell_at(50.0, 10.0, (100.0, 100.0), 2.0, (8, 4)), None);
        assert_eq!(
            v.cell_at(1.0, 26.0, (100.0, 100.0), 2.0, (8, 4)),
            Some((0, 3))
        );
        assert_eq!(
            v.cell_at(99.0, 74.0, (100.0, 100.0), 2.0, (8, 4)),
            Some((7, 0))
        );
    }

    #[test]
    fn zoom_keeps_the_point_under_the_cursor_and_pan_follows_the_pointer() {
        let mut v = ImageView::default();
        let vp = (400.0, 200.0);
        let before = v.image_uv(300.0 / 400.0, 50.0 / 200.0, 2.0, 2.0);
        v.zoom_at(4.0, 300.0, 50.0, vp, 2.0);
        let after = v.image_uv(300.0 / 400.0, 50.0 / 200.0, 2.0, 2.0);
        assert!((v.zoom - 4.0).abs() < 1e-6);
        assert!((before[0] - after[0]).abs() < 1e-5 && (before[1] - after[1]).abs() < 1e-5);
        // Dragging right by a quarter of the visible width moves the centre left by it.
        let c0 = v.center;
        v.pan(100.0, 0.0, vp, 2.0);
        assert!((c0[0] - v.center[0] - 0.25 / 4.0).abs() < 1e-5);
        v.zoom_at(1e9, 0.0, 0.0, vp, 2.0);
        assert_eq!(v.zoom, ImageView::MAX_ZOOM);
        v.pan(-1e9, -1e9, vp, 2.0);
        assert_eq!(v.center, [1.0, 1.0]);
    }

    #[test]
    fn cpu_colorize_matches_the_colormap_and_hides_nan() {
        let f = colorize(
            &[0.0, 1.0, f32::NAN, 0.5],
            2,
            2,
            Colormap::Coolwarm,
            (0.0, 1.0),
        );
        assert_eq!(&f.data[0..3], &Colormap::Coolwarm.sample_u8(0.0));
        assert_eq!(&f.data[4..7], &Colormap::Coolwarm.sample_u8(1.0));
        assert_eq!(&f.data[8..12], &[0, 0, 0, 0]);
        assert_eq!(f.data[15], 255);
    }

    /// The shader's value -> colour path against `ColorScale` on the CPU: every built-in
    /// map, a custom map from stops, bands, reverse, a log scale and the colours of values
    /// outside the range and of NaN.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_colour_scale_matches_the_cpu_reference() {
        use crate::GpuContext;
        use crate::colormap::{CustomMap, Interp, Stop, registry};
        use crate::offscreen::{OFFSCREEN_FORMAT, OffscreenTarget};
        let Ok(ctx) = GpuContext::new_headless() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        // 8 x 1 cells: below, the range in steps, above, NaN.
        let values = [-0.5f32, 0.01, 0.2, 0.45, 0.7, 1.0, 1.8, f32::NAN];
        let (nx, w, h) = (values.len() as u32, 160u32, 20u32);
        let mut r = FieldImageRenderer::new(&ctx.device, OFFSCREEN_FORMAT);
        r.upload(&ctx.device, &ctx.queue, 0, nx, 1, &values);
        let target = OffscreenTarget::new(&ctx.device, w, h);
        let slot = registry::upsert(CustomMap {
            id: "gpu-test-map".into(),
            name: "GPU test".into(),
            interp: Interp::Oklab,
            stops: vec![
                Stop {
                    pos: 0.0,
                    color: [0.0, 0.0, 0.4],
                },
                Stop {
                    pos: 0.5,
                    color: [1.0, 1.0, 1.0],
                },
                Stop {
                    pos: 1.0,
                    color: [0.6, 0.0, 0.0],
                },
            ],
        })
        .expect("a free custom slot");
        let mut scales: Vec<(ColorScale, (f32, f32))> = Colormap::BUILTIN
            .iter()
            .map(|m| (ColorScale::from(*m), (0.0, 1.0)))
            .collect();
        let custom = ColorScale::from(Colormap::Custom(slot));
        scales.push((custom, (0.0, 1.0)));
        scales.push((ColorScale { steps: 4, ..custom }, (0.0, 1.0)));
        scales.push((
            ColorScale {
                reverse: true,
                steps: 5,
                ..ColorScale::from(Colormap::Turbo)
            },
            (0.0, 1.0),
        ));
        scales.push((
            ColorScale {
                log: true,
                ..ColorScale::from(Colormap::Magma)
            },
            (0.01, 1.0),
        ));
        scales.push((
            ColorScale {
                below: Some([0.0, 0.0, 1.0]),
                above: Some([1.0, 0.0, 0.0]),
                nan: [1.0, 0.0, 1.0, 1.0],
                ..ColorScale::from(Colormap::Greys)
            },
            (0.0, 1.0),
        ));
        let background = [0.2, 0.3, 0.4, 1.0];
        let mut seen = std::collections::HashSet::new();
        for (scale, range) in scales {
            let layer = ImageLayer {
                scale,
                range,
                cell_grid: false,
                aspect: Some(w as f32 / h as f32),
                ..ImageLayer::new(0, (0.0, 0.0, w as f32, h as f32))
            };
            r.render(
                &ctx.device,
                &ctx.queue,
                target.view(),
                w,
                h,
                &[layer],
                background,
            );
            let frame = target.read_back(&ctx).unwrap();
            let cell_w = w / nx;
            let mut row = Vec::new();
            for (i, v) in values.iter().enumerate() {
                let at = ((h / 2 * w + i as u32 * cell_w + cell_w / 2) * 4) as usize;
                let got = [frame.data[at], frame.data[at + 1], frame.data[at + 2]];
                let c = scale.color(*v, range);
                let want = if c[3] == 0.0 {
                    [51, 77, 102] // the background
                } else {
                    [0, 1, 2].map(|k| (c[k] * 255.0).round() as u8)
                };
                assert!(
                    got.iter().zip(&want).all(|(p, q)| p.abs_diff(*q) <= 3),
                    "{scale:?} at {v}: {got:?} vs {want:?}"
                );
                row.push(got);
            }
            seen.insert(row);
        }
        // every scale drew something different
        assert_eq!(seen.len(), Colormap::BUILTIN.len() + 5);
        // a custom map edited in place reaches the next frame
        registry::upsert(CustomMap {
            id: "gpu-test-map".into(),
            name: "GPU test".into(),
            interp: Interp::Srgb,
            stops: vec![
                Stop {
                    pos: 0.0,
                    color: [0.0, 1.0, 0.0],
                },
                Stop {
                    pos: 1.0,
                    color: [0.0, 1.0, 0.0],
                },
            ],
        });
        let layer = ImageLayer {
            scale: custom,
            cell_grid: false,
            aspect: Some(w as f32 / h as f32),
            ..ImageLayer::new(0, (0.0, 0.0, w as f32, h as f32))
        };
        r.render(
            &ctx.device,
            &ctx.queue,
            target.view(),
            w,
            h,
            &[layer],
            background,
        );
        let frame = target.read_back(&ctx).unwrap();
        let at = ((h / 2 * w + 3 * (w / nx) + 10) * 4) as usize;
        assert_eq!(&frame.data[at..at + 3], &[0, 255, 0]);
        registry::remove("gpu-test-map");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_image_matches_the_cpu_reference_and_splits_by_scissor() {
        use crate::GpuContext;
        use crate::offscreen::{OFFSCREEN_FORMAT, OffscreenTarget};
        let Ok(ctx) = GpuContext::new_headless() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        // 4 x 2 grid: a ramp along x on slot 0, a constant on slot 1.
        let (nx, ny) = (4usize, 2usize);
        let ramp: Vec<f32> = (0..nx)
            .flat_map(|x| (0..ny).map(move |_| x as f32 / 3.0))
            .collect();
        let mut r = FieldImageRenderer::new(&ctx.device, OFFSCREEN_FORMAT);
        r.upload(
            &ctx.device,
            &ctx.queue,
            0,
            nx as u32,
            ny as u32,
            &grid_to_texels(&ramp, nx, ny),
        );
        r.upload(&ctx.device, &ctx.queue, 1, nx as u32, ny as u32, &[1.0; 8]);
        assert_eq!((r.uploads, r.upload_bytes), (2, 64));
        assert_eq!(r.slot_size(0), Some((4, 2)));
        let (w, h) = (160u32, 80u32);
        let target = OffscreenTarget::new(&ctx.device, w, h);
        let full = (0.0, 0.0, w as f32, h as f32);
        let layer = |slot: usize, scissor| ImageLayer {
            scissor,
            cell_grid: false,
            ..ImageLayer::new(slot, full)
        };
        let px = |f: &RgbaFrame, x: u32, y: u32| -> [u8; 3] {
            let i = ((y * w + x) * 4) as usize;
            [f.data[i], f.data[i + 1], f.data[i + 2]]
        };
        let near = |a: [u8; 3], b: [u8; 3]| a.iter().zip(&b).all(|(p, q)| p.abs_diff(*q) <= 3);

        r.render(
            &ctx.device,
            &ctx.queue,
            target.view(),
            w,
            h,
            &[layer(0, None)],
            [0.0, 0.0, 0.0, 1.0],
        );
        let frame = target.read_back(&ctx).unwrap();
        // The image fills the 2:1 viewport; cell x is 40 px wide.
        for x in 0..nx as u32 {
            let want = Colormap::Viridis.sample_u8(x as f32 / 3.0);
            assert!(
                near(px(&frame, x * 40 + 20, 40), want),
                "cell {x}: {:?} vs {want:?}",
                px(&frame, x * 40 + 20, 40)
            );
        }

        // Split: left half slot 0, right half slot 1, same viewport.
        let layers = [
            layer(0, Some((0, 0, w / 2, h))),
            layer(1, Some((w / 2, 0, w / 2, h))),
        ];
        r.render(
            &ctx.device,
            &ctx.queue,
            target.view(),
            w,
            h,
            &layers,
            [0.0, 0.0, 0.0, 1.0],
        );
        let frame = target.read_back(&ctx).unwrap();
        assert!(near(px(&frame, 20, 40), Colormap::Viridis.sample_u8(0.0)));
        assert!(near(px(&frame, 140, 40), Colormap::Viridis.sample_u8(1.0)));
        assert!(
            near(px(&frame, 100, 40), Colormap::Viridis.sample_u8(1.0)),
            "right half shows slot 1"
        );

        // Zoomed 2x on the left edge: the first cell now covers the left half of the view.
        let mut zoomed = layer(0, None);
        zoomed.view = ImageView {
            center: [0.25, 0.5],
            zoom: 2.0,
        };
        r.render(
            &ctx.device,
            &ctx.queue,
            target.view(),
            w,
            h,
            &[zoomed],
            [0.0, 0.0, 0.0, 1.0],
        );
        let frame = target.read_back(&ctx).unwrap();
        assert!(near(px(&frame, 70, 40), Colormap::Viridis.sample_u8(0.0)));
        assert!(near(
            px(&frame, 90, 40),
            Colormap::Viridis.sample_u8(1.0 / 3.0)
        ));
    }
}
