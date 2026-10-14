// SPDX-License-Identifier: Apache-2.0
//! Render a frame into CPU memory. Used by the desktop WebView fallback (the native
//! renderer cannot draw into wry's WebView, so frames are streamed as images) and for
//! headless thumbnails.

#[cfg(not(target_arch = "wasm32"))]
use crate::GpuContext;
#[cfg(not(target_arch = "wasm32"))]
use crate::camera::OrbitCamera;
#[cfg(not(target_arch = "wasm32"))]
use crate::renderer::{RenderSettings, SurfaceRenderer};

pub const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Tightly packed RGBA8 frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Owns the offscreen colour texture and readback buffer, reused across frames.
#[cfg(not(target_arch = "wasm32"))]
pub struct OffscreenTarget {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_row: u32,
}

#[cfg(not(target_arch = "wasm32"))]
impl OffscreenTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen colour"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let padded_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            width,
            height,
            texture,
            view,
            readback,
            padded_row,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The colour attachment, for callers that draw several passes (split views)
    /// before reading back with [`OffscreenTarget::render_region`].
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Render and read the frame back. Blocks on the GPU (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render(
        &self,
        ctx: &GpuContext,
        renderer: &mut SurfaceRenderer,
        camera: &OrbitCamera,
        settings: &RenderSettings,
    ) -> Result<RgbaFrame, crate::VizError> {
        self.render_region(ctx, renderer, camera, settings, 0, None)
    }

    /// Like [`OffscreenTarget::render`] but into a region (see
    /// [`SurfaceRenderer::render_region`]); the whole target is read back.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn render_region(
        &self,
        ctx: &GpuContext,
        renderer: &mut SurfaceRenderer,
        camera: &OrbitCamera,
        settings: &RenderSettings,
        slot: usize,
        region: Option<(u32, u32, u32, u32, bool)>,
    ) -> Result<RgbaFrame, crate::VizError> {
        renderer.render_region(
            &ctx.device,
            &ctx.queue,
            &self.view,
            self.width,
            self.height,
            camera,
            settings,
            slot,
            region,
        );
        self.read_back(ctx)
    }

    /// Copy whatever was last drawn into [`OffscreenTarget::view`] to CPU memory. Blocks
    /// on the GPU (native only). Any renderer can draw into the view first.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_back(&self, ctx: &GpuContext) -> Result<RgbaFrame, crate::VizError> {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(Some(encoder.finish()));
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| crate::VizError::Surface(e.to_string()))?;
        rx.recv()
            .map_err(|e| crate::VizError::Surface(e.to_string()))?
            .map_err(|e| crate::VizError::Surface(e.to_string()))?;
        let mut data = Vec::with_capacity((self.width * self.height * 4) as usize);
        {
            let view = slice
                .get_mapped_range()
                .map_err(|e| crate::VizError::Surface(e.to_string()))?;
            for row in 0..self.height {
                let start = (row * self.padded_row) as usize;
                data.extend_from_slice(&view[start..start + (self.width * 4) as usize]);
            }
        }
        self.readback.unmap();
        Ok(RgbaFrame {
            width: self.width,
            height: self.height,
            data,
        })
    }
}

/// Encode a frame as PNG (fast compression; this runs at interaction rate).
#[cfg(feature = "png")]
pub fn encode_png(frame: &RgbaFrame) -> Result<Vec<u8>, crate::VizError> {
    let mut out = Vec::with_capacity(frame.data.len() / 3);
    {
        let mut enc = png::Encoder::new(&mut out, frame.width, frame.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.set_compression(png::Compression::Fastest);
        let mut w = enc
            .write_header()
            .map_err(|e| crate::VizError::Surface(e.to_string()))?;
        w.write_image_data(&frame.data)
            .map_err(|e| crate::VizError::Surface(e.to_string()))?;
    }
    Ok(out)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::fixtures::ahmed_body;

    #[test]
    fn offscreen_frame_renders_the_fixture() {
        // Skipped when no adapter is available (CI without a GPU / software rasteriser).
        let Ok(ctx) = GpuContext::new_headless() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mesh = ahmed_body(0);
        let scalar = mesh.point_field("p").unwrap();
        let mut r = SurfaceRenderer::new(&ctx.device, OFFSCREEN_FORMAT, 1);
        r.upload(&ctx.device, &mesh, &scalar);
        let mut cam = OrbitCamera::default();
        let (lo, hi) = mesh.bounds();
        cam.fit(lo, hi);
        let settings = RenderSettings {
            range: mesh.field_range("p").unwrap(),
            ..Default::default()
        };
        let target = OffscreenTarget::new(&ctx.device, 160, 120);
        let frame = target.render(&ctx, &mut r, &cam, &settings).unwrap();
        assert_eq!(frame.data.len(), 160 * 120 * 4);
        // The centre pixel is on the body, so it is not the background colour.
        let c = (60 * 160 + 80) * 4;
        let px = &frame.data[c..c + 3];
        assert!(px != [14u8, 17, 22].as_slice(), "{px:?}");
    }

    #[test]
    fn scalar_update_recolours_without_reuploading_and_tiles_draw_each_slot() {
        let Ok(ctx) = GpuContext::new_headless() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mesh = ahmed_body(0);
        let n = mesh.points.len();
        let mut r = SurfaceRenderer::new(&ctx.device, OFFSCREEN_FORMAT, 1);
        r.upload_slot(&ctx.device, 0, &mesh, &vec![0.0; n]);
        r.upload_slot(&ctx.device, 1, &mesh, &vec![1.0; n]);
        assert_eq!(r.loaded_slots(), 2);
        let mut cam = OrbitCamera::z_up();
        let (lo, hi) = mesh.bounds();
        cam.fit(lo, hi);
        let settings = RenderSettings::default();
        let (w, h) = (240u32, 100u32);
        let target = OffscreenTarget::new(&ctx.device, w, h);
        let centre = |f: &RgbaFrame, cx: u32| -> [u8; 3] {
            let i = ((50 * w + cx) * 4) as usize;
            [f.data[i], f.data[i + 1], f.data[i + 2]]
        };
        let tile = |slot: usize, x: f32| crate::renderer::Tile {
            slot,
            viewport: (x, 0.0, 120.0, 100.0),
            camera: cam,
            settings,
        };
        // Slot 2 was never uploaded: its tile is skipped, not an error.
        let tiles = [tile(0, 0.0), tile(1, 120.0), tile(2, 0.0)];
        let tris = r.render_tiles(
            &ctx.device,
            &ctx.queue,
            target.view(),
            w,
            h,
            &tiles,
            settings.background,
        );
        assert_eq!(tris, 2 * mesh.faces.len() as u64);
        let frame = target.read_back(&ctx).unwrap();
        let (left, right) = (centre(&frame, 60), centre(&frame, 180));
        // viridis(0) is dark purple, viridis(1) is yellow: the two tiles differ strongly.
        assert!(
            right[0] > left[0].saturating_add(60),
            "{left:?} vs {right:?}"
        );

        // A new time step for slot 0: only the scalar buffer is written.
        assert!(r.update_scalar(&ctx.queue, 0, &vec![1.0; n]));
        assert!(
            !r.update_scalar(&ctx.queue, 0, &vec![1.0; n - 1]),
            "length mismatch is refused"
        );
        assert!(
            !r.update_scalar(&ctx.queue, 7, &vec![1.0; n]),
            "empty slot is refused"
        );
        r.render_tiles(
            &ctx.device,
            &ctx.queue,
            target.view(),
            w,
            h,
            &tiles[..2],
            settings.background,
        );
        let frame = target.read_back(&ctx).unwrap();
        assert_eq!(centre(&frame, 60), centre(&frame, 180));
        assert_eq!(centre(&frame, 180), right);

        r.clear_slot(1);
        assert_eq!(r.loaded_slots(), 1);
    }
}
