// SPDX-License-Identifier: Apache-2.0
//! The colormap atlas on the GPU: one texture row per colormap (built-in, then custom),
//! sampled by `colormap.wgsl`. A renderer owns one and calls [`ColormapAtlas::sync`]
//! before drawing, which uploads again when the custom maps changed.

use crate::colormap::{self, ColorScale, LUT_SIZE, registry};

/// `scale_color` and `lut_at`, appended to the shaders that colour by value.
pub const WGSL: &str = include_str!("colormap.wgsl");

pub struct ColormapAtlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// Registry generation the texture holds; `None` before the first upload.
    uploaded: Option<u64>,
}

impl ColormapAtlas {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormap atlas"),
            size: wgpu::Extent3d {
                width: LUT_SIZE as u32,
                height: colormap::atlas_rows(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Display colours as stored: the shaders treat them like the old polynomials.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            uploaded: None,
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Upload the atlas when it was never uploaded or the custom maps changed since.
    pub fn sync(&mut self, queue: &wgpu::Queue) {
        let generation = registry::generation();
        if self.uploaded == Some(generation) {
            return;
        }
        let (bytes, rows) = colormap::atlas();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(LUT_SIZE as u32 * 4),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: LUT_SIZE as u32,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        self.uploaded = Some(generation);
    }

    /// The bind group layout entry of the atlas at `binding` (fragment stage).
    pub fn layout_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }
}

/// The uniform fields of a [`ColorScale`]: what `colormap.wgsl` reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleUniform {
    pub colormap: u32,
    pub steps: u32,
    pub flags: u32,
    pub below: [f32; 4],
    pub above: [f32; 4],
    pub nan: [f32; 4],
}

impl ScaleUniform {
    pub fn new(scale: &ColorScale, range: (f32, f32)) -> Self {
        let rgb1 = |c: Option<[f32; 3]>| c.map_or([0.0; 4], |c| [c[0], c[1], c[2], 1.0]);
        let mut flags = 0u32;
        if scale.reverse {
            flags |= 1;
        }
        if scale.log && ColorScale::log_valid(range) {
            flags |= 2;
        }
        if scale.below.is_some() {
            flags |= 4;
        }
        if scale.above.is_some() {
            flags |= 8;
        }
        Self {
            colormap: scale.map.index(),
            steps: if scale.steps >= 2 { scale.steps } else { 0 },
            flags,
            below: rgb1(scale.below),
            above: rgb1(scale.above),
            nan: scale.nan,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colormap::Colormap;

    #[test]
    fn scale_uniform_carries_the_flags_the_shader_reads() {
        let plain = ScaleUniform::new(&ColorScale::from(Colormap::Coolwarm), (0.0, 1.0));
        assert_eq!(plain.colormap, Colormap::Coolwarm.index());
        assert_eq!((plain.steps, plain.flags), (0, 0));
        let scale = ColorScale {
            map: Colormap::Turbo,
            reverse: true,
            steps: 8,
            log: true,
            below: Some([0.0, 0.0, 1.0]),
            above: None,
            nan: [1.0, 0.0, 1.0, 1.0],
        };
        let u = ScaleUniform::new(&scale, (0.1, 10.0));
        assert_eq!((u.steps, u.flags), (8, 1 | 2 | 4));
        assert_eq!(u.below, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(u.nan, [1.0, 0.0, 1.0, 1.0]);
        // a log scale needs a positive range; one band is no banding
        let u = ScaleUniform::new(&ColorScale { steps: 1, ..scale }, (-1.0, 10.0));
        assert_eq!((u.steps, u.flags & 2), (0, 0));
    }
}
