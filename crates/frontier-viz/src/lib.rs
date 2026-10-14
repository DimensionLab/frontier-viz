// SPDX-License-Identifier: Apache-2.0
//! Rendering core.
//!
//! Scales from "no GPU" (wgpu's software/`Gl` fallback or CPU reduction paths) to a
//! single GPU (desktop app / browser via WebGPU) to multi-GPU clusters (server-side
//! rendering + video streaming, planned). The same crate compiles to wasm for the web
//! build of the Dioxus app and natively for the desktop app and the headless server.
//!
//! Host-agnostic by design: nothing here knows about Dioxus. Hosts give the renderer a
//! `wgpu::TextureView` (a `<canvas>` surface on web, an offscreen texture on desktop /
//! headless) and call [`SurfaceRenderer::render`].
//!
//! - [`surface`]: `SurfaceMesh` (points / faces / scalar fields), normals, LOD, picking.
//! - [`fixtures`]: generated Ahmed-body mesh with a synthetic pressure field (offline demo).
//! - [`colormap`]: the colormaps (built-in tables, custom maps from colour stops), value to
//!   colour ([`ColorScale`]: bands, log scale, out-of-range and NaN colours), import / export.
//! - [`lut`]: the colormaps as one GPU lookup texture, shared by the shaders below.
//! - [`field`]: per-field defaults (pressure -> coolwarm, symmetric range), legend ticks.
//! - [`camera`]: orbit camera and the small matrix math it needs.
//! - [`renderer`]: the wgpu pipeline (lit triangles + colormap + wireframe overlay), with
//!   mesh slots for split views and small multiples and a scalar-only update per time step.
//! - [`image`]: regular-grid fields as colormapped R32Float textures with pan / zoom.
//! - [`arrow`] (feature `arrow`): Arrow IPC stream -> `SurfaceMesh`.
//! - [`offscreen`]: render to an RGBA buffer (+ PNG with feature `png`).

pub mod camera;
pub mod colormap;
pub mod field;
pub mod fixtures;
pub mod image;
pub mod lut;
pub mod offscreen;
pub mod renderer;
pub mod surface;

#[cfg(feature = "arrow")]
pub mod arrow;

pub use camera::{Mat4, OrbitCamera, Vec3};
pub use colormap::{ColorScale, Colormap};
pub use field::RangeMode;
pub use image::{FieldImageRenderer, ImageLayer, ImageView};
pub use renderer::{RenderSettings, SurfaceRenderer, Tile};
pub use surface::{Hit, LodSet, SurfaceMesh};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VizError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    #[error("device request failed: {0}")]
    Device(String),
    #[error("surface error: {0}")]
    Surface(String),
    #[error("mesh decode error: {0}")]
    Decode(String),
}

/// Describes the GPU (or software fallback) a renderer will use.
#[derive(Debug, Clone)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}

/// Owned GPU context shared by all renderers.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a headless context, preferring a discrete GPU and falling back to
    /// whatever wgpu can find (including software rasterizers).
    pub fn new_headless() -> Result<Self, VizError> {
        pollster::block_on(Self::new_headless_async())
    }

    pub async fn new_headless_async() -> Result<Self, VizError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        Self::from_instance(instance, None).await
    }

    /// Create a context whose adapter is compatible with `surface` (web canvas path).
    pub async fn from_instance(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, VizError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                ..Default::default()
            })
            .await
            .map_err(|_| VizError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("frontier-viz"),
                ..Default::default()
            })
            .await
            .map_err(|e| VizError::Device(e.to_string()))?;
        tracing::info!(adapter = ?adapter.get_info().name, "wgpu context created");
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    pub fn summary(&self) -> AdapterSummary {
        let info = self.adapter.get_info();
        AdapterSummary {
            name: info.name,
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
        }
    }
}

/// Enumerate adapters without creating a device. Useful for `frontier doctor`.
pub fn enumerate_adapters() -> Vec<AdapterSummary> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .into_iter()
        .map(|a| {
            let info = a.get_info();
            AdapterSummary {
                name: info.name,
                backend: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
            }
        })
        .collect()
}
