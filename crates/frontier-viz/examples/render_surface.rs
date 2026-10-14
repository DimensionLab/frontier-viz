// SPDX-License-Identifier: Apache-2.0
//! Render an Arrow surface (as served by `GET .../surface`) to a PNG, headless, with the
//! same per-field defaults as the app (pressure -> coolwarm, symmetric range).
//!
//!   cargo run -p frontier-viz --features png --example render_surface -- surf.arrow out.png [field] [w] [h] [--smooth] [--wire]

use frontier_viz::field::{defaults_for, fmt_tick, range_for};
use frontier_viz::offscreen::{OFFSCREEN_FORMAT, OffscreenTarget, encode_png};
use frontier_viz::{GpuContext, OrbitCamera, RenderSettings, SurfaceRenderer, arrow};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flags: Vec<&String> = args.iter().filter(|a| a.starts_with("--")).collect();
    let args: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let input = args
        .get(1)
        .expect("usage: render_surface <arrow> <png> [field] [w] [h]");
    let output = args.get(2).expect("output png");
    let field = args
        .get(3)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "p".into());
    let w: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1200);
    let h: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(800);
    let smooth = flags.iter().any(|f| *f == "--smooth");
    let wireframe = flags.iter().any(|f| *f == "--wire");

    let bytes = std::fs::read(input).expect("read input");
    let mesh = arrow::decode(&bytes).expect("decode arrow surface");
    let name = mesh.resolve_field(&field).expect("field present");
    let scalar = mesh.point_field(&name).expect("point field");
    let (colormap, mode) = defaults_for(&name);
    let range = range_for(&mesh, &name, mode);
    eprintln!(
        "faces={} points={} field={name} colormap={} range=({}, {}) shading={}",
        mesh.faces.len(),
        mesh.points.len(),
        colormap.as_str(),
        fmt_tick(range.0),
        fmt_tick(range.1),
        if smooth { "smooth" } else { "flat" }
    );

    let ctx = GpuContext::new_headless().expect("gpu");
    let mut r = SurfaceRenderer::new(&ctx.device, OFFSCREEN_FORMAT, 4);
    r.upload(&ctx.device, &mesh, &scalar);
    let mut cam = OrbitCamera::z_up();
    let (lo, hi) = mesh.bounds();
    cam.fit(lo, hi);
    let settings = RenderSettings {
        scale: colormap.into(),
        range,
        smooth,
        wireframe,
        ..Default::default()
    };
    let target = OffscreenTarget::new(&ctx.device, w, h);
    let frame = target
        .render(&ctx, &mut r, &cam, &settings)
        .expect("render");
    std::fs::write(output, encode_png(&frame).expect("png")).expect("write png");
}
