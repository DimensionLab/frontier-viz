// SPDX-License-Identifier: Apache-2.0
// Regular-grid scalar field: one R32Float texel per cell -> colormap, with pan / zoom.

struct Uniforms {
    // xy: image-space centre of the view (0..1), z: zoom (1 = fit), w: viewport aspect (w / h)
    view: vec4<f32>,
    // xy: texture size in texels, z: physical aspect of the image (width / height), w: unused
    size: vec4<f32>,
    range: vec2<f32>,
    colormap: u32,
    flags: u32,        // bit0: target is sRGB (linearise), bit1: bilinear, bit2: cell grid when magnified
    background: vec4<f32>,
    below: vec4<f32>,
    above: vec4<f32>,
    nan: vec4<f32>,
    steps: u32,        // colour bands (0: continuous)
    scale_flags: u32,  // see colormap.wgsl
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var field: texture_2d<f32>;
@group(0) @binding(2) var lut: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    // One triangle covering the viewport.
    var out: VsOut;
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    out.clip = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

// Colormap values are display (sRGB) colours and there is no lighting here, so the image
// must reach the screen unchanged: an sRGB target encodes on write, so linearise first.
fn encode(c: vec3<f32>) -> vec3<f32> {
    if ((u.flags & 1u) != 0u) {
        let x = max(c, vec3<f32>(0.0));
        let lo = x / 12.92;
        let hi = pow((x + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
        return select(hi, lo, x <= vec3<f32>(0.04045));
    }
    return c;
}

fn texel(p: vec2<i32>) -> f32 {
    let hi = vec2<i32>(i32(u.size.x) - 1, i32(u.size.y) - 1);
    return textureLoad(field, clamp(p, vec2<i32>(0), hi), 0).r;
}

// Screen uv (0..1, y down) -> image uv (0..1, y down). Mirrors `ImageView::image_uv`.
fn image_uv(uv: vec2<f32>) -> vec2<f32> {
    let va = u.view.w;
    let ia = max(u.size.z, 1e-6);
    let h = u.view.z * min(1.0, va / ia);
    let w = h * ia;
    let p = (uv - vec2<f32>(0.5)) * vec2<f32>(va, 1.0);
    return p / vec2<f32>(w, h) + u.view.xy;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let q = image_uv(in.uv);
    let t = q * u.size.xy;
    // Derivatives must be taken before any branch (WGSL requires uniform control flow).
    let px = fwidth(t);
    let bg = vec4<f32>(encode(u.background.rgb), u.background.a);
    if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) {
        return bg;
    }
    var v: f32;
    if ((u.flags & 2u) != 0u) {
        let f = t - vec2<f32>(0.5);
        let i = vec2<i32>(floor(f));
        let w = f - floor(f);
        let a = texel(i);
        let b = texel(i + vec2<i32>(1, 0));
        let c = texel(i + vec2<i32>(0, 1));
        let d = texel(i + vec2<i32>(1, 1));
        v = mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
    } else {
        v = texel(vec2<i32>(floor(t)));
    }
    // NaN (masked cells, e.g. inside a solid) shows as background unless it has a colour.
    let mapped = scale_color(v);
    if (mapped.a == 0.0) {
        return bg;
    }
    var rgb = mapped.rgb;
    if ((u.flags & 4u) != 0u) {
        // Cell outlines once a cell is larger than ~12 px: makes the resolution visible.
        if (px.x < 0.085 && px.y < 0.085) {
            let g = abs(fract(t) - vec2<f32>(0.5));
            let edge = max(g.x + px.x, g.y + px.y);
            if (edge > 0.5) {
                rgb = mix(rgb, vec3<f32>(0.0), 0.28);
            }
        }
    }
    return vec4<f32>(encode(rgb), 1.0);
}
