// SPDX-License-Identifier: Apache-2.0
// Value -> colour, shared by the surface and the image shaders. The including shader
// declares `u` (with `range`, `colormap`, `steps`, `scale_flags`, `below`, `above`,
// `nan`) and `lut`: the colormap atlas, 256 texels wide, one row per colormap. Mirrors
// `ColorScale` in colormap.rs.
//
// scale_flags: bit0 reverse, bit1 log scale, bit2 `below` set, bit3 `above` set.

fn lut_at(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0) * 255.0;
    let i = i32(floor(x));
    let j = min(i + 1, 255);
    let row = i32(u.colormap);
    let a = textureLoad(lut, vec2<i32>(i, row), 0).rgb;
    let b = textureLoad(lut, vec2<i32>(j, row), 0).rgb;
    return mix(a, b, x - f32(i));
}

// rgb and, in w, whether the value has a colour at all (0 for a NaN whose colour is
// transparent: the caller draws its background).
fn scale_color(v: f32) -> vec4<f32> {
    // By its bits: compilers that assume finite maths drop `v != v`.
    if ((bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u) {
        return u.nan;
    }
    var f: f32;
    if ((u.scale_flags & 2u) != 0u) {
        if (v <= 0.0) {
            f = -1.0;
        } else {
            f = log(v / u.range.x) / log(u.range.y / u.range.x);
        }
    } else {
        let span = u.range.y - u.range.x;
        if (span == 0.0) {
            f = 0.5;
        } else {
            f = (v - u.range.x) / span;
        }
    }
    if (f < 0.0 && (u.scale_flags & 4u) != 0u) {
        return vec4<f32>(u.below.rgb, 1.0);
    }
    if (f > 1.0 && (u.scale_flags & 8u) != 0u) {
        return vec4<f32>(u.above.rgb, 1.0);
    }
    var t = clamp(f, 0.0, 1.0);
    if (u.steps >= 2u) {
        let n = f32(u.steps);
        t = (min(floor(t * n), n - 1.0) + 0.5) / n;
    }
    if ((u.scale_flags & 1u) != 0u) {
        t = 1.0 - t;
    }
    return vec4<f32>(lut_at(t), 1.0);
}
