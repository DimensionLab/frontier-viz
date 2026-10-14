// SPDX-License-Identifier: Apache-2.0
// Surface mesh: scalar field -> colormap, Lambert + rim lighting, optional wireframe pass.

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    light_dir: vec4<f32>,
    range: vec2<f32>,
    colormap: u32,
    flags: u32,       // bit0: srgb-encode output, bit1: wire pass, bit2: smooth (vertex) normals
    wire_offset: f32,
    steps: u32,       // colour bands (0: continuous)
    scale_flags: u32, // see colormap.wgsl
    _pad0: f32,
    below: vec4<f32>,
    above: vec4<f32>,
    nan: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var lut: texture_2d<f32>;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) scalar: f32,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) scalar: f32,
    @location(2) world: vec3<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    var p = in.pos;
    if ((u.flags & 2u) != 0u) {
        p = p + in.normal * u.wire_offset;
    }
    out.clip = u.view_proj * vec4<f32>(p, 1.0);
    if ((u.flags & 2u) != 0u) {
        // Wire pass: a small pull towards the camera so the lines sit on the fill
        // without a normal offset (moving meshes carry no normals).
        out.clip.z = out.clip.z - 0.0004 * out.clip.w;
    }
    out.normal = in.normal;
    out.scalar = in.scalar;
    out.world = p;
    return out;
}

fn encode(c: vec3<f32>) -> vec3<f32> {
    if ((u.flags & 1u) != 0u) {
        return pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    }
    return c;
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let mapped = scale_color(in.scalar);
    // A value without a colour (NaN left transparent) is drawn as neutral grey: the
    // surface is still there.
    let base = mix(vec3<f32>(0.5), mapped.rgb, mapped.a);
    let l = normalize(u.light_dir.xyz);
    let v = normalize(u.camera_pos.xyz - in.world);
    var n: vec3<f32>;
    if ((u.flags & 4u) != 0u) {
        // Smooth: area-weighted vertex normals (smears across creases on decimated LODs).
        n = normalize(in.normal);
        if (dot(n, v) < 0.0) {
            n = -n;
        }
    } else {
        // Flat: the face normal from screen-space derivatives of the world position, so
        // sliver triangles and sharp edges of decimated meshes shade correctly. Always
        // flipped towards the camera (meshes are not consistently wound).
        n = normalize(cross(dpdx(in.world), dpdy(in.world)));
        if (dot(n, v) < 0.0) {
            n = -n;
        }
    }
    let diffuse = max(dot(n, l), 0.0);
    let h = normalize(l + v);
    let spec = pow(max(dot(n, h), 0.0), 48.0) * 0.18;
    let rim = pow(1.0 - max(dot(n, v), 0.0), 3.0) * 0.15;
    var lit = base * (0.42 + 0.58 * diffuse) + vec3<f32>(spec + rim);
    if (!front) {
        // Inside of the surface (visible through holes / open boundaries): darker and
        // desaturated so gaps read as gaps instead of bright patches.
        let grey = dot(lit, vec3<f32>(0.3, 0.59, 0.11));
        lit = mix(lit, vec3<f32>(grey), 0.6) * 0.35;
    }
    return vec4<f32>(encode(lit), 1.0);
}

@fragment
fn fs_wire(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(encode(vec3<f32>(0.04, 0.05, 0.07)), 0.85);
}
