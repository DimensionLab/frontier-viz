// SPDX-License-Identifier: Apache-2.0
//! Offline fixtures: an Ahmed-body surface (the AhmedML geometry) with a synthetic
//! pressure-coefficient field. Used when the server is unreachable and in tests.

use crate::surface::SurfaceMesh;
use std::collections::BTreeMap;
use std::f32::consts::PI;

/// Ahmed body (Ahmed et al. 1984) in metres: length 1.044, width 0.389, height 0.288,
/// 25 deg rear slant, front edges rounded with r = 0.1. The body sits with its length
/// along +x, width along z and height along y (y up, flow along -x so the nose faces +x).
///
/// `resolution` scales the tessellation: 0 ~ 3.5k faces, 1 ~ 15k, 2 ~ 55k, 3 ~ 110k.
pub fn ahmed_body(resolution: u8) -> SurfaceMesh {
    let (profile_segments, width_segments) = match resolution {
        0 => (48, 36),
        1 => (100, 76),
        2 => (190, 150),
        _ => (270, 210),
    };
    let length = 1.044f32;
    let height = 0.288f32;
    let width = 0.389f32;
    let r = 0.1f32;
    let slant_len = 0.222f32;
    let slant = 25f32.to_radians();

    // Closed 2D profile in the (x, y) plane, counter-clockwise starting at the bottom
    // rear corner, parameterised by arc length so tessellation is uniform.
    let x_rear = -length * 0.5;
    let x_front = length * 0.5;
    let y_bot = 0.0;
    let y_top = height;
    let slant_dx = slant_len * slant.cos();
    let slant_dy = slant_len * slant.sin();
    let mut segs: Vec<(f32, f32)> = Vec::new();
    // bottom edge: rear -> front (minus rounding)
    segs.push((x_rear, y_bot));
    segs.push((x_front - r, y_bot));
    // front bottom round: quarter circle centred (x_front - r, y_bot + r)
    let arc = 24;
    for i in 1..=arc {
        let a = -PI * 0.5 + (i as f32 / arc as f32) * PI * 0.5;
        segs.push((x_front - r + r * a.cos(), y_bot + r + r * a.sin()));
    }
    // front vertical
    segs.push((x_front, y_top - r));
    for i in 1..=arc {
        let a = (i as f32 / arc as f32) * PI * 0.5;
        segs.push((x_front - r + r * a.cos(), y_top - r + r * a.sin()));
    }
    // top edge to slant start
    segs.push((x_rear + slant_dx, y_top));
    // slant down to rear top
    segs.push((x_rear, y_top - slant_dy));
    // rear vertical back to start
    segs.push((x_rear, y_bot));

    // Resample by arc length.
    let mut cum = vec![0.0f32];
    for w in segs.windows(2) {
        let d = ((w[1].0 - w[0].0).powi(2) + (w[1].1 - w[0].1).powi(2)).sqrt();
        cum.push(cum.last().copied().unwrap_or(0.0) + d);
    }
    let total = *cum.last().unwrap_or(&1.0);
    let mut profile: Vec<(f32, f32)> = Vec::with_capacity(profile_segments);
    for i in 0..profile_segments {
        let s = total * i as f32 / profile_segments as f32;
        let k = cum
            .iter()
            .position(|c| *c > s)
            .unwrap_or(cum.len() - 1)
            .max(1);
        let t = ((s - cum[k - 1]) / (cum[k] - cum[k - 1]).max(1e-9)).clamp(0.0, 1.0);
        let (a, b) = (segs[k - 1], segs[k]);
        profile.push((a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t));
    }

    // Extrude along z.
    let np = profile.len();
    let nz = width_segments + 1;
    let mut points = Vec::with_capacity(np * nz + 2);
    for zi in 0..nz {
        let z = -width * 0.5 + width * zi as f32 / width_segments as f32;
        for (x, y) in &profile {
            points.push([*x, *y, z]);
        }
    }
    let mut faces = Vec::with_capacity(np * width_segments * 2 + np * 2);
    for zi in 0..width_segments {
        for pi in 0..np {
            let a = (zi * np + pi) as u32;
            let b = (zi * np + (pi + 1) % np) as u32;
            let c = ((zi + 1) * np + (pi + 1) % np) as u32;
            let d = ((zi + 1) * np + pi) as u32;
            faces.push([a, c, b]);
            faces.push([a, d, c]);
        }
    }
    // Side caps as fans from the profile centroid (profile is convex).
    let cx = profile.iter().map(|p| p.0).sum::<f32>() / np as f32;
    let cy = profile.iter().map(|p| p.1).sum::<f32>() / np as f32;
    let c0 = points.len() as u32;
    points.push([cx, cy, -width * 0.5]);
    let c1 = points.len() as u32;
    points.push([cx, cy, width * 0.5]);
    for pi in 0..np {
        let a = pi as u32;
        let b = ((pi + 1) % np) as u32;
        faces.push([c0, a, b]);
        let off = (width_segments * np) as u32;
        faces.push([c1, off + b, off + a]);
    }

    let mut mesh = SurfaceMesh {
        points,
        faces,
        fields: BTreeMap::new(),
        lod: 0,
    };
    let p = synthetic_cp(&mesh, x_rear, x_front, y_top, slant_dx, slant_dy, width, r);
    let wss: Vec<f32> = p.iter().map(|cp| (1.0 - cp).max(0.0) * 0.012).collect();
    mesh.fields.insert("p".into(), p);
    mesh.fields.insert("wallShearStress".into(), wss);
    mesh
}

/// Plausible Cp: stagnation on the nose, suction around the rounded front edges,
/// recovery along the body, a slant with the C-pillar vortex pair and a low-pressure base.
#[allow(clippy::too_many_arguments)]
fn synthetic_cp(
    mesh: &SurfaceMesh,
    x_rear: f32,
    x_front: f32,
    y_top: f32,
    slant_dx: f32,
    slant_dy: f32,
    width: f32,
    r: f32,
) -> Vec<f32> {
    let normals = mesh.vertex_normals();
    mesh.points
        .iter()
        .zip(&normals)
        .enumerate()
        .map(|(i, (p, n))| {
            let [x, y, z] = *p;
            let nx = n[0];
            // flow along -x: faces with normal +x face the oncoming flow
            let stag = nx.max(0.0).powi(2);
            let mut cp = stag * 1.0;
            // suction on the rounded edges (normal tilted away from x)
            let edge = (1.0 - nx.abs()).clamp(0.0, 1.0) * ((x - (x_front - r)) / r).clamp(0.0, 1.0);
            cp -= 1.1 * edge * edge;
            // mild recovery along the body
            let along = ((x_front - x) / (x_front - x_rear)).clamp(0.0, 1.0);
            cp -= 0.12 * (1.0 - along) * (nx.abs() < 0.5) as i32 as f32;
            // slant region
            let on_slant =
                x < x_rear + slant_dx + 1e-4 && y > y_top - slant_dy - 1e-4 && n[1] > 0.3;
            if on_slant {
                let s = ((x_rear + slant_dx - x) / slant_dx).clamp(0.0, 1.0);
                let side = (z.abs() - (width * 0.5 - 0.06)).max(0.0) / 0.06;
                cp = -0.55
                    - 0.35 * (1.0 - s)
                    - 0.9
                        * (-((1.0 - side) * 3.0).powi(2)).exp()
                        * (z.abs() > width * 0.3) as i32 as f32;
            }
            // base
            if nx < -0.7 {
                cp = -0.22 + 0.05 * ((y / y_top) * PI).sin();
            }
            // deterministic noise so the field looks measured, not analytic
            let h = ((i as u32).wrapping_mul(2654435761) >> 8) as f32 / (1u32 << 24) as f32;
            cp + (h - 0.5) * 0.03
        })
        .collect()
}

/// Metadata matching the fixture, in the shape of `samples/<id>/meta.json`.
pub fn ahmed_meta(sample_id: &str) -> serde_json::Value {
    serde_json::json!({
        "id": sample_id,
        "solver": "OpenFOAM",
        "solver_version": "v2212",
        "turbulence_model": "HiFi LES-like",
        "fidelity": "high",
        "geometry": {"ride_height_m": 0.05, "slant_angle_deg": 25.0, "length_m": 1.044, "width_m": 0.389, "height_m": 0.288},
        "conditions": {"u_inf_m_s": 40.0, "re": 2.8e6},
        "units": {"p": "Pa (Cp here)", "wallShearStress": "Pa"},
        "provenance": {"source": "fixture:frontier-viz::fixtures::ahmed_body", "license": "CC-BY-SA-4.0"}
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_is_a_closed_manifold_with_sane_fields() {
        let m = ahmed_body(0);
        assert!(
            m.faces.len() > 3000 && m.faces.len() < 6000,
            "{}",
            m.faces.len()
        );
        assert!(
            m.faces
                .iter()
                .all(|f| f.iter().all(|i| (*i as usize) < m.points.len()))
        );
        // Every edge is shared by exactly two faces on a closed surface.
        let mut count = std::collections::HashMap::new();
        for f in &m.faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                *count
                    .entry(if a < b { (a, b) } else { (b, a) })
                    .or_insert(0) += 1;
            }
        }
        assert!(count.values().all(|c| *c == 2));
        let (lo, hi) = m.field_range("p").unwrap();
        assert!(lo < -0.1 && hi > 0.5, "{lo} {hi}");
        let raw_min = m.fields["p"].iter().copied().fold(f32::MAX, f32::min);
        assert!(raw_min < -0.6, "{raw_min}");
        let (bmin, bmax) = m.bounds();
        assert!((bmax.x - bmin.x - 1.044).abs() < 1e-3);
    }

    #[test]
    fn resolutions_scale() {
        assert!(ahmed_body(3).faces.len() > ahmed_body(1).faces.len());
        assert!(ahmed_body(3).faces.len() > 90_000);
    }
}
