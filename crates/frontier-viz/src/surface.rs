// SPDX-License-Identifier: Apache-2.0
//! Triangle surface mesh with per-point scalar fields, plus the CPU helpers a viewer
//! needs: normals, bounds, LOD selection / decimation, picking and JSON interop.

use crate::VizError;
use crate::camera::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct SurfaceMesh {
    pub points: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    /// Scalar fields keyed by name. Length is `points.len()` (point data) or
    /// `faces.len()` (cell data); [`SurfaceMesh::point_field`] normalises both.
    #[serde(default)]
    pub fields: BTreeMap<String, Vec<f32>>,
    /// LOD level this mesh represents (0 = full resolution).
    #[serde(default)]
    pub lod: u8,
}

/// Result of a ray cast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub face: usize,
    pub t: f32,
    pub point: [f32; 3],
    /// Barycentric weights of the face's three vertices.
    pub bary: [f32; 3],
}

impl SurfaceMesh {
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut hi = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        for p in &self.points {
            let v = Vec3::from_array(*p);
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if self.points.is_empty() {
            (Vec3::ZERO, Vec3::ZERO)
        } else {
            (lo, hi)
        }
    }

    /// Area-weighted vertex normals.
    pub fn vertex_normals(&self) -> Vec<[f32; 3]> {
        let mut n = vec![Vec3::ZERO; self.points.len()];
        for f in &self.faces {
            let [a, b, c] = f.map(|i| Vec3::from_array(self.points[i as usize]));
            let fnorm = (b - a).cross(c - a);
            for i in f {
                n[*i as usize] = n[*i as usize] + fnorm;
            }
        }
        n.into_iter().map(|v| v.normalize().to_array()).collect()
    }

    /// Whether the face winding points outwards (signed volume test). Decimated or
    /// solver-exported meshes are often wound inwards; the renderer flips those so
    /// back faces (the inside, seen through holes) can be shaded differently.
    pub fn winding_is_outward(&self) -> bool {
        let (lo, hi) = self.bounds();
        let c = (lo + hi) * 0.5;
        let mut acc = 0.0f64;
        for f in &self.faces {
            let [a, b, cc] = f.map(|i| Vec3::from_array(self.points[i as usize]));
            let n = (b - a).cross(cc - a);
            let centroid = (a + b + cc) * (1.0 / 3.0);
            acc += (centroid - c).dot(n) as f64;
        }
        acc >= 0.0
    }

    /// Resolve a requested field name: exact match, else the first field starting with
    /// it (`p` -> `pMean`), else the first point field. Returns the actual name.
    pub fn resolve_field(&self, name: &str) -> Option<String> {
        if self.fields.contains_key(name) {
            return Some(name.to_string());
        }
        self.fields
            .keys()
            .find(|k| k.starts_with(name))
            .or_else(|| {
                self.fields
                    .iter()
                    .find(|(_, v)| v.len() == self.points.len())
                    .map(|(k, _)| k)
            })
            .cloned()
    }

    /// Field as point data (cell data is averaged onto vertices).
    pub fn point_field(&self, name: &str) -> Option<Vec<f32>> {
        let f = self.fields.get(name)?;
        if f.len() == self.points.len() {
            return Some(f.clone());
        }
        if f.len() != self.faces.len() {
            return None;
        }
        let mut acc = vec![0.0f32; self.points.len()];
        let mut cnt = vec![0u32; self.points.len()];
        for (face, v) in self.faces.iter().zip(f) {
            for i in face {
                acc[*i as usize] += v;
                cnt[*i as usize] += 1;
            }
        }
        Some(
            acc.iter()
                .zip(&cnt)
                .map(|(a, c)| if *c > 0 { a / *c as f32 } else { 0.0 })
                .collect(),
        )
    }

    /// Robust (2nd..98th percentile) range of a field, for colour mapping.
    pub fn field_range(&self, name: &str) -> Option<(f32, f32)> {
        crate::field::robust_range(self.fields.get(name)?)
    }

    /// Closest ray/triangle intersection (Moller-Trumbore). Linear in faces: fine for
    /// the <= 100k-face LODs a single viewer shows.
    pub fn pick(&self, origin: Vec3, dir: Vec3) -> Option<Hit> {
        let mut best: Option<Hit> = None;
        for (fi, f) in self.faces.iter().enumerate() {
            let [a, b, c] = f.map(|i| Vec3::from_array(self.points[i as usize]));
            let e1 = b - a;
            let e2 = c - a;
            let p = dir.cross(e2);
            let det = e1.dot(p);
            if det.abs() < 1e-9 {
                continue;
            }
            let inv = 1.0 / det;
            let s = origin - a;
            let u = s.dot(p) * inv;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(e1);
            let v = dir.dot(q) * inv;
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let t = e2.dot(q) * inv;
            if t <= 1e-6 || best.is_some_and(|h| h.t <= t) {
                continue;
            }
            best = Some(Hit {
                face: fi,
                t,
                point: (origin + dir * t).to_array(),
                bary: [1.0 - u - v, u, v],
            });
        }
        best
    }

    /// Interpolated point-field value at a hit.
    pub fn value_at(&self, field: &[f32], hit: &Hit) -> Option<f32> {
        value_at_flat(&self.faces_flat(), field, hit)
    }

    fn faces_flat(&self) -> Vec<u32> {
        self.faces.iter().flat_map(|f| *f).collect()
    }
}

/// Closest ray/triangle intersection over flat GPU-layout arrays (`xyz` per vertex,
/// three indices per triangle): the picking the playback engine does on the frame it is
/// showing, without a `SurfaceMesh` copy of it.
pub fn pick_flat(positions: &[f32], indices: &[u32], origin: Vec3, dir: Vec3) -> Option<Hit> {
    let p = |i: u32| {
        let i = i as usize * 3;
        Vec3::new(positions[i], positions[i + 1], positions[i + 2])
    };
    let mut best: Option<Hit> = None;
    for (fi, f) in indices.as_chunks::<3>().0.iter().enumerate() {
        if f.iter().any(|&i| i as usize * 3 + 3 > positions.len()) {
            continue;
        }
        let (a, b, c) = (p(f[0]), p(f[1]), p(f[2]));
        let e1 = b - a;
        let e2 = c - a;
        let pv = dir.cross(e2);
        let det = e1.dot(pv);
        if det.abs() < 1e-9 {
            continue;
        }
        let inv = 1.0 / det;
        let s = origin - a;
        let u = s.dot(pv) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = dir.dot(q) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = e2.dot(q) * inv;
        if t <= 1e-6 || best.is_some_and(|h| h.t <= t) {
            continue;
        }
        best = Some(Hit {
            face: fi,
            t,
            point: (origin + dir * t).to_array(),
            bary: [1.0 - u - v, u, v],
        });
    }
    best
}

/// Interpolated per-vertex value at a hit over a flat index list.
pub fn value_at_flat(indices: &[u32], field: &[f32], hit: &Hit) -> Option<f32> {
    let f = indices.get(hit.face * 3..hit.face * 3 + 3)?;
    Some(
        f.iter()
            .zip(hit.bary)
            .map(|(i, w)| field.get(*i as usize).copied().unwrap_or(0.0) * w)
            .sum(),
    )
}

impl SurfaceMesh {
    /// Vertex-clustering decimation to roughly `target_faces`. Fields are averaged per
    /// cluster. Fast (a few passes) and good enough for far-away LODs; the server's
    /// quadric decimation produces the shipped LOD levels.
    pub fn decimate(&self, target_faces: usize) -> SurfaceMesh {
        if self.faces.len() <= target_faces || self.faces.is_empty() {
            return self.clone();
        }
        let (lo, hi) = self.bounds();
        let extent = (hi - lo).length().max(1e-6);
        // Faces scale with (extent / cell)^2 on a surface; start from that estimate and
        // refine the cell size against the measured result.
        let mut cell = extent / (target_faces as f32).sqrt();
        let mut best = self.cluster(lo, cell);
        for _ in 0..5 {
            let ratio = best.faces.len() as f32 / target_faces as f32;
            if (0.7..=1.2).contains(&ratio) {
                break;
            }
            cell *= ratio.sqrt();
            let next = self.cluster(lo, cell);
            if next.faces.is_empty() {
                break;
            }
            best = next;
        }
        best.lod = self.lod + 1;
        best
    }

    fn cluster(&self, lo: Vec3, cell: f32) -> SurfaceMesh {
        let key = |p: &[f32; 3]| {
            (
                ((p[0] - lo.x) / cell) as i32,
                ((p[1] - lo.y) / cell) as i32,
                ((p[2] - lo.z) / cell) as i32,
            )
        };
        let mut cluster_of: HashMap<(i32, i32, i32), u32> = HashMap::new();
        let mut remap = vec![0u32; self.points.len()];
        let mut points: Vec<[f32; 3]> = Vec::new();
        let mut counts: Vec<u32> = Vec::new();
        let field_names: Vec<&String> = self
            .fields
            .iter()
            .filter(|(_, v)| v.len() == self.points.len())
            .map(|(k, _)| k)
            .collect();
        let mut fields: Vec<Vec<f32>> = vec![Vec::new(); field_names.len()];
        for (i, p) in self.points.iter().enumerate() {
            let k = key(p);
            let c = *cluster_of.entry(k).or_insert_with(|| {
                points.push([0.0; 3]);
                counts.push(0);
                for f in &mut fields {
                    f.push(0.0);
                }
                (points.len() - 1) as u32
            });
            remap[i] = c;
            let cu = c as usize;
            for (acc, v) in points[cu].iter_mut().zip(p) {
                *acc += v;
            }
            counts[cu] += 1;
            for (fi, name) in field_names.iter().enumerate() {
                fields[fi][cu] += self.fields[*name][i];
            }
        }
        for (i, c) in counts.iter().enumerate() {
            let inv = 1.0 / (*c as f32);
            for p in points[i].iter_mut() {
                *p *= inv;
            }
            for f in &mut fields {
                f[i] *= inv;
            }
        }
        let mut faces = Vec::new();
        for f in &self.faces {
            let m = [
                remap[f[0] as usize],
                remap[f[1] as usize],
                remap[f[2] as usize],
            ];
            if m[0] != m[1] && m[1] != m[2] && m[0] != m[2] {
                faces.push(m);
            }
        }
        SurfaceMesh {
            points,
            faces,
            fields: field_names.into_iter().cloned().zip(fields).collect(),
            lod: self.lod,
        }
    }

    /// `surface.json`: `{"points":[[x,y,z],..],"faces":[[a,b,c],..],"fields":{"p":[..]}}`.
    pub fn from_json(s: &str) -> Result<SurfaceMesh, VizError> {
        serde_json::from_str(s).map_err(|e| VizError::Decode(e.to_string()))
    }

    pub fn to_json(&self) -> Result<String, VizError> {
        serde_json::to_string(self).map_err(|e| VizError::Decode(e.to_string()))
    }

    /// Interleaved GPU vertices: position(3) + normal(3) + scalar(1). Normals are
    /// negated when `flip` so they point outwards for inward-wound meshes.
    pub fn gpu_vertices(&self, scalar: &[f32], flip: bool) -> Vec<[f32; 7]> {
        let normals = self.vertex_normals();
        let sign = if flip { -1.0 } else { 1.0 };
        self.points
            .iter()
            .zip(&normals)
            .enumerate()
            .map(|(i, (p, n))| {
                let s = scalar.get(i).copied().unwrap_or(0.0);
                [p[0], p[1], p[2], n[0] * sign, n[1] * sign, n[2] * sign, s]
            })
            .collect()
    }

    /// Geometry-only GPU vertices: position(3) + normal(3). The colouring scalar goes in a
    /// separate buffer so a new time step does not re-upload (or recompute) this.
    pub fn gpu_geometry(&self, flip: bool) -> Vec<[f32; 6]> {
        let normals = self.vertex_normals();
        let sign = if flip { -1.0 } else { 1.0 };
        self.points
            .iter()
            .zip(&normals)
            .map(|(p, n)| [p[0], p[1], p[2], n[0] * sign, n[1] * sign, n[2] * sign])
            .collect()
    }

    /// Per-point values from a frame that is either per point or per face (cell values
    /// are averaged onto the vertices). `None` when the length matches neither.
    pub fn to_point_values(&self, values: &[f32]) -> Option<Vec<f32>> {
        if values.len() == self.points.len() {
            return Some(values.to_vec());
        }
        if values.len() != self.faces.len() {
            return None;
        }
        let mut acc = vec![0.0f32; self.points.len()];
        let mut cnt = vec![0u32; self.points.len()];
        for (face, v) in self.faces.iter().zip(values) {
            for i in face {
                acc[*i as usize] += v;
                cnt[*i as usize] += 1;
            }
        }
        Some(
            acc.iter()
                .zip(&cnt)
                .map(|(a, c)| if *c > 0 { a / *c as f32 } else { 0.0 })
                .collect(),
        )
    }

    /// Unique edges as a line-list index buffer (wireframe overlay), ordered by vertex
    /// index. Sort + dedup of packed keys: every mesh upload builds this (a transient
    /// surface uploads one per time step), and hashing three edges per face was the
    /// largest part of an upload.
    pub fn edge_indices(&self) -> Vec<u32> {
        let mut keys: Vec<u64> = Vec::with_capacity(self.faces.len() * 3);
        for f in &self.faces {
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
}

/// Levels 0..=3 of the same sample, as stored in `surface.zarr/lod/<level>`.
#[derive(Debug, Clone, Default)]
pub struct LodSet {
    pub levels: Vec<SurfaceMesh>,
}

impl LodSet {
    /// Target face counts per level (`docs/spec/dataset-format.md`).
    pub const TARGET_FACES: [usize; 4] = [usize::MAX, 100_000, 20_000, 5_000];

    /// Build levels 1..=3 by decimating level 0.
    pub fn from_full(full: SurfaceMesh) -> Self {
        let mut levels = vec![full];
        for lvl in 1..4 {
            let prev = &levels[lvl - 1];
            let mut m = prev.decimate(Self::TARGET_FACES[lvl]);
            m.lod = lvl as u8;
            levels.push(m);
        }
        Self { levels }
    }

    /// Pick the level for a viewer that is `width_px` x `height_px` on screen: budget of
    /// one face per ~4 px^2 of the viewer's diagonal squared, never below level 3.
    pub fn level_for_pixels(&self, width_px: f32, height_px: f32) -> usize {
        if self.levels.is_empty() {
            return 0;
        }
        let diag_sq = width_px * width_px + height_px * height_px;
        let budget = (diag_sq * 0.25) as usize;
        let mut best = self.levels.len() - 1;
        for (i, m) in self.levels.iter().enumerate() {
            if m.faces.len() <= budget {
                best = i;
                break;
            }
        }
        best
    }

    pub fn get(&self, level: usize) -> Option<&SurfaceMesh> {
        self.levels
            .get(level.min(self.levels.len().saturating_sub(1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> SurfaceMesh {
        SurfaceMesh {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            faces: vec![[0, 1, 2], [0, 2, 3]],
            fields: BTreeMap::from([("p".to_string(), vec![0.0, 1.0, 2.0, 3.0])]),
            lod: 0,
        }
    }

    #[test]
    fn normals_point_along_z() {
        let n = quad().vertex_normals();
        assert!(n.iter().all(|v| (v[2] - 1.0).abs() < 1e-6));
    }

    #[test]
    fn pick_hits_the_quad_and_interpolates() {
        let m = quad();
        let hit = m
            .pick(Vec3::new(0.5, 0.5, 5.0), Vec3::new(0.0, 0.0, -1.0))
            .unwrap();
        assert!((hit.t - 5.0).abs() < 1e-5);
        let v = m.value_at(&m.point_field("p").unwrap(), &hit).unwrap();
        // Centre of the first triangle (0,1,2) -> p = (0+1+2)/3... but the hit at (0.5,0.5)
        // is on the shared diagonal; either triangle gives a value between 0 and 3.
        assert!((0.0..=3.0).contains(&v));
        assert!(
            m.pick(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, -1.0))
                .is_none()
        );
    }

    #[test]
    fn cell_data_is_averaged_to_points() {
        let mut m = quad();
        m.fields.insert("c".into(), vec![1.0, 3.0]);
        let p = m.point_field("c").unwrap();
        assert_eq!(p[0], 2.0); // shared by both faces
        assert_eq!(p[1], 1.0);
    }

    #[test]
    fn winding_orientation_is_detected() {
        let body = crate::fixtures::ahmed_body(0);
        let outward = body.winding_is_outward();
        let mut flipped = body.clone();
        for f in &mut flipped.faces {
            f.swap(1, 2);
        }
        assert_ne!(outward, flipped.winding_is_outward());
        let v = body.gpu_vertices(&body.fields["p"], !outward);
        assert_eq!(v.len(), body.points.len());
    }

    #[test]
    fn json_roundtrip_and_edges() {
        let m = quad();
        let back = SurfaceMesh::from_json(&m.to_json().unwrap()).unwrap();
        assert_eq!(m, back);
        assert_eq!(m.edge_indices().len(), 10); // 5 unique edges
        // Each edge once, whatever the winding of the faces that share it.
        let edges: Vec<(u32, u32)> = m.edge_indices().chunks(2).map(|e| (e[0], e[1])).collect();
        let mut expected = std::collections::BTreeSet::new();
        for f in &m.faces {
            for (a, b) in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])] {
                expected.insert((a.min(b), a.max(b)));
            }
        }
        assert_eq!(edges, expected.into_iter().collect::<Vec<_>>());
    }

    #[test]
    fn decimation_reduces_faces_and_lod_selection_works() {
        let full = crate::fixtures::ahmed_body(2);
        let set = LodSet::from_full(full);
        assert_eq!(set.levels.len(), 4);
        for w in set.levels.windows(2) {
            assert!(w[1].faces.len() <= w[0].faces.len());
        }
        assert!(
            set.levels[3].faces.len() <= 8_000,
            "{}",
            set.levels[3].faces.len()
        );
        assert_eq!(set.level_for_pixels(4000.0, 4000.0), 0);
        assert!(set.level_for_pixels(60.0, 40.0) >= 2);
    }
}
