// SPDX-License-Identifier: Apache-2.0
//! Per-field viewer defaults and legend helpers shared by the app, the example renderer
//! and the thumbnails: pressure-like fields get a diverging colormap centred on zero,
//! everything else a sequential one over its robust range.

use crate::colormap::Colormap;
use crate::surface::SurfaceMesh;

/// How the scalar range mapped to the colormap is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RangeMode {
    /// Robust 2nd..98th percentile of the field.
    Auto,
    /// `[-m, m]` with `m` the robust 98th percentile of `|value|`.
    Symmetric,
    Manual(f32, f32),
}

impl RangeMode {
    pub fn parse(s: &str, lo: f32, hi: f32) -> RangeMode {
        match s {
            "symmetric" => RangeMode::Symmetric,
            "manual" => RangeMode::Manual(lo, hi),
            _ => RangeMode::Auto,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            RangeMode::Auto => "auto",
            RangeMode::Symmetric => "symmetric",
            RangeMode::Manual(..) => "manual",
        }
    }
}

/// Pressure, pressure coefficient and their averaged / predicted variants.
pub fn is_pressure_like(name: &str) -> bool {
    let base = name
        .strip_prefix("field:")
        .or_else(|| name.strip_prefix("fields/"))
        .unwrap_or(name);
    let base = base.trim_end_matches("_pred").trim_end_matches("_true");
    matches!(
        base,
        "p" | "pMean" | "static(p)_coeffMean" | "static(p)_coeff" | "Cp" | "cp" | "p_err"
    ) || base.starts_with("p_") && !base.starts_with("p_std")
}

/// Signed differences (`*_err`) are diverging too.
pub fn is_diverging(name: &str) -> bool {
    is_pressure_like(name) || name.ends_with("_err")
}

/// Default colormap and range mode for a field name.
pub fn defaults_for(name: &str) -> (Colormap, RangeMode) {
    if is_diverging(name) {
        (Colormap::Coolwarm, RangeMode::Symmetric)
    } else {
        (Colormap::Viridis, RangeMode::Auto)
    }
}

/// Robust symmetric magnitude: 98th percentile of `|v|`.
///
/// Linear in the number of values (a selection, not a sort): viewers ask for the range of
/// meshes with 10^5..10^6 vertices.
pub fn symmetric_range(values: &[f32]) -> Option<(f32, f32)> {
    let mut v: Vec<f32> = values
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .collect();
    if v.is_empty() {
        return None;
    }
    let i = ((v.len() as f32 * 0.98) as usize).min(v.len() - 1);
    let (_, m, _) = v.select_nth_unstable_by(i, f32::total_cmp);
    let m = *m;
    let m = if m > 0.0 {
        m
    } else {
        v.iter().copied().fold(0.0f32, f32::max)
    };
    if m > 0.0 { Some((-m, m)) } else { None }
}

/// Robust 2nd..98th percentile of `values` (min..max when those coincide), ignoring
/// non-finite entries. Linear time.
pub fn robust_range(values: &[f32]) -> Option<(f32, f32)> {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    let n = v.len();
    let i_lo = (n as f32 * 0.02) as usize;
    let i_hi = ((n as f32 * 0.98) as usize).min(n - 1);
    let (below, hi, _) = v.select_nth_unstable_by(i_hi, f32::total_cmp);
    let hi = *hi;
    let lo = if i_lo < i_hi {
        *below.select_nth_unstable_by(i_lo, f32::total_cmp).1
    } else {
        hi
    };
    if hi > lo {
        Some((lo, hi))
    } else {
        let min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Some((min, max))
    }
}

/// The `pct`-th to `(100 - pct)`-th percentile of the finite values (`pct` in 0..50; 0 is
/// min..max). `None` without finite values; min..max when the percentiles coincide.
pub fn percentile_range(values: &[f32], pct: f32) -> Option<(f32, f32)> {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    let n = v.len();
    let p = (pct.clamp(0.0, 49.9) / 100.0) as f64;
    let i_lo = ((n as f64 * p) as usize).min(n - 1);
    let i_hi = ((n as f64 * (1.0 - p)) as usize).min(n - 1).max(i_lo);
    let (below, hi, _) = v.select_nth_unstable_by(i_hi, f32::total_cmp);
    let hi = *hi;
    let lo = if i_lo < i_hi {
        *below.select_nth_unstable_by(i_lo, f32::total_cmp).1
    } else {
        hi
    };
    if hi > lo {
        Some((lo, hi))
    } else {
        let min = v.iter().copied().fold(f32::INFINITY, f32::min);
        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Some((min, max))
    }
}

/// Resolve the range to use for a field of `mesh` under `mode`.
pub fn range_for(mesh: &SurfaceMesh, field: &str, mode: RangeMode) -> (f32, f32) {
    match mode {
        RangeMode::Manual(lo, hi) if hi > lo => (lo, hi),
        RangeMode::Symmetric => mesh
            .fields
            .get(field)
            .and_then(|v| symmetric_range(v))
            .or_else(|| mesh.field_range(field))
            .unwrap_or((-1.0, 1.0)),
        _ => mesh.field_range(field).unwrap_or((0.0, 1.0)),
    }
}

/// "Nice" tick values inside `[lo, hi]` (1/2/5 steps), at most `max_ticks`.
pub fn legend_ticks(lo: f32, hi: f32, max_ticks: usize) -> Vec<f32> {
    if hi <= lo || hi.is_nan() || lo.is_nan() || max_ticks == 0 {
        return vec![lo];
    }
    let span = (hi - lo) as f64;
    let raw = span / max_ticks.max(1) as f64;
    let mag = 10f64.powf(raw.log10().floor());
    let norm = raw / mag;
    let step = if norm <= 1.0 {
        1.0
    } else if norm <= 2.0 {
        2.0
    } else if norm <= 5.0 {
        5.0
    } else {
        10.0
    } * mag;
    let first = (lo as f64 / step).ceil() * step;
    let mut out = Vec::new();
    let mut t = first;
    while t <= hi as f64 + step * 1e-6 && out.len() < max_ticks + 1 {
        out.push(t as f32);
        t += step;
    }
    if out.is_empty() {
        out.push(lo);
    }
    out
}

/// Compact number formatting for legends (3 significant digits, no trailing zeros).
pub fn fmt_tick(v: f32) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    let s = if a >= 1e4 || a < 1e-3 {
        format!("{v:.2e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 10.0 {
        format!("{v:.1}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.3}")
    };
    if s.contains('.') && !s.contains('e') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Unit label for well-known CFD field names (OpenFOAM kinematic pressure etc.).
pub fn units_for(name: &str) -> &'static str {
    let base = name
        .trim_end_matches("_pred")
        .trim_end_matches("_true")
        .trim_end_matches("_err");
    match base {
        "p" | "pMean" => "m²/s²",
        "static(p)_coeffMean" | "static(p)_coeff" | "Cp" | "cp" => "",
        "wallShearStress" | "wallShearStressMean" => "m²/s²",
        "yPlus" | "yPlusMean" => "",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_by_selection_match_ranges_by_sorting() {
        // Reference: the percentile picked from a fully sorted copy.
        let reference = |values: &[f32]| {
            let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
            v.sort_by(f32::total_cmp);
            let lo = v[(v.len() as f32 * 0.02) as usize];
            let hi = v[((v.len() as f32 * 0.98) as usize).min(v.len() - 1)];
            if hi > lo {
                (lo, hi)
            } else {
                (v[0], v[v.len() - 1])
            }
        };
        let mut x = 0x2545_f491u32;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for n in [1usize, 2, 3, 49, 50, 51, 1000, 20_001] {
            let mut values: Vec<f32> = (0..n).map(|_| next() * 37.0).collect();
            if n > 3 {
                values[1] = f32::NAN;
                values[2] = f32::INFINITY;
            }
            assert_eq!(robust_range(&values), Some(reference(&values)), "n = {n}");
            let abs: Vec<f32> = values.iter().map(|v| v.abs()).collect();
            let m = reference(&abs).1;
            assert_eq!(symmetric_range(&values), Some((-m, m)), "n = {n}");
        }
        assert_eq!(robust_range(&[5.0; 10]), Some((5.0, 5.0)));
        assert_eq!(robust_range(&[f32::NAN]), None);
        assert_eq!(symmetric_range(&[0.0, 0.0]), None);
        assert_eq!(symmetric_range(&[]), None);
    }

    #[test]
    fn percentile_ranges_clip_both_ends() {
        let v: Vec<f32> = (0..=1000).map(|i| i as f32).collect();
        assert_eq!(percentile_range(&v, 0.0), Some((0.0, 1000.0)));
        assert_eq!(percentile_range(&v, 10.0), Some((100.0, 900.0)));
        // the same clip as the robust range at 2 %
        assert_eq!(percentile_range(&v, 2.0), robust_range(&v));
        let mut spiky = v.clone();
        spiky.push(1e9);
        spiky.push(f32::NAN);
        let (lo, hi) = percentile_range(&spiky, 5.0).unwrap();
        assert!(lo > 40.0 && hi < 960.0, "{lo} {hi}");
        assert_eq!(percentile_range(&[3.0; 5], 5.0), Some((3.0, 3.0)));
        assert_eq!(percentile_range(&[], 5.0), None);
    }

    #[test]
    fn pressure_like_names_get_coolwarm_symmetric() {
        for n in [
            "p",
            "pMean",
            "static(p)_coeffMean",
            "Cp",
            "p_pred",
            "p_true",
            "p_err",
        ] {
            assert_eq!(
                defaults_for(n),
                (Colormap::Coolwarm, RangeMode::Symmetric),
                "{n}"
            );
        }
        for n in ["yPlusMean", "wallShearStressMean", "p_std", "cell_area"] {
            assert_eq!(defaults_for(n).0, Colormap::Viridis, "{n}");
        }
    }

    #[test]
    fn symmetric_range_is_centred_and_robust() {
        let mut v: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0 - 0.7).collect();
        v.push(1e6);
        let (lo, hi) = symmetric_range(&v).unwrap();
        assert_eq!(lo, -hi);
        assert!(hi < 1.0 && hi > 0.6, "{hi}");
    }

    #[test]
    fn ticks_are_nice_and_inside_range() {
        let t = legend_ticks(-0.37, 0.37, 5);
        assert!(t.len() >= 3 && t.len() <= 6, "{t:?}");
        assert!(t.iter().all(|x| *x >= -0.37 && *x <= 0.37));
        assert!(t.contains(&0.0));
        assert_eq!(fmt_tick(0.0), "0");
        assert_eq!(fmt_tick(0.25), "0.25");
        assert_eq!(fmt_tick(1234.0), "1234");
    }
}
