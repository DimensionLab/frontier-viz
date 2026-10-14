// SPDX-License-Identifier: Apache-2.0
//! Colormaps: the built-in scientific maps, user-defined maps and the value -> colour scale.
//!
//! Every map is a [`LUT_SIZE`]-entry sRGB table ([`Lut`]). The GPU reads the same tables
//! from [`atlas`] (row = [`Colormap::index`]) and [`Colormap::sample`] is the linear
//! interpolation of that table, so CPU legends and GPU pixels agree. A shader reproduces
//! `sample(t)` with a linear-filtered lookup at `u = (0.5 + t * 255) / 256`.

use std::sync::OnceLock;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

pub const LUT_SIZE: usize = 256;
/// Number of custom-map slots in the registry and rows reserved for them in the atlas.
pub const MAX_CUSTOM: usize = 32;
/// sRGB display colours, alpha 255.
pub type Lut = [[u8; 4]; LUT_SIZE];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    Uniform,
    Diverging,
    Sequential,
    Cyclic,
    Classic,
    Custom,
}

impl Family {
    pub const ALL: &'static [Family] = &[
        Family::Uniform,
        Family::Diverging,
        Family::Sequential,
        Family::Cyclic,
        Family::Classic,
        Family::Custom,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Family::Uniform => "Perceptually uniform",
            Family::Diverging => "Diverging",
            Family::Sequential => "Sequential",
            Family::Cyclic => "Cyclic",
            Family::Classic => "Classic",
            Family::Custom => "Custom",
        }
    }

    /// The built-in maps of this family, in menu order (empty for `Custom`).
    pub fn builtins(&self) -> Vec<Colormap> {
        Colormap::BUILTIN
            .iter()
            .copied()
            .filter(|m| m.family() == *self)
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Colormap {
    #[default]
    Viridis,
    Plasma,
    Inferno,
    Magma,
    Cividis,
    Turbo,
    Coolwarm,
    RdBu,
    BrBG,
    Balance,
    Blues,
    Greys,
    YlGnBu,
    Thermal,
    Oranges,
    Twilight,
    Phase,
    Jet,
    Rainbow,
    /// A slot of the [`registry`].
    Custom(u16),
}

impl Colormap {
    /// All built-in maps; the position in this list is the atlas row.
    pub const BUILTIN: &'static [Colormap] = &[
        Colormap::Viridis,
        Colormap::Plasma,
        Colormap::Inferno,
        Colormap::Magma,
        Colormap::Cividis,
        Colormap::Turbo,
        Colormap::Coolwarm,
        Colormap::RdBu,
        Colormap::BrBG,
        Colormap::Balance,
        Colormap::Blues,
        Colormap::Greys,
        Colormap::YlGnBu,
        Colormap::Thermal,
        Colormap::Oranges,
        Colormap::Twilight,
        Colormap::Phase,
        Colormap::Jet,
        Colormap::Rainbow,
    ];

    /// Case-insensitive wire name or label; `custom:<id>` goes through the registry.
    /// Anything unknown is viridis.
    pub fn parse(s: &str) -> Colormap {
        let s = s.trim();
        if let Some(prefix) = s.get(..7)
            && prefix.eq_ignore_ascii_case("custom:")
        {
            return Colormap::custom(&s[7..]).unwrap_or_default();
        }
        if ["grays", "gray", "grey"]
            .iter()
            .any(|alias| s.eq_ignore_ascii_case(alias))
        {
            return Colormap::Greys;
        }
        Colormap::BUILTIN
            .iter()
            .copied()
            .find(|m| s.eq_ignore_ascii_case(m.as_str()))
            .unwrap_or_default()
    }

    /// The registered custom map with this id.
    pub fn custom(id: &str) -> Option<Colormap> {
        registry::slot_of(id).map(Colormap::Custom)
    }

    /// Lower-case wire name; `"custom"` for every custom map (see [`Colormap::name`]).
    pub fn as_str(&self) -> &'static str {
        match self {
            Colormap::Viridis => "viridis",
            Colormap::Plasma => "plasma",
            Colormap::Inferno => "inferno",
            Colormap::Magma => "magma",
            Colormap::Cividis => "cividis",
            Colormap::Turbo => "turbo",
            Colormap::Coolwarm => "coolwarm",
            Colormap::RdBu => "rdbu",
            Colormap::BrBG => "brbg",
            Colormap::Balance => "balance",
            Colormap::Blues => "blues",
            Colormap::Greys => "greys",
            Colormap::YlGnBu => "ylgnbu",
            Colormap::Thermal => "thermal",
            Colormap::Oranges => "oranges",
            Colormap::Twilight => "twilight",
            Colormap::Phase => "phase",
            Colormap::Jet => "jet",
            Colormap::Rainbow => "rainbow",
            Colormap::Custom(_) => "custom",
        }
    }

    /// Wire name that [`Colormap::parse`] reads back: `custom:<id>` for a custom map.
    pub fn name(&self) -> String {
        match self {
            Colormap::Custom(slot) => match registry::by_slot(*slot) {
                Some(map) => format!("custom:{}", map.id),
                None => "custom".to_string(),
            },
            _ => self.as_str().to_string(),
        }
    }

    /// Display name, in the casing of the map's origin (matplotlib, ColorBrewer, cmocean).
    pub fn label(&self) -> String {
        match self {
            Colormap::RdBu => "RdBu".to_string(),
            Colormap::BrBG => "BrBG".to_string(),
            Colormap::Blues => "Blues".to_string(),
            Colormap::Greys => "Greys".to_string(),
            Colormap::YlGnBu => "YlGnBu".to_string(),
            Colormap::Oranges => "Oranges".to_string(),
            Colormap::Custom(slot) => registry::by_slot(*slot)
                .map(|map| map.name)
                .unwrap_or_else(|| "custom".to_string()),
            _ => self.as_str().to_string(),
        }
    }

    pub fn family(&self) -> Family {
        match self {
            Colormap::Viridis
            | Colormap::Plasma
            | Colormap::Inferno
            | Colormap::Magma
            | Colormap::Cividis
            | Colormap::Turbo => Family::Uniform,
            Colormap::Coolwarm | Colormap::RdBu | Colormap::BrBG | Colormap::Balance => {
                Family::Diverging
            }
            Colormap::Blues
            | Colormap::Greys
            | Colormap::YlGnBu
            | Colormap::Thermal
            | Colormap::Oranges => Family::Sequential,
            Colormap::Twilight | Colormap::Phase => Family::Cyclic,
            Colormap::Jet | Colormap::Rainbow => Family::Classic,
            Colormap::Custom(_) => Family::Custom,
        }
    }

    /// False for the maps whose lightness is not even: turbo, jet, rainbow and custom maps.
    pub fn perceptually_uniform(&self) -> bool {
        !matches!(
            self,
            Colormap::Turbo | Colormap::Jet | Colormap::Rainbow | Colormap::Custom(_)
        )
    }

    /// Row in the [`atlas`].
    pub fn index(&self) -> u32 {
        match self {
            Colormap::Viridis => 0,
            Colormap::Plasma => 1,
            Colormap::Inferno => 2,
            Colormap::Magma => 3,
            Colormap::Cividis => 4,
            Colormap::Turbo => 5,
            Colormap::Coolwarm => 6,
            Colormap::RdBu => 7,
            Colormap::BrBG => 8,
            Colormap::Balance => 9,
            Colormap::Blues => 10,
            Colormap::Greys => 11,
            Colormap::YlGnBu => 12,
            Colormap::Thermal => 13,
            Colormap::Oranges => 14,
            Colormap::Twilight => 15,
            Colormap::Phase => 16,
            Colormap::Jet => 17,
            Colormap::Rainbow => 18,
            Colormap::Custom(slot) => Colormap::BUILTIN.len() as u32 + u32::from(*slot),
        }
    }

    /// sRGB in 0..1 for `t` in 0..1 (clamped): the linear interpolation of [`Colormap::lut`].
    pub fn sample(&self, t: f32) -> [f32; 3] {
        match self {
            Colormap::Custom(slot) => registry::with_lut(*slot, |lut| sample_lut(lut, t))
                .unwrap_or_else(|| sample_lut(grey_lut(), t)),
            _ => sample_lut(self.builtin_lut(), t),
        }
    }

    pub fn sample_u8(&self, t: f32) -> [u8; 3] {
        let c = self.sample(t);
        [to_u8(c[0]), to_u8(c[1]), to_u8(c[2])]
    }

    /// The table of this map. An empty custom slot is a grey ramp, like its atlas row.
    /// Take it once and use [`sample_lut`] when colouring many values.
    pub fn lut(&self) -> Lut {
        match self {
            Colormap::Custom(slot) => {
                registry::with_lut(*slot, |lut| *lut).unwrap_or_else(|| *grey_lut())
            }
            _ => *self.builtin_lut(),
        }
    }

    /// CSS `linear-gradient` for a legend bar, sampled at `stops` points.
    pub fn css_gradient(&self, stops: usize) -> String {
        ColorScale::from(*self).css_gradient(stops)
    }

    fn builtin_lut(&self) -> &'static Lut {
        static LUTS: OnceLock<Vec<Lut>> = OnceLock::new();
        LUTS.get_or_init(|| Colormap::BUILTIN.iter().map(Colormap::build).collect())
            .get(self.index() as usize)
            .unwrap_or_else(|| grey_lut())
    }

    fn build(&self) -> Lut {
        match self {
            Colormap::Viridis => lut_from(|t| poly6(&VIRIDIS, t)),
            Colormap::Plasma => lut_from(|t| poly6(&PLASMA, t)),
            Colormap::Inferno => lut_from(|t| poly6(&INFERNO, t)),
            Colormap::Magma => lut_from(|t| poly6(&MAGMA, t)),
            Colormap::Cividis => table_lut(&even_stops(CIVIDIS), Interp::Oklab),
            Colormap::Turbo => table_lut(&even_stops(TURBO), Interp::Oklab),
            Colormap::Coolwarm => lut_from(coolwarm),
            Colormap::RdBu => table_lut(&even_stops(RDBU), Interp::Srgb),
            Colormap::BrBG => table_lut(&even_stops(BRBG), Interp::Srgb),
            Colormap::Balance => table_lut(&balance_stops(), Interp::Oklab),
            Colormap::Blues => table_lut(&even_stops(BLUES), Interp::Srgb),
            Colormap::Greys => table_lut(&even_stops(GREYS), Interp::Srgb),
            Colormap::YlGnBu => table_lut(&even_stops(YLGNBU), Interp::Srgb),
            Colormap::Thermal => table_lut(&even_stops(THERMAL), Interp::Oklab),
            Colormap::Oranges => table_lut(&even_stops(ORANGES), Interp::Srgb),
            Colormap::Twilight => table_lut(&even_stops(TWILIGHT), Interp::Oklab),
            Colormap::Phase => table_lut(&even_stops(PHASE), Interp::Oklab),
            Colormap::Jet => lut_from(jet),
            Colormap::Rainbow => lut_from(rainbow),
            Colormap::Custom(_) => *grey_lut(),
        }
    }
}

/// Linear interpolation of a table at `t` in 0..1 (clamped; NaN reads as 0).
pub fn sample_lut(lut: &Lut, t: f32) -> [f32; 3] {
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    let x = t * (LUT_SIZE - 1) as f32;
    let i = (x.floor() as usize).min(LUT_SIZE - 1);
    let j = (i + 1).min(LUT_SIZE - 1);
    let f = x - i as f32;
    let (a, b) = (lut[i], lut[j]);
    let mix = |k: usize| (f32::from(a[k]) + (f32::from(b[k]) - f32::from(a[k])) * f) / 255.0;
    [mix(0), mix(1), mix(2)]
}

fn to_u8(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn lut_from(f: impl Fn(f32) -> [f32; 3]) -> Lut {
    let mut lut = [[0, 0, 0, 255]; LUT_SIZE];
    for (i, px) in lut.iter_mut().enumerate() {
        let c = f(i as f32 / (LUT_SIZE - 1) as f32);
        *px = [to_u8(c[0]), to_u8(c[1]), to_u8(c[2]), 255];
    }
    lut
}

fn table_lut(stops: &[Stop], interp: Interp) -> Lut {
    lut_from(|t| sample_stops(stops, interp, t))
}

fn grey_lut() -> &'static Lut {
    static GREY: OnceLock<Lut> = OnceLock::new();
    GREY.get_or_init(|| lut_from(|t| [t, t, t]))
}

fn rgb(hex: u32) -> [f32; 3] {
    [
        ((hex >> 16) & 0xff) as f32 / 255.0,
        ((hex >> 8) & 0xff) as f32 / 255.0,
        (hex & 0xff) as f32 / 255.0,
    ]
}

fn even_stops(colors: &[u32]) -> Vec<Stop> {
    let last = colors.len().saturating_sub(1).max(1) as f32;
    colors
        .iter()
        .enumerate()
        .map(|(i, c)| Stop {
            pos: i as f32 / last,
            color: rgb(*c),
        })
        .collect()
}

type Poly6 = [[f32; 3]; 7];

fn poly6(c: &Poly6, t: f32) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let mut acc = c[6][k];
        for row in c[..6].iter().rev() {
            acc = acc * t + row[k];
        }
        *o = acc.clamp(0.0, 1.0);
    }
    out
}

// Degree-6 polynomial fits of the matplotlib maps by Matt Zucker (shadertoy WlfXRN, CC0);
// maximum error against matplotlib 2 % (3 % for inferno).
const VIRIDIS: Poly6 = [
    [0.277_727_3, 0.005_407_344_5, 0.334_099_8],
    [0.105_093_04, 1.404_613_5, 1.384_590_2],
    [-0.330_861_83, 0.214_847_56, 0.095_095_16],
    [-4.634_230_5, -5.799_101, -19.332_441],
    [6.228_27, 14.179_933, 56.690_55],
    [4.776_385, -13.745_145, -65.353_035],
    [-5.435_456, 4.645_852_6, 26.312_435],
];
const PLASMA: Poly6 = [
    [0.058_732_346, 0.023_336_709, 0.543_340_2],
    [2.176_514_6, 0.238_383_41, 0.753_960_43],
    [-2.689_460_5, -7.455_851, 3.1108],
    [6.130_348, 42.346_188, -28.518_854],
    [-11.107_436, -82.666_31, 60.139_847],
    [10.023_066, 71.413_62, -54.072_186],
    [-3.658_713_8, -22.931_534, 18.191_908],
];
const INFERNO: Poly6 = [
    [0.000_218_940_37, 0.001_651_004_7, -0.019_480_899],
    [0.106_513_42, 0.563_956_44, 3.932_712_3],
    [11.602_493, -3.972_854, -15.942_394],
    [-41.703_995, 17.4364, 44.354_145],
    [77.162_93, -33.402_36, -81.807_31],
    [-71.319_43, 32.626_064, 73.209_52],
    [25.131_126, -12.242_669, -23.070_326],
];
const MAGMA: Poly6 = [
    [-0.002_136_485, -0.000_749_655_07, -0.005_386_127_6],
    [0.251_660_56, 0.677_523_26, 2.494_026_7],
    [8.353_717, -3.577_719_4, 0.314_467_9],
    [-27.668_734, 14.264_73, -13.649_213],
    [52.176_14, -27.943_605, 12.944_169],
    [-50.768_524, 29.046_583, 4.234_153],
    [18.655_704, -11.489_774, -5.601_961_6],
];

// matplotlib 3.11 `cividis`, 17 evenly spaced samples.
const CIVIDIS: &[u32] = &[
    0x00224e, 0x002e6a, 0x1a386f, 0x32436d, 0x434e6c, 0x535a6d, 0x61656f, 0x6f7073, 0x7d7c78,
    0x8c8878, 0x9b9476, 0xaba072, 0xbcae6c, 0xcdbb63, 0xdec958, 0xf0d846, 0xfee838,
];
// matplotlib 3.11 `turbo` (Anton Mikhailov's table), 33 evenly spaced samples. The published
// polynomial approximation is off by up to 0.12 at the ends, so the table is used instead.
const TURBO: &[u32] = &[
    0x30123b, 0x392a73, 0x4040a2, 0x4456c7, 0x466be3, 0x4680f6, 0x4294ff, 0x37a8fa, 0x28bceb,
    0x1ccdd8, 0x18ddc2, 0x1fe9af, 0x32f298, 0x4ef97d, 0x6dfe62, 0x8bff4b, 0xa4fc3c, 0xb9f635,
    0xcdec34, 0xdfdf37, 0xeecf3a, 0xf8be39, 0xfdac34, 0xfe962b, 0xfb7e21, 0xf46617, 0xeb500e,
    0xdf3f08, 0xd02f05, 0xbe2102, 0xa91601, 0x920b01, 0x7a0403,
];
// matplotlib 3.11 `twilight`, 33 evenly spaced samples (last set equal to first).
const TWILIGHT: &[u32] = &[
    0xe2d9e2, 0xd7d7dd, 0xc4ced4, 0xacc2cc, 0x95b5c7, 0x81a6c3, 0x7297c1, 0x6887be, 0x6276ba,
    0x5f64b5, 0x5e51ad, 0x5d3da1, 0x592a8f, 0x511b77, 0x45135c, 0x381145, 0x2f1436, 0x3a113a,
    0x4a1342, 0x5f174a, 0x741e4f, 0x872750, 0x983550, 0xa64550, 0xb25652, 0xbb6958, 0xc27c63,
    0xc89073, 0xcca389, 0xd1b6a3, 0xd8c7be, 0xdfd4d6, 0xe2d9e2,
];
// ColorBrewer 2.0 class tables (11-class diverging, 9-class sequential), interpolated in
// sRGB like matplotlib does. RdBu runs red (low) to blue (high).
const RDBU: &[u32] = &[
    0x67001f, 0xb2182b, 0xd6604d, 0xf4a582, 0xfddbc7, 0xf7f7f7, 0xd1e5f0, 0x92c5de, 0x4393c3,
    0x2166ac, 0x053061,
];
const BRBG: &[u32] = &[
    0x543005, 0x8c510a, 0xbf812d, 0xdfc27d, 0xf6e8c3, 0xf5f5f5, 0xc7eae5, 0x80cdc1, 0x35978f,
    0x01665e, 0x003c30,
];
const BLUES: &[u32] = &[
    0xf7fbff, 0xdeebf7, 0xc6dbef, 0x9ecae1, 0x6baed6, 0x4292c6, 0x2171b5, 0x08519c, 0x08306b,
];
const GREYS: &[u32] = &[
    0xffffff, 0xf0f0f0, 0xd9d9d9, 0xbdbdbd, 0x969696, 0x737373, 0x525252, 0x252525, 0x000000,
];
const YLGNBU: &[u32] = &[
    0xffffd9, 0xedf8b1, 0xc7e9b4, 0x7fcdbb, 0x41b6c4, 0x1d91c0, 0x225ea8, 0x253494, 0x081d58,
];
const ORANGES: &[u32] = &[
    0xfff5eb, 0xfee6ce, 0xfdd0a2, 0xfdae6b, 0xfd8d3c, 0xf16913, 0xd94801, 0xa63603, 0x7f2704,
];
// cmocean (Thyng et al. 2016) maps as 12 evenly spaced samples: approximations of the
// originals, not the full 256-entry tables.
const THERMAL: &[u32] = &[
    0x032333, 0x0d3064, 0x35329b, 0x5d3e99, 0x7e4d8f, 0x9e5987, 0xc16479, 0xe17161, 0xf68b45,
    0xfbad3c, 0xf6d346, 0xe7fa5a,
];
const PHASE: &[u32] = &[
    0xa7770c, 0xc56033, 0xd94360, 0xdd26a3, 0xc43be0, 0x9961f4, 0x5f7fe4, 0x2890b7, 0x0f9788,
    0x27994f, 0x778d11, 0xa7770c,
];
const BALANCE: &[u32] = &[
    0x171c42, 0x293a8f, 0x0b66bd, 0x4590b9, 0x8eb5c2, 0xd2d8db, 0xe6d2cc, 0xd59d89, 0xc46548,
    0xac2b24, 0x780e28, 0x3c0911,
];
/// The near-white centre of cmocean `balance`, which 12 even samples straddle.
const BALANCE_CENTRE: u32 = 0xf1eceb;

fn balance_stops() -> Vec<Stop> {
    let mut stops = even_stops(BALANCE);
    stops.insert(
        BALANCE.len() / 2,
        Stop {
            pos: 0.5,
            color: rgb(BALANCE_CENTRE),
        },
    );
    stops
}

/// Moreland "cool to warm" (blue -> light grey -> red), piecewise-linear in sRGB.
fn coolwarm(t: f32) -> [f32; 3] {
    const LO: [f32; 3] = [0.230, 0.299, 0.754];
    const MID: [f32; 3] = [0.865, 0.865, 0.865];
    const HI: [f32; 3] = [0.706, 0.016, 0.150];
    if t < 0.5 {
        lerp3(LO, MID, t * 2.0)
    } else {
        lerp3(MID, HI, (t - 0.5) * 2.0)
    }
}

fn piecewise(points: &[(f32, f32)], t: f32) -> f32 {
    let mut prev = points[0];
    for &p in points {
        if t <= p.0 {
            let span = p.0 - prev.0;
            return if span > 0.0 {
                prev.1 + (p.1 - prev.1) * (t - prev.0) / span
            } else {
                p.1
            };
        }
        prev = p;
    }
    prev.1
}

/// The classic jet, as defined by matplotlib's segment data.
fn jet(t: f32) -> [f32; 3] {
    const R: &[(f32, f32)] = &[
        (0.0, 0.0),
        (0.35, 0.0),
        (0.66, 1.0),
        (0.89, 1.0),
        (1.0, 0.5),
    ];
    const G: &[(f32, f32)] = &[
        (0.0, 0.0),
        (0.125, 0.0),
        (0.375, 1.0),
        (0.64, 1.0),
        (0.91, 0.0),
        (1.0, 0.0),
    ];
    const B: &[(f32, f32)] = &[
        (0.0, 0.5),
        (0.11, 1.0),
        (0.34, 1.0),
        (0.65, 0.0),
        (1.0, 0.0),
    ];
    [piecewise(R, t), piecewise(G, t), piecewise(B, t)]
}

/// Full-saturation HSV sweep from blue (240 deg) down to red (0 deg), ParaView's legacy rainbow.
fn rainbow(t: f32) -> [f32; 3] {
    let h = (1.0 - t) * 4.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    match h as u32 {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        _ => [x, 0.0, 1.0],
    }
}

fn lerp3(a: [f32; 3], b: [f32; 3], u: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * u,
        a[1] + (b[1] - a[1]) * u,
        a[2] + (b[2] - a[2]) * u,
    ]
}

fn srgb_to_linear(c: f32) -> f64 {
    let c = f64::from(c).clamp(0.0, 1.0);
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f64) -> f32 {
    let c = c.clamp(0.0, 1.0);
    let v = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    v as f32
}

/// sRGB (0..1) to OKLab `[L, a, b]` (Björn Ottosson, 2020).
pub fn oklab_from_srgb(c: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (
        srgb_to_linear(c[0]),
        srgb_to_linear(c[1]),
        srgb_to_linear(c[2]),
    );
    let l = (0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b).cbrt();
    let m = (0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b).cbrt();
    let s = (0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b).cbrt();
    [
        (0.210_454_255_3 * l + 0.793_617_785 * m - 0.004_072_046_8 * s) as f32,
        (1.977_998_495_1 * l - 2.428_592_205 * m + 0.450_593_709_9 * s) as f32,
        (0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766 * s) as f32,
    ]
}

/// OKLab to sRGB (0..1); colours outside the sRGB gamut are clamped per channel.
pub fn srgb_from_oklab(c: [f32; 3]) -> [f32; 3] {
    let (ll, a, b) = (f64::from(c[0]), f64::from(c[1]), f64::from(c[2]));
    let l = (ll + 0.396_337_777_4 * a + 0.215_803_757_3 * b).powi(3);
    let m = (ll - 0.105_561_345_8 * a - 0.063_854_172_8 * b).powi(3);
    let s = (ll - 0.089_484_177_5 * a - 1.291_485_548 * b).powi(3);
    [
        linear_to_srgb(4.076_741_662_1 * l - 3.307_711_591_3 * m + 0.230_969_929_2 * s),
        linear_to_srgb(-1.268_438_004_6 * l + 2.609_757_401_1 * m - 0.341_319_396_5 * s),
        linear_to_srgb(-0.004_196_086_3 * l - 0.703_418_614_7 * m + 1.707_614_701 * s),
    ]
}

/// `#rgb`, `#rrggbb` or `#rrggbbaa` (alpha ignored), with or without the `#`.
pub fn parse_hex(s: &str) -> Option<[f32; 3]> {
    let s = s.trim();
    let s = s.strip_prefix('#').unwrap_or(s);
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let channel = |range: std::ops::Range<usize>| {
        u8::from_str_radix(&s[range], 16)
            .ok()
            .map(|v| f32::from(v) / 255.0)
    };
    match s.len() {
        3 => {
            let nibble = |i: usize| {
                u8::from_str_radix(&s[i..=i], 16)
                    .ok()
                    .map(|v| f32::from(v * 17) / 255.0)
            };
            Some([nibble(0)?, nibble(1)?, nibble(2)?])
        }
        6 | 8 => Some([channel(0..2)?, channel(2..4)?, channel(4..6)?]),
        _ => None,
    }
}

/// `#rrggbb`, lower case.
pub fn to_hex(c: [f32; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", to_u8(c[0]), to_u8(c[1]), to_u8(c[2]))
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ColormapError {
    #[error("Not a colormap: expected JSON")]
    NotJson,
    #[error("Not a colormap: expected colour stops, a ParaView preset or a list of colours")]
    Unrecognized,
    #[error("No colour stops found")]
    NoStops,
    #[error("Invalid colour: {0}")]
    BadColor(String),
    #[error("Invalid colormap: {0}")]
    Invalid(String),
}

/// A control colour of a custom map: `pos` in 0..1, `color` sRGB in 0..1.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Stop {
    pub pos: f32,
    #[serde(serialize_with = "ser_color", deserialize_with = "de_color")]
    pub color: [f32; 3],
}

fn ser_color<S: Serializer>(c: &[f32; 3], s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&to_hex(*c))
}

fn de_color<'de, D: Deserializer<'de>>(d: D) -> Result<[f32; 3], D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Hex(String),
        Rgb([f32; 3]),
    }
    match Repr::deserialize(d)? {
        Repr::Hex(s) => {
            parse_hex(&s).ok_or_else(|| serde::de::Error::custom(format!("invalid colour {s:?}")))
        }
        Repr::Rgb(c) => Ok(c),
    }
}

/// The space in which neighbouring stops are blended.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum Interp {
    #[default]
    Oklab,
    Srgb,
}

/// A user-defined colormap. On the wire the stop colours are `#rrggbb` strings.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CustomMap {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub interp: Interp,
    pub stops: Vec<Stop>,
}

/// Colour at `t` for stops sorted by position. Of several stops at one position the last
/// one applies from that position on, which gives a hard edge.
fn sample_stops(stops: &[Stop], interp: Interp, t: f32) -> [f32; 3] {
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else {
        return [0.0; 3];
    };
    let t = if t.is_nan() { 0.0 } else { t.clamp(0.0, 1.0) };
    if t < first.pos {
        return first.color;
    }
    let i = stops.partition_point(|s| s.pos <= t).saturating_sub(1);
    let Some(b) = stops.get(i + 1) else {
        return last.color;
    };
    let a = &stops[i];
    let u = (t - a.pos) / (b.pos - a.pos);
    if u <= 0.0 {
        return a.color;
    }
    match interp {
        Interp::Srgb => lerp3(a.color, b.color, u),
        Interp::Oklab => {
            srgb_from_oklab(lerp3(oklab_from_srgb(a.color), oklab_from_srgb(b.color), u))
        }
    }
}

impl CustomMap {
    /// The same map with usable stops: sorted by position, positions and colours clamped to
    /// 0..1, non-finite stops dropped, and at least two stops (a single stop is duplicated,
    /// none becomes black to white).
    pub fn normalized(&self) -> CustomMap {
        let mut stops: Vec<Stop> = self
            .stops
            .iter()
            .filter(|s| s.pos.is_finite() && s.color.iter().all(|c| c.is_finite()))
            .map(|s| Stop {
                pos: s.pos.clamp(0.0, 1.0),
                color: s.color.map(|c| c.clamp(0.0, 1.0)),
            })
            .collect();
        stops.sort_by(|a, b| a.pos.total_cmp(&b.pos));
        match stops.len() {
            0 => {
                stops = vec![
                    Stop {
                        pos: 0.0,
                        color: [0.0; 3],
                    },
                    Stop {
                        pos: 1.0,
                        color: [1.0; 3],
                    },
                ];
            }
            1 => {
                let color = stops[0].color;
                stops = vec![Stop { pos: 0.0, color }, Stop { pos: 1.0, color }];
            }
            _ => {}
        }
        CustomMap {
            id: self.id.clone(),
            name: self.name.clone(),
            interp: self.interp,
            stops,
        }
    }

    /// sRGB in 0..1 at `t` in 0..1 (clamped), blended between the neighbouring stops. This is
    /// the exact colour; [`Colormap::sample`] of a registered map reads the 8-bit table.
    pub fn sample(&self, t: f32) -> [f32; 3] {
        let map = self.normalized();
        sample_stops(&map.stops, map.interp, t)
    }

    pub fn lut(&self) -> Lut {
        let map = self.normalized();
        table_lut(&map.stops, map.interp)
    }

    pub fn css_gradient(&self, stops: usize) -> String {
        let lut = self.lut();
        gradient_css(stops, |t| sample_lut(&lut, t))
    }

    /// Pretty JSON in the native format:
    /// `{"id", "name", "interp", "stops": [{"pos": 0.0, "color": "#rrggbb"}]}`.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Reads colormaps from text: the native format (one object or an array), a ParaView
    /// JSON colour preset (`RGBPoints`), or a matplotlib-style list of colours. The returned
    /// maps are normalised and have distinct, non-empty ids.
    pub fn import(text: &str) -> Result<Vec<CustomMap>, ColormapError> {
        let value: Value = serde_json::from_str(text.trim()).map_err(|_| ColormapError::NotJson)?;
        let mut maps = match &value {
            Value::Object(_) => vec![import_object(&value)?],
            Value::Array(items) if items.is_empty() => return Err(ColormapError::NoStops),
            Value::Array(items) if items.iter().all(Value::is_object) => {
                // ParaView preset files mix colour maps with entries that have none
                // (indexed colours, annotations): those are skipped.
                let mut maps = Vec::new();
                let mut first_err = None;
                for item in items {
                    match import_object(item) {
                        Ok(map) => maps.push(map),
                        Err(e) => {
                            first_err.get_or_insert(e);
                        }
                    }
                }
                if maps.is_empty() {
                    return Err(first_err.unwrap_or(ColormapError::NoStops));
                }
                maps
            }
            Value::Array(items) => vec![import_list(items)?],
            _ => return Err(ColormapError::Unrecognized),
        };
        let mut seen: Vec<String> = Vec::new();
        for map in &mut maps {
            let base = map.id.clone();
            let mut n = 2;
            while seen.contains(&map.id) {
                map.id = format!("{base}-{n}");
                n += 1;
            }
            seen.push(map.id.clone());
        }
        Ok(maps)
    }
}

fn slug(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_alphanumeric() {
            out.extend(ch.to_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    let out = out.trim_end_matches('-');
    if out.is_empty() {
        "imported".to_string()
    } else {
        out.to_string()
    }
}

fn import_object(value: &Value) -> Result<CustomMap, ColormapError> {
    let mut map = if let Some(points) = value.get("RGBPoints") {
        import_paraview(value, points)?
    } else if value.get("stops").is_some() {
        serde_json::from_value::<CustomMap>(value.clone())
            .map_err(|e| ColormapError::Invalid(e.to_string()))?
    } else {
        return Err(ColormapError::Unrecognized);
    };
    if map.stops.is_empty() {
        return Err(ColormapError::NoStops);
    }
    if map.name.trim().is_empty() {
        map.name = if map.id.trim().is_empty() {
            "Imported".to_string()
        } else {
            map.id.clone()
        };
    }
    if map.id.trim().is_empty() {
        map.id = slug(&map.name);
    }
    Ok(map.normalized())
}

fn import_paraview(value: &Value, points: &Value) -> Result<CustomMap, ColormapError> {
    let numbers: Vec<f64> = points
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default();
    let (quads, _) = numbers.as_chunks::<4>();
    if quads.is_empty() {
        return Err(ColormapError::NoStops);
    }
    // The x of an RGBPoint is in data units.
    let lo = quads.iter().map(|q| q[0]).fold(f64::INFINITY, f64::min);
    let hi = quads.iter().map(|q| q[0]).fold(f64::NEG_INFINITY, f64::max);
    let stops = quads
        .iter()
        .enumerate()
        .map(|(i, q)| Stop {
            pos: if hi > lo {
                ((q[0] - lo) / (hi - lo)) as f32
            } else {
                even_pos(i, quads.len())
            },
            color: [q[1] as f32, q[2] as f32, q[3] as f32],
        })
        .collect();
    let text = |key: &str| value.get(key).and_then(Value::as_str).unwrap_or_default();
    Ok(CustomMap {
        id: String::new(),
        name: text("Name").to_string(),
        interp: if text("ColorSpace").eq_ignore_ascii_case("rgb") {
            Interp::Srgb
        } else {
            Interp::Oklab
        },
        stops,
    })
}

fn even_pos(i: usize, n: usize) -> f32 {
    if n > 1 {
        i as f32 / (n - 1) as f32
    } else {
        0.0
    }
}

/// A colour of a matplotlib-style list; numeric triples may still be on a 0..255 scale.
struct ListColor {
    pos: Option<f32>,
    color: [f32; 3],
    numeric: bool,
}

fn list_color(value: &Value) -> Result<([f32; 3], bool), ColormapError> {
    match value {
        Value::String(s) => parse_hex(s)
            .map(|c| (c, false))
            .ok_or_else(|| ColormapError::BadColor(s.clone())),
        Value::Array(a) if a.len() >= 3 => {
            let channel = |v: &Value| {
                v.as_f64()
                    .map(|x| x as f32)
                    .ok_or_else(|| ColormapError::BadColor(v.to_string()))
            };
            Ok(([channel(&a[0])?, channel(&a[1])?, channel(&a[2])?], true))
        }
        other => Err(ColormapError::BadColor(other.to_string())),
    }
}

fn import_list(items: &[Value]) -> Result<CustomMap, ColormapError> {
    let mut colors = Vec::with_capacity(items.len());
    for item in items {
        let entry = match item {
            Value::Array(a) if a.len() == 2 && !a[1].is_number() => {
                let pos = a[0].as_f64().ok_or(ColormapError::Unrecognized)? as f32;
                let (color, numeric) = list_color(&a[1])?;
                ListColor {
                    pos: Some(pos),
                    color,
                    numeric,
                }
            }
            Value::Array(_) | Value::String(_) => {
                let (color, numeric) = list_color(item)?;
                ListColor {
                    pos: None,
                    color,
                    numeric,
                }
            }
            _ => return Err(ColormapError::Unrecognized),
        };
        colors.push(entry);
    }
    let bytes = colors
        .iter()
        .any(|c| c.numeric && c.color.iter().any(|x| *x > 1.0));
    let positioned = colors.iter().all(|c| c.pos.is_some());
    let n = colors.len();
    let stops = colors
        .iter()
        .enumerate()
        .map(|(i, c)| Stop {
            pos: match c.pos {
                Some(pos) if positioned => pos,
                _ => even_pos(i, n),
            },
            color: if bytes && c.numeric {
                c.color.map(|x| x / 255.0)
            } else {
                c.color
            },
        })
        .collect();
    Ok(CustomMap {
        id: "imported".to_string(),
        name: "Imported".to_string(),
        interp: Interp::default(),
        stops,
    }
    .normalized())
}

fn css_percent(x: f32) -> String {
    let s = format!("{:.2}", x * 100.0);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn css_stop(c: [f32; 3], at: f32) -> String {
    format!(
        "rgb({},{},{}) {}%",
        to_u8(c[0]),
        to_u8(c[1]),
        to_u8(c[2]),
        css_percent(at)
    )
}

fn gradient_css(stops: usize, color: impl Fn(f32) -> [f32; 3]) -> String {
    let stops = stops.max(2);
    let parts: Vec<String> = (0..stops)
        .map(|i| {
            let t = i as f32 / (stops - 1) as f32;
            css_stop(color(t), t)
        })
        .collect();
    format!("linear-gradient(to right, {})", parts.join(", "))
}

/// Process-wide table of custom maps. The app fills it from the user's and the project's
/// saved maps; a map keeps its slot (and so its [`Colormap::Custom`] value and atlas row)
/// for as long as it stays registered. Maps are stored [normalised](CustomMap::normalized).
pub mod registry {
    use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

    use super::{CustomMap, Lut, MAX_CUSTOM};

    struct Entry {
        map: CustomMap,
        lut: Lut,
    }

    impl Entry {
        fn new(map: CustomMap) -> Entry {
            let lut = map.lut();
            Entry { map, lut }
        }
    }

    pub(super) struct Table {
        slots: [Option<Entry>; MAX_CUSTOM],
        generation: u64,
    }

    impl Table {
        pub(super) const fn new() -> Table {
            Table {
                slots: [const { None }; MAX_CUSTOM],
                generation: 0,
            }
        }

        pub(super) fn generation(&self) -> u64 {
            self.generation
        }

        pub(super) fn slot_of(&self, id: &str) -> Option<u16> {
            self.slots
                .iter()
                .position(|s| s.as_ref().is_some_and(|e| e.map.id == id))
                .map(|i| i as u16)
        }

        pub(super) fn by_slot(&self, slot: u16) -> Option<&CustomMap> {
            self.entry(slot).map(|e| &e.map)
        }

        fn entry(&self, slot: u16) -> Option<&Entry> {
            self.slots.get(usize::from(slot)).and_then(Option::as_ref)
        }

        pub(super) fn list(&self) -> Vec<CustomMap> {
            self.slots.iter().flatten().map(|e| e.map.clone()).collect()
        }

        pub(super) fn set(&mut self, maps: Vec<CustomMap>) {
            let mut wanted: Vec<CustomMap> = Vec::new();
            for map in maps {
                if wanted.len() < MAX_CUSTOM && !wanted.iter().any(|m| m.id == map.id) {
                    wanted.push(map.normalized());
                }
            }
            let mut next: [Option<Entry>; MAX_CUSTOM] = [const { None }; MAX_CUSTOM];
            let mut fresh = Vec::new();
            for map in wanted {
                match self.slot_of(&map.id) {
                    Some(slot) => next[usize::from(slot)] = Some(Entry::new(map)),
                    None => fresh.push(map),
                }
            }
            for map in fresh {
                if let Some(slot) = next.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Entry::new(map));
                }
            }
            self.slots = next;
            self.generation += 1;
        }

        pub(super) fn upsert(&mut self, map: CustomMap) -> Option<u16> {
            let slot = match self.slot_of(&map.id) {
                Some(slot) => slot,
                None => self.slots.iter().position(Option::is_none)? as u16,
            };
            self.slots[usize::from(slot)] = Some(Entry::new(map.normalized()));
            self.generation += 1;
            Some(slot)
        }

        pub(super) fn remove(&mut self, id: &str) {
            if let Some(slot) = self.slot_of(id) {
                self.slots[usize::from(slot)] = None;
                self.generation += 1;
            }
        }
    }

    static TABLE: RwLock<Table> = RwLock::new(Table::new());

    fn read() -> RwLockReadGuard<'static, Table> {
        TABLE.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write() -> RwLockWriteGuard<'static, Table> {
        TABLE.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Replaces all maps. An id that was already registered keeps its slot; of several maps
    /// with one id the first counts; maps beyond [`MAX_CUSTOM`] are dropped.
    pub fn set(maps: Vec<CustomMap>) {
        write().set(maps);
    }

    /// Adds a map or replaces the one with the same id. Returns its slot, `None` when full.
    pub fn upsert(map: CustomMap) -> Option<u16> {
        write().upsert(map)
    }

    pub fn remove(id: &str) {
        write().remove(id);
    }

    pub fn get(id: &str) -> Option<CustomMap> {
        let table = read();
        table
            .slot_of(id)
            .and_then(|slot| table.by_slot(slot).cloned())
    }

    pub fn slot_of(id: &str) -> Option<u16> {
        read().slot_of(id)
    }

    pub fn by_slot(slot: u16) -> Option<CustomMap> {
        read().by_slot(slot).cloned()
    }

    /// The registered maps in slot order.
    pub fn list() -> Vec<CustomMap> {
        read().list()
    }

    /// Bumped by every change; renderers re-upload the atlas when it differs.
    pub fn generation() -> u64 {
        read().generation()
    }

    pub(super) fn with_lut<R>(slot: u16, f: impl FnOnce(&Lut) -> R) -> Option<R> {
        read().entry(slot).map(|e| f(&e.lut))
    }

    pub(super) fn luts() -> Vec<Option<Lut>> {
        read()
            .slots
            .iter()
            .map(|s| s.as_ref().map(|e| e.lut))
            .collect()
    }
}

/// Number of rows of the [`atlas`].
pub fn atlas_rows() -> u32 {
    (Colormap::BUILTIN.len() + MAX_CUSTOM) as u32
}

/// RGBA8 atlas, [`LUT_SIZE`] wide, one row per colormap: the [`Colormap::BUILTIN`] rows, then
/// [`MAX_CUSTOM`] custom rows (an empty slot is a grey ramp). Returns `(bytes, rows)`.
pub fn atlas() -> (Vec<u8>, u32) {
    let rows = atlas_rows();
    let mut bytes = Vec::with_capacity(rows as usize * LUT_SIZE * 4);
    for map in Colormap::BUILTIN {
        bytes.extend_from_slice(map.builtin_lut().as_flattened());
    }
    for lut in registry::luts() {
        bytes.extend_from_slice(lut.as_ref().unwrap_or_else(|| grey_lut()).as_flattened());
    }
    (bytes, rows)
}

/// Everything between a value and a colour, shared by the shaders and the CPU paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorScale {
    pub map: Colormap,
    /// Flips the colours; the axis keeps its direction.
    pub reverse: bool,
    /// 0 (or 1) = continuous, N >= 2 = N equal bands, each in the map's colour at its centre.
    pub steps: u32,
    /// Log10 scale; only applied when [`ColorScale::log_valid`] holds for the range.
    pub log: bool,
    /// Colour of values below the range; `None` clamps to the end colour.
    pub below: Option<[f32; 3]>,
    /// Colour of values above the range; `None` clamps to the end colour.
    pub above: Option<[f32; 3]>,
    /// Colour of NaN; alpha 0 leaves the background.
    pub nan: [f32; 4],
}

impl Default for ColorScale {
    fn default() -> Self {
        ColorScale {
            map: Colormap::Viridis,
            reverse: false,
            steps: 0,
            log: false,
            below: None,
            above: None,
            nan: [0.0; 4],
        }
    }
}

impl From<Colormap> for ColorScale {
    fn from(map: Colormap) -> Self {
        ColorScale {
            map,
            ..ColorScale::default()
        }
    }
}

impl ColorScale {
    /// Whether a log scale can show this range.
    pub fn log_valid(range: (f32, f32)) -> bool {
        range.0 > 0.0 && range.1 > range.0
    }

    /// Unclamped position of `v` on the legend bar: 0 at `range.0`, 1 at `range.1`. On a log
    /// scale a value <= 0 is at minus infinity. An empty range puts every value at 0.5.
    pub fn fraction(&self, v: f32, range: (f32, f32)) -> f32 {
        let (lo, hi) = range;
        if v.is_nan() {
            return f32::NAN;
        }
        if self.log && Self::log_valid(range) {
            if v <= 0.0 {
                return f32::NEG_INFINITY;
            }
            return (v / lo).log10() / (hi / lo).log10();
        }
        let span = hi - lo;
        if span == 0.0 {
            return 0.5;
        }
        (v - lo) / span
    }

    /// Fraction (clamped to 0..1) to table coordinate: bands first, then reverse.
    pub fn map_t(&self, f: f32) -> f32 {
        let mut t = if f.is_nan() { 0.0 } else { f.clamp(0.0, 1.0) };
        if self.steps >= 2 {
            let n = self.steps as f32;
            t = ((t * n).floor().min(n - 1.0) + 0.5) / n;
        }
        if self.reverse { 1.0 - t } else { t }
    }

    /// RGBA of a value: the NaN colour, the out-of-range colours, or the map (alpha 1).
    pub fn color(&self, v: f32, range: (f32, f32)) -> [f32; 4] {
        let f = self.fraction(v, range);
        if f.is_nan() {
            return self.nan;
        }
        let c = match (self.below, self.above) {
            (Some(below), _) if f < 0.0 => below,
            (_, Some(above)) if f > 1.0 => above,
            _ => self.map.sample(self.map_t(f)),
        };
        [c[0], c[1], c[2], 1.0]
    }

    /// CSS `linear-gradient` of the legend bar: `stops` samples when continuous, one
    /// hard-edged segment per band otherwise. Honours `reverse`.
    pub fn css_gradient(&self, stops: usize) -> String {
        if self.steps < 2 {
            return gradient_css(stops, |t| self.map.sample(self.map_t(t)));
        }
        let n = self.steps as f32;
        let parts: Vec<String> = (0..self.steps)
            .flat_map(|k| {
                let k = k as f32;
                let c = self.map.sample(self.map_t((k + 0.5) / n));
                [css_stop(c, k / n), css_stop(c, (k + 1.0) / n)]
            })
            .collect();
        format!("linear-gradient(to right, {})", parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
    }

    fn stop(pos: f32, hex: &str) -> Stop {
        Stop {
            pos,
            color: parse_hex(hex).expect("hex"),
        }
    }

    fn map(id: &str, stops: Vec<Stop>) -> CustomMap {
        CustomMap {
            id: id.to_string(),
            name: format!("Map {id}"),
            interp: Interp::Oklab,
            stops,
        }
    }

    fn chroma(c: [f32; 3]) -> f32 {
        let lab = oklab_from_srgb(c);
        (lab[1] * lab[1] + lab[2] * lab[2]).sqrt()
    }

    #[test]
    fn luts_are_opaque_and_agree_with_sample() {
        for m in Colormap::BUILTIN {
            let lut = m.lut();
            assert_eq!(lut.len(), LUT_SIZE);
            assert!(lut.iter().all(|px| px[3] == 255), "{m:?}");
            assert_eq!(&lut[0][..3], &m.sample_u8(0.0), "{m:?}");
            assert_eq!(&lut[LUT_SIZE - 1][..3], &m.sample_u8(1.0), "{m:?}");
            for i in (0..LUT_SIZE).step_by(15) {
                let t = i as f32 / (LUT_SIZE - 1) as f32;
                assert_eq!(&lut[i][..3], &m.sample_u8(t), "{m:?} at {i}");
            }
            let between = m.sample(100.5 / 255.0);
            let mean = |k: usize| (f32::from(lut[100][k]) + f32::from(lut[101][k])) / 510.0;
            assert!(close(between, [mean(0), mean(1), mean(2)], 1e-3), "{m:?}");
            assert_eq!(m.sample(-3.0), m.sample(0.0));
            assert_eq!(m.sample(7.0), m.sample(1.0));
            assert_eq!(m.sample(f32::NAN), m.sample(0.0));
        }
    }

    #[test]
    fn maps_match_matplotlib() {
        // matplotlib 3.11 at t = 0, 0.25, 0.5, 0.75, 1.
        let reference: &[(Colormap, [[f32; 3]; 5])] = &[
            (
                Colormap::Viridis,
                [
                    [0.267, 0.005, 0.329],
                    [0.230, 0.322, 0.546],
                    [0.128, 0.567, 0.551],
                    [0.369, 0.789, 0.383],
                    [0.993, 0.906, 0.144],
                ],
            ),
            (
                Colormap::Plasma,
                [
                    [0.050, 0.030, 0.528],
                    [0.495, 0.012, 0.658],
                    [0.798, 0.280, 0.470],
                    [0.973, 0.586, 0.252],
                    [0.940, 0.975, 0.131],
                ],
            ),
            (
                Colormap::Inferno,
                [
                    [0.001, 0.000, 0.014],
                    [0.342, 0.062, 0.429],
                    [0.736, 0.216, 0.330],
                    [0.978, 0.558, 0.035],
                    [0.988, 0.998, 0.645],
                ],
            ),
            (
                Colormap::Magma,
                [
                    [0.001, 0.000, 0.014],
                    [0.317, 0.072, 0.485],
                    [0.716, 0.215, 0.475],
                    [0.987, 0.536, 0.382],
                    [0.987, 0.991, 0.750],
                ],
            ),
            (
                Colormap::Cividis,
                [
                    [0.000, 0.135, 0.305],
                    [0.264, 0.308, 0.423],
                    [0.489, 0.485, 0.471],
                    [0.736, 0.681, 0.424],
                    [0.996, 0.909, 0.218],
                ],
            ),
            (
                Colormap::Turbo,
                [
                    [0.190, 0.072, 0.232],
                    [0.158, 0.736, 0.923],
                    [0.644, 0.990, 0.234],
                    [0.984, 0.493, 0.128],
                    [0.480, 0.016, 0.011],
                ],
            ),
            (
                Colormap::Twilight,
                [
                    [0.886, 0.850, 0.888],
                    [0.384, 0.461, 0.731],
                    [0.185, 0.079, 0.213],
                    [0.698, 0.338, 0.322],
                    [0.886, 0.850, 0.886],
                ],
            ),
            (
                Colormap::RdBu,
                [
                    [0.404, 0.000, 0.122],
                    [0.899, 0.514, 0.408],
                    [0.966, 0.967, 0.968],
                    [0.409, 0.669, 0.815],
                    [0.020, 0.188, 0.380],
                ],
            ),
            (
                Colormap::BrBG,
                [
                    [0.329, 0.188, 0.020],
                    [0.813, 0.636, 0.336],
                    [0.957, 0.960, 0.960],
                    [0.346, 0.692, 0.653],
                    [0.000, 0.235, 0.188],
                ],
            ),
            (
                Colormap::Blues,
                [
                    [0.969, 0.984, 1.000],
                    [0.775, 0.858, 0.937],
                    [0.417, 0.681, 0.838],
                    [0.127, 0.440, 0.707],
                    [0.031, 0.188, 0.420],
                ],
            ),
            (
                Colormap::Greys,
                [
                    [1.000, 1.000, 1.000],
                    [0.850, 0.850, 0.850],
                    [0.586, 0.586, 0.586],
                    [0.317, 0.317, 0.317],
                    [0.000, 0.000, 0.000],
                ],
            ),
            (
                Colormap::YlGnBu,
                [
                    [1.000, 1.000, 0.851],
                    [0.778, 0.913, 0.706],
                    [0.253, 0.711, 0.768],
                    [0.134, 0.365, 0.657],
                    [0.031, 0.114, 0.345],
                ],
            ),
            (
                Colormap::Oranges,
                [
                    [1.000, 0.961, 0.922],
                    [0.992, 0.815, 0.634],
                    [0.991, 0.551, 0.233],
                    [0.846, 0.281, 0.004],
                    [0.498, 0.153, 0.016],
                ],
            ),
            (
                Colormap::Jet,
                [
                    [0.000, 0.000, 0.500],
                    [0.000, 0.504, 1.000],
                    [0.490, 1.000, 0.478],
                    [1.000, 0.582, 0.000],
                    [0.500, 0.000, 0.000],
                ],
            ),
        ];
        for (m, want) in reference {
            for (i, w) in want.iter().enumerate() {
                let got = m.sample(i as f32 / 4.0);
                assert!(close(got, *w, 0.03), "{m:?} at {i}: {got:?} vs {w:?}");
            }
        }
    }

    #[test]
    fn coolwarm_keeps_the_moreland_points() {
        assert!(close(
            Colormap::Coolwarm.sample(0.0),
            [0.230, 0.299, 0.754],
            0.003
        ));
        assert!(close(
            Colormap::Coolwarm.sample(1.0),
            [0.706, 0.016, 0.150],
            0.003
        ));
        let m = Colormap::Coolwarm.sample(0.5);
        assert!(
            (m[0] - m[1]).abs() < 0.005 && (m[1] - m[2]).abs() < 0.005,
            "{m:?}"
        );
        assert!((m[0] - 0.865).abs() < 0.01);
    }

    #[test]
    fn classic_and_cmocean_maps_have_the_expected_ends() {
        assert!(close(Colormap::Rainbow.sample(0.0), [0.0, 0.0, 1.0], 0.01));
        assert!(close(Colormap::Rainbow.sample(0.5), [0.0, 1.0, 0.0], 0.02));
        assert!(close(Colormap::Rainbow.sample(1.0), [1.0, 0.0, 0.0], 0.01));
        // Dark blue -> near white -> dark red.
        let (lo, mid, hi) = (
            Colormap::Balance.sample(0.0),
            Colormap::Balance.sample(0.5),
            Colormap::Balance.sample(1.0),
        );
        assert!(lo[2] > lo[0] && hi[0] > hi[2]);
        assert!(mid.iter().all(|c| *c > 0.9), "{mid:?}");
        // Dark blue -> yellow.
        let l = |t: f32| oklab_from_srgb(Colormap::Thermal.sample(t))[0];
        assert!(l(0.0) < 0.3 && l(1.0) > 0.9);
    }

    #[test]
    fn lightness_is_monotone_for_sequential_uniform_maps() {
        for m in [
            Colormap::Viridis,
            Colormap::Cividis,
            Colormap::Greys,
            Colormap::Thermal,
        ] {
            let lut = m.lut();
            let l: Vec<f32> = lut
                .iter()
                .map(|px| {
                    oklab_from_srgb([
                        f32::from(px[0]) / 255.0,
                        f32::from(px[1]) / 255.0,
                        f32::from(px[2]) / 255.0,
                    ])[0]
                })
                .collect();
            let sign = (l[LUT_SIZE - 1] - l[0]).signum();
            assert!((l[LUT_SIZE - 1] - l[0]).abs() > 0.5, "{m:?}");
            // One 8-bit step of rounding noise is allowed.
            assert!(l.windows(2).all(|w| (w[1] - w[0]) * sign > -0.004), "{m:?}");
        }
    }

    #[test]
    fn cyclic_maps_close_the_loop() {
        for m in [Colormap::Twilight, Colormap::Phase] {
            assert_eq!(m.family(), Family::Cyclic);
            let lut = m.lut();
            assert_eq!(lut[0], lut[LUT_SIZE - 1], "{m:?}");
        }
    }

    #[test]
    fn names_parse_back_and_indices_are_contiguous() {
        for (i, m) in Colormap::BUILTIN.iter().enumerate() {
            assert_eq!(m.index(), i as u32, "{m:?}");
            assert_eq!(Colormap::parse(m.as_str()), *m);
            assert_eq!(Colormap::parse(&m.name()), *m);
            assert_eq!(Colormap::parse(&m.label()), *m);
            assert_eq!(Colormap::parse(&m.as_str().to_uppercase()), *m);
            assert_eq!(m.as_str(), m.label().to_lowercase());
            assert_ne!(m.family(), Family::Custom);
        }
        let names: HashSet<&str> = Colormap::BUILTIN.iter().map(|m| m.as_str()).collect();
        assert_eq!(names.len(), Colormap::BUILTIN.len());
        assert_eq!(
            Colormap::Custom(3).index(),
            Colormap::BUILTIN.len() as u32 + 3
        );
        assert_eq!(Colormap::parse("RdBu"), Colormap::RdBu);
        assert_eq!(Colormap::parse(" brbg "), Colormap::BrBG);
        assert_eq!(Colormap::parse("gray"), Colormap::Greys);
        assert_eq!(Colormap::parse("no such map"), Colormap::Viridis);
        assert_eq!(Colormap::parse(""), Colormap::Viridis);
        assert_eq!(Colormap::default(), Colormap::Viridis);
    }

    #[test]
    fn families_and_uniformity() {
        assert_eq!(Family::Uniform.label(), "Perceptually uniform");
        assert_eq!(
            Family::Uniform.builtins(),
            vec![
                Colormap::Viridis,
                Colormap::Plasma,
                Colormap::Inferno,
                Colormap::Magma,
                Colormap::Cividis,
                Colormap::Turbo
            ]
        );
        assert_eq!(
            Family::Classic.builtins(),
            vec![Colormap::Jet, Colormap::Rainbow]
        );
        assert!(Family::Custom.builtins().is_empty());
        let total: usize = Family::ALL.iter().map(|f| f.builtins().len()).sum();
        assert_eq!(total, Colormap::BUILTIN.len());
        assert!(Colormap::Viridis.perceptually_uniform());
        assert!(Colormap::RdBu.perceptually_uniform());
        for m in [
            Colormap::Turbo,
            Colormap::Jet,
            Colormap::Rainbow,
            Colormap::Custom(0),
        ] {
            assert!(!m.perceptually_uniform(), "{m:?}");
        }
        assert_eq!(Colormap::Custom(0).family(), Family::Custom);
        assert_eq!(Colormap::Custom(0).as_str(), "custom");
    }

    #[test]
    fn css_gradient_keeps_its_format() {
        let css = Colormap::Coolwarm.css_gradient(5);
        assert!(
            css.starts_with("linear-gradient(to right, rgb(59,76,192) 0%, "),
            "{css}"
        );
        assert!(css.ends_with(" 100%)"), "{css}");
        assert_eq!(css.matches("rgb(").count(), 5);
        assert!(Colormap::Viridis.css_gradient(9).contains(" 12.5%"));
        assert_eq!(Colormap::Viridis.css_gradient(0).matches("rgb(").count(), 2);
    }

    #[test]
    fn oklab_round_trips_and_hex_parses() {
        for c in [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.6, 0.9],
            [1.0, 0.0, 0.0],
        ] {
            assert!(close(srgb_from_oklab(oklab_from_srgb(c)), c, 1e-3), "{c:?}");
        }
        assert!((oklab_from_srgb([1.0, 1.0, 1.0])[0] - 1.0).abs() < 1e-3);
        // Outside the gamut: clamped, not wrapped.
        let c = srgb_from_oklab([0.7, 0.5, 0.5]);
        assert!(c.iter().all(|x| (0.0..=1.0).contains(x)));
        assert_eq!(parse_hex("#ff0080"), Some([1.0, 0.0, 128.0 / 255.0]));
        assert_eq!(parse_hex("FF0080"), parse_hex("#ff0080ff"));
        assert_eq!(parse_hex("#f08"), Some([1.0, 0.0, 136.0 / 255.0]));
        assert_eq!(parse_hex("#ff00"), None);
        assert_eq!(parse_hex("#gg0000"), None);
        assert_eq!(parse_hex("red"), None);
        assert_eq!(to_hex([1.0, 0.0, 128.0 / 255.0]), "#ff0080");
    }

    #[test]
    fn oklab_midpoint_of_black_and_white_is_mid_grey() {
        let mut m = map("bw", vec![stop(0.0, "#000000"), stop(1.0, "#ffffff")]);
        let oklab = m.sample(0.5);
        assert!((oklab_from_srgb(oklab)[0] - 0.5).abs() < 0.01);
        assert!((oklab[0] - oklab[1]).abs() < 1e-3 && (oklab[1] - oklab[2]).abs() < 1e-3);
        m.interp = Interp::Srgb;
        let srgb = m.sample(0.5);
        assert!(close(srgb, [0.5; 3], 1e-6));
        assert!((srgb[0] - oklab[0]).abs() > 0.05, "{oklab:?}");
    }

    #[test]
    fn oklab_midpoint_of_blue_and_yellow_keeps_its_colour() {
        let mut m = map("by", vec![stop(0.0, "#0000ff"), stop(1.0, "#ffff00")]);
        let oklab = m.sample(0.5);
        m.interp = Interp::Srgb;
        let srgb = m.sample(0.5);
        assert!(chroma(srgb) < 0.01, "{srgb:?}");
        assert!(chroma(oklab) > 0.05, "{oklab:?}");
        assert_eq!(m.sample(0.0), [0.0, 0.0, 1.0]);
        assert_eq!(m.sample(1.0), [1.0, 1.0, 0.0]);
    }

    #[test]
    fn duplicate_positions_give_a_hard_edge() {
        let m = map(
            "edge",
            vec![
                stop(0.0, "#ff0000"),
                stop(0.5, "#ff0000"),
                stop(0.5, "#0000ff"),
                stop(1.0, "#0000ff"),
            ],
        );
        assert!(close(m.sample(0.499), [1.0, 0.0, 0.0], 1e-3));
        assert!(close(m.sample(0.5), [0.0, 0.0, 1.0], 1e-3));
        let lut = m.lut();
        assert_eq!(lut[127], [255, 0, 0, 255]);
        assert_eq!(lut[128], [0, 0, 255, 255]);
    }

    #[test]
    fn stops_are_normalised() {
        let m = map(
            "messy",
            vec![
                stop(1.5, "#ffffff"),
                Stop {
                    pos: f32::NAN,
                    color: [1.0, 0.0, 0.0],
                },
                Stop {
                    pos: 0.5,
                    color: [f32::INFINITY, 0.0, 0.0],
                },
                Stop {
                    pos: -0.2,
                    color: [-1.0, 0.0, 2.0],
                },
            ],
        );
        let n = m.normalized();
        assert_eq!(
            n.stops,
            vec![
                Stop {
                    pos: 0.0,
                    color: [0.0, 0.0, 1.0]
                },
                Stop {
                    pos: 1.0,
                    color: [1.0, 1.0, 1.0]
                }
            ]
        );
        assert_eq!((n.id.as_str(), n.name.as_str()), ("messy", "Map messy"));
        assert_eq!(n.normalized(), n);
        assert_eq!(m.sample(0.0), [0.0, 0.0, 1.0]);

        // Stops that do not reach the ends hold their colour outside.
        let inner = map("inner", vec![stop(0.25, "#ff0000"), stop(0.75, "#0000ff")]);
        assert_eq!(inner.sample(0.1), [1.0, 0.0, 0.0]);
        assert_eq!(inner.sample(0.9), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn one_stop_is_a_flat_map_and_none_is_black_to_white() {
        let one = map("one", vec![stop(0.3, "#336699")]);
        assert_eq!(one.normalized().stops.len(), 2);
        let c = parse_hex("#336699").expect("hex");
        for t in [0.0, 0.3, 0.7, 1.0] {
            assert!(close(one.sample(t), c, 1e-3));
        }
        let none = map("none", vec![]);
        assert_eq!(none.sample(0.0), [0.0; 3]);
        assert_eq!(none.sample(1.0), [1.0; 3]);
        assert!(
            none.css_gradient(3)
                .starts_with("linear-gradient(to right, rgb(0,0,0) 0%, ")
        );
    }

    #[test]
    fn bands_give_exactly_n_colours() {
        let scale = ColorScale {
            steps: 5,
            ..ColorScale::default()
        };
        let colors: HashSet<[u8; 3]> = (0..=1000)
            .map(|i| {
                let c = scale.color(i as f32 / 1000.0, (0.0, 1.0));
                [to_u8(c[0]), to_u8(c[1]), to_u8(c[2])]
            })
            .collect();
        assert_eq!(colors.len(), 5);
        assert!((scale.map_t(0.0) - 0.1).abs() < 1e-6);
        assert!((scale.map_t(0.39) - 0.3).abs() < 1e-6);
        assert!((scale.map_t(1.0) - 0.9).abs() < 1e-6);
        let continuous = ColorScale {
            steps: 1,
            ..ColorScale::default()
        };
        assert_eq!(continuous.map_t(0.37), 0.37);
    }

    #[test]
    fn reverse_mirrors_the_colours_but_not_the_axis() {
        let plain = ColorScale::from(Colormap::Plasma);
        let reversed = ColorScale {
            reverse: true,
            ..plain
        };
        let range = (-2.0, 6.0);
        for v in [-2.0f32, -1.0, 0.5, 2.0, 6.0] {
            let mirrored = range.0 + range.1 - v;
            let (a, b) = (reversed.color(v, range), plain.color(mirrored, range));
            assert!(close([a[0], a[1], a[2]], [b[0], b[1], b[2]], 1e-5), "{v}");
            assert_eq!(reversed.fraction(v, range), plain.fraction(v, range));
        }
        let banded = ColorScale {
            steps: 4,
            reverse: true,
            ..plain
        };
        assert!((banded.map_t(0.1) - 0.875).abs() < 1e-6);
    }

    #[test]
    fn log_scale_centres_the_geometric_mean() {
        let scale = ColorScale {
            log: true,
            ..ColorScale::default()
        };
        let range = (0.01, 100.0);
        assert!(ColorScale::log_valid(range));
        assert!((scale.fraction(1.0, range) - 0.5).abs() < 1e-5);
        assert!((scale.fraction(0.01, range)).abs() < 1e-5);
        assert!((scale.fraction(100.0, range) - 1.0).abs() < 1e-5);
        assert!((scale.fraction(1000.0, range) - 1.25).abs() < 1e-5);
        assert_eq!(scale.fraction(0.0, range), f32::NEG_INFINITY);
        // A range a log scale cannot show falls back to linear.
        assert!(!ColorScale::log_valid((0.0, 1.0)));
        assert!(!ColorScale::log_valid((-1.0, 1.0)));
        assert!(!ColorScale::log_valid((2.0, 2.0)));
        assert!((scale.fraction(0.0, (-1.0, 1.0)) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fraction_is_unclamped() {
        let scale = ColorScale::default();
        assert_eq!(scale.fraction(15.0, (0.0, 10.0)), 1.5);
        assert_eq!(scale.fraction(-5.0, (0.0, 10.0)), -0.5);
        assert_eq!(scale.fraction(2.5, (0.0, 10.0)), 0.25);
        assert_eq!(scale.fraction(3.0, (3.0, 3.0)), 0.5);
        assert!(scale.fraction(f32::NAN, (0.0, 1.0)).is_nan());
    }

    #[test]
    fn out_of_range_and_nan_colours() {
        let clamped = ColorScale::default();
        let lo = Colormap::Viridis.sample(0.0);
        let hi = Colormap::Viridis.sample(1.0);
        assert_eq!(clamped.color(-9.0, (0.0, 1.0)), [lo[0], lo[1], lo[2], 1.0]);
        assert_eq!(clamped.color(9.0, (0.0, 1.0)), [hi[0], hi[1], hi[2], 1.0]);
        assert_eq!(clamped.color(f32::NAN, (0.0, 1.0)), [0.0; 4]);

        let marked = ColorScale {
            below: Some([0.0, 0.0, 1.0]),
            above: Some([1.0, 0.0, 1.0]),
            nan: [0.5, 0.5, 0.5, 1.0],
            ..ColorScale::default()
        };
        assert_eq!(marked.color(-0.001, (0.0, 1.0)), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(marked.color(1.001, (0.0, 1.0)), [1.0, 0.0, 1.0, 1.0]);
        assert_eq!(marked.color(f32::NAN, (0.0, 1.0)), [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(marked.color(0.0, (0.0, 1.0)), [lo[0], lo[1], lo[2], 1.0]);
        assert_eq!(marked.color(1.0, (0.0, 1.0)), [hi[0], hi[1], hi[2], 1.0]);
        // The low side stays the low side when the colours are reversed.
        let reversed = ColorScale {
            reverse: true,
            ..marked
        };
        assert_eq!(reversed.color(-1.0, (0.0, 1.0)), [0.0, 0.0, 1.0, 1.0]);
        // Zero is below every log range.
        let log = ColorScale {
            log: true,
            ..marked
        };
        assert_eq!(log.color(0.0, (1.0, 10.0)), [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn scale_gradient_honours_bands_and_reverse() {
        let rgb_of = |c: [u8; 3]| format!("rgb({},{},{})", c[0], c[1], c[2]);
        let reversed = ColorScale {
            reverse: true,
            ..ColorScale::default()
        };
        let css = reversed.css_gradient(5);
        let first = rgb_of(Colormap::Viridis.sample_u8(1.0));
        assert!(
            css.starts_with(&format!("linear-gradient(to right, {first} 0%, ")),
            "{css}"
        );

        let banded = ColorScale {
            steps: 3,
            ..ColorScale::default()
        };
        let css = banded.css_gradient(9);
        assert_eq!(css.matches("rgb(").count(), 6);
        assert_eq!(css.matches(" 33.33%").count(), 2);
        assert_eq!(css.matches(" 66.67%").count(), 2);
        let centre = rgb_of(Colormap::Viridis.sample_u8(0.5));
        assert!(
            css.contains(&format!("{centre} 33.33%, {centre} 66.67%")),
            "{css}"
        );
    }

    #[test]
    fn native_json_round_trips() {
        let m = CustomMap {
            id: "sunset".to_string(),
            name: "Sunset".to_string(),
            interp: Interp::Srgb,
            stops: vec![
                stop(0.0, "#20124d"),
                stop(0.25, "#c2185b"),
                stop(1.0, "#ffe082"),
            ],
        };
        let json = m.to_json();
        assert!(json.contains("\"color\": \"#c2185b\""), "{json}");
        assert!(json.contains("\"interp\": \"srgb\""), "{json}");
        assert!(json.find("\"id\"") < json.find("\"interp\""));
        assert!(json.find("\"interp\"") < json.find("\"stops\""));
        assert_eq!(CustomMap::import(&json), Ok(vec![m.clone()]));

        let other = map("other", vec![stop(0.0, "#000000"), stop(1.0, "#00ff00")]);
        let both = format!("[{}, {}]", m.to_json(), other.to_json());
        assert_eq!(CustomMap::import(&both), Ok(vec![m, other]));
    }

    #[test]
    fn native_json_accepts_arrays_and_missing_fields() {
        let text = r##"{"name": "Hot Spot!", "stops": [
            {"pos": 1, "color": [1, 0.5, 0]},
            {"pos": 0, "color": "#000"}
        ]}"##;
        let maps = CustomMap::import(text).expect("import");
        assert_eq!(maps.len(), 1);
        assert_eq!(maps[0].id, "hot-spot");
        assert_eq!(maps[0].name, "Hot Spot!");
        assert_eq!(maps[0].interp, Interp::Oklab);
        assert_eq!(
            maps[0].stops,
            vec![
                Stop {
                    pos: 0.0,
                    color: [0.0; 3]
                },
                Stop {
                    pos: 1.0,
                    color: [1.0, 0.5, 0.0]
                }
            ]
        );
        // Same id twice in one file.
        let twice = format!("[{text}, {text}]");
        let ids: Vec<String> = CustomMap::import(&twice)
            .expect("import")
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert_eq!(ids, ["hot-spot", "hot-spot-2"]);
    }

    #[test]
    fn paraview_presets_import_with_data_unit_positions() {
        let text = r#"[
            {
                "ColorSpace": "RGB",
                "Name": "Cool to Warm (test)",
                "NanColor": [1, 1, 0],
                "RGBPoints": [-10, 0.23, 0.299, 0.754, 5, 0.865, 0.865, 0.865, 20, 0.706, 0.016, 0.15]
            },
            {
                "ColorSpace": "Diverging",
                "Name": "Two",
                "RGBPoints": [0, 0, 0, 0, 1, 1, 1, 1]
            },
            { "Name": "Categories", "IndexedColors": [1, 0, 0, 0, 1, 0] }
        ]"#;
        let maps = CustomMap::import(text).expect("import");
        assert_eq!(maps.len(), 2);
        assert_eq!(maps[0].id, "cool-to-warm-test");
        assert_eq!(maps[0].name, "Cool to Warm (test)");
        assert_eq!(maps[0].interp, Interp::Srgb);
        let pos: Vec<f32> = maps[0].stops.iter().map(|s| s.pos).collect();
        assert_eq!(pos, [0.0, 0.5, 1.0]);
        assert_eq!(maps[0].stops[1].color, [0.865, 0.865, 0.865]);
        assert!(close(maps[0].sample(0.25), [0.5475, 0.582, 0.8095], 1e-4));
        assert_eq!(maps[1].interp, Interp::Oklab);

        let single = r#"{"Name": "One", "RGBPoints": [3, 1, 0, 0]}"#;
        let one = CustomMap::import(single).expect("import");
        assert_eq!(one[0].stops.len(), 2);
        assert_eq!(one[0].sample(0.5), [1.0, 0.0, 0.0]);
        assert_eq!(
            CustomMap::import(r#"{"Name": "Empty", "RGBPoints": []}"#),
            Err(ColormapError::NoStops)
        );
    }

    #[test]
    fn matplotlib_lists_import() {
        let black_red_white = vec![
            Stop {
                pos: 0.0,
                color: [0.0, 0.0, 0.0],
            },
            Stop {
                pos: 0.5,
                color: [1.0, 0.0, 0.0],
            },
            Stop {
                pos: 1.0,
                color: [1.0, 1.0, 1.0],
            },
        ];
        for text in [
            "[[0, 0, 0], [1, 0, 0], [1, 1, 1]]",
            "[[0, 0, 0], [255, 0, 0], [255, 255, 255]]",
            "[[0, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]]",
            r##"["#000000", "#ff0000", "#fff"]"##,
            "[[0.5, [1, 0, 0]], [0, [0, 0, 0]], [1, [1, 1, 1]]]",
            "[[0, [0, 0, 0]], [0.5, [255, 0, 0]], [1, [255, 255, 255]]]",
            r##"[[0, "#000000"], [0.5, "#ff0000"], [1.0, "#ffffff"]]"##,
        ] {
            let maps = CustomMap::import(text).expect(text);
            assert_eq!(maps.len(), 1, "{text}");
            assert_eq!(maps[0].stops, black_red_white, "{text}");
            assert_eq!(
                (maps[0].id.as_str(), maps[0].name.as_str()),
                ("imported", "Imported")
            );
            assert_eq!(maps[0].interp, Interp::Oklab);
        }
        let uneven =
            CustomMap::import(r##"[[0, "#000"], [0.2, "#f00"], [1, "#fff"]]"##).expect("import");
        assert_eq!(uneven[0].stops[1].pos, 0.2);
    }

    #[test]
    fn garbage_is_an_error() {
        assert_eq!(CustomMap::import("hello"), Err(ColormapError::NotJson));
        assert_eq!(CustomMap::import(""), Err(ColormapError::NotJson));
        assert_eq!(CustomMap::import("{}"), Err(ColormapError::Unrecognized));
        assert_eq!(CustomMap::import("42"), Err(ColormapError::Unrecognized));
        assert_eq!(CustomMap::import("[]"), Err(ColormapError::NoStops));
        assert_eq!(
            CustomMap::import("[1, 2, 3]"),
            Err(ColormapError::Unrecognized)
        );
        assert_eq!(
            CustomMap::import("[[1, 2]]"),
            Err(ColormapError::BadColor("[1,2]".into()))
        );
        assert_eq!(
            CustomMap::import(r#"["tomato"]"#),
            Err(ColormapError::BadColor("tomato".into()))
        );
        assert_eq!(
            CustomMap::import(r#"[{"a": 1}]"#),
            Err(ColormapError::Unrecognized)
        );
        assert_eq!(
            CustomMap::import(r#"{"id": "x", "stops": []}"#),
            Err(ColormapError::NoStops)
        );
        assert!(matches!(
            CustomMap::import(r#"{"stops": [{"pos": 0, "color": "nope"}]}"#),
            Err(ColormapError::Invalid(_))
        ));
        assert_eq!(
            ColormapError::NotJson.to_string(),
            "Not a colormap: expected JSON"
        );
        assert_eq!(ColormapError::NoStops.to_string(), "No colour stops found");
    }

    #[test]
    fn table_set_keeps_slots_and_drops_the_overflow() {
        let two = |id: &str| map(id, vec![stop(0.0, "#000000"), stop(1.0, "#ffffff")]);
        let mut table = registry::Table::new();
        table.set(vec![two("a"), two("b"), two("c")]);
        let (a, b, c) = (table.slot_of("a"), table.slot_of("b"), table.slot_of("c"));
        assert_eq!((a, b, c), (Some(0), Some(1), Some(2)));
        let generation = table.generation();

        // "a" is gone, "d" is new and listed first: "b" and "c" stay where they were.
        table.set(vec![two("d"), two("c"), two("b"), two("b")]);
        assert!(table.generation() > generation);
        assert_eq!(table.slot_of("a"), None);
        assert_eq!((table.slot_of("b"), table.slot_of("c")), (b, c));
        assert_eq!(table.slot_of("d"), Some(0));
        assert_eq!(table.list().len(), 3);

        let many: Vec<CustomMap> = (0..MAX_CUSTOM + 5).map(|i| two(&format!("m{i}"))).collect();
        table.set(many);
        assert_eq!(table.list().len(), MAX_CUSTOM);
        assert!(table.slot_of(&format!("m{}", MAX_CUSTOM - 1)).is_some());
        assert_eq!(table.slot_of(&format!("m{MAX_CUSTOM}")), None);
        assert_eq!(table.upsert(two("one too many")), None);
        assert_eq!(table.upsert(two("m3")), table.slot_of("m3"));

        table.remove("m3");
        let freed = table.upsert(two("fits now"));
        assert!(freed.is_some());
        assert_eq!(
            table.by_slot(freed.unwrap_or(0)).map(|m| m.id.as_str()),
            Some("fits now")
        );
        table.set(Vec::new());
        assert!(table.list().is_empty());
    }

    #[test]
    fn registry_upsert_remove_and_generation() {
        let id = "test-registry-upsert";
        let other_id = "test-registry-neighbour";
        let first = map(id, vec![stop(0.0, "#000000"), stop(1.0, "#ff0000")]);
        let start = registry::generation();
        let slot = registry::upsert(first.clone()).expect("free slot");
        let after_insert = registry::generation();
        assert!(after_insert > start);
        assert_eq!(registry::slot_of(id), Some(slot));
        assert_eq!(registry::get(id), Some(first.clone()));
        assert_eq!(registry::by_slot(slot), Some(first));
        assert!(registry::list().iter().any(|m| m.id == id));

        let other = registry::upsert(map(other_id, vec![stop(0.5, "#00ff00")])).expect("free slot");
        assert_ne!(other, slot);
        // Stored normalised.
        assert_eq!(registry::get(other_id).map(|m| m.stops.len()), Some(2));

        let changed = map(id, vec![stop(0.0, "#000000"), stop(1.0, "#0000ff")]);
        assert_eq!(registry::upsert(changed.clone()), Some(slot));
        assert!(registry::generation() > after_insert);
        assert_eq!(registry::get(id), Some(changed));
        assert_eq!(Colormap::Custom(slot).sample_u8(1.0), [0, 0, 255]);
        assert_eq!(registry::slot_of(other_id), Some(other));

        let before_remove = registry::generation();
        registry::remove(id);
        assert!(registry::generation() > before_remove);
        assert_eq!(registry::get(id), None);
        assert_eq!(registry::slot_of(id), None);
        assert_eq!(registry::slot_of(other_id), Some(other));
        registry::remove(other_id);
    }

    #[test]
    fn custom_maps_resolve_by_name_and_fill_their_atlas_row() {
        let id = "test-registry-atlas";
        let custom = CustomMap {
            id: id.to_string(),
            name: "Atlas Test".to_string(),
            interp: Interp::Oklab,
            stops: vec![
                stop(0.0, "#102030"),
                stop(0.4, "#e040fb"),
                stop(1.0, "#fff59d"),
            ],
        };
        let slot = registry::upsert(custom.clone()).expect("free slot");
        let cm = Colormap::parse(&format!("custom:{id}"));
        assert_eq!(cm, Colormap::Custom(slot));
        assert_eq!(Colormap::parse(&format!("Custom:{id}")), cm);
        assert_eq!(Colormap::custom(id), Some(cm));
        assert_eq!(cm.name(), format!("custom:{id}"));
        assert_eq!(Colormap::parse(&cm.name()), cm);
        assert_eq!(cm.label(), "Atlas Test");
        assert_eq!(cm.as_str(), "custom");
        assert_eq!(cm.family(), Family::Custom);
        assert_eq!(cm.index(), Colormap::BUILTIN.len() as u32 + u32::from(slot));
        assert_eq!(cm.lut(), custom.lut());
        assert_eq!(cm.sample_u8(0.0), [0x10, 0x20, 0x30]);
        assert!(close(cm.sample(0.4), custom.sample(0.4), 0.01));
        assert_eq!(cm.css_gradient(4), custom.css_gradient(4));

        let (bytes, rows) = atlas();
        assert_eq!(rows, atlas_rows());
        assert_eq!(rows as usize, Colormap::BUILTIN.len() + MAX_CUSTOM);
        assert_eq!(bytes.len(), rows as usize * LUT_SIZE * 4);
        let row = |m: Colormap| {
            let start = m.index() as usize * LUT_SIZE * 4;
            bytes[start..start + LUT_SIZE * 4].to_vec()
        };
        assert_eq!(row(cm), custom.lut().as_flattened());
        for m in Colormap::BUILTIN {
            assert_eq!(row(*m), m.lut().as_flattened(), "{m:?}");
        }
        // No test fills the registry, so the last slot is a grey ramp.
        let unused = Colormap::Custom(MAX_CUSTOM as u16 - 1);
        if registry::by_slot(MAX_CUSTOM as u16 - 1).is_none() {
            let grey = row(unused);
            assert_eq!(&grey[..4], &[0, 0, 0, 255]);
            assert_eq!(&grey[LUT_SIZE * 4 - 4..], &[255, 255, 255, 255]);
            assert_eq!(unused.sample_u8(0.5), [128, 128, 128]);
            assert_eq!(unused.name(), "custom");
        }

        registry::remove(id);
        assert_eq!(Colormap::parse(&format!("custom:{id}")), Colormap::Viridis);
        assert_eq!(Colormap::parse("custom:"), Colormap::Viridis);
    }
}
