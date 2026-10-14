// SPDX-License-Identifier: Apache-2.0
//! Orbit camera plus the handful of linear-algebra types it needs (kept local so the
//! crate stays dependency-light and wasm-friendly).

use std::ops::{Add, Mul, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn from_array(a: [f32; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn normalize(self) -> Vec3 {
        let l = self.length();
        if l > 1e-12 { self * (1.0 / l) } else { self }
    }
    pub fn min(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    pub fn max(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

/// Column-major 4x4 matrix (matches WGSL `mat4x4<f32>` layout).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4(pub [[f32; 4]; 4]);

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    pub fn to_cols_array_2d(self) -> [[f32; 4]; 4] {
        self.0
    }

    /// Right-handed perspective, depth 0..1 (wgpu convention).
    pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        let nf = 1.0 / (near - far);
        Mat4([
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, far * nf, -1.0],
            [0.0, 0.0, near * far * nf, 0.0],
        ])
    }

    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let f = (target - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        Mat4([
            [s.x, u.x, -f.x, 0.0],
            [s.y, u.y, -f.y, 0.0],
            [s.z, u.z, -f.z, 0.0],
            [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
        ])
    }

    pub fn matmul(self, o: Mat4) -> Mat4 {
        let a = self.0;
        let b = o.0;
        let mut r = [[0.0f32; 4]; 4];
        for (c, col) in r.iter_mut().enumerate() {
            for (rw, cell) in col.iter_mut().enumerate() {
                *cell = (0..4).map(|k| a[k][rw] * b[c][k]).sum();
            }
        }
        Mat4(r)
    }

    pub fn transform_point(self, p: Vec3) -> Vec3 {
        let m = self.0;
        let x = m[0][0] * p.x + m[1][0] * p.y + m[2][0] * p.z + m[3][0];
        let y = m[0][1] * p.x + m[1][1] * p.y + m[2][1] * p.z + m[3][1];
        let z = m[0][2] * p.x + m[1][2] * p.y + m[2][2] * p.z + m[3][2];
        let w = m[0][3] * p.x + m[1][3] * p.y + m[2][3] * p.z + m[3][3];
        if w.abs() > 1e-12 {
            Vec3::new(x / w, y / w, z / w)
        } else {
            Vec3::new(x, y, z)
        }
    }
}

const ORBIT_RAD_PER_PX: f32 = 0.008;

/// How pointer input maps to camera motion. The default is what ParaView and most CAD
/// and CFD post-processors do: the scene follows the hand when orbiting and panning, and
/// rolling the wheel away from the user zooms in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Navigation {
    /// Horizontal drags turn the camera instead of the object.
    pub invert_orbit_x: bool,
    /// Vertical drags turn the camera instead of the object.
    pub invert_orbit_y: bool,
    /// Rolling the wheel away from the user zooms out.
    pub invert_zoom: bool,
}

impl Navigation {
    /// Presets by name: `follow` (default), `camera` (drags move the camera, as in a
    /// turntable or first-person view), `custom` (the three switches as given).
    pub fn preset(name: &str, custom: Navigation) -> Navigation {
        match name {
            "camera" => Navigation {
                invert_orbit_x: true,
                invert_orbit_y: true,
                invert_zoom: false,
            },
            "custom" => custom,
            _ => Navigation::default(),
        }
    }
}

/// Orbit camera around `target`. Yaw/pitch in radians, `distance` in world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    /// World up axis. CFD surfaces are usually z-up ([`OrbitCamera::z_up`]).
    pub up: Vec3,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 3.0,
            yaw: -0.7,
            pitch: 0.45,
            fov_y: 40f32.to_radians(),
            near: 0.01,
            far: 100.0,
            up: Vec3::new(0.0, 1.0, 0.0),
        }
    }
}

impl OrbitCamera {
    /// Z-up camera looking at the body from the front-left-above (yaw/pitch chosen so
    /// an Ahmed body's nose and one flank are visible).
    pub fn z_up() -> Self {
        Self {
            up: Vec3::new(0.0, 0.0, 1.0),
            yaw: -2.35,
            pitch: 0.42,
            ..Default::default()
        }
    }

    /// Orthonormal basis `(right, up, forward0)` around the up axis.
    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let w = self.up.normalize();
        let seed = if w.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let r = seed.cross(w).normalize();
        let f = w.cross(r);
        (r, w, f)
    }

    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let (r, w, f) = self.basis();
        self.target + (r * (cp * cy) + w * sp + f * (cp * sy)) * self.distance
    }

    pub fn view(&self) -> Mat4 {
        Mat4::look_at(self.eye(), self.target, self.up)
    }

    pub fn proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective(self.fov_y, aspect.max(1e-3), self.near, self.far)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect).matmul(self.view())
    }

    /// Drag rotate: `dx`, `dy` in pixels (screen, y down). The surface under the pointer
    /// follows the hand: a drag to the left turns the near side of the object to the left,
    /// a drag down turns it down.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.orbit_with(Navigation::default(), dx, dy);
    }

    /// [`OrbitCamera::orbit`] under the user's navigation preferences.
    pub fn orbit_with(&mut self, nav: Navigation, dx: f32, dy: f32) {
        let sx = if nav.invert_orbit_x { -1.0 } else { 1.0 };
        let sy = if nav.invert_orbit_y { -1.0 } else { 1.0 };
        self.yaw -= sx * dx * ORBIT_RAD_PER_PX;
        self.pitch = (self.pitch + sy * dy * ORBIT_RAD_PER_PX).clamp(-1.5, 1.5);
    }

    /// Drag pan in the view plane: `dx`, `dy` in pixels, `height_px` of the viewport. The
    /// scene moves with the hand.
    pub fn pan(&mut self, dx: f32, dy: f32, height_px: f32) {
        let world_per_px = 2.0 * self.distance * (self.fov_y * 0.5).tan() / height_px.max(1.0);
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(self.up).normalize();
        let up = right.cross(fwd);
        self.target = self.target - right * (dx * world_per_px) + up * (dy * world_per_px);
    }

    /// Wheel zoom: positive `delta` (wheel towards the user) zooms out.
    pub fn zoom(&mut self, delta: f32) {
        self.zoom_with(Navigation::default(), delta);
    }

    /// [`OrbitCamera::zoom`] under the user's navigation preferences.
    pub fn zoom_with(&mut self, nav: Navigation, delta: f32) {
        let delta = if nav.invert_zoom { -delta } else { delta };
        self.distance =
            (self.distance * (delta * 0.0015).exp()).clamp(self.near * 4.0, self.far * 0.5);
    }

    /// Frame an axis-aligned bounding box.
    pub fn fit(&mut self, min: Vec3, max: Vec3) {
        self.target = (min + max) * 0.5;
        let radius = (max - min).length() * 0.5;
        self.distance = radius / (self.fov_y * 0.5).sin() * 1.05;
        self.near = (self.distance * 0.01).max(1e-4);
        self.far = self.distance * 20.0;
    }

    /// World-space ray through a screen pixel (origin top-left).
    pub fn ray(&self, px: f32, py: f32, width: f32, height: f32) -> (Vec3, Vec3) {
        let aspect = width / height.max(1.0);
        let ndc_x = (px / width.max(1.0)) * 2.0 - 1.0;
        let ndc_y = 1.0 - (py / height.max(1.0)) * 2.0;
        let t = (self.fov_y * 0.5).tan();
        let fwd = (self.target - self.eye()).normalize();
        let right = fwd.cross(self.up).normalize();
        let up = right.cross(fwd);
        let dir = (fwd + right * (ndc_x * t * aspect) + up * (ndc_y * t)).normalize();
        (self.eye(), dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_at_maps_target_to_negative_z() {
        let v = Mat4::look_at(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        );
        let p = v.transform_point(Vec3::ZERO);
        assert!((p.z + 5.0).abs() < 1e-5 && p.x.abs() < 1e-5 && p.y.abs() < 1e-5);
    }

    #[test]
    fn fit_puts_bounds_in_front_of_camera() {
        let mut c = OrbitCamera::default();
        c.fit(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let vp = c.view_proj(1.5);
        for corner in [Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, -1.0, -1.0)] {
            let p = vp.transform_point(corner);
            assert!(
                p.x.abs() <= 1.0 && p.y.abs() <= 1.0 && (0.0..=1.0).contains(&p.z),
                "{p:?}"
            );
        }
    }

    #[test]
    fn z_up_camera_keeps_z_vertical() {
        let mut c = OrbitCamera::z_up();
        c.fit(Vec3::new(-1.2, -0.2, 0.0), Vec3::new(0.0, 0.2, 0.3));
        let vp = c.view_proj(1.5);
        let lo = vp.transform_point(Vec3::new(-0.6, 0.0, 0.0));
        let hi = vp.transform_point(Vec3::new(-0.6, 0.0, 0.3));
        assert!(hi.y > lo.y, "z should map upwards on screen: {lo:?} {hi:?}");
        // The nose (-x face) faces the camera.
        assert!(c.eye().x < -0.6, "{:?}", c.eye());
    }

    /// Where the point of the bounds nearest to the camera lands on screen, in pixels
    /// (y down) of an 800 x 600 view.
    fn near_point_px(c: &OrbitCamera, p: Vec3) -> (f32, f32) {
        let n = c.view_proj(800.0 / 600.0).transform_point(p);
        ((n.x * 0.5 + 0.5) * 800.0, (0.5 - n.y * 0.5) * 600.0)
    }

    #[test]
    fn orbit_moves_the_near_surface_with_the_hand() {
        for mut c in [OrbitCamera::default(), OrbitCamera::z_up()] {
            c.fit(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
            // A point on the sphere around the target, on the camera's side.
            let p = c.target + (c.eye() - c.target).normalize();
            let (x0, y0) = near_point_px(&c, p);
            let mut left = c;
            left.orbit(-40.0, 0.0);
            let (x1, y1) = near_point_px(&left, p);
            assert!(
                x1 < x0 - 5.0,
                "drag left moves the surface left: {x0} -> {x1}"
            );
            // (a turn about the vertical at a pitched view carries the point on a flat arc)
            assert!((y1 - y0).abs() < (x1 - x0).abs() * 0.2, "{y0} -> {y1}");
            let mut right = c;
            right.orbit(40.0, 0.0);
            assert!(near_point_px(&right, p).0 > x0 + 5.0);
            let mut down = c;
            down.orbit(0.0, 40.0);
            let (x2, y2) = near_point_px(&down, p);
            assert!(
                y2 > y0 + 5.0,
                "drag down moves the surface down: {y0} -> {y2}"
            );
            assert!((x2 - x0).abs() < (y2 - y0).abs() * 0.2, "{x0} -> {x2}");
            let mut up = c;
            up.orbit(0.0, -40.0);
            assert!(near_point_px(&up, p).1 < y0 - 5.0);
            // The switches turn each axis around on its own.
            let nav = Navigation {
                invert_orbit_x: true,
                ..Default::default()
            };
            let mut inv = c;
            inv.orbit_with(nav, -40.0, 40.0);
            let (xi, yi) = near_point_px(&inv, p);
            assert!(
                xi > x0 + 5.0 && yi > y0 + 5.0,
                "({x0}, {y0}) -> ({xi}, {yi})"
            );
        }
    }

    #[test]
    fn pan_moves_the_scene_with_the_hand_and_wheel_away_zooms_in() {
        let mut c = OrbitCamera::z_up();
        c.fit(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let p = c.target;
        let (x0, y0) = near_point_px(&c, p);
        let mut moved = c;
        moved.pan(30.0, 20.0, 600.0);
        let (x1, y1) = near_point_px(&moved, p);
        assert!(
            (x1 - x0 - 30.0).abs() < 1.0 && (y1 - y0 - 20.0).abs() < 1.0,
            "({x1}, {y1})"
        );
        // A wheel rolled away from the user reports a negative delta.
        let mut z = c;
        z.zoom(-100.0);
        assert!(z.distance < c.distance);
        let mut z = c;
        z.zoom_with(
            Navigation {
                invert_zoom: true,
                ..Default::default()
            },
            -100.0,
        );
        assert!(z.distance > c.distance);
        assert_eq!(
            Navigation::preset("follow", Navigation::default()),
            Navigation::default()
        );
        assert!(Navigation::preset("camera", Navigation::default()).invert_orbit_x);
    }

    #[test]
    fn centre_ray_hits_target() {
        let mut c = OrbitCamera::default();
        c.fit(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let (o, d) = c.ray(400.0, 300.0, 800.0, 600.0);
        let to_target = (c.target - o).normalize();
        assert!(d.dot(to_target) > 0.9999);
    }
}
