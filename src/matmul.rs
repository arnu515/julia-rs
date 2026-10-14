//! y = x·Wᵀ + b with W packed once at load time for a 6×16 register-tile kernel, in f32 or in
//! integers (AVX-VNNI).

use std::{
    arch::{
        asm,
        x86_64::{
            __m256i, _mm256_cvtps_epi32, _mm256_max_ps, _mm256_min_ps, _mm256_packs_epi32,
            _mm256_packus_epi16, _mm256_permute4x64_epi64, _mm256_permutevar8x32_epi32,
        },
    },
    simd::{
        StdFloat,
        cmp::SimdOrd,
        f32x8, i8x32, i16x16, i32x8,
        num::{SimdFloat, SimdInt, SimdUint},
    },
    str::FromStr,
};

use color_eyre::eyre::bail;

use rayon::prelude::*;

/// Outputs per packed panel: two f32x8 accumulators per token.
const NR: usize = 16;
/// Tokens per kernel call: MR × 2 accumulators + 2 weight vectors + 1 broadcast = 15 of the 16
/// ymm registers.
const MR: usize = 6;
/// Tokens × outputs per parallel task (multiples of MR and NR). A task's weights (96 × inp)
/// stay in its core's L2 while its 48 tokens stream past, and small inputs still give every core
/// several tasks. (Sizing token blocks to the pool instead, so W is read fewer times, measured
/// no faster with several requests at once and 10% slower for a single one.)
const TB: usize = 8 * MR;
const CB: usize = 6 * NR;
/// Inputs summed in fresh accumulators before adding to the running total. One long f32 chain
/// over 1152 inputs loses ~4x the accuracy of PyTorch; blocks of 64 match it for ~2% speed.
const KC: usize = 64;

/// Largest integer in [`Precision::I16`]: 13-bit values. 12 bits changed logits by 0.09 on average
/// over 100 benchmark questions, 13 bits by 0.05, and 14 would need an f32 flush every 32 inputs.
const L16: f32 = 4095.0;
/// Input pairs summed in i32 before converting to f32: |each product| <= L16², and 128 of them
/// stay below 2³¹, so the integer sums can't overflow.
const KP16: usize = 64;
const _: () = assert!(2 * KP16 as u64 * (L16 as u64).pow(2) < 1 << 31);
/// Largest weight in [`Precision::I8`] (symmetric int8).
const L8: f32 = 127.0;

/// Number format of the packed weights and of the activations multiplied with them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    /// Exact f32 (the reference).
    #[default]
    F32,
    /// Weights with one scale per output, activations with one per token, both as 13-bit integers
    /// in i16. About bf16's accuracy and 2× the multiply-adds per instruction (`vpdpwssd`).
    I16,
    /// Weights as int8 with one scale per output, activations as u8 with a scale and zero point
    /// per token: 4× the multiply-adds per instruction (`vpdpbusd`), but lossy. On 100 benchmark
    /// questions, 9 answers differed from f32's and logits moved by 1.1 on average.
    I8,
}

impl FromStr for Precision {
    type Err = color_eyre::Report;
    fn from_str(s: &str) -> color_eyre::Result<Self> {
        Ok(match s {
            "f32" => Self::F32,
            "i16" => Self::I16,
            "i8" => Self::I8,
            _ => bail!("unknown precision {s:?} (f32, i16, i8)"),
        })
    }
}

/// A weight matrix `[out, inp]` repacked as panels of 16 outputs, so the kernel reads 16
/// outputs' weights for one input index (f32), pair (i16) or quad (i8) as two contiguous vectors.
/// The last panel is zero-padded.
pub struct Packed {
    panels: Panels,
    inp: usize,
    out: usize,
}

enum Panels {
    /// `[inp][16]` per panel.
    F32(Vec<[f32; NR]>),
    /// `[inp / 2][16][2]` per panel (`vpdpwssd` sums adjacent pairs), inputs padded to even, and
    /// one scale per output (padded with zeros).
    I16 {
        panels: Vec<[i16; 2 * NR]>,
        scale: Vec<f32>,
    },
    /// `[inp / 4][16][4]` per panel (`vpdpbusd` sums groups of 4), inputs padded to a multiple of
    /// 4, one scale per output and each output's sum of integer weights, for the zero points.
    I8 {
        panels: Vec<[i8; 4 * NR]>,
        scale: Vec<f32>,
        sum: Vec<i32>,
    },
}

impl Packed {
    /// Packs `w` (`[out, inp]`, row-major as in PyTorch), in parallel over panels.
    pub fn new(w: &[f32], inp: usize, precision: Precision) -> Self {
        let out = w.len() / inp;
        assert!(
            inp > 0 && w.len() == out * inp,
            "Packed: {} weights are not [_, {inp}]",
            w.len()
        );
        let rows = |p: usize| w[p * NR * inp..].chunks_exact(inp).take(NR).enumerate();
        let panels = match precision {
            Precision::F32 => {
                let mut panels = vec![[0f32; NR]; out.div_ceil(NR) * inp];
                (panels.par_chunks_exact_mut(inp).enumerate()).for_each(|(p, panel)| {
                    for (c, row) in rows(p) {
                        for (dst, &v) in panel.iter_mut().zip(row) {
                            dst[c] = v;
                        }
                    }
                });
                Panels::F32(panels)
            }
            Precision::I16 => {
                let kp = inp.div_ceil(2);
                let mut panels = vec![[0i16; 2 * NR]; out.div_ceil(NR) * kp];
                let mut scale = vec![0f32; out.div_ceil(NR) * NR];
                let chunks = panels
                    .par_chunks_exact_mut(kp)
                    .zip(scale.par_chunks_mut(NR));
                chunks.enumerate().for_each(|(p, (panel, scale))| {
                    for (c, row) in rows(p) {
                        let (s, inv) = scales(row, L16);
                        scale[c] = s;
                        for (k, &v) in row.iter().enumerate() {
                            panel[k / 2][2 * c + k % 2] = (v * inv).round_ties_even() as i16;
                        }
                    }
                });
                Panels::I16 { panels, scale }
            }
            Precision::I8 => {
                // u8 · i8 products are <= 255 · 127, so i32 sums can't overflow below 66k inputs.
                assert!(inp < 1 << 16, "Packed: {inp} inputs could overflow i8 sums");
                let kq = inp.div_ceil(4);
                let mut panels = vec![[0i8; 4 * NR]; out.div_ceil(NR) * kq];
                let (mut scale, mut sum) = (vec![0f32; out.div_ceil(NR) * NR], vec![0; 0]);
                sum.resize(scale.len(), 0);
                let chunks = (panels.par_chunks_exact_mut(kq))
                    .zip(scale.par_chunks_mut(NR))
                    .zip(sum.par_chunks_mut(NR));
                chunks.enumerate().for_each(|(p, ((panel, scale), sum))| {
                    for (c, row) in rows(p) {
                        let (s, inv) = scales(row, L8);
                        scale[c] = s;
                        for (k, &v) in row.iter().enumerate() {
                            let q = (v * inv).round_ties_even() as i8;
                            panel[k / 4][4 * c + k % 4] = q;
                            sum[c] += q as i32;
                        }
                    }
                });
                Panels::I8 { panels, scale, sum }
            }
        };
        Self { panels, inp, out }
    }
}

/// Largest |x|, vectorised.
fn max_abs(x: &[f32]) -> f32 {
    // As integers, |x| bit patterns order like the floats, and the max is one vpmaxsd (f32 max
    // compiles to a NaN-handling compare + blend).
    let (v, tail) = x.as_chunks::<8>();
    let m = (v.iter()).fold(i32x8::splat(0), |m, &v| {
        m.simd_max(f32x8::from_array(v).abs().to_bits().cast())
    });
    let m = f32::from_bits(m.reduce_max() as u32);
    tail.iter().fold(m, |m, v| m.max(v.abs()))
}

/// Symmetric scale of a row for integers up to `l`, and its inverse: x ≈ scale · round(x ·
/// inverse).
fn scales(x: &[f32], l: f32) -> (f32, f32) {
    let m = max_abs(x);
    if m > 0.0 { (m / l, l / m) } else { (0.0, 0.0) }
}

/// min(x, 0) and max(x, 0), vectorised.
fn min_max(x: &[f32]) -> (f32, f32) {
    let (v, tail) = x.as_chunks::<8>();
    let zero = f32x8::splat(0.0);
    let (lo, hi) = v.iter().fold((zero, zero), |(lo, hi), &v| {
        // SAFETY: AVX (crate target features, checked in `main`). Plain vminps/vmaxps; the
        // portable versions add NaN handling.
        unsafe {
            let v = f32x8::from_array(v).into();
            (
                _mm256_min_ps(lo.into(), v).into(),
                _mm256_max_ps(hi.into(), v).into(),
            )
        }
    });
    let (lo, hi) = (lo.reduce_min(), hi.reduce_max());
    tail.iter()
        .fold((lo, hi), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

/// The rows of `x` (`[n, inp]`) as 13-bit integers, each adjacent pair packed into one i32 (the
/// operand `vpdpwssd` broadcasts; odd `inp` is zero-padded), with one scale per row.
fn quantize_i16(x: &[f32], inp: usize) -> (Vec<i32>, Vec<f32>) {
    let (n, kp) = (x.len() / inp, inp.div_ceil(2));
    let (mut q, mut s) = (vec![0i32; n * kp], vec![0f32; n]);
    let rows = q
        .par_chunks_exact_mut(kp)
        .zip(&mut s)
        .zip(x.par_chunks_exact(inp));
    rows.with_min_len(16).for_each(|((q, s), x)| {
        let inv;
        (*s, inv) = scales(x, L16);
        // SAFETY: an i32 slice reinterpreted as twice as many i16s (x86 is little-endian, so
        // input 2k is the low half of pair k).
        let q = unsafe { std::slice::from_raw_parts_mut(q.as_mut_ptr().cast::<i16>(), 2 * kp) };
        let (xv, tail) = x.as_chunks::<16>();
        let (qv, qtail) = q.as_chunks_mut::<16>();
        let inv8 = f32x8::splat(inv);
        for (q, x) in qv.iter_mut().zip(xv) {
            let (a, b) = x.split_at(8);
            *q = to_i16(f32x8::from_slice(a) * inv8, f32x8::from_slice(b) * inv8).to_array();
        }
        for (q, &x) in qtail.iter_mut().zip(tail) {
            *q = (x * inv).round_ties_even() as i16;
        }
    });
    (q, s)
}

/// The rows of `x` (`[n, inp]`) as u8 with a scale and zero point per row (x ≈ scale · (u -
/// zero)), each 4 adjacent inputs packed into one i32 (the operand `vpdpbusd` broadcasts; inputs
/// past `inp` are 0 and meet zero-padded weights).
fn quantize_u8(x: &[f32], inp: usize) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let (n, kq) = (x.len() / inp, inp.div_ceil(4));
    let (mut q, mut s, mut z) = (vec![0i32; n * kq], vec![0f32; n], vec![0i32; n]);
    let rows = (q.par_chunks_exact_mut(kq).zip(&mut s).zip(&mut z)).zip(x.par_chunks_exact(inp));
    rows.with_min_len(16).for_each(|(((q, s), z), x)| {
        let (lo, hi) = min_max(x);
        if hi == lo {
            return; // all zeros: scale 0
        }
        *s = (hi - lo) / 255.0;
        let inv = 255.0 / (hi - lo);
        *z = (-lo * inv).round_ties_even() as i32;
        let zf = *z as f32;
        // SAFETY: an i32 slice reinterpreted as 4× as many u8s (little-endian, so input 4k + i
        // is byte i of quad k).
        let q = unsafe { std::slice::from_raw_parts_mut(q.as_mut_ptr().cast::<u8>(), 4 * kq) };
        let (xv, tail) = x.as_chunks::<32>();
        let (qv, qtail) = q.as_chunks_mut::<32>();
        let u = |x: &[f32]| f32x8::from_slice(x) * f32x8::splat(inv) + f32x8::splat(zf);
        for (q, x) in qv.iter_mut().zip(xv) {
            *q = to_u8([u(&x[..8]), u(&x[8..16]), u(&x[16..24]), u(&x[24..])]);
        }
        for (q, &x) in qtail.iter_mut().zip(tail) {
            *q = (x * inv + zf).round_ties_even().clamp(0.0, 255.0) as u8;
        }
    });
    (q, s, z)
}

/// `y`'s base pointer, shared by the tasks of one `linear` call. Each task writes only its own
/// (token block × output block) and those blocks partition `y`, so no element is written twice
/// or read, and the `&mut y` borrow outlives the parallel loop.
#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    /// SAFETY: `i..i + len` is in bounds, in the calling task's block, and not otherwise
    /// borrowed while the slice lives.
    unsafe fn slice<'a>(self, i: usize, len: usize) -> &'a mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.0.add(i), len) }
    }
}

/// y[n, out] = x[n, inp] · Wᵀ + b, in parallel over blocks of tokens × outputs.
pub fn linear(x: &[f32], w: &Packed, b: Option<&[f32]>, y: &mut [f32]) {
    let (inp, out) = (w.inp, w.out);
    let n = x.len() / inp;
    // A mismatched y would otherwise silently skip rows.
    assert!(
        x.len() == n * inp && y.len() == n * out && b.is_none_or(|b| b.len() == out),
        "linear: bad shapes (x {}, y {}, W [{out}, {inp}])",
        x.len(),
        y.len()
    );
    let y = Out(y.as_mut_ptr());
    match &w.panels {
        Panels::F32(panels) => tiles(n, out, |t, m, j| {
            let panel = &panels[j / NR * inp..][..inp];
            match m {
                MR => store(tile_f32::<MR>(x, panel, t), out, b, y, t, j),
                _ => store(tile_f32::<1>(x, panel, t), out, b, y, t, j),
            }
        }),
        Panels::I16 { panels, scale } => {
            let (xq, sx) = quantize_i16(x, inp);
            let kp = inp.div_ceil(2);
            tiles(n, out, |t, m, j| {
                let (panel, sw) = (&panels[j / NR * kp..][..kp], &scale[j..][..NR]);
                match m {
                    MR => store(tile_i16::<MR>(&xq, panel, &sx, sw, t), out, b, y, t, j),
                    _ => store(tile_i16::<1>(&xq, panel, &sx, sw, t), out, b, y, t, j),
                }
            })
        }
        Panels::I8 { panels, scale, sum } => {
            let (xq, sx, zx) = quantize_u8(x, inp);
            let kq = inp.div_ceil(4);
            tiles(n, out, |t, m, j| {
                let panel = &panels[j / NR * kq..][..kq];
                let w = (panel, &scale[j..][..NR], &sum[j..][..NR]);
                match m {
                    MR => store(tile_i8::<MR>(&xq, &sx, &zx, w, t), out, b, y, t, j),
                    _ => store(tile_i8::<1>(&xq, &sx, &zx, w, t), out, b, y, t, j),
                }
            })
        }
    }
}

/// Runs `tile(t, m, j)` for every `m`-token (`MR` or 1) × 16-output tile of an `[n, out]` result,
/// in parallel over blocks of tokens × outputs.
fn tiles(n: usize, out: usize, tile: impl Fn(usize, usize, usize) + Sync) {
    let nc = out.div_ceil(CB);
    (0..n.div_ceil(TB) * nc).into_par_iter().for_each(|task| {
        let (t0, j0) = (task / nc * TB, task % nc * CB);
        let (t1, j1) = ((t0 + TB).min(n), (j0 + CB).min(out));
        // Panel by panel, so each panel stays in L1 for all the block's tokens.
        for j in (j0..j1).step_by(NR) {
            let mut t = t0;
            while t + MR <= t1 {
                tile(t, MR, j);
                t += MR;
            }
            for t in t..t1 {
                tile(t, 1, j);
            }
        }
    });
}

/// x·Wᵀ for tokens `t..t + M` and one f32 panel's 16 outputs.
#[inline(always)]
fn tile_f32<const M: usize>(x: &[f32], panel: &[[f32; NR]], t: usize) -> [[f32x8; 2]; M] {
    let inp = panel.len();
    let xr: [&[f32]; M] = std::array::from_fn(|r| &x[(t + r) * inp..][..inp]);
    let mut sum = [[f32x8::splat(0.0); 2]; M];
    for (kb, panel) in panel.chunks(KC).enumerate() {
        let xb: [&[f32]; M] = std::array::from_fn(|r| &xr[r][kb * KC..][..panel.len()]);
        let mut acc = [[f32x8::splat(0.0); 2]; M];
        for (k, w) in panel.iter().enumerate() {
            let (w0, w1) = (f32x8::from_slice(&w[..8]), f32x8::from_slice(&w[8..]));
            for r in 0..M {
                let xv = f32x8::splat(xb[r][k]);
                acc[r][0] = xv.mul_add(w0, acc[r][0]);
                acc[r][1] = xv.mul_add(w1, acc[r][1]);
            }
        }
        for (s, a) in sum.iter_mut().zip(acc) {
            s[0] += a[0];
            s[1] += a[1];
        }
    }
    sum
}

/// [`tile_f32`] on 13-bit integers: `x` holds packed input pairs and `sx` the token scales, `sw`
/// the panel's output scales.
#[inline(always)]
fn tile_i16<const M: usize>(
    x: &[i32],
    panel: &[[i16; 2 * NR]],
    sx: &[f32],
    sw: &[f32],
    t: usize,
) -> [[f32x8; 2]; M] {
    let kp = panel.len();
    let xr: [&[i32]; M] = std::array::from_fn(|r| &x[(t + r) * kp..][..kp]);
    let mut sum = [[f32x8::splat(0.0); 2]; M];
    for (kb, panel) in panel.chunks(KP16).enumerate() {
        let xb: [&[i32]; M] = std::array::from_fn(|r| &xr[r][kb * KP16..][..panel.len()]);
        let mut acc = [[__m256i::from(i32x8::splat(0)); 2]; M];
        for (k, w) in panel.iter().enumerate() {
            let w0 = __m256i::from(i16x16::from_slice(&w[..NR]));
            let w1 = __m256i::from(i16x16::from_slice(&w[NR..]));
            for r in 0..M {
                let xv = __m256i::from(i32x8::splat(xb[r][k]));
                acc[r][0] = dpwssd(acc[r][0], xv, w0);
                acc[r][1] = dpwssd(acc[r][1], xv, w1);
            }
        }
        for (s, a) in sum.iter_mut().zip(acc) {
            s[0] += i32x8::from(a[0]).cast::<f32>();
            s[1] += i32x8::from(a[1]).cast::<f32>();
        }
    }
    let (s0, s1) = (f32x8::from_slice(&sw[..8]), f32x8::from_slice(&sw[8..]));
    for (r, s) in sum.iter_mut().enumerate() {
        let st = f32x8::splat(sx[t + r]);
        s[0] *= st * s0;
        s[1] *= st * s1;
    }
    sum
}

/// [`tile_f32`] on u8 activations (`x` packed in quads, `sx` scales, `zx` zero points) and one i8
/// panel (with its output scales and weight sums).
#[inline(always)]
fn tile_i8<const M: usize>(
    x: &[i32],
    sx: &[f32],
    zx: &[i32],
    (panel, sw, sum): (&[[i8; 4 * NR]], &[f32], &[i32]),
    t: usize,
) -> [[f32x8; 2]; M] {
    let kq = panel.len();
    let xr: [&[i32]; M] = std::array::from_fn(|r| &x[(t + r) * kq..][..kq]);
    let mut acc = [[__m256i::from(i32x8::splat(0)); 2]; M];
    for (k, w) in panel.iter().enumerate() {
        let w0 = __m256i::from(i8x32::from_slice(&w[..2 * NR]));
        let w1 = __m256i::from(i8x32::from_slice(&w[2 * NR..]));
        for r in 0..M {
            let xv = __m256i::from(i32x8::splat(xr[r][k]));
            acc[r][0] = dpbusd(acc[r][0], xv, w0);
            acc[r][1] = dpbusd(acc[r][1], xv, w1);
        }
    }
    // Σ (u - z)·w = Σ u·w - z·Σ w, exactly in i32; then scale.
    let (c0, c1) = (i32x8::from_slice(&sum[..8]), i32x8::from_slice(&sum[8..]));
    let (s0, s1) = (f32x8::from_slice(&sw[..8]), f32x8::from_slice(&sw[8..]));
    std::array::from_fn(|r| {
        let (z, s) = (i32x8::splat(zx[t + r]), f32x8::splat(sx[t + r]));
        let a = |acc: __m256i, c: i32x8| (i32x8::from(acc) - z * c).cast::<f32>();
        [a(acc[r][0], c0) * (s * s0), a(acc[r][1], c1) * (s * s1)]
    })
}

/// a and b rounded to the nearest integers (ties to even), in order. Values must fit i16.
#[inline(always)]
fn to_i16(a: f32x8, b: f32x8) -> i16x16 {
    // SAFETY: the crate is built with AVX2 (.cargo/config.toml), and `main` checks for it.
    unsafe {
        let (a, b) = (_mm256_cvtps_epi32(a.into()), _mm256_cvtps_epi32(b.into()));
        // packs interleaves 128-bit halves (a0-3 b0-3 a4-7 b4-7); the permute restores order.
        _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packs_epi32(a, b)).into()
    }
}

/// 32 values rounded to the nearest integers (ties to even) and clamped to u8, in order.
#[inline(always)]
fn to_u8(v: [f32x8; 4]) -> [u8; 32] {
    // SAFETY: AVX2 (crate target features, checked in `main`).
    unsafe {
        let [a, b, c, d] = v.map(|v| _mm256_cvtps_epi32(v.into()));
        // Saturating packs work per 128-bit half: dwords come out a0 b0 c0 d0 a1 b1 c1 d1, where
        // a0 is a's first 4 values.
        let p = _mm256_packus_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d));
        let order = i32x8::from_array([0, 4, 1, 5, 2, 6, 3, 7]).into();
        std::mem::transmute(_mm256_permutevar8x32_epi32(p, order))
    }
}

/// acc + Σ adjacent i16 pairs of a·b, per i32 lane.
#[inline(always)]
fn dpwssd(mut acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
    // Inline asm, since LLVM splits `_mm256_dpwssd_avx_epi32` into vpmaddwd + vpaddd, two µops
    // instead of one, which halved the kernel's speed.
    // SAFETY: the crate is built with AVX-VNNI (.cargo/config.toml), and `main` checks for it.
    unsafe {
        asm!(
            "{{vex}} vpdpwssd {acc}, {a}, {b}",
            acc = inout(ymm_reg) acc,
            a = in(ymm_reg) a,
            b = in(ymm_reg) b,
            options(pure, nomem, nostack),
        );
    }
    acc
}

/// acc + Σ groups of 4 u8 a × i8 b, per i32 lane.
#[inline(always)]
fn dpbusd(mut acc: __m256i, a: __m256i, b: __m256i) -> __m256i {
    // Inline asm for the same reason as `dpwssd`.
    // SAFETY: the crate is built with AVX-VNNI (.cargo/config.toml), and `main` checks for it.
    unsafe {
        asm!(
            "{{vex}} vpdpbusd {acc}, {a}, {b}",
            acc = inout(ymm_reg) acc,
            a = in(ymm_reg) a,
            b = in(ymm_reg) b,
            options(pure, nomem, nostack),
        );
    }
    acc
}

/// Writes a tile's sums plus bias to outputs `j..j + 16` (fewer in a padded last panel) of
/// tokens `t..t + M`.
#[inline(always)]
fn store<const M: usize>(
    acc: [[f32x8; 2]; M],
    out: usize,
    b: Option<&[f32]>,
    y: Out,
    t: usize,
    j: usize,
) {
    let cols = (out - j).min(NR);
    for (r, acc) in acc.iter().enumerate() {
        // SAFETY: token t + r and outputs j..j + cols are inside this task's block.
        let yr = unsafe { y.slice((t + r) * out + j, cols) };
        if cols == NR {
            let zero = f32x8::splat(0.0);
            let (b0, b1) = b.map_or((zero, zero), |b| {
                (f32x8::from_slice(&b[j..]), f32x8::from_slice(&b[j + 8..]))
            });
            (acc[0] + b0).copy_to_slice(&mut yr[..8]);
            (acc[1] + b1).copy_to_slice(&mut yr[8..]);
        } else {
            let v = [acc[0].to_array(), acc[1].to_array()].concat();
            for (c, y) in yr.iter_mut().enumerate() {
                *y = v[c] + b.map_or(0.0, |b| b[j + c]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{L8, L16, Packed, Precision, linear};

    /// The exact product, and how far `precision` may stray from it: rounding each integer to the
    /// nearest step moves x by ≤ sx/2 and w by ≤ sw/2, so a product by ≤ |x|·sw/2 + |w|·sx/2 +
    /// sx·sw/4. (f32: a 1e-5 margin only.)
    fn naive(
        x: &[f32],
        w: &[f32],
        b: Option<&[f32]>,
        inp: usize,
        precision: Precision,
    ) -> Vec<(f32, f32)> {
        let max = |r: &[f32]| r.iter().fold(0f32, |m, v| m.max(v.abs())) as f64;
        let range = |r: &[f32]| {
            let (lo, hi) = r
                .iter()
                .fold((0f32, 0f32), |(l, h), &v| (l.min(v), h.max(v)));
            (hi - lo) as f64
        };
        let steps = |x: &[f32], w: &[f32]| match precision {
            Precision::F32 => (0.0, 0.0),
            Precision::I16 => (max(x) / L16 as f64, max(w) / L16 as f64),
            Precision::I8 => (range(x) / 255.0, max(w) / L8 as f64),
        };
        let (n, out) = (x.len() / inp, w.len() / inp);
        let mut y = Vec::with_capacity(n * out);
        for xr in x.chunks_exact(inp) {
            for (o, wr) in w.chunks_exact(inp).enumerate() {
                let (sx, sw) = steps(xr, wr);
                let (mut dot, mut tol) = (0f64, 1e-5);
                for (&a, &c) in xr.iter().zip(wr) {
                    let (a, c) = (a as f64, c as f64);
                    dot += a * c;
                    tol += a.abs() * sw / 2.0 + c.abs() * sx / 2.0 + sx * sw / 4.0;
                }
                y.push((dot as f32 + b.map_or(0.0, |b| b[o]), tol as f32));
            }
        }
        y
    }

    /// `linear` redone in scalar code: the same rounding to integers, exact integer sums, then the
    /// scales. Also returns a tolerance for the kernel's f32 rounding of its integer sums.
    fn emulate(x: &[f32], w: &[f32], inp: usize, precision: Precision) -> Vec<(f32, f32)> {
        let max = |r: &[f32]| r.iter().fold(0f32, |m, v| m.max(v.abs()));
        // (integers, scale, zero point) of a row.
        let sym = |r: &[f32], l: f32| {
            let m = max(r);
            let inv = if m > 0.0 { l / m } else { 0.0 };
            let q = r.iter().map(|v| (v * inv).round_ties_even() as i64);
            (q.collect::<Vec<_>>(), m / l, 0)
        };
        let asym = |r: &[f32]| {
            let (lo, hi) = r
                .iter()
                .fold((0f32, 0f32), |(l, h), &v| (l.min(v), h.max(v)));
            if hi == lo {
                return (vec![0; r.len()], 0.0, 0);
            }
            let inv = 255.0 / (hi - lo);
            let z = (-lo * inv).round_ties_even();
            let q = r
                .iter()
                .map(|v| (v * inv + z).round_ties_even().clamp(0.0, 255.0) as i64);
            (q.collect(), (hi - lo) / 255.0, z as i64)
        };
        let mut y = vec![];
        for xr in x.chunks_exact(inp) {
            let (qx, sx, z) = match precision {
                Precision::I8 => asym(xr),
                _ => sym(xr, L16),
            };
            for wr in w.chunks_exact(inp) {
                let (qw, sw, _) = sym(wr, if precision == Precision::I8 { L8 } else { L16 });
                let dot: i64 = qx.iter().zip(&qw).map(|(a, c)| (a - z) * c).sum();
                let abs: i64 = qx.iter().zip(&qw).map(|(a, c)| ((a - z) * c).abs()).sum();
                let s = sx as f64 * sw as f64;
                y.push((
                    (dot as f64 * s) as f32,
                    (1e-6 * abs as f64 * s + 1e-6) as f32,
                ));
            }
        }
        y
    }

    #[test]
    fn matches_naive_and_emulation() {
        let mut s = 1u32;
        let mut rnd = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    (s >> 8) as f32 / (1 << 24) as f32 - 0.5
                })
                .collect()
        };
        // Model shapes; out == 1 (scorer); 1 token; counts that leave partial 6-token tiles,
        // padded 16-output panels and partial task blocks; odd inp (a padded i16 pair).
        let mut cases: Vec<_> = [
            (37, 384, 1152),
            (40, 1152, 384),
            (3, 384, 1),
            (1, 16, 3),
            (4, 8, 2),
            (150, 384, 2304),
            (130, 64, 200),
            (7, 13, 21),
        ]
        .into_iter()
        .map(|(n, inp, out)| (inp, rnd(n * inp), rnd(out * inp), rnd(out)))
        .collect();
        // Every integer at its limit and all products of one sign: the largest i32 sums possible
        // (i8: u8 255, from all-positive x, and zero point 0, in the second).
        cases.push((
            1152,
            vec![1.0; 12 * 1152],
            vec![-1.0; 20 * 1152],
            vec![0.0; 20],
        ));
        cases.push((
            1152,
            vec![-1.0; 12 * 1152],
            vec![1.0; 20 * 1152],
            vec![0.0; 20],
        ));
        for (inp, x, w, b) in &cases {
            for precision in [Precision::F32, Precision::I16, Precision::I8] {
                let packed = Packed::new(w, *inp, precision);
                // Task blocks are fixed, but run the tests on several pool sizes anyway.
                for (threads, b) in [1, 3, 22]
                    .into_iter()
                    .flat_map(|t| [(t, None), (t, Some(&b[..]))])
                {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let mut y = vec![0.0; x.len() / inp * (w.len() / inp)];
                    pool.install(|| linear(x, &packed, b, &mut y));
                    for (i, (&got, (want, tol))) in
                        y.iter().zip(naive(x, w, b, *inp, precision)).enumerate()
                    {
                        assert!(
                            (got - want).abs() <= tol,
                            "{precision:?} {}x{inp}->{} bias={} threads={threads}: y[{i}] = {got}, \
                             want {want} ± {tol}",
                            x.len() / inp,
                            w.len() / inp,
                            b.is_some(),
                        );
                    }
                    if precision == Precision::F32 || b.is_some() {
                        continue;
                    }
                    for (i, (&got, (want, tol))) in
                        y.iter().zip(emulate(x, w, *inp, precision)).enumerate()
                    {
                        assert!(
                            (got - want).abs() <= tol,
                            "{precision:?} {}x{inp}->{} threads={threads}: y[{i}] = {got}, \
                             emulated {want} ± {tol}",
                            x.len() / inp,
                            w.len() / inp,
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "bad shapes")]
    fn short_output_panics() {
        linear(
            &[0.0; 16],
            &Packed::new(&[0.0; 24], 8, Precision::F32),
            None,
            &mut [0.0; 5],
        );
    }
}
