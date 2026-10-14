//! y = x·Wᵀ + b with W packed once at load time for a 6×16 register-tile kernel.

use std::simd::{StdFloat, f32x8};

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

/// A weight matrix `[out, inp]` repacked as panels of 16 outputs, each `[inp][16]`, so the kernel
/// reads 16 outputs' weights for one input index as two contiguous vectors. The last panel is
/// zero-padded.
pub struct Packed {
    panels: Vec<[f32; NR]>,
    inp: usize,
    out: usize,
}

impl Packed {
    /// Packs `w` (`[out, inp]`, row-major as in PyTorch), in parallel over panels.
    pub fn new(w: &[f32], inp: usize) -> Self {
        let out = w.len() / inp;
        assert!(
            inp > 0 && w.len() == out * inp,
            "Packed: {} weights are not [_, {inp}]",
            w.len()
        );
        let mut panels = vec![[0f32; NR]; out.div_ceil(NR) * inp];
        panels
            .par_chunks_exact_mut(inp)
            .enumerate()
            .for_each(|(p, panel)| {
                for (c, row) in w[p * NR * inp..].chunks_exact(inp).take(NR).enumerate() {
                    for (dst, &v) in panel.iter_mut().zip(row) {
                        dst[c] = v;
                    }
                }
            });
        Self { panels, inp, out }
    }
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
    let nc = out.div_ceil(CB);
    let y = Out(y.as_mut_ptr());
    (0..n.div_ceil(TB) * nc).into_par_iter().for_each(|task| {
        let (t0, j0) = (task / nc * TB, task % nc * CB);
        let (t1, j1) = ((t0 + TB).min(n), (j0 + CB).min(out));
        // Panel by panel, so each panel stays in L1 for all the block's tokens.
        for j in (j0..j1).step_by(NR) {
            let panel = &w.panels[j / NR * inp..][..inp];
            let mut t = t0;
            while t + MR <= t1 {
                tile::<MR>(x, panel, out, b, y, t, j);
                t += MR;
            }
            for t in t..t1 {
                tile::<1>(x, panel, out, b, y, t, j);
            }
        }
    });
}

/// Outputs `j..j + 16` (fewer in a padded last panel) for tokens `t..t + M`.
#[inline(always)]
fn tile<const M: usize>(
    x: &[f32],
    panel: &[[f32; NR]],
    out: usize,
    b: Option<&[f32]>,
    y: Out,
    t: usize,
    j: usize,
) {
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
    let acc = sum;
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
    use super::{Packed, linear};

    fn naive(x: &[f32], w: &[f32], b: Option<&[f32]>, inp: usize) -> Vec<f32> {
        let (n, out) = (x.len() / inp, w.len() / inp);
        let mut y = vec![0.0; n * out];
        for i in 0..n {
            for o in 0..out {
                let dot: f64 = (0..inp)
                    .map(|k| x[i * inp + k] as f64 * w[o * inp + k] as f64)
                    .sum();
                y[i * out + o] = dot as f32 + b.map_or(0.0, |b| b[o]);
            }
        }
        y
    }

    #[test]
    fn matches_naive() {
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
        // padded 16-output panels and partial task blocks; an odd inp.
        for (n, inp, out) in [
            (37, 384, 1152),
            (40, 1152, 384),
            (3, 384, 1),
            (1, 16, 3),
            (4, 8, 2),
            (150, 384, 2304),
            (130, 64, 200),
            (7, 13, 21),
        ] {
            let (x, w, b) = (rnd(n * inp), rnd(out * inp), rnd(out));
            let packed = Packed::new(&w, inp);
            // Task blocks are fixed, but run the tests on several pool sizes anyway.
            for (threads, b) in [1, 3, 22]
                .into_iter()
                .flat_map(|t| [(t, None), (t, Some(&b[..]))])
            {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut y = vec![0.0; n * out];
                pool.install(|| linear(&x, &packed, b, &mut y));
                let want = naive(&x, &w, b, inp);
                let err = y
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(
                    err < 1e-5,
                    "n={n} {inp}->{out} bias={} threads={threads}: max err {err}",
                    b.is_some()
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "bad shapes")]
    fn short_output_panics() {
        linear(&[0.0; 16], &Packed::new(&[0.0; 24], 8), None, &mut [0.0; 5]);
    }
}
