use std::simd::{StdFloat, f32x8, num::SimdFloat};

use rayon::prelude::*;

// For R*C FMA, we need R * C accumulator registers, C weight vectors while we
// go through the loop, and 1 token vector, loaded one at a time. So the number
// of registers needed is R*C + C + 1.
// avx2 has 16 ymm registers, so R*C + C + 1 <= 16.
//
// R = 4, C = 3 (i.e. 4x3) utilises this constraint effectively.
const R: usize = 4;

/// Tokens × outputs per parallel task. A task's block of W (CB rows) stays in its core's L2 while
/// its TB tokens stream past; small inputs still give enough tasks for every core. CB is a
/// multiple of 3 so blocks split evenly into C = 3 tiles.
const TB: usize = 64;
const CB: usize = 96;

/// `y`'s base pointer, shared by the tasks of one `linear` call. Each task writes only its own
/// (token block × output block) and those blocks partition `y`, so no element is written twice
/// or read, and the `&mut y` borrow outlives the parallel loop.
#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    /// SAFETY: `i` is in bounds and in the calling task's block.
    unsafe fn write(self, i: usize, v: f32) {
        unsafe { self.0.add(i).write(v) }
    }
}

// y[n*out] = x[n*inp] * w^T[inp*out] + b, in parallel over blocks of tokens × outputs.
pub fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, inp: usize, y: &mut [f32]) {
    let (n, out) = (x.len() / inp, w.len() / inp);
    // A mismatched y would otherwise silently skip rows (chunks_exact/zip stop early).
    assert!(
        inp.is_multiple_of(8)
            && x.len() == n * inp
            && w.len() == out * inp
            && y.len() == n * out
            && b.is_none_or(|b| b.len() == out),
        "linear: bad shapes (x {}, w {}, y {}, inp {inp})",
        x.len(),
        w.len(),
        y.len()
    );

    let (x8, []) = x.as_chunks::<8>() else {
        unreachable!()
    };
    let (w8, []) = w.as_chunks::<8>() else {
        unreachable!()
    };
    let k = inp / 8;
    let nc = out.div_ceil(CB);
    let y = Out(y.as_mut_ptr());
    (0..n.div_ceil(TB) * nc).into_par_iter().for_each(|task| {
        let (t0, j0) = (task / nc * TB, task % nc * CB);
        let (t1, j1) = ((t0 + TB).min(n), (j0 + CB).min(out));
        let (xs, ws) = (&x8[t0 * k..t1 * k], &w8[j0 * k..j1 * k]);
        if out.is_multiple_of(3) {
            block::<3>(xs, ws, b, k, out, y, t0, j0);
        } else {
            block::<1>(xs, ws, b, k, out, y, t0, j0);
        }
    });
}

/// Tokens `t0..` (`x`) × outputs `j0..` (`w`), as R-token tiles plus single-token leftovers.
#[allow(clippy::too_many_arguments)]
fn block<const C: usize>(
    x: &[[f32; 8]],
    w: &[[f32; 8]],
    b: Option<&[f32]>,
    k: usize,
    out: usize,
    y: Out,
    t0: usize,
    j0: usize,
) {
    // round the block's token count down to the nearest R multiple.
    let m = x.len() / k / R * R;
    let (xm, xt) = x.split_at(m * k);
    rows::<R, C>(xm, w, b, k, out, y, t0, j0);
    rows::<1, C>(xt, w, b, k, out, y, t0 + m, j0);
}

fn tile<const R: usize, const C: usize>(x: [&[[f32; 8]]; R], w: [&[[f32; 8]]; C]) -> [[f32; C]; R] {
    let mut acc = [[f32x8::splat(0.0); C]; R];
    let l = x[0].len();
    for i in 0..l {
        let wv: [f32x8; C] = std::array::from_fn(|c| f32x8::from_array(w[c][i]));
        for r in 0..R {
            let xv = f32x8::from_array(x[r][i]);
            for c in 0..C {
                acc[r][c] = xv.mul_add(wv[c], acc[r][c]);
            }
        }
    }
    acc.map(|r| r.map(|i| i.reduce_sum()))
}

#[allow(clippy::too_many_arguments)]
fn rows<const R: usize, const C: usize>(
    x: &[[f32; 8]],
    w: &[[f32; 8]],
    b: Option<&[f32]>,
    k: usize,
    out: usize,
    y: Out,
    t0: usize,
    j0: usize,
) {
    for (g, xs) in x.chunks_exact(R * k).enumerate() {
        let xr = std::array::from_fn::<_, R, _>(|i| &xs[(i * k)..((i + 1) * k)]);
        for (i, ws) in w.chunks_exact(C * k).enumerate() {
            let wr = std::array::from_fn::<_, C, _>(|i| &ws[(i * k)..((i + 1) * k)]);
            let t = tile(xr, wr);
            for r in 0..R {
                for c in 0..C {
                    let j = j0 + i * C + c;
                    // SAFETY: token t0 + g*R + r and output j are inside this task's block.
                    unsafe {
                        y.write(
                            (t0 + g * R + r) * out + j,
                            t[r][c] + b.map_or(0.0, |b| b[j]),
                        )
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::linear;

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
        // Covers the 4-row main part + tail, out % 3 == 0 and out == 1 (scorer), 1 token.
        for (n, inp, out) in [
            (37, 384, 1152),
            (40, 1152, 384),
            (3, 384, 1),
            (1, 16, 3),
            (4, 8, 2),
            // several token/output blocks, including partial ones (and the C = 1 path)
            (150, 384, 2304),
            (130, 64, 200),
        ] {
            let (x, w, b) = (rnd(n * inp), rnd(out * inp), rnd(out));
            for b in [None, Some(&b[..])] {
                let mut y = vec![0.0; n * out];
                linear(&x, &w, b, inp, &mut y);
                let want = naive(&x, &w, b, inp);
                let err = y
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(
                    err < 1e-5,
                    "n={n} {inp}->{out} bias={}: max err {err}",
                    b.is_some()
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "bad shapes")]
    fn short_output_panics() {
        linear(&[0.0; 16], &[0.0; 24], None, 8, &mut [0.0; 5]);
    }
}
