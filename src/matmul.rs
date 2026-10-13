use std::simd::{StdFloat, f32x8, num::SimdFloat};

// y[n*out] = x[n*inp] * w^T[inp*out] + b
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

    // For R*C FMA, we need R * C accumulator registers, C weight vectors while we
    // go through the loop, and 1 token vector, loaded one at a time. So the number
    // of registers needed is R*C + C + 1.
    // avx2 has 16 ymm registers, so R*C + C + 1 <= 16.
    //
    // R = 4, C = 3 (i.e. 4x3) utilises this constraint effectively.
    const R: usize = 4;
    // round down N to the nearest R multiple.
    let m = (n / R) * R;

    let (xm, xt) = x8.split_at(m * k);
    let (ym, yt) = y.split_at_mut(m * out);

    if out.is_multiple_of(3) {
        rows::<R, 3>(xm, w8, b, k, out, ym);
        rows::<1, 3>(xt, w8, b, k, out, yt);
    } else {
        rows::<R, 1>(xm, w8, b, k, out, ym);
        rows::<1, 1>(xt, w8, b, k, out, yt);
    }
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

fn rows<const R: usize, const C: usize>(
    x: &[[f32; 8]],
    w: &[[f32; 8]],
    b: Option<&[f32]>,
    k: usize,
    out: usize,
    y: &mut [f32],
) {
    for (xs, ys) in x.chunks_exact(R * k).zip(y.chunks_exact_mut(R * out)) {
        let xr = std::array::from_fn::<_, R, _>(|i| &xs[(i * k)..((i + 1) * k)]);
        for (i, ws) in w.chunks_exact(C * k).enumerate() {
            let wr = std::array::from_fn::<_, C, _>(|i| &ws[(i * k)..((i + 1) * k)]);
            let t = tile(xr, wr);
            for r in 0..R {
                for c in 0..C {
                    let j = i * C + c;
                    ys[r * out + j] = t[r][c] + b.map_or(0.0, |b| b[j]);
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
                let dot: f64 = (0..inp).map(|k| x[i * inp + k] as f64 * w[o * inp + k] as f64).sum();
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
        for (n, inp, out) in [(37, 384, 1152), (40, 1152, 384), (3, 384, 1), (1, 16, 3), (4, 8, 2)] {
            let (x, w, b) = (rnd(n * inp), rnd(out * inp), rnd(out));
            for b in [None, Some(&b[..])] {
                let mut y = vec![0.0; n * out];
                linear(&x, &w, b, inp, &mut y);
                let want = naive(&x, &w, b, inp);
                let err = y.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                assert!(err < 1e-5, "n={n} {inp}->{out} bias={}: max err {err}", b.is_some());
            }
        }
    }

    #[test]
    #[should_panic(expected = "bad shapes")]
    fn short_output_panics() {
        linear(&[0.0; 16], &[0.0; 24], None, 8, &mut [0.0; 5]);
    }
}
