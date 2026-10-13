//! Multi-head attention over a fused `[q | k | v]` buffer, plus ModernBERT's RoPE.
//! The encoder applies `Rope` first; the head's attention uses no RoPE.

use std::simd::{
    Select, StdFloat, f32x8, u32x8,
    cmp::SimdPartialOrd,
    num::SimdFloat,
};

use crate::config::HEAD_DIM as HD;

const HALF: usize = HD / 2;
const QB: usize = 4; // queries per tile
const KB: usize = 16; // keys per tile (2 x f32x8)

/// RoPE cos/sin for positions `0..n`, `[n][HD / 2]`. Rotate-half convention (dim `j` pairs with
/// `j + HD/2`); angles in f32 like HF's `ModernBertRotaryEmbedding`.
pub struct Rope {
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    pub fn new(theta: f32, n: usize) -> Self {
        let inv: Vec<f32> = (0..HALF)
            .map(|j| 1.0 / theta.powf((2 * j) as f32 / HD as f32))
            .collect();
        let (mut cos, mut sin) = (Vec::with_capacity(n * HALF), Vec::with_capacity(n * HALF));
        for p in 0..n {
            for &f in &inv {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Self { cos, sin }
    }

    /// Rotates every q and k head of `qkv` (`[n, 3d]`, row `p` at position `p`). V is untouched.
    pub fn apply(&self, qkv: &mut [f32], d: usize) {
        assert!(qkv.len() / (3 * d) * HALF <= self.cos.len(), "rope table too short");
        let (cos, []) = self.cos.as_chunks::<HALF>() else {
            unreachable!()
        };
        let (sin, []) = self.sin.as_chunks::<HALF>() else {
            unreachable!()
        };
        for ((row, c), s) in qkv.chunks_exact_mut(3 * d).zip(cos).zip(sin) {
            let (c, s) = (c.as_chunks::<8>().0, s.as_chunks::<8>().0);
            let (heads, []) = row[..2 * d].as_chunks_mut::<HD>() else {
                unreachable!()
            };
            for head in heads {
                let (x1, x2) = head.split_at_mut(HALF);
                let (x1, x2) = (x1.as_chunks_mut::<8>().0, x2.as_chunks_mut::<8>().0);
                for (((a, b), &c), &s) in x1.iter_mut().zip(x2).zip(c).zip(s) {
                    let (va, vb) = (f32x8::from_array(*a), f32x8::from_array(*b));
                    let (c, s) = (f32x8::from_array(c), f32x8::from_array(s));
                    *a = (va * c - vb * s).to_array(); // x1·c − x2·s
                    *b = vb.mul_add(c, va * s).to_array(); // x2·c + x1·s
                }
            }
        }
    }
}

#[derive(Default)]
pub struct Scratch {
    kt: Vec<f32>,     // K^T for the current head: HD rows, stride n + KB
    scores: Vec<f32>, // QB score rows
}

/// 2^x for x <= 0. ~2e-7 relative error.
#[inline(always)]
fn exp2(x: f32x8) -> f32x8 {
    let lo = f32x8::splat(-126.0);
    let x = x.simd_lt(lo).select(lo, x);
    // Adding 1.5·2^23 rounds x to the nearest integer n, which lands in the low mantissa bits
    // of t. This is ~5x faster than floor() + a float->int cast.
    const MAGIC: f32 = 12582912.0;
    let t = x + f32x8::splat(MAGIC);
    let n = t - f32x8::splat(MAGIC);
    let f = x - n; // [-0.5, 0.5]
    // Taylor series of 2^f; truncation error ~1.2e-7 on this range.
    let mut p = f32x8::splat(1.540353e-4);
    p = p.mul_add(f, f32x8::splat(1.333356e-3));
    p = p.mul_add(f, f32x8::splat(9.618129e-3));
    p = p.mul_add(f, f32x8::splat(5.550411e-2));
    p = p.mul_add(f, f32x8::splat(2.402265e-1));
    p = p.mul_add(f, f32x8::splat(6.931472e-1));
    p = p.mul_add(f, f32x8::splat(1.0));
    // 2^n: exponent field n + 127 (wrapping u32 math handles negative n).
    let bits = (t.to_bits() - u32x8::splat(0x4B40_0000 - 127)) << u32x8::splat(23);
    p * f32x8::from_bits(bits)
}

/// In-place p = 2^(p - max(p)); returns the sum.
fn softmax_exp2(p: &mut [f32]) -> f32 {
    let (chunks, tail) = p.as_chunks_mut::<8>();
    let mut m = f32x8::splat(f32::NEG_INFINITY);
    for c in chunks.iter() {
        // plain compare+select: simd_max's NaN handling made the softmax ~30% slower
        let c = f32x8::from_array(*c);
        m = c.simd_gt(m).select(c, m);
    }
    let max = tail.iter().fold(m.reduce_max(), |a, &b| a.max(b));

    let mx = f32x8::splat(max);
    let mut sum = f32x8::splat(0.0);
    for c in chunks.iter_mut() {
        let e = exp2(f32x8::from_array(*c) - mx);
        sum += e;
        *c = e.to_array();
    }
    let mut sum = sum.reduce_sum();
    for x in tail {
        *x = (*x - max).exp2();
        sum += *x;
    }
    sum
}

/// Scaled dot-product attention, no mask other than the optional window.
/// qkv: `[n, 3d]` (`[q | k | v]` per token, each head-major), RoPE already applied if needed.
/// window: keys `j` with `|i - j| <= w` (ModernBERT local layers: `local_attention / 2`).
/// o: `[n, d]`.
pub fn attention(
    qkv: &[f32],
    n: usize,
    d: usize,
    window: Option<usize>,
    o: &mut [f32],
    s: &mut Scratch,
) {
    assert_eq!(qkv.len(), n * 3 * d);
    assert_eq!(o.len(), n * d);
    assert_eq!(d % HD, 0);

    // Fold 1/sqrt(HD) and log2(e) into Q so softmax is a bare exp2.
    let qscale = std::f32::consts::LOG2_E / (HD as f32).sqrt();
    let ks = n + KB; // padded so a 16-key tile never runs past the end
    s.kt.clear();
    s.kt.resize(HD * ks, 0.0);

    let range = |i: usize| match window {
        Some(w) => (i.saturating_sub(w), (i + w + 1).min(n)),
        None => (0, n),
    };

    for h in 0..d / HD {
        let (qo, ko, vo) = (h * HD, d + h * HD, 2 * d + h * HD);
        let row = |j: usize, off: usize| &qkv[j * 3 * d + off..][..HD];

        // 1. Transpose this head's K so SIMD lanes index keys.
        for j in 0..n {
            for (t, &x) in row(j, ko).iter().enumerate() {
                s.kt[t * ks + j] = x;
            }
        }

        for i0 in (0..n).step_by(QB) {
            let nq = QB.min(n - i0);
            let lo = range(i0).0;
            let hi = range(i0 + nq - 1).1;
            let span = hi - lo;
            let ss = span.next_multiple_of(KB);
            s.scores.clear();
            s.scores.resize(QB * ss, 0.0);

            // Scaled queries; rows past n duplicate the last one (discarded).
            let mut q = [[0f32; HD]; QB];
            for (r, qr) in q.iter_mut().enumerate() {
                let src = row(i0 + r.min(nq - 1), qo);
                for (dst, &x) in qr.iter_mut().zip(src) {
                    *dst = x * qscale;
                }
            }

            // 2. S = Q K^T in 4x16 tiles: 8 independent FMA chains, no reductions.
            for jb in (0..span).step_by(KB) {
                let mut acc = [[f32x8::splat(0.0); 2]; QB];
                for t in 0..HD {
                    let k = &s.kt[t * ks + lo + jb..][..KB];
                    let k0 = f32x8::from_slice(&k[..8]);
                    let k1 = f32x8::from_slice(&k[8..]);
                    for r in 0..QB {
                        let qv = f32x8::splat(q[r][t]);
                        acc[r][0] = qv.mul_add(k0, acc[r][0]);
                        acc[r][1] = qv.mul_add(k1, acc[r][1]);
                    }
                }
                for (r, a) in acc.iter().enumerate() {
                    a[0].copy_to_slice(&mut s.scores[r * ss + jb..][..8]);
                    a[1].copy_to_slice(&mut s.scores[r * ss + jb + 8..][..8]);
                }
            }

            // 3. Softmax per query over its own window; zero the rest of the tile's span so
            //    all QB queries can share one pass over V.
            let mut inv = [0f32; QB];
            for (r, inv) in inv.iter_mut().enumerate().take(nq) {
                let (a, b) = range(i0 + r);
                let p = &mut s.scores[r * ss..][..span];
                *inv = 1.0 / softmax_exp2(&mut p[a - lo..b - lo]);
                p[..a - lo].fill(0.0);
                p[b - lo..].fill(0.0);
            }

            // 4. O = P V, 16 output lanes at a time: each V load feeds QB queries.
            for c in (0..HD).step_by(16) {
                let mut acc = [[f32x8::splat(0.0); 2]; QB];
                for j in 0..span {
                    let v = &qkv[(lo + j) * 3 * d + vo + c..][..16];
                    let (v0, v1) = (f32x8::from_slice(&v[..8]), f32x8::from_slice(&v[8..]));
                    for r in 0..QB {
                        let p = f32x8::splat(s.scores[r * ss + j]);
                        acc[r][0] = p.mul_add(v0, acc[r][0]);
                        acc[r][1] = p.mul_add(v1, acc[r][1]);
                    }
                }
                for r in 0..nq {
                    let out = &mut o[(i0 + r) * d + h * HD + c..][..16];
                    let iv = f32x8::splat(inv[r]);
                    (acc[r][0] * iv).copy_to_slice(&mut out[..8]);
                    (acc[r][1] * iv).copy_to_slice(&mut out[8..]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HD, Scratch, attention};

    /// Scalar f64 reference: softmax(q·k / sqrt(HD)) v per head, keys limited to the window.
    fn naive(qkv: &[f32], n: usize, d: usize, window: Option<usize>) -> Vec<f32> {
        let mut o = vec![0.0; n * d];
        let at = |j: usize, c: usize| qkv[j * 3 * d + c] as f64;
        for h in 0..d / HD {
            for i in 0..n {
                let keys = match window {
                    Some(w) => i.saturating_sub(w)..(i + w + 1).min(n),
                    None => 0..n,
                };
                let s: Vec<f64> = keys
                    .clone()
                    .map(|j| (0..HD).map(|t| at(i, h * HD + t) * at(j, d + h * HD + t)).sum::<f64>() / (HD as f64).sqrt())
                    .collect();
                let max = s.iter().copied().fold(f64::MIN, f64::max);
                let p: Vec<f64> = s.iter().map(|x| (x - max).exp()).collect();
                let sum: f64 = p.iter().sum();
                for t in 0..HD {
                    let v: f64 = keys.clone().zip(&p).map(|(j, p)| p * at(j, 2 * d + h * HD + t)).sum();
                    o[i * d + h * HD + t] = (v / sum) as f32;
                }
            }
        }
        o
    }

    #[test]
    fn matches_naive() {
        let mut s = 1u32;
        let mut rnd = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    4.0 * ((s >> 8) as f32 / (1 << 24) as f32 - 0.5)
                })
                .collect()
        };
        let d = 2 * HD;
        let mut scratch = Scratch::default();
        // n not a multiple of the 4-query / 16-key tiles; windows narrower than n.
        for (n, window) in [(1, None), (37, None), (37, Some(64)), (150, Some(64)), (150, Some(5))] {
            let qkv = rnd(n * 3 * d);
            let mut o = vec![0.0; n * d];
            attention(&qkv, n, d, window, &mut o, &mut scratch);
            let want = naive(&qkv, n, d, window);
            let err = o.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(err < 1e-5, "n={n} window={window:?}: max err {err}");
        }
    }
}
