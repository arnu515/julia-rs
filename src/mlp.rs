// Copyright (C) 2026 arnu515
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! ModernBERT's GeGLU MLP activation, and exact (erf) GELU.

use std::f32::consts::LOG2_E;
use std::simd::{Select, StdFloat, cmp::SimdPartialOrd, f32x8, num::SimdFloat};

use rayon::prelude::*;

use crate::avx2::exp2;

/// Exact GELU, x·Φ(x), with Φ from Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7·|x|).
// The A&S coefficients are kept as published.
#[allow(clippy::excessive_precision)]
#[inline(always)]
pub fn gelu(x: f32x8) -> f32x8 {
    // A&S: erf(z) = 1 - poly(t)·e^(-z²), t = 1 / (1 + p·z), z >= 0. With z = |x|/√2,
    // Φ(x) = 1 - q for x >= 0 and q for x < 0, where q = ½·poly(t)·e^(-x²/2).
    // p/√2 = 0.2316419, and the ½ is folded into the coefficients a1..a5.
    let t = f32x8::splat(1.0) / x.abs().mul_add(f32x8::splat(0.2316419), f32x8::splat(1.0));
    let mut p = f32x8::splat(0.5 * 1.061405429);
    p = p.mul_add(t, f32x8::splat(0.5 * -1.453152027));
    p = p.mul_add(t, f32x8::splat(0.5 * 1.421413741));
    p = p.mul_add(t, f32x8::splat(0.5 * -0.284496736));
    p = p.mul_add(t, f32x8::splat(0.5 * 0.254829592));
    let q = p * t * exp2(-(x * x) * f32x8::splat(0.5 * LOG2_E));
    x * x
        .simd_ge(f32x8::splat(0.0))
        .select(f32x8::splat(1.0) - q, q)
}

/// In-place exact GELU (the scorer's `nn.GELU`).
pub fn gelu_inplace(x: &mut [f32]) {
    let (x8, []) = x.as_chunks_mut::<8>() else {
        panic!("gelu_inplace: len {} not a multiple of 8", x.len())
    };
    for c in x8 {
        *c = gelu(f32x8::from_array(*c)).to_array();
    }
}

/// GeGLU on `hh` = x·Wiᵀ, rows of `[inp | gate]` (each `inter` wide): `m` = GELU(inp) ⊙ gate,
/// `[n, inter]`, in parallel over rows.
pub fn geglu(hh: &[f32], inter: usize, m: &mut [f32]) {
    assert!(
        inter.is_multiple_of(8) && hh.len() == 2 * m.len() && m.len().is_multiple_of(inter),
        "geglu: bad shapes (hh {}, m {}, inter {inter})",
        hh.len(),
        m.len()
    );
    let rows = hh
        .par_chunks_exact(2 * inter)
        .zip(m.par_chunks_exact_mut(inter));
    rows.with_min_len(8).for_each(|(h, m)| {
        let (inp, gate) = h.split_at(inter);
        let (m, []) = m.as_chunks_mut::<8>() else {
            unreachable!()
        };
        for ((m, a), g) in m
            .iter_mut()
            .zip(inp.as_chunks::<8>().0)
            .zip(gate.as_chunks::<8>().0)
        {
            *m = (gelu(f32x8::from_array(*a)) * f32x8::from_array(*g)).to_array();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gelu_ref(x: f32) -> f32 {
        let x = x as f64;
        (0.5 * x * (1.0 + (x / std::f64::consts::SQRT_2).erf())) as f32
    }

    #[test]
    fn gelu_matches_erf() {
        let xs: Vec<f32> = (-160_000..160_000).map(|i| i as f32 / 8000.0).collect(); // [-20, 20]
        let mut ys = xs.clone();
        gelu_inplace(&mut ys);
        for (&x, &y) in xs.iter().zip(&ys) {
            let err = (y - gelu_ref(x)).abs();
            assert!(
                err <= 2e-7 * x.abs().max(1.0),
                "gelu({x}) = {y}, want {}",
                gelu_ref(x)
            );
        }
    }

    #[test]
    fn geglu_matches_naive() {
        let (n, inter) = (5, 24);
        let hh: Vec<f32> = (0..n * 2 * inter)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 12.0)
            .collect();
        let want: Vec<f32> = (0..n * inter)
            .map(|i| {
                let (r, j) = (i / inter, i % inter);
                gelu_ref(hh[r * 2 * inter + j]) * hh[r * 2 * inter + inter + j]
            })
            .collect();
        let mut got = vec![0.0; n * inter];
        geglu(&hh, inter, &mut got);
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 1e-5, "max err {err}");
    }
}
