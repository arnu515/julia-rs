//! Julia's decision head (`julia/model.py`): type embedding, `nn.TransformerEncoderLayer`s
//! (pre-norm, biases, ReLU, global attention without RoPE) and the per-option scorer.

use rayon::prelude::*;
use serde::Deserialize;

use crate::{
    HeadLinears, Model,
    attn::{Scratch, attention_q},
    f32_vec_add, layernorm,
    matmul::linear,
    mlp::gelu_inplace,
    norm_rows,
    safetensors::HeadLayer,
};

/// `nn.LayerNorm`'s default eps. Julia's head and scorer don't set it, so it isn't in any config.
const TORCH_LN_EPS: f32 = 1e-5;

/// Question types, as numbered by `QTYPES` in Julia's `data.py`. A request's `type` defaults to
/// `choice`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QType {
    #[default]
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    /// The name used in requests and in the sequence's `"{type} question: ..."` prefix.
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
}

/// What a question's logits mean (`julia/typed.py`).
#[derive(Debug, PartialEq)]
pub enum Answer {
    /// Index of the most probable option.
    Choice(usize),
    /// Expected rubric level, sum of `i * p_i`.
    Score(f32),
    /// Probability of `true` (options are `[false, true]`).
    Noul(f32),
}

impl Model {
    /// Encoder then head: one logit per marker position. `tap` sees every stage (see `encode`).
    pub fn forward(
        &self,
        ids: &[u32],
        qtype: QType,
        markers: &[usize],
        mut tap: impl FnMut(&str, &[f32]),
    ) -> Vec<f32> {
        let x = self.encode(ids, &mut tap);
        self.head(x, qtype, markers, tap)
    }

    /// The head on the encoder output `x` (`[n, d]`): one logit per marker position. Only marker
    /// rows are read at the end, so the last layer computes just those (Julia's exact
    /// `_selected_head`); its tap gets the `[markers.len(), d]` marker rows.
    pub fn head(
        &self,
        mut x: Vec<f32>,
        qtype: QType,
        markers: &[usize],
        mut tap: impl FnMut(&str, &[f32]),
    ) -> Vec<f32> {
        let (data, d, ff) = (self.data(), self.config.hidden, self.config.head_ff);
        let (n, k) = (x.len() / d, markers.len());
        assert!(
            markers.iter().all(|&m| m < n),
            "head: marker past {n} tokens: {markers:?}"
        );

        let te = &self.weights.type_emb.as_f32(data)[qtype as usize * d..][..d];
        x.par_chunks_exact_mut(d)
            .with_min_len(16)
            .for_each(|row| f32_vec_add(row, te));
        let Some((last, dense)) = self.weights.head.split_last() else {
            return self.score(gather(&x, markers, d));
        };

        let (mut h, mut o) = (vec![0f32; n * d], vec![0f32; n * d]);
        let mut kv = vec![0f32; n * 2 * d];
        let mut f = vec![0f32; n * ff];
        let mut s = Scratch::default();
        let packed = &self.packed.head;
        for (i, (l, p)) in dense.iter().zip(packed).enumerate() {
            norm_rows(
                &x,
                Some(l.norm1_w.as_f32(data)),
                Some(l.norm1_b.as_f32(data)),
                TORCH_LN_EPS,
                &mut h,
            );
            let (bq, bkv) = l.in_proj_b.as_f32(data).split_at(d);
            linear(&h, &p.q, Some(bq), &mut o);
            linear(&h, &p.kv, Some(bkv), &mut kv);
            attention_q(&o, d, &kv, 2 * d, n, d, None, &mut h, &mut s);
            linear(&h, &p.out_proj, Some(l.out_proj_b.as_f32(data)), &mut o);
            f32_vec_add(&mut x, &o);
            self.head_mlp(l, p, &mut x, &mut h, &mut f);
            tap(&format!("head{i}"), &x);
        }

        // Last layer: K and V from every row, everything else for the marker rows only.
        let (l, p) = (last, &packed[dense.len()]);
        norm_rows(
            &x,
            Some(l.norm1_w.as_f32(data)),
            Some(l.norm1_b.as_f32(data)),
            TORCH_LN_EPS,
            &mut h,
        );
        let (bq, bkv) = l.in_proj_b.as_f32(data).split_at(d);
        linear(&h, &p.kv, Some(bkv), &mut kv);
        let q = &mut o[..k * d];
        linear(&gather(&h, markers, d), &p.q, Some(bq), q);
        let a = &mut h[..k * d];
        attention_q(q, d, &kv, 2 * d, n, d, None, a, &mut s);
        let mut xs = gather(&x, markers, d);
        let po = &mut o[..k * d];
        linear(a, &p.out_proj, Some(l.out_proj_b.as_f32(data)), po);
        f32_vec_add(&mut xs, po);
        self.head_mlp(l, p, &mut xs, &mut h[..k * d], &mut f[..k * ff]);
        tap(&format!("head{}", dense.len()), &xs);

        self.score(xs)
    }

    /// x += linear2(ReLU(linear1(norm2(x)))). `h` is `[rows, d]` and `f` `[rows, head_ff]` scratch.
    fn head_mlp(
        &self,
        l: &HeadLayer,
        p: &HeadLinears,
        x: &mut [f32],
        h: &mut [f32],
        f: &mut [f32],
    ) {
        let data = self.data();
        norm_rows(
            x,
            Some(l.norm2_w.as_f32(data)),
            Some(l.norm2_b.as_f32(data)),
            TORCH_LN_EPS,
            h,
        );
        linear(h, &p.linear1, Some(l.linear1_b.as_f32(data)), f);
        f.par_chunks_mut(4096).for_each(|f| {
            for v in f {
                // not f32::max, whose NaN handling stops it compiling to a plain vmaxps
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        });
        linear(f, &p.linear2, Some(l.linear2_b.as_f32(data)), h);
        f32_vec_add(x, h);
    }

    /// The scorer on marker rows `z` (`[k, d]`): LayerNorm, Linear, GELU, Linear -> `k` logits.
    fn score(&self, mut z: Vec<f32>) -> Vec<f32> {
        let (data, d, sc) = (self.data(), self.config.hidden, &self.weights.scorer);
        let (nw, nb) = (sc.norm_w.as_f32(data), sc.norm_b.as_f32(data));
        for row in z.chunks_exact_mut(d) {
            layernorm(row, nw, Some(nb), TORCH_LN_EPS);
        }
        let [lin1, lin2] = &self.packed.scorer;
        let mut y = vec![0f32; z.len()];
        linear(&z, lin1, Some(sc.lin1_b.as_f32(data)), &mut y);
        gelu_inplace(&mut y);
        let mut logits = vec![0f32; z.len() / d];
        linear(&y, lin2, Some(sc.lin2_b.as_f32(data)), &mut logits);
        logits
    }
}

/// Rows `rows` of `x` (`[_, d]`), packed.
fn gather(x: &[f32], rows: &[usize], d: usize) -> Vec<f32> {
    rows.iter()
        .flat_map(|&r| &x[r * d..][..d])
        .copied()
        .collect()
}

/// Softmax probabilities of `logits` and what they mean for `qtype` (`julia/typed.py`). The
/// probabilities are exact; Julia's `display_probabilities` rounding is cosmetic and not applied.
pub fn answer(qtype: QType, logits: &[f32]) -> (Answer, Vec<f32>) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits.iter().map(|&z| (z - max).exp()).collect();
    let sum: f32 = p.iter().sum();
    p.iter_mut().for_each(|v| *v /= sum);
    let a = match qtype {
        QType::Choice => {
            Answer::Choice((0..p.len()).fold(0, |b, i| if p[i] > p[b] { i } else { b }))
        }
        QType::Score => Answer::Score(p.iter().enumerate().map(|(i, v)| i as f32 * v).sum()),
        QType::Noul => Answer::Noul(p[1]),
    };
    (a, p)
}

#[cfg(test)]
mod tests {
    use super::{Answer, QType, answer};

    #[test]
    fn answers() {
        let (a, p) = answer(QType::Choice, &[1.0, 3.0, 2.0]);
        assert_eq!(a, Answer::Choice(1));
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6 && p[1] > p[2] && p[2] > p[0]);
        let Answer::Score(s) = answer(QType::Score, &[0.0, 0.0, 0.0]).0 else {
            panic!()
        };
        assert!((s - 1.0).abs() < 1e-6);
        let Answer::Noul(t) = answer(QType::Noul, &[0.0, 2.0f32.ln()]).0 else {
            panic!()
        };
        assert!((t - 2.0 / 3.0).abs() < 1e-6);
    }
}
