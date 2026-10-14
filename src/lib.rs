#![feature(portable_simd)]
#![cfg_attr(test, feature(float_erf))]

use std::{
    fs::File,
    ops::Range,
    path::Path,
    simd::{StdFloat, f32x8, num::SimdFloat},
};

use color_eyre::eyre::{bail, ensure};
use memmap2::{Mmap, UncheckedAdvice};
use rayon::prelude::*;

pub mod attn;
pub mod avx2;
pub mod config;
pub mod head;
pub mod matmul;
pub mod mlp;
pub mod safetensors;
pub mod sequence;
pub mod tokenizer;

use config::ModelConfig;
use safetensors::Weights;
use tokenizer::Tokenizer;

use crate::{
    attn::{Rope, Scratch, attention},
    matmul::{Packed, Precision, linear, linear_add},
    mlp::geglu,
};

pub struct Model {
    pub file: File,
    pub map: Mmap,
    /// Offset of the safetensors data section within `map`.
    pub data_start: usize,
    pub config: ModelConfig,
    pub weights: Weights,
    /// Every linear weight, packed for `matmul::linear` at load time.
    pub packed: PackedWeights,
    pub tokenizer: Tokenizer,
}

pub struct EncoderLinears {
    pub wqkv: Packed,
    pub wo: Packed,
    pub wi: Packed,
    pub wo_mlp: Packed,
}

/// `in_proj` is split into its q rows and k|v rows, so the head can compute queries for a
/// subset of rows (`attention_q`).
pub struct HeadLinears {
    pub q: Packed,
    pub kv: Packed,
    pub out_proj: Packed,
    pub linear1: Packed,
    pub linear2: Packed,
}

pub struct PackedWeights {
    pub encoder: Vec<EncoderLinears>,
    pub head: Vec<HeadLinears>,
    pub scorer: [Packed; 2],
}

impl PackedWeights {
    /// Packs every linear weight of `w` (in `map`'s data section, from `data_start`), then drops
    /// the original's pages from memory.
    fn new(
        w: &Weights,
        map: &Mmap,
        data_start: usize,
        cfg: &ModelConfig,
        precision: Precision,
    ) -> Self {
        let (d, inter, ff) = (cfg.hidden, cfg.intermediate, cfg.head_ff);
        let data = &map[data_start..];
        let release =
            |t: &safetensors::Tensor| release(map, data_start + t.start..data_start + t.end);
        let p = |t: &safetensors::Tensor, inp: usize| {
            let packed = Packed::new(t.as_f32(data), inp, precision);
            release(t);
            packed
        };
        Self {
            encoder: (w.encoder.layers.iter())
                .map(|l| EncoderLinears {
                    wqkv: p(&l.wqkv, d),
                    wo: p(&l.wo, d),
                    wi: p(&l.wi, d),
                    wo_mlp: p(&l.wo_mlp, inter),
                })
                .collect(),
            head: (w.head.iter())
                .map(|l| {
                    let (q, kv) = l.in_proj_w.as_f32(data).split_at(d * d);
                    let (q, kv) = (Packed::new(q, d, precision), Packed::new(kv, d, precision));
                    release(&l.in_proj_w);
                    HeadLinears {
                        q,
                        kv,
                        out_proj: p(&l.out_proj_w, d),
                        linear1: p(&l.linear1_w, d),
                        linear2: p(&l.linear2_w, ff),
                    }
                })
                .collect(),
            scorer: [p(&w.scorer.lin1_w, d), p(&w.scorer.lin2_w, d)],
        }
    }
}

/// Drops the resident pages wholly inside `range` of `map`, a weight already packed. The map is a
/// read-only file mapping, so anything touching them again rereads the file. (Peak memory after
/// loading: 483 -> 305 MB in f32, 350 -> 171 MB in i8.)
fn release(map: &Mmap, range: Range<usize>) {
    const PAGE: usize = 4096;
    let (a, b) = (range.start.next_multiple_of(PAGE), range.end / PAGE * PAGE);
    if a < b {
        // SAFETY: a shared read-only file mapping reads back unchanged after MADV_DONTNEED. Best
        // effort: errors (larger pages, say) just leave the pages resident.
        let _ = unsafe { map.unchecked_advise_range(UncheckedAdvice::DontNeed, a, b - a) };
    }
}

impl Model {
    /// The data section; `Tensor` offsets are relative to this.
    pub fn data(&self) -> &[u8] {
        &self.map[self.data_start..]
    }

    pub fn embed(&self, tokens: &[u32]) -> Vec<f32> {
        let embeddings = self.weights.encoder.tok_embeddings.as_f32(self.data());
        let norms = self.weights.encoder.emb_norm.as_f32(self.data());
        let h = self.config.hidden;
        let mut x = vec![0f32; h * tokens.len()];
        x.par_chunks_exact_mut(h)
            .zip(tokens)
            .with_min_len(16)
            .for_each(|(row, &t)| {
                let t = t as usize;
                row.copy_from_slice(&embeddings[(t * h)..((t + 1) * h)]);
                layernorm(row, norms, None, self.config.norm_eps);
            });
        x
    }

    pub fn encode(&self, tokens: &[u32]) -> Vec<f32> {
        let (n, d, inter, eps) = (
            tokens.len(),
            self.config.hidden,
            self.config.intermediate,
            self.config.norm_eps,
        );
        let mut x = self.embed(tokens);
        let (mut h, mut o) = (vec![0f32; n * d], vec![0f32; n * d]);
        let mut qkv = vec![0.0; n * 3 * d];
        let (mut hh, mut m) = (vec![0.0; n * 2 * inter], vec![0.0; n * inter]);
        let mut s = Scratch::default();
        let rope_g = Rope::new(self.config.rope_theta_global, n);
        let rope_l = Rope::new(self.config.rope_theta_local, n);

        let data = self.data();
        for (l, p) in self.weights.encoder.layers.iter().zip(&self.packed.encoder) {
            // attention
            norm_rows(
                &x,
                l.attn_norm.as_ref().map(|t| t.as_f32(data)),
                None,
                eps,
                &mut h,
            );
            linear(&h, &p.wqkv, None, &mut qkv);
            let (rope, window) = if l.global {
                (&rope_g, None)
            } else {
                (&rope_l, Some(self.config.local_window / 2))
            };
            rope.apply(&mut qkv, d);
            attention(&qkv, n, d, window, &mut o, &mut s);
            linear_add(&o, &p.wo, None, &mut x);

            // MLP
            norm_rows(&x, Some(l.mlp_norm.as_f32(data)), None, eps, &mut h);
            linear(&h, &p.wi, None, &mut hh);
            geglu(&hh, inter, &mut m);
            linear_add(&m, &p.wo_mlp, None, &mut x);
        }
        {
            let norms = self.weights.encoder.final_norm.as_f32(data);
            norm_rows_inplace(&mut x, norms, None, eps);
        }

        x
    }
}

/// a += b, in parallel over 4096-element chunks.
pub(crate) fn f32_vec_add(a: &mut [f32], b: &[f32]) {
    assert!(a.len() == b.len());
    a.par_chunks_mut(4096)
        .zip(b.par_chunks(4096))
        .for_each(|(a, b)| {
            for (a, &b) in a.iter_mut().zip(b) {
                *a += b;
            }
        });
}

pub(crate) fn norm_rows(
    x: &[f32],
    norms: Option<&[f32]>,
    biases: Option<&[f32]>,
    eps: f32,
    out: &mut [f32],
) {
    out.copy_from_slice(x);
    if let Some(norms) = norms {
        norm_rows_inplace(out, norms, biases, eps);
    }
}

/// `layernorm` on every row of `x` (rows are `norms.len()` wide), in parallel.
pub(crate) fn norm_rows_inplace(x: &mut [f32], norms: &[f32], biases: Option<&[f32]>, eps: f32) {
    assert!(x.len().is_multiple_of(norms.len()));
    x.par_chunks_exact_mut(norms.len())
        .with_min_len(16)
        .for_each(|row| layernorm(row, norms, biases, eps));
}

pub(crate) fn layernorm(row: &mut [f32], norms: &[f32], biases: Option<&[f32]>, eps: f32) {
    assert!(row.len().is_multiple_of(8) && norms.len() == row.len());
    let n = row.len() as f32;
    let (chunks, []) = row.as_chunks::<8>() else {
        unreachable!()
    };
    let mean = chunks
        .iter()
        .map(|&c| f32x8::from_array(c))
        .sum::<f32x8>()
        .reduce_sum()
        / n;
    let mean = f32x8::splat(mean);
    let factor = {
        let mut acc = f32x8::splat(0.0);
        for &c in chunks {
            let d = f32x8::from_array(c) - mean;
            acc = d.mul_add(d, acc);
        }
        f32x8::splat(1.0 / (acc.reduce_sum() / n + eps).sqrt())
    };

    let (row, []) = row.as_chunks_mut::<8>() else {
        unreachable!()
    };
    let (norms, []) = norms.as_chunks::<8>() else {
        unreachable!()
    };
    if let Some(biases) = biases {
        let (biases, []) = biases.as_chunks::<8>() else {
            unreachable!()
        };
        for ((row, &norm), &bias) in row.iter_mut().zip(norms).zip(biases) {
            *row = ((f32x8::from_array(*row) - mean) * factor)
                .mul_add(f32x8::from_array(norm), f32x8::from_array(bias))
                .to_array();
        }
    } else {
        for (row, &norm) in row.iter_mut().zip(norms) {
            *row = ((f32x8::from_array(*row) - mean) * factor * f32x8::from_array(norm)).to_array();
        }
    }
}

pub struct ModelPaths<'a> {
    pub safetensors_path: &'a Path,
    /// ModernBERT `encoder/config.json`.
    pub encoder_config_json_path: &'a Path,
    pub julia_config_json_path: &'a Path,
    pub tokenizer_json_path: &'a Path,
    pub tokenizer_config_json_path: &'a Path,
}

/// Loads a checkpoint, packing its linear weights in `precision`.
pub fn load_model(paths: ModelPaths<'_>, precision: Precision) -> color_eyre::Result<Model> {
    let config = ModelConfig::load(paths.encoder_config_json_path, paths.julia_config_json_path)?;

    let file = File::open(paths.safetensors_path)?;
    let map = unsafe { Mmap::map(&file)? };

    let Some(len_bytes) = map.get(0..8) else {
        bail!("model.safetensors is too short");
    };
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    let data_start = 8usize.saturating_add(len);
    let Some(header) = map.get(8..data_start) else {
        bail!("model.safetensors header length {len} exceeds file size");
    };
    let weights = safetensors::parse_header(header, map.len() - data_start, &config)?;
    let packed = PackedWeights::new(&weights, &map, data_start, &config, precision);
    let tokenizer = Tokenizer::load(paths.tokenizer_json_path, paths.tokenizer_config_json_path)?;
    ensure!(
        tokenizer.vocab_size == config.vocab,
        "tokenizer has {} ids but encoder config says vocab_size is {}",
        tokenizer.vocab_size,
        config.vocab
    );

    Ok(Model {
        file,
        map,
        data_start,
        config,
        weights,
        packed,
        tokenizer,
    })
}
