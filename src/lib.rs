#![feature(portable_simd)]
#![cfg_attr(test, feature(float_erf))]

use std::{
    fs::File,
    path::Path,
    simd::{StdFloat, f32x8, num::SimdFloat},
};

use color_eyre::eyre::{bail, ensure};
use memmap2::Mmap;

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
    matmul::linear,
    mlp::geglu,
};

pub struct Model {
    pub file: File,
    pub map: Mmap,
    /// Offset of the safetensors data section within `map`.
    pub data_start: usize,
    pub config: ModelConfig,
    pub weights: Weights,
    pub tokenizer: Tokenizer,
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
        let mut x = Vec::with_capacity(h * tokens.len());
        for (i, &t) in tokens.iter().enumerate() {
            let t = t as usize;
            x.extend_from_slice(&embeddings[(t * h)..((t + 1) * h)]);
            layernorm(
                &mut x[(i * h)..((i + 1) * h)],
                norms,
                None,
                self.config.norm_eps,
            );
        }
        x
    }

    pub fn encode(&self, tokens: &[u32], mut tap: impl FnMut(&str, &[f32])) -> Vec<f32> {
        let (n, d, inter, eps) = (
            tokens.len(),
            self.config.hidden,
            self.config.intermediate,
            self.config.norm_eps,
        );
        let mut x = self.embed(tokens);
        let (mut h, mut o) = (vec![0f32; n * d], vec![0f32; n * d]);
        let mut qkv = vec![0.0; n * 3 * d];
        let mut hh = vec![0.0; n * 2 * inter];
        let mut s = Scratch::default();
        let rope_g = Rope::new(self.config.rope_theta_global, n);
        let rope_l = Rope::new(self.config.rope_theta_local, n);

        let data = self.data();
        for (i, l) in self.weights.encoder.layers.iter().enumerate() {
            // attention
            norm_rows(
                &x,
                l.attn_norm.as_ref().map(|t| t.as_f32(data)),
                None,
                eps,
                &mut h,
            );
            linear(&h, l.wqkv.as_f32(data), None, d, &mut qkv);
            let (rope, window) = if l.global {
                (&rope_g, None)
            } else {
                (&rope_l, Some(self.config.local_window / 2))
            };
            rope.apply(&mut qkv, d);
            attention(&qkv, n, d, window, &mut o, &mut s);
            linear(&o, l.wo.as_f32(data), None, d, &mut h);
            f32_vec_add(&mut x, &h);

            // MLP
            norm_rows(&x, Some(l.mlp_norm.as_f32(data)), None, eps, &mut h);
            linear(&h, l.wi.as_f32(data), None, d, &mut hh);
            let m = geglu(&mut hh, inter);
            linear(m, l.wo_mlp.as_f32(data), None, inter, &mut h);
            f32_vec_add(&mut x, &h);

            tap(&format!("layer{i:02}"), &x);
        }
        {
            let norms = self.weights.encoder.final_norm.as_f32(data);
            for row in x.chunks_exact_mut(d) {
                layernorm(row, norms, None, eps);
            }
            tap("final_norm", &x);
        }

        x
    }
}

pub(crate) fn f32_vec_add(a: &mut [f32], b: &[f32]) {
    assert!(a.len() == b.len());
    for (a, &b) in a.iter_mut().zip(b) {
        *a += b;
    }
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
        assert!(out.len().is_multiple_of(norms.len()));
        for chunk in out.chunks_exact_mut(norms.len()) {
            layernorm(chunk, norms, biases, eps);
        }
    }
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

pub fn load_model(paths: ModelPaths<'_>) -> color_eyre::Result<Model> {
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
        tokenizer,
    })
}
