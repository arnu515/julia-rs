#![feature(portable_simd)]

use std::{
    fs::File,
    simd::{StdFloat, f32x8, num::SimdFloat},
};

use color_eyre::eyre::{bail, ensure};
use memmap2::Mmap;

pub mod config;
pub mod matmul;
pub mod safetensors;
pub mod tokenizer;

use config::ModelConfig;
use safetensors::Weights;
use tokenizer::Tokenizer;

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
}

fn layernorm(row: &mut [f32], norms: &[f32], biases: Option<&[f32]>, eps: f32) {
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
    pub safetensors_path: &'a str,
    /// ModernBERT `encoder/config.json`.
    pub encoder_config_json_path: &'a str,
    pub julia_config_json_path: &'a str,
    pub tokenizer_json_path: &'a str,
    pub tokenizer_config_json_path: &'a str,
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
