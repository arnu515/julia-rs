use std::fs::File;

use color_eyre::eyre::{bail, ensure};
use memmap2::Mmap;

pub mod config;
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
