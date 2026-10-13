use std::fs::File;

use memmap2::Mmap;

pub mod safetensors;
pub mod tokenizer;

use safetensors::Weights;
use tokenizer::Tokenizer;

pub struct Model {
    pub file: File,
    pub map: Mmap,
    /// Offset of the safetensors data section within `map`.
    pub data_start: usize,
    pub weights: Weights,
    pub tokenizer: Tokenizer,
}

impl Model {
    /// The data section; `Tensor` offsets are relative to this.
    pub fn data(&self) -> &[u8] {
        &self.map[self.data_start..]
    }
}

pub struct ModelConfig<'a> {
    pub safetensors_path: &'a str,
    pub tokenizer_json_path: &'a str,
    pub tokenizer_config_json_path: &'a str,
}

pub fn load_model(config: ModelConfig<'_>) -> color_eyre::Result<Model> {
    let file = File::open(config.safetensors_path)?;
    let map = unsafe { Mmap::map(&file)? };

    let Some(len_bytes) = map.get(0..8) else {
        color_eyre::eyre::bail!("model.safetensors is too short");
    };
    let len = u64::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    let data_start = 8usize.saturating_add(len);
    let Some(header) = map.get(8..data_start) else {
        color_eyre::eyre::bail!("model.safetensors header length {len} exceeds file size");
    };
    let weights = safetensors::parse_header(header, map.len() - data_start)?;
    let tokenizer = Tokenizer::load(
        config.tokenizer_json_path,
        config.tokenizer_config_json_path,
    )?;

    Ok(Model {
        file,
        map,
        data_start,
        weights,
        tokenizer,
    })
}
