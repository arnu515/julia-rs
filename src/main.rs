use std::arch;

use julia_rs::{ModelConfig, load_model};

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    if !arch::is_x86_feature_detected!("avxvnni") {
        panic!("AVX-VNNI is required to run this program");
    }

    load_model(ModelConfig {
        safetensors_path: "/path/to/Julia-1/model.safetensors",
        tokenizer_json_path: "/path/to/Julia-1/tokenizer/tokenizer.json",
        tokenizer_config_json_path: "/path/to/Julia-1/tokenizer/tokenizer_config.json",
    })?;
    Ok(())
}
