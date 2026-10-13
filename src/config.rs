//! Model dimensions, read from `encoder/config.json` (ModernBERT) and `julia_config.json` (head).

use color_eyre::eyre::{bail, ensure};
use serde::Deserialize;

/// Question types (`choice`, `score`, `noul`). Defined in Julia's code (`QTYPES` in `data.py`),
/// not in any config file; the Rust side has to know the mapping anyway.
pub const N_QTYPES: usize = 3;

#[derive(Debug)]
pub struct ModelConfig {
    /// Hidden size D, shared by the encoder and head.
    pub hidden: usize,
    pub vocab: usize,
    pub n_layers: usize,
    /// Encoder MLP width; `mlp.Wi` is `[2 * intermediate, hidden]` (input half, then gate half).
    pub intermediate: usize,
    pub n_heads: usize,
    /// Per encoder layer: global (full) attention, or sliding-window attention.
    pub global: Vec<bool>,
    /// Sliding window size in tokens (total, so each token sees `local_window / 2` either side).
    pub local_window: usize,
    pub rope_theta_global: f32,
    pub rope_theta_local: f32,
    pub norm_eps: f32,

    pub head_layers: usize,
    /// Head `nn.TransformerEncoderLayer(width, max(1, width // 64), 4 * width)` in `julia/model.py`.
    pub head_heads: usize,
    pub head_ff: usize,
}

#[derive(Deserialize)]
struct EncoderJson {
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    layer_types: Vec<String>,
    local_attention: usize,
    rope_parameters: RopeJson,
    norm_eps: f32,
    norm_bias: bool,
    attention_bias: bool,
    mlp_bias: bool,
}

#[derive(Deserialize)]
struct RopeJson {
    full_attention: RopeThetaJson,
    sliding_attention: RopeThetaJson,
}

#[derive(Deserialize)]
struct RopeThetaJson {
    rope_theta: f32,
}

#[derive(Deserialize)]
struct JuliaJson {
    head_layers: usize,
}

impl ModelConfig {
    pub fn load(encoder_config_json: &str, julia_config_json: &str) -> color_eyre::Result<Self> {
        let enc: EncoderJson =
            serde_json::from_str(&std::fs::read_to_string(encoder_config_json)?)?;
        let julia: JuliaJson = serde_json::from_str(&std::fs::read_to_string(julia_config_json)?)?;

        // The weight structs assume ModernBERT without biases.
        ensure!(
            !enc.norm_bias && !enc.attention_bias && !enc.mlp_bias,
            "encoder config has biases enabled; only bias-free ModernBERT is supported"
        );
        ensure!(
            enc.layer_types.len() == enc.num_hidden_layers,
            "layer_types has {} entries but num_hidden_layers is {}",
            enc.layer_types.len(),
            enc.num_hidden_layers
        );
        ensure!(
            enc.hidden_size.is_multiple_of(enc.num_attention_heads),
            "hidden_size {} is not divisible by num_attention_heads {}",
            enc.hidden_size,
            enc.num_attention_heads
        );
        let global = enc
            .layer_types
            .iter()
            .map(|t| match t.as_str() {
                "full_attention" => Ok(true),
                "sliding_attention" => Ok(false),
                _ => bail!("unknown layer type `{t}`"),
            })
            .collect::<color_eyre::Result<_>>()?;

        Ok(Self {
            hidden: enc.hidden_size,
            vocab: enc.vocab_size,
            n_layers: enc.num_hidden_layers,
            intermediate: enc.intermediate_size,
            n_heads: enc.num_attention_heads,
            global,
            local_window: enc.local_attention,
            rope_theta_global: enc.rope_parameters.full_attention.rope_theta,
            rope_theta_local: enc.rope_parameters.sliding_attention.rope_theta,
            norm_eps: enc.norm_eps,
            head_layers: julia.head_layers,
            head_heads: (enc.hidden_size / 64).max(1),
            head_ff: 4 * enc.hidden_size,
        })
    }
}
