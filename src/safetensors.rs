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

use std::collections::HashMap;

use eyre::{bail, ensure, eyre};
use serde::Deserialize;

use crate::config::{ModelConfig, N_QTYPES};

/// Byte range of a tensor inside the data section (after the 8-byte length + header).
#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    pub start: usize,
    pub end: usize,
}

impl Tensor {
    /// View this tensor as f32s. `data` must be the data section of the file.
    pub fn as_f32<'m>(&self, data: &'m [u8]) -> &'m [f32] {
        let bytes = &data[self.start..self.end];
        // SAFETY: every bit pattern is a valid f32; alignment is checked below.
        let (pre, floats, post) = unsafe { bytes.align_to::<f32>() };
        assert!(
            pre.is_empty() && post.is_empty(),
            "tensor is not 4-byte aligned"
        );
        floats
    }
}

#[derive(Debug)]
pub struct EncoderLayer {
    pub attn_norm: Option<Tensor>, // None for layer 0
    pub wqkv: Tensor,              // [3*D, D]
    pub wo: Tensor,                // [D, D]
    pub mlp_norm: Tensor,          // [D]
    pub wi: Tensor,                // [2*intermediate, D]
    pub wo_mlp: Tensor,            // [D, intermediate]
    pub global: bool,              // full attention (else sliding window), from `layer_types`
}

#[derive(Debug)]
pub struct Encoder {
    pub tok_embeddings: Tensor,    // [VOCAB, D]
    pub emb_norm: Tensor,          // [D]
    pub layers: Vec<EncoderLayer>, // n_layers
    pub final_norm: Tensor,        // [D]
}

#[derive(Debug)]
pub struct HeadLayer {
    pub in_proj_w: Tensor,  // [3*D, D]
    pub in_proj_b: Tensor,  // [3*D]
    pub out_proj_w: Tensor, // [D, D]
    pub out_proj_b: Tensor, // [D]
    pub linear1_w: Tensor,  // [head_ff, D]
    pub linear1_b: Tensor,  // [head_ff]
    pub linear2_w: Tensor,  // [D, head_ff]
    pub linear2_b: Tensor,  // [D]
    pub norm1_w: Tensor,    // [D]
    pub norm1_b: Tensor,    // [D]
    pub norm2_w: Tensor,    // [D]
    pub norm2_b: Tensor,    // [D]
}

#[derive(Debug)]
pub struct Scorer {
    pub norm_w: Tensor, // scorer.0, [D]
    pub norm_b: Tensor,
    pub lin1_w: Tensor, // scorer.1, [D, D]
    pub lin1_b: Tensor,
    pub lin2_w: Tensor, // scorer.3, [1, D]
    pub lin2_b: Tensor, // [1]
}

#[derive(Debug)]
pub struct Weights {
    pub encoder: Encoder,
    pub type_emb: Tensor,     // [N_QTYPES, D]
    pub head: Vec<HeadLayer>, // head_layers
    pub scorer: Scorer,
}

#[derive(Deserialize)]
struct RawTensor<'a> {
    dtype: &'a str,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

#[derive(Deserialize)]
struct Header<'a> {
    #[serde(rename = "__metadata__", borrow)]
    metadata: HashMap<&'a str, &'a str>,
    #[serde(flatten, borrow)]
    tensors: HashMap<&'a str, RawTensor<'a>>,
}

/// `data_len` is the size of the data section (file length - 8 - header length).
/// Every tensor's shape is checked against the dimensions in `cfg`.
pub fn parse_header(raw: &[u8], data_len: usize, cfg: &ModelConfig) -> eyre::Result<Weights> {
    let header: Header = serde_json::from_slice(raw)?;
    let d = cfg.hidden;

    if header.metadata.len() != 2
        || header.metadata.get("family") != Some(&"julia")
        || header.metadata.get("format") != Some(&"pt")
    {
        bail!(
            "unexpected __metadata__ {:?}, expected family=julia, format=pt",
            header.metadata
        );
    }

    let get = |name: &str, shape: &[usize]| -> eyre::Result<Tensor> {
        let t = header
            .tensors
            .get(name)
            .ok_or_else(|| eyre!("missing tensor `{name}`"))?;
        ensure!(
            t.dtype == "F32",
            "tensor `{name}` has dtype {}, expected F32",
            t.dtype
        );
        ensure!(
            t.shape == shape,
            "tensor `{name}` has shape {:?}, expected {shape:?}",
            t.shape
        );
        let (start, end) = t.data_offsets;
        let bytes = shape.iter().product::<usize>() * size_of::<f32>();
        ensure!(
            end <= data_len && end.checked_sub(start) == Some(bytes),
            "tensor `{name}` has bad data_offsets [{start}, {end}] (data section is {data_len} bytes)"
        );
        Ok(Tensor { start, end })
    };

    let enc_layer = |i: usize| -> eyre::Result<EncoderLayer> {
        let p = format!("encoder.layers.{i}.");
        let attn_norm = format!("{p}attn_norm.weight");
        Ok(EncoderLayer {
            attn_norm: match (i, header.tensors.contains_key(attn_norm.as_str())) {
                (0, false) => None,
                (0, true) => bail!("`{attn_norm}` should not exist"),
                _ => Some(get(&attn_norm, &[d])?),
            },
            wqkv: get(&format!("{p}attn.Wqkv.weight"), &[3 * d, d])?,
            wo: get(&format!("{p}attn.Wo.weight"), &[d, d])?,
            mlp_norm: get(&format!("{p}mlp_norm.weight"), &[d])?,
            wi: get(&format!("{p}mlp.Wi.weight"), &[2 * cfg.intermediate, d])?,
            wo_mlp: get(&format!("{p}mlp.Wo.weight"), &[d, cfg.intermediate])?,
            global: cfg.global[i],
        })
    };

    let head_layer = |j: usize| -> eyre::Result<HeadLayer> {
        let p = format!("head.layers.{j}.");
        Ok(HeadLayer {
            in_proj_w: get(&format!("{p}self_attn.in_proj_weight"), &[3 * d, d])?,
            in_proj_b: get(&format!("{p}self_attn.in_proj_bias"), &[3 * d])?,
            out_proj_w: get(&format!("{p}self_attn.out_proj.weight"), &[d, d])?,
            out_proj_b: get(&format!("{p}self_attn.out_proj.bias"), &[d])?,
            linear1_w: get(&format!("{p}linear1.weight"), &[cfg.head_ff, d])?,
            linear1_b: get(&format!("{p}linear1.bias"), &[cfg.head_ff])?,
            linear2_w: get(&format!("{p}linear2.weight"), &[d, cfg.head_ff])?,
            linear2_b: get(&format!("{p}linear2.bias"), &[d])?,
            norm1_w: get(&format!("{p}norm1.weight"), &[d])?,
            norm1_b: get(&format!("{p}norm1.bias"), &[d])?,
            norm2_w: get(&format!("{p}norm2.weight"), &[d])?,
            norm2_b: get(&format!("{p}norm2.bias"), &[d])?,
        })
    };

    let weights = Weights {
        encoder: Encoder {
            tok_embeddings: get("encoder.embeddings.tok_embeddings.weight", &[cfg.vocab, d])?,
            emb_norm: get("encoder.embeddings.norm.weight", &[d])?,
            layers: (0..cfg.n_layers)
                .map(enc_layer)
                .collect::<eyre::Result<_>>()?,
            final_norm: get("encoder.final_norm.weight", &[d])?,
        },
        type_emb: get("type_emb.weight", &[N_QTYPES, d])?,
        head: (0..cfg.head_layers)
            .map(head_layer)
            .collect::<eyre::Result<_>>()?,
        scorer: Scorer {
            norm_w: get("scorer.0.weight", &[d])?,
            norm_b: get("scorer.0.bias", &[d])?,
            lin1_w: get("scorer.1.weight", &[d, d])?,
            lin1_b: get("scorer.1.bias", &[d])?,
            lin2_w: get("scorer.3.weight", &[1, d])?,
            lin2_b: get("scorer.3.bias", &[1])?,
        },
    };

    // Every used tensor was found by name above, so matching counts means there are no unknown keys.
    // `act_head.*` and `temperature` are present in the file but unused.
    let n_used = 3 + 1 + 6 + cfg.n_layers * 6 - 1 + cfg.head_layers * 12;
    let used = header
        .tensors
        .keys()
        .filter(|k| !k.starts_with("act_head.") && **k != "temperature")
        .count();
    ensure!(
        used == n_used,
        "header has {} unexpected tensors",
        used - n_used
    );

    Ok(weights)
}
