use std::collections::HashMap;

use color_eyre::eyre::{bail, ensure, eyre};
use serde::Deserialize;

pub const D: usize = 384;
pub const VOCAB: usize = 256_000;
pub const N_TYPES: usize = 3;
pub const N_ENC: usize = 22;
pub const N_HEAD: usize = 2;
pub const ENC_MLP_HIDDEN: usize = 1152;
pub const HEAD_FF: usize = 1536;

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
    pub wi: Tensor,                // [2*ENC_MLP_HIDDEN, D]
    pub wo_mlp: Tensor,            // [D, ENC_MLP_HIDDEN]
    pub global: bool,              // i % 3 == 0
}

#[derive(Debug)]
pub struct Encoder {
    pub tok_embeddings: Tensor, // [VOCAB, D]
    pub emb_norm: Tensor,       // [D]
    pub layers: [EncoderLayer; N_ENC],
    pub final_norm: Tensor, // [D]
}

#[derive(Debug)]
pub struct HeadLayer {
    pub in_proj_w: Tensor,  // [3*D, D]
    pub in_proj_b: Tensor,  // [3*D]
    pub out_proj_w: Tensor, // [D, D]
    pub out_proj_b: Tensor, // [D]
    pub linear1_w: Tensor,  // [HEAD_FF, D]
    pub linear1_b: Tensor,  // [HEAD_FF]
    pub linear2_w: Tensor,  // [D, HEAD_FF]
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
    pub type_emb: Tensor, // [N_TYPES, D]
    pub head: [HeadLayer; N_HEAD],
    pub scorer: Scorer,
}

/// Number of tensors we read. `act_head.*` and `temperature` are present in the file but unused.
const N_USED: usize = 3 + 1 + 6 + N_ENC * 6 - 1 + N_HEAD * 12;

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
pub fn parse_header(raw: &[u8], data_len: usize) -> color_eyre::Result<Weights> {
    let header: Header = serde_json::from_slice(raw)?;

    if header.metadata.len() != 2
        || header.metadata.get("family") != Some(&"julia")
        || header.metadata.get("format") != Some(&"pt")
    {
        bail!(
            "unexpected __metadata__ {:?}, expected family=julia, format=pt",
            header.metadata
        );
    }

    let get = |name: &str, shape: &[usize]| -> color_eyre::Result<Tensor> {
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

    let enc_layer = |i: usize| -> color_eyre::Result<EncoderLayer> {
        let p = format!("encoder.layers.{i}.");
        let attn_norm = format!("{p}attn_norm.weight");
        Ok(EncoderLayer {
            attn_norm: match (i, header.tensors.contains_key(attn_norm.as_str())) {
                (0, false) => None,
                (0, true) => bail!("`{attn_norm}` should not exist"),
                _ => Some(get(&attn_norm, &[D])?),
            },
            wqkv: get(&format!("{p}attn.Wqkv.weight"), &[3 * D, D])?,
            wo: get(&format!("{p}attn.Wo.weight"), &[D, D])?,
            mlp_norm: get(&format!("{p}mlp_norm.weight"), &[D])?,
            wi: get(&format!("{p}mlp.Wi.weight"), &[2 * ENC_MLP_HIDDEN, D])?,
            wo_mlp: get(&format!("{p}mlp.Wo.weight"), &[D, ENC_MLP_HIDDEN])?,
            global: i.is_multiple_of(3),
        })
    };

    let head_layer = |j: usize| -> color_eyre::Result<HeadLayer> {
        let p = format!("head.layers.{j}.");
        Ok(HeadLayer {
            in_proj_w: get(&format!("{p}self_attn.in_proj_weight"), &[3 * D, D])?,
            in_proj_b: get(&format!("{p}self_attn.in_proj_bias"), &[3 * D])?,
            out_proj_w: get(&format!("{p}self_attn.out_proj.weight"), &[D, D])?,
            out_proj_b: get(&format!("{p}self_attn.out_proj.bias"), &[D])?,
            linear1_w: get(&format!("{p}linear1.weight"), &[HEAD_FF, D])?,
            linear1_b: get(&format!("{p}linear1.bias"), &[HEAD_FF])?,
            linear2_w: get(&format!("{p}linear2.weight"), &[D, HEAD_FF])?,
            linear2_b: get(&format!("{p}linear2.bias"), &[D])?,
            norm1_w: get(&format!("{p}norm1.weight"), &[D])?,
            norm1_b: get(&format!("{p}norm1.bias"), &[D])?,
            norm2_w: get(&format!("{p}norm2.weight"), &[D])?,
            norm2_b: get(&format!("{p}norm2.bias"), &[D])?,
        })
    };

    let weights = Weights {
        encoder: Encoder {
            tok_embeddings: get("encoder.embeddings.tok_embeddings.weight", &[VOCAB, D])?,
            emb_norm: get("encoder.embeddings.norm.weight", &[D])?,
            layers: (0..N_ENC)
                .map(enc_layer)
                .collect::<color_eyre::Result<Vec<_>>>()?
                .try_into()
                .unwrap(),
            final_norm: get("encoder.final_norm.weight", &[D])?,
        },
        type_emb: get("type_emb.weight", &[N_TYPES, D])?,
        head: (0..N_HEAD)
            .map(head_layer)
            .collect::<color_eyre::Result<Vec<_>>>()?
            .try_into()
            .unwrap(),
        scorer: Scorer {
            norm_w: get("scorer.0.weight", &[D])?,
            norm_b: get("scorer.0.bias", &[D])?,
            lin1_w: get("scorer.1.weight", &[D, D])?,
            lin1_b: get("scorer.1.bias", &[D])?,
            lin2_w: get("scorer.3.weight", &[1, D])?,
            lin2_b: get("scorer.3.bias", &[1])?,
        },
    };

    // Every used tensor was found by name above, so matching counts means there are no unknown keys.
    let used = header
        .tensors
        .keys()
        .filter(|k| !k.starts_with("act_head.") && **k != "temperature")
        .count();
    ensure!(
        used == N_USED,
        "header has {} unexpected tensors",
        used - N_USED
    );

    Ok(weights)
}
