//! Reads requests (Julia's JSONL rows) from stdin and writes one JSON result per line.

use std::{arch, io::Write, path::PathBuf};

use color_eyre::eyre::{Context, bail, eyre};
use julia_rs::{
    ModelPaths,
    head::Answer,
    load_model,
    sequence::{DEFAULT_HEAD_LENGTH, Request},
};
use serde_json::json;

const USAGE: &str =
    "usage: julia-rs <checkpoint dir> [--max-length N] [--head-length N] < requests.jsonl";

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    if !arch::is_x86_feature_detected!("avxvnni") {
        panic!("AVX-VNNI is required to run this program");
    }

    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or_else(|| eyre!(USAGE))?);
    let (mut max_length, mut head_length) = (None, DEFAULT_HEAD_LENGTH);
    while let Some(flag) = args.next() {
        let value: usize = args
            .next()
            .ok_or_else(|| eyre!(USAGE))?
            .parse()
            .wrap_err(USAGE)?;
        match flag.as_str() {
            "--max-length" => max_length = Some(value),
            "--head-length" => head_length = value,
            _ => bail!(USAGE),
        }
    }

    let path = |p: &str| dir.join(p);
    let model = load_model(ModelPaths {
        safetensors_path: &path("model.safetensors"),
        encoder_config_json_path: &path("encoder/config.json"),
        julia_config_json_path: &path("julia_config.json"),
        tokenizer_json_path: &path("tokenizer/tokenizer.json"),
        tokenizer_config_json_path: &path("tokenizer/tokenizer_config.json"),
    })?;
    let max_length = max_length.unwrap_or(model.config.max_positions);

    let mut out = std::io::stdout().lock();
    for (i, line) in std::io::stdin().lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let at = || format!("line {}", i + 1);
        let req: Request = serde_json::from_str(&line).wrap_err_with(at)?;
        let p = model
            .predict(&req, max_length, head_length)
            .wrap_err_with(at)?;
        let mut result = json!({
            "logits": p.logits,
            "probabilities": p.probabilities,
            "truncated": p.truncated,
        });
        match p.answer {
            Answer::Choice(i) => result["choice"] = i.into(),
            Answer::Score(s) => result["score"] = s.into(),
            Answer::Noul(t) => result["noul"] = t.into(),
        }
        writeln!(out, "{result}")?;
    }
    Ok(())
}
