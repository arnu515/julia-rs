//! Reads requests (Julia's JSONL rows) from stdin and writes one JSON result per line.

use std::{
    arch,
    collections::BTreeMap,
    io::Write,
    path::PathBuf,
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering::Relaxed},
        mpsc,
    },
};

use color_eyre::eyre::{Context, bail, eyre};
use julia_rs::{
    ModelPaths,
    head::Answer,
    load_model,
    sequence::{DEFAULT_HEAD_LENGTH, Request},
};
use serde_json::json;

const USAGE: &str = "usage: julia-rs <checkpoint dir> [--max-length N] [--head-length N] \
[--threads N] < requests.jsonl
  --threads defaults to RAYON_NUM_THREADS, else every logical CPU. One per physical performance
  core is usually fastest: hyperthreads and slow cores mostly contend for cache and memory.";

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    if !arch::is_x86_feature_detected!("avxvnni") {
        panic!("AVX-VNNI is required to run this program");
    }

    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().ok_or_else(|| eyre!(USAGE))?);
    let (mut max_length, mut head_length, mut threads) = (None, DEFAULT_HEAD_LENGTH, 0);
    while let Some(flag) = args.next() {
        let value: usize = args
            .next()
            .ok_or_else(|| eyre!(USAGE))?
            .parse()
            .wrap_err(USAGE)?;
        match flag.as_str() {
            "--max-length" => max_length = Some(value),
            "--head-length" => head_length = value,
            "--threads" => threads = value,
            _ => bail!(USAGE),
        }
    }

    // 0 keeps rayon's default.
    let shared = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;

    let path = |p: &str| dir.join(p);
    let model = load_model(ModelPaths {
        safetensors_path: &path("model.safetensors"),
        encoder_config_json_path: &path("encoder/config.json"),
        julia_config_json_path: &path("julia_config.json"),
        tokenizer_json_path: &path("tokenizer/tokenizer.json"),
        tokenizer_config_json_path: &path("tokenizer/tokenizer_config.json"),
    })?;
    let max_length = max_length.unwrap_or(model.config.max_positions);

    let run = |i: usize, line: &str| -> color_eyre::Result<String> {
        let at = || format!("line {}", i + 1);
        let req: Request = serde_json::from_str(line).wrap_err_with(at)?;
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
        Ok(result.to_string())
    };

    // One worker per thread. A request that is alone (nothing queued or running) spreads its
    // steps over the shared pool for low latency; otherwise each worker runs its request on a
    // private 1-thread pool, since whole requests in parallel beat sharing one pool (~20% more
    // throughput). A writer puts results back in input order and stops at the first error.
    let (job_tx, job_rx) = mpsc::channel::<(usize, usize, String)>();
    let (out_tx, out_rx) = mpsc::channel::<(usize, color_eyre::Result<String>)>();
    let (job_rx, queued, running) = (Mutex::new(job_rx), AtomicUsize::new(0), AtomicUsize::new(0));
    std::thread::scope(|ts| {
        let writer = ts.spawn(move || -> color_eyre::Result<()> {
            let mut out = std::io::stdout().lock();
            let (mut next, mut pending) = (0, BTreeMap::new());
            for (seq, result) in out_rx {
                pending.insert(seq, result);
                while let Some(result) = pending.remove(&next) {
                    writeln!(out, "{}", result?)?;
                    next += 1;
                }
            }
            Ok(())
        });
        for _ in 0..shared.current_num_threads() {
            let (out_tx, job_rx, queued, running, shared, run) =
                (out_tx.clone(), &job_rx, &queued, &running, &shared, &run);
            ts.spawn(move || {
                let private = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .unwrap();
                loop {
                    let Ok((seq, i, line)) = job_rx.lock().unwrap().recv() else {
                        break;
                    };
                    let backlog = queued.fetch_sub(1, Relaxed) > 1;
                    let others = running.fetch_add(1, Relaxed) > 0;
                    let pool = if backlog || others { &private } else { shared };
                    let result = pool.install(|| run(i, &line));
                    running.fetch_sub(1, Relaxed);
                    // the writer is gone only after an earlier error
                    let _ = out_tx.send((seq, result));
                }
            });
        }
        drop(out_tx);
        let read = (|| -> color_eyre::Result<()> {
            let lines = std::io::stdin().lines().enumerate();
            let mut seq = 0;
            for (i, line) in lines {
                let line = line?;
                if !line.trim().is_empty() {
                    queued.fetch_add(1, Relaxed);
                    job_tx.send((seq, i, line))?;
                    seq += 1;
                }
            }
            Ok(())
        })();
        drop(job_tx);
        let written = writer.join().unwrap();
        read.and(written)
    })
}
