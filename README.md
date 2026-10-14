# julia-rs

A CPU inference engine for [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1), written in
Rust with hand-written AVX2 / AVX-VNNI kernels. Runs in fp32, int16, or int8 precision.

## Requirements

- An x86-64 CPU with **AVX2, FMA and AVX-VNNI** (Intel Alder Lake / Meteor Lake or newer, AMD Zen 5
  or newer). The binary checks for AVX-VNNI at startup.
  - Please feel free to open a pull-request with other combinations (e.g. AVX-512, AVX2 + FMA, no VNNI, etc.)
- Linux (tested). Other Unix-likes should work; the memory release after packing is best effort.
- Rust nightly (pinned in `rust-toolchain.toml`).
- These files downloaded from the Julia-1 HF repo: `model.safetensors`, `julia_config.json`,
  `encoder/config.json`, `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`.

## Build

```sh
cargo build --release
```

`.cargo/config.toml` enables `avx2,fma,avxvnni,f16c,bmi2` for the whole crate, so the binary only
runs on CPUs that have them.

## Usage

```sh
julia-rs <checkpoint dir> [--max-length N] [--head-length N] [--threads N] [--workers N] \
    [--precision f32|i16|i8] < requests.jsonl > results.jsonl
```

Each input line is one request of this sorts:

```json
{"state": {"task": "Rotate the expired TLS certificate.", "tool_errors": 0},
 "question": "What should the observability system do with this trace?",
 "type": "choice",
 "options": ["Let the agent proceed.", "Queue for human review.", "Halt the agent now."]}
```

- `state`: text or any JSON value.
- `type`: `choice` (default), `score` (options are ordered levels) or `noul` (exactly two options, ordered `[false, true]`).
- `options`: 2–20 non-empty descriptions.

Each output line, in input order:

```json
{"logits": [...], "probabilities": [...], "truncated": false, "choice": 1}
```

with `choice` (index of the most probable option), `score` (expected level, `Σ i·pᵢ`) or `noul`
(probability of `true`), depending on the type. `truncated` says whether the state was cut to fit
`--max-length`. The run stops at the first invalid request and reports its line number.

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--precision` | `f32` | Number format of the linear layers (see below). |
| `--max-length` | model maximum | Token budget for the whole sequence; the state is truncated to fit. |
| `--head-length` | 256 | Token budget for the question and options (`TransformerEngine`'s default). |
| `--threads` | `RAYON_NUM_THREADS`, else all logical CPUs | Threads for a request that runs alone. |
| `--workers` | `--threads` | Requests run at once (one thread each) when several are waiting. |

The typed-decisions benchmark uses `--max-length 1024 --head-length 512`.

### Choosing threads on a hybrid CPU

> Examples are for Core Ultra 9 185H

- **Latency (one request at a time):** one thread per physical P core.

  ```sh
  taskset -c 0,1,3,6,8,10 julia-rs ckpt --threads 6 < requests.jsonl
  ```

- **Throughput (many requests):** one thread per P core and E core.

  ```sh
  taskset -c 0,1,3,6,8,10,12-19 julia-rs ckpt --threads 6 --workers 14 < requests.jsonl
  ```

Please measure for your workload and pick what's best!

### As a library

```rust
use julia_rs::{ModelPaths, load_model, matmul::Precision, sequence::Request};

let model = load_model(ModelPaths { /* the five checkpoint paths */ }, Precision::I16)?;
let request: Request = serde_json::from_str(line)?;
let prediction = model.predict(&request, 1024, 512)?; // logits, probabilities, answer
```

## Precision modes

| Mode | Linear layers | Accuracy cost | Speed |
| --- | --- | --- | --- |
| `f32` | f32 weights and activations, FMA | reference | 1× |
| `i16` | 13-bit integers in i16 (one scale per weight row and per token), `vpdpwssd` | negligible | ~1.5–1.6× |
| `i8` | int8 weights × u8 activations (scale + zero point per token), `vpdpbusd` | −0.70 pp | ~2.3–2.7× |

Everything else (embeddings, LayerNorm, attention, softmax, GeGLU, RoPE, residuals, the scorer's
GELU) are in f32 in every mode.

## Benchmarks

> [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)

| Run | Correct / 2,000 | Accuracy | Choice | Noul | Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Julia-1 model card (H200, BF16) | 1,463 | **73.15%** | 428 (71.33%) | 484 (80.67%) | 551 (68.88%) |
| Julia-1 (PyTorch CPU, FP32) | 1,451 | 72.55% | 426 (71.00%) | 483 (80.50%) | 542 (67.75%) |
| **julia-rs `f32`** | **1,451** | **72.55%** | 426 (71.00%) | 483 (80.50%) | 542 (67.75%) |
| julia-rs `i16` | 1,452 | 72.60% | 426 (71.00%) | 484 (80.67%) | 542 (67.75%) |
| julia-rs `i8` | 1,437 | 71.85% | 420 (70.00%) | 481 (80.17%) | 536 (67.00%) |
| Jev reference (from the model card) | — | 72.70% | | | |

> All numbers, except those from model cards, were run on my Core Ultra 9 185H Laptop in performance mode.

### Throughput over all 2,000 questions

Wall time through the CLI, including the 0.3s load, with `--threads 6`.

| Mode | 6 P-cores (`--workers 6`) | + 8 E-cores (`--workers 14`) | Peak memory (P / P+E) |
| --- | ---: | ---: | ---: |
| `f32` | 123.3 s · 16.2 q/s | 92.0 s · 21.7 q/s | 410 / 523 MB |
| `i16` | 76.3 s · 26.2 q/s | 53.8 s · 37.2 q/s | 332 / 458 MB |
| `i8` | 45.9 s · 43.6 q/s | 34.0 s · 58.8 q/s | 282 / 410 MB |
| PyTorch reference, CPU FP32 (4 threads) | 601.7 s · 3.3 q/s | | |

### Latency: one request at a time

100 questions from the suite, each run alone (`Model::predict`). We pool three runs.

| Mode | 6 P-cores: mean | p50 | p95 | 1 thread: mean | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `f32` | 68.1 ms | 66.1 ms | 104.1 ms | 258.0 ms | 256.2 ms | 408.3 ms |
| `i16` | 45.6 ms | 45.1 ms | 69.7 ms | 155.6 ms | 155.1 ms | 246.8 ms |
| `i8` | 29.9 ms | 29.3 ms | 45.6 ms | 95.8 ms | 95.6 ms | 154.6 ms |

### Load time and memory

Loading (parse, tokenizer, packing every linear weight) takes 0.31–0.32 s in every mode. Peak
memory after load is 305 MB (`f32`), 216 MB (`i16`) and 171 MB (`i8`). With work in flight it
grows by each worker's buffers (see the throughput table).

### Kernels (single core)

| | f32 | i16 | i8 |
| --- | ---: | ---: | ---: |
| Linear layers, model shapes | 116–148 GFLOP/s | 215–267 GFLOP/s | 378–494 GFLOP/s |
| Instruction peak measured on a P-core | 76 GMAC/s (FMA) | 150 GMAC/s (`vpdpwssd`) | 305 GMAC/s (`vpdpbusd`) |

Attention runs at 78–101 GFLOP/s on one core.

## Optimisations

- **Packed weights, 6×16 register-tile matmul** (`matmul.rs`):
  - Each weight matrix is repacked once at load into panels of 16 outputs, so the kernel reads
    two contiguous vectors per input.
  - It keeps 6 tokens × 16 outputs in 12 accumulators (15 of the 16 ymm registers).
  - f32 sums are blocked every 64 inputs to keep PyTorch-level accuracy; one long chain over
    1,152 inputs was 4× less accurate.
  - About 80% of one core's FMA peak (121–148 GFLOP/s single-threaded).
- **Integer linear layers:**
  - **`i16`:** 13-bit values with `vpdpwssd`. The i32 sums are converted to f32 every 128 inputs,
    so they can't overflow.
  - **`i8`:** `vpdpbusd`, with the per-token zero point removed exactly in i32 using each output's
    weight sum.
  - **Inline asm:** both instructions are emitted as inline asm, because LLVM splits the
    intrinsics into two instructions each, which halved the kernel's speed.
  - **Speed:** 215–267 GFLOP/s (`i16`) and 378–494 GFLOP/s (`i8`) on one core.
- **Why 13 bits:** the model is unusually sensitive to quantisation. Plain int8 flips 9% of answers,
  and int8 weights alone flip 5%. Some activation channels reach 15,000× the median.
  - **Simulated before building:** per-group scales, SmoothQuant, keeping outlier channels exact,
    and many mixed-precision policies. None was both fast and accurate.
  - **Bit-width sweep:** 12–13-bit values were the cheapest that kept every answer unchanged.
- **Attention** (`attn.rs`):
  - Q·Kᵀ and P·V run in 6-query × 16-key register tiles.
  - The softmax is a bare `exp2`, with log₂e and 1/√d folded into Q.
  - P is transposed so each key's probabilities are read together.
  - Each head's K (transposed, in 16-key blocks) and V are copied once to be read sequentially.
  - Sliding-window layers compute only the window.
  - 78–101 GFLOP/s on one core, up from 55–59.
- **Threading** (rayon):
  - **One request:** each step is split across threads: matmuls into 48-token × 96-output
    blocks, attention by query block (all heads per task), and norms and elementwise ops by row.
  - **Many requests:** each worker runs a whole request on its own single-thread pool, which
    gave about 20% more throughput than sharing one pool. A writer thread restores input order.
- **Fused residual adds:** `wo`, `wo_mlp`, `out_proj` and `linear2` add straight into the residual
  stream.
- **Memory:** after packing, the f32 weights' pages are dropped from memory (`MADV_DONTNEED` on
  the read-only file mapping). Peak memory after load is 305 / 215 / 171 MB in f32 / i16 / i8,
  down from 483 / 394 / 350 MB.
- **Head:** the last head layer computes queries, attention and MLP only for the option-marker
  rows.

Tried and dropped:

- **Adaptive matmul block sizes:** single requests 10% slower, no throughput gain.
- **A persistent thread team per encoder layer** (each thread keeps its token rows through the
  layer, 2 barriers instead of about 13 parallel regions): only 2–9% faster on 6 threads, with a
  worse `i8` p95, for about 150 lines of `unsafe`. The remaining 6-thread loss is mostly hardware:
  the all-core clock is 10–18% lower, and there are more cache misses.
- **Reusing work buffers:** after load, 100 requests cause only about 650 page faults.
- **Mixed int8/i16 policies:** the useful ones save only 8–20% of the multiply-adds and triple
  the logit error.
