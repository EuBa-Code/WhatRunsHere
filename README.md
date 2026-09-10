# WhatLLM

**Which language models will actually run well on your machine — and what happens when they do.**

Most sizing tools answer one question: does the file fit in the memory you have?
That question is easy, and it is not the one that matters. A model that fits at
an empty prompt runs out of memory at 32k context. A model that "fits" with a
tenth of its layers in system RAM runs at a fifth of the speed. A model that fits
comfortably may be a worse choice than a smaller one you could run at higher
precision. And whether any of it beats simply paying for an API depends on your
electricity bill and how much you actually use it.

WhatLLM answers those questions instead.

## What makes it different

**It measures rather than assumes.** Rather than looking an accelerator up in a
table of known cards, WhatLLM derives NVIDIA bandwidth from the bus width and
memory clock the device itself reports — which reproduces the datasheet figure
for parts that do not exist yet — and probes host memory directly. Hardware
nobody has catalogued works as well as hardware everybody has, and the trap that
sinks name tables does not apply: a mobile part shares its model number with a
desktop card of a different bus width, and reading the width from the device
cannot get that wrong.

**It models memory in full.** Weights, attention cache, activations, runtime
overhead and allocator slack — not just the file size. The cache term is
architecture-exact: grouped-query, multi-query, DeepSeek's latent attention,
Gemma's sliding windows and hybrid linear-attention stacks all have genuinely
different slopes, and treating them alike is how a 671B model gets reported as
needing more cache than a 70B one when it needs less.

**It knows throughput decays.** Every generated token reads the whole attention
cache. At long context that cache rivals the weights, and the same hardware runs
at half the speed it showed on an empty prompt. WhatLLM reports the curve, not
the best point on it.

**It prices the decision.** Local versus hosted, including electricity and
hardware amortisation, with the break-even stated as a monthly request count you
can check against your own usage.

**It says what to change.** Not just a verdict but the specific move: quantize
the cache to `q8_0` and you gain 40k of context; drop one rung on the weight
ladder and the whole model stays on the GPU at three times the speed for two
points of quality.

Where a real measurement exists — a real file size, a real probe, a real
benchmark run — it always wins over a formula, and every number carries a label
saying which of the two produced it.

## Status

The engine is built, measured, validated against real hardware, and tested. The
desktop application is not.

| Component | State |
|---|---|
| `whatllm-core` — memory, throughput, quality, cost, fit solver | Working |
| `whatllm-hw` — hardware detection | Working (Windows, Linux, macOS) |
| `whatllm-probe` — on-device calibration | Working (host memory bandwidth) |
| `whatllm-cli` — `doctor`, `probe`, `fit`, `plan`, `cost` | Working |
| Model catalog | 65 models, 836 measured builds |
| GPU compute probe (for time-to-first-token) | Not started |
| Catalog updates over the network | Not started |
| Desktop application (Tauri 2) | Not started |

118 tests, no network access in any of them, and `cargo clippy --all-targets`
clean under `pedantic`.

## How accurate is it

Both halves of the model are checked against reality rather than argued for.

### Memory: 1.15% against real files

The weight-size model is validated against published GGUF builds: **1.15% mean
absolute error across 59 real files**, spanning dense and mixture-of-experts
models from 4B to 32B and every rung from `Q2_K` to `Q8_0`. Three corrections
came out of that, each a place where a plausible-looking model goes wrong:

- A model with tied embeddings stores one tensor serving as both embedding table
  and output projection, and llama.cpp quantizes it at the **output** precision.
  Assuming the embedding precision underestimates a tied 4B by up to 8%.
- The bits-per-weight figures quoted in quantization tables are whole-file
  averages for models with small vocabularies, not body figures. Using them
  directly overshoots `Q3_K_L` by 2.5% and `Q2_K` by 11%; solving for the real
  values against two published builds fixes both.
- `tie_word_embeddings` being absent from a `config.json` does not mean false.
  It means "use this architecture's default", and Gemma's default is true while
  Llama's is false. Reading absence as false mis-sizes every Gemma model.

### Throughput: fitted to 136 measurements on real machines

The decode model says time per token is `bytes / bandwidth + overhead`, which is
a straight line. Fitting it across the dozens of different models each machine
ran recovers both unknowns without assuming either — and the fit quality is the
test, because a wrong bytes-per-token figure would not fall on a line.

| Machine | Models | Fitted bandwidth | R² | Overhead | Of datasheet |
|---|---:|---:|---:|---:|---:|
| RTX 2080 | 15 | 382 GB/s | 0.990 | 1.85 ms | 85% |
| Apple M4 Pro | 31 | 236 GB/s | 0.984 | 2.38 ms | 86% |
| Radeon RX 7900 XT | 46 | 603 GB/s | 0.953 | 1.67 ms | 75% |
| RTX 5070 | 29 | 493 GB/s | 0.898 | 0.57 ms | 73% |
| GTX 1050 Ti | 15 | 43 GB/s | 0.880 | 3.87 ms | 38% |

Three things came out of it that the model did not previously know:

- **Per-token overhead is 1.85 ms, not 0.2.** The old figure was nearly ten
  times too low, which flattered every small model.
- **Sparse models reach 79% of the speed their active parameters predict.**
  Reading eight experts out of a hundred and twenty-eight is a gather, not a
  stream, and a gather does not reach streaming bandwidth. Counting only active
  parameters overstates every mixture-of-experts model.
- **Gemma 2 cannot use flash attention.** It caps its attention logits and the
  flash kernels have nowhere to apply the cap, so llama.cpp materialises the
  full score matrix — quadratic in context. This was found, not looked up: on
  an RTX 2080 sixteen models landed within a few percent of one line and Gemma 2
  9B ran at a third of the speed its size predicts. Modelling it took that
  machine's fit from R² 0.53 to 0.99.

The benchmark corpus is llmfit's community submissions, MIT licensed, used with
thanks. The validation runs as a test: `cargo test -p whatllm-cli --test
throughput_validation -- --nocapture`.

## Using it

```sh
whatllm doctor          # what this machine is, and what has been measured
whatllm probe           # measure memory bandwidth, ~5 seconds
whatllm fit             # rank the catalog for this machine
whatllm plan <model>    # memory breakdown, curves, and what to change
whatllm cost <model>    # local against hosted, with the break-even volume
```

Every command takes `--json`. `--context`, `--parallel`, `--use`, `--prefer` and
`--runtime` shape the question.

## Building

Requires a Rust toolchain. On Windows that also means the Visual Studio C++
build tools, for the linker.

```sh
cargo test
cargo clippy --all-targets
cargo run --release -p whatllm-cli -- doctor
```

To rebuild the model catalog from HuggingFace (contributors only; the catalog
ships with the binary and is also read from `~/.whatllm/catalog.json` if newer):

```sh
python tools/build_catalog.py
```

## Prior art

WhatLLM began as a response to [llmfit](https://github.com/AlexsJones/llmfit) by
Alex Jones, which framed the problem well and is worth your time. The
disagreements are technical, not personal: they are set out above and argued in
the module documentation, which is where the reasoning for each modelling
decision lives.

Three things here came from reading llmfit's source rather than from thinking
about the problem, and are credited in the code that uses them:

- **NVIDIA's unified-memory parts.** GB10 and the Jetson line share one pool
  with the CPU. WhatLLM was treating every NVIDIA device as discrete, which
  counted a DGX Spark's memory twice. That was a bug, and llmfit had already
  found the cases.
- **How to probe memory honestly.** Private per-thread buffers put pages on the
  right node on a multi-node machine; a barrier makes the threads measure
  concurrently instead of letting an early one finish against an uncontended
  controller. Adding both moved this machine's figure from 40.3 GB/s to a
  truthful 37.6. WhatLLM still measures a read rather than a copy, because
  decode streams weights in and writes nothing back.
- **A measurement can be wrong.** A result from a machine under contention comes
  back far too low and one taken in cache far too high, and either is worse than
  no result because it will be believed. `whatllm probe` now refuses to store
  an implausible figure.
- **A rebuild can quietly shrink the catalog.** `build_catalog.py` reads every
  model from HuggingFace, and when that network is unwell the failed entries
  would simply be absent from the file it writes — invisible, because a smaller
  catalog looks exactly like a correct one. llmfit's scraper carries a comment
  naming the day this happened to them across 1,764 models. A failed entry is
  now carried forward from the previous build, and past a threshold the rebuild
  writes nothing at all. `tools/test_catalog_safety.py` holds it to that.
- **An update should add, not replace.** A catalog in `~/.whatllm` is merged
  over the built-in one rather than substituted for it, so adding one model by
  hand cannot remove the sixty-five the binary already knew.

llmfit's community benchmark corpus, MIT licensed, is what the throughput model
is validated against. No llmfit code is copied here.

## Licence

MIT or Apache-2.0, at your option.
