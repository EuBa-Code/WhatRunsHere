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

The engine is built, measured, validated against real hardware, and tested, and
the desktop application runs on it.

| Component | State |
|---|---|
| `whatllm-core` — memory, throughput, quality, cost, fit solver | Working |
| `whatllm-hw` — hardware detection | Working (Windows, Linux, macOS) |
| `whatllm-probe` — on-device calibration | Working (host memory bandwidth) |
| `whatllm-state` — catalog and calibration on disk | Working |
| `whatllm-cli` — `doctor`, `probe`, `fit`, `plan`, `cost` | Working |
| Desktop application — Tauri 2, React 19, Tailwind v4 | Working (Machine, Models, Plan, Cost) |
| Model catalog | 56 models, 701 measured builds |
| Continuous validation | size and throughput models checked on every rebuild |
| Nightly catalog refresh | GitHub Actions, refuses to shrink the catalog |
| GPU compute probe (for time-to-first-token) | Not started |
| Catalog updates over the network | Not started |
| Provider detection — what is already downloaded | Not started |

147 tests, no network access in any of them, and `cargo clippy --all-targets`
clean under `pedantic`.

**The application makes no network calls, and this is enforced rather than
asserted.** Its content security policy admits only the bundled assets and the
IPC channel; the three typefaces are carried in the binary rather than fetched
from a font CDN, which would have been the one request the whole engine exists
to avoid.

Every figure it shows carries a rule beneath it saying where the figure came
from — solid for something measured on this machine, dashed for a
manufacturer's specification, dotted for an assumption nothing has replaced. It
is the same `Confidence` the CLI prints, drawn instead of named.

## How accurate is it

Both halves of the model are checked against reality rather than argued for.

### Memory: 1.5% against 701 real files, checked on every rebuild

The catalog records the exact byte count of every published build it knows
about, which makes the check free and offline: compute each model's size from
its architecture and compare against what the file actually weighs. Across
**701 builds of 56 models** the mean absolute error is **1.5%**.

That number is not a measurement somebody took once. `size_validation` runs it
on every rebuild of the catalog and on every pull request, so an architecture
derived wrongly fails there rather than shipping. It found five real defects the
first time it ran:

- **`mmproj-` and `mtp-` files were being recorded as builds.** A multimodal
  projector and a multi-token-prediction head carry the same quantization suffix
  as the weights. Someone asking for gemma-4-12B's F16 was being sent to a
  122 MB vision encoder.
- **A tied model's vocabulary tensor is sometimes written twice.**
  `tie_word_embeddings` says the *weights* are shared; it does not say what the
  GGUF converter wrote. Qwen3 0.6B, 1.7B and Qwen3.5-2B are tied and ship it
  twice; Qwen3 4B, equally tied, ships it once. On a 0.6B that is a fifth of the
  file, and nothing in the metadata distinguishes them — so the builder now
  reads it off the published sizes instead.
- **Not every feed-forward block is gated.** The builder assumed three matrices
  everywhere. For the models that use two, that overstates the feed-forward by
  half — a third of the whole file. Also now read off the files.
- **gpt-oss keeps its experts in MXFP4 whatever the build is called.** Its F16,
  Q8_0 and Q6_K files are 13.79, 12.11 and 12.04 GB: differences a whole-model
  quantization cannot produce, because 91% of the model never changes. Treating
  its "F16" as sixteen bits overstates it threefold.
- **Hybrid stacks were being described as ordinary transformers.** Qwen3.5,
  Qwen3.6 and LFM2 mix linear-attention or convolution blocks in among the
  attention ones. The memory model can carry them, but only when told how many
  parameters each block holds, and that does not follow from the config. They
  are rejected with the reason rather than sized wrongly.

Three earlier corrections, from validating against published GGUF builds by
hand, are what got the model to a percent in the first place:

- A tied model's shared tensor is quantized at the **output** precision, not the
  embedding's — assuming otherwise underestimates a tied 4B by up to 8%.
- The bits-per-weight figures in quantization tables are whole-file averages for
  small vocabularies, not body figures; using them overshoots `Q3_K_L` by 2.5%
  and `Q2_K` by 11%.
- `tie_word_embeddings` being absent does not mean false. It means "use this
  architecture's default", and Gemma's default is true.

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

The desktop application additionally needs Node 22, and on Linux the system
webview (`libwebkit2gtk-4.1-dev` and the packages beside it in `ci.yml`).

```sh
npm --prefix ui ci
npm --prefix ui run build     # the window will not compile without this
cargo run -p whatllm-app      # run it
npx --prefix ui tauri build   # an installer for this platform
```

`tauri::generate_context!` reads the built front end at compile time, so a
fresh checkout cannot build the window until `npm run build` has produced
`ui/dist`.

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

Several things here came from reading llmfit's source rather than from thinking
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
  hand cannot remove the ones the binary already knew.
- **The memory the operating system reports is not the memory installed.**
  Firmware on an AMD APU hands a block to the integrated GPU before the kernel
  starts, and the OS never sees it: a 128 GB Ryzen AI MAX+ with 96 GB carved out
  reports around 31 GB. Sizing against that rejects every model the machine was
  bought to run. WhatLLM reads the DIMM capacity out of the SMBIOS table Windows
  caches in the registry — no WMI query, no spawned `powershell` — and counts
  the difference back in.
- **A large integrated part looks exactly like a card.** That same 96 GB is
  reported by the adapter as 96 GB of "video memory" under the name
  `AMD Radeon Graphics`, which is also what a discrete Instinct MI50 calls
  itself. The memory floor that settles the MI50 gets this one backwards and
  describes the machine as a 96 GB GPU beside 31 GB of RAM, when it is one pool
  of 128. The measured carveout tells them apart, because it is the same memory
  seen from the other side.
- **macOS will not let Metal wire all of the memory.** Apple Silicon has one
  pool, but a compute job gets roughly three quarters of it on the smaller
  machines, and an allocation past that fails rather than paging. WhatLLM asks
  Metal for `recommendedMaxWorkingSetSize` rather than reproducing the default
  as a formula, so the answer stays right across macOS releases and follows an
  owner who has raised `iogpu.wired_limit_mb` by hand.
- **A stranger's metadata can block your own commits.** A publisher who once
  pasted an access token where a repository name belonged leaves it in the
  upstream metadata for good, and GitHub's secret scanning then rejects the push
  carrying it — so the nightly catalog rebuild stops committing rather than
  merely being wrong. It blocked llmfit on 2026-08-03. `build_catalog.py` now
  drops such an entry and names it redacted.
- **A generic name can hide a serious card.** A 32 GB Instinct MI50 reports
  itself as `AMD Radeon Graphics`, the same string an integrated Cezanne uses.
  WhatLLM was reading the name and sizing it against system memory. Reported
  memory now settles it before any name does: nothing integrated owns five
  gigabytes.
- **Integrated graphics can be too old to be a target at all.** A Coffee Lake
  UHD 630 was being reported as a unified-memory GPU with the whole RAM pool
  behind it while every load ran on the CPU. Intel parts predating Xe are now
  named as what they are, and the answer given is the CPU.
- **A 32-bit memory field cannot be repaired.** Windows reports video memory
  through fields four bytes wide, and the ceiling value sits *between* a real
  2 GB card and a real 2 GiB one — so no threshold separates the three. A
  narrow reading is discarded rather than believed, and the card is left out
  with a note. A compile-time assertion keeps anyone from inventing the
  threshold later.
- **A bug report should be a test.** llmfit's `doctor` captures the raw output
  of everything it consulted, so a report pastes straight into the suite as a
  regression fixture. `whatllm doctor --json` now carries the unclassified
  adapter list alongside the verdict, and `classify_raw` replays it without
  touching a machine.

llmfit's community benchmark corpus, MIT licensed, is what the throughput model
is validated against. No llmfit code is copied here.

## Licence

MIT or Apache-2.0, at your option.
