# WhatLLM

**Which language models will actually run well on your machine. Measured, not
looked up.**

Any tool that answers this needs two numbers about your hardware: how much
memory it has, and how fast that memory is. The second one is what decides how
fast a model generates, and nearly everything reads it out of a table of known
cards.

A table is wrong in a specific and common way. An RTX 5070 Laptop GPU has a
128-bit bus and 8 GB; the desktop card that shares the number has 192-bit and
12 GB. Keyed on the model number, a table hands back the desktop figure for the
laptop (llmfit measured that overestimate at up to 1.8× across the 30, 40 and
50 mobile lines), and the error lands in every tokens-per-second figure built on
top of it.

WhatLLM does not keep that table. It derives bandwidth from the bus width and
memory clock the device itself reports, which reproduces the published figure
for every part NVIDIA has shipped and will keep reproducing it for parts that do
not exist yet. Then it measures system memory directly, with a benchmark that
takes about a second, and says so: every figure it shows carries a mark saying
whether it was measured here, taken from a specification, or assumed.

The rest follows from getting that right. A model that fits at an empty prompt
runs out of memory at 32k context. A model that "fits" with a tenth of its
layers in system RAM runs at a fifth of the speed. A model that fits comfortably
may be a worse choice than a smaller one at higher precision. And whether any of
it beats paying for an API depends on your electricity bill and how much you
actually use it.

## What makes it different

**It measures rather than assumes.** Rather than looking an accelerator up in a
table of known cards, WhatLLM derives NVIDIA bandwidth from the bus width and
memory clock the device itself reports, which reproduces the datasheet figure
for parts that do not exist yet, and probes host memory directly. Hardware
nobody has catalogued works as well as hardware everybody has, and the trap that
sinks name tables does not apply: a mobile part shares its model number with a
desktop card of a different bus width, and reading the width from the device
cannot get that wrong.

**It models memory in full.** Weights, attention cache, activations, runtime
overhead and allocator slack, not just the file size. The cache term is
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

Where a real measurement exists (a real file size, a real probe, a real
benchmark run) it always wins over a formula, and every number carries a label
saying which of the two produced it.

## Status

The engine is built, measured, validated against real hardware, and tested, and
the desktop application runs on it.

| Component | State |
|---|---|
| `whatllm-core`: memory, throughput, quality, cost, fit solver | Working |
| `whatllm-hw`: hardware detection | Working (Windows, Linux, macOS) |
| `whatllm-probe`: on-device calibration | Working (host memory bandwidth) |
| `whatllm-state`: catalog and calibration on disk | Working |
| `whatllm-cli`: `doctor`, `probe`, `fit`, `plan`, `installed`, `cost` | Working |
| Desktop application: Tauri 2, React 19, Tailwind v4 | Working (Machine, Models, Plan, Cost) |
| Handing the model to its runtime: a download, a Modelfile, or the command that works | Working (llama.cpp, LM Studio, Ollama, vLLM, MLX) |
| Model catalog | 97 models, 1196 measured builds |
| Continuous validation | size and throughput models checked on every rebuild |
| Nightly catalog refresh | GitHub Actions, refuses to shrink the catalog |
| GPU compute probe (for time-to-first-token) | Not started |
| Catalog updates over the network | Not started |
| What is already on this machine: WhatLLM's folder, LM Studio, Ollama, llama.cpp's cache, the HuggingFace cache | Working (disk only, identified by exact size) |

188 Rust tests and 10 Python tests, no network access in any of them, and `cargo clippy --all-targets`
clean under `pedantic`.

**Nothing here reaches the network unless you press a button that says it
will.** Every answer (what your machine is, what fits, how fast, what it costs)
is computed offline from the catalog compiled into the binary. The one
exception is downloading a model's weights, which happens only when you ask for
them and only from the repository the catalog recorded.

That boundary is enforced rather than asserted. The window's content security
policy admits the bundled assets and the IPC channel and nothing else, so the
page itself cannot make a request at all: a transfer can only originate in the
Rust process, from the one module that does it. The three typefaces are carried
in the binary rather than fetched from a font CDN, which would have been a
request nobody asked for.

Every figure it shows carries a rule beneath it saying where the figure came
from: solid for something measured on this machine, dashed for a
manufacturer's specification, dotted for an assumption nothing has replaced. It
is the same `Confidence` the CLI prints, drawn instead of named.

The quality score is anchored to published evaluations, copied from the
publisher's model card (MMLU-Pro, GPQA Diamond, LiveCodeBench, MATH-500,
IFEval) and, where the card has none, from the Open LLM Leaderboard's raw
columns. Each entry in `catalog/sources.json` names where its figures were
read. Where nothing is published the score is inferred from size and says so,
with the fraction of the use case that real evaluations back.

## How accurate is it

Both halves of the model are checked against reality rather than argued for.

### Memory: 1.4% against 1169 real files, checked on every rebuild

The catalog records the exact byte count of every published build it knows
about, which makes the check free and offline: compute each model's size from
its architecture and compare against what the file actually weighs. Across
**1169 builds of 97 models** the mean absolute error is **1.4%**.

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
  file, and nothing in the metadata distinguishes them, so the builder now
  reads it off the published sizes instead.
- **Not every feed-forward block is gated.** The builder assumed three matrices
  everywhere. For the models that use two, that overstates the feed-forward by
  half, which is a third of the whole file. Also now read off the files.
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
  embedding's. Assuming otherwise underestimates a tied 4B by up to 8%.
- The bits-per-weight figures in quantization tables are whole-file averages for
  small vocabularies, not body figures; using them overshoots `Q3_K_L` by 2.5%
  and `Q2_K` by 11%.
- `tie_word_embeddings` being absent does not mean false. It means "use this
  architecture's default", and Gemma's default is true.

### Throughput: fitted to 136 measurements on real machines

The decode model says time per token is `bytes / bandwidth + overhead`, which is
a straight line. Fitting it across the dozens of different models each machine
ran recovers both unknowns without assuming either, and the fit quality is the
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
  full score matrix, which is quadratic in context. This was found, not looked up: on
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
whatllm plan <model>    # memory breakdown, curves, what to change, how to run it
whatllm installed       # what is already on this disk, and how each would run
whatllm cost <model>    # local against hosted, with the break-even volume
```

`installed` reads the places each local runtime keeps its files (WhatLLM's
own folder, LM Studio's, Ollama's store, llama.cpp's cache and the HuggingFace
cache) and identifies each file against the catalog by its exact byte count,
which across 1196 builds is as good as a fingerprint. A file it recognises is
sized for exactly the build it is; one it does not is named from its own
header and declared not sizeable, rather than guessed at. Nothing is asked of
any running program.

Every command takes `--json`. `--context`, `--parallel`, `--use`, `--prefer` and
`--runtime` shape the question.

`plan` ends with how to run it, and that answer changes with `--runtime`,
because the catalog holds GGUF and not every runtime runs GGUF. For llama.cpp
it is the `llama-server` line with the context, the layer split and the cache
format that were sized, so the thing that runs is the thing that was measured.
For Ollama it is a Modelfile and the import. For LM Studio it is the path
inside LM Studio's own folder, so the model appears in its list with nothing
to move. For vLLM it is the `vllm serve` line and no download at all, because
vLLM fetches the original weights itself. For MLX it is a search, because the
catalog does not carry a conversion and guessing one would be exactly the
approximation this tool refuses. The window does the same, with a button in
place of the download command.

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

## It stays free

Free, under MIT or Apache 2.0, with no feature gating, no licensing code and no
telemetry. The architecture has no place to put a check, which is deliberate.

That is a commitment rather than a stage. A project that never promised to stay
free and later charges has broken nothing; one that promised and then charged
has broken the only thing it had. So the promise is made here, where it can be
held to.

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
  would simply be absent from the file it writes, invisible because a smaller
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
  caches in the registry (no WMI query, no spawned `powershell`) and counts
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
  carrying it, so the nightly catalog rebuild stops committing rather than
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
  2 GB card and a real 2 GiB one, so no threshold separates the three. A
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
