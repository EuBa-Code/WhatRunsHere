#!/usr/bin/env python3
"""Build WhatLLM's model catalog from HuggingFace.

Deliberately not a scraper. Scraping the HuggingFace index yields tens of
thousands of entries, most of them broken re-uploads and merge experiments, and
no amount of ranking recovers a useful list from that. This reads a curated set
of model ids instead, and for each one pulls two things that cannot be guessed:

  * the exact architecture, from ``config.json`` -- layer count, head counts,
    head width, sliding-window pattern, mixture-of-experts layout;
  * the real size of every published GGUF build, from the file listing.

The second is the point. A computed size is accurate to about a percent; a real
file size is exact, and WhatLLM prefers it wherever it exists.

Usage::

    python tools/build_catalog.py                 # write catalog/catalog.json
    python tools/build_catalog.py --check         # validate without writing
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request
from datetime import date
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "catalog"
SOURCES = CATALOG_DIR / "sources.json"
OUTPUT = CATALOG_DIR / "catalog.json"

USER_AGENT = {"User-Agent": "whatllm-catalog/0.1 (+https://github.com/eugeniobarberini/whatllm)"}
CATALOG_VERSION = 1

# A rebuild reads the whole catalog from a network that is sometimes unwell.
# One or two models can genuinely lose their config.json; a dozen at once is an
# outage upstream, and writing that result would shrink the catalog quietly and
# permanently. A failed entry is therefore carried forward from the previous
# build rather than dropped, and past these limits the rebuild writes nothing.
#
# The hazard is not hypothetical. llmfit records a scrape that dropped
# architecture metadata for 1,764 models in a single run; the guard it added
# afterwards is what this imitates.
RESCUE_LIMIT = 5
BUILD_LOSS_LIMIT = 0.15

# Quantization names WhatLLM models. Builds outside this set are skipped rather
# than guessed at.
KNOWN_QUANTS = {
    "F16", "BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
    "MXFP4", "IQ4_XS", "Q3_K_L", "Q3_K_M", "Q3_K_S", "IQ3_S", "IQ3_XXS",
    "Q2_K", "IQ2_XS", "IQ2_XXS",
}

# Repositories ship companions beside the weights, and they carry the same
# quantization suffix: `mmproj-` is a multimodal projector (the vision encoder),
# `mtp-` a multi-token-prediction head, `draft-` a speculative-decoding draft.
# Recorded as builds, they send someone downloading "the F16" to a 122 MB vision
# encoder, and they poison any check that compares a computed size to a real one.
AUXILIARY_PREFIXES = ("mmproj-", "mtp-", "draft-", "lora-", "adapter-")

# Bits per weight outside this band is not a quantization of the model it is
# filed under. This is the net a prefix list cannot be: it catches shards, wrong
# files, and companions nobody has thought of a name for yet.
MIN_BITS_PER_WEIGHT = 1.0
MAX_BITS_PER_WEIGHT = 17.5

# Share of a repository's candidate files that must land inside that band.
#
# A publisher does not upload eleven broken files and one good one. When most
# of a repository disagrees with the architecture derived from `config.json`,
# it is the architecture that is wrong, and the survivors are the few small
# enough to squeak through rather than evidence of anything. Keeping them would
# leave an entry that looks valid and sizes every future question wrongly.
MIN_BUILD_ACCEPTANCE = 2 / 3

# Nominal bits per weight, used for one job only: deciding whether a
# repository's files store the vocabulary tensor once or twice. The two
# spellings differ by a whole vocabulary tensor -- twenty percent of a small
# model -- so a coarse figure separates them easily. The precise table lives in
# the Rust model and is not duplicated here.
NOMINAL_BPW = {
    "BF16": 16.0, "F16": 16.0, "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7,
    "Q5_K_S": 5.5, "Q4_K_M": 4.9, "Q4_K_S": 4.6, "IQ4_XS": 4.3,
    "Q3_K_L": 4.2, "Q3_K_M": 3.9, "Q3_K_S": 3.5, "Q2_K": 3.0, "MXFP4": 4.25,
}

QUANT_IN_FILENAME = re.compile(
    r"[-.](BF16|F16|MXFP4(?:_MOE)?|IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d)"
    r"(?:-f(?:p)?(?:16|32))?\.gguf$",
    re.IGNORECASE,
)


class CatalogError(RuntimeError):
    """A model could not be described accurately enough to publish."""


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def language_model_config(config: dict) -> dict:
    """Unwrap the text config from a multimodal wrapper."""
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(config.get(key), dict) and "num_hidden_layers" in config[key]:
            return config[key]
    return config


def attention_kind(config: dict) -> dict:
    """Describe how this architecture carries state between tokens."""
    heads = config["num_attention_heads"]
    hidden = config["hidden_size"]

    # DeepSeek-style latent attention, which caches a low-rank projection
    # instead of full keys and values.
    if config.get("kv_lora_rank"):
        return {
            "kind": "latent",
            "q_lora_rank": config.get("q_lora_rank"),
            "kv_lora_rank": config["kv_lora_rank"],
            "qk_nope_head_dim": config.get("qk_nope_head_dim", 128),
            "qk_rope_head_dim": config.get("qk_rope_head_dim", 64),
            "v_head_dim": config.get("v_head_dim", 128),
        }

    return {
        "kind": "grouped",
        "kv_heads": config.get("num_key_value_heads", heads),
        "head_dim": config.get("head_dim") or hidden // heads,
    }


def layer_layout(config: dict) -> dict:
    """Build the repeating layer cycle, including sliding-window patterns."""
    n_layers = config["num_hidden_layers"]
    kind = attention_kind(config)

    window = config.get("sliding_window")
    # Several architectures carry a window value but disable it.
    if config.get("use_sliding_window") is False:
        window = None
    # Gemma 3 names the period; Gemma 2 alternates every other layer.
    period = config.get("sliding_window_pattern") or config.get("layer_types_period")

    if window and period and period > 1:
        cycle = [{"attention": kind, "window": window} for _ in range(period - 1)]
        cycle.append({"attention": kind})
        return {"n_layers": n_layers, "cycle": cycle}

    # An explicit per-layer list, as newer configs provide.
    layer_types = config.get("layer_types")
    if window and isinstance(layer_types, list) and layer_types:
        cycle = [
            {"attention": kind, "window": window}
            if str(t).startswith("sliding")
            else {"attention": kind}
            for t in layer_types
        ]
        return {"n_layers": n_layers, "cycle": cycle}

    if window:
        return {"n_layers": n_layers, "cycle": [{"attention": kind, "window": window}]}

    return {"n_layers": n_layers, "cycle": [{"attention": kind}]}


def moe_spec(config: dict) -> dict | None:
    """Mixture-of-experts layout, when the model is sparse."""
    experts = (
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
    )
    if not experts:
        return None

    per_token = (
        config.get("num_experts_per_tok")
        or config.get("experts_per_token")
        or config.get("n_experts_per_tok")
        or 2
    )
    expert_intermediate = (
        config.get("moe_intermediate_size")
        or config.get("expert_intermediate_size")
        or config["intermediate_size"]
    )
    shared_count = config.get("n_shared_experts") or (
        1 if config.get("shared_expert_intermediate_size") else 0
    )
    shared_intermediate = config.get("shared_expert_intermediate_size") or (
        expert_intermediate if shared_count else 0
    )
    return {
        "experts": experts,
        "experts_per_token": per_token,
        "expert_intermediate": expert_intermediate,
        "shared_experts": shared_count,
        "shared_intermediate": shared_intermediate,
        # DeepSeek keeps a dense prefix; most designs do not.
        "dense_layers": config.get("first_k_dense_replace", 0),
    }


# Architectures whose config class defaults ``tie_word_embeddings`` to true.
# The field being absent does not mean false: it means "use the model class
# default", and that default differs by family. Gemma ties and omits the field;
# Llama does not tie and omits it just as often.
TIES_BY_DEFAULT = ("gemma",)


def tied_embeddings(config: dict, outer: dict) -> bool:
    """Whether the output projection shares storage with the embedding table."""
    for source in (config, outer):
        if "tie_word_embeddings" in source:
            return bool(source["tie_word_embeddings"])
    model_type = str(config.get("model_type", outer.get("model_type", ""))).lower()
    return model_type.startswith(TIES_BY_DEFAULT)


def norms_per_layer(config: dict) -> int:
    """Gemma normalises before and after both sub-blocks; most models do not."""
    model_type = str(config.get("model_type", "")).lower()
    return 4 if "gemma" in model_type else 2


NATIVE_QUANT_NAMES = {"mxfp4": "MXFP4"}


def native_expert_quant(config: dict) -> str | None:
    """The format this model's experts were trained in, if any.

    ``quantization_config.modules_to_not_convert`` lists what stays in full
    precision; when the experts are absent from that list, they are the part
    that was quantized, and no repack changes them back.
    """
    quantization = config.get("quantization_config") or {}
    method = str(quantization.get("quant_method", "")).lower()
    name = NATIVE_QUANT_NAMES.get(method)
    if not name:
        return None
    untouched = " ".join(quantization.get("modules_to_not_convert") or [])
    if "expert" in untouched.lower() or "mlp\"" in untouched.lower():
        return None
    return name


def architecture(outer: dict) -> dict:
    config = language_model_config(outer)
    for required in ("hidden_size", "num_attention_heads", "num_hidden_layers", "vocab_size"):
        if required not in config:
            raise CatalogError(f"config is missing `{required}`")

    # A hybrid stack mixes blocks that are not attention in among the ones that
    # are: linear attention in Qwen3.5, state-space blocks in Jamba and
    # Falcon-H1, short convolutions in LFM2. The memory model carries them --
    # see `AttentionKind::Recurrent` -- but only when told how many parameters
    # and how much state each block holds, and neither follows from
    # `config.json`. Describing one as though every layer were ordinary
    # attention gets its size wrong by a sixth.
    HYBRID_BLOCKS = ("linear", "mamba", "ssm", "conv", "recurrent")
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list) and any(
        any(marker in str(kind).lower() for marker in HYBRID_BLOCKS)
        for kind in layer_types
    ):
        kinds = sorted({str(k) for k in layer_types})
        raise CatalogError(
            f"hybrid stack ({', '.join(kinds)}); the per-block parameter count "
            "is not derivable from the config"
        )

    # Some designs vary a dimension per layer -- Gemma 3n's MatFormer gives each
    # layer its own feed-forward width. WhatLLM's architecture model carries one
    # value, so describing such a model here would mean averaging and calling
    # the result exact. Refusing is the honest option: a catalog that omits a
    # model is better than one that mis-sizes it.
    for field in ("hidden_size", "intermediate_size", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "vocab_size"):
        if isinstance(config.get(field), (list, tuple)):
            raise CatalogError(
                f"`{field}` varies per layer; this architecture is not modelled"
            )

    moe = moe_spec(config)
    arch = {
        "hidden_size": config["hidden_size"],
        "heads": config["num_attention_heads"],
        "intermediate_size": config["intermediate_size"],
        "vocab_size": config["vocab_size"],
        "layers": layer_layout(config),
        "ffn": "gated",
        "tied_embeddings": tied_embeddings(config, outer),
        "norms_per_layer": norms_per_layer(config),
        # Gemma 2 and a few others cap attention logits, which rules out flash
        # attention in llama.cpp and forces the quadratic score matrix.
        "softcapped_attention": bool(config.get("attn_logit_softcapping")),
        # A model trained quantized keeps its experts in that format through
        # every published build; only the tensors listed as not-converted change.
        "native_expert_quant": native_expert_quant(outer),
        "max_context": config.get("max_position_embeddings", 8192),
    }
    if moe:
        arch["moe"] = moe
    return arch


def approximate_params(arch: dict) -> tuple[int, int]:
    """Plausible range of stored parameters, for sanity-checking a file size.

    A range rather than a number, because a tied model may or may not have its
    shared vocabulary tensor written out twice: bartowski's bf16 build of
    Qwen3-0.6B does, which puts a perfectly good file at 20 bits per weight
    against the tied count. Checking the upper bound against the larger figure
    and the lower bound against the smaller admits both spellings while still
    rejecting a file that is not this model at all.

    Deliberately approximate. The exact count lives in the Rust model; this only
    has to tell a quantization of this model from a vision encoder filed beside
    it.
    """
    hidden = arch["hidden_size"]
    layers = arch["layers"]["n_layers"]
    heads = arch["heads"]
    vocab = arch["vocab_size"]
    spec = arch["layers"]["cycle"][0]["attention"]
    head_dim = spec.get("head_dim", hidden // max(heads, 1))
    kv_heads = spec.get("kv_heads", heads)
    attention = layers * (2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim)
    moe = arch.get("moe")
    if moe:
        ffn = layers * moe["experts"] * 3 * hidden * moe["expert_intermediate"]
    else:
        ffn = layers * 3 * hidden * arch["intermediate_size"]
    body = attention + ffn
    return body + vocab * hidden, body + vocab * hidden * 2


def observe_shape(arch: dict, builds: list) -> tuple[bool, str] | None:
    """Read two structural choices off the published files.

    Neither is reliably stated in the metadata, and both change a size
    materially:

    * **Vocabulary storage.** ``tie_word_embeddings`` says the *weights* are
      shared. It does not say what the GGUF converter wrote. Qwen3 0.6B, 1.7B
      and Qwen3.5-2B are all tied and all ship the tensor written twice; Qwen3
      4B, equally tied, ships it once. On a 0.6B that is a fifth of the file.
    * **Feed-forward shape.** A gated block holds three matrices, a plain one
      two. Assuming gated for a model that is not overstates its feed-forward by
      half, which on an 8B is a third of the whole file.

    So rather than guess, try the four arrangements and return whichever the
    measured sizes actually fit. ``None`` when there is not enough evidence.
    """
    body_without_ffn, ffn_unit, vocab = _param_terms(arch)
    if not vocab or len(builds) < 3:
        return None
    # The comparison below assumes one bits-per-weight across the whole model.
    # For a model whose experts stay in their native format whatever the build
    # is called, that assumption is false, and the detector would "correct" a
    # correct architecture to make its own arithmetic work.
    if arch.get("native_expert_quant"):
        return None

    best = None
    for tied in (True, False):
        for matrices, ffn in ((3, "gated"), (2, "standard")):
            body = body_without_ffn + ffn_unit * matrices
            vocab_tensors = 1 if tied else 2
            error = 0.0
            counted = 0
            for build in builds:
                bpw = NOMINAL_BPW.get(build["quant"])
                if not bpw:
                    continue
                predicted = (body + vocab * vocab_tensors) * bpw / 8
                error += abs(predicted - build["bytes"]) / build["bytes"]
                counted += 1
            if counted < 3:
                return None
            mean = error / counted
            if best is None or mean < best[0]:
                best = (mean, tied, ffn)
    return (best[1], best[2]) if best else None


def _param_terms(arch: dict) -> tuple[int, int, int]:
    """Parameters outside the feed-forward, one feed-forward matrix, one
    vocabulary tensor."""
    hidden = arch["hidden_size"]
    layers = arch["layers"]["n_layers"]
    heads = arch["heads"]
    spec = arch["layers"]["cycle"][0]["attention"]
    head_dim = spec.get("head_dim", hidden // max(heads, 1))
    kv_heads = spec.get("kv_heads", heads)
    attention = layers * (2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim)
    moe = arch.get("moe")
    if moe:
        ffn_unit = layers * moe["experts"] * hidden * moe["expert_intermediate"]
    else:
        ffn_unit = layers * hidden * arch["intermediate_size"]
    return attention + hidden * (2 * layers + 1), ffn_unit, arch["vocab_size"] * hidden


def gguf_builds(repo: str, arch: dict) -> tuple[list[dict], int, int]:
    """Every published GGUF build in a repository, with its real size.

    Returns the builds along with how many candidate files were considered and
    how many survived the plausibility band, so the caller can tell a repository
    with one odd file from an architecture that is simply wrong.
    """
    info = fetch_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
    params_min, params_max = approximate_params(arch)
    builds: dict[str, dict] = {}
    considered = 0
    accepted = 0
    for sibling in info.get("siblings", []):
        filename = sibling.get("rfilename", "")
        size = sibling.get("size")
        if not size:
            continue
        stem = filename.rsplit("/", 1)[-1].lower()
        if stem.startswith(AUXILIARY_PREFIXES):
            continue
        # Multi-part builds report only the first shard's size; skip them
        # rather than publish a size that is a fraction of the real one.
        if re.search(r"-0000\d-of-0000\d\.gguf$", filename):
            continue
        match = QUANT_IN_FILENAME.search(filename)
        if not match:
            continue
        quant = match.group(1).upper().removesuffix("_MOE")
        if quant not in KNOWN_QUANTS:
            continue
        # A file whose size makes no sense against the parameter count is not
        # a build of this model, whatever it is called.
        considered += 1
        low = size * 8 / params_max if params_max else 0
        high = size * 8 / params_min if params_min else 0
        if low < MIN_BITS_PER_WEIGHT or high > MAX_BITS_PER_WEIGHT:
            print(
                f"    ! skipped {filename}: {low:.2f}-{high:.2f} bits/weight",
                file=sys.stderr,
            )
            continue
        accepted += 1
        # Keep the smallest when a repo publishes variants of one name.
        if quant not in builds or size < builds[quant]["bytes"]:
            builds[quant] = {"quant": quant, "repo": repo, "file": filename, "bytes": size}
    return sorted(builds.values(), key=lambda b: -b["bytes"]), considered, accepted


def build_entry(source: dict) -> dict:
    config = fetch_json(f"https://huggingface.co/{source['id']}/resolve/main/config.json")
    entry = {
        "id": source["id"],
        "display_name": source["display_name"],
        "family": source["family"],
        "architecture": architecture(config),
        "benchmarks": source.get("benchmarks", {}),
        "builds": [],
        "license": source.get("license"),
        "released": source.get("released"),
    }
    considered = accepted = 0
    for repo in source.get("gguf_repos", []):
        try:
            found, seen, kept = gguf_builds(repo, entry["architecture"])
        except urllib.error.HTTPError as error:
            print(f"    ! {repo}: HTTP {error.code}", file=sys.stderr)
            continue
        entry["builds"].extend(found)
        considered += seen
        accepted += kept
    # One build per quantization; prefer whichever repo was listed first.
    seen: dict[str, dict] = {}
    for build in entry["builds"]:
        seen.setdefault(build["quant"], build)
    entry["builds"] = sorted(seen.values(), key=lambda b: -b["bytes"])

    # Let the files correct the config where the config does not say.
    observed = observe_shape(entry["architecture"], entry["builds"])
    if observed is not None:
        tied, ffn = observed
        arch = entry["architecture"]
        if tied != arch["tied_embeddings"]:
            print(
                f"    files store the vocabulary tensor "
                f"{'once' if tied else 'twice'} (config said "
                f"tie_word_embeddings={arch['tied_embeddings']})",
                file=sys.stderr,
            )
            arch["tied_embeddings"] = tied
        if ffn != arch["ffn"]:
            print(f"    files show a {ffn} feed-forward block", file=sys.stderr)
            arch["ffn"] = ffn

    # The published files are the ground truth. If a repository full of builds
    # produced none this model could plausibly own, the architecture derived
    # above is wrong, and shipping it would mean sizing every future question
    # about this model against a description that disagrees with reality.
    if considered and accepted < considered * MIN_BUILD_ACCEPTANCE:
        raise CatalogError(
            f"only {accepted} of {considered} published builds match the derived "
            "architecture; the architecture is wrong"
        )
    return entry


def load_previous() -> dict:
    """Entries from the catalog already on disk, keyed by model id."""
    if not OUTPUT.exists():
        return {}
    try:
        catalog = json.loads(OUTPUT.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {entry["id"]: entry for entry in catalog.get("models", [])}


def rescue_failures(models: list, previous: dict, failed: list) -> list:
    """Carry forward entries this run could not rebuild.

    Mutates ``models``. Returns the ids rescued, which the caller weighs
    against :data:`RESCUE_LIMIT`.
    """
    rescued = []
    known = {entry["id"] for entry in models}
    for model_id in failed:
        entry = previous.get(model_id)
        if entry is not None and model_id not in known:
            models.append(entry)
            rescued.append(model_id)
    return rescued


def assess_loss(models: list, previous: dict, rescued: list) -> list:
    """Reasons this rebuild should not be written.

    Empty when the rebuild is at least as complete as what it would replace.
    Separated from :func:`main` so the guard can be tested without a network.
    """
    reasons = []
    if len(rescued) > RESCUE_LIMIT:
        reasons.append(
            f"{len(rescued)} models had to be carried forward (limit {RESCUE_LIMIT}); "
            "that is an outage upstream, not that many publishers deleting files"
        )
    total_builds = sum(len(entry.get("builds", [])) for entry in models)
    previous_builds = sum(len(entry.get("builds", [])) for entry in previous.values())
    if previous_builds and total_builds < previous_builds * (1.0 - BUILD_LOSS_LIMIT):
        reasons.append(
            f"measured builds fell from {previous_builds} to {total_builds} "
            f"(limit {BUILD_LOSS_LIMIT:.0%})"
        )
    lost = sorted(set(previous) - {entry["id"] for entry in models})
    if lost:
        reasons.append("models disappeared entirely: " + ", ".join(lost))
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without writing")
    parser.add_argument("--only", help="build a single model id")
    parser.add_argument(
        "--allow-loss",
        action="store_true",
        help="write even when the rebuild lost models or builds",
    )
    args = parser.parse_args()

    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    if args.only:
        sources = [s for s in sources if s["id"] == args.only]
        if not sources:
            print(f"no source named {args.only}", file=sys.stderr)
            return 1

    # Two kinds of failure, which deserve opposite treatment. A network that
    # would not answer is transient: the previous entry is still the best
    # description available and is carried forward. A model this builder cannot
    # describe correctly is permanent: carrying its old entry forward would keep
    # shipping a description already known to be wrong.
    models, unreachable, rejected, excluded = [], [], [], []
    for source in sources:
        # A model can be excluded in place, with the reason beside it, rather
        # than deleted from the list and forgotten about. These are judgement
        # calls a rebuild must not silently re-litigate.
        if source.get("excluded"):
            print(f"  {source['id']}: excluded ({source['excluded']})", flush=True)
            excluded.append(source["id"])
            continue
        print(f"  {source['id']}", flush=True)
        try:
            entry = build_entry(source)
        except CatalogError as error:
            print(f"    ! rejected: {error}", file=sys.stderr)
            rejected.append(source["id"])
            continue
        except (urllib.error.URLError, KeyError) as error:
            print(f"    ! unreachable: {type(error).__name__}: {error}", file=sys.stderr)
            unreachable.append(source["id"])
            continue
        if not entry["builds"]:
            print("    ! no GGUF builds found", file=sys.stderr)
        models.append(entry)

    # Nothing that was in the catalog may leave it because the network had a
    # bad afternoon.
    previous = {} if args.only else load_previous()
    rescued = rescue_failures(models, previous, unreachable)
    # A deliberate removal is not a loss to be guarded against; an accidental
    # one is the whole point of the guard.
    for model_id in rejected + excluded:
        previous.pop(model_id, None)
    models.sort(key=lambda entry: entry["id"])

    catalog = {
        "version": CATALOG_VERSION,
        "generated": date.today().isoformat(),
        "models": models,
    }

    total_builds = sum(len(m["builds"]) for m in models)
    previous_builds = sum(len(m.get("builds", [])) for m in previous.values())
    print(f"\n{len(models)} models, {total_builds} measured builds", file=sys.stderr)
    if rejected:
        print("rejected as not describable: " + ", ".join(rejected), file=sys.stderr)
    if unreachable:
        print("unreachable: " + ", ".join(unreachable), file=sys.stderr)
    if rescued:
        print("carried forward: " + ", ".join(rescued), file=sys.stderr)

    refusals = assess_loss(models, previous, rescued)

    if refusals and not args.allow_loss:
        for reason in refusals:
            print("REFUSING TO WRITE: " + reason, file=sys.stderr)
        print(
            "The catalog on disk is unchanged. Re-run when the network is well, "
            "or pass --allow-loss if the loss is real.",
            file=sys.stderr,
        )
        return 1

    if not args.check:
        CATALOG_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(catalog, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {OUTPUT.relative_to(ROOT)}", file=sys.stderr)
    # A model that could not be reached but was carried forward is not a
    # failure: the catalog is still complete and correct. Only a gap nothing
    # could fill is worth waking anyone for.
    gaps = [model_id for model_id in unreachable if model_id not in rescued]
    if gaps:
        print("no entry at all for: " + ", ".join(gaps), file=sys.stderr)
    return 1 if gaps else 0


if __name__ == "__main__":
    raise SystemExit(main())
