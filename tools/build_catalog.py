#!/usr/bin/env python3
"""Build WhatRunsHere's model catalog from HuggingFace.

Deliberately not a scraper. Scraping the HuggingFace index yields tens of
thousands of entries, most of them broken re-uploads and merge experiments, and
no amount of ranking recovers a useful list from that. This reads a curated set
of model ids instead, and for each one pulls two things that cannot be guessed:

  * the exact architecture, from ``config.json`` -- layer count, head counts,
    head width, sliding-window pattern, mixture-of-experts layout;
  * the real size of every published GGUF build, from the file listing.

The second is the point. A computed size is accurate to about a percent; a real
file size is exact, and WhatRunsHere prefers it wherever it exists.

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

USER_AGENT = {"User-Agent": "whatrunshere-catalog/0.1 (+https://github.com/EuBa-Code/WhatRunsHere)"}
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

# Quantization names WhatRunsHere models. Builds outside this set are skipped rather
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


def gated_delta_net(config: dict) -> dict:
    """The linear-attention block of Qwen3-Next and Qwen3.5, as a recurrent
    layer: how many weights it stores and how much state it carries.

    Read off the block's own definition (`Qwen3NextGatedDeltaNet` in
    transformers). Each of its tensors follows from five config keys:

    * `in_proj_qkvz`: hidden x (2 key_dim + 2 value_dim), for q, k, v and z.
    * `in_proj_ba`: hidden x 2 value heads, the per-head beta and decay.
    * `conv1d`: a depthwise convolution over 2 key_dim + value_dim channels,
      `linear_conv_kernel_dim` taps each, no bias.
    * `A_log` and `dt_bias`: one scalar per value head each.
    * the gated norm: one weight per value-head width.
    * `out_proj`: value_dim x hidden.

    The state is one key_dim x value_dim matrix per value head, plus the
    kernel-minus-one columns the convolution keeps. It never grows with
    context, which is the whole point of the block, and it is held in full
    precision whatever the cache format.
    """
    hidden = config["hidden_size"]
    k_heads = config["linear_num_key_heads"]
    k_dim = config["linear_key_head_dim"]
    v_heads = config["linear_num_value_heads"]
    v_dim = config["linear_value_head_dim"]
    kernel = config.get("linear_conv_kernel_dim", 4)
    key_dim = k_heads * k_dim
    value_dim = v_heads * v_dim
    channels = 2 * key_dim + value_dim
    params = (
        hidden * (2 * key_dim + 2 * value_dim)
        + hidden * 2 * v_heads
        + channels * kernel
        + 2 * v_heads
        + v_dim
        + value_dim * hidden
    )
    state = v_heads * k_dim * v_dim + channels * (kernel - 1)
    return {"kind": "recurrent", "state_elems": state, "params": params}


def short_conv(config: dict) -> dict:
    """LFM2's short-convolution block, as a recurrent layer.

    From `Lfm2ShortConv`: an input projection to three times the width (the
    gate, the value and the convolution input), a depthwise convolution of
    `conv_L_cache` taps over the hidden width, and an output projection back.
    `conv_bias` adds a bias to each. The state is the `conv_L_cache - 1`
    columns the convolution keeps, per channel: a few kilobytes, which is why
    these models run on a phone at any context.
    """
    hidden = config["hidden_size"]
    taps = config.get("conv_L_cache", 3)
    bias = bool(config.get("conv_bias", False))
    params = (
        hidden * 3 * hidden
        + hidden * taps
        + hidden * hidden
        + (3 * hidden + hidden + hidden if bias else 0)
    )
    state = hidden * (taps - 1)
    return {"kind": "recurrent", "state_elems": state, "params": params}


def mamba2(config: dict) -> dict:
    """Granite 4's Mamba-2 block, as a recurrent layer.

    From `Mamba2Mixer`, which `GraniteMoeHybridMambaLayer` wraps: the inner
    width is `mamba_expand` times the hidden width, split into `mamba_n_heads`
    heads of `mamba_d_head`. One projection produces the gate, the
    convolution input (inner width plus two state widths per group) and one
    time-step scalar per head; a depthwise convolution of `mamba_d_conv` taps
    runs over that middle part; `A_log`, `D` and `dt_bias` hold one scalar per
    head; a gated norm covers the inner width; an output projection returns
    to the hidden width. The state is one `d_head x d_state` matrix per head,
    plus the convolution's `d_conv - 1` columns.
    """
    hidden = config["hidden_size"]
    inner = config.get("mamba_expand", 2) * hidden
    heads = config["mamba_n_heads"]
    d_state = config["mamba_d_state"]
    d_head = config.get("mamba_d_head", inner // heads)
    groups = config.get("mamba_n_groups", 1)
    d_conv = config.get("mamba_d_conv", 4)
    conv_dim = inner + 2 * groups * d_state
    proj_bias = bool(config.get("mamba_proj_bias", False))
    conv_bias = bool(config.get("mamba_conv_bias", True))
    in_width = inner + conv_dim + heads
    params = (
        hidden * in_width
        + (in_width if proj_bias else 0)
        + conv_dim * d_conv
        + (conv_dim if conv_bias else 0)
        + 3 * heads
        + inner
        + inner * hidden
        + (hidden if proj_bias else 0)
    )
    state = heads * d_head * d_state + conv_dim * (d_conv - 1)
    return {"kind": "recurrent", "state_elems": state, "params": params}


def recurrent_block(kind: str, config: dict) -> dict | None:
    """The recurrent description of one layer kind, when this can derive it."""
    kind = kind.lower()
    if kind == "linear_attention" and "linear_num_value_heads" in config:
        return gated_delta_net(config)
    if kind == "conv" and "conv_L_cache" in config:
        return short_conv(config)
    if kind == "mamba" and "mamba_n_heads" in config:
        return mamba2(config)
    return None


def hybrid_layer_types(config: dict) -> list | None:
    """The per-layer kinds of a hybrid stack, as listed or as implied.

    Qwen3.5 lists `layer_types` outright. Qwen3-Next's config predates the
    list and says only `full_attention_interval`; transformers derives the
    list from it as every interval-th layer full, the rest linear, and so
    does this. A config with neither is not a hybrid.
    """
    listed = config.get("layer_types")
    if isinstance(listed, list) and listed:
        return listed
    attention_at = config.get("full_attn_idxs")
    if isinstance(attention_at, list) and "conv_L_cache" in config:
        at = {int(i) for i in attention_at}
        return [
            "full_attention" if i in at else "conv"
            for i in range(config["num_hidden_layers"])
        ]
    interval = config.get("full_attention_interval")
    if interval and config.get("linear_num_value_heads"):
        return [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(config["num_hidden_layers"])
        ]
    return None


def shortest_period(specs: list) -> list:
    """The repeating unit of a per-layer list, which is the whole list when
    it does not repeat."""
    for period in range(1, len(specs) + 1):
        if all(specs[i] == specs[i % period] for i in range(len(specs))):
            return specs[:period]
    return specs


def layer_layout(config: dict) -> dict:
    """Build the repeating layer cycle, including sliding-window patterns and
    the linear-attention layers of a hybrid stack."""
    n_layers = config["num_hidden_layers"]
    kind = attention_kind(config)

    # A hybrid stack: linear-attention layers carrying constant state between
    # full-attention layers, in the order the config lists them. Qwen3.5
    # repeats three of one then one of the other; the cycle is read off the
    # list rather than assumed. Qwen3-Next gates its attention output without
    # saying so in its config; Qwen3.5 says so.
    layer_types = hybrid_layer_types(config)
    if isinstance(layer_types, list) and any(
        recurrent_block(str(t), config) for t in layer_types
    ):
        full = {"attention": kind}
        gated_by_design = str(config.get("model_type", "")).lower() == "qwen3_next"
        if config.get("attn_output_gate", gated_by_design):
            full["output_gate"] = True
        specs = []
        for t in layer_types:
            block = recurrent_block(str(t), config)
            specs.append({"attention": block} if block else full)
        return {"n_layers": n_layers, "cycle": shortest_period(specs)}

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


def intermediate_size(config: dict) -> int:
    """The dense feed-forward width, which two families compute rather than
    state.

    LFM2 names a nominal width and, when `block_auto_adjust_ff_dim` is set,
    uses two thirds of it, scaled by `block_ffn_dim_multiplier` and rounded
    up to `block_multiple_of`, exactly as `Lfm2MLP` does. Granite 4's hybrid
    config reserves `intermediate_size` for its experts and keeps the dense
    block's width in `shared_intermediate_size`, which for the dense sizes is
    the only feed-forward there is.
    """
    model_type = str(config.get("model_type", "")).lower()
    if model_type == "lfm2":
        width = config.get("intermediate_size") or config["block_ff_dim"]
        if config.get("block_auto_adjust_ff_dim"):
            width = int(2 * width / 3)
            multiplier = config.get("block_ffn_dim_multiplier")
            if multiplier is not None:
                width = int(multiplier * width)
            multiple = config.get("block_multiple_of", 256)
            width = multiple * ((width + multiple - 1) // multiple)
        return width
    if model_type == "granitemoehybrid" and not config.get("num_local_experts"):
        return config.get("shared_intermediate_size") or config["intermediate_size"]
    return config["intermediate_size"]


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
    # Qwen names the shared expert's width one way, Granite 4 another.
    shared_width = config.get("shared_expert_intermediate_size") or (
        config.get("shared_intermediate_size")
        if str(config.get("model_type", "")).lower() == "granitemoehybrid"
        else None
    )
    shared_count = config.get("n_shared_experts") or (1 if shared_width else 0)
    shared_intermediate = shared_width or (expert_intermediate if shared_count else 0)
    return {
        "experts": experts,
        "experts_per_token": per_token,
        "expert_intermediate": expert_intermediate,
        "shared_experts": shared_count,
        "shared_intermediate": shared_intermediate,
        # DeepSeek and LFM2 keep a dense prefix; most designs do not.
        "dense_layers": config.get("first_k_dense_replace")
        or config.get("num_dense_layers")
        or 0,
    }


# Architectures whose config class defaults ``tie_word_embeddings`` to true.
# The field being absent does not mean false: it means "use the model class
# default", and that default differs by family. Gemma ties and omits the field;
# Llama does not tie and omits it just as often.
TIES_BY_DEFAULT = ("gemma", "lfm2")


def tied_embeddings(config: dict, outer: dict) -> bool:
    """Whether the output projection shares storage with the embedding table."""
    for source in (config, outer):
        for key in ("tie_word_embeddings", "tie_embedding"):
            if key in source:
                return bool(source[key])
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
    # are. The memory model carries them as `AttentionKind::Recurrent`, but only
    # when told how many parameters and how much state each block holds. Three
    # blocks are derived from their config keys, each tensor by tensor from
    # the block's definition: Qwen3.5's Gated DeltaNet, LFM2's short
    # convolution and Granite 4's Mamba-2. Any other kind (Jamba, Falcon-H1)
    # is refused, because describing one as though every layer were ordinary
    # attention gets its size wrong by a sixth.
    HYBRID_BLOCKS = ("linear", "mamba", "ssm", "conv", "recurrent")
    layer_types = hybrid_layer_types(config)
    if isinstance(layer_types, list):
        hybrid = sorted({
            str(kind) for kind in layer_types
            if any(marker in str(kind).lower() for marker in HYBRID_BLOCKS)
        })
        derivable = all(recurrent_block(kind, config) for kind in hybrid)
        if hybrid and not derivable:
            kinds = sorted({str(k) for k in layer_types})
            raise CatalogError(
                f"hybrid stack ({', '.join(kinds)}); the per-block parameter count "
                "is not derivable from the config"
            )

    # Some designs vary a dimension per layer -- Gemma 3n's MatFormer gives each
    # layer its own feed-forward width. WhatRunsHere's architecture model carries one
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
        "intermediate_size": intermediate_size(config),
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
    vocab = arch["vocab_size"]
    attention = _attention_params(arch)
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


def _attention_params(arch: dict) -> int:
    """Parameters in every attention or recurrent block of the stack, walking
    the layer cycle as the Rust count does."""
    hidden = arch["hidden_size"]
    heads = arch["heads"]
    cycle = arch["layers"]["cycle"]
    total = 0
    for index in range(arch["layers"]["n_layers"]):
        layer = cycle[index % len(cycle)]
        spec = layer["attention"]
        if spec.get("kind") == "recurrent":
            total += spec["params"]
            continue
        head_dim = spec.get("head_dim", hidden // max(heads, 1))
        kv_heads = spec.get("kv_heads", heads)
        query = hidden * heads * head_dim
        # Query and output projections, a gate beside the query when there is
        # one, and the key and value projections.
        total += query * (3 if layer.get("output_gate") else 2) + 2 * hidden * kv_heads * head_dim
    return total


def _param_terms(arch: dict) -> tuple[int, int, int]:
    """Parameters outside the feed-forward, one feed-forward matrix, one
    vocabulary tensor.

    This counts parameters, which the Rust model also does, exactly and in more
    detail. Two implementations of one thing is how a project ends up applying
    a fix in one of them: llmfit carries a comment about a duplicate throughput
    estimator that silently missed three consecutive mixture-of-experts fixes
    and underestimated sparse models fourfold.

    What makes it acceptable here is that this one cannot be believed on its
    own. Everything it decides is checked afterwards by `size_validation`,
    which uses the exact Rust count against the real file sizes: if this
    approximation drifts, the choice it makes stops matching reality and the
    test fails. It is a proposal, not an authority.
    """
    hidden = arch["hidden_size"]
    layers = arch["layers"]["n_layers"]
    attention = _attention_params(arch)
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


#: Token shapes that must never reach the committed catalog.
#:
#: Repository names, file names and metadata are written by strangers, and a
#: publisher who once pasted a token where a name belonged has put it in the
#: upstream metadata for good. Two things then go wrong at once: the entry is
#: garbage, and GitHub's secret scanning rejects the push that carries it, so
#: a catalog rebuild that swallowed one would break the daily commit rather
#: than merely be wrong. llmfit had exactly that push blocked on 2026-08-03.
SECRET_SHAPES = re.compile(
    r"hf_[A-Za-z0-9]{28,}"                      # HuggingFace access token
    r"|ghp_[A-Za-z0-9]{30,}"                    # GitHub personal access token
    r"|gho_[A-Za-z0-9]{30,}"                    # GitHub OAuth token
    r"|github_pat_[A-Za-z0-9_]{22,}"            # GitHub fine-grained token
    r"|(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{32,}"   # OpenAI-style API key
)


def drop_secret_bearing(models: list) -> tuple[list, list]:
    """Split out entries whose serialised form contains a token shape.

    Returns ``(kept, dropped)``, where the dropped ids have the match itself
    replaced, so a caller can name what it dropped in a log that anyone may
    read without publishing the secret a second time.
    """
    kept, dropped = [], []
    for entry in models:
        if SECRET_SHAPES.search(json.dumps(entry)):
            dropped.append(SECRET_SHAPES.sub("<redacted>", entry["id"]))
        else:
            kept.append(entry)
    return kept, dropped


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

    models, leaked = drop_secret_bearing(models)
    for model_id in leaked:
        print(
            f"dropped an entry carrying a credential-shaped string: {model_id}",
            file=sys.stderr,
        )
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
