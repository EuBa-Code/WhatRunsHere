//! Architectural description of a transformer.
//!
//! Every other model in this crate is built on top of this one. The reason it is
//! this detailed is that the two questions people actually ask ("will it fit?"
//! and "how fast will it be?") are decided by tensor shapes, not by a parameter
//! count. Two 30B models can differ by an order of magnitude in KV cache size,
//! and a 671B model can have a *smaller* cache than a 70B one. A single
//! `parameter_count` field cannot express that; this module can.

use serde::{Deserialize, Serialize};

/// How one layer stores the state it carries forward between tokens.
///
/// This is the single biggest source of divergence between model families, and
/// the thing most sizing tools get wrong: they assume every layer caches
/// `2 * heads * head_dim` elements per token, which is true only for classic
/// multi-head attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttentionKind {
    /// Multi-head, grouped-query and multi-query attention, which differ only in
    /// how many key/value heads they keep.
    ///
    /// `kv_heads == heads` is MHA, `1 < kv_heads < heads` is GQA, and
    /// `kv_heads == 1` is MQA.
    Grouped {
        /// Number of key/value heads actually stored.
        kv_heads: u32,
        /// Width of one head. Not always `hidden_size / heads`.
        head_dim: u32,
    },

    /// Multi-head latent attention, as used by `DeepSeek` V2/V3.
    ///
    /// Keys and values are projected down to a shared low-rank latent that is
    /// what actually gets cached, alongside a small decoupled `RoPE` component.
    /// The cache is therefore `kv_lora_rank + qk_rope_head_dim` elements per
    /// layer per token, independent of head count.
    Latent {
        /// Rank of the query down-projection, when the model uses one.
        q_lora_rank: Option<u32>,
        /// Rank of the shared key/value latent. This is what is cached.
        kv_lora_rank: u32,
        /// Per-head width of the non-positional part of the query/key.
        qk_nope_head_dim: u32,
        /// Per-head width of the decoupled `RoPE` part. Also cached.
        qk_rope_head_dim: u32,
        /// Per-head width of the value.
        v_head_dim: u32,
    },

    /// Linear attention or a state-space block, as used by hybrid models such as
    /// Qwen3-Next, Jamba and Falcon-H1.
    ///
    /// The distinguishing property is that the carried state has a fixed size:
    /// context length does not enlarge it at all.
    Recurrent {
        /// Scalar elements of recurrent state, constant in context length.
        state_elems: u64,
        /// Weight parameters in the block, which vary too much by design to be
        /// derived from `hidden_size` alone.
        params: u64,
    },
}

impl AttentionKind {
    /// Scalar elements this layer appends to its cache for each new token.
    ///
    /// Zero for [`AttentionKind::Recurrent`], whose state is constant instead.
    pub const fn cache_elems_per_token(self) -> u64 {
        match self {
            Self::Grouped { kv_heads, head_dim } => 2 * kv_heads as u64 * head_dim as u64,
            Self::Latent {
                kv_lora_rank,
                qk_rope_head_dim,
                ..
            } => kv_lora_rank as u64 + qk_rope_head_dim as u64,
            Self::Recurrent { .. } => 0,
        }
    }

    /// Scalar elements of state this layer holds regardless of context length.
    pub const fn constant_state_elems(self) -> u64 {
        match self {
            Self::Recurrent { state_elems, .. } => state_elems,
            _ => 0,
        }
    }
}

/// One layer of the network: how it attends, and how far back it can see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSpec {
    /// The attention mechanism this layer uses.
    pub attention: AttentionKind,
    /// Sliding-window span in tokens, if the layer only attends locally.
    ///
    /// Gemma 2/3 and Mistral use this to stop the cache growing with context:
    /// a windowed layer never stores more than `window` tokens no matter how
    /// long the conversation gets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<u32>,
}

impl LayerSpec {
    /// A globally-attending layer with the given mechanism.
    pub const fn global(attention: AttentionKind) -> Self {
        Self {
            attention,
            window: None,
        }
    }

    /// A layer that only attends to the last `window` tokens.
    pub const fn windowed(attention: AttentionKind, window: u32) -> Self {
        Self {
            attention,
            window: Some(window),
        }
    }

    /// How many tokens this layer actually keeps at the given context length.
    pub fn cached_tokens(self, context: u32) -> u32 {
        match self.window {
            Some(w) => context.min(w),
            None => context,
        }
    }

    /// Total scalar elements this layer holds at the given context length.
    pub fn cache_elems(self, context: u32) -> u64 {
        self.attention.cache_elems_per_token() * u64::from(self.cached_tokens(context))
            + self.attention.constant_state_elems()
    }
}

/// The layer stack, stored as a repeating cycle rather than an explicit list.
///
/// Real architectures are periodic: Gemma 3 repeats five local layers then one
/// global, Qwen3-Next repeats three linear layers then one full-attention layer.
/// Encoding the period keeps the catalog small and makes the pattern legible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerLayout {
    /// Total number of layers in the stack.
    pub n_layers: u32,
    /// The repeating unit. A single entry means every layer is identical.
    pub cycle: Vec<LayerSpec>,
}

impl LayerLayout {
    /// A stack in which every layer is the same.
    pub fn uniform(n_layers: u32, spec: LayerSpec) -> Self {
        Self {
            n_layers,
            cycle: vec![spec],
        }
    }

    /// A stack built from a repeating pattern of layers.
    pub fn cycling(n_layers: u32, cycle: Vec<LayerSpec>) -> Self {
        Self { n_layers, cycle }
    }

    /// The spec for layer `index`, or `None` if the cycle is empty.
    pub fn layer(&self, index: u32) -> Option<LayerSpec> {
        if self.cycle.is_empty() {
            return None;
        }
        self.cycle.get((index as usize) % self.cycle.len()).copied()
    }

    /// Every layer in the stack, in order.
    pub fn iter(&self) -> impl Iterator<Item = LayerSpec> + '_ {
        (0..self.n_layers).filter_map(move |i| self.layer(i))
    }

    /// Scalar elements held across the whole stack at the given context length.
    pub fn cache_elems(&self, context: u32) -> u64 {
        self.iter().map(|l| l.cache_elems(context)).sum()
    }

    /// True when at least one layer keeps a bounded window or constant state,
    /// meaning the cache does not grow linearly with context forever.
    pub fn has_bounded_layers(&self) -> bool {
        self.iter()
            .any(|l| l.window.is_some() || l.attention.constant_state_elems() > 0)
    }
}

/// Shape of the feed-forward block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnKind {
    /// Gated activation (`SwiGLU`, GeGLU): gate, up and down projections.
    Gated,
    /// Classic two-matrix MLP: up and down projections.
    Standard,
}

impl FfnKind {
    /// Number of weight matrices per feed-forward block.
    pub const fn matrices(self) -> u64 {
        match self {
            Self::Gated => 3,
            Self::Standard => 2,
        }
    }
}

/// Mixture-of-experts configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeSpec {
    /// Total routed experts per `MoE` layer.
    pub experts: u32,
    /// Routed experts activated for each token.
    pub experts_per_token: u32,
    /// Inner width of one routed expert.
    pub expert_intermediate: u32,
    /// Always-on shared experts, if the design has them.
    #[serde(default)]
    pub shared_experts: u32,
    /// Inner width of one shared expert.
    #[serde(default)]
    pub shared_intermediate: u32,
    /// Leading layers that stay dense. `DeepSeek` V3 keeps the first three.
    #[serde(default)]
    pub dense_layers: u32,
}

impl MoeSpec {
    /// True when layer `index` is a mixture-of-experts layer.
    pub const fn is_moe_layer(&self, index: u32) -> bool {
        index >= self.dense_layers
    }
}

const fn default_norms_per_layer() -> u32 {
    2
}

/// Everything needed to compute a model's memory footprint and per-token
/// traffic exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Architecture {
    /// Residual stream width.
    pub hidden_size: u32,
    /// Number of query heads.
    pub heads: u32,
    /// Inner width of a dense feed-forward block.
    pub intermediate_size: u32,
    /// Vocabulary size. Large vocabularies dominate small models.
    pub vocab_size: u32,
    /// The layer stack.
    pub layers: LayerLayout,
    /// Feed-forward shape.
    pub ffn: FfnKind,
    /// Whether the output projection shares storage with the embedding table.
    #[serde(default)]
    pub tied_embeddings: bool,
    /// Mixture-of-experts configuration, when the model is sparse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<MoeSpec>,
    /// Normalisation tensors per layer. Two for most designs, four for Gemma 3.
    #[serde(default = "default_norms_per_layer")]
    pub norms_per_layer: u32,
    /// Whether the model applies a soft cap to its attention logits.
    ///
    /// Gemma 2 does, and it is not a detail. Flash-attention kernels cannot
    /// apply the cap, so llama.cpp falls back to materialising the full score
    /// matrix for these models, a term quadratic in context that turns a
    /// comfortable fit into an out-of-memory, and costs a large amount of
    /// speed besides.
    ///
    /// Found by fitting the decode model to community measurements: on an
    /// RTX 2080, sixteen models landed within a few percent of one line and
    /// Gemma 2 9B ran at a third of the speed its size predicts.
    #[serde(default)]
    pub softcapped_attention: bool,
    /// Longest context the model was trained or extended to serve.
    pub max_context: u32,
    /// The format this model's experts are always stored in, when it was
    /// trained quantized rather than quantized afterwards.
    ///
    /// gpt-oss is the case that forces this. Its experts are MXFP4 in the
    /// original weights, and every published build leaves them there: only the
    /// attention, router and vocabulary tensors (nine percent of the model)
    /// change between builds. Its `F16`, `Q8_0` and `Q6_K` files are 13.79, 12.11
    /// and 12.04 GB, differences a whole-model quantization cannot produce.
    /// Treating
    /// its "F16" as sixteen bits throughout overstates the file threefold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_expert_quant: Option<String>,
}

/// Where a model's stored parameters live, by tensor class.
///
/// Split this way because quantization is not uniform: llama.cpp keeps
/// embeddings and the output projection at higher precision than the body, so
/// a single bits-per-weight number misestimates exactly the models where it
/// matters most.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParamBreakdown {
    /// Token embedding table.
    pub embedding: u64,
    /// Output projection, or zero when tied to the embedding table.
    pub lm_head: u64,
    /// Attention projections across all layers.
    pub attention: u64,
    /// Dense feed-forward blocks across all layers.
    pub dense_ffn: u64,
    /// Every routed and shared expert, whether or not it is activated.
    pub expert_ffn: u64,
    /// Expert routing gates.
    pub router: u64,
    /// Normalisation tensors.
    pub norms: u64,
}

impl ParamBreakdown {
    /// Total stored parameters.
    pub const fn total(&self) -> u64 {
        self.embedding
            + self.lm_head
            + self.attention
            + self.dense_ffn
            + self.expert_ffn
            + self.router
            + self.norms
    }

    /// Parameters outside the embedding and output tensors.
    pub const fn body(&self) -> u64 {
        self.attention + self.dense_ffn + self.expert_ffn + self.router + self.norms
    }
}

/// Weights read from memory to produce one token during decode.
///
/// Deliberately different from [`ParamBreakdown`]. The embedding table is stored
/// but barely read: decode looks up a single row. The output projection is read
/// in full every token whether or not it has its own storage. For a small model
/// with a large vocabulary that difference is most of the answer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeTraffic {
    /// Attention projections.
    pub attention: u64,
    /// Dense feed-forward blocks.
    pub dense_ffn: u64,
    /// Only the experts the router actually selects.
    pub expert_ffn: u64,
    /// Routing gates.
    pub router: u64,
    /// Normalisation tensors.
    pub norms: u64,
    /// Output projection over the vocabulary, read in full every token.
    pub lm_head: u64,
}

impl DecodeTraffic {
    /// Total parameters read per generated token.
    pub const fn total(&self) -> u64 {
        self.attention + self.dense_ffn + self.expert_ffn + self.router + self.norms + self.lm_head
    }

    /// Parameters read per token excluding the output projection.
    pub const fn body(&self) -> u64 {
        self.attention + self.dense_ffn + self.expert_ffn + self.router + self.norms
    }
}

/// Reasons an [`Architecture`] cannot be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArchError {
    /// A dimension that must be positive was zero.
    #[error("architecture field `{field}` must be greater than zero")]
    ZeroDimension {
        /// Name of the offending field.
        field: &'static str,
    },
    /// The layer cycle was empty, so no layer could be described.
    #[error("layer layout has an empty cycle")]
    EmptyCycle,
    /// A mixture-of-experts model activated more experts than it has.
    #[error("routes {active} of {total} experts per token")]
    OverRouted {
        /// Experts activated per token.
        active: u32,
        /// Experts available.
        total: u32,
    },
    /// More dense prefix layers than layers.
    #[error("{dense} dense prefix layers in a {total}-layer stack")]
    DensePrefixTooLong {
        /// Dense prefix length.
        dense: u32,
        /// Layers in the stack.
        total: u32,
    },
}

impl Architecture {
    /// Check the architecture describes a model that could exist.
    ///
    /// # Errors
    /// Returns [`ArchError`] describing the first inconsistency found.
    pub fn validate(&self) -> Result<(), ArchError> {
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("heads", self.heads),
            ("vocab_size", self.vocab_size),
            ("max_context", self.max_context),
            ("n_layers", self.layers.n_layers),
        ] {
            if value == 0 {
                return Err(ArchError::ZeroDimension { field });
            }
        }
        if self.layers.cycle.is_empty() {
            return Err(ArchError::EmptyCycle);
        }
        if let Some(moe) = &self.moe {
            if moe.experts_per_token > moe.experts {
                return Err(ArchError::OverRouted {
                    active: moe.experts_per_token,
                    total: moe.experts,
                });
            }
            if moe.dense_layers > self.layers.n_layers {
                return Err(ArchError::DensePrefixTooLong {
                    dense: moe.dense_layers,
                    total: self.layers.n_layers,
                });
            }
        }
        Ok(())
    }

    /// Weight parameters in one layer's attention block.
    fn attention_params(&self, spec: LayerSpec) -> u64 {
        let hidden = u64::from(self.hidden_size);
        let heads = u64::from(self.heads);
        match spec.attention {
            AttentionKind::Grouped { kv_heads, head_dim } => {
                let head_dim = u64::from(head_dim);
                let q_dim = heads * head_dim;
                let kv_dim = u64::from(kv_heads) * head_dim;
                // q, k, v and the output projection.
                hidden * q_dim + 2 * hidden * kv_dim + q_dim * hidden
            }
            AttentionKind::Latent {
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
            } => {
                let kv_rank = u64::from(kv_lora_rank);
                let qk_nope = u64::from(qk_nope_head_dim);
                let qk_rope = u64::from(qk_rope_head_dim);
                let v_dim = u64::from(v_head_dim);
                let q_head = qk_nope + qk_rope;
                let query = match q_lora_rank {
                    Some(rank) => {
                        let rank = u64::from(rank);
                        hidden * rank + rank * heads * q_head
                    }
                    None => hidden * heads * q_head,
                };
                query
                    + hidden * (kv_rank + qk_rope)     // down-projection, cached
                    + kv_rank * heads * (qk_nope + v_dim) // up-projection
                    + heads * v_dim * hidden // output projection
            }
            AttentionKind::Recurrent { params, .. } => params,
        }
    }

    /// Parameters in one dense feed-forward block.
    fn dense_ffn_params(&self) -> u64 {
        self.ffn.matrices() * u64::from(self.hidden_size) * u64::from(self.intermediate_size)
    }

    /// Parameters in the experts of one mixture-of-experts layer, counting
    /// `experts` routed experts plus every shared expert.
    fn expert_layer_params(&self, moe: &MoeSpec, experts: u32) -> u64 {
        let hidden = u64::from(self.hidden_size);
        let matrices = self.ffn.matrices();
        let routed = u64::from(experts) * matrices * hidden * u64::from(moe.expert_intermediate);
        let shared =
            u64::from(moe.shared_experts) * matrices * hidden * u64::from(moe.shared_intermediate);
        routed + shared
    }

    /// Storage footprint in parameters, by tensor class.
    pub fn params(&self) -> ParamBreakdown {
        let hidden = u64::from(self.hidden_size);
        let vocab = u64::from(self.vocab_size);
        let mut out = ParamBreakdown {
            embedding: vocab * hidden,
            lm_head: if self.tied_embeddings {
                0
            } else {
                vocab * hidden
            },
            // The final pre-output normalisation.
            norms: hidden,
            ..ParamBreakdown::default()
        };

        for index in 0..self.layers.n_layers {
            let Some(spec) = self.layers.layer(index) else {
                continue;
            };
            out.attention += self.attention_params(spec);
            out.norms += u64::from(self.norms_per_layer) * hidden;
            match &self.moe {
                Some(moe) if moe.is_moe_layer(index) => {
                    out.expert_ffn += self.expert_layer_params(moe, moe.experts);
                    out.router += hidden * u64::from(moe.experts);
                }
                _ => out.dense_ffn += self.dense_ffn_params(),
            }
        }
        out
    }

    /// Weights read to generate one token, by tensor class.
    pub fn decode_traffic(&self) -> DecodeTraffic {
        let hidden = u64::from(self.hidden_size);
        let vocab = u64::from(self.vocab_size);
        let mut out = DecodeTraffic {
            // Read in full every token, tied or not.
            lm_head: vocab * hidden,
            norms: hidden,
            ..DecodeTraffic::default()
        };

        for index in 0..self.layers.n_layers {
            let Some(spec) = self.layers.layer(index) else {
                continue;
            };
            out.attention += self.attention_params(spec);
            out.norms += u64::from(self.norms_per_layer) * hidden;
            match &self.moe {
                Some(moe) if moe.is_moe_layer(index) => {
                    out.expert_ffn += self.expert_layer_params(moe, moe.experts_per_token);
                    out.router += hidden * u64::from(moe.experts);
                }
                _ => out.dense_ffn += self.dense_ffn_params(),
            }
        }
        out
    }

    /// Total stored parameters. Shorthand for `self.params().total()`.
    pub fn total_params(&self) -> u64 {
        self.params().total()
    }

    /// Parameters read per generated token. Shorthand for
    /// `self.decode_traffic().total()`.
    pub fn active_params(&self) -> u64 {
        self.decode_traffic().total()
    }

    /// True when the model routes to a subset of its experts.
    pub const fn is_sparse(&self) -> bool {
        self.moe.is_some()
    }

    /// Scalar elements of attention state held at the given context length.
    pub fn cache_elems(&self, context: u32) -> u64 {
        self.layers.cache_elems(context)
    }

    /// Scalar elements added to the cache by one more token, at the point where
    /// the context already holds `context` tokens.
    ///
    /// This is not constant: once a windowed layer is saturated it stops
    /// growing, so the marginal cost of context falls in steps.
    pub fn marginal_cache_elems(&self, context: u32) -> u64 {
        self.layers
            .iter()
            .filter(|layer| match layer.window {
                Some(window) => context < window,
                None => true,
            })
            .map(|layer| layer.attention.cache_elems_per_token())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Llama 3.1 8B: plain GQA, untied embeddings. The reference case.
    fn llama_3_1_8b() -> Architecture {
        Architecture {
            hidden_size: 4096,
            heads: 32,
            intermediate_size: 14336,
            vocab_size: 128_256,
            layers: LayerLayout::uniform(
                32,
                LayerSpec::global(AttentionKind::Grouped {
                    kv_heads: 8,
                    head_dim: 128,
                }),
            ),
            ffn: FfnKind::Gated,
            tied_embeddings: false,
            moe: None,
            norms_per_layer: 2,
            softcapped_attention: false,
            native_expert_quant: None,
            max_context: 131_072,
        }
    }

    /// Gemma 3 27B: five sliding-window layers to every global one, tied
    /// embeddings, and a vocabulary large enough to matter.
    fn gemma_3_27b() -> Architecture {
        let local = LayerSpec::windowed(
            AttentionKind::Grouped {
                kv_heads: 16,
                head_dim: 128,
            },
            1024,
        );
        let global = LayerSpec::global(AttentionKind::Grouped {
            kv_heads: 16,
            head_dim: 128,
        });
        Architecture {
            hidden_size: 5376,
            heads: 32,
            intermediate_size: 21504,
            vocab_size: 262_208,
            layers: LayerLayout::cycling(62, vec![local, local, local, local, local, global]),
            ffn: FfnKind::Gated,
            tied_embeddings: true,
            moe: None,
            norms_per_layer: 4,
            softcapped_attention: false,
            native_expert_quant: None,
            max_context: 131_072,
        }
    }

    /// Mixtral 8x7B: the canonical sparse model, with well-known totals.
    fn mixtral_8x7b() -> Architecture {
        Architecture {
            hidden_size: 4096,
            heads: 32,
            intermediate_size: 14336,
            vocab_size: 32_000,
            layers: LayerLayout::uniform(
                32,
                LayerSpec::global(AttentionKind::Grouped {
                    kv_heads: 8,
                    head_dim: 128,
                }),
            ),
            ffn: FfnKind::Gated,
            tied_embeddings: false,
            moe: Some(MoeSpec {
                experts: 8,
                experts_per_token: 2,
                expert_intermediate: 14336,
                shared_experts: 0,
                shared_intermediate: 0,
                dense_layers: 0,
            }),
            norms_per_layer: 2,
            softcapped_attention: false,
            native_expert_quant: None,
            max_context: 32_768,
        }
    }

    /// `DeepSeek` V3: latent attention, 256 experts, a dense prefix.
    fn deepseek_v3() -> Architecture {
        Architecture {
            hidden_size: 7168,
            heads: 128,
            intermediate_size: 18432,
            vocab_size: 129_280,
            layers: LayerLayout::uniform(
                61,
                LayerSpec::global(AttentionKind::Latent {
                    q_lora_rank: Some(1536),
                    kv_lora_rank: 512,
                    qk_nope_head_dim: 128,
                    qk_rope_head_dim: 64,
                    v_head_dim: 128,
                }),
            ),
            ffn: FfnKind::Gated,
            tied_embeddings: false,
            moe: Some(MoeSpec {
                experts: 256,
                experts_per_token: 8,
                expert_intermediate: 2048,
                shared_experts: 1,
                shared_intermediate: 2048,
                dense_layers: 3,
            }),
            norms_per_layer: 2,
            softcapped_attention: false,
            native_expert_quant: None,
            max_context: 163_840,
        }
    }

    /// Assert `actual` is within `tolerance` (as a fraction) of `expected`.
    fn assert_close(actual: u64, expected: u64, tolerance: f64, what: &str) {
        let delta = (actual as f64 - expected as f64).abs() / expected as f64;
        assert!(
            delta <= tolerance,
            "{what}: got {actual}, expected ~{expected} (off by {:.2}%)",
            delta * 100.0
        );
    }

    #[test]
    fn architectures_are_self_consistent() {
        for arch in [llama_3_1_8b(), gemma_3_27b(), mixtral_8x7b(), deepseek_v3()] {
            arch.validate().expect("reference architecture is valid");
        }
    }

    #[test]
    fn llama_8b_totals_match_the_published_count() {
        let arch = llama_3_1_8b();
        assert_close(arch.total_params(), 8_030_000_000, 0.005, "Llama 3.1 8B");
    }

    #[test]
    fn gemma_27b_totals_match_the_published_count() {
        let arch = gemma_3_27b();
        assert_close(arch.total_params(), 27_000_000_000, 0.02, "Gemma 3 27B");
    }

    #[test]
    fn mixtral_totals_and_active_params_match_the_published_counts() {
        let arch = mixtral_8x7b();
        assert_close(arch.total_params(), 46_700_000_000, 0.01, "Mixtral total");
        // The widely quoted figure for Mixtral is 12.9B active parameters.
        assert_close(arch.active_params(), 12_900_000_000, 0.02, "Mixtral active");
    }

    #[test]
    fn deepseek_v3_totals_and_active_params_match_the_published_counts() {
        let arch = deepseek_v3();
        assert_close(arch.total_params(), 671_000_000_000, 0.02, "V3 total");
        assert_close(arch.active_params(), 37_000_000_000, 0.05, "V3 active");
    }

    #[test]
    fn tied_embeddings_are_stored_once_but_still_read_every_token() {
        let arch = gemma_3_27b();
        let params = arch.params();
        assert_eq!(params.lm_head, 0, "tied model stores no separate lm_head");
        assert_eq!(
            arch.decode_traffic().lm_head,
            u64::from(arch.vocab_size) * u64::from(arch.hidden_size),
            "the output projection is still read in full"
        );
    }

    #[test]
    fn latent_attention_caches_far_less_than_grouped_attention() {
        // The headline claim: DeepSeek V3 is ten times the size of Llama 8B and
        // has 61 layers to its 32, yet holds less state per token.
        let v3 = deepseek_v3().cache_elems(1);
        let llama = llama_3_1_8b().cache_elems(1);
        assert!(
            v3 < llama,
            "V3 caches {v3} elems/token, Llama 8B caches {llama}"
        );
        // 61 layers x (512 + 64) elements.
        assert_eq!(v3, 61 * (512 + 64));
    }

    #[test]
    fn sliding_windows_stop_the_cache_growing() {
        let arch = gemma_3_27b();
        let at_8k = arch.cache_elems(8_192);
        let at_128k = arch.cache_elems(131_072);
        // Only the ten global layers keep growing past the 1024-token window,
        // so a sixteenfold context increase must cost far less than sixteenfold.
        let growth = at_128k as f64 / at_8k as f64;
        assert!(
            growth < 12.0,
            "cache grew {growth:.1}x for a 16x context increase"
        );

        // Past the window, local layers contribute nothing to the margin.
        let global_layers = arch.layers.iter().filter(|l| l.window.is_none()).count() as u64;
        assert_eq!(
            arch.marginal_cache_elems(2_048),
            global_layers * 2 * 16 * 128
        );
    }

    #[test]
    fn recurrent_layers_hold_constant_state() {
        let hybrid = Architecture {
            hidden_size: 2048,
            heads: 16,
            intermediate_size: 8192,
            vocab_size: 151_936,
            layers: LayerLayout::cycling(
                48,
                vec![
                    LayerSpec::global(AttentionKind::Recurrent {
                        state_elems: 65_536,
                        params: 12_000_000,
                    }),
                    LayerSpec::global(AttentionKind::Recurrent {
                        state_elems: 65_536,
                        params: 12_000_000,
                    }),
                    LayerSpec::global(AttentionKind::Recurrent {
                        state_elems: 65_536,
                        params: 12_000_000,
                    }),
                    LayerSpec::global(AttentionKind::Grouped {
                        kv_heads: 2,
                        head_dim: 128,
                    }),
                ],
            ),
            ffn: FfnKind::Gated,
            tied_embeddings: false,
            moe: None,
            norms_per_layer: 2,
            softcapped_attention: false,
            native_expert_quant: None,
            max_context: 262_144,
        };
        assert!(hybrid.layers.has_bounded_layers());
        let at_4k = hybrid.cache_elems(4_096);
        let at_64k = hybrid.cache_elems(65_536);
        // Twelve full-attention layers grow; thirty-six recurrent ones do not.
        assert!((at_64k as f64 / at_4k as f64) < 16.0);
    }

    #[test]
    fn over_routed_mixtures_are_rejected() {
        let mut arch = mixtral_8x7b();
        if let Some(moe) = arch.moe.as_mut() {
            moe.experts_per_token = 16;
        }
        assert!(matches!(
            arch.validate(),
            Err(ArchError::OverRouted {
                active: 16,
                total: 8
            })
        ));
    }
}
