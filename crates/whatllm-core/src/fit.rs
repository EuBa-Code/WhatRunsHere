//! Putting a model on a machine, and saying what happens.
//!
//! The solver walks every placement the hardware allows, every weight
//! quantization from best to worst, and every cache quantization, and keeps the
//! configuration that best matches what the user said they wanted. Then it says
//! why — and, more usefully, what to change.
//!
//! Two decisions here differ from the usual approach and are worth defending:
//!
//! **Speed has diminishing returns.** Going from 5 to 15 tokens per second
//! transforms a model from unusable to usable. Going from 80 to 120 changes
//! almost nothing, because both are already faster than anyone reads. Scoring
//! throughput linearly makes a solver prefer a small fast model over a large
//! capable one on the strength of speed nobody will notice. [`speed_utility`]
//! saturates instead.
//!
//! **Headroom is a risk, not a virtue.** Filling a card to 55% is not better
//! than filling it to 75%; how good the model is has already been counted. What
//! matters is that a configuration at 96% will fall over. So utilisation only
//! ever subtracts, and only near the top.

use crate::arch::Architecture;
use crate::hardware::{MemoryPool, PoolKind, SystemProfile};
use crate::memory::{self, LoadConfig, MemoryPlan, RuntimeProfile};
use crate::model::GgufBuild;
use crate::perf::{self, Calibration, Throughput, TrafficSplit};
use crate::quality::{self, Benchmarks, QualityAssessment, UseCase};
use crate::quant::{KvQuant, WeightQuant, WEIGHT_QUANTS};
use serde::{Deserialize, Serialize};

/// What the user is optimising for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preference {
    /// The most capable model that will run at all.
    Quality,
    /// The fastest usable answer.
    Speed,
    /// A sensible compromise.
    #[default]
    Balanced,
}

impl Preference {
    /// Tokens per second at which a configuration is about two-thirds as usable
    /// as an arbitrarily fast one.
    ///
    /// Comfortable reading is roughly seven to ten tokens per second, so the
    /// balanced setting sits just above it: fast enough to read along with,
    /// with no credit for going faster than anyone can follow.
    const fn speed_tolerance(self) -> f64 {
        match self {
            // Willing to wait for a better answer.
            Self::Quality => 5.0,
            Self::Balanced => 8.0,
            // Wants the answer now.
            Self::Speed => 20.0,
        }
    }
}

/// How reachable a model's ability is at a given generation speed, from 0 to 1.
///
/// A multiplier rather than a term added beside quality, and that is the
/// substantive choice in this module. Speed does not make a model cleverer; it
/// decides whether its cleverness is worth waiting for. A capable model at half
/// a token per second is unusable, and a weak one at two hundred tokens per
/// second is still weak.
///
/// Adding the two instead ranks a 0.6B above a 4B on a slow machine, on the
/// strength of speed that has nothing to do with what the 0.6B can do. That is
/// not a tuning problem; it is the wrong shape of model.
pub fn usability(tokens_per_second: f64, preference: Preference) -> f64 {
    if tokens_per_second <= 0.0 {
        return 0.0;
    }
    1.0 - (-tokens_per_second / preference.speed_tolerance()).exp()
}

/// Utilisation above which a configuration starts to look risky.
const RISK_ONSET: f64 = 0.85;
/// Utilisation at which a configuration is treated as certain to fail.
const RISK_CEILING: f64 = 0.98;
/// How much of a configuration's value the worst crowding takes away.
const MAX_CROWDING_DISCOUNT: f64 = 0.35;

/// The share of a configuration's value discounted for running near the edge.
///
/// Only ever subtracts, and only near the top: filling a card to 55% is not
/// better than filling it to 75%, because how good the model is has already
/// been counted. What matters is that a configuration at 96% will fall over.
pub fn crowding_risk(utilisation: f64) -> f64 {
    if utilisation <= RISK_ONSET {
        return 0.0;
    }
    let span = RISK_CEILING - RISK_ONSET;
    ((utilisation - RISK_ONSET) / span).clamp(0.0, 1.0) * MAX_CROWDING_DISCOUNT
}

/// How comfortably a configuration sits in its memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Room to spare. Longer contexts and other work will still fit.
    Comfortable,
    /// Fits with normal margins.
    Fits,
    /// Fits on paper, with little tolerance for anything else running.
    Tight,
    /// Does not fit.
    DoesNotFit,
}

impl Verdict {
    fn from_utilisation(utilisation: f64) -> Self {
        if utilisation > 1.0 {
            Self::DoesNotFit
        } else if utilisation > RISK_ONSET {
            Self::Tight
        } else if utilisation > 0.6 {
            Self::Fits
        } else {
            Self::Comfortable
        }
    }

    /// A label for display.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Comfortable => "Comfortable",
            Self::Fits => "Fits",
            Self::Tight => "Tight",
            Self::DoesNotFit => "Does not fit",
        }
    }

    /// Whether the model runs at all.
    pub const fn runs(self) -> bool {
        !matches!(self, Self::DoesNotFit)
    }
}

/// Where the model ends up running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum RunMode {
    /// Entirely in accelerator memory.
    Accelerated {
        /// How many devices hold it.
        devices: u32,
    },
    /// Entirely in a unified memory pool.
    Unified,
    /// Split: some layers on the accelerator, the rest in system memory.
    ///
    /// Reported separately because the performance cost is severe and
    /// non-obvious — see [`crate::perf`].
    Offloaded {
        /// Layers resident on the accelerator.
        gpu_layers: u32,
        /// Layers in the model.
        total_layers: u32,
    },
    /// Entirely in system memory.
    Cpu,
}

impl RunMode {
    /// A label for display.
    pub fn label(self) -> String {
        match self {
            Self::Accelerated { devices: 1 } => "GPU".to_owned(),
            Self::Accelerated { devices } => format!("GPU x{devices}"),
            Self::Unified => "Unified memory".to_owned(),
            Self::Offloaded {
                gpu_layers,
                total_layers,
            } => format!("Partial offload ({gpu_layers}/{total_layers} layers)"),
            Self::Cpu => "CPU".to_owned(),
        }
    }
}

/// Something worth doing about this configuration.
///
/// The point of the tool is not to hand down a verdict but to say what to
/// change, so these carry the size of the win rather than just naming it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "note", rename_all = "snake_case")]
pub enum FitNote {
    /// A quantized cache would free memory, and possibly buy context.
    QuantiseCache {
        /// The cache format currently assumed.
        from: KvQuant,
        /// The format being suggested.
        to: KvQuant,
        /// Bytes freed.
        saves_bytes: u64,
        /// Context reachable afterwards, when it goes up.
        unlocks_context: Option<u32>,
    },
    /// The requested context does not fit; this is what does.
    ContextCeiling {
        /// What was asked for.
        requested: u32,
        /// What is actually reachable.
        achievable: u32,
    },
    /// Flash attention is off and is costing a large, avoidable allocation.
    EnableFlashAttention {
        /// Bytes the score matrix is currently taking.
        saves_bytes: u64,
    },
    /// A smaller weight format would keep the whole model on the accelerator.
    ///
    /// Almost always the single highest-value change available: a partial
    /// offload can cost more speed than several rungs of quantization.
    QuantiseToAvoidOffload {
        /// The format that fits entirely.
        to: String,
        /// How much faster the fully-resident configuration runs.
        speed_multiplier: f64,
        /// Quality points given up to get there.
        quality_cost: f64,
    },
    /// There is room for a higher-fidelity format than the one chosen.
    RoomForBetterQuant {
        /// The better format.
        to: String,
        /// Quality points gained.
        quality_gain: f64,
    },
    /// The configuration fits, but with little margin.
    RunningClose {
        /// Fraction of the pool consumed.
        utilisation: f64,
    },
}

/// A complete answer for one model on one machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelFit {
    /// The weight format chosen.
    pub quant: WeightQuant,
    /// The cache format chosen.
    pub kv_quant: KvQuant,
    /// Where it runs.
    pub run_mode: RunMode,
    /// The pool it was placed in.
    pub pool: MemoryPool,
    /// The full memory accounting.
    pub memory: MemoryPlan,
    /// Fraction of the pool consumed.
    pub utilisation: f64,
    /// How comfortably it sits.
    pub verdict: Verdict,
    /// Speed, with its basis.
    pub throughput: Throughput,
    /// Quality, with its basis.
    pub quality: QualityAssessment,
    /// The longest context reachable in this placement.
    pub max_context: Option<u32>,
    /// The composite score the solver ranked by.
    pub score: f64,
    /// What to do about it.
    pub notes: Vec<FitNote>,
}

/// What is being asked of the model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitRequest {
    /// What it will be used for.
    pub use_case: UseCase,
    /// Context length wanted, in tokens.
    pub context: u32,
    /// Sequences to serve concurrently.
    pub parallel: u32,
    /// The runtime that will host it.
    pub runtime: RuntimeProfile,
    /// What to optimise for.
    pub preference: Preference,
}

impl Default for FitRequest {
    fn default() -> Self {
        Self {
            use_case: UseCase::General,
            context: 8_192,
            parallel: 1,
            runtime: RuntimeProfile::LLAMA_CPP,
            preference: Preference::Balanced,
        }
    }
}

/// Everything known about the model and the machine.
#[derive(Debug, Clone, Copy)]
pub struct FitContext<'a> {
    /// The model's shape.
    pub arch: &'a Architecture,
    /// Its published evaluations, if any.
    pub benchmarks: &'a Benchmarks,
    /// The machine.
    pub system: &'a SystemProfile,
    /// What the machine can sustain, and how that was established.
    pub calibration: &'a Calibration,
    /// Published builds with measured file sizes.
    ///
    /// When one matches the format under consideration its exact size is used
    /// in place of the computed estimate, which is the difference between an
    /// answer good to a percent and an answer that is simply right.
    pub builds: &'a [GgufBuild],
}

impl<'a> FitContext<'a> {
    /// A context with no measured builds behind it.
    pub fn new(
        arch: &'a Architecture,
        benchmarks: &'a Benchmarks,
        system: &'a SystemProfile,
        calibration: &'a Calibration,
    ) -> Self {
        Self {
            arch,
            benchmarks,
            system,
            calibration,
            builds: &[],
        }
    }

    /// The same context with measured file sizes available.
    #[must_use]
    pub fn with_builds(mut self, builds: &'a [GgufBuild]) -> Self {
        self.builds = builds;
        self
    }

    /// The measured size of a published build of `quant`, if there is one.
    fn measured_bytes(&self, quant: &str) -> Option<u64> {
        self.builds
            .iter()
            .find(|build| build.quant.eq_ignore_ascii_case(quant))
            .map(|build| build.bytes)
    }
}

/// A placement being evaluated, before it is scored.
struct Candidate {
    run_mode: RunMode,
    pool: MemoryPool,
    /// Bytes available to the accelerator in an offloaded placement.
    device_bytes: u64,
    on_accelerator: bool,
    /// Whether device and host memory are the same physical pool.
    ///
    /// They frequently are — every laptop with integrated graphics — and it
    /// matters here because "offload the rest to system RAM" is not a placement
    /// on such a machine. It is the same memory counted twice.
    unified: bool,
}

/// Every placement this machine allows, best first.
fn candidates(system: &SystemProfile) -> Vec<Candidate> {
    let mut out = Vec::new();
    for pool in system.pools() {
        let run_mode = match pool.kind {
            PoolKind::Unified => RunMode::Unified,
            PoolKind::Device => RunMode::Accelerated {
                devices: pool.devices.len() as u32,
            },
            PoolKind::Host => RunMode::Cpu,
        };
        out.push(Candidate {
            run_mode,
            device_bytes: pool.usable_bytes,
            on_accelerator: pool.kind != PoolKind::Host,
            unified: pool.kind == PoolKind::Unified,
            pool,
        });
    }
    out
}

/// Build the load configuration for a placement.
fn load_config(
    ctx: &FitContext,
    request: &FitRequest,
    quant: WeightQuant,
    kv: KvQuant,
) -> LoadConfig {
    LoadConfig {
        context: request.context,
        parallel: request.parallel.max(1),
        ubatch: 512,
        weight_quant: quant,
        kv_quant: kv,
        measured_weight_bytes: ctx.measured_bytes(quant.id),
        runtime: request.runtime,
    }
}

/// The largest number of layers that fits on a device of `device_bytes`,
/// with the rest of the model in system memory.
///
/// Activations, runtime overhead and headroom all sit on the device whatever
/// the split, so a partial offload does not get to shed them.
fn best_offload_layers(
    arch: &Architecture,
    plan: &MemoryPlan,
    device_bytes: u64,
    host_bytes: u64,
) -> Option<u32> {
    let total_layers = arch.layers.n_layers;
    let fixed = plan.activations.total() + plan.runtime_overhead + plan.headroom;
    if fixed >= device_bytes {
        return None;
    }
    let movable = plan.weights + plan.kv_cache;
    let budget = device_bytes - fixed;

    let mut best = None;
    for layers in 1..=total_layers {
        let share = (movable as f64) * f64::from(layers) / f64::from(total_layers);
        let on_device = share.round() as u64;
        if on_device > budget {
            break;
        }
        if movable.saturating_sub(on_device) <= host_bytes {
            best = Some(layers);
        }
    }
    // A "partial offload" that holds every layer is simply an accelerated run;
    // the caller has already considered that placement.
    best.filter(|&layers| layers < total_layers)
}

/// Score a fitted configuration.
fn composite_score(
    preference: Preference,
    quality: &QualityAssessment,
    throughput: &Throughput,
    utilisation: f64,
) -> f64 {
    quality.quantized
        * usability(throughput.decode_tps, preference)
        * (1.0 - crowding_risk(utilisation))
}

/// Evaluate one weight and cache format in one placement.
fn evaluate(
    ctx: &FitContext,
    request: &FitRequest,
    candidate: &Candidate,
    quant: WeightQuant,
    kv: KvQuant,
) -> Option<ModelFit> {
    let cfg = load_config(ctx, request, quant, kv);
    let plan = memory::plan(ctx.arch, &cfg);

    let host_bytes = ctx.system.memory.available_bytes;
    let (run_mode, split, pool_bytes) = if plan.fits_in(candidate.device_bytes) {
        let weights = perf::weight_traffic_bytes(ctx.arch, &quant);
        let cache = memory::kv_cache_bytes(ctx.arch, request.context, kv);
        let split = if candidate.on_accelerator {
            TrafficSplit::all_accelerator(weights, cache)
        } else {
            TrafficSplit::all_host(weights, cache)
        };
        (candidate.run_mode, split, candidate.device_bytes)
    } else if candidate.on_accelerator && !candidate.unified {
        // Too big for the device alone; see whether a split works. Only a
        // discrete device can split: on unified memory there is nowhere else
        // for the remainder to go.
        let layers = best_offload_layers(ctx.arch, &plan, candidate.device_bytes, host_bytes)?;
        let weights = perf::weight_traffic_bytes(ctx.arch, &quant);
        let cache = memory::kv_cache_bytes(ctx.arch, request.context, kv);
        (
            RunMode::Offloaded {
                gpu_layers: layers,
                total_layers: ctx.arch.layers.n_layers,
            },
            TrafficSplit::by_layers(weights, cache, layers, ctx.arch.layers.n_layers),
            candidate.device_bytes.saturating_add(host_bytes),
        )
    } else {
        return None;
    };

    let utilisation = plan.utilisation(pool_bytes);
    if utilisation > 1.0 {
        return None;
    }

    let throughput = perf::throughput(ctx.arch, &quant, ctx.calibration, &split, request.context);
    let quality = quality::assess(
        ctx.benchmarks,
        request.use_case,
        ctx.arch.total_params(),
        ctx.arch.active_params(),
        &quant,
        kv,
    );

    Some(ModelFit {
        quant,
        kv_quant: kv,
        run_mode,
        pool: candidate.pool.clone(),
        memory: plan,
        utilisation,
        verdict: Verdict::from_utilisation(utilisation),
        throughput,
        quality,
        max_context: memory::max_context_for(ctx.arch, &cfg, pool_bytes),
        score: composite_score(request.preference, &quality, &throughput, utilisation),
        notes: Vec::new(),
    })
}

/// The weight formats worth considering for this model.
///
/// When the catalog knows which builds were actually published, only those are
/// offered. Recommending a format nobody uploaded is advice that cannot be
/// acted on, and it happens easily: `MXFP4` exists for gpt-oss and for almost
/// nothing else, yet it sits on the ladder and fits beautifully.
fn candidate_quants(ctx: &FitContext) -> Vec<&'static WeightQuant> {
    if ctx.builds.is_empty() {
        return WEIGHT_QUANTS.iter().collect();
    }
    let published: Vec<&'static WeightQuant> = WEIGHT_QUANTS
        .iter()
        .filter(|quant| {
            ctx.builds
                .iter()
                .any(|build| build.quant.eq_ignore_ascii_case(quant.id))
        })
        .collect();
    // A catalog entry listing only formats this build does not model should
    // fall back to the full ladder rather than produce no answer at all.
    if published.is_empty() {
        WEIGHT_QUANTS.iter().collect()
    } else {
        published
    }
}

/// Find the best way to run this model on this machine.
///
/// Returns `None` only when no combination of placement, weight format and
/// cache format fits at the requested context.
pub fn solve(ctx: &FitContext, request: &FitRequest) -> Option<ModelFit> {
    let mut best: Option<ModelFit> = None;

    let quants = candidate_quants(ctx);
    for candidate in candidates(ctx.system) {
        for quant in &quants {
            for &kv in KvQuant::LADDER {
                let Some(fit) = evaluate(ctx, request, &candidate, **quant, kv) else {
                    continue;
                };
                if best.as_ref().is_none_or(|b| fit.score > b.score) {
                    best = Some(fit);
                }
            }
        }
    }

    let mut best = best?;
    best.notes = advise(ctx, request, &best);
    Some(best)
}

/// What to change about a configuration that already works.
fn advise(ctx: &FitContext, request: &FitRequest, fit: &ModelFit) -> Vec<FitNote> {
    let mut notes = Vec::new();
    let pool_bytes = fit.pool.usable_bytes;

    // The requested context may not have been reachable.
    if let Some(ceiling) = fit.max_context {
        if ceiling < request.context {
            notes.push(FitNote::ContextCeiling {
                requested: request.context,
                achievable: ceiling,
            });
        }
    }

    // A partial offload is the expensive case; check whether stepping down the
    // weight ladder avoids it entirely.
    let quants = candidate_quants(ctx);
    if let RunMode::Offloaded { .. } = fit.run_mode {
        for quant in &quants {
            let cfg = load_config(ctx, request, **quant, fit.kv_quant);
            if !memory::plan(ctx.arch, &cfg).fits_in(pool_bytes) {
                continue;
            }
            let weights = perf::weight_traffic_bytes(ctx.arch, quant);
            let cache = memory::kv_cache_bytes(ctx.arch, request.context, fit.kv_quant);
            let resident = perf::throughput(
                ctx.arch,
                quant,
                ctx.calibration,
                &TrafficSplit::all_accelerator(weights, cache),
                request.context,
            );
            if resident.decode_tps > fit.throughput.decode_tps * 1.15 {
                notes.push(FitNote::QuantiseToAvoidOffload {
                    to: quant.id.to_owned(),
                    speed_multiplier: resident.decode_tps / fit.throughput.decode_tps.max(0.001),
                    quality_cost: quant.quality_cost - fit.quant.quality_cost,
                });
            }
            break;
        }
    }

    // Quantizing the cache is cheap in quality and often buys real context.
    if fit.kv_quant == KvQuant::F16 {
        let target = KvQuant::Q8_0;
        let before = memory::kv_cache_bytes(ctx.arch, request.context, KvQuant::F16);
        let after = memory::kv_cache_bytes(ctx.arch, request.context, target);
        let saved = before.saturating_sub(after) * u64::from(request.parallel.max(1));
        if saved > 256 * 1024 * 1024 {
            let cfg = load_config(ctx, request, fit.quant, target);
            let unlocked = memory::max_context_for(ctx.arch, &cfg, pool_bytes)
                .filter(|&c| fit.max_context.is_none_or(|current| c > current));
            notes.push(FitNote::QuantiseCache {
                from: KvQuant::F16,
                to: target,
                saves_bytes: saved,
                unlocks_context: unlocked,
            });
        }
    }

    // A materialised score matrix is pure waste when it can be avoided.
    if fit.memory.activations.attention_scores > 0 {
        notes.push(FitNote::EnableFlashAttention {
            saves_bytes: fit.memory.activations.attention_scores,
        });
    }

    // If a better format still fits, say so rather than leaving quality behind.
    // The ladder is ordered best first, so the first fitting entry above the
    // chosen one is the best available upgrade.
    let chosen = quants
        .iter()
        .position(|q| q.id == fit.quant.id)
        .unwrap_or(0);
    if let Some(better) = quants[..chosen].iter().find(|q| {
        let cfg = load_config(ctx, request, ***q, fit.kv_quant);
        memory::plan(ctx.arch, &cfg).fits_in(pool_bytes)
    }) {
        notes.push(FitNote::RoomForBetterQuant {
            to: better.id.to_owned(),
            quality_gain: (fit.quant.quality_cost - better.quality_cost)
                * quality::quantization_sensitivity(ctx.arch.total_params()),
        });
    }

    if fit.verdict == Verdict::Tight {
        notes.push(FitNote::RunningClose {
            utilisation: fit.utilisation,
        });
    }

    notes
}

#[cfg(test)]
mod tests {
    // These assert exact zeroes and ones the code produces by construction.
    #![allow(clippy::float_cmp)]

    use super::*;
    use crate::arch::{AttentionKind, FfnKind, LayerLayout, LayerSpec};
    use crate::hardware::{Accelerator, Backend, CpuArch, CpuInfo, HostMemory, Vendor};
    use crate::perf::{Confidence, DeviceThroughput};

    const GIB: u64 = 1024 * 1024 * 1024;

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

    fn llama_3_3_70b() -> Architecture {
        Architecture {
            hidden_size: 8192,
            heads: 64,
            intermediate_size: 28672,
            vocab_size: 128_256,
            layers: LayerLayout::uniform(
                80,
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

    fn machine(vram_gib: u64, ram_gib: u64) -> SystemProfile {
        SystemProfile {
            cpu: CpuInfo {
                brand: "AMD Ryzen 9 7950X".to_owned(),
                physical_cores: 16,
                logical_cores: 32,
                arch: CpuArch::X86_64,
            },
            memory: HostMemory {
                total_bytes: ram_gib * GIB,
                available_bytes: ram_gib * GIB * 3 / 4,
                channels: Some(2),
                speed_mts: Some(5600),
                uma_carveout_bytes: None,
            },
            accelerators: vec![Accelerator {
                index: 0,
                name: format!("RTX (test, {vram_gib} GiB)"),
                vendor: Vendor::Nvidia,
                backend: Backend::Cuda,
                total_bytes: vram_gib * GIB,
                reserved_bytes: GIB,
                unified: false,
                drives_display: true,
                peak_bandwidth_gbps: Some(1008.0),
                peak_tflops_fp16: Some(165.0),
            }],
            os: "Windows 11".to_owned(),
        }
    }

    fn cpu_only(ram_gib: u64) -> SystemProfile {
        SystemProfile {
            accelerators: Vec::new(),
            ..machine(0, ram_gib)
        }
    }

    fn calibration() -> Calibration {
        Calibration {
            accelerator: Some(DeviceThroughput::from_vendor_spec(1008.0, Some(165.0))),
            host: DeviceThroughput::from_vendor_spec(89.6, None),
            overhead_ms_per_token: 0.2,
            source: Confidence::VendorSpec,
        }
    }

    fn benchmarks() -> Benchmarks {
        Benchmarks {
            mmlu_pro: Some(48.0),
            gpqa_diamond: Some(32.0),
            ifeval: Some(80.0),
            arena_elo: Some(1180.0),
            ..Benchmarks::default()
        }
    }

    #[test]
    fn usability_saturates_rather_than_rising_forever() {
        let preference = Preference::Balanced;
        let unusable = usability(1.0, preference);
        let usable = usability(15.0, preference);
        let fast = usability(80.0, preference);
        assert_eq!(usability(0.0, preference), 0.0);
        assert!(unusable < usable && usable < fast);
        assert!(fast < 1.0, "the curve approaches one, it does not reach it");
        // Making an unusable model usable is worth far more than making a fast
        // one faster.
        assert!(
            fast - usable < usable - unusable,
            "saturation is not doing its job"
        );
    }

    #[test]
    fn a_speed_preference_is_less_forgiving_of_a_slow_model() {
        let slow = 4.0;
        assert!(
            usability(slow, Preference::Quality) > usability(slow, Preference::Balanced),
            "a quality preference should tolerate waiting"
        );
        assert!(
            usability(slow, Preference::Balanced) > usability(slow, Preference::Speed),
            "a speed preference should not"
        );
    }

    #[test]
    fn speed_scales_ability_rather_than_standing_in_for_it() {
        // The case an additive score gets wrong: a small fast model against a
        // larger slower one, on a machine where neither is quick.
        let tiny = QualityAssessment {
            full_precision: 26.0,
            quantized: 26.0,
            degradation: 0.0,
            coverage: 1.0,
            basis: crate::quality::QualityBasis::Measured,
        };
        let capable = QualityAssessment {
            full_precision: 48.0,
            quantized: 48.0,
            ..tiny
        };
        let at = |quality: &QualityAssessment, tps: f64| {
            quality.quantized * usability(tps, Preference::Balanced) * (1.0 - crowding_risk(0.5))
        };
        assert!(
            at(&capable, 9.4) > at(&tiny, 31.9),
            "a 4B at 9 tok/s should beat a 0.6B at 32 tok/s for general use"
        );
    }

    #[test]
    fn crowding_only_discounts_near_the_edge() {
        assert_eq!(crowding_risk(0.4), 0.0);
        assert_eq!(crowding_risk(0.85), 0.0);
        assert!(crowding_risk(0.95) > 0.0);
        assert!(crowding_risk(0.98) > crowding_risk(0.90));
        assert!(crowding_risk(1.0) <= MAX_CROWDING_DISCOUNT);
    }

    #[test]
    fn only_published_formats_are_offered_when_the_catalog_knows_them() {
        use crate::model::GgufBuild;
        let arch = llama_3_1_8b();
        let system = machine(24, 64);
        let calibration = calibration();
        let benchmarks = benchmarks();
        let builds = vec![
            GgufBuild {
                quant: "Q4_K_M".to_owned(),
                repo: "someone/model-GGUF".to_owned(),
                file: "model-Q4_K_M.gguf".to_owned(),
                bytes: 4_920_000_000,
            },
            GgufBuild {
                quant: "Q8_0".to_owned(),
                repo: "someone/model-GGUF".to_owned(),
                file: "model-Q8_0.gguf".to_owned(),
                bytes: 8_540_000_000,
            },
        ];
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration).with_builds(&builds);
        let fit = solve(&ctx, &FitRequest::default()).expect("fits");
        assert!(
            builds.iter().any(|b| b.quant == fit.quant.id),
            "chose {}, which nobody published",
            fit.quant.id
        );
        assert!(
            fit.memory.weights_measured,
            "a published build's real size should have been used"
        );
    }

    #[test]
    fn unified_memory_is_never_split_against_itself() {
        // An integrated machine: one pool, and nowhere else to put anything.
        let arch = llama_3_3_70b();
        let mut system = machine(0, 32);
        system.accelerators = vec![Accelerator {
            index: 0,
            name: "Integrated Graphics".to_owned(),
            vendor: Vendor::Intel,
            backend: Backend::Vulkan,
            total_bytes: 32 * GIB,
            reserved_bytes: 4 * GIB,
            unified: true,
            drives_display: true,
            peak_bandwidth_gbps: None,
            peak_tflops_fp16: None,
        }];
        let calibration = calibration();
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        if let Some(fit) = solve(&ctx, &FitRequest::default()) {
            assert!(
                !matches!(fit.run_mode, RunMode::Offloaded { .. }),
                "there is nowhere to offload to: {:?}",
                fit.run_mode
            );
        }
    }

    #[test]
    fn an_8b_on_a_24gb_card_runs_fully_on_the_gpu() {
        let arch = llama_3_1_8b();
        let system = machine(24, 64);
        let calibration = calibration();
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        let fit = solve(&ctx, &FitRequest::default()).expect("an 8B fits in 23 GiB");
        assert!(matches!(fit.run_mode, RunMode::Accelerated { devices: 1 }));
        assert!(fit.verdict.runs());
        assert!(
            fit.throughput.decode_tps > 40.0,
            "expected GPU-class speed, got {:.1}",
            fit.throughput.decode_tps
        );
    }

    #[test]
    fn quality_preference_picks_a_higher_fidelity_format_than_speed_does() {
        let arch = llama_3_1_8b();
        let system = machine(24, 64);
        let calibration = calibration();
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        let for_quality = solve(
            &ctx,
            &FitRequest {
                preference: Preference::Quality,
                ..FitRequest::default()
            },
        )
        .expect("fits");
        let for_speed = solve(
            &ctx,
            &FitRequest {
                preference: Preference::Speed,
                ..FitRequest::default()
            },
        )
        .expect("fits");
        assert!(
            for_quality.quant.body_bpw >= for_speed.quant.body_bpw,
            "quality chose {} ({} bpw), speed chose {} ({} bpw)",
            for_quality.quant.id,
            for_quality.quant.body_bpw,
            for_speed.quant.id,
            for_speed.quant.body_bpw
        );
        assert!(for_speed.throughput.decode_tps >= for_quality.throughput.decode_tps);
    }

    #[test]
    fn a_70b_on_a_12gb_card_offloads_and_is_told_to_quantise_instead() {
        let arch = llama_3_3_70b();
        let system = machine(12, 128);
        let calibration = calibration();
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        let fit = solve(&ctx, &FitRequest::default()).expect("a 70B fits somewhere in 128 GiB");
        assert!(
            matches!(fit.run_mode, RunMode::Offloaded { .. } | RunMode::Cpu),
            "a 70B cannot be fully resident in 11 GiB, got {:?}",
            fit.run_mode
        );
        assert!(
            fit.throughput.decode_tps < 20.0,
            "a spilled 70B should be slow, got {:.1} tok/s",
            fit.throughput.decode_tps
        );
    }

    #[test]
    fn a_model_that_does_not_fit_anywhere_returns_nothing() {
        let arch = llama_3_3_70b();
        // Four gigabytes of RAM and no accelerator.
        let system = cpu_only(4);
        let calibration = calibration();
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        assert!(solve(&ctx, &FitRequest::default()).is_none());
    }

    #[test]
    fn a_long_context_request_reports_the_reachable_ceiling() {
        let arch = llama_3_1_8b();
        let system = machine(12, 32);
        let calibration = calibration();
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        let fit = solve(
            &ctx,
            &FitRequest {
                context: 131_072,
                ..FitRequest::default()
            },
        );
        // Either it does not fit at all, or it fits and admits the shortfall.
        if let Some(fit) = fit {
            if fit.max_context.is_some_and(|c| c < 131_072) {
                assert!(
                    fit.notes
                        .iter()
                        .any(|n| matches!(n, FitNote::ContextCeiling { .. })),
                    "a short ceiling must be reported"
                );
            }
        }
    }

    #[test]
    fn a_cache_quantisation_saving_is_offered_when_it_is_worth_having() {
        let arch = llama_3_1_8b();
        let system = machine(24, 64);
        let calibration = calibration();
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        let fit = solve(
            &ctx,
            &FitRequest {
                context: 65_536,
                ..FitRequest::default()
            },
        )
        .expect("fits");
        if fit.kv_quant == KvQuant::F16 {
            assert!(
                fit.notes
                    .iter()
                    .any(|n| matches!(n, FitNote::QuantiseCache { .. })),
                "8 GiB of fp16 cache at 64k should prompt a suggestion"
            );
        }
    }

    #[test]
    fn disabling_flash_attention_earns_a_warning() {
        let arch = llama_3_1_8b();
        let system = machine(24, 64);
        let calibration = calibration();
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        let fit = solve(
            &ctx,
            &FitRequest {
                context: 32_768,
                runtime: RuntimeProfile::LLAMA_CPP_NO_FLASH,
                ..FitRequest::default()
            },
        )
        .expect("fits");
        assert!(
            fit.notes
                .iter()
                .any(|n| matches!(n, FitNote::EnableFlashAttention { .. })),
            "a materialised score matrix must be called out"
        );
    }

    #[test]
    fn concurrency_reduces_the_reachable_context() {
        let arch = llama_3_1_8b();
        let system = machine(24, 64);
        let calibration = calibration();
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        let single = solve(&ctx, &FitRequest::default()).expect("fits");
        let many = solve(
            &ctx,
            &FitRequest {
                parallel: 16,
                ..FitRequest::default()
            },
        )
        .expect("fits");
        assert!(
            many.memory.kv_cache > single.memory.kv_cache,
            "sixteen sequences must cost more cache than one"
        );
    }

    #[test]
    fn a_cpu_only_machine_still_produces_an_answer() {
        let arch = llama_3_1_8b();
        let system = cpu_only(32);
        let calibration = Calibration {
            accelerator: None,
            ..calibration()
        };
        let benchmarks = benchmarks();
        let ctx = FitContext::new(&arch, &benchmarks, &system, &calibration);
        let fit = solve(&ctx, &FitRequest::default()).expect("an 8B fits in 24 GiB of RAM");
        assert_eq!(fit.run_mode, RunMode::Cpu);
        assert!(
            fit.throughput.decode_tps < 30.0,
            "CPU inference should not look like GPU inference"
        );
    }
}
