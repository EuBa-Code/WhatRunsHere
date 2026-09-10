//! What the window may ask the engine.
//!
//! Six commands, each a thin wrapper: detection and the catalog are read once
//! at startup and held, and everything else is computed on demand from them.
//! Nothing here decides anything — the shapes below are the engine's own types,
//! serialised as they stand, so a figure shown in the window is the same figure
//! `whatllm --json` prints.
//!
//! Every shape here serialises in snake case, which is the engine's own and
//! the command line's. Renaming these to camel case for the window would
//! rename only the outermost object: the types underneath come from
//! `whatllm-core` and cannot be recased from here without changing what
//! `whatllm --json` prints. One casing throughout keeps the seam out of the
//! middle of an object, where nothing would have caught it.
//!
//! Two things are deliberately not thin, and both for the same reason. The
//! curves come from `memory::context_curve` and `perf::decode_curve` rather
//! than from a loop written here, and [`measure`] takes the write lock for the
//! second a probe runs. A second sampler would be a second thing to keep
//! correct; a rank computed halfway through a calibration swap would describe
//! two machines at once.

// Tauri resolves a command's arguments by type, and `State` must be taken by
// value for that to work. Clippy reads every one of these as a needless copy.
#![allow(clippy::needless_pass_by_value)]

use serde::{Deserialize, Serialize};
use std::sync::RwLock;
use whatllm_core::cost::{
    self, ApiPricing, CostComparison, EnergyProfile, HardwareInvestment, Workload,
};
use whatllm_core::fit::{self, FitContext, FitNote, FitRequest, ModelFit, RunMode, Verdict};
use whatllm_core::hardware::Accelerator;
use whatllm_core::memory::{self, LoadConfig, RuntimeProfile};
use whatllm_core::model::{Catalog, ModelEntry};
use whatllm_core::perf::{self, Calibration, Confidence};
use whatllm_core::quant::KvQuant;

// Re-exported so a caller building a [`Sizing`] does not have to reach past
// this module into `whatllm-core` for two of its five fields.
pub use whatllm_core::fit::Preference;
pub use whatllm_core::quality::UseCase;

use whatllm_probe::ProbeOptions;
use whatllm_state::{CachedCalibration, CatalogSource};

/// Everything the window reasons about, read once and held.
pub struct Engine {
    inner: RwLock<State>,
}

struct State {
    detection: whatllm_hw::Detection,
    calibration: Calibration,
    cached: Option<CachedCalibration>,
    catalog: Catalog,
    source: CatalogSource,
}

impl Engine {
    /// Detect the machine and load the catalog.
    ///
    /// # Panics
    /// Only if the catalog compiled into this binary does not parse, which
    /// would mean the build shipped a malformed one. There is no useful
    /// recovery from that and no honest answer to give without it.
    pub fn new() -> Self {
        let detection = whatllm_hw::detect();
        let cached = whatllm_state::load_calibration(&detection.system);
        let calibration = whatllm_hw::calibration(
            &detection.system,
            cached.as_ref().map(|c| c.host_bytes_per_s),
        );
        // Only an explicitly requested file can fail to load, and none is
        // requested here: the copy compiled into the binary is the floor.
        let (catalog, source) =
            whatllm_state::load_catalog(None).expect("the built-in catalog always parses");

        Self {
            inner: RwLock::new(State {
                detection,
                calibration,
                cached,
                catalog,
                source,
            }),
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

/// A lock this process poisoned is a bug in this process, and continuing past
/// it would mean reasoning about a half-written machine.
fn read(engine: &Engine) -> std::sync::RwLockReadGuard<'_, State> {
    engine
        .inner
        .read()
        .expect("the engine lock was poisoned by a panic elsewhere")
}

/// How the window asks for a model to be sized.
///
/// Mirrors the command line's sizing flags one for one, so the two front ends
/// cannot drift into answering different questions.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Sizing {
    /// Context length in tokens.
    pub context: u32,
    /// Concurrent sequences to serve.
    pub parallel: u32,
    /// What the model is for.
    pub use_case: UseCase,
    /// What to optimise for.
    pub preference: Preference,
    /// The runtime that will host it.
    pub runtime: RuntimeName,
}

/// The runtimes a model can be hosted by.
///
/// Named rather than described: a whole [`RuntimeProfile`] arriving from the
/// window would be a way to ask about a runtime that does not exist.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeName {
    /// llama.cpp, Ollama or LM Studio, with flash attention.
    LlamaCpp,
    /// The same without it, which costs a great deal at long context.
    LlamaCppNoFlash,
    /// vLLM.
    Vllm,
    /// MLX on Apple Silicon.
    Mlx,
}

impl From<RuntimeName> for RuntimeProfile {
    fn from(value: RuntimeName) -> Self {
        match value {
            RuntimeName::LlamaCpp => Self::LLAMA_CPP,
            RuntimeName::LlamaCppNoFlash => Self::LLAMA_CPP_NO_FLASH,
            RuntimeName::Vllm => Self::VLLM,
            RuntimeName::Mlx => Self::MLX,
        }
    }
}

impl Sizing {
    fn request(self) -> FitRequest {
        FitRequest {
            use_case: self.use_case,
            context: self.context,
            parallel: self.parallel.max(1),
            runtime: self.runtime.into(),
            preference: self.preference,
        }
    }
}

/// The machine, and what is known about it.
#[derive(Debug, Serialize)]
pub struct Machine {
    /// What was detected, caveats included.
    pub detection: whatllm_hw::Detection,
    /// The throughput figures every estimate is built on.
    pub calibration: Calibration,
    /// The stored measurement behind that calibration, when there is one.
    pub measurement: Option<CachedCalibration>,
    /// Every pool a model could be placed in, largest first.
    pub pools: Vec<whatllm_core::hardware::MemoryPool>,
    /// The processor and each accelerator with trademark marks removed.
    ///
    /// Sent alongside the raw names rather than instead of them: the raw
    /// string is the evidence a misdetection report needs, and the clean one
    /// is what a person should be shown.
    pub display: Names,
    /// Where the catalog came from.
    pub catalog_source: String,
    /// How many models it holds.
    pub catalog_size: usize,
    /// The date it was built.
    pub catalog_generated: String,
}

/// Names as a person should read them.
#[derive(Debug, Serialize)]
pub struct Names {
    /// Processor.
    pub cpu: String,
    /// Each accelerator, in driver order.
    pub accelerators: Vec<String>,
    /// Each memory pool's label, in the same order as `pools`.
    pub pools: Vec<String>,
}

/// Report the machine.
#[tauri::command]
pub fn machine(engine: tauri::State<'_, Engine>) -> Machine {
    machine_of(&engine)
}

/// The body of [`machine`], reachable without a window.
pub fn machine_of(engine: &Engine) -> Machine {
    use whatllm_core::hardware::display_name;

    let state = read(engine);
    let system = &state.detection.system;
    Machine {
        display: Names {
            cpu: display_name(&system.cpu.brand),
            accelerators: system
                .accelerators
                .iter()
                .map(Accelerator::display_name)
                .collect(),
            pools: system
                .pools()
                .iter()
                .map(|pool| display_name(&pool.label))
                .collect(),
        },
        pools: system.pools(),
        detection: state.detection.clone(),
        calibration: state.calibration,
        measurement: state.cached.clone(),
        catalog_source: state.source.to_string(),
        catalog_size: state.catalog.models.len(),
        catalog_generated: state.catalog.generated.clone(),
    }
}

/// One model, as a list needs it.
///
/// Not the whole [`ModelEntry`]: the catalog carries every published build and
/// the full benchmark set for each model, which is most of 233 KB and none of
/// what a list shows.
#[derive(Debug, Serialize)]
pub struct CatalogEntry {
    /// Catalog id, which is also the `HuggingFace` repository.
    pub id: String,
    /// Name to show a person.
    pub name: String,
    /// Family, for grouping.
    pub family: String,
    /// Total stored parameters.
    pub parameters: u64,
    /// Parameters read per token, which is fewer for a sparse model.
    pub active_parameters: u64,
    /// Longest context the model was trained for.
    pub max_context: u32,
    /// Whether the model routes to experts.
    pub sparse: bool,
    /// Release date, as an ISO calendar date.
    pub released: Option<String>,
    /// Licence identifier.
    pub license: Option<String>,
    /// Published builds, by quantization name.
    pub builds: Vec<String>,
}

/// The whole catalog, in list form.
#[tauri::command]
pub fn catalog(engine: tauri::State<'_, Engine>) -> Vec<CatalogEntry> {
    catalog_of(&engine)
}

/// The body of [`catalog`], reachable without a window.
pub fn catalog_of(engine: &Engine) -> Vec<CatalogEntry> {
    let state = read(engine);
    state
        .catalog
        .models
        .iter()
        .map(|model| CatalogEntry {
            id: model.id.clone(),
            name: model.display_name.clone(),
            family: model.family.clone(),
            parameters: model.architecture.total_params(),
            active_parameters: model.architecture.active_params(),
            max_context: model.architecture.max_context,
            sparse: model.architecture.moe.is_some(),
            released: model.released.clone(),
            license: model.license.clone(),
            builds: model.builds.iter().map(|b| b.quant.clone()).collect(),
        })
        .collect()
}

/// One model's footprint, with the totals the engine derives from it.
///
/// [`MemoryPlan`] carries the parts and computes the totals in methods, so a
/// window handed the raw structure would have to add them up itself. It would
/// get the same answer today and a different one the first time a term is
/// added. The engine's own arithmetic is called here instead.
#[derive(Debug, Serialize)]
pub struct Footprint {
    /// Model weights.
    pub weights: u64,
    /// True when `weights` came from a published file rather than an estimate.
    pub weights_measured: bool,
    /// Attention cache across every concurrent sequence.
    pub kv_cache: u64,
    /// Graph working set, summed.
    pub activations: u64,
    /// The attention score matrix alone, which is zero under flash attention
    /// and gigabytes without it. Reported apart because it is the term that
    /// surprises people.
    pub attention_scores: u64,
    /// Runtime fixed overhead.
    pub runtime_overhead: u64,
    /// Allocator slack deliberately left free.
    pub headroom: u64,
    /// Everything that must be resident, headroom excluded.
    pub resident: u64,
    /// What the pool must supply for this to load and stay up.
    pub required: u64,
}

impl From<&whatllm_core::memory::MemoryPlan> for Footprint {
    fn from(plan: &whatllm_core::memory::MemoryPlan) -> Self {
        Self {
            weights: plan.weights,
            weights_measured: plan.weights_measured,
            kv_cache: plan.kv_cache,
            activations: plan.activations.total(),
            attention_scores: plan.activations.attention_scores,
            runtime_overhead: plan.runtime_overhead,
            headroom: plan.headroom,
            resident: plan.resident(),
            required: plan.required(),
        }
    }
}

/// What the solver decided, flattened for display.
#[derive(Debug, Serialize)]
pub struct FitView {
    /// The weight format chosen, by name.
    pub quant: String,
    /// Effective bits per weight in the body of the model.
    pub bits_per_weight: f64,
    /// The cache format chosen.
    pub kv_quant: KvQuant,
    /// Where it runs.
    pub run_mode: RunMode,
    /// The pool it was placed in.
    pub pool: whatllm_core::hardware::MemoryPool,
    /// The full memory accounting.
    pub memory: Footprint,
    /// Fraction of the pool consumed.
    pub utilisation: f64,
    /// How comfortably it sits.
    pub verdict: Verdict,
    /// Generated tokens per second at the evaluated context.
    pub decode_tps: f64,
    /// Prompt tokens per second. Null when nothing measured compute, which is
    /// deliberately different from zero.
    pub prefill_tps: Option<f64>,
    /// How trustworthy those two are.
    pub confidence: Confidence,
    /// Quality after quantization, 0-100.
    pub quality: f64,
    /// Quality at full precision, for comparison.
    pub quality_full_precision: f64,
    /// Points lost to quantization.
    pub degradation: f64,
    /// Fraction of the use case backed by real evaluations.
    pub quality_coverage: f64,
    /// How many catalog models this one scores at least as well as, and out of
    /// how many.
    ///
    /// A bare 53 out of 100 says nothing: the scale is a weighted average of
    /// published benchmark accuracies, and nothing scores near 100 on those.
    /// Rank inside the catalog is the anchor the engine can give honestly,
    /// because it compares like with like under the same weighting. Filled in
    /// by [`rank`], which is the only place the whole field is in view; `None`
    /// wherever one model was solved alone.
    pub quality_rank: Option<QualityRank>,
    /// The longest context reachable in this placement.
    pub max_context: Option<u32>,
    /// The composite score the solver ranked by.
    pub score: f64,
    /// What to do about it.
    pub notes: Vec<FitNote>,
}

impl From<ModelFit> for FitView {
    fn from(fit: ModelFit) -> Self {
        Self {
            quant: fit.quant.id.to_owned(),
            bits_per_weight: fit.quant.body_bpw,
            kv_quant: fit.kv_quant,
            run_mode: fit.run_mode,
            pool: fit.pool,
            memory: (&fit.memory).into(),
            utilisation: fit.utilisation,
            verdict: fit.verdict,
            decode_tps: fit.throughput.decode_tps,
            prefill_tps: fit.throughput.prefill_tps,
            confidence: fit.throughput.confidence,
            quality: fit.quality.quantized,
            quality_full_precision: fit.quality.full_precision,
            degradation: fit.quality.degradation,
            quality_coverage: fit.quality.coverage,
            max_context: fit.max_context,
            score: fit.score,
            notes: fit.notes,
            // Only meaningful against a field, so [`rank`] fills it in.
            quality_rank: None,
        }
    }
}

/// Where a model's quality sits among the models it was ranked beside.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct QualityRank {
    /// Models scoring no higher than this one.
    pub at_or_below: usize,
    /// Models compared, this one included.
    pub of: usize,
}

/// One point on either curve, paired so the window plots one series.
#[derive(Debug, Serialize)]
pub struct CurveSample {
    /// Context length in tokens.
    pub context: u32,
    /// What the model needs there.
    pub required: u64,
    /// Generated tokens per second there.
    pub decode_tps: f64,
}

/// One model placed on this machine.
#[derive(Debug, Serialize)]
pub struct RankedModel {
    /// Catalog id.
    pub id: String,
    /// Name to show a person.
    pub name: String,
    /// Family, for grouping.
    pub family: String,
    /// Total stored parameters.
    pub parameters: u64,
    /// Parameters read per token, which is fewer for a sparse model.
    pub active_parameters: u64,
    /// Whether the model routes to experts.
    pub sparse: bool,
    /// Longest context the model was trained for.
    pub max_trained_context: u32,
    /// What the solver decided.
    pub fit: FitView,
}

/// Rank the catalog for this machine.
///
/// Every model the solver can place is returned, including the ones it places
/// badly. A list that silently dropped them could not answer "why is that one
/// not here", which is the next question someone asks.
#[tauri::command]
pub fn rank(engine: tauri::State<'_, Engine>, sizing: Sizing) -> Vec<RankedModel> {
    rank_of(&engine, sizing)
}

/// The body of [`rank`], reachable without a window.
pub fn rank_of(engine: &Engine, sizing: Sizing) -> Vec<RankedModel> {
    let state = read(engine);
    let request = sizing.request();

    let mut ranked: Vec<RankedModel> = state
        .catalog
        .models
        .iter()
        .filter_map(|model| {
            solve(&state, model, &request).map(|fit| RankedModel {
                id: model.id.clone(),
                name: model.display_name.clone(),
                family: model.family.clone(),
                parameters: model.architecture.total_params(),
                active_parameters: model.architecture.active_params(),
                sparse: model.architecture.moe.is_some(),
                max_trained_context: model.architecture.max_context,
                fit: fit.into(),
            })
        })
        .collect();

    // Quality rank, before the list is reordered by fit score. The two are
    // different questions — the best model here is rarely the best model —
    // and reading rank off the fit order would answer the wrong one.
    let mut scores: Vec<f64> = ranked.iter().map(|m| m.fit.quality).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let of = scores.len();
    for model in &mut ranked {
        let at_or_below = scores.partition_point(|&s| s <= model.fit.quality);
        model.fit.quality_rank = Some(QualityRank { at_or_below, of });
    }

    ranked.sort_by(|a, b| {
        b.fit
            .score
            .partial_cmp(&a.fit.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

fn solve(state: &State, model: &ModelEntry, request: &FitRequest) -> Option<ModelFit> {
    let ctx = FitContext::new(
        &model.architecture,
        &model.benchmarks,
        &state.detection.system,
        &state.calibration,
    )
    .with_builds(&model.builds);
    fit::solve(&ctx, request)
}

/// Everything about running one model here.
#[derive(Debug, Serialize)]
pub struct Plan {
    /// Catalog id.
    pub id: String,
    /// Name to show a person.
    pub name: String,
    /// Family, for grouping.
    pub family: String,
    /// Licence identifier.
    pub license: Option<String>,
    /// Release date, as an ISO calendar date.
    pub released: Option<String>,
    /// What the solver decided.
    pub fit: FitView,
    /// Every published build, so the window can show what else exists.
    pub builds: Vec<Build>,
    /// Footprint and speed against context length, on one ladder.
    pub curve: Vec<CurveSample>,
    /// The pool the curve is drawn against, so the window can mark the
    /// context at which the model stops fitting without recomputing it.
    pub pool_bytes: u64,
}

/// One published build of a model.
#[derive(Debug, Serialize)]
pub struct Build {
    /// Quantization name.
    pub quant: String,
    /// Exact file size in bytes, as published.
    pub bytes: u64,
    /// A command that downloads it.
    pub command: String,
    /// Whether this is the build the solver chose.
    pub chosen: bool,
}

/// The full breakdown for one model.
#[tauri::command]
pub fn plan(engine: tauri::State<'_, Engine>, id: String, sizing: Sizing) -> Option<Plan> {
    plan_of(&engine, &id, sizing)
}

/// The body of [`plan`], reachable without a window.
pub fn plan_of(engine: &Engine, id: &str, sizing: Sizing) -> Option<Plan> {
    let state = read(engine);
    let model = state.catalog.find(id)?;
    let request = sizing.request();
    let fit = solve(&state, model, &request)?;
    let arch = &model.architecture;

    // The configuration the solver settled on, handed back to the engine's own
    // samplers. Writing the loop here instead would be a second implementation
    // of the thing the validations check.
    let load = LoadConfig {
        context: request.context,
        parallel: request.parallel.max(1),
        ubatch: 512,
        weight_quant: fit.quant,
        kv_quant: fit.kv_quant,
        measured_weight_bytes: model.build(fit.quant.id).map(|b| b.bytes),
        runtime: request.runtime,
    };

    // Both curves walk the same power-of-two ladder, so they are zipped into
    // one series rather than left for the window to line up by index.
    let memory_curve = memory::context_curve(arch, &load);
    let speed_curve = perf::decode_curve(
        arch,
        &fit.quant,
        &state.calibration,
        fit.kv_quant,
        !matches!(fit.run_mode, RunMode::Cpu),
        arch.max_context,
    );
    let curve = memory_curve
        .iter()
        .map(|point| CurveSample {
            context: point.context,
            required: point.plan.required(),
            // Matched on context rather than position: the two samplers agree
            // today, and a mismatch tomorrow should show as a missing point
            // and not as a footprint paired with the wrong speed.
            decode_tps: speed_curve
                .iter()
                .find(|s| s.context == point.context)
                .map_or(f64::NAN, |s| s.decode_tps),
        })
        .collect();

    Some(Plan {
        id: model.id.clone(),
        name: model.display_name.clone(),
        family: model.family.clone(),
        license: model.license.clone(),
        released: model.released.clone(),
        builds: model
            .builds
            .iter()
            .map(|build| Build {
                quant: build.quant.clone(),
                bytes: build.bytes,
                command: build.download_command(),
                chosen: build.quant.eq_ignore_ascii_case(fit.quant.id),
            })
            .collect(),
        pool_bytes: fit.pool.usable_bytes,
        curve,
        fit: fit.into(),
    })
}

/// What a month of use is assumed to look like, and what it is priced against.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct CostQuery {
    /// Requests per month.
    pub requests: u64,
    /// Prompt tokens per request.
    pub input: u32,
    /// Generated tokens per request.
    pub output: u32,
    /// Electricity price per kilowatt-hour.
    pub price_per_kwh: f64,
    /// Whole-system draw while generating, in watts.
    pub watts: f64,
    /// Hardware to amortise. Zero means it is already owned.
    pub hardware_cost: f64,
    /// API price per million input tokens.
    pub api_input: f64,
    /// API price per million output tokens.
    pub api_output: f64,
}

/// Compare running one model here against paying an API for it.
#[tauri::command]
pub fn cost(
    engine: tauri::State<'_, Engine>,
    id: String,
    sizing: Sizing,
    query: CostQuery,
) -> Option<CostComparison> {
    cost_of(&engine, &id, sizing, query)
}

/// The body of [`cost`], reachable without a window.
pub fn cost_of(
    engine: &Engine,
    id: &str,
    sizing: Sizing,
    query: CostQuery,
) -> Option<CostComparison> {
    let state = read(engine);
    let model = state.catalog.find(id)?;
    let fit = solve(&state, model, &sizing.request())?;

    Some(cost::compare(
        &Workload {
            requests_per_month: query.requests,
            input_tokens: query.input,
            output_tokens: query.output,
        },
        fit.throughput.decode_tps,
        // Passed through as it stands, `None` included. Prompt processing is
        // uncounted rather than charged at zero, and the comparison says so.
        fit.throughput.prefill_tps,
        &EnergyProfile {
            load_watts: query.watts,
            price_per_kwh: query.price_per_kwh,
        },
        // A price of zero means the hardware is already on the desk, so there
        // is nothing to write down. Amortising nothing over three years and
        // amortising a purchase already made are different claims.
        &if query.hardware_cost > 0.0 {
            HardwareInvestment::purchase(query.hardware_cost)
        } else {
            HardwareInvestment::OWNED
        },
        &ApiPricing {
            input_per_mtok: query.api_input,
            output_per_mtok: query.api_output,
        },
    ))
}

/// What a probe found.
#[derive(Debug, Serialize)]
pub struct Measurement {
    /// Sustained streaming bandwidth, in bytes per second.
    pub bytes_per_s: f64,
    /// The same on one thread, which says how much of it parallelism bought.
    pub single_thread_bytes_per_s: f64,
    /// Threads used.
    pub threads: usize,
    /// How long the measurement took, in seconds.
    pub seconds: f64,
    /// Whether the figure is one the hardware could plausibly have produced.
    ///
    /// An implausible measurement is reported and not stored. A machine under
    /// heavy load measures its own contention, and writing that to the cache
    /// would quietly spoil every later estimate.
    pub plausible: bool,
    /// The calibration as it now stands.
    pub calibration: Calibration,
}

/// Measure this machine, and keep the result.
///
/// # Panics
/// If another command panicked while holding the engine lock. Continuing past
/// that would mean reasoning about a half-written machine.
#[tauri::command]
pub fn measure(engine: tauri::State<'_, Engine>, quick: bool) -> Measurement {
    let options = if quick {
        ProbeOptions::quick()
    } else {
        ProbeOptions::default()
    };
    // Measured before the lock is taken: holding it across a second of
    // streaming would freeze the window for no reason, and the probe reads
    // nothing the engine owns.
    let host = whatllm_probe::measure_host_bandwidth(options);

    let mut state = engine
        .inner
        .write()
        .expect("the engine lock was poisoned by a panic elsewhere");

    if host.plausible {
        state.calibration =
            whatllm_hw::calibration(&state.detection.system, Some(host.bytes_per_s));
        // Written, then read back, so the window is shown what is on disk
        // rather than what was meant to be: a measurement the home directory
        // refused must not appear to have been kept.
        if whatllm_state::save_calibration(
            &state.detection.system,
            host.bytes_per_s,
            host.single_thread_bytes_per_s,
        )
        .is_ok()
        {
            state.cached = whatllm_state::load_calibration(&state.detection.system);
        }
    }

    Measurement {
        bytes_per_s: host.bytes_per_s,
        single_thread_bytes_per_s: host.single_thread_bytes_per_s,
        threads: host.threads,
        seconds: host.elapsed.as_secs_f64(),
        plausible: host.plausible,
        calibration: state.calibration,
    }
}
