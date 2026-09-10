//! `WhatLLM` on the command line.
//!
//! Five questions, five commands: what is this machine, how fast is it really,
//! what will run on it, what happens when one of them does, and whether any of
//! it beats paying for an API.

#![forbid(unsafe_code)]

mod render;
mod state;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use render::Style;
use std::path::PathBuf;
use whatllm_core::cost::{self, ApiPricing, EnergyProfile, HardwareInvestment, Workload};
use whatllm_core::fit::{self, FitContext, FitNote, FitRequest, ModelFit, Preference, RunMode};
use whatllm_core::memory::{self, LoadConfig, RuntimeProfile};
use whatllm_core::model::{Catalog, ModelEntry};
use whatllm_core::perf::{self, Calibration};
use whatllm_core::quality::UseCase;
use whatllm_probe::ProbeOptions;

/// What the model will be used for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum UseCaseArg {
    General,
    Coding,
    Reasoning,
    Math,
    Chat,
    Agentic,
    LongContext,
    Multilingual,
}

impl From<UseCaseArg> for UseCase {
    fn from(value: UseCaseArg) -> Self {
        match value {
            UseCaseArg::General => Self::General,
            UseCaseArg::Coding => Self::Coding,
            UseCaseArg::Reasoning => Self::Reasoning,
            UseCaseArg::Math => Self::Math,
            UseCaseArg::Chat => Self::Chat,
            UseCaseArg::Agentic => Self::Agentic,
            UseCaseArg::LongContext => Self::LongContext,
            UseCaseArg::Multilingual => Self::Multilingual,
        }
    }
}

/// What to optimise for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum PreferenceArg {
    Quality,
    Speed,
    Balanced,
}

impl From<PreferenceArg> for Preference {
    fn from(value: PreferenceArg) -> Self {
        match value {
            PreferenceArg::Quality => Self::Quality,
            PreferenceArg::Speed => Self::Speed,
            PreferenceArg::Balanced => Self::Balanced,
        }
    }
}

/// The runtime the model will be hosted by.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum RuntimeArg {
    /// llama.cpp, Ollama or LM Studio with flash attention.
    LlamaCpp,
    /// The same without flash attention, which costs a great deal at long
    /// context.
    LlamaCppNoFlash,
    /// vLLM.
    Vllm,
    /// MLX on Apple Silicon.
    Mlx,
}

impl From<RuntimeArg> for RuntimeProfile {
    fn from(value: RuntimeArg) -> Self {
        match value {
            RuntimeArg::LlamaCpp => Self::LLAMA_CPP,
            RuntimeArg::LlamaCppNoFlash => Self::LLAMA_CPP_NO_FLASH,
            RuntimeArg::Vllm => Self::VLLM,
            RuntimeArg::Mlx => Self::MLX,
        }
    }
}

/// Options shared by every command that sizes a model.
#[derive(Debug, Clone, clap::Args)]
struct SizingArgs {
    /// Context length in tokens.
    #[arg(short, long, default_value_t = 8192)]
    context: u32,
    /// Concurrent sequences to serve.
    #[arg(short, long, default_value_t = 1)]
    parallel: u32,
    /// What the model is for.
    #[arg(short, long, value_enum, default_value_t = UseCaseArg::General)]
    r#use: UseCaseArg,
    /// What to optimise for.
    #[arg(long, value_enum, default_value_t = PreferenceArg::Balanced)]
    prefer: PreferenceArg,
    /// The runtime that will host it.
    #[arg(short, long, value_enum, default_value_t = RuntimeArg::LlamaCpp)]
    runtime: RuntimeArg,
}

impl SizingArgs {
    fn request(&self) -> FitRequest {
        FitRequest {
            use_case: self.r#use.into(),
            context: self.context,
            parallel: self.parallel.max(1),
            runtime: self.runtime.into(),
            preference: self.prefer.into(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "whatllm",
    version,
    about = "Which language models will actually run well on your machine",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Emit JSON instead of a report.
    #[arg(long, global = true)]
    json: bool,

    /// Never colour the output.
    #[arg(long, global = true)]
    no_color: bool,

    /// Read the model catalog from this file.
    #[arg(long, global = true, value_name = "PATH")]
    catalog: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Report what this machine is, and what has been measured about it.
    Doctor,

    /// Measure how fast this machine really streams from memory.
    Probe {
        /// Measure quickly and less precisely.
        #[arg(long)]
        quick: bool,
    },

    /// Rank the catalog for this machine.
    Fit {
        /// Only consider models matching this text.
        query: Option<String>,
        #[command(flatten)]
        sizing: SizingArgs,
        /// Show this many models.
        #[arg(short = 'n', long, default_value_t = 20)]
        limit: usize,
    },

    /// Everything about running one model here.
    Plan {
        /// Model id or a fragment of its name.
        model: String,
        #[command(flatten)]
        sizing: SizingArgs,
    },

    /// Compare running one model here against paying for an API.
    Cost {
        /// Model id or a fragment of its name.
        model: String,
        #[command(flatten)]
        sizing: SizingArgs,
        /// Requests per month.
        #[arg(long, default_value_t = 3000)]
        requests: u64,
        /// Prompt tokens per request.
        #[arg(long, default_value_t = 8000)]
        input: u32,
        /// Generated tokens per request.
        #[arg(long, default_value_t = 1200)]
        output: u32,
        /// Electricity price per kilowatt-hour.
        #[arg(long, default_value_t = 0.25)]
        price_per_kwh: f64,
        /// Whole-system draw while generating, in watts.
        #[arg(long, default_value_t = 450.0)]
        watts: f64,
        /// Cost of hardware to amortise. Zero means it is already owned.
        #[arg(long, default_value_t = 0.0)]
        hardware_cost: f64,
        /// API price per million input tokens.
        #[arg(long, default_value_t = 0.30)]
        api_input: f64,
        /// API price per million output tokens.
        #[arg(long, default_value_t = 1.20)]
        api_output: f64,
    },
}

/// The machine, its calibration, and the catalog: everything a command needs.
struct Session {
    detection: whatllm_hw::Detection,
    calibration: Calibration,
    cached: Option<state::CachedCalibration>,
    catalog: Catalog,
    source: state::CatalogSource,
    style: Style,
    json: bool,
}

impl Session {
    fn open(cli: &Cli) -> Result<Self> {
        let detection = whatllm_hw::detect();
        let cached = state::load_calibration(&detection.system);
        let calibration = whatllm_hw::calibration(
            &detection.system,
            cached.as_ref().map(|c| c.host_bytes_per_s),
        );
        let (catalog, source) = state::load_catalog(cli.catalog.as_deref())?;
        Ok(Self {
            detection,
            calibration,
            cached,
            catalog,
            source,
            style: Style::detect(cli.no_color),
            json: cli.json,
        })
    }

    /// Resolve a model from an id or a fragment of its name.
    fn resolve(&self, query: &str) -> Result<&ModelEntry> {
        if let Some(model) = self.catalog.find(query) {
            return Ok(model);
        }
        let hits = self.catalog.search(query);
        match hits.len() {
            0 => anyhow::bail!("no model matches `{query}`. Try `whatllm fit` to see the catalog."),
            _ => Ok(hits[0]),
        }
    }

    fn solve(&self, model: &ModelEntry, request: &FitRequest) -> Option<ModelFit> {
        let ctx = FitContext::new(
            &model.architecture,
            &model.benchmarks,
            &self.detection.system,
            &self.calibration,
        )
        .with_builds(&model.builds);
        fit::solve(&ctx, request)
    }

    /// The one-line summary of the machine that heads every report.
    fn machine_line(&self) -> String {
        let system = &self.detection.system;
        let accelerator = system
            .largest_accelerator()
            .map_or_else(|| "no accelerator".to_owned(), |a| a.name.clone());
        format!(
            "{} · {} · {}",
            system.cpu.brand,
            render::bytes(system.memory.total_bytes),
            accelerator
        )
    }

    fn calibration_line(&self) -> String {
        let bandwidth = self.calibration.host.bandwidth_bytes_per_s / 1e9;
        match &self.cached {
            Some(cached) => format!(
                "{:.0} GB/s measured {}",
                cached.host_bytes_per_s / 1e9,
                state::age(cached.measured_at)
            ),
            None => format!("{bandwidth:.0} GB/s assumed — run `whatllm probe` to measure"),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let session = Session::open(&cli)?;

    match cli.command.unwrap_or(Command::Fit {
        query: None,
        sizing: SizingArgs {
            context: 8192,
            parallel: 1,
            r#use: UseCaseArg::General,
            prefer: PreferenceArg::Balanced,
            runtime: RuntimeArg::LlamaCpp,
        },
        limit: 20,
    }) {
        Command::Doctor => doctor(&session),
        Command::Probe { quick } => probe(&session, quick),
        Command::Fit {
            query,
            sizing,
            limit,
        } => show_fit(&session, query.as_deref(), &sizing, limit),
        Command::Plan { model, sizing } => plan(&session, &model, &sizing),
        Command::Cost {
            model,
            sizing,
            requests,
            input,
            output,
            price_per_kwh,
            watts,
            hardware_cost,
            api_input,
            api_output,
        } => {
            let workload = Workload {
                requests_per_month: requests,
                input_tokens: input,
                output_tokens: output,
            };
            let energy = EnergyProfile {
                load_watts: watts,
                price_per_kwh,
            };
            let hardware = if hardware_cost > 0.0 {
                HardwareInvestment::purchase(hardware_cost)
            } else {
                HardwareInvestment::OWNED
            };
            let pricing = ApiPricing {
                input_per_mtok: api_input,
                output_per_mtok: api_output,
            };
            show_cost(
                &session, &model, &sizing, &workload, &energy, &hardware, &pricing,
            )
        }
    }
}

/// What this machine is.
fn doctor(session: &Session) -> Result<()> {
    let style = session.style;
    let system = &session.detection.system;

    if session.json {
        println!("{}", serde_json::to_string_pretty(&session.detection)?);
        return Ok(());
    }

    println!("{}", render::heading(style, "Machine"));
    println!(
        "{}",
        render::field(
            style,
            14,
            "Processor",
            &format!(
                "{}  ·  {} cores / {} threads",
                system.cpu.brand, system.cpu.physical_cores, system.cpu.logical_cores
            )
        )
    );
    println!(
        "{}",
        render::field(
            style,
            14,
            "Memory",
            &format!(
                "{} installed, {} available",
                render::bytes(system.memory.total_bytes),
                render::bytes(system.memory.available_bytes)
            )
        )
    );
    for accelerator in &system.accelerators {
        let kind = if accelerator.unified {
            "unified memory"
        } else {
            "dedicated memory"
        };
        println!(
            "{}",
            render::field(
                style,
                14,
                "Accelerator",
                &format!(
                    "{}  ·  {}  ·  {} usable, {}",
                    accelerator.name,
                    accelerator.backend.label(),
                    render::bytes(accelerator.usable_bytes()),
                    kind
                )
            )
        );
    }
    println!("{}", render::field(style, 14, "System", &system.os));
    println!(
        "{}",
        render::field(style, 14, "Throughput", &session.calibration_line())
    );

    println!("{}", render::heading(style, "Memory pools"));
    for pool in system.pools() {
        println!(
            "{}",
            render::field(
                style,
                14,
                &pool.label,
                &format!("{} usable", render::bytes(pool.usable_bytes))
            )
        );
    }

    if !session.detection.notes.is_empty() {
        println!("{}", render::heading(style, "Notes"));
        for note in &session.detection.notes {
            println!("  {} {}", style.dim("·"), describe_detection(note));
        }
    }

    println!(
        "\n{}\n",
        style.dim(&format!(
            "Catalog: {} models from {}",
            session.catalog.models.len(),
            session.source
        ))
    );
    Ok(())
}

/// Put a detection caveat into words.
fn describe_detection(note: &whatllm_hw::DetectionNote) -> String {
    use whatllm_hw::DetectionNote as N;
    match note {
        N::IgnoredVirtualAdapter { name } => {
            format!("ignored {name} — a display, not something that computes")
        }
        N::IntegratedGraphics {
            name,
            claimed_vram_bytes,
        } => match claimed_vram_bytes {
            Some(bytes) => format!(
                "{name} is integrated; the {} of \"video memory\" the system reports \
                 is an aperture, not memory it owns. Its pool is system RAM.",
                render::bytes(*bytes)
            ),
            None => format!("{name} is integrated; its pool is system RAM"),
        },
        N::NvidiaUnavailable { .. } => "no NVIDIA driver found".to_owned(),
        N::BandwidthUnknown { device } => {
            format!("{device}'s bandwidth is unknown until something measures it")
        }
        N::PlatformUnsupported { target } => {
            format!("no adapter enumeration for {target} yet")
        }
    }
}

/// Measure the machine.
fn probe(session: &Session, quick: bool) -> Result<()> {
    let style = session.style;
    let options = if quick {
        ProbeOptions::quick()
    } else {
        ProbeOptions::default()
    };

    if !session.json {
        println!(
            "\n  Measuring memory bandwidth with {} threads over {}...",
            options.threads,
            render::bytes(options.total_bytes() as u64)
        );
    }

    let result = whatllm_probe::measure_host_bandwidth(options);

    if session.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }

    // A wrong measurement is worse than none, because everything downstream
    // will believe it. Say so, and change nothing.
    if !result.plausible {
        println!(
            "
  {}
  {}
",
            style.warn(&format!(
                "Measured {:.1} GB/s, which is not a memory bandwidth.",
                result.gigabytes_per_s()
            )),
            style.dim(
                "The machine was busy, or is throttled. Nothing has been saved;                  try again when it is idle."
            )
        );
        return Ok(());
    }

    let path = state::save_calibration(
        &session.detection.system,
        result.bytes_per_s,
        result.single_thread_bytes_per_s,
    )?;

    println!("{}", render::heading(style, "Measured"));
    println!(
        "{}",
        render::field(
            style,
            18,
            "Sustained read",
            &style.bold(&format!("{:.1} GB/s", result.gigabytes_per_s()))
        )
    );
    println!(
        "{}",
        render::field(
            style,
            18,
            "Single thread",
            &format!("{:.1} GB/s", result.single_thread_bytes_per_s / 1e9)
        )
    );
    println!(
        "{}",
        render::field(
            style,
            18,
            "Parallel gain",
            &format!(
                "{:.1}x across {} threads",
                result.parallel_speedup(),
                result.threads
            )
        )
    );
    println!(
        "\n{}\n",
        style.dim(&format!(
            "Saved to {}. Every estimate is now calibrated to this machine.",
            path.display()
        ))
    );
    Ok(())
}

/// One row of the ranked table.
#[derive(serde::Serialize)]
struct FitRow<'a> {
    id: &'a str,
    display_name: &'a str,
    size: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    fit: Option<ModelFit>,
}

/// Rank the catalog.
fn show_fit(
    session: &Session,
    query: Option<&str>,
    sizing: &SizingArgs,
    limit: usize,
) -> Result<()> {
    let style = session.style;
    let request = sizing.request();

    let models: Vec<&ModelEntry> = match query {
        Some(text) => session.catalog.search(text),
        None => session.catalog.models.iter().collect(),
    };
    if models.is_empty() {
        anyhow::bail!("no model matches `{}`", query.unwrap_or(""));
    }

    let mut rows: Vec<FitRow> = models
        .iter()
        .map(|model| FitRow {
            id: &model.id,
            display_name: &model.display_name,
            size: model.size_label(),
            fit: session.solve(model, &request),
        })
        .collect();
    rows.sort_by(|a, b| {
        let score = |row: &FitRow| row.fit.as_ref().map_or(f64::MIN, |f| f.score);
        score(b)
            .partial_cmp(&score(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rows.truncate(limit);

    if session.json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }

    println!("{}", render::heading(style, "What runs here"));
    println!(
        "  {}\n  {}",
        style.dim(&session.machine_line()),
        style.dim(&format!(
            "{} · {} context · {} · optimising for {:?}",
            session.calibration_line(),
            render::tokens(request.context),
            request.use_case.label(),
            sizing.prefer
        ))
    );
    println!();
    println!(
        "  {}",
        style.dim(&format!(
            "{:<30} {:>9} {:>9} {:>10} {:>8} {:>9}  {}",
            "MODEL", "SIZE", "FORMAT", "MEMORY", "TOK/S", "QUALITY", "VERDICT"
        ))
    );
    println!("  {}", render::rule(style, 96));

    for row in &rows {
        let Some(fit) = &row.fit else {
            println!(
                "  {:<30} {:>9} {}",
                style.dim(row.display_name),
                style.dim(&row.size),
                style.dim("— does not fit at this context")
            );
            continue;
        };
        let measured = if fit.memory.weights_measured {
            style.good("✓")
        } else {
            style.dim("~")
        };
        let verdict = match fit.verdict {
            whatllm_core::fit::Verdict::Comfortable | whatllm_core::fit::Verdict::Fits => {
                style.good(fit.verdict.label())
            }
            whatllm_core::fit::Verdict::Tight => style.warn(fit.verdict.label()),
            whatllm_core::fit::Verdict::DoesNotFit => style.bad(fit.verdict.label()),
        };
        println!(
            "  {:<30} {:>9} {:>9} {:>8}{} {:>8} {:>9}  {} {}",
            row.display_name,
            row.size,
            fit.quant.id,
            render::bytes(fit.memory.required()),
            measured,
            format!("{:.1}", fit.throughput.decode_tps),
            format!("{:.0}", fit.quality.quantized),
            verdict,
            style.dim(&fit.run_mode.label()),
        );
    }

    println!(
        "\n  {}\n",
        style.dim(
            "✓ size measured from a real file  ·  ~ computed  ·  `whatllm plan <model>` for detail"
        )
    );
    Ok(())
}

/// Everything about one model here.
fn plan(session: &Session, query: &str, sizing: &SizingArgs) -> Result<()> {
    let style = session.style;
    let model = session.resolve(query)?;
    let request = sizing.request();

    let Some(fit) = session.solve(model, &request) else {
        anyhow::bail!(
            "{} does not fit on this machine at {} of context, in any format",
            model.display_name,
            render::tokens(request.context)
        );
    };

    if session.json {
        println!("{}", serde_json::to_string_pretty(&fit)?);
        return Ok(());
    }

    let arch = &model.architecture;
    println!("{}", render::heading(style, &model.display_name));
    println!(
        "  {}",
        style.dim(&format!(
            "{} · {} · {} layers · {} · {}",
            model.id,
            model.size_label(),
            arch.layers.n_layers,
            if arch.is_sparse() { "sparse" } else { "dense" },
            model.license.as_deref().unwrap_or("licence unknown")
        ))
    );

    // Memory, which is the whole point.
    println!("{}", render::heading(style, "Memory"));
    let pool = fit.pool.usable_bytes;
    let parts = [
        ("Weights", fit.memory.weights),
        ("Attention cache", fit.memory.kv_cache),
        ("Activations", fit.memory.activations.total()),
        ("Runtime overhead", fit.memory.runtime_overhead),
        ("Headroom", fit.memory.headroom),
    ];
    for (label, value) in parts {
        if value == 0 {
            continue;
        }
        println!(
            "  {:<18} {:>9}  {}",
            style.dim(label),
            render::bytes(value),
            style.dim(&render::meter(value as f64 / pool as f64, 28))
        );
    }
    println!("  {}", render::rule(style, 60));
    println!(
        "  {:<18} {:>9}  of {} in {}  ({:.0}%)",
        style.bold("Total"),
        style.bold(&render::bytes(fit.memory.required())),
        render::bytes(pool),
        fit.pool.label,
        fit.utilisation * 100.0
    );
    if fit.memory.weights_measured {
        println!(
            "  {}",
            style.dim("Weight size read from the published file, not estimated.")
        );
    }

    // How it runs.
    println!("{}", render::heading(style, "Performance"));
    println!(
        "{}",
        render::field(
            style,
            18,
            "Generation",
            &format!(
                "{} tok/s at {} context   {}",
                style.bold(&format!("{:.1}", fit.throughput.decode_tps)),
                render::tokens(request.context),
                style.dim(fit.throughput.confidence.label())
            )
        )
    );
    if let Some(ttft) = fit.throughput.ttft_ms(request.context) {
        println!(
            "{}",
            render::field(
                style,
                18,
                "First token",
                &format!(
                    "{:.1} s for a full {} prompt",
                    ttft / 1000.0,
                    render::tokens(request.context)
                )
            )
        );
    }
    println!(
        "{}",
        render::field(
            style,
            18,
            "Placement",
            &format!("{} · {}", fit.run_mode.label(), fit.quant.id)
        )
    );

    // The curves, which are the honest answer to "how fast is it".
    let load = LoadConfig {
        context: request.context,
        parallel: request.parallel.max(1),
        ubatch: 512,
        weight_quant: fit.quant,
        kv_quant: fit.kv_quant,
        measured_weight_bytes: model.build(fit.quant.id).map(|b| b.bytes),
        runtime: request.runtime,
    };
    let memory_curve = memory::context_curve(arch, &load);
    let speed_curve = perf::decode_curve(
        arch,
        &fit.quant,
        &session.calibration,
        fit.kv_quant,
        !matches!(fit.run_mode, RunMode::Cpu),
        arch.max_context,
    );
    if memory_curve.len() > 2 {
        let memory_values: Vec<f64> = memory_curve
            .iter()
            .map(|p| p.plan.required() as f64)
            .collect();
        let speed_values: Vec<f64> = speed_curve.iter().map(|p| p.decode_tps).collect();
        println!("{}", render::heading(style, "Across context lengths"));
        println!(
            "  {:<12} {} {}",
            style.dim("Memory"),
            render::sparkline(&memory_values),
            style.dim(&format!(
                "{} at 1k → {} at {}",
                render::bytes(memory_curve[0].plan.required()),
                render::bytes(memory_curve.last().map_or(0, |p| p.plan.required())),
                render::tokens(memory_curve.last().map_or(0, |p| p.context))
            ))
        );
        println!(
            "  {:<12} {} {}",
            style.dim("Speed"),
            render::sparkline(&speed_values),
            style.dim(&format!(
                "{:.1} tok/s at 1k → {:.1} at {}",
                speed_curve.first().map_or(0.0, |p| p.decode_tps),
                speed_curve.last().map_or(0.0, |p| p.decode_tps),
                render::tokens(speed_curve.last().map_or(0, |p| p.context))
            ))
        );
        if let Some(ceiling) = fit.max_context {
            println!(
                "  {:<12} {}",
                style.dim("Ceiling"),
                format!("{} tokens in this placement", render::tokens(ceiling))
            );
        }
    }

    // Quality, and what quantization took from it.
    println!("{}", render::heading(style, "Quality"));
    println!(
        "{}",
        render::field(
            style,
            18,
            "Score",
            &format!(
                "{} of 100 for {}   {}",
                style.bold(&format!("{:.0}", fit.quality.quantized)),
                request.use_case.label(),
                style.dim(fit.quality.basis.label())
            )
        )
    );
    if fit.quality.degradation > 0.1 {
        println!(
            "{}",
            render::field(
                style,
                18,
                "Given up",
                &format!("{:.1} points to {}", fit.quality.degradation, fit.quant.id)
            )
        );
    }

    if !fit.notes.is_empty() {
        println!("{}", render::heading(style, "Worth changing"));
        for note in &fit.notes {
            println!("  {} {}", style.warn("→"), describe_note(note, style));
        }
    }

    if let Some(build) = model.build(fit.quant.id) {
        println!("{}", render::heading(style, "Get it"));
        println!("  {}", style.dim(&build.download_command()));
    }
    println!();
    Ok(())
}

/// Put a fit recommendation into words.
fn describe_note(note: &FitNote, style: Style) -> String {
    match note {
        FitNote::QuantiseCache {
            to,
            saves_bytes,
            unlocks_context,
            ..
        } => {
            let base = format!(
                "Quantize the attention cache to {} and save {}",
                style.bold(to.id()),
                render::bytes(*saves_bytes)
            );
            match unlocks_context {
                Some(context) => {
                    format!("{base}, reaching {} of context", render::tokens(*context))
                }
                None => format!("{base}. It costs almost nothing in quality."),
            }
        }
        FitNote::ContextCeiling {
            requested,
            achievable,
        } => format!(
            "{} of context does not fit here; {} does",
            render::tokens(*requested),
            style.bold(&render::tokens(*achievable))
        ),
        FitNote::EnableFlashAttention { saves_bytes } => format!(
            "Turn on flash attention: the score matrix is taking {} for nothing",
            style.bold(&render::bytes(*saves_bytes))
        ),
        FitNote::QuantiseToAvoidOffload {
            to,
            speed_multiplier,
            quality_cost,
        } => format!(
            "Drop to {} and the whole model stays on the accelerator: {} faster \
             for {:.1} points of quality",
            style.bold(to),
            style.bold(&format!("{speed_multiplier:.1}x")),
            quality_cost
        ),
        FitNote::RoomForBetterQuant { to, quality_gain } => format!(
            "There is room for {}: {:.1} points of quality for free",
            style.bold(to),
            quality_gain
        ),
        FitNote::RunningClose { utilisation } => format!(
            "Running at {:.0}% of the pool — anything else on this machine will \
             push it over",
            utilisation * 100.0
        ),
    }
}

/// Local against hosted.
fn show_cost(
    session: &Session,
    query: &str,
    sizing: &SizingArgs,
    workload: &Workload,
    energy: &EnergyProfile,
    hardware: &HardwareInvestment,
    pricing: &ApiPricing,
) -> Result<()> {
    let style = session.style;
    let model = session.resolve(query)?;
    let request = sizing.request();
    let fit = session
        .solve(model, &request)
        .with_context(|| format!("{} does not fit on this machine", model.display_name))?;

    let comparison = cost::compare(
        workload,
        fit.throughput.decode_tps,
        fit.throughput.prefill_tps,
        energy,
        hardware,
        pricing,
    );

    if session.json {
        println!("{}", serde_json::to_string_pretty(&comparison)?);
        return Ok(());
    }

    println!(
        "{}",
        render::heading(
            style,
            &format!("{} — local against hosted", model.display_name)
        )
    );
    println!(
        "  {}",
        style.dim(&format!(
            "{} requests a month · {} in, {} out · {:.1} tok/s here",
            workload.requests_per_month,
            render::tokens(workload.input_tokens),
            render::tokens(workload.output_tokens),
            fit.throughput.decode_tps
        ))
    );
    println!();
    println!(
        "{}",
        render::field(
            style,
            22,
            "Compute time",
            &format!(
                "{:.0} hours a month{}",
                comparison.compute_hours_per_month,
                if comparison.prefill_included {
                    ""
                } else {
                    ", generation only"
                }
            )
        )
    );
    if !comparison.prefill_included {
        println!(
            "  {}",
            style.dim(
                "Prompt processing is not counted: nothing has measured this                  machine's compute throughput, and charging it at zero would                  understate both the time and the bill."
            )
        );
    }
    println!(
        "{}",
        render::field(
            style,
            22,
            "Electricity",
            &format!("{:.2} a month", comparison.local_energy)
        )
    );
    if comparison.local_amortisation > 0.0 {
        println!(
            "{}",
            render::field(
                style,
                22,
                "Hardware, amortised",
                &format!("{:.2} a month", comparison.local_amortisation)
            )
        );
    }
    println!("  {}", render::rule(style, 60));
    println!(
        "{}",
        render::field(
            style,
            22,
            "Running it here",
            &style.bold(&format!(
                "{:.2} a month   ({:.3} per million tokens)",
                comparison.local_total, comparison.local_per_mtok
            ))
        )
    );
    println!(
        "{}",
        render::field(
            style,
            22,
            "Paying the API",
            &style.bold(&format!(
                "{:.2} a month   ({:.3} per million tokens)",
                comparison.api_total, comparison.api_per_mtok
            ))
        )
    );

    println!();
    let verdict = match comparison.verdict {
        cost::CostVerdict::LocalCheaper { by_percent } => style.good(&format!(
            "Local is {by_percent:.0}% cheaper — {:.2} a month saved",
            comparison.monthly_saving()
        )),
        cost::CostVerdict::ApiCheaper { by_percent } => style.warn(&format!(
            "The API is {by_percent:.0}% cheaper at this volume",
        )),
        cost::CostVerdict::TooCloseToCall => style.dim(
            "Within a few percent. Decide on privacy, latency or availability \
             instead of price.",
        ),
    };
    println!("  {verdict}");
    if let Some(breakeven) = comparison.breakeven_requests_per_month {
        println!(
            "  {}",
            style.dim(&format!(
                "Break-even at {breakeven} requests a month; you are planning {}.",
                workload.requests_per_month
            ))
        );
    }
    println!();
    Ok(())
}
