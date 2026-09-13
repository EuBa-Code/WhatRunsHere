//! The physics of running a language model on your own hardware.
//!
//! This crate answers four questions from first principles, with no network
//! access and no I/O of any kind:
//!
//! - **Will it fit?** [`memory`] accounts for weights, attention cache,
//!   activations, runtime overhead and allocator slack, using the exact tensor
//!   shapes in [`arch`] rather than a parameter count.
//! - **How fast will it be?** [`perf`] models decode as bandwidth-bound and
//!   prefill as compute-bound, including the cache traffic that makes
//!   throughput decay as a conversation grows.
//! - **On what?** [`hardware`] describes the machine and the memory pools a
//!   model can actually be placed into: usable bytes, not sticker capacity.
//! - **What will it cost?** [`cost`] compares running it here against paying
//!   an API, electricity and hardware amortisation included.
//! - **And then what?** [`launch`] says how to hand the model to the program
//!   that will run it, with the configuration that was sized carried into the
//!   command rather than left for the reader to reconstruct.
//!
//! Everything is deterministic and unit-tested against published figures for
//! real models. Where a measurement exists (a real file size, a real probe of
//! real silicon) it is always preferred over a formula, and the result records
//! which of the two it used.

#![forbid(unsafe_code)]

pub mod arch;
pub mod cost;
pub mod fit;
pub mod hardware;
pub mod launch;
pub mod memory;
pub mod model;
pub mod perf;
pub mod quality;
pub mod quant;

pub use arch::{Architecture, AttentionKind, FfnKind, LayerLayout, LayerSpec, MoeSpec};
pub use cost::{ApiPricing, CostComparison, EnergyProfile, HardwareInvestment, Workload};
pub use fit::{FitContext, FitNote, FitRequest, ModelFit, Preference, RunMode, Verdict};
pub use hardware::{Accelerator, Backend, MemoryPool, SystemProfile, Vendor};
pub use launch::{Host, Launch, LaunchNote, LaunchRequest, Platform, Weights};
pub use memory::{LoadConfig, MemoryPlan, RuntimeProfile};
pub use model::{Catalog, GgufBuild, ModelEntry};
pub use perf::{Calibration, Confidence, DeviceThroughput, Throughput, TrafficSplit};
pub use quality::{Benchmarks, QualityAssessment, QualityBasis, UseCase};
pub use quant::{weight_quant, KvQuant, QuantFamily, WeightQuant};
