//! The figures the download page shows, computed by the engine itself.
//!
//! A page that demonstrates what this tool answers has to get those answers
//! from somewhere, and the tempting shortcut — a few plausible numbers written
//! into the JavaScript — would make the site a second implementation of the
//! thing the whole project refuses to have. So it is not one. This runs the
//! real solver over a handful of described machines and writes what it decided
//! to `site/demo.json`, which the page reads and does no arithmetic on.
//!
//! The machines are described, not detected. Every figure in them is a
//! published specification, and the page says so: a visitor is looking at what
//! `WhatLLM` would conclude about a machine of that shape, not at a measurement
//! of anyone's actual hardware. That distinction is the product's whole
//! argument and it would be a poor thing to blur in the shop window.
//!
//! The result is written straight into `site/index.html`, between the markers
//! around its data block, rather than beside it as a file the page fetches.
//! Two reasons: the page then works from a file:// URL and inside any host
//! that will not let it fetch, and there is one copy of the numbers rather
//! than a file plus a pasted duplicate that drift apart.
//!
//! Run with:
//! ```sh
//! cargo run --release -p whatllm-cli --example demo_data
//! ```

use whatllm_core::fit::{self, FitContext, FitRequest, Preference};
use whatllm_core::hardware::{
    Accelerator, Backend, CpuArch, CpuInfo, HostMemory, SystemProfile, Vendor,
};
use whatllm_core::memory::RuntimeProfile;
use whatllm_core::perf::{Calibration, Confidence, DeviceThroughput};
use whatllm_core::quality::UseCase;

const GIB: u64 = 1024 * 1024 * 1024;

/// One machine the page can be switched to.
struct Machine {
    /// Short label for the control.
    tab: &'static str,
    /// What it is, in full.
    name: &'static str,
    /// A line of specification, shown under the name.
    detail: &'static str,
    profile: SystemProfile,
    calibration: Calibration,
}

fn cpu(brand: &str, cores: u32) -> CpuInfo {
    CpuInfo {
        brand: brand.to_owned(),
        physical_cores: cores,
        logical_cores: cores * 2,
        arch: CpuArch::X86_64,
    }
}

fn memory(total_gib: u64) -> HostMemory {
    HostMemory {
        total_bytes: total_gib * GIB,
        // Three quarters free is an ordinary desktop with a browser open.
        available_bytes: total_gib * GIB * 3 / 4,
        channels: None,
        speed_mts: None,
        uma_carveout_bytes: None,
    }
}

/// A discrete card, described from its published specification.
fn card(name: &str, vram_gib: u64, bandwidth_gbps: f64, tflops: f64) -> Accelerator {
    Accelerator {
        index: 0,
        name: name.to_owned(),
        vendor: Vendor::Nvidia,
        backend: Backend::Cuda,
        total_bytes: vram_gib * GIB,
        // A card driving a display has some of it spoken for before anything
        // of yours loads.
        reserved_bytes: GIB,
        unified: false,
        drives_display: true,
        peak_bandwidth_gbps: Some(bandwidth_gbps),
        peak_tflops_fp16: Some(tflops),
    }
}

fn unified(name: &str, pool_gib: u64, bandwidth_gbps: f64) -> Accelerator {
    Accelerator {
        index: 0,
        name: name.to_owned(),
        vendor: Vendor::Apple,
        backend: Backend::Metal,
        total_bytes: pool_gib * GIB,
        reserved_bytes: 2 * GIB,
        unified: true,
        drives_display: true,
        peak_bandwidth_gbps: Some(bandwidth_gbps),
        peak_tflops_fp16: None,
    }
}

fn calibrated(bandwidth_gbps: f64, tflops: Option<f64>, on_accelerator: bool) -> Calibration {
    let device = DeviceThroughput::from_vendor_spec(bandwidth_gbps, tflops);
    Calibration {
        accelerator: on_accelerator.then_some(device),
        host: if on_accelerator {
            DeviceThroughput::from_vendor_spec(60.0, None)
        } else {
            device
        },
        // The figure the throughput validation measured. See perf.rs.
        overhead_ms_per_token: if on_accelerator { 1.8 } else { 2.5 },
        // Described from a specification, never measured — which is exactly
        // what the page has to admit.
        source: Confidence::VendorSpec,
    }
}

fn machines() -> Vec<Machine> {
    vec![
        Machine {
            tab: "Laptop",
            name: "A laptop with no graphics card",
            detail: "16 GB of memory, integrated graphics — the machine most people already have",
            profile: SystemProfile {
                cpu: cpu("Intel Core Ultra 7", 8),
                memory: memory(16),
                accelerators: vec![],
                os: "Windows 11".to_owned(),
            },
            calibration: calibrated(60.0, None, false),
        },
        Machine {
            tab: "RTX 4060 Ti",
            name: "A mid-range card",
            detail: "NVIDIA RTX 4060 Ti, 16 GB — 288 GB/s, published",
            profile: SystemProfile {
                cpu: cpu("AMD Ryzen 7 7700X", 8),
                memory: memory(32),
                accelerators: vec![card("NVIDIA GeForce RTX 4060 Ti", 16, 288.0, 44.0)],
                os: "Windows 11".to_owned(),
            },
            calibration: calibrated(288.0, Some(44.0), true),
        },
        Machine {
            tab: "RTX 4090",
            name: "The card people buy for this",
            detail: "NVIDIA RTX 4090, 24 GB — 1008 GB/s, published",
            profile: SystemProfile {
                cpu: cpu("AMD Ryzen 9 7950X", 16),
                memory: memory(64),
                accelerators: vec![card("NVIDIA GeForce RTX 4090", 24, 1008.0, 165.0)],
                os: "Windows 11".to_owned(),
            },
            calibration: calibrated(1008.0, Some(165.0), true),
        },
        Machine {
            tab: "MacBook Pro",
            name: "Apple Silicon, one pool of memory",
            detail: "M3 Max, 36 GB unified — 300 GB/s, published",
            profile: SystemProfile {
                cpu: CpuInfo {
                    brand: "Apple M3 Max".to_owned(),
                    physical_cores: 14,
                    logical_cores: 14,
                    arch: CpuArch::Aarch64,
                },
                memory: memory(36),
                accelerators: vec![unified("Apple M3 Max", 36, 300.0)],
                os: "macOS 15".to_owned(),
            },
            calibration: calibrated(300.0, None, false),
        },
        Machine {
            tab: "Ryzen AI MAX+",
            name: "128 GB, most of it given to the graphics",
            detail: "Ryzen AI MAX+ 395 — 96 GB carved out in firmware, 256 GB/s",
            profile: SystemProfile {
                cpu: cpu("AMD Ryzen AI MAX+ 395", 16),
                memory: HostMemory {
                    total_bytes: 128 * GIB,
                    available_bytes: 120 * GIB,
                    channels: None,
                    speed_mts: None,
                    uma_carveout_bytes: Some(96 * GIB),
                },
                accelerators: vec![Accelerator {
                    index: 0,
                    name: "AMD Radeon 8060S Graphics".to_owned(),
                    vendor: Vendor::Amd,
                    backend: Backend::Vulkan,
                    total_bytes: 128 * GIB,
                    reserved_bytes: 8 * GIB,
                    unified: true,
                    drives_display: true,
                    peak_bandwidth_gbps: Some(256.0),
                    peak_tflops_fp16: None,
                }],
                os: "Windows 11".to_owned(),
            },
            calibration: calibrated(256.0, None, false),
        },
    ]
}

fn main() {
    let catalog = whatllm_state::load_catalog(None)
        .expect("the built-in catalog always parses")
        .0;

    let request = FitRequest {
        use_case: UseCase::General,
        context: 8192,
        parallel: 1,
        runtime: RuntimeProfile::LLAMA_CPP,
        preference: Preference::Balanced,
    };

    let machines: Vec<serde_json::Value> = machines()
        .iter()
        .map(|machine| {
            let mut ranked: Vec<(f64, serde_json::Value)> = catalog
                .models
                .iter()
                .filter_map(|model| {
                    let ctx = FitContext::new(
                        &model.architecture,
                        &model.benchmarks,
                        &machine.profile,
                        &machine.calibration,
                    )
                    .with_builds(&model.builds);
                    let fit = fit::solve(&ctx, &request)?;
                    if !fit.verdict.runs() {
                        return None;
                    }
                    let entry = serde_json::json!({
                        "name": model.display_name,
                        "params": model.architecture.total_params(),
                        "quant": fit.quant.id,
                        "required": fit.memory.required(),
                        "tps": (fit.throughput.decode_tps * 100.0).round() / 100.0,
                        "quality": (fit.quality.quantized * 10.0).round() / 10.0,
                        "verdict": format!("{:?}", fit.verdict).to_lowercase(),
                        "weights": fit.memory.weights,
                        "cache": fit.memory.kv_cache,
                        "other": fit.memory.activations.total() + fit.memory.runtime_overhead,
                        "headroom": fit.memory.headroom,
                        // The pool this model was placed in, which is what the
                        // bar must be drawn against. The machine's largest pool
                        // is a different thing: on a machine with a card it is
                        // usually system memory, and a 24 GB card shown against
                        // 56 GB of RAM would be a lie by framing.
                        "pool": fit.pool.usable_bytes,
                        "where": fit.run_mode.label(),
                    });
                    Some((fit.score, entry))
                })
                .collect();

            ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
            serde_json::json!({
                "tab": machine.tab,
                "name": machine.name,
                "detail": machine.detail,
                "runs": ranked.len(),
                "top": ranked.iter().take(3).map(|(_, e)| e).collect::<Vec<_>>(),
            })
        })
        .collect();

    let document = serde_json::json!({
        "generated": catalog.generated,
        "context": 8192,
        "machines": machines,
    });

    // Three rows per machine is what the page draws; the rest would be dead
    // weight in a file that ships inside the HTML.
    let compact = serde_json::to_string(&document).expect("the document serialises");

    let page = std::path::Path::new("site/index.html");
    let html = std::fs::read_to_string(page)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", page.display()));

    let open = "<script id=\"demo\" type=\"application/json\">";
    let start = html
        .find(open)
        .map(|i| i + open.len())
        .expect("site/index.html has no demo data block to write into");
    let end = html[start..]
        .find("</script>")
        .map(|i| start + i)
        .expect("the demo data block is not closed");

    let updated = format!(
        "{}
{compact}
{}",
        &html[..start],
        &html[end..]
    );
    if updated == html {
        eprintln!("{} is already up to date", page.display());
        return;
    }
    std::fs::write(page, &updated)
        .unwrap_or_else(|e| panic!("could not write {}: {e}", page.display()));
    eprintln!(
        "wrote {} bytes of engine output into {}",
        compact.len(),
        page.display()
    );
}
