//! What this machine is.
//!
//! Detection is separated from measurement on purpose. This crate answers
//! "what is here and how much memory can a model claim", quickly and without
//! running anything. How fast that memory actually is comes from
//! `whatllm-probe`, and is a different question with a different answer.
//!
//! Two things it refuses to do, both of which cost accuracy elsewhere:
//!
//! - **It does not count virtual display adapters.** A machine reached over
//!   remote desktop reports one, and it computes nothing.
//! - **It does not believe the video memory figure for integrated graphics.**
//!   Windows will tell you an integrated GPU has 2 GB. It has none; it borrows
//!   system RAM. Reporting the borrowed pool once, as unified memory, is the
//!   only accounting that does not either understate or double-count it.
//!
//! Everything it could not establish is reported in [`Detection::notes`] rather
//! than filled in with a plausible number.

#![forbid(unsafe_code)]

pub mod classify;
pub mod nvidia;
pub mod platform;

use classify::AdapterClass;
use serde::{Deserialize, Serialize};
use sysinfo::System;
use whatllm_core::hardware::{Accelerator, Backend, CpuArch, CpuInfo, HostMemory, SystemProfile};
use whatllm_core::perf::{Calibration, Confidence, DeviceThroughput};

/// Something detection could not establish, or deliberately disregarded.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "note", rename_all = "snake_case")]
pub enum DetectionNote {
    /// A display adapter was found that cannot run compute work.
    IgnoredVirtualAdapter {
        /// What the driver called it.
        name: String,
    },
    /// Integrated graphics too old for any inference runtime to use.
    LegacyIntegratedGraphics {
        /// What the driver called it.
        name: String,
    },
    /// An integrated GPU was found; its pool is system memory.
    IntegratedGraphics {
        /// What the driver called it.
        name: String,
        /// The video memory figure the system claimed, which is an aperture
        /// size rather than memory the device owns.
        claimed_vram_bytes: Option<u64>,
    },
    /// No NVIDIA driver, which on most machines is simply the truth.
    NvidiaUnavailable {
        /// What NVML said.
        reason: String,
    },
    /// A card's memory could only be read from a field too narrow to hold it.
    MemorySizeUnreadable {
        /// The device in question.
        device: String,
    },
    /// A device's bandwidth is unknown until something measures it.
    BandwidthUnknown {
        /// The device in question.
        device: String,
    },
    /// Firmware kept memory back from the operating system for an integrated
    /// GPU, and it has been counted back in.
    FirmwareMemoryCarveout {
        /// How much, in bytes.
        bytes: u64,
        /// What the operating system reported without it, for comparison.
        os_reported_bytes: u64,
    },
    /// The platform will not let a compute job hold the whole shared pool.
    ComputeMemoryCapped {
        /// The device in question.
        device: String,
        /// The most it may hold, in bytes.
        bytes: u64,
        /// The pool it is drawn from, for comparison.
        pool_bytes: u64,
    },
    /// This platform has no adapter enumeration yet.
    PlatformUnsupported {
        /// The target the binary was built for.
        target: String,
    },
}

/// The machine, plus what could not be established about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// The profile the rest of the system reasons about.
    pub system: SystemProfile,
    /// Caveats, in the order they were found.
    pub notes: Vec<DetectionNote>,
    /// What each platform reported, before any of it was interpreted.
    ///
    /// The verdict is what the tool acts on; this is what produced it. A report
    /// of a misdetection is only actionable with both, and with both it can be
    /// replayed as a test. See [`classify_raw`].
    #[serde(default)]
    pub raw_adapters: Vec<platform::RawAdapter>,
}

/// Classify a captured adapter list without touching the machine.
///
/// The path a bug report takes back into the test suite: paste the
/// `raw_adapters` from a `doctor --json` dump into a test, and this returns the
/// same judgement the reporter's machine made.
///
/// `carveout_bytes` comes from the same dump's `system.memory`, and passing it
/// matters on exactly the machines a report is most likely to come from:
/// without it, a replay of a large integrated part reaches the opposite
/// verdict to the one being investigated.
pub fn classify_raw(
    adapters: &[platform::RawAdapter],
    carveout_bytes: Option<u64>,
) -> Vec<(String, AdapterClass)> {
    adapters
        .iter()
        .map(|adapter| {
            let vendor = adapter.pci_vendor.map_or_else(
                || classify::vendor_from_name(&adapter.name),
                classify::vendor_from_pci_id,
            );
            (
                adapter.name.clone(),
                classify::classify_with_carveout(
                    &adapter.name,
                    vendor,
                    adapter.claimed_vram_bytes,
                    carveout_bytes,
                ),
            )
        })
        .collect()
}

/// The instruction set this binary was built for.
fn cpu_arch() -> CpuArch {
    match std::env::consts::ARCH {
        "x86_64" => CpuArch::X86_64,
        "aarch64" => CpuArch::Aarch64,
        _ => CpuArch::Other,
    }
}

/// How much memory firmware kept from the operating system, if enough to be a
/// deliberate graphics carveout rather than ordinary reserve.
///
/// Every machine loses a little between the DIMMs and the kernel: ACPI tables,
/// the framebuffer the firmware itself was drawing to, memory holes. That is
/// tens or a couple of hundred megabytes, and it is not a carveout. A BIOS
/// graphics allocation is chosen from a menu whose smallest entry is a
/// gigabyte, so a gigabyte is where one stops being the other.
///
/// The development machine measured 315 MB of ordinary reserve against 32 GiB
/// installed, which is the shape this threshold has to clear.
fn uma_carveout_bytes(os_total: u64, installed: Option<u64>) -> Option<u64> {
    /// Smallest BIOS graphics allocation offered on any board that offers one.
    const SMALLEST_CARVEOUT: u64 = 1024 * 1024 * 1024;

    let gap = installed?.checked_sub(os_total)?;
    (gap >= SMALLEST_CARVEOUT).then_some(gap)
}

/// Turn one platform-reported adapter into an accelerator, or into a note
/// saying why it is not one.
///
/// Separate from [`detect`] so that a single adapter's interpretation can be
/// exercised on its own, which is what makes a captured report replayable.
fn interpret(
    adapter: platform::RawAdapter,
    index: u32,
    memory: &HostMemory,
    notes: &mut Vec<DetectionNote>,
) -> Option<Accelerator> {
    let vendor = adapter.pci_vendor.map_or_else(
        || classify::vendor_from_name(&adapter.name),
        classify::vendor_from_pci_id,
    );
    let class = classify::classify_with_carveout(
        &adapter.name,
        vendor,
        adapter.claimed_vram_bytes,
        memory.uma_carveout_bytes,
    );

    match class {
        AdapterClass::Virtual => {
            notes.push(DetectionNote::IgnoredVirtualAdapter { name: adapter.name });
            None
        }
        AdapterClass::LegacyIntegrated => {
            notes.push(DetectionNote::LegacyIntegratedGraphics { name: adapter.name });
            None
        }
        AdapterClass::Integrated => {
            notes.push(DetectionNote::IntegratedGraphics {
                name: adapter.name.clone(),
                claimed_vram_bytes: adapter.claimed_vram_bytes,
            });
            // The pool is system memory, reported once, but only as much of
            // it as the platform will let a compute job hold. macOS enforces
            // such a ceiling and nothing else does; where one is reported it
            // is still clamped to the memory that exists, since the sysctl
            // behind it can be set to a nonsense figure by hand.
            let pool = adapter
                .compute_memory_limit_bytes
                .map_or(memory.total_bytes, |limit| limit.min(memory.total_bytes));
            if pool < memory.total_bytes {
                notes.push(DetectionNote::ComputeMemoryCapped {
                    device: adapter.name.clone(),
                    bytes: pool,
                    pool_bytes: memory.total_bytes,
                });
            }
            Some(Accelerator {
                index,
                name: adapter.name,
                vendor,
                backend: classify::backend_for(vendor, class),
                total_bytes: pool,
                // What other processes hold, measured against the same pool:
                // subtracting a whole-machine figure from a capped one would
                // count the cap itself as memory in use.
                reserved_bytes: pool.saturating_sub(memory.available_bytes.min(pool)),
                unified: true,
                drives_display: true,
                // Integrated graphics read the same DIMMs the CPU does, so
                // there is no separate figure to quote. The probe supplies the
                // one that matters.
                peak_bandwidth_gbps: None,
                peak_tflops_fp16: None,
            })
        }
        AdapterClass::Discrete => {
            // A saturated field cannot say how much memory a card has, and
            // guessing four gigabytes for a card that has twenty-four would
            // reject every model it can comfortably run. A capacity that cannot
            // be read is left at zero, and the caller drops the device with a
            // note rather than sizing against a number that is not one.
            if adapter.vram_from_narrow_field {
                notes.push(DetectionNote::MemorySizeUnreadable {
                    device: adapter.name.clone(),
                });
            }
            notes.push(DetectionNote::BandwidthUnknown {
                device: adapter.name.clone(),
            });
            Some(Accelerator {
                index,
                name: adapter.name,
                vendor,
                backend: classify::backend_for(vendor, class),
                total_bytes: if adapter.vram_from_narrow_field {
                    0
                } else {
                    adapter.claimed_vram_bytes.unwrap_or(0)
                },
                reserved_bytes: 0,
                unified: false,
                drives_display: false,
                peak_bandwidth_gbps: None,
                peak_tflops_fp16: None,
            })
        }
    }
}

/// Inspect the machine. Fast, and runs nothing.
pub fn detect() -> Detection {
    let mut sys = System::new_all();
    sys.refresh_all();

    let logical = sys.cpus().len() as u32;
    let cpu = CpuInfo {
        brand: sys
            .cpus()
            .first()
            .map_or_else(|| "Unknown CPU".to_owned(), |c| c.brand().trim().to_owned()),
        physical_cores: System::physical_core_count()
            .map_or(logical.max(1), |n| n as u32)
            .max(1),
        logical_cores: logical.max(1),
        arch: cpu_arch(),
    };

    let os_total = sys.total_memory();
    let carveout = uma_carveout_bytes(os_total, platform::installed_memory_bytes());
    let memory = HostMemory {
        // Firmware handed the carveout to the integrated GPU before the kernel
        // started, so the operating system does not count it. A model can use
        // it (that is the entire point of it) so both figures get it back.
        total_bytes: os_total + carveout.unwrap_or(0),
        available_bytes: sys.available_memory() + carveout.unwrap_or(0),
        // Left unset: no platform reports these reliably, and the probe
        // measures the thing they would only have let us estimate.
        channels: None,
        speed_mts: None,
        uma_carveout_bytes: carveout,
    };

    let mut notes = Vec::new();
    if let Some(bytes) = carveout {
        notes.push(DetectionNote::FirmwareMemoryCarveout {
            bytes,
            os_reported_bytes: os_total,
        });
    }
    let mut accelerators = match nvidia::detect() {
        Ok(devices) => devices,
        Err(error) => {
            notes.push(DetectionNote::NvidiaUnavailable {
                reason: error.to_string(),
            });
            Vec::new()
        }
    };

    let raw = platform::adapters();
    let captured = raw.clone();
    if raw.is_empty() && accelerators.is_empty() {
        notes.push(DetectionNote::PlatformUnsupported {
            target: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        });
    }

    for adapter in raw {
        // NVML already described NVIDIA devices, with real free-memory figures
        // the registry cannot supply.
        let vendor = adapter.pci_vendor.map_or_else(
            || classify::vendor_from_name(&adapter.name),
            classify::vendor_from_pci_id,
        );
        if vendor == whatllm_core::hardware::Vendor::Nvidia && !accelerators.is_empty() {
            continue;
        }
        let index = accelerators.len() as u32;
        if let Some(accelerator) = interpret(adapter, index, &memory, &mut notes) {
            accelerators.push(accelerator);
        }
    }

    // A device whose memory nobody could establish cannot be sized against.
    accelerators.retain(|a| a.total_bytes > 0);

    Detection {
        system: SystemProfile {
            cpu,
            memory,
            accelerators,
            os: System::long_os_version()
                .or_else(System::name)
                .unwrap_or_else(|| std::env::consts::OS.to_owned()),
        },
        notes,
        raw_adapters: captured,
    }
}

/// Last-resort host bandwidth, in bytes per second, when nothing measured it.
///
/// Deliberately conservative. Being pleasantly surprised by a measurement is
/// better than promising a speed the machine cannot reach.
fn assumed_host_bandwidth(arch: CpuArch) -> f64 {
    match arch {
        // Apple Silicon's unified memory is in a different class from a typical
        // x86 desktop's two DDR channels.
        CpuArch::Aarch64 => 100e9,
        CpuArch::X86_64 | CpuArch::Other => 30e9,
    }
}

/// Build a calibration for this machine.
///
/// `measured_host_bytes_per_s` comes from `whatllm-probe` when it has run. Pass
/// `None` and every number downstream is labelled as the guess it is.
pub fn calibration(system: &SystemProfile, measured_host_bytes_per_s: Option<f64>) -> Calibration {
    let (host, host_confidence) = match measured_host_bytes_per_s {
        Some(measured) if measured > 0.0 => (
            DeviceThroughput::from_probe(measured, None),
            Confidence::Calibrated,
        ),
        _ => (
            DeviceThroughput {
                bandwidth_bytes_per_s: assumed_host_bandwidth(system.cpu.arch),
                compute_flops: None,
            },
            Confidence::Fallback,
        ),
    };

    let primary = system.largest_accelerator();
    let (accelerator, accelerator_confidence) = match primary {
        // Unified memory is the same DIMMs the CPU reads. Quoting a separate
        // figure for it would be inventing one.
        Some(device) if device.unified => (Some(host), host_confidence),
        Some(device) => match device.vendor_throughput() {
            Some(throughput) => (Some(throughput), Confidence::VendorSpec),
            None => (None, Confidence::Fallback),
        },
        None => (None, host_confidence),
    };

    let on_accelerator =
        accelerator.is_some() && primary.is_some_and(|d| d.backend != Backend::Cpu && !d.unified);

    Calibration {
        accelerator,
        host,
        // Measured by fitting the decode model to community benchmarks: the
        // intercept lands between 0.6 ms on an RTX 5070 and 3.9 ms on a 2016
        // Pascal card, with a median of 1.85. The 0.2 ms this used to assume
        // was nearly ten times too low, which flattered every small model.
        overhead_ms_per_token: if on_accelerator { 1.8 } else { 2.5 },
        // The weaker of the two decides: an estimate is only as trustworthy as
        // its least trustworthy input.
        source: host_confidence.max(accelerator_confidence),
    }
}

/// The operating system this process runs on, as the launch commands need it.
///
/// It decides how a path is quoted and whether a host runs here at all. Read
/// from the build rather than from the OS string detection reports, because a
/// command is typed into the shell of the machine the binary was built for.
pub const fn platform() -> whatllm_core::launch::Platform {
    use whatllm_core::launch::Platform;
    if cfg!(windows) {
        Platform::Windows
    } else if cfg!(target_os = "macos") {
        Platform::MacOs
    } else if cfg!(target_os = "linux") {
        Platform::Linux
    } else {
        Platform::Other
    }
}

/// Whether this machine is Apple Silicon, the one place MLX runs.
///
/// A binary built for it is running on it. One built for x86 and running
/// under Rosetta still sees the Apple accelerator detection found, so either
/// answer counts.
pub fn apple_silicon(system: &SystemProfile) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        || system
            .accelerators
            .iter()
            .any(|a| a.vendor == whatllm_core::hardware::Vendor::Apple)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_describes_the_machine_it_runs_on() {
        let found = detect();
        let system = &found.system;

        assert!(!system.cpu.brand.is_empty());
        assert!(system.cpu.logical_cores >= 1);
        assert!(system.cpu.physical_cores >= 1);
        assert!(
            system.cpu.physical_cores <= system.cpu.logical_cores,
            "{} physical cores against {} logical",
            system.cpu.physical_cores,
            system.cpu.logical_cores
        );
        assert!(system.memory.total_bytes > 0, "a machine has memory");
        assert!(system.memory.available_bytes <= system.memory.total_bytes);
        assert!(!system.os.is_empty());
    }

    #[test]
    fn every_reported_accelerator_is_usable() {
        for device in &detect().system.accelerators {
            assert!(!device.name.is_empty());
            assert!(device.total_bytes > 0, "{} reported no memory", device.name);
            assert!(device.usable_bytes() <= device.total_bytes);
            assert!(
                device.backend.is_accelerated(),
                "{} was kept but cannot compute",
                device.name
            );
        }
    }

    #[test]
    fn unified_memory_is_never_offered_twice() {
        let system = detect().system;
        if system.accelerators.iter().any(|a| a.unified) {
            let pools = system.pools();
            assert_eq!(
                pools.len(),
                1,
                "a unified machine has one pool, found {}",
                pools.len()
            );
            assert!(
                pools[0].usable_bytes <= system.memory.total_bytes,
                "the pool cannot exceed installed memory"
            );
        }
    }

    #[test]
    fn a_detection_carries_the_evidence_behind_its_verdict() {
        let found = detect();
        // Every accelerator kept, and every adapter dismissed, must be
        // traceable to something a platform actually reported.
        for accelerator in &found.system.accelerators {
            assert!(
                found
                    .raw_adapters
                    .iter()
                    .any(|raw| raw.name == accelerator.name)
                    || accelerator.vendor == whatllm_core::hardware::Vendor::Nvidia,
                "{} appears in the verdict with nothing behind it",
                accelerator.name
            );
        }
        // And the capture can be replayed without touching the machine.
        let replayed = classify_raw(&found.raw_adapters, found.system.memory.uma_carveout_bytes);
        assert_eq!(replayed.len(), found.raw_adapters.len());
    }

    #[test]
    fn a_captured_report_replays_to_the_same_judgement() {
        // The shape a bug report takes: the `raw_adapters` from a doctor dump,
        // pasted verbatim. This one is from the machine WhatLLM was built on.
        let captured: Vec<platform::RawAdapter> = serde_json::from_str(
            r#"[
              {"name":"Parsec Virtual Display Adapter","provider":"Parsec Cloud, Inc.",
               "pci_vendor":null,"claimed_vram_bytes":null,"vram_from_narrow_field":false},
              {"name":"Intel(R) Graphics","provider":"Intel Corporation",
               "pci_vendor":null,"claimed_vram_bytes":2147479552,"vram_from_narrow_field":false}
            ]"#,
        )
        .expect("a captured report parses");

        let judged = classify_raw(&captured, None);
        assert_eq!(judged[0].1, AdapterClass::Virtual);
        assert_eq!(judged[1].1, AdapterClass::Integrated);
    }

    #[test]
    fn a_graphics_carveout_is_told_apart_from_ordinary_firmware_reserve() {
        const GIB: u64 = 1024 * 1024 * 1024;

        // The development machine: 32 GiB installed, 315 MB kept by firmware.
        // Every machine loses something, and it is not a carveout.
        assert_eq!(
            uma_carveout_bytes(34_029_121_536, Some(34_359_738_368)),
            None
        );

        // A Ryzen AI MAX+ 395 with 96 GB configured for graphics.
        assert_eq!(
            uma_carveout_bytes(32 * GIB, Some(128 * GIB)),
            Some(96 * GIB)
        );

        // Without a firmware figure there is nothing to compare against, and
        // a total larger than the installed capacity is not arithmetic worth
        // trusting either.
        assert_eq!(uma_carveout_bytes(32 * GIB, None), None);
        assert_eq!(uma_carveout_bytes(32 * GIB, Some(16 * GIB)), None);
    }

    #[test]
    fn a_carved_out_machine_reports_one_pool_of_everything_installed() {
        const GIB: u64 = 1024 * 1024 * 1024;

        // What detect() builds on a 128 GB machine with 96 GB carved out: the
        // adapter is integrated, so the pool is the whole of system memory,
        // which now includes the carveout the OS could not see.
        let memory = HostMemory {
            total_bytes: 128 * GIB,
            available_bytes: 120 * GIB,
            channels: None,
            speed_mts: None,
            uma_carveout_bytes: Some(96 * GIB),
        };
        let adapter = platform::RawAdapter {
            name: "AMD Radeon(TM) Graphics".to_owned(),
            provider: None,
            pci_vendor: Some(0x1002),
            claimed_vram_bytes: Some(96 * GIB),
            vram_from_narrow_field: false,
            compute_memory_limit_bytes: None,
        };

        let mut notes = Vec::new();
        let device = interpret(adapter, 0, &memory, &mut notes).expect("an accelerator");
        assert!(device.unified, "a carveout is system memory, not a card");

        let profile = SystemProfile {
            cpu: CpuInfo {
                brand: "AMD Ryzen AI MAX+ 395".to_owned(),
                physical_cores: 16,
                logical_cores: 32,
                arch: CpuArch::X86_64,
            },
            memory,
            accelerators: vec![device],
            os: "Windows 11".to_owned(),
        };
        let pools = profile.pools();
        assert_eq!(pools.len(), 1, "one pool, not a card beside the RAM");
        assert!(
            pools[0].usable_bytes > 100 * GIB,
            "the machine can hold a 70B model; it offered {} bytes",
            pools[0].usable_bytes
        );
    }

    #[test]
    fn a_measurement_outranks_an_assumption() {
        let system = detect().system;
        let assumed = calibration(&system, None);
        let measured = calibration(&system, Some(60e9));
        assert_eq!(assumed.source, Confidence::Fallback);
        assert!(measured.source <= Confidence::Calibrated);
        assert!(measured.host.bandwidth_bytes_per_s > assumed.host.bandwidth_bytes_per_s);
    }

    #[test]
    fn a_unified_machine_quotes_one_bandwidth_for_both_devices() {
        let system = detect().system;
        if system.accelerators.iter().any(|a| a.unified) {
            let calibration = calibration(&system, Some(60e9));
            let accelerator = calibration
                .accelerator
                .expect("a unified machine has an accelerator");
            assert!(
                (accelerator.bandwidth_bytes_per_s - calibration.host.bandwidth_bytes_per_s).abs()
                    < 1.0,
                "unified memory must not be given two different speeds"
            );
        }
    }
}
