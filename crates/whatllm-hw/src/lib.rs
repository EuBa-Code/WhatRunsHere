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
    /// A device's bandwidth is unknown until something measures it.
    BandwidthUnknown {
        /// The device in question.
        device: String,
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
}

/// The instruction set this binary was built for.
fn cpu_arch() -> CpuArch {
    match std::env::consts::ARCH {
        "x86_64" => CpuArch::X86_64,
        "aarch64" => CpuArch::Aarch64,
        _ => CpuArch::Other,
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

    let memory = HostMemory {
        total_bytes: sys.total_memory(),
        available_bytes: sys.available_memory(),
        // Left unset: no platform reports these reliably, and the probe
        // measures the thing they would only have let us estimate.
        channels: None,
        speed_mts: None,
    };

    let mut notes = Vec::new();
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
    if raw.is_empty() && accelerators.is_empty() {
        notes.push(DetectionNote::PlatformUnsupported {
            target: format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        });
    }

    for adapter in raw {
        let vendor = adapter.pci_vendor.map_or_else(
            || classify::vendor_from_name(&adapter.name),
            classify::vendor_from_pci_id,
        );
        let class = classify::classify(&adapter.name, vendor);

        match class {
            AdapterClass::Virtual => {
                notes.push(DetectionNote::IgnoredVirtualAdapter { name: adapter.name });
            }
            // NVML already described these, with real free-memory figures the
            // registry cannot supply.
            _ if vendor == whatllm_core::hardware::Vendor::Nvidia && !accelerators.is_empty() => {}
            AdapterClass::Integrated => {
                notes.push(DetectionNote::IntegratedGraphics {
                    name: adapter.name.clone(),
                    claimed_vram_bytes: adapter.claimed_vram_bytes,
                });
                accelerators.push(Accelerator {
                    index: accelerators.len() as u32,
                    name: adapter.name,
                    vendor,
                    backend: classify::backend_for(vendor, class),
                    // The pool is system memory, reported once.
                    total_bytes: memory.total_bytes,
                    reserved_bytes: memory.total_bytes.saturating_sub(memory.available_bytes),
                    unified: true,
                    drives_display: true,
                    // Integrated graphics read the same DIMMs the CPU does, so
                    // there is no separate figure to quote. The probe supplies
                    // the one that matters.
                    peak_bandwidth_gbps: None,
                    peak_tflops_fp16: None,
                });
            }
            AdapterClass::Discrete => {
                notes.push(DetectionNote::BandwidthUnknown {
                    device: adapter.name.clone(),
                });
                accelerators.push(Accelerator {
                    index: accelerators.len() as u32,
                    name: adapter.name,
                    vendor,
                    backend: classify::backend_for(vendor, class),
                    total_bytes: adapter.claimed_vram_bytes.unwrap_or(0),
                    reserved_bytes: 0,
                    unified: false,
                    drives_display: false,
                    peak_bandwidth_gbps: None,
                    peak_tflops_fp16: None,
                });
            }
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
