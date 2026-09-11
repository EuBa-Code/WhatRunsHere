//! NVIDIA devices, through NVML.
//!
//! Worth reading for one reason: this module contains no table of graphics
//! cards. Memory bandwidth is derived from the bus width and memory clock the
//! device itself reports, which reproduces the published figure for every part
//! NVIDIA has shipped (384 bits at 10501 MHz is the 1008 GB/s on the RTX 4090's
//! spec sheet, 5120 bits at 1593 MHz is the A100 80GB's 2039 GB/s) and, more to
//! the point, will keep reproducing it for parts that do not exist yet.
//!
//! The same goes for how much memory is free. NVML reports what is actually
//! allocated right now, which on a card driving a display is most of a gigabyte
//! before anything of yours loads. That is a measurement, not an allowance.

use nvml_wrapper::enum_wrappers::device::Clock;
use nvml_wrapper::Nvml;
use whatllm_core::hardware::{Accelerator, Backend, Vendor};

/// Why NVIDIA detection produced nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NvidiaError {
    /// The NVML library could not be loaded, which is the normal state on a
    /// machine with no NVIDIA driver installed.
    NotAvailable(String),
    /// NVML loaded but refused to answer.
    QueryFailed(String),
}

impl std::fmt::Display for NvidiaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAvailable(why) => write!(f, "NVML unavailable: {why}"),
            Self::QueryFailed(why) => write!(f, "NVML query failed: {why}"),
        }
    }
}

impl std::error::Error for NvidiaError {}

/// Whether this is one of NVIDIA's unified-memory parts.
///
/// Most NVIDIA hardware has its own memory. The Grace Blackwell superchips
/// (GB10, as in DGX Spark) and the Jetson line do not: they share one pool with
/// the CPU, and treating their reported memory as dedicated video memory both
/// double-counts it against system RAM and mis-models where a spilled layer
/// would go.
///
/// The list comes from llmfit, which learned these cases in the field.
///
/// A list is the wrong shape for this and will lag new parts: a unified part it
/// does not know gets treated as discrete, and its memory is then counted
/// twice, once as video memory and once as system RAM. The obvious cross-check
/// does not rescue it either. "Reported memory equals system RAM" describes a
/// DGX Spark and equally describes a 24 GB machine with a 24 GB card, and
/// getting that backwards would mis-model a discrete card badly. Until NVML
/// exposes the addressing mode through a safe binding, a list is all there is.
fn is_unified_memory_part(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    // Grace Blackwell superchips.
    lower.contains("gb10")
        || lower.contains("gb20")
        // Jetson and Tegra integrated GPUs run the legacy driver stack, which
        // suffixes the reported name this way.
        || lower.contains("nvgpu")
        || lower.contains("jetson")
        || lower.contains("orin")
        || lower.contains("tegra")
}

/// Peak memory bandwidth in GB/s from the bus width and effective clock.
///
/// NVML reports the memory clock already doubled for the data rate on GDDR
/// parts, and the factor of two below covers the remaining double-pumping. The
/// result matches published figures across GDDR6, GDDR6X and HBM.
fn bandwidth_gbps(bus_width_bits: u32, memory_clock_mhz: u32) -> f64 {
    f64::from(bus_width_bits) / 8.0 * f64::from(memory_clock_mhz) * 2.0 / 1000.0
}

/// Half-precision tensor throughput in TFLOPs, estimated from shader
/// throughput.
///
/// Shader FLOPs follow exactly from core count and clock. Tensor throughput
/// does not (the ratio is an architectural choice) but it has sat at roughly
/// two times dense fp16-with-fp32-accumulate across recent consumer parts, and
/// an estimate that tracks the hardware beats no time-to-first-token estimate
/// at all. Superseded the moment a probe measures the real thing.
fn tensor_tflops_fp16(cores: u32, graphics_clock_mhz: u32) -> f64 {
    let shader_tflops = f64::from(cores) * f64::from(graphics_clock_mhz) * 2.0 / 1e6;
    shader_tflops * 2.0
}

/// Every NVIDIA device NVML can see.
///
/// # Errors
/// Returns [`NvidiaError::NotAvailable`] when there is no NVIDIA driver, which
/// is not a failure so much as an answer.
pub fn detect() -> Result<Vec<Accelerator>, NvidiaError> {
    let nvml = Nvml::init().map_err(|e| NvidiaError::NotAvailable(e.to_string()))?;
    let count = nvml
        .device_count()
        .map_err(|e| NvidiaError::QueryFailed(e.to_string()))?;

    let mut found = Vec::new();
    for index in 0..count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };
        let name = device
            .name()
            .unwrap_or_else(|_| format!("NVIDIA device {index}"));
        let memory = device
            .memory_info()
            .map_err(|e| NvidiaError::QueryFailed(e.to_string()))?;

        // Everything not free is unavailable to us, whether the driver reserved
        // it or another process did.
        let reserved = memory.total.saturating_sub(memory.free);

        let peak_bandwidth_gbps = match (
            device.memory_bus_width(),
            device.max_clock_info(Clock::Memory),
        ) {
            (Ok(bus), Ok(clock)) if bus > 0 && clock > 0 => Some(bandwidth_gbps(bus, clock)),
            _ => None,
        };

        let peak_tflops_fp16 = match (device.num_cores(), device.max_clock_info(Clock::Graphics)) {
            (Ok(cores), Ok(clock)) if cores > 0 && clock > 0 => {
                Some(tensor_tflops_fp16(cores, clock))
            }
            _ => None,
        };

        let unified = is_unified_memory_part(&name);
        found.push(Accelerator {
            index,
            name,
            vendor: Vendor::Nvidia,
            backend: Backend::Cuda,
            total_bytes: memory.total,
            reserved_bytes: reserved,
            unified,
            // A card with memory already committed at idle is almost certainly
            // painting something. This is inference from a measurement rather
            // than a display-topology query, and it is right far more often
            // than assuming device zero drives the desktop.
            drives_display: reserved > 128 * 1024 * 1024,
            peak_bandwidth_gbps,
            peak_tflops_fp16,
        });
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandwidth_reproduces_published_figures() {
        // RTX 4090: 384-bit GDDR6X, NVML reports 10501 MHz. Spec: 1008 GB/s.
        let ada = bandwidth_gbps(384, 10_501);
        assert!((ada - 1008.0).abs() < 2.0, "got {ada:.1} GB/s");

        // A100 80GB: 5120-bit HBM2e at 1593 MHz. Spec: 2039 GB/s.
        let hopper = bandwidth_gbps(5120, 1593);
        assert!((hopper - 2039.0).abs() < 5.0, "got {hopper:.1} GB/s");

        // RTX 3090: 384-bit GDDR6X at 9751 MHz. Spec: 936 GB/s.
        let ampere = bandwidth_gbps(384, 9_751);
        assert!((ampere - 936.0).abs() < 3.0, "got {ampere:.1} GB/s");
    }

    #[test]
    fn tensor_throughput_lands_near_the_published_figure() {
        // RTX 4090: 16384 cores at 2520 MHz gives 82.6 TFLOPs of shader work
        // and a published 165 TFLOPs of dense fp16 tensor throughput.
        let ada = tensor_tflops_fp16(16_384, 2_520);
        assert!((ada - 165.0).abs() < 15.0, "got {ada:.1} TFLOPs");
    }

    #[test]
    fn unified_memory_parts_are_recognised() {
        for name in [
            "NVIDIA GB10",
            "NVIDIA GB200",
            "Orin (nvgpu)",
            "NVIDIA Jetson AGX Orin",
        ] {
            assert!(is_unified_memory_part(name), "{name} shares system memory");
        }
        for name in [
            "NVIDIA GeForce RTX 4090",
            "NVIDIA A100-SXM4-80GB",
            "NVIDIA RTX 5880 Ada Generation",
        ] {
            assert!(!is_unified_memory_part(name), "{name} has its own memory");
        }
    }

    #[test]
    fn an_absent_driver_is_reported_as_absent_not_as_a_failure() {
        // On a machine without NVIDIA hardware this is the expected path, and
        // it must not panic or be mistaken for a broken installation.
        match detect() {
            Ok(devices) => {
                for device in &devices {
                    assert_eq!(device.vendor, Vendor::Nvidia);
                    assert!(device.total_bytes > 0);
                    assert!(device.usable_bytes() <= device.total_bytes);
                }
            }
            Err(NvidiaError::NotAvailable(_)) => {}
            Err(other) => panic!("unexpected NVML failure: {other}"),
        }
    }
}
