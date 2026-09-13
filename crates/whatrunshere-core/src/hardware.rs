//! A description of the machine, and of the memory pools a model can go into.
//!
//! Detection lives in a separate crate; this one only defines what is detected,
//! so the models above it stay pure and testable.
//!
//! One distinction is worth stating explicitly, because it is the source of a
//! great deal of confusion: a 24 GB card does not offer 24 GB. Whatever drives
//! your monitors has already taken a slice for the compositor and every window
//! on screen, and on Windows the display driver reserves more on top. Sizing
//! against the sticker number tells people a model will load when it will not.
//! [`Accelerator::usable_bytes`] is the number that matters, and it is always
//! the one used downstream.

use crate::perf::DeviceThroughput;
use serde::{Deserialize, Serialize};

/// Who made the silicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Vendor {
    /// NVIDIA.
    Nvidia,
    /// AMD.
    Amd,
    /// Intel.
    Intel,
    /// Apple.
    Apple,
    /// Qualcomm.
    Qualcomm,
    /// Anything else.
    Other,
}

/// The compute path an accelerator is reached through.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// NVIDIA CUDA.
    Cuda,
    /// AMD `ROCm` / HIP.
    Rocm,
    /// Apple Metal.
    Metal,
    /// Vulkan compute.
    Vulkan,
    /// Intel oneAPI / SYCL.
    Sycl,
    /// `DirectML` on Windows.
    DirectMl,
    /// No accelerator: the CPU does the work.
    Cpu,
}

impl Backend {
    /// A human-readable name.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
            Self::Metal => "Metal",
            Self::Vulkan => "Vulkan",
            Self::Sycl => "oneAPI",
            Self::DirectMl => "DirectML",
            Self::Cpu => "CPU",
        }
    }

    /// Whether this backend runs on a discrete or integrated accelerator.
    pub const fn is_accelerated(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

/// Instruction set family, which sets the CPU fallback's ballpark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuArch {
    /// x86-64.
    X86_64,
    /// 64-bit ARM.
    Aarch64,
    /// Anything else.
    Other,
}

/// The host processor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Marketing name as reported by the OS.
    pub brand: String,
    /// Physical cores.
    pub physical_cores: u32,
    /// Logical processors, including simultaneous multithreading.
    pub logical_cores: u32,
    /// Instruction set family.
    pub arch: CpuArch,
}

/// System memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostMemory {
    /// Installed capacity.
    pub total_bytes: u64,
    /// Free right now. What a model can actually claim without swapping.
    pub available_bytes: u64,
    /// Populated memory channels, when the platform reports them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channels: Option<u32>,
    /// Rated transfer rate in MT/s, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed_mts: Option<u32>,
    /// Memory firmware handed to an integrated GPU before the kernel started,
    /// and which the operating system therefore does not report at all.
    ///
    /// Present only where it was measured, as the difference between the DIMM
    /// capacity firmware enumerates and the total the operating system
    /// manages. It is recorded rather than merely folded into the totals
    /// because it is also the evidence that the adapter claiming that much
    /// memory is an integrated part and not a card: see
    /// [`crate::hardware`]'s classification counterpart in `whatrunshere-hw`.
    ///
    /// [`Self::total_bytes`] and [`Self::available_bytes`] already include it.
    /// A model can use this memory (that is what it was carved out for), so
    /// leaving it out of both would describe a machine that cannot run what it
    /// plainly can.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uma_carveout_bytes: Option<u64>,
}

impl HostMemory {
    /// Bytes a model can claim with the machine otherwise quiet.
    ///
    /// [`Self::available_bytes`] says what is free this second, which swings
    /// with whatever happens to be open. Answering "what can this machine run"
    /// from it means the answer changes when a browser opens, which is not a
    /// property anyone wants from a planning tool. This is installed capacity
    /// less a reserve for the operating system and its caches, floored at
    /// whatever is genuinely free right now.
    pub fn claimable_bytes(&self) -> u64 {
        const MINIMUM_RESERVE: u64 = 3 * 1024 * 1024 * 1024;
        let reserve = (self.total_bytes / 8).max(MINIMUM_RESERVE);
        self.total_bytes
            .saturating_sub(reserve)
            .max(self.available_bytes)
    }

    /// Theoretical peak bandwidth from channel count and transfer rate.
    ///
    /// A DDR channel is 64 bits, so each channel carries eight bytes per
    /// transfer. Real code sees well under this; the discount is applied where
    /// the number is consumed, not here.
    pub fn peak_bandwidth_gbps(&self) -> Option<f64> {
        let channels = f64::from(self.channels?);
        let rate = f64::from(self.speed_mts?);
        Some(channels * rate * 8.0 / 1000.0)
    }
}

/// One GPU, NPU or integrated accelerator.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Accelerator {
    /// Index within the machine, as the driver enumerates it.
    pub index: u32,
    /// Device name as reported by the driver.
    pub name: String,
    /// Manufacturer.
    pub vendor: Vendor,
    /// Compute path.
    pub backend: Backend,
    /// Memory attached to the device, or the share of unified memory the
    /// platform will let a compute job take.
    pub total_bytes: u64,
    /// Memory already committed: the desktop compositor, other processes, and
    /// on Windows the display driver's own reservation.
    ///
    /// Measured wherever the driver will report it rather than assumed.
    #[serde(default)]
    pub reserved_bytes: u64,
    /// Whether device and host memory are the same physical pool, as on Apple
    /// Silicon and integrated graphics.
    #[serde(default)]
    pub unified: bool,
    /// Whether this device is driving a display.
    #[serde(default)]
    pub drives_display: bool,
    /// Datasheet memory bandwidth, used only when no probe has run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_bandwidth_gbps: Option<f64>,
    /// Datasheet half-precision throughput, used only when no probe has run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_tflops_fp16: Option<f64>,
}

impl Accelerator {
    /// The device name with trademark marks removed. See [`display_name`].
    pub fn display_name(&self) -> String {
        display_name(&self.name)
    }

    /// Memory a model can actually claim on this device.
    pub const fn usable_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.reserved_bytes)
    }

    /// Throughput derived from the datasheet, when there is one.
    pub fn vendor_throughput(&self) -> Option<DeviceThroughput> {
        let bandwidth = self.peak_bandwidth_gbps?;
        Some(DeviceThroughput::from_vendor_spec(
            bandwidth,
            self.peak_tflops_fp16,
        ))
    }
}

/// A driver-reported name, as it should be shown to a person.
///
/// Drivers write trademark marks into the middle of product names. Windows
/// reports `Intel(R) Core(TM) 7 150U` and `AMD Radeon(TM) 780M Graphics`, and
/// they are noise in every context but a legal one. Nobody writes their own
/// processor's name that way, so printing it back is the surest sign a tool is
/// echoing a string it never looked at.
///
/// Only the marks and the resulting double spaces are removed. The name is not
/// otherwise rewritten: an abbreviation table would eventually shorten a part
/// it had never seen into something wrong, and the name is also the thing
/// someone searches for when a detection looks off.
///
/// The raw string stays in [`Accelerator::name`] and in the detection's captured
/// adapters, so nothing needed for a bug report is lost.
pub fn display_name(raw: &str) -> String {
    let mut folded = raw.to_owned();
    for mark in ["(R)", "(r)", "(TM)", "(tm)", "(C)", "(c)", "®", "™", "©"] {
        folded = folded.replace(mark, " ");
    }
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Which physical memory a pool refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PoolKind {
    /// Dedicated accelerator memory.
    Device,
    /// System memory, reached by the CPU.
    Host,
    /// One pool serving both, as on Apple Silicon.
    Unified,
}

/// Somewhere a model's bytes can live.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryPool {
    /// A label for the pool, such as the device name.
    pub label: String,
    /// What kind of memory it is.
    pub kind: PoolKind,
    /// Bytes a model can claim.
    pub usable_bytes: u64,
    /// Indices of the accelerators backing it, empty for host memory.
    pub devices: Vec<u32>,
}

/// The whole machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemProfile {
    /// Host processor.
    pub cpu: CpuInfo,
    /// System memory.
    pub memory: HostMemory,
    /// Every accelerator found, in driver order.
    pub accelerators: Vec<Accelerator>,
    /// Operating system name and version, for reproducing reports.
    pub os: String,
}

impl SystemProfile {
    /// Whether anything beyond the CPU is available.
    pub fn has_accelerator(&self) -> bool {
        !self.accelerators.is_empty()
    }

    /// The accelerator with the most usable memory.
    pub fn largest_accelerator(&self) -> Option<&Accelerator> {
        self.accelerators.iter().max_by_key(|a| a.usable_bytes())
    }

    /// The backend a model would run on by default.
    pub fn primary_backend(&self) -> Backend {
        self.largest_accelerator()
            .map_or(Backend::Cpu, |a| a.backend)
    }

    /// Combined usable memory across every accelerator.
    ///
    /// Only meaningful when the devices can actually be pooled. See
    /// [`Self::can_pool_devices`].
    pub fn aggregate_accelerator_bytes(&self) -> u64 {
        self.accelerators
            .iter()
            .map(Accelerator::usable_bytes)
            .sum()
    }

    /// Whether the accelerators can hold one model between them.
    ///
    /// Splitting a model across cards needs them to share a backend. Mixing an
    /// NVIDIA card with an integrated Intel one does not give you their sum,
    /// however encouraging the arithmetic looks.
    pub fn can_pool_devices(&self) -> bool {
        let mut backends = self.accelerators.iter().map(|a| a.backend);
        let Some(first) = backends.next() else {
            return false;
        };
        first.is_accelerated() && backends.all(|b| b == first)
    }

    /// Every pool a model could be placed in, largest first.
    ///
    /// Unified-memory machines get a single pool rather than a device pool and
    /// a host pool that double-count the same silicon.
    pub fn pools(&self) -> Vec<MemoryPool> {
        let mut pools = Vec::new();

        if let Some(unified) = self.accelerators.iter().find(|a| a.unified) {
            pools.push(MemoryPool {
                label: unified.name.clone(),
                kind: PoolKind::Unified,
                // Two ceilings, and the lower one binds. `total_bytes` is what
                // the platform will let a compute job hold, which on macOS is
                // well under the machine's memory and everywhere else is the
                // machine's memory. `claimable_bytes` is what is left after a
                // reserve for the operating system.
                //
                // Deliberately not `usable_bytes()`: that subtracts whatever
                // is committed this second, and an answer to "what can this
                // machine run" should not change because a browser opened.
                usable_bytes: unified.total_bytes.min(self.memory.claimable_bytes()),
                devices: vec![unified.index],
            });
            return pools;
        }

        for accelerator in &self.accelerators {
            pools.push(MemoryPool {
                label: accelerator.name.clone(),
                kind: PoolKind::Device,
                usable_bytes: accelerator.usable_bytes(),
                devices: vec![accelerator.index],
            });
        }

        if self.accelerators.len() > 1 && self.can_pool_devices() {
            pools.push(MemoryPool {
                label: format!("{} devices combined", self.accelerators.len()),
                kind: PoolKind::Device,
                usable_bytes: self.aggregate_accelerator_bytes(),
                devices: self.accelerators.iter().map(|a| a.index).collect(),
            });
        }

        pools.push(MemoryPool {
            label: "System memory".to_owned(),
            kind: PoolKind::Host,
            usable_bytes: self.memory.claimable_bytes(),
            devices: Vec::new(),
        });

        pools.sort_by_key(|p| std::cmp::Reverse(p.usable_bytes));
        pools
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn trademark_marks_are_stripped_without_rewriting_the_name() {
        use super::display_name;

        // The strings Windows and Linux actually report.
        assert_eq!(
            display_name("Intel(R) Core(TM) 7 150U"),
            "Intel Core 7 150U"
        );
        assert_eq!(
            display_name("AMD Radeon(TM) 780M Graphics"),
            "AMD Radeon 780M Graphics"
        );
        assert_eq!(
            display_name("NVIDIA® GeForce RTX 4090"),
            "NVIDIA GeForce RTX 4090"
        );
        assert_eq!(
            display_name("Intel(R)  Arc(TM) A770 Graphics"),
            "Intel Arc A770 Graphics"
        );

        // Names with nothing to strip are returned as they came, and a model
        // number that happens to contain a letter is not a trademark mark.
        assert_eq!(display_name("Apple M3 Max"), "Apple M3 Max");
        assert_eq!(
            display_name("NVIDIA A100-SXM4-80GB"),
            "NVIDIA A100-SXM4-80GB"
        );
        assert_eq!(
            display_name("AMD Ryzen AI MAX+ 395"),
            "AMD Ryzen AI MAX+ 395"
        );
    }

    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn cpu() -> CpuInfo {
        CpuInfo {
            brand: "AMD Ryzen 9 7950X".to_owned(),
            physical_cores: 16,
            logical_cores: 32,
            arch: CpuArch::X86_64,
        }
    }

    fn gpu(index: u32, name: &str, gib: u64, drives_display: bool) -> Accelerator {
        Accelerator {
            index,
            name: name.to_owned(),
            vendor: Vendor::Nvidia,
            backend: Backend::Cuda,
            total_bytes: gib * GIB,
            // A display-driving card on Windows loses roughly a gigabyte.
            reserved_bytes: if drives_display { GIB } else { 0 },
            unified: false,
            drives_display,
            peak_bandwidth_gbps: Some(1008.0),
            peak_tflops_fp16: Some(165.0),
        }
    }

    fn workstation(accelerators: Vec<Accelerator>) -> SystemProfile {
        SystemProfile {
            cpu: cpu(),
            memory: HostMemory {
                total_bytes: 64 * GIB,
                available_bytes: 48 * GIB,
                channels: Some(2),
                speed_mts: Some(5600),
                uma_carveout_bytes: None,
            },
            accelerators,
            os: "Windows 11".to_owned(),
        }
    }

    #[test]
    fn a_display_driving_card_offers_less_than_its_sticker_size() {
        let card = gpu(0, "RTX 4090", 24, true);
        assert_eq!(card.total_bytes, 24 * GIB);
        assert_eq!(card.usable_bytes(), 23 * GIB);
    }

    #[test]
    fn host_bandwidth_follows_from_channels_and_transfer_rate() {
        let memory = HostMemory {
            total_bytes: 64 * GIB,
            available_bytes: 48 * GIB,
            channels: Some(2),
            speed_mts: Some(5600),
            uma_carveout_bytes: None,
        };
        // Two channels at 5600 MT/s carry 89.6 GB/s at peak.
        let bandwidth = memory.peak_bandwidth_gbps().expect("both fields present");
        assert!((bandwidth - 89.6).abs() < 0.1, "got {bandwidth}");

        let unknown = HostMemory {
            channels: None,
            ..memory
        };
        assert_eq!(unknown.peak_bandwidth_gbps(), None);
    }

    #[test]
    fn matching_cards_pool_and_mismatched_ones_do_not() {
        let matched = workstation(vec![
            gpu(0, "RTX 4090", 24, true),
            gpu(1, "RTX 4090", 24, false),
        ]);
        assert!(matched.can_pool_devices());
        assert_eq!(matched.aggregate_accelerator_bytes(), 47 * GIB);

        let mixed = workstation(vec![
            gpu(0, "RTX 4090", 24, true),
            Accelerator {
                vendor: Vendor::Intel,
                backend: Backend::Sycl,
                ..gpu(1, "Intel UHD Graphics", 2, false)
            },
        ]);
        assert!(
            !mixed.can_pool_devices(),
            "cards on different backends must not be summed"
        );
    }

    #[test]
    fn pooling_offers_the_combined_option_only_when_it_is_real() {
        let matched = workstation(vec![
            gpu(0, "RTX 4090", 24, true),
            gpu(1, "RTX 4090", 24, false),
        ]);
        let pools = matched.pools();
        let combined = pools
            .iter()
            .find(|p| p.devices.len() == 2)
            .expect("two cards on the same backend must offer a combined pool");
        assert_eq!(combined.usable_bytes, 47 * GIB);
        assert_eq!(
            pools
                .iter()
                .filter(|p| p.kind == PoolKind::Device)
                .map(|p| p.usable_bytes)
                .max(),
            Some(47 * GIB),
            "the combined pool is the largest device pool"
        );
        assert!(
            pools
                .windows(2)
                .all(|w| w[0].usable_bytes >= w[1].usable_bytes),
            "pools must be ordered by capacity"
        );

        let single = workstation(vec![gpu(0, "RTX 4090", 24, true)]);
        assert!(
            single.pools().iter().all(|p| p.devices.len() <= 1),
            "one card cannot be combined with itself"
        );
    }

    #[test]
    fn the_planning_figure_does_not_swing_with_whatever_is_open() {
        let quiet = HostMemory {
            total_bytes: 32 * GIB,
            available_bytes: 28 * GIB,
            channels: None,
            speed_mts: None,
            uma_carveout_bytes: None,
        };
        let busy = HostMemory {
            available_bytes: 6 * GIB,
            ..quiet
        };
        assert_eq!(
            quiet.claimable_bytes(),
            busy.claimable_bytes(),
            "a busy machine must not look like a smaller one"
        );
        assert_eq!(quiet.claimable_bytes(), 28 * GIB);

        // A machine with genuinely more free than the reserve implies keeps the
        // larger figure rather than being talked down to it.
        let generous = HostMemory {
            available_bytes: 31 * GIB,
            ..quiet
        };
        assert_eq!(generous.claimable_bytes(), 31 * GIB);
    }

    #[test]
    fn unified_memory_is_one_pool_not_two() {
        let mac = SystemProfile {
            cpu: CpuInfo {
                brand: "Apple M3 Max".to_owned(),
                physical_cores: 16,
                logical_cores: 16,
                arch: CpuArch::Aarch64,
            },
            memory: HostMemory {
                total_bytes: 128 * GIB,
                available_bytes: 110 * GIB,
                channels: None,
                speed_mts: None,
                uma_carveout_bytes: None,
            },
            accelerators: vec![Accelerator {
                index: 0,
                name: "Apple M3 Max".to_owned(),
                vendor: Vendor::Apple,
                backend: Backend::Metal,
                total_bytes: 128 * GIB,
                reserved_bytes: 18 * GIB,
                unified: true,
                drives_display: true,
                peak_bandwidth_gbps: Some(400.0),
                peak_tflops_fp16: None,
            }],
            os: "macOS 15".to_owned(),
        };
        let pools = mac.pools();
        assert_eq!(pools.len(), 1, "unified memory must not be counted twice");
        assert_eq!(pools[0].kind, PoolKind::Unified);
    }

    #[test]
    fn a_machine_without_an_accelerator_still_offers_its_ram() {
        let headless = workstation(Vec::new());
        assert!(!headless.has_accelerator());
        assert_eq!(headless.primary_backend(), Backend::Cpu);
        let pools = headless.pools();
        assert_eq!(pools.len(), 1);
        assert_eq!(pools[0].kind, PoolKind::Host);
        // 64 GiB installed, an eighth reserved for the system.
        assert_eq!(pools[0].usable_bytes, 56 * GIB);
    }
}
