//! Deciding what a graphics adapter actually is from what the driver calls it.
//!
//! Two classifications matter and both are routinely got wrong:
//!
//! **Virtual adapters.** A machine reached over Parsec, RDP, Citrix or a
//! hypervisor console reports a display adapter that cannot run a single
//! kernel. Counting it as an accelerator is worse than ignoring it: the tool
//! confidently recommends a GPU placement onto a device that does not compute.
//!
//! **Integrated graphics.** Windows reports an aperture size for integrated
//! GPUs (2 GB is typical) and both WMI and the registry will hand it to you
//! as though it were video memory. It is not. Integrated graphics have no
//! memory of their own; they share system RAM. Sizing a model against that
//! 2 GB figure rejects models the machine could comfortably run, and sizing
//! against it *plus* system RAM double-counts the same silicon.

use whatllm_core::hardware::{Backend, Vendor};

/// What kind of device the driver is describing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterClass {
    /// A real accelerator with its own memory.
    Discrete,
    /// A real accelerator sharing system memory.
    Integrated,
    /// Integrated graphics old enough that no inference runtime will use them.
    ///
    /// Reported apart from [`Self::Integrated`] because the honest answer for
    /// such a part is that the model runs on the CPU. Calling it an accelerator
    /// and handing it the system memory pool describes a placement that does
    /// not exist: llmfit's issue #964 is a Coffee Lake UHD 630 reported as a
    /// 47 GB unified-memory GPU while every load ran at 100% CPU.
    LegacyIntegrated,
    /// Not an accelerator at all: a remote-desktop or hypervisor display.
    Virtual,
}

/// Dedicated memory above which a device is discrete whatever it is called.
///
/// Names are not reliable here. A 32 GB Instinct MI50 can report itself as
/// `AMD Radeon Graphics`, the same generic string an integrated Cezanne uses,
/// and llmfit hit exactly that (#638). Integrated parts report an aperture of
/// a gigabyte or two; nothing integrated owns five.
///
/// That last sentence stopped being true, which is why this floor is no longer
/// consulted alone. See [`classify_with_carveout`].
const DISCRETE_MEMORY_FLOOR: u64 = 5 * 1024 * 1024 * 1024;

/// Intel integrated parts predating the Xe architecture.
///
/// `HD Graphics` with any number, and `UHD Graphics` below 700, are Gen 9.5 or
/// older. From `UHD Graphics 7xx` and `Iris Xe` onwards the Vulkan and SYCL
/// back ends work. Erring towards CPU costs nothing, because an integrated GPU
/// reads the same memory the CPU does and the generation estimate is identical
/// either way. Erring the other way promises a placement that does not run.
fn is_pre_xe_intel(lower: &str) -> bool {
    let Some(rest) = lower.split_once("hd graphics").map(|(_, rest)| rest.trim()) else {
        return false;
    };
    let number: String = rest.chars().take_while(char::is_ascii_digit).collect();
    match number.parse::<u32>() {
        // Four digits is the old `HD Graphics 4000` series.
        Ok(model) if model >= 1000 => true,
        Ok(model) => model < 700,
        // A bare "HD Graphics" with no number is older still.
        Err(_) => true,
    }
}

/// Substrings that identify an adapter as a software or remoting display.
const VIRTUAL_MARKERS: &[&str] = &[
    "virtual display",
    "virtual monitor",
    "parsec",
    "citrix",
    "vmware svga",
    "virtualbox",
    "hyper-v video",
    "microsoft basic display",
    "microsoft basic render",
    "microsoft remote display",
    "rdpdd",
    "idd driver",
    "splashtop",
    "usb display",
    "displaylink",
    "meta virtual",
];

/// Substrings that identify an integrated part despite an otherwise
/// discrete-looking vendor and name.
const INTEGRATED_MARKERS: &[&str] = &[
    "uhd graphics",
    "hd graphics",
    "iris",
    "integrated",
    "vega 3",
    "vega 6",
    "vega 7",
    "vega 8",
    "vega 11",
    "radeon graphics",
    "890m",
    "880m",
    "780m",
    "760m",
    "680m",
    "660m",
];

/// Substrings that mark a discrete part whose name might otherwise be read as
/// integrated. Checked before the integrated markers.
const DISCRETE_OVERRIDES: &[&str] = &["arc a", "arc b", "radeon rx", "radeon pro", "firepro"];

/// Fold a driver-reported name into something worth matching against.
///
/// Drivers pepper names with trademark marks, as in "Intel(R) Arc(TM) A770",
/// and they land in the middle of exactly the phrases worth recognising,
/// turning "arc a770" into "arc(tm) a770". Stripping them once here is more
/// robust than keeping a marker list that enumerates every punctuation variant.
fn normalise(name: &str) -> String {
    let mut folded = name.to_ascii_lowercase();
    for mark in ["(r)", "(tm)", "(c)", "®", "™", "©"] {
        folded = folded.replace(mark, " ");
    }
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Classify an adapter from its reported name, vendor and reported memory.
///
/// `dedicated_bytes` is what the driver claims the device owns, when the
/// platform supplies it. It settles the cases a name cannot.
pub fn classify(name: &str, vendor: Vendor, dedicated_bytes: Option<u64>) -> AdapterClass {
    classify_with_carveout(name, vendor, dedicated_bytes, None)
}

/// Classify an adapter, given also what firmware kept back from the operating
/// system for graphics.
///
/// The memory floor above cannot stand alone once integrated parts are allowed
/// to be large. A Ryzen AI MAX+ can be configured in firmware to hand 96 GB of
/// its 128 GB to the integrated GPU, which then reports 96 GB of "video
/// memory" under a name, `AMD Radeon Graphics`, indistinguishable from the
/// generic string a discrete Instinct card uses. Both report far more than the
/// floor, and only one of them is a card.
///
/// A measured carveout separates them, because it is the same memory seen from
/// the other side: the operating system is missing exactly what the adapter
/// claims to own. A discrete card's memory is unrelated to anything the
/// operating system lost, so no discrete part is ever explained this way.
///
/// The comparison needs no tolerance in the direction that matters. The
/// carveout is measured as installed capacity less the operating system's
/// total, so it also contains the ordinary firmware reserve. It is at least
/// the graphics allocation and never less, and `<=` is therefore safe.
pub fn classify_with_carveout(
    name: &str,
    vendor: Vendor,
    dedicated_bytes: Option<u64>,
    carveout_bytes: Option<u64>,
) -> AdapterClass {
    let lower = normalise(name);

    if VIRTUAL_MARKERS.iter().any(|m| lower.contains(m)) {
        return AdapterClass::Virtual;
    }

    // Checked ahead of the memory floor so that a machine carrying both an
    // APU and a card keeps the card: a discrete part on a machine with a
    // larger carveout would otherwise be explained away by it.
    if DISCRETE_OVERRIDES.iter().any(|m| lower.contains(m)) {
        return AdapterClass::Discrete;
    }

    let explained_by_carveout = match (dedicated_bytes, carveout_bytes) {
        (Some(claimed), Some(carveout)) => claimed <= carveout,
        _ => false,
    };

    // Memory settles it before any name does. Nothing integrated owns this
    // much, unless the operating system is missing exactly that much, in
    // which case it does not own it either; it was lent it.
    if !explained_by_carveout && dedicated_bytes.is_some_and(|b| b >= DISCRETE_MEMORY_FLOOR) {
        return AdapterClass::Discrete;
    }

    if vendor == Vendor::Intel && is_pre_xe_intel(&lower) {
        return AdapterClass::LegacyIntegrated;
    }

    match vendor {
        // Apple has never shipped a discrete GPU in an Apple Silicon machine.
        Vendor::Apple | Vendor::Qualcomm => AdapterClass::Integrated,
        Vendor::Nvidia => AdapterClass::Discrete,
        Vendor::Intel | Vendor::Amd | Vendor::Other => {
            if INTEGRATED_MARKERS.iter().any(|m| lower.contains(m)) {
                AdapterClass::Integrated
            } else if vendor == Vendor::Intel {
                // Intel's bare "Intel(R) Graphics", used for recent mobile
                // parts, is integrated. Anything discrete carries "Arc".
                AdapterClass::Integrated
            } else {
                AdapterClass::Discrete
            }
        }
    }
}

/// Guess the vendor from a driver-reported name or PCI vendor id.
pub fn vendor_from_name(name: &str) -> Vendor {
    let lower = normalise(name);
    if lower.contains("nvidia") || lower.contains("geforce") || lower.contains("quadro") {
        Vendor::Nvidia
    } else if lower.contains("amd") || lower.contains("radeon") || lower.contains("ati ") {
        Vendor::Amd
    } else if lower.contains("intel") || lower.contains("arc ") {
        Vendor::Intel
    } else if lower.contains("apple") {
        Vendor::Apple
    } else if lower.contains("qualcomm") || lower.contains("adreno") {
        Vendor::Qualcomm
    } else {
        Vendor::Other
    }
}

/// Vendor from a PCI vendor id, which is unambiguous where a name is not.
pub fn vendor_from_pci_id(id: u16) -> Vendor {
    match id {
        0x10de => Vendor::Nvidia,
        0x1002 | 0x1022 => Vendor::Amd,
        0x8086 => Vendor::Intel,
        0x106b => Vendor::Apple,
        0x5143 => Vendor::Qualcomm,
        _ => Vendor::Other,
    }
}

/// The backend a device of this vendor and class would realistically be driven
/// through.
pub fn backend_for(vendor: Vendor, class: AdapterClass) -> Backend {
    match (vendor, class) {
        (_, AdapterClass::Virtual | AdapterClass::LegacyIntegrated) => Backend::Cpu,
        (Vendor::Nvidia, _) => Backend::Cuda,
        (Vendor::Apple, _) => Backend::Metal,
        (Vendor::Amd, AdapterClass::Discrete) => Backend::Rocm,
        // Integrated AMD and every Intel part are reached most reliably through
        // Vulkan; oneAPI works on Intel but needs a runtime few machines have.
        (Vendor::Amd | Vendor::Intel | Vendor::Qualcomm | Vendor::Other, _) => Backend::Vulkan,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remoting_and_hypervisor_displays_are_not_accelerators() {
        for name in [
            "Parsec Virtual Display Adapter",
            "Microsoft Basic Display Adapter",
            "Citrix Indirect Display Adapter",
            "VMware SVGA 3D",
            "Hyper-V Video",
            "DisplayLink USB Device",
        ] {
            assert_eq!(
                classify(name, vendor_from_name(name), None),
                AdapterClass::Virtual,
                "{name} should have been rejected"
            );
        }
    }

    #[test]
    fn intel_mobile_graphics_are_integrated_but_arc_is_not() {
        assert_eq!(
            classify("Intel(R) Graphics", Vendor::Intel, None),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("Intel(R) Iris(R) Xe Graphics", Vendor::Intel, None),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("Intel(R) UHD Graphics 770", Vendor::Intel, None),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("Intel(R) Arc(TM) A770 Graphics", Vendor::Intel, None),
            AdapterClass::Discrete
        );
    }

    #[test]
    fn amd_apu_graphics_are_integrated_but_radeon_rx_is_not() {
        assert_eq!(
            classify("AMD Radeon(TM) 780M Graphics", Vendor::Amd, None),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("AMD Radeon(TM) Graphics", Vendor::Amd, None),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("AMD Radeon RX 7900 XTX", Vendor::Amd, None),
            AdapterClass::Discrete
        );
        assert_eq!(
            classify("AMD Radeon Pro W7900", Vendor::Amd, None),
            AdapterClass::Discrete
        );
    }

    #[test]
    fn every_nvidia_part_is_discrete() {
        for name in ["NVIDIA GeForce RTX 4090", "NVIDIA A100-SXM4-80GB"] {
            assert_eq!(classify(name, Vendor::Nvidia, None), AdapterClass::Discrete);
        }
    }

    #[test]
    fn trademark_marks_do_not_hide_the_name_underneath() {
        assert_eq!(
            normalise("Intel(R)  Arc(TM) A770 Graphics"),
            "intel arc a770 graphics"
        );
        assert_eq!(normalise("AMD Radeon™ Graphics"), "amd radeon graphics");
        assert_eq!(
            normalise("NVIDIA® GeForce RTX 4090"),
            "nvidia geforce rtx 4090"
        );
    }

    #[test]
    fn vendors_are_recognised_from_names_and_from_pci_ids() {
        assert_eq!(vendor_from_name("NVIDIA GeForce RTX 4090"), Vendor::Nvidia);
        assert_eq!(vendor_from_name("AMD Radeon RX 7900 XTX"), Vendor::Amd);
        assert_eq!(vendor_from_name("Intel(R) Graphics"), Vendor::Intel);
        assert_eq!(vendor_from_name("Apple M3 Max"), Vendor::Apple);
        assert_eq!(vendor_from_name("Something Else"), Vendor::Other);

        assert_eq!(vendor_from_pci_id(0x10de), Vendor::Nvidia);
        assert_eq!(vendor_from_pci_id(0x1002), Vendor::Amd);
        assert_eq!(vendor_from_pci_id(0x8086), Vendor::Intel);
        assert_eq!(vendor_from_pci_id(0x1234), Vendor::Other);
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn a_large_dedicated_pool_outranks_a_generic_name() {
        // llmfit's #638: a 32 GB Instinct MI50 reporting itself as the same
        // generic string an integrated Cezanne uses. Sizing it against system
        // memory would be badly wrong in both directions.
        assert_eq!(
            classify("AMD Radeon Graphics", Vendor::Amd, Some(32 * GIB)),
            AdapterClass::Discrete
        );
        // The integrated part with the same name keeps its classification.
        assert_eq!(
            classify("AMD Radeon Graphics", Vendor::Amd, Some(512 * 1024 * 1024)),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("AMD Radeon Graphics", Vendor::Amd, None),
            AdapterClass::Integrated
        );
    }

    #[test]
    fn a_firmware_carveout_marks_a_large_integrated_part_as_integrated() {
        // A Ryzen AI MAX+ 395 with 128 GB installed and 96 GB configured as
        // its graphics carveout. The adapter claims 96 GB under a generic
        // name, which is more than the discrete floor and would otherwise be
        // read as a card, leaving the machine described as a 96 GB GPU beside
        // 31 GB of system RAM, when it has one pool of 128.
        assert_eq!(
            classify_with_carveout(
                "AMD Radeon(TM) Graphics",
                Vendor::Amd,
                Some(96 * GIB),
                Some(96 * GIB),
            ),
            AdapterClass::Integrated
        );
        // The same adapter on a machine with no measured carveout keeps the
        // old reading: without that evidence there is nothing to distinguish
        // it from the Instinct card of #638.
        assert_eq!(
            classify_with_carveout("AMD Radeon(TM) Graphics", Vendor::Amd, Some(96 * GIB), None),
            AdapterClass::Discrete
        );
    }

    #[test]
    fn a_carveout_does_not_explain_away_a_card_sitting_beside_it() {
        // Both are present on the same machine, so both are offered the same
        // carveout figure. Only the integrated part is explained by it.
        let carveout = Some(96 * GIB);
        assert_eq!(
            classify_with_carveout(
                "AMD Radeon(TM) Graphics",
                Vendor::Amd,
                Some(96 * GIB),
                carveout
            ),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify_with_carveout(
                "AMD Radeon RX 7900 XTX",
                Vendor::Amd,
                Some(24 * GIB),
                carveout
            ),
            AdapterClass::Discrete
        );
        // An NVIDIA card in the same machine likewise.
        assert_eq!(
            classify_with_carveout(
                "NVIDIA GeForce RTX 4090",
                Vendor::Nvidia,
                Some(24 * GIB),
                carveout
            ),
            AdapterClass::Discrete
        );
    }

    #[test]
    fn the_ordinary_reserve_every_machine_loses_is_not_a_carveout() {
        // Passing a small firmware reserve as though it were a carveout must
        // not turn a real card into integrated graphics.
        assert_eq!(
            classify_with_carveout(
                "AMD Radeon Graphics",
                Vendor::Amd,
                Some(32 * GIB),
                Some(315 * 1024 * 1024),
            ),
            AdapterClass::Discrete
        );
    }

    #[test]
    fn intel_graphics_older_than_xe_are_not_inference_targets() {
        for name in [
            "Intel(R) HD Graphics 4000",
            "Intel(R) UHD Graphics 630",
            "Intel(R) HD Graphics 530",
            "Intel(R) HD Graphics",
        ] {
            assert_eq!(
                classify(name, Vendor::Intel, None),
                AdapterClass::LegacyIntegrated,
                "{name} predates a usable compute back end"
            );
        }
        for name in [
            "Intel(R) UHD Graphics 770",
            "Intel(R) Iris(R) Xe Graphics",
            "Intel(R) Graphics",
        ] {
            assert_eq!(
                classify(name, Vendor::Intel, None),
                AdapterClass::Integrated,
                "{name} is Xe or later and does work"
            );
        }
    }

    #[test]
    fn a_legacy_integrated_part_is_told_to_run_on_the_cpu() {
        assert_eq!(
            backend_for(Vendor::Intel, AdapterClass::LegacyIntegrated),
            Backend::Cpu
        );
    }

    #[test]
    fn a_virtual_adapter_never_gets_an_accelerated_backend() {
        let backend = backend_for(Vendor::Nvidia, AdapterClass::Virtual);
        assert_eq!(backend, Backend::Cpu);
        assert!(!backend.is_accelerated());
    }

    #[test]
    fn integrated_amd_goes_through_vulkan_and_discrete_amd_through_rocm() {
        assert_eq!(
            backend_for(Vendor::Amd, AdapterClass::Integrated),
            Backend::Vulkan
        );
        assert_eq!(
            backend_for(Vendor::Amd, AdapterClass::Discrete),
            Backend::Rocm
        );
    }
}
