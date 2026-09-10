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
//! GPUs — 2 GB is typical — and both WMI and the registry will hand it to you
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
    /// Not an accelerator at all: a remote-desktop or hypervisor display.
    Virtual,
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
/// Drivers pepper names with trademark marks — "Intel(R) Arc(TM) A770" — and
/// they land in the middle of exactly the phrases worth recognising, turning
/// "arc a770" into "arc(tm) a770". Stripping them once here is more robust than
/// keeping a marker list that enumerates every punctuation variant.
fn normalise(name: &str) -> String {
    let mut folded = name.to_ascii_lowercase();
    for mark in ["(r)", "(tm)", "(c)", "®", "™", "©"] {
        folded = folded.replace(mark, " ");
    }
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Classify an adapter from its reported name and vendor.
pub fn classify(name: &str, vendor: Vendor) -> AdapterClass {
    let lower = normalise(name);

    if VIRTUAL_MARKERS.iter().any(|m| lower.contains(m)) {
        return AdapterClass::Virtual;
    }

    match vendor {
        // Apple has never shipped a discrete GPU in an Apple Silicon machine.
        Vendor::Apple | Vendor::Qualcomm => AdapterClass::Integrated,
        Vendor::Nvidia => AdapterClass::Discrete,
        Vendor::Intel | Vendor::Amd | Vendor::Other => {
            if DISCRETE_OVERRIDES.iter().any(|m| lower.contains(m)) {
                AdapterClass::Discrete
            } else if INTEGRATED_MARKERS.iter().any(|m| lower.contains(m)) {
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
        (_, AdapterClass::Virtual) => Backend::Cpu,
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
                classify(name, vendor_from_name(name)),
                AdapterClass::Virtual,
                "{name} should have been rejected"
            );
        }
    }

    #[test]
    fn intel_mobile_graphics_are_integrated_but_arc_is_not() {
        assert_eq!(
            classify("Intel(R) Graphics", Vendor::Intel),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("Intel(R) Iris(R) Xe Graphics", Vendor::Intel),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("Intel(R) UHD Graphics 770", Vendor::Intel),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("Intel(R) Arc(TM) A770 Graphics", Vendor::Intel),
            AdapterClass::Discrete
        );
    }

    #[test]
    fn amd_apu_graphics_are_integrated_but_radeon_rx_is_not() {
        assert_eq!(
            classify("AMD Radeon(TM) 780M Graphics", Vendor::Amd),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("AMD Radeon(TM) Graphics", Vendor::Amd),
            AdapterClass::Integrated
        );
        assert_eq!(
            classify("AMD Radeon RX 7900 XTX", Vendor::Amd),
            AdapterClass::Discrete
        );
        assert_eq!(
            classify("AMD Radeon Pro W7900", Vendor::Amd),
            AdapterClass::Discrete
        );
    }

    #[test]
    fn every_nvidia_part_is_discrete() {
        for name in ["NVIDIA GeForce RTX 4090", "NVIDIA A100-SXM4-80GB"] {
            assert_eq!(classify(name, Vendor::Nvidia), AdapterClass::Discrete);
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
