//! Enumerating non-NVIDIA graphics adapters, per operating system.
//!
//! Each platform returns the same shape — a name, a vendor hint, and whatever
//! the system claims about dedicated memory — and leaves the judgement of what
//! that means to [`crate::classify`]. The claim in particular is not to be
//! trusted: Windows reports 2 GB of "video memory" for integrated graphics that
//! own none.

/// What a platform managed to learn about one adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawAdapter {
    /// Driver-reported device name.
    pub name: String,
    /// Driver or vendor string, when the platform supplies one.
    pub provider: Option<String>,
    /// PCI vendor id, when the platform supplies one. More reliable than a
    /// name.
    pub pci_vendor: Option<u16>,
    /// Dedicated video memory the system claims, in bytes.
    ///
    /// Meaningful for discrete parts and misleading for integrated ones.
    pub claimed_vram_bytes: Option<u64>,
}

/// Read a little-endian unsigned integer from a registry blob.
#[cfg(windows)]
fn le_bytes_to_u64(bytes: &[u8]) -> Option<u64> {
    match bytes.len() {
        4 => Some(u64::from(u32::from_le_bytes(bytes.try_into().ok()?))),
        8 => Some(u64::from_le_bytes(bytes.try_into().ok()?)),
        _ => None,
    }
}

/// Windows enumerates display adapters under the display class key.
///
/// Preferred over WMI's `Win32_VideoController`, whose `AdapterRAM` is a signed
/// 32-bit field and therefore wraps for any card with 4 GB or more — the exact
/// cards worth asking about.
#[cfg(windows)]
pub fn adapters() -> Vec<RawAdapter> {
    use winreg::enums::HKEY_LOCAL_MACHINE;
    use winreg::RegKey;

    const DISPLAY_CLASS: &str =
        r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";

    let Ok(class) = RegKey::predef(HKEY_LOCAL_MACHINE).open_subkey(DISPLAY_CLASS) else {
        return Vec::new();
    };

    let mut found = Vec::new();
    for key_name in class.enum_keys().flatten() {
        // Adapter instances are four-digit keys; Properties and Configuration
        // are not adapters.
        if !key_name.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let Ok(entry) = class.open_subkey(&key_name) else {
            continue;
        };
        let Ok(name) = entry.get_value::<String, _>("DriverDesc") else {
            continue;
        };

        // The 64-bit value is present on discrete parts and authoritative. The
        // binary one is a fallback, and on integrated parts describes an
        // aperture rather than memory.
        let claimed_vram_bytes = entry
            .get_value::<u64, _>("HardwareInformation.qwMemorySize")
            .ok()
            .or_else(|| {
                entry
                    .get_raw_value("HardwareInformation.MemorySize")
                    .ok()
                    .and_then(|value| le_bytes_to_u64(&value.bytes))
            })
            .filter(|&bytes| bytes > 0);

        found.push(RawAdapter {
            name,
            provider: entry.get_value::<String, _>("ProviderName").ok(),
            pci_vendor: None,
            claimed_vram_bytes,
        });
    }
    found
}

/// Linux exposes adapters through the DRM subsystem.
#[cfg(target_os = "linux")]
pub fn adapters() -> Vec<RawAdapter> {
    use std::fs;

    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return Vec::new();
    };

    let mut found = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Cards are `card0`, `card1`; connectors are `card0-DP-1`.
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let device = entry.path().join("device");

        let read = |file: &str| fs::read_to_string(device.join(file)).ok();
        let pci_vendor = read("vendor")
            .and_then(|raw| u16::from_str_radix(raw.trim().trim_start_matches("0x"), 16).ok());
        // AMD exposes its video memory here; Intel and others do not.
        let claimed_vram_bytes = read("mem_info_vram_total")
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .filter(|&bytes| bytes > 0);

        let label = read("label")
            .map(|raw| raw.trim().to_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| match pci_vendor {
                Some(0x10de) => "NVIDIA device".to_owned(),
                Some(0x1002) => "AMD device".to_owned(),
                Some(0x8086) => "Intel device".to_owned(),
                _ => name.to_string(),
            });

        found.push(RawAdapter {
            name: label,
            provider: None,
            pci_vendor,
            claimed_vram_bytes,
        });
    }
    found
}

/// Apple Silicon has exactly one accelerator and it shares the machine's
/// memory, so there is nothing to enumerate — only to name.
#[cfg(target_os = "macos")]
pub fn adapters() -> Vec<RawAdapter> {
    use std::process::Command;

    let chip = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());

    match chip {
        Some(name) if name.starts_with("Apple") => vec![RawAdapter {
            name,
            provider: Some("Apple".to_owned()),
            pci_vendor: Some(0x106b),
            // Deliberately absent: unified memory is not video memory, and the
            // caller supplies the real pool from system RAM.
            claimed_vram_bytes: None,
        }],
        // An Intel Mac's discrete GPU is not reachable this way; report nothing
        // rather than something wrong.
        _ => Vec::new(),
    }
}

/// Platforms without an implementation report nothing rather than guessing.
#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
pub fn adapters() -> Vec<RawAdapter> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn binary_memory_blobs_decode_at_both_widths() {
        // 0x7FFFF000, the 2 GB aperture Windows reports for integrated parts.
        assert_eq!(
            le_bytes_to_u64(&[0x00, 0xF0, 0xFF, 0x7F]),
            Some(2_147_479_552)
        );
        assert_eq!(
            le_bytes_to_u64(&[0, 0, 0, 0, 0x06, 0, 0, 0]),
            Some(25_769_803_776)
        );
        assert_eq!(le_bytes_to_u64(&[1, 2, 3]), None);
    }

    #[test]
    fn enumeration_never_panics_on_the_host_it_runs_on() {
        for adapter in adapters() {
            assert!(!adapter.name.is_empty(), "an adapter must have a name");
        }
    }
}
