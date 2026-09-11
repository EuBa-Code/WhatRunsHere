//! Enumerating non-NVIDIA graphics adapters, per operating system.
//!
//! Each platform returns the same shape (a name, a vendor hint, and whatever
//! the system claims about dedicated memory) and leaves the judgement of what
//! that means to [`crate::classify`]. The claim in particular is not to be
//! trusted: Windows reports 2 GB of "video memory" for integrated graphics that
//! own none.

/// What a platform managed to learn about one adapter.
///
/// Serializable, and carried through to `whatllm doctor --json`, so that a
/// report of "my card was detected wrongly" arrives with the evidence rather
/// than only the verdict. Pasted into a test, one of these is a regression
/// fixture for the classification that got it wrong, which is the shape a bug
/// report should have.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
    /// True when that figure came out of a field only 32 bits wide.
    ///
    /// Windows reports video memory in several places and the old ones are
    /// narrow: `Win32_VideoController.AdapterRAM` is a signed 32-bit integer,
    /// and `HardwareInformation.MemorySize` is sometimes only four bytes. Four
    /// bytes cannot express more than four gibibytes, which is less than every
    /// card worth asking about, and what comes back instead is a ceiling value
    /// that looks exactly like a real small capacity: `0x7FFF_F000` sits
    /// between a genuine 2 GB card and a genuine 2 GiB one, so no threshold
    /// separates them.
    ///
    /// Rather than invent one, the narrow field is simply not used as a
    /// capacity for a discrete card. It remains useful for telling an
    /// integrated part's aperture, which is all it was ever describing.
    pub vram_from_narrow_field: bool,
    /// The most of a shared pool the platform will let a compute job hold, in
    /// bytes, where the platform enforces such a limit.
    ///
    /// Distinct from [`Self::claimed_vram_bytes`], which is a claim about
    /// memory a device owns. This is a claim about memory a device may
    /// *borrow*, and it is the binding figure on Apple Silicon: the pool is
    /// the whole machine's memory, and macOS will not let Metal wire all of
    /// it. On a 32 GB Mac the limit is 24 GB, and a plan sized against 32
    /// describes a load that fails.
    ///
    /// `None` everywhere the platform imposes no such ceiling, which is
    /// everywhere but macOS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compute_memory_limit_bytes: Option<u64>,
}

/// The reason the narrow field is discarded rather than repaired, checked where
/// it is claimed rather than argued for in prose.
///
/// The signed 32-bit ceiling sits between a real 2 GB card and a real 2 GiB
/// one, so no threshold drawn here can tell all three apart. Should someone
/// later be tempted to invent one, this stops compiling.
const _: () = {
    const CEILING: u64 = 0x7FFF_F000;
    const TWO_GB: u64 = 2_000_000_000;
    const TWO_GIB: u64 = 2 * 1024 * 1024 * 1024;
    assert!(TWO_GB < CEILING && CEILING < TWO_GIB);
};

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
/// 32-bit field and therefore wraps for any card with 4 GB or more: the exact
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

        // The 64-bit value is present on discrete parts and authoritative.
        let precise = entry
            .get_value::<u64, _>("HardwareInformation.qwMemorySize")
            .ok()
            .filter(|&bytes| bytes > 0);
        // The binary one is a fallback. On integrated parts it describes an
        // aperture rather than memory; at four bytes wide it may also have
        // saturated, and there is no way to tell that from a small card.
        let fallback = entry
            .get_raw_value("HardwareInformation.MemorySize")
            .ok()
            .and_then(|value| {
                let width = value.bytes.len();
                le_bytes_to_u64(&value.bytes).map(|bytes| (bytes, width))
            })
            .filter(|&(bytes, _)| bytes > 0);

        let (claimed_vram_bytes, vram_from_narrow_field) = match (precise, fallback) {
            (Some(bytes), _) => (Some(bytes), false),
            (None, Some((bytes, width))) => (Some(bytes), width <= 4),
            (None, None) => (None, false),
        };

        found.push(RawAdapter {
            name,
            provider: entry.get_value::<String, _>("ProviderName").ok(),
            pci_vendor: None,
            claimed_vram_bytes,
            vram_from_narrow_field,
            compute_memory_limit_bytes: None,
        });
    }
    found
}

/// Memory installed in the machine, read from firmware rather than from the
/// operating system.
///
/// The operating system reports the memory it is allowed to manage, which is
/// not the memory that is present. Where firmware has handed a block to an
/// integrated GPU before the kernel started, that block is simply missing from
/// the total: a 128 GB Ryzen AI MAX with 96 GB configured as its graphics
/// carveout reports around 31 GB of system RAM. Sizing a model against that
/// figure rejects every model the machine was bought to run.
///
/// SMBIOS knows the real capacity because it enumerates the DIMM slots, and
/// Windows caches the entire SMBIOS table in the registry, so this is a
/// registry read, not a WMI query or a spawned `powershell`.
///
/// Returns `None` where the table cannot be read or describes no populated
/// slot, and the caller keeps the operating system's figure. The difference
/// between the two is what [`crate::detect`] treats as a carveout.
#[cfg(windows)]
pub fn installed_memory_bytes() -> Option<u64> {
    use winreg::enums::HKEY_LOCAL_MACHINE;
    use winreg::RegKey;

    const SMBIOS: &str = r"SYSTEM\CurrentControlSet\services\mssmbios\Data";

    let key = RegKey::predef(HKEY_LOCAL_MACHINE)
        .open_subkey(SMBIOS)
        .ok()?;
    let blob = key.get_raw_value("SMBiosData").ok()?;
    // The registry value prefixes the table with a header of its own: calling
    // method, SMBIOS major and minor version, DMI revision, then a 32-bit
    // length. The structures start after those eight bytes.
    sum_populated_dimms(blob.bytes.get(8..)?)
}

/// Sum the capacity of every populated memory device in an SMBIOS table.
///
/// Split from the registry read so the walk can be exercised against a table
/// written by hand. That is the only way to reach the wide-capacity path on a
/// machine whose own DIMMs are small enough to fit the narrow field, and the
/// wide path is the one that matters: it is the machines with a great deal of
/// memory that have a great deal of it carved out.
#[cfg(windows)]
fn sum_populated_dimms(table: &[u8]) -> Option<u64> {
    /// SMBIOS structure type 17, `Memory Device`: one DIMM slot.
    const MEMORY_DEVICE: u8 = 17;
    /// Structure type 127 closes the table.
    const END_OF_TABLE: u8 = 127;
    /// Type, length and a 16-bit handle precede every structure's fields.
    const HEADER_LEN: usize = 4;

    let mut total: u64 = 0;
    let mut populated = 0usize;
    let mut at = 0usize;

    while at + HEADER_LEN <= table.len() {
        let kind = table[at];
        let length = table[at + 1] as usize;
        if kind == END_OF_TABLE {
            break;
        }
        // A structure shorter than its own header means the table is not what
        // it claims to be. Walking on would read arbitrary bytes as capacities.
        if length < HEADER_LEN {
            break;
        }
        let structure = table.get(at..at.checked_add(length)?)?;
        if kind == MEMORY_DEVICE {
            if let Some(bytes) = dimm_capacity_bytes(structure) {
                total += bytes;
                populated += 1;
            }
        }

        // Variable-length strings follow the fields, each NUL-terminated, and
        // the set ends with an extra NUL. A structure with no strings at all
        // still carries the pair.
        let mut cursor = at + length;
        loop {
            match table.get(cursor..cursor.checked_add(2)?) {
                Some([0, 0]) => {
                    cursor += 2;
                    break;
                }
                Some(_) => cursor += 1,
                // The table ended mid-structure. Report what was counted
                // rather than discarding it.
                None => return (populated > 0).then_some(total),
            }
        }
        at = cursor;
    }

    (populated > 0).then_some(total)
}

/// Capacity of one `Memory Device` structure, in bytes.
///
/// `None` for an empty slot and for a slot whose size the firmware does not
/// know, which are different from a slot holding nothing and must not be
/// counted as zero-capacity DIMMs. A table of nothing but unknowns would
/// otherwise sum to zero and look authoritative.
#[cfg(windows)]
fn dimm_capacity_bytes(device: &[u8]) -> Option<u64> {
    /// 16-bit capacity field.
    const SIZE: usize = 12;
    /// 32-bit capacity field, used when the narrow one cannot hold the value.
    const EXTENDED_SIZE: usize = 28;
    /// Written into the narrow field when the real figure is in the wide one.
    const SIZE_IS_EXTENDED: u16 = 0x7FFF;
    /// Written when the firmware does not know the capacity.
    const SIZE_UNKNOWN: u16 = 0xFFFF;

    let raw = u16::from_le_bytes(device.get(SIZE..SIZE + 2)?.try_into().ok()?);
    if raw == 0 || raw == SIZE_UNKNOWN {
        return None;
    }
    if raw == SIZE_IS_EXTENDED {
        // Anything from 32 GB up. Bit 31 of the wide field is reserved, and
        // the value is in megabytes.
        let wide = u32::from_le_bytes(
            device
                .get(EXTENDED_SIZE..EXTENDED_SIZE + 4)?
                .try_into()
                .ok()?,
        );
        let megabytes = u64::from(wide & 0x7FFF_FFFF);
        return (megabytes > 0).then(|| megabytes * 1024 * 1024);
    }
    // Bit 15 selects the unit: clear for megabytes, set for kilobytes.
    let value = u64::from(raw & 0x7FFF);
    Some(if raw & 0x8000 == 0 {
        value * 1024 * 1024
    } else {
        value * 1024
    })
}

/// Firmware memory capacity is not readable without privileges here.
///
/// Linux keeps the SMBIOS tables under `/sys/firmware/dmi`, readable only by
/// root, and macOS has no equivalent carveout to recover: Apple Silicon's
/// memory is one pool the operating system reports in full.
#[cfg(not(windows))]
pub fn installed_memory_bytes() -> Option<u64> {
    None
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
            // sysfs reports a 64-bit value; nothing here saturates.
            vram_from_narrow_field: false,
            // Linux imposes no ceiling on how much of a shared pool a compute
            // job may hold; it is ordinary memory, allocated as needed.
            compute_memory_limit_bytes: None,
        });
    }
    found
}

/// How much of the shared pool macOS will let Metal wire, in bytes.
///
/// Apple Silicon's memory is one pool, but a compute job does not get all of
/// it. The kernel holds a wired limit well below the installed total (around
/// three quarters of it on the smaller machines) and an allocation past that
/// point fails rather than paging. A tool that reads the pool as the machine's
/// full memory promises loads that do not happen, and it promises them on the
/// machines people most often buy for this.
///
/// Metal is asked rather than the limit computed, for the same reason NVIDIA
/// bandwidth is derived rather than looked up: `recommendedMaxWorkingSetSize`
/// is the figure the kernel is actually enforcing, so it stays right when the
/// default changes with a macOS release, and it follows an owner who has
/// raised `iogpu.wired_limit_mb` by hand. A formula reproducing today's
/// default would be wrong in both of those cases.
///
/// `None` when there is no Metal device, which on Apple Silicon means
/// something is wrong enough that guessing would not help.
#[cfg(target_os = "macos")]
fn metal_working_set_limit_bytes() -> Option<u64> {
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice};

    let device = MTLCreateSystemDefaultDevice()?;
    let bytes = device.recommendedMaxWorkingSetSize();
    (bytes > 0).then_some(bytes)
}

/// Apple Silicon has exactly one accelerator and it shares the machine's
/// memory, so there is nothing to enumerate, only to name.
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
            vram_from_narrow_field: false,
            compute_memory_limit_bytes: metal_working_set_limit_bytes(),
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

    /// One SMBIOS `Memory Device` structure carrying `size` in the narrow
    /// field and `extended` in the wide one, followed by its empty string set.
    #[cfg(windows)]
    fn memory_device(size: u16, extended: u32) -> Vec<u8> {
        let mut bytes = vec![0u8; 40];
        bytes[0] = 17; // type
        bytes[1] = 40; // length
        bytes[12..14].copy_from_slice(&size.to_le_bytes());
        bytes[28..32].copy_from_slice(&extended.to_le_bytes());
        bytes.extend_from_slice(&[0, 0]); // no strings
        bytes
    }

    #[cfg(windows)]
    #[test]
    fn dimm_capacities_are_summed_across_both_field_widths() {
        const GIB: u64 = 1024 * 1024 * 1024;

        // A 128 GB Ryzen AI MAX: four 32 GB DIMMs, each too large for the
        // 16-bit field and so carrying 0x7FFF and a megabyte count beside it.
        let mut table = Vec::new();
        for _ in 0..4 {
            table.extend(memory_device(0x7FFF, 32 * 1024));
        }
        table.push(127); // end of table
        table.extend_from_slice(&[4, 0, 0, 0, 0]);
        assert_eq!(sum_populated_dimms(&table), Some(128 * GIB));

        // 16 GB fits the narrow field, in megabytes.
        let mut table = memory_device(16 * 1024, 0);
        table.extend(memory_device(16 * 1024, 0));
        table.push(127);
        table.extend_from_slice(&[4, 0, 0, 0, 0]);
        assert_eq!(sum_populated_dimms(&table), Some(32 * GIB));
    }

    #[cfg(windows)]
    #[test]
    fn empty_and_unknown_slots_are_not_counted_as_zero_capacity_dimms() {
        const GIB: u64 = 1024 * 1024 * 1024;

        // Two populated slots and two empty ones, which is the ordinary state
        // of a four-slot board.
        let mut table = memory_device(8 * 1024, 0);
        table.extend(memory_device(8 * 1024, 0));
        table.extend(memory_device(0, 0));
        table.extend(memory_device(0, 0));
        table.push(127);
        table.extend_from_slice(&[4, 0, 0, 0, 0]);
        assert_eq!(sum_populated_dimms(&table), Some(16 * GIB));

        // A table of nothing but unknowns sums to zero, which would look like
        // an authoritative answer of "no memory". It must decline instead.
        let mut table = memory_device(0xFFFF, 0);
        table.extend(memory_device(0xFFFF, 0));
        table.push(127);
        table.extend_from_slice(&[4, 0, 0, 0, 0]);
        assert_eq!(sum_populated_dimms(&table), None);
    }

    #[cfg(windows)]
    #[test]
    fn strings_between_structures_do_not_derail_the_walk() {
        const GIB: u64 = 1024 * 1024 * 1024;

        // Real tables carry manufacturer, serial and part number strings after
        // each structure's fields. Miscounting them reads the next structure
        // from the wrong offset.
        let mut table = vec![0u8; 40];
        table[0] = 17;
        table[1] = 40;
        table[12..14].copy_from_slice(&(16u16 * 1024).to_le_bytes());
        table.extend_from_slice(b"Corsair\0CMK32GX4\0\0");
        table.extend(memory_device(16 * 1024, 0));
        table.push(127);
        table.extend_from_slice(&[4, 0, 0, 0, 0]);

        assert_eq!(sum_populated_dimms(&table), Some(32 * GIB));
    }

    #[cfg(windows)]
    #[test]
    fn a_truncated_or_nonsense_table_is_declined_rather_than_misread() {
        assert_eq!(sum_populated_dimms(&[]), None);
        // A length shorter than the header itself.
        assert_eq!(sum_populated_dimms(&[17, 2, 0, 0, 0, 0]), None);
        // A structure that runs off the end of the buffer.
        assert_eq!(sum_populated_dimms(&[17, 40, 0, 0, 1, 2, 3]), None);
    }

    #[cfg(windows)]
    #[test]
    fn firmware_capacity_is_never_smaller_than_what_the_os_reports() {
        // On the machine this runs on, whatever it is: the installed total is
        // the OS total plus anything hidden from it, so it cannot be less. A
        // parser reading the wrong offsets would fail here first.
        let Some(installed) = installed_memory_bytes() else {
            return;
        };
        let os_total = sysinfo::System::new_all().total_memory();
        assert!(
            installed >= os_total,
            "firmware reports {installed} bytes installed, the OS sees {os_total}"
        );
    }

    #[test]
    fn enumeration_never_panics_on_the_host_it_runs_on() {
        for adapter in adapters() {
            assert!(!adapter.name.is_empty(), "an adapter must have a name");
        }
    }
}
