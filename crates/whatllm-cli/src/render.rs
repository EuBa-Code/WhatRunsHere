//! Terminal presentation.
//!
//! A report that is hard to read is a report that gets skimmed, and the whole
//! value here is in the parts people skim past: which number was measured,
//! which was guessed, and where the configuration is about to fall over.

use std::fmt::Write as _;
use std::io::IsTerminal;

/// Whether to emit colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Style {
    /// False when writing to a pipe, or when the user asked for plain output.
    pub colour: bool,
}

impl Style {
    /// Colour if standard output is a terminal and `NO_COLOR` is unset.
    pub fn detect(forced_off: bool) -> Self {
        Self {
            colour: !forced_off
                && std::io::stdout().is_terminal()
                && std::env::var_os("NO_COLOR").is_none(),
        }
    }

    /// Wrap `text` in an ANSI code, or return it unchanged.
    pub fn paint(self, code: &str, text: &str) -> String {
        if self.colour {
            format!("\x1b[{code}m{text}\x1b[0m")
        } else {
            text.to_owned()
        }
    }

    /// De-emphasised text.
    pub fn dim(self, text: &str) -> String {
        self.paint("2", text)
    }

    /// Emphasised text.
    pub fn bold(self, text: &str) -> String {
        self.paint("1", text)
    }

    /// Something that went well.
    pub fn good(self, text: &str) -> String {
        self.paint("32", text)
    }

    /// Something to look at.
    pub fn warn(self, text: &str) -> String {
        self.paint("33", text)
    }

    /// Something that does not work.
    pub fn bad(self, text: &str) -> String {
        self.paint("31", text)
    }

    /// The accent used for headings.
    pub fn accent(self, text: &str) -> String {
        self.paint("36", text)
    }
}

/// Format a byte count in binary units.
///
/// Binary rather than decimal because every tool this one sits beside uses
/// binary: llama.cpp reports MiB, nvidia-smi reports MiB, and a machine sold
/// as having 32 GB has 32 GiB. Printing its 34.4 decimal gigabytes would be
/// correct and would read as a bug. Published file sizes are converted on the
/// way in, so a model llama.cpp loads as 4.58 GiB is shown here as 4.58 GiB.
pub fn bytes(value: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let value = value as f64;
    if value >= GIB {
        let gib = value / GIB;
        if gib >= 100.0 {
            format!("{gib:.0} GiB")
        } else {
            format!("{gib:.1} GiB")
        }
    } else if value >= MIB {
        format!("{:.0} MiB", value / MIB)
    } else {
        format!("{:.0} KiB", value / 1024.0)
    }
}

/// Format a token count as `8k`, `128k`, or the number itself.
pub fn tokens(count: u32) -> String {
    if count >= 1000 && count % 1024 == 0 {
        format!("{}k", count / 1024)
    } else if count >= 1000 {
        format!("{:.0}k", f64::from(count) / 1000.0)
    } else {
        count.to_string()
    }
}

/// A compact horizontal bar for a 0..=1 fraction.
pub fn meter(fraction: f64, width: usize) -> String {
    const BLOCKS: [char; 9] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];
    let fraction = fraction.clamp(0.0, 1.0);
    let eighths = (fraction * width as f64 * 8.0).round() as usize;
    let full = eighths / 8;
    let remainder = eighths % 8;

    let mut out = String::with_capacity(width);
    for _ in 0..full.min(width) {
        out.push('█');
    }
    if full < width && remainder > 0 {
        out.push(BLOCKS[remainder]);
    }
    while out.chars().count() < width {
        out.push('·');
    }
    out
}

/// A sparkline over a series, for curves that belong inline.
pub fn sparkline(values: &[f64]) -> String {
    const LEVELS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() {
        return String::new();
    }
    let max = values.iter().copied().fold(f64::MIN, f64::max);
    let min = values.iter().copied().fold(f64::MAX, f64::min);
    let span = (max - min).max(f64::EPSILON);
    values
        .iter()
        .map(|value| {
            let level = ((value - min) / span * 7.0).round() as usize;
            LEVELS[level.min(7)]
        })
        .collect()
}

/// A section heading.
pub fn heading(style: Style, text: &str) -> String {
    format!("\n{}\n", style.bold(&style.accent(text)))
}

/// A `label: value` line, with the label padded to `width`.
pub fn field(style: Style, width: usize, label: &str, value: &str) -> String {
    let mut out = String::new();
    let _ = write!(out, "  {:<width$}  {}", style.dim(label), value);
    out
}

/// A horizontal rule.
pub fn rule(style: Style, width: usize) -> String {
    style.dim(&"─".repeat(width))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN: Style = Style { colour: false };

    #[test]
    fn byte_counts_read_the_way_the_neighbouring_tools_print_them() {
        // A published 4.92 GB file is the 4.58 GiB llama.cpp reports on load.
        assert_eq!(bytes(4_920_000_000), "4.6 GiB");
        // 32 GiB of RAM must not print as 34 of anything.
        assert_eq!(bytes(34_359_738_368), "32.0 GiB");
        assert_eq!(bytes(128 * 1024 * 1024 * 1024), "128 GiB");
        assert_eq!(bytes(560 * 1024 * 1024), "560 MiB");
        assert_eq!(bytes(4_096), "4 KiB");
    }

    #[test]
    fn token_counts_are_shortened_without_losing_the_number() {
        assert_eq!(tokens(8_192), "8k");
        assert_eq!(tokens(131_072), "128k");
        assert_eq!(tokens(512), "512");
        assert_eq!(tokens(40_960), "40k");
    }

    #[test]
    fn a_meter_is_always_exactly_the_width_asked_for() {
        for fraction in [0.0, 0.01, 0.33, 0.5, 0.999, 1.0, 2.0, -1.0] {
            assert_eq!(
                meter(fraction, 12).chars().count(),
                12,
                "width wrong at {fraction}"
            );
        }
    }

    #[test]
    fn a_full_meter_is_full_and_an_empty_one_is_empty() {
        assert!(meter(1.0, 8).chars().all(|c| c == '█'));
        assert!(meter(0.0, 8).chars().all(|c| c == '·'));
    }

    #[test]
    fn a_sparkline_tracks_the_shape_of_the_series() {
        let rising = sparkline(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rising.chars().count(), 4);
        assert_eq!(rising.chars().next(), Some('▁'));
        assert_eq!(rising.chars().last(), Some('█'));

        // A flat series must not divide by zero.
        let flat = sparkline(&[5.0, 5.0, 5.0]);
        assert_eq!(flat.chars().count(), 3);
        assert!(sparkline(&[]).is_empty());
    }

    #[test]
    fn plain_output_carries_no_escape_codes() {
        assert_eq!(PLAIN.good("ok"), "ok");
        assert_eq!(PLAIN.bold("ok"), "ok");
        assert!(!heading(PLAIN, "Section").contains('\x1b'));
    }

    #[test]
    fn coloured_output_closes_every_sequence_it_opens() {
        let styled = Style { colour: true };
        let painted = styled.good("ok");
        assert!(painted.starts_with("\x1b["));
        assert!(painted.ends_with("\x1b[0m"));
    }
}
