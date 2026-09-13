//! The front of a GGUF file: enough to say what it is.
//!
//! A GGUF file opens with a small table of key-value metadata before any
//! tensor, and the keys under `general.` say what the model is called, which
//! architecture it has and which format its weights are in. Reading them costs
//! a few megabytes at most (the tokenizer's vocabulary sits in the same table
//! and has to be stepped over), which is nothing beside the file.
//!
//! This is read only for a file the catalog does not recognise by size. It
//! says what the file is; it does not size it. Turning the header into an
//! [`whatrunshere_core::arch::Architecture`] is a different and larger job, and
//! guessing a footprint from a name is what this project refuses to do.

use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// The GGUF versions whose string and array lengths are 64-bit. Version 1
/// used 32-bit lengths and was superseded in 2023.
const SUPPORTED_VERSIONS: [u32; 2] = [2, 3];

/// A cap on how much of a file the reader will step through looking for the
/// general keys, so a malformed header cannot turn a listing into a wait.
const BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// What the header said about the file.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Header {
    /// `general.name`, the model's own name as the converter wrote it.
    pub name: Option<String>,
    /// `general.architecture`, such as `llama`, `qwen3` or `clip`.
    pub architecture: Option<String>,
    /// `general.size_label`, such as `30B-A3B`.
    pub size_label: Option<String>,
    /// The weight format, from `general.file_type`, when it is one of the
    /// numbered formats llama.cpp defines. Unknown numbers are left unnamed
    /// rather than mapped to the nearest name.
    pub format: Option<String>,
    /// Which shard this is of a split model, as `(this, of)`, when it is one.
    pub shard: Option<(u32, u32)>,
}

/// Read the header of the file at `path`.
///
/// # Errors
/// When the file cannot be opened, is not a GGUF, is a version this reader
/// does not know, or is malformed.
pub fn read_header(path: &Path) -> Result<Header, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse(&mut BufReader::new(file)).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parse a header from anything readable, which is what the tests hand in.
///
/// # Errors
/// As [`read_header`].
pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<Header, String> {
    let mut cursor = Counting {
        inner: reader,
        consumed: 0,
    };
    let mut magic = [0u8; 4];
    cursor
        .read_exact(&mut magic)
        .map_err(|_| "shorter than a header".to_owned())?;
    if &magic != b"GGUF" {
        return Err("not a GGUF file".to_owned());
    }
    let version = cursor.u32()?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(format!(
            "GGUF version {version} is not one this reader knows"
        ));
    }
    let _tensors = cursor.u64()?;
    let pairs = cursor.u64()?;

    let mut header = Header::default();
    let mut shard_no = None;
    let mut shard_of = None;
    for _ in 0..pairs {
        if cursor.consumed > BUDGET_BYTES {
            return Err("the metadata table is larger than any real one".to_owned());
        }
        let key = cursor.string()?;
        let kind = cursor.u32()?;
        match key.as_str() {
            "general.name" => header.name = Some(cursor.string_value(kind)?),
            "general.architecture" => header.architecture = Some(cursor.string_value(kind)?),
            "general.size_label" => header.size_label = Some(cursor.string_value(kind)?),
            "general.file_type" => header.format = format_name(cursor.integer_value(kind)?),
            "split.no" => shard_no = Some(cursor.integer_value(kind)? as u32),
            "split.count" => shard_of = Some(cursor.integer_value(kind)? as u32),
            _ => cursor.skip_value(kind)?,
        }
    }
    if let (Some(no), Some(of)) = (shard_no, shard_of) {
        // Written zero-based; said one-based, as the filenames are.
        header.shard = Some((no + 1, of));
    }
    Ok(header)
}

/// The weight formats llama.cpp numbers in `general.file_type`.
///
/// Only the numbers that name a whole-file format. A number outside the table
/// is left unnamed.
fn format_name(file_type: u64) -> Option<String> {
    let name = match file_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ4_XS",
        29 => "IQ1_M",
        30 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4",
        _ => return None,
    };
    Some(name.to_owned())
}

/// Value type tags, from the GGUF specification.
mod tag {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

/// A reader that counts what it has consumed, so the budget can be enforced.
struct Counting<'a, R> {
    inner: &'a mut R,
    consumed: u64,
}

impl<R: Read + Seek> Counting<'_, R> {
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.inner.read_exact(buf)?;
        self.consumed += buf.len() as u64;
        Ok(())
    }

    fn skip(&mut self, bytes: u64) -> Result<(), String> {
        let offset = i64::try_from(bytes).map_err(|_| "a length past any file".to_owned())?;
        self.inner
            .seek(SeekFrom::Current(offset))
            .map_err(|e| format!("could not skip ahead: {e}"))?;
        self.consumed += bytes;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)
            .map_err(|_| "cut short".to_owned())?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, String> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)
            .map_err(|_| "cut short".to_owned())?;
        Ok(u64::from_le_bytes(b))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u64()?;
        if len > 1 << 20 {
            return Err("a string longer than any key or name".to_owned());
        }
        let mut bytes = vec![0u8; len as usize];
        self.read_exact(&mut bytes)
            .map_err(|_| "cut short".to_owned())?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// A value that must be a string.
    fn string_value(&mut self, kind: u32) -> Result<String, String> {
        if kind == tag::STRING {
            self.string()
        } else {
            self.skip_value(kind)?;
            Err(format!("expected a string, found type {kind}"))
        }
    }

    /// A value that must be an integer of some width.
    fn integer_value(&mut self, kind: u32) -> Result<u64, String> {
        match kind {
            tag::UINT8 | tag::INT8 | tag::BOOL => {
                let mut b = [0u8; 1];
                self.read_exact(&mut b)
                    .map_err(|_| "cut short".to_owned())?;
                Ok(u64::from(b[0]))
            }
            tag::UINT16 | tag::INT16 => {
                let mut b = [0u8; 2];
                self.read_exact(&mut b)
                    .map_err(|_| "cut short".to_owned())?;
                Ok(u64::from(u16::from_le_bytes(b)))
            }
            tag::UINT32 | tag::INT32 => self.u32().map(u64::from),
            tag::UINT64 | tag::INT64 => self.u64(),
            _ => {
                self.skip_value(kind)?;
                Err(format!("expected an integer, found type {kind}"))
            }
        }
    }

    /// Step over a value of any type without keeping it.
    fn skip_value(&mut self, kind: u32) -> Result<(), String> {
        match kind {
            tag::UINT8 | tag::INT8 | tag::BOOL => self.skip(1),
            tag::UINT16 | tag::INT16 => self.skip(2),
            tag::UINT32 | tag::INT32 | tag::FLOAT32 => self.skip(4),
            tag::UINT64 | tag::INT64 | tag::FLOAT64 => self.skip(8),
            tag::STRING => {
                let len = self.u64()?;
                self.skip(len)
            }
            tag::ARRAY => {
                let element = self.u32()?;
                let count = self.u64()?;
                match element {
                    // Fixed-width elements are one seek. A count that
                    // overflows is a malformed header, and the seek says so.
                    tag::UINT8 | tag::INT8 | tag::BOOL => self.skip(count),
                    tag::UINT16 | tag::INT16 => self.skip(count.saturating_mul(2)),
                    tag::UINT32 | tag::INT32 | tag::FLOAT32 => self.skip(count.saturating_mul(4)),
                    tag::UINT64 | tag::INT64 | tag::FLOAT64 => self.skip(count.saturating_mul(8)),
                    // Strings carry their own lengths, so each is a read and a
                    // seek. The vocabulary is the one array that is large.
                    _ => {
                        for _ in 0..count {
                            self.skip_value(element)?;
                        }
                        Ok(())
                    }
                }
            }
            other => Err(format!("unknown value type {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a header the way a converter would: magic, version, counts, then
    /// pairs, with the vocabulary in the way of the keys that matter.
    fn header(pairs: &[(&str, Value)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
        for (key, value) in pairs {
            out.extend_from_slice(&(key.len() as u64).to_le_bytes());
            out.extend_from_slice(key.as_bytes());
            value.write(&mut out);
        }
        out
    }

    enum Value {
        Str(&'static str),
        U32(u32),
        U16(u16),
        F32(f32),
        Strings(usize),
        Ints(usize),
    }

    impl Value {
        fn write(&self, out: &mut Vec<u8>) {
            match self {
                Self::Str(s) => {
                    out.extend_from_slice(&tag::STRING.to_le_bytes());
                    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                    out.extend_from_slice(s.as_bytes());
                }
                Self::U32(v) => {
                    out.extend_from_slice(&tag::UINT32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Self::U16(v) => {
                    out.extend_from_slice(&tag::UINT16.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Self::F32(v) => {
                    out.extend_from_slice(&tag::FLOAT32.to_le_bytes());
                    out.extend_from_slice(&v.to_le_bytes());
                }
                Self::Strings(n) => {
                    out.extend_from_slice(&tag::ARRAY.to_le_bytes());
                    out.extend_from_slice(&tag::STRING.to_le_bytes());
                    out.extend_from_slice(&(*n as u64).to_le_bytes());
                    for i in 0..*n {
                        let token = format!("tok{i}");
                        out.extend_from_slice(&(token.len() as u64).to_le_bytes());
                        out.extend_from_slice(token.as_bytes());
                    }
                }
                Self::Ints(n) => {
                    out.extend_from_slice(&tag::ARRAY.to_le_bytes());
                    out.extend_from_slice(&tag::INT32.to_le_bytes());
                    out.extend_from_slice(&(*n as u64).to_le_bytes());
                    for i in 0..*n {
                        let value = i32::try_from(i).expect("a small test array");
                        out.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }
    }

    #[test]
    fn the_general_keys_are_read_past_the_vocabulary() {
        let bytes = header(&[
            ("general.architecture", Value::Str("qwen3moe")),
            ("general.name", Value::Str("Qwen3 30B A3B Instruct 2507")),
            ("qwen3moe.block_count", Value::U32(48)),
            ("tokenizer.ggml.tokens", Value::Strings(20_000)),
            ("tokenizer.ggml.token_type", Value::Ints(20_000)),
            ("general.size_label", Value::Str("30B-A3B")),
            ("general.file_type", Value::U32(15)),
            ("general.quantization_version", Value::U16(2)),
            ("qwen3moe.rope.freq_base", Value::F32(1e6)),
        ]);
        let found = parse(&mut Cursor::new(bytes)).expect("parses");
        assert_eq!(found.architecture.as_deref(), Some("qwen3moe"));
        assert_eq!(found.name.as_deref(), Some("Qwen3 30B A3B Instruct 2507"));
        assert_eq!(found.size_label.as_deref(), Some("30B-A3B"));
        assert_eq!(found.format.as_deref(), Some("Q4_K_M"));
        assert_eq!(found.shard, None);
    }

    #[test]
    fn a_shard_says_which_it_is_one_based() {
        let bytes = header(&[
            ("general.architecture", Value::Str("llama")),
            ("split.no", Value::U16(0)),
            ("split.count", Value::U16(3)),
        ]);
        let found = parse(&mut Cursor::new(bytes)).expect("parses");
        assert_eq!(found.shard, Some((1, 3)));
    }

    #[test]
    fn an_unnumbered_format_is_left_unnamed() {
        let bytes = header(&[("general.file_type", Value::U32(999))]);
        let found = parse(&mut Cursor::new(bytes)).expect("parses");
        assert_eq!(found.format, None);
    }

    #[test]
    fn what_is_not_a_gguf_is_refused_by_name() {
        assert!(parse(&mut Cursor::new(b"GGML....".to_vec()))
            .unwrap_err()
            .contains("not a GGUF"));
        assert!(parse(&mut Cursor::new(b"GG".to_vec()))
            .unwrap_err()
            .contains("shorter"));

        let mut v1 = header(&[]);
        v1[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert!(parse(&mut Cursor::new(v1))
            .unwrap_err()
            .contains("version 1"));
    }

    #[test]
    fn a_header_cut_short_is_an_error_not_a_hang() {
        let mut bytes = header(&[("general.name", Value::Str("Something"))]);
        bytes.truncate(bytes.len() - 4);
        assert!(parse(&mut Cursor::new(bytes)).is_err());
    }
}
