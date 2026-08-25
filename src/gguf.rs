//! Minimal GGUF (<https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>)
//! header reader — reads just the metadata key/value section and the
//! tensor-info array (name/shape/type — never the tensor *data* itself),
//! which is all `cmd::show` needs to report a GGUF model's architecture,
//! parameter count, quantization, context length, and chat template
//! without extracting/loading the whole file.
//!
//! Deliberately narrower than a full GGUF library: no writing, no
//! alignment/padding handling past the tensor-info array, no support for
//! the ancient, tensor-count-as-u32 GGUF v1 format (real models in the
//! wild are v2/v3).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{bail, Context};

/// One parsed GGUF metadata value. See
/// <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#file-structure>
/// for the underlying `gguf_metadata_value_type` this mirrors.
#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Any non-negative integer variant, widened to `u64`.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v as u64),
            Value::U16(v) => Some(v as u64),
            Value::U32(v) => Some(v as u64),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// One entry of the tensor-info array — everything but the tensor's own
/// data, which this reader never touches.
#[derive(Debug, Clone)]
struct TensorInfo {
    #[allow(dead_code)]
    name: String,
    dims: Vec<u64>,
    ggml_type: u32,
}

/// Parsed GGUF header, ready for `cmd::show` to render.
#[derive(Debug, Clone, Default)]
pub struct Info {
    pub metadata: HashMap<String, Value>,
    pub tensor_count: u64,
    /// Total element count across every tensor (i.e. the model's
    /// parameter count) — independent of quantization, matching how
    /// `ollama show`'s own "Parameters" figure counts elements, not
    /// bytes.
    pub parameter_count: u64,
    /// The most common quantization type name among 2-D-or-higher
    /// tensors (the actual weight matrices — 1-D bias/norm tensors are
    /// excluded since they're almost always kept at F32/F16 regardless of
    /// the model's overall quantization and would otherwise skew a
    /// per-tensor-count vote toward "unquantized").
    pub quantization: Option<String>,
}

impl Info {
    /// Convenience accessor for a string-valued key.
    pub fn str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(Value::as_str)
    }

    /// Convenience accessor for an integer-valued key.
    pub fn u64(&self, key: &str) -> Option<u64> {
        self.metadata.get(key).and_then(Value::as_u64)
    }

    /// `general.architecture` — e.g. "llama", "qwen3", "gemma3" — used to
    /// look up architecture-prefixed keys like `{arch}.context_length`.
    pub fn architecture(&self) -> Option<&str> {
        self.str("general.architecture")
    }

    /// `{architecture}.context_length`, if both are present.
    pub fn context_length(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.u64(&format!("{arch}.context_length"))
    }

    /// `{architecture}.embedding_length`, if both are present.
    pub fn embedding_length(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.u64(&format!("{arch}.embedding_length"))
    }

    /// `{architecture}.block_count` (i.e. number of transformer layers).
    pub fn block_count(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.u64(&format!("{arch}.block_count"))
    }
}

struct Reader<R> {
    inner: R,
}

impl<R: Read> Reader<R> {
    fn u8(&mut self) -> anyhow::Result<u8> {
        let mut b = [0u8; 1];
        self.inner.read_exact(&mut b)?;
        Ok(b[0])
    }
    fn i8(&mut self) -> anyhow::Result<i8> {
        Ok(self.u8()? as i8)
    }
    fn bool_(&mut self) -> anyhow::Result<bool> {
        Ok(self.u8()? != 0)
    }
    fn u16(&mut self) -> anyhow::Result<u16> {
        let mut b = [0u8; 2];
        self.inner.read_exact(&mut b)?;
        Ok(u16::from_le_bytes(b))
    }
    fn i16(&mut self) -> anyhow::Result<i16> {
        let mut b = [0u8; 2];
        self.inner.read_exact(&mut b)?;
        Ok(i16::from_le_bytes(b))
    }
    fn u32(&mut self) -> anyhow::Result<u32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn i32(&mut self) -> anyhow::Result<i32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }
    fn f32(&mut self) -> anyhow::Result<f32> {
        let mut b = [0u8; 4];
        self.inner.read_exact(&mut b)?;
        Ok(f32::from_le_bytes(b))
    }
    fn u64(&mut self) -> anyhow::Result<u64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn i64(&mut self) -> anyhow::Result<i64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(i64::from_le_bytes(b))
    }
    fn f64(&mut self) -> anyhow::Result<f64> {
        let mut b = [0u8; 8];
        self.inner.read_exact(&mut b)?;
        Ok(f64::from_le_bytes(b))
    }

    /// A GGUF string: a `u64` byte length followed by (not
    /// NUL-terminated) UTF-8 bytes — decoded lossily, since a stray
    /// invalid byte in, say, a license string must never fail the whole
    /// read.
    fn string(&mut self) -> anyhow::Result<String> {
        let len = self.u64()? as usize;
        // A malformed/corrupt length must not attempt a multi-GiB
        // allocation on `show`'s behalf — real GGUF strings (names,
        // templates, license text) are never anywhere close to this.
        if len > 64 * 1024 * 1024 {
            bail!("implausible GGUF string length: {len}");
        }
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn value(&mut self, ty: u32) -> anyhow::Result<Value> {
        self.value_at(ty, 0)
    }

    /// `value`'s real implementation, tracking array-nesting `depth` —
    /// an array's own element type (read from the file, not validated
    /// against anything) can itself be `9` (array), so a hostile or
    /// merely corrupt GGUF file that repeats a nested-array header can
    /// otherwise recurse once per ~12 input bytes. A real stack overflow
    /// aborts the whole process below any `anyhow::Result` this reader
    /// could otherwise report, taking `llmman show` down with it instead
    /// of just failing to read one untrusted (e.g. pulled-from-a-registry)
    /// file — bounding the depth turns that into an ordinary error.
    fn value_at(&mut self, ty: u32, depth: u32) -> anyhow::Result<Value> {
        if depth > 8 {
            bail!("GGUF value nesting too deep (> 8 levels)");
        }
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.i8()?),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.i16()?),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.i32()?),
            6 => Value::F32(self.f32()?),
            7 => Value::Bool(self.bool_()?),
            8 => Value::String(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let len = self.u64()?;
                if len > 10_000_000 {
                    bail!("implausible GGUF array length: {len}");
                }
                let mut items = Vec::with_capacity(len.min(1024) as usize);
                for _ in 0..len {
                    items.push(self.value_at(elem_ty, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.i64()?),
            12 => Value::F64(self.f64()?),
            other => bail!("unknown GGUF value type: {other}"),
        })
    }
}

/// Reads `path`'s GGUF header: the magic/version, every metadata
/// key/value pair, and the tensor-info array (name/shape/type only, never
/// tensor data) — enough to derive parameter count and dominant
/// quantization without loading the model itself.
pub fn read_info(path: &Path) -> anyhow::Result<Info> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut r = Reader {
        inner: BufReader::new(file),
    };

    let mut magic = [0u8; 4];
    r.inner.read_exact(&mut magic).context("read GGUF magic")?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file: {}", path.display());
    }
    let version = r.u32().context("read GGUF version")?;
    if version < 2 {
        bail!("unsupported GGUF version {version} (only v2+ is supported)");
    }

    let tensor_count = r.u64().context("read tensor_count")?;
    let metadata_kv_count = r.u64().context("read metadata_kv_count")?;

    let mut metadata = HashMap::with_capacity(metadata_kv_count.min(4096) as usize);
    for _ in 0..metadata_kv_count {
        let key = r.string().context("read metadata key")?;
        let value_type = r.u32().context("read metadata value type")?;
        let value = r
            .value(value_type)
            .with_context(|| format!("read metadata value for {key:?}"))?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(65536) as usize);
    for _ in 0..tensor_count {
        let name = r.string().context("read tensor name")?;
        let n_dims = r.u32().context("read tensor n_dims")?;
        if n_dims > 8 {
            bail!("implausible tensor rank: {n_dims}");
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(r.u64().context("read tensor dim")?);
        }
        let ggml_type = r.u32().context("read tensor type")?;
        let _offset = r.u64().context("read tensor offset")?;
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
        });
    }

    let parameter_count = tensors
        .iter()
        .map(|t| {
            t.dims
                .iter()
                .fold(1u128, |acc, &d| acc.saturating_mul(d as u128))
        })
        .fold(0u128, |acc, n| acc.saturating_add(n))
        .min(u64::MAX as u128) as u64;

    let quantization = dominant_quantization(&tensors);

    Ok(Info {
        metadata,
        tensor_count,
        parameter_count,
        quantization,
    })
}

/// The `ggml_type` name most representative of a model's actual
/// quantization — the modal type (by total element count, not tensor
/// count, so a handful of huge matrices outweigh many tiny ones) among
/// tensors with rank ≥ 2 (real weight matrices; 1-D bias/norm tensors are
/// excluded — see [`Info::quantization`]'s own doc comment).
fn dominant_quantization(tensors: &[TensorInfo]) -> Option<String> {
    let mut totals: HashMap<u32, u128> = HashMap::new();
    for t in tensors {
        if t.dims.len() < 2 {
            continue;
        }
        let elems = t
            .dims
            .iter()
            .fold(1u128, |acc, &d| acc.saturating_mul(d as u128));
        // Saturating, not `+=`: a corrupt tensor's `dims` can already
        // saturate `elems` to `u128::MAX` (see the fold above); a second
        // tensor sharing the same `ggml_type` would then overflow a plain
        // `+=` and panic in a debug build, aborting `llmman show` instead
        // of just reporting the file's real (if implausible) contents.
        let entry = totals.entry(t.ggml_type).or_insert(0);
        *entry = entry.saturating_add(elems);
    }
    let (ty, _) = totals.into_iter().max_by_key(|(_, n)| *n)?;
    Some(ggml_type_name(ty).to_string())
}

/// `ggml_type` id → name, mirroring `ggml.c`'s own `type_traits[].type_name`
/// table. Deprecated/removed ids (4, 5, 31-33) map to a placeholder rather
/// than panicking or erroring — an old file naming one is still a file we
/// should be able to at least partially describe.
fn ggml_type_name(ty: u32) -> &'static str {
    match ty {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a minimal-but-valid GGUF file in memory: one string
    /// metadata key, one u32 metadata key, and one 2-D tensor, then
    /// writes it to a temp file and reads it back through [`read_info`].
    fn write_test_gguf() -> std::path::PathBuf {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&2u64.to_le_bytes()); // metadata_kv_count

        // general.architecture = "llama"
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        write_string(&mut buf, "llama");

        // llama.context_length = 4096 (u32)
        write_string(&mut buf, "llama.context_length");
        buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        buf.extend_from_slice(&4096u32.to_le_bytes());

        // one tensor: "blk.0.weight", 2 dims [4, 8], type Q4_K (12)
        write_string(&mut buf, "blk.0.weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&8u64.to_le_bytes());
        buf.extend_from_slice(&12u32.to_le_bytes()); // ggml_type = Q4_K
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        let path = std::env::temp_dir().join(format!(
            "llmman-gguf-test-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut f = File::create(&path).unwrap();
        f.write_all(&buf).unwrap();
        path
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn reads_metadata_and_computes_parameter_count_and_quantization() {
        let path = write_test_gguf();
        let info = read_info(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(info.architecture(), Some("llama"));
        assert_eq!(info.context_length(), Some(4096));
        assert_eq!(info.tensor_count, 1);
        assert_eq!(info.parameter_count, 32); // 4 * 8
        assert_eq!(info.quantization, Some("Q4_K".to_string()));
    }

    /// Regression test: two tensors whose declared dimensions each
    /// individually saturate `elems` to `u128::MAX` (a corrupt/hostile
    /// file, not a real one) must not overflow-panic when
    /// `dominant_quantization` sums their per-type totals — see that
    /// function's own comment on why `saturating_add` replaced a plain
    /// `+=` there.
    #[test]
    fn read_info_does_not_overflow_when_summing_saturated_element_counts() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count

        for name in ["blk.0.weight", "blk.1.weight"] {
            write_string(&mut buf, name);
            buf.extend_from_slice(&3u32.to_le_bytes()); // n_dims
                                                        // Three u64::MAX dims: their product vastly exceeds
                                                        // u128::MAX, so `elems` saturates to u128::MAX for this
                                                        // tensor alone.
            for _ in 0..3 {
                buf.extend_from_slice(&u64::MAX.to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type = F32, same for both
            buf.extend_from_slice(&0u64.to_le_bytes()); // offset
        }

        let path = std::env::temp_dir().join(format!(
            "llmman-gguf-saturate-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut f = File::create(&path).unwrap();
        f.write_all(&buf).unwrap();

        // Must not panic (a debug-build overflow would abort the whole
        // process, not just this call) and must report a sane result.
        let info = read_info(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(info.parameter_count, u64::MAX);
        assert_eq!(info.quantization, Some("F32".to_string()));
    }

    #[test]
    fn read_info_rejects_a_non_gguf_file() {
        let path =
            std::env::temp_dir().join(format!("llmman-gguf-nonmagic-{}.bin", std::process::id()));
        std::fs::write(&path, b"not a gguf file at all").unwrap();
        let err = read_info(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(err.to_string().contains("not a GGUF file"));
    }

    #[test]
    fn ggml_type_name_covers_common_quantizations() {
        assert_eq!(ggml_type_name(0), "F32");
        assert_eq!(ggml_type_name(12), "Q4_K");
        assert_eq!(ggml_type_name(14), "Q6_K");
        assert_eq!(ggml_type_name(9999), "unknown");
    }

    /// Regression test: a GGUF file whose metadata contains a
    /// deeply-nested array-of-array-of-... chain must be rejected with an
    /// ordinary error, not recurse until the process's stack overflows —
    /// see `Reader::value_at`'s own doc comment. Built with 12 levels of
    /// nesting (past the 8-level bound) around a single UINT8 leaf.
    #[test]
    fn read_info_rejects_metadata_nested_past_the_depth_limit_instead_of_overflowing_the_stack() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count

        write_string(&mut buf, "test.nested");
        buf.extend_from_slice(&9u32.to_le_bytes()); // top-level value type: ARRAY
        for _ in 0..12 {
            buf.extend_from_slice(&9u32.to_le_bytes()); // element type: ARRAY (nested)
            buf.extend_from_slice(&1u64.to_le_bytes()); // length: 1
        }
        // Innermost leaf: one UINT8 array of length 1.
        buf.extend_from_slice(&0u32.to_le_bytes()); // element type: UINT8
        buf.extend_from_slice(&1u64.to_le_bytes()); // length: 1
        buf.push(0u8);

        let path = std::env::temp_dir().join(format!(
            "llmman-gguf-nested-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut f = File::create(&path).unwrap();
        f.write_all(&buf).unwrap();

        let err = read_info(&path).unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(
            err.chain()
                .any(|c| c.to_string().contains("nesting too deep")),
            "got: {err:#}"
        );
    }
}
