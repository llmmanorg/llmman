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
use std::io::{BufReader, Read, Seek};
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
    /// An array of at most [`MAX_ARRAY_ELEMENTS`] elements.
    Array(Vec<Value>),
    /// A longer array, by its length: its elements were skipped. A tokenizer's
    /// `tokens` and `merges` cost tens of megabytes as `Value`s.
    LargeArray(u64),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// The byte size of a value of type `ty` when every value of it has the
/// same one: the numbers and booleans, not strings or arrays.
fn fixed_size(ty: u32) -> Option<i64> {
    match ty {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

/// Elements past this many are not read: the array is a [`Value::LargeArray`].
/// Ollama's `/api/show` drops longer arrays too (`fs/gguf/metadata.go`).
pub const MAX_ARRAY_ELEMENTS: usize = 1024;

/// What a hostile or corrupt header is held to, read or skipped.
const MAX_DEPTH: u32 = 8;
const MAX_STRING_BYTES: u64 = 64 * 1024 * 1024;
const MAX_ARRAY_LEN: u64 = 10_000_000;

fn check_depth(depth: u32) -> anyhow::Result<()> {
    if depth > MAX_DEPTH {
        bail!("GGUF value nesting too deep (> {MAX_DEPTH} levels)");
    }
    Ok(())
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

    /// Whether this is an embedding model: it has `{architecture}.pooling_type`,
    /// ollama's own test.
    pub fn is_embedding(&self) -> bool {
        self.architecture()
            .is_some_and(|arch| self.u64(&format!("{arch}.pooling_type")).is_some())
    }

    /// Whether this is a decision model (`{architecture}.decision.type`, as
    /// llama.cpp and ollama test it): it answers typed questions through
    /// `/v1/systemone` and generates no text.
    pub fn is_decision(&self) -> bool {
        self.architecture().is_some_and(|arch| {
            self.str(&format!("{arch}.decision.type"))
                .is_some_and(|kind| !kind.is_empty())
        })
    }

    /// Whether this is a Canary speech model: `general.architecture` is
    /// `canary`, the header `speech::model::load` requires. Such a GGUF
    /// transcribes audio and nothing else, so listings, `/api/show` and
    /// the daemon's refusals all read this one answer.
    pub fn is_speech(&self) -> bool {
        self.architecture() == Some("canary")
    }

    /// `{architecture}.block_count` (i.e. number of transformer layers).
    pub fn block_count(&self) -> Option<u64> {
        let arch = self.architecture()?;
        self.u64(&format!("{arch}.block_count"))
    }
}

struct Reader<R> {
    inner: BufReader<R>,
    /// An array of more elements than this is skipped, not read.
    max_array_elements: u64,
}

impl<R: Read + Seek> Reader<R> {
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
        let len = self.string_len()?;
        let mut buf = vec![0u8; len as usize];
        self.inner.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    /// A string's byte length. A malformed/corrupt one must not attempt a
    /// multi-GiB allocation on `show`'s behalf: real GGUF strings (names,
    /// templates, license text) are never anywhere close to
    /// [`MAX_STRING_BYTES`].
    fn string_len(&mut self) -> anyhow::Result<u64> {
        let len = self.u64()?;
        if len > MAX_STRING_BYTES {
            bail!("implausible GGUF string length: {len}");
        }
        Ok(len)
    }

    /// An array's element type and length, the length held to
    /// [`MAX_ARRAY_LEN`].
    fn array_header(&mut self) -> anyhow::Result<(u32, u64)> {
        let elem_ty = self.u32()?;
        let len = self.u64()?;
        if len > MAX_ARRAY_LEN {
            bail!("implausible GGUF array length: {len}");
        }
        Ok((elem_ty, len))
    }

    fn value(&mut self, ty: u32) -> anyhow::Result<Value> {
        self.value_at(ty, 0)
    }

    /// Steps over a value of type `ty` without building it, under the same
    /// bounds as [`Self::value_at`]. A skipped value is not checked to be
    /// there: a seek can pass the end of the file, which [`read_info`]
    /// checks for once the whole header is read.
    fn skip_value_at(&mut self, ty: u32, depth: u32) -> anyhow::Result<()> {
        check_depth(depth)?;
        if let Some(size) = fixed_size(ty) {
            self.inner.seek_relative(size)?;
            return Ok(());
        }
        match ty {
            8 => {
                let len = self.string_len()?;
                self.inner.seek_relative(len as i64)?;
            }
            9 => {
                let (elem_ty, len) = self.array_header()?;
                self.skip_elements(elem_ty, len, depth + 1)?;
            }
            other => bail!("unknown GGUF value type: {other}"),
        }
        Ok(())
    }

    /// Steps over `len` values of type `ty`: one seek when they all have
    /// the same size.
    fn skip_elements(&mut self, ty: u32, len: u64, depth: u32) -> anyhow::Result<()> {
        // No elements, no nesting: `value_at` accepts an empty array at the
        // deepest level too.
        if len > 0 {
            check_depth(depth)?;
        }
        match fixed_size(ty) {
            Some(size) => self.inner.seek_relative(size * len as i64)?,
            None => {
                for _ in 0..len {
                    self.skip_value_at(ty, depth)?;
                }
            }
        }
        Ok(())
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
        check_depth(depth)?;
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
                let (elem_ty, len) = self.array_header()?;
                if len > self.max_array_elements {
                    self.skip_elements(elem_ty, len, depth + 1)?;
                    return Ok(Value::LargeArray(len));
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
///
/// An array past [`MAX_ARRAY_ELEMENTS`] is a [`Value::LargeArray`]: its
/// length, not its elements. A caller that needs them (a tokenizer's
/// pieces) reads with [`read_info_full`].
pub fn read_info(path: &Path) -> anyhow::Result<Info> {
    read_info_up_to(path, MAX_ARRAY_ELEMENTS as u64)
}

/// [`read_info`] with every array read, however long: a [`Value::Array`]
/// each, as before the long ones were skipped.
pub fn read_info_full(path: &Path) -> anyhow::Result<Info> {
    read_info_up_to(path, u64::MAX)
}

fn read_info_up_to(path: &Path, max_array_elements: u64) -> anyhow::Result<Info> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut r = Reader {
        inner: BufReader::new(file),
        max_array_elements,
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

    // A skip seeks without reading, and a seek past the end of the file
    // succeeds: a header cut off inside a skipped array, with no read after
    // it, would otherwise pass.
    let end = r.inner.stream_position()?;
    if end > file_len {
        bail!("GGUF header runs past the end of the file ({end} > {file_len} bytes)");
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

/// `general.file_type`'s name — llama.cpp's `llama_ftype`, transcribed
/// from `include/llama.h` at the pinned `LLAMA_CPP_RELEASE`. The file
/// type keeps the mixed-quant variant that [`dominant_quantization`]
/// cannot see: `Q4_K_M` and `Q4_K_S` are both modal `Q4_K` by tensor
/// type, and differ only in which tensors got the larger one. `None` for
/// a value this llama.cpp has no name for, including the removed ones
/// and `LLAMA_FTYPE_GUESSED`. New quantizations land in that enum (39-41
/// are recent), so this table goes stale silently — which is why callers
/// fall back to [`dominant_quantization`] rather than treating a `None`
/// as "unquantized": a stale entry costs the `_M`/`_S` variant, not the
/// answer.
pub fn file_type_name(ftype: u64) -> Option<&'static str> {
    Some(match ftype {
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
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        39 => "NVFP4",
        40 => "Q1_0",
        41 => "Q2_0",
        _ => return None,
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
pub(crate) fn ggml_type_name(ty: u32) -> &'static str {
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

/// Test fixture (also for `cmd::serve`): a minimal GGUF with
/// `general.architecture = "llama"`, `llama.context_length = 4096`, any
/// extra `UINT32` keys, and one 2-D Q4_K tensor. The caller removes it.
#[cfg(test)]
pub(crate) fn write_test_gguf_with(extra_u32: &[(&str, u32)]) -> std::path::PathBuf {
    write_test_gguf_with_strings(extra_u32, &[])
}

/// [`write_test_gguf_with`] plus `STRING` keys.
#[cfg(test)]
pub(crate) fn write_test_gguf_with_strings(
    extra_u32: &[(&str, u32)],
    extra_str: &[(&str, &str)],
) -> std::path::PathBuf {
    use std::io::Write;

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes()); // version
    buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    let kv_count = 2 + extra_u32.len() + extra_str.len();
    buf.extend_from_slice(&(kv_count as u64).to_le_bytes()); // metadata_kv_count

    // general.architecture = "llama"
    write_string(&mut buf, "general.architecture");
    buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
    write_string(&mut buf, "llama");

    // llama.context_length = 4096 (u32)
    write_string(&mut buf, "llama.context_length");
    buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
    buf.extend_from_slice(&4096u32.to_le_bytes());

    for (key, value) in extra_u32 {
        write_string(&mut buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes()); // UINT32
        buf.extend_from_slice(&value.to_le_bytes());
    }

    for (key, value) in extra_str {
        write_string(&mut buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        write_string(&mut buf, value);
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_gguf() -> std::path::PathBuf {
        write_test_gguf_with(&[])
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

    #[test]
    fn is_decision_needs_a_decision_type_under_the_architecture() {
        let info = |strings: &[(&str, &str)]| {
            let path = write_test_gguf_with_strings(&[], strings);
            let info = read_info(&path).unwrap();
            std::fs::remove_file(&path).ok();
            info
        };
        assert!(info(&[("llama.decision.type", "clef")]).is_decision());
        assert!(!info(&[]).is_decision());
        assert!(!info(&[("llama.decision.type", "")]).is_decision());
        // Another architecture's key says nothing about this one.
        assert!(!info(&[("clef.decision.type", "clef")]).is_decision());
    }

    #[test]
    fn is_speech_keys_off_the_canary_architecture() {
        let info = |strings: &[(&str, &str)]| {
            let path = write_test_gguf_with_strings(&[], strings);
            let info = read_info(&path).unwrap();
            std::fs::remove_file(&path).ok();
            info
        };
        assert!(info(&[("general.architecture", "canary")]).is_speech());
        assert!(!info(&[("general.architecture", "llama")]).is_speech());
        assert!(!info(&[]).is_speech());
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

    /// The variant the modal tensor type cannot see: both are Q4_K by
    /// tensor type and differ only in the file type.
    #[test]
    fn file_type_name_keeps_the_mixed_quant_variant() {
        assert_eq!(file_type_name(14), Some("Q4_K_S"));
        assert_eq!(file_type_name(15), Some("Q4_K_M"));
        assert_eq!(file_type_name(0), Some("F32"));
        assert_eq!(file_type_name(32), Some("BF16"));
        // Removed from gguf files, and "not specified in the model file".
        assert_eq!(file_type_name(33), None);
        assert_eq!(file_type_name(1024), None);
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

    /// A header's bytes: `kvs` (key, type, raw value bytes), then, when
    /// `close`, a `llama.context_length`, whose value is read correctly only
    /// if every skip before it landed exactly on the next key.
    fn gguf_bytes(kvs: &[(&str, u32, Vec<u8>)], close: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&(kvs.len() as u64 + 1 + u64::from(close)).to_le_bytes());
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_string(&mut buf, "llama");
        for (key, ty, value) in kvs {
            write_string(&mut buf, key);
            buf.extend_from_slice(&ty.to_le_bytes());
            buf.extend_from_slice(value);
        }
        if close {
            write_string(&mut buf, "llama.context_length");
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&4096u32.to_le_bytes());
        }
        buf
    }

    fn write_temp_gguf(buf: &[u8]) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(buf).unwrap();
        file
    }

    /// A header with `kvs` in front of a closing `llama.context_length`.
    fn gguf_with_kvs(kvs: &[(&str, u32, Vec<u8>)]) -> tempfile::NamedTempFile {
        write_temp_gguf(&gguf_bytes(kvs, true))
    }

    /// An ARRAY value's bytes: its element type, its length, then `body`.
    fn array_bytes(elem_ty: u32, len: u64, body: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&elem_ty.to_le_bytes());
        v.extend_from_slice(&len.to_le_bytes());
        v.extend_from_slice(body);
        v
    }

    /// `llmman serve` kept tens of megabytes for good after one
    /// `/api/show`, most of it a tokenizer's arrays of strings read as
    /// `Value`s. Past the ceiling they are skipped, and what follows must
    /// still be read from the right place.
    #[test]
    fn read_info_skips_arrays_past_the_ceiling_and_stays_aligned() {
        let n = MAX_ARRAY_ELEMENTS as u64 + 1;
        let mut strings = Vec::new();
        for i in 0..n {
            write_string(&mut strings, &format!("token-{i}"));
        }
        // an array of arrays, long enough to be skipped, each of 2 bytes
        let mut nested = Vec::new();
        for _ in 0..n {
            nested.extend_from_slice(&array_bytes(0, 2, &[7, 7]));
        }
        let file = gguf_with_kvs(&[
            ("tok.tokens", 9, array_bytes(8, n, &strings)),
            ("tok.types", 9, array_bytes(5, 5000, &vec![0u8; 5000 * 4])),
            ("tok.nested", 9, array_bytes(9, n, &nested)),
            ("tok.small", 9, array_bytes(4, 2, &[1, 0, 0, 0, 2, 0, 0, 0])),
            (
                "tok.exact",
                9,
                array_bytes(0, MAX_ARRAY_ELEMENTS as u64, &vec![0u8; MAX_ARRAY_ELEMENTS]),
            ),
            ("tok.after", 4, 99u32.to_le_bytes().to_vec()),
        ]);
        let info = read_info(file.path()).unwrap();

        assert!(matches!(info.metadata["tok.tokens"], Value::LargeArray(l) if l == n));
        assert!(matches!(
            info.metadata["tok.types"],
            Value::LargeArray(5000)
        ));
        assert!(matches!(info.metadata["tok.nested"], Value::LargeArray(l) if l == n));
        // an array at the ceiling is still read
        assert!(
            matches!(&info.metadata["tok.exact"], Value::Array(a) if a.len() == MAX_ARRAY_ELEMENTS)
        );
        match &info.metadata["tok.small"] {
            Value::Array(a) => assert_eq!(a.len(), 2),
            other => panic!("{other:?}"),
        }
        assert_eq!(info.u64("tok.after"), Some(99));
        assert_eq!(info.context_length(), Some(4096), "the key after the skips");
    }

    /// A skipped array's elements are still held to the reader's bounds.
    #[test]
    fn read_info_still_rejects_a_bad_array_it_would_skip() {
        let n = MAX_ARRAY_ELEMENTS as u64 + 1;
        // unknown element type
        let file = gguf_with_kvs(&[("tok.bad", 9, array_bytes(99, n, &[]))]);
        let err = read_info(file.path()).unwrap_err();
        assert!(
            format!("{err:#}").contains("unknown GGUF value type"),
            "{err:#}"
        );

        // an implausible string length inside it
        let mut body = Vec::new();
        body.extend_from_slice(&(u64::MAX).to_le_bytes());
        let file = gguf_with_kvs(&[("tok.bad", 9, array_bytes(8, n, &body))]);
        let err = read_info(file.path()).unwrap_err();
        assert!(
            format!("{err:#}").contains("implausible GGUF string length"),
            "{err:#}"
        );

        // an implausible array length
        let file = gguf_with_kvs(&[("tok.bad", 9, array_bytes(0, 10_000_001, &[]))]);
        let err = read_info(file.path()).unwrap_err();
        assert!(
            format!("{err:#}").contains("implausible GGUF array length"),
            "{err:#}"
        );
    }

    /// `speech::model::load` needs a tokenizer's pieces as an array, however
    /// many: Canary's are 16,384. `read_info_full` reads them all.
    #[test]
    fn read_info_full_reads_the_arrays_read_info_skips() {
        let n = 16_384u64;
        let mut strings = Vec::new();
        for i in 0..n {
            write_string(&mut strings, &format!("piece-{i}"));
        }
        let file = gguf_with_kvs(&[("tok.tokens", 9, array_bytes(8, n, &strings))]);
        let skipped = read_info(file.path()).unwrap();
        let full = read_info_full(file.path()).unwrap();

        assert!(matches!(skipped.metadata["tok.tokens"], Value::LargeArray(l) if l == n));
        match &full.metadata["tok.tokens"] {
            Value::Array(a) => {
                assert_eq!(a.len() as u64, n);
                assert_eq!(a[16_383].as_str(), Some("piece-16383"));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(full.context_length(), Some(4096), "the key after it");
    }

    /// A seek past the end of the file succeeds, so a header cut off inside
    /// a skipped array, with no read after it, must still be rejected.
    #[test]
    fn read_info_rejects_a_header_that_ends_inside_a_skipped_array() {
        let n = MAX_ARRAY_ELEMENTS as u64 + 1;
        let one_short = |each: Vec<u8>, last: Vec<u8>| {
            let mut body = each.repeat(n as usize - 1);
            body.extend(last);
            body
        };
        let mut a_string = Vec::new();
        write_string(&mut a_string, "a");
        let mut a_cut_string = 1000u64.to_le_bytes().to_vec(); // 1000 bytes, 5 there
        a_cut_string.extend_from_slice(b"short");
        let a_cut_array = array_bytes(4, 1_000_000, &[]);
        let cut = [
            // each is the last value in the file, with no tensors after it
            ("u32s", array_bytes(4, n, &[0u8; 8])),
            (
                "strings",
                array_bytes(8, n, &one_short(a_string, a_cut_string)),
            ),
            (
                "arrays",
                array_bytes(9, n, &one_short(array_bytes(0, 0, &[]), a_cut_array)),
            ),
        ];
        for (what, value) in cut {
            let file = write_temp_gguf(&gguf_bytes(&[("tok.cut", 9, value)], false));
            let err = read_info(file.path());
            let err = format!("{:#}", err.unwrap_err());
            assert!(err.contains("past the end of the file"), "{what}: {err}");
        }

        // The same array, complete and last, is read: the end is not past it.
        let kv = (
            "tok.whole",
            9,
            array_bytes(4, n, &vec![0u8; n as usize * 4]),
        );
        let file = write_temp_gguf(&gguf_bytes(&[kv], false));
        let info = read_info(file.path()).unwrap();
        assert!(matches!(info.metadata["tok.whole"], Value::LargeArray(l) if l == n));
    }

    /// An empty array has no elements to nest: at the deepest level it is
    /// read, so it is skipped too, inside an array long enough to be skipped.
    #[test]
    fn read_info_skips_an_empty_array_at_the_deepest_level() {
        // Depths 1 to 7 are arrays of one array; the one at depth 8 is empty.
        let mut chain = array_bytes(4, 0, &[]);
        for _ in 1..8 {
            chain = array_bytes(9, 1, &chain);
        }
        let n = MAX_ARRAY_ELEMENTS as u64 + 1;
        let body: Vec<u8> = (0..n).flat_map(|_| chain.clone()).collect();
        let file = gguf_with_kvs(&[("tok.deep", 9, array_bytes(9, n, &body))]);
        let info = read_info(file.path());
        let info = info.unwrap();
        assert!(matches!(info.metadata["tok.deep"], Value::LargeArray(l) if l == n));
        assert_eq!(info.context_length(), Some(4096));
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
