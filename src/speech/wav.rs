//! RIFF/WAVE reader for the one encoding the speech front end accepts:
//! PCM16 little-endian, mono, 16 kHz. Everything else is rejected with a
//! typed error so callers can tell "resample this" from "not a WAV".
//!
//! Layout (RIFF 1991 spec, as written by sox, ffmpeg and the fixture
//! `fleurs-en.wav`): a 12-byte `RIFF <size> WAVE` header followed by
//! chunks of `<id:4><len:u32 LE><body>`, each body padded to an even
//! length. The `fmt ` chunk must come before `data`; chunks with any other
//! id (`LIST`, `fact`, ...) are skipped. The RIFF size field is used only
//! to bound the chunk walk: trailing bytes beyond it are ignored, a file
//! shorter than it is [`WavError::Truncated`].

/// Why [`parse`] rejected the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WavError {
    /// Not a `RIFF .... WAVE` file, a chunk header is malformed, or the
    /// `fmt ` chunk's block alignment or byte rate contradicts its format.
    BadHeader,
    /// The named chunk (`fmt ` or `data`) was not found before the end.
    MissingChunk(&'static str),
    /// The input ends before the RIFF header says it should, or the data
    /// chunk ends in the middle of a sample.
    Truncated,
    /// A chunk declares more bytes than remain.
    OversizedChunk,
    /// The input is longer than the caller's `max_bytes`.
    TooLarge,
    /// Format tag other than 1 (PCM): 3 is float, 0xFFFE extensible.
    UnsupportedFormat(u16),
    /// Channel count other than 1.
    UnsupportedChannels(u16),
    /// Bits per sample other than 16.
    UnsupportedBits(u16),
    /// Sample rate other than 16000 Hz.
    UnsupportedRate(u32),
}

impl std::fmt::Display for WavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadHeader => f.write_str("not a RIFF/WAVE file or malformed chunk header"),
            Self::MissingChunk(id) => write!(f, "missing {:?} chunk", id.trim_end()),
            Self::Truncated => f.write_str("file is shorter than its headers declare"),
            Self::OversizedChunk => f.write_str("chunk length exceeds the available bytes"),
            Self::TooLarge => f.write_str("input is longer than the caller's limit"),
            Self::UnsupportedFormat(tag) => {
                write!(f, "unsupported format tag {tag} (only PCM, tag 1)")
            }
            Self::UnsupportedChannels(n) => write!(f, "unsupported channel count {n} (only mono)"),
            Self::UnsupportedBits(n) => write!(f, "unsupported {n} bits per sample (only 16)"),
            Self::UnsupportedRate(hz) => write!(f, "unsupported sample rate {hz} Hz (only 16000)"),
        }
    }
}

impl std::error::Error for WavError {}

/// The only sample rate the front end accepts.
pub const SAMPLE_RATE: u32 = 16_000;

const FORMAT_PCM: u16 = 1;
const CHANNELS: u16 = 1;
const BITS_PER_SAMPLE: u16 = 16;
const BYTES_PER_SAMPLE: usize = 2;
/// `nBlockAlign` and `nAvgBytesPerSec` of the one accepted encoding.
const BLOCK_ALIGN: u16 = BYTES_PER_SAMPLE as u16;
const BYTE_RATE: u32 = SAMPLE_RATE * BYTES_PER_SAMPLE as u32;
/// Size of the `fmt ` chunk fields this reader needs (through bits per
/// sample). PCM writers emit exactly 16; extensible ones more.
const FMT_MIN_LEN: usize = 16;

/// Decodes `bytes` as a PCM16 LE mono 16 kHz WAV into samples.
///
/// `max_bytes` caps the whole input: anything longer is rejected before
/// parsing starts, which also bounds the returned allocation.
pub fn parse(bytes: &[u8], max_bytes: usize) -> Result<Vec<i16>, WavError> {
    if bytes.len() > max_bytes {
        return Err(WavError::TooLarge);
    }
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::BadHeader);
    }
    let riff_len = u32_at(bytes, 4) as usize;
    // The RIFF size counts everything after its own 8-byte header.
    let riff_end = riff_len.checked_add(8).ok_or(WavError::BadHeader)?;
    if riff_end < 12 {
        return Err(WavError::BadHeader);
    }
    if riff_end > bytes.len() {
        return Err(WavError::Truncated);
    }
    let riff = &bytes[..riff_end];

    let mut fmt_seen = false;
    let mut pos = 12;
    while pos < riff.len() {
        if riff.len() - pos < 8 {
            return Err(WavError::BadHeader);
        }
        let id = &riff[pos..pos + 4];
        let len = u32_at(riff, pos + 4) as usize;
        let body_start = pos + 8;
        let body_end = body_start
            .checked_add(len)
            .filter(|end| *end <= riff.len())
            .ok_or(WavError::OversizedChunk)?;
        let body = &riff[body_start..body_end];
        match id {
            b"fmt " => {
                check_format(body)?;
                fmt_seen = true;
            }
            b"data" => {
                if !fmt_seen {
                    return Err(WavError::MissingChunk("fmt "));
                }
                return decode_pcm16(body);
            }
            _ => {}
        }
        // Chunk bodies are padded to an even length; the pad byte is not
        // counted in `len`. A writer may omit the pad after the last
        // chunk, which the loop condition tolerates.
        pos = body_end + (len & 1);
    }
    Err(WavError::MissingChunk(if fmt_seen {
        "data"
    } else {
        "fmt "
    }))
}

/// Checks the `fmt ` chunk body against the one accepted encoding.
fn check_format(body: &[u8]) -> Result<(), WavError> {
    if body.len() < FMT_MIN_LEN {
        return Err(WavError::BadHeader);
    }
    let format_tag = u16_at(body, 0);
    let channels = u16_at(body, 2);
    let sample_rate = u32_at(body, 4);
    let byte_rate = u32_at(body, 8);
    let block_align = u16_at(body, 12);
    let bits_per_sample = u16_at(body, 14);
    if format_tag != FORMAT_PCM {
        return Err(WavError::UnsupportedFormat(format_tag));
    }
    if channels != CHANNELS {
        return Err(WavError::UnsupportedChannels(channels));
    }
    if bits_per_sample != BITS_PER_SAMPLE {
        return Err(WavError::UnsupportedBits(bits_per_sample));
    }
    if sample_rate != SAMPLE_RATE {
        return Err(WavError::UnsupportedRate(sample_rate));
    }
    // The header must agree with itself: the samples are read as two
    // bytes each, whatever the block alignment says.
    if block_align != BLOCK_ALIGN || byte_rate != BYTE_RATE {
        return Err(WavError::BadHeader);
    }
    Ok(())
}

fn decode_pcm16(body: &[u8]) -> Result<Vec<i16>, WavError> {
    if !body.len().is_multiple_of(BYTES_PER_SAMPLE) {
        return Err(WavError::Truncated);
    }
    let (pairs, _) = body.as_chunks::<BYTES_PER_SAMPLE>();
    Ok(pairs.iter().map(|pair| i16::from_le_bytes(*pair)).collect())
}

/// Callers check `at + 2 <= bytes.len()` through the chunk length checks.
fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

/// Callers check `at + 4 <= bytes.len()` through the chunk length checks.
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    const NO_CAP: usize = usize::MAX;
    const SAMPLES: [i16; 4] = [0, -1, 32767, -32768];

    /// A `fmt ` chunk body with the field layout shared by every PCM
    /// writer: tag, channels, rate, byte rate, block align, bits.
    fn fmt_body(tag: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let block_align = channels * bits / 8;
        let mut body = Vec::new();
        body.extend_from_slice(&tag.to_le_bytes());
        body.extend_from_slice(&channels.to_le_bytes());
        body.extend_from_slice(&rate.to_le_bytes());
        body.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
        body.extend_from_slice(&block_align.to_le_bytes());
        body.extend_from_slice(&bits.to_le_bytes());
        body
    }

    fn pcm_body(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
        chunk_declaring(id, body.len() as u32, body)
    }

    /// A chunk whose length field is `declared` regardless of the body
    /// actually present, with the pad byte when the body is odd.
    fn chunk_declaring(id: &[u8; 4], declared: u32, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(id);
        out.extend_from_slice(&declared.to_le_bytes());
        out.extend_from_slice(body);
        if body.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    /// Wraps chunks in a RIFF/WAVE header with a consistent size field.
    fn riff(chunks: &[Vec<u8>]) -> Vec<u8> {
        let payload: Vec<u8> = chunks.concat();
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&((payload.len() + 4) as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(&payload);
        out
    }

    fn wav_with_fmt(fmt: Vec<u8>) -> Vec<u8> {
        riff(&[chunk(b"fmt ", &fmt), chunk(b"data", &pcm_body(&SAMPLES))])
    }

    fn good_wav() -> Vec<u8> {
        wav_with_fmt(fmt_body(1, 1, 16_000, 16))
    }

    #[test]
    fn parses_pcm16_mono_16k() {
        assert_eq!(parse(&good_wav(), NO_CAP).unwrap(), SAMPLES);
    }

    #[test]
    fn builder_matches_the_fixture_header_layout() {
        // Byte-for-byte the header of fleurs-en.wav with its sizes
        // replaced, so the builder and the real producer agree.
        let wav = good_wav();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(&wav[16..20], &16u32.to_le_bytes());
        assert_eq!(
            &wav[20..36],
            &[1, 0, 1, 0, 0x80, 0x3e, 0, 0, 0, 0x7d, 0, 0, 2, 0, 0x10, 0]
        );
        assert_eq!(&wav[36..40], b"data");
    }

    #[test]
    fn rejects_stereo() {
        let wav = wav_with_fmt(fmt_body(1, 2, 16_000, 16));
        assert_eq!(parse(&wav, NO_CAP), Err(WavError::UnsupportedChannels(2)));
    }

    #[test]
    fn rejects_8_bit() {
        let wav = wav_with_fmt(fmt_body(1, 1, 16_000, 8));
        assert_eq!(parse(&wav, NO_CAP), Err(WavError::UnsupportedBits(8)));
    }

    #[test]
    fn rejects_44100_hz() {
        let wav = wav_with_fmt(fmt_body(1, 1, 44_100, 16));
        assert_eq!(parse(&wav, NO_CAP), Err(WavError::UnsupportedRate(44_100)));
    }

    #[test]
    fn rejects_float_format_tag() {
        let wav = wav_with_fmt(fmt_body(3, 1, 16_000, 32));
        assert_eq!(parse(&wav, NO_CAP), Err(WavError::UnsupportedFormat(3)));
    }

    #[test]
    fn rejects_extensible_format_tag() {
        let mut fmt = fmt_body(0xFFFE, 1, 16_000, 16);
        // cbSize plus the 22-byte extension an extensible fmt carries.
        fmt.extend_from_slice(&22u16.to_le_bytes());
        fmt.extend_from_slice(&[0; 22]);
        let wav = wav_with_fmt(fmt);
        assert_eq!(
            parse(&wav, NO_CAP),
            Err(WavError::UnsupportedFormat(0xFFFE))
        );
    }

    #[test]
    fn rejects_truncated_data_chunk() {
        let mut wav = good_wav();
        wav.truncate(wav.len() - 3);
        assert_eq!(parse(&wav, NO_CAP), Err(WavError::Truncated));
    }

    #[test]
    fn rejects_data_chunk_ending_mid_sample() {
        let mut data = pcm_body(&SAMPLES);
        data.push(0x7f);
        let wav = riff(&[
            chunk(b"fmt ", &fmt_body(1, 1, 16_000, 16)),
            chunk(b"data", &data),
        ]);
        assert_eq!(parse(&wav, NO_CAP), Err(WavError::Truncated));
    }

    #[test]
    fn rejects_declared_data_length_beyond_the_file() {
        let fmt = chunk(b"fmt ", &fmt_body(1, 1, 16_000, 16));
        for declared in [9, 1_000, u32::MAX] {
            let data = chunk_declaring(b"data", declared, &pcm_body(&SAMPLES));
            let wav = riff(&[fmt.clone(), data]);
            assert_eq!(
                parse(&wav, NO_CAP),
                Err(WavError::OversizedChunk),
                "declared {declared}"
            );
        }
    }

    #[test]
    fn skips_unknown_odd_length_chunk_before_data() {
        let wav = riff(&[
            chunk(b"fmt ", &fmt_body(1, 1, 16_000, 16)),
            chunk(b"LIST", b"INFOISFT\x05\x00\x00\x00sox\x00\x00"),
            chunk(b"fact", &[1, 2, 3]),
            chunk(b"data", &pcm_body(&SAMPLES)),
        ]);
        assert_eq!(parse(&wav, NO_CAP).unwrap(), SAMPLES);
    }

    #[test]
    fn rejects_input_above_max_bytes() {
        let wav = good_wav();
        assert_eq!(parse(&wav, wav.len() - 1), Err(WavError::TooLarge));
        assert_eq!(parse(&wav, wav.len()).unwrap(), SAMPLES);
    }

    #[test]
    fn rejects_non_riff_and_short_input() {
        assert_eq!(parse(b"", NO_CAP), Err(WavError::BadHeader));
        assert_eq!(
            parse(b"RIFF\x00\x00\x00\x00WAV", NO_CAP),
            Err(WavError::BadHeader)
        );
        let mut wav = good_wav();
        wav[8..12].copy_from_slice(b"AVI ");
        assert_eq!(parse(&wav, NO_CAP), Err(WavError::BadHeader));
    }

    #[test]
    fn reports_missing_chunks() {
        let fmt = chunk(b"fmt ", &fmt_body(1, 1, 16_000, 16));
        let data = chunk(b"data", &pcm_body(&SAMPLES));
        assert_eq!(
            parse(&riff(std::slice::from_ref(&fmt)), NO_CAP),
            Err(WavError::MissingChunk("data"))
        );
        assert_eq!(
            parse(&riff(std::slice::from_ref(&data)), NO_CAP),
            Err(WavError::MissingChunk("fmt "))
        );
        assert_eq!(
            parse(&riff(&[data, fmt]), NO_CAP),
            Err(WavError::MissingChunk("fmt "))
        );
    }

    #[test]
    fn rejects_inconsistent_block_align_or_byte_rate() {
        // Bytes 8..12 of the fmt body are the byte rate, 12..14 the block
        // alignment.
        for (at, bad) in [(8, &0u32.to_le_bytes()[..]), (12, &4u16.to_le_bytes()[..])] {
            let mut fmt = fmt_body(1, 1, 16_000, 16);
            fmt[at..at + bad.len()].copy_from_slice(bad);
            assert_eq!(
                parse(&wav_with_fmt(fmt), NO_CAP),
                Err(WavError::BadHeader),
                "field at {at}"
            );
        }
    }

    #[test]
    fn rejects_short_fmt_chunk() {
        let wav = riff(&[
            chunk(b"fmt ", &[1, 0, 1, 0]),
            chunk(b"data", &pcm_body(&SAMPLES)),
        ]);
        assert_eq!(parse(&wav, NO_CAP), Err(WavError::BadHeader));
    }

    #[test]
    fn ignores_bytes_beyond_the_riff_size() {
        let mut wav = good_wav();
        wav.extend_from_slice(b"junk");
        assert_eq!(parse(&wav, NO_CAP).unwrap(), SAMPLES);
    }

    #[test]
    fn parses_the_fixture_when_present() {
        // Same reference dump directory as the mel parity test.
        let Some(dir) = std::env::var_os("LLMMAN_CANARY_REF_DIR") else {
            println!("skip: LLMMAN_CANARY_REF_DIR unset");
            return;
        };
        let path = std::path::Path::new(&dir).join("fleurs-en.wav");
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(parse(&bytes, NO_CAP).unwrap().len(), 168_960);
    }
}
