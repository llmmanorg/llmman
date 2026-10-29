//! Log-mel feature extraction matching NeMo's
//! `AudioToMelSpectrogramPreprocessor` (features.py in NeMo v2.7.3) as
//! Canary and Parakeet models expect it.
//!
//! Building blocks live here in `f64`: the FFT, the symmetric Hann window
//! that `torch.hann_window(periodic=False)` produces, and the Slaney mel
//! filterbank that `librosa.filters.mel(htk=False, norm="slaney")`
//! produces. No crate dependency, safe Rust only.

use std::f64::consts::PI;

/// How the log-mel matrix is normalized before it reaches the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalize {
    /// Leave the log-mel values as they are.
    None,
    /// Per mel bin: subtract the mean over the frames and divide by the
    /// unbiased standard deviation plus `1e-5` (NeMo `per_feature`).
    PerFeature,
}

/// Front-end parameters. The defaults are NeMo's Canary/Parakeet recipe;
/// a GGUF loader can fill them from `stt.frontend.*` metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct MelConfig {
    /// Number of mel bins per frame.
    pub n_mels: usize,
    /// Input sample rate in Hz.
    pub sample_rate: u32,
    /// FFT length per frame; must be a power of two.
    pub n_fft: usize,
    /// Hann window length in samples, centered inside the FFT frame.
    pub win_length: usize,
    /// Step between consecutive frames in samples.
    pub hop: usize,
    /// Pre-emphasis coefficient: `y[n] = x[n] - preemph * x[n-1]`.
    pub preemph: f64,
    /// Lowest filterbank edge in Hz.
    pub f_min: f64,
    /// Highest filterbank edge in Hz.
    pub f_max: f64,
    /// Added to the mel energy before the natural log.
    pub log_zero_guard: f64,
    /// Normalization applied after the log.
    pub normalize: Normalize,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            n_mels: 128,
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop: 160,
            preemph: 0.97,
            f_min: 0.0,
            f_max: 8000.0,
            log_zero_guard: 2f64.powi(-24),
            normalize: Normalize::PerFeature,
        }
    }
}

/// One complex sample of the FFT buffer.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Complex {
    re: f64,
    im: f64,
}

/// In-place iterative radix-2 forward DFT, `X[k] = sum x[n] e^{-2 pi i kn/N}`
/// (no scaling). `buf.len()` must be a power of two.
fn fft(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(n.is_power_of_two(), "fft length {n} is not a power of two");
    if n < 2 {
        return;
    }
    let shift = n.leading_zeros() + 1;
    for i in 0..n {
        let j = i.reverse_bits() >> shift;
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        for k in 0..half {
            // Direct evaluation per twiddle keeps the error independent of
            // the stage count, unlike accumulating w^k by multiplication.
            let (sin, cos) = (-2.0 * PI * k as f64 / len as f64).sin_cos();
            let w = Complex { re: cos, im: sin };
            for start in (0..n).step_by(len) {
                let a = buf[start + k];
                let b = buf[start + k + half];
                let t = Complex {
                    re: b.re * w.re - b.im * w.im,
                    im: b.re * w.im + b.im * w.re,
                };
                buf[start + k] = Complex {
                    re: a.re + t.re,
                    im: a.im + t.im,
                };
                buf[start + k + half] = Complex {
                    re: a.re - t.re,
                    im: a.im - t.im,
                };
            }
        }
        len *= 2;
    }
}

/// Symmetric Hann window `0.5 - 0.5 cos(2 pi i / (n - 1))`, the shape
/// `torch.hann_window(n, periodic=False)` returns. A length of 1 is `[1.0]`.
fn hann_symmetric(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}

/// Slaney mel scale: 200/3 Hz per mel up to 1000 Hz, then logarithmic
/// with 27 mels per factor 6.4 in frequency (`librosa.hz_to_mel`,
/// `htk=False`).
const MEL_F_SP: f64 = 200.0 / 3.0;
const MEL_MIN_LOG_HZ: f64 = 1000.0;
const MEL_MIN_LOG_MEL: f64 = MEL_MIN_LOG_HZ / MEL_F_SP;

fn mel_log_step() -> f64 {
    6.4f64.ln() / 27.0
}

fn hz_to_mel(hz: f64) -> f64 {
    if hz >= MEL_MIN_LOG_HZ {
        MEL_MIN_LOG_MEL + (hz / MEL_MIN_LOG_HZ).ln() / mel_log_step()
    } else {
        hz / MEL_F_SP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    if mel >= MEL_MIN_LOG_MEL {
        MEL_MIN_LOG_HZ * (mel_log_step() * (mel - MEL_MIN_LOG_MEL)).exp()
    } else {
        MEL_F_SP * mel
    }
}

/// The `n_mels + 2` triangle edges in Hz, evenly spaced in mel between
/// `f_min` and `f_max` (`librosa.mel_frequencies`).
fn mel_edges(cfg: &MelConfig) -> Vec<f64> {
    let n = cfg.n_mels + 2;
    let lo = hz_to_mel(cfg.f_min);
    let hi = hz_to_mel(cfg.f_max);
    (0..n)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n - 1) as f64))
        .collect()
}

/// Mel filterbank of shape `[n_mels][n_fft / 2 + 1]` (row-major), the
/// matrix `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax, htk=False,
/// norm="slaney")` returns: a triangle per row between consecutive mel
/// edges, sampled at the FFT bin centers `k * sr / n_fft` and scaled by
/// `2 / (edge[i + 2] - edge[i])` so every triangle has unit area.
fn slaney_filterbank(cfg: &MelConfig) -> Vec<f64> {
    let n_bins = cfg.n_fft / 2 + 1;
    let edges = mel_edges(cfg);
    let mut weights = vec![0.0; cfg.n_mels * n_bins];
    for (i, row) in weights.chunks_exact_mut(n_bins).enumerate() {
        let (lo, mid, hi) = (edges[i], edges[i + 1], edges[i + 2]);
        let enorm = 2.0 / (hi - lo);
        for (k, w) in row.iter_mut().enumerate() {
            let hz = k as f64 * f64::from(cfg.sample_rate) / cfg.n_fft as f64;
            let lower = (hz - lo) / (mid - lo);
            let upper = (hi - hz) / (hi - mid);
            *w = lower.min(upper).max(0.0) * enorm;
        }
    }
    weights
}

/// NeMo's `CONSTANT`: added to the per-bin standard deviation so a
/// constant bin does not divide by zero.
const STD_FLOOR: f64 = 1e-5;

/// Largest `n_fft` `validate` accepts. With `MAX_FILTERBANK_WEIGHTS` it
/// bounds every allocation that the config sizes (window, scratch,
/// filterbank) when the config comes from model metadata.
const MAX_N_FFT: usize = 1 << 16;

/// Largest filterbank (`n_mels * (n_fft / 2 + 1)` weights, 128 MiB of
/// f64) `validate` accepts. `n_fft` and `n_mels` are each bounded alone,
/// but their product is the one allocation that grows quadratically.
const MAX_FILTERBANK_WEIGHTS: usize = 1 << 24;

/// Why `features` rejected its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelError {
    /// Per-feature normalization needs at least two frames (it divides by
    /// `frames - 1`), so the input must hold at least `2 * hop` samples.
    TooShort,
    /// A `MelConfig` field the pipeline cannot work with; the message
    /// names the constraint.
    InvalidConfig(&'static str),
}

impl std::fmt::Display for MelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooShort => f.write_str(
                "input yields fewer than two frames, which per-feature normalization needs",
            ),
            Self::InvalidConfig(what) => write!(f, "invalid mel config: {what}"),
        }
    }
}

impl std::error::Error for MelError {}

/// Rejects configs the pipeline would panic on or produce NaN for. The
/// config may come from model metadata, so it is checked, not trusted.
fn validate(cfg: &MelConfig) -> Result<(), MelError> {
    if cfg.n_mels == 0 {
        return Err(MelError::InvalidConfig("n_mels must be at least 1"));
    }
    if cfg.n_fft < 2 || cfg.n_fft > MAX_N_FFT || !cfg.n_fft.is_power_of_two() {
        return Err(MelError::InvalidConfig(
            "n_fft must be a power of two between 2 and 65536",
        ));
    }
    if cfg.n_mels > cfg.n_fft / 2 + 1 {
        return Err(MelError::InvalidConfig(
            "n_mels must not exceed n_fft / 2 + 1",
        ));
    }
    if cfg.n_mels.saturating_mul(cfg.n_fft / 2 + 1) > MAX_FILTERBANK_WEIGHTS {
        return Err(MelError::InvalidConfig(
            "n_mels * (n_fft / 2 + 1) must not exceed 2^24",
        ));
    }
    if cfg.win_length == 0 || cfg.win_length > cfg.n_fft {
        return Err(MelError::InvalidConfig(
            "win_length must be between 1 and n_fft",
        ));
    }
    if cfg.hop == 0 {
        return Err(MelError::InvalidConfig("hop must be at least 1"));
    }
    if cfg.sample_rate == 0 {
        return Err(MelError::InvalidConfig("sample_rate must be at least 1"));
    }
    let nyquist = f64::from(cfg.sample_rate) / 2.0;
    if !(cfg.f_min >= 0.0 && cfg.f_min < cfg.f_max && cfg.f_max <= nyquist) {
        return Err(MelError::InvalidConfig(
            "f_min and f_max must satisfy 0 <= f_min < f_max <= sample_rate / 2",
        ));
    }
    if !cfg.preemph.is_finite() {
        return Err(MelError::InvalidConfig("preemph must be finite"));
    }
    if !(cfg.log_zero_guard > 0.0 && cfg.log_zero_guard.is_finite()) {
        return Err(MelError::InvalidConfig(
            "log_zero_guard must be finite and positive",
        ));
    }
    Ok(())
}

/// Number of frames `features` produces for `len` samples: `len / hop`,
/// NeMo's `get_seq_len` (`(len + 2 * (n_fft / 2) - n_fft) / hop`, no
/// `+ 1`). The centered STFT yields one more column; NeMo zeroes it after
/// normalization and hands the encoder the wider tensor with `len / hop`
/// as its length, so the zero column only reaches encoder outputs past
/// the valid length, which NeMo masks. It is left out here and out of
/// the statistics. 168960 samples give 1056 frames with the default
/// config.
///
/// # Panics
/// When `cfg.hop` is 0; `features` rejects such a config first.
pub fn frame_count(len: usize, cfg: &MelConfig) -> usize {
    len / cfg.hop
}

/// Log-mel features of 16-bit PCM, frame-major: element `t * n_mels + m`
/// is mel bin `m` of frame `t`, for `frame_count(pcm.len(), cfg)` frames.
///
/// The steps mirror NeMo's `FilterbankFeatures.forward` in eval mode (no
/// dither) and run in `f64` until the final cast: samples scaled by
/// `1 / 32768`; pre-emphasis `y[0] = x[0]`, `y[n] = x[n] - preemph *
/// x[n - 1]` over the whole signal; `n_fft / 2` zeros on each side
/// (`torch.stft` with `center=True, pad_mode="constant"`); frame `t`
/// starts at padded sample `t * hop` and is multiplied by the symmetric
/// Hann window of `win_length` centered in the `n_fft` frame (the STFT's
/// last frame, beyond `len / hop`, is not computed; see `frame_count`);
/// power spectrum `|X|^2` over `n_fft / 2 + 1` bins; Slaney filterbank;
/// `ln(x + log_zero_guard)`; then, for `Normalize::PerFeature`, per mel
/// bin `(x - mean) / (std + 1e-5)` with the mean and unbiased std
/// (divisor `frames - 1`) over all frames.
///
/// Allocations: the output (`frames * n_mels` f32), the padded
/// pre-emphasized signal (`len + n_fft` f64), one complex FFT scratch of
/// `n_fft` reused across frames, the window (`win_length` f64), the
/// filterbank (`n_mels * (n_fft / 2 + 1)` f64), one log-mel row
/// (`n_mels` f64) and the per-bin mean and variance accumulators
/// (`2 * n_mels` f64). There is no f64 copy of the matrix: with
/// per-feature normalization the frames are computed twice, once to
/// accumulate the statistics (Welford) and once to write the normalized
/// values. The second pass repeats the same operations, so it reproduces
/// the first pass bit for bit.
///
/// Single-threaded and deterministic: the same input and config give the
/// same bits on every run.
///
/// # Errors
/// `InvalidConfig` when a `cfg` field is out of range (see `validate`)
/// and `TooShort` when per-feature normalization gets fewer than two
/// frames, that is fewer than `2 * hop` samples.
pub fn features(pcm: &[i16], cfg: &MelConfig) -> Result<Vec<f32>, MelError> {
    validate(cfg)?;
    let frames = frame_count(pcm.len(), cfg);
    if cfg.normalize == Normalize::PerFeature && frames < 2 {
        return Err(MelError::TooShort);
    }
    let pad = cfg.n_fft / 2;

    // Pre-emphasis first, then constant zero padding; `prev` starts at 0
    // so y[0] = x[0] exactly.
    let mut signal = vec![0.0f64; pcm.len() + 2 * pad];
    let mut prev = 0.0;
    for (dst, &s) in signal[pad..].iter_mut().zip(pcm) {
        let x = f64::from(s) / 32768.0;
        *dst = x - cfg.preemph * prev;
        prev = x;
    }

    let mut framer = Framer {
        cfg,
        signal: &signal,
        window: hann_symmetric(cfg.win_length),
        fb: slaney_filterbank(cfg),
        scratch: vec![Complex::default(); cfg.n_fft],
        row: vec![0.0; cfg.n_mels],
    };
    let mut out = vec![0.0f32; frames * cfg.n_mels];

    if cfg.normalize == Normalize::None {
        for (t, dst) in out.chunks_exact_mut(cfg.n_mels).enumerate() {
            for (d, &v) in dst.iter_mut().zip(framer.log_mel(t)) {
                *d = v as f32;
            }
        }
        return Ok(out);
    }

    // Welford's update keeps a constant bin at exactly zero variance.
    let mut mean = vec![0.0f64; cfg.n_mels];
    let mut m2 = vec![0.0f64; cfg.n_mels];
    for t in 0..frames {
        let k = (t + 1) as f64;
        for ((mu, acc), &v) in mean.iter_mut().zip(&mut m2).zip(framer.log_mel(t)) {
            let delta = v - *mu;
            *mu += delta / k;
            *acc += delta * (v - *mu);
        }
    }
    for acc in &mut m2 {
        *acc = (*acc / (frames - 1) as f64).sqrt() + STD_FLOOR;
    }
    let std = m2;
    for (t, dst) in out.chunks_exact_mut(cfg.n_mels).enumerate() {
        let row = framer.log_mel(t);
        for (((d, &v), &mu), &s) in dst.iter_mut().zip(row).zip(&mean).zip(&std) {
            *d = ((v - mu) / s) as f32;
        }
    }
    Ok(out)
}

/// Per-frame state of `features`: the padded signal plus the buffers
/// reused by every frame.
struct Framer<'a> {
    cfg: &'a MelConfig,
    signal: &'a [f64],
    window: Vec<f64>,
    fb: Vec<f64>,
    scratch: Vec<Complex>,
    row: Vec<f64>,
}

impl Framer<'_> {
    /// Log-mel row of frame `t`, valid until the next call.
    fn log_mel(&mut self, t: usize) -> &[f64] {
        let cfg = self.cfg;
        let n_bins = cfg.n_fft / 2 + 1;
        // torch.stft centers a short window in the frame, extra zero on
        // the right.
        let win_offset = (cfg.n_fft - cfg.win_length) / 2;
        let start = t * cfg.hop + win_offset;
        let frame = &self.signal[start..start + cfg.win_length];
        self.scratch.fill(Complex::default());
        for ((dst, &x), &w) in self.scratch[win_offset..]
            .iter_mut()
            .zip(frame)
            .zip(&self.window)
        {
            dst.re = x * w;
        }
        fft(&mut self.scratch);
        // Power spectrum in place: only the real part of the first
        // n_bins entries is read below.
        for c in &mut self.scratch[..n_bins] {
            c.re = c.re * c.re + c.im * c.im;
        }
        for (dst, weights) in self.row.iter_mut().zip(self.fb.chunks_exact(n_bins)) {
            let energy: f64 = weights
                .iter()
                .zip(&self.scratch)
                .map(|(w, c)| w * c.re)
                .sum();
            *dst = (energy + cfg.log_zero_guard).ln();
        }
        &self.row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bin spacing in Hz for the default config.
    const BIN_HZ: f64 = 16_000.0 / 512.0;

    /// Deterministic values in [-1, 1) from a 64-bit LCG (Knuth MMIX
    /// constants), so the test needs no rand crate.
    fn pseudo_random(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    }

    fn naive_dft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut acc = Complex::default();
                for (i, v) in x.iter().enumerate() {
                    let (sin, cos) = (-2.0 * PI * (k * i) as f64 / n as f64).sin_cos();
                    acc.re += v.re * cos - v.im * sin;
                    acc.im += v.re * sin + v.im * cos;
                }
                acc
            })
            .collect()
    }

    #[test]
    fn fft_matches_naive_dft() {
        let re = pseudo_random(512, 1);
        let im = pseudo_random(512, 2);
        let input: Vec<Complex> = re
            .iter()
            .zip(&im)
            .map(|(&re, &im)| Complex { re, im })
            .collect();
        let expected = naive_dft(&input);
        let mut got = input;
        fft(&mut got);
        for (k, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g.re - e.re).abs() < 1e-9,
                "bin {k} re {} vs {}",
                g.re,
                e.re
            );
            assert!(
                (g.im - e.im).abs() < 1e-9,
                "bin {k} im {} vs {}",
                g.im,
                e.im
            );
        }
    }

    #[test]
    fn fft_of_cosine_peaks_in_its_bin() {
        let n = 512;
        let bin = 37;
        let mut buf: Vec<Complex> = (0..n)
            .map(|i| Complex {
                re: (2.0 * PI * (bin * i) as f64 / n as f64).cos(),
                im: 0.0,
            })
            .collect();
        fft(&mut buf);
        let mag: Vec<f64> = buf.iter().map(|c| c.re.hypot(c.im)).collect();
        assert!((mag[bin] - n as f64 / 2.0).abs() < 1e-9);
        assert!((mag[n - bin] - n as f64 / 2.0).abs() < 1e-9);
        for (k, m) in mag.iter().enumerate() {
            if k != bin && k != n - bin {
                assert!(*m < 1e-9, "leak {m} in bin {k}");
            }
        }
    }

    #[test]
    fn fft_of_length_one_is_identity() {
        let mut one = [Complex { re: 3.0, im: -1.0 }];
        fft(&mut one);
        assert_eq!(one[0], Complex { re: 3.0, im: -1.0 });
    }

    #[test]
    #[should_panic(expected = "not a power of two")]
    fn fft_rejects_non_power_of_two() {
        let mut buf = vec![Complex::default(); 400];
        fft(&mut buf);
    }

    #[test]
    fn hann_is_symmetric_with_zero_endpoints() {
        for n in [400usize, 401] {
            let w = hann_symmetric(n);
            assert_eq!(w.len(), n);
            assert_eq!(w[0], 0.0);
            assert!(w[n - 1].abs() < 1e-15);
            for i in 0..n {
                assert!((w[i] - w[n - 1 - i]).abs() < 1e-15, "n={n} i={i}");
            }
        }
        let odd = hann_symmetric(401);
        assert!((odd[200] - 1.0).abs() < 1e-15);
        assert_eq!(hann_symmetric(1), vec![1.0]);
        assert!(hann_symmetric(0).is_empty());
    }

    #[test]
    fn mel_scale_round_trips() {
        // 1000 / (200 / 3) is 15 minus one ulp in f64, as in librosa.
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-12);
        assert!((mel_to_hz(15.0) - 1000.0).abs() < 1e-9);
        assert!((mel_to_hz(hz_to_mel(6400.0)) - 6400.0).abs() < 1e-9);
        assert!((hz_to_mel(6400.0) - 42.0).abs() < 1e-12);
        for hz in [0.0, 123.4, 999.9, 1000.1, 3456.7, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 1e-9, "hz={hz}");
        }
    }

    #[test]
    fn mel_edges_are_linear_below_1khz() {
        let cfg = MelConfig::default();
        let edges = mel_edges(&cfg);
        assert_eq!(edges.len(), 130);
        assert_eq!(edges[0], 0.0);
        assert!((edges[129] - 8000.0).abs() < 1e-9);
        let mel_8k = 15.0 + 8f64.ln() * 27.0 / 6.4f64.ln();
        let hz_step = mel_8k / 129.0 * (200.0 / 3.0);
        for (i, pair) in edges.windows(2).enumerate() {
            if pair[1] < 1000.0 {
                let step = pair[1] - pair[0];
                assert!((step - hz_step).abs() < 1e-9, "edge {i}: {step}");
            }
        }
        // The log region starts after the linear one and grows faster.
        let mut last = 0.0;
        for pair in edges.windows(2) {
            let step = pair[1] - pair[0];
            assert!(step >= last - 1e-9);
            last = step;
        }
    }

    #[test]
    fn filterbank_rows_are_contiguous_triangles() {
        let cfg = MelConfig::default();
        let n_bins = cfg.n_fft / 2 + 1;
        let fb = slaney_filterbank(&cfg);
        assert_eq!(fb.len(), cfg.n_mels * n_bins);
        let edges = mel_edges(&cfg);
        for (i, row) in fb.chunks_exact(n_bins).enumerate() {
            let nonzero: Vec<usize> = (0..n_bins).filter(|&k| row[k] > 1e-12).collect();
            assert!(!nonzero.is_empty(), "row {i} is empty");
            let (first, last) = (nonzero[0], nonzero[nonzero.len() - 1]);
            assert_eq!(nonzero.len(), last - first + 1, "row {i} has a gap");
            let (lo, mid, hi) = (edges[i], edges[i + 1], edges[i + 2]);
            assert!(first as f64 * BIN_HZ > lo && (last as f64) * BIN_HZ < hi);
            let peak = (0..n_bins)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .unwrap();
            assert!(
                (peak as f64 * BIN_HZ - mid).abs() <= BIN_HZ,
                "row {i} peaks in bin {peak}, center {mid} Hz"
            );
            assert!(row[peak] <= 2.0 / (hi - lo) + 1e-12);
        }
    }

    #[test]
    fn filterbank_peak_is_slaney_scaled_at_exact_bin_hit() {
        // 31 mels between 0 and 1000 Hz put every edge on a bin center
        // (32 intervals of 31.25 Hz), so each row peaks at exactly
        // 2 / (edge[i + 2] - edge[i]) = 2 / 62.5.
        let cfg = MelConfig {
            n_mels: 31,
            f_max: 1000.0,
            ..MelConfig::default()
        };
        let n_bins = cfg.n_fft / 2 + 1;
        let fb = slaney_filterbank(&cfg);
        let edges = mel_edges(&cfg);
        for (i, row) in fb.chunks_exact(n_bins).enumerate() {
            let enorm = 2.0 / (edges[i + 2] - edges[i]);
            assert!((enorm - 2.0 / 62.5).abs() < 1e-12);
            assert!(
                (row[i + 1] - enorm).abs() < 1e-12,
                "row {i}: {}",
                row[i + 1]
            );
            // The edge bins sit on the triangle feet; rounding in the
            // mel round trip may leave a weight of a few ulps there.
            assert!(row[i].abs() < 1e-12);
            assert!(row[i + 2].abs() < 1e-12);
        }
    }

    /// Slaney scaling gives every triangle unit area, so the bin sum times
    /// the bin spacing approximates 1. The Riemann sum of a unimodal
    /// function is off by at most one sample of its peak, `2 / base`
    /// here, which bounds every row; rows at least four bins wide land
    /// within 5% (the widest miss, 7%, is row 73 at 2073 Hz with a
    /// 102 Hz base).
    #[test]
    fn filterbank_rows_have_unit_area() {
        let cfg = MelConfig::default();
        let n_bins = cfg.n_fft / 2 + 1;
        let fb = slaney_filterbank(&cfg);
        let edges = mel_edges(&cfg);
        let mut wide = 0;
        for (i, row) in fb.chunks_exact(n_bins).enumerate() {
            let base = edges[i + 2] - edges[i];
            let error = (row.iter().sum::<f64>() * BIN_HZ - 1.0).abs();
            assert!(error <= 2.0 * BIN_HZ / base, "row {i} error {error}");
            if base >= 4.0 * BIN_HZ {
                assert!(error < 0.05, "row {i} error {error}");
                wide += 1;
            }
        }
        assert!(wide > 40, "only {wide} rows span four bins");
    }

    /// One second of a 1 kHz sine at 16 kHz, amplitude 8000.
    fn sine_1khz() -> Vec<i16> {
        (0..16_000)
            .map(|i| (8000.0 * (2.0 * PI * 1000.0 * i as f64 / 16_000.0).sin()).round() as i16)
            .collect()
    }

    /// Per-bin mean and unbiased std of a frame-major matrix.
    fn column_stats(m: &[f32], n_mels: usize) -> (Vec<f64>, Vec<f64>) {
        let frames = m.len() / n_mels;
        let mut mean = vec![0.0; n_mels];
        for row in m.chunks_exact(n_mels) {
            for (acc, &v) in mean.iter_mut().zip(row) {
                *acc += f64::from(v) / frames as f64;
            }
        }
        let mut std = vec![0.0; n_mels];
        for row in m.chunks_exact(n_mels) {
            for ((acc, &mu), &v) in std.iter_mut().zip(&mean).zip(row) {
                *acc += (f64::from(v) - mu).powi(2) / (frames - 1) as f64;
            }
        }
        for s in &mut std {
            *s = s.sqrt();
        }
        (mean, std)
    }

    #[test]
    fn frame_count_matches_nemo_get_seq_len() {
        let cfg = MelConfig::default();
        assert_eq!(frame_count(168_960, &cfg), 1056);
        assert_eq!(frame_count(0, &cfg), 0);
        assert_eq!(frame_count(159, &cfg), 0);
        assert_eq!(frame_count(160, &cfg), 1);
        assert_eq!(frame_count(320, &cfg), 2);
    }

    #[test]
    fn one_sample_is_too_short_for_per_feature() {
        let cfg = MelConfig::default();
        assert_eq!(features(&[1], &cfg), Err(MelError::TooShort));
        assert_eq!(features(&[1; 319], &cfg), Err(MelError::TooShort));
        assert_eq!(features(&[1; 320], &cfg).unwrap().len(), 2 * 128);
        // Without normalization a single frame is fine, and no sample at
        // all gives no frame.
        assert!(features(
            &[],
            &MelConfig {
                normalize: Normalize::None,
                ..cfg.clone()
            },
        )
        .unwrap()
        .is_empty());
        let raw = features(
            &[1; 160],
            &MelConfig {
                normalize: Normalize::None,
                ..cfg
            },
        )
        .unwrap();
        assert_eq!(raw.len(), 128);
    }

    #[test]
    fn invalid_config_is_rejected() {
        let bad = |cfg: MelConfig| features(&[0; 320], &cfg).unwrap_err();
        let base = MelConfig::default();
        for (cfg, what) in [
            (
                MelConfig {
                    n_fft: 400,
                    ..base.clone()
                },
                "n_fft",
            ),
            (
                MelConfig {
                    win_length: 513,
                    ..base.clone()
                },
                "win_length",
            ),
            (
                MelConfig {
                    hop: 0,
                    ..base.clone()
                },
                "hop",
            ),
            (
                MelConfig {
                    n_mels: 0,
                    ..base.clone()
                },
                "n_mels",
            ),
            (
                MelConfig {
                    f_max: 9000.0,
                    ..base.clone()
                },
                "f_max",
            ),
            (
                MelConfig {
                    log_zero_guard: 0.0,
                    ..base.clone()
                },
                "log_zero_guard",
            ),
            (
                MelConfig {
                    n_fft: 1 << 17,
                    ..base.clone()
                },
                "n_fft",
            ),
            (
                MelConfig {
                    n_mels: 258,
                    ..base.clone()
                },
                "n_mels",
            ),
            // 8 GiB of filterbank if it were allocated.
            (
                MelConfig {
                    n_fft: MAX_N_FFT,
                    n_mels: MAX_N_FFT / 2 + 1,
                    ..base.clone()
                },
                "n_mels",
            ),
        ] {
            match bad(cfg) {
                MelError::InvalidConfig(msg) => assert!(msg.contains(what), "{msg}"),
                other => panic!("expected InvalidConfig for {what}, got {other:?}"),
            }
        }
    }

    #[test]
    fn all_zero_input_normalizes_to_all_zero() {
        let cfg = MelConfig::default();
        let pcm = vec![0i16; 16_000];
        let out = features(&pcm, &cfg).unwrap();
        assert_eq!(out.len(), 100 * 128);
        assert!(out.iter().all(|&v| v == 0.0), "{:?}", &out[..4]);
        // Before normalization every value is the log floor.
        let raw = features(
            &pcm,
            &MelConfig {
                normalize: Normalize::None,
                ..cfg
            },
        )
        .unwrap();
        let floor = 2f64.powi(-24).ln() as f32;
        assert!(raw.iter().all(|&v| v == floor), "{:?}", &raw[..4]);
    }

    /// Pins frame direction, hop alignment and window centering: frame
    /// `t` is centered on sample `t * hop`, so an impulse at sample 1000
    /// (pre-emphasis adds a second tap at 1001) lands 40 samples from the
    /// center of frame 6 and 120 from frame 7, and outside the 400-sample
    /// Hann of every other frame. A reversed frame loop moves the energy
    /// to frames 92 and 93; dropping `win_offset` from the frame start
    /// shifts the centers by 56 samples and makes frame 7 louder than 6.
    #[test]
    fn impulse_lands_in_the_frames_centered_on_it() {
        let cfg = MelConfig {
            normalize: Normalize::None,
            ..MelConfig::default()
        };
        let mut pcm = vec![0i16; 16_000];
        pcm[1000] = i16::MAX;
        let raw = features(&pcm, &cfg).unwrap();
        let floor = cfg.log_zero_guard.ln() as f32;
        let energy = |t: usize| -> f64 {
            raw[t * cfg.n_mels..(t + 1) * cfg.n_mels]
                .iter()
                .map(|&v| f64::from(v) - f64::from(floor))
                .sum()
        };
        for t in 0..frame_count(pcm.len(), &cfg) {
            if t == 6 || t == 7 {
                continue;
            }
            assert_eq!(energy(t), 0.0, "frame {t} is not at the floor");
        }
        assert!(energy(7) > 0.0, "frame 7 is at the floor");
        assert!(
            energy(6) > energy(7),
            "frame 6 ({}) is not louder than frame 7 ({})",
            energy(6),
            energy(7)
        );
    }

    #[test]
    fn sine_1khz_is_finite_and_normalized() {
        let cfg = MelConfig::default();
        let pcm = sine_1khz();
        let frames = frame_count(pcm.len(), &cfg);
        assert_eq!(frames, 100);
        let out = features(&pcm, &cfg).unwrap();
        assert_eq!(out.len(), frames * cfg.n_mels);
        assert!(out.iter().all(|v| v.is_finite()));

        let raw = features(
            &pcm,
            &MelConfig {
                normalize: Normalize::None,
                ..cfg.clone()
            },
        )
        .unwrap();
        // The loudest bin (by mean log energy) is the triangle centered
        // nearest 1 kHz.
        let (raw_mean, raw_std) = column_stats(&raw, cfg.n_mels);
        let edges = mel_edges(&cfg);
        let expected = (0..cfg.n_mels)
            .min_by(|&a, &b| {
                (edges[a + 1] - 1000.0)
                    .abs()
                    .total_cmp(&(edges[b + 1] - 1000.0).abs())
            })
            .unwrap();
        let loudest = (0..cfg.n_mels)
            .max_by(|&a, &b| raw_mean[a].total_cmp(&raw_mean[b]))
            .unwrap();
        assert!(
            loudest.abs_diff(expected) <= 1,
            "loudest bin {loudest}, expected {expected}"
        );

        let (mean, std) = column_stats(&out, cfg.n_mels);
        let mut checked = 0;
        for m in 0..cfg.n_mels {
            assert!(mean[m].abs() < 1e-4, "bin {m} mean {}", mean[m]);
            // std / (std + 1e-5) is 1 within 1% once the raw std is
            // above 1e-3; quieter bins are dominated by the floor.
            if raw_std[m] > 1e-3 {
                assert!((std[m] - 1.0).abs() < 1e-2, "bin {m} std {}", std[m]);
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} bins carry energy");
    }

    #[test]
    fn features_are_bit_identical_across_runs() {
        let cfg = MelConfig::default();
        let pcm = sine_1khz();
        let a = features(&pcm, &cfg).unwrap();
        let b = features(&pcm, &cfg).unwrap();
        assert_eq!(a, b);
    }

    /// Compares against NeMo's preprocessor output for the fixture when
    /// `LLMMAN_CANARY_REF_DIR` points at a reference dump; skips when it
    /// is unset and fails when the dump is incomplete.
    /// The reference is `[n_mels, T]` (mel-major); T may exceed our frame
    /// count by right-padded zero columns, which NeMo writes after
    /// normalization (see `frame_count`).
    #[test]
    fn parity_matches_nemo_reference() {
        let Some(dir) = std::env::var_os("LLMMAN_CANARY_REF_DIR") else {
            println!("skip: LLMMAN_CANARY_REF_DIR is not set");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let sidecar = dir.join("enc.mel.in.json");
        assert!(sidecar.is_file(), "{} is missing", sidecar.display());

        let cfg = MelConfig::default();
        let wav = std::fs::read(dir.join("fleurs-en.wav")).unwrap();
        let pcm = crate::speech::wav::parse(&wav, 64 << 20).unwrap();
        let ours = features(&pcm, &cfg).unwrap();
        let frames = ours.len() / cfg.n_mels;
        assert_eq!(frames, 1056, "fixture frame count");

        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&sidecar).unwrap()).unwrap();
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .expect("shape array")
            .iter()
            .map(|v| usize::try_from(v.as_u64().expect("shape entry")).unwrap())
            .collect();
        assert_eq!(shape.len(), 2, "shape {shape:?}");
        let (n_mels, t_ref) = (shape[0], shape[1]);
        assert_eq!(n_mels, cfg.n_mels, "shape {shape:?}");
        assert!(
            t_ref >= frames,
            "reference has {t_ref} frames, we have {frames}"
        );

        let raw = std::fs::read(dir.join("enc.mel.in.f32")).unwrap();
        assert_eq!(raw.len(), n_mels * t_ref * 4, "enc.mel.in.f32 length");
        let reference: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();

        // Per-feature normalization divides each bin by its raw log std
        // plus 1e-5, so a bin at the log floor (std 0.019 for mel 127 on
        // this fixture) multiplies a one-ulp f32 difference of the
        // reference's log (1.9e-6 at |ln 2^-24| = 16.6) by about 50.
        // The max bound is therefore taken in the raw log domain, each
        // diff scaled back by its bin's divisor: 1e-5 is about five f32
        // ulps at the floor and stays under the 4e-5 table seed for
        // every bin whose raw std is at least 0.25. p99 stays in the
        // normalized domain at the seed. Measured 2026-10-29 against the
        // NeMo 2.7.3 dump: raw-domain max 5.98e-6 (mel 44, std 1.35),
        // normalized max 1.29e-4 (mel 127), p99 6.7e-6. The normalized
        // max keeps a ceiling of 2e-4 (1.5x the measured value, the
        // tolerance table's rule) so a gross regression fails even when
        // p99 and the raw-domain max do not catch it.
        let raw = features(
            &pcm,
            &MelConfig {
                normalize: Normalize::None,
                ..cfg.clone()
            },
        )
        .unwrap();
        let (_, raw_std) = column_stats(&raw, n_mels);

        let mut diffs = Vec::with_capacity(frames * n_mels);
        let mut worst = (0.0f64, 0usize, 0usize);
        let mut worst_raw = (0.0f64, 0usize, 0usize);
        for m in 0..n_mels {
            let row = &reference[m * t_ref..(m + 1) * t_ref];
            let scale = raw_std[m] + STD_FLOOR;
            for (t, &r) in row[..frames].iter().enumerate() {
                let d = (f64::from(ours[t * n_mels + m]) - f64::from(r)).abs();
                assert!(d.is_finite(), "non-finite diff at frame {t} mel {m}");
                if d > worst.0 {
                    worst = (d, t, m);
                }
                if d * scale > worst_raw.0 {
                    worst_raw = (d * scale, t, m);
                }
                diffs.push(d);
            }
            for (t, &r) in row[frames..].iter().enumerate() {
                assert_eq!(r, 0.0, "padded frame {} mel {m} is not zero", frames + t);
            }
        }
        diffs.sort_by(f64::total_cmp);
        let max = diffs[diffs.len() - 1];
        let p99 = diffs[(diffs.len() - 1) * 99 / 100];
        println!(
            "mel parity: max|diff| = {max:.3e} at frame {} mel {} (raw std {:.3e}), \
             p99|diff| = {p99:.3e}, raw-domain max = {:.3e} at frame {} mel {} (raw std {:.3e})",
            worst.1,
            worst.2,
            raw_std[worst.2],
            worst_raw.0,
            worst_raw.1,
            worst_raw.2,
            raw_std[worst_raw.2],
        );
        assert!(max <= 2e-4, "max|diff| {max:.3e} exceeds 2e-4");
        assert!(
            p99 <= 4e-5,
            "p99|diff| {p99:.3e} exceeds 4e-5 (max {max:.3e})"
        );
        assert!(
            worst_raw.0 <= 1e-5,
            "raw-domain max|diff| {:.3e} at frame {} mel {} exceeds 1e-5",
            worst_raw.0,
            worst_raw.1,
            worst_raw.2
        );
    }
}
