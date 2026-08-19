//! NVIDIA Parakeet backend (ONNX models) via `parakeet-rs` / ONNX Runtime.
//!
//! Uses the **TDT** model (`ParakeetTDT`), multilingual with language
//! auto-detection and predicted punctuation. The model is a **directory**
//! containing `encoder-model.onnx`, `decoder_joint-model.onnx` and `vocab.txt`.
//!
//! Unlike whisper.cpp, Parakeet supports neither a decoding prompt
//! (`initial_prompt`), nor beam search, nor translation, nor an internal VAD:
//! [`AsrEngine::capabilities`] says so, and the corresponding
//! [`TranscribeOptions`] are ignored.

use parakeet_rs::{ExecutionConfig, ExecutionProvider, ParakeetTDT, TimestampMode, Transcriber};

use super::AsrEngine;
use crate::error::AsrError;
use crate::types::{
    DevicePreference, EngineCapabilities, EngineConfig, LanguageSupport, Segment,
    TranscribeControl, TranscribeOptions, TranscriptionResult,
};

/// Parakeet TDT backend. See [`AsrEngine`].
pub struct ParakeetEngine {
    model: ParakeetTDT,
    caps: EngineCapabilities,
}

impl ParakeetEngine {
    /// Loads a Parakeet TDT model from the **directory** [`EngineConfig::model_path`].
    ///
    /// [`EngineConfig::device`] picks the ONNX Runtime execution provider:
    /// `Cpu` forces CPU; `Auto`/`Gpu` request DirectML when the crate is built
    /// with `parakeet-directml`, otherwise fall back to CPU. GPU providers fall
    /// back to CPU anyway when the hardware/driver is missing (`ort`'s
    /// guarantee). [`EngineConfig::crash_marker`] is ignored: `ort` returns
    /// `Result`s, it does not `abort()` the process.
    pub fn load(config: &EngineConfig) -> Result<Self, AsrError> {
        let dir = config.model_path.as_path();
        if !dir.is_dir() {
            return Err(AsrError::ModelNotFound(format!(
                "parakeet model directory not found: {}",
                dir.display()
            )));
        }
        // Windows MAX_PATH (~260): parakeet-rs opens `encoder-model.onnx` and
        // `decoder_joint-model.onnx` inside this directory, so the directory path
        // plus the longest leaf (~24 chars) must stay well under the limit or
        // ONNX Runtime fails to open the file with an opaque error. Fail early
        // with a clear message, mirroring the whisper backend's guard. Only
        // checked when the path is UTF-8 measurable; a non-UTF-8 path is left to
        // the loader (parakeet-rs takes a `Path`, not a `&str`).
        if cfg!(windows)
            && let Some(p) = dir.to_str()
            && p.len() > 230
        {
            return Err(AsrError::Load(format!(
                "parakeet model directory path too long ({} chars, Windows limit ~260): {}",
                p.len(),
                dir.display()
            )));
        }

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 8);

        let exec = ExecutionConfig::new()
            .with_execution_provider(select_provider(config.device))
            .with_intra_threads(n_threads as usize);

        let model = ParakeetTDT::from_pretrained(dir, Some(exec))
            .map_err(|e| AsrError::Load(format!("parakeet load: {e}")))?;

        Ok(ParakeetEngine {
            model,
            caps: ParakeetEngine::capabilities_static(),
        })
    }

    fn capabilities_static() -> EngineCapabilities {
        EngineCapabilities {
            name: "parakeet-tdt",
            supports_prompt: false,
            supports_beam: false,
            supports_translate: false,
            supports_internal_vad: false,
            languages: LanguageSupport::Set(tdt_v3_languages()),
        }
    }
}

impl AsrEngine for ParakeetEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        _opts: &TranscribeOptions,
        control: &TranscribeControl,
    ) -> Result<TranscriptionResult, AsrError> {
        if audio.is_empty() {
            return Ok(TranscriptionResult::default());
        }
        // Enforce the audio contract before feature extraction: a NaN/Inf sample
        // poisons Parakeet's mel spectrogram into garbage text. Borrows through
        // untouched when already clean.
        let audio = super::sanitize_audio(audio);
        let audio = audio.as_ref();

        // Parakeet runs the encoder over the WHOLE sequence at once (no internal
        // windowing, unlike whisper.cpp), and ONNX Runtime's arena never returns
        // its peak to the OS: on a long clip the RAM balloons without bound and
        // stays high. So we split into windows <= 32 s, cutting on the least-bad
        // energy dip, and free each window between passes so the arena settles on
        // a single window's peak. Every sample is transcribed exactly once — no
        // overlap — so no word is duplicated at a seam (same RMS-cut rationale as
        // the streaming path; an overlap+dedup approach is unreliable without the
        // token context Parakeet does not carry between calls).
        let mut text = String::new();
        let mut segments: Vec<Segment> = Vec::new();
        let total = audio.len();
        let mut start = 0;
        while start < audio.len() {
            // Cooperative cancellation: checked at each window boundary, i.e.
            // before the expensive encoder pass. Granularity is one window
            // (~30 s), which is the finest this backend can offer without an
            // in-model hook. Report progress as the fraction of samples already
            // consumed (0 % on the first window, 100 % once the loop ends).
            if control.is_cancelled() {
                return Err(AsrError::Cancelled);
            }
            control.report_progress((start as u64 * 100 / total as u64) as u8);
            let rest = &audio[start..];
            let cut = find_min_rms_cut(rest);
            // `transcribe_samples` takes an owned `Vec<f32>`; we hand it a copy of
            // the window, freed at the end of the iteration. Mono 16 kHz is our
            // input contract. `Sentences` groups tokens into sentence-level spans.
            let out = self
                .model
                .transcribe_samples(
                    rest[..cut].to_vec(),
                    crate::types::SAMPLE_RATE,
                    1,
                    Some(TimestampMode::Sentences),
                )
                .map_err(|e| AsrError::Transcribe(format!("{e}")))?;
            let piece = out.text.trim();
            if !piece.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(piece);
            }
            // Token times are relative to the window; shift them onto the full
            // input's timeline by the window's start offset.
            let offset = start as f32 / crate::types::SAMPLE_RATE as f32;
            segments.extend(
                out.tokens
                    .into_iter()
                    .filter(|t| !t.text.trim().is_empty())
                    .map(|t| Segment {
                        text: t.text.trim().to_string(),
                        start: t.start + offset,
                        end: t.end + offset,
                    }),
            );
            start += cut;
        }
        control.report_progress(100);

        // The model auto-detects the language but does not expose it in the
        // result -> `detected_language = None`.
        Ok(TranscriptionResult {
            text,
            detected_language: None,
            segments,
        })
    }

    fn capabilities(&self) -> &EngineCapabilities {
        &self.caps
    }
}

/// Maps a [`DevicePreference`] to an ONNX Runtime execution provider.
///
/// `DirectML` is only nameable when the crate is built with the
/// `parakeet-directml` feature (the `parakeet-rs` enum variant is itself gated);
/// otherwise we stay on CPU.
fn select_provider(device: DevicePreference) -> ExecutionProvider {
    match device {
        DevicePreference::Cpu => ExecutionProvider::Cpu,
        DevicePreference::Auto | DevicePreference::Gpu => {
            #[cfg(feature = "parakeet-directml")]
            {
                ExecutionProvider::DirectML
            }
            #[cfg(not(feature = "parakeet-directml"))]
            {
                ExecutionProvider::Cpu
            }
        }
    }
}

/// Languages of the `nvidia/parakeet-tdt-0.6b-v3` model (25 European languages:
/// the official EU languages minus Irish, plus Russian and Ukrainian), as
/// lowercase English names to stay consistent with whisper's language detection.
///
/// Source: NVIDIA parakeet-tdt-0.6b-v3 model card (reconfirmed). It is a
/// capability metadatum, not a guarantee extracted programmatically from the
/// loaded model.
fn tdt_v3_languages() -> Vec<String> {
    [
        "bulgarian", "croatian", "czech", "danish", "dutch", "english", "estonian",
        "finnish", "french", "german", "greek", "hungarian", "italian", "latvian",
        "lithuanian", "maltese", "polish", "portuguese", "romanian", "russian",
        "slovak", "slovenian", "spanish", "swedish", "ukrainian",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Hard cap on a transcription window's length, in samples (30 s). No window
/// ever exceeds this, which is what bounds the ONNX Runtime arena.
const WINDOW_SAMPLES: usize = 30 * crate::types::SAMPLE_RATE as usize;
/// A clip up to this length (32 s) is transcribed in a single window. Set above
/// `WINDOW_SAMPLES` on purpose: whenever we DO split, the remainder after a cut
/// in the [25 s, 30 s] band is then always >= 2 s. A very short trailing window
/// (a few hundred ms) makes the model hallucinate; this removes that case by
/// construction, at the cost of one window occasionally reaching 32 s.
const SINGLE_WINDOW_MAX_SAMPLES: usize = 32 * crate::types::SAMPLE_RATE as usize;
/// Start of the cut-point search band (25 s). We only look for an energy dip
/// between 25 and 30 s: past 25 s of speech we want to cut, but at the best spot
/// in that band.
const SEARCH_START_SAMPLES: usize = 25 * crate::types::SAMPLE_RATE as usize;
/// RMS analysis block size (100 ms).
const RMS_BLOCK: usize = crate::types::SAMPLE_RATE as usize / 10;

/// Cut position (in samples) of the next window within `rest`.
///
/// Returns `rest.len()` when everything fits in a single window (last pass, or a
/// short clip: identical behaviour to a single call). Otherwise it searches the
/// [25 s, 30 s] band for the 100 ms block of lowest energy (RMS) and cuts at its
/// start: we land on the least-bad breath rather than mid-word. If no dip stands
/// out (continuous speech), the hard cap at 30 s applies — the window is always
/// bounded. Always `>= 25 s` once `rest` exceeds `SINGLE_WINDOW_MAX_SAMPLES`, so
/// the calling loop always makes progress.
fn find_min_rms_cut(rest: &[f32]) -> usize {
    if rest.len() <= SINGLE_WINDOW_MAX_SAMPLES {
        return rest.len();
    }
    // Default hard cap: kept if the band yields no block (impossible here — the
    // band is 5 s = 50 blocks — but keeps the window bounded whatever happens).
    let mut best_start = WINDOW_SAMPLES;
    let mut best_rms = f32::INFINITY;
    let mut pos = SEARCH_START_SAMPLES;
    while pos + RMS_BLOCK <= WINDOW_SAMPLES {
        let block = &rest[pos..pos + RMS_BLOCK];
        let rms = (block.iter().map(|s| s * s).sum::<f32>() / RMS_BLOCK as f32).sqrt();
        if rms < best_rms {
            best_rms = rms;
            best_start = pos;
        }
        pos += RMS_BLOCK;
    }
    best_start
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_declare_no_whisper_only_options() {
        let caps = ParakeetEngine::capabilities_static();
        assert_eq!(caps.name, "parakeet-tdt");
        assert!(!caps.supports_prompt);
        assert!(!caps.supports_beam);
        assert!(!caps.supports_translate);
        assert!(!caps.supports_internal_vad);
        match caps.languages {
            LanguageSupport::Set(langs) => {
                assert_eq!(langs.len(), 25, "parakeet-tdt-v3 = 25 langues");
                assert!(langs.iter().any(|l| l == "french"));
            }
            LanguageSupport::Any => panic!("Parakeet has a closed language set"),
        }
    }

    #[test]
    fn load_reports_missing_model_directory() {
        let cfg = EngineConfig {
            model_path: "::/no/such/parakeet/dir".into(),
            device: DevicePreference::Cpu,
            default_language: None,
            crash_marker: None,
        };
        assert!(matches!(
            ParakeetEngine::load(&cfg),
            Err(AsrError::ModelNotFound(_))
        ));
    }

    const RATE: usize = crate::types::SAMPLE_RATE as usize;

    /// `secs` seconds of a signal at amplitude `amp` (0.0 = silence).
    fn tone(secs: f32, amp: f32) -> Vec<f32> {
        let n = (RATE as f32 * secs) as usize;
        (0..n).map(|i| if i % 2 == 0 { amp } else { -amp }).collect()
    }

    #[test]
    fn cut_takes_the_whole_clip_when_it_fits_in_one_window() {
        // <= 32 s: a single call, cut = full length (identical to the old path).
        assert_eq!(find_min_rms_cut(&tone(10.0, 0.3)), 10 * RATE);
        assert_eq!(find_min_rms_cut(&tone(30.0, 0.3)), 30 * RATE);
        assert_eq!(find_min_rms_cut(&tone(32.0, 0.3)), 32 * RATE);
    }

    #[test]
    fn cut_falls_on_the_quietest_block_in_the_search_band() {
        // 33 s of loud speech with a silence gap at 26.0 s: the cut must land in
        // that gap (the lowest-energy 100 ms block of the [25 s, 30 s] band).
        let mut audio = tone(26.0, 0.3);
        audio.extend(tone(0.2, 0.0)); // silence at 26.0 s
        audio.extend(tone(6.8, 0.3)); // total 33 s
        let cut = find_min_rms_cut(&audio);
        let cut_s = cut as f32 / RATE as f32;
        assert!(
            (26.0..=26.3).contains(&cut_s),
            "cut expected near 26 s (the gap), got {cut_s} s"
        );
    }

    #[test]
    fn cut_stays_within_the_search_band_on_continuous_speech() {
        // Continuous speech with no dip: the cut stays bounded to [25 s, 30 s],
        // never beyond (ONNX arena guaranteed flat).
        let cut = find_min_rms_cut(&tone(40.0, 0.3));
        assert!(
            (SEARCH_START_SAMPLES..=WINDOW_SAMPLES).contains(&cut),
            "cut {cut} outside band [{SEARCH_START_SAMPLES}, {WINDOW_SAMPLES}]"
        );
    }

    #[test]
    fn a_split_never_leaves_a_tiny_trailing_window() {
        // Whenever the clip is split, the remainder must be >= 2 s, so the last
        // window never degrades into a hallucination-prone sub-second clip. This
        // is what the 32 s single-window threshold buys.
        for secs in [30.1f32, 32.1, 33.0, 45.0, 60.0] {
            let audio = tone(secs, 0.2);
            let cut = find_min_rms_cut(&audio);
            if cut < audio.len() {
                let tail = audio.len() - cut;
                assert!(tail >= 2 * RATE, "tail {tail} samples < 2 s at {secs} s");
            }
        }
    }

    #[test]
    fn cut_always_makes_progress() {
        // Every cut is > 0, so the windowing loop always advances.
        for secs in [1.0f32, 15.0, 29.9, 30.0, 32.1, 61.0, 120.0] {
            assert!(find_min_rms_cut(&tone(secs, 0.2)) > 0, "null cut at {secs} s");
        }
    }
}
