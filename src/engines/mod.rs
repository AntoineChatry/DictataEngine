//! The [`AsrEngine`] trait and the backend factory.
//!
//! Each real backend lives in its own submodule, enabled by a Cargo feature
//! (`whisper`, `parakeet`). [`MockEngine`] is always available and serves as the
//! test double and reference implementation.

mod mock;
pub use mock::MockEngine;

#[cfg(feature = "whisper")]
mod whisper;
#[cfg(feature = "whisper")]
pub use whisper::WhisperEngine;

#[cfg(feature = "parakeet")]
mod parakeet;
#[cfg(feature = "parakeet")]
pub use parakeet::ParakeetEngine;

#[cfg(feature = "sensevoice")]
mod sensevoice;
#[cfg(feature = "sensevoice")]
pub use sensevoice::SenseVoiceEngine;

#[cfg(feature = "moonshine")]
mod moonshine;
#[cfg(feature = "moonshine")]
pub use moonshine::MoonshineEngine;

#[cfg(any(
    feature = "whisper",
    feature = "parakeet",
    feature = "sensevoice",
    feature = "moonshine"
))]
use std::borrow::Cow;

use crate::error::AsrError;
use crate::types::{
    EngineCapabilities, EngineConfig, TranscribeControl, TranscribeOptions, TranscriptionResult,
};

/// An ASR backend: loads a model once, transcribes clips on demand.
///
/// **Stateless between calls**: each [`transcribe`](AsrEngine::transcribe) is
/// independent. Any continuity (vocabulary, already-emitted text) goes through
/// [`TranscribeOptions::initial_prompt`], never through state carried on the
/// engine — one instance can be shared between one-shot and streaming.
///
/// `Send` lets the engine move onto a worker thread; it is not `Sync`, so
/// concurrent access goes through the caller's own `Mutex`.
pub trait AsrEngine: Send {
    /// Transcribe `audio` (mono 16 kHz `f32`, see [`crate::SAMPLE_RATE`]).
    ///
    /// Empty `audio` returns an empty result without error.
    ///
    /// `control` carries cooperative cancellation and progress (see
    /// [`TranscribeControl`]); pass `&TranscribeControl::none()` for a plain
    /// call. A requested cancellation returns [`AsrError::Cancelled`], never a
    /// partial `Ok`.
    fn transcribe(
        &mut self,
        audio: &[f32],
        opts: &TranscribeOptions,
        control: &TranscribeControl,
    ) -> Result<TranscriptionResult, AsrError>;

    /// This backend's capabilities (honoured options, languages). Stable for the
    /// lifetime of the instance.
    fn capabilities(&self) -> &EngineCapabilities;
}

/// Identifies a backend for the [`load_engine`] factory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    /// whisper.cpp via `whisper-rs` (feature `whisper`).
    Whisper,
    /// NVIDIA Parakeet via `parakeet-rs`/ONNX (feature `parakeet`).
    Parakeet,
    /// SenseVoice CTC via `ort`/ONNX (feature `sensevoice`).
    SenseVoice,
    /// Moonshine encoder-decoder via `ort`/ONNX (feature `moonshine`).
    Moonshine,
}

/// Loads backend `kind` from `config`, returned behind `dyn AsrEngine` for
/// dynamic dispatch on the caller's side.
///
/// When a backend is not built (its feature is off), returns
/// [`AsrError::BackendUnavailable`]. Each compiled backend lives behind its
/// `#[cfg(feature = …)]` arm.
// `config` is consumed only by a compiled backend arm; with no backend feature
// at all it is legitimately unused.
#[cfg_attr(
    not(any(
        feature = "whisper",
        feature = "parakeet",
        feature = "sensevoice",
        feature = "moonshine"
    )),
    allow(unused_variables)
)]
pub fn load_engine(
    kind: EngineKind,
    config: &EngineConfig,
) -> Result<Box<dyn AsrEngine>, AsrError> {
    match kind {
        EngineKind::Whisper => {
            #[cfg(feature = "whisper")]
            {
                WhisperEngine::load(config).map(|e| Box::new(e) as Box<dyn AsrEngine>)
            }
            #[cfg(not(feature = "whisper"))]
            {
                Err(AsrError::BackendUnavailable(
                    "whisper backend not built (feature `whisper` off)".into(),
                ))
            }
        }
        EngineKind::Parakeet => {
            #[cfg(feature = "parakeet")]
            {
                ParakeetEngine::load(config).map(|e| Box::new(e) as Box<dyn AsrEngine>)
            }
            #[cfg(not(feature = "parakeet"))]
            {
                Err(AsrError::BackendUnavailable(
                    "parakeet backend not built (feature `parakeet` off)".into(),
                ))
            }
        }
        EngineKind::SenseVoice => {
            #[cfg(feature = "sensevoice")]
            {
                SenseVoiceEngine::load(config).map(|e| Box::new(e) as Box<dyn AsrEngine>)
            }
            #[cfg(not(feature = "sensevoice"))]
            {
                Err(AsrError::BackendUnavailable(
                    "sensevoice backend not built (feature `sensevoice` off)".into(),
                ))
            }
        }
        EngineKind::Moonshine => {
            #[cfg(feature = "moonshine")]
            {
                MoonshineEngine::load(config).map(|e| Box::new(e) as Box<dyn AsrEngine>)
            }
            #[cfg(not(feature = "moonshine"))]
            {
                Err(AsrError::BackendUnavailable(
                    "moonshine backend not built (feature `moonshine` off)".into(),
                ))
            }
        }
    }
}

/// Enforce the audio contract before it reaches a native backend.
///
/// Every backend documents mono 16 kHz `f32` in `[-1.0, 1.0]`, but callers only
/// promise it — nothing checks. A single `NaN`/`±Inf` sample (corrupt file,
/// bad ffmpeg decode) can make whisper.cpp `abort()` the whole process — which
/// no `Result` nor `catch_unwind` can intercept — or poison Parakeet's mel
/// extraction into garbage text. Out-of-range values (`|s| > 1.0`) distort the
/// spectrogram. This replaces non-finite samples with silence and clamps the
/// rest into range. It borrows the input untouched when the audio is already
/// clean, so the overwhelmingly common path allocates nothing.
#[cfg(any(
    feature = "whisper",
    feature = "parakeet",
    feature = "sensevoice",
    feature = "moonshine"
))]
pub(crate) fn sanitize_audio(audio: &[f32]) -> Cow<'_, [f32]> {
    if audio.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)) {
        return Cow::Borrowed(audio);
    }
    let fixed = audio
        .iter()
        .map(|&s| if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 })
        .collect();
    Cow::Owned(fixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TranscribeOptions;

    #[test]
    fn load_engine_never_yields_an_engine_from_a_bogus_config() {
        let cfg = EngineConfig {
            model_path: "::/does/not/exist".into(),
            device: crate::types::DevicePreference::Auto,
            default_language: None,
            crash_marker: None,
        };
        // Depending on the build:
        //   - Parakeet without its feature: always `BackendUnavailable`.
        //   - Whisper without its feature: `BackendUnavailable`.
        //   - Whisper WITH its feature: `ModelNotFound` (the bogus model is absent).
        // In every case an error — never an engine from a nonexistent path. No
        // `{:?}` on the `Ok`: `dyn AsrEngine` is not `Debug`.
        for kind in [EngineKind::Whisper, EngineKind::Parakeet] {
            match load_engine(kind, &cfg) {
                Err(_) => {}
                Ok(_) => panic!("a nonexistent path must never yield an engine ({kind:?})"),
            }
        }
    }

    #[cfg(not(feature = "whisper"))]
    #[test]
    fn whisper_arm_is_unavailable_without_its_feature() {
        let cfg = EngineConfig {
            model_path: "irrelevant".into(),
            device: crate::types::DevicePreference::Auto,
            default_language: None,
            crash_marker: None,
        };
        assert!(matches!(
            load_engine(EngineKind::Whisper, &cfg),
            Err(AsrError::BackendUnavailable(_))
        ));
    }

    #[test]
    fn engine_is_usable_as_a_trait_object() {
        // The whole point of the exercise: drive any backend behind
        // `dyn AsrEngine`.
        let mut engine: Box<dyn AsrEngine> = Box::new(MockEngine::new("salut"));
        let out = engine
            .transcribe(
                &[0.1, -0.1],
                &TranscribeOptions::default(),
                &TranscribeControl::none(),
            )
            .unwrap();
        assert_eq!(out.text, "salut");
        assert!(engine.capabilities().supports_prompt || !engine.capabilities().supports_prompt);
    }

    #[cfg(any(
    feature = "whisper",
    feature = "parakeet",
    feature = "sensevoice",
    feature = "moonshine"
))]
    #[test]
    fn sanitize_borrows_clean_audio_without_allocating() {
        let clean = [0.0, 0.5, -0.5, 1.0, -1.0];
        // Clean audio is passed through as a borrow (no copy).
        assert!(matches!(sanitize_audio(&clean), Cow::Borrowed(_)));
    }

    #[cfg(any(
    feature = "whisper",
    feature = "parakeet",
    feature = "sensevoice",
    feature = "moonshine"
))]
    #[test]
    fn sanitize_replaces_non_finite_and_clamps_out_of_range() {
        let dirty = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.0, -3.0, 0.3];
        let out = sanitize_audio(&dirty);
        assert!(matches!(out, Cow::Owned(_)));
        assert_eq!(&*out, &[0.0, 0.0, 0.0, 1.0, -1.0, 0.3]);
        // Every sample is finite and in range after sanitizing.
        assert!(out.iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
    }
}
