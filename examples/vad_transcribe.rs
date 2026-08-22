//! Manual smoke test of the VAD-gated transcription pipeline.
//!
//! Runs the shared Silero VAD in front of any backend, so only speech regions
//! are transcribed and the result is stitched back onto the original timeline.
//!
//! Usage:
//!   cargo run --example vad_transcribe --features "vad,<backend>" -- \
//!       <backend> <model_dir_or_file> <silero_vad.onnx> <audio.wav>
//!
//! `<backend>` is one of: whisper parakeet sensevoice moonshine zipformer. The
//! matching backend feature must be enabled (e.g. `--features "vad,zipformer"`),
//! otherwise the engine load reports `BackendUnavailable`. `audio.wav` must be
//! 16-bit mono PCM at 16 kHz. Set `EXPECT_SUBSTR` to assert the output contains a
//! substring (case-insensitive), turning this into a reproducible check.

use dictata_engine::pipeline::transcribe_with_vad;
use dictata_engine::vad::{SileroVad, VadConfig};
use dictata_engine::{
    DevicePreference, EngineConfig, EngineKind, TranscribeControl, TranscribeOptions, load_engine,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let backend = args.next().expect("usage: <backend> <model> <silero.onnx> <audio.wav>");
    let model = args.next().expect("usage: <backend> <model> <silero.onnx> <audio.wav>");
    let silero = args.next().expect("usage: <backend> <model> <silero.onnx> <audio.wav>");
    let wav = args.next().expect("usage: <backend> <model> <silero.onnx> <audio.wav>");

    let kind = match backend.to_lowercase().as_str() {
        "whisper" => EngineKind::Whisper,
        "parakeet" => EngineKind::Parakeet,
        "sensevoice" => EngineKind::SenseVoice,
        "moonshine" => EngineKind::Moonshine,
        "zipformer" => EngineKind::Zipformer,
        other => panic!("unknown backend {other:?}"),
    };

    let audio = read_wav_mono_16k_f32(&wav);
    println!(
        "audio: {} samples ({:.2} s @ 16 kHz)",
        audio.len(),
        audio.len() as f32 / 16_000.0
    );

    let config = EngineConfig {
        model_path: model.into(),
        device: DevicePreference::Cpu,
        default_language: None,
        crash_marker: None,
    };
    let mut engine = load_engine(kind, &config).expect("loading backend");
    println!("backend: {}", engine.capabilities().name);

    let mut vad = SileroVad::load(&silero, VadConfig::default()).expect("loading Silero VAD");

    let t0 = std::time::Instant::now();
    let result = transcribe_with_vad(
        engine.as_mut(),
        &mut vad,
        &audio,
        &TranscribeOptions::default(),
        &TranscribeControl::none(),
    )
    .expect("vad-gated transcription");
    println!("done in {:.2} s", t0.elapsed().as_secs_f32());

    println!("language detected : {:?}", result.detected_language);
    println!("--- text ---\n{}\n-------------", result.text);
    println!("segments ({}):", result.segments.len());
    for s in &result.segments {
        println!("  [{:6.2}s -> {:6.2}s] {}", s.start, s.end, s.text);
    }

    if let Ok(expect) = std::env::var("EXPECT_SUBSTR")
        && !expect.trim().is_empty()
    {
        if result.text.to_lowercase().contains(&expect.to_lowercase()) {
            println!("assertion OK: output contains {expect:?}");
        } else {
            eprintln!("assertion FAILED: output does not contain {expect:?}");
            std::process::exit(1);
        }
    }
}

/// Reads a 16-bit mono 16 kHz PCM WAV file into a `Vec<f32>` without external dependencies.
fn read_wav_mono_16k_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("reading WAV file");
    assert!(bytes.len() > 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE");

    let mut channels = 1u16;
    let mut sample_rate = 16_000u32;
    let mut bits = 16u16;
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]])
            as usize;
        let body = pos + 8;
        match id {
            b"fmt " => {
                channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
                sample_rate = u32::from_le_bytes([
                    bytes[body + 4],
                    bytes[body + 5],
                    bytes[body + 6],
                    bytes[body + 7],
                ]);
                bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
            }
            b"data" => {
                assert_eq!(bits, 16, "this smoke test only supports 16-bit PCM");
                let end = (body + size).min(bytes.len());
                let samples: Vec<f32> = bytes[body..end]
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                let mono = if channels > 1 {
                    samples
                        .chunks(channels as usize)
                        .map(|f| f.iter().sum::<f32>() / channels as f32)
                        .collect()
                } else {
                    samples
                };
                assert_eq!(
                    sample_rate, 16_000,
                    "this smoke test expects 16 kHz (got {sample_rate})"
                );
                return mono;
            }
            _ => {}
        }
        pos = body + size + (size & 1);
    }
    panic!("chunk `data` not found in {path}");
}
