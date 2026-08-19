//! Manual smoke test of the Whisper backend through the `AsrEngine` trait.
//!
//! Usage:
//!   cargo run --example whisper_smoke --features whisper -- <model.bin> <audio.wav>
//!
//! `audio.wav` must be 16-bit mono PCM at 16 kHz (the `jfk.wav` file from
//! whisper.cpp is). The example does not depend on any audio crate: it reads
//! the WAV file itself.


use dictata_engine::engines::WhisperEngine;
use dictata_engine::{AsrEngine, DevicePreference, EngineConfig, TranscribeControl, TranscribeOptions};

fn main() {
    let mut args = std::env::args().skip(1);
    let model = args.next().expect("usage: <model.bin> <audio.wav>");
    let wav = args.next().expect("usage: <model.bin> <audio.wav>");
    let device = match args.next().as_deref() {
        Some("gpu") => DevicePreference::Gpu,
        _ => DevicePreference::Cpu,
    };
    println!("device requested: {device:?}");

    let audio = read_wav_mono_16k_f32(&wav);
    println!(
        "audio: {} samples ({:.2} s @ 16 kHz)",
        audio.len(),
        audio.len() as f32 / 16_000.0
    );

    let config = EngineConfig {
        model_path: model.into(),
        device,
        default_language: None,
        crash_marker: None,
    };

    let t0 = std::time::Instant::now();
    let mut engine = WhisperEngine::load(&config).expect("loading the model");
    println!("model loaded in {:.2} s", t0.elapsed().as_secs_f32());
    println!("capabilities: {:?}", engine.capabilities());

    let t1 = std::time::Instant::now();
    let result = engine
        .transcribe(&audio, &TranscribeOptions::default(), &TranscribeControl::none())
        .expect("transcription");
    println!("transcribed in {:.2} s", t1.elapsed().as_secs_f32());

    println!("detected language: {:?}", result.detected_language);
    println!("--- texte ---\n{}\n-------------", result.text);
}

/// Reads a 16-bit mono PCM WAV file into a `Vec<f32>` without external dependencies.
/// Walks through the RIFF chunks until `data`, reads the `fmt ` chunk on the way.
fn read_wav_mono_16k_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("loading the WAV file");
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
                assert_eq!(bits, 16, "only 16-bit PCM is supported by this smoke test");
                let end = (body + size).min(bytes.len());
                let samples: Vec<f32> = bytes[body..end]
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
                    .collect();
                // Repli mono if the file was multi-channel (average per frame).
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
        // Chunks are aligned to 2-byte boundaries.
        pos = body + size + (size & 1);
    }
    panic!("chunk `data` not found in {path}");
}
