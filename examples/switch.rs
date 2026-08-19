//! Test harness for dynamic dispatch: loads each backend through the
//! `load_engine` factory, transcribes the same audio, and **swaps the same
//! `Box<dyn AsrEngine>`** from one backend to another — without the caller
//! knowing the concrete type. This demonstrates “backend dynamically enabled
//! at runtime” (not a UI).
//!
//! Usage (both backends):
//!   cargo run --example switch --features "whisper parakeet" -- [audio.wav] [whisper_model.bin] [parakeet_dir]
//!
//! Without arguments, uses the local fixtures. A backend whose feature was not
//! compiled is shown as “unavailable” and the switch continues.


use dictata_engine::{
    load_engine, AsrEngine, DevicePreference, EngineConfig, EngineKind, LanguageSupport,
    TranscribeControl, TranscribeOptions,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let audio_path = args
        .next()
        .unwrap_or_else(|| "models/jfk.wav".to_string());
    let whisper_model = args
        .next()
        .unwrap_or_else(|| "models/ggml-tiny.bin".to_string());
    let parakeet_dir = args
        .next()
        .unwrap_or_else(|| "models/parakeet-tdt-v3-int8".to_string());

    let audio = read_wav_mono_16k_f32(&audio_path);
    println!(
        "audio : {}  ({:.2} s @ 16 kHz)\n",
        audio_path,
        audio.len() as f32 / 16_000.0
    );

// The “plan” of backends to try, each with its own config. The factory
// returns a `Box<dyn AsrEngine>`: the rest of the code is identical regardless
// of the backend.
    let plan = [
        (EngineKind::Whisper, whisper_model),
        (EngineKind::Parakeet, parakeet_dir),
    ];

    // A SINGLE active engine slot, reassigned on each iteration: this is
    // literally the runtime backend “switch”.
    let mut active: Option<Box<dyn AsrEngine>> = None;
    let opts = TranscribeOptions::default(); // language auto, greedy decoding

    for (kind, model_path) in plan {
        println!("=== switch -> {kind:?} ===");
        let config = EngineConfig {
            model_path: model_path.clone().into(),
            device: DevicePreference::Cpu, // Reproducible, with no GPU dependency.
            default_language: None,
            crash_marker: None,
        };

        let t_load = std::time::Instant::now();
        match load_engine(kind, &config) {
            Ok(engine) => {
                // The previous backend is released here (the old Box is dropped).
                active = Some(engine);
                println!("  model : {model_path}");
                println!("  loaded in {:.2} s", t_load.elapsed().as_secs_f32());
            }
            Err(e) => {
                println!("  unavailable : {e}\n");
                continue;
            }
        }

        let engine = active.as_mut().expect("engine loaded at current iteration");

        let caps = engine.capabilities();
        println!(
            "  capabilities : prompt={} beam={} translate={} vad={} | languages={}",
            caps.supports_prompt,
            caps.supports_beam,
            caps.supports_translate,
            caps.supports_internal_vad,
            match &caps.languages {
                LanguageSupport::Any => "all".to_string(),
                LanguageSupport::Set(v) => format!("{} languages", v.len()),
            }
        );

        let t_tx = std::time::Instant::now();
        match engine.transcribe(&audio, &opts, &TranscribeControl::none()) {
            Ok(res) => {
                let secs = t_tx.elapsed().as_secs_f32();
                let rtf = (audio.len() as f32 / 16_000.0) / secs.max(1e-6);
                println!(
                    "  transcribed in {secs:.2} s  (~{rtf:.0}x real-time)",
                );
                println!("  detected language : {:?}", res.detected_language);
                println!("  text : {}\n", res.text);
            }
            Err(e) => println!("  transcription failed : {e}\n"),
        }
    }

    // Final proof that the slot contains the LAST loaded backend,
    // and that we can still use it after all the switches.
    if let Some(engine) = active.as_mut() {
        println!(
            "backend actif en fin de parcours : {}",
            engine.capabilities().name
        );
    }
}

/// Reads a 16-bit mono 16 kHz WAV PCM file into a `Vec<f32>` without external dependencies.
fn read_wav_mono_16k_f32(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("Read WAV file");
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
                assert_eq!(bits, 16, "only 16-bit PCM is supported by this harness");
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
                    "this harness expects 16 kHz (obtained {sample_rate})"
                );
                return mono;
            }
            _ => {}
        }
        pos = body + size + (size & 1);
    }
    panic!("chunk `data` not found in {path}");
}
