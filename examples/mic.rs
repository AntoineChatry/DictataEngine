//! REAL-WORLD TEST: speak into the microphone, then BOTH backends
//! transcribe your live audio and switch from one to the other. This proves
//! that this is not a file replay — it's your voice, captured right now.
//!
//! Usage:
//!   cargo run --example mic --features "whisper parakeet" -- [whisper_model.bin] [parakeet_dir]
//!
//! Flow: Press Enter to start → speak → press Enter to stop →
//! Whisper transcription, then switch to Parakeet, using the SAME recording.

use std::io::{BufRead, Write};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use dictata_engine::{
    load_engine, AsrEngine, DevicePreference, EngineConfig, EngineKind, LanguageSupport,
    TranscribeControl, TranscribeOptions, SAMPLE_RATE,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let whisper_model = args
        .next()
        .unwrap_or_else(|| "models/ggml-tiny.bin".to_string());
    let parakeet_dir = args
        .next()
        .unwrap_or_else(|| "models/parakeet-tdt-v3-int8".to_string());

    let audio = record_from_mic();
    println!(
        "\ncaptured : {} samples ({:.2} s @ 16 kHz)\n",
        audio.len(),
        audio.len() as f32 / SAMPLE_RATE as f32
    );
    if audio.len() < 4000 {
        println!("(audio very short — speak a bit longer for a real test)\n");
    }

    // Same live audio, two backends, one reassigned engine slot.
    let plan = [
        (EngineKind::Whisper, whisper_model),
        (EngineKind::Parakeet, parakeet_dir),
    ];
    let mut active: Option<Box<dyn AsrEngine>> = None;
    let opts = TranscribeOptions::default(); // language auto

    for (kind, model_path) in plan {
        println!("=== {kind:?} ===");
        let config = EngineConfig {
            model_path: model_path.clone().into(),
            device: DevicePreference::Cpu,
            default_language: None,
            crash_marker: None,
        };
        match load_engine(kind, &config) {
            Ok(engine) => active = Some(engine),
            Err(e) => {
                println!("  unavailable : {e}\n");
                continue;
            }
        }
        let engine = active.as_mut().unwrap();
        let caps = engine.capabilities();
        let langs = match &caps.languages {
            LanguageSupport::Any => "all".to_string(),
            LanguageSupport::Set(v) => format!("{} languages", v.len()),
        };
        println!("  ({}, {langs})", caps.name);
        match engine.transcribe(&audio, &opts, &TranscribeControl::none()) {
            Ok(res) => {
                println!("  detected language : {:?}", res.detected_language);
                println!("  text : {}\n", res.text);
            }
            Err(e) => println!("  failure : {e}\n"),
        }
    }

    match active.as_mut() {
        Some(engine) => println!(
            "backend active in fin of test : {}",
            engine.capabilities().name
        ),
        None => println!("no backend compiled : relaunch with --features \"whisper parakeet\""),
    }
}

/// Capture the microphone by default between two presses of Enter, and return
/// the audio in mono 16 kHz f32 (downmix + linear resampling).
fn record_from_mic() -> Vec<f32> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("no default microphone available");
    let cfg = device
        .default_input_config()
        .expect("microphone configuration unavailable");
    let native_rate = cfg.sample_rate();
    let channels = cfg.channels() as usize;
    let sample_format = cfg.sample_format();
    println!(
        "micro : {} — {} Hz, {} channels, {:?}",
        device.description().map(|d| d.name().to_string()).unwrap_or_default(),
        native_rate,
        channels,
        sample_format
    );

    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let stream_cfg: cpal::StreamConfig = cfg.into();
    let err_fn = |e| eprintln!("error audio stream : {e}");

    let buf_cb = buffer.clone();
    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            stream_cfg,
            move |data: &[f32], _| feed(&buf_cb, data, channels),
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            stream_cfg,
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                feed(&buf_cb, &f, channels)
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            stream_cfg,
            move |data: &[u16], _| {
                let f: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0 - 1.0).collect();
                feed(&buf_cb, &f, channels)
            },
            err_fn,
            None,
        ),
        other => panic!("unsupported audio format: {other:?}"),
    }
    .expect("opening the microphone stream");

    prompt("Press Enter to START recording...");
    buffer.lock().unwrap().clear(); // ignore the audio before the start
    stream.play().expect("démarrage du flux");
    prompt(">>> Speak, then Press Enter to STOP...");
    drop(stream); // stop the capture

    let native = std::mem::take(&mut *buffer.lock().unwrap());
    resample_to_16k(&native, native_rate)
}

/// Downmix to mono and accumulate.
fn feed(buffer: &Arc<Mutex<Vec<f32>>>, data: &[f32], channels: usize) {
    if channels == 0 {
        return;
    }
    let mut g = buffer.lock().unwrap_or_else(|p| p.into_inner());
    for frame in data.chunks(channels) {
        g.push(frame.iter().copied().sum::<f32>() / channels as f32);
    }
}

/// Linear resampling of a complete buffer to 16 kHz.
fn resample_to_16k(input: &[f32], from_rate: u32) -> Vec<f32> {
    if from_rate == SAMPLE_RATE || input.is_empty() {
        return input.to_vec();
    }
    let ratio = SAMPLE_RATE as f64 / from_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 / ratio;
        let idx = src.floor() as usize;
        let frac = (src - idx as f64) as f32;
        let a = input.get(idx).copied().unwrap_or(0.0);
        let b = input.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

fn prompt(msg: &str) {
    print!("{msg} ");
    let _ = std::io::stdout().flush();
    let mut line = String::new();
    let _ = std::io::stdin().lock().read_line(&mut line);
}
