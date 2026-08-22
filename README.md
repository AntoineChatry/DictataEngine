<div align="center">

# dictata-engine

**Multi-backend, offline, on-device speech-to-text for Rust.**

One trait. Several ASR engines behind it. No cloud, no network — everything runs locally.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![ASR](https://img.shields.io/badge/ASR-offline%20%7C%20on--device-brightgreen.svg)

</div>

---

`dictata-engine` puts a single, minimal trait — [`AsrEngine`](#the-trait) — in front of
interchangeable transcription backends:

- **whisper.cpp** via [`whisper-rs`](https://crates.io/crates/whisper-rs) — broad language
  coverage, prompting, translation, internal VAD.
- **NVIDIA Parakeet TDT** via [`parakeet-rs`](https://crates.io/crates/parakeet-rs) / ONNX
  Runtime — fast, punctuated, 25-language transducer decoding.
- **SenseVoice** via ONNX Runtime — multilingual CTC (zh / en / ja / ko / yue) with a pure-Rust
  kaldi-fbank front-end.
- **Moonshine** via ONNX Runtime — a compact English encoder-decoder for short-form speech.
- **Zipformer** via ONNX Runtime — an offline RNN-T (transducer) with a pure-Rust kaldi-fbank
  front-end; language coverage follows whatever model you load (English, multilingual, …).

Pick a backend with a Cargo feature, load a model, hand it audio, get text. The heavy native
dependency of each backend is gated behind its feature, so the base crate pulls neither
whisper.cpp nor ONNX Runtime.

```rust
use dictata_engine::{
    load_engine, EngineKind, EngineConfig, DevicePreference, TranscribeOptions, TranscribeControl,
};

let mut engine = load_engine(EngineKind::Parakeet, &EngineConfig {
    model_path: "models/parakeet-tdt-v3".into(), // directory name is yours to choose
    device: DevicePreference::Auto,
    default_language: None,
    crash_marker: None,
})?;

let result = engine.transcribe(
    &audio, // mono 16 kHz f32
    &TranscribeOptions::default(),
    &TranscribeControl::none(), // no cancellation, no progress
)?;
println!("{}", result.text);
# Ok::<(), dictata_engine::AsrError>(())
```

The whole point of the abstraction: the code above is **identical** whether
`EngineKind::Whisper` or `EngineKind::Parakeet` is running. Swap backends at runtime by
reassigning a single `Box<dyn AsrEngine>` — see [`examples/switch.rs`](examples/switch.rs).

---

## At a glance

- 🦀 **Pure-Rust core** — native backends only build when you ask for them.
- 🔌 **Runtime backend swap** — everything hides behind `Box<dyn AsrEngine>`.
- 🧭 **Honest capabilities** — `engine.capabilities()` says what a backend actually honours,
  so the app can grey out unsupported options instead of guessing.
- 🧱 **Stateless between calls** — one instance is safely shared between one-shot and streaming.
- 🛡️ **Defensive input** — the audio buffer is sanitized (NaN/Inf → silence, clamped to range)
  before it reaches native code that could otherwise crash or produce garbage.
- 🧠 **Bounded memory on long clips** — the Parakeet, SenseVoice and Moonshine paths window
  internally (≤ 32 s per window, cut on the lowest-energy point) so ONNX Runtime's arena never
  balloons on a multi-minute file.
- ⏱️ **Timestamps** — every result carries timed `segments` (seconds) for subtitles, alignment
  or click-to-seek; ignore them and read `text` when you only want the words.
- 🛑 **Cancellable with progress** — pass a `TranscribeControl` to cancel a long transcription
  cooperatively (returns `AsrError::Cancelled`) and receive `0..=100` progress; whisper.cpp
  reports through its native callbacks, Parakeet between its internal windows.
- 🗣️ **Shared VAD** — an optional Silero voice-activity detector (feature `vad`) every backend
  can reuse to find speech regions and skip silence, independent of whisper.cpp's internal VAD.
- 👥 **Speaker diarization** — an optional Sortformer pass (feature `diarize`, Parakeet-only)
  that labels *who spoke when* as speaker-tagged time spans, beside the trait — compose it with
  `transcribe` for speaker-attributed text.

---

## Backends & features

| Feature             | Backend                        | Pulls in                  | Acceleration                |
| ------------------- | ------------------------------ | ------------------------- | --------------------------- |
| *(none)*            | `MockEngine` only              | nothing                   | —                           |
| `whisper`           | whisper.cpp via `whisper-rs`   | whisper.cpp (C/C++ build) | CPU                         |
| `vulkan`            | whisper.cpp + Vulkan           | whisper.cpp, Vulkan       | Vulkan (AMD/NVIDIA/Intel)   |
| `parakeet`          | Parakeet TDT via `parakeet-rs` | ONNX Runtime (`ort`)      | CPU                         |
| `parakeet-directml` | Parakeet + DirectML            | ONNX Runtime, DirectML    | DirectML (any Windows GPU)  |
| `sensevoice`        | SenseVoice CTC via `ort`       | ONNX Runtime (`ort`)      | CPU                         |
| `moonshine`         | Moonshine (EN) via `ort`       | ONNX Runtime (`ort`)      | CPU                         |
| `zipformer`         | Zipformer RNN-T via `ort`      | ONNX Runtime (`ort`)      | CPU                         |
| `diarize`           | Sortformer diarization         | ONNX Runtime (`ort`)      | CPU                         |

Features are **additive** — build with `whisper` and `parakeet` at once if you want both.
`MockEngine` is always available and needs no feature; it is the reference implementation and
the test double.

```toml
[dependencies]
dictata-engine = { git = "https://github.com/AntoineChatry/dictata-engine", features = ["parakeet"] }
```

### What each backend honours

`TranscribeOptions` is a superset; a backend cleanly ignores what it does not support. Query
`capabilities()` up front to know which is which.

| Capability                | whisper.cpp        | Parakeet TDT           | SenseVoice          | Moonshine           | Zipformer               |
| ------------------------- | :----------------: | :--------------------: | :-----------------: | :-----------------: | :---------------------: |
| Languages                 | any (auto-detect)  | 25 (fixed set)         | 5 (fixed set)       | English only        | per model               |
| Initial prompt            | ✅                 | ❌                     | ❌                  | ❌                  | ❌                      |
| Beam search               | ✅                 | ❌                     | ❌                  | ❌                  | ❌                      |
| Translate → English       | ✅                 | ❌                     | ❌                  | ❌                  | ❌                      |
| Internal VAD              | ✅                 | ❌                     | ❌                  | ❌                  | ❌                      |
| Word/segment timestamps   | ✅                 | ✅                     | ❌                  | ❌                  | ❌                      |
| Detected language         | ✅                 | ❌                     | ✅                  | ✅ *(fixed EN)*     | ❌                      |
| Cancellation + progress   | ✅ *(per step)*    | ✅ *(per ~30 s window)*| ✅ *(per call)*     | ✅ *(per token)*    | ✅ *(per encoder frame)*|

---

## Build requirements

The base crate is pure Rust. **The heavy toolchains only kick in when you enable a backend
feature.**

### `whisper` / `vulkan`
`whisper-rs` compiles whisper.cpp from source, so you need a C/C++ toolchain:

- **CMake**
- A C/C++ compiler: MSVC (Windows), clang or gcc (Linux/macOS)
- **`libclang`** (bindgen) — usually part of the LLVM install
- For `vulkan`: the **Vulkan SDK** present at build time

### `parakeet` / `parakeet-directml`
`parakeet-rs` uses ONNX Runtime through `ort`. By default `ort` downloads a prebuilt ONNX
Runtime binary; to use a system install, follow the [`ort` docs](https://ort.pyke.io/).
`parakeet-directml` targets Windows and runs on any Direct3D 12 GPU (AMD/NVIDIA/Intel), falling
back to CPU when no suitable device is present.

> **Platform note.** Development and testing target Windows, but the CPU paths (`whisper`,
> `parakeet`) are portable. The only Windows-specific code is the `MAX_PATH` guard in the
> loaders and the DirectML feature.

---

## Audio contract

Every backend consumes the **same** input:

> **mono**, **16 kHz**, **`f32`** samples in `[-1.0, 1.0]`.

The engine does **not** resample on its own — that stays the caller's job (typically the capture
stage or an ffmpeg decode). For convenience, the optional **`resample`** feature ships a
dependency-free helper that converts arbitrary-rate, multi-channel PCM to the contract:

```rust
# #[cfg(feature = "resample")]
let audio = dictata_engine::resample::to_engine_audio(&interleaved, 48_000, 2); // -> mono 16 kHz
```

It uses channel averaging + linear interpolation (negligible CPU, no extra crate); good enough
for 16 kHz ASR. Bring your own resampler (e.g. `rubato`) if you need windowed-sinc quality.

Out-of-contract input is defended against, not silently trusted: `transcribe` sanitizes the
buffer first (`NaN`/`±Inf` → silence, out-of-range clamped), because a single bad sample can
otherwise make whisper.cpp `abort()` the process or poison Parakeet's feature extraction.

---

## Voice activity detection (VAD)

Optional, behind the **`vad`** feature: a **Silero** VAD that any backend can share. Parakeet has
no VAD of its own and whisper's is buried in whisper.cpp — this is the one detector both can use
to locate speech and skip silence before transcribing. It pulls ONNX Runtime directly (pinned to
the same `ort` as Parakeet, so a `parakeet` + `vad` build shares a single runtime).

```rust
# #[cfg(feature = "vad")]
# fn demo(audio: &[f32]) -> Result<(), dictata_engine::AsrError> {
use dictata_engine::vad::{SileroVad, VadConfig};

let mut vad = SileroVad::load("models/silero_vad.onnx", VadConfig::default())?;
for seg in vad.segments(audio)? {           // sample indices into `audio`, end-exclusive
    let (start, end) = (seg.start, seg.end); // ÷ SAMPLE_RATE for seconds
    println!("speech {start}..{end}");
}
let speech_only = vad.collect_speech(audio)?; // silence dropped, ready for transcribe
# Ok(()) }
```

For the common case — *run the VAD in front of a backend, transcribe only the speech, keep the
original timeline* — the crate wires the two together for you with `pipeline::transcribe_with_vad`,
so you don't re-implement the offset bookkeeping. It transcribes each speech span, drops the
silence, and stitches every segment back onto the **original** timeline; backends that emit no
timestamps (Zipformer, SenseVoice, Moonshine) get one segment synthesized per span, so the result
always carries a coherent timeline. Cancellation and progress work as usual (progress tracks the
fraction of *speech* already transcribed).

```rust
# #[cfg(feature = "vad")]
# fn demo(engine: &mut dyn dictata_engine::AsrEngine, audio: &[f32]) -> Result<(), dictata_engine::AsrError> {
use dictata_engine::pipeline::transcribe_with_vad;
use dictata_engine::vad::{SileroVad, VadConfig};
use dictata_engine::{TranscribeOptions, TranscribeControl};

let mut vad = SileroVad::load("models/silero_vad.onnx", VadConfig::default())?;
let result = transcribe_with_vad(
    engine, &mut vad, audio,
    &TranscribeOptions::default(),
    &TranscribeControl::none(),
)?;
println!("{}", result.text);
# Ok(()) }
```

Bring a **Silero v5** ONNX model (`silero_vad.onnx`, ~2 MB, from
[snakers4/silero-vad](https://github.com/snakers4/silero-vad)) — not bundled, like the ASR
models. `VadConfig` exposes the usual knobs (`threshold`, `min_speech_ms`, `min_silence_ms`,
`speech_pad_ms`); the defaults match Silero's own reference.

---

## Speaker diarization

Optional, behind the **`diarize`** feature (Parakeet-only): *who spoke when*, as a standalone
pass — deliberately **outside** the `AsrEngine` trait, since it produces speaker-labelled spans,
not text. Like the shared VAD it lives beside the trait; compose the two — diarize to get the
spans, then transcribe each — when you want speaker-attributed text. Backed by NVIDIA's streaming
**Sortformer v2** (up to 4 speakers) through `parakeet-rs`, so it shares Parakeet's ONNX Runtime.

```rust
# #[cfg(feature = "diarize")]
# fn demo(audio: &[f32]) -> Result<(), dictata_engine::AsrError> {
use dictata_engine::diarize::{Diarizer, DiarizeConfig};

let mut diarizer = Diarizer::load("models/sortformer_4spk_v2.onnx", DiarizeConfig::default())?;
for seg in diarizer.diarize(audio)? {        // sample indices into `audio`, end-exclusive
    let (start, end) = (seg.start, seg.end); // ÷ SAMPLE_RATE for seconds
    println!("speaker {} {start}..{end}", seg.speaker);
}
# Ok(()) }
```

Bring a **Sortformer v2** ONNX model (`diar_streaming_sortformer_4spk-v2`, from
[nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2))
— not bundled, like the ASR models. `DiarizeConfig` exposes the segmentation knobs (`onset`,
`offset`, `min_speech_ms`, `min_gap_ms`); the defaults match NVIDIA's tuned CallHome preset. This
offline pass is tuned for **turn-taking** speech (meetings, interviews, multi-speaker dictation),
not heavy cross-talk.

---

## Models

Models are **not** bundled (they are hundreds of MB to several GB, with their own licenses).
Download them separately. The paths below are examples — the file/directory name is yours.

**whisper** — a single ggml `.bin` **file**, e.g. from
[`ggerganov/whisper.cpp`](https://huggingface.co/ggerganov/whisper.cpp):

```
models/ggml-base.bin
```

**Parakeet** — a **directory** (a HuggingFace bundle) containing at least:

```
models/parakeet-tdt-v3/
├── encoder-model.onnx
├── decoder_joint-model.onnx
└── vocab.txt
```

**SenseVoice** — a **directory** (a sherpa-onnx export) containing:

```
models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue/
├── model.int8.onnx   (or model.onnx)
└── tokens.txt
```

**Moonshine** — a **directory** (a sherpa-onnx "v1" export, e.g.
[`csukuangfj/sherpa-onnx-moonshine-base-en-int8`](https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8))
containing the four split graphs plus the SentencePiece vocabulary (`.int8.onnx`
variants are preferred, plain `.onnx` also works):

```
models/sherpa-onnx-moonshine-base-en-int8/
├── preprocess.onnx
├── encode.int8.onnx          (or encode.onnx)
├── uncached_decode.int8.onnx (or uncached_decode.onnx)
├── cached_decode.int8.onnx   (or cached_decode.onnx)
└── tokens.txt
```

**Zipformer** — a **directory** (a sherpa-onnx offline zipformer transducer export, e.g.
[`k2-fsa/sherpa-onnx-zipformer-gigaspeech-2023-12-12`](https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-gigaspeech-2023-12-12))
containing the three transducer graphs plus `tokens.txt`. File names vary between exports, so the
loader matches by substring (`encoder` / `decoder` / `joiner`) and prefers the `int8` variant:

```
models/sherpa-onnx-zipformer-gigaspeech-2023-12-12/
├── encoder-epoch-30-avg-1.int8.onnx (or the plain .onnx)
├── decoder-epoch-30-avg-1.int8.onnx
├── joiner-epoch-30-avg-1.int8.onnx
└── tokens.txt
```

`EngineConfig::model_path` points at the **file** (whisper) or the **directory** (Parakeet,
SenseVoice, Moonshine, Zipformer).

---

## Usage

### The trait

```rust
pub trait AsrEngine: Send {
    fn transcribe(
        &mut self,
        audio: &[f32],
        opts: &TranscribeOptions,
        control: &TranscribeControl,
    ) -> Result<TranscriptionResult, AsrError>;
    fn capabilities(&self) -> &EngineCapabilities;
}
```

`TranscriptionResult` carries the `text`, the backend-`detected_language` (when available), and
timed `segments` (`{ text, start, end }` in seconds) for subtitles or alignment.

`TranscribeControl` carries an optional `cancel` flag (`Arc<AtomicBool>` — share the one your
app already uses) and an optional `on_progress` sink (`Fn(u8)`, `0..=100`). Pass
`&TranscribeControl::none()` for a plain call; a requested cancellation returns
`AsrError::Cancelled`, never a partial `Ok`. Cancellation is **cooperative**: whisper.cpp checks
between decode steps, Parakeet between its ~30 s windows.

Engines are **stateless between calls**: every `transcribe` is independent. Any continuity
(custom vocabulary, already-emitted text) travels through `TranscribeOptions::initial_prompt`,
never through state carried on the engine — so one instance can be shared between a one-shot
path and a streaming path. The trait is **synchronous**; run it on a worker thread (there is no
async runtime).

### Loading

Use the `load_engine` factory for dynamic dispatch, or a concrete type (`WhisperEngine::load`,
`ParakeetEngine::load`) when you know the backend at compile time.

```rust
let engine: Box<dyn AsrEngine> = load_engine(EngineKind::Whisper, &config)?;
```

### Streaming

The crate does not orchestrate streaming — it gives you the primitive. The caller cuts audio
into chunks (e.g. on speech pauses) and calls `transcribe` per chunk. On a long **one-shot**
clip the Parakeet, SenseVoice and Moonshine backends also window internally (≤ 32 s per window,
cutting on the lowest-energy point) and free each window between passes, so ONNX Runtime's arena
stays bounded to a single window's peak instead of growing with the whole clip.

---

## Examples

The smoke examples read the WAV themselves (16-bit PCM, mono, 16 kHz), so they carry no
audio-crate dependency.

```bash
# Parakeet smoke test — model directory + a mono 16 kHz WAV (+ optional "gpu")
cargo run --example parakeet_smoke --features parakeet -- models/parakeet-tdt-v3 audio.wav

# whisper smoke test — ggml .bin + WAV
cargo run --example whisper_smoke --features whisper -- models/ggml-base.bin audio.wav

# Zipformer smoke test — model directory + a mono 16 kHz WAV
cargo run --example zipformer_smoke --features zipformer -- models/sherpa-onnx-zipformer-gigaspeech-2023-12-12 audio.wav

# VAD-gated transcription — Silero in front of any backend (here Zipformer), speech only
cargo run --example vad_transcribe --features "vad,zipformer" -- zipformer models/sherpa-onnx-zipformer-gigaspeech-2023-12-12 models/silero_vad.onnx audio.wav

# Speaker diarization — who spoke when (Sortformer .onnx + WAV)
cargo run --example diarize_smoke --features diarize -- models/diar_streaming_sortformer_4spk-v2.onnx audio.wav

# Swap both backends behind one Box<dyn AsrEngine>, on the same audio
cargo run --example switch --features "whisper parakeet" -- audio.wav models/ggml-tiny.bin models/parakeet-tdt-v3

# Live microphone: record between two Enter presses, then transcribe with BOTH backends
# (uses the cpal dev-dependency; args optional)
cargo run --example mic --features "whisper parakeet" -- models/ggml-tiny.bin models/parakeet-tdt-v3
```

---

## Testing

```bash
cargo test --features "whisper parakeet"
```

Unit tests cover the trait, the `load_engine` factory, audio sanitizing, the Parakeet
windowing math, and the VAD pipeline's segment placement (timeline shift vs. synthesized
segment). They do **not** require a model — the model-dependent checks are the smoke examples
above.

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE).

Models are **not** covered by this license; each carries its own terms (check the model card
before redistributing).
