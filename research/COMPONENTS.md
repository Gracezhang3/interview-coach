# Component spec cards

One card per component in the voice-analysis pipeline, so we can decide which models
can realistically be combined into the final real-time system.

**All numbers are measured** on one machine: Apple M1 Pro (8 performance + 2 efficiency cores), 16 GB RAM,
macOS, Python 3.12, **CPU only (no GPU)**. Raw results are in [`results/components.json`](results/components.json)
(profiled by [`profile_components.py`](profile_components.py), each component in its own process)
and [`results/benchmark.json`](results/benchmark.json) (accuracy, from [`benchmark.py`](benchmark.py)).
"Not measured" means exactly that.

Test audio: 7 synthetic interview answers (macOS text-to-speech, 4 accents, 18–32 s each, 16 kHz mono)
with exact reference transcripts, plus noisy copies at 10 dB and 0 dB SNR. Synthetic speech is cleaner
than real speech, so treat accuracy as optimistic. See *Limits* at the end.

---

## 1. Whisper speech-to-text (via faster-whisper)

| | |
|---|---|
| **Role** | Answer audio → transcript with a start/end time for every word |
| **Implementation** | [faster-whisper 1.2.1](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 re-implementation of [OpenAI Whisper](https://github.com/openai/whisper)), int8 weights |
| **Input** | Mono float32 PCM at **16 kHz**, any length. Internally split into **30 s windows** → log-Mel spectrogram of **80 × 3000** (tiny–medium) or **128 × 3000** (large-v3, distil-large-v3), i.e. 10 ms hop |
| **Output** | List of words, each `{word, start_s, end_s, probability}` (N words × 4); plus segment text. ~2.9 words/s of speech |
| **Licence** | Whisper code + weights: **MIT**. faster-whisper: MIT. CTranslate2: MIT. distil-large-v3 weights: MIT |
| **GPU** | Not required. CUDA supported by CTranslate2 (not measured, no NVIDIA GPU here). No Apple-GPU (Metal) path in CTranslate2 |

### Latency and memory per size (CPU, int8, beam 5, word timestamps)

| Model | Params | Weights on disk | Load | 5 s chunk | 5 s chunk, greedy¹ | 28 s answer | Peak RAM | WER clean / 0 dB² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tiny | 39 M | 72 MB | 0.3 s | 0.27 s | 0.17 s | 1.0 s | 0.53 GB | 2.3% / 16.6% |
| base | 74 M | 138 MB | 0.4 s | 0.43 s | 0.33 s | 1.2 s | 0.65 GB | 3.7% / 18.3% |
| **small** | 244 M | 461 MB | 0.8 s | 1.19 s | 1.02 s | 4.4 s | **1.02 GB** | **0.9% / 4.6%** |
| medium | 769 M | 1.4 GB | 1.9 s | 3.67 s | 2.90 s | 8.8 s | 2.10 GB | 0.3% / 1.7% |
| large-v3 | 1550 M | 2.9 GB | 4.3 s | 5.91 s | 5.12 s | 12.6 s | 3.35 GB | 0.0% / 1.7% |
| distil-large-v3 | 756 M | 1.4 GB | 1.7 s | 3.68 s | 3.59 s | 4.5 s | 1.91 GB | 0.6% / 2.3% |

¹ beam size 1, no word timestamps: the cheapest setting, for live captions.
² clean = 4 answers; 0 dB = 2 answers with noise as loud as the voice. WER after lowercasing and removing punctuation.
Load time excludes the one-off download. Peak RAM is the whole process (Python + runtime ≈ 0.1 GB of it).

Note the **5 s chunk** column: a short chunk costs almost as much as a long one for the larger
models (large-v3: 5.9 s for 5 s of audio) because every call pads to a 30 s window. tiny, base
and small keep up with live 5 s chunks easily; medium and distil-large-v3 only just (3.7 s per 5 s);
large-v3 cannot on this CPU.

### Success example
`clean_us.wav`, Whisper small, word-perfect (0 errors / 83 words), including the fillers:
```
  0.00– 0.14  Tell     0.14– 0.34  me       0.34– 0.60  about    0.60– 1.04  yourself.
  1.54– 1.84  Sure.    2.18– 2.36  Um,      2.60– 2.88  so       2.88– 3.14  I'm …
```

### Failure case
`technical_in.wav` (Indian-English voice), Whisper **base**: 11.7% WER from two classic errors.
- **Deletion:** the first sentence, *"Walk me through a project you built."*, is missing entirely.
- **Hallucination:** *"…under one second with only **10 seconds and leave** a small drop in accuracy"*: four invented words.

small and larger models got this clip word-perfect. Separately, at 0 dB noise tiny and base get roughly
1 word in 6 wrong.

**Known open issue: filler words.** Whisper tends to drop "um"/"uh" in real speech. We prompt it with
a disfluent prompt (`"Um, uh, so, like, you know…"`) to keep them. Our synthetic voices pronounce "um" as a
clear word, so fillers survived with or without the prompt (95% kept, [`filler_ablation.py`](filler_ablation.py)).
**The effect on real speech is unmeasured.**

---

## 2. Transcript metrics (our code, deterministic)

| | |
|---|---|
| **Role** | Word timestamps → speaking-behaviour metrics |
| **Implementation** | `transcript_metrics()` in [`analyze.py`](analyze.py), pure Python |
| **Input** | Whisper's word list: N × `{word, start_s, end_s}` plus total audio duration |
| **Output** | 11 numbers: words, speaking rate (wpm, total time), articulation rate (wpm, speech time only), filler count and per minute, repetitions, pause count / total / mean / max (gaps ≥ 0.3 s), pause ratio |
| **Latency** | **0.1 ms** per 83-word answer |
| **CPU / GPU / memory** | CPU, negligible memory (< 1 MB of data) |
| **Licence** | Ours (repository) |

**Success example**: `clean_us.wav` (28.3 s, Whisper small): 83 words, **176 wpm** speaking rate, **200 wpm**
articulation rate, 0 repetitions, **8 pauses**, mean 0.40 s, max 0.52 s, 11% of the time paused.
It found the 3 real fillers: "um", "uh", "you know".

**Failure case**: it is only as good as its input and its word lists.
- The same clip scores **4 fillers, not 3**: *"performance **actually** matters"* is counted, but there
  "actually" is meaningful, not a filler.
- *"I would like to work on systems I like"* → **2 fillers**, but both "like"s are verbs. Word-list filler
  detection cannot tell filler "like" from verb "like". That needs context (a language model).
- Any word Whisper drops becomes a fake pause: base's dropped first sentence above would be read as
  2–3 s of silence.
- "Repetitions" only catches exact repeats ("I I", "we we were"). Real self-corrections
  (*"we went, I mean, I went"*) are not detected.

---

## 3. openSMILE: vocal features (eGeMAPS v02)

| | |
|---|---|
| **Role** | Audio → standard acoustic voice descriptors |
| **Implementation** | [opensmile-python 2.6.0](https://github.com/audeering/opensmile-python) (wraps [openSMILE 3](https://github.com/audeering/opensmile)), feature set **eGeMAPSv02** ([Eyben et al., 2016](https://doi.org/10.1109/TAFFC.2015.2457417)) |
| **Input** | Mono PCM (float32), any sample rate (we use 16 kHz), any length |
| **Output** | **Low-level descriptors**: **frames × 25** at 100 frames/s (2829 × 25 for 28.3 s): F0, loudness, jitter, shimmer, HNR, formants F1–F3, spectral slope, alpha ratio, Hammarberg index, MFCC 1–4, …<br>**Functionals**: **1 × 88** per clip: means, std, percentiles, voiced/unvoiced segment stats, rate of loudness peaks |
| **Latency** | 28.3 s clip: **0.39 s** functionals + **0.41 s** LLDs. 5 s chunk (both): **0.15 s**. Load 0.18 s |
| **CPU / GPU / memory** | CPU only (C++). **Peak RAM 0.15 GB** for the whole process |
| **Licence** | ⚠️ **audEERING Research License: non-commercial use only.** Teaching, academic research and evaluation are fine. *"Any further commercial use requires you to obtain a commercial license"* from audEERING. **Decide this before productising.** Open alternatives for the same features: Praat/Parselmouth (GPL-3.0), librosa (ISC) |

**Success example**: `clean_us.wav`: F0 mean **169 Hz**, F0 variation **4.1 semitones** (SD), voiced
**78%** of frames, 2.8 voiced segments/s, loudness 1.04 sones, jitter 1.27%, shimmer 0.91 dB, HNR 9.0 dB,
MFCC 1–4 means 26.0 / 6.4 / 12.6 / −6.3. The F0 contour follows each sentence's falling intonation
(see `media/walkthrough.png`).

**Failure case**: **background noise breaks voicing and pitch statistics.** Same answer, clean vs. 0 dB noise:

| feature | `slow_au` (clean) | `noisy0db_au` | effect |
|---|---:|---:|---|
| voiced ratio | 80.5% | **39.9%** | half the speech is no longer detected as voiced |
| F0 variation | 2.98 st | **1.39 st** | a lively speaker now looks **monotone** |
| jitter | 0.80% | **1.67%** | doubles: would look like a strained voice |
| HNR | 10.5 dB | 6.8 dB | |
| loudness | 0.95 | **2.10** | measures the noise, not the speaker |
| F0 mean | 192.5 Hz | 192.4 Hz | robust |

Also: jitter, shimmer and HNR on a **synthetic** voice say nothing about a person. They are only
here to show the pipeline works.

---

## 4. Browser live analyser (Web Audio API + our JavaScript)

| | |
|---|---|
| **Role** | Live coaching while the user talks: loudness, pitch, pauses, "um"-sound detection |
| **Implementation** | `frame()`, `detectPitch()`, `trackHesitation()` in [`static/app.js`](../static/app.js): RMS level + normalised-autocorrelation pitch (70–400 Hz, decimated to ~16 kHz) |
| **Input** | Microphone via `getUserMedia` → `AnalyserNode`. Every **50 ms**: one window of **2048 float32 samples** at the device rate (48 kHz → 43 ms) |
| **Output** | Per frame `{t, dB, pitch_Hz or null}` (3 numbers, 20 frames/s), plus derived gauges and nudges |
| **Latency** | **0.09 ms** per voiced frame, 0.003 ms per silent frame (measured in Chromium): **0.2% of the 50 ms budget**. End-to-end ≈ one 50 ms hop |
| **CPU / GPU / memory** | Any laptop browser, CPU, single thread. Page JS heap **1.8 MB** used |
| **Licence** | Ours. Web Audio API is a W3C standard in all modern browsers |

**Success example**: synthetic 170 Hz voiced frame → **170.2 Hz**, −12.4 dB. On the demo answer the live
gauges read 168–171 wpm and "lively (3.3 st)", matching the offline numbers (176 wpm, 4.1 st).

**Failure case**:
- **Octave error in noise:** the same 170 Hz frame with added noise → **84.7 Hz** (half the true pitch).
  That is a classic autocorrelation failure, and it inflates "pitch variety".
- The "um"-sound detector (flat pitch ≥ 0.45 s) flagged **2** hesitations in the demo answer, which
  contains **1** spoken "um": a false positive on a held vowel.

---

## 5. Live transcript: Web Speech API (browser)

| | |
|---|---|
| **Role** | Live words for the pace gauge and filler highlighting while the user speaks |
| **Implementation** | `webkitSpeechRecognition`, continuous, interim results |
| **Input** | The microphone stream (the browser handles it, no format control) |
| **Output** | Interim and final text strings. **No word timestamps, no confidence per word** |
| **Latency** | Not measured (the mic is blocked in the test browser). Interim words typically appear within a second |
| **CPU / GPU / memory** | **Runs on Google's servers in Chrome: audio leaves the device.** Requires network |
| **Licence** | Browser feature; covered by Google's terms in Chrome. Not available in Firefox |

**Success example**: in Chrome, words appear as the user speaks and drive the live wpm and filler highlights.

**Failure case**: unavailable outside Chrome, Edge and Safari. In the Claude desktop app's built-in
browser it fails with `not-allowed`. Like Whisper, it **drops most "um"/"uh"**, which is why the browser
analyser has its own audio-based "um" detector.

---

## 6. Interviewer + content feedback: Claude (optional)

| | |
|---|---|
| **Role** | Asks follow-up questions, rates answer content (STAR), writes the summary |
| **Implementation** | `/api/interview` in [`server.py`](../server.py), Anthropic Messages API, model `claude-sonnet-5-5`, JSON output. Without an API key, a question bank + rule-based STAR check is used |
| **Input** | JSON: role, interview type, Q&A history (≈ 100–400 words per answer) + each answer's delivery metrics |
| **Output** | JSON: `{reaction, feedback{stars 1–5, comment, tip}, next_question}` or a final `summary` |
| **Latency** | **Not measured** (no API key in this environment). It runs between questions, not inside the live loop |
| **CPU / GPU / memory** | Cloud API: no local compute. Needs network; transcripts leave the device |
| **Licence** | Commercial API, paid per token, under Anthropic's terms. Rule-based fallback: ours |

**Success example (rule-based fallback, measured)**: demo answer → *"★★☆☆☆ 106 words. STAR check:
situation ✗, action ✗, result ✗. Tip: Open with the situation in one sentence…"*. That's correct: the demo
answer is a speech about cooking, not an interview story.

**Failure case (fallback)**: the STAR check is keyword-based. It counted "a *third* of what you pay" as a
signpost and would miss a well-structured answer that never says "result" or uses a number.

---

## Can these run together in real time?

Two loops with different budgets:

**Live loop (while speaking, every 50 ms):** browser analyser (0.09 ms) + Web Speech API. Comfortable
on any laptop. To drop the Google dependency, Whisper **tiny or base** can transcribe 5 s chunks in
0.27–0.43 s (5–9% of real time); **small** in 1.2 s (24%); medium and distil-large-v3 in 3.7 s (74%, little
headroom); large-v3 cannot keep up on CPU.

**After each answer (the user waits for feedback):** for a 30 s answer on CPU:

| combination | time after answer | RAM |
|---|---:|---:|
| Whisper small + transcript metrics + openSMILE | ≈ 4.4 + 0.0 + 0.8 = **≈ 5 s** | ≈ 1.2 GB |
| Whisper distil-large-v3 + metrics + openSMILE | ≈ 5.3 s | ≈ 2.1 GB |
| Whisper large-v3 + metrics + openSMILE | ≈ 13.4 s | ≈ 3.5 GB |
| + Claude feedback | + network round trip (not measured) | cloud |

**Recommendation for the next iteration:** keep the browser live loop. After each answer, run
**Whisper small + transcript metrics + openSMILE** on a local server (≈ 5 s, ≈ 1.2 GB RAM, no GPU).
Resolve the **openSMILE licence** before any commercial use, or swap in Parselmouth/librosa.

## What needs a pretrained neural model (not covered by the cards above)

Deterministic (cards 2–4): speaking/articulation rate, pause statistics, filler-word counts from text, loudness,
F0 mean and variation, voiced ratio, jitter, shimmer, HNR, MFCCs.

Needs a learned model: the transcript itself (Whisper); telling filler "like" from verb "like"; detecting
self-corrections and restarts; robust voice-activity detection in noise (e.g. Silero VAD); perceived
confidence, nervousness or engagement (e.g. embeddings from wav2vec 2.0 / HuBERT / WavLM plus a classifier
trained on labelled interviews). None of these are measured yet.

## Limits

- **Synthetic speech.** All test clips are text-to-speech: no real hesitations, breathing or self-corrections,
  and they are cleaner than a real microphone. WER is optimistic, and voice-quality features (jitter, shimmer, HNR)
  are not meaningful. **Next step: 10–20 real recorded answers with hand-checked transcripts.**
- **Small test set:** 7 clips, ~3 minutes of audio. Differences under ~1% WER are noise.
- **One machine, CPU only.** No GPU numbers.
