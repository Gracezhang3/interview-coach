#!/usr/bin/env python3
"""Offline analysis pipeline: Whisper ASR with timestamps + openSMILE vocal features.

    .venv/bin/python research/analyze.py research/clips/clean_us.wav --model base

Prints a word-timestamped transcript, deterministic speech metrics derived from it,
and the openSMILE eGeMAPS summary (pitch, loudness, jitter, shimmer, voicing, MFCC).
Writes <clip>.<model>.json next to the clip unless --no-save.
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import opensmile
import soundfile as sf
from faster_whisper import WhisperModel

FILLERS = {"um", "uh", "uhm", "erm", "ah", "hmm", "mm", "like", "basically", "actually", "literally"}
FILLER_PHRASES = ["you know", "i mean", "kind of", "sort of"]
# Whisper only emits "um"/"uh" if asked for them via the initial prompt.
DISFLUENT_PROMPT = "Um, uh, so, like, you know, I mean... hmm. Okay, so."


# --------------------------------------------------------------------- ASR
_models = {}


def load_model(size, device="cpu", compute_type="int8"):
    key = (size, device, compute_type)
    if key not in _models:
        t = time.perf_counter()
        _models[key] = WhisperModel(size, device=device, compute_type=compute_type)
        _models[key].load_seconds = time.perf_counter() - t
    return _models[key]


def transcribe(path, size="base", keep_fillers=True, beam_size=5):
    model = load_model(size)
    audio, sr = sf.read(str(path), dtype="float32")        # decode ourselves: avoids the PyAV dependency
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
    t = time.perf_counter()
    segments, info = model.transcribe(
        audio, language="en", beam_size=beam_size, word_timestamps=True,
        vad_filter=False, initial_prompt=DISFLUENT_PROMPT if keep_fillers else None,
        condition_on_previous_text=False,
    )
    words = [{"w": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2), "p": round(w.probability, 2)}
             for s in segments for w in (s.words or [])]
    secs = time.perf_counter() - t
    return {
        "model": size, "words": words,
        "text": " ".join(w["w"] for w in words),
        "audio_seconds": round(info.duration, 2),
        "transcribe_seconds": round(secs, 2),
        "real_time_factor": round(secs / info.duration, 3) if info.duration else None,
        "model_load_seconds": round(model.load_seconds, 2),
    }


# --------------------------------------------------------------------- transcript metrics
def norm(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def transcript_metrics(words, audio_seconds):
    toks = [t for w in words for t in norm(w["w"])]
    n = len(toks)
    text = " ".join(toks)
    fillers = sum(1 for t in toks if t in FILLERS) + sum(len(re.findall(rf"\b{p}\b", text)) for p in FILLER_PHRASES)
    # repetition / self-correction: an immediately repeated word or bigram ("I I", "we we were")
    reps = sum(1 for i in range(1, n) if toks[i] == toks[i - 1])
    reps += sum(1 for i in range(2, n - 1) if toks[i:i + 2] == toks[i - 2:i])
    # pauses from word timestamps
    gaps = [(words[i]["start"] - words[i - 1]["end"]) for i in range(1, len(words))]
    pauses = [g for g in gaps if g >= 0.3]
    speaking = sum(w["end"] - w["start"] for w in words) + sum(g for g in gaps if 0 < g < 0.3)
    minutes = audio_seconds / 60 if audio_seconds else 1
    return {
        "words": n,
        "speaking_rate_wpm": round(n / minutes, 1),                    # over total time, incl. pauses
        "articulation_rate_wpm": round(n / (speaking / 60), 1) if speaking else None,  # over speech only
        "filler_count": fillers, "fillers_per_min": round(fillers / minutes, 1),
        "repetitions": reps,
        "pause_count": len(pauses),
        "pause_total_seconds": round(sum(pauses), 2),
        "pause_mean_seconds": round(statistics.mean(pauses), 2) if pauses else 0,
        "pause_max_seconds": round(max(pauses), 2) if pauses else 0,
        "pause_ratio": round(sum(pauses) / audio_seconds, 3) if audio_seconds else None,
    }


# --------------------------------------------------------------------- openSMILE
_smile = {}


def smile(level):
    if level not in _smile:
        _smile[level] = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=level)
    return _smile[level]


def vocal_features(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    t = time.perf_counter()
    # functionals: one row of 88 clip-level statistics
    func = smile(opensmile.FeatureLevel.Functionals).process_signal(audio, sr).iloc[0]
    # LLDs: 25 descriptors every 10 ms, used for the time series + voiced ratio
    lld = smile(opensmile.FeatureLevel.LowLevelDescriptors).process_signal(audio, sr)
    secs = time.perf_counter() - t
    f0 = lld["F0semitoneFrom27.5Hz_sma3nz"].to_numpy()
    voiced = f0 > 0
    to_hz = lambda st: 27.5 * 2 ** (st / 12)
    return {
        "opensmile_seconds": round(secs, 2),
        "feature_set": "eGeMAPSv02 (88 functionals, 25 LLDs @ 10 ms)",
        "f0_mean_hz": round(float(to_hz(func["F0semitoneFrom27.5Hz_sma3nz_amean"])), 1),
        "f0_std_semitones": round(float(func["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"] * func["F0semitoneFrom27.5Hz_sma3nz_amean"]), 2),
        "f0_range_semitones": round(float(func["F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2"]), 2),
        "loudness_mean": round(float(func["loudness_sma3_amean"]), 3),
        "loudness_std_norm": round(float(func["loudness_sma3_stddevNorm"]), 3),
        "loudness_range": round(float(func["loudness_sma3_pctlrange0-2"]), 3),
        "jitter_local": round(float(func["jitterLocal_sma3nz_amean"]), 4),
        "shimmer_local_db": round(float(func["shimmerLocaldB_sma3nz_amean"]), 3),
        "hnr_db": round(float(func["HNRdBACF_sma3nz_amean"]), 2),
        "voiced_ratio": round(float(voiced.mean()), 3),
        "voiced_segments_per_sec": round(float(func["VoicedSegmentsPerSec"]), 2),
        "mean_voiced_segment_ms": round(float(func["MeanVoicedSegmentLengthSec"]) * 1000, 0),
        "mean_unvoiced_segment_ms": round(float(func["MeanUnvoicedSegmentLength"]) * 1000, 0),
        "mfcc1_4_mean": [round(float(func[f"mfcc{i}_sma3_amean"]), 2) for i in range(1, 5)],
        "series": {   # 10 ms frames, for plotting
            "t": [round(float(x), 2) for x in lld.index.get_level_values("start").total_seconds()],
            "f0_hz": [round(float(to_hz(v)), 1) if v > 0 else None for v in f0],
            "loudness": [round(float(v), 3) for v in lld["Loudness_sma3"]],
        },
    }


# --------------------------------------------------------------------- main
def analyze(path, model="base", keep_fillers=True):
    asr = transcribe(path, model, keep_fillers)
    return {
        "clip": Path(path).name,
        "asr": asr,
        "transcript_metrics": transcript_metrics(asr["words"], asr["audio_seconds"]),
        "vocal": vocal_features(path),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--model", default="base", help="tiny | base | small | medium | large-v3 | distil-large-v3")
    ap.add_argument("--no-fillers", action="store_true", help="don't prompt Whisper to keep um/uh")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    r = analyze(a.clip, a.model, not a.no_fillers)
    print(f"\n== {r['clip']}  ({a.model}: {r['asr']['transcribe_seconds']}s for {r['asr']['audio_seconds']}s audio, "
          f"RTF {r['asr']['real_time_factor']})\n")
    for w in r["asr"]["words"]:
        print(f"  {w['start']:6.2f}–{w['end']:6.2f}  {w['w']}")
    print("\n-- transcript metrics");  print(json.dumps(r["transcript_metrics"], indent=2))
    v = {k: x for k, x in r["vocal"].items() if k != "series"}
    print("\n-- openSMILE eGeMAPS");   print(json.dumps(v, indent=2))
    if not a.no_save:
        out = Path(a.clip).with_suffix(f".{a.model}.json")
        out.write_text(json.dumps(r, indent=1))
        print(f"\nsaved {out}")
