#!/usr/bin/env python3
"""Profile each server-side component: I/O shapes, latency, memory, and a failure case.

    .venv/bin/python research/profile_components.py

Each measurement runs in a fresh subprocess so peak memory (max RSS) belongs to that
component alone. Writes research/results/components.json.
"""
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
CLIPS = HERE / "clips"
OUT = HERE / "results" / "components.json"
WHISPER = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]


def max_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 1e6 if sys.platform == "darwin" else r / 1e3, 1)   # bytes on macOS, KB on Linux


# ------------------------------------------------------------------ children
def child_whisper(size):
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel
    base_rss = max_rss_mb()
    t = time.perf_counter()
    m = WhisperModel(size, device="cpu", compute_type="int8")
    load = time.perf_counter() - t
    after_load = max_rss_mb()
    audio, sr = sf.read(str(CLIPS / "clean_us.wav"), dtype="float32")
    def run(x):
        t = time.perf_counter()
        segs, info = m.transcribe(x, language="en", beam_size=5, word_timestamps=True, condition_on_previous_text=False)
        words = [w for s in segs for w in (s.words or [])]
        return time.perf_counter() - t, words
    run(np.zeros(16000, dtype="float32"))                      # warm-up
    t5 = [run(audio[: 5 * sr])[0] for _ in range(3)]
    t_full, words = run(audio)
    t_greedy = time.perf_counter()
    list(m.transcribe(audio[: 5 * sr], language="en", beam_size=1, word_timestamps=False)[0])
    t_greedy = time.perf_counter() - t_greedy
    fe = m.feature_extractor
    return {
        "load_s": round(load, 2),
        "latency_5s_chunk_s": round(sorted(t5)[1], 2),            # median of 3, beam 5 + word timestamps
        "latency_5s_chunk_greedy_s": round(t_greedy, 2),           # beam 1, no word timestamps
        "latency_full_s": round(t_full, 2), "audio_s": round(len(audio) / sr, 2),
        "rss_runtime_mb": base_rss, "rss_after_load_mb": after_load, "rss_peak_mb": max_rss_mb(),
        "n_mels": int(fe.mel_filters.shape[0]), "mel_frames_per_30s": fe.nb_max_frames,
        "words_out": len(words),
        "example_words": [{"w": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2), "p": round(w.probability, 2)} for w in words[:6]],
    }


def child_opensmile(_):
    import opensmile
    import soundfile as sf
    base_rss = max_rss_mb()
    t = time.perf_counter()
    fn = opensmile.Smile(opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.Functionals)
    ll = opensmile.Smile(opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.LowLevelDescriptors)
    load = time.perf_counter() - t
    audio, sr = sf.read(str(CLIPS / "clean_us.wav"), dtype="float32")
    fn.process_signal(audio[:sr], sr)                           # warm-up
    t = time.perf_counter(); F = fn.process_signal(audio, sr); t_fn = time.perf_counter() - t
    t = time.perf_counter(); L = ll.process_signal(audio, sr); t_ll = time.perf_counter() - t
    t = time.perf_counter(); ll.process_signal(audio[: 5 * sr], sr); fn.process_signal(audio[: 5 * sr], sr); t5 = time.perf_counter() - t
    return {
        "load_s": round(load, 2), "audio_s": round(len(audio) / sr, 2),
        "latency_functionals_s": round(t_fn, 2), "latency_lld_s": round(t_ll, 2), "latency_5s_chunk_s": round(t5, 3),
        "functionals_shape": list(F.shape), "lld_shape": list(L.shape),
        "lld_names": list(L.columns),
        "rss_runtime_mb": base_rss, "rss_peak_mb": max_rss_mb(),
    }


def child_noise_failure(_):
    sys.path.insert(0, str(HERE))
    from analyze import vocal_features
    keys = ["f0_mean_hz", "f0_std_semitones", "voiced_ratio", "jitter_local", "shimmer_local_db", "hnr_db", "loudness_mean"]
    out = {}
    for clip in ["slow_au", "noisy0db_au"]:
        v = vocal_features(CLIPS / f"{clip}.wav")
        out[clip] = {k: v[k] for k in keys}
    return out


def child_metrics(_):
    sys.path.insert(0, str(HERE))
    from analyze import transcript_metrics
    words = json.loads((CLIPS / "clean_us.small.json").read_text())["asr"]["words"]
    t = time.perf_counter()
    for _ in range(1000):
        m = transcript_metrics(words, 28.34)
    per = (time.perf_counter() - t) / 1000
    # failure: "like" as a verb is still counted as a filler, and a dropped sentence is read as a pause
    fake = [{"w": w, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, w in enumerate("I would like to work on systems I like".split())]
    return {"latency_ms": round(per * 1000, 3), "input_words": len(words), "output_keys": len(m), "example": m,
            "failure_like_as_verb": {"text": "I would like to work on systems I like", "filler_count": transcript_metrics(fake, 4)["filler_count"]}}


CHILDREN = {"whisper": child_whisper, "opensmile": child_opensmile, "noise": child_noise_failure, "metrics": child_metrics}


# ------------------------------------------------------------------ parent
def spawn(kind, arg=""):
    p = subprocess.run([sys.executable, __file__, "--child", kind, arg], capture_output=True, text=True)
    if p.returncode:
        print(p.stderr[-2000:]); raise SystemExit(f"{kind} {arg} failed")
    return json.loads(p.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        print(json.dumps(CHILDREN[sys.argv[2]](sys.argv[3] if len(sys.argv) > 3 else "")))
        raise SystemExit
    res = {"machine": {"cpu": "Apple M1 Pro (8P+2E cores)", "ram_gb": 16, "os": platform.platform(),
                       "python": platform.python_version(), "gpu_used": False}}
    res["whisper"] = {}
    for m in WHISPER:
        r = spawn("whisper", m)
        res["whisper"][m] = r
        print(f"whisper {m:16s} load {r['load_s']:5.1f}s  5s-chunk {r['latency_5s_chunk_s']:5.2f}s "
              f"(greedy {r['latency_5s_chunk_greedy_s']:.2f}s)  full {r['latency_full_s']:5.1f}s  "
              f"peak RAM {r['rss_peak_mb']:7.0f} MB  mels {r['n_mels']}x{r['mel_frames_per_30s']}", flush=True)
    res["opensmile"] = spawn("opensmile"); print("opensmile", {k: v for k, v in res["opensmile"].items() if k != "lld_names"})
    res["opensmile_noise_failure"] = spawn("noise"); print("noise", res["opensmile_noise_failure"])
    res["transcript_metrics"] = spawn("metrics"); print("metrics", res["transcript_metrics"]["latency_ms"], "ms;", res["transcript_metrics"]["failure_like_as_verb"])
    OUT.write_text(json.dumps(res, indent=1))
    print("saved", OUT)
