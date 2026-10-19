#!/usr/bin/env python3
"""Benchmark Whisper model sizes on the test clips: accuracy (WER) vs. latency vs. size.

    .venv/bin/python research/benchmark.py            # all models, all clips
    .venv/bin/python research/benchmark.py --models tiny base

Writes research/results/benchmark.json and prints a summary table.
WER is computed against the reference text in <clip>.txt with the usual
normalisation (lowercase, no punctuation, numbers spelled as in the reference).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import jiwer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import transcribe, transcript_metrics, FILLERS, load_model  # noqa: E402

CLIPS = Path(__file__).parent / "clips"
OUT = Path(__file__).parent / "results"
MODELS = ["tiny", "base", "small", "medium", "large-v3", "distil-large-v3"]
PARAMS_M = {"tiny": 39, "base": 74, "small": 244, "medium": 769, "large-v3": 1550, "distil-large-v3": 756}

NUMS = {"two": "2", "five": "5", "forty": "40", "one": "1", "fifty": "50", "eight": "8", "three": "3",
        "second": "second"}


def normalise(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    toks = [NUMS.get(t, t) for t in s.split()]
    return " ".join(toks)


def wer_report(ref, hyp):
    r, h = normalise(ref), normalise(hyp)
    o = jiwer.process_words(r, h)
    return {"wer": round(o.wer, 4), "sub": o.substitutions, "del": o.deletions, "ins": o.insertions,
            "ref_words": len(r.split())}


def filler_recall(ref, hyp):
    """How many of the reference's um/uh/like/you-know made it through the ASR."""
    def count(s):
        t = normalise(s).split()
        return sum(1 for w in t if w in FILLERS) + len(re.findall(r"\byou know\b|\bi mean\b", " ".join(t)))
    return count(ref), count(hyp)


def run(models, clips):
    rows = []
    for m in models:
        load_model(m)
        print(f"\n### {m}  (load {load_model(m).load_seconds:.1f}s)", flush=True)
        for clip in clips:
            ref = clip.with_suffix(".txt").read_text()
            r = transcribe(clip, m)
            w = wer_report(ref, r["text"])
            fr, fh = filler_recall(ref, r["text"])
            tm = transcript_metrics(r["words"], r["audio_seconds"])
            row = {"model": m, "params_M": PARAMS_M.get(m), "clip": clip.stem, "audio_s": r["audio_seconds"],
                   "latency_s": r["transcribe_seconds"], "rtf": r["real_time_factor"], **w,
                   "fillers_ref": fr, "fillers_hyp": fh, "wpm": tm["speaking_rate_wpm"], "text": r["text"]}
            rows.append(row)
            print(f"  {clip.stem:14s} WER {w['wer']*100:5.1f}%  {r['transcribe_seconds']:5.1f}s  RTF {r['real_time_factor']:.2f}  "
                  f"fillers {fh}/{fr}", flush=True)
    return rows


def summarise(rows):
    out = []
    for m in dict.fromkeys(r["model"] for r in rows):
        rs = [r for r in rows if r["model"] == m]
        tot_err = sum(r["sub"] + r["del"] + r["ins"] for r in rs)
        tot_ref = sum(r["ref_words"] for r in rs)
        clean = [r for r in rs if not r["clip"].startswith("noisy")]
        noisy = [r for r in rs if r["clip"] == "noisy_us"]
        hard = [r for r in rs if r["clip"].startswith("noisy0db")]
        group_wer = lambda g: round(sum(r["sub"] + r["del"] + r["ins"] for r in g) / sum(r["ref_words"] for r in g), 4) if g else None
        out.append({
            "model": m, "params_M": rs[0]["params_M"],
            "wer_all": round(tot_err / tot_ref, 4),
            "wer_clean": round(sum(r["sub"] + r["del"] + r["ins"] for r in clean) / sum(r["ref_words"] for r in clean), 4),
            "wer_noisy": round(noisy[0]["wer"], 4) if noisy else None,   # 10 dB SNR
            "wer_noisy0db": group_wer(hard),                              # 0 dB SNR
            "rtf_mean": round(float(np.mean([r["rtf"] for r in clean])), 3),
            "latency_per_30s": round(float(np.mean([r["rtf"] for r in clean])) * 30, 1),
            "filler_recall": round(sum(r["fillers_hyp"] for r in rs) / max(1, sum(r["fillers_ref"] for r in rs)), 2),
        })
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--clips", nargs="+", default=None)
    ap.add_argument("--append", action="store_true", help="merge into the existing benchmark.json")
    a = ap.parse_args()
    clips = [CLIPS / f"{c}.wav" for c in a.clips] if a.clips else sorted(CLIPS.glob("*.wav"))
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    rows = run(a.models, clips)
    if a.append and (OUT / "benchmark.json").exists():
        old = json.loads((OUT / "benchmark.json").read_text())["rows"]
        new_keys = {(r["model"], r["clip"]) for r in rows}
        rows = [r for r in old if (r["model"], r["clip"]) not in new_keys] + rows
    summary = summarise(rows)
    (OUT / "benchmark.json").write_text(json.dumps({"rows": rows, "summary": summary,
                                                    "machine": "Apple M1 Pro, 16 GB, CPU int8 via faster-whisper/CTranslate2"}, indent=1))
    print(f"\n{'model':16s} {'params':>7s} {'WER clean':>10s} {'10dB':>7s} {'0dB':>7s} {'RTF':>6s} {'s/30s':>6s} {'fillers':>8s}")
    fmt = lambda v: f"{v*100:6.1f}%" if v is not None else "     —"
    for s in summary:
        print(f"{s['model']:16s} {s['params_M']:>6}M {s['wer_clean']*100:9.1f}% {fmt(s['wer_noisy'])} {fmt(s['wer_noisy0db'])} "
              f"{s['rtf_mean']:6.2f} {s['latency_per_30s']:6.1f} {s['filler_recall']*100:7.0f}%")
    print(f"\ntotal {time.time()-t0:.0f}s; saved {OUT/'benchmark.json'}")
