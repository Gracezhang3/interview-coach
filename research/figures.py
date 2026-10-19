#!/usr/bin/env python3
"""Charts for the slides, from research/results/*.json.

    .venv/bin/python research/figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

R = Path(__file__).parent / "results"
M = Path(__file__).parent / "media"
INK, MUTED, GRID, BG = "#1d2433", "#5b6475", "#e4e7ee", "#fbfaf7"
BLUE, ORANGE = "#2f6fdb", "#e0782f"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})


def walkthrough(clip_json, wav, out):
    d = json.loads(Path(clip_json).read_text())
    words, s = d["asr"]["words"], d["vocal"]["series"]
    audio, sr = sf.read(wav)
    t = np.arange(len(audio)) / sr
    T = min(9.2, t[-1])                      # first ~9 s keeps the words readable
    fig, ax = plt.subplots(3, 1, figsize=(16, 7.6), sharex=True, gridspec_kw={"height_ratios": [1.25, 1, 1]})
    # 1. waveform + words + pauses
    a = ax[0]
    k = t <= T
    a.plot(t[k], audio[k], color="#9aa6bd", lw=0.4)
    for i in range(1, len(words)):
        g0, g1 = words[i - 1]["end"], words[i]["start"]
        if g1 - g0 >= 0.3 and g1 <= T:
            a.axvspan(g0, g1, color=ORANGE, alpha=0.18, lw=0)
    for w in words:
        if w["start"] > T: break
        filler = w["w"].lower().strip(",.") in ("um", "uh", "like")
        a.text((w["start"] + w["end"]) / 2, (0.95, 0.80)[words.index(w) % 2], w["w"].strip(",."),
               transform=a.get_xaxis_transform(), ha="center", fontsize=14,
               color=ORANGE if filler else INK, fontweight="bold" if filler else "normal")
    a.set_yticks([]); a.set_ylabel("audio +\nwords")
    # 2. F0
    a = ax[1]
    tt = np.array(s["t"]); f0 = np.array([v if v else np.nan for v in s["f0_hz"]], float)
    k = tt <= T
    a.plot(tt[k], f0[k], ".", ms=3, color=BLUE)
    a.set_ylabel("pitch F0\n(Hz)"); a.set_ylim(80, 320); a.grid(axis="y", color=GRID)
    # 3. loudness
    a = ax[2]
    lo = np.array(s["loudness"])
    a.fill_between(tt[k], lo[k], color=BLUE, alpha=0.35, lw=0)
    a.set_ylabel("loudness\n(sones)"); a.set_xlabel("time (s)"); a.grid(axis="y", color=GRID)
    a.set_xlim(0, T)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def tradeoff(bench_json, out):
    s = json.loads(Path(bench_json).read_text())["summary"]
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in s:
        x, y = r["latency_per_30s"], r["wer_all"] * 100
        size = 120 + r["params_M"] * 0.9
        ax.scatter(x, y, s=size, color=BLUE, alpha=0.75, edgecolor="white", lw=2, zorder=3)
        off = {"medium": (-8, 30), "large-v3": (22, -6)}.get(r["model"], (14, 6))
        ax.annotate(f"{r['model']}\n{r['params_M']}M", (x, y), xytext=off, textcoords="offset points",
                    fontsize=14, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 30]); ax.set_xticklabels(["1", "2", "5", "10", "20", "30"])
    ax.set_xlabel("seconds to transcribe a 30 s answer (CPU, log scale)")
    ax.set_ylabel("word error rate, all 7 clips (%)")
    ax.axvline(30, color=ORANGE, ls="--", lw=1.5)
    ax.text(30, ax.get_ylim()[1] * 0.96, " slower than\n real time", color=ORANGE, fontsize=13, va="top")
    ax.grid(color=GRID, zorder=0)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    M.mkdir(exist_ok=True)
    clips = Path(__file__).parent / "clips"
    walkthrough(clips / "clean_us.small.json", clips / "clean_us.wav", M / "walkthrough.png")
    if (R / "benchmark.json").exists():
        tradeoff(R / "benchmark.json", M / "tradeoff.png")
    print("wrote", *sorted(p.name for p in M.glob("*.png")))
