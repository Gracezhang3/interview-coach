#!/usr/bin/env python3
"""Does Whisper keep "um"/"uh"? Compare transcription with and without a disfluent initial prompt.

    .venv/bin/python research/filler_ablation.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import transcribe  # noqa: E402
from benchmark import filler_recall, wer_report  # noqa: E402

CLIPS = Path(__file__).parent / "clips"
rows = []
for model in ["base", "small"]:
    for clip in ["clean_us", "fast_uk", "slow_au"]:
        ref = (CLIPS / f"{clip}.txt").read_text()
        for prompted in (False, True):
            r = transcribe(CLIPS / f"{clip}.wav", model, keep_fillers=prompted)
            fr, fh = filler_recall(ref, r["text"])
            rows.append({"model": model, "clip": clip, "prompted": prompted, "fillers_ref": fr, "fillers_hyp": fh,
                         "wer": wer_report(ref, r["text"])["wer"], "text": r["text"]})
            print(f"{model:6s} {clip:10s} prompt={'yes' if prompted else 'no '}  fillers {fh}/{fr}  WER {rows[-1]['wer']*100:.1f}%")

summary = {}
for p in (False, True):
    rs = [r for r in rows if r["prompted"] == p]
    summary["prompted" if p else "default"] = round(sum(r["fillers_hyp"] for r in rs) / sum(r["fillers_ref"] for r in rs), 2)
print("filler recall:", summary)
out = Path(__file__).parent / "results" / "filler_ablation.json"
out.write_text(json.dumps({"rows": rows, "recall": summary}, indent=1))
print("saved", out)
