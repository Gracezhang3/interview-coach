# Interview Coach (prototype)

A live mock interview. The interviewer asks a question out loud and you answer. While you talk,
a coach panel tracks your delivery in real time and nudges you. After each answer you get feedback,
and at the end a summary.

## Run

```bash
python3 server.py
```

Open http://localhost:8000 in **Chrome**, which you need for live transcription. Headphones help.
To use Claude as the interviewer, which gives tailored follow-up questions and content feedback:

```bash
ANTHROPIC_API_KEY=sk-ant-... python3 server.py
```

If you don't set a key, the server uses a built-in question bank and rule-based STAR feedback.
You can override the model with `CLAUDE_MODEL=...`.

## Realtime coaching (in the browser, updated every 50 ms)

| Gauge | How | Nudge |
|---|---|---|
| Answer time | from question start; scored from first word | over 2 min: "wrap up" |
| Pace | rolling 20 s words/min from the live transcript | over 175: "slow down", under 100: "pick up the pace" |
| Filler words | regex on the live transcript (um, like, you know, basically…) | each new filler |
| "Um" sounds | voiced sound ≥ 0.45 s with flat pitch (from audio, since Chrome drops "um") | each hesitation |
| Volume | rolling 3 s level vs. your own baseline; phrase-end fade detection | "speak up", "you trailed off" |
| Vocal variety | rolling 10 s pitch spread in semitones (autocorrelation) | under 1.8 st: "monotone" |
| Pauses | adaptive silence threshold; gaps ≥ 0.4 s | good pauses praised; over 3 s: "keep going" |

With auto-advance on, 4 seconds of silence ends the answer. Space also ends it.

**▶ Demo answer** plays a recorded sample through the same live pipeline, so you can try it without a mic.

## Files

- `server.py`: interviewer API (Claude, or the fallback question bank) plus static file server. Standard library only.
- `static/app.js`: audio analysis, live detectors, nudges, interview flow
- `static/index.html`, `static/style.css`
- `static/samples/`: text-to-speech sample answer for the demo

## Known limits

- The mic is blocked in the Claude app's built-in browser, so use real Chrome.
- The "um" detector and all score thresholds are rough guesses and haven't been calibrated against real speakers.
- Without headphones, the mic can pick up the interviewer's voice. Analysis pauses while the interviewer speaks.
