// Interview Coach — live mock interview with realtime delivery coaching.
// Mic -> AnalyserNode -> 50ms frames {t, db, pitch} -> live detectors + nudges.
// Speech recognition -> live transcript -> pace + filler detection.
// Each finished answer -> full metrics -> server (/api/interview) -> feedback + next question.

const $ = (id) => document.getElementById(id);
const HOP = 0.05;          // seconds between analysis frames
const WIN = 2048;          // samples per analysis window
const ANSWER_LO = 45, ANSWER_HI = 120;   // target answer length (s)

// ================================================================ DSP
function rmsDb(buf) {
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.max(-100, 10 * Math.log10(s / buf.length + 1e-12));
}

// Normalised autocorrelation pitch detector on a decimated (~16kHz) signal, 70–400 Hz.
function detectPitch(buf, sr) {
  const step = Math.max(1, Math.round(sr / 16000));
  const x = [];
  for (let i = 0; i < buf.length; i += step) x.push(buf[i]);
  const fs = sr / step, minLag = Math.floor(fs / 400), maxLag = Math.ceil(fs / 70);
  const n = x.length - maxLag;
  if (n < 64) return null;
  let bestR = 0, bestLag = 0;
  for (let lag = minLag; lag <= maxLag; lag++) {
    let c = 0, e0 = 0, e1 = 0;
    for (let i = 0; i < n; i++) { c += x[i] * x[i + lag]; e0 += x[i] * x[i]; e1 += x[i + lag] * x[i + lag]; }
    const r = c / Math.sqrt(e0 * e1 + 1e-12);
    if (r > bestR) { bestR = r; bestLag = lag; }
  }
  return bestR > 0.6 ? fs / bestLag : null;
}

function frame(buf, sr, t) {
  const db = rmsDb(buf);
  return { t, db, pitch: db > -60 ? detectPitch(buf, sr) : null };
}

// ================================================================ helpers
const pct = (arr, p) => { if (!arr.length) return 0; const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);
const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };
const clamp = (v, lo = 0, hi = 100) => Math.round(Math.max(lo, Math.min(hi, v)));
const semis = (ps) => { const m = pct(ps, 0.5); return ps.map((p) => 12 * Math.log2(p / m)).filter((s) => Math.abs(s) < 12); };
const countWords = (t) => (t.match(/[a-zA-Z0-9']+/g) || []).length;
const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n);

// ================================================================ filler words
const FILLERS = ["um+", "uh+", "uhm", "erm?", "ah+", "hmm+", "you know", "i mean", "basically", "actually", "literally", "kind of", "sort of", "like"];
// "like" only counts when it isn't a verb/comparison ("I like", "looks like", ...)
const LIKE_OK = /\b(i|you|we|they|would|feel|feels|felt|look|looks|looked|seem|seems|sounds|just|really|much|more|something|anything)\s+like$/i;

function findFillers(text) {
  const hits = [];
  const re = new RegExp(`\\b(${FILLERS.join("|")})\\b`, "gi");
  let m;
  while ((m = re.exec(text))) {
    const w = m[1].toLowerCase();
    if (w === "like" && LIKE_OK.test(text.slice(0, m.index + 4))) continue;
    if (w === "you know" && /^\s+(exactly|what|how|that|why|where|when|who|it|the)\b/i.test(text.slice(m.index + 8))) continue;
    hits.push({ word: w, index: m.index, len: m[1].length });
  }
  return hits;
}

function highlightFillers(text) {
  let out = "", last = 0;
  findFillers(text).forEach((f) => { out += esc(text.slice(last, f.index)) + `<mark>${esc(text.slice(f.index, f.index + f.len))}</mark>`; last = f.index + f.len; });
  return out + esc(text.slice(last));
}

// ================================================================ per-answer scoring
function analyze(frames, transcript, duration, hesitations = 0) {
  const dbs = frames.map((f) => f.db);
  const floor = pct(dbs, 0.1), peak = pct(dbs, 0.95);
  const thr = Math.max(floor + 6, floor + 0.3 * (peak - floor), peak - 35);
  const voiced = frames.map((f) => f.db > thr);

  const segs = [];
  let cur = null;
  frames.forEach((f, i) => {
    if (!voiced[i]) return;
    if (cur && f.t - cur.end <= 0.25) cur.end = f.t + HOP;
    else { cur = { start: f.t, end: f.t + HOP }; segs.push(cur); }
  });
  const pauses = [];
  for (let i = 1; i < segs.length; i++) {
    const gap = segs[i].start - segs[i - 1].end;
    if (gap >= 0.4 && gap < 4) pauses.push(gap);   // ignore the end-of-answer silence
  }
  const minutes = Math.max(duration, 1) / 60;

  const words = countWords(transcript);
  const wpm = words ? Math.round(words / minutes) : null;
  const fillers = findFillers(transcript);
  const fpm = (fillers.length + hesitations) / minutes;

  const segDb = (from, to) => mean(frames.filter((f, i) => voiced[i] && f.t >= from && f.t < to).map((f) => f.db));
  const longSegs = segs.filter((g) => g.end - g.start >= 1);
  let fades = 0;
  longSegs.forEach((g) => { const cut = g.start + 0.75 * (g.end - g.start); if (segDb(g.start, cut) - segDb(cut, g.end) > 6) fades++; });
  const fadeRate = longSegs.length ? fades / longSegs.length : 0;
  const loudStd = segs.length > 1 ? std(segs.map((g) => segDb(g.start, g.end)).filter((v) => !isNaN(v))) : 0;

  const pitches = frames.filter((f, i) => voiced[i] && f.pitch).map((f) => f.pitch);
  const st = semis(pitches);
  const pitchStd = st.length >= 20 ? std(st) : null;

  const longPauses = pauses.filter((p) => p > 3).length;
  const ppm = pauses.length / minutes;
  const dOff = duration < ANSWER_LO ? (ANSWER_LO - duration) / ANSWER_LO : duration > ANSWER_HI ? (duration - ANSWER_HI) / ANSWER_HI : 0;

  const scores = {
    duration: clamp(100 - dOff * 150, 20),
    pace: wpm == null ? null : clamp(wpm >= 130 && wpm <= 160 ? 100 : 100 - Math.abs(wpm - (wpm < 130 ? 130 : 160)) * 1.2, 20),
    fillers: words ? clamp(100 - fpm * 12, 20) : null,
    volume: segs.length ? clamp(100 - fadeRate * 60 - Math.max(0, loudStd - 4) * 5, 20) : null,
    variety: pitchStd == null ? null : clamp(40 + (pitchStd - 1) * 20, 20),
    pausing: segs.length ? clamp((ppm >= 4 && ppm <= 14 ? 100 : ppm < 4 ? 100 - (4 - ppm) * 12 : 100 - (ppm - 14) * 5) - longPauses * 8, 20) : null,
  };
  const weights = { duration: 0.5, pace: 1, fillers: 1, volume: 1, variety: 1, pausing: 1 };
  let ws = 0, wt = 0;
  for (const k in scores) if (scores[k] != null) { ws += scores[k] * weights[k]; wt += weights[k]; }

  return {
    overall: wt ? Math.round(ws / wt) : 0, scores,
    durationSec: +duration.toFixed(1), words, wpm,
    fillerCount: fillers.length, hesitations, fillersPerMin: +fpm.toFixed(1),
    pauseCount: pauses.length, pausesPerMin: +ppm.toFixed(1), longPauses,
    fadeRate: +fadeRate.toFixed(2), loudnessStdDb: +loudStd.toFixed(1),
    medianPitchHz: pitches.length ? Math.round(pct(pitches, 0.5)) : null,
    pitchStdSemitones: pitchStd == null ? null : +pitchStd.toFixed(2),
  };
}

// ================================================================ session
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let S = null;            // the running interview
const dbHistory = [];    // rolling ~20s of frame levels (excluding interviewer speech) -> adaptive silence threshold

function voicingThreshold() {
  const floor = pct(dbHistory, 0.1), peak = pct(dbHistory, 0.95);
  return Math.max(floor + 6, floor + 0.3 * (peak - floor), peak - 35);
}

async function startInterview() {
  S = {
    role: $("role").value.trim(), type: $("itype").value, numQ: +$("numQ").value,
    voice: $("voice").checked, auto: $("autoAdvance").checked,
    history: [], state: "idle", ended: false,
  };
  S.ac = new AudioContext();
  S.an = S.ac.createAnalyser(); S.an.fftSize = WIN;
  S.buf = new Float32Array(WIN);
  $("warn").hidden = true;
  try {
    S.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: false } });
    S.ac.createMediaStreamSource(S.stream).connect(S.an);
  } catch (e) {
    showWarn(`Microphone unavailable (${e.message}). You can still try “▶ Demo answer”.`);
  }
  $("setup").hidden = true; $("summary").hidden = true; $("interview").hidden = false;
  $("history").innerHTML = ""; $("feed").innerHTML = "";
  S.timer = setInterval(tick, HOP * 1000);
  nextTurn();
}

function setState(state) {
  S.state = state;
  const pill = $("state");
  pill.className = "pill" + (state === "answering" ? " live" : "");
  pill.textContent = { thinking: "Interviewer is thinking…", speaking: "Interviewer speaking…", answering: "Live: your turn", idle: "" }[state] || "";
  $("doneBtn").disabled = state === "thinking";
  $("skipBtn").disabled = state !== "answering";
  $("demoBtn").disabled = state === "thinking";
  $("doneBtn").innerHTML = state === "speaking" ? "Start answering now <kbd>Space</kbd>" : "Done answering <kbd>Space</kbd>";
}

async function nextTurn() {
  setState("thinking");
  const final = S.ended || S.history.length >= S.numQ;
  let r;
  try {
    const res = await fetch("/api/interview", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({
        role: S.role, type: S.type, total: S.numQ, final,
        history: S.history.map(({ q, a, metrics, feedback }) => ({ q, a, metrics, feedback })),
      }),
    });
    r = await res.json();
  } catch (e) {
    showWarn(`Couldn't reach the local server: ${e}`);
    return;
  }
  if (!S) return;
  if (r.warning) showWarn(r.warning);
  const last = S.history.at(-1);
  if (last && r.feedback) { last.feedback = r.feedback; renderFeedback(last); }
  if (final) return finish(r);
  if (S.ended) return nextTurn();             // "End interview" was pressed while we were waiting
  askQuestion(r.next_question, r.reaction);
}

// ---------------------------------------------------------------- interviewer voice
let speakDone = null;
function speak(text) {
  return new Promise((resolve) => {
    if (!S.voice || !window.speechSynthesis) return resolve();
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US"; u.rate = 1.02;
    const voice = speechSynthesis.getVoices().find((v) => /en-US/.test(v.lang) && /Samantha|Google US|Natural/i.test(v.name));
    if (voice) u.voice = voice;
    let done = false;
    speakDone = () => { if (!done) { done = true; speakDone = null; resolve(); } };
    u.onend = u.onerror = speakDone;
    setTimeout(() => speakDone?.(), (countWords(text) / 2.6 + 2) * 1000);   // Chrome sometimes never fires onend
    speechSynthesis.speak(u);
  });
}

async function askQuestion(q, reaction) {
  S.q = q;
  $("qnum").textContent = `Question ${S.history.length + 1} of ${S.numQ}`;
  $("question").textContent = q;
  $("answerBox").innerHTML = `<span class="muted">${S.voice ? "Listen to the question…" : "Get ready…"}</span>`;
  resetGauges();
  setState("speaking");
  await speak([reaction, q].filter(Boolean).join(" "));
  if (S?.state === "speaking" && S.q === q) startAnswer();
}

function resetGauges() {
  ["gPace", "gVol", "gPitch", "nPause"].forEach((id) => ($(id).textContent = "—"));
  $("gTime").textContent = "0:00"; $("nTime").textContent = "aim for 45s – 2 min";
  $("nPace").textContent = "130–160 wpm is ideal";
  $("zTime").style.left = $("zPace").style.left = "0";
  $("gFill").textContent = "0"; $("nFill").textContent = "none yet";
  $("gPause").textContent = "0";
  $("feed").innerHTML = "";
}

function skipSpeaking() {
  speechSynthesis.cancel();
  speakDone?.();
}

// ---------------------------------------------------------------- answering
function startAnswer() {
  S.live = {
    t0: S.ac.currentTime, frames: [], text: "", interim: "", words: 0, wordsAt: [],
    firstVoice: null, silence: null, seg: [], run: [], runFlagged: false, longSilenceNudged: false,
    baseline: null, voicedDbs: [],
    fillerCount: 0, hesitations: 0, pauses: 0, pauseLens: [], longPauses: 0, fades: 0,
    nudgeAt: {}, nudges: 0, demoUsed: false,
  };
  setState("answering");
  $("answerBox").innerHTML = `<span class="muted">Listening… start whenever you're ready.</span>`;
  startRecognition();
}

function setAnswerText(finalText, interim = "") {
  const L = S.live;
  L.text = finalText; L.interim = interim;
  const all = (finalText + " " + interim).trim();
  L.words = countWords(all);
  $("answerBox").innerHTML = highlightFillers(finalText) + (interim ? ` <span class="interim">${esc(interim)}</span>` : "") || `<span class="muted">Listening…</span>`;
  $("answerBox").scrollTop = 1e6;
  const hits = findFillers(all);
  if (hits.length > L.fillerCount) {
    const w = hits.at(-1).word;
    nudge("filler", `Filler word: “${w}”. Try a silent pause instead.`, "warn", 4);
    $("fillGauge").classList.remove("flash"); void $("fillGauge").offsetWidth; $("fillGauge").classList.add("flash");
  }
  L.fillerCount = Math.max(L.fillerCount, hits.length);
  renderFillerGauge(hits);
}

function renderFillerGauge(hits) {
  const L = S.live;
  const counts = {};
  (hits || findFillers(L.text)).forEach((h) => (counts[h.word] = (counts[h.word] || 0) + 1));
  $("gFill").textContent = L.fillerCount + L.hesitations;
  const parts = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([w, c]) => `“${w}” ×${c}`);
  if (L.hesitations) parts.push(`~${L.hesitations} “um” sound${L.hesitations > 1 ? "s" : ""} (from audio)`);
  $("nFill").textContent = parts.join(", ") || "none yet";
}

async function finishAnswer(skipped = false) {
  if (S?.state !== "answering") return;
  setState("thinking");
  stopRecognition();
  stopDemo();
  await sleep(skipped ? 0 : 700);            // let the last recognition results arrive
  const L = S.live;
  // score only the speech: from the first voiced frame to the start of the final silence
  const start = Math.max(0, (L.firstVoice ?? 0) - 0.2);
  const end = L.silence != null ? L.silence : S.ac.currentTime - L.t0;
  const frames = L.frames.filter((f) => f.t >= start && f.t <= end).map((f) => ({ ...f, t: f.t - start }));
  const text = (L.text + " " + L.interim).trim();
  const spoke = text || L.voicedDbs.length >= 20;      // at least ~1s of voice
  const entry = { q: S.q, a: spoke ? text : skipped ? "(skipped)" : "(no answer)", live: { nudges: L.nudges } };
  if (spoke) entry.metrics = { ...analyze(frames, text, Math.max(1, end - start), L.hesitations), thinkingTimeSec: +start.toFixed(1) };
  S.history.push(entry);
  renderHistoryEntry(entry);
  nextTurn();
}

// ---------------------------------------------------------------- realtime loop
function tick() {
  if (!S) return;
  S.an.getFloatTimeDomainData(S.buf);
  const f = frame(S.buf, S.ac.sampleRate, 0);
  if (S.state !== "speaking") { dbHistory.push(f.db); if (dbHistory.length > 400) dbHistory.shift(); }
  const thr = voicingThreshold();
  const voiced = f.db > thr;
  drawVolume(f.db, thr);
  if (S.state !== "answering") return;

  const L = S.live;
  const t = (f.t = S.ac.currentTime - L.t0);
  L.frames.push(f);
  if (S.demo) updateDemoText();
  L.wordsAt.push({ t, w: L.words });

  if (voiced) {
    if (L.silence != null) {                  // a pause just ended
      const gap = t - L.silence;
      if (gap >= 0.4) {
        L.pauses++; L.pauseLens.push(gap);
        if (gap > 3) L.longPauses++;
        if (gap >= 0.8 && gap <= 2.5 && L.words >= 10) nudge("goodpause", "Nice pause. It lets your point land.", "good", 25);
      }
      L.silence = null; L.longSilenceNudged = false;
    }
    if (L.firstVoice == null) L.firstVoice = t;
    L.seg.push(f.db);
    L.voicedDbs.push(f.db);
    if (!L.baseline && L.voicedDbs.length >= 60) L.baseline = pct(L.voicedDbs, 0.5);
    trackHesitation(f.pitch);
  } else {
    if (L.silence == null) L.silence = t;
    const silent = t - L.silence;
    if (silent >= 0.3 && L.seg.length) endSegment();
    if (silent >= 0.3) { L.run = []; L.runFlagged = false; }
    if (silent > 3 && L.words > 0 && !L.longSilenceNudged) {
      L.longSilenceNudged = true;
      nudge("silence", S.auto ? "Long pause. Keep going, or stay quiet to move on." : "Long pause. Keep going, or press Space if you're done.", "warn", 0);
    }
    if (S.auto && silent >= 4 && L.words >= 15 && !S.demo) return finishAnswer();
  }

  updateLiveGauges(t, voiced);
}

function endSegment() {
  const L = S.live, seg = L.seg;
  L.seg = [];
  if (seg.length < 20) return;               // only judge phrases >= 1s
  const cut = Math.floor(seg.length * 0.75);
  if (mean(seg.slice(0, cut)) - mean(seg.slice(cut)) > 6) {
    L.fades++;
    nudge("fade", "You trailed off. Keep your volume up to the last word.", "warn", 12);
  }
}

// A sustained voiced sound with almost flat pitch (≥0.45s) is very likely "uhhh"/"ummm".
function trackHesitation(pitch) {
  const L = S.live;
  L.run.push(pitch);
  if (L.run.length > 9) L.run.shift();
  if (L.run.length < 9) return;
  const ps = L.run.filter(Boolean);
  const spread = ps.length >= 8 ? std(ps.map((p) => 12 * Math.log2(p / ps[0]))) : 99;
  if (spread < 0.5 && !L.runFlagged) {
    L.runFlagged = true; L.hesitations++;
    nudge("hes", "That sounded like an “um”. A silent pause sounds more confident.", "warn", 6);
    renderFillerGauge();
  } else if (spread > 1.5) L.runFlagged = false;
}

function updateLiveGauges(t, voiced) {
  const L = S.live;
  // time
  $("gTime").textContent = fmt(t);
  $("zTime").style.left = `calc(${Math.min(100, (t / 150) * 100)}% - 2px)`;
  $("nTime").textContent = t < ANSWER_LO ? "aim for 45s – 2 min" : t <= ANSWER_HI ? "good length" : "running long, so wrap up";
  if (t > ANSWER_HI) nudge("long", "Over 2 minutes. Land your result and wrap up.", "warn", 30);

  // pace (rolling 20s)
  const spoken = L.firstVoice != null ? t - L.firstVoice : 0;
  if (spoken >= 6 && L.words > 0) {
    const start = L.wordsAt.find((x) => x.t >= Math.max(t - 20, L.firstVoice));
    const span = t - start.t;
    const wpm = span > 4 ? Math.round(((L.words - start.w) / span) * 60) : null;
    if (wpm != null) {
      $("gPace").textContent = `${wpm} wpm`;
      $("zPace").style.left = `calc(${Math.max(0, Math.min(100, ((wpm - 60) / 160) * 100))}% - 2px)`;
      $("nPace").textContent = wpm > 160 ? "fast, so slow down" : wpm < 130 ? (wpm < 100 ? "slow, so add energy" : "a little slow") : "ideal";
      if (spoken > 12 && wpm > 175) nudge("fast", `Slow down a little (${wpm} wpm).`, "warn", 15);
      if (spoken > 12 && wpm < 100 && L.words > 10) nudge("slow", `Pick up the pace a bit (${wpm} wpm).`, "warn", 20);
      if (spoken > 30 && wpm >= 130 && wpm <= 160) nudge("goodpace", "Great pace. Keep it up.", "good", 999);
    }
  }

  // volume vs. your own baseline (rolling 3s)
  const recent = L.frames.slice(-60);
  const thr = voicingThreshold();
  const rv = recent.filter((f) => f.db > thr).map((f) => f.db);
  if (L.baseline && rv.length >= 20) {
    const d = mean(rv) - L.baseline;
    $("gVol").textContent = d < -6 ? "getting quiet" : d > 6 ? "louder" : "steady";
    if (d < -8) nudge("quiet", "Speak up. Your volume is dropping.", "warn", 12);
  }

  // vocal variety (rolling 10s)
  const pitches = L.frames.slice(-200).filter((f) => f.db > thr && f.pitch).map((f) => f.pitch);
  if (pitches.length >= 40) {
    const sd = std(semis(pitches));
    $("gPitch").textContent = sd < 1.8 ? `flat (${sd.toFixed(1)} st)` : sd < 3 ? `ok (${sd.toFixed(1)} st)` : `lively (${sd.toFixed(1)} st)`;
    if (pitches.length >= 80 && sd < 1.8) nudge("mono", "A bit monotone. Raise your pitch on key words.", "warn", 20);
  }
  drawPitch(L.frames.slice(-200), thr);

  // pauses
  $("gPause").textContent = L.pauses;
  $("nPause").textContent = L.silence != null && t - L.silence >= 0.4 && L.voicedDbs.length
    ? `pausing… ${(t - L.silence).toFixed(1)}s`
    : L.pauseLens.length ? `avg ${mean(L.pauseLens).toFixed(1)}s · ${L.longPauses} over 3s` : "—";
}

// ---------------------------------------------------------------- nudges
let toastTimer = null;
function nudge(key, msg, kind = "warn", cooldown = 12) {
  const L = S.live;
  if (!L) return;
  const t = S.ac.currentTime - L.t0;
  if (L.nudgeAt[key] != null && t - L.nudgeAt[key] < cooldown) return;
  if (cooldown >= 999 && L.nudgeAt[key] != null) return;
  L.nudgeAt[key] = t; L.nudges++;
  const li = document.createElement("li");
  li.className = kind;
  li.innerHTML = `<small>${fmt(t)}</small>${esc(msg)}`;
  $("feed").prepend(li);
  while ($("feed").children.length > 12) $("feed").lastChild.remove();
  const toast = $("toast");
  toast.className = `toast show ${kind}`;
  toast.textContent = msg;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (toast.className = "toast"), 3500);
}

// ---------------------------------------------------------------- canvases
function prepCanvas(c) {
  const dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = +c.getAttribute("height");
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, W, H };
}

const volTrace = [];
function drawVolume(db, thr) {
  volTrace.push(db);
  if (volTrace.length > 160) volTrace.shift();
  const { ctx, W, H } = prepCanvas($("volCanvas"));
  ctx.clearRect(0, 0, W, H);
  const y = (d) => H - ((d + 80) / 75) * H;
  const bw = W / 160;
  ctx.fillStyle = cssVar("--vol");
  volTrace.forEach((d, i) => { const h = Math.max(1, H - y(d)); ctx.globalAlpha = d > thr ? 1 : 0.3; ctx.fillRect(i * bw, H - h, Math.max(1, bw - 0.5), h); });
  ctx.globalAlpha = 1;
  if (S?.live?.baseline) {                    // your normal speaking level
    ctx.strokeStyle = cssVar("--good"); ctx.setLineDash([3, 3]); ctx.beginPath();
    ctx.moveTo(0, y(S.live.baseline)); ctx.lineTo(W, y(S.live.baseline)); ctx.stroke(); ctx.setLineDash([]);
  }
}

function drawPitch(frames, thr) {
  const { ctx, W, H } = prepCanvas($("pitchCanvas"));
  ctx.clearRect(0, 0, W, H);
  const ps = frames.filter((f) => f.db > thr && f.pitch).map((f) => f.pitch);
  if (ps.length < 5) return;
  const med = pct(ps, 0.5);
  ctx.strokeStyle = cssVar("--line"); ctx.beginPath(); ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2); ctx.stroke();
  ctx.fillStyle = cssVar("--pit");
  frames.forEach((f, i) => {
    if (!(f.db > thr && f.pitch)) return;
    const s = 12 * Math.log2(f.pitch / med);
    if (Math.abs(s) < 10) ctx.fillRect((i / 200) * W, H / 2 - (s / 10) * (H / 2 - 3), 2, 2);
  });
}

// ---------------------------------------------------------------- speech recognition
function startRecognition() {
  if (!SR) { showWarn("Live transcription needs Chrome, Edge or Safari. Delivery metrics (volume, tone, pauses, “um” sounds) still work."); return; }
  const r = new SR();
  r.continuous = true; r.interimResults = true; r.lang = "en-US";
  let finalText = "";
  r.onresult = (e) => {
    if (!S?.live || S.live.demoUsed) return;
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) finalText += e.results[i][0].transcript.trim() + ". ";
      else interim += e.results[i][0].transcript;
    }
    setAnswerText(finalText.trim(), interim.trim());
  };
  r.onerror = (e) => {
    if (e.error === "no-speech" || e.error === "aborted") return;
    r.dead = true;
    showWarn(`Speech recognition error “${e.error}”. Open http://localhost:8000 in Chrome for live transcripts. Delivery metrics still work.`);
  };
  r.onend = () => { if (S?.rec === r && S.state === "answering" && !r.dead) try { r.start(); } catch {} };  // Chrome stops after silence
  try { r.start(); } catch {}
  S.rec = r;
}

function stopRecognition() {
  const r = S.rec;
  S.rec = null;
  try { r?.stop(); } catch {}
}

// ---------------------------------------------------------------- demo answer (no mic needed)
// Plays a recorded sample through the same analyser and streams its transcript in sync.
async function startDemo() {
  if (S.state === "speaking") { skipSpeaking(); await sleep(50); }
  if (S.state !== "answering" || S.demo) return;
  const words = (await fetch("samples/sample.txt").then((r) => r.text())).trim().split(/\s+/);
  const el = new Audio("samples/sample.wav");
  const src = S.ac.createMediaElementSource(el);
  src.connect(S.an); src.connect(S.ac.destination);
  await S.ac.resume();
  S.demo = { el, words };
  S.live.demoUsed = true;
  el.onended = () => { S.demo = null; src.disconnect(); };
  el.play();
}

function updateDemoText() {
  const { el, words } = S.demo;
  if (!el.duration) return;
  const n = Math.round((el.currentTime / el.duration) * words.length);
  if (n !== countWords(S.live.text)) setAnswerText(words.slice(0, n).join(" "));
}

function stopDemo() {
  if (S.demo) { S.demo.el.pause(); S.demo = null; }
}

// ---------------------------------------------------------------- rendering
function showWarn(msg) { $("warn").hidden = false; $("warn").textContent = msg; }
const stars = (n) => { n = Math.max(1, Math.min(5, n | 0)); return "★".repeat(n) + "☆".repeat(5 - n); };
const cls = (s) => (s >= 80 ? "good" : s >= 60 ? "mid" : "bad");

function renderHistoryEntry(e) {
  const m = e.metrics;
  const div = document.createElement("div");
  div.className = "qa";
  div.innerHTML = `
    <div class="q">Q${S.history.length}. ${esc(e.q)}</div>
    <div class="a">${e.a.startsWith("(") ? `<i>${esc(e.a.slice(1, -1))}</i>` : highlightFillers(e.a) || "<i>no transcript captured</i>"}</div>
    ${m ? `<div class="chips">
      <span class="chip">Delivery <b class="${cls(m.overall)}">${m.overall}</b></span>
      <span class="chip">${fmt(m.durationSec)} (thought for ${m.thinkingTimeSec}s)</span>
      <span class="chip">${m.wpm ?? "—"} wpm</span>
      <span class="chip">${m.fillerCount + m.hesitations} fillers</span>
      <span class="chip">tone ${m.pitchStdSemitones ?? "—"} st</span>
      <span class="chip">${m.pauseCount} pauses</span>
    </div>` : ""}
    <div class="fb"><span class="muted">Getting feedback…</span></div>`;
  e.el = div;
  $("history").prepend(div);
}

function renderFeedback(e) {
  const f = e.feedback;
  e.el.querySelector(".fb").innerHTML =
    `<span class="stars">${stars(f.stars)}</span> ${esc(f.comment || "")}<br>💡 <b>Tip:</b> ${esc(f.tip || "")}`;
}

function teardown() {
  clearInterval(S.timer);
  stopRecognition();
  stopDemo();
  speechSynthesis.cancel();
  S.stream?.getTracks().forEach((t) => t.stop());
  S.ac.close();
}

async function finish(r) {
  const s = S;
  teardown();
  S = null;
  $("interview").hidden = true;
  const answered = s.history.filter((h) => h.metrics);
  if (!answered.length) { $("setup").hidden = false; return; }
  if (r.reaction && s.voice) { const v = s.voice; S = { voice: v }; await speak(r.reaction).catch(() => {}); S = null; }
  $("summary").hidden = false;
  const overall = Math.round(mean(answered.map((h) => h.metrics.overall)));
  const avgStars = mean(answered.map((h) => h.feedback?.stars || 0));
  $("sOverall").textContent = overall;
  $("sOverallMsg").textContent = `Average delivery score over ${answered.length} answer(s). Content averaged ${avgStars.toFixed(1)}/5.`;
  const sm = r.summary;
  $("sAI").innerHTML = sm ? `
    <p>${esc(sm.overall || "")}</p>
    <h3>Strengths</h3><ul>${(sm.strengths || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    <h3>To improve</h3><ul>${(sm.improvements || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>
    <div class="tip">💡 <b>Practice next:</b> ${esc(sm.tip || "")}</div>
    <p class="status">${r.source === "claude" ? `Written by Claude (${esc(r.model)})` : "Rule-based summary. Set ANTHROPIC_API_KEY for a Claude interviewer."}</p>` : "";
  $("sTable").innerHTML = `<tr><th>Question</th><th>Content</th><th>Delivery</th><th>Time</th><th>Pace</th><th>Fillers</th><th>Tone</th></tr>` +
    s.history.map((h) => `<tr>
      <td>${esc(h.q)}</td>
      <td class="stars">${h.feedback ? stars(h.feedback.stars) : "—"}</td>
      <td>${h.metrics ? `<b class="${cls(h.metrics.overall)}">${h.metrics.overall}</b>` : "—"}</td>
      <td>${h.metrics ? fmt(h.metrics.durationSec) : "—"}</td>
      <td>${h.metrics?.wpm ?? "—"}</td>
      <td>${h.metrics ? h.metrics.fillerCount + h.metrics.hesitations : "—"}</td>
      <td>${h.metrics?.pitchStdSemitones ?? "—"}</td></tr>`).join("");
}

// ================================================================ wiring
$("startBtn").onclick = () => startInterview().catch((e) => showWarn(String(e)));
$("doneBtn").onclick = () => (S?.state === "speaking" ? skipSpeaking() : finishAnswer());
$("skipBtn").onclick = () => finishAnswer(true);
$("demoBtn").onclick = () => startDemo().catch((e) => showWarn(`Demo failed: ${e.message}`));
$("endBtn").onclick = () => {
  if (!S) return;
  S.ended = true;
  if (S.state === "answering") finishAnswer();
  else if (S.state === "speaking") { speechSynthesis.cancel(); setState("thinking"); nextTurn(); }
};
$("againBtn").onclick = () => { $("summary").hidden = true; $("setup").hidden = false; };
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || !S || /INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)) return;
  e.preventDefault();
  $("doneBtn").click();
});
window.speechSynthesis?.getVoices();   // warm up the voice list

fetch("/api/status").then((r) => r.json()).then((s) => {
  $("aiStatus").textContent = s.ai
    ? `Interviewer: Claude (${s.model}). It asks follow-ups based on your answers.`
    : "Interviewer: built-in question bank with rule-based feedback. Set ANTHROPIC_API_KEY and restart to use Claude, which asks tailored follow-ups.";
}).catch(() => {});
