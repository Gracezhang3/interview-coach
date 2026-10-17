#!/usr/bin/env python3
"""Local server for the live mock-interview coach.

Serves ./static and exposes:
  GET  /api/status     -> {"ai": bool, "model": str}
  POST /api/interview  -> the interviewer's next turn: reaction to the last answer,
                          feedback on it, and the next question (or a final summary)

Standard library only. Set ANTHROPIC_API_KEY for a Claude interviewer that asks
tailored follow-up questions; without it a question bank + rule-based feedback is used.
"""
import json
import os
import random
import re
import urllib.error
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain, zip_longest
from pathlib import Path

PORT = int(os.environ.get("PORT", "8000"))
STATIC_DIR = Path(__file__).parent / "static"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

# ---------------------------------------------------------------- Claude interviewer

PROMPT = """You are a friendly but rigorous job interviewer conducting a spoken mock interview.
Role: {role}
Interview type: {kind}
Planned number of questions: {total}

Conversation so far. Each answer was transcribed live by browser speech recognition
(expect small errors, and "um"/"uh" are usually missing) and comes with delivery
metrics measured from the candidate's audio:
{history}

{task}

Reply with ONLY a JSON object, no prose before or after."""

TASK_FIRST = """Ask the opening question. Return:
{"next_question": "<the question, as you'd say it aloud>"}"""

TASK_NEXT = """Respond to the last answer and ask question {n} of {total}. Return:
{{
  "reaction": "<one short, natural sentence you'd say aloud before the next question; acknowledge, don't grade>",
  "feedback": {{"stars": <1-5 for content of the last answer>,
               "comment": "<one sentence coaching on the last answer's content: structure (STAR), specificity, relevance; quote a short phrase if useful>",
               "tip": "<one concrete thing to do better next time>"}},
  "next_question": "<next question; a follow-up that probes the last answer is fine if it was vague>"
}}"""

TASK_FINAL = """The interview is over. Respond to the last answer (if any) and summarize. Return:
{
  "reaction": "<one short closing sentence you'd say aloud>",
  "feedback": {"stars": 1-5, "comment": "<one sentence on the last answer>", "tip": "<one tip>"},
  "summary": {"overall": "<two sentences: overall impression, as a hiring manager would put it>",
              "strengths": ["<2-3 specific strengths, content and delivery>"],
              "improvements": ["<2-3 specific, actionable improvements>"],
              "tip": "<the single most valuable thing to practice next>"}
}
If there were no answers, return {"summary": null}."""


def format_history(history):
    if not history:
        return "(no questions asked yet)"
    lines = []
    for i, h in enumerate(history, 1):
        m = h.get("metrics") or {}
        lines.append(f"Q{i}: {h.get('q', '')}")
        lines.append(f"A{i}: {h.get('a') or '(no answer)'}")
        if m:
            lines.append(
                f"Delivery A{i}: {m.get('durationSec')}s, {m.get('wpm')} wpm, "
                f"{m.get('fillerCount', 0)} filler words + {m.get('hesitations', 0)} hesitation sounds, "
                f"pitch spread {m.get('pitchStdSemitones')} semitones, "
                f"{round((m.get('fadeRate') or 0) * 100)}% phrases trailing off, delivery score {m.get('overall')}/100"
            )
    return "\n".join(lines)


def claude_turn(req):
    history = req.get("history", [])
    total = req.get("total", 5)
    if req.get("final"):
        task = TASK_FINAL
    elif not history:
        task = TASK_FIRST
    else:
        task = TASK_NEXT.format(n=len(history) + 1, total=total)
    prompt = PROMPT.format(role=req.get("role") or "general", kind=req.get("type", "mixed"),
                           total=total, history=format_history(history), task=task)
    body = json.dumps({"model": MODEL, "max_tokens": 1000,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    http_req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": API_KEY,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(http_req, timeout=60) as resp:
        data = json.load(resp)
    text = "".join(b.get("text", "") for b in data.get("content", []))
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


# ---------------------------------------------------------------- offline fallback

QUESTION_BANK = {
    "behavioral": [
        "Tell me about yourself, and what draws you to this {role} role.",
        "Tell me about a time you had a conflict with a teammate. How did you handle it?",
        "Describe a time you failed at something. What did you learn?",
        "Tell me about a time you had to meet a tight deadline.",
        "Give me an example of when you led something without formal authority.",
        "Tell me about a time you disagreed with your manager or teacher.",
        "What accomplishment are you most proud of, and why?",
    ],
    "general": [
        "Tell me about yourself, and what draws you to this {role} role.",
        "What would you say is your greatest strength?",
        "What's your biggest weakness, and what are you doing about it?",
        "Why should we hire you over other candidates?",
        "Where do you see yourself in five years?",
        "What motivates you to do your best work?",
    ],
    "technical": [
        "Walk me through a project you built from start to finish.",
        "Explain a technical concept you know well to someone non-technical.",
        "Tell me about a hard bug or problem you solved. How did you approach it?",
        "Describe a trade-off you made in a project. What did you choose and why?",
        "How do you keep your skills up to date?",
    ],
}

REACTIONS = ["Thanks, that's helpful.", "Got it, thank you.", "Okay, interesting.",
             "Thanks for walking me through that.", "Great, I appreciate that."]

LABELS = {"pace": "pace", "fillers": "few filler words", "volume": "steady volume",
          "variety": "vocal variety", "pausing": "use of pauses", "duration": "answer length"}
TIPS = {
    "pace": "Aim for 130-160 words per minute; breathe at the end of each sentence.",
    "fillers": "When you need time to think, pause silently instead of saying 'um' or 'like'.",
    "volume": "Keep your energy through the last word of each sentence instead of trailing off.",
    "variety": "Lift your pitch on the one key word in each sentence so you sound engaged.",
    "pausing": "Pause for a beat after your key point so it lands.",
    "duration": "Aim for 1-2 minutes per answer: enough for a full STAR story, short enough to stay sharp.",
}


def questions_for(kind, role):
    if kind == "mixed":
        b, g, t = QUESTION_BANK["behavioral"], QUESTION_BANK["general"][1:], QUESTION_BANK["technical"]
        qs = [b[0]] + [q for q in chain.from_iterable(zip_longest(b[1:], t, g)) if q]
    else:
        qs = QUESTION_BANK.get(kind, QUESTION_BANK["general"])
    return [q.format(role=role or "this") for q in qs]


def answer_feedback(h):
    a = (h.get("a") or "").strip()
    if not a or a in ("(skipped)", "(no answer)"):
        return {"stars": 1, "comment": "No answer was captured for this question.",
                "tip": "Even a short, honest answer beats skipping. Buy time with 'Let me think about that for a second.'"}
    low = a.lower()
    words = len(re.findall(r"[a-z']+", low))
    situation = bool(re.search(r"\b(when i|at my|last (year|summer|semester|month)|in my (role|job|class|internship|team)|we were|there was a time|during)\b", low))
    action = bool(re.search(r"\bi (led|built|decided|created|organized|designed|wrote|talked|worked|started|took|made|helped|reached|proposed|managed|fixed|learned|set up|asked|suggested)\b", low))
    result = bool(re.search(r"\b(result|outcome|increased|reduced|improved|saved|grew|won|launched|finished|delivered|ended up)\b|\d", low))
    stars = max(1, min(5, 1 + situation + action + result + (60 <= words <= 320)))
    check = ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in
                      (("situation", situation), ("action", action), ("result", result)))
    if not situation:
        tip = "Open with the situation in one sentence: where you were and what was at stake."
    elif not action:
        tip = "Say 'I' more: what did you specifically do, step by step?"
    elif not result:
        tip = "End with the result. A number or concrete outcome makes the story land."
    elif words < 60:
        tip = "Go deeper: add one specific detail or example."
    elif words > 320:
        tip = "Tighten it: aim for 60-120 seconds and cut background detail."
    else:
        tip = "Solid structure. Now quantify the impact with a number."
    return {"stars": stars, "comment": f"{words} words. STAR check: {check}.", "tip": tip}


def summary_fallback(history):
    answered = [h for h in history if h.get("metrics")]
    if not answered:
        return None
    avg = {}
    for k in LABELS:
        vals = [h["metrics"].get("scores", {}).get(k) for h in answered]
        vals = [v for v in vals if v is not None]
        if vals:
            avg[k] = sum(vals) / len(vals)
    ranked = sorted(avg.items(), key=lambda kv: kv[1])
    stars = [h.get("feedback", {}).get("stars") or answer_feedback(h)["stars"] for h in answered]
    mean_stars = sum(stars) / len(stars)
    strengths = [f"Consistently good {LABELS[k]} (avg {round(v)}/100)." for k, v in reversed(ranked) if v >= 75][:2]
    if mean_stars >= 3.5:
        strengths.append("Your answers mostly had a clear situation, action and result.")
    improvements = [TIPS[k] for k, v in ranked if v < 80][:2]
    if mean_stars < 3.5:
        improvements.append("Use STAR: situation, task, what YOU did, then the measurable result.")
    delivery = round(sum(h["metrics"].get("overall", 0) for h in answered) / len(answered))
    return {
        "overall": f"You answered {len(answered)} question(s) with an average delivery score of {delivery}/100 "
                   f"and content rated {mean_stars:.1f}/5.",
        "strengths": strengths or ["You completed a full mock interview, which is great practice."],
        "improvements": improvements or ["Keep practicing with harder, more specific questions."],
        "tip": TIPS[ranked[0][0]] if ranked else TIPS["duration"],
    }


def fallback_turn(req):
    history = req.get("history", [])
    out = {}
    if history:
        out["reaction"] = random.choice(REACTIONS)
        out["feedback"] = answer_feedback(history[-1])
    if req.get("final"):
        out["reaction"] = "Thanks so much, that's the end of our interview."
        out["summary"] = summary_fallback(history)
    else:
        asked = {h.get("q") for h in history}
        pool = [q for q in questions_for(req.get("type", "mixed"), req.get("role")) if q not in asked]
        out["next_question"] = pool[0] if pool else "Is there anything else you'd like me to know about you?"
    return out


# ---------------------------------------------------------------- HTTP

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            return self._json(200, {"ai": bool(API_KEY), "model": MODEL if API_KEY else None})
        return super().do_GET()

    def do_POST(self):
        if self.path != "/api/interview":
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "invalid JSON"})
        if API_KEY:
            try:
                return self._json(200, {"source": "claude", "model": MODEL, **claude_turn(req)})
            except (urllib.error.URLError, ValueError, KeyError) as e:
                detail = e.read().decode()[:300] if isinstance(e, urllib.error.HTTPError) else str(e)
                print("Claude call failed, using fallback:", detail)
                return self._json(200, {"source": "rules", "warning": f"AI call failed: {detail}", **fallback_turn(req)})
        return self._json(200, {"source": "rules", **fallback_turn(req)})

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"Interview coach running at http://localhost:{PORT}")
    print("Interviewer:", f"Claude ({MODEL})" if API_KEY else "question bank + rule-based feedback (set ANTHROPIC_API_KEY for Claude)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
