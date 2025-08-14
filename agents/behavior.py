from typing import Dict, Any
import re
from utils.text import tokenize

SOFT_KEYWORDS = {
    "collaboration": ["team", "teammate", "pair", "collaborat", "mentor", "cross-functional", "stakeholder"],
    "communication": ["communicat", "present", "explain", "document", "clarify", "write", "talk"],
    "problem_solving": ["debug", "root cause", "investigate", "optimiz", "design", "tradeoff", "prototype"],
    "leadership": ["lead", "own", "drive", "coordinate", "organize", "initiative"],
    "adaptability": ["adapt", "learn", "shift", "change", "uncertain", "ambigu"],
}

class BehavioralAnalyzerAgent:
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self._sentiment = None
        if self.use_llm:
            try:
                from transformers import pipeline
                self._sentiment = pipeline("sentiment-analysis")
            except Exception as e:
                print("[Behavior] LLM sentiment unavailable; falling back to keywords. Reason:", e)
                self._sentiment = None

    def analyze(self, transcript: str) -> Dict[str,Any]:
        tokens = tokenize(transcript)
        counts = {theme: sum(any(tok.startswith(stem) for tok in tokens) for stem in stems)
                  for theme, stems in SOFT_KEYWORDS.items()}
        tone = "neutral"
        if self._sentiment:
            try:
                res = self._sentiment(transcript[:2000])[0]
                tone = "constructive" if res["label"].lower().startswith("pos") else "cautious" if res["label"].lower().startswith("neg") else "neutral"
            except Exception:
                tone = "neutral"
        else:
            pos = sum(1 for t in tokens if t in {"clarify","help","learn","thanks","appreciate","together","solution","progress","listen","respect","deadline","ownership"})
            neg = sum(1 for t in tokens if t in {"blame","delay","excuse","argue","stuck","problem","late"})
            tone = "constructive" if pos>=neg else "cautious"

        key_quotes = []
        for sent in re.split(r"[\.!\?]\s+", transcript):
            s = sent.strip()
            if any(k in s.lower() for k in ["team", "debug", "communicat", "design"]) and len(s)>0:
                key_quotes.append(s)
            if len(key_quotes)>=3: break

        return {
            "themes": counts,
            "tone": tone,
            "signals": {
                "collaboration": "present" if counts["collaboration"]>0 else "unclear",
                "communication": "present" if counts["communication"]>0 else "unclear",
                "problem_solving": "present" if counts["problem_solving"]>0 else "unclear",
            },
            "notes": "Content-only analysis; no demographic or affinity factors used.",
            "illustrative_quotes": key_quotes
        }
