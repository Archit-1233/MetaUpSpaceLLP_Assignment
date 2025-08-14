from dataclasses import dataclass
from typing import List, Dict, Any
from utils.text import tokenize, keyword_count, year_from, confidence_from_counts

# Canonical skills and aliases
SKILL_CANON = {
    "python": ["python", "py"],
    "pytorch": ["pytorch", "torch"],
    "tensorflow": ["tensorflow", "tf"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy", "np"],
    "matplotlib": ["matplotlib", "plt"],
    "mlops": ["mlops", "ml ops"],
    "nlp": ["nlp", "natural language processing"],
    "cv": ["computer vision", "cv", "opencv"],
    "llm": ["llm", "large language model", "gpt", "transformer"],
    "prompt engineering": ["prompt engineering", "prompting"],
    "xgboost": ["xgboost", "lightgbm", "catboost"],
    "sql": ["sql", "postgres", "mysql", "sqlite"],
    "docker": ["docker", "container"],
    "kubernetes": ["kubernetes", "k8s"],
    "flask": ["flask", "fastapi"],
    "django": ["django"],
    "java": ["java"],
    "c++": ["c++", "cpp"],
    "git": ["git", "github", "gitlab"],
}

@dataclass
class Candidate:
    id: str
    name: str
    linkedin_summary: str
    work_history: List[Dict[str, Any]]
    github: Dict[str, Any]

class CandidateProfilerAgent:
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self._zs = None
        if self.use_llm:
            try:
                from transformers import pipeline
                self._zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
            except Exception as e:
                print("[Profiler] LLM unavailable, falling back to heuristics. Reason:", e)
                self._zs = None

    def _llm_skill_scores(self, text: str, candidate_labels: List[str]) -> Dict[str, float]:
        if not self._zs:
            return {}
        res = self._zs(text, candidate_labels=candidate_labels, multi_label=True)
        return {lab: float(scr) for lab, scr in zip(res["labels"], res["scores"])}

    def extract_skills(self, candidate: Candidate) -> Dict[str, Dict[str, Any]]:
        blobs = [candidate.linkedin_summary]
        for repo in candidate.github.get("repos", []):
            blobs.append(repo.get("description",""))
            blobs.extend(repo.get("topics", []))
            blobs.extend(repo.get("languages", []))
        for job in candidate.work_history:
            blobs.append(job.get("title",""))
            blobs.append(job.get("description",""))
            blobs.append(job.get("tech",""))

        fulltext = " ".join(blobs)
        skills = {}

        # Heuristic counts
        for canon, variants in SKILL_CANON.items():
            count = keyword_count(fulltext, variants)
            if count>0:
                last_role_text = (candidate.work_history[-1]["description"] + " " + candidate.work_history[-1].get("tech","")).lower() if candidate.work_history else ""
                boost = 0.05 if any(v in last_role_text for v in variants) else 0.0
                conf = confidence_from_counts(count, recency_boost=boost)
                skills[canon] = {"confidence": conf, "evidence_count": count, "source":"heuristic"}

        # Optional zero-shot adjustment
        if self._zs:
            labels = list(SKILL_CANON.keys())
            zs_scores = self._llm_skill_scores(fulltext, labels)
            for lab, scr in zs_scores.items():
                if scr < 0.2:
                    continue
                if lab not in skills:
                    skills[lab] = {"confidence": round(0.5 + 0.48*scr,2), "evidence_count": 0, "source":"llm"}
                else:
                    skills[lab]["confidence"] = round(max(skills[lab]["confidence"], 0.5 + 0.48*scr), 2)
                    skills[lab]["source"] = skills[lab]["source"] + "+llm"

        return dict(sorted(skills.items(), key=lambda kv: (-kv[1]["confidence"], kv[0])))

    def summarize_progression(self, candidate: Candidate) -> str:
        if not candidate.work_history:
            return "No work history provided."
        parts = []
        for job in candidate.work_history:
            yr_from = year_from(job.get("from",""))
            yr_to = year_from(job.get("to","")) or 2025
            dur_yrs = max(1, (yr_to - (yr_from or yr_to)))
            parts.append(f"{job.get('title','Role')} at {job.get('company','Company')} ({yr_from or '?'}–{yr_to} ~{dur_yrs}y): {job.get('impact','Contributed to key deliverables.')}")
        return " → ".join(parts)

    def build_report(self, candidate: Candidate) -> Dict[str, Any]:
        skills = self.extract_skills(candidate)
        top_skills = sorted(skills.items(), key=lambda kv: -kv[1]["confidence"])[:8]
        return {
            "candidate_id": candidate.id,
            "name": candidate.name,
            "top_skills": [{ "skill": s, **meta } for s, meta in top_skills],
            "all_skills": skills,
            "career_progression": self.summarize_progression(candidate),
        }
