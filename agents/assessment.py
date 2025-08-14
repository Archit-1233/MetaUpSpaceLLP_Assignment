from dataclasses import dataclass
from typing import Dict, Any, List
import random

@dataclass
class JobRole:
    id: str
    title: str
    level: str
    must_have_skills: List[str]
    nice_to_have_skills: List[str]

class AssessmentDesignerAgent:
    def _level_complexity(self, level: str) -> Dict[str,int]:
        mapping = {
            "intern": {"coding":1, "ml":1, "system":0},
            "junior": {"coding":2, "ml":1, "system":1},
            "mid": {"coding":3, "ml":2, "system":2},
            "senior": {"coding":3, "ml":3, "system":3},
        }
        return mapping.get(level.lower(), mapping["junior"])

    def _gen_coding_problem(self, skills: List[str], complexity: int) -> Dict[str,Any]:
        topics = [
            ("strings", "Normalize noisy text and compute token frequencies."),
            ("graphs", "Build a dependency resolver with cycle detection."),
            ("dp", "Memoize an inference cost function; analyze complexity."),
            ("dataframes", "Implement group-by aggregations without external libs."),
        ]
        topic = random.choice(topics)
        if "python" in skills: lang = "Python"
        elif "java" in skills: lang = "Java"
        elif "c++" in skills: lang = "C++"
        else: lang = "Any language"
        return {"type":"coding","language":lang,"complexity":complexity,"title":topic[0].title(),"prompt":topic[1]+" Add tests and discuss trade-offs."}

    def _gen_ml_problem(self, skills: List[str], complexity: int) -> Dict[str,Any]:
        if "cv" in skills:
            prompt = "Design a small image classifier; compare CNN from scratch vs transfer learning; metrics + augmentation."
        elif "nlp" in skills or "llm" in skills:
            prompt = "Design a text-classification pipeline; compare tf-idf+linear vs transformer fine-tune; eval & inference cost."
        else:
            prompt = "Choose a dataset; build E2E supervised model; compare 3 models; explain validation and monitoring."
        return {"type":"ml","complexity":complexity,"prompt":prompt}

    def _gen_system_prompt(self, level: str, skills: List[str], complexity: int) -> Dict[str,Any]:
        focus = "LLM app with retrieval" if "llm" in skills else "model training + serving"
        return {"type":"system_design","complexity":complexity,"prompt":f"Design a {focus} system for 100k MAU: data flow, storage, scaling, CI/CD, observability, cost."}

    def _rubric(self, level: str) -> Dict[str,Any]:
        weights = {"Problem-solving approach": 0.4, "Code quality": 0.3, "Communication": 0.3}
        if level.lower() in ["senior","mid"]:
            weights = {"Problem-solving approach": 0.35, "Code quality": 0.25, "Communication": 0.2, "Architecture & Tradeoffs": 0.2}
        return {"weights": weights, "scale": "0–4 per criterion"}

    def _bias_mitigation(self) -> List[str]:
        return [
            "Score only on rubric; ignore identity or affiliations.",
            "Use identical prompts, time windows, and scales for comparable candidates.",
            "Give rubric-aligned feedback; avoid personality labels.",
            "Double-mark borderline cases with a second reviewer.",
            "Mask PII where feasible during evaluation.",
        ]

    def design_package(self, candidate_report: Dict[str,Any], job: JobRole) -> Dict[str,Any]:
        level = job.level
        skills_sorted = [x["skill"] for x in candidate_report["top_skills"]]
        complexity = self._level_complexity(level)
        challenges = []
        if complexity["coding"]>0:
            challenges.append(self._gen_coding_problem(skills_sorted, complexity["coding"]))
        if complexity["ml"]>0:
            challenges.append(self._gen_ml_problem(skills_sorted, complexity["ml"]))
        if complexity["system"]>0:
            challenges.append(self._gen_system_prompt(level, skills_sorted, complexity["system"]))
        return {
            "candidate_id": candidate_report["candidate_id"],
            "job_id": job.id,
            "job_title": job.title,
            "level": level,
            "technical_challenges": challenges,
            "evaluation_framework": self._rubric(level),
            "bias_mitigation_protocol": self._bias_mitigation(),
        }
