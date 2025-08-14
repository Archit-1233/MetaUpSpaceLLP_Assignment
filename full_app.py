# app_final.py
# Final polished MAIRS demo (Streamlit)
# Run:
#   python3 -m venv .venv
#   source .venv/bin/activate      # Windows: .venv\Scripts\activate
#   pip install streamlit matplotlib
#   streamlit run app_final.py

import streamlit as st
import re, json, random
from dataclasses import dataclass
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import io
from datetime import datetime

st.set_page_config(page_title="MAIRS — Final Demo", layout="wide")

# ---------------------------
# Utilities
# ---------------------------
STOPWORDS = set(
    "a an the and or but if on in at for with without to of from is are was were be been being by as into through during his her their our your i you he she they we me him her them us this that these those it its it's".split()
)

def tokenize(text: str):
    text = (text or "").lower()
    return [t for t in re.findall(r"[a-zA-Z\+\#\d][a-zA-Z\+\#\d\-]*", text) if t not in STOPWORDS]

def keyword_count(text: str, stems):
    t = (text or "").lower()
    return sum(t.count(s) for s in stems)

def year_from(s: str):
    m = re.search(r"(20\d{2}|19\d{2})", s or "")
    return int(m.group(1)) if m else None

def confidence_from_counts(total_mentions: int, recency_boost: float = 0.0) -> float:
    base = 1 - (0.7 ** max(int(total_mentions), 0))
    conf = 0.5 + 0.48 * base + recency_boost
    return round(min(conf, 0.98), 2)

# ---------------------------
# Skill ontology
# ---------------------------
SKILL_CANON = {
    "python": ["python","py"],
    "pytorch": ["pytorch","torch"],
    "tensorflow": ["tensorflow","tf"],
    "scikit-learn": ["scikit-learn","sklearn"],
    "pandas": ["pandas"],
    "numpy": ["numpy","np"],
    "mlops": ["mlops","ml ops"],
    "nlp": ["nlp","natural language processing"],
    "cv": ["computer vision","cv","opencv"],
    "llm": ["llm","large language model","gpt","transformer"],
    "prompt engineering": ["prompt engineering","prompting"],
    "xgboost": ["xgboost","lightgbm","catboost"],
    "sql": ["sql","postgres","mysql","sqlite"],
    "docker": ["docker","container"],
    "kubernetes": ["kubernetes","k8s"],
    "flask": ["flask","fastapi"],
    "git": ["git","github","gitlab"],
    "java": ["java"],
    "c++": ["c++","cpp"],
}

# ---------------------------
# Synthetic dataset (6 candidates)
# ---------------------------
DATA = { 
  "candidates": [
    {"id":"cand_001","name":"Riya Sharma",
     "linkedin_summary":"AI/ML enthusiast with CV & NLP internships. Python, PyTorch, scikit-learn, pandas. Built small Flask APIs and RAG prototypes.",
     "work_history":[
       {"company":"VisionLabs","title":"Computer Vision Intern","from":"2022","to":"2023","description":"Trained CNNs and fine-tuned vision transformers for defect detection; collaborated with teammates; documented results.","tech":"Python, PyTorch, OpenCV","impact":"Improved F1 by 7%"},
       {"company":"NLPWorks","title":"NLP Research Intern","from":"2023","to":"2024","description":"Built text classifiers; prompt engineering for LLMs; created FastAPI service; wrote unit tests.","tech":"Python, Transformers, FastAPI","impact":"Shipped API used by 3 teams"}
     ],
     "github": {"repos":[{"name":"cv-defects","description":"Image defect detection (transfer learning)","languages":["Python"],"topics":["computer vision","cnn"]},{"name":"rag-lab","description":"RAG + prompts","languages":["Python"],"topics":["llm","prompt engineering"]}]}
    },
    {"id":"cand_002","name":"Arjun Verma",
     "linkedin_summary":"Data generalist: sklearn, XGBoost, SQL; strong with ETL and MLOps. Built monitoring and dashboards.",
     "work_history":[
       {"company":"DataFlow","title":"ML Intern","from":"2021","to":"2022","description":"Implemented feature store; optimized XGBoost; mentored juniors.","tech":"Python, scikit-learn, XGBoost, SQL, Docker","impact":"Cut training time by 30%"},
       {"company":"RetailAI","title":"Junior ML Engineer","from":"2022","to":"2024","description":"Owned ETL pipelines; deployed models with Docker + CI.","tech":"Python, SQL, Docker","impact":"Stabilized pipelines"}
     ],
     "github": {"repos":[{"name":"etl-utils","description":"Pandas utilities","languages":["Python"],"topics":["pandas","data quality"]}]}
    },
    {"id":"cand_003","name":"Sana Kapoor",
     "linkedin_summary":"NLP engineer with transformer fine-tuning and infra experience. Docker, FastAPI, monitoring.",
     "work_history":[
       {"company":"Textify","title":"NLP Intern","from":"2022","to":"2023","description":"Fine-tuned transformers; built inference endpoints.","tech":"Python, Transformers, Docker","impact":"Reduced latency by 20%"},
       {"company":"InsightAI","title":"ML Engineer","from":"2023","to":"2024","description":"Built retraining pipelines and dashboards.","tech":"Python, SQL, MLflow","impact":"99.5% uptime"}
     ],
     "github": {"repos":[{"name":"nlp-utils","description":"Tokenizer experiments","languages":["Python"],"topics":["nlp","transformer"]}]}
    },
    {"id":"cand_004","name":"Vikram Singh",
     "linkedin_summary":"Frontend & Android dev turned ML hobbyist. TFLite & mobile optimizations.",
     "work_history":[
       {"company":"Mobify","title":"Android Developer","from":"2019","to":"2022","description":"Built Android apps; mentored interns.","tech":"Kotlin, Android Studio","impact":"3 apps on Play Store"},
       {"company":"EdgeAI","title":"ML Intern","from":"2022","to":"2023","description":"Converted models to TFLite and optimized inference.","tech":"TensorFlow, TFLite","impact":"Reduced model size by 60%"}
     ],
     "github": {"repos":[{"name":"edge-app","description":"Mobile demo","languages":["Python"],"topics":["tflite","cv"]}]}
    },
    {"id":"cand_005","name":"Megha Patel",
     "linkedin_summary":"Data analyst and BI developer. Strong with SQL, Excel, Power BI; building dashboards and ETL.",
     "work_history":[
       {"company":"RetailInsights","title":"Data Analyst","from":"2020","to":"2023","description":"Built Power BI dashboards; automated ETL jobs.","tech":"SQL, Power BI, Python","impact":"Enabled weekly KPIs"}
     ],
     "github": {"repos":[{"name":"dashboards","description":"BI samples","languages":["PowerBI"],"topics":["sql","powerbi"]}]}
    },
    {"id":"cand_006","name":"Aman Roy",
     "linkedin_summary":"Research intern exploring RL and simulation. Comfortable with PyTorch and experiment tracking.",
     "work_history":[
       {"company":"SimLab","title":"Research Intern","from":"2023","to":"2024","description":"Set up RL baselines and evaluation.","tech":"Python, PyTorch, Gym","impact":"Established eval pipeline"}
     ],
     "github": {"repos":[{"name":"rl-baselines","description":"RL experiments","languages":["Python"],"topics":["rl"]}]}
    }
  ],
  "jobs": [
    {"id":"job_ai_intern_001","title":"AI Intern – Applied ML/LLMs","level":"intern",
     "must_have_skills":["python","scikit-learn","pytorch","nlp or cv"], "nice_to_have_skills":["llm","mlops","docker","sql"]}
  ],
  "transcripts": {
    "cand_001":"I like to clarify requirements early and keep the team in the loop. When a training job failed, I paired with a teammate to debug the root cause.",
    "cand_002":"I usually organize stakeholders, communicate risks, and design incremental rollouts. When pipelines were late, I focused on solutions.",
    "cand_003":"I prototype quickly, write runbooks and collaborate with infra and product teams to make decisions.",
    "cand_004":"I mentor interns and explain UI tradeoffs. I integrated models on mobile with backend engineers.",
    "cand_005":"I prepare dashboards and present weekly to stakeholders, collect feedback and iterate.",
    "cand_006":"I run experiments, tune hyperparameters, and document what worked to replicate with peers."
  },
  "market_data": {
    "regions":["India","EU","US"],
    "salary_benchmarks": {
      "India":{"intern":[20000,50000],"junior":[600000,1200000]},
      "EU":{"intern":[800,1800],"junior":[35000,60000]},
      "US":{"intern":[1500,3500],"junior":[80000,120000]}
    },
    "demand_by_skill":{"python":95,"pytorch":70,"tensorflow":55,"llm":80,"nlp":65,"cv":60,"sql":75,"docker":60,"mlops":50},
    "channel_effectiveness": {
      "LinkedIn":{"python":0.8,"llm":0.7,"sql":0.85,"docker":0.7},
      "GitHub":{"python":0.9,"pytorch":0.85,"cv":0.8,"llm":0.75},
      "Kaggle":{"python":0.75,"nlp":0.7,"cv":0.8},
      "Reddit":{"mlops":0.6,"llm":0.5}
    }
  }
}

# ---------------------------
# Agent implementations (same design as earlier)
# ---------------------------
@dataclass
class Candidate:
    id: str; name: str; linkedin_summary: str; work_history: List[Dict[str,Any]]; github: Dict[str,Any]

class CandidateProfilerAgent:
    def extract_skills(self, cand: Candidate) -> Dict[str, Dict[str,Any]]:
        blobs = [cand.linkedin_summary]
        for repo in cand.github.get("repos", []):
            blobs.append(repo.get("description",""))
            blobs.extend(repo.get("topics", []))
            blobs.extend(repo.get("languages", []))
        for job in cand.work_history:
            blobs.append(job.get("title",""))
            blobs.append(job.get("description",""))
            blobs.append(job.get("tech",""))
        full = " ".join(blobs).lower()
        skills = {}
        for canon, variants in SKILL_CANON.items():
            count = keyword_count(full, variants)
            if count>0:
                last_role_text = (cand.work_history[-1].get("description","") + " " + cand.work_history[-1].get("tech","")).lower() if cand.work_history else ""
                boost = 0.05 if any(v in last_role_text for v in variants) else 0.0
                skills[canon] = {"confidence":confidence_from_counts(count, boost), "evidence_count":count}
        # sort by confidence desc
        return dict(sorted(skills.items(), key=lambda kv:-kv[1]["confidence"]))

    def summarize_progression(self, cand: Candidate) -> str:
        if not cand.work_history: return "No work history"
        parts=[]
        for job in cand.work_history:
            fr = year_from(job.get("from","")) or "?"
            to = year_from(job.get("to","")) or "present"
            parts.append(f"{job.get('title')} at {job.get('company')} ({fr}–{to}): {job.get('impact','')}")
        return " → ".join(parts)

    def build_report(self, cand: Candidate) -> Dict[str,Any]:
        skills = self.extract_skills(cand)
        top_skills = [{"skill":s, **meta} for s,meta in list(skills.items())[:8]]
        return {"candidate_id":cand.id, "name":cand.name, "top_skills":top_skills, "all_skills":skills, "career_progression":self.summarize_progression(cand)}

class AssessmentDesignerAgent:
    def _level_complexity(self, level:str):
        mapping = {"intern":{"coding":1,"ml":1,"system":0},"junior":{"coding":2,"ml":1,"system":1},"mid":{"coding":3,"ml":2,"system":2},"senior":{"coding":3,"ml":3,"system":3}}
        return mapping.get(level.lower(), mapping["junior"])
    
    def design_package(self, report:Dict[str,Any], job:Dict[str,Any]):
        level = job.get("level","intern")
        skills = [s["skill"] for s in report["top_skills"]]
        comp = self._level_complexity(level)
        challenges=[]
        if comp["coding"]>0:
            topics=[
                ("Normalize & tokens","Normalize noisy text and compute token frequencies; include tests and complexity notes."),
                ("Dependency resolver","Build a dependency resolver with cycle detection and justify choices."),
                ("Group-by aggregator","Implement group-by aggregations for CSV data without pandas.")
            ]
            t = random.choice(topics)
            lang = "Python" if "python" in skills else "Any"
            challenges.append({"type":"coding","language":lang,"complexity":comp["coding"],"title":t[0],"prompt":t[1]})
        if comp["ml"]>0:
            if "cv" in skills:
                prompt="Design an image-classification pipeline; compare CNN vs transfer learning; discuss augmentation & metrics."
            elif "nlp" in skills or "llm" in skills:
                prompt="Design a text-classification pipeline; compare tf-idf+linear vs transformer fine-tune; discuss evaluation & inference cost."
            else:
                prompt="Choose a dataset; build E2E supervised model; compare models; explain validation & monitoring."
            challenges.append({"type":"ml","complexity":comp["ml"],"prompt":prompt})
        if comp["system"]>0:
            focus = "LLM app with retrieval" if "llm" in skills else "model training + serving"
            challenges.append({"type":"system_design","complexity":comp["system"],"prompt":f"Design a {focus} system for 100k MAU: data flow, scaling, CI/CD, observability, cost."})
        rubric={"weights":{"Problem-solving approach":0.4,"Code quality":0.3,"Communication":0.3},"scale":"0-4 per criterion"}
        if level.lower() in ["mid","senior"]:
            rubric={"weights":{"Problem-solving approach":0.35,"Code quality":0.25,"Communication":0.2,"Architecture & Tradeoffs":0.2},"scale":"0-4 per criterion"}
        bias=[
            "Score only on rubric; ignore identity or affiliations.",
            "Use identical prompts & time windows for comparable candidates.",
            "Provide rubric-aligned feedback; double-mark borderline cases.",
            "Mask PII where feasible."
        ]
        return {"candidate_id":report["candidate_id"], "job_id":job.get("id"), "job_title":job.get("title"), "level":level, "technical_challenges":challenges, "evaluation_framework":rubric, "bias_mitigation_protocol":bias}

class BehavioralAnalyzerAgent:
    SOFT_KEYWORDS = {
        "collaboration":["team","teammate","pair","collaborat","mentor","stakeholder"],
        "communication":["communicat","present","explain","document","clarify","write","talk"],
        "problem_solving":["debug","root cause","investigat","optimiz","design","tradeoff","prototype"]
    }
    def analyze(self, transcript:str):
        tokens = tokenize(transcript)
        counts = {k: sum(any(tok.startswith(stem) for tok in tokens) for stem in stems) for k,stems in self.SOFT_KEYWORDS.items()}
        pos = sum(1 for t in tokens if t in {"clarify","help","learn","thanks","appreciate","together","solution","progress","listen","respect"})
        neg = sum(1 for t in tokens if t in {"blame","delay","excuse","argue","stuck","problem","late"})
        tone = "constructive" if pos>=neg else "cautious"
        quotes=[]
        for sent in re.split(r"[\.!\?]\s+", transcript):
            s=sent.strip()
            if any(k in s.lower() for k in ["team","debug","communicat","design"]) and s:
                quotes.append(s)
            if len(quotes)>=3: break
        signals={"collaboration":"present" if counts["collaboration"]>0 else "unclear", "communication":"present" if counts["communication"]>0 else "unclear", "problem_solving":"present" if counts["problem_solving"]>0 else "unclear"}
        # a simple aggregate soft-skill rating (0-10) — explainable
        soft_score = min(10, 4 + counts["collaboration"]*2 + counts["communication"]*1 + counts["problem_solving"]*2)
        return {"themes":counts, "tone":tone, "signals":signals, "illustrative_quotes":quotes, "soft_skill_score":soft_score, "notes":"Content-only analysis; no demographic signals used."}

class MarketIntelAgent:
    def __init__(self, market:Dict[str,Any]): self.market=market
    def summarize(self):
        demand = sorted(self.market["demand_by_skill"].items(), key=lambda kv:-kv[1])[:6]
        medians = {region:{lvl:int(sum(bounds)//2) for lvl,bounds in levels.items()} for region,levels in self.market["salary_benchmarks"].items()}
        channel_scores={}
        for ch, per in self.market["channel_effectiveness"].items():
            score = sum(per.get(skill,0.0) * demand_val for skill,demand_val in self.market["demand_by_skill"].items())
            channel_scores[ch]=score
        ranked = sorted(channel_scores.items(), key=lambda kv:-kv[1])
        recs=[{"channel":ch,"score":round(sc,2),"top_skills":[k for k,_ in sorted(self.market["channel_effectiveness"][ch].items(), key=lambda kv:-kv[1])[:3]]} for ch,sc in ranked]
        return {"top_demand_skills":demand, "median_salary":medians, "sourcing_recommendations":recs}

# ---------------------------
# UI orchestration
# ---------------------------
st.title("MAIRS — Multi-Agent Intelligent Recruitment System (Final Demo)")

# sidebar controls
with st.sidebar:
    st.header("Controls")
    candidate_names = [c["name"] for c in DATA["candidates"]]
    selected = st.selectbox("Select candidate", candidate_names, index=0)
    job = DATA["jobs"][0]
    show_json = st.checkbox("Show raw JSON tab", True)
    regenerate = st.button("Regenerate assessment (shuffle)")

# fetch candidate
blob = next(c for c in DATA["candidates"] if c["name"]==selected)
candidate = Candidate(**blob)

# run agents
profiler = CandidateProfilerAgent()
assessor = AssessmentDesignerAgent()
behavior = BehavioralAnalyzerAgent()
market_agent = MarketIntelAgent(DATA["market_data"])

report = profiler.build_report(candidate)
assessment = assessor.design_package(report, job)
behavior_report = behavior.analyze(DATA["transcripts"].get(candidate.id,""))
market_summary = market_agent.summarize()

# layout: tabs
tabs = st.tabs(["Summary","Skills","Assessment","Behavioral","Market","Raw JSON" if show_json else "Raw JSON (hidden)"])

# Summary tab
with tabs[0]:
    st.header(f"{candidate.name} — Talent Snapshot")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Career Progression")
        st.write(report["career_progression"])
        st.subheader("Top Skills")
        st.write(", ".join([f"{s['skill']} ({s['confidence']})" for s in report["top_skills"]]) or "No skills detected")
    with col2:
        st.metric("Top skill count", len(report["top_skills"]))
        st.metric("Soft skill score", behavior_report["soft_skill_score"])
        st.markdown("**Job**")
        st.write(job["title"])
        st.write(f"Level: {job['level']}")
    st.download_button("Download candidate report (JSON)", json.dumps({"talent_report":report,"assessment":assessment,"behavior":behavior_report,"market":market_summary}, indent=2), file_name=f"{candidate.id}_full_report.json")

# Skills tab
with tabs[1]:
    st.header("Extracted Skills (confidence and evidence)")
    skills = report["all_skills"]
    if not skills:
        st.info("No skills extracted.")
    else:
        for skill,meta in skills.items():
            conf = meta["confidence"]
            cnt = meta["evidence_count"]
            col_left, col_right = st.columns([4,1])
            with col_left:
                st.write(f"**{skill}** — confidence {conf} — evidence {cnt}")
                # draw a small bar using matplotlib for a nicer look
                prog = conf
                st.progress(int(prog*100)/100.0)
            with col_right:
                st.caption("")  # small spacer

    st.markdown("Explanation: confidence = heuristic function of mention count (and recency boost). More evidence -> higher confidence (capped).")

# Assessment tab
with tabs[2]:
    st.header("Adaptive Assessment Package")
    st.subheader("Technical Challenges")
    for i,ch in enumerate(assessment["technical_challenges"], start=1):
        st.markdown(f"**{i}. [{ch['type'].upper()}] {ch.get('title','').strip()}**")
        st.write(ch.get("prompt",""))
        st.caption(f"Complexity: {ch.get('complexity')}, Language: {ch.get('language','Any')}")
    st.subheader("Evaluation Framework")
    for k,v in assessment["evaluation_framework"]["weights"].items():
        st.write(f"- {k}: {int(v*100)}%")
    st.write("Scale:", assessment["evaluation_framework"]["scale"])
    st.subheader("Bias Mitigation Protocol")
    for b in assessment["bias_mitigation_protocol"]:
        st.write("-", b)

# Behavioral tab
with tabs[3]:
    st.header("Behavioral & Cultural Fit Analysis")
    st.subheader("Simulated transcript")
    st.write(DATA["transcripts"].get(candidate.id,"(no transcript)"))
    st.subheader("Detected themes & signals")
    for k,v in behavior_report["themes"].items():
        st.write(f"- {k}: {v}")
    st.write("Signals:", behavior_report["signals"])
    st.write("Tone:", behavior_report["tone"])
    st.write("Soft-skill score (0-10):", behavior_report["soft_skill_score"])
    if behavior_report["illustrative_quotes"]:
        st.subheader("Illustrative quotes")
        for q in behavior_report["illustrative_quotes"]:
            st.write(">", q)
    st.markdown("> Note: Analysis uses content-only signals. No demographic or affinity features are analyzed.")

# Market tab
with tabs[4]:
    st.header("Market Intelligence & Sourcing Recommendations")
    st.subheader("Top demand skills")
    st.write(", ".join([f"{k} ({v})" for k,v in market_summary["top_demand_skills"]]))
    st.subheader("Median salary by region")
    st.write(market_summary["median_salary"])
    st.subheader("Sourcing recommendations")
    for rec in market_summary["sourcing_recommendations"]:
        st.write(f"- {rec['channel']}: score {rec['score']} — good for {', '.join(rec['top_skills'])}")
    # salary chart
    fig, ax = plt.subplots()
    regions = list(market_summary["median_salary"].keys())
    levels = list(next(iter(market_summary["median_salary"].values())).keys())
    for lvl in levels:
        vals = [market_summary["median_salary"][r][lvl] for r in regions]
        ax.plot(regions, vals, marker='o', label=lvl)
    ax.set_title("Median salary by region & level")
    ax.set_ylabel("Median salary (units vary by region)")
    ax.legend()
    st.pyplot(fig)

# Raw JSON
with tabs[5]:
    if show_json:
        st.header("Raw JSON (combined)")
        st.json({"talent_report":report,"assessment":assessment,"behavior":behavior_report,"market":market_summary})
    else:
        st.info("Raw JSON hidden (enable in sidebar).")

st.markdown("---")
st.caption("Prototype ready for the assignment: clear mapping from requirements -> evidence in the UI.")
