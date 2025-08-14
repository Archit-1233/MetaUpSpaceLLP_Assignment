import json
import streamlit as st

from agents.profiler import CandidateProfilerAgent, Candidate
from agents.assessment import AssessmentDesignerAgent, JobRole
from agents.behavior import BehavioralAnalyzerAgent
from agents.market import MarketIntelAgent, MarketData

DATA_PATH = "data/sample_data.json"

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

st.set_page_config(page_title="MAIRS – Multi-Agent Recruitment", layout="wide")
st.title("🧠 MAIRS – Multi-Agent Intelligent Recruitment System")

blob = load_data(DATA_PATH)
cands = {c["name"]: c for c in blob["candidates"]}
jobs = {j["title"]: j for j in blob["jobs"]}

col1, col2, col3 = st.columns([2,2,1])
with col1:
    cand_name = st.selectbox("Candidate", list(cands.keys()))
with col2:
    job_title = st.selectbox("Job Role", list(jobs.keys()))
with col3:
    use_llm = st.toggle("Use LLM", value=False, help="Uses free Hugging Face pipelines locally")

cand = Candidate(**cands[cand_name])
job = JobRole(**jobs[job_title])
transcripts = blob["transcripts"]
transcript = transcripts.get(cand.id, "")

profiler = CandidateProfilerAgent(use_llm=use_llm)
assessor = AssessmentDesignerAgent()
behavior = BehavioralAnalyzerAgent(use_llm=use_llm)

prof = profiler.build_report(cand)
assess = assessor.design_package(prof, job)
beh = behavior.analyze(transcript)

st.subheader("Talent Intelligence Report")
st.json(prof)

st.subheader("Assessment Package")
st.json(assess)

st.subheader("Behavioral & Cultural Insights")
st.json(beh)

st.subheader("Market Intelligence")
market = MarketData(**blob["market_data"])
market_agent = MarketIntelAgent(market)
st.json(market_agent.summarize())

st.markdown("---")
st.caption("Content-only analysis; no demographic/affinity factors considered.")
