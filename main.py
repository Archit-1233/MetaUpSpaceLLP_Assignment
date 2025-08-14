import json, os, argparse

from agents.profiler import CandidateProfilerAgent, Candidate
from agents.assessment import AssessmentDesignerAgent, JobRole
from agents.behavior import BehavioralAnalyzerAgent
from agents.market import MarketIntelAgent, MarketData

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_md(report, path):
    prof = report["Talent Intelligence Report"]
    assess = report["Assessment Package"]
    beh = report["Behavioral Insights"]
    lines = []
    lines.append(f"# Talent Intelligence Report – {prof['name']}")
    lines.append("## Top Skills")
    for s in prof["top_skills"]:
        lines.append(f"- {s['skill']} (confidence {s['confidence']}, evidence {s['evidence_count']}, source {s['source']})")
    lines.append("## Career Progression")
    lines.append(prof["career_progression"])
    lines.append("\n# Assessment Package")
    lines.append(f"**Role:** {assess['job_title']} ({assess['level']})")
    lines.append("## Technical Challenges")
    for ch in assess["technical_challenges"]:
        lines.append(f"- **{ch['type'].title()}** (complexity {ch['complexity']}) – {ch.get('title','')}")
        lines.append(f"  - {ch['prompt']}")
    lines.append("## Evaluation Framework")
    for k,v in assess["evaluation_framework"]["weights"].items():
        lines.append(f"- {k}: {int(v*100)}%")
    lines.append("Scale: " + assess["evaluation_framework"]["scale"])
    lines.append("## Bias Mitigation Protocol")
    for g in assess["bias_mitigation_protocol"]:
        lines.append(f"- {g}")
    lines.append("\n# Behavioral & Cultural Insights")
    lines.append(f"Tone: **{beh['tone']}**")
    lines.append("Signals: " + ", ".join(f"{k}={v}" for k,v in beh["signals"].items()))
    lines.append("Themes: " + ", ".join(f"{k}={v}" for k,v in beh["themes"].items()))
    if beh.get("illustrative_quotes"):
        lines.append("### Illustrative Quotes")
        for q in beh["illustrative_quotes"]:
            lines.append(f"> {q}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.json")
    parser.add_argument("--output", default="output")
    parser.add_argument("--use-llm", default="false", choices=["true","false"])
    args = parser.parse_args()

    use_llm = args.use_llm.lower() == "true"
    os.makedirs(args.output, exist_ok=True)

    blob = load_data(args.data)
    candidates = [Candidate(**c) for c in blob["candidates"]]
    jobs = [JobRole(**j) for j in blob["jobs"]]
    transcripts = blob["transcripts"]
    md = blob["market_data"]
    market = MarketData(**md)

    profiler = CandidateProfilerAgent(use_llm=use_llm)
    assessor = AssessmentDesignerAgent()
    behavior = BehavioralAnalyzerAgent(use_llm=use_llm)
    market_agent = MarketIntelAgent(market)

    # Market once
    save_json(market_agent.summarize(), os.path.join(args.output, "market_summary.json"))

    for cand in candidates:
        job = jobs[0]
        transcript = transcripts.get(cand.id, "")
        prof = profiler.build_report(cand)
        assess = assessor.design_package(prof, job)
        beh = behavior.analyze(transcript)
        report = {"Talent Intelligence Report": prof, "Assessment Package": assess, "Behavioral Insights": beh}
        save_json(report, os.path.join(args.output, f"{cand.id}_full_report.json"))
        save_md(report, os.path.join(args.output, f"{cand.id}_summary.md"))
    print(f"Reports written to {args.output}")

if __name__ == "__main__":
    main()
