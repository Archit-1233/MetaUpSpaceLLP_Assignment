from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

@dataclass
class MarketData:
    regions: List[str]
    salary_benchmarks: Dict[str, Dict[str, Tuple[int,int]]]  # region -> level -> (p25,p75)
    demand_by_skill: Dict[str, int]
    channel_effectiveness: Dict[str, Dict[str, float]]

class MarketIntelAgent:
    def __init__(self, market: MarketData):
        self.market = market

    def summarize(self) -> Dict[str,Any]:
        top_demand = sorted(self.market.demand_by_skill.items(), key=lambda kv: -kv[1])[:5]
        medians = {region: {lvl: int(sum(bounds)//2) for lvl, bounds in levels.items()} 
                   for region, levels in self.market.salary_benchmarks.items()}
        recs = self._sourcing_recommendations()
        return {"top_demand_skills": top_demand, "median_salary": medians, "sourcing_recommendations": recs}

    def _sourcing_recommendations(self) -> List[Dict[str,Any]]:
        channel_scores = {}
        for channel, per_skill in self.market.channel_effectiveness.items():
            score = 0.0
            for skill, demand in self.market.demand_by_skill.items():
                score += per_skill.get(skill, 0.0) * demand
            channel_scores[channel] = score
        ranked = sorted(channel_scores.items(), key=lambda kv: -kv[1])
        tips = []
        for ch, sc in ranked:
            focus = sorted(self.market.channel_effectiveness[ch].items(), key=lambda kv: -kv[1])[:3]
            tips.append({"channel": ch, "relative_score": round(sc,2), "best_for_skills": [k for k,_ in focus]})
        return tips
