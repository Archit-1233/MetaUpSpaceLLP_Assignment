import re
STOPWORDS=set(''' a an the and or but if on in at for with without to of from is are was were be been being by as into through during his her their our your i you he she they we me him her them us this that these those it its it's
              '''.split())

def tokenize(text:str):
    text=(text or "").lower()
    return [t for t in re.findall(r"[a-zA-Z\+\#\d][a-zA-Z\+\#\d\-]*", text) if t not in STOPWORDS]

def keyword_count(text:str,stems):
    t=(text or "").lower()
    return sum(t.count(s) for s in stems)

def year_from(s: str):
    m = re.search(r"(20\d{2}|19\d{2})", s or "")
    return int(m.group(1)) if m else None

def confidence_from_counts(total_mentions: int, recency_boost: float = 0.0) -> float:
    base = 1 - (0.7 ** max(total_mentions, 0))
    conf = 0.5 + 0.48 * base + recency_boost
    return round(min(conf, 0.98), 2)

