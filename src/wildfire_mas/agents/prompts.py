"""Prompts. Intent-Aware Language Tuning (IALT) lives here: every LLM component is
conditioned on the planner's intent label (style, ordering, length, vocabulary limits)."""
from __future__ import annotations

INTENT_DEFS = {
    "situational_awareness": "Is there fire / what is happening now near a place, site or record? Needs live data.",
    "hazard_advice": "What hazards are present and what should crews do? Needs rule-derived hazards and actions.",
    "data_lookup": "A specific stored value or record attribute (scores, thresholds, detections, weather rows).",
    "explanation": "General wildfire knowledge or how something works (FWI, fire behaviour, PPE). Documents.",
    "trend": "How something changed or will change over time (FWI trend, fire growth).",
    "chitchat": "Greetings, thanks, help about the assistant.",
}

PLANNER_SYSTEM = f"""You are the planner of a wildfire screening system. Classify the user's question.
Intents:
{chr(10).join(f'- {k}: {v}' for k, v in INTENT_DEFS.items())}
Urgency is "high" when people or assets may be in danger now (evacuation, nearby fire, crew safety).
Extract a location only if the question names one (place name, coordinates, or "here"/"this site"
when a record is selected). Never invent coordinates.
Reply JSON only:
{{"intent": str, "urgency": "low"|"normal"|"high", "place": str|null, "lat": float|null, "lon": float|null,
  "days": int|null, "cq": str|null, "site": str|null, "date": str|null, "event": str|null, "topic": str}}
"cq" is the best matching competency question id from the menu for data_lookup questions, else null;
the labelled examples are strong hints (questions about the selected record usually map to CQ01-CQ26).
"site", "date" (YYYY-MM-DD) and "event" only when written in the question; otherwise null.
"topic" is a short search query for documents (used for explanation and hazard guidance)."""

PLANNER_LLM_PLAN = """Write an execution plan as JSON for the question below.
Agents: resolve (find site/coordinates), live_data (FIRMS hotspots: params days, radius_km;
FWI: params kind="fwi"), kg (params cq + cq params), risk (run rules, collect hazards),
rag (params query). Steps run in parallel unless "after" lists dependencies; list steps in order.
Reply JSON: {"steps": [{"id": "s1", "agent": str, "task": str, "params": {}, "after": []}],
"synthesis": "what the answer must contain"}"""

SPARQL_SYSTEM = """You write SPARQL 1.1 SELECT queries for the FFOnto knowledge graph in GraphDB.
Use ONLY the classes and properties listed below. Read-only. Add LIMIT 100.
Reply JSON: {"sparql": str, "explanation": str}"""

RAG_GRADER_SYSTEM = """Grade whether each passage helps answer the question. Be strict.
Reply JSON: {"grades": [{"id": str, "relevant": bool}], "rewrite": str|null}
"rewrite" is a better search query if fewer than two passages are relevant."""

# ---- synthesis (CRM): one response schema per intent ----------------------------
RESPONSE_SCHEMAS = {
    "situational_awareness": ["Situation", "Fire danger", "Hazards", "Recommended actions", "Data gaps"],
    "hazard_advice": ["Hazards", "Recommended actions", "Supporting evidence", "Data gaps"],
    "data_lookup": ["Answer", "Context", "Data gaps"],
    "trend": ["Current value", "Trend", "What it means", "Data gaps"],
    "explanation": ["Explanation", "Key figures", "Related hazards and actions"],
}

STYLE = {  # IALT: wording follows intent + urgency + role
    ("high", "incident_commander"): "Short imperative sentences. Lead with the most severe hazard and the action. No background.",
    ("high", "*"): "Short, plain sentences. Put dangers and actions first. No background.",
    ("normal", "*"): "Clear, compact sentences. Facts first, then interpretation.",
    ("low", "*"): "Explanatory prose is fine, but stay concise.",
}


def style_for(urgency: str, role: str) -> str:
    return STYLE.get((urgency, role)) or STYLE.get((urgency, "*")) or STYLE[("normal", "*")]


SYNTH_SYSTEM = """You write the answer of a wildfire screening system from EVIDENCE ONLY.
Rules:
1. Every sentence must cite one or more evidence ids in "cites", e.g. ["LIVE1","R09-1"].
2. Never state a number, place, time, hazard or action that is not in the evidence.
3. If the evidence does not answer part of the question, say so in "Data gaps" (cite nothing there).
4. Hazards marked SAFETY-CRITICAL must appear in the answer, with their actions if given.
5. Satellite hotspots that are NOT likely wildfires (industrial, volcanic, offshore, persistent,
   low confidence) must not be described as wildfires.
6. Use only these section titles, in this order: {sections}
Style: {style}
Reply JSON: {{"sections": [{{"title": str, "sentences": [{{"text": str, "cites": [str]}}]}}]}}"""

VERIFIER_SYSTEM = """Check each sentence against the evidence it cites. A sentence is supported only if
every fact in it (numbers, places, times, hazards, actions) appears in the cited evidence.
Reply JSON: {"unsupported": [{"index": int, "reason": str}]}"""
