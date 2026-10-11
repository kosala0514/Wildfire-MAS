"""Planner agent: intent + urgency + slots (LLM, few-shot), then a typed, validated plan.

planner_mode=template  deterministic plan templates per intent (default; reproducible)
planner_mode=llm       the LLM writes the plan; it is validated against the Plan schema and
                       replaced by the template plan if invalid (recorded in the trace)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..kg.cq import CQCatalog
from ..llm import LLMClient
from ..rag import tokenize
from .prompts import PLANNER_LLM_PLAN, PLANNER_SYSTEM
from .state import INTENTS, Plan, PlanStep, Request

_KEYWORDS = [
    ("chitchat", r"^\s*(hi|hello|hey|thanks|thank you|help|who are you)\b"),
    ("hazard_advice", r"\b(hazard|danger(ous)?|what should (we|i|crews?)|safe|ppe|evacuat|risk to|protect)\b"),
    ("trend", r"\b(trend|increas|decreas|chang|over the (last|past)|growing|forecast|next \d+ days)\b"),
    ("situational_awareness", r"\b(fire[s]? (near|around|in)|active fire|hotspot|burning|now|currently|today|latest)\b"),
    ("data_lookup", r"\b(score|threshold|value of|what (is|was) the|how many|list|which records?)\b"),
    ("explanation", r"\b(what is|explain|how does|why|definition|meaning)\b"),
]


def heuristic_intent(q: str) -> str:
    for intent, pat in _KEYWORDS:
        if re.search(pat, q, re.I):
            return intent
    return "explanation"


def _coords(q: str) -> tuple[float, float] | None:
    m = re.search(r"(-?\d{1,2}\.\d+)\s*[,; ]\s*(-?\d{1,3}\.\d+)", q)
    return (float(m.group(1)), float(m.group(2))) if m else None


# FFOnto CQs run for a selected record, per intent (the record's event is found automatically)
RECORD_CQS = {
    "hazard_advice": ["CQ24", "CQ25", "CQ17"],     # hazards + what they threaten, actions, missing PPE
    "situational_awareness": ["CQ05", "CQ24"],     # detections vs thresholds, hazards
    "trend": [],
}


class Planner:
    def __init__(self, llm: LLMClient, catalog: CQCatalog, settings, bridge=None):
        self.llm = llm
        self.catalog = catalog
        self.s = settings
        self.mode = settings.get("orchestrator.planner_mode", "template")
        self.fewshot = self._load_fewshot(settings.path(settings.get("paths.fewshot_file")))
        self.bridge = bridge if (bridge is not None and bridge.configured) else None
        self.core = self.bridge.registry() if self.bridge else {}
        self.core_menu = self.bridge.menu() if self.bridge else ""

    @staticmethod
    def _load_fewshot(path: Path | None) -> list[dict]:
        if not path or not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def _examples(self, q: str, k: int = 8) -> list[dict]:
        if self.bridge is not None and self.s.get("ffonto_v2.use_examples", True):
            near = self.bridge.nearest_examples(q, k=k, exclude_self=bool(self.s.get("ffonto_v2.exclude_self")))
            if near:   # ForestFire-VQA labelled questions, embedding-nearest (bge-m3)
                return [{"question": e["question"], "intent": e["intent"],
                         "cq": None if e["intent"] == "explanation" else e["label"]} for e in near]
        qt = set(tokenize(q))
        scored = sorted(self.fewshot, key=lambda e: -len(qt & set(tokenize(e["question"]))) /
                        (len(qt | set(tokenize(e["question"]))) or 1))
        return scored[:k]

    async def classify(self, req: Request) -> dict:
        examples = self._examples(req.question)
        ex_txt = "\n".join(json.dumps({"q": e["question"], "intent": e["intent"], "cq": e.get("cq")})
                           for e in examples)
        menu = self.catalog.menu() + (("\n" + self.core_menu) if self.core_menu else "")
        user = (f"Competency questions:\n{menu}\n\nLabelled examples:\n{ex_txt}\n\n"
                f"Selected record: {req.record or 'none'}\nQuestion: {req.question}")
        try:
            out = await self.llm.chat_json("planner", PLANNER_SYSTEM, user)
            if out.get("intent") not in INTENTS:
                raise ValueError(f"bad intent {out.get('intent')}")
            out["_source"] = "llm"
        except Exception as e:  # noqa: BLE001 - planner must never block the pipeline
            out = {"intent": heuristic_intent(req.question), "urgency": "normal", "topic": req.question,
                   "_source": f"heuristic ({e.__class__.__name__})"}
        c = _coords(req.question)
        if c and out.get("lat") is None:
            out["lat"], out["lon"] = c
        if out.get("cq") and out["cq"] not in self.catalog.by_id and out["cq"] not in self.core:
            out["cq"] = None
        out["_hints"] = [e["cq"] for e in examples if e.get("cq")][:3]   # for text-to-SPARQL
        out.setdefault("urgency", "normal")
        out.setdefault("topic", req.question)
        return out

    async def plan(self, req: Request) -> tuple[Plan, dict]:
        cls = await self.classify(req)
        plan = self.template(cls["intent"], cls, req)
        if self.mode == "llm" and cls["intent"] != "chitchat":
            try:
                out = await self.llm.chat_json(
                    "planner", PLANNER_SYSTEM,
                    f"{PLANNER_LLM_PLAN}\nIntent: {cls['intent']}\nCQs:\n{self.catalog.menu()}\n"
                    f"Question: {req.question}")
                plan = Plan(intent=cls["intent"], urgency=cls.get("urgency", "normal"),
                            steps=[PlanStep(**s) for s in out["steps"]], synthesis=out.get("synthesis", ""))
                cls["_plan_source"] = "llm"
            except Exception as e:  # noqa: BLE001
                cls["_plan_source"] = f"template (llm plan rejected: {str(e)[:120]})"
        else:
            cls["_plan_source"] = "template"
        return plan, cls

    # ------------------------------------------------------------------ templates
    def template(self, intent: str, cls: dict, req: Request) -> Plan:
        urg = cls.get("urgency", "normal")
        days = int(cls.get("days") or (5 if intent == "trend" else 2))
        topic = cls.get("topic") or req.question
        has_loc = bool(req.record or req.lat is not None or req.place or cls.get("place")
                       or cls.get("lat") is not None or req.area)
        steps: list[PlanStep] = []

        def add(agent, task, params=None, after=None, optional=False):
            sid = f"s{len(steps) + 1}"
            steps.append(PlanStep(id=sid, agent=agent, task=task, params=params or {},
                                  after=after or [], optional=optional))
            return sid

        if intent == "chitchat":
            return Plan(intent="chitchat", urgency="low", steps=[], synthesis="reply briefly")

        if intent == "explanation":
            add("rag", "find passages, tables and figures", {"query": topic})
            return Plan(intent=intent, urgency=urg, steps=steps,
                        synthesis="explain using documents and ontology facts; cite figures")

        if intent == "data_lookup":
            cq = cls.get("cq")
            narrow = {k: cls.get(k) for k in ("site", "date", "event") if cls.get(k)}
            after = []
            if has_loc and not (cq and cq in self.core):   # FFOnto CQs take the record directly
                after = [add("resolve", "resolve record / place to site and coordinates",
                             {"place": cls.get("place"), "lat": cls.get("lat"), "lon": cls.get("lon")})]
            add("kg", f"answer with {cq or 'text-to-SPARQL (no CQ fits)'}",
                {"cq": cq, "question": req.question, "cq_params": narrow, "hints": cls.get("_hints", [])}, after)
            return Plan(intent=intent, urgency=urg, steps=steps, synthesis="the requested value(s) with context")

        # situational_awareness / hazard_advice / trend: live pipeline
        res = add("resolve", "resolve record / place to site and coordinates",
                  {"place": cls.get("place"), "lat": cls.get("lat"), "lon": cls.get("lon")})
        hs = add("live_data", "FIRMS hotspots around the location", {"kind": "hotspots", "days": days}, [res])
        fw = add("live_data", "current weather and FWI at the location", {"kind": "fwi"}, [res], optional=True)
        # with a selected record, a missing location must not block the record's own FFOnto CQs
        rk = add("risk", "run rules R09-R12 and collect hazards for the site", {}, [hs, fw],
                 optional=bool(req.record))
        add("kg", "live hazards and mitigating actions for the site", {"cq": "CQL02"}, [rk], optional=True)
        if intent == "trend":
            add("kg", "FWI trend at the site", {"cq": "CQL03"}, [rk], optional=True)
            add("kg", "live fire events near the site and growth", {"cq": "CQL04"}, [rk], optional=True)
        if intent == "situational_awareness":
            add("kg", "live fire events near the site", {"cq": "CQL04"}, [rk], optional=True)
        if req.record:
            # FFOnto's own CQs about the selected record (offline image/audio evidence) next to live data;
            # they run after the rules so CQ24/CQ25 also list the live hazards of the record's event
            for cid in RECORD_CQS.get(intent, []):
                if cid in self.core:
                    add("kg", f"{cid} ({self.core[cid]['title']}) for the selected record", {"cq": cid},
                        [rk], optional=True)
            add("kg", "satellite vs image conflict for the record", {"cq": "CQL08"}, [rk], optional=True)
        if intent == "hazard_advice":
            add("rag", "guidance documents for the hazards", {"query": topic, "after_hazards": True}, [rk],
                optional=True)
        synth = {"situational_awareness": "situation, danger level, hazards, actions; cite all",
                 "hazard_advice": "hazards with actions first; cite all",
                 "trend": "current value, trend direction and meaning; cite all"}[intent]
        return Plan(intent=intent, urgency=urg, steps=steps, synthesis=synth)
