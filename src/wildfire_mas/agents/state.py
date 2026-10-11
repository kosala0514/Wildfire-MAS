"""Shared state, plan schema and evidence model used by all agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Intent = Literal["situational_awareness", "hazard_advice", "data_lookup", "explanation", "trend", "chitchat"]
INTENTS = ["situational_awareness", "hazard_advice", "data_lookup", "explanation", "trend", "chitchat"]
AgentName = Literal["resolve", "live_data", "kg", "risk", "rag"]


class PlanStep(BaseModel):
    id: str
    agent: AgentName
    task: str
    params: dict[str, Any] = Field(default_factory=dict)
    after: list[str] = Field(default_factory=list)
    optional: bool = False       # failure does not block dependants


class Plan(BaseModel):
    intent: Intent
    urgency: Literal["low", "normal", "high"] = "normal"
    steps: list[PlanStep] = Field(default_factory=list)
    synthesis: str = ""

    @field_validator("steps")
    @classmethod
    def _dag(cls, steps: list[PlanStep]) -> list[PlanStep]:
        ids = [s.id for s in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step ids")
        seen: set[str] = set()
        for s in steps:  # steps must be listed in a valid topological order
            missing = [a for a in s.after if a not in seen]
            if missing:
                raise ValueError(f"step {s.id} depends on unknown/later steps {missing}")
            seen.add(s.id)
        return steps

    def waves(self) -> list[list[PlanStep]]:
        done: set[str] = set()
        left = list(self.steps)
        out = []
        while left:
            wave = [s for s in left if set(s.after) <= done]
            if not wave:
                raise ValueError("plan has a cycle")
            out.append(wave)
            done |= {s.id for s in wave}
            left = [s for s in left if s.id not in done]
        return out


@dataclass
class Evidence:
    id: str                      # citation tag, e.g. KG1, LIVE2, R09-1, D3, F1, O1
    kind: Literal["kg", "live", "rule", "doc", "figure", "ontology"]
    summary: str                 # compact text given to the synthesizer
    source: str                  # human-readable provenance (CQ id, endpoint, document...)
    step_id: str
    data: Any = None             # rows / raw payload for the UI
    query: str | None = None     # SPARQL used, if any
    safety_critical: bool = False  # rule-derived hazards: never filtered out (CRM override)
    concepts: list[str] = field(default_factory=list)


@dataclass
class StepResult:
    step_id: str
    agent: str
    ok: bool
    evidence: list[Evidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # IORS rounds, relaxations, fallbacks
    error: str | None = None
    context_updates: dict = field(default_factory=dict)


@dataclass
class Request:
    question: str
    record: str | None = None            # selected image/audio record IRI
    lat: float | None = None
    lon: float | None = None
    place: str | None = None
    area: str | None = None              # 'w,s,e,n' | 'world' | ISO3
    role: str = "analyst"                # incident_commander | analyst | public
    image_path: str | None = None


class EvidenceLedger:
    """Assigns citation IDs and keeps evidence in order."""

    PREFIX = {"kg": "KG", "live": "LIVE", "doc": "D", "figure": "F", "ontology": "O"}

    def __init__(self):
        self.items: list[Evidence] = []
        self._n: dict[str, int] = {}

    def new_id(self, kind: str, rule: str | None = None) -> str:
        key = rule if kind == "rule" and rule else self.PREFIX.get(kind, kind.upper())
        self._n[key] = self._n.get(key, 0) + 1
        return f"{key}-{self._n[key]}" if kind == "rule" else f"{key}{self._n[key]}"

    def add(self, ev: Evidence) -> Evidence:
        self.items.append(ev)
        return ev

    def ids(self) -> set[str]:
        return {e.id for e in self.items}

    def get(self, eid: str) -> Evidence | None:
        return next((e for e in self.items if e.id == eid), None)
