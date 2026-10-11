"""Orchestrator: plan -> execute (parallel waves) -> synthesize -> verify -> finalize.

Built as a LangGraph StateGraph. If LangGraph is not installed, the same node functions
run in a minimal sequential runner, so tests and model sweeps never depend on it.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from .agents.planner import Planner
from .agents.state import EvidenceLedger, Plan, Request, StepResult
from .agents.synthesis import Answer, Synthesizer, Verifier
from .agents.workers import AGENTS, Deps
from .config import Settings
from .kg.cq import CQCatalog
from .llm import LLMClient, start_call_log
from .tools.bus import BaseBus


class GraphState(TypedDict, total=False):
    request: Request
    plan: Plan
    classification: dict
    ctx: dict
    ledger: EvidenceLedger
    results: dict[str, StepResult]
    answer: Answer
    problems: list[str]
    rewrites: int
    timings: dict[str, float]


@dataclass
class Result:
    question: str
    answer_markdown: str
    sections: list[dict]
    intent: str
    urgency: str
    classification: dict
    plan: dict
    steps: list[dict]
    evidence: list[dict]
    tool_calls: list[dict]
    llm_calls: list[dict]
    issues: list[str]
    timings: dict[str, float]
    context: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


class Orchestrator:
    def __init__(self, settings: Settings, bus: BaseBus, llm: LLMClient, catalog: CQCatalog | None = None,
                 bridge=None):
        self.s = settings
        self.bus = bus
        self.llm = llm
        self.catalog = catalog or CQCatalog.load(settings)
        self.bridge = bridge if bridge is not None else _make_bridge(settings, llm)
        core = self.bridge.registry() if self.bridge is not None and self.bridge.configured else {}
        self.planner = Planner(llm, self.catalog, settings, self.bridge)
        self.synth = Synthesizer(llm)
        self.verifier = Verifier(llm, use_llm=bool(settings.get("orchestrator.verifier_llm", False)))
        self.deps = Deps(bus, llm, settings, self.catalog, core)
        self.max_rewrites = int(settings.get("orchestrator.verifier_rewrites", 1))
        self._graph = self._build_graph()

    # ------------------------------------------------------------------ nodes
    async def n_plan(self, st: GraphState) -> GraphState:
        t0 = time.perf_counter()
        plan, cls = await self.planner.plan(st["request"])
        return {"plan": plan, "classification": cls, "ctx": {}, "ledger": EvidenceLedger(),
                "results": {}, "rewrites": 0, "timings": {**st.get("timings", {}), "plan": time.perf_counter() - t0}}

    async def n_execute(self, st: GraphState) -> GraphState:
        t0 = time.perf_counter()
        plan, req, ctx, led = st["plan"], st["request"], st["ctx"], st["ledger"]
        results: dict[str, StepResult] = {}
        sem = asyncio.Semaphore(int(self.s.get("orchestrator.max_parallel_steps", 4)))
        timeout = float(self.s.get("orchestrator.step_timeout_s", 90))
        optional = {s.id for s in plan.steps if s.optional}

        async def run(step):
            blocked = [a for a in step.after if a in results and not results[a].ok and a not in optional]
            if blocked:
                return StepResult(step.id, step.agent, False, error=f"skipped: dependency {blocked} failed")
            async with sem:
                try:
                    return await asyncio.wait_for(AGENTS[step.agent](step, req, ctx, led, self.deps), timeout)
                except asyncio.TimeoutError:
                    return StepResult(step.id, step.agent, False, error=f"timed out after {timeout:.0f}s")
                except Exception as e:  # noqa: BLE001 - one failed step must not sink the answer
                    return StepResult(step.id, step.agent, False, error=f"{e.__class__.__name__}: {e}")

        for wave in plan.waves():
            for res in await asyncio.gather(*(run(s) for s in wave)):
                results[res.step_id] = res
        return {"results": results, "timings": {**st["timings"], "execute": time.perf_counter() - t0}}

    async def n_synthesize(self, st: GraphState) -> GraphState:
        t0 = time.perf_counter()
        req, plan = st["request"], st["plan"]
        feedback = "\n".join(st.get("problems") or []) or None
        ev = st["ledger"].items
        failed = [f"{r.step_id} ({r.agent}): {r.error}" for r in st["results"].values() if not r.ok]
        q = req.question + (("\n(Steps that failed, report as data gaps: " + "; ".join(failed) + ")") if failed else "")
        ans = await self.synth.write(q, plan.intent, plan.urgency, req.role, ev, feedback)
        rewrites = st.get("rewrites", 0) + (1 if feedback else 0)
        return {"answer": ans, "rewrites": rewrites,
                "timings": {**st["timings"], "synthesize": st["timings"].get("synthesize", 0) + time.perf_counter() - t0}}

    async def n_verify(self, st: GraphState) -> GraphState:
        t0 = time.perf_counter()
        problems = await self.verifier.check(st["answer"], st["ledger"].items)
        return {"problems": problems, "timings": {**st["timings"], "verify": time.perf_counter() - t0}}

    def route_after_verify(self, st: GraphState) -> str:
        if st.get("problems") and st["answer"].source == "llm" and st.get("rewrites", 0) < self.max_rewrites:
            return "synthesize"
        return "finalize"

    async def n_finalize(self, st: GraphState) -> GraphState:
        ans = st["answer"]
        if st.get("problems"):
            ans.issues += [f"verifier: {p}" for p in st["problems"]]
        Verifier.enforce(ans, st["ledger"].items)
        return {"answer": ans}

    async def n_chitchat(self, st: GraphState) -> GraphState:
        text = ("I screen wildfire situations using the FFOnto knowledge graph, live NASA FIRMS hotspots, "
                "computed fire-weather danger, and cited documents. Ask about fires near a place or site, "
                "hazards for a selected record, FWI trends, or wildfire concepts.")
        return {"answer": Answer([{"title": "Reply", "sentences": [{"text": text, "cites": []}]}], "template"),
                "problems": [], "results": {}}

    def route_after_plan(self, st: GraphState) -> str:
        return "chitchat" if st["plan"].intent == "chitchat" else "execute"

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None
        g = StateGraph(GraphState)
        g.add_node("plan", self.n_plan)
        g.add_node("execute", self.n_execute)
        g.add_node("synthesize", self.n_synthesize)
        g.add_node("verify", self.n_verify)
        g.add_node("finalize", self.n_finalize)
        g.add_node("chitchat", self.n_chitchat)
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", self.route_after_plan, {"chitchat": "chitchat", "execute": "execute"})
        g.add_edge("execute", "synthesize")
        g.add_edge("synthesize", "verify")
        g.add_conditional_edges("verify", self.route_after_verify, {"synthesize": "synthesize", "finalize": "finalize"})
        g.add_edge("finalize", END)
        g.add_edge("chitchat", END)
        return g.compile()

    async def _run_fallback(self, st: GraphState) -> GraphState:
        st.update(await self.n_plan(st))
        if self.route_after_plan(st) == "chitchat":
            st.update(await self.n_chitchat(st))
            return st
        st.update(await self.n_execute(st))
        while True:
            st.update(await self.n_synthesize(st))
            st.update(await self.n_verify(st))
            if self.route_after_verify(st) == "finalize":
                break
        st.update(await self.n_finalize(st))
        return st

    # ------------------------------------------------------------------ public
    async def ask(self, req: Request) -> Result:
        llm_log = start_call_log()
        self.bus.calls.clear()
        t0 = time.perf_counter()
        init: GraphState = {"request": req, "timings": {}}
        st = await (self._graph.ainvoke(init) if self._graph is not None else self._run_fallback(init))
        timings = {**st.get("timings", {}), "total": time.perf_counter() - t0}
        plan: Plan = st["plan"]
        led: EvidenceLedger | None = st.get("ledger")
        return Result(
            question=req.question,
            answer_markdown=st["answer"].markdown(),
            sections=st["answer"].sections,
            intent=plan.intent,
            urgency=plan.urgency,
            classification=st.get("classification", {}),
            plan=plan.model_dump(),
            steps=[{"id": r.step_id, "agent": r.agent, "ok": r.ok, "error": r.error, "notes": r.notes,
                    "evidence": [e.id for e in r.evidence]} for r in (st.get("results") or {}).values()],
            evidence=[{"id": e.id, "kind": e.kind, "source": e.source, "summary": e.summary, "query": e.query,
                       "safety_critical": e.safety_critical, "step": e.step_id, "data": _trim(e.data)}
                      for e in (led.items if led else [])],
            tool_calls=[asdict(c) for c in self.bus.calls],
            llm_calls=[asdict(c) for c in llm_log],
            issues=st["answer"].issues,
            timings={k: round(v, 3) for k, v in timings.items()},
            context=st.get("ctx", {}),
        )


def _make_bridge(settings: Settings, llm: LLMClient):
    """ffonto-v2 bridge when ffonto_v2.path is set and its qa package imports; else None."""
    from .kg.ffonto_bridge import FfontoBridge

    sparql_model = None
    try:
        slot = llm.slot_config("sparql")
        sparql_model = slot.model if slot.provider == "ollama" else None
    except Exception:  # noqa: BLE001
        pass
    b = FfontoBridge(settings, sparql_model)
    return b if b.configured and b.available() else None


def _trim(data: Any, n: int = 50) -> Any:
    if isinstance(data, list):
        return data[:n]
    if isinstance(data, dict) and isinstance(data.get("hotspots"), list):
        return {**data, "hotspots": data["hotspots"][:n]}
    return data


def build(settings: Settings | None = None, transport: str = "inprocess", llm: LLMClient | None = None,
          toolkit=None, slot_overrides: dict | None = None) -> Orchestrator:
    from .config import get_settings
    from .tools.bus import InProcessBus, MCPBus
    from .tools.toolkit import Toolkit

    settings = settings or get_settings()
    llm = llm or LLMClient(settings, slot_overrides)
    if transport == "mcp":
        bus: BaseBus = MCPBus()
        bridge = None
    else:
        tk = toolkit or Toolkit(settings, llm=llm)
        bus = InProcessBus(tk)
        bridge = tk.bridge if tk.bridge.configured and tk.bridge.available() else None
    return Orchestrator(settings, bus, llm, bridge=bridge)
