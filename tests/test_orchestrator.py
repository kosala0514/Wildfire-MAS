"""End-to-end orchestration with scripted tools and a scripted model (no network, no GPU)."""
import asyncio
import json
import re

from fakes import ScriptedBus

from wildfire_mas.agents.state import Plan, PlanStep, Request
from wildfire_mas.llm import ScriptedLLM
from wildfire_mas.orchestrator import Orchestrator

SITE = "http://purl.org/ffonto/data/querysite/abc"


def _hotspots(args):
    r = args.get("radius_km", 10)
    found = r >= 25
    hs = [{"id": "h1", "latitude": 7.35, "longitude": 80.70, "acq_datetime": "2026-10-08T07:48:00+00:00",
           "category": "likely_vegetation_fire", "frp": 14.2, "confidence_raw": "h", "source": "VIIRS_NOAA20_NRT",
           "reasons": ["no static/volcanic/offshore match"], "wildfire_evidence": True}] if found else []
    hs.append({"id": "h9", "latitude": 7.29, "longitude": 80.64, "acq_datetime": "2026-10-08T07:48:00+00:00",
               "category": "persistent_static_source", "frp": 30.0, "confidence_raw": "n",
               "source": "VIIRS_NOAA20_NRT", "reasons": ["detected on 14 distinct days"], "wildfire_evidence": False})
    return {"area": "bbox", "days": args.get("days", 2), "sources": ["VIIRS_NOAA20_NRT"],
            "fetched_at": "2026-10-09T05:00:00+00:00", "from_cache": False, "endpoints": [],
            "total": len(hs), "likely_wildfire": int(found),
            "by_category": {"persistent_static_source": 1, **({"likely_vegetation_fire": 1} if found else {})},
            "hotspots": hs, "truncated": 0, "stored_triples": 20, "warnings": []}


def _bus():
    def run_cq(a):
        cq = a["cq_id"]
        if cq == "CQL09":
            return {"ok": True, "rows": [], "sufficient": False}
        if cq == "CQL02":
            return {"ok": True, "sufficient": True, "sparql": "SELECT ...", "rows": [
                {"hazard": "nearby active fire", "rule": "R09", "actions": "monitor the fire front | alert crews"}]}
        return {"ok": True, "sufficient": False, "sparql": "SELECT ...", "rows": []}

    return ScriptedBus({
        "weather.geocode": {"results": [{"name": "Kandy", "admin1": "Central", "country": "Sri Lanka",
                                         "lat": 7.2906, "lon": 80.6337}]},
        "kg.run_cq": run_cq,
        "kg.register_site": {"site": SITE, "label": "query location: Kandy", "lat": 7.2906, "lon": 80.6337},
        "firms.get_hotspots": _hotspots,
        "weather.fwi": {"today": {"date": "2026-10-09", "fwi": 24.3, "ffmc": 90.1, "dmc": 30.2, "dc": 210.0,
                                  "isi": 9.8, "bui": 45.0, "temp": 31.0, "rh": 48.0, "wind": 14.0, "rain": 0.0},
                        "danger": "High", "trend": [{"date": "2026-10-08", "fwi": 22.0}, {"date": "2026-10-09", "fwi": 24.3}],
                        "source": "open-meteo"},
        "kg.run_rules": {"derived": {"R09": 1}, "triples": 9, "warnings": [], "hazards": [
            {"id": "hz1", "rule": "R09", "type": "NearbyActiveFire", "site": SITE,
             "explanation": "1 likely vegetation-fire hotspot within 10 km of query location: Kandy; closest 8.7 km.",
             "supported_by": ["h1"]}]},
    })


def _llm(settings, synth_replies):
    planner = json.dumps({"intent": "situational_awareness", "urgency": "high", "place": "Kandy",
                          "lat": None, "lon": None, "days": None, "cq": None, "topic": "fires near Kandy"})
    return ScriptedLLM(settings, {"planner": [planner], "synthesizer": synth_replies})


def _ids(messages):
    return re.findall(r"^\[([A-Z0-9-]+)\]", messages[-1]["content"], re.M)


def test_situational_question_end_to_end(settings):
    def synth_first(messages):  # omits the safety-critical hazard on purpose
        ids = [i for i in _ids(messages) if not i.startswith("R09")]
        return json.dumps({"sections": [
            {"title": "Situation", "sentences": [{"text": "One likely vegetation fire about 8.7 km away.", "cites": ids[:2]}]},
            {"title": "Data gaps", "sentences": [{"text": "No on-scene imagery.", "cites": []}]}]})

    def synth_second(messages):
        assert "safety-critical evidence R09-1" in messages[-1]["content"]
        ids = _ids(messages)
        return json.dumps({"sections": [
            {"title": "Situation", "sentences": [{"text": "One likely vegetation fire about 8.7 km away.", "cites": [ids[1]]}]},
            {"title": "Hazards", "sentences": [{"text": "Nearby active fire.", "cites": ["R09-1"]}]},
            {"title": "Recommended actions", "sentences": [{"text": "Monitor the fire front and alert crews.", "cites": ["KG1"]}]}]})

    llm = _llm(settings, [])
    replies = iter([synth_first, synth_second])
    llm.script["synthesizer"] = lambda m: next(replies)(m)
    orch = Orchestrator(settings, _bus(), llm)
    orch._graph = None  # exercise the built-in runner (LangGraph path uses the same nodes)
    res = asyncio.run(orch.ask(Request("Are there any fires near Kandy right now?")))

    assert res.intent == "situational_awareness" and res.urgency == "high"
    assert [s["agent"] for s in res.plan["steps"]][:4] == ["resolve", "live_data", "live_data", "risk"]
    hot = next(s for s in res.steps if s["agent"] == "live_data" and any("IORS" in n for n in s["notes"]))
    assert "10 km" in hot["notes"][0] and "25 km" in hot["notes"][0]  # widened once, then stopped
    assert "R09-1" in res.answer_markdown and "Hazards" in res.answer_markdown
    ev = {e["id"]: e for e in res.evidence}
    assert ev["R09-1"]["safety_critical"]
    assert any("persistent static source" in e["summary"] for e in res.evidence)
    assert sum(1 for c in res.llm_calls if c["slot"] == "synthesizer") == 2  # one verifier-driven rewrite
    assert not [i for i in res.issues if "appended" in i]


def test_synth_failure_falls_back_to_extractive_and_keeps_hazards(settings):
    llm = _llm(settings, ["this is not json", "still not json"])
    orch = Orchestrator(settings, _bus(), llm)
    orch._graph = None
    res = asyncio.run(orch.ask(Request("Are there any fires near Kandy right now?")))
    assert any("extractive" in i for i in res.issues)
    assert "R09-1" in res.answer_markdown
    for s in res.sections:
        for x in s["sentences"]:
            assert x["cites"] or s["title"] == "Data gaps"


def test_planner_fallback_and_chitchat(settings):
    llm = ScriptedLLM(settings, {"planner": ["garbage", "garbage"]})
    orch = Orchestrator(settings, ScriptedBus({}), llm)
    orch._graph = None
    res = asyncio.run(orch.ask(Request("hello")))
    assert res.intent == "chitchat" and res.classification["_source"].startswith("heuristic")
    assert "FIRMS" in res.answer_markdown


def test_failed_step_becomes_data_gap(settings):
    bus = _bus()
    bus.handlers["weather.fwi"] = lambda a: (_ for _ in ()).throw(RuntimeError("open-meteo down"))
    seen = {}

    def synth(messages):
        seen["q"] = messages[-1]["content"]
        return json.dumps({"sections": [{"title": "Hazards", "sentences": [{"text": "Nearby active fire.", "cites": ["R09-1"]}]}]})

    orch = Orchestrator(settings, bus, _llm(settings, synth))
    orch._graph = None
    res = asyncio.run(orch.ask(Request("Are there any fires near Kandy right now?")))
    fwi = next(s for s in res.steps if s["agent"] == "live_data" and not s["ok"])
    assert "open-meteo down" in fwi["error"]
    assert "Steps that failed" in seen["q"]
    assert next(s for s in res.steps if s["agent"] == "risk")["ok"]  # optional dependency did not block


def test_plan_schema_rejects_bad_dag():
    import pytest

    with pytest.raises(Exception):
        Plan(intent="trend", steps=[PlanStep(id="s1", agent="kg", task="x", after=["s2"]),
                                    PlanStep(id="s2", agent="kg", task="y")])
    p = Plan(intent="trend", steps=[PlanStep(id="a", agent="resolve", task="x"),
                                    PlanStep(id="b", agent="live_data", task="y", after=["a"]),
                                    PlanStep(id="c", agent="live_data", task="z", after=["a"]),
                                    PlanStep(id="d", agent="risk", task="w", after=["b", "c"])])
    assert [[s.id for s in w] for w in p.waves()] == [["a"], ["b", "c"], ["d"]]
