"""Integration with the real ffonto-v2 project (runs only when FFONTO_V2_PATH points to it).

Checks that the live module speaks FFOnto: every term in the live CQs and rule inputs is defined
in FFOnto core/vocab or the live module, FFOnto's own CQ01-CQ26 load through the bridge, and the
KG agent narrows them by record / event exactly as ffonto-v2 does.
"""
import os
import re
from pathlib import Path

V2 = os.environ.get("FFONTO_V2_PATH")


def _settings(settings):
    settings.raw.setdefault("ffonto_v2", {})["path"] = V2
    return settings


def test_live_terms_exist_in_ffonto(settings):
    if not V2:
        return
    from wildfire_mas.kg.cq import CQCatalog, render
    from wildfire_mas.kg.guard import build_guard
    from wildfire_mas.kg.rules_live import align_namespaces

    s = _settings(settings)
    g = build_guard(s)
    bad = {}
    for cq in CQCatalog.load(s).by_id.values():
        params = {k: ("http://purl.org/ffonto/data/x" if v.get("type") == "iri" else 1)
                  for k, v in cq.params.items()}
        params.update(lat=7.3, lon=80.6, radius_km=10, lat_min=7, lat_max=8, lon_min=80, lon_max=81)
        res = g.check(render(cq, params))
        if not res.ok:
            bad[cq.id] = res.errors
    for f in sorted((s.root / "rules").glob("inputs_*.rq")):
        res = g.check(align_namespaces(f.read_text(), s))
        if not res.ok:
            bad[f.name] = res.errors
    for f in sorted((Path(V2) / "queries").glob("CQ*.rq")):   # FFOnto's own queries: guard must accept them
        res = g.check(f.read_text())
        if not res.ok:
            bad[f.name] = res.errors
    assert not bad, bad


def test_bridge_loads_ffonto_cqs_and_narrows(settings, monkeypatch=None):
    if not V2:
        return
    from wildfire_mas.kg.ffonto_bridge import FfontoBridge, intent_for_label

    s = _settings(settings)
    b = FfontoBridge(s)
    reg = b.registry()
    assert {"CQ01", "CQ17", "CQ24", "CQ25", "CQ26", "CQ07b"} <= set(reg) and len(reg) == 27
    assert "record" in reg["CQ17"]["focus"] and "eventCode" in reg["CQ24"]["focus"]
    seen = []
    kg = b.qa["kg"]
    real_select = kg.select

    def fake_select(q, timeout=60):
        seen.append(q)
        if "ffonto:eventCode ?e" in q and "LIMIT 1" in q:          # kg.event_of(record)
            return ["e"], [{"e": "ev_demo_081"}]
        return ["record", "missingRequiredPPE"], [{"record": "ffd:record/x", "missingRequiredPPE": "Gloves"}]

    kg.select = fake_select
    try:
        rec = "http://purl.org/ffonto/data/record/00000000-0000-4000-8000-000000000081"
        r17 = b.run_cq("CQ17", record=rec)
        assert r17["ok"] and r17["sparql"].rstrip().endswith(f"VALUES ?record {{ <{rec}> }}")
        r24 = b.run_cq("CQ24", record=rec)       # no ?record column: narrowed by the record's event
        assert 'VALUES ?eventCode { "ev_demo_081" }' in r24["sparql"]
    finally:
        kg.select = real_select
    assert intent_for_label("CQ24") == "hazard_advice" and intent_for_label("OOS") == "explanation"
    assert intent_for_label("CQ09") == "data_lookup"


def test_ffonto_labelled_questions(settings):
    if not V2:
        return
    from wildfire_mas.kg.ffonto_bridge import FfontoBridge

    ex = FfontoBridge(_settings(settings)).all_examples()
    assert len(ex) > 900
    labels = {e["label"] for e in ex}
    assert "OOS" in labels and "CQ24" in labels
    assert not re.search(r"\bCQ\d", " ".join(e["intent"] for e in ex))


def test_planner_uses_ffonto_cqs_for_a_record(settings):
    if not V2:
        return
    import asyncio
    import json

    from wildfire_mas.agents.planner import Planner
    from wildfire_mas.agents.state import Request
    from wildfire_mas.kg.cq import CQCatalog
    from wildfire_mas.kg.ffonto_bridge import FfontoBridge
    from wildfire_mas.llm import ScriptedLLM

    s = _settings(settings)
    seen = {}

    def planner_reply(messages):
        seen["prompt"] = messages[-1]["content"]
        return json.dumps({"intent": "hazard_advice", "urgency": "high", "cq": "CQ24"})

    p = Planner(ScriptedLLM(s, {"planner": planner_reply}), CQCatalog.load(s), s, FfontoBridge(s))
    rec = "http://purl.org/ffonto/data/record/00000000-0000-4000-8000-000000000081"
    plan, cls = asyncio.run(p.plan(Request("What hazards are present and what should we do?", record=rec)))
    assert "CQ24: Which hazards exist" in seen["prompt"] and "CQL02" in seen["prompt"]
    kg_cqs = [st.params.get("cq") for st in plan.steps if st.agent == "kg"]
    assert {"CQ24", "CQ25", "CQ17", "CQL02", "CQL08"} <= set(kg_cqs), kg_cqs
    risk = next(st for st in plan.steps if st.agent == "risk")
    assert risk.optional    # a record without a location must not block CQ24/CQ25/CQ17

    # data lookup on an FFOnto CQ goes straight to the KG agent (no live resolve step)
    p2 = Planner(ScriptedLLM(s, {"planner": [json.dumps({"intent": "data_lookup", "cq": "CQ05"})]}),
                 CQCatalog.load(s), s, FfontoBridge(s))
    plan2, _ = asyncio.run(p2.plan(Request("Is the fire score above its threshold?", record=rec)))
    assert [st.agent for st in plan2.steps] == ["kg"] and plan2.steps[0].params["cq"] == "CQ05"
