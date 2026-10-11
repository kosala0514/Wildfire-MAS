"""SPARQL guard, CQ rendering / sufficiency, and live rules R09-R12."""
from datetime import datetime, timedelta, timezone

from fakes import RoutingStore

from wildfire_mas.kg.cq import CQCatalog, apply_post_filter, render, sufficient
from wildfire_mas.kg.guard import build_guard
from wildfire_mas.kg.rules_live import LiveRuleEngine

FFO = "http://purl.org/ffonto#"


def test_guard_blocks_writes_service_and_unknown_terms(settings):
    g = build_guard(settings)
    ok = g.check(f"PREFIX ffo: <{FFO}> SELECT ?h WHERE {{ ?h a ffo:SatelliteHotspot ; ffo:frpMW ?f }}")
    assert ok.ok, ok.errors
    assert ok.query.rstrip().endswith("LIMIT 500")
    assert not g.check(f"PREFIX ffo: <{FFO}> DELETE WHERE {{ ?s ffo:frpMW ?o }}").ok
    assert not g.check("SELECT * WHERE { SERVICE <http://evil.example/sparql> { ?s ?p ?o } }").ok
    bad = g.check(f"PREFIX ffo: <{FFO}> SELECT ?x WHERE {{ ?x ffo:madeUpProperty ?y }}")
    assert not bad.ok and any("madeUpProperty" in e for e in bad.errors)
    capped = g.check(f"PREFIX ffo: <{FFO}> SELECT ?h WHERE {{ ?h a ffo:SatelliteHotspot }} LIMIT 99999")
    assert capped.ok and "LIMIT 500" in capped.query
    # undeclared standard prefix is added automatically
    auto = g.check("SELECT ?c ?l WHERE { ?c skos:prefLabel ?l }")
    assert auto.ok and "PREFIX skos:" in auto.query
    # strings and data IRIs are fine
    s = g.check(f'PREFIX ffo: <{FFO}> SELECT ?h WHERE {{ ?h ffo:fromProduct "DELETE:x" . '
                f'FILTER(?h != <{settings.ns["data"]}hotspot/abc>) }}')
    assert s.ok, s.errors


def test_cq_render_types_and_bbox(settings):
    cat = CQCatalog.load(settings)
    cq = cat.get("CQL01")
    q = render(cq, {"lat": 7.3, "lon": 80.63})
    assert "{{" not in q and "^^<http://www.w3.org/2001/XMLSchema#dateTime>" in q
    assert "7.21" in q  # lat_min for 10 km default radius
    import pytest

    with pytest.raises(ValueError):
        render(cat.get("CQL02"), {"site": "not an iri"})
    rows = [{"h": "a", "lat": 7.30, "lon": 80.64}, {"h": "b", "lat": 7.38, "lon": 80.70}]
    kept = apply_post_filter(cq, {"lat": 7.3, "lon": 80.63, "radius_km": 5}, rows)
    assert [r["h"] for r in kept] == ["a"] and kept[0]["distance_km"] < 2
    assert sufficient(cq, kept)[0] and not sufficient(cq, [])[0]
    assert {"CQL02", "CQL03"} <= {c.id for c in cat.for_intent("trend") + cat.for_intent("hazard_advice")}


def _rule_store(now):
    site = "http://purl.org/ffonto/data/site/s1"
    t1, t2 = now - timedelta(hours=12), now - timedelta(hours=1)
    hotspots = [
        {"h": "h1", "lat": 7.300, "lon": 80.630, "t": t1.isoformat(), "frp": 5.0, "product": "VIIRS_NOAA20_NRT"},
        {"h": "h2", "lat": 7.302, "lon": 80.633, "t": t2.isoformat(), "frp": 9.0, "product": "VIIRS_NOAA20_NRT"},
        {"h": "h3", "lat": 7.305, "lon": 80.636, "t": t2.isoformat(), "frp": 7.0, "product": "VIIRS_NOAA20_NRT"},
        {"h": "far", "lat": 8.900, "lon": 81.000, "t": t2.isoformat(), "frp": 3.0, "product": "MODIS_NRT"},
    ]
    return site, RoutingStore([
        ("?h ?lat ?lon ?t ?frp ?product", hotspots),
        ("?site ?label ?lat ?lon ?wkt", [{"site": site, "label": "Knuckles ridge", "lat": 7.33, "lon": 80.65}]),
        ("?site ?a ?date ?fwi ?danger", [{"site": site, "a": "fwi1", "date": "2026-10-08", "fwi": 31.0,
                                          "danger": "http://purl.org/ffonto/vocab#FWIDangerHigh"}]),
        ("?rec ?wkt ?t ?score ?threshold ?event", [{"rec": "img_081", "wkt": "POINT(80.633 7.302)",
                                                    "t": t2.isoformat(), "score": 0.12, "threshold": 0.3,
                                                    "event": "ev_081"}]),
    ])


def test_rules_r09_to_r12(settings):
    now = datetime.now(timezone.utc)
    site, store = _rule_store(now)
    run = LiveRuleEngine(settings, store).run(write=True)
    by_rule = {}
    for h in run.hazards:
        by_rule.setdefault(h["rule"], []).append(h)
    assert set(by_rule) == {"R09", "R10", "R11", "R12"}, by_rule.keys()
    r09 = by_rule["R09"][0]
    assert r09["site"] == site and r09["hotspotCount"] == 3  # 'far' hotspot excluded
    assert by_rule["R10"][0]["type"] == "SpreadingFire" and by_rule["R10"][0]["frpGrowthRatio"] == 3.2
    assert by_rule["R11"][0]["raisedDangerClass"] == "VeryHigh"
    assert by_rule["R12"][0]["event"] == "ev_081" and by_rule["R12"][0]["distanceKm"] < 1
    assert run.derived["R10_event"] == 2  # main cluster + far hotspot
    assert store.cleared == [settings.graph("inferred_live")]
    graph, ttl = store.inserted[0]
    assert graph == settings.graph("inferred_live") and "ffo:LiveFireEvent" in ttl and 'ffo:derivedByRule "R11"' in ttl
    # FFOnto's hazard model, so CQ24/CQ25 list live hazards
    assert "a ffo:Hazard" in ttl and "ffo:hasHazardType ffv:ElevatedDangerNearActiveFire" in ttl
    assert "prov:wasDerivedFrom" in ttl and "rdfs:comment" in ttl and "ffo:raisedDangerClass ffv:FWIDangerVeryHigh" in ttl
    assert 'ffo:eventCode "live-' in ttl and "ffo:inEvent" in ttl


def test_rules_no_hazard_when_fire_is_far(settings):
    now = datetime.now(timezone.utc)
    site, store = _rule_store(now)
    store.routes[0] = ("?h ?lat ?lon ?t ?frp ?product",
                       [{"h": "far", "lat": 8.9, "lon": 81.0, "t": now.isoformat(), "frp": 3.0, "product": "MODIS_NRT"}])
    run = LiveRuleEngine(settings, store).run(write=False)
    assert not [h for h in run.hazards if h["rule"] in ("R09", "R11")]
