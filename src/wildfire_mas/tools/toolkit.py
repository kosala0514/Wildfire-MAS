"""Tool implementations shared by the MCP servers and the in-process bus.

Write access to the KG is deliberately narrow: only `firms_get_hotspots(store=True)`,
`weather_*(store=True)` and `kg_run_rules()` write, and only into the `live` and
`inferred_live` named graphs. There is no generic "insert" tool.
"""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..kg.cq import CQCatalog, apply_post_filter, render, sufficient
from ..kg.guard import build_guard
from ..kg.rules_live import LiveRuleEngine
from ..kg.store import GraphStore, make_store
from ..live.classify import HotspotClassifier
from ..live.firms import FirmsClient, FirmsError, Hotspot
from ..live.geo import bbox_around, parse_bbox
from ..live.lift import LiveLifter
from ..live.weather import WeatherClient
from ..rag import make_rag

COUNTRY = re.compile(r"^[A-Za-z]{3}$")


class Toolkit:
    def __init__(self, settings: Settings | None = None, store: GraphStore | None = None,
                 firms: FirmsClient | None = None, weather: WeatherClient | None = None,
                 rag=None, llm=None):
        self.s = settings or get_settings()
        self._store = store
        self._firms = firms
        self._weather = weather
        self._rag = rag
        self._llm = llm

    # lazily built dependencies (so an MCP server only builds what it serves)
    @cached_property
    def store(self) -> GraphStore:
        return self._store or make_store(self.s)

    @cached_property
    def firms(self) -> FirmsClient:
        return self._firms or FirmsClient(self.s)

    @cached_property
    def weather(self) -> WeatherClient:
        return self._weather or WeatherClient(self.s)

    @cached_property
    def classifier(self) -> HotspotClassifier:
        return HotspotClassifier.from_settings(self.s)

    @cached_property
    def lifter(self) -> LiveLifter:
        return LiveLifter(self.s)

    @cached_property
    def catalog(self) -> CQCatalog:
        return CQCatalog.load(self.s)

    @cached_property
    def guard(self):
        return build_guard(self.s, self.store)

    @cached_property
    def rag(self):
        return self._rag or make_rag(self.s, self.bridge)

    @cached_property
    def bridge(self):
        """ffonto-v2 qa package (CQ01-CQ26, text-to-SPARQL plus, multimodal RAG), if configured."""
        from ..kg.ffonto_bridge import FfontoBridge

        sparql_model = None
        try:
            slot = self.llm.slot_config("sparql")
            sparql_model = slot.model if slot.provider == "ollama" else None
        except Exception:  # noqa: BLE001
            pass
        return FfontoBridge(self.s, sparql_model)

    def _core(self) -> dict:
        return self.bridge.registry() if self.bridge.configured else {}

    @cached_property
    def llm(self):
        if self._llm is None:
            from ..llm import LLMClient

            self._llm = LLMClient(self.s)
        return self._llm

    # ------------------------------------------------------------------ FIRMS
    def firms_get_hotspots(self, area: str | list[float] | None = None, lat: float | None = None,
                           lon: float | None = None, radius_km: float | None = None, days: int = 1,
                           sources: list[str] | None = None, persistence: bool = True,
                           store: bool = True, max_items: int = 50) -> dict:
        """area: 'w,s,e,n' | [w,s,e,n] | 'world' | 3-letter country code (e.g. 'LKA').
        Or give lat/lon/radius_km. Default: the configured region (Sri Lanka)."""
        if lat is not None and lon is not None:
            area_v: Any = bbox_around(float(lat), float(lon), float(radius_km or 25))
        elif area in (None, "", "default"):
            area_v = tuple(self.s.get("region.default_bbox"))
        elif isinstance(area, str) and (area.lower() == "world"):
            area_v = "world"
        elif isinstance(area, str) and COUNTRY.match(area):
            area_v = area.upper()
        else:
            area_v = parse_bbox(area)
        days = int(days)
        if isinstance(area_v, str) and area_v != "world":
            res = self.firms.get_country(area_v, days, sources)
        else:
            res = self.firms.get_area(area_v, days, sources)
        history: list[Hotspot] | None = None
        warnings: list[str] = list(res.warnings)
        if persistence and res.hotspots and area_v != "world":
            try:
                win = int(self.s.get("hotspot_filter.persistence_window_days", 30))
                hist_src = [(sources or self.firms.default_sources)[0]]
                if isinstance(area_v, str):
                    history = self.firms.get_country(area_v, win, hist_src).hotspots
                else:
                    history = self.firms.get_area(area_v, win, hist_src).hotspots
            except FirmsError as e:
                warnings.append(f"persistence check skipped: {e}")
        self.classifier.classify(res.hotspots, history)
        stored = 0
        if store and res.hotspots:
            doc = self.lifter.hotspots(res)
            self.store.insert_turtle(doc.serialize(), self.s.graph("live"))
            stored = len(doc)
        cats = Counter(h.category for h in res.hotspots)
        ranked = sorted(res.hotspots, key=lambda h: (not h.wildfire_evidence, -(h.frp or 0)))
        return {
            "area": res.area, "days": days, "sources": res.sources, "fetched_at": res.fetched_at,
            "from_cache": res.from_cache, "endpoints": res.urls_redacted,
            "total": len(res.hotspots),
            "likely_wildfire": sum(1 for h in res.hotspots if h.wildfire_evidence),
            "by_category": dict(cats),
            "hotspots": [h.to_dict() for h in ranked[:max_items]],
            "truncated": max(0, len(ranked) - max_items),
            "stored_triples": stored, "warnings": warnings,
        }

    def firms_key_status(self) -> dict:
        return self.firms.key_status()

    # ------------------------------------------------------------------ weather
    def weather_current(self, lat: float, lon: float, site: str | None = None, store: bool = True) -> dict:
        w = self.weather.current(float(lat), float(lon))
        if store:
            self.store.insert_turtle(self.lifter.weather(w, site).serialize(), self.s.graph("live"))
        return w.__dict__

    def weather_fwi(self, lat: float, lon: float, site: str | None = None, store: bool = True) -> dict:
        r = self.weather.fwi(float(lat), float(lon))
        if store:
            # replace earlier versions of the same days (Open-Meteo revises recent hours)
            g = self.s.graph("live")
            for day in r.trend:
                iri = self.lifter.fwi_day_iri(r.lat, r.lon, day.date)
                self.store.update(f"DELETE WHERE {{ GRAPH <{g}> {{ <{iri}> ?p ?o }} }}")
            self.store.insert_turtle(self.lifter.fwi(r, site).serialize(), g)
        return r.to_dict()

    def geo_geocode(self, name: str) -> dict:
        return {"query": name, "results": self.weather.geocode(name)}

    # ------------------------------------------------------------------ KG
    def kg_list_cqs(self, intent: str | None = None) -> dict:
        cqs = self.catalog.for_intent(intent) if intent else list(self.catalog.by_id.values())
        out = [{"id": c.id, "question": c.question, "params": c.params, "required_vars": c.required_vars,
                "source": "live"} for c in cqs]
        if not intent:
            out += [{"id": cid, "question": c["title"], "params": {k: {"type": "string"} for k in c["focus"]},
                     "examples": c["examples"][:3], "source": "ffonto-v2"} for cid, c in self._core().items()]
        return {"cqs": out}

    def kg_run_cq(self, cq_id: str, params: dict) -> dict:
        if cq_id not in self.catalog.by_id and cq_id in self._core():
            # FFOnto CQ01-CQ26: tested queries from ffonto-v2, narrowed by record / event / site / date
            p = params or {}
            res = self.bridge.run_cq(cq_id, record=p.get("record"), event=p.get("eventCode") or p.get("event"),
                                     site=p.get("site"), date=p.get("date"))
            if res["ok"]:
                n = len(res["rows"])
                res.update(sufficient=n > 0, sufficiency=f"{n} row(s)", source="ffonto-v2")
            return res
        cq = self.catalog.get(cq_id)
        query = render(cq, params)
        g = self.guard.check(query)
        if not g.ok:
            return {"cq": cq_id, "ok": False, "errors": g.errors, "sparql": query, "rows": []}
        rows = self.store.select(g.query)
        rows = apply_post_filter(cq, params, rows)
        ok, note = sufficient(cq, rows)
        return {"cq": cq_id, "ok": True, "sparql": g.query, "rows": rows, "sufficient": ok,
                "sufficiency": note, "warnings": g.warnings}

    def kg_run_sparql(self, query: str) -> dict:
        g = self.guard.check(query)
        if not g.ok:
            return {"ok": False, "errors": g.errors, "sparql": query, "rows": []}
        if g.query.lstrip().upper().startswith("ASK") or " ASK " in g.query.upper():
            return {"ok": True, "sparql": g.query, "boolean": self.store.ask(g.query), "rows": []}
        return {"ok": True, "sparql": g.query, "rows": self.store.select(g.query), "warnings": g.warnings}

    def kg_text2sparql(self, question: str, record: str | None = None, hints: list[str] | None = None,
                       date: str | None = None, event: str | None = None, mode: str | None = None) -> dict:
        """ffonto-v2 text-to-SPARQL ("plus" by default): used when no CQ fits the question."""
        if not self.bridge.configured:
            return {"ok": False, "errors": ["ffonto_v2.path not configured"], "rows": [], "sparql": None}
        return self.bridge.text2sparql(question, record, hints, date, event, mode)

    def kg_register_site(self, label: str, lat: float, lon: float) -> dict:
        """Register an ad-hoc query location as a Site in the live graph, so rules R09/R11 and the
        site CQs treat a place name or coordinates exactly like a known site."""
        import hashlib

        from ..kg.turtle import Lit

        data = self.s.ns["data"]
        site = f"{data}querysite/{hashlib.sha1(f'{float(lat):.4f},{float(lon):.4f}'.encode()).hexdigest()[:12]}"
        d = self.lifter.doc()
        d.add_iri(site, "a", "ffo:Site")
        d.add(site, "rdfs:label", Lit(f"query location: {label}", lang="en"))
        d.add(site, "ffo:siteName", f"query location: {label}")   # FFOnto's CQ08/CQ09 show sites by siteName
        d.add(site, "ffo:latitude", round(float(lat), 6))
        d.add(site, "ffo:longitude", round(float(lon), 6))
        d.add(site, "ffo:isQueryLocation", True)
        self.store.insert_turtle(d.serialize(), self.s.graph("live"))
        return {"site": site, "label": f"query location: {label}", "lat": float(lat), "lon": float(lon)}

    def kg_run_rules(self) -> dict:
        run = LiveRuleEngine(self.s, self.store).run(write=True)
        return {"derived": dict(run.derived), "triples": run.triples, "hazards": run.hazards,
                "warnings": run.warnings, "graph": self.s.graph("inferred_live"),
                "ran_at": datetime.now(timezone.utc).isoformat()}

    # ------------------------------------------------------------------ RAG + vision
    def rag_search(self, query: str, k: int | None = None, kinds: list[str] | None = None) -> dict:
        hits = self.rag.search(query, k or int(self.s.get("rag.top_k", 6)) * (2 if kinds else 1))
        if kinds:
            hits = [h for h in hits if h.get("kind", "text") in kinds]
        return {"query": query, "hits": hits[: k or int(self.s.get("rag.top_k", 6))]}

    def vision_read_figure(self, image_path: str, question: str) -> dict:
        import asyncio

        from ..llm import image_part

        p = Path(image_path)
        if not p.is_absolute():
            p = self.s.root / p
        prompt = ("Read this figure to answer the question. Report only values, labels and trends "
                  "that are visible. If the figure does not show it, say 'not shown'.\n"
                  f"Question: {question}\nReply JSON: {{\"answer\": str, \"values\": [{{\"label\": str, "
                  "\"value\": str}}], \"visible\": bool}}")
        out = asyncio.run(self.llm.chat_json(
            "figure_reader", "You read charts and maps precisely.",
            [{"type": "text", "text": prompt}, image_part(p)]))
        return {"image_path": str(image_path), **out}

    def vision_scene_graph(self, image_path: str, concepts: list[str]) -> dict:
        import asyncio

        from ..llm import image_part

        prompt = ("List what is visible in this wildfire scene using ONLY these concepts:\n"
                  + ", ".join(concepts)
                  + "\nReply JSON: {\"entities\": [{\"concept\": str, \"count\": int}], "
                    "\"relations\": [{\"subject\": str, \"predicate\": str, \"object\": str}]}")
        p = Path(image_path)
        out = asyncio.run(self.llm.chat_json(
            "scene_graph", "You describe images using a controlled vocabulary only.",
            [{"type": "text", "text": prompt}, image_part(p if p.is_absolute() else self.s.root / p)]))
        return out


# names exposed to agents -> Toolkit method
TOOL_METHODS = {
    "firms.get_hotspots": "firms_get_hotspots",
    "firms.key_status": "firms_key_status",
    "weather.current": "weather_current",
    "weather.fwi": "weather_fwi",
    "weather.geocode": "geo_geocode",
    "kg.list_cqs": "kg_list_cqs",
    "kg.run_cq": "kg_run_cq",
    "kg.run_sparql": "kg_run_sparql",
    "kg.run_rules": "kg_run_rules",
    "kg.register_site": "kg_register_site",
    "kg.text2sparql": "kg_text2sparql",
    "rag.search": "rag_search",
    "vision.read_figure": "vision_read_figure",
    "vision.scene_graph": "vision_scene_graph",
}
