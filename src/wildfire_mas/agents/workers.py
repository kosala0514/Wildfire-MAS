"""Worker agents. Each one runs a plan step, calls tools only through the bus, and returns
evidence items with citation IDs. Iterative Optimization of Retrieval (IORS) lives here:
each agent widens its search along a configured ladder and stops as soon as the evidence
is sufficient."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..kg.cq import CQCatalog
from ..live.geo import haversine_km, parse_wkt_point
from ..llm import LLMClient
from ..tools.bus import BaseBus, ToolError
from .prompts import RAG_GRADER_SYSTEM, SPARQL_SYSTEM
from .state import Evidence, EvidenceLedger, PlanStep, Request, StepResult


@dataclass
class Deps:
    bus: BaseBus
    llm: LLMClient
    settings: Any
    catalog: CQCatalog
    core_cqs: dict = field(default_factory=dict)    # FFOnto CQ01-CQ26 (ffonto-v2), id -> {title, focus, ...}


def short(v: Any) -> str:
    s = str(v)
    if s.startswith("http"):
        return re.split(r"[#/]", s.rstrip("/"))[-1]
    return s


def rows_text(rows: list[dict], limit: int = 15) -> str:
    if not rows:
        return "(no rows)"
    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines = [" | ".join(cols)]
    for r in rows[:limit]:
        lines.append(" | ".join(_cell(r.get(c)) for c in cols))
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more rows")
    return "\n".join(lines)


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    s = " ".join(short(x) for x in str(v).split(" ")) if str(v).startswith("http") else str(v)
    return s[:120]


# ============================================================================ resolve
async def resolve_agent(step: PlanStep, req: Request, ctx: dict, led: EvidenceLedger, d: Deps) -> StepResult:
    r = StepResult(step.id, "resolve", True)
    p = step.params
    lat = req.lat if req.lat is not None else p.get("lat")
    lon = req.lon if req.lon is not None else p.get("lon")
    place = req.place or p.get("place")
    if req.record:
        out = await d.bus.call("kg.run_cq", {"cq_id": "CQL05", "params": {"record": req.record}})
        row = (out.get("rows") or [{}])[0]
        ctx.update(event=row.get("event"), eventCode=row.get("eventCode"),
                   label=f"record {short(req.record)}", record_site=row.get("site"))
        if row.get("wkt"):
            try:
                lat, lon = parse_wkt_point(row["wkt"])
            except (ValueError, IndexError):
                r.notes.append(f"record centerWKT not readable: {row['wkt']}")
        parts = [f"Selected record {short(req.record)}"]
        if lat is not None:
            parts.append(f"was captured at {lat:.4f}, {lon:.4f}")
        if row.get("captureTime"):
            parts.append(f"on {str(row['captureTime'])[:16].replace('T', ' ')} UTC")
        if row.get("eventCode"):
            parts.append(f"(fire event {row['eventCode']})")
        if lat is None:
            parts.append("has no location (no ffonto:centerWKT), so live satellite and weather data cannot be matched to it")
            ctx["no_location"] = True
        r.evidence.append(led.add(Evidence(led.new_id("kg"), "kg", " ".join(parts) + ".", "CQL05", step.id,
                                           out.get("rows"), out.get("sparql"))))
    if lat is None and place:
        g = await d.bus.call("weather.geocode", {"name": place})
        if g["results"]:
            top = g["results"][0]
            lat, lon = float(top["lat"]), float(top["lon"])
            label = ", ".join(x for x in (top["name"], top.get("admin1"), top.get("country")) if x)
            ctx["label"] = label
            r.evidence.append(led.add(Evidence(
                led.new_id("live"), "live", f"'{place}' resolved to {label} at {lat:.4f}, {lon:.4f}",
                "Open-Meteo geocoding", step.id, g["results"][:3])))
        else:
            r.notes.append(f"place '{place}' not found by geocoder")
    if lat is None:
        area = req.area or "default"
        default_label = f"{d.settings.get('region.default_name')} (default region)"
        ctx["area"] = area
        ctx["label"] = ctx.get("label") or (default_label if area == "default" else area)
        r.notes.append(f"no point location; using area {area}")
        r.context_updates = dict(ctx)
        return r
    ctx.update(lat=float(lat), lon=float(lon))
    ctx.setdefault("label", f"{float(lat):.4f}, {float(lon):.4f}")
    if not ctx.get("site"):
        sites = (await d.bus.call("kg.run_cq", {"cq_id": "CQL09", "params": {}})).get("rows", [])
        best = None
        for s in sites:
            try:
                slat, slon = (float(s["lat"]), float(s["lon"])) if s.get("lat") is not None else parse_wkt_point(s["wkt"])
            except (KeyError, ValueError, TypeError):
                continue
            km = haversine_km(lat, lon, slat, slon)
            if km <= 5 and (best is None or km < best[1]):
                best = (s, km)
        if best:
            ctx.update(site=best[0]["site"], label=best[0].get("label") or short(best[0]["site"]))
            r.notes.append(f"matched known site {ctx['label']} ({best[1]:.1f} km)")
        else:
            reg = await d.bus.call("kg.register_site", {"label": ctx["label"], "lat": lat, "lon": lon})
            ctx.update(site=reg["site"])
            r.notes.append("registered query location as a site in the live graph")
    r.context_updates = dict(ctx)
    return r


# ============================================================================ live data
async def live_data_agent(step: PlanStep, req: Request, ctx: dict, led: EvidenceLedger, d: Deps) -> StepResult:
    kind = step.params.get("kind", "hotspots")
    r = StepResult(step.id, "live_data", True)
    if ctx.get("no_location"):
        r.ok, r.error = False, "the selected record has no location, so live data cannot be matched to it"
        return r
    if kind == "fwi":
        if ctx.get("lat") is None:
            r.ok, r.error = False, "FWI needs a point location"
            return r
        f = await d.bus.call("weather.fwi", {"lat": ctx["lat"], "lon": ctx["lon"], "site": ctx.get("site")})
        t = f["today"]
        trend = ", ".join(f"{x['date'][5:]}: {x['fwi']:.1f}" for x in f["trend"])
        ctx["danger"] = f["danger"]
        r.evidence.append(led.add(Evidence(
            led.new_id("live"), "live",
            f"Fire Weather Index at {ctx.get('label')} on {t['date']}: FWI {t['fwi']:.1f} (danger class "
            f"{f['danger']}); FFMC {t['ffmc']:.1f}, DMC {t['dmc']:.1f}, DC {t['dc']:.1f}, ISI {t['isi']:.1f}, "
            f"BUI {t['bui']:.1f}; noon weather {t['temp']:.1f} C, RH {t['rh']:.0f}%, wind {t['wind']:.0f} km/h, "
            f"24 h rain {t['rain']:.1f} mm. FWI last days: {trend}.",
            f"{f['source']}", step.id, f)))
        return r

    days = int(step.params.get("days", 2))
    if ctx.get("lat") is None:
        out = await d.bus.call("firms.get_hotspots", {"area": ctx.get("area"), "days": days})
        r.evidence.append(led.add(_hotspot_evidence(led, out, step.id, ctx, None)))
        return r
    # IORS ladder: widen radius, then days, until at least one likely wildfire (or ladder ends)
    radii = [float(x) for x in d.settings.get("iors.radius_ladder_km", [10, 25, 50])]
    day_ladder = sorted({days, *[int(x) for x in d.settings.get("iors.day_ladder", [1, 2, 5]) if int(x) >= days]})
    max_rounds = int(d.settings.get("iors.max_rounds", 3))
    attempts = [(rad, days) for rad in radii] + [(radii[-1], dd) for dd in day_ladder if dd > days]
    out = None
    tried = []
    for rad, dd in attempts[: max(max_rounds, 1) + 1]:
        out = await d.bus.call("firms.get_hotspots", {"lat": ctx["lat"], "lon": ctx["lon"], "radius_km": rad, "days": dd})
        tried.append(f"{rad:g} km/{dd} d: {out['likely_wildfire']} likely wildfire(s) of {out['total']}")
        out["_radius_km"], out["_days"] = rad, dd
        if out["likely_wildfire"] > 0:
            break
    r.notes.append("IORS rounds: " + "; ".join(tried))
    ctx["hotspot_radius_km"] = out["_radius_km"]
    r.evidence.append(led.add(_hotspot_evidence(led, out, step.id, ctx, tried)))
    return r


def _hotspot_evidence(led: EvidenceLedger, out: dict, step_id: str, ctx: dict, tried: list[str] | None) -> Evidence:
    lat, lon = ctx.get("lat"), ctx.get("lon")
    where = (f"within {out['_radius_km']:g} km of {ctx.get('label')}" if lat is not None
             else f"in {ctx.get('label') or out['area']}")
    cats = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in out["by_category"].items()) or "none"
    lines = [f"NASA FIRMS ({', '.join(out['sources'])}) {where} over the last {out.get('_days', out['days'])} day(s): "
             f"{out['total']} thermal detection(s), {out['likely_wildfire']} classified as likely vegetation fire. "
             f"Categories: {cats}."]
    if tried and len(tried) > 1:
        lines.append("Search widened step by step: " + "; ".join(tried) + ".")
    for h in out["hotspots"][:8]:
        dist = f", {haversine_km(lat, lon, h['latitude'], h['longitude']):.1f} km away" if lat is not None else ""
        lines.append(f"- {h['acq_datetime'][:16].replace('T', ' ')} UTC at {h['latitude']:.3f}, {h['longitude']:.3f}"
                     f"{dist}; {h['category'].replace('_', ' ')}; FRP {h['frp'] if h['frp'] is not None else 'n/a'} MW; "
                     f"confidence {h['confidence_raw']} ({h['source']}). Reason: {h['reasons'][0] if h['reasons'] else ''}")
    if out.get("truncated"):
        lines.append(f"... {out['truncated']} more detections not listed.")
    return Evidence(led.new_id("live"), "live", "\n".join(lines),
                    f"NASA FIRMS, fetched {out['fetched_at'][:16]}Z", step_id, out)


# ============================================================================ risk
async def risk_agent(step: PlanStep, req: Request, ctx: dict, led: EvidenceLedger, d: Deps) -> StepResult:
    r = StepResult(step.id, "risk", True)
    if ctx.get("no_location"):
        r.ok, r.error = False, "the selected record has no location, so live rules cannot be applied to it"
        return r
    out = await d.bus.call("kg.run_rules", {})
    site = ctx.get("site")
    hazards = [h for h in out["hazards"] if not site or h.get("site") == site]
    if not site:
        hazards = hazards[:10]
    r.notes.append(f"rules derived {dict(out['derived'])}; {len(hazards)} relevant to this request")
    r.notes += out.get("warnings", [])
    types = []
    for h in hazards:
        types.append(h["type"])
        r.evidence.append(led.add(Evidence(
            led.new_id("rule", h["rule"]), "rule",
            f"[{h['rule']}] {re.sub(r'(?<!^)(?=[A-Z])', ' ', h['type']).lower()}: {h['explanation']}",
            f"rule {h['rule']} over the live graph", step.id, h, safety_critical=True, concepts=[h["type"]])))
    if not hazards:
        r.evidence.append(led.add(Evidence(
            led.new_id("kg"), "kg",
            f"Live rules R09-R12 derived no hazard for {ctx.get('label') or 'this area'} "
            f"(R09 radius {d.settings.get('rules.R09_near_km')} km).",
            "rules R09-R12", step.id, out)))
    ctx["hazard_types"] = sorted(set(types))
    return r


# ============================================================================ KG
def _cq_params(cq, req: Request, ctx: dict, step: PlanStep) -> dict:
    p: dict[str, Any] = {}
    for name in cq.params:
        if name in step.params.get("cq_params", {}):
            p[name] = step.params["cq_params"][name]
        elif name == "record":
            p[name] = req.record
        elif name in ("site", "lat", "lon", "event"):
            p[name] = ctx.get(name)
        elif name == "radius_km" and ctx.get("hotspot_radius_km"):
            p[name] = ctx["hotspot_radius_km"]
    return {k: v for k, v in p.items() if v is not None}


async def kg_agent(step: PlanStep, req: Request, ctx: dict, led: EvidenceLedger, d: Deps) -> StepResult:
    r = StepResult(step.id, "kg", True)
    cq_id = step.params.get("cq")
    if cq_id and cq_id not in d.catalog.by_id and cq_id in d.core_cqs:
        return await _core_cq(cq_id, step, req, ctx, led, d, r)
    if cq_id:
        cq = d.catalog.get(cq_id)
        params = _cq_params(cq, req, ctx, step)
        missing = [n for n, spec in cq.params.items() if n not in params and "default" not in spec]
        if missing:
            r.ok, r.error = False, f"{cq_id} needs {missing}, not available"
            return r
        out = await d.bus.call("kg.run_cq", {"cq_id": cq_id, "params": params})
        rounds = [f"{cq_id} {params}: {out.get('sufficiency', out.get('errors'))}"]
        # IORS: walk the CQ's relaxation ladder until sufficient
        if out.get("ok") and not out.get("sufficient") and cq.relax:
            for name, ladder in cq.relax.items():
                cur = params.get(name, cq.params.get(name, {}).get("default"))
                for val in [v for v in ladder if cur is None or v > cur]:
                    params[name] = val
                    out = await d.bus.call("kg.run_cq", {"cq_id": cq_id, "params": params})
                    rounds.append(f"{name}={val}: {out.get('sufficiency')}")
                    if out.get("sufficient"):
                        break
                if out.get("sufficient"):
                    break
        r.notes.append("IORS: " + " -> ".join(rounds))
        if not out.get("ok"):
            r.ok, r.error = False, "; ".join(out.get("errors", []))
            return r
        summary = (f"{cq_id} ({cq.question}) for {ctx.get('label') or short(req.record or '')}:\n"
                   + (rows_text(out["rows"]) if out["rows"] else "no evidence in the knowledge graph"))
        r.evidence.append(led.add(Evidence(led.new_id("kg"), "kg", summary, cq_id, step.id,
                                           out["rows"], out.get("sparql"))))
        return r
    if d.core_cqs:
        return await _ffonto_text2sparql(step, req, ctx, led, d, r)
    return await _nl_to_sparql(step, req, ctx, led, d, r)


async def _core_cq(cq_id, step, req, ctx, led, d: Deps, r: StepResult) -> StepResult:
    """FFOnto CQ01-CQ26 via ffonto-v2: tested query narrowed by record / event / site / date.
    IORS for these CQs = drop the narrowest row filter (date, then site) when nothing matches."""
    meta = d.core_cqs[cq_id]
    given = step.params.get("cq_params", {}) or {}
    params = {"record": req.record, "event": given.get("event") or ctx.get("eventCode"),
              "site": given.get("site"), "date": given.get("date")}
    params = {k: v for k, v in params.items() if v}
    out = await d.bus.call("kg.run_cq", {"cq_id": cq_id, "params": params})
    rounds = [f"{cq_id} {_short_params(params)}: {out.get('sufficiency', out.get('errors'))}"]
    for drop in ("date", "site"):
        if out.get("ok") and not out.get("rows") and drop in params:
            params.pop(drop)
            out = await d.bus.call("kg.run_cq", {"cq_id": cq_id, "params": params})
            rounds.append(f"without {drop}: {out.get('sufficiency')}")
    r.notes.append("IORS: " + " -> ".join(rounds))
    if not out.get("ok"):
        r.ok, r.error = False, "; ".join(out.get("errors", []))
        return r
    applied = ", ".join(f"{k} {v}" for k, v in (out.get("filters") or {}).items()) or "no narrowing"
    summary = (f"{cq_id} ({meta['title']}), {applied}:\n"
               + (rows_text(out["rows"]) if out["rows"] else "no evidence in the knowledge graph"))
    r.evidence.append(led.add(Evidence(led.new_id("kg"), "kg", summary, f"{cq_id} (FFOnto)", step.id,
                                       out["rows"], out.get("sparql"))))
    return r


def _short_params(p: dict) -> dict:
    return {k: (short(v) if isinstance(v, str) else v) for k, v in p.items()}


async def _ffonto_text2sparql(step, req, ctx, led, d: Deps, r: StepResult) -> StepResult:
    """No CQ fits: ffonto-v2 text-to-SPARQL "plus" (context binding, schema linking, template
    adaptation from the nearest CQs, execution feedback)."""
    given = step.params.get("cq_params", {}) or {}
    out = await d.bus.call("kg.text2sparql", {
        "question": step.params.get("question") or req.question, "record": req.record,
        "hints": step.params.get("hints") or [], "date": given.get("date"),
        "event": given.get("event") or ctx.get("eventCode")})
    if not out.get("ok"):
        r.ok, r.error = False, f"text-to-SPARQL failed: {out.get('errors')}"
        return r
    r.notes.append(f"answered with text-to-SPARQL ({out.get('mode')}; hints {step.params.get('hints') or []})")
    r.evidence.append(led.add(Evidence(
        led.new_id("kg"), "kg", "Generated query result (no CQ matched):\n" +
        (rows_text(out["rows"]) if out["rows"] else "no evidence in the knowledge graph"),
        "text-to-SPARQL (FFOnto)", step.id, out["rows"], out.get("sparql"))))
    return r


async def _nl_to_sparql(step, req, ctx, led, d: Deps, r: StepResult) -> StepResult:
    """Fallback when no CQ fits: LLM writes SPARQL, guard checks it, one repair attempt."""
    terms = await d.bus.call("kg.run_sparql", {"query": f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?t ?type ?label WHERE {{ GRAPH <{d.settings.graph('ontology')}> {{
          ?t a ?type . OPTIONAL {{ ?t rdfs:label ?label }} FILTER(isIRI(?t)) }} }} LIMIT 400"""})
    vocab = "\n".join(f"{row['t']} ({short(row['type'])})" for row in terms.get("rows", []))
    ns = d.settings.ns
    user = (f"Prefixes: ffo: <{ns['ffo']}> ffv: <{ns['ffv']}>\nTerms:\n{vocab}\n\n"
            f"Selected record: {req.record or 'none'}; site: {ctx.get('site') or 'none'}\n"
            f"Question: {step.params.get('question') or req.question}")
    q = (await d.llm.chat_json("sparql", SPARQL_SYSTEM, user)).get("sparql", "")
    out = await d.bus.call("kg.run_sparql", {"query": q})
    if not out.get("ok"):
        r.notes.append(f"SPARQL rejected: {out.get('errors')}; repairing once")
        q = (await d.llm.chat_json("sparql", SPARQL_SYSTEM,
                                   user + f"\n\nYour query:\n{q}\nwas rejected: {out.get('errors')}. Fix it.")).get("sparql", "")
        out = await d.bus.call("kg.run_sparql", {"query": q})
    if not out.get("ok"):
        r.ok, r.error = False, f"LLM SPARQL rejected after repair: {out.get('errors')}"
        return r
    r.notes.append("answered with LLM-written SPARQL (no CQ matched)")
    r.evidence.append(led.add(Evidence(
        led.new_id("kg"), "kg", "LLM-written query result:\n" + (rows_text(out["rows"]) if out["rows"]
                                                                  else "no evidence in the knowledge graph"),
        "guarded LLM SPARQL", step.id, out["rows"], out.get("sparql"))))
    return r


# ============================================================================ RAG
async def rag_agent(step: PlanStep, req: Request, ctx: dict, led: EvidenceLedger, d: Deps) -> StepResult:
    r = StepResult(step.id, "rag", True)
    query = step.params.get("query") or req.question
    if step.params.get("after_hazards") and ctx.get("hazard_types"):
        query += " " + " ".join(re.sub(r"(?<!^)(?=[A-Z])", " ", t) for t in ctx["hazard_types"])
    k = int(d.settings.get("rag.top_k", 6))
    hits = (await d.bus.call("rag.search", {"query": query, "k": k}))["hits"]
    kept = hits
    # CRAG-style grading + one rewrite (IORS for documents)
    if hits:
        try:
            listing = "\n".join(f"[{h['id']}] {h.get('text', '')[:500]}" for h in hits)
            g = await d.llm.chat_json("rag_grader", RAG_GRADER_SYSTEM, f"Question: {req.question}\n\n{listing}")
            ok_ids = {x["id"] for x in g.get("grades", []) if x.get("relevant")}
            kept = [h for h in hits if h["id"] in ok_ids]
            r.notes.append(f"CRAG: {len(kept)}/{len(hits)} passages relevant")
            if len(kept) < int(d.settings.get("rag.grade_min_relevant", 2)) and g.get("rewrite"):
                more = (await d.bus.call("rag.search", {"query": g["rewrite"], "k": k}))["hits"]
                seen = {h["id"] for h in kept}
                kept += [h for h in more if h["id"] not in seen][: k - len(kept)]
                r.notes.append(f"rewrote query to '{g['rewrite']}'")
        except Exception as e:  # noqa: BLE001 - grading is an optimisation, not required
            r.notes.append(f"grading skipped: {e.__class__.__name__}")
            kept = hits
    for h in kept:
        kind = h.get("kind", "text")
        if kind == "figure":
            ev = Evidence(led.new_id("figure"), "figure",
                          f"Figure from {h.get('source')} p.{h.get('page')}: {h.get('text', '')[:700]}",
                          f"{h.get('source')} p.{h.get('page')}", step.id, h, concepts=h.get("concepts", []))
            if h.get("image_path"):
                try:
                    fig = await d.bus.call("vision.read_figure", {"image_path": h["image_path"], "question": req.question})
                    ev.summary += f"\nRe-read for this question: {fig.get('answer')}"
                except ToolError as e:
                    r.notes.append(f"figure re-read failed: {e}")
            r.evidence.append(led.add(ev))
        else:
            r.evidence.append(led.add(Evidence(
                led.new_id("doc"), "doc", f"{'Table' if kind == 'table' else 'Passage'} from {h.get('source')} "
                f"p.{h.get('page')}: {h.get('text', '')[:900]}", f"{h.get('source')} p.{h.get('page')}",
                step.id, h, concepts=h.get("concepts", []))))
    # ontology facts [O]: definitions and hazard -> action links for concepts in play
    labels = [re.sub(r"(?<!^)(?=[A-Z])", " ", t).lower() for t in ctx.get("hazard_types", [])]
    labels += [w for w in re.findall(r"[a-z]{4,}", req.question.lower())
               if w not in {"what", "which", "should", "does", "about", "there", "with", "near", "from"}][:6]
    if labels:
        rx = "|".join(re.escape(x) for x in dict.fromkeys(labels))
        try:  # ontology facts are a bonus: a failure here must not fail the document step
            o = await d.bus.call("kg.run_sparql", {"query": f"""
                PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
                PREFIX ffo: <{d.settings.ns['ffo']}>
                SELECT ?c ?label ?def (GROUP_CONCAT(DISTINCT ?al; separator="; ") AS ?actions) WHERE {{
                  ?c skos:prefLabel ?label . OPTIONAL {{ ?c skos:definition ?def }}
                  OPTIONAL {{ ?c ffo:mitigatedBy/skos:prefLabel ?al }}
                  FILTER(REGEX(STR(?label), "{rx}", "i")) }} GROUP BY ?c ?label ?def LIMIT 12"""})
        except ToolError as e:
            o = {}
            r.notes.append(f"ontology facts lookup failed: {str(e)[:150]}")
        if o.get("ok") and o.get("rows"):
            r.evidence.append(led.add(Evidence(led.new_id("ontology"), "ontology",
                                               "Ontology facts:\n" + rows_text(o["rows"]), "FFOnto vocabulary",
                                               step.id, o["rows"], o.get("sparql"))))
        elif o.get("ok") is False:
            r.notes.append(f"ontology facts query rejected: {o.get('errors')}")
    if not r.evidence:
        r.notes.append("no relevant documents found")
    return r


AGENTS = {"resolve": resolve_agent, "live_data": live_data_agent, "risk": risk_agent, "kg": kg_agent, "rag": rag_agent}
