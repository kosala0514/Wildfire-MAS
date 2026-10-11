"""Live-data rules R09-R12 (portable Python engine; no GeoSPARQL plugin required).

Each run reads its inputs with the editable queries in rules/inputs_*.rq, computes the
derivations, and *replaces* the inferred-live named graph, so results are idempotent and
always reflect the current live graph. Hazards use FFOnto's own hazard model
(ffonto:Hazard + hasHazardType + inEvent + prov:wasDerivedFrom + rdfs:comment), so FFOnto's CQ24
and CQ25 list live hazards and their actions next to the image/audio hazards of rules R06/R07.
Every hazard also records its rule ID, which the synthesizer cites.

R09 NearbyActiveFire     likely-vegetation hotspot within R09_near_km of a Site
R10 LiveFireEvent        hotspots clustered within R10_cluster_km and R10_cluster_hours;
    SpreadingFire        latest-pass FRP / previous-pass FRP >= R10_frp_growth_ratio
R11 ElevatedDangerNearActiveFire  site has an R09 hazard AND its latest FWI danger class
                         >= R11_min_danger (danger level raised by one class)
R12 SatelliteSceneConflict  an image's Fire observation is below its threshold, but a
                         likely-vegetation hotspot lies within R12_match_km of the image's
                         centerWKT (and within R12_match_hours of its captureTime, when it has one)
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..config import Settings
from ..live.fwi import class_rank
from ..live.geo import haversine_km, parse_wkt_point
from .store import GraphStore
from .turtle import Lit, TurtleDoc

XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"

DEFAULT_FFO = "http://purl.org/ffonto#"
DEFAULT_FFV = "http://purl.org/ffonto/vocab#"


def align_namespaces(text: str, settings: Settings) -> str:
    return text.replace(DEFAULT_FFO, settings.ns["ffo"]).replace(DEFAULT_FFV, settings.ns["ffv"])


def _dt(v) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _h(*p) -> str:
    return hashlib.sha1("|".join(map(str, p)).encode()).hexdigest()[:16]


@dataclass
class RuleRun:
    derived: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    hazards: list[dict] = field(default_factory=list)
    triples: int = 0
    warnings: list[str] = field(default_factory=list)


class LiveRuleEngine:
    def __init__(self, settings: Settings, store: GraphStore):
        self.s = settings
        self.store = store
        self.cfg = settings.get("rules")
        self.classes = settings.get("danger.fwi_classes")
        self.rules_dir = settings.root / settings.get("paths.rules_dir")
        self.data = settings.ns["data"]

    def _q(self, name: str) -> list[dict]:
        text = align_namespaces((self.rules_dir / name).read_text(), self.s)
        return self.store.select(text)

    # -- inputs ----------------------------------------------------------------------
    def _sites(self) -> list[dict]:
        out = []
        for r in self._q("inputs_sites.rq"):
            if "lat" in r and "lon" in r:
                lat, lon = float(r["lat"]), float(r["lon"])
            else:
                lat, lon = parse_wkt_point(r["wkt"])
            out.append({"site": r["site"], "label": r.get("label") or r["site"].rsplit("/", 1)[-1],
                        "lat": lat, "lon": lon})
        return out

    def _hotspots(self) -> list[dict]:
        return [{"h": r["h"], "lat": float(r["lat"]), "lon": float(r["lon"]), "t": _dt(r["t"]),
                 "frp": float(r.get("frp") or 0.0), "product": r.get("product", "")}
                for r in self._q("inputs_hotspots.rq")]

    def _latest_fwi(self) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for r in self._q("inputs_fwi.rq"):
            if r["site"] not in latest:
                latest[r["site"]] = {"a": r["a"], "date": r["date"], "fwi": float(r["fwi"]),
                                     "danger": str(r["danger"]).rsplit("#", 1)[-1].replace("FWIDanger", "")}
        return latest

    # -- run -------------------------------------------------------------------------
    def run(self, write: bool = True) -> RuleRun:
        run = RuleRun()
        d = TurtleDoc({"ffo": self.s.ns["ffo"], "ffv": self.s.ns["ffv"], "data": self.data})
        now = datetime.now(timezone.utc)
        sites = self._sites()
        hotspots = self._hotspots()
        r09_by_site = self._r09(d, now, sites, hotspots, run)
        self._r10(d, now, sites, hotspots, run)
        self._r11(d, now, sites, r09_by_site, run)
        try:
            self._r12(d, now, sites, hotspots, run)
        except Exception as e:  # noqa: BLE001 - R12 depends on aligned image terms
            run.warnings.append(f"R12 skipped: {e}")
        run.triples = len(d)
        if write:
            g = self.s.graph("inferred_live")
            self.store.clear_graph(g)
            if len(d):
                self.store.insert_turtle(d.serialize(), g)
        return run

    def _hazard(self, d: TurtleDoc, run: RuleRun, rule: str, concept: str, now: datetime,
                key: str, supported_by: list[str], explanation: str,
                site: str | None = None, event: str | None = None, **values) -> str:
        hz = f"{self.data}livehazard/{rule}/{_h(key)}"
        d.add_iri(hz, "a", "ffo:LiveHazard")
        d.add_iri(hz, "a", "ffo:Hazard")          # explicit, so CQ24/CQ25 match without inference
        d.add_iri(hz, "ffo:hasHazardType", f"ffv:{concept}")
        d.add(hz, "ffo:derivedByRule", rule)
        d.add(hz, "ffo:derivedAt", now)
        d.add(hz, "rdfs:comment", explanation)
        d.add_iri(hz, "ffo:affectsSite", site)
        d.add_iri(hz, "ffo:inEvent", event)
        for s in supported_by:
            d.add_iri(hz, "prov:wasDerivedFrom", s)
        for k, v in values.items():
            if v is None:
                continue
            if k == "raisedDangerClass":
                d.add_iri(hz, "ffo:raisedDangerClass", f"ffv:FWIDanger{v}")
            elif k in ("distanceKm", "frpGrowthRatio"):
                d.add(hz, f"ffo:{k}", Lit(repr(float(v)), XSD_DOUBLE))
            # other values (e.g. hotspotCount) stay in the run report only
        run.derived[rule] += 1
        run.hazards.append({"id": hz, "rule": rule, "type": concept, "site": site, "event": event,
                            "explanation": explanation, "supported_by": supported_by, **values})
        return hz

    def _r09(self, d, now, sites, hotspots, run) -> dict[str, list[tuple[dict, float]]]:
        near_km = float(self.cfg["R09_near_km"])
        by_site: dict[str, list[tuple[dict, float]]] = {}
        for s in sites:
            near = []
            for h in hotspots:
                km = haversine_km(s["lat"], s["lon"], h["lat"], h["lon"])
                if km <= near_km:
                    near.append((h, km))
            if not near:
                continue
            near.sort(key=lambda x: x[1])
            by_site[s["site"]] = near
            closest_h, closest_km = near[0]
            self._hazard(
                d, run, "R09", "NearbyActiveFire", now, s["site"], [h["h"] for h, _ in near[:20]],
                f"{len(near)} likely vegetation-fire hotspot(s) within {near_km:g} km of {s['label']}; "
                f"closest {closest_km:.1f} km, detected {closest_h['t']:%Y-%m-%d %H:%M} UTC.",
                site=s["site"], distanceKm=round(closest_km, 2), hotspotCount=len(near))
        return by_site

    def _r10(self, d, now, sites, hotspots, run) -> None:
        km_lim = float(self.cfg["R10_cluster_km"])
        hr_lim = timedelta(hours=float(self.cfg["R10_cluster_hours"]))
        n = len(hotspots)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        order = sorted(range(n), key=lambda i: hotspots[i]["t"])
        for a_idx, i in enumerate(order):
            for j in order[a_idx + 1:]:
                if hotspots[j]["t"] - hotspots[i]["t"] > hr_lim:
                    break
                if haversine_km(hotspots[i]["lat"], hotspots[i]["lon"], hotspots[j]["lat"], hotspots[j]["lon"]) <= km_lim:
                    parent[find(j)] = find(i)
        clusters: dict[int, list[dict]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(hotspots[i])
        near_km = float(self.cfg["R09_near_km"])
        for members in clusters.values():
            members.sort(key=lambda h: h["t"])
            ev = f"{self.data}liveevent/{_h(members[0]['h'])}"
            d.add_iri(ev, "a", "ffo:LiveFireEvent")
            d.add_iri(ev, "a", "ffo:FireEvent")
            d.add(ev, "ffo:eventCode", f"live-{_h(members[0]['h'])[:10]}")
            d.add(ev, "ffo:firstSeen", members[0]["t"])
            d.add(ev, "ffo:lastSeen", members[-1]["t"])
            d.add(ev, "ffo:hotspotCount", len(members))
            total = round(sum(h["frp"] for h in members), 2)
            d.add(ev, "ffo:totalFrpMW", Lit(repr(float(total)), XSD_DOUBLE))
            d.add(ev, "ffo:derivedByRule", "R10")
            clat = sum(h["lat"] for h in members) / len(members)
            clon = sum(h["lon"] for h in members) / len(members)
            d.add(ev, "ffo:latitude", round(clat, 6))
            d.add(ev, "ffo:longitude", round(clon, 6))
            for h in members:
                d.add_iri(ev, "ffo:hasHotspot", h["h"])
            near_sites = [s for s in sites if haversine_km(s["lat"], s["lon"], clat, clon) <= near_km]
            for s in near_sites:
                d.add_iri(ev, "ffo:nearSite", s["site"])
            run.derived["R10_event"] += 1
            # passes: group by (product, acquisition time within 30 min)
            passes: list[list[dict]] = []
            for h in members:
                if passes and h["product"] == passes[-1][0]["product"] and \
                        h["t"] - passes[-1][0]["t"] <= timedelta(minutes=30):
                    passes[-1].append(h)
                else:
                    passes.append([h])
            if len(passes) >= 2:
                prev, last = sum(h["frp"] for h in passes[-2]), sum(h["frp"] for h in passes[-1])
                ratio = (last / prev) if prev > 0 else None
                grew_count = len(passes[-1]) > len(passes[-2])
                if (ratio and ratio >= float(self.cfg["R10_frp_growth_ratio"])) or (ratio is None and grew_count):
                    self._hazard(
                        d, run, "R10", "SpreadingFire", now, ev, [h["h"] for h in passes[-1] + passes[-2]],
                        f"Fire radiative power rose from {prev:.1f} to {last:.1f} MW between the last two "
                        f"satellite passes ({passes[-2][0]['t']:%m-%d %H:%M} -> {passes[-1][0]['t']:%m-%d %H:%M} UTC).",
                        event=ev, site=near_sites[0]["site"] if near_sites else None,
                        frpGrowthRatio=round(ratio, 2) if ratio else None)

    def _r11(self, d, now, sites, r09_by_site, run) -> None:
        latest = self._latest_fwi()
        min_rank = class_rank(self.cfg["R11_min_danger"], self.classes)
        names = [c["name"] for c in self.classes]
        label = {s["site"]: s["label"] for s in sites}
        for site, near in r09_by_site.items():
            f = latest.get(site)
            if not f or f["danger"] not in names:
                continue
            rank = class_rank(f["danger"], self.classes)
            if rank < min_rank:
                continue
            raised = names[min(rank + 1, len(names) - 1)]
            self._hazard(
                d, run, "R11", "ElevatedDangerNearActiveFire", now, site, [f["a"], near[0][0]["h"]],
                f"FWI {f['fwi']:.1f} ({f['danger']}) on {f['date']} at {label.get(site, site)} with an active "
                f"fire {near[0][1]:.1f} km away; danger raised to {raised}.",
                site=site, raisedDangerClass=raised)

    def _r12(self, d, now, sites, hotspots, run) -> None:
        km_lim = float(self.cfg["R12_match_km"])
        hr_lim = timedelta(hours=float(self.cfg["R12_match_hours"]))
        for r in self._q("inputs_images.rq"):
            if float(r["score"]) >= float(r["threshold"]):
                continue  # the image already says fire: no conflict of this kind
            try:
                ilat, ilon = parse_wkt_point(r["wkt"])
            except (ValueError, IndexError):
                run.warnings.append(f"R12: unreadable centerWKT on {r['rec']}")
                continue
            t = _dt(r["t"]) if r.get("t") else None
            for h in hotspots:
                km = haversine_km(ilat, ilon, h["lat"], h["lon"])
                if km > km_lim or (t is not None and abs(h["t"] - t) > hr_lim):
                    continue
                when = f" and {hr_lim.total_seconds() / 3600:g} h of its capture time" if t else \
                    " (the image has no capture time, so time was not compared)"
                self._hazard(
                    d, run, "R12", "SatelliteSceneConflict", now, r["rec"] + h["h"], [r["rec"], h["h"]],
                    f"Image {r['rec'].rsplit('/', 1)[-1]} scored fire {float(r['score']):.2f} (below its "
                    f"threshold {float(r['threshold']):.2f}), but a satellite hotspot lies {km:.1f} km from the "
                    f"image location{when}.",
                    event=r.get("event"), distanceKm=round(km, 2))
                break
