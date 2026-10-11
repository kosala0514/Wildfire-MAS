"""CQ catalog: typed templates, rendering, sufficiency checks, and the CQ test runner."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import yaml

from ..config import Settings
from ..live.geo import bbox_around, haversine_km
from .rules_live import align_namespaces

_TOKEN = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_IRI_OK = re.compile(r"^[^\s<>\"{}|\\^`]+$")


@dataclass
class CQ:
    id: str
    question: str
    sparql: str
    intents: list[str] = field(default_factory=list)
    params: dict[str, dict] = field(default_factory=dict)
    required_vars: list[str] = field(default_factory=list)
    min_rows: int = 1
    relax: dict[str, list] = field(default_factory=dict)
    post_filter: dict | None = None
    expect: dict | None = None

    def describe(self) -> str:
        ps = ", ".join(f"{k}:{v.get('type')}" for k, v in self.params.items())
        return f"{self.id}: {self.question} (params: {ps or 'none'})"


class CQCatalog:
    def __init__(self, cqs: list[CQ]):
        self.by_id = {c.id: c for c in cqs}

    @classmethod
    def load(cls, settings: Settings) -> "CQCatalog":
        cqs = []
        for f in sorted((settings.root / settings.get("paths.cq_dir")).glob("*.yaml")):
            for item in yaml.safe_load(f.read_text()) or []:
                item["sparql"] = align_namespaces(item["sparql"], settings)
                if item.get("expect"):
                    import json as _json

                    item["expect"] = _json.loads(align_namespaces(_json.dumps(item["expect"]), settings))
                cqs.append(CQ(**item))
        return cls(cqs)

    def get(self, cq_id: str) -> CQ:
        if cq_id not in self.by_id:
            raise KeyError(f"unknown CQ {cq_id}")
        return self.by_id[cq_id]

    def for_intent(self, intent: str) -> list[CQ]:
        return [c for c in self.by_id.values() if intent in c.intents]

    def menu(self, intent: str | None = None) -> str:
        cqs = self.for_intent(intent) if intent else list(self.by_id.values())
        return "\n".join(c.describe() for c in cqs)


def _fmt(value: Any, typ: str) -> str:
    if typ == "iri":
        v = str(value).strip().strip("<>")
        if not _IRI_OK.match(v) or ":" not in v:
            raise ValueError(f"invalid IRI parameter: {value!r}")
        return f"<{v}>"
    if typ == "float":
        return repr(float(value))
    if typ == "int":
        return str(int(value))
    if typ == "datetime":
        dt = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return f'"{dt.isoformat()}"^^<http://www.w3.org/2001/XMLSchema#dateTime>'
    s = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{s}"'


def resolve_params(cq: CQ, params: dict) -> dict:
    """Apply defaults and derived params (since, bbox)."""
    p = {k: v for k, v in params.items() if v is not None}
    for name, spec in cq.params.items():
        if name not in p and "default" in spec:
            p[name] = spec["default"]
    if "days" in p and "since" not in p:
        p["since"] = datetime.now(timezone.utc) - timedelta(days=float(p["days"]))
    if {"lat", "lon", "radius_km"} <= p.keys() and "lat_min" not in p:
        w, s, e, n = bbox_around(float(p["lat"]), float(p["lon"]), float(p["radius_km"]))
        p.update(lat_min=s, lat_max=n, lon_min=w, lon_max=e)
    missing = [n for n in cq.params if n not in p]
    if missing:
        raise ValueError(f"{cq.id} missing parameters: {missing}")
    return p


def render(cq: CQ, params: dict) -> str:
    p = resolve_params(cq, params)
    types = {k: v.get("type", "string") for k, v in cq.params.items()}
    types.update({"since": "datetime", "lat_min": "float", "lat_max": "float",
                  "lon_min": "float", "lon_max": "float"})

    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in p:
            raise ValueError(f"{cq.id}: no value for {{{{{name}}}}}")
        return _fmt(p[name], types.get(name, "string"))

    return _TOKEN.sub(sub, cq.sparql)


def apply_post_filter(cq: CQ, params: dict, rows: list[dict]) -> list[dict]:
    pf = cq.post_filter
    if not pf or pf.get("kind") != "distance":
        return rows
    p = resolve_params(cq, params)
    lat0, lon0, rad = float(p[pf["lat"]]), float(p[pf["lon"]]), float(p[pf["radius"]])
    out = []
    for r in rows:
        try:
            km = haversine_km(lat0, lon0, float(r[pf["row_lat"]]), float(r[pf["row_lon"]]))
        except (KeyError, TypeError, ValueError):
            continue
        if km <= rad:
            out.append({**r, "distance_km": round(km, 2)})
    return sorted(out, key=lambda r: r["distance_km"])


def sufficient(cq: CQ, rows: list[dict]) -> tuple[bool, str]:
    good = [r for r in rows if all(r.get(v) not in (None, "") for v in cq.required_vars)]
    if len(good) >= cq.min_rows:
        return True, f"{len(good)} row(s) with {cq.required_vars} bound"
    return False, f"only {len(good)} row(s) with {cq.required_vars} bound (need {cq.min_rows})"
