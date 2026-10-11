"""Label FIRMS thermal anomalies: which ones are plausible vegetation fires?

FIRMS detects *thermal anomalies*. Besides wildfires these include volcanoes, gas
flares, refineries, steel plants, power stations, offshore platforms, agricultural
or waste burning, and occasional false alarms. Nothing is deleted here: every hotspot
gets a category plus the reasons, so the KG keeps the full record and the agents can
explain why a detection was or wasn't treated as wildfire evidence.

Checks, in order (first decisive match wins, all reasons are kept):
  1. FIRMS `type` column (standard-processing products only):
     0 presumed vegetation fire, 1 active volcano, 2 other static land source, 3 offshore
  2. Known static sources file (volcanoes, flares, plants) within `static_match_km`
  3. Offshore (optional land-mask GeoJSON)
  4. Persistence: same ~400 m cell detected on >= N distinct days in the window
     -> very likely an industrial / static heat source, not a moving wildfire
  5. Low confidence
  6. Otherwise: likely vegetation fire (may still be agricultural burning; FIRMS cannot
     tell the two apart without land-cover data)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .firms import Hotspot
from .geo import grid_cell, haversine_km, point_in_polygon

CATEGORIES = {
    "likely_vegetation_fire": True,
    "low_confidence": False,
    "persistent_static_source": False,
    "static_land_source": False,
    "volcanic": False,
    "offshore": False,
}


@dataclass
class StaticSource:
    name: str
    lat: float
    lon: float
    kind: str


def load_static_sources(path: Path | None) -> list[StaticSource]:
    if not path or not path.exists():
        return []
    out = []
    with path.open() as f:
        rows = (r for r in f if r.strip() and not r.lstrip().startswith("#"))
        for r in csv.DictReader(rows):
            out.append(StaticSource(r["name"], float(r["lat"]), float(r["lon"]), r.get("kind", "static")))
    return out


def load_land_polygons(path: Path | None) -> list[list]:
    if not path or not path.exists():
        return []
    gj = json.loads(path.read_text())
    rings = []
    for feat in gj.get("features", [gj]):
        geom = feat.get("geometry", feat)
        if geom["type"] == "Polygon":
            rings.append(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            rings.extend(poly[0] for poly in geom["coordinates"])
    return rings


class HotspotClassifier:
    def __init__(self, cfg: dict, static_sources: list[StaticSource] | None = None,
                 land_rings: list[list] | None = None):
        self.cfg = cfg
        self.static = static_sources or []
        self.land = land_rings or []

    @classmethod
    def from_settings(cls, settings) -> "HotspotClassifier":
        c = settings.get("hotspot_filter")
        return cls(c, load_static_sources(settings.path(c.get("static_sources_file"))),
                   load_land_polygons(settings.path(c.get("land_mask_file"))))

    def persistence_counts(self, history: Iterable[Hotspot]) -> dict[tuple[int, int], int]:
        cell = float(self.cfg.get("persistence_cell_deg", 0.004))
        days: dict[tuple[int, int], set] = defaultdict(set)
        for h in history:
            c = grid_cell(h.latitude, h.longitude, cell)
            # count the cell and its 8 neighbours so a source on a cell edge still matches
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    days[(c[0] + dy, c[1] + dx)].add(h.acq_datetime.date())
        return {k: len(v) for k, v in days.items()}

    def classify(self, hotspots: list[Hotspot], history: Iterable[Hotspot] | None = None) -> list[Hotspot]:
        cfg = self.cfg
        counts = self.persistence_counts(history) if history is not None else {}
        cell = float(cfg.get("persistence_cell_deg", 0.004))
        min_days = int(cfg.get("persistence_min_days", 8))
        match_km = float(cfg.get("static_match_km", 1.0))
        for h in hotspots:
            reasons: list[str] = []
            cat: str | None = None
            if h.fire_type is not None:
                mapping = {1: "volcanic", 2: "static_land_source", 3: "offshore"}
                if h.fire_type in mapping:
                    cat = mapping[h.fire_type]
                    reasons.append(f"FIRMS type={h.fire_type} ({cat})")
                else:
                    reasons.append("FIRMS type=0 (presumed vegetation fire)")
            if cat is None and self.static:
                near = min(self.static, key=lambda s: haversine_km(h.latitude, h.longitude, s.lat, s.lon))
                d = haversine_km(h.latitude, h.longitude, near.lat, near.lon)
                if d <= match_km:
                    cat = "volcanic" if near.kind == "volcano" else "static_land_source"
                    reasons.append(f"{d:.2f} km from known {near.kind} '{near.name}'")
            if cat is None and self.land:
                if not any(point_in_polygon(h.latitude, h.longitude, r) for r in self.land):
                    cat = "offshore"
                    reasons.append("outside land mask")
            if cat is None and counts:
                n = counts.get(grid_cell(h.latitude, h.longitude, cell), 0)
                if n >= min_days:
                    cat = "persistent_static_source"
                    reasons.append(f"detected on {n} distinct days in the last "
                                   f"{cfg.get('persistence_window_days', 30)} days at this location")
            if cat is None:
                weak = False
                if h.instrument and "MODIS" in h.instrument.upper():
                    weak = h.confidence_score * 100 < float(cfg.get("min_modis_confidence", 30))
                elif cfg.get("viirs_low_confidence_is_weak", True):
                    weak = h.confidence_raw.strip().lower() in {"l", "low"}
                if weak:
                    cat = "low_confidence"
                    reasons.append(f"low detection confidence ({h.confidence_raw})")
            if cat is None:
                cat = "likely_vegetation_fire"
                reasons.append("no static/volcanic/offshore match; confidence acceptable "
                               "(could still be agricultural burning)")
            h.category = cat
            h.wildfire_evidence = CATEGORIES[cat]
            h.reasons = reasons
        return hotspots
