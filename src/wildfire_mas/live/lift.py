"""Lift live data (FIRMS hotspots, weather, FWI) into FFOnto RDF.

Layers follow FFOnto, reusing its classes and properties:
  DATA   ffo:SatelliteHotspot
         ffo:LiveWeatherRecord (a ffonto:WeatherRecord: atSite, observationDate, airTemperatureC,
         relativeHumidityPct, windSpeedKmh, rainMm, ffmc, dmc, dc, isi, bui, fwi, previousRecord)
  CLAIMS ffo:HotspotClassification (a ffonto:ModelObservation; ffonto:confidence as xsd:double)
  PROV   prov:Activity per fetch (endpoint stored with the MAP_KEY redacted)
Numbers that FFOnto types as xsd:double are written as xsd:double (mixing decimal and
double on one property would make the ontology inconsistent under its range axioms).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from ..config import Settings
from ..kg.turtle import Lit, TurtleDoc
from .firms import FirmsResult, Hotspot
from .geo import wkt_point
from .weather import FWIReport, WeatherNow

GEO_WKT = "http://www.opengis.net/ont/geosparql#wktLiteral"

CATEGORY_CONCEPT = {
    "likely_vegetation_fire": "ffv:LikelyVegetationFire",
    "low_confidence": "ffv:LowConfidenceDetection",
    "persistent_static_source": "ffv:PersistentStaticHeatSource",
    "static_land_source": "ffv:StaticLandHeatSource",
    "volcanic": "ffv:VolcanicHeatSource",
    "offshore": "ffv:OffshoreHeatSource",
}


XSD_DOUBLE = "http://www.w3.org/2001/XMLSchema#double"
XSD_DATETIME = "http://www.w3.org/2001/XMLSchema#dateTime"


def _dbl(v):
    """xsd:double literal (FFOnto's range for measurements and confidences)."""
    return None if v is None else Lit(repr(float(v)), XSD_DOUBLE)


def _fl(v):
    return None if v is None else float(v)


def _h(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


class LiveLifter:
    def __init__(self, settings: Settings):
        self.s = settings
        self.data = settings.ns["data"]

    def doc(self) -> TurtleDoc:
        return TurtleDoc({"ffo": self.s.ns["ffo"], "ffv": self.s.ns["ffv"], "data": self.data})

    def _geom(self, d: TurtleDoc, subj: str, lat: float, lon: float) -> None:
        g = subj + "/geom" if subj.startswith("http") else subj + "_geom"
        d.add_iri(subj, "geo:hasGeometry", g)
        d.add_iri(g, "a", "geo:Geometry")
        d.add(g, "geo:asWKT", Lit(wkt_point(lat, lon), GEO_WKT))
        d.add(subj, "ffo:latitude", round(lat, 6))
        d.add(subj, "ffo:longitude", round(lon, 6))

    def fetch_activity(self, d: TurtleDoc, kind: str, endpoint: str, product: str | None = None) -> str:
        now = datetime.now(timezone.utc)
        act = f"{self.data}fetch/{_h(kind, endpoint, now.isoformat())}"
        d.add_iri(act, "a", "prov:Activity")
        d.add(act, "prov:startedAtTime", now)
        d.add(act, "ffo:sourceEndpoint", endpoint)
        d.add(act, "ffo:sourceKind", kind)
        if product:
            d.add(act, "ffo:sourceProduct", product)
        return act

    def hotspots(self, res: FirmsResult, d: TurtleDoc | None = None) -> TurtleDoc:
        d = d or self.doc()
        act = self.fetch_activity(d, "NASA FIRMS", "; ".join(res.urls_redacted), ",".join(res.sources))
        for h in res.hotspots:
            self.hotspot(d, h, act)
        return d

    def hotspot(self, d: TurtleDoc, h: Hotspot, activity: str) -> str:
        s = f"{self.data}hotspot/{h.key}"
        d.add_iri(s, "a", "ffo:SatelliteHotspot")
        d.add(s, "ffo:acquiredAt", h.acq_datetime)
        self._geom(d, s, h.latitude, h.longitude)
        d.add(s, "ffo:fromProduct", h.source)
        d.add(s, "ffo:satellite", h.satellite)
        d.add(s, "ffo:instrument", h.instrument)
        d.add(s, "ffo:frpMW", _dbl(h.frp))
        d.add(s, "ffo:brightnessK", _dbl(h.brightness))
        d.add(s, "ffo:brightness2K", _dbl(h.brightness_2))
        d.add(s, "ffo:dayNight", h.daynight)
        d.add(s, "ffo:firmsConfidence", h.confidence_raw)
        if h.fire_type is not None:
            d.add(s, "ffo:firmsType", h.fire_type)
        d.add_iri(s, "prov:wasGeneratedBy", activity)
        # CLAIMS layer: the detection as a model claim with confidence + our classification
        c = f"{self.data}claim/hotspot/{h.key}"
        d.add_iri(c, "a", "ffo:HotspotClassification")
        d.add_iri(c, "ffo:aboutHotspot", s)
        d.add(c, "ffo:confidence", _dbl(round(h.confidence_score, 3)))
        if h.category:
            d.add_iri(c, "ffo:detectionCategory", CATEGORY_CONCEPT[h.category])
            d.add(c, "ffo:isWildfireEvidence", bool(h.wildfire_evidence))
            d.add_all(c, "ffo:classificationReason", h.reasons)
        d.add_iri(c, "prov:wasGeneratedBy", activity)
        return s

    def _weather_record(self, d: TurtleDoc, s: str, site: str | None, date_iso: str, act: str) -> None:
        d.add_iri(s, "a", "ffo:LiveWeatherRecord")
        d.add_iri(s, "a", "ffo:WeatherRecord")   # explicit, so CQ08/CQ09 match without inference
        d.add_iri(s, "ffo:atSite", site)
        d.add(s, "ffo:observationDate", Lit(f"{date_iso[:10]}T00:00:00Z", XSD_DATETIME))
        d.add_iri(s, "prov:wasGeneratedBy", act)

    def weather(self, w: WeatherNow, site: str | None = None, d: TurtleDoc | None = None) -> TurtleDoc:
        d = d or self.doc()
        act = self.fetch_activity(d, "Open-Meteo current", "https://api.open-meteo.com/v1/forecast")
        s = f"{self.data}liveweather/now/{_h(w.lat, w.lon, w.time_local)}"
        self._weather_record(d, s, site, w.time_local or datetime.now(timezone.utc).isoformat(), act)
        d.add(s, "ffo:observedAtLocal", w.time_local)
        d.add(s, "ffo:airTemperatureC", _dbl(w.temp_c))
        d.add(s, "ffo:relativeHumidityPct", _dbl(w.rh_pct))
        d.add(s, "ffo:windSpeedKmh", _dbl(w.wind_kmh))
        d.add(s, "ffo:rainMm", _dbl(w.precip_mm))
        return d

    def fwi(self, r: FWIReport, site: str | None = None, d: TurtleDoc | None = None) -> TurtleDoc:
        """One ffonto:WeatherRecord per day (noon weather + FWI codes), chained with previousRecord."""
        from .fwi import danger_class

        d = d or self.doc()
        act = self.fetch_activity(d, "FWI computation", r.source)
        classes = self.s.get("danger.fwi_classes")
        prev = None
        for day in r.trend:
            s = self.fwi_day_iri(r.lat, r.lon, day.date)
            self._weather_record(d, s, site, day.date, act)
            d.add(s, "ffo:airTemperatureC", _dbl(day.temp))
            d.add(s, "ffo:relativeHumidityPct", _dbl(day.rh))
            d.add(s, "ffo:windSpeedKmh", _dbl(day.wind))
            d.add(s, "ffo:rainMm", _dbl(day.rain))
            for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi"):
                d.add(s, f"ffo:{k}", _dbl(getattr(day, k)))
            d.add_iri(s, "ffo:dangerClass", f"ffv:FWIDanger{danger_class(day.fwi, classes)}")
            d.add_iri(s, "ffo:previousRecord", prev)
            prev = s
        return d

    def fwi_day_iri(self, lat: float, lon: float, date: str) -> str:
        return f"{self.data}liveweather/fwi/{_h(round(lat, 4), round(lon, 4), date)}"
