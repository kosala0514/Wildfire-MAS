"""NASA FIRMS client (area / country / world), with caching and snapshot replay.

FIRMS returns CSV only. Day range is 1..5 per request; longer windows are fetched
in 5-day pages using the optional [DATE] path segment.

The MAP_KEY is read from FIRMS_MAP_KEY and is never written to logs, cache metadata,
snapshots, or the knowledge graph (URLs are stored with the key redacted).
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import requests

from ..config import Settings
from .geo import BBox

NRT_SOURCES = {
    "MODIS_NRT", "VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "LANDSAT_NRT",
}
SP_SOURCES = {"MODIS_SP", "VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"}
ALL_SOURCES = NRT_SOURCES | SP_SOURCES


class FirmsError(RuntimeError):
    pass


@dataclass
class Hotspot:
    source: str
    latitude: float
    longitude: float
    acq_datetime: datetime          # UTC
    satellite: str | None
    instrument: str | None
    confidence_raw: str             # 'l'/'n'/'h' (VIIRS) or '0'..'100' (MODIS)
    confidence_score: float         # normalised 0..1
    frp: float | None               # fire radiative power, MW
    brightness: float | None        # K (MODIS brightness / VIIRS bright_ti4)
    brightness_2: float | None      # K (MODIS bright_t31 / VIIRS bright_ti5)
    daynight: str | None
    fire_type: int | None = None    # SP products only: 0 veg fire, 1 volcano, 2 static land, 3 offshore
    version: str | None = None
    # filled by classify.py
    category: str | None = None
    wildfire_evidence: bool | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        raw = f"{self.source}|{self.latitude:.5f}|{self.longitude:.5f}|{self.acq_datetime.isoformat()}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["acq_datetime"] = self.acq_datetime.isoformat()
        d["id"] = self.key
        return d


@dataclass
class FirmsResult:
    hotspots: list[Hotspot]
    sources: list[str]
    area: str
    day_range: int
    end_date: str | None
    fetched_at: str
    urls_redacted: list[str]
    from_cache: bool
    warnings: list[str] = field(default_factory=list)   # e.g. a source FIRMS rejected


def _conf_score(raw: str, instrument: str | None) -> float:
    r = str(raw).strip().lower()
    if r in {"l", "low"}:
        return 0.3
    if r in {"n", "nominal"}:
        return 0.6
    if r in {"h", "high"}:
        return 0.9
    try:
        return max(0.0, min(1.0, float(r) / 100.0))
    except ValueError:
        return 0.0


def _f(v) -> float | None:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_firms_csv(text: str, source: str) -> list[Hotspot]:
    text = text.strip()
    if not text:
        return []
    head = text.splitlines()[0].lower()
    if "latitude" not in head:
        raise FirmsError(f"FIRMS returned an error instead of CSV: {text[:200]}")
    df = pd.read_csv(io.StringIO(text), dtype={"acq_time": str, "confidence": str, "version": str})
    out: list[Hotspot] = []
    for row in df.to_dict(orient="records"):
        t = str(row.get("acq_time", "0")).split(".")[0].zfill(4)
        dt = datetime.strptime(f"{row['acq_date']} {t}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
        instrument = row.get("instrument")
        is_modis = "brightness" in row
        ftype = row.get("type")
        out.append(Hotspot(
            source=source,
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            acq_datetime=dt,
            satellite=None if pd.isna(row.get("satellite")) else str(row.get("satellite")),
            instrument=None if instrument is None or pd.isna(instrument) else str(instrument),
            confidence_raw=str(row.get("confidence", "")),
            confidence_score=_conf_score(row.get("confidence", ""), instrument),
            frp=_f(row.get("frp")),
            brightness=_f(row.get("brightness") if is_modis else row.get("bright_ti4")),
            brightness_2=_f(row.get("bright_t31") if is_modis else row.get("bright_ti5")),
            daynight=None if pd.isna(row.get("daynight")) else str(row.get("daynight")),
            fire_type=None if ftype is None or pd.isna(ftype) else int(ftype),
            version=None if pd.isna(row.get("version")) else str(row.get("version")),
        ))
    return out


class FirmsClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        c = settings.get("firms")
        self.settings = settings
        self.map_key: str | None = c.get("map_key")
        self.base = c["base_url"].rstrip("/")
        self.default_sources: list[str] = c["default_sources"]
        self.max_days = int(c.get("max_day_range", 5))
        self.cache_dir = settings.path(c.get("cache_dir"))
        self.ttl = int(c.get("cache_ttl_s", 900))
        self.replay_dir = settings.path(c.get("replay_dir"))
        self.record_dir = settings.path(c.get("record_dir"))
        self.http = session or requests.Session()

    # -- low level ------------------------------------------------------------------
    def _redact(self, url: str) -> str:
        return url.replace(self.map_key, "<MAP_KEY>") if self.map_key else url

    @staticmethod
    def _hash(redacted_url: str) -> str:
        return hashlib.sha1(redacted_url.encode()).hexdigest()

    def _get_text(self, url: str) -> tuple[str, bool]:
        red = self._redact(url)
        h = self._hash(red)
        if self.replay_dir:
            f = self.replay_dir / f"{h}.csv"
            if not f.exists():
                raise FirmsError(f"replay snapshot missing for {red}")
            return f.read_text(), True
        if self.cache_dir:
            f = self.cache_dir / f"{h}.csv"
            if f.exists() and time.time() - f.stat().st_mtime < self.ttl:
                return f.read_text(), True
        if not self.map_key:
            raise FirmsError("FIRMS_MAP_KEY is not set (put it in .env)")
        r = self.http.get(url, timeout=60)
        if r.status_code != 200:
            raise FirmsError(f"FIRMS HTTP {r.status_code}: {r.text[:200]}")
        text = r.text
        for d in (self.cache_dir, self.record_dir):
            if d:
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{h}.csv").write_text(text)
                (d / f"{h}.json").write_text(json.dumps(
                    {"url": red, "fetched_at": datetime.now(timezone.utc).isoformat()}))
        return text, False

    # -- endpoints --------------------------------------------------------------------
    def _check_source(self, source: str) -> None:
        if source not in ALL_SOURCES:
            raise FirmsError(f"unknown FIRMS source {source}; use one of {sorted(ALL_SOURCES)}")

    def area_url(self, source: str, area: str, days: int, end: date | None = None) -> str:
        """`area` is 'w,s,e,n' or 'world'. With `end`, FIRMS returns [end-days+1 .. end]."""
        self._check_source(source)
        days = max(1, min(self.max_days, int(days)))
        url = f"{self.base}/api/area/csv/{self.map_key or 'NOKEY'}/{source}/{area}/{days}"
        if end is not None:
            start = end - timedelta(days=days - 1)  # FIRMS [DATE] is the window start
            url += f"/{start.isoformat()}"
        return url

    def country_url(self, source: str, country: str, days: int, end: date | None = None) -> str:
        self._check_source(source)
        days = max(1, min(self.max_days, int(days)))
        url = f"{self.base}/api/country/csv/{self.map_key or 'NOKEY'}/{source}/{country.upper()}/{days}"
        if end is not None:
            url += f"/{(end - timedelta(days=days - 1)).isoformat()}"
        return url

    def _paged(self, make_url, source: str, days: int, end: date | None) -> tuple[list[Hotspot], list[str], bool]:
        """Fetch any number of days as 5-day pages."""
        hs: list[Hotspot] = []
        urls: list[str] = []
        all_cached = True
        if days <= self.max_days and end is None:
            url = make_url(source, days, None)
            text, cached = self._get_text(url)
            return parse_firms_csv(text, source), [self._redact(url)], cached
        end = end or datetime.now(timezone.utc).date()
        remaining = days
        cursor = end
        while remaining > 0:
            n = min(self.max_days, remaining)
            url = make_url(source, n, cursor)
            text, cached = self._get_text(url)
            all_cached &= cached
            hs.extend(parse_firms_csv(text, source))
            urls.append(self._redact(url))
            cursor -= timedelta(days=n)
            remaining -= n
        return hs, urls, all_cached

    def _collect(self, srcs: list[str], make_url, days: int, end: date | None, label: str) -> FirmsResult:
        """Fetch every source; one source that FIRMS rejects is reported, not fatal."""
        hs: list[Hotspot] = []
        urls: list[str] = []
        warnings: list[str] = []
        cached_all = True
        used: list[str] = []
        for s in srcs:
            if s == "LANDSAT_NRT" and label == "world":
                continue  # US/Canada only
            try:
                part, u, c = self._paged(make_url, s, days, end)
            except FirmsError as e:
                warnings.append(f"{s}: {e}")
                continue
            hs.extend(part)
            urls.extend(u)
            used.append(s)
            cached_all &= c
        if not used:
            raise FirmsError(f"every FIRMS source failed for {label}: " + "; ".join(warnings))
        return FirmsResult(_dedupe(hs), used, label, days, end.isoformat() if end else None,
                           datetime.now(timezone.utc).isoformat(), urls, cached_all, warnings)

    def get_area(self, area: BBox | str, days: int = 1, sources: Iterable[str] | None = None,
                 end: date | None = None) -> FirmsResult:
        area_s = area if isinstance(area, str) else ",".join(f"{v:g}" for v in area)
        return self._collect(list(sources or self.default_sources),
                             lambda src, d, e: self.area_url(src, area_s, d, e), days, end, area_s)

    def get_country(self, country: str, days: int = 1, sources: Iterable[str] | None = None,
                    end: date | None = None) -> FirmsResult:
        return self._collect(list(sources or self.default_sources),
                             lambda src, d, e: self.country_url(src, country, d, e), days, end, country.upper())

    def key_status(self) -> dict:
        if not self.map_key:
            raise FirmsError("FIRMS_MAP_KEY is not set")
        r = self.http.get(f"{self.base}/mapserver/mapkey_status/", params={"MAP_KEY": self.map_key}, timeout=30)
        r.raise_for_status()
        return r.json()


def _dedupe(hs: list[Hotspot]) -> list[Hotspot]:
    seen: dict[str, Hotspot] = {}
    for h in hs:
        seen.setdefault(h.key, h)
    return sorted(seen.values(), key=lambda h: h.acq_datetime)
