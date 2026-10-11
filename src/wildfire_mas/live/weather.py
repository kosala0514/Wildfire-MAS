"""Current weather + FWI from Open-Meteo (free, no key), and place-name geocoding."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import requests

from ..config import Settings
from .fwi import FWIDay, danger_class, run_series


@dataclass
class WeatherNow:
    lat: float
    lon: float
    time_local: str
    timezone: str
    temp_c: float | None
    rh_pct: float | None
    wind_kmh: float | None
    precip_mm: float | None
    source: str = "open-meteo"


@dataclass
class FWIReport:
    lat: float
    lon: float
    today: FWIDay
    danger: str
    trend: list[FWIDay]          # last 7 days, oldest first
    spinup_days: int
    source: str = "open-meteo hourly; FWI computed locally (Van Wagner 1987)"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class WeatherClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        c = settings.get("weather")
        self.base = c["base_url"]
        self.geo = c["geocode_url"]
        self.spinup = int(c.get("fwi_spinup_days", 30))
        self.ttl = int(c.get("cache_ttl_s", 1800))
        self.classes = settings.get("danger.fwi_classes")
        self.http = session or requests.Session()
        self._mem: dict[str, tuple[float, dict]] = {}

    def _get(self, url: str, params: dict) -> dict:
        key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        hit = self._mem.get(key)
        if hit and time.time() - hit[0] < self.ttl:
            return hit[1]
        r = self.http.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        self._mem[key] = (time.time(), data)
        return data

    def current(self, lat: float, lon: float) -> WeatherNow:
        d = self._get(self.base, {
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        })
        c = d.get("current", {})
        return WeatherNow(lat, lon, c.get("time", ""), d.get("timezone", ""), c.get("temperature_2m"),
                          c.get("relative_humidity_2m"), c.get("wind_speed_10m"), c.get("precipitation"))

    def hourly(self, lat: float, lon: float, past_days: int) -> dict:
        return self._get(self.base, {
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            "past_days": min(92, past_days), "forecast_days": 1,
        })

    def fwi(self, lat: float, lon: float) -> FWIReport:
        days = noon_series(self.hourly(lat, lon, self.spinup + 1))
        # drop days that are not complete yet (today before noon has no noon value)
        series = run_series(days, lat)
        if not series:
            raise RuntimeError("no complete weather days returned")
        today = series[-1]
        return FWIReport(lat, lon, today, danger_class(today.fwi, self.classes), series[-7:], len(series))

    def geocode(self, name: str, count: int = 3) -> list[dict]:
        d = self._get(self.geo, {"name": name, "count": count, "language": "en", "format": "json"})
        return [
            {"name": r.get("name"), "country": r.get("country"), "admin1": r.get("admin1"),
             "lat": r.get("latitude"), "lon": r.get("longitude"), "country_code": r.get("country_code")}
            for r in d.get("results", []) or []
        ]


def noon_series(hourly_json: dict) -> list[dict]:
    """Noon-local temp/RH/wind + rain summed over the 24 h ending at noon."""
    h = hourly_json["hourly"]
    times = h["time"]
    now_local = None
    off = hourly_json.get("utc_offset_seconds")
    if off is not None:
        now_local = datetime.now(timezone.utc).timestamp() + off
    rain_by_noon: dict[str, float] = defaultdict(float)
    noon: dict[str, dict] = {}
    for i, t in enumerate(times):
        dt = datetime.fromisoformat(t)
        p = h["precipitation"][i] or 0.0
        # hour 13:00 (day D) .. 12:00 (day D+1) belongs to day D+1's FWI rainfall
        target = dt if dt.hour <= 12 else dt.replace(hour=0) + _one_day()
        rain_by_noon[target.date().isoformat()] += p
        if dt.hour == 12:
            if now_local is not None and dt.replace(tzinfo=timezone.utc).timestamp() > now_local:
                continue  # noon not reached yet
            vals = (h["temperature_2m"][i], h["relative_humidity_2m"][i], h["wind_speed_10m"][i])
            if None in vals:
                continue
            noon[dt.date().isoformat()] = {"temp": vals[0], "rh": vals[1], "wind": vals[2]}
    days = []
    for d in sorted(noon):
        days.append({"date": d, **noon[d], "rain": round(rain_by_noon.get(d, 0.0), 2)})
    return days[1:]  # first day has an incomplete rain window


def _one_day():
    from datetime import timedelta

    return timedelta(days=1)
