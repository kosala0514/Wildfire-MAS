"""Canadian Forest Fire Weather Index (FWI) System.

Standard equations (Van Wagner 1987), with the latitude-dependent day-length
adjustments of Wang et al. (2015), so the codes work in the tropics (e.g. Sri Lanka)
as well as the Mediterranean (Montesinho, Algeria).

Inputs per day are *noon local* temperature (C), relative humidity (%), 10 m wind
speed (km/h) and 24 h rainfall (mm) ending at noon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# DMC effective day length (Le) by month
_LE_N30 = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
_LE_N10 = [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8]
_LE_S10 = [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2]
_LE_S30 = [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8]
# DC day length factor (Lf) by month
_LF_N = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
_LF_S = [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]

START = {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0}


@dataclass
class FWIState:
    ffmc: float = START["ffmc"]
    dmc: float = START["dmc"]
    dc: float = START["dc"]


@dataclass
class FWIDay:
    date: str
    temp: float
    rh: float
    wind: float
    rain: float
    ffmc: float
    dmc: float
    dc: float
    isi: float
    bui: float
    fwi: float
    dsr: float


def _day_length_dmc(lat: float, month: int) -> float:
    if lat > 30:
        return _LE_N30[month - 1]
    if lat > 10:
        return _LE_N10[month - 1]
    if lat > -10:
        return 9.0
    if lat > -30:
        return _LE_S10[month - 1]
    return _LE_S30[month - 1]


def _day_length_dc(lat: float, month: int) -> float:
    if lat > 20:
        return _LF_N[month - 1]
    if lat <= -20:
        return _LF_S[month - 1]
    return 1.4


def ffmc_calc(temp: float, rh: float, wind: float, rain: float, ffmc_prev: float) -> float:
    rh = min(100.0, max(0.0, rh))
    mo = 147.2 * (101.0 - ffmc_prev) / (59.5 + ffmc_prev)
    if rain > 0.5:
        rf = rain - 0.5
        mr = mo + 42.5 * rf * math.exp(-100.0 / (251.0 - mo)) * (1.0 - math.exp(-6.93 / rf))
        if mo > 150.0:
            mr += 0.0015 * (mo - 150.0) ** 2 * math.sqrt(rf)
        mo = min(mr, 250.0)
    ed = 0.942 * rh ** 0.679 + 11.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
    if mo > ed:
        ko = 0.424 * (1.0 - (rh / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind) * (1.0 - (rh / 100.0) ** 8)
        kd = ko * 0.581 * math.exp(0.0365 * temp)
        m = ed + (mo - ed) * 10.0 ** (-kd)
    else:
        ew = 0.618 * rh ** 0.753 + 10.0 * math.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - math.exp(-0.115 * rh))
        if mo < ew:
            k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * math.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0) ** 8)
            kw = k1 * 0.581 * math.exp(0.0365 * temp)
            m = ew - (ew - mo) * 10.0 ** (-kw)
        else:
            m = mo
    return min(101.0, max(0.0, 59.5 * (250.0 - m) / (147.2 + m)))


def dmc_calc(temp: float, rh: float, rain: float, dmc_prev: float, month: int, lat: float) -> float:
    rh = min(100.0, max(0.0, rh))
    t = max(temp, -1.1)
    rk = 1.894 * (t + 1.1) * (100.0 - rh) * _day_length_dmc(lat, month) * 1e-4
    if rain > 1.5:
        rw = 0.92 * rain - 1.27
        wmi = 20.0 + 280.0 / math.exp(0.023 * dmc_prev)
        if dmc_prev <= 33.0:
            b = 100.0 / (0.5 + 0.3 * dmc_prev)
        elif dmc_prev <= 65.0:
            b = 14.0 - 1.3 * math.log(dmc_prev)
        else:
            b = 6.2 * math.log(dmc_prev) - 17.2
        wmr = wmi + 1000.0 * rw / (48.77 + b * rw)
        pr = 43.43 * (5.6348 - math.log(wmr - 20.0))
    else:
        pr = dmc_prev
    return max(0.0, max(pr, 0.0) + rk)


def dc_calc(temp: float, rain: float, dc_prev: float, month: int, lat: float) -> float:
    t = max(temp, -2.8)
    pe = max(0.0, (0.36 * (t + 2.8) + _day_length_dc(lat, month)) / 2.0)
    if rain > 2.8:
        rw = 0.83 * rain - 1.27
        smi = 800.0 * math.exp(-dc_prev / 400.0)
        dr = max(0.0, dc_prev - 400.0 * math.log(1.0 + 3.937 * rw / smi))
    else:
        dr = dc_prev
    return max(0.0, dr + pe)


def isi_calc(ffmc: float, wind: float) -> float:
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    ff = 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)
    return 0.208 * math.exp(0.05039 * wind) * ff


def bui_calc(dmc: float, dc: float) -> float:
    if dmc == 0 and dc == 0:
        return 0.0
    if dmc <= 0.4 * dc:
        bui = 0.8 * dmc * dc / (dmc + 0.4 * dc)
    else:
        bui = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)
    return max(0.0, bui)


def fwi_calc(isi: float, bui: float) -> float:
    fd = 0.626 * bui ** 0.809 + 2.0 if bui <= 80.0 else 1000.0 / (25.0 + 108.64 * math.exp(-0.023 * bui))
    b = 0.1 * isi * fd
    return math.exp(2.72 * (0.434 * math.log(b)) ** 0.647) if b > 1.0 else b


def step(state: FWIState, date: str, temp: float, rh: float, wind: float, rain: float, lat: float) -> FWIDay:
    month = int(date[5:7])
    ffmc = ffmc_calc(temp, rh, wind, rain, state.ffmc)
    dmc = dmc_calc(temp, rh, rain, state.dmc, month, lat)
    dc = dc_calc(temp, rain, state.dc, month, lat)
    isi = isi_calc(ffmc, wind)
    bui = bui_calc(dmc, dc)
    fwi = fwi_calc(isi, bui)
    state.ffmc, state.dmc, state.dc = ffmc, dmc, dc
    return FWIDay(date, temp, rh, wind, rain, round(ffmc, 2), round(dmc, 2), round(dc, 2),
                  round(isi, 2), round(bui, 2), round(fwi, 2), round(0.0272 * fwi ** 1.77, 2))


def run_series(days: list[dict], lat: float, start: FWIState | None = None) -> list[FWIDay]:
    """days: [{'date','temp','rh','wind','rain'}...] in chronological order."""
    st = start or FWIState()
    return [step(st, d["date"], d["temp"], d["rh"], d["wind"], d["rain"], lat) for d in days]


def danger_class(fwi: float, classes: list[dict]) -> str:
    for c in classes:
        if fwi < float(c["max"]):
            return c["name"]
    return classes[-1]["name"]


def class_rank(name: str, classes: list[dict]) -> int:
    return [c["name"] for c in classes].index(name)
