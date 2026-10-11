"""Small geodesy helpers (no GIS dependencies)."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

EARTH_KM = 6371.0088
BBox = tuple[float, float, float, float]  # west, south, east, north


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(a)))


def bbox_around(lat: float, lon: float, radius_km: float) -> BBox:
    dlat = radius_km / 111.32
    dlon = radius_km / (111.32 * max(0.01, math.cos(math.radians(lat))))
    return (
        round(max(-180.0, lon - dlon), 5),
        round(max(-90.0, lat - dlat), 5),
        round(min(180.0, lon + dlon), 5),
        round(min(90.0, lat + dlat), 5),
    )


def parse_bbox(value: str | Sequence[float]) -> BBox:
    if isinstance(value, str):
        parts = [float(x) for x in value.replace(" ", "").split(",")]
    else:
        parts = [float(x) for x in value]
    if len(parts) != 4:
        raise ValueError("bbox needs 4 numbers: west,south,east,north")
    w, s, e, n = parts
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError(f"invalid bbox {parts}")
    return (w, s, e, n)


def in_bbox(lat: float, lon: float, bbox: BBox) -> bool:
    w, s, e, n = bbox
    return s <= lat <= n and w <= lon <= e


def wkt_point(lat: float, lon: float) -> str:
    return f"POINT({lon:.6f} {lat:.6f})"


def parse_wkt_point(wkt: str) -> tuple[float, float]:
    """Returns (lat, lon) from 'POINT(lon lat)' (optionally with a CRS IRI prefix)."""
    body = wkt[wkt.upper().index("POINT") :]
    inside = body[body.index("(") + 1 : body.index(")")].split()
    lon, lat = float(inside[0]), float(inside[1])
    return lat, lon


def grid_cell(lat: float, lon: float, cell_deg: float) -> tuple[int, int]:
    return (math.floor(lat / cell_deg), math.floor(lon / cell_deg))


def point_in_polygon(lat: float, lon: float, ring: Iterable[Sequence[float]]) -> bool:
    """Ray casting; ring is [[lon, lat], ...] as in GeoJSON."""
    pts = list(ring)
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i][0], pts[i][1]
        xj, yj = pts[j][0], pts[j][1]
        if (yi > lat) != (yj > lat):
            x_cross = (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
            if lon < x_cross:
                inside = not inside
        j = i
    return inside
