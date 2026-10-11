"""FWI reference values, FIRMS parsing, hotspot classification, lifting to RDF."""
from datetime import datetime, timedelta, timezone

from wildfire_mas.live.classify import HotspotClassifier, StaticSource
from wildfire_mas.live.firms import FirmsClient, Hotspot, parse_firms_csv
from wildfire_mas.live.fwi import FWIState, danger_class, step
from wildfire_mas.live.geo import bbox_around, haversine_km, parse_wkt_point, wkt_point
from wildfire_mas.live.lift import LiveLifter
from wildfire_mas.live.weather import noon_series


def test_fwi_matches_van_wagner_reference():
    # Standard test values (Van Wagner & Pickett 1985): day 1 from start codes 85/6/15
    def close(got, ref):
        return all(abs(g - r) <= 0.06 for g, r in zip(got, ref))

    st = FWIState()
    d = step(st, "2000-04-13", 17, 42, 25, 0, 46)
    assert close((d.ffmc, d.dmc, d.dc, d.isi, d.bui, d.fwi), (87.7, 8.5, 19.0, 10.9, 8.5, 10.1))
    d2 = step(st, "2000-04-14", 20, 21, 25, 2.4, 46)
    assert close((d2.ffmc, d2.dmc, d2.dc, d2.fwi), (86.2, 10.4, 23.6, 9.3))


def test_fwi_tropics_runs_and_danger_class(settings):
    d = step(FWIState(), "2026-03-10", 33, 45, 15, 0, 7.3)  # Sri Lanka dry season
    assert d.fwi > 0
    classes = settings.get("danger.fwi_classes")
    assert danger_class(4.0, classes) == "VeryLow"
    assert danger_class(25.0, classes) == "High"
    assert danger_class(80.0, classes) == "Extreme"


def test_parse_viirs_and_modis(fixtures):
    v = parse_firms_csv((fixtures / "firms_viirs.csv").read_text(), "VIIRS_NOAA20_NRT")
    assert len(v) == 4 and v[0].acq_datetime == datetime(2026, 10, 8, 7, 48, tzinfo=timezone.utc)
    assert v[1].confidence_score == 0.9 and v[1].brightness == 355.9 and v[1].fire_type is None
    m = parse_firms_csv((fixtures / "firms_modis_sp.csv").read_text(), "MODIS_SP")
    assert [h.fire_type for h in m] == [1, 0, 2, 0]
    assert m[0].confidence_score == 0.85


def test_firms_error_text_raises():
    import pytest

    from wildfire_mas.live.firms import FirmsError

    with pytest.raises(FirmsError):
        parse_firms_csv("Invalid MAP_KEY.", "VIIRS_SNPP_NRT")


def test_classifier_labels_non_wildfires(settings, fixtures):
    cfg = settings.get("hotspot_filter")
    # a refinery-like static source at the 4th VIIRS point
    clf = HotspotClassifier(cfg, [StaticSource("test refinery", 6.9520, 79.8940, "industrial")])
    hs = parse_firms_csv((fixtures / "firms_viirs.csv").read_text(), "VIIRS_NOAA20_NRT")
    clf.classify(hs)
    cats = [h.category for h in hs]
    assert cats == ["likely_vegetation_fire", "likely_vegetation_fire", "low_confidence", "static_land_source"]
    assert [h.wildfire_evidence for h in hs] == [True, True, False, False]

    m = parse_firms_csv((fixtures / "firms_modis_sp.csv").read_text(), "MODIS_SP")
    HotspotClassifier(cfg).classify(m)
    assert [h.category for h in m] == ["volcanic", "likely_vegetation_fire", "static_land_source", "low_confidence"]


def test_persistence_flags_static_heat(settings):
    cfg = settings.get("hotspot_filter")
    base = datetime(2026, 10, 8, 7, 48, tzinfo=timezone.utc)
    mk = lambda lat, lon, t: Hotspot("VIIRS_SNPP_NRT", lat, lon, t, "N", "VIIRS", "n", 0.6, 5.0, 330, 300, "D")
    history = [mk(8.0201, 79.7201, base - timedelta(days=i)) for i in range(12)]  # 12 distinct days
    now = [mk(8.0202, 79.7203, base), mk(7.5, 80.5, base)]
    HotspotClassifier(cfg).classify(now, history)
    assert now[0].category == "persistent_static_source" and now[1].category == "likely_vegetation_fire"


def test_firms_urls_and_redaction(settings):
    settings.raw["firms"]["map_key"] = "SECRETKEY123"
    c = FirmsClient(settings)
    u = c.area_url("VIIRS_SNPP_NRT", "79.5,5.85,82,9.9", 3)
    assert u.endswith("/api/area/csv/SECRETKEY123/VIIRS_SNPP_NRT/79.5,5.85,82,9.9/3")
    from datetime import date

    u2 = c.area_url("MODIS_NRT", "world", 5, end=date(2026, 10, 8))
    assert u2.endswith("/MODIS_NRT/world/5/2026-10-04")  # FIRMS [DATE] is the window start
    assert "SECRETKEY123" not in c._redact(u)


def test_geo_helpers():
    assert abs(haversine_km(7.29, 80.63, 7.29, 80.73) - 11.0) < 0.2
    w, s, e, n = bbox_around(7.29, 80.63, 10)
    assert s < 7.29 < n and w < 80.63 < e
    assert parse_wkt_point(wkt_point(7.5, 80.25)) == (7.5, 80.25)


def test_lift_hotspots_turtle(settings, fixtures):
    from wildfire_mas.live.firms import FirmsResult

    hs = parse_firms_csv((fixtures / "firms_viirs.csv").read_text(), "VIIRS_NOAA20_NRT")
    HotspotClassifier(settings.get("hotspot_filter")).classify(hs)
    res = FirmsResult(hs, ["VIIRS_NOAA20_NRT"], "79.5,5.85,82,9.9", 1, None, "2026-10-08T08:00:00+00:00",
                      ["https://firms.modaps.eosdis.nasa.gov/api/area/csv/<MAP_KEY>/VIIRS_NOAA20_NRT/x/1"], False)
    ttl = LiveLifter(settings).hotspots(res).serialize()
    assert ttl.count("a ffo:SatelliteHotspot") == 4
    assert ttl.count("a ffo:HotspotClassification") == 4 and "ffo:aboutHotspot" in ttl
    assert '"0.9"^^<http://www.w3.org/2001/XMLSchema#double>' in ttl     # FFOnto ranges are xsd:double
    assert "ffv:LowConfidenceDetection" in ttl and "<MAP_KEY>" in ttl
    assert '"POINT(80.635100 7.301200)"^^<http://www.opengis.net/ont/geosparql#wktLiteral>' in ttl
    assert '"true"^^xsd:boolean' in ttl


def test_noon_series_rain_window():
    times, temp, rh, wind, pr = [], [], [], [], []
    start = datetime(2026, 10, 1, 0, 0)
    for i in range(72):
        t = start + timedelta(hours=i)
        times.append(t.strftime("%Y-%m-%dT%H:%M"))
        temp.append(30.0)
        rh.append(60.0)
        wind.append(10.0)
        pr.append(1.0 if (t.day == 2 and t.hour == 18) else 0.0)  # rain after noon on day 2
    days = noon_series({"hourly": {"time": times, "temperature_2m": temp, "relative_humidity_2m": rh,
                                   "wind_speed_10m": wind, "precipitation": pr}})
    assert [x["date"] for x in days] == ["2026-10-02", "2026-10-03"]
    assert days[0]["rain"] == 0.0 and days[1]["rain"] == 1.0  # counts toward the next day's FWI
