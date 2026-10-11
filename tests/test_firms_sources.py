"""One FIRMS source rejected (HTTP 400) must not fail the whole request."""
import pytest

from wildfire_mas.live.firms import FirmsClient, FirmsError


class FakeResp:
    def __init__(self, code, text):
        self.status_code, self.text = code, text


class FakeHTTP:
    def __init__(self, ok_csv):
        self.ok_csv = ok_csv

    def get(self, url, timeout=60):
        return FakeResp(400, "Invalid API call.") if "NOAA21" in url else FakeResp(200, self.ok_csv)


def test_bad_source_is_a_warning_not_a_crash(settings, fixtures):
    settings.raw["firms"].update(map_key="KEY", cache_dir=None)
    c = FirmsClient(settings, session=FakeHTTP((fixtures / "firms_viirs.csv").read_text()))
    res = c.get_country("BRA", 1, ["VIIRS_NOAA21_NRT", "VIIRS_NOAA20_NRT"])
    assert res.sources == ["VIIRS_NOAA20_NRT"] and len(res.hotspots) == 4
    assert "VIIRS_NOAA21_NRT" in res.warnings[0] and "Invalid API call" in res.warnings[0]
    with pytest.raises(FirmsError):
        c.get_country("BRA", 1, ["VIIRS_NOAA21_NRT"])
