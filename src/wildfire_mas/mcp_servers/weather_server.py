"""weather-mcp: current weather, computed FWI (with danger class), and geocoding."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ..tools.toolkit import Toolkit
from . import in_thread, run

server = MCPServer("weather-mcp", instructions="Open-Meteo weather; FWI computed locally (Van Wagner 1987).")
_tk = Toolkit()


@server.tool()
async def current(lat: float, lon: float, site: str | None = None, store: bool = True) -> dict:
    """Current temperature, humidity, wind and precipitation at a point."""
    return await in_thread(_tk.weather_current, lat=lat, lon=lon, site=site, store=store)


@server.tool()
async def fwi(lat: float, lon: float, site: str | None = None, store: bool = True) -> dict:
    """Today's FWI components, danger class and 7-day trend at a point."""
    return await in_thread(_tk.weather_fwi, lat=lat, lon=lon, site=site, store=store)


@server.tool()
async def geocode(name: str) -> dict:
    """Resolve a place name to coordinates."""
    return await in_thread(_tk.geo_geocode, name=name)


if __name__ == "__main__":
    run(server)
