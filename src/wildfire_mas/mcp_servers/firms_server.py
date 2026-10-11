"""firms-mcp: NASA FIRMS hotspots, labelled (wildfire vs non-wildfire) and stored in the live graph."""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ..tools.toolkit import Toolkit
from . import in_thread, run

server = MCPServer(
    "firms-mcp",
    instructions=("NASA FIRMS active-fire / thermal-anomaly detections. FIRMS also detects volcanoes, "
                  "industrial heat and flares; every hotspot is labelled and only "
                  "'likely_vegetation_fire' counts as wildfire evidence."),
)
_tk = Toolkit()


@server.tool()
async def get_hotspots(area: str | None = None, lat: float | None = None, lon: float | None = None,
                       radius_km: float | None = None, days: int = 1, sources: list[str] | None = None,
                       persistence: bool = True, store: bool = True, max_items: int = 50) -> dict:
    """Hotspots for an area ('w,s,e,n', 'world', ISO3 country like 'LKA') or a point + radius,
    over 1..N days. Default area is the configured region (Sri Lanka). Labels each hotspot and,
    with store=true, writes it to the KG live graph."""
    return await in_thread(_tk.firms_get_hotspots, area=area, lat=lat, lon=lon, radius_km=radius_km,
                           days=days, sources=sources, persistence=persistence, store=store,
                           max_items=max_items)


@server.tool()
async def key_status() -> dict:
    """Current FIRMS MAP_KEY transaction usage."""
    return await in_thread(_tk.firms_key_status)


if __name__ == "__main__":
    run(server)
