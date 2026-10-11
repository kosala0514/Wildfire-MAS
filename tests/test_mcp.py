"""The MCP layer: servers expose the toolkit, MCPBus calls them (in-process transport)."""
import asyncio

from fakes import RoutingStore

from wildfire_mas.tools.bus import MCPBus
from wildfire_mas.tools.toolkit import Toolkit


def test_mcp_servers_roundtrip(settings):
    import wildfire_mas.mcp_servers.graphdb_server as gs
    import wildfire_mas.mcp_servers.rag_server as rs

    store = RoutingStore([("?site ?label ?lat ?lon ?wkt", [{"site": "http://purl.org/ffonto/data/site/s1",
                                                            "label": "Ridge", "lat": 7.3, "lon": 80.6}])])
    tk = Toolkit(settings, store=store)
    tk.guard.allowed_terms |= {settings.ns["ffo"] + t for t in ("Site", "siteName")}  # FFOnto core terms (ffonto_v2.path)
    gs._tk = tk
    rs._tk = tk

    async def go():
        bus = MCPBus({"graphdb": gs.server, "rag": rs.server})
        try:
            cqs = await bus.call("kg.list_cqs", {"intent": "trend"})
            sites = await bus.call("kg.run_cq", {"cq_id": "CQL09", "params": {}})
            bad = await bus.call("kg.run_sparql", {"query": "DELETE WHERE { ?s ?p ?o }"})
            hits = await bus.call("rag.search", {"query": "how does FWI relate to fire spread", "k": 2})
            return cqs, sites, bad, hits
        finally:
            await bus.aclose()

    cqs, sites, bad, hits = asyncio.run(go())
    assert any(c["id"] == "CQL03" for c in cqs["cqs"])
    assert sites["ok"] and sites["rows"][0]["label"] == "Ridge" and "LIMIT" in sites["sparql"]
    assert not bad["ok"] and "read-only" in bad["errors"][0]
    assert hits["hits"][0]["id"] == "demo_fwi_1"


def test_toolkit_register_site_writes_only_live_graph(settings):
    store = RoutingStore([])
    out = Toolkit(settings, store=store).kg_register_site("Kandy", 7.2906, 80.6337)
    graph, ttl = store.inserted[0]
    assert graph == settings.graph("live") and "a ffo:Site" in ttl and "ffo:isQueryLocation" in ttl
    assert out["site"].startswith(settings.ns["data"] + "querysite/")


def test_mcp_bus_parallel_calls_then_close(settings):
    """Agents call tools from parallel tasks; the bus must still close cleanly (anyio cancel scopes)."""
    import wildfire_mas.mcp_servers.graphdb_server as gs
    import wildfire_mas.mcp_servers.rag_server as rs

    tk = Toolkit(settings, store=RoutingStore([]))
    gs._tk = tk
    rs._tk = tk

    async def go():
        bus = MCPBus({"graphdb": gs.server, "rag": rs.server})
        try:
            return await asyncio.gather(*(bus.call("rag.search", {"query": q, "k": 1})
                                          for q in ("fwi", "propane tank", "gas flares")),
                                        bus.call("kg.list_cqs", {}))
        finally:
            await bus.aclose()

    out = asyncio.run(go())
    assert out[0]["hits"][0]["id"] == "demo_fwi_1" and len(out[3]["cqs"]) >= 9
