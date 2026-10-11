"""graphdb-mcp: FFOnto CQ01-CQ26 (via ffonto-v2) + live CQs, guarded SPARQL, text-to-SPARQL, live rules.

All queries pass the SPARQL guard here (read-only, row limit, ontology terms only),
so no agent prompt can bypass it. There is no generic write tool.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from ..tools.toolkit import Toolkit
from . import in_thread, run

server = MCPServer("graphdb-mcp", instructions="FFOnto knowledge graph in GraphDB.")
_tk = Toolkit()


@server.tool()
async def list_cqs(intent: str | None = None) -> dict:
    """Competency questions available (optionally only those serving an intent)."""
    return await in_thread(_tk.kg_list_cqs, intent=intent)


@server.tool()
async def run_cq(cq_id: str, params: dict) -> dict:
    """Fill and run a tested CQ template. Returns rows, the SPARQL used and a sufficiency verdict."""
    return await in_thread(_tk.kg_run_cq, cq_id=cq_id, params=params)


@server.tool()
async def run_sparql(query: str) -> dict:
    """Run a read-only SPARQL query after the guard (ontology terms only, LIMIT enforced)."""
    return await in_thread(_tk.kg_run_sparql, query=query)


@server.tool()
async def text2sparql(question: str, record: str | None = None, hints: list[str] | None = None,
                      date: str | None = None, event: str | None = None, mode: str | None = None) -> dict:
    """ffonto-v2 text-to-SPARQL ("plus": context binding, schema linking, template adaptation,
    execution feedback) for questions no CQ covers. `hints` = likely CQ ids."""
    return await in_thread(_tk.kg_text2sparql, question=question, record=record, hints=hints,
                           date=date, event=event, mode=mode)


@server.tool()
async def register_site(label: str, lat: float, lon: float) -> dict:
    """Register a query location as a Site in the live graph (the only site-level write)."""
    return await in_thread(_tk.kg_register_site, label=label, lat=lat, lon=lon)


@server.tool()
async def run_rules() -> dict:
    """Run live rules R09-R12 and rewrite the inferred-live graph; returns derived hazards."""
    return await in_thread(_tk.kg_run_rules)


if __name__ == "__main__":
    run(server)
