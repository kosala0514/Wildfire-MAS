"""Test doubles: a routing fake graph store and a scripted tool bus."""
from __future__ import annotations

from wildfire_mas.tools.bus import BaseBus


class RoutingStore:
    """select() returns canned rows for the first route whose marker appears in the query."""

    def __init__(self, routes: list[tuple[str, list[dict]]]):
        self.routes = routes
        self.queries: list[str] = []
        self.inserted: list[tuple[str, str]] = []
        self.cleared: list[str] = []

    def select(self, query: str) -> list[dict]:
        self.queries.append(query)
        for marker, rows in self.routes:
            if marker in query:
                return rows
        return []

    def ask(self, query: str) -> bool:
        return False

    def update(self, update: str) -> None:
        self.queries.append(update)

    def insert_turtle(self, turtle: str, graph: str) -> None:
        self.inserted.append((graph, turtle))

    def clear_graph(self, graph: str) -> None:
        self.cleared.append(graph)


class ScriptedBus(BaseBus):
    """Tool bus returning canned outputs: handlers[tool] = dict | callable(args) -> dict."""

    def __init__(self, handlers: dict):
        super().__init__()
        self.handlers = handlers

    async def _call(self, tool: str, args: dict) -> dict:
        h = self.handlers.get(tool)
        if h is None:
            raise RuntimeError(f"no handler for {tool}")
        return h(args) if callable(h) else h
