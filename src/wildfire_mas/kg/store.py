"""Graph store backends: GraphDB (RDF4J REST) and an in-memory rdflib store for demos/tests.

Both expose the same small interface used by the agents and the MCP server:
    select(query) -> list[dict]      (values converted to Python types)
    ask(query) -> bool
    update(sparql_update) -> None    (write path; only live-data and rule engine use it)
    insert_turtle(turtle, graph)     (bulk load into a named graph)
    clear_graph(graph)
"""
from __future__ import annotations

from typing import Any, Protocol

import requests

_NUM = {
    "http://www.w3.org/2001/XMLSchema#integer": int,
    "http://www.w3.org/2001/XMLSchema#int": int,
    "http://www.w3.org/2001/XMLSchema#long": int,
    "http://www.w3.org/2001/XMLSchema#decimal": float,
    "http://www.w3.org/2001/XMLSchema#double": float,
    "http://www.w3.org/2001/XMLSchema#float": float,
}
_BOOL = "http://www.w3.org/2001/XMLSchema#boolean"


def convert_binding(b: dict) -> Any:
    v = b.get("value")
    dt = b.get("datatype")
    if dt in _NUM:
        try:
            return _NUM[dt](v)
        except ValueError:
            return v
    if dt == _BOOL:
        return v in ("true", "1")
    return v


class GraphStore(Protocol):
    def select(self, query: str) -> list[dict]: ...
    def ask(self, query: str) -> bool: ...
    def update(self, update: str) -> None: ...
    def insert_turtle(self, turtle: str, graph: str) -> None: ...
    def clear_graph(self, graph: str) -> None: ...


class GraphDBStore:
    def __init__(self, url: str, repository: str, user: str | None = None,
                 password: str | None = None, timeout_s: float = 30):
        self.endpoint = f"{url.rstrip('/')}/repositories/{repository}"
        self.http = requests.Session()
        if user:
            self.http.auth = (user, password or "")
        self.timeout = timeout_s

    def _query(self, query: str) -> dict:
        r = self.http.post(self.endpoint, data={"query": query},
                           headers={"Accept": "application/sparql-results+json"}, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"GraphDB query failed ({r.status_code}): {r.text[:300]}")
        return r.json()

    def select(self, query: str) -> list[dict]:
        data = self._query(query)
        return [{k: convert_binding(v) for k, v in row.items()} for row in data["results"]["bindings"]]

    def ask(self, query: str) -> bool:
        return bool(self._query(query).get("boolean"))

    def update(self, update: str) -> None:
        r = self.http.post(f"{self.endpoint}/statements", data={"update": update}, timeout=self.timeout)
        if r.status_code not in (200, 204):
            raise RuntimeError(f"GraphDB update failed ({r.status_code}): {r.text[:300]}")

    def insert_turtle(self, turtle: str, graph: str) -> None:
        r = self.http.post(f"{self.endpoint}/statements", params={"context": f"<{graph}>"},
                           data=turtle.encode(), headers={"Content-Type": "text/turtle"}, timeout=self.timeout)
        if r.status_code not in (200, 204):
            raise RuntimeError(f"GraphDB insert failed ({r.status_code}): {r.text[:300]}")

    def clear_graph(self, graph: str) -> None:
        self.update(f"CLEAR SILENT GRAPH <{graph}>")


class MemoryStore:
    """rdflib Dataset with default-graph-as-union, so queries without GRAPH see everything
    (same behaviour as GraphDB's default query over all named graphs)."""

    def __init__(self):
        from rdflib import Dataset

        self.ds = Dataset(default_union=True)

    def load_file(self, path: str, graph: str) -> None:
        from rdflib import URIRef

        self.ds.graph(URIRef(graph)).parse(path)

    def select(self, query: str) -> list[dict]:
        res = self.ds.query(query)
        out = []
        for row in res:
            d = {}
            for var in res.vars:
                val = row[var]
                if val is None:
                    continue
                py = val.toPython() if hasattr(val, "toPython") else val
                if hasattr(py, "isoformat"):
                    py = py.isoformat()
                elif type(py).__name__ == "Decimal":
                    py = float(py)
                d[str(var)] = py if isinstance(py, (int, float, bool)) else str(val)
            out.append(d)
        return out

    def ask(self, query: str) -> bool:
        return bool(self.ds.query(query).askAnswer)

    def update(self, update: str) -> None:
        self.ds.update(update)

    def insert_turtle(self, turtle: str, graph: str) -> None:
        from rdflib import URIRef

        self.ds.graph(URIRef(graph)).parse(data=turtle, format="turtle")

    def clear_graph(self, graph: str) -> None:
        from rdflib import URIRef

        self.ds.remove_graph(self.ds.graph(URIRef(graph)))


def make_store(settings) -> GraphStore:
    g = settings.get("graphdb")
    if g.get("backend") == "memory":
        from .guard import ontology_files

        store = MemoryStore()
        for f in ontology_files(settings):          # live module + FFOnto (ffonto_v2.path)
            graph = settings.graph("vocab" if "vocab" in f.name else "ontology")
            store.load_file(str(f), graph)
        return store
    return GraphDBStore(g["url"] or "http://localhost:7200", g["repository"] or "ffonto",
                        g.get("user"), g.get("password"), g.get("timeout_s", 30))
