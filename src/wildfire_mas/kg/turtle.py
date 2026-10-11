"""Minimal Turtle writer (no rdflib needed), with stable prefixes."""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Iterable

XSD = "http://www.w3.org/2001/XMLSchema#"
STD_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "xsd": XSD,
    "owl": "http://www.w3.org/2002/07/owl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "prov": "http://www.w3.org/ns/prov#",
    "geo": "http://www.opengis.net/ont/geosparql#",
}

_PNAME = re.compile(r"^[A-Za-z][\w-]*:[\w.-]*$")


class Lit:
    def __init__(self, value: Any, datatype: str | None = None, lang: str | None = None):
        self.value, self.datatype, self.lang = value, datatype, lang


def iri(v: str) -> str:
    return v if _PNAME.match(v) else f"<{v}>"


def lit(v: Any) -> str:
    if isinstance(v, Lit):
        s = _escape(str(v.value))
        if v.lang:
            return f'"{s}"@{v.lang}'
        return f'"{s}"^^{iri(v.datatype)}' if v.datatype else f'"{s}"'
    if isinstance(v, bool):
        return f'"{str(v).lower()}"^^xsd:boolean'
    if isinstance(v, int):
        return f'"{v}"^^xsd:integer'
    if isinstance(v, float):
        return f'"{v!r}"^^xsd:decimal' if "e" not in repr(v) else f'"{v!r}"^^xsd:double'
    if isinstance(v, datetime):
        return f'"{v.isoformat()}"^^xsd:dateTime'
    if isinstance(v, date):
        return f'"{v.isoformat()}"^^xsd:date'
    return f'"{_escape(str(v))}"'


def _escape(s: str) -> str:
    return (s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r"))


class TurtleDoc:
    """Collects triples as subject -> [(pred, obj_str)] and serialises them."""

    def __init__(self, prefixes: dict[str, str]):
        self.prefixes = {**STD_PREFIXES, **prefixes}
        self.subjects: dict[str, list[tuple[str, str]]] = {}

    def add(self, s: str, p: str, o: Any, *, is_iri: bool = False) -> None:
        if o is None:
            return
        obj = iri(o) if is_iri else lit(o)
        self.subjects.setdefault(iri(s), []).append((iri(p) if p != "a" else "a", obj))

    def add_iri(self, s: str, p: str, o: str | None) -> None:
        if o:
            self.add(s, p, o, is_iri=True)

    def add_all(self, s: str, p: str, objs: Iterable[Any]) -> None:
        for o in objs:
            self.add(s, p, o)

    def __len__(self) -> int:
        return sum(len(v) for v in self.subjects.values())

    def serialize(self, with_prefixes: bool = True) -> str:
        out = []
        if with_prefixes:
            out += [f"@prefix {k}: <{v}> ." for k, v in self.prefixes.items()]
            out.append("")
        for s, pos in self.subjects.items():
            body = " ;\n    ".join(f"{p} {o}" for p, o in pos)
            out.append(f"{s} {body} .")
        return "\n".join(out) + "\n"

    def sparql_prefixes(self) -> str:
        return "\n".join(f"PREFIX {k}: <{v}>" for k, v in self.prefixes.items())

    def as_insert_data(self, graph: str) -> str:
        triples = []
        for s, pos in self.subjects.items():
            body = " ;\n    ".join(f"{p} {o}" for p, o in pos)
            triples.append(f"{s} {body} .")
        return f"{self.sparql_prefixes()}\nINSERT DATA {{ GRAPH <{graph}> {{\n" + "\n".join(triples) + "\n} }"
