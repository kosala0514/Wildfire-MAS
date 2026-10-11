"""SPARQL guard for LLM-written queries: read-only, bounded, ontology terms only.

Same policy as the existing system (step 10a), now enforced inside the GraphDB MCP
server so no agent or prompt can bypass it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_UPDATE_KW = re.compile(
    r"\b(INSERT|DELETE|LOAD|CLEAR|DROP|CREATE|ADD|MOVE|COPY|WITH)\b", re.I)
_SERVICE = re.compile(r"\bSERVICE\b", re.I)
_FORM = re.compile(r"\b(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", re.I)
_PREFIX_DECL = re.compile(r"PREFIX\s+([A-Za-z][\w-]*)?:\s*<([^>]*)>", re.I)
_FULL_IRI = re.compile(r"<([^<>\s]+)>")
_PNAME = re.compile(r"(?<![\w<\"'])([A-Za-z][\w-]*)?:([A-Za-z_][\w.-]*)")
_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*$", re.I)
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_COMMENT = re.compile(r"#[^\n]*")


@dataclass
class GuardResult:
    ok: bool
    query: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SparqlGuard:
    def __init__(self, allowed_terms: set[str], own_namespaces: list[str], extra_prefixes: list[str],
                 max_rows: int = 500, allow_service: bool = False, default_prefixes: dict[str, str] | None = None):
        self.allowed_terms = allowed_terms          # full IRIs of ontology + vocab terms
        self.own_ns = own_namespaces                # e.g. ffo, ffv namespaces (terms must exist)
        self.extra = extra_prefixes                 # namespaces allowed wholesale (rdf, rdfs, skos, geo...)
        self.data_ns: list[str] = []                # instance namespaces (always allowed)
        self.max_rows = max_rows
        self.allow_service = allow_service
        self.default_prefixes = default_prefixes or {}

    def _strip(self, q: str) -> str:
        q = _STRING.sub('""', q)
        # remove comments not inside IRIs
        return "\n".join(_COMMENT.sub("", line) if "<" not in line else line for line in q.splitlines())

    def check(self, query: str) -> GuardResult:
        errors: list[str] = []
        warnings: list[str] = []
        body = self._strip(query)
        if _UPDATE_KW.search(re.sub(r"<[^>]*>", "", body)):
            errors.append("query is not read-only (update keyword found)")
        if not self.allow_service and _SERVICE.search(body):
            errors.append("SERVICE (federated query) is not allowed")
        if not _FORM.search(body):
            errors.append("no SELECT/ASK/CONSTRUCT/DESCRIBE form")

        prefixes = dict(self.default_prefixes)
        for pfx, ns in _PREFIX_DECL.findall(body):
            prefixes[pfx or ""] = ns
        body_no_decl = _PREFIX_DECL.sub("", body)

        unknown: set[str] = set()
        for m in _FULL_IRI.finditer(body_no_decl):
            self._check_iri(m.group(1), unknown)
        for pfx, local in _PNAME.findall(body_no_decl):
            if pfx in ("http", "https", "urn", "mailto"):
                continue
            if pfx not in prefixes:
                errors.append(f"undeclared prefix '{pfx}:'")
                continue
            self._check_iri(prefixes[pfx] + local, unknown)
        if unknown:
            errors.append("terms not in the ontology/vocabulary: " + ", ".join(sorted(unknown)[:10]))

        # rewrite with prefixes + bounded LIMIT
        q = query.strip()
        missing = [p for p in prefixes if p in self.default_prefixes
                   and not re.search(rf"PREFIX\s+{re.escape(p)}:", q, re.I)
                   and re.search(rf"(?<![\w<]){re.escape(p)}:", q)]
        if missing:
            q = "\n".join(f"PREFIX {p}: <{self.default_prefixes[p]}>" for p in missing) + "\n" + q
        if re.search(r"\bSELECT\b", body, re.I):
            m = _LIMIT.search(q)
            if not m:
                q += f"\nLIMIT {self.max_rows}"
                warnings.append(f"added LIMIT {self.max_rows}")
            elif int(m.group(1)) > self.max_rows:
                q = _LIMIT.sub(f"LIMIT {self.max_rows}", q)
                warnings.append(f"LIMIT capped at {self.max_rows}")
        if not errors:
            errors += self._syntax(q)
        return GuardResult(not errors, q, errors, warnings)

    def _check_iri(self, iri: str, unknown: set[str]) -> None:
        if any(iri.startswith(ns) for ns in self.extra + self.data_ns):
            return
        if any(iri.startswith(ns) for ns in self.own_ns):
            if iri not in self.allowed_terms:
                unknown.add(iri)
            return
        unknown.add(iri)

    @staticmethod
    def _syntax(q: str) -> list[str]:
        try:
            from rdflib.plugins.sparql import prepareQuery
        except ImportError:
            return []  # syntax is then checked by GraphDB itself
        try:
            prepareQuery(q)
            return []
        except Exception as e:  # noqa: BLE001
            return [f"syntax error: {str(e)[:200]}"]


_DECL = re.compile(r"(?m)^\s*([A-Za-z][\w-]*):([\w.-]+)\s+(?:a|rdf:type)\s")
_TTL_PREFIX = re.compile(r"@prefix\s+([\w-]*):\s*<([^>]*)>\s*\.", re.I)


def terms_from_turtle(text: str) -> set[str]:
    """Cheap term harvest from Turtle files: every subject declared with `a`."""
    prefixes = dict(_TTL_PREFIX.findall(text))
    out = set()
    for pfx, local in _DECL.findall(text):
        if pfx in prefixes:
            out.add(prefixes[pfx] + local)
    return out


def ontology_files(settings) -> list:
    """The live module's .ttl files plus FFOnto's own (from ffonto_v2.path), shapes excluded."""
    from pathlib import Path

    files = sorted((settings.root / settings.get("paths.ontology_dir")).glob("*.ttl"))
    v2 = settings.get("ffonto_v2.path")
    if v2 and (Path(v2).expanduser() / "ontology").is_dir():
        files += sorted((Path(v2).expanduser() / "ontology").glob("*.ttl"))
    return [f for f in files if "shapes" not in f.name]


def build_guard(settings, store=None) -> SparqlGuard:
    """Allowed terms = terms declared in ontology/*.ttl, plus (if a store is given) every
    class/property/concept already in the ontology + vocab graphs of GraphDB."""
    ns = settings.ns
    terms: set[str] = set()
    for f in ontology_files(settings):
        terms |= terms_from_turtle(f.read_text(encoding="utf-8"))
    if store is not None:
        q = f"""SELECT DISTINCT ?t WHERE {{
          VALUES ?g {{ <{settings.graph('ontology')}> <{settings.graph('vocab')}> }}
          GRAPH ?g {{ ?t a ?k }} FILTER(isIRI(?t)) }}"""
        try:
            terms |= {r["t"] for r in store.select(q)}
        except Exception:  # noqa: BLE001 - store offline: fall back to file terms
            pass
    gcfg = settings.get("sparql_guard")
    std = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",
        "owl": "http://www.w3.org/2002/07/owl#",
        "skos": "http://www.w3.org/2004/02/skos/core#",
        "prov": "http://www.w3.org/ns/prov#",
        "sosa": "http://www.w3.org/ns/sosa/",
        "dcterms": "http://purl.org/dc/terms/",
        "geo": "http://www.opengis.net/ont/geosparql#",
        "geof": "http://www.opengis.net/def/function/geosparql/",
        "uom": "http://www.opengis.net/def/uom/OGC/1.0/",
        "ffo": ns["ffo"], "ffv": ns["ffv"], "data": ns["data"],
        # FFOnto's own prefix names, so its queries and LLM SPARQL written in its style pass too
        "ffonto": ns["ffo"], "ffontov": ns["ffv"], "ffd": ns["data"],
    }
    g = SparqlGuard(terms, [ns["ffo"], ns["ffv"]], gcfg.get("extra_allowed_prefixes", []),
                    gcfg.get("max_rows", 500), gcfg.get("allow_service", False), std)
    g.data_ns = [ns["data"]] + list(settings.get("ontology.graphs").values())
    return g
