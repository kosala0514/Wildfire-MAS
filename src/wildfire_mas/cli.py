"""Command line: wfmas ask | hotspots | fwi | rules | test-cqs | shacl | key-status"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import get_settings


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_ask(a) -> int:
    from .agents.state import Request
    from .orchestrator import build

    orch = build(transport=a.transport)
    req = Request(a.question, record=a.record, lat=a.lat, lon=a.lon, place=a.place, area=a.area, role=a.role)

    async def go():
        try:
            return await orch.ask(req)
        finally:
            await orch.bus.aclose()

    res = asyncio.run(go())
    print(res.answer_markdown or "(no answer)")
    print(f"\n-- intent {res.intent} | urgency {res.urgency} | {res.timings.get('total', 0):.1f}s | "
          f"{len(res.tool_calls)} tool calls | {len(res.llm_calls)} LLM calls")
    for e in res.evidence:
        print(f"[{e['id']}] {e['source']}")
    for i in res.issues:
        print(f"! {i}")
    if a.json:
        with open(a.json, "w") as f:
            f.write(res.to_json())
        print(f"full trace written to {a.json}")
    return 0


def cmd_hotspots(a) -> int:
    from .tools.toolkit import Toolkit

    out = Toolkit().firms_get_hotspots(area=a.area, lat=a.lat, lon=a.lon, radius_km=a.radius_km, days=a.days,
                                       persistence=not a.no_persistence, store=not a.no_store,
                                       max_items=a.max_items)
    if a.brief:
        out = {k: v for k, v in out.items() if k != "hotspots"}
    _print(out)
    return 0


def cmd_fwi(a) -> int:
    from .tools.toolkit import Toolkit

    _print(Toolkit().weather_fwi(a.lat, a.lon, store=not a.no_store))
    return 0


def cmd_rules(a) -> int:
    from .tools.toolkit import Toolkit

    _print(Toolkit().kg_run_rules())
    return 0


def cmd_key(a) -> int:
    from .tools.toolkit import Toolkit

    _print(Toolkit().firms_key_status())
    return 0


def cmd_test_cqs(a) -> int:
    from .tools.toolkit import Toolkit

    tk = Toolkit()
    failed = 0
    tested = 0
    for cq in tk.catalog.by_id.values():
        if not cq.expect:
            continue
        tested += 1
        out = tk.kg_run_cq(cq.id, cq.expect.get("params", {}))
        rows = out.get("rows", [])
        ok = out.get("ok") and len(rows) >= int(cq.expect.get("min_rows", 1))
        for var, needle in (cq.expect.get("contains") or {}).items():
            ok = ok and any(needle in str(r.get(var, "")) for r in rows)
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {cq.id} ({len(rows)} rows){'' if ok else ' ' + str(out.get('errors', ''))}")
    print(f"{tested - failed}/{tested} CQ tests passed")
    return 1 if failed else 0


def cmd_ontology(a) -> int:
    """Load / unload the live-module .ttl files into GraphDB, or clear the live data graphs."""
    from pathlib import Path

    from .kg.store import make_store

    s = get_settings()
    store = make_store(s)
    if a.action == "clear-live":
        for g in ("live", "inferred_live"):
            store.clear_graph(s.graph(g))
            print(f"cleared {s.graph(g)}")
        return 0
    src = Path(a.dir) if a.dir else s.root / "ontology"
    pairs = [(src / "ffonto-live.ttl", s.graph("ontology")), (src / "ffonto-live-vocab.ttl", s.graph("vocab"))]
    for f, graph in pairs:
        if not f.exists():
            print(f"skip {f}: not found")
            continue
        if a.action == "load":
            store.insert_turtle(f.read_text(encoding="utf-8"), graph)
            print(f"loaded {f.name} into {graph}")
            continue
        from rdflib import BNode, Graph

        g = Graph().parse(f, format="turtle")
        triples = [t for t in g if not any(isinstance(x, BNode) for x in t)]
        for i in range(0, len(triples), 500):
            body = "\n".join(f"{t[0].n3()} {t[1].n3()} {t[2].n3()} ." for t in triples[i:i + 500])
            store.update(f"DELETE DATA {{ GRAPH <{graph}> {{\n{body}\n}} }}")
        print(f"removed {len(triples)} triples of {f.name} from {graph}")
    return 0


def cmd_golden_import(a) -> int:
    """Part A of the golden set from ffonto-v2's labelled ForestFire-VQA questions."""
    import random

    from .kg.ffonto_bridge import FfontoBridge

    s = get_settings()
    b = FfontoBridge(s)
    if not b.configured:
        print("set FFONTO_V2_PATH (or ffonto_v2.path) to your ffonto-v2 folder first")
        return 1
    items = b.all_examples()
    rnd = random.Random(a.seed)
    order = list(range(len(items)))
    rnd.shuffle(order)
    n_dev = int(len(items) * a.dev)
    dev = set(order[:n_dev])
    out = s.path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, e in enumerate(items):
            oos = (e["label"] or "").upper() == "OOS"
            f.write(json.dumps({
                "id": f"A{i + 1:04d}", "part": "A", "question": e["question"],
                "gold_intent": e["intent"], "gold_cq": None if oos else e["label"],
                "gold_route": "rag" if oos else "kg", "modality": e.get("modality"),
                "gold_answer": e.get("answer") or "", "source": e.get("source") or "",
                "split": "dev" if i in dev else "test"}, ensure_ascii=False) + "\n")
    n_kg = sum(1 for e in items if (e["label"] or "").upper() != "OOS")
    print(f"wrote {len(items)} items ({n_kg} KG, {len(items) - n_kg} general knowledge; "
          f"{n_dev} dev / {len(items) - n_dev} test) to {out}")
    print("labels are silver (keyword rules + review in FFVQA_CQ_mapping.xlsx): review the confusion pairs "
          "after the first run, as docs/EVALUATION.md in ffonto-v2 says")
    return 0


def cmd_shacl(a) -> int:
    """Validate the live + inferred-live graphs against ontology/shapes-live.ttl (needs pyshacl)."""
    import requests
    from pyshacl import validate
    from rdflib import Graph

    s = get_settings()
    g = Graph()
    if s.get("graphdb.backend") == "memory":
        print("shacl: memory backend holds no persistent data; use GraphDB")
        return 1
    ep = f"{s.get('graphdb.url').rstrip('/')}/repositories/{s.get('graphdb.repository')}/statements"
    for name in ("live", "inferred_live", "vocab"):
        r = requests.get(ep, params={"context": f"<{s.graph(name)}>"}, headers={"Accept": "text/turtle"}, timeout=120)
        r.raise_for_status()
        g.parse(data=r.text, format="turtle")
    conforms, _, text = validate(g, shacl_graph=str(s.root / "ontology" / "shapes-live.ttl"), inference="none")
    print(text)
    return 0 if conforms else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wfmas")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask", help="ask the multi-agent system a question")
    p.add_argument("question")
    p.add_argument("--record")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--place")
    p.add_argument("--area", help="'w,s,e,n', 'world' or ISO3 country code")
    p.add_argument("--role", default="analyst", choices=["analyst", "incident_commander", "public"])
    p.add_argument("--transport", default="inprocess", choices=["inprocess", "mcp"])
    p.add_argument("--json", help="write the full trace (plan, evidence, calls) to this file")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("hotspots", help="fetch + label FIRMS hotspots (and store them in the live graph)")
    p.add_argument("--area", help="'w,s,e,n', 'world' or ISO3 (default: configured region)")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--radius-km", type=float, default=25)
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--max-items", type=int, default=50)
    p.add_argument("--no-store", action="store_true")
    p.add_argument("--no-persistence", action="store_true")
    p.add_argument("--brief", action="store_true")
    p.set_defaults(fn=cmd_hotspots)

    p = sub.add_parser("fwi", help="compute today's FWI at a point")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--no-store", action="store_true")
    p.set_defaults(fn=cmd_fwi)

    sub.add_parser("rules", help="run live rules R09-R12").set_defaults(fn=cmd_rules)
    sub.add_parser("key-status", help="FIRMS MAP_KEY usage").set_defaults(fn=cmd_key)
    sub.add_parser("test-cqs", help="run CQ tests (entries with an expect block)").set_defaults(fn=cmd_test_cqs)
    sub.add_parser("shacl", help="validate live graphs with SHACL").set_defaults(fn=cmd_shacl)

    p = sub.add_parser("ontology", help="load/unload the live-module ontology in GraphDB, or clear live data")
    p.add_argument("action", choices=["load", "unload", "clear-live"])
    p.add_argument("--dir", help="folder with ffonto-live.ttl / ffonto-live-vocab.ttl (default: ontology/)")
    p.set_defaults(fn=cmd_ontology)

    p = sub.add_parser("golden-import", help="build golden Part A from ffonto-v2's labelled questions")
    p.add_argument("--out", default="eval/datasets/golden_partA.jsonl")
    p.add_argument("--dev", type=float, default=0.3, help="share of items in the dev split")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_golden_import)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
