"""Model sweep: vary the model in ONE slot, keep every other slot at its baseline.

    python -m wildfire_mas.evaluation.harness --slot planner --pool text \
        --dataset eval/datasets/intent.jsonl --repeats 3

    python -m wildfire_mas.evaluation.harness --slot scene_graph \
        --models qwen2.5vl:7b,llava:7b,gemma3:12b --dataset eval/datasets/scene.jsonl

Outputs in runs/<timestamp>_<slot>/:
    items.jsonl     every item x model x repeat (prediction, metrics, latency, tokens)
    summary.csv     one row per model: mean + 95% bootstrap CI per metric, latency, tokens
    summary.md      the same as a markdown table, plus McNemar tests vs the best model
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..agents.planner import Planner
from ..agents.prompts import SPARQL_SYSTEM
from ..agents.state import Evidence, Request
from ..agents.synthesis import GAP_TITLES, Synthesizer, Verifier
from ..config import Settings, load_settings
from ..kg.cq import CQCatalog
from ..llm import LLMClient, image_part, start_call_log
from .metrics import (bootstrap_ci, macro_f1, mape, mcnemar, numbers_unsupported, rows_match, set_prf)

Task = Callable[[dict, "Ctx"], Awaitable[dict]]


class Ctx:
    def __init__(self, settings: Settings, llm: LLMClient, toolkit=None):
        self.s = settings
        self.llm = llm
        self.catalog = CQCatalog.load(settings)
        self._tk = toolkit
        from ..kg.ffonto_bridge import FfontoBridge

        b = FfontoBridge(settings)
        self.bridge = b if b.configured and b.available() else None

    @property
    def tk(self):
        if self._tk is None:
            from ..tools.toolkit import Toolkit

            self._tk = Toolkit(self.s, llm=self.llm)
        return self._tk


# ------------------------------------------------------------------------------ tasks
async def task_planner(item: dict, c: Ctx) -> dict:
    """item: {question, intent|gold_intent, cq|gold_cq?, record?, modality?} (golden_partA.jsonl works as is)."""
    p = Planner(c.llm, c.catalog, c.s, c.bridge)
    gold_intent = item.get("gold_intent", item.get("intent"))
    gold_cq = item.get("gold_cq", item.get("cq"))
    record = item.get("record")
    if not record and item.get("modality") and "general" not in str(item["modality"]).lower():
        record = f"(a selected {item['modality']} record)"   # as in ffonto-v2's routing evaluation
    out = await p.classify(Request(item["question"], record=record))
    r = {"pred_intent": out["intent"], "gold_intent": gold_intent, "pred_cq": out.get("cq"), "gold_cq": gold_cq,
         "intent_correct": out["intent"] == gold_intent, "valid_json": out["_source"] == "llm",
         "source": out["_source"]}
    if gold_cq:
        r["cq_correct"] = out.get("cq") == gold_cq
    # route as in ffonto-v2 stage 1: KG vs general knowledge
    if item.get("gold_route"):
        r["route_correct"] = ("rag" if out["intent"] == "explanation" else "kg") == item["gold_route"]
    r["correct"] = r.get("cq_correct", r["intent_correct"])
    return r


async def task_sparql(item: dict, c: Ctx) -> dict:
    """item: {question, record?, gold_cq + gold_params | gold_sparql}. Needs the graph store."""
    tk = c.tk
    gold = (tk.kg_run_cq(item["gold_cq"], item.get("gold_params", {}))["rows"] if item.get("gold_cq")
            else tk.kg_run_sparql(item["gold_sparql"])["rows"])
    terms = tk.kg_run_sparql(f"""PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?t ?type WHERE {{ GRAPH <{c.s.graph('ontology')}> {{ ?t a ?type FILTER(isIRI(?t)) }} }} LIMIT 400""")
    vocab = "\n".join(f"{r['t']} ({r['type']})" for r in terms.get("rows", []))
    ns = c.s.ns
    user = (f"Prefixes: ffo: <{ns['ffo']}> ffv: <{ns['ffv']}>\nTerms:\n{vocab}\n\n"
            f"Selected record: {item.get('record') or 'none'}\nQuestion: {item['question']}")
    repaired = False
    try:
        q = (await c.llm.chat_json("sparql", SPARQL_SYSTEM, user)).get("sparql", "")
    except Exception:  # noqa: BLE001
        q = ""
    out = tk.kg_run_sparql(q) if q else {"ok": False, "errors": ["no JSON"]}
    if not out.get("ok") and q:
        repaired = True
        try:
            q = (await c.llm.chat_json("sparql", SPARQL_SYSTEM, user + f"\n\nYour query:\n{q}\nwas rejected: "
                                       f"{out.get('errors')}. Fix it.")).get("sparql", "")
            out = tk.kg_run_sparql(q)
        except Exception:  # noqa: BLE001
            pass
    exec_ok = bool(out.get("ok")) and rows_match(gold, out.get("rows", []))
    return {"valid": bool(out.get("ok")), "repaired": repaired, "exec_correct": exec_ok, "correct": exec_ok,
            "pred_sparql": q, "errors": out.get("errors")}


async def task_synthesizer(item: dict, c: Ctx) -> dict:
    """item: {question, intent, urgency?, role?, evidence: [{id, kind, summary, safety_critical?}]}"""
    ev = [Evidence(e["id"], e["kind"], e["summary"], e.get("source", "dataset"), "s0",
                   safety_critical=bool(e.get("safety_critical"))) for e in item["evidence"]]
    ans = await Synthesizer(c.llm).write(item["question"], item["intent"], item.get("urgency", "normal"),
                                         item.get("role", "analyst"), ev)
    ids = {e.id: e for e in ev}
    claims = [(t, s) for t, _, s in ans.sentences() if t not in GAP_TITLES]
    grounded = [s for _, s in claims if s.get("cites") and all(x in ids for x in s["cites"])]
    cited = {x for _, s in claims for x in s.get("cites", [])}
    safety = [e.id for e in ev if e.safety_critical]
    num_bad = sum(len(numbers_unsupported(s["text"], " ".join(ids[x].summary for x in s["cites"] if x in ids)))
                  for _, s in claims if s.get("cites"))
    problems = await Verifier().check(ans, ev)
    hazard_recall = (len(set(safety) & cited) / len(safety)) if safety else None
    return {"valid_json": ans.source == "llm", "n_claims": len(claims),
            "grounded_rate": len(grounded) / len(claims) if claims else 0.0,
            "hazard_recall": hazard_recall, "unsupported_numbers": num_bad,
            "verifier_problems": len(problems),
            "correct": ans.source == "llm" and not problems and (hazard_recall in (None, 1.0)),
            "answer": ans.markdown()}


async def task_scene_graph(item: dict, c: Ctx) -> dict:
    """item: {image, gold: {concept: count}}; vocabulary from eval/datasets/scene_vocab.txt"""
    vocab_file = c.s.root / "eval" / "datasets" / "scene_vocab.txt"
    vocab = [v.strip() for v in vocab_file.read_text().splitlines() if v.strip() and not v.startswith("#")]
    prompt = ("List what is visible in this wildfire scene using ONLY these concepts:\n" + ", ".join(vocab)
              + "\nReply JSON: {\"entities\": [{\"concept\": str, \"count\": int}], "
                "\"relations\": [{\"subject\": str, \"predicate\": str, \"object\": str}]}")
    img = Path(item["image"])
    img = img if img.is_absolute() else c.s.root / img
    try:
        out = await c.llm.chat_json("scene_graph", "You describe images using a controlled vocabulary only.",
                                    [{"type": "text", "text": prompt}, image_part(img)])
        ents = {e["concept"]: int(e.get("count", 1)) for e in out.get("entities", []) if isinstance(e, dict)}
        valid = True
    except Exception:  # noqa: BLE001
        ents, valid = {}, False
    gold = item["gold"]
    p, r, f1 = set_prf(gold, ents)
    in_vocab = [k for k in ents if k in vocab]
    halluc = [k for k in ents if k not in gold]
    counts = [ents[k] == gold[k] for k in gold if k in ents]
    return {"valid_json": valid, "precision": p, "recall": r, "f1": f1,
            "vocab_compliance": len(in_vocab) / len(ents) if ents else 1.0,
            "hallucination_rate": len(halluc) / len(ents) if ents else 0.0,
            "count_accuracy": sum(counts) / len(counts) if counts else None,
            "correct": f1 >= 0.8, "pred": ents}


async def task_figure_reader(item: dict, c: Ctx) -> dict:
    """item: {image, question, gold_value?: float, gold_trend?: 'up'|'down'|'flat'}"""
    img = Path(item["image"])
    img = img if img.is_absolute() else c.s.root / img
    prompt = (f"Question: {item['question']}\nReply JSON: {{\"value\": number|null, \"trend\": "
              "\"up\"|\"down\"|\"flat\"|null, \"answer\": str}}. Use null if the figure does not show it.")
    try:
        out = await c.llm.chat_json("figure_reader", "You read charts precisely.",
                                    [{"type": "text", "text": prompt}, image_part(img)])
        valid = True
    except Exception:  # noqa: BLE001
        out, valid = {}, False
    r: dict[str, Any] = {"valid_json": valid}
    if "gold_value" in item:
        v = out.get("value")
        try:
            v = float(v) if v is not None else None
        except (TypeError, ValueError):
            v = None
        r["pred_value"] = v
        r["ape"] = abs(v - item["gold_value"]) / abs(item["gold_value"]) * 100 if v is not None and item["gold_value"] else None
        r["correct"] = r["ape"] is not None and r["ape"] <= 5
    if "gold_trend" in item:
        r["trend_correct"] = out.get("trend") == item["gold_trend"]
        r["correct"] = r.get("correct", True) and r["trend_correct"]
    return r


async def task_e2e(item: dict, c: Ctx) -> dict:
    """item: {question, record?, place?, lat?, lon?, intent?, hazards?: [type,...]}.
    Use firms.replay_dir with frozen snapshots for reproducible runs."""
    from ..orchestrator import build

    orch = build(c.s, llm=c.llm, toolkit=c.tk)
    res = await orch.ask(Request(item["question"], record=item.get("record"), place=item.get("place"),
                                 lat=item.get("lat"), lon=item.get("lon"), role=item.get("role", "analyst")))
    cited = {x for s in res.sections for sent in s["sentences"] for x in sent.get("cites", [])}
    rule_types = {e["data"]["type"] for e in res.evidence if e["kind"] == "rule" and e["id"] in cited}
    exp = set(item.get("hazards", []))
    claims = [sent for s in res.sections if s["title"] not in GAP_TITLES for sent in s["sentences"]]
    r = {"intent_correct": (res.intent == item["intent"]) if item.get("intent") else None,
         "hazard_recall": len(exp & rule_types) / len(exp) if exp else None,
         "grounded_rate": sum(1 for x in claims if x.get("cites")) / len(claims) if claims else 0.0,
         "tool_calls": len(res.tool_calls), "issues": len(res.issues), "answer": res.answer_markdown}
    r["correct"] = bool(r["intent_correct"] in (None, True) and r["hazard_recall"] in (None, 1.0))
    return r


TASKS: dict[str, tuple[str, Task]] = {   # slot -> (candidate pool, task)
    "planner": ("text", task_planner),
    "sparql": ("text", task_sparql),
    "synthesizer": ("text", task_synthesizer),
    "verifier": ("text", task_synthesizer),
    "scene_graph": ("vision", task_scene_graph),
    "figure_reader": ("vision", task_figure_reader),
    "e2e": ("text", task_e2e),
}


# ------------------------------------------------------------------------------ runner
async def run_sweep(settings: Settings, slot: str, models: list[dict], items: list[dict], repeats: int = 1,
                    out_dir: Path | None = None, toolkit=None) -> dict:
    _, task = TASKS[slot]
    # leave-one-out: a test question is never shown to the planner as its own few-shot example
    settings.raw.setdefault("ffonto_v2", {})["exclude_self"] = True
    out_dir = out_dir or settings.root / settings.get("paths.runs_dir", "runs") / \
        f"{datetime.now():%Y%m%d_%H%M%S}_{slot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model: dict[str, list[dict]] = {}
    skipped: dict[str, str] = {}
    with (out_dir / "items.jsonl").open("w") as f:
        for m in models:
            name = m["model"]
            # e2e swaps the whole text stack; otherwise only the slot under test changes
            slots = ["planner", "sparql", "synthesizer", "verifier", "rag_grader"] if slot == "e2e" else [slot]
            llm = LLMClient(settings, {s: {**m, **({"json": True} if s != "sparql" else {})} for s in slots})
            problem = preflight(llm, slots[0])
            if problem:
                print(f"SKIP {name}: {problem}", flush=True)
                skipped[name] = problem
                continue
            ctx = Ctx(settings, llm, toolkit)
            rows = []
            for rep in range(repeats):
                for i, item in enumerate(items):
                    log = start_call_log()
                    t0 = time.perf_counter()
                    try:
                        r = await task(item, ctx)
                        err = None
                    except Exception as e:  # noqa: BLE001
                        r, err = {"correct": False}, f"{e.__class__.__name__}: {e}"
                    llm_errors = [c.error for c in log if not c.ok and c.error]
                    if llm_errors and not err:
                        err = f"LLM: {llm_errors[0][:300]}"
                    r.update(model=name, item=i, repeat=rep, error=err,
                             latency_s=round(time.perf_counter() - t0, 3),
                             prompt_tokens=sum(c.prompt_tokens or 0 for c in log),
                             completion_tokens=sum(c.completion_tokens or 0 for c in log),
                             llm_calls=len(log))
                    rows.append(r)
                    f.write(json.dumps(r, default=str) + "\n")
            per_model[name] = rows
    if not per_model:
        msg = "no model could be reached:\n" + "\n".join(f"  {n}: {p}" for n, p in skipped.items())
        (out_dir / "summary.md").write_text(f"# Slot: {slot}\n\n{msg}\n")
        raise SystemExit(msg)
    summary = summarize(slot, per_model, items)
    summary["skipped"] = skipped
    write_summary(out_dir, slot, summary)
    return {"out_dir": str(out_dir), "summary": summary}


def preflight(llm: LLMClient, slot: str) -> str | None:
    """One tiny call before a model's sweep: an unreachable server or a missing model tag is reported
    instead of silently scoring the fallback path for every item."""
    try:
        llm.chat_sync(slot, [{"role": "user", "content": 'Reply with the JSON {"ok": true}. Question: ping'}],
                      json_mode=True)
        return None
    except Exception as e:  # noqa: BLE001
        text = str(e)
        resp = getattr(e, "response", None)
        if resp is not None:
            try:
                text = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception:  # noqa: BLE001
                pass
        if "Connection refused" in text or "Max retries" in text:
            text += " (is Ollama running? check OLLAMA_BASE_URL)"
        elif "not found" in text.lower():
            text += " (pull it: ollama pull <tag>)"
        return text[:400]


METRICS = {
    "planner": ["intent_correct", "cq_correct", "route_correct", "valid_json"],
    "sparql": ["exec_correct", "valid", "repaired"],
    "synthesizer": ["grounded_rate", "hazard_recall", "valid_json", "unsupported_numbers", "verifier_problems"],
    "verifier": ["grounded_rate", "hazard_recall", "valid_json"],
    "scene_graph": ["f1", "precision", "recall", "vocab_compliance", "hallucination_rate", "count_accuracy", "valid_json"],
    "figure_reader": ["ape", "trend_correct", "valid_json"],
    "e2e": ["intent_correct", "hazard_recall", "grounded_rate", "tool_calls", "issues"],
}


def summarize(slot: str, per_model: dict[str, list[dict]], items: list[dict]) -> dict:
    out: dict[str, Any] = {"models": {}}
    for name, rows in per_model.items():
        s: dict[str, Any] = {"n": len(rows), "errors": sum(1 for r in rows if r.get("error"))}
        for m in METRICS[slot] + ["correct"]:
            vals = [float(r[m]) for r in rows if r.get(m) is not None]
            mean, lo, hi = bootstrap_ci(vals)
            s[m] = {"mean": mean, "ci95": [lo, hi]}
        if slot == "planner":
            s["intent_macro_f1"] = macro_f1([r["gold_intent"] for r in rows if "gold_intent" in r],
                                            [r["pred_intent"] for r in rows if "pred_intent" in r])
            cq_rows = [r for r in rows if r.get("gold_cq")]
            s["cq_macro_f1"] = macro_f1([r["gold_cq"] for r in cq_rows], [str(r.get("pred_cq")) for r in cq_rows])
        if slot == "figure_reader":
            golds = [items[r["item"]].get("gold_value") for r in rows if "pred_value" in r]
            s["mape"] = mape(golds, [r["pred_value"] for r in rows if "pred_value" in r])
        lat = [r["latency_s"] for r in rows]
        s["latency_p50"] = statistics.median(lat) if lat else None
        s["latency_p95"] = sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None
        s["tokens_per_item"] = statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows) if rows else None
        out["models"][name] = s
    # paired significance vs the best model (on repeat 0)
    best = max(out["models"], key=lambda n: out["models"][n]["correct"]["mean"] or 0)
    out["best"] = best
    base = {r["item"]: bool(r.get("correct")) for r in per_model[best] if r["repeat"] == 0}
    out["mcnemar_vs_best"] = {}
    for name, rows in per_model.items():
        if name == best:
            continue
        other = {r["item"]: bool(r.get("correct")) for r in rows if r["repeat"] == 0}
        keys = sorted(set(base) & set(other))
        out["mcnemar_vs_best"][name] = mcnemar([base[k] for k in keys], [other[k] for k in keys])
    return out


def write_summary(out_dir: Path, slot: str, summary: dict) -> None:
    cols = METRICS[slot] + ["correct"]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n", "errors", *[f"{c}_mean" for c in cols], *[f"{c}_ci95" for c in cols],
                    "latency_p50", "latency_p95", "tokens_per_item"])
        for name, s in summary["models"].items():
            w.writerow([name, s["n"], s["errors"], *[_f(s[c]["mean"]) for c in cols],
                        *[f"[{_f(s[c]['ci95'][0])}, {_f(s[c]['ci95'][1])}]" for c in cols],
                        _f(s["latency_p50"]), _f(s["latency_p95"]), _f(s["tokens_per_item"])])
    lines = [f"# Slot: {slot}", "", "| model | " + " | ".join(cols) + " | p50 s | tokens/item |",
             "|" + "---|" * (len(cols) + 3)]
    for name, s in summary["models"].items():
        cells = [f"{_f(s[c]['mean'])} [{_f(s[c]['ci95'][0])}-{_f(s[c]['ci95'][1])}]" for c in cols]
        lines.append(f"| {name}{' (best)' if name == summary['best'] else ''} | " + " | ".join(cells)
                     + f" | {_f(s['latency_p50'])} | {_f(s['tokens_per_item'])} |")
    broken = {n: s["errors"] for n, s in summary["models"].items() if s["n"] and s["errors"] > 0.5 * s["n"]}
    if broken or summary.get("skipped"):
        lines += ["", "WARNING: results below are not valid model comparisons:", ""]
        lines += [f"- {n}: skipped ({p})" for n, p in summary.get("skipped", {}).items()]
        lines += [f"- {n}: {e} of {summary['models'][n]['n']} items failed (see `error` in items.jsonl)"
                  for n, e in broken.items()]
    if summary["mcnemar_vs_best"]:
        lines += ["", "McNemar (exact) vs best, repeat 0:", ""]
        for name, t in summary["mcnemar_vs_best"].items():
            lines.append(f"- {name}: best-only correct {t['b']}, other-only correct {t['c']}, p = {t['p']:.4f}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _f(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wfmas-eval")
    ap.add_argument("--slot", required=True, choices=sorted(TASKS))
    ap.add_argument("--dataset", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--models", help="comma-separated model tags (provider from --provider)")
    g.add_argument("--pool", choices=["text", "vision"], help="use candidates.<pool> from config/models.yaml")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all",
                    help="golden sets: evaluate only this split")
    a = ap.parse_args(argv)
    s = load_settings()
    if a.models:
        models = [{"provider": a.provider, "model": m.strip()} for m in a.models.split(",")]
    else:
        models = s.models["candidates"][a.pool or TASKS[a.slot][0]]
    items = [json.loads(x) for x in Path(a.dataset).read_text().splitlines() if x.strip()]
    if a.split != "all":
        items = [i for i in items if i.get("split", a.split) == a.split]
    if a.limit:
        items = items[: a.limit]
    res = asyncio.run(run_sweep(s, a.slot, models, items, a.repeats))
    print((Path(res["out_dir"]) / "summary.md").read_text())
    print(f"results in {res['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
