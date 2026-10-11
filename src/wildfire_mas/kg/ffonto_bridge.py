"""Bridge to the ffonto-v2 project's `qa` package: the MAS's GraphDB agent backend.

ffonto-v2 already has a tested KG question-answering layer; the MAS reuses it instead of
re-implementing it:
  qa.cqs.run_cq          CQ01-CQ26 from queries/*.rq, narrowed by record / event (VALUES) and
                         site / date (row filter); the record's event is found automatically
  qa.pipeline.text_to_sparql   text-to-SPARQL "plus" (context binding, schema linking, template
                         adaptation, execution feedback; each switchable with T2S_* env vars)
  qa.rag.retrieve_mm     multimodal document retrieval (bge-m3: text, tables, figures)
  qa.examples            the labelled ForestFire-VQA questions (docs/FFVQA_CQ_mapping.xlsx)

Set `ffonto_v2.path` (env FFONTO_V2_PATH) to the project folder. Without it the MAS falls back
to its built-in CQ catalog, SPARQL guard and BM25 index.
"""
from __future__ import annotations

import os
import sys
from functools import cached_property
from pathlib import Path
from typing import Any

# CQ label -> planner intent (OOS = out of scope of the KG = general knowledge)
HAZARD_CQS = {"CQ17", "CQ24", "CQ25"}


def intent_for_label(label: str | None) -> str:
    if not label or label.upper() in {"OOS", "RAG", "NONE"}:
        return "explanation"
    if label in HAZARD_CQS:
        return "hazard_advice"
    return "data_lookup"


class BridgeUnavailable(RuntimeError):
    pass


class FfontoBridge:
    def __init__(self, settings, sparql_model: str | None = None):
        self.s = settings
        raw = settings.get("ffonto_v2.path") or ""
        self.root = Path(raw).expanduser().resolve() if raw else None
        self.sparql_model = sparql_model
        self.error: str | None = None

    # ------------------------------------------------------------------ loading
    @property
    def configured(self) -> bool:
        return bool(self.root and (self.root / "qa" / "cqs.py").exists())

    @cached_property
    def qa(self) -> dict[str, Any]:
        if not self.configured:
            raise BridgeUnavailable(f"ffonto_v2.path not set or no qa package at {self.root}")
        g = self.s.get("graphdb") or {}
        # the qa package reads its settings from the environment at import time:
        # point it at the same GraphDB and Ollama as the MAS
        os.environ.setdefault("GRAPHDB_URL", g.get("url") or "http://localhost:7200")
        os.environ.setdefault("GRAPHDB_REPO", g.get("repository") or "ffonto")
        base = (self.s.models.get("providers", {}).get("ollama", {}) or {}).get("base_url") or ""
        if base and "OLLAMA_URL" not in os.environ:
            os.environ["OLLAMA_URL"] = base.rstrip("/").removesuffix("/v1")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        try:
            import qa.config as config
            import qa.cqs as cqs
            import qa.kg as kg
        except Exception as e:  # noqa: BLE001
            self.error = f"could not import ffonto-v2 qa package: {e}"
            raise BridgeUnavailable(self.error) from e
        if self.sparql_model:
            config.SPARQL_MODEL = self.sparql_model     # model sweeps reach text-to-SPARQL too
        return {"config": config, "cqs": cqs, "kg": kg}

    def available(self) -> bool:
        try:
            self.qa
            return True
        except BridgeUnavailable:
            return False

    def _mod(self, name: str):
        self.qa
        import importlib

        return importlib.import_module(f"qa.{name}")

    # ------------------------------------------------------------------ CQs
    def registry(self) -> dict[str, dict]:
        """{cq_id: {title, examples, columns, focus}} for CQ01-CQ26 (empty if unavailable)."""
        if not self.available():
            return {}
        return {cid: {"title": c.title, "examples": list(c.examples), "columns": list(c.columns),
                      "focus": sorted(c.focus)}
                for cid, c in self.qa["cqs"].REGISTRY.items()}

    def menu(self) -> str:
        lines = []
        for cid, c in self.registry().items():
            ex = "; ".join(c["examples"][:2])
            narrow = f" [narrow by: {', '.join(c['focus'])}]" if c["focus"] else ""
            lines.append(f"{cid}: {c['title']}{narrow} (e.g. {ex})")
        return "\n".join(lines)

    def run_cq(self, cq_id: str, record: str | None = None, event: str | None = None,
               site: str | None = None, date: str | None = None) -> dict:
        q = self.qa
        try:
            res = q["cqs"].run_cq(cq_id, record=record or None, event=event or None,
                                  site=site or None, date=date or None)
        except q["kg"].KGError as e:
            return {"cq": cq_id, "ok": False, "errors": [str(e)], "rows": [], "sparql": None}
        return {"cq": cq_id, "ok": True, "title": res["title"], "sparql": res["query"], "columns": res["columns"],
                "rows": res["rows"], "filters": res["filters"]}

    def text2sparql(self, question: str, record: str | None = None, hints: list[str] | None = None,
                    date: str | None = None, event: str | None = None, mode: str | None = None) -> dict:
        q = self.qa
        pipeline = self._mod("pipeline")
        llm = self._mod("llm")
        try:
            sparql, cols, rows, applied = pipeline.text_to_sparql(
                question, record or None, [h for h in (hints or []) if h in q["cqs"].REGISTRY],
                date=date or None, event=event or None, mode=mode)
        except (q["kg"].KGError, llm.LLMError) as e:
            return {"ok": False, "errors": [str(e)], "rows": [], "sparql": None}
        return {"ok": True, "sparql": sparql, "columns": cols, "rows": rows, "filters": applied,
                "mode": mode or q["config"].T2S_MODE}

    # ------------------------------------------------------------------ RAG
    def rag_index_exists(self) -> bool:
        return self.configured and (self.root / "qa" / "index" / "rag_chunks.jsonl").exists()

    def rag_search(self, query: str, k_text: int = 5, k_fig: int = 3) -> list[dict]:
        rag = self._mod("rag")
        passages, figures = rag.retrieve_mm(query, k_text=k_text, k_fig=k_fig)
        out = []
        for p in passages:
            out.append({"id": p.get("source"), "kind": p.get("type", "text"), "text": p.get("text", ""),
                        "source": p.get("doc") or p.get("source"), "page": p.get("page"),
                        "score": round(float(p.get("score", 0)), 3)})
        for f in figures:
            out.append({"id": f.get("source"), "kind": "figure", "text": f.get("text", ""),
                        "source": f.get("doc") or f.get("source"), "page": f.get("page"),
                        "image_path": f.get("image"), "caption": f.get("caption"),
                        "concepts": [c.get("label") for c in f.get("concepts", [])],
                        "score": round(float(f.get("score", 0)), 3)})
        return out

    # ------------------------------------------------------------------ labelled examples
    @cached_property
    def _example_index(self):
        try:
            return self._mod("examples").ExampleIndex.load()   # needs `python -m qa.build_index` once
        except Exception:  # noqa: BLE001
            return None

    def nearest_examples(self, question: str, k: int = 8, exclude_self: bool = False) -> list[dict] | None:
        """Embedding-nearest labelled questions as {question, label, intent}; None if unavailable."""
        idx = self._example_index
        if idx is None:
            return None
        try:
            hits = idx.nearest(question, k=k, exclude_text=question if exclude_self else None)
        except Exception:  # noqa: BLE001 - Ollama embeddings unavailable
            return None
        return [{"question": h["question"], "label": h["cq"], "intent": intent_for_label(h["cq"]),
                 "score": round(float(h.get("score", 0)), 3)} for h in hits]

    def all_examples(self) -> list[dict]:
        """Every labelled question from the mapping workbook (for the golden set and fallbacks)."""
        ex = self._mod("examples")
        return [{"question": e["question"], "label": e["cq"], "intent": intent_for_label(e["cq"]),
                 "modality": e.get("modality"), "answer": e.get("answer"), "source": e.get("source")}
                for e in ex.load_from_mapping()]
