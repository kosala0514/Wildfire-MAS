"""RAG backends. Plug in your existing bge-m3 index via `rag.backend: custom`.

Chunk format (JSONL), same three content kinds as your index:
  {"id": "fwi_manual_p12_c3", "kind": "text|table|figure", "text": "...", "source": "FWI manual",
   "page": 12, "image_path": "figs/fwi_p12.png", "concepts": ["FWI", "FireSpread"]}
For figures, `text` is the Qwen2.5-VL description (with extracted chart values).
"""
from __future__ import annotations

import importlib
import json
import math
import re
from collections import Counter
from typing import Protocol

_TOK = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of and or to in on for is are was were be by with at from as that this it what how "
            "which who when where why do does did can should would could".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in _STOP]


class RagBackend(Protocol):
    def search(self, query: str, k: int = 6) -> list[dict]: ...


class BM25Backend:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.docs = [tokenize(c.get("text", "") + " " + " ".join(c.get("concepts", []))) for c in chunks]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    @classmethod
    def from_jsonl(cls, path) -> "BM25Backend":
        chunks = []
        if path and path.exists():
            chunks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return cls(chunks)

    def search(self, query: str, k: int = 6) -> list[dict]:
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [{**self.chunks[i], "score": round(s, 3)} for s, i in scored[:k]]


class FfontoRag:
    """The ffonto-v2 multimodal index (bge-m3 over text chunks, tables and VLM-described figures)."""

    def __init__(self, bridge, k_fig: int = 3):
        self.bridge = bridge
        self.k_fig = k_fig

    def search(self, query: str, k: int = 6) -> list[dict]:
        return self.bridge.rag_search(query, k_text=k, k_fig=self.k_fig)


def make_rag(settings, bridge=None) -> RagBackend:
    c = settings.get("rag")
    if bridge is not None and settings.get("ffonto_v2.use_rag", True) and bridge.rag_index_exists():
        return FfontoRag(bridge)
    if c.get("backend") == "custom" and c.get("custom_class"):
        mod, cls = c["custom_class"].split(":")
        return getattr(importlib.import_module(mod), cls)(settings)
    return BM25Backend.from_jsonl(settings.path(c.get("chunks_file")))
