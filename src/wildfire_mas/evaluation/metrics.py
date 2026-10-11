"""Metrics and statistics for the model comparison."""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Iterable, Sequence

_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?)?|\b\d{1,2}:\d{2}(?::\d{2})?\b")


def accuracy(gold: Sequence, pred: Sequence) -> float:
    return sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold))


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / max(1, len(f1s))


def set_prf(gold: Iterable[str], pred: Iterable[str]) -> tuple[float, float, float]:
    g, p = set(gold), set(pred)
    tp = len(g & p)
    prec = tp / len(p) if p else (1.0 if not g else 0.0)
    rec = tp / len(g) if g else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def rows_match(gold: list[dict], pred: list[dict]) -> bool:
    """Execution accuracy: same multiset of row value-sets (column names may differ)."""
    norm = lambda rows: Counter(frozenset(_norm(v) for v in r.values()) for r in rows)  # noqa: E731
    return norm(gold) == norm(pred)


def _norm(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    s = str(v)
    try:
        return f"{float(s):.4g}"
    except ValueError:
        return s


def numbers_unsupported(text: str, evidence_text: str) -> list[str]:
    """Numbers stated in a sentence but absent from its cited evidence (hallucination proxy)."""
    ev_clean = _DATETIME.sub(" ", evidence_text)
    ev = {_norm(n) for n in _NUM.findall(ev_clean)}
    out = []
    for n in _NUM.findall(_DATETIME.sub(" ", text)):
        if _norm(n) not in ev:
            out.append(n)
    return out


def mape(gold: Sequence[float], pred: Sequence[float | None]) -> float:
    errs = [abs(p - g) / abs(g) for g, p in zip(gold, pred) if p is not None and g != 0]
    return 100 * sum(errs) / len(errs) if errs else float("nan")


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return sum(vals) / len(vals), means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test on paired right/wrong outcomes."""
    b = sum(x and not y for x, y in zip(a_correct, b_correct))  # A right, B wrong
    c = sum(y and not x for x, y in zip(a_correct, b_correct))  # B right, A wrong
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"b": b, "c": c, "p": p}
