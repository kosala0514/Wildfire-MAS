"""Synthesizer (Contextual Response Mapping) and verifier.

CRM: the planner's intent selects a response schema (section list); evidence is mapped
to those sections and packed under a size budget, with rule-derived hazards always kept
(safety override). The LLM writes sentence-level JSON where every sentence carries
citation IDs. If the LLM fails, an extractive answer is assembled from the evidence
summaries instead, so the system degrades to "show the evidence", never to invention.

Verifier: deterministic checks (every sentence cited, all cites exist, every
safety-critical item cited) plus an optional LLM entailment check. Failures go back to
the synthesizer once with feedback; anything still failing is removed or appended
deterministically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm import LLMClient
from .prompts import RESPONSE_SCHEMAS, SYNTH_SYSTEM, VERIFIER_SYSTEM, style_for
from .state import Evidence

GAP_TITLES = {"Data gaps"}

SECTION_HINT = {  # CRM: which section each evidence kind usually feeds
    "rule": "Hazards",
    "live": "Situation",
    "kg": "Supporting evidence",
    "doc": "Recommended actions",
    "figure": "Key figures",
    "ontology": "Recommended actions",
}

KIND_PRIORITY = {"rule": 0, "live": 1, "kg": 2, "ontology": 3, "figure": 4, "doc": 5}


@dataclass
class Answer:
    sections: list[dict]
    source: str                      # "llm" | "extractive"
    issues: list[str] = field(default_factory=list)

    def sentences(self):
        for s in self.sections:
            for i, sent in enumerate(s.get("sentences", [])):
                yield s["title"], i, sent

    def markdown(self) -> str:
        out = []
        for s in self.sections:
            sents = s.get("sentences", [])
            if not sents:
                continue
            out.append(f"**{s['title']}**")
            body = " ".join(f"{x['text'].strip()}" + (f" [{', '.join(x['cites'])}]" if x.get("cites") else "")
                            for x in sents)
            out.append(body)
            out.append("")
        return "\n".join(out).strip()


def pack_evidence(evidence: list[Evidence], budget_chars: int = 12000,
                  sections: list[str] | None = None) -> tuple[str, list[Evidence]]:
    """CRM packing: safety-critical first, then by kind priority, within a size budget."""
    ordered = sorted(evidence, key=lambda e: (not e.safety_critical, KIND_PRIORITY.get(e.kind, 9)))
    kept, used = [], 0
    for e in ordered:
        size = len(e.summary) + 60
        if e.safety_critical or used + size <= budget_chars:
            kept.append(e)
            used += size
    kept.sort(key=lambda e: evidence.index(e))
    blocks = []
    for e in kept:
        flag = " SAFETY-CRITICAL" if e.safety_critical else ""
        hint = section_for(e, sections) if sections else SECTION_HINT.get(e.kind, "-")
        blocks.append(f"[{e.id}]{flag} ({e.kind}; source: {e.source}; suggested section: {hint})\n{e.summary}")
    return "\n\n".join(blocks), kept


class Synthesizer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def write(self, question: str, intent: str, urgency: str, role: str, evidence: list[Evidence],
                    feedback: str | None = None) -> Answer:
        sections = RESPONSE_SCHEMAS.get(intent, RESPONSE_SCHEMAS["data_lookup"])
        packed, kept = pack_evidence(evidence, sections=sections)
        system = SYNTH_SYSTEM.format(sections=", ".join(sections), style=style_for(urgency, role))
        user = f"QUESTION: {question}\n\nEVIDENCE:\n{packed}"
        if feedback:
            user += f"\n\nYOUR PREVIOUS ANSWER HAD PROBLEMS, FIX THEM:\n{feedback}"
        try:
            out = await self.llm.chat_json("synthesizer", system, user)
            secs = [s for s in out.get("sections", []) if isinstance(s, dict) and s.get("title")]
            for s in secs:
                s["sentences"] = [x for x in s.get("sentences", []) if isinstance(x, dict) and x.get("text")]
                for x in s["sentences"]:
                    x["cites"] = [str(c).strip("[] ") for c in x.get("cites", []) or []]
            if not secs:
                raise ValueError("empty answer")
            return Answer(secs, "llm")
        except Exception as e:  # noqa: BLE001
            ans = extractive_answer(intent, kept or evidence)
            ans.issues.append(f"LLM synthesis failed ({e.__class__.__name__}); extractive answer used")
            return ans


def section_for(e: Evidence, sections: list[str]) -> str:
    """CRM mapping of one evidence item onto the intent's response schema."""
    text = e.summary.lower()
    wants: list[str]
    if "no evidence in the knowledge graph" in text or "derived no hazard" in text:
        wants = ["Data gaps"]
    elif e.kind == "rule":
        wants = ["Hazards"]
    elif e.kind == "live" and "fire weather index" in text:
        wants = ["Fire danger", "Current value", "Trend", "Context"]
    elif e.kind == "live":
        wants = ["Situation", "Current value", "Context", "Supporting evidence"]
    elif e.kind == "kg" and ("actions" in text or "mitigat" in text):
        wants = ["Recommended actions", "Hazards"]
    elif e.kind == "kg" and "fwi" in text:
        wants = ["Trend", "Fire danger", "Context"]
    elif e.kind == "kg":
        wants = ["Answer", "Situation", "Supporting evidence", "Context"]
    elif e.kind == "figure":
        wants = ["Key figures", "What it means", "Supporting evidence"]
    else:  # doc, ontology
        wants = ["Recommended actions", "Explanation", "What it means", "Related hazards and actions", "Context"]
    return next((w for w in wants if w in sections), sections[0])


def _flatten(summary: str, limit: int = 600) -> str:
    lines = [ln.strip("- ").strip() for ln in summary.split("\n") if ln.strip()]
    text = "; ".join(lines)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def extractive_answer(intent: str, evidence: list[Evidence]) -> Answer:
    """No-LLM answer: each evidence item becomes one cited sentence in its CRM section."""
    sections = RESPONSE_SCHEMAS.get(intent, RESPONSE_SCHEMAS["data_lookup"])
    buckets: dict[str, list[dict]] = {s: [] for s in sections}
    for e in evidence:
        buckets[section_for(e, sections)].append({"text": _flatten(e.summary), "cites": [e.id]})
    return Answer([{"title": t, "sentences": s} for t, s in buckets.items() if s], "extractive")


class Verifier:
    def __init__(self, llm: LLMClient | None = None, use_llm: bool = False):
        self.llm = llm
        self.use_llm = use_llm and llm is not None

    async def check(self, ans: Answer, evidence: list[Evidence]) -> list[str]:
        ids = {e.id for e in evidence}
        problems: list[str] = []
        cited: set[str] = set()
        flat = list(ans.sentences())
        for title, i, s in flat:
            cites = s.get("cites", [])
            cited |= set(cites)
            if title in GAP_TITLES:
                continue
            if not cites:
                problems.append(f"'{title}' sentence {i + 1} has no citation: \"{s['text'][:80]}\"")
            bad = [c for c in cites if c not in ids]
            if bad:
                problems.append(f"'{title}' sentence {i + 1} cites unknown evidence {bad}")
        for e in evidence:
            if e.safety_critical and e.id not in cited:
                problems.append(f"safety-critical evidence {e.id} is not mentioned: {e.summary[:100]}")
        if self.use_llm and not problems:
            by_id = {e.id: e for e in evidence}
            listing = "\n".join(
                f"{n}. {s['text']}\n   cited: " + " || ".join(by_id[c].summary[:400] for c in s.get("cites", []) if c in by_id)
                for n, (_, _, s) in enumerate(flat) if s.get("cites"))
            try:
                out = await self.llm.chat_json("verifier", VERIFIER_SYSTEM, listing)
                for u in out.get("unsupported", []):
                    idx = int(u.get("index", -1))
                    if 0 <= idx < len(flat):
                        problems.append(f"sentence \"{flat[idx][2]['text'][:80]}\" not supported: {u.get('reason')}")
            except Exception as e:  # noqa: BLE001
                ans.issues.append(f"LLM verifier skipped ({e.__class__.__name__})")
        return problems

    @staticmethod
    def enforce(ans: Answer, evidence: list[Evidence]) -> Answer:
        """Last line of defence after the rewrite budget is spent."""
        ids = {e.id for e in evidence}
        removed = 0
        for s in ans.sections:
            keep = []
            for x in s.get("sentences", []):
                x["cites"] = [c for c in x.get("cites", []) if c in ids]
                if x["cites"] or s["title"] in GAP_TITLES:
                    keep.append(x)
                else:
                    removed += 1
            s["sentences"] = keep
        if removed:
            ans.issues.append(f"removed {removed} uncited sentence(s)")
        cited = {c for _, _, x in ans.sentences() for c in x.get("cites", [])}
        missing = [e for e in evidence if e.safety_critical and e.id not in cited]
        if missing:
            hz = next((s for s in ans.sections if s["title"] == "Hazards"), None)
            if hz is None:
                hz = {"title": "Hazards", "sentences": []}
                ans.sections.insert(0, hz)
            for e in missing:
                hz["sentences"].append({"text": e.summary.split("\n")[0], "cites": [e.id]})
            ans.issues.append(f"appended {len(missing)} safety-critical item(s) the writer omitted")
        return ans
