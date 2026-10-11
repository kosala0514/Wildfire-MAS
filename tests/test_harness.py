"""Model sweep harness with fake models (one good, one bad) and the metric helpers."""
import asyncio
import json

from wildfire_mas.evaluation import harness
from wildfire_mas.evaluation.metrics import bootstrap_ci, macro_f1, mcnemar, numbers_unsupported, rows_match
from wildfire_mas.llm import LLMCall, LLMClient, _record


GOLD: dict[str, str] = {}


class FakeModels(LLMClient):
    """'good' answers correctly, 'bad' always says explanation; synth behaviour likewise."""

    def chat_sync(self, slot, messages, json_mode=None):
        model = self.slot_config(slot).model
        _record(LLMCall(slot, model, 0.01, 100, 20, True))
        text = messages[-1]["content"]
        if slot == "planner":
            q = text.rsplit("Question:", 1)[1].strip()
            intent = GOLD.get(q, "explanation") if model == "good" else "explanation"
            return json.dumps({"intent": intent, "urgency": "normal"})
        if slot == "synthesizer":
            if model == "good":
                return json.dumps({"sections": [{"title": "Hazards", "sentences": [
                    {"text": "Nearby active fire 3.4 km away.", "cites": ["R09-1"]},
                    {"text": "Danger raised to very high.", "cites": ["R11-1"]}]}]})
            return json.dumps({"sections": [{"title": "Hazards", "sentences": [
                {"text": "Fire 7 km away.", "cites": ["R09-1"]}, {"text": "Probably fine.", "cites": []}]}]})
        raise KeyError(slot)


def test_planner_sweep(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "LLMClient", FakeModels)
    items = [json.loads(x) for x in (settings.root / "eval/datasets/intent.jsonl").read_text().splitlines() if x]
    GOLD.update({i["question"]: i["intent"] for i in items})
    res = asyncio.run(harness.run_sweep(settings, "planner", [{"provider": "ollama", "model": "good"},
                                                              {"provider": "ollama", "model": "bad"}],
                                        items, repeats=1, out_dir=tmp_path))
    s = res["summary"]
    assert s["best"] == "good"
    assert s["models"]["good"]["intent_correct"]["mean"] == 1.0 > s["models"]["bad"]["intent_correct"]["mean"]
    assert s["models"]["good"]["intent_macro_f1"] == 1.0
    assert s["mcnemar_vs_best"]["bad"]["b"] >= 6   # items with a gold CQ count only if the CQ is right
    assert (tmp_path / "summary.md").read_text().count("|") > 10 and (tmp_path / "items.jsonl").exists()


def test_synth_sweep_measures_grounding_and_hazard_recall(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "LLMClient", FakeModels)
    items = [json.loads(x) for x in (settings.root / "eval/datasets/synth.jsonl").read_text().splitlines() if x][:1]
    res = asyncio.run(harness.run_sweep(settings, "synthesizer", [{"provider": "ollama", "model": "good"},
                                                                  {"provider": "ollama", "model": "bad"}],
                                        items, out_dir=tmp_path))
    g, b = res["summary"]["models"]["good"], res["summary"]["models"]["bad"]
    assert g["grounded_rate"]["mean"] == 1.0 and g["hazard_recall"]["mean"] == 1.0
    assert b["grounded_rate"]["mean"] == 0.5 and b["hazard_recall"]["mean"] == 0.5
    assert b["unsupported_numbers"]["mean"] == 1.0  # "7 km" is not in the evidence
    assert g["tokens_per_item"] == 120


def test_metric_helpers():
    assert macro_f1(["a", "b", "a"], ["a", "b", "b"]) < 1.0
    assert mcnemar([True] * 10, [False] * 10)["p"] < 0.01
    assert mcnemar([True, False], [True, False])["p"] == 1.0
    mean, lo, hi = bootstrap_ci([1, 0, 1, 1])
    assert lo <= mean <= hi
    assert rows_match([{"x": "a", "n": 1}], [{"y": "a", "m": 1.0}])
    assert numbers_unsupported("FWI 31.0 at 3.4 km", "FWI 31 ... closest 3.4 km") == []
    assert numbers_unsupported("FWI 35", "FWI 31") == ["35"]


class DownModels(FakeModels):
    """'down' behaves like an unreachable Ollama / a tag that was never pulled."""

    def chat_sync(self, slot, messages, json_mode=None):
        if self.slot_config(slot).model == "down":
            raise ConnectionError("Connection refused")
        return super().chat_sync(slot, messages, json_mode)


def test_unreachable_model_is_skipped_not_scored(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "LLMClient", DownModels)
    items = [json.loads(x) for x in (settings.root / "eval/datasets/intent.jsonl").read_text().splitlines() if x]
    GOLD.update({i["question"]: i["intent"] for i in items})
    res = asyncio.run(harness.run_sweep(settings, "planner", [{"provider": "ollama", "model": "good"},
                                                              {"provider": "ollama", "model": "down"}],
                                        items, out_dir=tmp_path))
    assert list(res["summary"]["models"]) == ["good"] and "down" in res["summary"]["skipped"]
    assert "WARNING" in (tmp_path / "summary.md").read_text()
    import pytest
    with pytest.raises(SystemExit):
        asyncio.run(harness.run_sweep(settings, "planner", [{"provider": "ollama", "model": "down"}],
                                      items, out_dir=tmp_path / "b"))
