"""Slot-based LLM/VLM client over OpenAI-compatible endpoints (Ollama, vLLM, LM Studio...).

Every call is recorded (slot, model, latency, tokens) so the evaluation harness and the
answer trace can report cost per plan.
"""
from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from .config import Settings

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)

# Collected per request (contextvar so parallel requests don't mix).
_calls: contextvars.ContextVar[list | None] = contextvars.ContextVar("llm_calls", default=None)


@dataclass
class LLMCall:
    slot: str
    model: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    ok: bool
    error: str | None = None


@dataclass
class SlotConfig:
    slot: str
    provider: str
    model: str
    base_url: str
    api_key: str = "EMPTY"
    json: bool = False
    vision: bool = False
    temperature: float = 0.0
    seed: int | None = 42
    max_tokens: int = 1536
    timeout_s: float = 180
    extra: dict = field(default_factory=dict)


def start_call_log() -> list:
    log: list = []
    _calls.set(log)
    return log


def _record(call: LLMCall) -> None:
    log = _calls.get()
    if log is not None:
        log.append(call)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply: strips <think>, code fences, leading prose."""
    text = _THINK.sub("", text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON in model output: {text[:200]!r}")


def image_part(path_or_bytes: str | Path | bytes, mime: str = "image/png") -> dict:
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    if not isinstance(path_or_bytes, bytes):
        suffix = Path(path_or_bytes).suffix.lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, mime)
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


class LLMClient:
    """Calls the model configured for a slot. Swap models in config/models.yaml."""

    def __init__(self, settings: Settings, slot_overrides: dict[str, dict] | None = None):
        self.settings = settings
        self.slot_overrides = slot_overrides or {}

    def slot_config(self, slot: str) -> SlotConfig:
        m = self.settings.models
        spec = dict(m["slots"].get(slot) or {})
        spec.update(self.slot_overrides.get(slot, {}))
        if not spec:
            raise KeyError(f"no model configured for slot '{slot}'")
        prov = m["providers"][spec["provider"]]
        d = m.get("defaults", {})
        return SlotConfig(
            slot=slot,
            provider=spec["provider"],
            model=spec["model"],
            base_url=(prov.get("base_url") or "").rstrip("/"),
            api_key=prov.get("api_key") or "EMPTY",
            json=bool(spec.get("json", False)),
            vision=bool(spec.get("vision", False)),
            temperature=spec.get("temperature", d.get("temperature", 0)),
            seed=spec.get("seed", d.get("seed")),
            max_tokens=spec.get("max_tokens", d.get("max_tokens", 1536)),
            timeout_s=spec.get("timeout_s", d.get("timeout_s", 180)),
            extra=spec.get("extra", {}),
        )

    # -- sync core -------------------------------------------------------------------
    def chat_sync(self, slot: str, messages: list[dict], json_mode: bool | None = None) -> str:
        cfg = self.slot_config(slot)
        body: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            **cfg.extra,
        }
        if cfg.seed is not None:
            body["seed"] = cfg.seed
        if json_mode if json_mode is not None else cfg.json:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{cfg.base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                timeout=cfg.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            _record(LLMCall(slot, cfg.model, time.perf_counter() - t0,
                            usage.get("prompt_tokens"), usage.get("completion_tokens"), True))
            return _THINK.sub("", text).strip()
        except Exception as e:  # noqa: BLE001
            _record(LLMCall(slot, cfg.model, time.perf_counter() - t0, None, None, False, str(e)))
            raise

    # -- async helpers ---------------------------------------------------------------
    async def chat(self, slot: str, system: str, user: str | list, json_mode: bool | None = None) -> str:
        content = user if isinstance(user, list) else [{"type": "text", "text": user}]
        if all(p.get("type") == "text" for p in content):
            content = "\n".join(p["text"] for p in content)  # plain string for text-only models
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        return await asyncio.to_thread(self.chat_sync, slot, messages, json_mode)

    async def chat_json(self, slot: str, system: str, user: str | list, retries: int = 1) -> Any:
        last_err: Exception | None = None
        prompt = user
        for _ in range(retries + 1):
            text = await self.chat(slot, system, prompt, json_mode=True)
            try:
                return extract_json(text)
            except ValueError as e:
                last_err = e
                note = "\n\nYour previous reply was not valid JSON. Reply with JSON only."
                prompt = (prompt + note) if isinstance(prompt, str) else [*prompt, {"type": "text", "text": note}]
        raise last_err  # type: ignore[misc]


class ScriptedLLM(LLMClient):
    """Test double: returns canned replies per slot (callable or list)."""

    def __init__(self, settings: Settings, script: dict[str, Callable[[list[dict]], str] | list[str]]):
        super().__init__(settings)
        self.script = script
        self.seen: list[tuple[str, list[dict]]] = []

    def chat_sync(self, slot: str, messages: list[dict], json_mode: bool | None = None) -> str:
        self.seen.append((slot, messages))
        s = self.script.get(slot)
        if s is None:
            raise KeyError(f"ScriptedLLM has no reply for slot {slot}")
        out = s(messages) if callable(s) else s.pop(0)
        _record(LLMCall(slot, "scripted", 0.0, 0, 0, True))
        return out
