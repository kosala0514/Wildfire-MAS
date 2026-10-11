"""Settings loader: YAML files with ${ENV_VAR} substitution."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _load_dotenv(root: Path) -> None:
    env_file = root / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    except ImportError:  # minimal fallback parser
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _subst(value: Any) -> Any:
    if isinstance(value, str):
        m = _ENV.fullmatch(value.strip())
        if m:
            return os.environ.get(m.group(1)) or m.group(2) or None
        return _ENV.sub(lambda mm: os.environ.get(mm.group(1)) or mm.group(2) or "", value)
    if isinstance(value, dict):
        return {k: _subst(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_subst(v) for v in value]
    return value


def project_root() -> Path:
    env = os.environ.get("WFMAS_HOME")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "config" / "settings.yaml").exists():
            return p
    return Path(__file__).resolve().parents[2]


class Settings:
    """Thin wrapper over the merged YAML dicts."""

    def __init__(self, raw: dict, models: dict, root: Path):
        self.raw = raw
        self.models = models
        self.root = root

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def path(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    # convenience
    @property
    def ns(self) -> dict[str, str]:
        o = self.raw["ontology"]
        return {"ffo": o["base"], "ffv": o["vocab"], "data": o["data"]}

    def graph(self, name: str) -> str:
        return self.raw["ontology"]["graphs"][name]


def load_settings(root: Path | None = None, overrides: dict | None = None) -> Settings:
    root = root or project_root()
    _load_dotenv(root)
    raw = _subst(yaml.safe_load((root / "config" / "settings.yaml").read_text()))
    models = _subst(yaml.safe_load((root / "config" / "models.yaml").read_text()))
    if overrides:
        _deep_update(raw, overrides)
    return Settings(raw, models, root)


def _deep_update(base: dict, upd: dict) -> None:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
