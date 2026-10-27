"""Tiny typed config layer over YAML.

We deliberately avoid heavyweight config frameworks (Hydra, OmegaConf) so the
repo stays trivially runnable on a fresh Colab kernel. A ``Config`` is a plain
dataclass-like dotted-access dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """A dict that also supports attribute access and nested dotted access."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Read ``a.b.c`` from a nested config, returning ``default`` if absent."""
        node: Any = self
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> Config:
    """Load a YAML config, applying an optional ``defaults: <file>`` parent.

    A config may declare ``defaults: default.yaml`` (relative to its own
    directory) to inherit shared settings, keeping per-experiment YAML short.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    parent_name = raw.pop("defaults", None)
    if parent_name:
        parent = load_config(path.parent / parent_name)
        raw = _merge(parent, raw)
    return Config(raw)
