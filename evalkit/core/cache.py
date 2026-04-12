"""
evalkit.core.cache — Deterministic disk cache for LLM responses.

Avoids redundant (and costly) API calls by hashing the request
fingerprint ``sha256(model + prompt + temperature)`` and persisting
the raw response as a JSON file under ``~/.evalkit/cache/``.

Public API
----------
* ``get(key)``  – return the cached dict or ``None``.
* ``set(key, response)`` – write a dict to the cache.
* ``make_key(model, prompt, temperature)`` – build the hash key.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from evalkit.config import CACHE_DIR


def _cache_path(key: str) -> Path:
    """Return the filesystem path for a given cache key."""
    return CACHE_DIR / f"{key}.json"


def make_key(model: str, prompt: str, temperature: float = 0.0) -> str:
    """
    Derive a deterministic cache key from the request parameters.

    Parameters
    ----------
    model : str
        Model identifier, e.g. ``"gpt-4o-mini"``.
    prompt : str
        The user prompt sent to the model.
    temperature : float
        Sampling temperature.

    Returns
    -------
    str
        A hex-encoded SHA-256 digest.
    """
    raw = f"{model}|{prompt}|{temperature}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(key: str) -> dict[str, Any] | None:
    """
    Retrieve a cached response.

    Parameters
    ----------
    key : str
        Cache key produced by :func:`make_key`.

    Returns
    -------
    dict | None
        The parsed JSON response, or ``None`` on cache miss.
    """
    path = _cache_path(key)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def set(key: str, response: dict[str, Any]) -> None:
    """
    Persist a response to the disk cache.

    Parameters
    ----------
    key : str
        Cache key produced by :func:`make_key`.
    response : dict
        Serialisable response payload to store.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(response, fh, ensure_ascii=False, indent=2)
