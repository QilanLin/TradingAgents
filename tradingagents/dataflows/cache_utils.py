import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .config import get_config


def _get_cache_root() -> Path:
    config = get_config()
    cache_root = Path(config["data_cache_dir"]) / "vendor_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def get_cache_ttl_seconds() -> int:
    config = get_config()
    return int(config.get("data_cache_ttl_seconds", 24 * 60 * 60))


def _cache_key_digest(key_payload: dict[str, Any]) -> str:
    encoded = json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(namespace: str, key_payload: dict[str, Any]) -> Path:
    namespace_dir = _get_cache_root() / namespace
    namespace_dir.mkdir(parents=True, exist_ok=True)
    return namespace_dir / f"{_cache_key_digest(key_payload)}.json"


def _load_entry(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cached_text(
    namespace: str,
    key_payload: dict[str, Any],
    *,
    allow_stale: bool = False,
    ttl_seconds: int | None = None,
) -> str | None:
    path = _cache_path(namespace, key_payload)
    entry = _load_entry(path)
    if not entry:
        return None

    effective_ttl = get_cache_ttl_seconds() if ttl_seconds is None else ttl_seconds
    age_seconds = time.time() - float(entry.get("stored_at", 0.0))

    if age_seconds > effective_ttl and not allow_stale:
        return None

    return entry.get("payload")


def save_cached_text(namespace: str, key_payload: dict[str, Any], payload: str) -> str:
    path = _cache_path(namespace, key_payload)
    tmp_path = path.with_suffix(".tmp")
    entry = {
        "stored_at": time.time(),
        "payload": payload,
    }
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, ensure_ascii=False)
    os.replace(tmp_path, path)
    return str(path)


def get_or_fetch_cached_text(
    namespace: str,
    key_payload: dict[str, Any],
    fetch_fn: Callable[[], str],
    *,
    ttl_seconds: int | None = None,
    fallback_exceptions: Iterable[type[BaseException]] = (Exception,),
) -> str:
    cached = load_cached_text(
        namespace,
        key_payload,
        allow_stale=False,
        ttl_seconds=ttl_seconds,
    )
    if cached is not None:
        return cached

    stale = load_cached_text(
        namespace,
        key_payload,
        allow_stale=True,
        ttl_seconds=ttl_seconds,
    )

    try:
        payload = fetch_fn()
    except tuple(fallback_exceptions):
        if stale is not None:
            return stale
        raise

    save_cached_text(namespace, key_payload, payload)
    return payload
