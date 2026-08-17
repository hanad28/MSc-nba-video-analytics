"""
cache_utils.py

Pickle-based caching for expensive pipeline outputs (detections, tracks,
team assignments), avoiding repeated inference during development.
Fingerprint sidecars ({cache_path}.meta.json) record the inputs, model
identity and parameters that produced a cache, so same-length staleness
is detected mechanically instead of by manual deletion.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import pickle


def save_cache(data, cache_path: str) -> None:
    """Save data to a pickle file at cache_path, creating parent directories if needed."""
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except (OSError, pickle.PicklingError) as error:
        raise IOError(f'Cannot write cache to {cache_path}: {error}') from error


def load_cache(cache_path: str):
    """Load and return data from a pickle file at cache_path."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f'Cache file does not exist: {cache_path}')

    try:
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as error:
        # A truncated or otherwise unreadable cache must never be mistaken for
        # a cache miss -- that would silently re-run inference and overwrite
        # evidence. Surface it so the file can be inspected or deleted by hand.
        raise IOError(
            f'Cache at {cache_path} is unreadable ({type(error).__name__}: {error}). '
            f'Delete it to force fresh inference.'
        ) from error


@functools.lru_cache(maxsize=None)
def file_digest(path: str) -> str:
    """Return the sha256 hex digest of a file, memoised so a large checkpoint is hashed once per process."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def data_digest(data: object) -> str:
    """Return the sha256 hex digest of an in-memory object's pickled bytes, for fingerprinting stage inputs."""
    return hashlib.sha256(pickle.dumps(data)).hexdigest()


def _meta_path(cache_path: str) -> str:
    """Return the fingerprint sidecar path for a cache file."""
    return f'{cache_path}.meta.json'


def _short(value: object) -> object:
    """Abbreviate long strings (digests) so mismatch messages stay readable."""
    if isinstance(value, str) and len(value) > 16:
        return f'{value[:8]}…'
    return value


def save_cache_with_meta(data, cache_path: str, fingerprint: dict) -> None:
    """Save data to a pickle cache plus a sibling .meta.json fingerprint sidecar used for staleness detection."""
    save_cache(data, cache_path)
    with open(_meta_path(cache_path), 'w') as f:
        json.dump(fingerprint, f, indent=2, sort_keys=True)


def load_valid_cache(cache_path: str, fingerprint: dict) -> object | None:
    """Return the cached object only when every fingerprint key matches the sidecar; otherwise None, printing why."""
    if not os.path.exists(cache_path):
        return None

    meta_path = _meta_path(cache_path)
    if not os.path.exists(meta_path):
        # Read the pickle before discarding it: every pre-sidecar cache takes this
        # branch on its first run, and returning None without opening the file would
        # bypass load_cache's guard against mistaking a corrupt cache for a miss:
        # the regenerating stage would silently overwrite the evidence.
        load_cache(cache_path)
        print(
            f'[cache] {cache_path} has no fingerprint sidecar — treating the cache '
            f'as unverifiable, regenerating.'
        )
        return None

    try:
        with open(meta_path, 'r') as f:
            stored = json.load(f)
    except json.JSONDecodeError as error:
        # Mirrors load_cache's contract: an unreadable sidecar must not be
        # mistaken for a mismatch, which would silently overwrite evidence.
        raise IOError(
            f'Fingerprint sidecar at {meta_path} is unreadable ({error}). '
            f'Delete it and the cache to force fresh inference.'
        ) from error

    mismatched = [key for key in fingerprint if stored.get(key) != fingerprint[key]]
    if mismatched:
        for key in mismatched:
            print(
                f'[cache] {cache_path} is stale: {key} '
                f'{_short(stored.get(key))} -> {_short(fingerprint[key])} — regenerating.'
            )
        return None

    return load_cache(cache_path)
