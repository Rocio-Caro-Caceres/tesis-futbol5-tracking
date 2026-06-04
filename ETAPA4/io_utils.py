"""
io_utils.py
===========
Helpers de I/O compartidos por el pipeline de Etapa 4. Mantener el resto del
paquete libre de dependencias a la API concreta de pandas / parquet / consola.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# ---------------------------------------------------------------------------
# Logging ligero (no usamos logging std para no contaminar stderr en tests)
# ---------------------------------------------------------------------------
def info(msg: str) -> None:
    print(f"[etapa4] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[etapa4][WARN] {msg}", file=sys.stderr, flush=True)


@contextmanager
def stage(name: str):
    info(f">> {name}")
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    info(f"<< {name} ({dt:.1f}s)")


# ---------------------------------------------------------------------------
# Paths y Parquet
# ---------------------------------------------------------------------------
def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def read_parquet(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# JSON safe (NaN/Inf -> null) para dumps de metadata
# ---------------------------------------------------------------------------
def _sanitize(o: Any) -> Any:
    if isinstance(o, float):
        if o != o or o in (float("inf"), float("-inf")):
            return None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(obj), f, ensure_ascii=False, indent=2)


def read_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Numericas
# ---------------------------------------------------------------------------
def coerce_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    """Coerciona a numerico; los que no se pueden pasan a NaN (no al default)."""
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Hash util
# ---------------------------------------------------------------------------
def file_sha256(path: Path | str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
