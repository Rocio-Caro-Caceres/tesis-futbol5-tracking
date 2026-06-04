"""
metrica_loader.py
=================

Adaptador de datos Metrica Sports al sistema de coordenadas absolutas
del campo (metros, origen en el centro).

Dos puntos de entrada:

1. ``load_via_kloppy(path, ...)``  -- high-level. Carga un archivo de
   tracking de Metrica Sports usando kloppy y devuelve un DataFrame
   largo con coordenadas en metros. Requiere ``kloppy`` instalado.

2. ``from_kloppy_dataset(dataset, ...)``  -- mid-level. Toma un
   ``kloppy.domain.TrackingDataset`` ya construido (por ejemplo desde
   otro provider: sportec, tracab, opta, ...) y produce el mismo
   DataFrame. Tambien requiere kloppy.

3. ``rescale_norm_to_meters(x, y, field)``  -- low-level. Solo la
   matematica de reescalado. No requiere kloppy. Es la primitiva que
   usan las dos funciones anteriores y que reutiliza
   ``tracking/homography.py`` para mantener la misma convencion.

Convencion de coordenadas (alineada con tracking/homography.py):

    u, v in [0, 1]  ->  x_m, y_m in metros
    x_m = (u - 0.5) * field.length_m
    y_m = (v - 0.5) * field.width_m

Origen en el centro, +x a la derecha, +y arriba.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    try:
        from kloppy.domain import TrackingDataset
    except ImportError:  # pragma: no cover - solo type hints
        TrackingDataset = Any  # type: ignore[assignment]


DEFAULT_FIELD_LENGTH_M: float = 105.0
DEFAULT_FIELD_WIDTH_M: float = 68.0


@dataclass(frozen=True)
class FieldDims:
    """Dimensiones del campo en metros. Default: 105x68 (FIFA 11-a-side)."""

    length_m: float = DEFAULT_FIELD_LENGTH_M
    width_m: float = DEFAULT_FIELD_WIDTH_M


def rescale_norm_to_meters(
    x_norm: np.ndarray | float,
    y_norm: np.ndarray | float,
    field: FieldDims | None = None,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    Reescala coordenadas normalizadas ``[0, 1]`` a metros absolutos
    con origen en el centro del campo.

    Parameters
    ----------
    x_norm, y_norm : array-like
        Coordenadas normalizadas. Acepta escalares, listas o ndarrays.
    field : FieldDims, optional
        Dimensiones del campo. Si es None, usa 105x68.

    Returns
    -------
    (x_m, y_m) : tupla del mismo tipo que la entrada.
        ``x_m = (x_norm - 0.5) * length_m``,
        ``y_m = (y_norm - 0.5) * width_m``.
    """
    f = field or FieldDims()
    x_arr = np.asarray(x_norm, dtype=np.float64)
    y_arr = np.asarray(y_norm, dtype=np.float64)
    x_m = (x_arr - 0.5) * f.length_m
    y_m = (y_arr - 0.5) * f.width_m
    if np.isscalar(x_norm) and np.isscalar(y_norm):
        return float(x_m), float(y_m)
    return x_m, y_m


def _require_kloppy():
    """Importa kloppy perezosamente. Lanza un error claro si no esta."""
    try:
        import kloppy  # noqa: F401
        return kloppy
    except ImportError as e:
        raise ImportError(
            "kloppy es necesario para esta funcion. "
            "Instala con: pip install kloppy"
        ) from e


def from_kloppy_dataset(
    dataset: "TrackingDataset",
    *,
    field: FieldDims | None = None,
    match_id: str | None = None,
) -> pd.DataFrame:
    """
    Convierte un ``kloppy.TrackingDataset`` a un DataFrame largo en metros.

    Espera que el dataset ya este en coordenadas normalizadas ``[0, 1]``
    (KloppyCoordinateSystem o equivalente). Si tu dataset esta en otro
    sistema, llama primero a ``dataset.transform(to_coordinate_system(...))``.

    Columnas de salida (una fila por par (frame, jugador)):

        match_id, frame, time_s, period,
        team, track_id, x_norm, y_norm, x_m, y_m

    Parameters
    ----------
    dataset : kloppy.domain.TrackingDataset
        Dataset ya cargado.
    field : FieldDims, optional
        Dimensiones del campo. Default 105x68.
    match_id : str, optional
        Override del match_id. Si es None, intenta leerlo de
        ``dataset.metadata.match_id`` o del nombre del primer frame.
    """
    _require_kloppy()  # sanity check: kloppy instalado

    f = field or FieldDims()
    md = getattr(dataset, "metadata", None)
    resolved_match_id = (
        match_id
        or (getattr(md, "match_id", None) if md else None)
        or "unknown_match"
    )

    records: list[dict[str, Any]] = []
    for frame in dataset.frames:
        frame_id = int(getattr(frame, "frame_id", 0))
        period_id = int(getattr(getattr(frame, "period", None), "id", 0))

        # kloppy >=3 usa frame.time como timedelta; <3 usaba timestamp.
        t_attr = getattr(frame, "time", None) or getattr(frame, "timestamp", None)
        if t_attr is None:
            time_s = float(frame_id) / 25.0
        elif hasattr(t_attr, "total_seconds"):
            time_s = float(t_attr.total_seconds())
        else:
            time_s = float(t_attr)

        for player in frame.players:
            team_obj = getattr(player, "team", None)
            team = (
                str(getattr(team_obj, "name", "unknown")).lower()
                if team_obj is not None
                else "unknown"
            )
            x_n = float(player.x)
            y_n = float(player.y)
            x_m, y_m = rescale_norm_to_meters(x_n, y_n, f)
            records.append(
                {
                    "match_id": resolved_match_id,
                    "frame": frame_id,
                    "time_s": time_s,
                    "period": period_id,
                    "team": team,
                    "track_id": str(getattr(player, "player_id", "")),
                    "x_norm": x_n,
                    "y_norm": y_n,
                    "x_m": x_m,
                    "y_m": y_m,
                }
            )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    return df


def load_via_kloppy(
    path: str | Path,
    *,
    field: FieldDims | None = None,
    match_id: str | None = None,
    sample_rate: float | None = None,
    only_alive: bool = True,
) -> pd.DataFrame:
    """
    Carga un archivo de tracking de Metrica Sports con kloppy y devuelve
    un DataFrame largo con coordenadas en metros.

    Parameters
    ----------
    path : str | Path
        Ruta al archivo CSV/HDF5/XML de Metrica Sports. El provider
        exacto se selecciona segun la extension / estructura; kloppy
        detecta el formato mas comun (open data de Metrica).
    field : FieldDims, optional
        Dimensiones del campo (default 105x68).
    match_id : str, optional
        Override del match_id en la columna de salida.
    sample_rate : float, optional
        Si se da, submuestrea a esa tasa (Hz).
    only_alive : bool
        Si True, descarta frames donde la pelota no esta "en juego"
        (solo aplicable si el provider distingue frames alive/dead).

    Notas
    -----
    La API exacta de kloppy para Metrica varia entre versiones. Esta
    funcion intenta primero ``metrica.load`` y, si no existe, hace
    fallback a la API generica de kloppy. Si tu version de kloppy
    requiere argumentos distintos, usa ``from_kloppy_dataset`` pasando
    el dataset que tu mismo cargues.
    """
    kloppy = _require_kloppy()
    f = field or FieldDims()
    p = Path(path)
    resolved_match_id = match_id or p.stem

    dataset = None
    # 1) Intento: provider dedicado metrica.load
    metrica_mod = getattr(kloppy, "metrica", None)
    if metrica_mod is not None and hasattr(metrica_mod, "load"):
        try:
            kwargs: dict[str, Any] = {}
            if sample_rate is not None:
                kwargs["sample_rate"] = sample_rate
            if "only_alive" in getattr(metrica_mod.load, "__code__", metrica_mod.load).co_varnames:
                kwargs["only_alive"] = only_alive
            dataset = metrica_mod.load(raw_data=str(p), **kwargs)
        except Exception:
            dataset = None

    # 2) Fallback: open data de Metrica
    if dataset is None and metrica_mod is not None and hasattr(metrica_mod, "load_open_data"):
        try:
            dataset = metrica_mod.load_open_data(match_id=resolved_match_id, **{
                "sample_rate": sample_rate,
            } if sample_rate is not None else {})
        except Exception:
            dataset = None

    if dataset is None:
        raise RuntimeError(
            f"No se pudo cargar {p} con kloppy {kloppy.__version__}. "
            "Prueba cargando el dataset manualmente y pasando el "
            "resultado a from_kloppy_dataset()."
        )

    return from_kloppy_dataset(dataset, field=f, match_id=resolved_match_id)
