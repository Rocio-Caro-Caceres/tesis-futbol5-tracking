"""
distance.py
===========

Distancia euclidea acumulada por track_id, con umbral de corte para
omitir pasos pequenos que corresponden al jugador estatico (jitter del
detector o micro-movimientos).

    step[i] = ||(x[i], y[i]) - (x[i-1], y[i-1])||_2
    if step[i] < min_step  ->  tratar como 0
    cumdist[i] = sum_{k=1..i} step[k]

El umbral por defecto es 0.03 m, alineado con la consigna de la etapa 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MIN_STEP_M: float = 0.03


def cumulative_distance(
    df: pd.DataFrame,
    *,
    min_step: float = DEFAULT_MIN_STEP_M,
    group_col: str = "track_id",
    x_col: str = "x_m",
    y_col: str = "y_m",
    time_col: str = "time_s",
    out_col: str = "cumdist_m",
    step_col: str = "step_m",
) -> pd.DataFrame:
    """
    Anade columnas ``step_m`` (paso instantaneo, NaN en la primera fila
    de cada grupo) y ``cumdist_m`` (distancia acumulada por grupo) al
    DataFrame de tracking.

    Parameters
    ----------
    df : DataFrame
        Tracking en formato largo, ordenado por ``group_col`` y tiempo.
    min_step : float
        Umbral de paso en metros. Pasos con ``step < min_step`` se
        cuentan como 0 (jugador estatico).
    group_col : str
        Columna por la que agrupar (un objeto trackeado).
    x_col, y_col : str
        Coordenadas en metros.
    time_col : str
        Columna temporal (segundos). Solo se usa para decidir el orden.
    out_col : str
        Nombre de la columna de salida con la distancia acumulada.
    step_col : str
        Nombre de la columna de salida con el paso instantaneo.

    Returns
    -------
    DataFrame (copia) con ``step_col`` y ``out_col`` anadidas.
    """
    if min_step < 0:
        raise ValueError("min_step debe ser >= 0")

    out = df.copy()
    n = len(out)
    step = np.full(n, np.nan, dtype=np.float64)
    cum = np.zeros(n, dtype=np.float64)

    sort_cols = [c for c in (group_col, time_col, "frame") if c in out.columns]

    if group_col is None or group_col not in out.columns:
        ordered = out.sort_values(sort_cols) if sort_cols else out
        x = ordered[x_col].to_numpy()
        y = ordered[y_col].to_numpy()
        s, c = _cumdist_1d(x, y, min_step)
        step[ordered.index.to_numpy()] = s
        cum[ordered.index.to_numpy()] = c
    else:
        for _, idx in out.groupby(group_col, sort=False).groups.items():
            idx_arr = np.asarray(idx)
            sub = out.loc[idx_arr]
            sort_idx = sub.sort_values(sort_cols).index.to_numpy() if sort_cols else idx_arr
            x = out.loc[sort_idx, x_col].to_numpy()
            y = out.loc[sort_idx, y_col].to_numpy()
            s, c = _cumdist_1d(x, y, min_step)
            step[sort_idx] = s
            cum[sort_idx] = c

    out[step_col] = step
    out[out_col] = cum
    return out


def _cumdist_1d(
    x: np.ndarray, y: np.ndarray, min_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula paso y acumulado sobre una secuencia ordenada.

    Si ``x`` o ``y`` tienen NaN en una posicion, el paso en esa
    posicion queda en NaN en la salida ``step`` y se cuenta como 0
    para la acumulada (no se propaga NaN hacia adelante). Esto refleja
    la convencion de que un jugador sin posicion observable esta
    "estatico" a los efectos del conteo de distancia.
    """
    n = x.size
    step = np.full(n, np.nan, dtype=np.float64)
    cum = np.zeros(n, dtype=np.float64)
    if n < 2:
        return step, cum
    dx = np.diff(x)
    dy = np.diff(y)
    s = np.sqrt(dx * dx + dy * dy)
    s_for_sum = np.where(np.isnan(s), 0.0, s)
    s_for_sum = np.where(s_for_sum < min_step, 0.0, s_for_sum)
    step[1:] = s
    cum[1:] = np.cumsum(s_for_sum)
    return step, cum


def total_distance_per_track(
    df: pd.DataFrame,
    *,
    group_col: str = "track_id",
    cumdist_col: str = "cumdist_m",
) -> pd.DataFrame:
    """
    Resume la distancia total por track_id (ultimo valor de la columna
    ``cumdist_m`` dentro de cada grupo, que es la acumulada).
    """
    if cumdist_col not in df.columns:
        raise KeyError(
            f"Columna '{cumdist_col}' no encontrada. Llama primero a cumulative_distance()."
        )
    out = (
        df.groupby(group_col, as_index=False)[cumdist_col]
        .max()
        .rename(columns={cumdist_col: "total_distance_m"})
    )
    return out
