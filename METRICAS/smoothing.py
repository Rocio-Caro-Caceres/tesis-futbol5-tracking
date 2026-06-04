"""
smoothing.py
============

Filtro de media movil CENTRAL (simetrica) con ventana parametrica.

Objetivo: suavizar las coordenadas ``X, Y`` de los centroides de los
jugadores antes de calcular metricas derivadas (velocidad, aceleracion),
para mitigar las oscilaciones producidas por jitter del detector.

A diferencia de la media movil causal de ``Metrica_Velocities.py`` (que
mira solo el pasado), esta implementacion es **simetrica**: cada salida
``y[i]`` promedia ``w/2`` muestras a la izquierda y ``w/2`` a la
derecha. Esto preserva la fase de la senial (no introduce lag), que es
lo que se quiere cuando el filtro se aplica a *posiciones* y no a
*velocidades*.

Parametros
----------
- ``window`` : longitud de la ventana (se recomienda impar; cualquier
  entero >= 1 es valido).
- ``min_periods`` : cantidad minima de valores no-NaN en la ventana
  para emitir un valor no-NaN. Si la ventana tiene mas NaN que
  ``window - min_periods``, la salida es NaN.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def central_moving_average(
    series: Iterable[float] | np.ndarray,
    window: int = 5,
    *,
    min_periods: int | None = None,
) -> np.ndarray:
    """
    Media movil central de longitud ``window`` sobre ``series``.

    Para ``window = 2k+1``:
        y[i] = mean(series[i-k : i+k+1])   para k <= i <= n-1-k
        y[i] = NaN                           en los bordes

    Para ``window = 2k`` se redondea hacia la izquierda:
        y[i] = mean(series[i-k : i+k])       para k <= i <= n-k

    Parameters
    ----------
    series : array-like
        Senial 1D a suavizar.
    window : int
        Longitud de la ventana. ``1`` devuelve la entrada tal cual.
    min_periods : int, optional
        Minimo de muestras no-NaN requeridas para emitir un valor
        finito. Por defecto exige la ventana completa (``window``).
        Si la ventana tiene mas NaN que ``window - min_periods``,
        el output es NaN en esa posicion.

    Returns
    -------
    np.ndarray de la misma longitud que ``series`` (dtype float64).
    """
    arr = np.asarray(series, dtype=np.float64)
    n = arr.size
    w = int(window)

    if w < 1:
        raise ValueError("window debe ser >= 1")
    if w == 1:
        return arr.copy()
    if w > n:
        return np.full(n, np.nan, dtype=np.float64)

    k_left = w // 2
    k_right = w - k_left - 1
    minp = int(min_periods) if min_periods is not None else w
    if minp < 1 or minp > w:
        raise ValueError(f"min_periods debe estar en [1, {w}]")

    out = np.empty(n, dtype=np.float64)
    if k_left > 0:
        out[:k_left] = np.nan
    if k_right > 0:
        out[n - k_right:] = np.nan

    windows = sliding_window_view(arr, w, axis=0)  # (n - w + 1, w)
    n_valid = windows.shape[0]
    if n_valid <= 0:
        return out

    if minp <= 1:
        out[k_left : k_left + n_valid] = windows.mean(axis=1)
    else:
        nan_count = np.isnan(windows).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(
                nan_count <= (w - minp),
                np.nansum(windows, axis=1) / w,
                np.nan,
            )
        out[k_left : k_left + n_valid] = means

    return out


def smooth_positions(
    df: pd.DataFrame,
    *,
    window: int = 5,
    x_col: str = "x_m",
    y_col: str = "y_m",
    out_x: str = "x_m_smooth",
    out_y: str = "y_m_smooth",
    group_col: str = "track_id",
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Aplica la media movil central a las columnas X, Y por grupo
    (por defecto, por ``track_id``).

    Mantiene el orden del DataFrame de entrada: la salida tiene
    exactamente las mismas filas, en el mismo orden, mas dos columnas
    nuevas con las coordenadas suavizadas.

    Parameters
    ----------
    df : DataFrame
        Tracking en formato largo.
    window : int
        Longitud de la ventana de suavizado.
    x_col, y_col : str
        Columnas de entrada con las coordenadas (en metros).
    out_x, out_y : str
        Nombres de las columnas de salida.
    group_col : str
        Columna para agrupar. Si es None, suaviza sobre todo el
        DataFrame como un unico grupo (no recomendado: no respeta
        discontinuidades entre jugadores).
    min_periods : int, optional
        Minimo de muestras no-NaN por ventana (ver
        :func:`central_moving_average`).

    Returns
    -------
    DataFrame (copia) con las columnas ``out_x`` y ``out_y`` anadidas.
    """
    out = df.copy()
    if group_col is None or group_col not in out.columns:
        sx = central_moving_average(out[x_col].to_numpy(), window, min_periods=min_periods)
        sy = central_moving_average(out[y_col].to_numpy(), window, min_periods=min_periods)
        out[out_x] = sx
        out[out_y] = sy
        return out

    sx_all = np.full(len(out), np.nan, dtype=np.float64)
    sy_all = np.full(len(out), np.nan, dtype=np.float64)
    for _, idx in out.groupby(group_col, sort=False).groups.items():
        idx_arr = np.asarray(idx)
        sub = out.loc[idx_arr]
        sx_all[idx_arr] = central_moving_average(
            sub[x_col].to_numpy(), window, min_periods=min_periods
        )
        sy_all[idx_arr] = central_moving_average(
            sub[y_col].to_numpy(), window, min_periods=min_periods
        )
    out[out_x] = sx_all
    out[out_y] = sy_all
    return out
