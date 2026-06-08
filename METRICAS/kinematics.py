"""
kinematics.py
=============

Calculo cinematico por track_id: velocidad, rapidez y aceleracion.

Las ecuaciones de diferenciacion provienen de
``LaurieOnTracking repo/Metrica_Velocities.py``:

    vx = dx / dt
    vy = dy / dt
    speed = sqrt(vx^2 + vy^2)

Laurie usa diferencias hacia ATRAS (``series.diff() / dt``) y luego
suaviza con Savitzky-Golay. Ac usamos **diferencias centrales** (que
son de segundo orden en precision en vez de primer orden), porque el
filtro de posicion se aplica ANTES en :func:`METRICAS.smoothing` y por
lo tanto no necesitamos que la diferenciacion misma sea suavizante.

Para la aceleracion volvemos a aplicar diferencias centrales sobre las
velocidades resultantes.

    vx[i] = (x[i+1] - x[i-1]) / (t[i+1] - t[i-1])   para 1 <= i <= n-2
    ax[i] = (vx[i+1] - vx[i-1]) / (t[i+1] - t[i-1])  para 2 <= i <= n-3

Los bordes quedan en NaN (no hay vecindad completa).

Opcionalmente se aplica un cap de velocidad maxima (12 m/s por
defecto, mismo valor que usa Laurie) para descartar outliers
provocados por errores de deteccion: cualquier componente de velocidad
cuya magnitud combinada supere ``maxspeed`` se reemplaza por NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MAX_SPEED_MPS: float = 12.0
KINEMATICS_COLUMNS = (
    "vx_mps",
    "vy_mps",
    "speed_mps",
    "ax_mps2",
    "ay_mps2",
    "accel_mps2",
)


def _central_diff_1d(
    series: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Diferencias centrales sobre una serie 1D indexada por tiempo.

    Devuelve (vel, acc) con NaN en los bordes.

        vel[i] = (series[i+1] - series[i-1]) / (t[i+1] - t[i-1])
        acc[i] = d(vel)/dt  (via np.gradient, central O(h^2) en interior)

    vel tiene datos en [1, n-2]; acc combina NaN de vel con el resultado
    de np.gradient (diferencias de primer orden en los bordes, que
    quedan NaN porque vel[0] y vel[n-1] lo son).
    """
    n = series.size
    vel = np.full(n, np.nan, dtype=np.float64)
    acc = np.full(n, np.nan, dtype=np.float64)
    if n < 3:
        return vel, acc

    dt_total = t[2:] - t[:-2]
    valid_dt = dt_total > 0
    vel[1:-1] = np.where(valid_dt, (series[2:] - series[:-2]) / dt_total, np.nan)

    # Aceleracion via np.gradient: diferencias centrales O(h^2) en el
    # interior; bordes con diferencias de primer orden (quedan NaN porque
    # vel[0] / vel[n-1] son NaN).
    acc[:] = np.gradient(vel, t)
    return vel, acc


def _grouped_central_diff(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    time_col: str,
    group_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica diferencias centrales a X e Y por grupo. Devuelve (vx, vy)."""
    n = len(df)
    vx = np.full(n, np.nan, dtype=np.float64)
    vy = np.full(n, np.nan, dtype=np.float64)

    if group_col is None or group_col not in df.columns:
        vx[:], _ = _central_diff_1d(df[x_col].to_numpy(), df[time_col].to_numpy())
        vy[:], _ = _central_diff_1d(df[y_col].to_numpy(), df[time_col].to_numpy())
        return vx, vy

    for _, idx in df.groupby(group_col, sort=False).groups.items():
        idx_arr = np.asarray(idx)
        sub = df.loc[idx_arr]
        t = sub[time_col].to_numpy()
        vx_sub, _ = _central_diff_1d(sub[x_col].to_numpy(), t)
        vy_sub, _ = _central_diff_1d(sub[y_col].to_numpy(), t)
        vx[idx_arr] = vx_sub
        vy[idx_arr] = vy_sub
    return vx, vy


def _grouped_accel(
    df: pd.DataFrame,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    time_col: str,
    group_col: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica diferencias centrales a vx y vy (ya calculados). Devuelve (ax, ay)."""
    n = len(df)
    ax = np.full(n, np.nan, dtype=np.float64)
    ay = np.full(n, np.nan, dtype=np.float64)
    if group_col is None or group_col not in df.columns:
        ax[:], _ = _central_diff_1d(vx, df[time_col].to_numpy())
        ay[:], _ = _central_diff_1d(vy, df[time_col].to_numpy())
        return ax, ay
    for _, idx in df.groupby(group_col, sort=False).groups.items():
        idx_arr = np.asarray(idx)
        t = df.loc[idx_arr, time_col].to_numpy()
        ax_sub, _ = _central_diff_1d(vx[idx_arr], t)
        ay_sub, _ = _central_diff_1d(vy[idx_arr], t)
        ax[idx_arr] = ax_sub
        ay[idx_arr] = ay_sub
    return ax, ay


def remove_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las columnas cinematicas si ya estan presentes."""
    drop = [c for c in KINEMATICS_COLUMNS if c in df.columns]
    return df.drop(columns=drop) if drop else df.copy()


def compute_kinematics(
    df: pd.DataFrame,
    *,
    fps: float = 25.0,
    max_speed: float = DEFAULT_MAX_SPEED_MPS,
    smoothing: bool = True,
    smooth_window: int = 5,
    group_col: str = "track_id",
    time_col: str = "time_s",
    x_col: str = "x_m",
    y_col: str = "y_m",
) -> pd.DataFrame:
    """
    Calcula velocidad, rapidez y aceleracion por track_id sobre un
    DataFrame en formato largo.

    Parameters
    ----------
    df : DataFrame
        Debe contener ``x_col``, ``y_col`` y, o bien ``time_col``
        (float, segundos), o bien una columna ``frame`` (int) de la
        cual se derivan tiempos uniformes con ``fps``.
    fps : float
        Frames por segundo. Solo se usa si ``time_col`` no esta en
        el DataFrame (en ese caso se crea ``time_s = frame / fps``).
    max_speed : float
        Tope de rapidez (m/s). Las muestras con speed > max_speed se
        marcan como NaN (outliers de deteccion). 0 o negativo desactiva.
    smoothing : bool
        Si True, aplica una media movil central previa a X e Y
        (ventana ``smooth_window``) antes de diferenciar. Por defecto
        True, como pide la etapa 2.
    smooth_window : int
        Longitud de la ventana de suavizado (impar recomendado).
    group_col : str
        Columna para agrupar (un valor unico por objeto trackeado).
    time_col : str
        Nombre de la columna temporal en segundos.
    x_col, y_col : str
        Columnas de coordenadas en metros.

    Returns
    -------
    DataFrame (copia) con las columnas adicionales:
        vx_mps, vy_mps, speed_mps, ax_mps2, ay_mps2, accel_mps2
    """
    out = df.copy()

    # Tiempo en segundos
    if time_col not in out.columns:
        if "frame" not in out.columns:
            raise ValueError(
                f"DataFrame sin columna '{time_col}' ni 'frame' para derivar tiempo."
            )
        out[time_col] = out["frame"].astype(float) / float(fps)
    out[time_col] = out[time_col].astype(float)

    # Suavizado previo
    if smoothing:
        from .smoothing import central_moving_average

        if group_col is None or group_col not in out.columns:
            out[x_col] = central_moving_average(out[x_col].to_numpy(), smooth_window)
            out[y_col] = central_moving_average(out[y_col].to_numpy(), smooth_window)
        else:
            for _, idx in out.groupby(group_col, sort=False).groups.items():
                idx_arr = np.asarray(idx)
                out.loc[idx_arr, x_col] = central_moving_average(
                    out.loc[idx_arr, x_col].to_numpy(), smooth_window
                )
                out.loc[idx_arr, y_col] = central_moving_average(
                    out.loc[idx_arr, y_col].to_numpy(), smooth_window
                )

    # Velocidad por diferencias centrales
    vx, vy = _grouped_central_diff(
        out, x_col=x_col, y_col=y_col, time_col=time_col, group_col=group_col
    )
    speed = np.sqrt(vx * vx + vy * vy)

    # Cap de velocidad maxima
    if max_speed and max_speed > 0:
        mask = speed > max_speed
        vx = np.where(mask, np.nan, vx)
        vy = np.where(mask, np.nan, vy)
        speed = np.where(mask, np.nan, speed)

    # Aceleracion: diferencias centrales sobre vx, vy
    ax, ay = _grouped_accel(out, vx, vy, time_col=time_col, group_col=group_col)
    accel = np.sqrt(ax * ax + ay * ay)

    out["vx_mps"] = vx
    out["vy_mps"] = vy
    out["speed_mps"] = speed
    out["ax_mps2"] = ax
    out["ay_mps2"] = ay
    out["accel_mps2"] = accel
    return out
