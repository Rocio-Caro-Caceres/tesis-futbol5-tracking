"""
heatmap.py
==========

Generador de mapas de calor (densidad de posiciones en el campo).

Pipeline:

    1. Discretizacion: ``np.histogram2d`` arma una grilla regular de
       ``bins x bins`` celdas sobre las dimensiones del campo.
    2. Suavizado: ``scipy.ndimage.gaussian_filter`` aplica un kernel
       gaussiano 2D (``sigma`` en unidades de bins) sobre la grilla
       cruda para obtener la matriz de densidad.
    3. Export: ``save_heatmap_npy`` (formato binario numpy) y
       ``save_heatmap_csv`` (matriz transpuesta a filas x columnas).

scipy se importa perezosamente para que el resto del paquete funcione
sin scipy instalado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrica_loader import DEFAULT_FIELD_LENGTH_M, DEFAULT_FIELD_WIDTH_M, FieldDims


def _require_scipy():
    try:
        import scipy.ndimage  # noqa: F401
        return scipy.ndimage
    except ImportError as e:
        raise ImportError(
            "scipy es necesario para build_heatmap. "
            "Instala con: pip install scipy"
        ) from e


def build_heatmap(
    df: pd.DataFrame,
    *,
    x_col: str = "x_m",
    y_col: str = "y_m",
    bins: int | tuple[int, int] = 50,
    field: FieldDims | None = None,
    sigma: float = 1.5,
    normalize: bool = False,
) -> dict[str, Any]:
    """
    Construye la matriz de densidad de posiciones.

    Parameters
    ----------
    df : DataFrame
        Tracking en formato largo. Solo se usan las columnas ``x_col``
        e ``y_col``; se descartan filas con NaN en alguna de las dos.
    x_col, y_col : str
        Columnas con las coordenadas (metros).
    bins : int | (nx, ny)
        Numero de bins por eje. Default 50x50.
    field : FieldDims, optional
        Dimensiones del campo. Default 105x68. Las celdas del histograma
        cubren el rango ``[-L/2, +L/2] x [-W/2, +W/2]``.
    sigma : float
        Desviacion estandar del kernel gaussiano, en unidades de bins.
        ``0`` desactiva el suavizado.
    normalize : bool
        Si True, divide la densidad por su suma para que integre a 1.

    Returns
    -------
    dict con:
        H : (ny, nx) ndarray
            Matriz de densidad (counts o densidad normalizada). Las filas
            corresponden al eje Y (ancho del campo) y las columnas al
            eje X (largo del campo). Pensada para ``imshow(..., origin='lower')``
            o ``pcolormesh``.
        xedges, yedges : ndarrays
            Bordes de los bins en metros.
        x_m, y_m : ndarrays
            Centros de los bins en metros (utiles para graficar).
        bins : tuple[int, int]
        field : FieldDims
    """
    nd = _require_scipy()

    f = field or FieldDims(length_m=DEFAULT_FIELD_LENGTH_M, width_m=DEFAULT_FIELD_WIDTH_M)
    if isinstance(bins, int):
        nx = ny = int(bins)
    else:
        nx, ny = (int(b) for b in bins)
    if nx < 1 or ny < 1:
        raise ValueError("bins debe ser >= 1")

    sub = df[[x_col, y_col]].dropna()
    x = sub[x_col].to_numpy(dtype=np.float64)
    y = sub[y_col].to_numpy(dtype=np.float64)
    if x.size == 0:
        raise ValueError("DataFrame vacio tras descartar NaN en X / Y.")

    x_range = (-f.length_m / 2.0, f.length_m / 2.0)
    y_range = (-f.width_m / 2.0, f.width_m / 2.0)

    # np.histogram2d devuelve (nx, ny); transponemos a (ny, nx) para que
    # la convencion del modulo sea filas = Y, columnas = X.
    H_xy, xedges, yedges = np.histogram2d(x, y, bins=[nx, ny], range=[x_range, y_range])
    H = H_xy.T

    if sigma and sigma > 0:
        H = nd.gaussian_filter(H, sigma=float(sigma), mode="constant", cval=0.0)

    if normalize:
        total = H.sum()
        if total > 0:
            H = H / total

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    return {
        "H": H,
        "xedges": xedges,
        "yedges": yedges,
        "x_m": x_centers,
        "y_m": y_centers,
        "bins": (nx, ny),
        "field": f,
    }


def save_heatmap_npy(heatmap: dict[str, Any], path: str | Path) -> Path:
    """Guarda la matriz ``H`` y los ejes en un .npz."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        H=heatmap["H"],
        xedges=heatmap["xedges"],
        yedges=heatmap["yedges"],
        x_m=heatmap["x_m"],
        y_m=heatmap["y_m"],
    )
    return p


def save_heatmap_csv(heatmap: dict[str, Any], path: str | Path) -> Path:
    """
    Guarda la matriz densa en CSV: una cabecera con los centros de los
    bins en X (en metros) y una columna ``y_m`` con los centros de los
    bins en Y. El resto son los counts/densidad.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    H = np.asarray(heatmap["H"])
    x_centers = np.asarray(heatmap["x_m"])
    y_centers = np.asarray(heatmap["y_m"])
    df = pd.DataFrame(H, index=y_centers, columns=x_centers)
    df.index.name = "y_m"
    df.to_csv(p, float_format="%.6f")
    return p
