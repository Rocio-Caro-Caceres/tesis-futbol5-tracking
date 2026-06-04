"""
homography.py
=============

Transformacion de coordenadas pixel -> coordenadas normalizadas del campo.

En etapa 1 dejamos una homografia IDENTIDAD parametrizable. Esto significa:
    x_norm = x / width
    y_norm = y / height

La idea es que la DB ya tenga coordenadas normalizadas en [0,1] aunque la
homografia real (4 puntos del campo en pixeles) aun no este calibrada.

Cuando se calibre la homografia real:
    1. Calcular matriz H de 3x3 con cv2.getPerspectiveTransform(src, dst).
    2. Almacenar la matriz en match_summary.homography.
    3. En este modulo, leer H y aplicarla: dst = H @ [x, y, 1].
    4. Llevar a metros: x_m = (x_norm - 0.5) * field_length_m,
       y_m = (y_norm - 0.5) * field_width_m, con origen en el centro
       y convencion +x derecha, +y arriba (como LaurieOnTracking).

La conversion a PostGIS Point en EPSG:3857 se hace en el ingest,
no aca, para mantener este modulo puro (sin dependencias de PostGIS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FieldDims:
    """Dimensiones del campo en metros (futbol 5 = 42x25 aprox, futbol 11 = 105x68)."""
    length_m: float = 105.0
    width_m: float = 68.0


@dataclass
class PixelToField:
    """
    Conversor pixel -> campo. Si 'H' es None, usa la identidad (normaliza por
    tamano de frame); si esta provisto, aplica la homografia perspectiva.
    """
    H: np.ndarray | None          # matriz 3x3, fila mayor
    width: int
    height: int
    field: FieldDims = FieldDims()

    def to_norm(self, x: float, y: float) -> tuple[float, float]:
        """Devuelve (x_norm, y_norm) en [0, 1] sobre el campo."""
        if self.H is None:
            # Identidad parametrizada por el tamano del frame
            return (x / max(self.width, 1), y / max(self.height, 1))

        # Homografia perspectiva: [u, v, w]^T = H @ [x, y, 1]^T
        p = self.H @ np.array([x, y, 1.0], dtype=np.float64)
        w = p[2] if abs(p[2]) > 1e-9 else 1e-9
        return (float(p[0] / w), float(p[1] / w))

    def to_meters(self, x: float, y: float) -> tuple[float, float]:
        """
        Devuelve (x_m, y_m) en metros con origen en el centro de la cancha,
        +x hacia la derecha, +y hacia arriba.
        Por convencion de Metrica/Laurie: tras aplicar H, las coords ya estan
        en [0,1], asi que (0.5, 0.5) = centro.
        """
        u, v = self.to_norm(x, y)
        x_m = (u - 0.5) * self.field.length_m
        y_m = (v - 0.5) * self.field.width_m
        return (x_m, y_m)

    def to_all(self, x: float, y: float) -> dict:
        """Helper: devuelve los tres pares (x_norm, y_norm) y (x_m, y_m)."""
        u, v = self.to_norm(x, y)
        x_m, y_m = self.to_meters(x, y)
        return {
            "x_norm": u,
            "y_norm": v,
            "x_m":    x_m,
            "y_m":    y_m,
        }

    @staticmethod
    def from_match_summary(row: dict | None, width: int, height: int,
                           field: FieldDims | None = None) -> "PixelToField":
        """
        Construye el conversor desde una fila de match_summary.
        'row' puede tener 'homography' como lista/array de 9 elementos (3x3 aplanada)
        o None.
        """
        H = None
        if row and row.get("homography") is not None:
            arr = np.asarray(row["homography"], dtype=np.float64)
            if arr.size == 9:
                H = arr.reshape(3, 3)
        return PixelToField(
            H=H,
            width=width,
            height=height,
            field=field or FieldDims(
                length_m=float(row["field_length_m"]) if row and row.get("field_length_m") else 105.0,
                width_m=float(row["field_width_m"])   if row and row.get("field_width_m")   else 68.0,
            ),
        )
