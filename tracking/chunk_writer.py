"""
chunk_writer.py
===============

Volcado atomico de los datos de tracking en chunks JSON, un archivo por
ventana temporal. Cada chunk respeta el formato:

    {
      "match_id": "partido_f5_2026-06-03",
      "schema_version": 1,
      "metadata": {
        "fps": 24,
        "width": 1920,
        "height": 1080,
        "field_length_m": 105.0,
        "field_width_m": 68.0,
        "start_frame": 0,
        "end_frame": 149,
        "homography": null | [9 floats],
        "created_at": "2026-06-03T22:00:00Z"
      },
      "tracking": [
        { "frame": 0, "track_id": 1, "object_type": "player",
          "team": "home", "x": 640, "y": 360,
          "x_norm": 0.5, "y_norm": 0.5, "x_m": 52.5, "y_m": 34.0,
          "bbox_x1": 600, "bbox_y1": 320, "bbox_x2": 680, "bbox_y2": 400,
          "confidence": 0.87, "in_occlusion": false }
      ],
      "ball": [
        { "frame": 0, "x": 320, "y": 240,
          "x_norm": 0.25, "y_norm": 0.22, "x_m": 26.25, "y_m": 15.0,
          "confidence": 0.5 }
      ]
    }

Reglas:
- Cada chunk cubre CHUNK_FRAMES frames (default 150 ~= 6 s a 24 fps).
- Escritura atomica: primero a .tmp, luego rename.
- Numeracion con padding 5 digitos para que el orden lexicografico
  coincida con el orden temporal.
- La pelota se separa en su propia lista para simplificar la insercion
  (object_type='ball', track_id=0).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .homography import PixelToField


SCHEMA_VERSION = 1
BALL_TRACK_ID = 0


@dataclass
class ChunkMetadata:
    fps: float
    width: int
    height: int
    field_length_m: float = 105.0
    field_width_m: float = 68.0
    homography: list[float] | None = None  # 3x3 aplanada, 9 floats
    start_frame: int = 0
    end_frame: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "field_length_m": self.field_length_m,
            "field_width_m": self.field_width_m,
            "homography": self.homography,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "created_at": self.created_at,
        }


class ChunkWriter:
    """
    Acumula registros de tracking y pelota en memoria y los vuelca a disco
    cuando se completa una ventana de CHUNK_FRAMES frames.
    """

    def __init__(
        self,
        match_id: str,
        output_dir: str | Path,
        metadata: ChunkMetadata,
        chunk_frames: int = 150,
        overwrite: bool = False,
    ) -> None:
        self.match_id = match_id
        self.output_dir = Path(output_dir) / match_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_template = metadata
        self.chunk_frames = int(chunk_frames)
        self.overwrite = overwrite

        self._tracking: list[dict[str, Any]] = []
        self._ball: list[dict[str, Any]] = []
        self._chunk_index: int = 0
        self._start_frame: int | None = None

    # ------------------------------------------------------------------ API
    def add_player(
        self,
        *,
        frame: int,
        track_id: int,
        team: str,
        x: float,
        y: float,
        bbox: tuple[int, int, int, int] | None,
        confidence: float | None,
        in_occlusion: bool,
        converter: PixelToField,
    ) -> None:
        coords = converter.to_all(float(x), float(y))
        rec: dict[str, Any] = {
            "frame":        int(frame),
            "track_id":     int(track_id),
            "object_type":  "player",
            "team":         team if team in ("home", "away") else "unknown",
            "x":            float(x),
            "y":            float(y),
            "x_norm":       coords["x_norm"],
            "y_norm":       coords["y_norm"],
            "x_m":          coords["x_m"],
            "y_m":          coords["y_m"],
            "confidence":   float(confidence) if confidence is not None else None,
            "in_occlusion": bool(in_occlusion),
        }
        if bbox is not None:
            rec["bbox_x1"] = int(bbox[0])
            rec["bbox_y1"] = int(bbox[1])
            rec["bbox_x2"] = int(bbox[2])
            rec["bbox_y2"] = int(bbox[3])
        self._mark_frame(frame)
        self._tracking.append(rec)

    def add_ball(
        self,
        *,
        frame: int,
        x: float,
        y: float,
        confidence: float | None,
        converter: PixelToField,
    ) -> None:
        coords = converter.to_all(float(x), float(y))
        rec = {
            "frame":       int(frame),
            "track_id":    BALL_TRACK_ID,
            "object_type": "ball",
            "x":           float(x),
            "y":           float(y),
            "x_norm":      coords["x_norm"],
            "y_norm":      coords["y_norm"],
            "x_m":         coords["x_m"],
            "y_m":         coords["y_m"],
            "confidence":  float(confidence) if confidence is not None else None,
        }
        self._mark_frame(frame)
        self._ball.append(rec)

    def maybe_flush(self) -> Path | None:
        """Vuelca el chunk si se completo la ventana. Devuelve la ruta escrita o None."""
        if self._start_frame is None:
            return None
        if (self._last_frame - self._start_frame + 1) < self.chunk_frames:
            return None
        return self.flush()

    def flush(self) -> Path:
        """Fuerza el volcado del chunk actual. Devuelve la ruta escrita."""
        if self._start_frame is None:
            # Nada acumulado: escribimos un chunk vacio igualmente? Mejor no.
            raise RuntimeError("ChunkWriter.flush() llamado sin datos.")

        meta = ChunkMetadata(
            fps=self.metadata_template.fps,
            width=self.metadata_template.width,
            height=self.metadata_template.height,
            field_length_m=self.metadata_template.field_length_m,
            field_width_m=self.metadata_template.field_width_m,
            homography=self.metadata_template.homography,
            start_frame=self._start_frame,
            end_frame=self._last_frame,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        payload = {
            "match_id": self.match_id,
            "schema_version": SCHEMA_VERSION,
            "metadata": meta.to_dict(),
            "tracking": self._tracking,
            "ball":     self._ball,
        }

        path = self.output_dir / f"chunk_{self._chunk_index:05d}.json"
        self._write_atomic(path, payload)

        self._tracking.clear()
        self._ball.clear()
        self._chunk_index += 1
        self._start_frame = None
        self._last_frame = -1
        return path

    def close(self) -> Path | None:
        """Vuelca lo que quede acumulado. Devuelve la ruta o None si estaba vacio."""
        if self._start_frame is None:
            return None
        return self.flush()

    # ------------------------------------------------------------- internals
    def _mark_frame(self, frame: int) -> None:
        if self._start_frame is None:
            self._start_frame = frame
        self._last_frame = frame

    def _write_atomic(self, path: Path, payload: dict[str, Any]) -> None:
        if path.exists() and not self.overwrite:
            raise FileExistsError(
                f"El chunk {path} ya existe. Pase overwrite=True para sobrescribir."
            )
        # NamedTemporaryFile en el mismo directorio para que el rename sea atomico.
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            finally:
                raise
