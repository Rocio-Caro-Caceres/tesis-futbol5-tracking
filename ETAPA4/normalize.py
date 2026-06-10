"""
normalize.py
============
Convierte el formato SoccerNet (tracking.jsonl.gz + Labels-v2.json) a:

  1. Filas atomicas compatibles con `tracking_events` (1 fila por objeto
     por frame): se guardan en Parquet (`<match>__h<N>_tracking.parquet`)
     y, si se pide, se ingestan en la DB.

  2. Filas de `event_training_labels`: filtradas a Pases / Duelos / Faltas
     (las 3 clases de Etapa 4), con `start_x_m`/`start_y_m` resueltos
     haciendo JOIN contra el tracking de la mitad correspondiente.

Convenciones:
  * match_id del tracking es `<root>__h<1|2>` (sintetico por mitad) para
    evitar colisionar frames con el PK de tracking_events. Configurable
    via `match_id_strategy`.
  * Las coordenadas de SoccerNet son normalizadas en [0, 1] (default) o
    en metros (configurable). Se proyectan a metros via FieldDims.
  * No se descarga video.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .config import (
    DownloadConfig,
    LABEL_ALIASES,
    NormalizeConfig,
    TARGET_LABELS,
)
from .io_utils import info, stage, warn, write_parquet
from .soccernet_io import (
    GameDir,
    GameDirMot,
    iter_games,
    iter_games_mot,
    read_labels,
    read_tracking_jsonl,
    read_tracking_mot,
)


# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------
# Tamano de imagen asumido para derivar pixeles a partir de la posicion
# normalizada. SoccerNet publica anotaciones sobre imagenes 1050x680 en el
# split de action spotting; tracking_challenge usa 1920x1080. Default 1050x680
# es seguro para labels-v2 (los pixeles son solo un proxy, los features ML
# trabajan en metros).
DEFAULT_IMG_W = 1050.0
DEFAULT_IMG_H = 680.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(x: Any, default: float | None = None) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _half_from_game_time(game_time: str) -> int:
    """'1 - 12:34' -> 1, '2 - 45:00' -> 2."""
    if not isinstance(game_time, str):
        return 0
    head = game_time.split("-", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return 0


def _resolve_label_type(raw: str) -> str:
    """Mapea 'Pass ', 'pass', 'Duel', etc. a la forma canonica."""
    return LABEL_ALIASES.get(raw, LABEL_ALIASES.get(raw.strip(), raw))


# ---------------------------------------------------------------------------
# Tracking -> filas atomicas
# ---------------------------------------------------------------------------
def _track_to_atomic_rows(
    game: GameDir,
    half: int,
    cfg: NormalizeConfig,
    img_w: float,
    img_h: float,
) -> list[dict[str, Any]]:
    """
    Convierte una mitad completa de tracking jsonl en filas atomicas.

    Cada linea SoccerNet se interpreta como una deteccion individual. El
    resultado es una lista de dicts con la MISMA forma que una fila
    de tracking_events. NO se deduplica (SoccerNet ya da una fila por
    objeto por frame, no por track).
    """
    track_path = game.tracking_paths[half]
    fps = cfg.fps

    rows: list[dict[str, Any]] = []
    for rec in read_tracking_jsonl(track_path):
        role = str(rec.get("role", "player"))
        jersey = rec.get("jersey")
        if jersey is None:
            # Si no hay jersey no podemos asociar el track
            continue
        track_id = int(jersey)
        if role == cfg.ball_role_marker:
            track_id = 0
            obj_type = "ball"
            team = "unknown"
        else:
            obj_type = "player"
            team = str(rec.get("team", "unknown"))
            if team not in ("home", "away"):
                team = "unknown"
            if team == "away":
                # Offset para evitar colision de jerseys entre equipos
                track_id = track_id + 100

        pos = rec.get("position") or rec.get("xy")
        if not pos or len(pos) < 2:
            continue
        nx = _safe_float(pos[0])
        ny = _safe_float(pos[1])
        if nx is None or ny is None:
            continue

        if cfg.tracking_normalized:
            x_norm = nx
            y_norm = ny
            x_m = nx * cfg.field_dims.length_m
            y_m = ny * cfg.field_dims.width_m
        else:
            x_norm = nx / cfg.field_dims.length_m
            y_norm = ny / cfg.field_dims.width_m
            x_m = nx
            y_m = ny
        # Pixeles proxy (no fiel al video, pero no nulos)
        x_px = x_norm * img_w
        y_px = y_norm * img_h

        frame = int(rec.get("frame", 0))
        visibility = str(rec.get("visibility", "visible")).lower()
        in_occl = visibility in ("invisible", "uncertain")
        confidence = _safe_float(rec.get("confidence"), default=1.0)

        rows.append({
            "match_id":      f"{game.match_id}__h{half}",
            "frame":         frame,
            "timestamp_ms":  int(round(frame * 1000.0 / fps)),
            "object_type":   obj_type,
            "track_id":      track_id,
            "team":          team,
            "x":             x_px,
            "y":             y_px,
            "x_norm":        x_norm,
            "y_norm":        y_norm,
            "x_m":           x_m,
            "y_m":           y_m,
            "confidence":    confidence,
            "in_occlusion":  in_occl,
            "source_chunk":  str(track_path),
        })
    return rows


# ---------------------------------------------------------------------------
# Labels -> filas de training labels
# ---------------------------------------------------------------------------
def _event_to_label_row(
    ev: dict[str, Any],
    game: GameDir,
    fps: float,
    tracking_index: dict[tuple[str, int], dict[int, tuple[float, float]]],
) -> dict[str, Any] | None:
    """
    Convierte un evento SoccerNet a una fila de event_training_labels.

    Soporta dos formatos de Labels-v2.json:
      1. Legacy con start/end/from/to (hipotetico).
      2. Real: gameTime, position (frame number), team.

    Devuelve None si el evento no califica como Pass/Duel/Foul o si faltan
    campos minimos (half, frame).
    """
    raw_label = ev.get("label", "")
    coarse = _resolve_label_type(str(raw_label))
    if coarse not in TARGET_LABELS:
        return None

    # --- Half ---
    start = ev.get("start") or {}
    half = int(start.get("half", 0) or _half_from_game_time(ev.get("gameTime", "")))

    # --- Frame ---
    # Formato legacy: start.frame
    # Formato real: position (string con numero de frame)
    frame = int(start.get("frame", 0) or 0)
    if frame <= 0:
        # Intentar desde "position" (formato real Labels-v2)
        pos_str = ev.get("position")
        if pos_str is not None:
            try:
                frame = int(pos_str)
            except (ValueError, TypeError):
                frame = 0
    if half not in (1, 2) or frame <= 0:
        return None

    team = str(ev.get("team", "unknown")).lower()
    if team not in ("home", "away"):
        team = "unknown"

    # --- Actor/Target (legacy format only) ---
    actor = ev.get("from") or {}
    target = ev.get("to") or ev.get("duel", {}).get("opponent") or {}
    actor_jersey = actor.get("jersey")
    target_jersey = target.get("jersey")
    actor_track = int(actor_jersey) if actor_jersey is not None else None
    target_track = int(target_jersey) if target_jersey is not None else None
    if team == "away":
        if actor_track is not None:
            actor_track += 100
        if target_track is not None:
            target_track += 100

    # --- Resolver posiciones desde el tracking ---
    sx = sy = None
    ex = ey = None
    mid = f"{game.match_id}__h{half}"

    # 1) Intentar con actor_track especifico
    actor_index = tracking_index.get((mid, frame), {})
    if actor_track is not None and actor_track in actor_index:
        sx, sy = actor_index[actor_track]

    # 2) Si no hay actor_track (formato real), usar el jugador mas cercano
    #    al centro del campo en ese frame como proxy del actor.
    if sx is None and actor_index:
        center_x = 52.5  # mitad del campo (105m)
        center_y = 34.0  # mitad del campo (68m)
        best_tid = None
        best_dist = float("inf")
        for tid, (mx, my) in actor_index.items():
            d = (mx - center_x) ** 2 + (my - center_y) ** 2
            if d < best_dist:
                best_dist = d
                best_tid = tid
        if best_tid is not None:
            sx, sy = actor_index[best_tid]
            actor_track = best_tid

    # 3) End position (legacy)
    end_block = ev.get("end") or {}
    end_frame = int(end_block.get("frame", frame) or frame)
    if end_frame != frame and actor_track is not None:
        end_pos = tracking_index.get((mid, end_frame), {}).get(actor_track, (None, None))
        ex, ey = end_pos

    if sx is None and ex is None:
        return None

    result = str(ev.get("result") or ev.get("subtype") or "").strip() or None
    visibility = str(ev.get("visibility", start.get("visibility", "visible"))).lower()

    return {
        "match_id":        game.match_id,         # root, NO suffix por mitad
        "half":            half,
        "frame":           frame,
        "timestamp_ms":    int(round(frame * 1000.0 / fps)),
        "fps":             fps,
        "event_type":      coarse,
        "event_label":     str(raw_label),
        "event_subtype":   str(ev.get("subtype") or ev.get("type") or "").strip() or None,
        "team":            team,
        "actor_track_id":  actor_track,
        "target_track_id": target_track,
        "start_x_m":       sx,
        "start_y_m":       sy,
        "end_x_m":         ex,
        "end_y_m":         ey,
        "visibility":      visibility,
        "result":          result,
        "source":          "soccernet",
        "raw":             ev,
    }


def _build_tracking_index(
    tracking_df: pd.DataFrame,
) -> dict[tuple[str, int], dict[int, tuple[float, float]]]:
    """
    Construye un indice: (match_id_synthetic, frame) -> {track_id: (x_m, y_m)}.

    Es costoso (N frames) pero se hace una vez por match. Sirve para que
    _event_to_label_row resuelva start_x_m/start_y_m rapido.
    """
    idx: dict[tuple[str, int], dict[int, tuple[float, float]]] = {}
    if tracking_df.empty:
        return idx
    for r in tracking_df.itertuples(index=False):
        if pd.isna(r.x_m) or pd.isna(r.y_m):
            continue
        key = (r.match_id, int(r.frame))
        idx.setdefault(key, {})[int(r.track_id)] = (float(r.x_m), float(r.y_m))
    return idx


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def normalize_one(
    game: GameDir,
    cfg: NormalizeConfig,
) -> dict[str, Any]:
    """
    Normaliza un unico partido. Devuelve un dict con paths escritos y counts.
    """
    out = Path(cfg.output_dir) / game.match_id
    out.mkdir(parents=True, exist_ok=True)

    img_w = DEFAULT_IMG_W
    img_h = DEFAULT_IMG_H

    # 1) Tracking por mitad
    all_track_rows: list[dict[str, Any]] = []
    for half in sorted(game.tracking_paths.keys()):
        rows = _track_to_atomic_rows(game, half, cfg, img_w, img_h)
        all_track_rows.extend(rows)
    tracking_df = pd.DataFrame(all_track_rows)
    if not tracking_df.empty:
        # Renombrar match_id con sufijo por mitad ya viene de _track_to_atomic_rows
        # Asegurar tipos
        for col in ("x", "y", "x_norm", "y_norm", "x_m", "y_m"):
            if col in tracking_df.columns:
                tracking_df[col] = pd.to_numeric(tracking_df[col], errors="coerce")
    tracking_path = out / "tracking.parquet"
    if not tracking_df.empty:
        write_parquet(tracking_df, tracking_path)

    # 2) Tracking index para resolver posiciones de los eventos
    tracking_index = _build_tracking_index(tracking_df)

    # 3) Labels (combinar combinado y por mitad si existen)
    label_files: list[Path] = []
    if game.labels_path is not None:
        label_files.append(game.labels_path)
    for half in sorted(game.half_labels.keys()):
        label_files.append(game.half_labels[half])

    label_rows: list[dict[str, Any]] = []
    for lp in label_files:
        for ev in read_labels(lp):
            row = _event_to_label_row(ev, game, cfg.fps, tracking_index)
            if row is not None:
                label_rows.append(row)
    labels_df = pd.DataFrame(label_rows)
    if not labels_df.empty:
        # Deduplicar por (match_id, half, frame, event_type) quedandonos con
        # la primera aparicion (en caso de duplicados entre combinado y halves).
        labels_df = labels_df.drop_duplicates(
            subset=["match_id", "half", "frame", "event_type"], keep="first"
        )
    labels_path = out / "events.parquet"
    if not labels_df.empty:
        write_parquet(labels_df, labels_path)

    return {
        "match_id":      game.match_id,
        "tracking_rows": int(len(tracking_df)),
        "label_rows":    int(len(labels_df)),
        "tracking_path": str(tracking_path) if not tracking_df.empty else None,
        "labels_path":   str(labels_path) if not labels_df.empty else None,
    }


def normalize_all(
    norm_cfg: NormalizeConfig,
    dl_cfg: DownloadConfig,
) -> list[dict[str, Any]]:
    """
    Itera todos los partidos que `dl_cfg` sepa localizar y los normaliza
    con `norm_cfg`. Devuelve una lista de summaries por partido.
    """
    summaries: list[dict[str, Any]] = []
    n = 0
    for game in iter_games(dl_cfg):
        if not game.has_minimum():
            warn(f"Skip {game.match_id}: no tiene tracking+labels minimo")
            continue
        s = normalize_one(game, norm_cfg)
        summaries.append(s)
        n += 1
        if n % 25 == 0:
            info(f"  {n} partidos normalizados...")
    info(f"Total partidos normalizados: {n}")
    return summaries


# ---------------------------------------------------------------------------
# MOT20 tracking -> filas atomicas
# ---------------------------------------------------------------------------
def _track_to_atomic_rows_mot(
    clip: GameDirMot,
    cfg: NormalizeConfig,
) -> list[dict[str, Any]]:
    """
    Convierte un clip MOT20 (gt.txt) en filas atomicas compatibles
    con tracking_events.

    El formato MOT20 da bounding boxes en pixeles del frame del video
    broadcast (1920x1080 tipicamente). Convertimos el centro del bbox
    a coordenadas normalizadas [0,1] y luego a metros via FieldDims.

    Convenciones:
      - track_id se conserva tal cual del MOT (no hay jersey/team info).
      - object_type es "player" para todos (MOT no distingue pelota).
      - team es "unknown" (sin info de equipos en MOT).
    """
    fps = cfg.fps
    img_w = float(clip.img_w)
    img_h = float(clip.img_h)

    rows: list[dict[str, Any]] = []
    for rec in read_tracking_mot(clip.mot_gt_path):
        frame = rec["frame"]
        track_id = rec["track_id"]
        conf = rec["confidence"]

        # Centro del bounding box
        cx = rec["bbox_x"] + rec["width"] / 2.0
        cy = rec["bbox_y"] + rec["height"] / 2.0

        # Normalizar a [0, 1] por dimensiones del frame
        x_norm = cx / img_w
        y_norm = cy / img_h

        # A metros via field_dims
        x_m = x_norm * cfg.field_dims.length_m
        y_m = y_norm * cfg.field_dims.width_m

        # Pixeles del frame (proxy, no del campo)
        x_px = cx
        y_px = cy

        rows.append({
            "match_id":      f"{clip.match_id}__h1",   # suffix por mitad (siempre h1 para clips MOT)
            "frame":         frame,
            "timestamp_ms":  int(round(frame * 1000.0 / fps)),
            "object_type":   "player",
            "track_id":      track_id,
            "team":          "unknown",
            "x":             x_px,
            "y":             y_px,
            "x_norm":        x_norm,
            "y_norm":        y_norm,
            "x_m":           x_m,
            "y_m":           y_m,
            "confidence":    conf,
            "in_occlusion":  False,
            "source_chunk":  str(clip.mot_gt_path),
        })
    return rows


def normalize_one_mot(
    clip: GameDirMot,
    cfg: NormalizeConfig,
) -> dict[str, Any]:
    """
    Normaliza un unico clip MOT. Devuelve un dict con paths escritos y counts.
    """
    out = Path(cfg.output_dir) / clip.match_id
    out.mkdir(parents=True, exist_ok=True)

    # 1) Tracking
    all_track_rows = _track_to_atomic_rows_mot(clip, cfg)
    tracking_df = pd.DataFrame(all_track_rows)
    if not tracking_df.empty:
        for col in ("x", "y", "x_norm", "y_norm", "x_m", "y_m"):
            if col in tracking_df.columns:
                tracking_df[col] = pd.to_numeric(tracking_df[col], errors="coerce")
    tracking_path = out / "tracking.parquet"
    if not tracking_df.empty:
        write_parquet(tracking_df, tracking_path)

    # 2) Labels (si hay match con Labels-v2.json)
    label_rows: list[dict[str, Any]] = []
    if clip.labels_path is not None and clip.labels_path.is_file():
        tracking_index = _build_tracking_index(tracking_df)
        # Crear un GameDir dummy para reusar _event_to_label_row
        # El match_id del GameDir debe ser el root (sin __h suffix)
        # porque _event_to_label_row agrega __h{half} internamente.
        dummy_game = GameDir(
            match_id=clip.match_id,  # root sin suffix
            path=clip.path,
            labels_path=clip.labels_path,
            half_labels={},
            tracking_paths={},
        )
        for ev in read_labels(clip.labels_path):
            row = _event_to_label_row(ev, dummy_game, cfg.fps, tracking_index)
            if row is not None:
                label_rows.append(row)

    labels_df = pd.DataFrame(label_rows)
    if not labels_df.empty:
        labels_df = labels_df.drop_duplicates(
            subset=["match_id", "half", "frame", "event_type"], keep="first"
        )
    labels_path = out / "events.parquet"
    if not labels_df.empty:
        write_parquet(labels_df, labels_path)

    return {
        "match_id":      clip.match_id,
        "tracking_rows": int(len(tracking_df)),
        "label_rows":    int(len(labels_df)),
        "tracking_path": str(tracking_path) if not tracking_df.empty else None,
        "labels_path":   str(labels_path) if not labels_df.empty else None,
    }


def normalize_all_mot(
    norm_cfg: NormalizeConfig,
    dl_cfg: DownloadConfig,
) -> list[dict[str, Any]]:
    """
    Itera todos los clips MOT que `dl_cfg` sepa localizar y los normaliza
    con `norm_cfg`. Devuelve una lista de summaries por clip.
    """
    summaries: list[dict[str, Any]] = []
    n = 0
    for clip in iter_games_mot(dl_cfg):
        if not clip.has_minimum():
            warn(f"Skip {clip.match_id}: no tiene gt.txt")
            continue
        s = normalize_one_mot(clip, norm_cfg)
        summaries.append(s)
        n += 1
        if n % 50 == 0:
            info(f"  {n} clips normalizados...")
    info(f"Total clips MOT normalizados: {n}")
    return summaries


# ---------------------------------------------------------------------------
# Ingesta a la DB (opcional)
# ---------------------------------------------------------------------------
def ingest_to_db(
    normalized_dir: Path,
    match_id_filter: str | None = None,
) -> dict[str, int]:
    """
    Lee los Parquet de `normalized_dir` y los inserta en tracking_events
    y event_training_labels. Idempotente (ON CONFLICT en tracking_events).

    Devuelve conteos {tracking_events: N, event_training_labels: M}.
    """
    try:
        import psycopg  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Para ingestar a la DB hace falta psycopg. "
            "Instalar requirements-db.txt."
        ) from e

    from .config import NormalizeConfig  # noqa: F401
    from scripts.ingest import (
        TrackingRow,
        STAGING_DDL,
        STAGING_INSERT_SQL,
        bulk_insert_rows,
        build_dsn_from_env,
        frame_to_ms,
    )

    dsn = build_dsn_from_env()
    counts = {"tracking_events": 0, "event_training_labels": 0, "matches": 0}

    base = Path(normalized_dir)
    game_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if match_id_filter is not None:
        game_dirs = [d for d in game_dirs if d.name == match_id_filter]
        if not game_dirs:
            raise FileNotFoundError(f"No se encontro {match_id_filter} en {base}")

    etl_insert = """
        INSERT INTO event_training_labels (
            match_id, half, frame, timestamp_ms, fps,
            event_type, event_label, event_subtype, team,
            actor_track_id, target_track_id,
            start_x_m, start_y_m, end_x_m, end_y_m,
            visibility, result, source, raw
        ) VALUES (
            %(match_id)s, %(half)s, %(frame)s, %(timestamp_ms)s, %(fps)s,
            %(event_type)s, %(event_label)s, %(event_subtype)s, %(team)s,
            %(actor_track_id)s, %(target_track_id)s,
            %(start_x_m)s, %(start_y_m)s, %(end_x_m)s, %(end_y_m)s,
            %(visibility)s, %(result)s, %(source)s, %(raw)s::jsonb
        )
        ON CONFLICT (match_id, label_id) DO NOTHING;
    """

    with psycopg.connect(dsn, autocommit=False) as conn:
        for gd in game_dirs:
            info(f"Ingesting {gd.name} ...")
            tp = gd / "tracking.parquet"
            ep = gd / "events.parquet"
            n_match = 0
            if tp.exists():
                tdf = pd.read_parquet(tp)
                rows: list[TrackingRow] = []
                for r in tdf.itertuples(index=False):
                    rows.append(
                        TrackingRow(
                            match_id=str(r.match_id),
                            frame=int(r.frame),
                            timestamp_ms=int(r.timestamp_ms),
                            object_type=str(r.object_type),
                            track_id=int(r.track_id),
                            team=str(r.team),
                            x=float(r.x),
                            y=float(r.y),
                            bbox_x1=None, bbox_y1=None,
                            bbox_x2=None, bbox_y2=None,
                            x_norm=float(r.x_norm) if not pd.isna(r.x_norm) else None,
                            y_norm=float(r.y_norm) if not pd.isna(r.y_norm) else None,
                            x_m=float(r.x_m) if not pd.isna(r.x_m) else None,
                            y_m=float(r.y_m) if not pd.isna(r.y_m) else None,
                            confidence=float(r.confidence) if not pd.isna(r.confidence) else None,
                            in_occlusion=bool(r.in_occlusion),
                            source_chunk=str(r.source_chunk),
                        )
                    )
                with conn.cursor() as cur:
                    for i in range(0, len(rows), 5_000):
                        batch = rows[i:i + 5_000]
                        bulk_insert_rows(cur, batch)
                        counts["tracking_events"] += len(batch)
                conn.commit()
                info(f"  tracking_events: +{len(rows)} filas")
            if ep.exists():
                edf = pd.read_parquet(ep)
                with conn.cursor() as cur:
                    for r in edf.itertuples(index=False):
                        cur.execute(
                            etl_insert,
                            {
                                "match_id":        r.match_id,
                                "half":            int(r.half),
                                "frame":           int(r.frame),
                                "timestamp_ms":    int(r.timestamp_ms),
                                "fps":             float(r.fps),
                                "event_type":      r.event_type,
                                "event_label":     r.event_label,
                                "event_subtype":   r.event_subtype,
                                "team":            r.team,
                                "actor_track_id":  int(r.actor_track_id) if not pd.isna(r.actor_track_id) else None,
                                "target_track_id": int(r.target_track_id) if not pd.isna(r.target_track_id) else None,
                                "start_x_m":       float(r.start_x_m) if not pd.isna(r.start_x_m) else None,
                                "start_y_m":       float(r.start_y_m) if not pd.isna(r.start_y_m) else None,
                                "end_x_m":         float(r.end_x_m) if not pd.isna(r.end_x_m) else None,
                                "end_y_m":         float(r.end_y_m) if not pd.isna(r.end_y_m) else None,
                                "visibility":      r.visibility,
                                "result":          r.result,
                                "source":          r.source,
                                "raw":             json.dumps(r.raw, ensure_ascii=False),
                            },
                        )
                        counts["event_training_labels"] += 1
                conn.commit()
                info(f"  event_training_labels: +{len(edf)} filas")
            n_match += 1
        counts["matches"] = n_match
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Normalize SoccerNet -> tracking_events")
    p.add_argument("--output-dir", default="ETAPA4_data/normalized",
                   help="Directorio de salida (Parquet por partido).")
    p.add_argument("--raw-dir", default=None,
                   help="Carpeta raiz con SoccerNet ya bajado (omite download).")
    p.add_argument("--ingest", action="store_true",
                   help="Ademas de normalizar, ingestar en la DB.")
    p.add_argument("--match-id", default=None,
                   help="Filtrar ingesta a un solo match_id.")
    p.add_argument("--tracking-format", default="mot", choices=["mot", "jsonl"],
                   help="Formato de tracking: 'mot' (MOT20 CSV) o 'jsonl' (legacy).")
    args = p.parse_args(argv)

    from .config import DownloadConfig, FieldDims, NormalizeConfig
    norm_cfg = NormalizeConfig(
        field_dims=FieldDims(),
        output_dir=Path(args.output_dir),
    )
    dl_cfg = DownloadConfig(
        raw_dir=Path(args.raw_dir) if args.raw_dir else None,
        tracking_format=args.tracking_format,
    )

    with stage("normalize"):
        if args.tracking_format == "mot":
            summaries = normalize_all_mot(norm_cfg, dl_cfg)
        else:
            summaries = normalize_all(norm_cfg, dl_cfg)

    if args.ingest:
        with stage("ingest-to-db"):
            counts = ingest_to_db(Path(args.output_dir), match_id_filter=args.match_id)
            info(f"DB insert counts: {counts}")
    return 0
