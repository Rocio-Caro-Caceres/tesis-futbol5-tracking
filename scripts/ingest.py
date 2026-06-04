"""
ingest.py
=========

Ingesta idempotente de chunks JSON a PostgreSQL con TimescaleDB + PostGIS.

Estrategia de carga (la mas rapida posible sin perder idempotencia):

  1. Para cada chunks/<match_id>/chunk_*.json:
     a) Calcula SHA256 y consulta loaded_chunks. Si ya esta cargado
        con el mismo hash, se saltea.
     b) Parsea el JSON y aplana las listas 'tracking' y 'ball' en
        filas unicas (object_type = 'player' | 'ball').
     c) Inserta en bloque via COPY a una TEMP TABLE de staging
        (la geometria PostGIS se materializa en el INSERT final
        a partir de x_m / y_m con ST_SetSRID(ST_MakePoint, 3857)).
     d) INSERT INTO tracking_events ... ON CONFLICT (...) DO UPDATE
        para que re-correr el script no duplique filas.
     e) Registra el chunk en loaded_chunks.

  2. Para cada match.json:
        UPSERT en match_summary.

Uso:
    python -m scripts.ingest --match-id partido_f5
    python -m scripts.ingest --chunks-dir chunks --match-id partido_f5
    python -m scripts.ingest --all
    python -m scripts.ingest --rebuild-tracking    # wipe tracking_events

Variables de entorno (o ver db/.env):
    PGHOST (default localhost)
    PGPORT (default 5432)
    PGDATABASE (default futbol5)
    PGUSER     (default futbol5)
    PGPASSWORD (default futbol5)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import psycopg
from psycopg import sql


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 5_000  # filas por COPY batch en memoria

STAGING_DDL = """
    DROP TABLE IF EXISTS pg_temp.tracking_staging;
    CREATE TEMP TABLE tracking_staging (
        match_id      TEXT,
        frame         BIGINT,
        timestamp_ms  BIGINT,
        object_type   TEXT,
        track_id      INTEGER,
        team          TEXT,
        x             DOUBLE PRECISION,
        y             DOUBLE PRECISION,
        bbox_x1       INTEGER,
        bbox_y1       INTEGER,
        bbox_x2       INTEGER,
        bbox_y2       INTEGER,
        x_norm        DOUBLE PRECISION,
        y_norm        DOUBLE PRECISION,
        x_m           DOUBLE PRECISION,
        y_m           DOUBLE PRECISION,
        confidence    DOUBLE PRECISION,
        in_occlusion  BOOLEAN,
        source_chunk  TEXT
    ) ON COMMIT DROP;
"""

STAGING_INSERT_SQL = """
    INSERT INTO tracking_events (
        match_id, frame, timestamp_ms, object_type, track_id, team,
        x, y, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        x_norm, y_norm, x_m, y_m,
        confidence, in_occlusion, source_chunk,
        geom
    )
    SELECT
        s.match_id, s.frame, s.timestamp_ms, s.object_type, s.track_id, s.team::team_side,
        s.x, s.y, s.bbox_x1, s.bbox_y1, s.bbox_x2, s.bbox_y2,
        s.x_norm, s.y_norm, s.x_m, s.y_m,
        s.confidence, s.in_occlusion, s.source_chunk,
        CASE WHEN s.x_m IS NULL OR s.y_m IS NULL
             THEN NULL
             ELSE ST_SetSRID(ST_MakePoint(s.x_m, s.y_m), 3857)
        END
    FROM tracking_staging s
    ON CONFLICT (match_id, frame, object_type, track_id) DO UPDATE SET
        timestamp_ms = EXCLUDED.timestamp_ms,
        team         = EXCLUDED.team,
        x            = EXCLUDED.x,
        y            = EXCLUDED.y,
        bbox_x1      = EXCLUDED.bbox_x1,
        bbox_y1      = EXCLUDED.bbox_y1,
        bbox_x2      = EXCLUDED.bbox_x2,
        bbox_y2      = EXCLUDED.bbox_y2,
        x_norm       = EXCLUDED.x_norm,
        y_norm       = EXCLUDED.y_norm,
        x_m          = EXCLUDED.x_m,
        y_m          = EXCLUDED.y_m,
        geom         = EXCLUDED.geom,
        confidence   = EXCLUDED.confidence,
        in_occlusion = EXCLUDED.in_occlusion,
        source_chunk = EXCLUDED.source_chunk;
"""


# ---------------------------------------------------------------------------
# Modelo de fila
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TrackingRow:
    match_id:     str
    frame:        int
    timestamp_ms: int
    object_type:  str
    track_id:     int
    team:         str
    x:            float
    y:            float
    bbox_x1:      int | None
    bbox_y1:      int | None
    bbox_x2:      int | None
    bbox_y2:      int | None
    x_norm:       float | None
    y_norm:       float | None
    x_m:          float | None
    y_m:          float | None
    confidence:   float | None
    in_occlusion: bool
    source_chunk: str

    def as_tuple(self) -> tuple:
        return (
            self.match_id,
            self.frame,
            self.timestamp_ms,
            self.object_type,
            self.track_id,
            self.team,
            self.x,
            self.y,
            self.bbox_x1,
            self.bbox_y1,
            self.bbox_x2,
            self.bbox_y2,
            self.x_norm,
            self.y_norm,
            self.x_m,
            self.y_m,
            self.confidence,
            self.in_occlusion,
            self.source_chunk,
        )


def frame_to_ms(frame: int, fps: float) -> int:
    if not fps or fps <= 0:
        return int(frame) * 41  # ~24 fps fallback
    return int(round(frame * 1000.0 / fps))


def iter_chunk_rows(chunk_path: Path, match_id_fallback: str) -> Iterator[TrackingRow]:
    """Genera filas atomicas a partir de un chunk JSON."""
    with open(chunk_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    match_id = data.get("match_id") or match_id_fallback
    md = data.get("metadata", {}) or {}
    fps = float(md.get("fps") or 0.0)
    source = str(chunk_path)

    for rec in data.get("tracking", []) or []:
        frame = int(rec["frame"])
        yield TrackingRow(
            match_id=match_id,
            frame=frame,
            timestamp_ms=frame_to_ms(frame, fps),
            object_type=str(rec.get("object_type", "player")),
            track_id=int(rec.get("track_id", 0)),
            team=str(rec.get("team", "unknown")),
            x=float(rec["x"]),
            y=float(rec["y"]),
            bbox_x1=_to_int_or_none(rec.get("bbox_x1")),
            bbox_y1=_to_int_or_none(rec.get("bbox_y1")),
            bbox_x2=_to_int_or_none(rec.get("bbox_x2")),
            bbox_y2=_to_int_or_none(rec.get("bbox_y2")),
            x_norm=_to_float_or_none(rec.get("x_norm")),
            y_norm=_to_float_or_none(rec.get("y_norm")),
            x_m=_to_float_or_none(rec.get("x_m")),
            y_m=_to_float_or_none(rec.get("y_m")),
            confidence=_to_float_or_none(rec.get("confidence")),
            in_occlusion=bool(rec.get("in_occlusion", False)),
            source_chunk=source,
        )

    for rec in data.get("ball", []) or []:
        frame = int(rec["frame"])
        yield TrackingRow(
            match_id=match_id,
            frame=frame,
            timestamp_ms=frame_to_ms(frame, fps),
            object_type="ball",
            track_id=0,
            team="unknown",
            x=float(rec["x"]),
            y=float(rec["y"]),
            bbox_x1=None,
            bbox_y1=None,
            bbox_x2=None,
            bbox_y2=None,
            x_norm=_to_float_or_none(rec.get("x_norm")),
            y_norm=_to_float_or_none(rec.get("y_norm")),
            x_m=_to_float_or_none(rec.get("x_m")),
            y_m=_to_float_or_none(rec.get("y_m")),
            confidence=_to_float_or_none(rec.get("confidence")),
            in_occlusion=False,
            source_chunk=source,
        )


def _to_int_or_none(v):
    return None if v is None else int(v)


def _to_float_or_none(v):
    return None if v is None else float(v)


# ---------------------------------------------------------------------------
# Ingesta
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_chunk_already_loaded(cur, match_id: str, chunk_path: str, sha: str) -> bool:
    cur.execute(
        "SELECT chunk_sha256 FROM loaded_chunks WHERE match_id = %s AND chunk_path = %s",
        (match_id, chunk_path),
    )
    row = cur.fetchone()
    if row is None:
        return False
    return row[0] == sha


def bulk_insert_rows(cur, rows: list[TrackingRow]) -> int:
    """Inserta via COPY -> staging -> INSERT ... ON CONFLICT. Devuelve filas afectadas."""
    if not rows:
        return 0
    cur.execute(STAGING_DDL)
    with cur.copy("COPY tracking_staging FROM STDIN") as copy:
        for r in rows:
            copy.write_row(r.as_tuple())
    cur.execute(STAGING_INSERT_SQL)
    return cur.rowcount or 0


def upsert_match_summary(cur, match_path: Path) -> None:
    with open(match_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    homography = m.get("homography")
    cur.execute(
        """
        INSERT INTO match_summary (
            match_id, home_team, away_team, match_date, venue,
            home_score, away_score, duration_seconds, fps, width, height,
            field_length_m, field_width_m, homography, notes
        ) VALUES (
            %(match_id)s, %(home_team)s, %(away_team)s, %(match_date)s, %(venue)s,
            %(home_score)s, %(away_score)s, %(duration_seconds)s, %(fps)s, %(width)s, %(height)s,
            %(field_length_m)s, %(field_width_m)s, %(homography)s, %(notes)s
        )
        ON CONFLICT (match_id) DO UPDATE SET
            home_team        = EXCLUDED.home_team,
            away_team        = EXCLUDED.away_team,
            match_date       = EXCLUDED.match_date,
            venue            = EXCLUDED.venue,
            home_score       = EXCLUDED.home_score,
            away_score       = EXCLUDED.away_score,
            duration_seconds = EXCLUDED.duration_seconds,
            fps              = EXCLUDED.fps,
            width            = EXCLUDED.width,
            height           = EXCLUDED.height,
            field_length_m   = EXCLUDED.field_length_m,
            field_width_m    = EXCLUDED.field_width_m,
            homography       = EXCLUDED.homography,
            notes            = EXCLUDED.notes,
            updated_at       = NOW();
        """,
        {
            "match_id":        m.get("match_id"),
            "home_team":       m.get("home_team"),
            "away_team":       m.get("away_team"),
            "match_date":      m.get("match_date"),
            "venue":           m.get("venue"),
            "home_score":      m.get("home_score", 0),
            "away_score":      m.get("away_score", 0),
            "duration_seconds": m.get("duration_seconds"),
            "fps":             m.get("fps"),
            "width":           m.get("width"),
            "height":          m.get("height"),
            "field_length_m":  m.get("field_length_m", 105.0),
            "field_width_m":   m.get("field_width_m", 68.0),
            "homography":      homography,
            "notes":           m.get("notes"),
        },
    )


def ingest_match(conn, match_dir: Path) -> dict:
    """Procesa todos los chunks de un match_dir. Devuelve resumen."""
    summary = {
        "match_id": match_dir.name,
        "chunks_total": 0,
        "chunks_skipped": 0,
        "chunks_loaded": 0,
        "rows_loaded": 0,
        "errors": [],
    }
    if not match_dir.is_dir():
        return summary

    # match.json primero
    match_path = match_dir / "match.json"
    with conn.cursor() as cur:
        if match_path.exists():
            try:
                upsert_match_summary(cur, match_path)
                print(f"  [match.json] OK -> {match_path}")
            except Exception as e:
                summary["errors"].append(("match.json", str(e)))
                conn.rollback()
                print(f"  [match.json] ERROR: {e}")
                return summary
        else:
            print(f"  [match.json] no encontrado en {match_dir}, se omite")
        conn.commit()

    # Chunks
    chunks = sorted(match_dir.glob("chunk_*.json"))
    summary["chunks_total"] = len(chunks)
    print(f"  Encontrados {len(chunks)} chunks en {match_dir}")

    for chunk_path in chunks:
        sha = sha256_of_file(chunk_path)
        try:
            with conn.cursor() as cur:
                if is_chunk_already_loaded(cur, match_dir.name, str(chunk_path), sha):
                    print(f"  - {chunk_path.name}: ya cargado (mismo sha256), skip")
                    summary["chunks_skipped"] += 1
                    conn.commit()
                    continue

                # Materializar todas las filas del chunk en memoria (suficiente
                # para 150 frames * 14 objetos ~= 2.1k filas, no es problema)
                rows: list[TrackingRow] = list(
                    iter_chunk_rows(chunk_path, match_id_fallback=match_dir.name)
                )

                # Bulk insert en batches
                total = 0
                for i in range(0, len(rows), BATCH_SIZE):
                    batch = rows[i : i + BATCH_SIZE]
                    total += bulk_insert_rows(cur, batch)

                # Registrar
                cur.execute(
                    """
                    INSERT INTO loaded_chunks (match_id, chunk_path, chunk_sha256, row_count)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (match_id, chunk_path) DO UPDATE
                    SET chunk_sha256 = EXCLUDED.chunk_sha256,
                        row_count    = EXCLUDED.row_count,
                        loaded_at    = NOW();
                    """,
                    (match_dir.name, str(chunk_path), sha, total),
                )
            conn.commit()
            print(f"  + {chunk_path.name}: {total} filas (sha {sha[:8]}...)")
            summary["chunks_loaded"] += 1
            summary["rows_loaded"] += total
        except Exception as e:
            conn.rollback()
            msg = f"{chunk_path}: {e}"
            summary["errors"].append((chunk_path.name, str(e)))
            print(f"  ! {chunk_path.name}: ERROR {e}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_dsn_from_env() -> str:
    return psycopg.conninfo.make_conninfo(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "futbol5"),
        user=os.environ.get("PGUSER", "futbol5"),
        password=os.environ.get("PGPASSWORD", "futbol5"),
    )


def cmd_rebuild_tracking(conn) -> None:
    print("ATENCION: esto borra TODAS las filas de tracking_events y loaded_chunks.")
    resp = input("Continuar? [si/no]: ").strip().lower()
    if resp not in ("si", "s", "yes", "y"):
        print("Cancelado.")
        return
    with conn.cursor() as cur:
        cur.execute("TRUNCATE tracking_events, loaded_chunks;")
    conn.commit()
    print("OK.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ingesta chunks JSON a PostgreSQL")
    p.add_argument("--chunks-dir", default="chunks",
                   help="Directorio raiz con subdirs por match_id")
    p.add_argument("--match-id", default=None,
                   help="Si se pasa, procesa solo chunks/<chunks-dir>/<match-id>")
    p.add_argument("--all", action="store_true",
                   help="Procesa todos los match_id bajo chunks-dir")
    p.add_argument("--rebuild-tracking", action="store_true",
                   help="Borra tracking_events y loaded_chunks (USAR CON CUIDADO)")
    args = p.parse_args(argv)

    dsn = build_dsn_from_env()
    print(f"Conectando a {os.environ.get('PGHOST', 'localhost')}:"
          f"{os.environ.get('PGPORT', '5432')}/"
          f"{os.environ.get('PGDATABASE', 'futbol5')}")

    with psycopg.connect(dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension ORDER BY 1;")
            exts = [r[0] for r in cur.fetchall()]
            print(f"Extensiones instaladas: {exts}")
            if "timescaledb" not in exts or "postgis" not in exts:
                print("Faltan extensiones timescaledb/postgis. Revisar db/sql/.")
                return 2

        if args.rebuild_tracking:
            cmd_rebuild_tracking(conn)
            return 0

        chunks_root = Path(args.chunks_dir)
        if not chunks_root.exists():
            print(f"No existe el directorio {chunks_root.resolve()}")
            return 1

        if args.match_id:
            match_dirs = [chunks_root / args.match_id]
        elif args.all:
            match_dirs = sorted(d for d in chunks_root.iterdir() if d.is_dir())
        else:
            print("Especificar --match-id <id> o --all")
            return 1

        total_summary = {
            "matches": 0,
            "chunks_loaded": 0,
            "chunks_skipped": 0,
            "rows_loaded": 0,
        }
        for md in match_dirs:
            print(f"\n== {md.name} ==")
            s = ingest_match(conn, md)
            total_summary["matches"] += 1
            total_summary["chunks_loaded"] += s["chunks_loaded"]
            total_summary["chunks_skipped"] += s["chunks_skipped"]
            total_summary["rows_loaded"] += s["rows_loaded"]

        print("\n========== RESUMEN INGESTA ==========")
        print(f"Matches procesados:  {total_summary['matches']}")
        print(f"Chunks cargados:     {total_summary['chunks_loaded']}")
        print(f"Chunks saltados:     {total_summary['chunks_skipped']}")
        print(f"Filas totales:       {total_summary['rows_loaded']}")
        print("=====================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
