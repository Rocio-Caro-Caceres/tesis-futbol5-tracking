"""
verify_db.py
============

Smoke test de la base luego de la ingesta. Imprime:
  - Extensiones instaladas (TimescaleDB / PostGIS)
  - Tablas de la etapa 1 y conteos
  - Que tracking_events sea hypertable y sus chunks
  - Que los indices compuestos (match_id, track_id, timestamp_ms) existan
  - Una consulta analitica de ejemplo: trayectorias por jugador en un partido

Uso:
    python -m scripts.verify_db
    python -m scripts.verify_db --match-id partido_f5
"""

from __future__ import annotations

import argparse
import os
import sys

import psycopg


def build_dsn_from_env() -> str:
    return psycopg.conninfo.make_conninfo(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "futbol5"),
        user=os.environ.get("PGUSER", "futbol5"),
        password=os.environ.get("PGPASSWORD", "futbol5"),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verifica el estado de la DB")
    p.add_argument("--match-id", default=None)
    args = p.parse_args(argv)

    dsn = build_dsn_from_env()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            # Extensiones
            cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY 1;")
            exts = cur.fetchall()
            print("== Extensiones ==")
            for name, ver in exts:
                flag = "OK" if name in ("timescaledb", "postgis") else "   "
                print(f"  [{flag}] {name} {ver}")
            if not any(n == "timescaledb" for n, _ in exts):
                print("  FALTA timescaledb")
            if not any(n == "postgis" for n, _ in exts):
                print("  FALTA postgis")

            # Tablas
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = [r[0] for r in cur.fetchall()]
            print("\n== Tablas (public) ==")
            for t in tables:
                cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(t)))
                n = cur.fetchone()[0]
                print(f"  {t}: {n} filas")

            # Hypertable check
            cur.execute("""
                SELECT hypertable_name, num_chunks
                FROM timescaledb_information.hypertables;
            """)
            print("\n== Hypertables ==")
            for name, n in cur.fetchall():
                print(f"  {name}: {n} chunks")

            # Indices
            cur.execute("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                AND tablename IN ('tracking_events','game_events','match_summary')
                ORDER BY tablename, indexname;
            """)
            print("\n== Indices (Etapa 1) ==")
            for name, ddl in cur.fetchall():
                print(f"  {name}")
                print(f"    {ddl}")

            # Query analitica de ejemplo
            match_filter = ""
            params: tuple = ()
            if args.match_id:
                match_filter = "WHERE match_id = %s"
                params = (args.match_id,)

            cur.execute(f"""
                SELECT
                    object_type,
                    track_id,
                    count(*)            AS n_rows,
                    min(timestamp_ms)   AS t0_ms,
                    max(timestamp_ms)   AS t1_ms,
                    avg(x_m)            AS cx_m,
                    avg(y_m)            AS cy_m
                FROM tracking_events
                {match_filter}
                GROUP BY object_type, track_id
                ORDER BY object_type, track_id;
            """, params)
            print("\n== Trayectorias (resumen) ==")
            print(f"  {'type':<7} {'id':<4} {'rows':<7} {'t0_ms':<10} {'t1_ms':<10} "
                  f"{'cx_m':<8} {'cy_m':<8}")
            for ot, tid, n, t0, t1, cx, cy in cur.fetchall():
                print(f"  {ot:<7} {tid:<4} {n:<7} {t0:<10} {t1:<10} "
                      f"{(cx or 0):<8.2f} {(cy or 0):<8.2f}")

            # Ejemplo PostGIS: distancias recorridas (suma de segmentos entre frames)
            if args.match_id:
                cur.execute("""
                    WITH pts AS (
                        SELECT
                            object_type, track_id, frame,
                            x_m, y_m,
                            LEAD(x_m) OVER w AS nx,
                            LEAD(y_m) OVER w AS ny
                        FROM tracking_events
                        WHERE match_id = %s
                        WINDOW w AS (PARTITION BY object_type, track_id
                                     ORDER BY frame)
                    )
                    SELECT object_type, track_id,
                           round(sum(sqrt(power(nx-x_m,2) + power(ny-y_m,2)))::numeric, 2)
                                AS distance_m
                    FROM pts WHERE nx IS NOT NULL
                    GROUP BY object_type, track_id
                    ORDER BY object_type, track_id;
                """, (args.match_id,))
                print("\n== Distancia recorrida (metros) ==")
                for ot, tid, d in cur.fetchall():
                    print(f"  {ot:<7} id={tid:<3} {d} m")

    return 0


if __name__ == "__main__":
    sys.exit(main())
