# AGENTS.md

Compact reference for OpenCode sessions on this Windows-first tesis repo
(futbol-5 player tracking + SoccerNet ML).

## Stack & env

- Python 3.12, Windows 10/11 (also reachable from WSL at `/mnt/c/...`).
- NVIDIA + CUDA 12.1: `tracking.py` runs YOLOv8m (needs GPU, auto-detects
  and prints which it picked). `METRICAS` and `ETAPA4` are pure
  NumPy/PyTorch and CPU-friendly.
- Branch: `fran` (active); `origin/HEAD` → `develop`.
- **No pytest/ruff/mypy/CI.**
- `requirements.txt` is **UTF-16 encoded** — some tools choke on it.
  Use `pip install -r requirements.txt` directly (works fine).
- `LaurieOnTracking repo/` is read-only reference code. Do not modify.
- **README is stale** — only documents Etapa 1.

## Routing (which file owns what)

| Task | Edit |
|------|------|
| Tracker / re-id / video input | `tracking.py` (top-level) |
| DB schema, ingest idempotency | `scripts/ingest.py`, `db/sql/*.sql` |
| Smoothing, kinematics, distance, heatmap, possession, events | `METRICAS/` |
| SoccerNet download / normalize / features / train / bench | `ETAPA4/` |

## Data flows (4 of them)

`tracking.py` writes JSON chunks + annotated video. The other flows
are offline:

1. **JSON chunks** (`chunks/<MATCH_ID>/chunk_NNNNN.json` + `match.json`)
   — primary output. No legacy CSV is written (removed).
2. **JSON chunks → PostgreSQL** (Etapa 1): `scripts.ingest.py` →
   TimescaleDB + PostGIS.
3. **METRICAS** (Etapa 2 + 3, offline): `python -m METRICAS --input ...`
   → `players_metrics.csv`, `summary_*.json`, heatmaps, possession/events.
4. **SoccerNet ML** (Etapa 4, optional GPU): `python -m ETAPA4 download
   --normalize --features --train-xgb/lstm/benchmark`.

**CRITICAL GAP**: There is no converter from JSON chunks (flow 1) to
CSV that METRICAS (flow 3) can consume. This must be built before
METRICAS can run on real tracking data. See `issues.md` #10.

When asked to "add tracking output", ask which flow. For ML on tracking
data prefer flow 4.

## Layout quirks

- `tracking.py` (top-level) is the **entrypoint**; `tracking/` (package)
  contains only Etapa 1 helpers (`chunk_writer.py`, `homography.py`).
  Both names are intentional, do not rename one to dodge the collision.
- `scripts/` is a package: `python -m scripts.ingest`, `python -m
  scripts.verify_db`.
- `METRICAS/` and `ETAPA4/` are UPPERCASE packages — lowercase would
  shadow the module name.
- `db/sql/00_extensions.sql` MUST sort before `01_schema.sql`
  (alphabetical ordering in `docker-entrypoint-initdb.d`). `db/sql/`
  is bind-mounted; edits don't need a rebuild but only run on the
  FIRST init of `db_data`.

## Hardcoded knobs that bite

- **`tracking.py` (no CLI flags, edit the file):**
  - `video_path` line 58 — hardcoded Windows path; must edit to run.
  - `MINUTO_INICIO = 20` line 71 — skips first 20 min; lower for short test clips.
  - `MAX_JUGADORES = 13`, `DIST_MAX_REID = 200`, `IOU_OCLUSION = 0.4`
    lines 93/102/106 — tuned for the test video's camera angle, don't
    change without asking.
  - `CHUNK_FRAMES = 150`, `MATCH_ID = 'partido_f5'`, `TEAM_BY_ID = {}`
    lines 112/114/118. **Team detection not implemented**; everything
    writes `'unknown'`.
- **`scripts/ingest.py`:** DB defaults `PGHOST=localhost`, port `5432`,
  db/user/pass `futbol5`. Override via env or gitignored `db/.env`.
  `--rebuild-tracking` is destructive (truncates `tracking_events` +
  `loaded_chunks`); asks for typed confirmation, don't bypass.
- **`METRICAS/`:**
  - `FieldDims(length_m=105.0, width_m=68.0)` is FIFA 11-a-side
    default (README claims futsal 42×25 — that's stale, pass
    `FieldDims(42, 25)` explicitly for futsal).
  - `DEFAULT_MAX_SPEED_MPS = 12.0` in `kinematics.py:40` (Laurie cap;
    set to 0 to disable).
  - `DEFAULT_MIN_STEP_M = 0.03` in `distance.py:22` (static-player
    threshold).
  - `PipelineConfig.field_dims` / `EventsConfig.field_dims` are the
    canonical attributes (`.field` is a property alias kept for compat).
- **`ETAPA4/` (see `ETAPA4/config.py`):**
  - `DownloadConfig.password = "s0cc3rn3t"` (SoccerNet academic default).
  - `NormalizeConfig.fps = 25.0`, `tracking_normalized = True`
    (SoccerNet positions are in [0, 1]).
  - `FeaturesConfig.window_frames = 25`, `top_k_neighbors = 4`,
    `smooth_window = 3`.
  - `TrainConfig.lstm_hidden = 128`, `lstm_layers = 2`, `dropout = 0.3`,
    `lr = 1e-3`, `epochs = 30`, `patience = 5`.

## Database

- `docker compose down` keeps the volume; SQL init does NOT re-run.
  To re-apply schema: `docker compose down -v` (wipes data) or
  `python -m scripts.ingest --rebuild-tracking` (tracking only).
- For a brand new SQL file after first init: `psql -f db/sql/<NN>_<name>.sql`.
- `tracking_events` PK = `(match_id, frame, object_type, track_id)`;
  ingest is idempotent via `ON CONFLICT ... DO UPDATE`.
- `geom` (PostGIS) is built at INSERT from `x_m`/`y_m` via
  `ST_SetSRID(ST_MakePoint(x_m, y_m), 3857)`. Stays NULL when
  `x_m`/`y_m` are NULL (homography not yet calibrated).
- `tracking/homography.py` uses identity (pixel/frame size normalized),
  correct for layout but not real-world meters.
- **Schema migration pattern**: `SCHEMA_VERSION` lives in
  `tracking/chunk_writer.py`. Bump it AND add a branch in
  `scripts/ingest.py:iter_chunk_rows` when changing the JSON chunk
  shape. Version is stored in each chunk's `schema_version` field.

## Conventions that bite

### METRICAS

- `kloppy` is **optional** — `metrica_loader._require_kloppy()` raises
  a clear `ImportError` if missing. `pip install -r
  requirements-metricas.txt` installs it, but no module requires it.
- **Heatmap is `(ny, nx)`** — rows = Y, cols = X. `build_heatmap`
  transposes `np.histogram2d`. Do not "fix".
- **Central moving average has NaN at the edges** (`window//2` on each
  side). Central-difference velocity/acceleration are NaN at track
  start/end. `cumulative_distance` treats those NaN steps as 0 so
  `cumdist_m` is always finite.
- **Ball identification**: `track_id == 0` OR `object_type == 'ball'`.
  `compute_possession` / `detect_events` auto-detect via the second
  convention; they also auto-compute `speed_mps`/`accel_mps2` on
  pre-kinematics data.
- **Possession FSM** needs `frame, x_m, y_m` for players and ball plus
  any `team` string.
- **Save detection** needs an identifiable goalkeeper: pass
  `goalkeeper_track_ids={5, 7}` (or `--keeper-ids 5 7`), or
  `goalkeeper_team="away"`. Without one, no saves fire.
- **NaN in JSON output**: `summary_match.json` and `ETAPA4/**/metrics.json`
  sanitize `NaN`/`Inf` → `null`. Don't bypass with `default=str` —
  produces invalid JSON.

### ETAPA4 (SoccerNet ML)

- **match_id is synthetic per half**: tracking goes to
  `soccernet_<slug>__h<1|2>` so the `tracking_events` PK doesn't
  collide on overlapping frame numbers between halves. Events keep
  the root `<slug>` + a `half` column. The features pipeline joins
  via `f"{root}__h{half}"`.
- **track_id = jersey, with away offset +100** to avoid home/away
  jersey collisions. Assumes jerseys < 100. See
  `ETAPA4/normalize._track_to_atomic_rows`.
- **Split by match, never by frame**: `train_xgb.split_by_match`
  shuffles match_id. Splitting by frame would leak the same play
  into train+val.
- **Duel/Foul targets may be NULL** — `to` is not guaranteed; the
  opponent lives in `duel.opponent` (`_event_to_label_row` checks
  both).
- **`vx_proxy`** is a per-track scalar pre-computed for
  `_neighbor_features`. Do not confuse with the actor's actual
  velocity, which comes from `_compute_kinematics` via central diffs.
- **No video guard**: `soccernet_io.download` raises if
  `DownloadConfig.files` contains `.mp4`. Keep the guard.
- **Features pipeline emits `mask_seq.npy`**: a per-frame boolean mask
  `(N, T)` tracking actor presence in each window. `train-lstm` and
  `benchmark` load it if present; fall back to `x.abs().sum() > 0` if
  missing (backward compat).
- New table `event_training_labels` lives in
  `db/sql/05_etapa4_labels.sql`; `ml_performance_logs` in
  `db/sql/06_ml_performance.sql`. Same `psql -f` rule as the
  other schema files.

## Dev commands (canonical order)

```powershell
# Etapas 1-3
cd db; copy .env.example .env; docker compose up -d
docker compose logs -f db   # wait for "database system is ready"
cd ..
pip install -r requirements.txt
pip install -r requirements-db.txt
pip install -r requirements-metricas.txt
python tracking.py
python -m scripts.ingest   --match-id partido_f5
python -m scripts.verify_db --match-id partido_f5
python -m METRICAS --input videos/datos_finales_tesis.csv --output-dir out/
python -m METRICAS --input ... --possession --events --keeper-ids 5 7

# Etapa 4 (separate, no DB / no GPU required)
pip install -r requirements-etapa4.txt
python -m ETAPA4 download   --output-dir data/soccernet --splits train
python -m ETAPA4 normalize  --raw-dir data/soccernet --output-dir ETAPA4_data/normalized [--ingest]
python -m ETAPA4 features   --normalized-dir ETAPA4_data/normalized --output-dir ETAPA4_data/features
python -m ETAPA4 train-xgb  --features-dir ETAPA4_data/features --output-dir ETAPA4_data/models
python -m ETAPA4 train-lstm --features-dir ETAPA4_data/features --output-dir ETAPA4_data/models
python -m ETAPA4 benchmark  --features-dir ETAPA4_data/features --output-dir ETAPA4_data/models
```

## WSL vs Windows

- Repo lives on Windows (`/mnt/c/...`); bundled `.venv/` is a Windows
  venv (`Scripts\python.exe`). From WSL, either use PowerShell on the
  Windows side, or create a fresh Linux venv: `python3 -m venv
  .venv-wsl` (don't commit it).
- `.gitignore` lists `venv/` (no dot) but the committed venv is
  `.venv/` — new files there WILL be tracked. Either extend gitignore
  or use `pip install -r requirements*.txt` outside `.venv`.
- `tracking.py` reads absolute Windows paths. WSL can read the
  codebase but not the video file. Run `tracking.py` from PowerShell.

## MVP scope

The MVP delivers 4 metrics from futsal video: **heatmap, possession,
distance traveled, goal events**. See `mvp.md` for full spec and
`issues.md` for the task breakdown (8 done, 4 pending).

The user's role is the "middle" pipeline: take CV tracking data →
compute metrics → output stats for web. The web platform and the CV
processing itself are handled by other people.

## Things you should NOT do

- Modify `LaurieOnTracking repo/`.
- Commit `db/.env`, `chunks/`, `videos/*.avi|mp4`, `db/data/`,
  `__pycache__/`, `ETAPA4_data/`, `data/soccernet/`, or `*.csv`. Use
  an output dir outside the repo (e.g. `out/`) for heatmap `.npz`
  files (not covered by `.gitignore`).
- Add pytest/ruff/mypy config from scratch — the project has none.
- Bypass `--rebuild-tracking` confirmation, or change `MAX_JUGADORES`
  / `DIST_MAX_REID` / `IOU_OCLUSION` without asking.


## Integración con cerebro

**Este proyecto está conectado con cerebro** (`C:\Users\Francisco\Documents\GitHub\cerebro`).

### Al completar una tarea
1. Crear accionable en: `C:\Users\Francisco\Documents\GitHub\cerebro\vault\accionables\`
   - Nombre: `accionable-[proyecto]-[YYYY-MM-DD]-[descripcion].md`
2. Actualizar conocimiento en: `C:\Users\Francisco\Documents\GitHub\cerebro\vault\knowledge\projects\[proyecto].md`
3. Si la tarea estaba en Tareas Activas de cerebro, marcarla como completada

### Formato del accionable
```markdown
# [Título]

- **Proyecto**: [nombre]
- **Fecha**: YYYY-MM-DD
- **Agent**: [nombre]
- **Estado**: completado

## Qué se hizo
[descripción]

## Aprendizajes
[qué aprendiste]

## Referencias
[archivos, links]
```

### Commando /guardar
Cuando el usuario diga "/guardar" o cuando completes algo significativo, ejecutá el protocolo de integración con cerebro automáticamente.
