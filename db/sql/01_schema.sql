-- ============================================================================
-- 01_schema.sql
-- Esquema relacional atomico: una fila por objeto (jugador o pelota) por frame.
-- Idempotente: usa CREATE ... IF NOT EXISTS para que pueda re-ejecutarse.
-- ============================================================================

-- Extensiones: deben existir antes de cualquier CREATE TABLE que las use.
-- (Tambien las crea initdb-extensions.sh; este CREATE es idempotente.)
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;

-- Tipos enumerados
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'object_type') THEN
        CREATE TYPE object_type AS ENUM ('player', 'ball');
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'team_side') THEN
        CREATE TYPE team_side AS ENUM ('home', 'away', 'unknown');
    END IF;
END$$;

-- ----------------------------------------------------------------------------
-- match_summary: una fila por partido. Catalogo y metadatos del encuentro.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_summary (
    match_id          TEXT PRIMARY KEY,
    home_team         TEXT,
    away_team         TEXT,
    match_date        DATE,
    venue             TEXT,
    home_score        SMALLINT NOT NULL DEFAULT 0,
    away_score        SMALLINT NOT NULL DEFAULT 0,
    duration_seconds  INTEGER,
    fps               NUMERIC(8,3),
    width             INTEGER,
    height            INTEGER,
    field_length_m    NUMERIC(6,2) NOT NULL DEFAULT 105.00,  -- largo FIFA
    field_width_m     NUMERIC(6,2) NOT NULL DEFAULT 68.00,   -- ancho FIFA
    homography        DOUBLE PRECISION[9],                  -- matriz 3x3 aplanada
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes             TEXT
);

COMMENT ON TABLE match_summary IS
    'Catalogo de partidos. Un match_id por encuentro. La homografia se usa para pasar de pixeles a coordenadas normalizadas del campo.';

-- ----------------------------------------------------------------------------
-- tracking_events: UNA fila por objeto por frame (atomicidad pedida).
--   Particionada por tiempo (TimescaleDB hypertable en timestamp_ms).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tracking_events (
    match_id      TEXT        NOT NULL,
    frame         BIGINT      NOT NULL,
    timestamp_ms  BIGINT      NOT NULL,        -- frame * (1000/fps), en ms de partido
    object_type   object_type NOT NULL,        -- 'player' o 'ball'
    track_id      INTEGER     NOT NULL,        -- ID persistente (1..13) o 0 para ball
    team          team_side   NOT NULL DEFAULT 'unknown',
    -- Coordenadas en pixeles (origen top-left del frame)
    x             DOUBLE PRECISION NOT NULL,
    y             DOUBLE PRECISION NOT NULL,
    bbox_x1       INTEGER,
    bbox_y1       INTEGER,
    bbox_x2       INTEGER,
    bbox_y2       INTEGER,
    -- Coordenadas normalizadas en [0,1] del campo (util para visualizacion)
    x_norm        DOUBLE PRECISION,
    y_norm        DOUBLE PRECISION,
    -- Coordenadas en metros sobre el campo (origen centro de cancha, +x derecha, +y arriba)
    x_m           DOUBLE PRECISION,
    y_m           DOUBLE PRECISION,
    -- Geometria PostGIS (Point en Web Mercator para poder usar ST_Distance, ST_Contains, etc.)
    -- Se mantiene NULL hasta que la homografia del partido este calibrada.
    geom          GEOMETRY(Point, 3857),
    confidence    DOUBLE PRECISION,
    in_occlusion  BOOLEAN     NOT NULL DEFAULT FALSE,
    source_chunk  TEXT,                       -- ruta del chunk JSON del que provino
    CONSTRAINT pk_tracking_events
        PRIMARY KEY (match_id, frame, object_type, track_id)
);

COMMENT ON TABLE tracking_events IS
    'Atomica: 1 fila por objeto por frame. PK compuesta (match, frame, tipo, track_id) garantiza idempotencia al reinsertar chunks.';
COMMENT ON COLUMN tracking_events.timestamp_ms IS
    'Milisegundos desde el inicio del partido. Es la dimension de particion de la hypertable.';
COMMENT ON COLUMN tracking_events.geom IS
    'PostGIS Point en EPSG:3857 (Web Mercator, metros). NULL si la homografia aun no fue calibrada.';

-- ----------------------------------------------------------------------------
-- game_events: eventos de juego detectados (goles, pases, faltas, tiros...).
-- Se crea vacia en etapa 1; sera poblada en etapa 2 por un modulo de deteccion.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS game_events (
    event_id        BIGSERIAL,
    match_id        TEXT        NOT NULL,
    frame           BIGINT,
    timestamp_ms    BIGINT      NOT NULL,
    period          SMALLINT    NOT NULL DEFAULT 1,   -- 1 o 2 tiempo
    event_type      TEXT        NOT NULL,            -- 'pass','shot','goal','foul','recover','loss',...
    team            team_side   NOT NULL DEFAULT 'unknown',
    actor_track_id  INTEGER,                         -- jugador que inicia el evento
    target_track_id INTEGER,                         -- receptor (pases) o victima (faltas)
    start_x_m       DOUBLE PRECISION,
    start_y_m       DOUBLE PRECISION,
    end_x_m         DOUBLE PRECISION,
    end_y_m         DOUBLE PRECISION,
    start_geom      GEOMETRY(Point, 3857),
    end_geom        GEOMETRY(Point, 3857),
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_game_events
        PRIMARY KEY (match_id, event_id)
);

COMMENT ON TABLE game_events IS
    'Eventos discretos del partido. Se particiona por tiempo como hypertable. Se puebla en etapa 2.';

-- ----------------------------------------------------------------------------
-- loaded_chunks: registro de chunks ya ingestados. Permite hacer la ingesta
-- idempotente: si un chunk ya esta registrado, se lo saltea.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS loaded_chunks (
    match_id      TEXT      NOT NULL,
    chunk_path    TEXT      NOT NULL,
    chunk_sha256  CHAR(64)  NOT NULL,
    row_count     BIGINT    NOT NULL DEFAULT 0,
    loaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_loaded_chunks
        PRIMARY KEY (match_id, chunk_path)
);

COMMENT ON TABLE loaded_chunks IS
    'Auditoria de chunks procesados. PRIMARY KEY (match_id, chunk_path) garantiza idempotencia.';
