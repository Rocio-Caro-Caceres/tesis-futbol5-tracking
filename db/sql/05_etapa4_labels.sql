-- ============================================================================
-- 05_etapa4_labels.sql
-- Tabla para etiquetas de eventos de SoccerNet usadas como ground truth de
-- la etapa 4 (clasificador ML de Pases / Duelos / Faltas).
--
-- Aislada de game_events (que recibe eventos detectados en runtime) para no
-- mezclar labels sinteticos con detecciones reales.
--
-- Solo corre en el primer init de db_data (alphabetical ordering
-- de docker-entrypoint-initdb.d). Para aplicarlo a mano sobre una DB ya
-- levantada:  psql -f db/sql/05_etapa4_labels.sql
-- ============================================================================

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

CREATE TABLE IF NOT EXISTS event_training_labels (
    label_id        BIGSERIAL,
    match_id        TEXT         NOT NULL,
    half            SMALLINT     NOT NULL,        -- 1 o 2
    frame           BIGINT       NOT NULL,        -- frame de inicio del evento (en la mitad)
    timestamp_ms    BIGINT       NOT NULL,        -- frame * (1000 / fps) desde el kick-off de la mitad
    fps             NUMERIC(8,3) NOT NULL,
    event_type      TEXT         NOT NULL,        -- 'Pass' | 'Duel' | 'Foul'  (filtro para entrenamiento)
    event_label     TEXT         NOT NULL,        -- etiqueta cruda SoccerNet (mismo valor cuando coarse)
    event_subtype   TEXT,                         -- e.g. 'deep_completed', 'ground', 'aerial', 'foul.offsides'
    team            team_side    NOT NULL DEFAULT 'unknown',
    actor_track_id  INTEGER,                      -- jersey del emisor / ejecutor
    target_track_id INTEGER,                      -- jersey del receptor (Pass) o victima/oponente (Duel/Foul)
    start_x_m       DOUBLE PRECISION,             -- posicion en metros (origen centro de cancha, +x derecha, +y arriba)
    start_y_m       DOUBLE PRECISION,
    end_x_m         DOUBLE PRECISION,
    end_y_m         DOUBLE PRECISION,
    visibility      TEXT,                         -- 'visible' | 'uncertain' | 'invisible' (SoccerNet)
    result          TEXT,                         -- 'won' | 'lost' | 'completed' | 'incomplete' | ...
    source          TEXT         NOT NULL DEFAULT 'soccernet',
    raw             JSONB        NOT NULL DEFAULT '{}'::jsonb,  -- copia del evento SoccerNet original
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT pk_event_training_labels
        PRIMARY KEY (match_id, label_id)
);

COMMENT ON TABLE event_training_labels IS
    'Etiquetas de eventos de SoccerNet usadas como ground truth para entrenar el clasificador ML de Etapa 4. NO se mezcla con game_events (eventos detectados en runtime).';
COMMENT ON COLUMN event_training_labels.event_type IS
    'Clase coarse: Pass | Duel | Foul. El sub-tipo fino esta en event_subtype.';
COMMENT ON COLUMN event_training_labels.actor_track_id IS
    'Jersey del jugador que inicia el evento. Para Duelos/Faltas es quien gana/comete. track_id coincide con el usado en tracking_events para este match.';
COMMENT ON COLUMN event_training_labels.target_track_id IS
    'Jersey del receptor (Pass) o victima/oponente (Duel/Foul). NULL si el evento no lo declara.';
COMMENT ON COLUMN event_training_labels.start_x_m IS
    'Posicion en metros sobre el campo. Origen en el centro, +x derecha, +y arriba. Mismo sistema de coordenadas que tracking_events.x_m / y_m.';

CREATE INDEX IF NOT EXISTS ix_etl_event_type
    ON event_training_labels (event_type);

CREATE INDEX IF NOT EXISTS ix_etl_match_frame
    ON event_training_labels (match_id, frame);

CREATE INDEX IF NOT EXISTS ix_etl_match_half_frame
    ON event_training_labels (match_id, half, frame);
