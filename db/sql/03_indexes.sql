-- ============================================================================
-- 03_indexes.sql
-- Indices compuestos sobre la clave analitica (match_id, track_id, timestamp_ms)
-- para acelerar consultas de series temporales por jugador.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- tracking_events
-- ----------------------------------------------------------------------------

-- Indice principal: consulta tipica "dame la trayectoria del jugador X en el
-- partido M entre t1 y t2". Coincide con la PK para los lookups por (match, frame)
-- pero el ORDEN de las columnas adicionales esta optimizado para range scans.
CREATE INDEX IF NOT EXISTS idx_tracking_match_track_time
    ON tracking_events (match_id, track_id, timestamp_ms DESC);

-- Para consultas "dame todos los objetos de un frame": muy usado en anotacion
-- y depuracion.
CREATE INDEX IF NOT EXISTS idx_tracking_match_frame
    ON tracking_events (match_id, frame);

-- Indice espacial GIST sobre PostGIS geometry: habilita ST_DWithin, ST_Contains,
-- ST_Distance entre puntos del campo. Esencial para EPV / pitch control / zonas.
CREATE INDEX IF NOT EXISTS idx_tracking_geom
    ON tracking_events USING GIST (geom);

-- BRIN sobre timestamp_ms: muy chico (kilobytes por millon de filas) y
-- aprovecha la correlacion fisica con la hypertable. Acelera range scans
-- sobre el tiempo sin penalizar inserciones.
CREATE INDEX IF NOT EXISTS idx_tracking_time_brin
    ON tracking_events USING BRIN (timestamp_ms)
    WITH (pages_per_range = 32);

-- ----------------------------------------------------------------------------
-- game_events
-- ----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_events_match_time
    ON game_events (match_id, timestamp_ms DESC);

CREATE INDEX IF NOT EXISTS idx_events_match_type
    ON game_events (match_id, event_type, timestamp_ms DESC);

CREATE INDEX IF NOT EXISTS idx_events_geom_start
    ON game_events USING GIST (start_geom);

-- Indice GIN sobre metadata para filtros jsonb (etapa 2).
CREATE INDEX IF NOT EXISTS idx_events_metadata_gin
    ON game_events USING GIN (metadata jsonb_path_ops);

-- ----------------------------------------------------------------------------
-- match_summary
-- ----------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_match_date
    ON match_summary (match_date DESC);
