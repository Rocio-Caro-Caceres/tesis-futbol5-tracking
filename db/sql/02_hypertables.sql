-- ============================================================================
-- 02_hypertables.sql
-- Convierte las tablas de series temporales en hypertables de TimescaleDB.
-- Idempotente: CREATE_EXTENT_TABLE es seguro de re-ejecutar dentro de una
-- transaccion y check_exclusion verifica si ya es hypertable.
-- ============================================================================

-- tracking_events: particionada por timestamp_ms en chunks de 1 minuto.
-- Justificacion: 1 minuto de juego a 24 fps = 1440 frames; con hasta 14 objetos
-- (13 jugadores + pelota) son ~20k filas por chunk, tamanio adecuado para
-- compresion y consultas analiticas.
SELECT create_hypertable(
    'tracking_events',
    'timestamp_ms',
    chunk_time_interval => 60000,   -- 60_000 ms = 1 minuto
    if_not_exists       => TRUE,
    migrate_data        => FALSE
);

-- game_events: misma granularidad temporal.
SELECT create_hypertable(
    'game_events',
    'timestamp_ms',
    chunk_time_interval => 60000,
    if_not_exists       => TRUE,
    migrate_data        => FALSE
);

-- Habilitamos compresion nativa de TimescaleDB sobre la parte "vieja" de los
-- datos. Se configura para activarse automaticamente a partir de 7 dias.
ALTER TABLE tracking_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'match_id, object_type, track_id',
    timescaledb.compress_orderby   = 'timestamp_ms DESC'
);

ALTER TABLE game_events SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'match_id, event_type',
    timescaledb.compress_orderby   = 'timestamp_ms DESC'
);

-- Policy: comprimir chunks con mas de 7 dias de antiguedad.
-- (No se aplica retroactivamente a datos existentes.)
SELECT add_compression_policy('tracking_events', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('game_events',     INTERVAL '7 days', if_not_exists => TRUE);

-- Retencion: despues de 365 dias, dropear chunks. Ajustable segun storage.
-- Comentado en etapa 1 para no perder datos durante desarrollo.
-- SELECT add_retention_policy('tracking_events', INTERVAL '365 days', if_not_exists => TRUE);
