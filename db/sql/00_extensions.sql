-- ============================================================================
-- 00_extensions.sql
-- Habilita TimescaleDB y PostGIS en la base. Se ejecuta al inicializar
-- el volumen (docker-entrypoint-initdb.d), ANTES de los demas SQL.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;
