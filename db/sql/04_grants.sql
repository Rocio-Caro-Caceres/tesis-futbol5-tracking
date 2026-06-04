-- ============================================================================
-- 04_grants.sql
-- Grants minimos para la aplicacion. El usuario de la app NO es superuser:
-- TimescaleDB requiere superuser solo para create_hypertable, que se hace en
-- este script (etapa 1, no repetido en runtime).
-- ============================================================================

-- Crea rol de aplicacion si no existe (en runtime usaremos POSTGRES_USER)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'futbol5_app') THEN
        CREATE ROLE futbol5_app LOGIN PASSWORD 'futbol5_app';
    END IF;
END$$;

GRANT CONNECT ON DATABASE futbol5 TO futbol5_app;
GRANT USAGE  ON SCHEMA public TO futbol5_app;

GRANT SELECT, INSERT, UPDATE, DELETE
    ON ALL TABLES IN SCHEMA public
    TO futbol5_app;

GRANT USAGE, SELECT
    ON ALL SEQUENCES IN SCHEMA public
    TO futbol5_app;

-- Para que tablas futuras hereden los grants
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO futbol5_app;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO futbol5_app;
