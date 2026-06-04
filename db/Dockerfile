ARG TS_IMAGE=timescale/timescaledb:latest-pg16

FROM ${TS_IMAGE}

# PostGIS 3.4 (misma major version que pg16) sobre la imagen base de TimescaleDB.
# Mantenemos la imagen oficial de TimescaleDB y solo añadimos PostGIS via apt.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        postgresql-16-postgis-3 \
        postgresql-16-postgis-3-scripts \
    && rm -rf /var/lib/apt/lists/*

# Los .sql de db/sql/ se montan en runtime via docker-compose (volumen bind).
# Esto permite editar los scripts sin rebuildear la imagen.
