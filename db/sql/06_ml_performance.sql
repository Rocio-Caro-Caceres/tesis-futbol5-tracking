-- ============================================================================
-- 06_ml_performance.sql
-- Tabla de log para metricas de evaluacion de los modelos ML de Etapa 4.
-- Cada fila corresponde a una corrida de entrenamiento (XGBoost o LSTM).
--
-- Se persisten aqui los resultados que benchmark.py vuelca como JSON sueltos
-- para poder hacer consultas historicas entre ejecuciones.
-- ============================================================================

CREATE TABLE IF NOT EXISTS ml_performance_logs (
    run_id          VARCHAR(100) PRIMARY KEY,
    model_type      VARCHAR(20) NOT NULL,       -- 'XGBOOST' | 'LSTM'
    accuracy        NUMERIC(4,4) NOT NULL,      -- [0, 1]
    f1_macro        NUMERIC(4,4) NOT NULL,      -- [0, 1]
    inference_ms    NUMERIC(7,2) NOT NULL,       -- ms por muestra
    train_s         NUMERIC(7,2) NOT NULL,       -- segundos de entrenamiento
    n_train         INTEGER,                     -- muestras en train
    n_val           INTEGER,                     -- muestras en val
    n_features      INTEGER,                     -- dimension del feature vector
    model_path      TEXT,
    config_json     JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE ml_performance_logs IS
    'Metricas de evaluacion de modelos ML (Etapa 4). Una fila por corrida.';
COMMENT ON COLUMN ml_performance_logs.run_id IS
    'Identificador unico de la corrida. Ej: xgb_20260608_1425 o lstm_v2_seed42.';
COMMENT ON COLUMN ml_performance_logs.model_type IS
    'Tipo de modelo: XGBOOST o LSTM.';
COMMENT ON COLUMN ml_performance_logs.config_json IS
    'Hiperparametros de la corrida en JSONB para consultas flexibles.';
