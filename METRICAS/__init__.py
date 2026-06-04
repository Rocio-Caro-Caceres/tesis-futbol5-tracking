"""
METRICAS
========

Modulo de metricas de tracking (Etapas 2 y 3).

Componentes:
    metrica_loader  Adaptador de datos Metrica Sports (vía kloppy) que
                    reescala coordenadas normalizadas [0, 1] a metros
                    absolutos con origen en el centro del campo.
    smoothing       Media movil CENTRAL con ventana parametrica, robusta
                    a NaN. Suaviza las coordenadas X, Y antes de
                    calcular metricas derivadas.
    kinematics      Velocidad (vx, vy), rapidez absoluta y aceleracion
                    por track_id, calculadas por DIFERENCIAS CENTRALES
                    (variante central de las ecuaciones de
                    LaurieOnTracking/Metrica_Velocities).
    distance        Sumatoria acumulada de distancia euclidea con umbral
                    de corte (0.03 m por defecto) para omitir pasos
                    cuando el jugador esta estatico.
    heatmap         Discretizacion espacial 2D + filtro gaussiano
                    (scipy.ndimage.gaussian_filter) sobre el historico
                    de posiciones, con export a .npy y .csv.
    possession      FSM de posesion del balon basada en proximidad
                    jugador-balon y variacion del vector velocidad
                    del balon, con histeresis por contador de robo.
                    (Etapa 3.)
    events          Deteccion geometrica de eventos de campo: goles
                    (interseccion balon-poligono de arco) y atajadas
                    (proximidad al arco + intercepcion por guardameta).
                    (Etapa 3.)
    pipeline        Orquestador: ejecuta la cadena completa sobre un
                    DataFrame largo y ofrece un CLI
                    (`python -m METRICAS ...`).

Convenciones geometricas (consistentes con tracking/homography.py):
    - Origen en el centro del campo.
    - +x hacia la derecha, +y hacia arriba.
    - Campo por defecto: 105 m x 68 m (FIFA 11-a-side). Parametrizable
      via FieldDims para futbol 5 u otras dimensiones.

Dependencias:
    - numpy, pandas: requeridas.
    - scipy: requerida para gaussian_filter (heatmap). Se importa
      perezosamente.
    - kloppy: OPCIONAL, solo necesaria para metrica_loader.load_via_kloppy.
      Si no esta instalada, las demas funciones siguen funcionando.
"""

from .distance import (
    DEFAULT_MIN_STEP_M,
    cumulative_distance,
    total_distance_per_track,
)
from .events import (
    DEFAULT_COOLDOWN_S,
    DEFAULT_GOAL_DEPTH_M,
    DEFAULT_GOAL_WIDTH_M,
    DEFAULT_KEEPER_INTERCEPT_RADIUS_M,
    DEFAULT_SAVE_RESPONSE_S,
    DEFAULT_SHOT_SPEED_MPS,
    Event,
    EventsConfig,
    EventsResult,
    detect_events,
)
from .heatmap import build_heatmap, save_heatmap_csv, save_heatmap_npy
from .kinematics import compute_kinematics, remove_kinematics
from .metrica_loader import (
    DEFAULT_FIELD_LENGTH_M,
    DEFAULT_FIELD_WIDTH_M,
    FieldDims,
    from_kloppy_dataset,
    load_via_kloppy,
    rescale_norm_to_meters,
)
from .pipeline import PipelineConfig, PipelineResult, run_pipeline
from .possession import (
    DEFAULT_CONTROL_RADIUS_M,
    DEFAULT_POSSESSION_RADIUS_M,
    DEFAULT_STEAL_MIN_FRAMES,
    DEFAULT_STEAL_RADIUS_M,
    DEFAULT_TOUCH_ACCEL_MPS2,
    PossessionConfig,
    PossessionResult,
    compute_possession,
    possession_percentages,
)
from .smoothing import central_moving_average, smooth_positions

__all__ = [
    "DEFAULT_FIELD_LENGTH_M",
    "DEFAULT_FIELD_WIDTH_M",
    "FieldDims",
    "from_kloppy_dataset",
    "load_via_kloppy",
    "rescale_norm_to_meters",
    "central_moving_average",
    "smooth_positions",
    "compute_kinematics",
    "remove_kinematics",
    "cumulative_distance",
    "total_distance_per_track",
    "build_heatmap",
    "save_heatmap_csv",
    "save_heatmap_npy",
    "DEFAULT_POSSESSION_RADIUS_M",
    "DEFAULT_CONTROL_RADIUS_M",
    "DEFAULT_STEAL_RADIUS_M",
    "DEFAULT_STEAL_MIN_FRAMES",
    "DEFAULT_TOUCH_ACCEL_MPS2",
    "PossessionConfig",
    "PossessionResult",
    "compute_possession",
    "possession_percentages",
    "DEFAULT_GOAL_DEPTH_M",
    "DEFAULT_GOAL_WIDTH_M",
    "DEFAULT_SHOT_SPEED_MPS",
    "DEFAULT_SAVE_RESPONSE_S",
    "DEFAULT_KEEPER_INTERCEPT_RADIUS_M",
    "DEFAULT_COOLDOWN_S",
    "Event",
    "EventsConfig",
    "EventsResult",
    "detect_events",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]
