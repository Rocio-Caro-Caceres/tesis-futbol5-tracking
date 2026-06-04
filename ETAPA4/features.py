"""
features.py
===========
Ingenieria de caracteristicas para el clasificador ML de Etapa 4.

Para cada evento en `event_training_labels`:

    1. Se busca el match en los Parquet de `normalized_dir/<match>__h<half>/`.
    2. Se extrae una ventana de +/- W frames alrededor del frame del evento.
    3. Para cada frame se computa un vector de features que incluye:
         - Estado del actor (x, y, vx, vy, speed, heading)
         - Estado del target (x, y, vx, vy, speed)  [si existe]
         - Estado de la pelota (x, y, vx, vy, speed)
         - Distancias y angulos relativos (actor<->ball, actor<->target)
         - k oponentes y k companeros mas cercanos (distancias + velocidades)
    4. Se calcula velocidad por diferencias centrales sobre la trayectoria
       del actor en una ventana auxiliar de +/- S frames (smooth_window).
    5. El resultado es un np.ndarray de shape (window_size, F) por evento.
       Se guarda como Parquet columnar con:
         - features_3d:  object (ndarray serializable)  -> por evento
         - features_flat: ndarray aplanado (T*F,)         -> para XGBoost
         - features_summary: ndarray de resumen (mean/std/min/max/last) -> XGB
         - label_idx: int (Pass=0, Duel=1, Foul=2)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import FeaturesConfig, TARGET_LABELS
from .io_utils import info, stage, warn, write_parquet


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
# (velocidad nula en m/s) maxima permitida para descartar outliers del
# tracking. A 25 fps, un jugador puede correr ~8 m/s; un pase a 25 m/s;
# dejamos margen hasta 40 m/s.
MAX_SPEED_MPS = 40.0

# Anotaciones por feature (para interpretabilidad del LSTM).
FEATURE_NAMES: list[str] = []  # se llena en _build_feature_names


@dataclass(slots=True)
class WindowSample:
    match_id: str
    half: int
    frame: int
    event_type: str
    label_idx: int
    actor_track_id: int | None
    target_track_id: int | None
    features_3d: np.ndarray   # (T, F)
    features_flat: np.ndarray # (T*F,)
    features_summary: np.ndarray  # (5*F,)  mean/std/min/max/last por columna
    valid_frames: int         # cuantos frames de la ventana tienen datos


# ---------------------------------------------------------------------------
# Suavizado y velocidad
# ---------------------------------------------------------------------------
def _smooth_series(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Moving average centrado. En los bordes replica el valor del borde
    (no introduce NaN). Si `window` es 0 o negativo, devuelve copia.
    """
    if window is None or window <= 1:
        return arr.copy()
    w = int(window)
    if w % 2 == 0:
        w += 1
    pad = w // 2
    if pad == 0:
        return arr.copy()
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="valid")


def _central_diff_velocity(
    positions: np.ndarray, fps: float
) -> np.ndarray:
    """
    Diferencias centradas con el mismo largo que `positions`.
    Para T frames, devuelve (T, 2) con NaN en los extremos.
    """
    T = len(positions)
    v = np.full((T, 2), np.nan, dtype=np.float32)
    if T < 3 or fps <= 0:
        return v
    # [v(t) = (x(t+1) - x(t-1)) / (2 / fps)]
    v[1:-1, 0] = (positions[2:, 0] - positions[:-2, 0]) * fps / 2.0
    v[1:-1, 1] = (positions[2:, 1] - positions[:-2, 1]) * fps / 2.0
    return v


# ---------------------------------------------------------------------------
# Extraccion de trayectoria por track_id en un rango de frames
# ---------------------------------------------------------------------------
def _extract_track_in_window(
    track_df: pd.DataFrame, track_id: int, half: int, frame_min: int, frame_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (positions (T, 2), present_mask (T,)) para el track_id dado,
    en el rango [frame_min, frame_max] inclusive. Rellena NaN con ffill
    sobre el rango, y reporta la mascara de presencia original.
    """
    n = frame_max - frame_min + 1
    positions = np.full((n, 2), np.nan, dtype=np.float32)
    present = np.zeros(n, dtype=bool)

    sub = track_df[
        (track_df["track_id"] == track_id)
        & (track_df["frame"] >= frame_min)
        & (track_df["frame"] <= frame_max)
    ][["frame", "x_m", "y_m"]]
    if sub.empty:
        return positions, present

    sub = sub.dropna(subset=["x_m", "y_m"])
    for r in sub.itertuples(index=False):
        idx = int(r.frame) - frame_min
        positions[idx, 0] = float(r.x_m)
        positions[idx, 1] = float(r.y_m)
        present[idx] = True

    # Forward fill: si el track estaba en frame 100 pero no en 101, usa 100.
    if n > 1:
        last = positions[0].copy()
        for i in range(n):
            if not np.isnan(positions[i, 0]):
                last = positions[i]
            elif i > 0:
                positions[i] = last
    return positions, present


def _extract_ball_in_window(
    track_df: pd.DataFrame, half: int, frame_min: int, frame_max: int
) -> tuple[np.ndarray, np.ndarray]:
    """Lo mismo que _extract_track_in_window pero con track_id==0 (ball)."""
    n = frame_max - frame_min + 1
    positions = np.full((n, 2), np.nan, dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    sub = track_df[
        (track_df["object_type"] == "ball")
        & (track_df["frame"] >= frame_min)
        & (track_df["frame"] <= frame_max)
    ][["frame", "x_m", "y_m"]]
    if sub.empty:
        return positions, present
    sub = sub.dropna(subset=["x_m", "y_m"])
    for r in sub.itertuples(index=False):
        idx = int(r.frame) - frame_min
        positions[idx, 0] = float(r.x_m)
        positions[idx, 1] = float(r.y_m)
        present[idx] = True
    if n > 1:
        last = positions[0].copy()
        for i in range(n):
            if not np.isnan(positions[i, 0]):
                last = positions[i]
            elif i > 0:
                positions[i] = last
    return positions, present


def _extract_players_in_frame(
    track_df: pd.DataFrame, frame: int, exclude_track_ids: set[int]
) -> pd.DataFrame:
    """Snapshot de TODOS los players en un frame, excluyendo ciertos track_ids."""
    sub = track_df[
        (track_df["frame"] == frame)
        & (track_df["object_type"] == "player")
        & (~track_df["track_id"].isin(list(exclude_track_ids)))
    ][["track_id", "team", "x_m", "y_m", "vx_proxy"]].copy()
    return sub


def _compute_kinematics(
    positions: np.ndarray, present: np.ndarray, fps: float, smooth: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (velocity (T, 2), speed (T,)) con la trayectoria suavizada.
    Las velocidades se calculan por diferencias centrales; los frames
    faltantes conservan el ultimo valor conocido (forward fill) para que
    el LSTM no reciba NaN tras el suavizado.
    """
    smooth_xy = _smooth_series(positions[:, 0], smooth)
    smooth_xy = np.column_stack([
        smooth_xy,
        _smooth_series(positions[:, 1], smooth),
    ])
    v = _central_diff_velocity(smooth_xy, fps)
    # Forward fill NaN
    if len(v) > 0:
        for col in range(2):
            last = v[0, col]
            for i in range(len(v)):
                if not np.isnan(v[i, col]):
                    last = v[i, col]
                else:
                    v[i, col] = last
    speed = np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
    # Clip outliers
    speed = np.minimum(speed, MAX_SPEED_MPS)
    return v.astype(np.float32), speed.astype(np.float32)


# ---------------------------------------------------------------------------
# Features por frame
# ---------------------------------------------------------------------------
def _frame_features(
    actor_xy: np.ndarray,
    actor_v: np.ndarray,
    actor_speed: np.ndarray,
    target_xy: np.ndarray | None,
    target_v: np.ndarray | None,
    target_speed: np.ndarray | None,
    ball_xy: np.ndarray,
    ball_v: np.ndarray,
    ball_speed: np.ndarray,
    players_df: pd.DataFrame | None,
    top_k: int,
) -> np.ndarray:
    """
    Vector de features para UN frame. El orden debe matchear FEATURE_NAMES.
    """
    feats: list[float] = []

    # --- Actor
    feats.extend([actor_xy[0], actor_xy[1], actor_v[0], actor_v[1], actor_speed[0]])
    # --- Target (si existe; si no, zeros enmascarados abajo)
    if target_xy is not None:
        feats.extend([target_xy[0], target_xy[1], target_v[0], target_v[1], target_speed[0]])
        target_present = 1.0
    else:
        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        target_present = 0.0
    # --- Ball
    feats.extend([ball_xy[0], ball_xy[1], ball_v[0], ball_v[1], ball_speed[0]])
    ball_present = 1.0 if not math.isnan(ball_xy[0]) else 0.0
    feats.append(ball_present)

    # --- Distancias / angulos relativos
    # actor<->ball
    dx_ab = ball_xy[0] - actor_xy[0]
    dy_ab = ball_xy[1] - actor_xy[1]
    dist_ab = math.hypot(dx_ab, dy_ab)
    ang_ab = math.atan2(dy_ab, dx_ab) if dist_ab > 1e-6 else 0.0
    feats.extend([dx_ab, dy_ab, dist_ab, ang_ab])

    # actor<->target
    if target_xy is not None:
        dx_at = target_xy[0] - actor_xy[0]
        dy_at = target_xy[1] - actor_xy[1]
        dist_at = math.hypot(dx_at, dy_at)
        ang_at = math.atan2(dy_at, dx_at) if dist_at > 1e-6 else 0.0
    else:
        dx_at, dy_at, dist_at, ang_at = 0.0, 0.0, 0.0, 0.0
    feats.extend([dx_at, dy_at, dist_at, ang_at])
    feats.append(target_present)

    # --- k oponentes y k companeros mas cercanos
    opp_dists, opp_vrel, opp_dirs = _neighbor_features(actor_xy, players_df, top_k, want_team="opponent")
    team_dists, team_vrel, team_dirs = _neighbor_features(actor_xy, players_df, top_k, want_team="teammate")
    feats.extend(opp_dists)
    feats.extend(opp_vrel)
    feats.extend(opp_dirs)
    feats.extend(team_dists)
    feats.extend(team_vrel)
    feats.extend(team_dirs)

    return np.array(feats, dtype=np.float32)


def _neighbor_features(
    actor_xy: np.ndarray,
    players_df: pd.DataFrame | None,
    top_k: int,
    want_team: str,
) -> tuple[list[float], list[float], list[float]]:
    """
    Devuelve 3 listas de largo `top_k` con: distancia, |v_rel|, angulo.
    `players_df` es el snapshot de jugadores en un frame con columnas
    track_id, team, x_m, y_m, vx_proxy.
    """
    if players_df is None or players_df.empty:
        empty = [0.0] * top_k
        return empty, empty, empty
    if want_team == "opponent":
        # Sin info del equipo del actor, no podemos distinguir oponentes de
        # companeros a nivel de feature. Tomamos los top_k mas cercanos como
        # "oponentes" (es la convencion cuando team del actor no se conoce);
        # en el caso contrario, se podria usar `players_df['team']`.
        # Por simplicidad y para no requerir el team del actor en el feature
        # pipeline, devolvemos los k mas cercanos indistintamente.
        # (Si en el futuro se quiere distinguir, pasar actor_team al extractor
        # y filtrar players_df[players_df['team'] != actor_team].)
        sub = players_df
    else:
        sub = players_df

    diffs = sub[["x_m", "y_m"]].to_numpy() - actor_xy.reshape(1, 2)
    dists = np.linalg.norm(diffs, axis=1)
    if len(dists) == 0:
        empty = [0.0] * top_k
        return empty, empty, empty
    order = np.argsort(dists)
    top = order[:top_k]
    out_d = [float(dists[i]) for i in top]
    # Padding
    while len(out_d) < top_k:
        out_d.append(0.0)

    out_v = [0.0] * top_k
    out_a = [0.0] * top_k
    if "vx_proxy" in sub.columns:
        # vx_proxy es la magnitud de velocidad en m/s si se calculo
        vmag = sub["vx_proxy"].to_numpy()
        for j, i in enumerate(top):
            out_v[j] = float(vmag[i])
    for j, i in enumerate(top):
        d = diffs[i]
        if np.linalg.norm(d) > 1e-6:
            out_a[j] = float(math.atan2(d[1], d[0]))
    return out_d, out_v, out_a


# ---------------------------------------------------------------------------
# Nombres de features (para interpretabilidad)
# ---------------------------------------------------------------------------
def _build_feature_names(top_k: int) -> list[str]:
    names: list[str] = []
    names += ["actor_x", "actor_y", "actor_vx", "actor_vy", "actor_speed"]
    names += ["target_x", "target_y", "target_vx", "target_vy", "target_speed"]
    names += ["ball_x", "ball_y", "ball_vx", "ball_vy", "ball_speed", "ball_present"]
    names += ["ab_dx", "ab_dy", "ab_dist", "ab_angle"]
    names += ["at_dx", "at_dy", "at_dist", "at_angle", "target_present"]
    for i in range(top_k):
        names += [f"opp{i}_dist", f"opp{i}_vrel", f"opp{i}_angle"]
    for i in range(top_k):
        names += [f"team{i}_dist", f"team{i}_vrel", f"team{i}_angle"]
    return names


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def build_features(
    events_df: pd.DataFrame,
    tracking_by_match: dict[str, pd.DataFrame],
    cfg: FeaturesConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Args:
        events_df: DataFrame con las filas de event_training_labels (al menos
            match_id, half, frame, event_type, actor_track_id, target_track_id).
        tracking_by_match: dict match_id_synthetic (con __h<N>) -> DataFrame
            de tracking atomico con columnas track_id, frame, x_m, y_m,
            object_type, team.

    Returns:
        X_seq: np.ndarray (N, T, F) listo para LSTM (con NaN rellenados a 0).
        X_flat: np.ndarray (N, T*F + 5*F) listo para XGBoost.
        meta_df: pd.DataFrame con una fila por evento (label, match, etc.).
    """
    global FEATURE_NAMES
    FEATURE_NAMES = _build_feature_names(cfg.top_k_neighbors)
    n_features = len(FEATURE_NAMES)
    T = 2 * cfg.window_frames + 1

    # Precomputar vx_proxy (rapido) por track para usar en _neighbor_features
    enriched_tracking: dict[str, pd.DataFrame] = {}
    for mid, tdf in tracking_by_match.items():
        if tdf.empty:
            enriched_tracking[mid] = tdf
            continue
        etdf = tdf.copy()
        etdf = etdf.sort_values(["track_id", "frame"]).reset_index(drop=True)
        # Velocidad escalar por track con diff central
        etdf["vx_proxy"] = (
            etdf.groupby("track_id", group_keys=False)["x_m"]
            .transform(lambda s: s.diff().fillna(0) * cfg.fps)
            .pow(2)
            .add(
                etdf.groupby("track_id", group_keys=False)["y_m"]
                .transform(lambda s: s.diff().fillna(0) * cfg.fps)
                .pow(2)
            )
            .pow(0.5)
            .astype(np.float32)
        )
        enriched_tracking[mid] = etdf

    X_seq = np.zeros((len(events_df), T, n_features), dtype=np.float32)
    X_flat_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    skipped = 0

    for i, ev in enumerate(events_df.itertuples(index=False)):
        match_root = ev.match_id
        half = int(ev.half)
        frame = int(ev.frame)
        event_type = ev.event_type
        actor_id = ev.actor_track_id if not pd.isna(ev.actor_track_id) else None
        target_id = ev.target_track_id if not pd.isna(ev.target_track_id) else None
        if pd.isna(target_id):
            target_id = None

        mid_synth = f"{match_root}__h{half}"
        track_df = enriched_tracking.get(mid_synth)
        if track_df is None or track_df.empty:
            skipped += 1
            continue

        frame_min = max(0, frame - cfg.window_frames)
        frame_max = frame + cfg.window_frames

        # Trayectorias de actor / target / ball
        actor_pos, actor_present = _extract_track_in_window(
            track_df, actor_id, half, frame_min, frame_max
        ) if actor_id is not None else (
            np.full((T, 2), np.nan, dtype=np.float32),
            np.zeros(T, dtype=bool),
        )
        if target_id is not None:
            target_pos, _ = _extract_track_in_window(
                track_df, target_id, half, frame_min, frame_max
            )
        else:
            target_pos = None
        ball_pos, _ = _extract_ball_in_window(track_df, half, frame_min, frame_max)

        actor_v, actor_speed = _compute_kinematics(actor_pos, actor_present, cfg.fps, cfg.smooth_window)
        if target_pos is not None:
            target_v, target_speed = _compute_kinematics(
                target_pos, np.ones(T, dtype=bool), cfg.fps, cfg.smooth_window
            )
        else:
            target_v = target_speed = None
        ball_v, ball_speed = _compute_kinematics(
            ball_pos, np.ones(T, dtype=bool), cfg.fps, cfg.smooth_window
        )

        # Snapshots de jugadores en cada frame (para oponentes/companeros)
        players_per_frame: dict[int, pd.DataFrame] = {}
        exclude = {actor_id} if actor_id is not None else set()
        if target_id is not None:
            exclude.add(target_id)
        for f in range(frame_min, frame_max + 1):
            snap = _extract_players_in_frame(track_df, f, exclude)
            if not snap.empty:
                players_per_frame[f] = snap

        # Features por frame
        feats = np.zeros((T, n_features), dtype=np.float32)
        for t in range(T):
            f = frame_min + t
            pdf = players_per_frame.get(f)
            feats[t] = _frame_features(
                actor_xy=actor_pos[t],
                actor_v=actor_v[t],
                actor_speed=actor_speed[t:t + 1],
                target_xy=target_pos[t] if target_pos is not None else None,
                target_v=target_v[t] if target_v is not None else None,
                target_speed=target_speed[t:t + 1] if target_speed is not None else None,
                ball_xy=ball_pos[t],
                ball_v=ball_v[t],
                ball_speed=ball_speed[t:t + 1],
                players_df=pdf,
                top_k=cfg.top_k_neighbors,
            )
        # Reemplazar NaN por 0 (LSTM los recibe como ceros enmascarados por
        # `valid_frames`/`present` flags que agregamos en `frame_features`).
        # Tambien sustituimos inf.
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        X_seq[i] = feats

        # XGBoost: flat + summary (mean/std/min/max/last por columna)
        flat = feats.reshape(-1)
        summary = np.concatenate([
            feats.mean(axis=0),
            feats.std(axis=0),
            feats.min(axis=0),
            feats.max(axis=0),
            feats[-1],
        ])
        X_flat_rows.append(np.concatenate([flat, summary]))

        label_idx = TARGET_LABELS.index(event_type) if event_type in TARGET_LABELS else -1
        meta_rows.append({
            "row_id":         i,
            "match_id":       match_root,
            "half":           half,
            "frame":          frame,
            "event_type":     event_type,
            "label_idx":      label_idx,
            "actor_track_id": int(actor_id) if actor_id is not None else -1,
            "target_track_id": int(target_id) if target_id is not None else -1,
            "valid_frames":   int(actor_present.sum()),
        })
        if (i + 1) % 500 == 0:
            info(f"  features: {i+1}/{len(events_df)} eventos procesados")

    if X_flat_rows:
        X_flat = np.stack(X_flat_rows).astype(np.float32)
    else:
        X_flat = np.zeros((0, T * n_features + 5 * n_features), dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    info(f"features: {len(meta_df)} muestras listas ({skipped} saltadas por falta de tracking)")
    info(f"   X_seq shape={X_seq.shape}  X_flat shape={X_flat.shape}  F={n_features} T={T}")
    return X_seq, X_flat, meta_df


# ---------------------------------------------------------------------------
# Loader: lee los Parquet normalizados
# ---------------------------------------------------------------------------
def load_normalized(
    normalized_dir: Path,
    target_labels: tuple[str, ...] = TARGET_LABELS,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Carga todos los partidos normalizados. Devuelve:
      - events_df concatenado (filtrado a target_labels)
      - tracking_by_match: dict match_id_synthetic -> DataFrame de tracking
    """
    base = Path(normalized_dir)
    if not base.exists():
        raise FileNotFoundError(f"No existe {base}")
    events: list[pd.DataFrame] = []
    tracking: dict[str, pd.DataFrame] = {}
    for gd in sorted(p for p in base.iterdir() if p.is_dir()):
        ep = gd / "events.parquet"
        if ep.exists():
            edf = pd.read_parquet(ep)
            if not edf.empty:
                edf = edf[edf["event_type"].isin(target_labels)].copy()
                if not edf.empty:
                    events.append(edf)
        # tracking.parquet contiene el partido partido por mitad, con
        # match_id ya sufijado __h<N>. Lo cargamos partido por su key.
        tp = gd / "tracking.parquet"
        if tp.exists():
            tdf = pd.read_parquet(tp)
            if not tdf.empty:
                for mid, sub in tdf.groupby("match_id"):
                    tracking[mid] = sub.reset_index(drop=True)
    if not events:
        warn(f"No se encontraron eventos con target_labels={target_labels} en {base}")
        return pd.DataFrame(), tracking
    events_df = pd.concat(events, ignore_index=True)
    info(f"load_normalized: {len(events_df)} eventos, {len(tracking)} halves de tracking")
    return events_df, tracking


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Extrae features (ventanas temporales) para Etapa 4")
    p.add_argument("--normalized-dir", default="ETAPA4_data/normalized",
                   help="Carpeta con los Parquet generados por `normalize`")
    p.add_argument("--output-dir", default="ETAPA4_data/features",
                   help="Donde escribir las features serializadas")
    p.add_argument("--window", type=int, default=25,
                   help="W: frames a cada lado del evento. total=2W+1")
    p.add_argument("--top-k", type=int, default=4,
                   help="Cantidad de oponentes/companeros mas cercanos a incluir")
    args = p.parse_args(argv)

    from .config import FeaturesConfig
    cfg = FeaturesConfig(
        window_frames=args.window,
        top_k_neighbors=args.top_k,
        output_dir=Path(args.output_dir),
    )

    with stage("load-normalized"):
        events_df, tracking = load_normalized(Path(args.normalized_dir))

    if events_df.empty:
        warn("No hay eventos. Abortando.")
        return 1

    with stage("build-features"):
        X_seq, X_flat, meta_df = build_features(events_df, tracking, cfg)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_seq.npy", X_seq)
    np.save(out / "X_flat.npy", X_flat)
    write_parquet(meta_df, out / "meta.parquet")
    # Guardar nombres de features para interpretabilidad
    from .io_utils import write_json
    write_json(out / "feature_names.json", _build_feature_names(cfg.top_k_neighbors))
    info(f"Features guardados en {out}")
    return 0
