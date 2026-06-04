"""
possession.py
=============

Maquina de estados finitos (FSM) para estimar la posesion del balon
por frame, basada en heuristicas geometricas y cinematicas.

Heuristicas combinadas
----------------------

1. **Proximidad jugador-balon** (``possession_radius_m``). En cada
   frame se calcula la distancia euclidea del balon a todos los
   jugadores activos. El candidato a poseedor es el jugador mas
   cercano dentro de este radio.

2. **Variacion del vector velocidad del balon**
   (``touch_accel_mps2``). Una variacion brusca de la magnitud o
   direccion de ``v_ball`` (aceleracion instantanea superior al
   umbral) marca un *touch* o intercepcion. Se acredita al poseedor
   vigente en ese frame. Esto captura tanto controles suaves
   (desaceleracion al recibir un pase) como cambios de direccion
   abruptos (un pase/shot).

3. **Histeresis con contador de robo** (``steal_radius_m`` +
   ``steal_min_frames``). Para evitar que la posesion oscile entre
   equipos cuando dos jugadores disputan el balon, el cambio de
   posesion requiere que un jugador rival permanezca dentro del
   radio de robo durante N frames consecutivos.

Estados del FSM
---------------

- ``LOOSE``        nadie controla el balon (distancia > radio).
- ``<team>``       un equipo (con un track_id especifico) tiene la
                   posesion; el estado incluye tambien el
                   ``track_id`` del poseedor concreto.

Salida
------

- ``timeline``  : una fila por frame con ``possessor_track_id``,
                  ``possessor_team``, ``state``, ``is_touch``,
                  ``ball_speed_mps``, ``ball_accel_mps2``.
- ``per_track`` : agregados por track_id (tiempo en posesion,
                  numero de touches, posesiones iniciadas).
- ``spells``    : una fila por posesion ininterrumpida
                  (spell = secuencia de frames del mismo poseedor).

Convenciones geometricas (consistentes con el resto de METRICAS):
origen en el centro del campo, +x a la derecha, +y arriba.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


BALL_TRACK_ID: int = 0
LOOSE: str = "loose"

DEFAULT_POSSESSION_RADIUS_M: float = 2.0
DEFAULT_CONTROL_RADIUS_M: float = 1.0
DEFAULT_STEAL_RADIUS_M: float = 1.0
DEFAULT_STEAL_MIN_FRAMES: int = 3
DEFAULT_TOUCH_ACCEL_MPS2: float = 4.0
DEFAULT_MIN_POSSESSION_FRAMES: int = 3


@dataclass
class PossessionConfig:
    """Parametros del FSM de posesion. Todos tienen defaults razonables."""

    possession_radius_m: float = DEFAULT_POSSESSION_RADIUS_M
    control_radius_m: float = DEFAULT_CONTROL_RADIUS_M
    steal_radius_m: float = DEFAULT_STEAL_RADIUS_M
    steal_min_frames: int = DEFAULT_STEAL_MIN_FRAMES
    touch_accel_mps2: float = DEFAULT_TOUCH_ACCEL_MPS2
    min_possession_frames: int = DEFAULT_MIN_POSSESSION_FRAMES


@dataclass
class PossessionResult:
    """Salida de :func:`compute_possession`."""

    timeline: pd.DataFrame
    per_track: pd.DataFrame
    spells: pd.DataFrame
    config: PossessionConfig


@dataclass
class _FSM:
    """Estado mutable de la maquina. Una instancia por llamada."""

    possessor_track_id: int | None = None
    possessor_team: str = LOOSE
    frames_in_state: int = 0
    pending_steal_team: str | None = None
    pending_steal_track_id: int | None = None
    pending_steal_frames: int = 0
    last_touch_frame: int = -1
    last_touch_track_id: int | None = None
    spell_start_frame: int = -1


def _build_player_frame_index(
    players: pd.DataFrame,
) -> dict[int, dict[int, tuple[float, float, str]]]:
    """Indexa jugadores por ``(frame, track_id)`` para lookup O(1)."""
    out: dict[int, dict[int, tuple[float, float, str]]] = {}
    if players.empty:
        return out
    cols_needed = {"frame", "track_id", "x_m", "y_m"}
    missing = cols_needed - set(players.columns)
    if missing:
        raise KeyError(f"players DF sin columnas {missing}")
    team_col = "team" if "team" in players.columns else None
    for frame, tid, x, y, team in zip(
        players["frame"].to_numpy(),
        players["track_id"].to_numpy(),
        players["x_m"].to_numpy(),
        players["y_m"].to_numpy(),
        players[team_col].to_numpy() if team_col else (["unknown"] * len(players)),
    ):
        f = int(frame)
        t = int(tid)
        out.setdefault(f, {})[t] = (float(x), float(y), str(team))
    return out


def _build_ball_frame_index(ball: pd.DataFrame) -> dict[int, tuple[float, float, float, float]]:
    """Indexa balon por frame. Devuelve (x, y, speed, accel) por frame."""
    out: dict[int, tuple[float, float, float, float]] = {}
    if ball.empty:
        return out
    speed = ball["speed_mps"].to_numpy() if "speed_mps" in ball.columns else np.full(len(ball), np.nan)
    accel = ball["accel_mps2"].to_numpy() if "accel_mps2" in ball.columns else np.full(len(ball), np.nan)
    for frame, x, y, s, a in zip(
        ball["frame"].to_numpy(),
        ball["x_m"].to_numpy(),
        ball["y_m"].to_numpy(),
        speed,
        accel,
    ):
        out[int(frame)] = (float(x), float(y), float(s), float(a))
    return out


def _candidate_possessor(
    ball_xy: tuple[float, float],
    players_in_frame: dict[int, tuple[float, float, str]],
    cfg: PossessionConfig,
) -> tuple[int, str, float] | None:
    """Devuelve (track_id, team, dist) del jugador mas cercano dentro del
    radio de posesion, o None si nadie califica."""
    bx, by = ball_xy
    best: tuple[int, str, float] | None = None
    for tid, (px, py, team) in players_in_frame.items():
        d = float(np.hypot(px - bx, py - by))
        if d <= cfg.possession_radius_m and (best is None or d < best[2]):
            best = (tid, team, d)
    return best


def _update_fsm(
    fsm: _FSM,
    candidate: tuple[int, str, float] | None,
    ball_speed: float,
    ball_accel: float,
    cfg: PossessionConfig,
) -> tuple[str, int | None, bool]:
    """Avanza un frame del FSM. Devuelve (state, track_id, is_touch).

    ``is_touch`` se enciende cuando la aceleracion del balon supera el
    umbral **y** el poseedor actual esta a menos de ``control_radius_m``
    del balon, o cuando el poseedor cambia (cambio de tack = contacto).
    """
    cand_tid, cand_team, cand_dist = (None, None, None) if candidate is None else candidate

    if candidate is None:
        fsm.pending_steal_team = None
        fsm.pending_steal_track_id = None
        fsm.pending_steal_frames = 0
        fsm.possessor_track_id = None
        fsm.possessor_team = LOOSE
        fsm.frames_in_state += 1
        return LOOSE, None, False

    is_touch = bool(
        np.isfinite(ball_accel) and abs(ball_accel) >= cfg.touch_accel_mps2
        and cand_dist is not None
        and cand_dist <= cfg.control_radius_m
    )

    if fsm.possessor_team == LOOSE or fsm.possessor_track_id is None:
        fsm.possessor_track_id = cand_tid
        fsm.possessor_team = cand_team
        fsm.frames_in_state = 1
        fsm.spell_start_frame = -1
        fsm.pending_steal_team = None
        fsm.pending_steal_frames = 0
        return cand_team, cand_tid, is_touch

    same_player = cand_tid == fsm.possessor_track_id
    same_team = cand_team == fsm.possessor_team

    if same_player:
        fsm.frames_in_state += 1
        fsm.pending_steal_team = None
        fsm.pending_steal_frames = 0
        return fsm.possessor_team, fsm.possessor_track_id, is_touch

    if same_team:
        fsm.possessor_track_id = cand_tid
        fsm.possessor_team = cand_team
        fsm.frames_in_state = 1
        fsm.pending_steal_team = None
        fsm.pending_steal_frames = 0
        return fsm.possessor_team, fsm.possessor_track_id, is_touch

    if cand_dist <= cfg.steal_radius_m:
        if fsm.pending_steal_team == cand_team and fsm.pending_steal_track_id == cand_tid:
            fsm.pending_steal_frames += 1
        else:
            fsm.pending_steal_team = cand_team
            fsm.pending_steal_track_id = cand_tid
            fsm.pending_steal_frames = 1
        if fsm.pending_steal_frames >= cfg.steal_min_frames:
            fsm.possessor_track_id = cand_tid
            fsm.possessor_team = cand_team
            fsm.frames_in_state = 1
            fsm.pending_steal_team = None
            fsm.pending_steal_frames = 0
            return fsm.possessor_team, fsm.possessor_track_id, is_touch
        fsm.frames_in_state += 1
        return fsm.possessor_team, fsm.possessor_track_id, is_touch

    fsm.pending_steal_team = None
    fsm.pending_steal_frames = 0
    fsm.frames_in_state += 1
    return fsm.possessor_team, fsm.possessor_track_id, is_touch


def _split_long_df(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa el DataFrame largo en (players, ball) usando la convencion
    del proyecto: ``track_id == 0`` o ``object_type == 'ball'`` es balon,
    el resto son jugadores."""
    if "track_id" not in df.columns:
        raise KeyError("DataFrame sin columna 'track_id'")
    if "object_type" in df.columns:
        is_ball = df["object_type"].astype(str).str.lower() == "ball"
    else:
        is_ball = df["track_id"].to_numpy() == BALL_TRACK_ID
    ball = df.loc[is_ball].copy()
    players = df.loc[~is_ball].copy()
    return players, ball


def compute_possession(
    df: pd.DataFrame,
    *,
    fps: float | None = None,
    config: PossessionConfig | None = None,
) -> PossessionResult:
    """Ejecuta el FSM de posesion sobre un DataFrame largo (mezcla
    jugadores + balon, o solo jugadores, pasando el balon por
    separado).

    Parameters
    ----------
    df : DataFrame
        Tracking en formato largo. Debe contener ``frame``,
        ``track_id``, ``x_m``, ``y_m``. Si el balon esta incluido,
        debe tener ``track_id == 0`` o ``object_type == 'ball'``.
        Si ya se computo ``speed_mps`` y ``accel_mps2`` (tipicamente
        via :func:`compute_kinematics`), se usan para detectar
        touches; si no, se calculan internamente con diferencias
        centrales.
    fps : float, optional
        Solo se usa si ``speed_mps`` / ``accel_mps2`` no estan en el
        DataFrame.
    config : PossessionConfig, optional

    Returns
    -------
    PossessionResult con ``timeline``, ``per_track``, ``spells``.
    """
    cfg = config or PossessionConfig()
    players, ball = _split_long_df(df)

    if ball.empty and "x_m" not in df.columns:
        raise ValueError(
            "DataFrame vacio. compute_possession necesita al menos "
            "las posiciones de jugadores y (preferentemente) del balon."
        )

    if ball.empty:
        ball = pd.DataFrame(columns=["frame", "x_m", "y_m", "speed_mps", "accel_mps2"])

    if "speed_mps" not in ball.columns or ball["speed_mps"].isna().all():
        if fps is None:
            raise ValueError(
                "Falta 'speed_mps' en el balon y no se paso 'fps'."
            )
        ball = ball.sort_values("frame").reset_index(drop=True)
        t = ball["frame"].to_numpy(dtype=float) / float(fps)
        bx = ball["x_m"].to_numpy(dtype=float)
        by = ball["y_m"].to_numpy(dtype=float)
        vx, vy = _central_diff_pair(bx, t), _central_diff_pair(by, t)
        ball["speed_mps"] = np.sqrt(vx * vx + vy * vy)
        vxv = ball["speed_mps"].to_numpy()
        ball["accel_mps2"] = _central_diff_pair(vxv, t)

    player_index = _build_player_frame_index(players)
    ball_index = _build_ball_frame_index(ball)

    all_frames = sorted(set(player_index) | set(ball_index))
    if not all_frames:
        empty = pd.DataFrame(columns=[
            "frame", "time_s", "ball_x_m", "ball_y_m", "ball_speed_mps",
            "ball_accel_mps2", "possessor_team", "possessor_track_id",
            "state", "is_touch", "candidate_track_id", "candidate_team",
            "candidate_dist_m",
        ])
        return PossessionResult(
            timeline=empty,
            per_track=pd.DataFrame(columns=[
                "track_id", "team", "time_in_possession_s",
                "n_touches", "n_possessions", "n_frames",
            ]),
            spells=pd.DataFrame(columns=[
                "spell_id", "start_frame", "end_frame", "team",
                "track_id", "n_frames", "n_touches",
                "start_time_s", "end_time_s", "duration_s",
            ]),
            config=cfg,
        )

    fps_eff = float(fps) if fps else _infer_fps(all_frames, players, ball)
    fsm = _FSM()

    rows: list[dict[str, Any]] = []
    spells: list[dict[str, Any]] = []
    spell_id = 0
    cur_spell: dict[str, Any] | None = None

    for f in all_frames:
        ball_data = ball_index.get(f)
        if ball_data is None:
            ball_x = ball_y = ball_speed = ball_accel = np.nan
        else:
            ball_x, ball_y, ball_speed, ball_accel = ball_data

        players_in_frame = player_index.get(f, {})
        candidate = _candidate_possessor(
            (ball_x, ball_y) if np.isfinite(ball_x) else (np.nan, np.nan),
            players_in_frame,
            cfg,
        )

        state, track_id, is_touch = _update_fsm(
            fsm, candidate, ball_speed, ball_accel, cfg
        )

        if is_touch and track_id is not None:
            fsm.last_touch_frame = f
            fsm.last_touch_track_id = track_id

        if cur_spell is None or cur_spell["track_id"] != track_id or cur_spell["team"] != state:
            if cur_spell is not None:
                cur_spell["end_frame"] = f - 1
                cur_spell["end_time_s"] = (f - 1) / fps_eff
                cur_spell["duration_s"] = cur_spell["end_time_s"] - cur_spell["start_time_s"]
                if cur_spell["n_frames"] >= cfg.min_possession_frames:
                    spells.append(cur_spell)
                spell_id += 1
            if state != LOOSE and track_id is not None:
                cur_spell = {
                    "spell_id": spell_id,
                    "start_frame": f,
                    "end_frame": f,
                    "start_time_s": f / fps_eff,
                    "end_time_s": f / fps_eff,
                    "duration_s": 0.0,
                    "team": state,
                    "track_id": track_id,
                    "n_frames": 1,
                    "n_touches": 1 if is_touch else 0,
                }
            else:
                cur_spell = None
        elif cur_spell is not None:
            cur_spell["end_frame"] = f
            cur_spell["end_time_s"] = f / fps_eff
            cur_spell["n_frames"] += 1
            if is_touch:
                cur_spell["n_touches"] += 1

        cand_tid = candidate[0] if candidate else None
        cand_team = candidate[1] if candidate else None
        cand_dist = candidate[2] if candidate else np.nan

        rows.append(
            {
                "frame": f,
                "time_s": f / fps_eff,
                "ball_x_m": ball_x,
                "ball_y_m": ball_y,
                "ball_speed_mps": ball_speed,
                "ball_accel_mps2": ball_accel,
                "possessor_team": state,
                "possessor_track_id": track_id,
                "state": state,
                "is_touch": bool(is_touch),
                "candidate_track_id": cand_tid,
                "candidate_team": cand_team,
                "candidate_dist_m": cand_dist,
            }
        )

    if cur_spell is not None:
        cur_spell["end_frame"] = all_frames[-1]
        cur_spell["end_time_s"] = all_frames[-1] / fps_eff
        cur_spell["duration_s"] = cur_spell["end_time_s"] - cur_spell["start_time_s"]
        if cur_spell["n_frames"] >= cfg.min_possession_frames:
            spells.append(cur_spell)

    timeline = pd.DataFrame.from_records(rows)
    spells_df = pd.DataFrame.from_records(spells) if spells else pd.DataFrame(
        columns=[
            "spell_id", "start_frame", "end_frame", "team", "track_id",
            "n_frames", "n_touches", "start_time_s", "end_time_s", "duration_s",
        ]
    )
    per_track = _per_track_stats(timeline, spells_df, players, fps_eff)

    return PossessionResult(
        timeline=timeline, per_track=per_track, spells=spells_df, config=cfg
    )


def _central_diff_pair(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Diferencias centrales 1D con NaN en los bordes."""
    n = y.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 3:
        return out
    dt = t[2:] - t[:-2]
    valid = dt > 0
    out[1:-1] = np.where(valid, (y[2:] - y[:-2]) / dt, np.nan)
    return out


def _infer_fps(
    frames: list[int],
    players: pd.DataFrame,
    ball: pd.DataFrame,
) -> float:
    """Heuristica: usa el FPS del campo si lo encuentra; en su defecto,
    infiere desde el frame rate observado en la primera secuencia."""
    if "time_s" in ball.columns and ball["time_s"].notna().any():
        t = ball["time_s"].to_numpy()
        if t.size > 1:
            dt = np.diff(t)
            dt = dt[dt > 0]
            if dt.size:
                return float(1.0 / np.median(dt))
    if not frames or len(frames) < 2:
        return 25.0
    return 25.0


def _per_track_stats(
    timeline: pd.DataFrame,
    spells: pd.DataFrame,
    players: pd.DataFrame,
    fps: float,
) -> pd.DataFrame:
    """Resumen por track_id: tiempo en posesion, toques, posesiones iniciadas."""
    if timeline.empty:
        return pd.DataFrame(columns=[
            "track_id", "team", "time_in_possession_s",
            "n_touches", "n_possessions", "n_frames",
        ])

    team_lookup: dict[int, str] = {}
    if not players.empty and "team" in players.columns:
        for tid, t in zip(players["track_id"].to_numpy(), players["team"].to_numpy()):
            team_lookup.setdefault(int(tid), str(t))

    counts = (
        timeline[timeline["possessor_track_id"].notna()]
        .groupby("possessor_track_id")
        .size()
        .rename("n_frames")
    )
    touches = (
        timeline[timeline["is_touch"]]
        .groupby("possessor_track_id")
        .size()
        .rename("n_touches")
    )
    started = (
        spells.groupby("track_id").size().rename("n_possessions")
        if not spells.empty
        else pd.Series(dtype=int, name="n_possessions")
    )

    df = counts.to_frame()
    df = df.join(touches, how="left").join(started, how="left")
    df["n_touches"] = df["n_touches"].fillna(0).astype(int)
    df["n_possessions"] = df["n_possessions"].fillna(0).astype(int)
    df["time_in_possession_s"] = df["n_frames"].astype(float) / fps
    df = df.reset_index().rename(columns={"possessor_track_id": "track_id"})
    df["track_id"] = df["track_id"].astype(int)
    df["team"] = df["track_id"].map(team_lookup).fillna("unknown")
    return df[["track_id", "team", "time_in_possession_s",
               "n_touches", "n_possessions", "n_frames"]]


def possession_percentages(timeline: pd.DataFrame) -> dict[str, float]:
    """Devuelve porcentaje de frames en posesion por equipo y 'loose'."""
    if timeline.empty or "possessor_team" not in timeline.columns:
        return {}
    counts = timeline["possessor_team"].value_counts(normalize=True)
    return {str(k): float(v * 100.0) for k, v in counts.items()}
