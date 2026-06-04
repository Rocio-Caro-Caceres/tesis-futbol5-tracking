"""
events.py
=========

Deteccion geometrica de eventos de campo (goles y atajadas) a partir
del DataFrame de tracking y la linea temporal de posesion.

Eventos soportados
------------------

1. **Gol** (``event_type='goal'``)
   La coordenada del balon intersecta el poligono rectangular
   definido para el area de goal (la porcion del campo "detras" de
   la linea de meta, de profundidad ``goal_depth_m`` y ancho
   ``goal_width_m``). El equipo que anota se infiere a partir de
   ``home_attacks_x``: si el balon cruza el arco del lado -x, anota
   el equipo que ataca en +x (visitante si ``home_attacks_x=+1``).

2. **Atajada** (``event_type='save'``)
   Se dispara cuando se cumple la siguiente secuencia dentro de una
   ventana de ``save_response_s`` segundos:
     a) el balon se mueve hacia el arco con ``speed >
        shot_speed_mps``;
     b) un jugador identificado como guardameta (track_id en
        ``goalkeeper_track_ids``) esta a menos de
        ``keeper_intercept_radius_m`` del balon en el momento de
        maxima cercania;
     c) la rapidez del balon cae por debajo de
        ``shot_speed_mps * save_decel_factor`` dentro de la ventana.

   El equipo acreditado es el **defensor** (el que protege el arco
   amenazado).

Geometria
---------

- Campo con origen en el centro: ``+x`` derecha, ``+y`` arriba.
- Arcos centrados en ``y=0``, ancho ``goal_width_m``, profundidad
  ``goal_depth_m`` por detras de la linea de meta.
- ``home_attacks_x = +1.0``: home ataca el arco de +x, defiende el
  de -x. ``home_attacks_x = -1.0`` invierte.

Antiespuma
----------

Cada detector lleva un ``cooldown_s`` para evitar disparos
repetidos por el mismo evento fisico (balon rebotando dentro del
arco, multiples contactos cerca del portero, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .metrica_loader import FieldDims
from .possession import BALL_TRACK_ID, LOOSE


DEFAULT_GOAL_DEPTH_M: float = 2.0
DEFAULT_GOAL_WIDTH_M: float = 7.32
DEFAULT_SHOT_SPEED_MPS: float = 5.0
DEFAULT_SAVE_RESPONSE_S: float = 1.0
DEFAULT_SAVE_DECEL_FACTOR: float = 0.5
DEFAULT_KEEPER_INTERCEPT_RADIUS_M: float = 1.5
DEFAULT_COOLDOWN_S: float = 4.0
DEFAULT_GOAL_PROXIMITY_M: float = 1.0


@dataclass
class EventsConfig:
    """Parametros de la deteccion geometrica de eventos."""

    field_dims: FieldDims = field(default_factory=FieldDims)
    goal_depth_m: float = DEFAULT_GOAL_DEPTH_M
    goal_width_m: float = DEFAULT_GOAL_WIDTH_M
    shot_speed_mps: float = DEFAULT_SHOT_SPEED_MPS
    save_response_s: float = DEFAULT_SAVE_RESPONSE_S
    save_decel_factor: float = DEFAULT_SAVE_DECEL_FACTOR
    keeper_intercept_radius_m: float = DEFAULT_KEEPER_INTERCEPT_RADIUS_M
    goal_proximity_m: float = DEFAULT_GOAL_PROXIMITY_M
    cooldown_s: float = DEFAULT_COOLDOWN_S
    home_attacks_x: float = +1.0
    goalkeeper_track_ids: set[int] = field(default_factory=set)
    goalkeeper_team: str = "home"

    @property
    def field(self) -> FieldDims:
        return self.field_dims


@dataclass
class Event:
    """Un evento discreto detectado."""

    event_type: str
    frame: int
    time_s: float
    x_m: float
    y_m: float
    team: str
    actor_track_id: int | None
    target_track_id: int | None
    side: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventsResult:
    """Salida de :func:`detect_events`."""

    events: pd.DataFrame
    goals: pd.DataFrame
    saves: pd.DataFrame
    config: EventsConfig
    home_attacks_x: float


def _split_long_df(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "track_id" not in df.columns:
        raise KeyError("DataFrame sin columna 'track_id'")
    if "object_type" in df.columns:
        is_ball = df["object_type"].astype(str).str.lower() == "ball"
    else:
        is_ball = df["track_id"].to_numpy() == BALL_TRACK_ID
    return df.loc[~is_ball].copy(), df.loc[is_ball].copy()


def _ball_trajectory(
    ball: pd.DataFrame,
) -> pd.DataFrame:
    """Ordena el balon por frame y anade velocidad por frame si no existe."""
    if ball.empty:
        return ball
    out = ball.sort_values("frame").reset_index(drop=True).copy()
    if "speed_mps" not in out.columns:
        out["speed_mps"] = np.nan
    if "vx_mps" not in out.columns:
        out["vx_mps"] = np.nan
    if "vy_mps" not in out.columns:
        out["vy_mps"] = np.nan
    return out


def _goal_polygon(side: str, cfg: EventsConfig) -> np.ndarray:
    """Devuelve los 4 vertices del poligono del arco en sentido horario.

    ``side='+x'``  -> arco al borde derecho, poligono por DETRAS de la
    linea de meta (x > length/2).
    ``side='-x'``  -> arco al borde izquierdo, poligono por DETRAS de
    la linea de meta (x < -length/2).
    """
    L = cfg.field.length_m
    hw = cfg.goal_width_m / 2.0
    d = cfg.goal_depth_m
    if side == "+x":
        x_outer = L / 2.0 + d
        x_inner = L / 2.0
    elif side == "-x":
        x_outer = -L / 2.0 - d
        x_inner = -L / 2.0
    else:
        raise ValueError(f"side invalido: {side!r}")
    return np.array(
        [
            [x_inner, -hw],
            [x_outer, -hw],
            [x_outer, +hw],
            [x_inner, +hw],
        ],
        dtype=np.float64,
    )


def _point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Test de punto en poligono convexo (ray casting) sobre los 4
    vertices del arco. Robusto contra NaN."""
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _scoring_team(side: str, home_attacks_x: float) -> tuple[str, str]:
    """Dado el lado del arco, devuelve (scoring_team, defending_team)."""
    if side == "+x":
        if home_attacks_x > 0:
            return "home", "away"
        return "away", "home"
    if home_attacks_x > 0:
        return "away", "home"
    return "home", "away"


def _infer_fps(
    ball: pd.DataFrame,
    fallback: float | None = None,
) -> float:
    if "time_s" in ball.columns and ball["time_s"].notna().any():
        t = ball["time_s"].to_numpy()
        if t.size > 1:
            dt = np.diff(t)
            dt = dt[dt > 0]
            if dt.size:
                return float(1.0 / np.median(dt))
    if fallback:
        return float(fallback)
    return 25.0


def _central_diff_1d_arr(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Diferencias centrales 1D con NaN en los bordes."""
    n = y.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < 3:
        return out
    dt = t[2:] - t[:-2]
    valid = dt > 0
    out[1:-1] = np.where(valid, (y[2:] - y[:-2]) / dt, np.nan)
    return out


def _detect_goals(
    ball: pd.DataFrame,
    cfg: EventsConfig,
    fps: float,
) -> list[Event]:
    """Detecta transiciones de la pelota desde fuera hacia dentro de
    cualquiera de los dos poligonos de arco."""
    goals: list[Event] = []
    if ball.empty:
        return goals

    poly_pos = _goal_polygon("+x", cfg)
    poly_neg = _goal_polygon("-x", cfg)

    last_inside_pos = False
    last_inside_neg = False
    last_event_time = {"+x": -1e9, "-x": -1e9}

    for _, row in ball.iterrows():
        f = int(row["frame"])
        t = float(row.get("time_s", f / fps))
        x = float(row["x_m"])
        y = float(row["y_m"])
        if not (np.isfinite(x) and np.isfinite(y)):
            last_inside_pos = last_inside_neg = False
            continue

        inside_pos = _point_in_polygon(x, y, poly_pos)
        inside_neg = _point_in_polygon(x, y, poly_neg)

        if inside_pos and not last_inside_pos:
            if (t - last_event_time["+x"]) >= cfg.cooldown_s:
                scoring, _ = _scoring_team("+x", cfg.home_attacks_x)
                vx = float(row.get("vx_mps", np.nan) or 0.0)
                goals.append(Event(
                    event_type="goal",
                    frame=f, time_s=t, x_m=x, y_m=y,
                    team=scoring, actor_track_id=None, target_track_id=None,
                    side="+x",
                    metadata={"ball_vx_mps": vx},
                ))
                last_event_time["+x"] = t

        if inside_neg and not last_inside_neg:
            if (t - last_event_time["-x"]) >= cfg.cooldown_s:
                scoring, _ = _scoring_team("-x", cfg.home_attacks_x)
                vx = float(row.get("vx_mps", np.nan) or 0.0)
                goals.append(Event(
                    event_type="goal",
                    frame=f, time_s=t, x_m=x, y_m=y,
                    team=scoring, actor_track_id=None, target_track_id=None,
                    side="-x",
                    metadata={"ball_vx_mps": vx},
                ))
                last_event_time["-x"] = t

        last_inside_pos = inside_pos
        last_inside_neg = inside_neg

    return goals


def _player_lookup(
    players: pd.DataFrame,
) -> dict[int, dict[int, tuple[float, float, str]]]:
    out: dict[int, dict[int, tuple[float, float, str]]] = {}
    if players.empty:
        return out
    team_col = "team" if "team" in players.columns else None
    teams = players[team_col].to_numpy() if team_col else np.array(["unknown"] * len(players))
    for f, tid, x, y, team in zip(
        players["frame"].to_numpy(),
        players["track_id"].to_numpy(),
        players["x_m"].to_numpy(),
        players["y_m"].to_numpy(),
        teams,
    ):
        out.setdefault(int(f), {})[int(tid)] = (float(x), float(y), str(team))
    return out


def _keepers_in_frame(
    players_in_frame: dict[int, tuple[float, float, str]],
    cfg: EventsConfig,
) -> list[tuple[int, str, float, float]]:
    """Devuelve [(track_id, team, x, y), ...] de los guardametas en el frame."""
    out: list[tuple[int, str, float, float]] = []
    for tid, (x, y, team) in players_in_frame.items():
        if cfg.goalkeeper_track_ids and tid in cfg.goalkeeper_track_ids:
            out.append((tid, team, x, y))
        elif cfg.goalkeeper_team and team == cfg.goalkeeper_team:
            out.append((tid, team, x, y))
    return out


def _detect_saves(
    ball: pd.DataFrame,
    players: pd.DataFrame,
    cfg: EventsConfig,
    fps: float,
) -> list[Event]:
    """Heuristica: balon rapido hacia el arco + portero cerca + desaceleracion."""
    saves: list[Event] = []
    if ball.empty or players.empty:
        return saves

    L = cfg.field.length_m
    player_idx = _player_lookup(players)
    ball_sorted = _ball_trajectory(ball)

    frames = ball_sorted["frame"].to_numpy(dtype=int)
    xs = ball_sorted["x_m"].to_numpy(dtype=float)
    ys = ball_sorted["y_m"].to_numpy(dtype=float)
    speeds = ball_sorted["speed_mps"].to_numpy(dtype=float)
    vxs = ball_sorted["vx_mps"].to_numpy(dtype=float) if "vx_mps" in ball_sorted.columns else np.zeros(len(ball_sorted))
    times = ball_sorted["time_s"].to_numpy(dtype=float) if "time_s" in ball_sorted.columns else frames.astype(float) / fps

    last_save_time = -1e9

    for i in range(len(frames) - 1):
        x, y, vx, sp, t = xs[i], ys[i], vxs[i], speeds[i], times[i]
        f = int(frames[i])
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(sp)):
            continue
        if sp < cfg.shot_speed_mps:
            continue

        toward_pos = (x > 0 and vx > 0 and (L / 2.0 - x) <= cfg.goal_proximity_m)
        toward_neg = (x < 0 and vx < 0 and (x - (-L / 2.0)) <= cfg.goal_proximity_m)
        if not (toward_pos or toward_neg):
            continue

        keepers = _keepers_in_frame(player_idx.get(f, {}), cfg)
        if not keepers:
            continue
        kx, ky, ktid, kteam = _closest_keeper(x, y, keepers)
        if np.hypot(x - kx, y - ky) > cfg.keeper_intercept_radius_m:
            continue

        decel_target = cfg.shot_speed_mps * cfg.save_decel_factor
        end_t = t + cfg.save_response_s
        saved = False
        for j in range(i + 1, len(frames)):
            tj = times[j]
            if tj > end_t:
                break
            sj = speeds[j]
            if np.isfinite(sj) and sj < decel_target:
                saved = True
                f_save = int(frames[j])
                x_save, y_save = xs[j], ys[j]
                break
        if not saved:
            continue

        if (t - last_save_time) < cfg.cooldown_s:
            continue
        last_save_time = t

        side = "+x" if toward_pos else "-x"
        _, defending = _scoring_team(side, cfg.home_attacks_x)
        saves.append(Event(
            event_type="save",
            frame=f_save, time_s=tj,
            x_m=float(x_save), y_m=float(y_save),
            team=defending, actor_track_id=ktid,
            target_track_id=None, side=side,
            metadata={
                "keeper_team": kteam,
                "ball_speed_at_shot_mps": float(sp),
                "ball_speed_after_mps": float(sj),
                "keeper_intercept_dist_m": float(np.hypot(x - kx, y - ky)),
            },
        ))

    return saves


def _closest_keeper(
    bx: float, by: float,
    keepers: list[tuple[int, str, float, float]],
) -> tuple[float, float, int, str]:
    best: tuple[float, float, int, str] | None = None
    best_d = float("inf")
    for tid, team, x, y in keepers:
        d = float(np.hypot(x - bx, y - by))
        if d < best_d:
            best = (x, y, tid, team)
            best_d = d
    assert best is not None
    return best


def _events_to_df(events: list[Event], match_id: str | None = None) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=[
            "event_id", "match_id", "event_type", "frame", "time_s",
            "x_m", "y_m", "team", "actor_track_id", "target_track_id",
            "side", "metadata",
        ])
    rows = []
    for k, e in enumerate(events):
        rows.append({
            "event_id": k,
            "match_id": match_id,
            "event_type": e.event_type,
            "frame": e.frame,
            "time_s": e.time_s,
            "x_m": e.x_m,
            "y_m": e.y_m,
            "team": e.team,
            "actor_track_id": e.actor_track_id,
            "target_track_id": e.target_track_id,
            "side": e.side,
            "metadata": e.metadata,
        })
    return pd.DataFrame.from_records(rows)


def detect_events(
    df: pd.DataFrame,
    *,
    fps: float | None = None,
    config: EventsConfig | None = None,
    match_id: str | None = None,
    home_attacks_x: float | None = None,
) -> EventsResult:
    """Detecta goles y atajadas a partir de un DataFrame largo.

    Parameters
    ----------
    df : DataFrame
        Tracking en formato largo (mezcla jugadores + balon, o se
        filtra internamente por ``track_id == 0`` / ``object_type``).
    fps : float, optional
    config : EventsConfig, optional
    match_id : str, optional
        Se anade como columna ``match_id`` en la salida.
    home_attacks_x : float, optional
        Si se da, sobreescribe ``config.home_attacks_x``.

    Returns
    -------
    EventsResult
    """
    cfg = config or EventsConfig()
    if home_attacks_x is not None:
        cfg.home_attacks_x = float(home_attacks_x)

    players, ball = _split_long_df(df)
    fps_eff = _infer_fps(ball, fps)

    if not ball.empty:
        ball = ball.sort_values("frame").reset_index(drop=True)
        if "time_s" not in ball.columns:
            ball["time_s"] = ball["frame"].astype(float) / float(fps_eff)
        if "speed_mps" not in ball.columns or ball["speed_mps"].isna().all():
            t = ball["time_s"].to_numpy(dtype=float)
            bx = ball["x_m"].to_numpy(dtype=float)
            by = ball["y_m"].to_numpy(dtype=float)
            vx = _central_diff_1d_arr(bx, t)
            vy = _central_diff_1d_arr(by, t)
            ball["vx_mps"] = vx
            ball["vy_mps"] = vy
            speed = np.sqrt(vx * vx + vy * vy)
            ball["speed_mps"] = speed
            ball["accel_mps2"] = _central_diff_1d_arr(speed, t)

    goals = _detect_goals(ball, cfg, fps_eff)
    saves = _detect_saves(ball, players, cfg, fps_eff)
    all_events = sorted(goals + saves, key=lambda e: (e.frame, e.event_type))

    return EventsResult(
        events=_events_to_df(all_events, match_id=match_id),
        goals=_events_to_df(goals, match_id=match_id),
        saves=_events_to_df(saves, match_id=match_id),
        config=cfg,
        home_attacks_x=cfg.home_attacks_x,
    )
