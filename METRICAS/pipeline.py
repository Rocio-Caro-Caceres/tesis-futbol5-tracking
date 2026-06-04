"""
pipeline.py
===========

Orquestador del modulo de metricas cinematicas (Etapa 2).

Encadena las cinco etapas en un solo llamado:

    cargar datos Metrica (kloppy)
        -> reescalar a metros  (metrica_loader)
        -> suavizar X,Y        (smoothing)
        -> calcular v,|v|,a    (kinematics)
        -> calcular distancia  (distance)
        -> generar heatmaps    (heatmap)

Devuelve un ``PipelineResult`` con el DataFrame enriquecido, los
heatmaps por track_id y un resumen por jugador (distancia total,
velocidad maxima, etc.).

CLI
---

    python -m METRICAS --input sample.csv --output-dir out/
    python -m METRICAS --input data/ --fps 25 --smooth-window 5 \\
        --min-step 0.03 --bins 50 --sigma 1.5 --output-dir out/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    load_via_kloppy,
)
from .possession import (
    PossessionConfig,
    PossessionResult,
    compute_possession,
    possession_percentages,
)
from .smoothing import smooth_positions


@dataclass
class PipelineConfig:
    """Parametros del pipeline. Todos tienen valores por defecto razonables."""

    fps: float = 25.0
    max_speed_mps: float = 12.0
    smooth_window: int = 5
    smooth_min_periods: int | None = None
    min_step_m: float = DEFAULT_MIN_STEP_M
    heatmap_bins: int = 50
    heatmap_sigma: float = 1.5
    heatmap_normalize: bool = False
    field_dims: FieldDims = field(
        default_factory=lambda: FieldDims(
            length_m=DEFAULT_FIELD_LENGTH_M, width_m=DEFAULT_FIELD_WIDTH_M
        )
    )
    smoothing: bool = True
    compute_possession: bool = False
    possession: PossessionConfig = field(default_factory=PossessionConfig)
    compute_events: bool = False
    events: EventsConfig = field(default_factory=EventsConfig)
    home_attacks_x: float = +1.0
    goalkeeper_track_ids: tuple[int, ...] = ()

    @property
    def field(self) -> FieldDims:
        return self.field_dims


@dataclass
class PipelineResult:
    """Salida del pipeline."""

    metrics: pd.DataFrame
    per_track_summary: pd.DataFrame
    heatmaps: dict[str, dict[str, Any]]
    config: PipelineConfig
    match_id: str
    possession: PossessionResult | None = None
    events: EventsResult | None = None


def _load(input_path: Path, config: PipelineConfig, match_id: str | None) -> pd.DataFrame:
    """Carga el input. Si es CSV, asume que ya esta en metros (o lo detecta).
    Si es cualquier otra cosa, intenta kloppy."""
    if not input_path.exists():
        raise FileNotFoundError(f"No existe: {input_path}")
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        if "x_m" not in df.columns or "y_m" not in df.columns:
            if "x_norm" in df.columns and "y_norm" in df.columns:
                from .metrica_loader import rescale_norm_to_meters
                x_m, y_m = rescale_norm_to_meters(
                    df["x_norm"].to_numpy(), df["y_norm"].to_numpy(), config.field
                )
                df["x_m"] = x_m
                df["y_m"] = y_m
        return df
    return load_via_kloppy(input_path, field=config.field, match_id=match_id)


def _per_track_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calcula velocidad maxima, distancia total y aceleracion maxima por track_id."""
    if "track_id" not in metrics.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for track_id, sub in metrics.groupby("track_id", sort=False):
        n_obs = int(sub.shape[0])
        team = sub["team"].iloc[0] if "team" in sub.columns else None
        max_speed = _safe_nanmax(sub["speed_mps"].to_numpy()) if "speed_mps" in sub else np.nan
        max_accel = _safe_nanmax(sub["accel_mps2"].to_numpy()) if "accel_mps2" in sub else np.nan
        total = _safe_nanmax(sub["cumdist_m"].to_numpy()) if "cumdist_m" in sub else np.nan
        rows.append(
            {
                "track_id": track_id,
                "team": team,
                "n_obs": n_obs,
                "max_speed_mps": max_speed,
                "max_accel_mps2": max_accel,
                "total_distance_m": total,
            }
        )
    return pd.DataFrame(rows)


def _safe_nanmax(arr: np.ndarray) -> float:
    """np.nanmax que devuelve NaN silenciosamente si la entrada es todo NaN o vacia."""
    if arr.size == 0:
        return float("nan")
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _sanitize_json(obj: Any) -> Any:
    """Reemplaza NaN/Inf por None y recurre sobre dicts/listas."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    config: PipelineConfig | None = None,
    match_id: str | None = None,
) -> PipelineResult:
    """
    Ejecuta la cadena completa sobre un archivo de tracking.

    Parameters
    ----------
    input_path : str | Path
        Archivo CSV (en metros o con x_norm/y_norm) o formato soportado
        por kloppy (Metrica Sports).
    output_dir : str | Path, optional
        Si se da, escribe:
            ``<output_dir>/players_metrics.csv``
            ``<output_dir>/summary_per_track.csv``
            ``<output_dir>/summary_match.json``
            ``<output_dir>/heatmaps/<track_id>.npz`` y ``.csv``
    config : PipelineConfig, optional
    match_id : str, optional

    Returns
    -------
    PipelineResult
    """
    cfg = config or PipelineConfig()
    df = _load(Path(input_path), cfg, match_id)
    if df.empty:
        raise ValueError(f"Input {input_path} no produjo filas.")

    if "time_s" not in df.columns and "frame" in df.columns:
        df["time_s"] = df["frame"].astype(float) / float(cfg.fps)

    df = smooth_positions(
        df,
        window=cfg.smooth_window,
        min_periods=cfg.smooth_min_periods,
        x_col="x_m",
        y_col="y_m",
        out_x="x_m_smooth",
        out_y="y_m_smooth",
        group_col="track_id",
    )

    base_cols = list(df.columns)
    df = compute_kinematics(
        df,
        fps=cfg.fps,
        max_speed=cfg.max_speed_mps,
        smoothing=False,  # ya suavizamos arriba
        smooth_window=cfg.smooth_window,
        group_col="track_id",
        time_col="time_s",
        x_col="x_m_smooth",
        y_col="y_m_smooth",
    )

    df = cumulative_distance(
        df,
        min_step=cfg.min_step_m,
        group_col="track_id",
        x_col="x_m_smooth",
        y_col="y_m_smooth",
        time_col="time_s",
    )

    summary = _per_track_summary(df)
    if "cumdist_m" in df.columns and "total_distance_m" not in summary.columns:
        totals = total_distance_per_track(df)
        summary = summary.merge(totals, on="track_id", how="left")

    heatmaps: dict[str, dict[str, Any]] = {}
    if "track_id" in df.columns:
        for tid, sub in df.groupby("track_id"):
            try:
                heatmaps[str(tid)] = build_heatmap(
                    sub,
                    x_col="x_m_smooth",
                    y_col="y_m_smooth",
                    bins=cfg.heatmap_bins,
                    field=cfg.field,
                    sigma=cfg.heatmap_sigma,
                    normalize=cfg.heatmap_normalize,
                )
            except ValueError:
                continue

    possession_result: PossessionResult | None = None
    if cfg.compute_possession:
        possession_result = compute_possession(
            df, fps=cfg.fps, config=cfg.possession
        )
        if "track_id" in df.columns and "team" in df.columns:
            team_lookup: dict[int, str] = {}
            for tid, t in zip(df["track_id"].to_numpy(), df["team"].to_numpy()):
                team_lookup.setdefault(int(tid), str(t))
            possession_result.per_track["team"] = (
                possession_result.per_track["track_id"]
                .map(team_lookup)
                .fillna(possession_result.per_track["team"])
            )
        if not summary.empty and not possession_result.per_track.empty:
            summary = summary.merge(
                possession_result.per_track[
                    ["track_id", "time_in_possession_s", "n_touches", "n_possessions"]
                ],
                on="track_id",
                how="left",
            )
            for col in ("time_in_possession_s", "n_touches", "n_possessions"):
                if col in summary.columns:
                    summary[col] = summary[col].fillna(0)

    events_result: EventsResult | None = None
    if cfg.compute_events:
        events_cfg = EventsConfig(
            field_dims=cfg.field_dims,
            goalkeeper_track_ids=set(cfg.goalkeeper_track_ids),
            home_attacks_x=cfg.home_attacks_x,
        )
        events_result = detect_events(
            df, fps=cfg.fps, config=events_cfg,
            match_id=match_id, home_attacks_x=cfg.home_attacks_x,
        )

    resolved_match_id = match_id or str(df.get("match_id", pd.Series(["unknown"])).iloc[0])

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "players_metrics.csv", index=False, float_format="%.4f")
        summary.to_csv(out / "summary_per_track.csv", index=False)
        match_summary = {
            "match_id": resolved_match_id,
            "n_rows": int(len(df)),
            "n_tracks": int(summary.shape[0]) if not summary.empty else 0,
            "config": {
                "fps": cfg.fps,
                "smooth_window": cfg.smooth_window,
                "min_step_m": cfg.min_step_m,
                "max_speed_mps": cfg.max_speed_mps,
                "heatmap_bins": cfg.heatmap_bins,
                "heatmap_sigma": cfg.heatmap_sigma,
                "field": {
                    "length_m": cfg.field.length_m,
                    "width_m": cfg.field.width_m,
                },
            },
            "per_track": summary.to_dict(orient="records"),
        }
        if possession_result is not None:
            match_summary["possession"] = {
                "percentages": possession_percentages(possession_result.timeline),
                "n_spells": int(len(possession_result.spells)),
                "config": {
                    "possession_radius_m": cfg.possession.possession_radius_m,
                    "control_radius_m": cfg.possession.control_radius_m,
                    "steal_radius_m": cfg.possession.steal_radius_m,
                    "steal_min_frames": cfg.possession.steal_min_frames,
                    "touch_accel_mps2": cfg.possession.touch_accel_mps2,
                    "min_possession_frames": cfg.possession.min_possession_frames,
                },
            }
        if events_result is not None:
            match_summary["events"] = {
                "n_goals": int(len(events_result.goals)),
                "n_saves": int(len(events_result.saves)),
                "home_attacks_x": cfg.home_attacks_x,
            }
        (out / "summary_match.json").write_text(
            json.dumps(_sanitize_json(match_summary), indent=2, default=str),
            encoding="utf-8",
        )
        heat_dir = out / "heatmaps"
        for tid, hm in heatmaps.items():
            safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(tid))
            save_heatmap_npy(hm, heat_dir / f"{safe}.npz")
            save_heatmap_csv(hm, heat_dir / f"{safe}.csv")
        if possession_result is not None:
            possession_result.timeline.to_csv(
                out / "possession_timeline.csv", index=False, float_format="%.4f"
            )
            possession_result.per_track.to_csv(
                out / "possession_per_track.csv", index=False
            )
            possession_result.spells.to_csv(
                out / "possession_spells.csv", index=False
            )
        if events_result is not None:
            events_result.events.to_csv(out / "events.csv", index=False)
            if not events_result.goals.empty:
                events_result.goals.to_csv(out / "goals.csv", index=False)
            if not events_result.saves.empty:
                events_result.saves.to_csv(out / "saves.csv", index=False)

    return PipelineResult(
        metrics=df,
        per_track_summary=summary,
        heatmaps=heatmaps,
        config=cfg,
        match_id=resolved_match_id,
        possession=possession_result,
        events=events_result,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m METRICAS",
        description="Pipeline de metricas cinematicas basicas (Etapa 2).",
    )
    p.add_argument("--input", required=True,
                   help="Archivo de tracking (CSV en metros/norm, o formato kloppy).")
    p.add_argument("--output-dir", default=None,
                   help="Directorio de salida. Si se omite, no escribe archivos.")
    p.add_argument("--match-id", default=None,
                   help="ID del partido (default: nombre del archivo).")
    p.add_argument("--fps", type=float, default=25.0,
                   help="Frames por segundo (default 25).")
    p.add_argument("--field-length", type=float, default=DEFAULT_FIELD_LENGTH_M,
                   help="Largo del campo en metros (default 105).")
    p.add_argument("--field-width", type=float, default=DEFAULT_FIELD_WIDTH_M,
                   help="Ancho del campo en metros (default 68).")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="Ventana de la media movil central (default 5).")
    p.add_argument("--min-step", type=float, default=DEFAULT_MIN_STEP_M,
                   help="Umbral de paso para distancia acumulada (default 0.03 m).")
    p.add_argument("--max-speed", type=float, default=12.0,
                   help="Tope de velocidad (m/s, default 12). 0 desactiva.")
    p.add_argument("--bins", type=int, default=50,
                   help="Bins por eje del heatmap (default 50).")
    p.add_argument("--sigma", type=float, default=1.5,
                   help="Sigma del filtro gaussiano en bins (default 1.5).")
    p.add_argument("--no-smoothing", action="store_true",
                   help="Saltea la etapa de suavizado.")
    p.add_argument("--normalize-heatmap", action="store_true",
                   help="Normaliza el heatmap para que integre a 1.")
    p.add_argument("--possession", action="store_true",
                   help="Habilita la FSM de posesion (Etapa 3).")
    p.add_argument("--possession-radius", type=float,
                   default=PossessionConfig.possession_radius_m,
                   help="Radio maximo (m) para asignar posesion (default 2.0).")
    p.add_argument("--control-radius", type=float,
                   default=PossessionConfig.control_radius_m,
                   help="Radio (m) para considerar 'control' del balon (default 1.0).")
    p.add_argument("--steal-radius", type=float,
                   default=PossessionConfig.steal_radius_m,
                   help="Radio (m) de robo entre jugadores (default 1.0).")
    p.add_argument("--steal-min-frames", type=int,
                   default=PossessionConfig.steal_min_frames,
                   help="Frames consecutivos para confirmar un robo (default 3).")
    p.add_argument("--touch-accel", type=float,
                   default=PossessionConfig.touch_accel_mps2,
                   help="Umbral |dv_ball| (m/s^2) para marcar touch (default 4.0).")
    p.add_argument("--events", action="store_true",
                   help="Habilita deteccion geometrica de eventos (Etapa 3).")
    p.add_argument("--goal-depth", type=float,
                   default=DEFAULT_GOAL_DEPTH_M,
                   help="Profundidad del arco detras de la linea de meta (m, default 2.0).")
    p.add_argument("--goal-width", type=float,
                   default=DEFAULT_GOAL_WIDTH_M,
                   help="Ancho del arco (m, default 7.32 = FIFA).")
    p.add_argument("--shot-speed", type=float,
                   default=DEFAULT_SHOT_SPEED_MPS,
                   help="Velocidad minima de balon para 'tiro' (m/s, default 5.0).")
    p.add_argument("--save-response", type=float,
                   default=DEFAULT_SAVE_RESPONSE_S,
                   help="Ventana (s) para detectar una atajada (default 1.0).")
    p.add_argument("--keeper-radius", type=float,
                   default=DEFAULT_KEEPER_INTERCEPT_RADIUS_M,
                   help="Distancia maxima portero-balon en una atajada (m, default 1.5).")
    p.add_argument("--cooldown", type=float,
                   default=DEFAULT_COOLDOWN_S,
                   help="Tiempo minimo entre eventos iguales (s, default 4.0).")
    p.add_argument("--home-attacks-x", type=float, default=+1.0,
                   help="+1.0 si home ataca el arco +x, -1.0 si ataca el -x (default +1.0).")
    p.add_argument("--keeper-ids", type=int, nargs="*", default=(),
                   help="track_ids de guardametas (espacio-separados). Si vacio, se infiere por equipo.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = PipelineConfig(
        fps=args.fps,
        max_speed_mps=args.max_speed,
        smooth_window=args.smooth_window,
        min_step_m=args.min_step,
        heatmap_bins=args.bins,
        heatmap_sigma=args.sigma,
        heatmap_normalize=args.normalize_heatmap,
        field_dims=FieldDims(length_m=args.field_length, width_m=args.field_width),
        smoothing=not args.no_smoothing,
        compute_possession=args.possession,
        possession=PossessionConfig(
            possession_radius_m=args.possession_radius,
            control_radius_m=args.control_radius,
            steal_radius_m=args.steal_radius,
            steal_min_frames=args.steal_min_frames,
            touch_accel_mps2=args.touch_accel,
        ),
        compute_events=args.events,
        events=EventsConfig(
            field_dims=FieldDims(length_m=args.field_length, width_m=args.field_width),
            goal_depth_m=args.goal_depth,
            goal_width_m=args.goal_width,
            shot_speed_mps=args.shot_speed,
            save_response_s=args.save_response,
            keeper_intercept_radius_m=args.keeper_radius,
            cooldown_s=args.cooldown,
            home_attacks_x=args.home_attacks_x,
            goalkeeper_track_ids=set(args.keeper_ids or ()),
        ),
        home_attacks_x=args.home_attacks_x,
        goalkeeper_track_ids=tuple(args.keeper_ids or ()),
    )
    result = run_pipeline(
        args.input, output_dir=args.output_dir, config=cfg, match_id=args.match_id
    )
    extras = []
    if result.possession is not None:
        pct = possession_percentages(result.possession.timeline)
        extras.append(f"posesion={pct}")
    if result.events is not None:
        extras.append(
            f"goles={len(result.events.goals)} atajadas={len(result.events.saves)}"
        )
    print(
        f"OK: {result.metrics.shape[0]} filas, "
        f"{result.per_track_summary.shape[0]} tracks, "
        f"{len(result.heatmaps)} heatmaps"
        + (f" | {' '.join(extras)}" if extras else "")
    )
    if args.output_dir:
        print(f"Salida en: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
