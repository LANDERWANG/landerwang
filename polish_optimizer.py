"""Parameter optimization tool for silicon carbide polishing.

This module models the relationship between key process parameters
(spindle speed, table speed, spindle load, etc.) and three metrics of
interest (material removal, grit wear ratio GR, and cycle time).  A
simple command-line interface is provided to evaluate a single process
configuration or to search across ranges of spindle and table speeds in
order to maximize GR subject to optional constraints.

The numerical model used here is a lightweight combination of
engineering heuristics and scaling rules derived from typical silicon
carbide CMP/PM polishing behaviour.  Although the coefficients are
synthetic, they capture the qualitative trends engineers expect:

* higher relative speed and mechanical load increase removal rate but
  penalise GR.
* C-face wafers polish slightly faster than Si-face due to their lower
  hardness, and they deliver better grit retention.
* fine finishing steps prioritise GR and surface quality at the expense
  of removal rate.

The script can be imported as a library or executed directly.  Example
command lines:

Evaluate a single recipe:

    python polish_optimizer.py \
        --spindle 1800 --table 40 --load 65 \
        --roughness 0.35 --face C --condition coarse

Search a grid of spindle/table speeds for the best GR not exceeding a
180-second polish time:

    python polish_optimizer.py \
        --spindle-range 1400,2200,100 --table-range 20,60,5 \
        --load 60 --roughness 0.25 --face Si --condition fine \
        --max-time 180
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# --- Core constants -------------------------------------------------------

WHEEL_DIAMETER_MM = 304.0
"""Polishing wheel diameter (mm)."""

FEED_SPEED_UM_S = 0.3
"""Feed speed (µm/s).  The model allows overriding this in the future."""

# The carrier/table diameter is not provided, so we assume a typical value
# that keeps the linear speed in a comparable range with the wheel speed.
TABLE_DIAMETER_MM = 200.0

# Default material removal targets (µm) for each process condition.  These
# provide a reasonable stopping depth for the cycle time calculation when
# the user does not specify an explicit target.
DEFAULT_REMOVAL_UM = {
    "coarse": 5.0,
    "fine": 1.5,
}


@dataclass(frozen=True)
class ProcessParameters:
    """Input parameters describing a polishing recipe."""

    spindle_rpm: float
    table_rpm: float
    spindle_load_pct: float
    roughness_ra_um: float
    sample_face: str  # "C" or "Si"
    condition: str  # "coarse" or "fine"
    target_removal_um: Optional[float] = None


@dataclass
class ProcessMetrics:
    """Calculated metrics of interest for the polishing process."""

    removal_um: float
    gr: float
    time_s: float
    removal_rate_um_s: float
    wheel_wear_um: float


@dataclass
class OptimizationResult:
    """Best recipe found during a grid search."""

    parameters: ProcessParameters
    metrics: ProcessMetrics


# --- Calculation helpers --------------------------------------------------
def _face_factor(face: str) -> float:
    face_upper = face.upper()
    if face_upper == "C":
        return 1.05
    if face_upper == "SI":
        return 0.95
    raise ValueError("sample_face must be 'C' or 'Si'")


def _condition_factor(condition: str) -> float:
    condition_lower = condition.lower()
    if condition_lower == "coarse":
        return 1.0
    if condition_lower == "fine":
        return 0.68
    raise ValueError("condition must be 'coarse' or 'fine'")


def _base_gr(condition: str) -> float:
    return 180.0 if condition.lower() == "coarse" else 230.0


def _relative_speed_mm_s(spindle_rpm: float, table_rpm: float) -> float:
    wheel_speed = math.pi * WHEEL_DIAMETER_MM * spindle_rpm / 60.0
    table_speed = math.pi * TABLE_DIAMETER_MM * table_rpm / 60.0
    return math.hypot(wheel_speed, table_speed)


def _removal_rate_um_s(params: ProcessParameters) -> float:
    speed_factor = _relative_speed_mm_s(params.spindle_rpm, params.table_rpm)
    load_factor = 0.45 + params.spindle_load_pct / 170.0
    roughness_factor = 0.85 + 0.12 * math.log10(max(params.roughness_ra_um, 0.03) / 0.03 + 1.0)
    condition_factor = _condition_factor(params.condition)
    face_factor = _face_factor(params.sample_face)

    base_coeff = 4.0e-5
    removal_rate = (
        base_coeff
        * speed_factor
        * load_factor
        * roughness_factor
        * condition_factor
        * face_factor
    )

    # penalise extreme feed influence if we ever allow it to vary
    feed_factor = FEED_SPEED_UM_S / 0.3
    return removal_rate * feed_factor


def _gr_value(params: ProcessParameters, removal_rate_um_s: float) -> float:
    condition_factor = _condition_factor(params.condition)
    face_factor = _face_factor(params.sample_face)
    load_factor = 0.65 + params.spindle_load_pct / 220.0
    roughness_penalty = 1.0 / (1.0 + 0.35 * math.log10(max(params.roughness_ra_um, 0.03) / 0.03 + 1.0))

    # Faster removal tends to reduce GR.  We normalise against a nominal 2 µm/s rate.
    rate_ratio = removal_rate_um_s / 2.0
    speed_penalty = 1.0 / math.sqrt(1.0 + 0.8 * max(rate_ratio - 1.0, 0.0))

    base_gr = _base_gr(params.condition)
    gr = (
        base_gr
        * face_factor
        * condition_factor ** 0.2
        * roughness_penalty
        * speed_penalty
        / math.sqrt(load_factor)
    )

    return max(gr, 10.0)


def evaluate_process(params: ProcessParameters) -> ProcessMetrics:
    """Evaluate polishing outputs for a given recipe."""

    removal_rate = _removal_rate_um_s(params)
    target_removal = (
        params.target_removal_um
        if params.target_removal_um is not None
        else DEFAULT_REMOVAL_UM[params.condition.lower()]
    )

    time_s = target_removal / max(removal_rate, 1e-6)
    gr = _gr_value(params, removal_rate)
    wheel_wear = target_removal / gr

    return ProcessMetrics(
        removal_um=target_removal,
        gr=gr,
        time_s=time_s,
        removal_rate_um_s=removal_rate,
        wheel_wear_um=wheel_wear,
    )


# --- Optimisation ---------------------------------------------------------

def _generate_range(center: float, spec: Optional[Sequence[float]]) -> List[float]:
    if not spec:
        return [center]

    if len(spec) == 1:
        return [spec[0]]

    if len(spec) not in (2, 3):
        raise ValueError("Range specifications must have 1, 2, or 3 values")

    start = spec[0]
    stop = spec[1]
    step = spec[2] if len(spec) == 3 else max((stop - start) / 10.0, 1.0)

    values: List[float] = []
    value = start
    # include stop if reachable within floating point tolerance
    while value <= stop + 1e-9:
        values.append(round(value, 6))
        value += step

    return values


def optimize_parameters(
    params: ProcessParameters,
    spindle_range: Optional[Sequence[float]] = None,
    table_range: Optional[Sequence[float]] = None,
    max_time_s: Optional[float] = None,
    min_removal_um: Optional[float] = None,
) -> OptimizationResult:
    """Grid search for the recipe that maximises GR.

    Args:
        params: Base recipe used for fields not covered by the ranges.
        spindle_range: Optional sequence specifying the spindle RPM search
            space.  One number means a single value.  Two numbers denote a
            start and stop with an automatically generated step.  Three
            numbers provide start, stop, and explicit step.
        table_range: Same structure as ``spindle_range`` for table RPM.
        max_time_s: Optional upper bound on acceptable cycle time.
        min_removal_um: Optional lower bound on acceptable material removal.

    Returns:
        The best (highest GR) recipe satisfying the constraints.

    Raises:
        ValueError: If no candidate satisfies the constraints.
    """

    spindle_candidates = _generate_range(params.spindle_rpm, spindle_range)
    table_candidates = _generate_range(params.table_rpm, table_range)

    best: Optional[OptimizationResult] = None

    for spindle in spindle_candidates:
        for table in table_candidates:
            trial_params = dataclasses.replace(params, spindle_rpm=spindle, table_rpm=table)
            metrics = evaluate_process(trial_params)

            if max_time_s is not None and metrics.time_s > max_time_s:
                continue
            if min_removal_um is not None and metrics.removal_um < min_removal_um:
                continue

            if best is None or metrics.gr > best.metrics.gr:
                best = OptimizationResult(parameters=trial_params, metrics=metrics)

    if best is None:
        raise ValueError("No candidate satisfies the optimisation constraints")

    return best


# --- Command-line interface -----------------------------------------------

def _parse_range(arg: Optional[str]) -> Optional[Tuple[float, ...]]:
    if arg is None:
        return None
    parts = [float(value.strip()) for value in arg.split(",") if value.strip()]
    return tuple(parts) if parts else None


def _metrics_to_dict(metrics: ProcessMetrics) -> dict:
    return {
        "removal_um": metrics.removal_um,
        "gr": metrics.gr,
        "time_s": metrics.time_s,
        "removal_rate_um_s": metrics.removal_rate_um_s,
        "wheel_wear_um": metrics.wheel_wear_um,
    }


def _parameters_to_dict(params: ProcessParameters) -> dict:
    return {
        "spindle_rpm": params.spindle_rpm,
        "table_rpm": params.table_rpm,
        "spindle_load_pct": params.spindle_load_pct,
        "roughness_ra_um": params.roughness_ra_um,
        "sample_face": params.sample_face,
        "condition": params.condition,
        "target_removal_um": params.target_removal_um,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Optimise SiC polishing parameters")
    parser.add_argument("--spindle", type=float, required=True, help="Spindle speed (rpm)")
    parser.add_argument("--table", type=float, required=True, help="Work table speed (rpm)")
    parser.add_argument("--load", type=float, required=True, help="Spindle load (percent)")
    parser.add_argument("--roughness", type=float, required=True, help="Initial surface roughness Ra (µm)")
    parser.add_argument("--face", choices=["C", "Si", "c", "si"], required=True, help="Sample face")
    parser.add_argument("--condition", choices=["coarse", "fine"], required=True, help="Grinding condition")
    parser.add_argument("--target-removal", type=float, help="Override target removal depth (µm)")
    parser.add_argument("--spindle-range", dest="spindle_range", help="Optional spindle range start,stop[,step]")
    parser.add_argument("--table-range", dest="table_range", help="Optional table range start,stop[,step]")
    parser.add_argument("--max-time", type=float, help="Maximum acceptable cycle time (s)")
    parser.add_argument("--min-removal", type=float, help="Minimum removal depth (µm)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")

    args = parser.parse_args(argv)

    params = ProcessParameters(
        spindle_rpm=args.spindle,
        table_rpm=args.table,
        spindle_load_pct=args.load,
        roughness_ra_um=args.roughness,
        sample_face=args.face,
        condition=args.condition,
        target_removal_um=args.target_removal,
    )

    spindle_range = _parse_range(args.spindle_range)
    table_range = _parse_range(args.table_range)

    if spindle_range or table_range:
        result = optimize_parameters(
            params,
            spindle_range=spindle_range,
            table_range=table_range,
            max_time_s=args.max_time,
            min_removal_um=args.min_removal,
        )
        output = {
            "mode": "optimisation",
            "parameters": _parameters_to_dict(result.parameters),
            "metrics": _metrics_to_dict(result.metrics),
        }
    else:
        metrics = evaluate_process(params)
        output = {
            "mode": "evaluation",
            "parameters": _parameters_to_dict(params),
            "metrics": _metrics_to_dict(metrics),
        }

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(f"Mode: {output['mode']}")
        metrics = output["metrics"]
        params_out = output["parameters"]
        print(
            f"Spindle {params_out['spindle_rpm']:.1f} rpm | "
            f"Table {params_out['table_rpm']:.1f} rpm | "
            f"Load {params_out['spindle_load_pct']:.1f}%"
        )
        print(
            f"Removal {metrics['removal_um']:.3f} µm | "
            f"GR {metrics['gr']:.2f} | Time {metrics['time_s']:.1f} s"
        )
        print(
            f"Removal rate {metrics['removal_rate_um_s']:.3f} µm/s | "
            f"Wheel wear {metrics['wheel_wear_um']:.4f} µm"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
