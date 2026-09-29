"""Adaptive composition refinement for native FGM tables.

Every proposed bridge is a new certified flamelet.  The routine never creates
rows by interpolation: interpolation is used only to estimate where another
physical solve is needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from kflame.fgm.common import (
    build_adaptive_c_grid,
    build_global_indicator,
    validate_fgm_table,
)
from kflame.fgm.refine import _defects
from kflame.fgm import generate as native


def _select_bridges(
    phis: np.ndarray,
    interval_defects: np.ndarray,
    target_defect: float,
    max_bridges: int,
) -> np.ndarray:
    """Return logarithmic interval midpoints whose defect exceeds the target."""
    if phis.size < 3 or interval_defects.shape != (phis.size - 1,):
        raise ValueError("Adaptive refinement requires at least three ordered flamelets.")
    if not np.isfinite(interval_defects).all() or np.any(np.diff(phis) <= 0.0):
        raise ValueError("Invalid composition coordinates or interpolation defect.")

    selected = np.flatnonzero(interval_defects > target_defect)
    if selected.size > max_bridges:
        selected = selected[np.argsort(interval_defects[selected])[-max_bridges:]]
    return np.sqrt(phis[np.sort(selected)] * phis[np.sort(selected) + 1])


def _check_progress_coordinate(record: native.FlameRecord) -> None:
    span = float(record.beta[-1] - record.beta[0])
    if not np.isfinite(record.beta).all() or abs(span) <= 1.0e-14:
        raise RuntimeError(f"Degenerate progress coordinate at phi={record.phi:g}.")
    raw_c = (record.beta - record.beta[0]) / span
    if (
        not np.isfinite(raw_c).all()
        or raw_c.min() < -1.0e-7
        or raw_c.max() > 1.0 + 1.0e-7
        or np.diff(raw_c).min() < -1.0e-7
    ):
        raise RuntimeError(f"Progress coordinate is not monotone and bounded at phi={record.phi:g}.")


def _tabulate(
    records: list[native.FlameRecord],
    species_names: list[str],
    args: Any,
) -> tuple[dict[str, np.ndarray], dict[str, float | int | bool]]:
    c_fine = np.linspace(0.0, 1.0, max(101, int(args.c_fine)))
    indicator = build_global_indicator(
        records=records,
        species_names=species_names,
        indicator_species=native.parse_species_list(args.indicator_species),
        c_fine=c_fine,
        w_grad=float(args.indicator_weight_grad),
        w_conc=float(args.indicator_weight_conc),
        w_temp=float(args.indicator_weight_temp),
        w_qdot=float(args.indicator_weight_qdot),
    )
    c_grid = build_adaptive_c_grid(
        c_fine=c_fine,
        indicator=indicator,
        n_c=int(args.n_c),
        bias=float(args.refine_bias),
    )
    table = native.build_tables(records, species_names, c_grid)
    table["predictor_kind"] = np.asarray(table["predictor_kind"], dtype=str)
    table.update(
        species_names=np.asarray(species_names, dtype=str),
        used_progress_species=np.asarray(
            native.parse_progress_weights(args.progress_species), dtype=str
        ),
        indicator_species=np.asarray(
            native.parse_species_list(args.indicator_species), dtype=str
        ),
        indicator_fine_c=c_fine,
        indicator_fine_value=indicator,
    )
    return table, validate_fgm_table(table)


def build_adaptive_fgm(
    *,
    args: Any,
    initial_phis: np.ndarray,
    output_dir: Path,
    target_defect: float,
    max_bridges_per_round: int,
    max_rounds: int,
    max_flames: int,
) -> Path:
    """Solve and refine an FGM family until its internal defect is acceptable."""
    initial_phis = np.asarray(initial_phis, dtype=float)
    if (
        initial_phis.ndim != 1
        or initial_phis.size < 3
        or not np.isfinite(initial_phis).all()
        or np.any(initial_phis <= 0.0)
        or np.any(np.diff(initial_phis) <= 0.0)
    ):
        raise ValueError("Adaptive FGM requires at least three positive, increasing initial phis.")
    if not np.isfinite(target_defect) or target_defect <= 0.0:
        raise ValueError("target_defect must be finite and positive.")
    if max_bridges_per_round < 1 or max_rounds < 0 or max_flames < initial_phis.size:
        raise ValueError("Invalid adaptive FGM refinement budget.")

    started = time.perf_counter()
    requested_threads = int(args.numba_kinetics_threads)
    if requested_threads <= 0:
        requested_threads = min(4, max(1, os.cpu_count() or 1))
    args.numba_kinetics_threads_resolved = native.configure_numba_kinetics_threads(
        requested_threads
    )
    mechanism = native.load_mechanism(native._resolve_mechanism(args.mech))
    species_names = list(mechanism.species_names)
    progress_weights = native.parse_progress_weights(args.progress_species)
    options = native.make_solve_options(args)

    records: dict[float, native.FlameRecord] = {}
    states: dict[float, dict[str, np.ndarray]] = {}
    traces: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    pending = initial_phis.copy()
    table: dict[str, np.ndarray] | None = None
    validation: dict[str, float | int | bool] | None = None

    print(
        f"Adaptive phi refinement: target={100.0 * target_defect:.3g}% | "
        f"initial flamelets={initial_phis.size}",
        flush=True,
    )
    for round_index in range(max_rounds + 1):
        for phi in pending:
            phi = float(phi)
            lower = sorted(value for value in states if value < phi)
            previous = states[lower[-1]] if lower else None
            previous2 = states[lower[-2]] if len(lower) > 1 else None
            role = "requested" if round_index == 0 else "bridge"
            print(
                f"[{len(records) + 1}] Solving {role} flamelet phi={phi:.6g} ...",
                flush=True,
            )
            record, state, trace = native.solve_flame_native(
                phi=phi,
                args=args,
                mech_data=mechanism,
                opts=options,
                progress_weights=progress_weights,
                prev_solution=previous,
                prev_prev_solution=previous2,
            )
            record.requested = round_index == 0
            record.bridge = not record.requested
            accepted = bool(record.solve_ok and record.final_accepted)
            trace.update(phi=phi, round=round_index, role=role, accepted=accepted)
            traces.append(trace)
            print(
                f"  {'OK' if accepted else 'FAIL'} | Su={record.Su_m_per_s:.4f} m/s | "
                f"n={record.n_points} | t={record.solve_time_s:.2f}s",
                flush=True,
            )
            if not accepted:
                raise RuntimeError(f"Flamelet phi={phi:g} was not accepted.")
            _check_progress_coordinate(record)
            records[phi] = record
            states[phi] = state

        ordered = [records[phi] for phi in sorted(records)]
        table, validation = _tabulate(ordered, species_names, args)
        interval_defects, diagnostics = _defects(table)
        pending = _select_bridges(
            table["phi_grid"],
            interval_defects,
            target_defect,
            max_bridges_per_round,
        )
        maximum = float(np.max(interval_defects))
        rounds.append(
            {
                "round": round_index,
                "n_flames": len(records),
                "max_defect": maximum,
                "interval_defect": interval_defects.tolist(),
                "leave_one_out": diagnostics,
                "proposed_phi": pending.tolist(),
            }
        )
        print(
            f"Adaptive check: {len(records)} flamelets | "
            f"max defect={100.0 * maximum:.3f}% | bridges next={pending.size}",
            flush=True,
        )
        if pending.size == 0:
            break
        if round_index >= max_rounds or len(records) + pending.size > max_flames:
            raise RuntimeError("Adaptive FGM budget exhausted before reaching target_defect.")

    assert table is not None and validation is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "fgm_table.npz", **table)
    native.write_summary_csv(output_dir / "summary_phi.csv", ordered)
    if bool(args.save_raw_profiles):
        native.write_raw_profiles(output_dir, ordered, species_names)
    schedule = {
        "coordinate": "log(phi)",
        "selection": {
            "method": "leave-one-out maximum relative L2 defect",
            "target_defect": float(target_defect),
        },
        "phi_requested": initial_phis.tolist(),
        "phi_resolved": table["phi_grid"].tolist(),
        "requested": table["requested"].tolist(),
        "bridge": table["bridge"].tolist(),
    }
    (output_dir / "continuation_schedule.json").write_text(
        json.dumps(schedule, indent=2), encoding="utf-8"
    )
    (output_dir / "continuation_trace.json").write_text(
        json.dumps(traces, indent=2), encoding="utf-8"
    )
    (output_dir / "adaptive_rounds.json").write_text(
        json.dumps(rounds, indent=2), encoding="utf-8"
    )
    metadata = {
        "mode": "adaptive-phi-grid",
        "args": vars(args),
        "n_phi_requested": int(initial_phis.size),
        "n_phi_table": int(table["phi_grid"].size),
        "n_phi_bridge": int(table["bridge"].sum()),
        "n_c": int(table["c_grid"].size),
        "all_final_accepted": bool(np.all(table["final_accepted"])),
        "table_validation": validation,
        "adaptive_phi": {
            "target_defect": float(target_defect),
            "max_bridges_per_round": int(max_bridges_per_round),
            "rounds": len(rounds),
        },
        "runtime_s": time.perf_counter() - started,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"FGM accepted: {metadata['n_phi_table']} flamelets "
        f"({metadata['n_phi_bridge']} bridges), {metadata['n_c']} c points.",
        flush=True,
    )
    return output_dir
