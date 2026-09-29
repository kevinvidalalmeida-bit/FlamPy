"""Independent H2 retained flames for the FGM fidelity figure.

The twelve compositions are geometric midpoints of intervals in the saved
KFLAME averaged-transport construction.  They are never construction rows and
each is solved with a fresh physical initialisation (no continuation seed).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "TESIS_RESUL/herramientas/campanas"))

import numpy as np

from benchmark_fgm_campaign import atomic, clean, digest


def load_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def holdout_design(phi_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if phi_grid.ndim != 1 or phi_grid.size < 13 or np.any(np.diff(phi_grid) <= 0):
        raise ValueError("The construction phi grid must be strictly increasing and contain at least 13 rows.")
    interval = np.rint(np.linspace(0, phi_grid.size - 2, 12)).astype(int)
    if np.unique(interval).size != 12:
        raise ValueError("Could not select twelve distinct construction intervals.")
    phi = np.sqrt(phi_grid[interval] * phi_grid[interval + 1])
    if np.isclose(phi[:, None], phi_grid[None, :], rtol=0, atol=1e-12).any():
        raise AssertionError("A retained phi coincides with a construction row.")
    return interval, phi


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction", type=Path,
                        default=Path("TESIS_RESUL/corridas/FGM/thesis_fgm_h2_comparison/cases/H2_T300_p1_P_kflame/attempt-001"))
    parser.add_argument("--output", type=Path,
                        default=Path("TESIS_RESUL/corridas/reproduccion/thesis_fgm_h2_holdouts"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args_cli = parser.parse_args(argv)
    source = args_cli.construction.resolve()
    output = args_cli.output.resolve()
    table_path = source / "fgm_table.npz"
    manifest_path = source.parents[2] / "manifest.json"
    if not table_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Expected the accepted H2 P/KFLAME construction and its campaign manifest.")
    table = load_table(table_path)
    interval, phi = holdout_design(np.asarray(table["phi_grid"], dtype=float))
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    settings = dict(source_manifest["settings"])
    settings.update(transport_model="mixture-averaged", soret_enabled=False)
    proposed = dict(schema=1, kind="fgm-h2-independent-holdouts", source_construction=str(source),
                    source_table_sha256=digest(table_path), source_manifest_sha256=digest(manifest_path),
                    mechanism="h2o2.yaml", fuel="H2", oxidizer="O2:1,N2:3.76", T_in=settings["T_in"],
                    P=settings["P"], transport="mixture-averaged", soret=False,
                    progress_species=settings["progress_species"], settings=settings,
                    interval_index=interval.tolist(), holdout_phi=phi.tolist(), n_holdouts=12,
                    protocol="Independent cold starts; no construction profile, copy, secant predictor, or warmup is used.")
    if args_cli.dry_run:
        print(json.dumps(clean(proposed), indent=2, ensure_ascii=False))
        return
    if output.exists() and (output / "manifest.json").exists():
        old = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if old != proposed:
            raise ValueError("Existing holdout output has different provenance; choose another output directory.")
        if not args_cli.resume:
            raise ValueError("Holdout output exists; use --resume.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("New holdout output directory must be empty.")
    else:
        atomic(output / "manifest.json", proposed)

    from kflame.fgm import generate as g
    g.configure_numba_kinetics_threads(4)
    native_args = argparse.Namespace(**settings)
    native_args.mech = str((source.parents[2] / "inputs" / "h2o2.yaml").resolve())
    if not Path(native_args.mech).exists():
        raise FileNotFoundError(f"Saved mechanism copy missing: {native_args.mech}")
    mech = g.load_mechanism(native_args.mech)
    weights = g.parse_progress_weights(native_args.progress_species)
    solve_options = g.make_solve_options(native_args)
    profiles = output / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    traces = []
    for index, value in enumerate(phi):
        profile = profiles / f"{index:02d}.npz"
        if profile.exists():
            with np.load(profile, allow_pickle=False) as saved:
                accepted = bool(saved["solve_ok"].item() and saved["final_accepted"].item())
            if not accepted:
                raise ValueError(f"Existing retained profile {profile} is not accepted.")
            traces.append(dict(index=index, phi=float(value), reused_saved_profile=True))
            continue
        start = time.perf_counter()
        record, _state, trace = g.solve_flame_native(float(value), native_args, mech, solve_options, weights,
                                                     prev_solution=None, prev_prev_solution=None)
        elapsed = time.perf_counter() - start
        accepted = bool(record.solve_ok and record.final_accepted)
        trace.update(index=index, phi=float(value), construction_interval=int(interval[index]),
                     measured_s=float(elapsed), accepted=accepted, predictor_forced="none")
        atomic(profile, dict(dataclasses.asdict(record), species_names=np.asarray(mech.species_names, dtype=str)), npz=True)
        traces.append(trace)
        atomic(output / "trace.json", traces)
        print(f"holdout {index + 1}/12 phi={value:.8g} accepted={accepted} t={elapsed:.3f}s", flush=True)
        if not accepted:
            raise RuntimeError(f"Unaccepted retained flame at phi={value}.")
    artifacts = {str(p.relative_to(output)): digest(p) for p in sorted(profiles.glob("*.npz"))}
    atomic(output / "trace.json", traces)
    artifacts["trace.json"] = digest(output / "trace.json")
    atomic(output / "result.json", dict(status="accepted", n_holdouts=12, artifacts=artifacts))


if __name__ == "__main__":
    main()
