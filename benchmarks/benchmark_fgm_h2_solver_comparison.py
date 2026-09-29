"""Eight adaptive H2/air FGM: four transports and two flame solvers.

The H2 progress coordinate is beta_c = Y_H2O + 10 Y_HO2.  This runner is
separate from the CH4 campaign so manifests, mechanisms, and results cannot
be mixed accidentally.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import benchmark_fgm_adaptive_campaign as adaptive
from benchmark_fgm_campaign import digest


def design():
    cases = []
    for index, (tag, model, soret) in enumerate(adaptive.TRANSPORTS):
        order = ("kflame", "cantera") if index % 2 == 0 else ("cantera", "kflame")
        for solver in order:
            cases.append(dict(
                id=f"H2_T300_p1_{tag}_{solver}", fuel="H2", T_in_K=300,
                p_atm=1, P_Pa=101325.0, sweep="base", transport=tag,
                transport_model=model, soret_enabled=soret, solver=solver,
            ))
    return cases


def make_manifest(smoke=False):
    manifest = adaptive.make_manifest(smoke)
    settings = manifest["settings"]
    settings.update(
        mech="h2o2.yaml", fuel="H2", oxidizer="O2:1,N2:3.76",
        progress_species="H2O:1,HO2:10",
        indicator_species="H2,O2,H2O,OH,HO2,H2O2",
    )
    manifest.update(kind="fgm-h2-solver-comparison", conditions=design())
    manifest["environment"]["versions"]["cantera"] = importlib.metadata.version("cantera")
    manifest["protocol"].update(
        progress_variable="beta_c=Y_H2O+10*Y_HO2; each profile is normalized by its fresh and burnt endpoints",
        continuation="KFLAME native secant; Cantera set_initial_guess(previous SolutionArray); within-family lower-phi states only",
        comparison="same H2/air physics and table algorithm; independent adaptive row sets; compare total workflow and final row count",
        cantera="Cantera FreeFlame, auto=True, refine_grid=True; steady rtol=1e-4 atol=1e-9; transient rtol=1e-4 atol=1e-11",
        order="alternate solver order across the four transports; one fresh construction each",
        scope="eight H2/h2o2 base-state tables; no pressure/temperature sweep, no repetition uncertainty",
    )
    manifest["code"][Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/thesis_fgm_h2_comparison"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Separate technical check: three initial rows, Nc=61, 10%% target")
    parser.add_argument("--case", action="append")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--attempt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if args.worker:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        condition = next(c for c in manifest["conditions"] if c["id"] == args.worker)
        engine = None
        if condition["solver"] == "cantera":
            from fgm_cantera_adapter import CanteraEngine
            engine = CanteraEngine
        return adaptive.worker(out, args.worker, args.attempt, engine=engine)
    manifest = make_manifest(args.smoke)
    cases = manifest["conditions"]
    if args.case:
        unknown = set(args.case) - {c["id"] for c in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in args.case]
    print("4 transports x 2 flame solvers = 8 H2/h2o2 FGM; 300 K, 1 atm; no repetitions.")
    print(f"Progress: {manifest['protocol']['progress_variable']}")
    for case in cases:
        print(case["id"])
    if args.dry_run:
        if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()) != manifest:
            raise ValueError("Configuration/code/environment changed: use a new directory.")
        print("Dry run: no solves or writes.")
        return 0
    adaptive.ensure_manifest(out, manifest, args.resume)
    lock = out / "campaign.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Campaign locked: check active processes before removing {lock}")
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        ok = True
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] {case['id']}", flush=True)
            ok = adaptive.run_case(out, case["id"], args.retry_failed, entrypoint=Path(__file__).resolve()) and ok
        return 0 if ok else 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
