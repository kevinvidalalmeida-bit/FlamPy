"""Every flame and table variant with every mechanism up to GRI-Mech 3.0.

Run from the repository root after installing plotting support:

    python -m pip install -e ".[plots]"
    python examples/validate_variants.py
    python examples/validate_variants.py --mechanisms gri30.yaml --cases premixed burner
    python examples/validate_variants.py --plot-only runs/variants/<folder>

For H2-O2 (H2/air), DRM19, DRM22, FFCM-1 and GRI-Mech 3.0 (CH4/air) the
script solves, one after the other:

- premixed: free flame, burner (heat loss to the burner), both with optically
  thin radiation, and multicomponent transport with Soret;
- counterflow: mixture-averaged, radiating and multicomponent with Soret;
- tables: premixed FGM (Z, c) with adaptive phi, burner FGM (c, h),
  non-adiabatic FGM (Z, C, h), diffusion FGM (Z, C) and radiative diffusion
  FGM (Z, C, h).

The burner mass flux is half the free flame's rho_u*Su, so every fuel gets a
stabilized flame. Each case keeps its own output; summary.json records
acceptance, wall time (including JIT compilation on a cold Numba cache) and
the case metric, and the figure (status and time per mechanism and variant)
is drawn only from that file. A failing case is
recorded with its error and the run continues.
"""

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import kflame


# %% CASES
MECHANISMS = {
    "h2o2.yaml": dict(fuel="H2", phi=0.8, progress="H2O:1,HO2:10,OH:-1", streams=dict(fuel_mass_flux=.05, oxidizer_mass_flux=.15)),
    "drm19.yaml": dict(fuel="CH4", phi=1.0, progress="CO2:1,H2O:1,CO:1,H2:0.5", streams=dict(fuel_mass_flux=.12, oxidizer_mass_flux=.24)),
    "drm22.yaml": dict(fuel="CH4", phi=1.0, progress="CO2:1,H2O:1,CO:1,H2:0.5", streams=dict(fuel_mass_flux=.12, oxidizer_mass_flux=.24)),
    "ffcm1.yaml": dict(fuel="CH4", phi=1.0, progress="CO2:1,H2O:1,CO:1,H2:0.5", streams=dict(fuel_mass_flux=.12, oxidizer_mass_flux=.24)),
    "gri30.yaml": dict(fuel="CH4", phi=1.0, progress="CO2:1,H2O:1,CO:1,H2:0.5", streams=dict(fuel_mass_flux=.12, oxidizer_mass_flux=.24)),
}
CASES = ("premixed", "burner", "premixed_radiation", "burner_radiation", "premixed_multicomponent",
         "counterflow", "counterflow_radiation", "counterflow_multicomponent",
         "fgm_premixed", "fgm_burner", "fgm_nonadiabatic", "fgm_diffusion", "fgm_diffusion_heat_loss")
LABELS = dict(premixed="free", burner="burner", premixed_radiation="free + rad", burner_radiation="burner + rad",
              premixed_multicomponent="free MC+Soret", counterflow="counterflow", counterflow_radiation="CF + rad",
              counterflow_multicomponent="CF MC+Soret", fgm_premixed="FGM (Z,c)", fgm_burner="FGM (c,h)",
              fgm_nonadiabatic="FGM (Z,C,h)", fgm_diffusion="diff. FGM (Z,C)", fgm_diffusion_heat_loss="diff. FGM (Z,C,h)")


def run_case(case, mechanism, setup, folder, memo):
    """Solve one variant; returns its metric dictionary (raises on failure)."""
    fuel, phi = setup["fuel"], setup["phi"]
    premixed = dict(mechanism=mechanism, fuel=fuel, phi=phi)
    counterflow = dict(mechanism=mechanism, fuel=fuel, **setup["streams"])
    if case == "premixed":
        flame = kflame.solve_flame(**premixed, output=folder)
        memo["burner_flux"] = .5*float(flame["rho"][0]*flame["Su"])
        return dict(Su_m_s=float(flame["Su"]), nodes=len(flame["z"]))
    if case in ("burner", "burner_radiation"):
        flame = kflame.solve_burner_flame(**premixed, mass_flux=memo["burner_flux"], output=folder,
                                          radiation=case == "burner_radiation")
        loss = flame["heat_loss"]
        return dict(mass_flux=memo["burner_flux"], wall_loss_W_m2=float(loss["burner_heat_loss_W_m2"]),
                    radiation_W_m2=float(loss.get("radiative_heat_loss_W_m2", 0.)),
                    energy_closure=float(loss["relative_energy_closure_error"]), nodes=len(flame["z"]))
    if case == "premixed_radiation":
        flame = kflame.solve_flame(**premixed, radiation=True, output=folder)
        return dict(Su_m_s=float(flame["Su"]), nodes=len(flame["z"]))
    if case == "premixed_multicomponent":
        flame = kflame.solve_flame(**premixed, transport="multicomponent", soret=True, output=folder)
        return dict(Su_m_s=float(flame["Su"]), nodes=len(flame["z"]))
    if case.startswith("counterflow"):
        settings = dict(radiation=True) if case == "counterflow_radiation" else (
            dict(transport="multicomponent", soret=True) if case == "counterflow_multicomponent" else {})
        flame = kflame.solve_diffusion_flame(**counterflow, output=folder, **settings)
        return dict(Tmax_K=float(np.max(flame["T"])), nodes=len(flame["z"]))
    if case == "fgm_premixed":
        phis = (.6, .8, 1.) if fuel == "H2" else (.8, 1., 1.2)
        out = kflame.generate_fgm(phis=phis, mechanism=mechanism, fuel=fuel, progress_species=setup["progress"],
                                  species=("H2O", "OH") if fuel == "H2" else ("CO2", "H2O"),
                                  target_defect=.02, output=folder, export=False)
        meta = json.loads((Path(out)/"metadata.json").read_text(encoding="utf-8"))
        return dict(flamelets=int(meta["n_phi_table"]), bridges=int(meta["n_phi_bridge"]))
    if case == "fgm_burner":
        flux = memo["burner_flux"]
        out = kflame.generate_burner_fgm(mass_fluxes=(1.2*flux, .6*flux), progress_species=setup["progress"],
                                         progress_points=61, output=folder, **premixed)
        meta = json.loads((Path(out)/"metadata.json").read_text(encoding="utf-8"))
        return dict(burners=len(meta["rows"])-1)
    if case == "fgm_nonadiabatic":
        phis = (.6, .8, 1.) if fuel == "H2" else (.8, 1., 1.2)
        out = kflame.generate_nonadiabatic_fgm(phis=phis, mass_flux_fractions=(.6, .3), progress_species=setup["progress"],
                                               progress_points=61, output=folder, mechanism=mechanism, fuel=fuel)
        return dict(table=str(Path(out).name))
    if case == "fgm_diffusion":
        result = kflame.generate_diffusion_fgm(flow_scales=(.75, 1., 1.25), mixture_points=121, output=folder,
                                               **counterflow)
        return dict(flamelets=len(result["flamelets"]))
    if case == "fgm_diffusion_heat_loss":
        result = kflame.generate_diffusion_heat_loss_fgm(flow_scales=(.75, 1.25), radiation_scales=(0., 4.),
                                                         mixture_points=121, output=folder, **counterflow)
        return dict(flamelets=sum(len(row) for row in result["flamelets"]))
    raise ValueError(case)


def run(mechanisms, cases, output):
    summary = dict(created=datetime.now().isoformat(timespec="seconds"), mechanisms={},
                   cases=list(cases), labels=LABELS)
    for mechanism in mechanisms:
        setup = MECHANISMS[mechanism]
        memo = {}
        rows = summary["mechanisms"][mechanism] = dict(fuel=setup["fuel"], cases={})
        needed = list(cases)
        if any(case.startswith(("burner", "fgm_burner")) for case in needed) and "premixed" not in needed:
            needed.insert(0, "premixed")   # the burner flux comes from the free flame
        for case in needed:
            folder = output/Path(mechanism).stem/case
            print(f"{mechanism} {case}...", flush=True)
            tic = time.perf_counter()
            try:
                metric = run_case(case, mechanism, setup, folder, memo)
                status = "accepted"
            except Exception as error:
                metric = dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(limit=3))
                status = "failed"
            elapsed = time.perf_counter()-tic
            rows["cases"][case] = dict(status=status, wall_s=elapsed, **metric)
            print(f"  {status} in {elapsed:.1f} s {json.dumps({k: v for k, v in metric.items() if k != 'traceback'})}",
                  flush=True)
            (output/"summary.json").write_text(json.dumps(summary, indent=1)+"\n", encoding="utf-8")
    return summary


# %% PLOTS
def plot(summary, output):
    mechanisms = list(summary["mechanisms"])
    cases = summary["cases"]
    labels = summary.get("labels", LABELS)
    status = np.full((len(mechanisms), len(cases)), np.nan)
    times = np.full_like(status, np.nan)
    for i, mechanism in enumerate(mechanisms):
        for j, case in enumerate(cases):
            row = summary["mechanisms"][mechanism]["cases"].get(case)
            if row is not None:
                status[i, j] = 1. if row["status"] == "accepted" else 0.
                times[i, j] = row["wall_s"]
    figure, axis = plt.subplots(figsize=(1.05*len(cases)+2.5, .62*len(mechanisms)+2.2), layout="constrained")
    colours = np.where(np.isnan(status), .5, status)
    axis.imshow(colours, cmap="RdYlGn", vmin=-.2, vmax=1.2, aspect="auto")
    for i in range(len(mechanisms)):
        for j in range(len(cases)):
            if not np.isnan(times[i, j]):
                seconds = times[i, j]
                text = (f"{seconds:.1f} s" if seconds < 10. else f"{seconds:.0f} s") if status[i, j] == 1. else "failed"
                axis.text(j, i, text, ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(cases)), [labels.get(case, case) for case in cases], rotation=35, ha="right")
    axis.set_yticks(range(len(mechanisms)), [f"{Path(m).stem} ({summary['mechanisms'][m]['fuel']})" for m in mechanisms])
    axis.set_title("Flame and table variants: accepted (green) and wall time per case")
    stem = output/"variants"
    figure.savefig(stem.with_suffix(".png"), dpi=160)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    return stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mechanisms", nargs="+", choices=list(MECHANISMS), default=list(MECHANISMS))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--output", type=Path, help="New folder; default: a dated folder in runs/")
    parser.add_argument("--plot-only", type=Path, help="Folder (or summary.json) to redraw")
    args = parser.parse_args()
    if args.plot_only:
        path = args.plot_only if args.plot_only.suffix == ".json" else args.plot_only/"summary.json"
        print(f"Figure: {plot(json.loads(path.read_text(encoding='utf-8')), args.output or path.parent)}")
        return
    output = args.output or Path("runs")/"variants"/datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Figure: {plot(run(args.mechanisms, args.cases, output), output)}")


if __name__ == "__main__":
    main()
