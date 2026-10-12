"""Extreme and difficult operating conditions: robustness and wall time.

Run from the repository root:

    python examples/stress_extreme.py
    python examples/stress_extreme.py --only premixed_p40 counterflow_strain20
    python examples/stress_extreme.py --list

Each case uses the public API with its default numerical settings, so a
failure or a slow case points at the solver, not at a tuned setup. Cases:

- premixed CH4/air (GRI-Mech 3.0): 0.1 to 40 atm, very lean and rich,
  gas-turbine preheat at 20 atm, oxy-fuel, multicomponent with Soret and a
  radiating burner at high pressure;
- premixed H2 (H2-O2): very lean, rich, 20 atm, preheated at 10 atm and
  H2/O2 without nitrogen;
- counterflow: 10 and 40 atm, strain up to near extinction, preheated air
  and H2/air at 10 atm;
- tables at high pressure: premixed FGM (Z, c), burner FGM (c, h) with CH4
  and H2, and diffusion FGM (Z, C).

summary.json records status, wall time (including JIT compilation on a
cold Numba cache) and the case metric; a failing case keeps its error and
the run continues.
"""

import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

import kflame

ATM = 101325.
GRI = dict(mechanism="gri30.yaml", fuel="CH4")
H2 = dict(mechanism="h2o2.yaml", fuel="H2")
CF_GRI = dict(mechanism="gri30.yaml", fuel="CH4", fuel_mass_flux=.12, oxidizer_mass_flux=.24)
CF_H2 = dict(mechanism="h2o2.yaml", fuel="H2", fuel_mass_flux=.05, oxidizer_mass_flux=.15)


def _premixed(**kwargs):
    def run(folder):
        flame = kflame.solve_flame(output=folder, **kwargs)
        return dict(Su_m_s=float(flame["Su"]), Tmax_K=float(np.max(flame["T"])), nodes=len(flame["z"]),
                    width_m=float(flame["z"][-1]))
    return run


def _burner(free, fraction, **kwargs):
    def run(folder):
        flame = kflame.solve_flame(output=folder/"free", **free)
        flux = fraction*float(flame["rho"][0]*flame["Su"])
        burner = kflame.solve_burner_flame(mass_flux=flux, output=folder/"burner", **free, **kwargs)
        loss = burner["heat_loss"]
        return dict(mass_flux=flux, energy_closure=float(loss["relative_energy_closure_error"]),
                    wall_loss_W_m2=float(loss["burner_heat_loss_W_m2"]), nodes=len(burner["z"]))
    return run


def _counterflow(**kwargs):
    def run(folder):
        flame = kflame.solve_diffusion_flame(output=folder, **kwargs)
        return dict(Tmax_K=float(np.max(flame["T"])), nodes=len(flame["z"]),
                    strain_s1=float(flame.get("strain_rate_mean_s1", np.nan)) if "strain_rate_mean_s1" in flame else None)
    return run


def _fgm_premixed(**kwargs):
    def run(folder):
        out = kflame.generate_fgm(output=folder, export=False, target_defect=.02, **kwargs)
        meta = json.loads((Path(out)/"metadata.json").read_text(encoding="utf-8"))
        return dict(flamelets=int(meta["n_phi_table"]), bridges=int(meta["n_phi_bridge"]))
    return run


def _fgm_burner(free, fractions, **kwargs):
    def run(folder):
        flame = kflame.solve_flame(output=folder/"free", **free)
        flux = float(flame["rho"][0]*flame["Su"])
        out = kflame.generate_burner_fgm(mass_fluxes=tuple(f*flux for f in fractions), output=folder/"table",
                                         **free, **kwargs)
        meta = json.loads((Path(out)/"metadata.json").read_text(encoding="utf-8"))
        return dict(burners=len(meta["rows"])-1)
    return run


def _fgm_diffusion(**kwargs):
    def run(folder):
        result = kflame.generate_diffusion_fgm(output=folder, **kwargs)
        return dict(flamelets=len(result["flamelets"]))
    return run


CASES = {
    # Premixed CH4/air, GRI-Mech 3.0
    "premixed_p1": _premixed(**GRI, phi=1.),
    "premixed_p0.1": _premixed(**GRI, phi=1., pressure=.1*ATM),
    "premixed_p10": _premixed(**GRI, phi=1., pressure=10*ATM),
    "premixed_p20": _premixed(**GRI, phi=1., pressure=20*ATM),
    "premixed_p40": _premixed(**GRI, phi=1., pressure=40*ATM),
    "premixed_lean0.55": _premixed(**GRI, phi=.55),
    "premixed_rich1.6": _premixed(**GRI, phi=1.6),
    "premixed_turbine": _premixed(**GRI, phi=.6, temperature=700., pressure=20*ATM),
    "premixed_oxyfuel": _premixed(**GRI, phi=1., oxidizer="O2:1"),
    "premixed_mc_p20": _premixed(**GRI, phi=1., pressure=20*ATM, transport="multicomponent", soret=True),
    "premixed_rad_p10": _premixed(**GRI, phi=1., pressure=10*ATM, radiation=True),
    "burner_rad_p10": _burner(dict(**GRI, phi=1., pressure=10*ATM), .5, radiation=True),
    # Premixed H2, H2-O2
    "h2_lean0.35": _premixed(**H2, phi=.35),
    "h2_rich3": _premixed(**H2, phi=3.),
    "h2_p20": _premixed(**H2, phi=1., pressure=20*ATM),
    "h2_preheat_p10": _premixed(**H2, phi=.5, temperature=700., pressure=10*ATM),
    "h2_oxygen": _premixed(**H2, phi=1., oxidizer="O2:1"),
    "h2_burner_p10": _burner(dict(**H2, phi=.6, pressure=10*ATM), .5),
    # Counterflow
    "counterflow_p10": _counterflow(**CF_GRI, pressure=10*ATM),
    "counterflow_p40": _counterflow(**CF_GRI, pressure=40*ATM),
    "counterflow_strain5": _counterflow(**{**CF_GRI, "fuel_mass_flux": .6, "oxidizer_mass_flux": 1.2}),
    "counterflow_strain15": _counterflow(**{**CF_GRI, "fuel_mass_flux": 1.8, "oxidizer_mass_flux": 3.6}),
    "counterflow_preheat": _counterflow(**CF_GRI, oxidizer_temperature=1000.),
    "counterflow_h2_p10": _counterflow(**CF_H2, pressure=10*ATM),
    "counterflow_mc_p10": _counterflow(**CF_GRI, pressure=10*ATM, transport="multicomponent", soret=True),
    # Tables at high pressure
    "fgm_premixed_p20": _fgm_premixed(**GRI, pressure=20*ATM, phis=(.8, 1., 1.2),
                                      progress_species="CO2:1,H2O:1,CO:1,H2:0.5", species=("CO2", "H2O")),
    "fgm_burner_p20": _fgm_burner(dict(**GRI, phi=1., pressure=20*ATM), (.6, .3),
                                  progress_species="CO2:1,H2O:1,CO:1,H2:0.5", progress_points=61),
    "fgm_burner_h2_p10": _fgm_burner(dict(**H2, phi=.6, pressure=10*ATM), (.6, .3),
                                     progress_species="H2O:1,HO2:10,OH:-1", progress_points=61),
    "fgm_diffusion_p10": _fgm_diffusion(**CF_GRI, pressure=10*ATM, flow_scales=(.75, 1., 1.25), mixture_points=121),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(CASES), help="Subset of cases")
    parser.add_argument("--output", type=Path, help="New folder; default: a dated folder in runs/stress")
    parser.add_argument("--list", action="store_true", help="List the cases and exit")
    args = parser.parse_args()
    if args.list:
        print("\n".join(CASES))
        return
    output = args.output or Path("runs")/"stress"/datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(created=datetime.now().isoformat(timespec="seconds"), cases={})
    for name in args.only or CASES:
        print(f"{name}...", flush=True)
        tic = time.perf_counter()
        try:
            metric = CASES[name](output/name)
            status = "accepted"
        except Exception as error:
            metric = dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(limit=4))
            status = "failed"
        elapsed = time.perf_counter()-tic
        summary["cases"][name] = dict(status=status, wall_s=elapsed, **metric)
        print(f"  {status} in {elapsed:.1f} s {json.dumps({k: v for k, v in metric.items() if k != 'traceback'})}",
              flush=True)
        (output/"summary.json").write_text(json.dumps(summary, indent=1)+"\n", encoding="utf-8")
    failed = [name for name, row in summary["cases"].items() if row["status"] != "accepted"]
    print(f"{len(summary['cases'])-len(failed)}/{len(summary['cases'])} accepted; failed: {failed}")


if __name__ == "__main__":
    main()
