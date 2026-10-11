"""Solve the same flame with every installed mechanism and compare them.

Run from the repository root after installing plotting support:

    python -m pip install -e ".[plots]"
    python examples/example_mechanisms.py
    python examples/example_mechanisms.py --flame diffusion
    python examples/example_mechanisms.py --mechanisms gri30.yaml uscmech2.yaml

Every mechanism of the catalog that is installed (AramcoMech is a local
installation, see docs/mechanisms.md) solves the same CH4/air case. Each flame
keeps its own output folder; ``summary.json`` stores the profiles and timings,
and the figure is drawn only from that file. To redraw the published figure
without simulating:

    python examples/example_mechanisms.py --plot-only docs/assets/mechanisms/premixed-ch4.json
    python examples/example_mechanisms.py --plot-only docs/assets/mechanisms/diffusion-ch4.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import kflame
from kflame.chemistry.catalog import available_mechanisms


# %% CASE CONFIGURATION
# Same physical case for every mechanism: only the chemistry changes.
PREMIXED = dict(fuel="CH4", oxidizer="O2:1, N2:3.76", phi=1.0, temperature=300.0,
                pressure=101325.0, max_time=900.0)
DIFFUSION = dict(fuel="CH4", oxidizer="O2:1,N2:3.76", fuel_mass_flux=0.12,
                 oxidizer_mass_flux=0.24, pressure=101325.0, max_time=900.0)
# The hydrogen-only mechanism has no methane.
SKIP = ("h2o2.yaml",)


def solve(flame, mechanism, folder):
    """One flame; returns the small record kept in summary.json."""
    if flame == "premixed":
        result = kflame.solve_flame(mechanism=mechanism, output=folder, **PREMIXED)
        metric = dict(Su_m_s=float(result["Su"]))
        profile = dict(z_m=result["z"].tolist(), T_K=result["T"].tolist())
    else:
        result = kflame.solve_diffusion_flame(mechanism=mechanism, output=folder, **DIFFUSION)
        metric = dict(Tmax_K=float(np.max(result["T"])))
        profile = dict(z_m=result["z"].tolist(), T_K=result["T"].tolist(), Z=result["Z"].tolist())
    return dict(accepted=bool(result["accepted"]), runtime_s=float(result["runtime_s"]),
                nodes=int(len(result["z"])), **metric, profile=profile)


def run(flame, selected, output):
    rows = []
    summary = dict(flame=flame, case=PREMIXED if flame == "premixed" else DIFFUSION,
                   created=datetime.now().isoformat(timespec="seconds"), mechanisms=rows)
    for entry in available_mechanisms():
        if entry["filename"] in SKIP or (selected and entry["filename"] not in selected):
            continue
        if not entry["installed"]:
            print(f"skip {entry['name']}: not installed ({entry['path']})", flush=True)
            continue
        print(f"{entry['name']} ({entry['species']} species)...", flush=True)
        record = solve(flame, entry["filename"], output/Path(entry["filename"]).stem)
        rows.append(dict(name=entry["name"], filename=entry["filename"], species=entry["species"],
                         reactions=entry["reactions"], **record))
        value = record.get("Su_m_s", record.get("Tmax_K"))
        print(f"  {record['runtime_s']:.1f} s, {record['nodes']} nodes, "
              f"{'Su' if flame == 'premixed' else 'Tmax'} = {value:.6g}", flush=True)
        (output/"summary.json").write_text(json.dumps(summary, indent=1)+"\n", encoding="utf-8")
    return summary


# %% PLOTS
# Uses only summary.json: change colours, panels or limits freely.
def plot(summary, output):
    rows = summary["mechanisms"]
    premixed = summary["flame"] == "premixed"
    colors = plt.cm.viridis(np.linspace(0., .9, len(rows)))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")

    for row, color in zip(rows, colors):
        profile = row["profile"]
        if premixed:
            # Align the flames at the steepest temperature gradient.
            z, T = np.asarray(profile["z_m"]), np.asarray(profile["T_K"])
            front = z[np.argmax(np.gradient(T, z))]
            axes[0].plot(1e3*(z-front), T, color=color, label=row["name"])
        else:
            axes[0].plot(profile["Z"], profile["T_K"], color=color, label=row["name"])
    if premixed:
        axes[0].set(xlabel="Distance to the flame front [mm]", ylabel="Temperature [K]", xlim=(-1.5, 3.))
    else:
        axes[0].set(xlabel="Mixture fraction Z", ylabel="Temperature [K]", xlim=(0., .25))
    axes[0].legend(fontsize="small")

    names = [row["name"] for row in rows]
    values = [row["Su_m_s"]*100. if premixed else row["Tmax_K"] for row in rows]
    axes[1].barh(names, values, color=colors)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Laminar burning velocity [cm s$^{-1}$]" if premixed else "Peak temperature [K]")
    if premixed:
        axes[1].set_xlim(0., 1.15*max(values))
    else:
        axes[1].set_xlim(min(values)-50., max(values)+25.)
    for y, value in enumerate(values):
        axes[1].text(value, y, f" {value:.1f}" if premixed else f" {value:.0f}", va="center", fontsize="small")

    species = np.array([row["species"] for row in rows])
    times = np.array([row["runtime_s"] for row in rows])
    # Same colour per mechanism as in the other panels (legend on the left).
    axes[2].scatter(species, times, c=colors, s=55, edgecolors="black", linewidths=.6, zorder=3)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    # At least one decade on each axis keeps plain decade labels.
    axes[2].set(xlabel="Species", ylabel="Solver time [s]", xlim=(10., 1000.),
                ylim=(min(times.min()/3., 10**np.floor(np.log10(times.min()))),
                      max(times.max()*3., 10**np.ceil(np.log10(times.max())))))
    axes[2].grid(True, which="both", alpha=.3)

    case = summary["case"]
    title = (f"Premixed CH$_4$/air, $\\phi$ = {case['phi']}, {case['temperature']:.0f} K, 1 atm" if premixed
             else f"Counterflow CH$_4$/air, fluxes {case['fuel_mass_flux']}/{case['oxidizer_mass_flux']} kg m$^{{-2}}$ s$^{{-1}}$, 1 atm")
    figure.suptitle(title)
    stem = output/f"mechanisms_{summary['flame']}"
    figure.savefig(stem.with_suffix(".png"), dpi=160)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    return stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flame", choices=("premixed", "diffusion"), default="premixed")
    parser.add_argument("--mechanisms", nargs="+", help="Catalog filenames; default: every installed one")
    parser.add_argument("--output", type=Path, help="New folder; default: a dated folder in runs/")
    parser.add_argument("--plot-only", type=Path, help="Redraw from an existing summary.json")
    args = parser.parse_args()
    if args.plot_only:
        summary = json.loads(args.plot_only.read_text(encoding="utf-8"))
        output = args.output or args.plot_only.parent
        output.mkdir(parents=True, exist_ok=True)
        print(f"Figure: {plot(summary, output)}")
        return
    output = args.output or Path("runs")/"mechanisms"/f"{args.flame}-{datetime.now():%Y%m%d-%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    summary = run(args.flame, args.mechanisms, output)
    print(f"Summary: {output/'summary.json'}")
    print(f"Figure: {plot(summary, output)}")


if __name__ == "__main__":
    main()
