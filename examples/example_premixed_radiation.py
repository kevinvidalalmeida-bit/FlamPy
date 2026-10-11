"""Optically thin radiation in premixed free and burner-stabilized flames.

Run from the repository root after installing plotting support:

    python -m pip install -e ".[plots]"
    python examples/example_premixed_radiation.py
    python examples/example_premixed_radiation.py --plot-only runs/premixed_radiation/<folder>

Solves CH4/air (phi=1, 300 K, 1 atm, GRI-Mech 3.0) as an adiabatic free flame
and with radiation, and burner flames at several radiation scales. The model
is the optically thin H2O/CO2 loss of the counterflow solver (Planck-mean
polynomials of Cantera's Flow1D). summary.json keeps the profiles and heat
budgets; the figure is drawn only from that file.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import kflame


# %% CASE CONFIGURATION
FLAME = dict(mechanism="gri30.yaml", fuel="CH4", phi=1.0, temperature=300.0, pressure=101325.0, max_time=300.0)
BURNER_MASS_FLUX = 0.08            # kg/(m^2 s)
BURNER_SCALES = (0.0, 1.0, 3.0)    # radiation_scale; 0 = burner loss only


def solve(output):
    rows = {}
    for label, settings in (("free, adiabatic", {}), ("free, radiation", dict(radiation=True))):
        flame = kflame.solve_flame(**FLAME, output=output/label.replace(", ", "_"), **settings)
        rows[label] = dict(kind="free", Su_m_s=float(flame["Su"]), runtime_s=float(flame["runtime_s"]),
                           z_m=flame["z"].tolist(), T_K=flame["T"].tolist(),
                           qrad_W_m3=flame.get("radiative_heat_loss", np.zeros_like(flame["z"])).tolist())
        print(f"{label}: Su = {flame['Su']:.6g} m/s, {flame['runtime_s']:.1f} s", flush=True)
    for scale in BURNER_SCALES:
        label = f"burner, radiation x{scale:g}"
        settings = dict(radiation=True, radiation_scale=scale) if scale > 0 else {}
        flame = kflame.solve_burner_flame(**FLAME, mass_flux=BURNER_MASS_FLUX,
                                          output=output/f"burner_x{scale:g}", **settings)
        loss = flame["heat_loss"]
        rows[label] = dict(kind="burner", radiation_scale=scale, runtime_s=float(flame["runtime_s"]),
                           z_m=flame["z"].tolist(), T_K=flame["T"].tolist(),
                           wall_W_m2=float(loss["burner_heat_loss_W_m2"]),
                           radiation_W_m2=float(loss.get("radiative_heat_loss_W_m2", 0.)),
                           outlet_W_m2=float(loss["outlet_enthalpy_loss_W_m2"]),
                           closure=float(loss["relative_energy_closure_error"]))
        print(f"{label}: wall {loss['burner_heat_loss_W_m2']/1e3:.1f} kW/m2, radiation "
              f"{loss.get('radiative_heat_loss_W_m2', 0.)/1e3:.1f} kW/m2, energy closure "
              f"{100*loss['relative_energy_closure_error']:.2f} %", flush=True)
    summary = dict(case=FLAME, burner_mass_flux=BURNER_MASS_FLUX, created=datetime.now().isoformat(timespec="seconds"),
                   flames=rows)
    (output/"summary.json").write_text(json.dumps(summary, indent=1)+"\n", encoding="utf-8")
    return summary


# %% PLOTS
def plot(summary, output):
    flames = summary["flames"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    for label, style in (("free, adiabatic", dict(color="black")), ("free, radiation", dict(color="tab:red", ls="--"))):
        row = flames[label]
        z, T = np.asarray(row["z_m"]), np.asarray(row["T_K"])
        front = z[np.argmax(np.gradient(T, z))]
        axes[0].plot(1e3*(z-front), T, label=f"{label}: Su = {100*row['Su_m_s']:.2f} cm/s", **style)
    axes[0].set(xlabel="Distance to the flame front [mm]", ylabel="Temperature [K]", xlim=(-2., 40.),
                title="Free flame: radiating products keep cooling")
    axes[0].legend(fontsize="small")
    burners = [(label, row) for label, row in flames.items() if row["kind"] == "burner"]
    colors = plt.cm.plasma(np.linspace(0., .8, len(burners)))
    for (label, row), color in zip(burners, colors):
        axes[1].plot(1e3*np.asarray(row["z_m"]), row["T_K"], color=color, label=label.split(", ")[1])
    axes[1].set(xlabel="Distance from the burner [mm]", ylabel="Temperature [K]",
                title=f"Burner, mass flux {summary['burner_mass_flux']} kg m$^{{-2}}$ s$^{{-1}}$")
    axes[1].legend(fontsize="small")
    names = [label.split(", ")[1] for label, _ in burners]
    wall = np.array([row["wall_W_m2"] for _, row in burners])/1e3
    radiation = np.array([row["radiation_W_m2"] for _, row in burners])/1e3
    outlet = np.array([row["outlet_W_m2"] for _, row in burners])/1e3
    axes[2].bar(names, wall, color="tab:blue", label="conduction to the burner")
    axes[2].bar(names, radiation, bottom=wall, color="tab:orange", label="radiation")
    axes[2].plot(names, outlet, "kD", label=r"$\dot m\,(h_{feed}-h_{out})$")
    for x, (_, row) in enumerate(burners):
        axes[2].annotate(f"{100*row['closure']:.1f} %", (x, max(outlet[x], wall[x]+radiation[x])),
                         textcoords="offset points", xytext=(0, 8), ha="center", fontsize="small")
    axes[2].set(ylabel="Heat loss [kW/m$^2$]", title="Burner heat budget (label: energy closure)")
    axes[2].legend(fontsize="small")
    figure.suptitle("Premixed CH$_4$/air with optically thin radiation")
    stem = output/"premixed_radiation"
    figure.savefig(stem.with_suffix(".png"), dpi=160)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    return stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, help="New folder; default: a dated folder in runs/")
    parser.add_argument("--plot-only", type=Path, help="Folder (or summary.json) to redraw")
    args = parser.parse_args()
    if args.plot_only:
        path = args.plot_only if args.plot_only.suffix == ".json" else args.plot_only/"summary.json"
        print(f"Figure: {plot(json.loads(path.read_text(encoding='utf-8')), args.output or path.parent)}")
        return
    output = args.output or Path("runs")/"premixed_radiation"/datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Figure: {plot(solve(output), output)}")


if __name__ == "__main__":
    main()
