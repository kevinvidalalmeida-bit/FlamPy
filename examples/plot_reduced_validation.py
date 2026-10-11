"""Figures of the reduced burner validation against second-order native references.

Uses only saved files: the JSON written by
``examples.validate_against_native_references`` and its ``--profiles-out``
folder. To redraw the published figures:

    python -m examples.plot_reduced_validation --validation docs/assets/reduced-burner-regen/validation.json --profiles docs/assets/reduced-burner-regen/profiles --output output/figures/reduced-burner-regen

To reproduce the data (regenerated library, references and table as in
docs/heat-loss-review.md), then plot:

    python -m examples.validate_against_native_references --table runs/regen_table --profiles runs/regen_exponential --references runs/regen_references --output runs/regen_validation.json --profiles-out runs/regen_profiles --jobs 6
    python -m examples.plot_reduced_validation --validation runs/regen_validation.json --profiles runs/regen_profiles --output output/figures/reduced-burner-regen
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Cases drawn as profiles: the two closest to the adiabatic limit.
PROFILE_CASES = ("case_0.985000_0.950000", "case_1.275000_0.935000")


def load(validation):
    data = json.loads(Path(validation).read_text(encoding="utf-8"))
    rows = data["results"]
    for row in rows:
        row["r"] = float(row["case"].split("_")[-1])
        row["q_signed"] = (row["q_reduced_W_m2"]-row["q_reference_W_m2"])/row["q_reference_W_m2"]
    return data, rows


def plot_errors(data, rows, output):
    limits = data["limits"]
    phi = np.array([row["phi"] for row in rows])
    r = np.array([row["r"] for row in rows])
    passed = np.array([row["passed"] for row in rows])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")

    q = 100*np.array([row["q_signed"] for row in rows])
    unc = 100*np.array([row["reference_grid_uncertainty"] for row in rows])
    scatter = axes[0].scatter(r, q, c=phi, cmap="viridis", s=36, edgecolors=np.where(passed, "black", "red"),
                              linewidths=np.where(passed, .5, 1.8), zorder=3)
    axes[0].errorbar(r, q, yerr=unc, fmt="none", ecolor="0.5", elinewidth=.8, zorder=2)
    for bound in (-100*limits["heat_loss_relative"], 100*limits["heat_loss_relative"]):
        axes[0].axhline(bound, color="tab:red", ls="--", lw=1)
    axes[0].set(xlabel="Mass-flux fraction r", ylabel="Heat loss error, reduced - reference [%]",
                title="Heat to the burner (bars: reference mesh uncertainty)")
    figure.colorbar(scatter, ax=axes[0], label=r"$\phi$")

    dT_C = np.array([row["metrics"]["temperature_K"] for row in rows])
    dT_z = np.array([row["same_z_metrics"]["temperature_K"] for row in rows])
    axes[1].semilogy(r, dT_z, "s", mfc="none", color="0.45", label="same position z (reported)")
    axes[1].semilogy(r, dT_C, "o", color="tab:blue", label="equal progress C (criterion)")
    axes[1].axhline(limits["temperature_K"], color="tab:red", ls="--", lw=1, label="limit")
    axes[1].set(xlabel="Mass-flux fraction r", ylabel="Largest temperature error [K]",
                title="State error: position vs. progress")
    axes[1].legend(fontsize="small")

    shift = 1e6*np.array([row["metrics"]["stand_off_distance_m"] for row in rows])
    axes[2].plot(r, shift, "o", color="tab:green")
    axes[2].axhline(1e6*limits["stand_off_distance_m"], color="tab:red", ls="--", lw=1, label="limit")
    axes[2].set(xlabel="Mass-flux fraction r", ylabel=r"Front stand-off difference [$\mu$m]",
                title="Flame position")
    axes[2].legend(fontsize="small")
    figure.suptitle(f"Reduced FGM burner, {data['cases']} independent cases: "
                    f"{data['accepted']} converge, {data['passed']} pass")
    return _save(figure, output/"01_validation_errors")


def plot_profiles(profiles, rows, output):
    by_case = {row["case"]: row for row in rows}
    figure, axes = plt.subplots(3, len(PROFILE_CASES), figsize=(6.2*len(PROFILE_CASES), 11), layout="constrained")
    for column, case in enumerate(PROFILE_CASES):
        with np.load(Path(profiles)/f"{case}.npz") as saved:
            p = {key: saved[key] for key in saved.files}
        row = by_case[case]
        front = p["z_ref"][np.argmax(np.gradient(p["T_ref"], p["z_ref"]))]
        top = axes[0, column]
        top.plot(1e3*(p["z_ref"]-front), p["T_ref"], color="black", lw=2, label="detailed reference")
        top.plot(1e3*(p["z"]-front), p["T"], color="tab:orange", ls="--", lw=1.6, label="reduced FGM")
        top.set(xlim=(-.4, .8), xlabel="Distance to the reference front [mm]", ylabel="Temperature [K]",
                title=f"$\\phi$ = {row['phi']}, r = {row['r']}: "
                      f"shift {1e6*row['metrics']['stand_off_distance_m']:.1f} $\\mu$m")
        top.legend(fontsize="small", loc="lower right")
        bottom = axes[1, column]
        bottom.plot(p["C_ref"], p["T_ref"], color="black", lw=2, label="detailed reference")
        bottom.plot(p["C"], p["T"], color="tab:orange", ls="--", lw=1.6, label="reduced FGM")
        bottom.set(xlabel="Progress C", ylabel="Temperature [K]",
                   title=f"Equal C: {row['metrics']['temperature_K']:.1f} K "
                         f"(same z: {row['same_z_metrics']['temperature_K']:.1f} K)")
        bottom.legend(fontsize="small", loc="lower right")
        source = axes[2, column]
        source.plot(p["C_ref"], p["omega_C_ref"], color="black", lw=2, label="detailed reference")
        source.plot(p["C"], p["omega_C"], color="tab:orange", ls="--", lw=1.6, label="reduced FGM (tabulated)")
        source.set(xlabel="Progress C", ylabel=r"Progress source $\Omega_C$ [kg m$^{-3}$ s$^{-1}$]",
                   title=f"Peak error at equal C: {100*row['sources']['omega_C']['peak_relative']:.2f} %")
        source.legend(fontsize="small", loc="upper left")
    figure.suptitle("Near-adiabatic cases: the reduced flame state matches; the front sits a few microns away")
    return _save(figure, output/"02_near_adiabatic_profiles")


def _save(figure, stem):
    figure.savefig(stem.with_suffix(".png"), dpi=160)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    return stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data, rows = load(args.validation)
    for path in (plot_errors(data, rows, args.output), plot_profiles(args.profiles, rows, args.output)):
        print(f"Figure: {path}")


if __name__ == "__main__":
    main()
