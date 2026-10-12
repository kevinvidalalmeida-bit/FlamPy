"""Multicomponent transport of large mechanisms: complete vs pruned Dixon--Lewis.

Run from the repository root after installing plotting support:

    python -m pip install -e ".[plots]"
    python examples/validate_multicomponent_trace.py
    python examples/validate_multicomponent_trace.py --cases premixed-H2 counterflow-C12H26
    python examples/validate_multicomponent_trace.py --plot-only runs/multicomponent_trace/<folder>

From 200 species, transport='multicomponent' omits by default the collisions
between two trace species (mole fractions adding up to at most 1e-8 at each
face) and solves with Newton--Krylov preconditioned by the structured
mixture-form Jacobian (kflame.chemistry.multicomponent_trace). Each case is
solved twice with JetSurF 2.0 (348 species): with the complete Dixon--Lewis
systems and with the default model. The complete premixed flame uses the
established dense multicomponent path (multicomponent_trace=None); the
complete counterflow flame uses a vanishing tail (every species a major,
i.e. the complete systems) with the same Newton--Krylov, because the dense
column Jacobian of that path would need ~4 GB and O(ns^3) work per node. Fuels and conditions
differ on purpose (H2 to n-dodecane, lean to rich, 1 and 10 atm, premixed and
counterflow): the majors are chosen per face from the local composition, so
nothing is tuned to a fuel. summary.json keeps the flame speeds, peak
temperatures, profile differences and runtimes; the figure is drawn only from
that file.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import kflame


# %% CASES
MECHANISM = "jetsurf2.yaml"
# Vanishing tail: no species is a trace, so the pruned kernel solves the
# complete Dixon--Lewis systems (checked against the complete kernel to 1e-11).
COMPLETE_TAIL = 1e-300
CASES = {
    "premixed-H2": dict(kind="premixed", fuel="H2", phi=0.6),
    "premixed-CH4": dict(kind="premixed", fuel="CH4", phi=1.0),
    "premixed-C2H4": dict(kind="premixed", fuel="C2H4", phi=1.0),
    "premixed-C7H16-lean": dict(kind="premixed", fuel="NC7H16", phi=0.8, temperature=400.0),
    "premixed-C7H16-rich": dict(kind="premixed", fuel="NC7H16", phi=1.3, temperature=400.0),
    "premixed-C12H26-10atm": dict(kind="premixed", fuel="NC12H26", phi=1.0, temperature=450.0,
                                  pressure=1013250.0),
    "counterflow-CH4": dict(kind="counterflow", fuel="CH4"),
    "counterflow-C12H26": dict(kind="counterflow", fuel="NC12H26:0.2,N2:0.8", fuel_temperature=450.0),
}


def solve(name, case, mode, folder):
    settings = {k: v for k, v in case.items() if k != "kind"}
    if mode == "pruned":
        trace = "auto"
    else:
        trace = None if case["kind"] == "premixed" else COMPLETE_TAIL
    if case["kind"] == "premixed":
        flame = kflame.solve_flame(mechanism=MECHANISM, transport="multicomponent", multicomponent_trace=trace,
                                   output=folder, max_time=3600.0, **settings)
        metric = dict(Su_m_s=float(flame["Su"]))
    else:
        flame = kflame.solve_diffusion_flame(mechanism=MECHANISM, transport="multicomponent",
                                             multicomponent_trace=trace, output=folder, max_time=7200.0, **settings)
        metric = dict(Tmax_K=float(np.max(flame["T"])))
    return dict(runtime_s=float(flame["runtime_s"]), nodes=int(len(flame["z"])), **metric,
                z_m=flame["z"].tolist(), T_K=flame["T"].tolist())


def compare(rows):
    """Profile difference of the pruned flame against the complete one, on the complete grid."""
    complete, pruned = rows["complete"], rows["pruned"]
    z, T = np.asarray(complete["z_m"]), np.asarray(complete["T_K"])
    T_pruned = np.interp(z, pruned["z_m"], pruned["T_K"])
    out = dict(max_abs_T_difference_K=float(np.max(np.abs(T_pruned-T))),
               speedup=complete["runtime_s"]/pruned["runtime_s"])
    for key in ("Su_m_s", "Tmax_K"):
        if key in complete:
            out[f"relative_{key}_difference"] = abs(pruned[key]-complete[key])/abs(complete[key])
    return out


def run(selected, output):
    summary = dict(mechanism=MECHANISM, created=datetime.now().isoformat(timespec="seconds"),
                   note=("pruned = default 'auto' (tail 1e-8); complete = multicomponent_trace=None (premixed) "
                         "or a vanishing tail (counterflow, same Newton-Krylov)"), cases={})
    for name in selected:
        case = CASES[name]
        rows = {}
        for mode in ("pruned", "complete"):
            print(f"{name}, {mode}...", flush=True)
            rows[mode] = solve(name, case, mode, output/name/mode)
            print(f"  {rows[mode]['runtime_s']:.1f} s, {rows[mode]['nodes']} nodes", flush=True)
        rows["comparison"] = compare(rows)
        rows["case"] = case
        summary["cases"][name] = rows
        print(f"  {json.dumps(rows['comparison'])}", flush=True)
        (output/"summary.json").write_text(json.dumps(summary, indent=1)+"\n", encoding="utf-8")
    return summary


# %% PLOTS
def plot(summary, output):
    names = list(summary["cases"])
    comparisons = [summary["cases"][name]["comparison"] for name in names]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    labels = [name.replace("premixed-", "P ").replace("counterflow-", "CF ") for name in names]
    ypos = np.arange(len(names))
    relative = [c.get("relative_Su_m_s_difference", c.get("relative_Tmax_K_difference")) for c in comparisons]
    axes[0].barh(ypos, np.maximum(relative, 1e-16), color="tab:blue")
    axes[0].set(xscale="log", yticks=ypos, yticklabels=labels, xlabel="relative difference (Su or Tmax)",
                title="Flame speed / peak temperature")
    axes[1].barh(ypos, np.maximum([c["max_abs_T_difference_K"] for c in comparisons], 1e-12), color="tab:orange")
    axes[1].set(xscale="log", yticks=ypos, yticklabels=[], xlabel="max |T_pruned - T_complete| [K]",
                title="Temperature profile")
    complete = [summary["cases"][name]["complete"]["runtime_s"] for name in names]
    pruned = [summary["cases"][name]["pruned"]["runtime_s"] for name in names]
    axes[2].barh(ypos+.2, complete, height=.4, color="tab:gray", label="complete kernel")
    axes[2].barh(ypos-.2, pruned, height=.4, color="tab:green", label="pruned (default)")
    axes[2].set(xscale="log", yticks=ypos, yticklabels=[], xlabel="solve time [s]", title="Runtime")
    axes[2].legend(fontsize="small")
    for axis in axes:
        axis.invert_yaxis()
        axis.grid(alpha=.3, axis="x")
    figure.suptitle(f"Multicomponent transport, {summary['mechanism']}: complete vs pruned trace collisions")
    stem = output/"multicomponent_trace"
    figure.savefig(stem.with_suffix(".png"), dpi=160)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)
    return stem.with_suffix(".png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--output", type=Path, help="New folder; default: a dated folder in runs/")
    parser.add_argument("--plot-only", type=Path, help="Folder (or summary.json) to redraw")
    args = parser.parse_args()
    if args.plot_only:
        path = args.plot_only if args.plot_only.suffix == ".json" else args.plot_only/"summary.json"
        print(f"Figure: {plot(json.loads(path.read_text(encoding='utf-8')), args.output or path.parent)}")
        return
    output = args.output or Path("runs")/"multicomponent_trace"/datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Figure: {plot(run(args.cases, output), output)}")


if __name__ == "__main__":
    main()
