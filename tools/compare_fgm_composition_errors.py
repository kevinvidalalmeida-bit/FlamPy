"""All-species retained-flame composition diagnostics for an existing FGM table.

Reports a global L2 composition error and a conservative L-infinity envelope
over species whose maximum retained mass fraction exceeds a configurable floor.
No flame is resolved and no scientific input is modified.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from kflame.fgm.common import monotonicize_on_c


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files if name != "predictor_kind"}


def scalar(value):
    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def metric(values):
    return dict(mean_percent=float(values.mean()), p95_percent=float(np.quantile(values, .95)),
                max_percent=float(values.max()))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--construction", type=Path, required=True, help="Accepted attempt containing fgm_table.npz")
    p.add_argument("--holdouts", type=Path, required=True, help="Attempt or directory containing profiles/")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--species-floor", type=float, default=1e-5,
                   help="Maximum retained Y required for a species to enter the L-infinity envelope")
    p.add_argument("--label", default="FGM", help="Short identifier written in the report")
    args = p.parse_args(argv)
    source, held, out = args.construction.resolve(), args.holdouts.resolve(), args.output.resolve()
    table = read_npz(source / "fgm_table.npz")
    profiles = held / "profiles" if (held / "profiles").is_dir() else held
    records = [{key: scalar(value) for key, value in read_npz(path).items()}
               for path in sorted(profiles.glob("*.npz"))]
    if not records:
        raise ValueError("No retained profiles found.")
    names = [str(x) for x in table["species_names"]]
    if any([str(x) for x in r["species_names"]] != names for r in records):
        raise ValueError("Species lists differ between the FGM table and retained flames.")
    phi = np.array([float(r["phi"]) for r in records])
    if np.isclose(phi[:, None], table["phi_grid"][None, :], rtol=0, atol=1e-12).any():
        raise ValueError("Holdout leakage: retained composition equals a construction row.")
    # Persisted FGM arrays use (Z, species, c); interpolation coordinates are
    # (Z, c), hence move the trailing species dimension after c.
    table_y = np.moveaxis(table["Y"], 1, -1)
    interp = RegularGridInterpolator((table["Z_grid"], table["c_grid"]), table_y, bounds_error=True)
    c = np.linspace(0, 1, 1001)
    truth, predicted = [], []
    for r in records:
        if not (r["solve_ok"] and r["final_accepted"]):
            raise ValueError(f"Unaccepted retained flame at phi={r['phi']}")
        raw_c = (r["beta"] - r["beta"][0]) / (r["beta"][-1] - r["beta"][0])
        if np.min(np.diff(raw_c)) < -1e-7:
            raise ValueError(f"Nonmonotone retained progress at phi={r['phi']}")
        # The helper accepts scalar profiles.  Apply the identical c cleanup
        # to every species before sampling all of them on the common grid.
        reconstructed = []
        for k in range(len(names)):
            cu, fields = monotonicize_on_c(r["c"], {"Y": r["Y"][k]})
            reconstructed.append(np.interp(c, cu, fields["Y"]))
        truth.append(np.column_stack(reconstructed))
        predicted.append(interp(np.column_stack([np.full(c.size, r["Z"]), c])))
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    # One L2 value at each physical state: it is the relative error of the
    # complete composition vector, so all species participate simultaneously.
    e2 = 100 * np.linalg.norm(predicted - truth, axis=2) / np.maximum(np.linalg.norm(truth, axis=2), args.species_floor)
    species_scale = np.max(np.abs(truth), axis=(0, 1))
    included = species_scale > args.species_floor
    if not np.any(included):
        raise ValueError("No species passed the requested significance floor.")
    normalized = 100 * np.abs(predicted[:, :, included] - truth[:, :, included]) / species_scale[included]
    einf = normalized.max(axis=2)
    worst_index = np.unravel_index(int(np.argmax(normalized)), normalized.shape)
    included_indices = np.flatnonzero(included)
    worst_species_index = int(included_indices[worst_index[2]])
    rows = [dict(metric="e_Y_2", definition="relative L2 error of the complete mass-fraction vector", **metric(e2)),
            dict(metric="e_Y_infty", definition=f"maximum specieswise normalized error; Ymax > {args.species_floor:g}", **metric(einf))]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "composition_error_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    details = dict(label=args.label, n_holdouts=len(records), samples_per_holdout=len(c), species_total=len(names),
                   species_floor=args.species_floor, species_in_linf=int(included.sum()),
                   excluded_trace_species=[names[i] for i in np.flatnonzero(~included)],
                   phi=phi.tolist(), species_scales={names[i]: float(species_scale[i]) for i in np.flatnonzero(included)},
                   linf_worst_case=dict(species=names[worst_species_index], phi=float(phi[worst_index[0]]),
                       c=float(c[worst_index[1]]), reference=float(truth[worst_index[0], worst_index[1], worst_species_index]),
                       prediction=float(predicted[worst_index[0], worst_index[1], worst_species_index]),
                       species_scale=float(species_scale[worst_species_index]), error_percent=float(normalized[worst_index])),
                   metrics=rows)
    (out / "composition_error_summary.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    lines = [r"\begin{table}[H]\centering\small",
             r"\caption{Diagnosticos globales de interpolacion de la composicion completa en llamas de retencion independientes. Cada valor resume 1001 posiciones uniformes de $c$ por llama, con igual peso por llama. $e_{Y,2}$ usa simultaneamente todas las fracciones masicas; $e_{Y,\infty}$ es la envolvente del error normalizado por especie, excluyendo especies con $\max|Y_k|\leq"+f"{args.species_floor:g}"+r".}",
             r"\label{tab:"+args.label+r"-composition-errors}",
             r"\begin{tabular}{lrrr}\toprule Diagnostico & Media [\%] & P95 [\%] & Maximo [\%]\\\midrule"]
    tex_metric = {"e_Y_2": r"e_{Y,2}", "e_Y_infty": r"e_{Y,\infty}"}
    lines += [f"${tex_metric[row['metric']]}$ & {row['mean_percent']:.3g} & "
              f"{row['p95_percent']:.3g} & {row['max_percent']:.3g}\\\\" for row in rows]
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (out / "table_composition_errors.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
