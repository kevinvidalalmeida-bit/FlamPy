"""Offline figure review for Chapter 4: archived CSV/JSON/NPZ, no flame solves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
from scipy.stats import gaussian_kde
from pypdf import PdfReader, PdfWriter

import postprocess_flame_sweeps as pp


SOLVERS = ("native", "cantera")
NAMES = {"native": "KFlame", "cantera": "Cantera"}
COLORS = {"native": "#1769aa", "cantera": "#b84b32"}
FUEL_NAMES = {"CH4": r"CH$_4$ / GRI-Mech 3.0", "H2": r"H$_2$ / h2o2"}
MODES = [("mixture-averaged", False), ("mixture-averaged", True),
         ("multicomponent", False), ("multicomponent", True)]
MODE_NAMES = ["Promediado", "Promediado + Soret",
              "Multicomponente", "Multicomponente + Soret"]
MODE_COLORS = ["#1769aa", "#d07b28", "#2e8b70", "#8b5897"]
STATES = [(1, 300), (10, 300), (1, 500)]
STATE_COLORS = ["#1769aa", "#b84b32", "#2e8b70"]
CAPTIONS = {}
OUTPUTS = []


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    v = row.get(key)
    return None if v in (None, "", "null") else float(v)


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 9.5,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": False,
        "lines.linewidth": 1.6, "lines.markersize": 4,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })


def save(fig, out, name, caption):
    fig.savefig(out / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(out / (name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    OUTPUTS.append(name)
    CAPTIONS[name] = caption
    height = ".60" if name == "15_precision_speedup" else ".70"
    tex = (
        "\\begin{figure}[!htbp]\n\\centering\n"
        "\\includegraphics[width=\\textwidth,height=" + height + "\\textheight,keepaspectratio]"
        "{\\SweepReportRoot/revision_figures/" + name + ".pdf}\n"
        "\\caption{" + caption + "}\n\\label{fig:rev-" + name.replace("_", "-")
        + "}\n\\end{figure}\n"
    )
    (out / (name + ".tex")).write_text(tex, encoding="utf-8")


def solver_handles(color=".2"):
    return [Line2D([], [], color=color, ls="-" if b == "native" else "--",
                   label=NAMES[b]) for b in SOLVERS]


def row_label(ax, fuel):
    ax.annotate(FUEL_NAMES[fuel], xy=(0, .5), xycoords="axes fraction",
                xytext=(-76, 0), textcoords="offset points", rotation=90,
                ha="center", va="center", fontsize=10, fontweight="bold",
                annotation_clip=False)


def profile_figures(campaign, records, out):
    index = []
    for fuel in ("CH4", "H2"):
        fig, axes = plt.subplots(4, 3, figsize=(10.2, 9.1),
                                 gridspec_kw={"width_ratios": (1.0, 1.0, .50)})
        fig.subplots_adjust(left=.08, right=.97, top=.83, bottom=.08,
                            hspace=.60, wspace=.62)
        spcolors = {fuel: "#1769aa", "O2": "#d07b28",
                    "CO2": "#2e8b70", "H2O": "#8b5897"}
        species = [fuel, "O2"] + (["CO2"] if fuel == "CH4" else []) + ["H2O"]
        species_tex = {"CH4": r"CH$_4$", "H2": r"H$_2$",
                       "O2": r"O$_2$", "CO2": r"CO$_2$", "H2O": r"H$_2$O"}
        limits = (-.8, 2.) if fuel == "CH4" else (-.4, 1.)
        fields, selected_records = {}, {}
        for i, mode in enumerate(MODES):
            for backend in SOLVERS:
                rr = [r for r in records if r.get("usable") and r["phase"] == "main"
                      and r["variant"] == backend
                      and r["condition"]["fuel"] == fuel
                      and r["condition"]["phi"] == 1
                      and r["condition"]["pressure_atm"] == 1
                      and r["condition"]["temperature"] == 300
                      and (r["condition"]["transport"], r["condition"]["soret"]) == mode]
                record = min(rr, key=lambda r: r["repetition"])
                fields[i, backend] = pp.profile(campaign, record)
                selected_records[i, backend] = record
                index.append({"figure": "02_perfiles_" + fuel,
                              "case_id": record["case_id"], "solver": backend,
                              "repetition": record["repetition"], "folder": record["folder"]})
        profile_errors = {
            i: pp.comparison(campaign, selected_records[i, "native"],
                             selected_records[i, "cantera"])
            for i in range(len(MODES))
        }
        error_metrics = [("Su_percent", r"$|\Delta S_u|$", "#1769aa"),
                         ("T_L2_percent", r"$E_2(T)$", "#b84b32"),
                         ("qdot_L2_percent", r"$E_2(\dot q)$", ".25"),
                         ("max_species_L2_percent", r"$\max E_2(Y_k)$", "#8b5897")]
        error_values = np.array([profile_errors[i][key] for i in range(len(MODES))
                                 for key, _, _ in error_metrics])
        error_min = 10 ** np.floor(np.log10(error_values.min()))
        error_max = 10 ** np.ceil(np.log10(error_values.max()))
        # Fixed y-ranges within a mechanism preserve visible transport effects.
        umax = max(float(f["u"].max()) for f in fields.values()) * 1.05
        qmax = max(float(f["qdot"].max()) for f in fields.values()) * 1.e-9 * 1.10
        for i, mode in enumerate(MODES):
            at, ay, ae = axes[i]
            au, aq = at.twinx(), ay.twinx()
            for twin in (au, aq):
                twin.spines["right"].set_visible(True)
                twin.spines["top"].set_visible(False)
                twin.grid(False)
            for backend in SOLVERS:
                f = fields[i, backend]
                z = pp.aligned(f) * 1.e3
                ls = "-" if backend == "native" else "--"
                marker_options = {}
                if backend == "cantera":
                    targets = np.linspace(limits[0] + .05, limits[1] - .05, 9)
                    selected = sorted(set(int(np.argmin(abs(z - xx))) for xx in targets))
                    marker_options = dict(marker="o", mfc="white", ms=2.6, mew=.7,
                                          markevery=selected)
                at.plot(z, f["T"], color="#b84b32", ls=ls, **marker_options)
                au.plot(z, f["u"], color="#1769aa", ls=ls, **marker_options)
                names = list(f["species_names"])
                for sp in species:
                    ay.plot(z, f["Y"][names.index(sp)], color=spcolors[sp], ls=ls,
                            **marker_options)
                aq.plot(z, f["qdot"] * 1.e-9, color=".25", ls=ls, lw=1.3, **marker_options)
            at.set_ylim(250, 2500)
            au.set_ylim(0, umax)
            ay.set_ylim(0, .26)
            aq.set_ylim(0, qmax)
            at.set_ylabel(r"$T$ [K]", color="#b84b32")
            au.set_ylabel(r"$u$ [m/s]", color="#1769aa")
            ay.set_ylabel(r"$Y_k$ [-]")
            aq.set_ylabel(r"$\dot q$ [GW/m$^3$]", color=".25")
            at.tick_params(axis="y", colors="#b84b32")
            au.tick_params(axis="y", colors="#1769aa")
            aq.tick_params(axis="y", colors=".25")
            at.spines["left"].set_color("#b84b32")
            au.spines["right"].set_color("#1769aa")
            aq.spines["right"].set_color(".25")
            for ax in (at, ay, au, aq, ae):
                ax.spines["top"].set_visible(False)
            for ax in (at, ay):
                ax.set_xlim(limits)
                ax.set_title(MODE_NAMES[i], fontsize=10, pad=7)
                if i == 3:
                    ax.set_xlabel(r"$x-x_{1/2}$ [mm]")
            for ax in (au, aq):
                ax.set_xlim(limits)
            for j, (key, label, color) in enumerate(error_metrics):
                value = profile_errors[i][key]
                ae.plot([error_min, value], [j, j], color=color, lw=1.0)
                ae.scatter(value, j, s=20, color=color, zorder=3)
                ae.annotate(f"{value:.2g}", (value, j), xytext=(3, 0),
                            textcoords="offset points", va="center", fontsize=6.5,
                            color=color)
            ae.set_xscale("log")
            ae.set_xlim(error_min / 1.35, error_max * 1.9)
            ae.set_ylim(3.5, -.5)
            ae.set_yticks(range(len(error_metrics)), [label for _, label, _ in error_metrics])
            ae.tick_params(axis="y", labelsize=7, length=0)
            ae.tick_params(axis="x", labelsize=7)
            ae.spines["top"].set_visible(False)
            ae.spines["right"].set_visible(False)
            if i == 0:
                ae.set_title("Concordancia", fontsize=10, pad=7)
            if i == 3:
                ae.set_xlabel("Error relativo [%]", fontsize=8)
            else:
                ae.tick_params(axis="x", labelbottom=False)
        fig.text(.5, .985, FUEL_NAMES[fuel] + r" : $\phi=1$, 1 atm, 300 K",
                 ha="center", va="top", fontsize=12)
        fig.legend(handles=[Line2D([], [], color=".2", label="KFlame"),
                            Line2D([], [], color=".2", ls="--", marker="o", mfc="white",
                                   label="Cantera")], ncol=2, loc="upper center",
                   bbox_to_anchor=(.5, .965), frameon=False)
        fig.legend(handles=[Line2D([], [], color=spcolors[s], label=species_tex[s])
                            for s in species] + [Line2D([], [], color=".25",
                            label=r"$\dot q$ (eje derecho)")],
                   loc="upper center", bbox_to_anchor=(.5, .925), ncol=5, frameon=False)
        fig.text(.235, .875, "Temperatura y velocidad local", ha="center", fontsize=10)
        fig.text(.585, .875, "Especies y liberación de calor", ha="center", fontsize=10)
        save(fig, out, "02_perfiles_" + fuel,
             rf"Frente de llama de {'CH$_4$--aire (GRI-Mech 3.0)' if fuel == 'CH4' else 'H$_2$--aire (h2o2)'} "
             r"a $\phi=1$, 1 atm y 300 K. Las filas mantienen las cuatro combinaciones "
             r"de transporte; línea continua: KFlame, discontinua: Cantera. "
             r"Los círculos blancos señalan algunos nodos originales de Cantera para "
             r"distinguir los perfiles cuando las curvas coinciden. "
             r"En la columna izquierda, temperatura en el eje izquierdo y velocidad local "
             r"$u(x)$ en el derecho; $u(x)$ incluye la expansión de los gases, mientras "
             r"que la velocidad de propagación $S_u$ corresponde a la entrada. En la derecha, "
             r"fracciones másicas en el eje izquierdo y calor liberado volumétrico en el derecho. "
             r"La tercera columna resume la discrepancia relativa de los mismos perfiles: "
             r"$|\Delta S_u|$, $E_2(T)$, $E_2(\dot q)$ y el mayor $E_2(Y_k)$ entre especies activas; "
             r"cada segmento parte del menor valor mostrado y la abscisa es logarítmica. "
             r"Se amplía el frente alrededor del punto medio térmico $x_{1/2}$ de cada solución. "
             r"Los límites de los ejes se mantienen entre transportes dentro de cada mecanismo. "
             r"Cada perfil corresponde a una ejecución individual: la primera repetición aceptada de L3.")
    pp.write_csv(out / "perfiles_seleccionados.csv", index)
    # Keep the previously opened filename useful, now as two successive pages.
    writer = PdfWriter()
    for fuel in ("CH4", "H2"):
        writer.append(str(out / ("02_perfiles_" + fuel + ".pdf")))
    writer.write(str(out / "02_perfiles_concordancia.pdf"))
    shutil.copyfile(out / "02_perfiles_CH4.png", out / "02_perfiles_concordancia.png")


def summary_rows(summary, config):
    byid = {r["id"]: r for r in config}
    result = []
    for r in summary:
        if not int(r["n_pairs"]):
            continue
        c = dict(byid[r["case_id"]])
        c.update(phi=float(c["phi"]), pressure_atm=float(c["pressure_atm"]),
                 temperature=float(c["temperature"]),
                 soret=str(c["soret"]).lower() in ("true", "1", "yes"))
        med, sol = json.loads(r["paired_medians"]), json.loads(r["solvers"])
        result.append(dict(c, ratio=float(r["ratio"]), ci=json.loads(r["ci95"]),
                           native=med["native"], cantera=med["cantera"],
                           timing=sol,
                           Su_native=sol["native"]["Su"]["median"],
                           Su_cantera=sol["cantera"]["Su"]["median"]))
    return result


def representative(r):
    return (r["transport"], r["soret"]) == (MODES[0] if r["fuel"] == "CH4" else MODES[3])


def full_transport(rows):
    return len(rows) == 104


def sweep_scope(rows):
    if full_transport(rows):
        return r"Los tres barridos incluyen las cuatro combinaciones de transporte en ambos mecanismos. "
    return (r"Composición: cuatro transportes. Presión y temperatura: CH$_4$/GRI-Mech 3.0 "
            r"promediado sin Soret e H$_2$/h2o2 multicomponente con Soret, según el diseño original. ")


def errors_figure(errors, config, out):
    byid = {r["id"]: r["fuel"] for r in config}
    metrics = [("Su_percent", r"$S_u$"), ("T_L2_percent", r"$E_2(T)$"),
               ("qdot_L2_percent", r"$E_2(\dot q)$"),
               ("max_species_L2_percent", r"$\max_k E_2(Y_k)$")]
    grouped = {}
    for r in errors:
        grouped.setdefault(r["case_id"], []).append(r)
    aggregated = [dict(case_id=cid, fuel=byid[cid], pairs=len(rr),
                      **{key: float(np.median([number(r, key) for r in rr]))
                         for key, _ in metrics}) for cid, rr in grouped.items()]
    pp.write_csv(out / "concordancia_por_condicion.csv", aggregated)
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.19, right=.98, top=.96, bottom=.10, hspace=.27)
    rng = np.random.default_rng(20260927)
    for ax, fuel, color in zip(axes, ("CH4", "H2"), ("#1769aa", "#b84b32")):
        for j, (key, label) in enumerate(metrics):
            vals = np.array([r[key] for r in aggregated if r["fuel"] == fuel])
            if np.any(vals <= 0):
                raise ValueError("Log violin requires strictly positive discrepancies")
            logs = np.log10(vals)
            support = np.linspace(logs.min(), logs.max(), 150)
            density = gaussian_kde(logs)(support)
            width = .34 * density / density.max()
            ax.fill_betweenx(10 ** support, j-width, j+width, color=color, alpha=.22, lw=0)
            ax.scatter(j + rng.uniform(-.055, .055, len(vals)), vals,
                       s=10, color=color, alpha=.50, zorder=3)
            q1, med, q3 = np.quantile(vals, [.25, .5, .75])
            ax.plot([j, j], [q1, q3], color=color, lw=3)
            ax.scatter(j, med, s=24, facecolor="white", edgecolor=color, zorder=5)
        ax.set_yscale("log")
        ax.set_ylabel("Discrepancia [%]")
        ax.set_xticks(range(4), [v for _, v in metrics])
        ax.set_xlim(-.6, 3.6)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        row_label(ax, fuel)
    save(fig, out, "12_concordancia_global",
         f"Concordancia de las {len(config)} condiciones de L3: {len(config)//2} por mecanismo. "
         r"Cada punto representa la mediana de las discrepancias de sus cinco pares, "
         r"por lo que las repeticiones no se cuentan como condiciones adicionales. "
         r"Los violines muestran una densidad suavizada en $\log_{10}$ de la discrepancia; "
         r"tramo grueso: cuartiles; círculo blanco: mediana. "
         r"$S_u$: discrepancia relativa absoluta respecto a Cantera; "
         r"$E_2(f)=100\|f_{\mathrm{K}}-f_{\mathrm{C}}\|_2/\|f_{\mathrm{C}}\|_2$, "
         r"con perfiles alineados e interpolados en el intervalo común. "
         r"El máximo de especies incluye aquellas cuyo máximo de fracción másica alcanza $10^{-5}$.")


SWEEPS = [
    ("phi", r"$\phi$ [-]", lambda r: r["pressure_atm"] == 1 and r["temperature"] == 300),
    ("pressure_atm", r"$p$ [atm]", lambda r: r["phi"] == 1 and r["temperature"] == 300),
    ("temperature", r"$T_u$ [K]", lambda r: r["phi"] == 1 and r["pressure_atm"] == 1)]


def pressure_axis(ax, field):
    if field == "pressure_atm":
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def physical_figure(rows, out):
    fig, axes = plt.subplots(2, 3, figsize=(8.3, 5.5))
    fig.subplots_adjust(left=.12, right=.99, top=.77, bottom=.11, wspace=.42, hspace=.43)
    for i, fuel in enumerate(("CH4", "H2")):
        for j, (field, label, rule) in enumerate(SWEEPS):
            ax = axes[i, j]
            for k, mode in enumerate(MODES):
                if j and not full_transport(rows) and mode != (MODES[0] if fuel == "CH4" else MODES[3]):
                    continue
                rr = sorted([r for r in rows if r["fuel"] == fuel and rule(r)
                             and (r["transport"], r["soret"]) == mode],
                            key=lambda r: r[field])
                if len(rr) != 5:
                    raise ValueError(f"Incomplete physical sweep: {fuel}/{mode}/{field}")
                for backend in SOLVERS:
                    ax.plot([r[field] for r in rr], [r["Su_" + backend] for r in rr],
                            color=MODE_COLORS[k], ls="-" if backend == "native" else "--",
                            marker="o" if backend == "native" else "s", mfc="white", ms=3)
            ax.set_xlabel(label)
            ax.set_ylabel(r"$S_u$ [m/s]")
            pressure_axis(ax, field)
            if i == 0:
                ax.set_title(("Composición", "Presión", "Temperatura")[j])
        row_label(axes[i, 0], fuel)
    fig.legend(handles=[Line2D([], [], color=c, label=n)
                        for c, n in zip(MODE_COLORS, MODE_NAMES)],
               loc="upper center", ncol=2, frameon=False)
    fig.legend(handles=solver_handles(), loc="upper center",
               bbox_to_anchor=(.5, .89), ncol=2, frameon=False)
    save(fig, out, "13_respuesta_fisica",
         r"Velocidad de propagación en los barridos de composición, presión y temperatura. "
         + sweep_scope(rows) +
         r"El color identifica el transporte y el trazo el solver. "
         r"Al variar $\phi$ se mantienen 1 atm y 300 K; "
         r"al variar $p$, $\phi=1$ y 300 K; al variar $T_u$, $\phi=1$ y 1 atm. "
         r"Los puntos son medianas de cinco ejecuciones; las líneas unen los estados muestreados.")


def speedup_figure(rows, out):
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.5))
    fig.subplots_adjust(left=.11, right=.99, top=.80, bottom=.10, wspace=.40, hspace=.43)
    for i, fuel in enumerate(("CH4", "H2")):
        upper = 1.08 * max(r['ci'][1] for r in rows if r['fuel'] == fuel)
        for j, (field, label, rule) in enumerate(SWEEPS):
            ax = axes[i, j]
            for k, mode in enumerate(MODES):
                if j and not full_transport(rows) and mode != (MODES[0] if fuel == "CH4" else MODES[3]):
                    continue
                rr = sorted([r for r in rows if r["fuel"] == fuel
                             and (r["transport"], r["soret"]) == mode and rule(r)],
                            key=lambda r: r[field])
                x, y = np.array([r[field] for r in rr]), np.array([r["ratio"] for r in rr])
                ci = np.array([r["ci"] for r in rr])
                ax.plot(x, y, color=MODE_COLORS[k], lw=1.3)
                ax.vlines(x, ci[:, 0], ci[:, 1], color=MODE_COLORS[k], lw=.9)
                for xx, yy, interval in zip(x, y, ci):
                    includes_one = interval[0] <= 1 <= interval[1]
                    ax.plot(xx, yy, marker="o", color=MODE_COLORS[k],
                            mfc="white" if includes_one else MODE_COLORS[k], ms=4)
            ax.axhline(1, color=".5", lw=.8, ls=":")
            ax.set_xlabel(label)
            ax.set_ylabel(r"$S_t$ [-]")
            ax.set_ylim(0, upper)
            pressure_axis(ax, field)
            if i == 0:
                ax.set_title(("Composición", "Presión", "Temperatura")[j])
        row_label(axes[i, 0], fuel)
    handles = [Line2D([], [], color=c, marker="o", label=n)
               for c, n in zip(MODE_COLORS, MODE_NAMES)]
    handles.append(Line2D([], [], color=".3", marker="o", mfc="white", ls="none",
                          label="IC 95 % incluye 1"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    save(fig, out, "14_speedup_global",
         r"Razón de medianas $S_t=\mathrm{mediana}(t_{\mathrm{Cantera}})/"
         r"\mathrm{mediana}(t_{\mathrm{KFlame}})$. Un valor mayor que 1 corresponde a menor "
         r"tiempo de KFlame; la línea punteada marca igualdad temporal. "
         r"Las barras muestran el intervalo bootstrap pareado del 95\,\%, con 20000 "
         r"remuestreos y semilla fija, calculado a partir de cinco pares por condición. "
         r"Los círculos blancos indican intervalos que contienen 1. Composición: cuatro "
         r"transportes a 1 atm y 300 K; presión: $\phi=1$ y 300 K; temperatura: $\phi=1$ "
         r"y 1 atm. Los límites verticales son comunes a los tres barridos de cada mecanismo. " + sweep_scope(rows))


def accuracy_speedup(rows, errors, out):
    err = {}
    for r in errors:
        err.setdefault(r["case_id"], []).append(number(r, "Su_percent"))
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.22, right=.97, top=.85, bottom=.10, hspace=.27)
    markers = ["o", "s", "^", "D"]
    for ax, fuel in zip(axes, ("CH4", "H2")):
        for mode, color, marker in zip(MODES, MODE_COLORS, markers):
            rr = [r for r in rows if r["fuel"] == fuel and r["id"] in err
                  and (r["transport"], r["soret"]) == mode]
            ax.scatter([r["ratio"] for r in rr], [np.median(err[r["id"]]) for r in rr],
                       color=color, marker=marker, s=28, alpha=.8)
        ax.axvline(1, color=".5", lw=.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"Discrepancia de $S_u$ [%]")
        row_label(ax, fuel)
    axes[-1].set_xlabel(r"Razón de tiempos $S_t$ [-]")
    fig.legend(handles=[Line2D([], [], color=c, marker=m, ls="none", label=n)
                        for c, m, n in zip(MODE_COLORS, markers, MODE_NAMES)],
               loc="upper center", ncol=2, frameon=False)
    extreme = max(rows, key=lambda r: r["ratio"])
    below = min(rows, key=lambda r: r["ratio"])
    upper_error = 4 * max(float(np.median(v)) for v in err.values())
    axes[0].set_ylim(top=upper_error)
    for r, position in ((extreme, (.04, .96)), (below, (.45, .96))):
        mode = MODE_NAMES[MODES.index((r['transport'], r['soret']))]
        text = (rf"$\phi={r['phi']:g}$, {r['pressure_atm']:g} atm, {r['temperature']:g} K"
                + "\n" + mode.replace("Multicomponente", "Multi.").replace("Promediado", "Prom."))
        ax = axes[0 if r["fuel"] == "CH4" else 1]
        ax.annotate(text, (r["ratio"], np.median(err[r["id"]])),
                    xytext=position, textcoords="axes fraction",
                    ha="left", va="top", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": ".5", "lw": .7})
    save(fig, out, "15_precision_speedup",
         r"Concordancia de velocidad frente a rendimiento: un punto por condición "
         f"({len(rows)} en total, {len(rows)//2} por mecanismo). Las filas separan mecanismos con límites "
         r"comunes; color y símbolo identifican las cuatro combinaciones de transporte. "
         r"La ordenada es la mediana de las cinco discrepancias relativas "
         r"absolutas de $S_u$ respecto a Cantera; la abscisa es la razón de medianas de "
         r"tiempo Cantera/KFlame. Ambos ejes son logarítmicos. La línea vertical marca "
         r"igualdad de tiempos. Se identifican el mayor y el menor cociente temporal. Esta discrepancia entre solvers "
         r"es un indicador de concordancia, no una estimación del error de discretización.")


def cost_figure(rows, out):
    # The nested bootstrap timer is already included in the exclusive categories.
    # Identical colors encode the reported operation, without equating timer scope.
    categories = [("jacobian_s", "Jacobiano", "#377eb8"),
                  ("residual_s", "Residual", "#e59a42"),
                  ("factorization_s", "Factorización", "#4b9677"),
                  ("backsolve_s", "Sustituciones", "#9365a5"),
                  ("unassigned_s", "Tiempo restante", "#bcbcbc")]
    selections = [(fuel, [
        (f"{fuel}_phi1_p1_T300_{transport}_S{int(soret)}", label)
        for (transport, soret), label in zip(MODES, MODE_NAMES)])
        for fuel in ("CH4", "H2")]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.1))
    fig.subplots_adjust(left=.28, right=.96, top=.84, bottom=.08, hspace=.52)
    exported = []
    for ax, (fuel, cases) in zip(axes, selections):
        rr = [r for r in rows if r["case_id"] in [cid for cid, _ in cases]]
        max_time = max(number(r, "time_instrumented_s") for r in rr)
        for y, (cid, _) in enumerate(cases):
            for backend, direction in (("native", -1), ("cantera", 1)):
                matches = [r for r in rr if r["case_id"] == cid and r["backend"] == backend]
                if len(matches) != 1:
                    raise ValueError(f"Expected one instrumented record: {cid}/{backend}")
                r = matches[0]
                if str(r["exact_production_profile"]).lower() != "true":
                    raise ValueError(f"Instrumented profile differs from production: {cid}/{backend}")
                total, left = number(r, "time_instrumented_s"), 0.
                for key, label, color in categories:
                    value = number(r, key) or 0.
                    ax.barh(y, direction * value, left=direction * left,
                            height=.52, color=color, edgecolor="white", lw=.35)
                    left += value
                if not np.isclose(left, total, rtol=1.e-6, atol=1.e-8):
                    raise ValueError(f"Cost does not close for {cid}/{backend}")
                ax.text(direction * (total + .035 * max_time), y,
                        f"{total:.2f} s", ha="right" if direction < 0 else "left",
                        va="center", fontsize=9)
                exported.append(r)
        ax.axvline(0, color=".3", lw=.8)
        ax.set_xlim(-1.28 * max_time, 1.28 * max_time)
        ax.set_yticks(range(len(cases)), [label for _, label in cases])
        ax.set_ylim(len(cases)-.4, -.6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):g}"))
        ax.set_xlabel("Tiempo absoluto [s]")
        ax.set_title(FUEL_NAMES[fuel], loc="left", pad=24)
        ax.text(.25, 1.04, "KFlame", transform=ax.transAxes, ha="center")
        ax.text(.75, 1.04, "Cantera", transform=ax.transAxes, ha="center")
    fig.legend(handles=[Patch(color=c, label=n) for _, n, c in categories],
               loc="upper center", ncol=3, frameon=False)
    pp.write_csv(out / "coste_casos_seleccionados.csv", exported)
    assert len(exported) == 16
    save(fig, out, "16_origen_coste",
         r"Desglose de tiempo absoluto de los ocho estados centrales: cuatro transportes "
         r"por mecanismo, a $\phi=1$, 1 atm y 300 K. Barras hacia la "
         r"izquierda: KFlame; hacia la derecha: Cantera. Cada mecanismo usa una escala "
         r"lineal simétrica, idéntica para ambos solvers; las escalas de CH$_4$ e H$_2$ "
         r"son distintas para mostrar sus respectivos costes. Las ejecuciones instrumentadas "
         r"reproducen exactamente los perfiles de producción usados en la comparación "
         r"de las cuatro combinaciones de transporte. "
         r"Los segmentos suman el tiempo total anotado. La inicialización multicomponente "
         r"está incluida en las operaciones y no se suma de nuevo. "
         r"Cantera comunica tiempos de residual y Jacobiano; álgebra lineal y otras etapas "
         r"sin temporizador separado forman parte del tiempo restante. Los contadores de "
         r"Cantera pueden omitir etapas interrumpidas durante la ampliación del dominio. "
         r"Se muestra una ejecución instrumentada por caso y solver, sin atribuir a estos "
         r"tiempos la incertidumbre de las cinco repeticiones de producción.")


def sensitivity_figure(data, out):
    rows = [r for r in data["rows"] if r["usable"]]
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 5.9))
    fig.subplots_adjust(left=.14, right=.98, top=.84, bottom=.10, hspace=.43, wspace=.36)
    exported = []
    for i, fuel in enumerate(("CH4", "H2")):
        for state, color in zip(STATES, STATE_COLORS):
            for backend in SOLVERS:
                rr = [r for r in rows if r["fuel"] == fuel
                      and (r["pressure_atm"], r["temperature"]) == state and r["backend"] == backend]
                base = next(r for r in rr if r["test"] == "L3")
                mesh = sorted([r for r in rr if r["test"] != "domain-L3"], key=lambda r: r["level"])
                dom = next(r for r in rr if r["test"] == "domain-L3")
                for r in rr:
                    exported.append(dict(r,
                        Tb_change_percent=100 * (r["Tb"] - base["Tb"]) / base["Tb"]))
                for j, key in enumerate(("signed_change_L3_percent", "Tb")):
                    ax = axes[i, j]
                    values = [r[key] if j == 0 else 100 * (r[key] - base[key]) / base[key]
                              for r in mesh]
                    ax.plot([r["level"] for r in mesh], values, color=color,
                            ls="-" if backend == "native" else "--",
                            marker="o" if backend == "native" else "s", mfc="none", ms=4)
                    value = dom[key] if j == 0 else 100 * (dom[key] - base[key]) / base[key]
                    # Separate paired domain points slightly to keep both visible.
                    ax.plot(6 + (-.06 if backend == "native" else .06), value,
                            marker="o" if backend == "native" else "s",
                            color=color, mfc="none", ms=4)
        for j, ax in enumerate(axes[i]):
            ax.axvspan(5.65, 6.35, color=".95", zorder=-5)
            ax.axhline(0, color=".6", lw=.6)
            ax.set_xticks([2, 3, 4, 5, 6], ["L2", "L3", "L4", "L5", "L3\n(2L)"])
            ax.set_xlim(1.85, 6.4)
            ax.set_ylabel(r"$\Delta S_u^{L3}$ [%]" if j == 0 else r"$\Delta T_b^{L3}$ [%]")
            if i == 0:
                ax.set_title("Velocidad de llama" if j == 0 else "Temperatura quemada")
        row_label(axes[i, 0], fuel)
    handles = [Line2D([], [], color=c, label=f"{p} atm, {t} K")
               for (p, t), c in zip(STATES, STATE_COLORS)] + solver_handles()
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    pp.write_csv(out / "sensibilidad_indicadores.csv", exported)
    save(fig, out, "11_sensibilidad_L3",
         r"Sensibilidad espacial de los seis estados de referencia, agrupada por mecanismo. "
         r"Cada punto procede de una resolución individual por estado, nivel o dominio y solver. "
         r"Cada color identifica presión y temperatura; línea continua y círculos: KFlame; "
         r"discontinua y cuadrados: Cantera. Se mantiene $\phi=1$, transporte promediado "
         r"sin Soret en CH$_4$ y multicomponente con Soret en H$_2$. "
         r"$\Delta S_u^{L3}=100(S_u-S_u^{L3})/S_u^{L3}$ y "
         r"$\Delta T_b^{L3}=100(T_b-T_b^{L3})/T_b^{L3}$, donde $T_b$ es la temperatura "
         r"del extremo quemado. La referencia L3 se toma por separado para cada solver "
         r"y estado. La zona gris L3 (2L) corresponde al dominio inicial duplicado "
         r"respecto al mayor dominio final del par L3 y constituye un control independiente "
         r"del refinamiento. El objetivo descriptivo de 0.5\,\% se aplica a la velocidad; "
         r"no es una banda de incertidumbre. Nodos, anchura final y errores integrados "
         r"de perfil se conservan en sensibilidad\_indicadores.csv.")


def transport_figure(rows, out):
    fig, axes = plt.subplots(2, 4, figsize=(9.4, 5.3))
    fig.subplots_adjust(left=.12, right=.99, top=.79, bottom=.11, hspace=.44, wspace=.48)
    contrasts = [(0, 2, "Multi. frente a prom.\nsin Soret"),
                 (1, 3, "Multi. frente a prom.\ncon Soret"),
                 (0, 1, "Soret en\npromediado"), (2, 3, "Soret en\nmulticomponente")]
    for i, fuel in enumerate(("CH4", "H2")):
        rr = [r for r in rows if r["fuel"] == fuel and r["pressure_atm"] == 1 and r["temperature"] == 300]
        for j, (a, b, title) in enumerate(contrasts):
            ax = axes[i, j]
            for backend in SOLVERS:
                points = []
                for phi in sorted({r["phi"] for r in rr}):
                    group = {(r["transport"], r["soret"]): r for r in rr if r["phi"] == phi}
                    v0, v1 = group[MODES[a]]["Su_" + backend], group[MODES[b]]["Su_" + backend]
                    points.append((phi, 100 * (v1 - v0) / v0))
                ax.plot(*zip(*points), color=COLORS[backend],
                        ls="-" if backend == "native" else "--",
                        marker="o" if backend == "native" else "s", mfc="none")
            ax.axhline(0, color=".65", lw=.7)
            ax.set_xlabel(r"$\phi$ [-]")
            if j == 0:
                ax.set_ylabel(r"Cambio de $S_u$ [%]")
            ax.set_xticks([.7, 1., 1.4])
            if i == 0:
                ax.set_title(title, fontsize=10)
        row_label(axes[i, 0], fuel)
    fig.legend(handles=[Line2D([], [], color=COLORS[b], ls="-" if b == "native" else "--",
                              label=NAMES[b]) for b in SOLVERS],
               loc="upper center", ncol=2, frameon=False)
    save(fig, out, "06_transporte",
         r"Efecto de transporte a 1 atm y 300 K, evaluado dentro de cada solver mediante "
         r"$100(S_{u,\mathrm{nuevo}}-S_{u,\mathrm{base}})/S_{u,\mathrm{base}}$. "
         r"Primeras dos columnas: multicomponente respecto a promediado, primero sin "
         r"Soret y después con Soret en ambos transportes. Últimas dos columnas: "
         r"activación de Soret respecto al mismo transporte sin Soret. "
         r"Se emplean las cuatro combinaciones en ambos mecanismos, cinco composiciones "
         r"por combinación y las medianas de las cinco ejecuciones por solver. "
         r"Un cambio negativo indica una reducción de la velocidad de propagación.")


def ablation_figure(records, out):
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.2))
    fig.subplots_adjust(left=.15, right=.97, top=.89, bottom=.10, hspace=.68)
    for ax, fuel, variant, label in zip(axes, ("CH4", "H2"), ("blocks-off", "fits-off"),
                                       ("Sin sustitución\ncompilada", "Sin reutilización\nmolecular")):
        rr = [r for r in records if r["phase"] == "ablation" and r.get("usable")
              and r["condition"]["fuel"] == fuel]
        values = []
        for rep in sorted({r["repetition"] for r in rr}):
            pair = {r["variant"]: r for r in rr if r["repetition"] == rep}
            values.append([pair[variant]["time_s"], pair["native"]["time_s"]])
            ax.plot([0, 1], values[-1], color=".60", lw=1, marker="o", ms=4, alpha=.7)
        values = np.array(values)
        ax.plot([0, 1], np.median(values, axis=0), "-D", color="#1769aa", ms=5, lw=2)
        ax.set_xticks([0, 1], [label, "KFlame"])
        ax.set_xlim(-.3, 1.3)
        ax.set_ylabel("Tiempo [s]")
        ax.set_title(FUEL_NAMES[fuel], loc="left")
    fig.legend(handles=[Line2D([], [], color=".6", marker="o", label="Par de ejecuciones"),
                        Line2D([], [], color="#1769aa", marker="D", label="Mediana")],
               loc="upper center", ncol=2, frameon=False)
    save(fig, out, "08_ablaciones",
         r"Ablaciones de KFlame en estados centrales ($\phi=1$, 1 atm y 300 K). "
         r"CH$_4$/GRI-Mech 3.0: transporte promediado sin Soret, sustitución compilada "
         r"desactivada frente a configuración completa. H$_2$/h2o2: multicomponente con "
         r"Soret, reutilización molecular desactivada frente a configuración completa. "
         r"Cada segmento gris une dos ejecuciones del mismo par; cinco pares por ablación. "
         r"Los rombos azules indican medianas. La tabla de ablaciones conserva los "
         r"intervalos pareados y la comprobación de concordancia de perfiles.")


def timing_handles():
    return [Line2D([], [], color=COLORS[b], ls="-" if b == "native" else "--",
                   marker="o" if b == "native" else "s", mfc="white",
                   label=NAMES[b]) for b in SOLVERS] + [
        Line2D([], [], color=".5", marker=".", ls="none", label="Cinco ejecuciones"),
        Line2D([], [], color=".25", marker="|", ms=9, ls="none",
                   label="Barras: cuartiles")]


def plot_timing(ax, rows, field):
    """Show the saved five observations, median and IQR on the physical x axis."""
    for backend in SOLVERS:
        xx = np.array([r[field] for r in rows])
        stats = [r["timing"][backend] for r in rows]
        yy = np.array([r["median_s"] for r in stats])
        quartiles = np.array([r["quartiles_s"] for r in stats])
        if any(len(r["times"]) != 5 for r in stats):
            raise ValueError("Expected all five production times")
        color = COLORS[backend]
        ax.plot(xx, yy, color=color, ls="-" if backend == "native" else "--",
                marker="o" if backend == "native" else "s", mfc="white", ms=4,
                zorder=4)
        ax.vlines(xx, quartiles[:, 0], quartiles[:, 1], color=color, lw=2.3, zorder=3)
        for x, s in zip(xx, stats):
            # Symmetric display-only offsets expose repeated observations.
            shift = np.linspace(-.010, .010, 5)
            offsets = x * np.exp(shift * 1.3) if field == "pressure_atm" else (
                x + shift * (220 if field == "temperature" else 1))
            ax.scatter(offsets, s["times"], color=color, alpha=.45, s=9, zorder=2)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5),
                                                           numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    pressure_axis(ax, field)


def timing_figures(rows, out):
    fig, axes = plt.subplots(2, 3, figsize=(8.3, 5.4))
    fig.subplots_adjust(left=.13, right=.99, top=.83, bottom=.11,
                        wspace=.43, hspace=.43)
    for i, fuel in enumerate(("CH4", "H2")):
        rr = [r for r in rows if r["fuel"] == fuel and representative(r)]
        all_times = [t for r in rr for b in SOLVERS for t in r["timing"][b]["times"]]
        for j, (field, label, rule) in enumerate(SWEEPS):
            ax = axes[i, j]
            subset = sorted([r for r in rr if rule(r)], key=lambda r: r[field])
            plot_timing(ax, subset, field)
            ax.set_ylim(min(all_times) / 1.25, max(all_times) * 1.25)
            ax.set_xlabel(label)
            ax.set_ylabel("Tiempo [s]")
            if i == 0:
                ax.set_title(("Composición", "Presión", "Temperatura")[j])
        row_label(axes[i, 0], fuel)
    fig.legend(handles=timing_handles(), ncol=2, loc="upper center", frameon=False)
    save(fig, out, "07_tiempos",
         r"Tiempos absolutos de producción frente a composición, presión y temperatura, "
         r"como complemento de la razón temporal $S_t$. CH$_4$/GRI-Mech 3.0: promediado "
         r"sin Soret; H$_2$/h2o2: multicomponente con Soret. Composición a 1 atm y 300 K; "
         r"presión a $\phi=1$ y 300 K; temperatura a $\phi=1$ y 1 atm. "
         r"Los puntos pequeños muestran las cinco ejecuciones, con desplazamientos "
         r"horizontales únicamente visuales; las líneas unen sus medianas y las barras "
         r"abarcan los cuartiles 25 y 75. El eje temporal es logarítmico y sus límites "
         r"son comunes a los tres barridos de cada mecanismo. Se conservan todas las "
         r"observaciones, incluidas las más lentas; se excluyen las ejecuciones instrumentadas. "
         r"Los cuartiles describen dispersión de tiempos, no un intervalo de confianza.")

    timing_detail(rows, out, 0, "09_tiempos_transportes")
    if full_transport(rows):
        timing_detail(rows, out, 1, "09_tiempos_presion")
        timing_detail(rows, out, 2, "09_tiempos_temperatura")


def timing_detail(rows, out, sweep_index, name):
    field, label, rule = SWEEPS[sweep_index]
    fig, axes = plt.subplots(2, 4, figsize=(9.2, 5.4))
    fig.subplots_adjust(left=.12, right=.99, top=.81, bottom=.11,
                        wspace=.28, hspace=.42)
    for i, fuel in enumerate(("CH4", "H2")):
        rr = [r for r in rows if r["fuel"] == fuel and rule(r)]
        times = [t for r in rr for b in SOLVERS for t in r["timing"][b]["times"]]
        for j, mode in enumerate(MODES):
            ax = axes[i, j]
            subset = sorted([r for r in rr if (r["transport"], r["soret"]) == mode],
                            key=lambda r: r[field])
            if len(subset) != 5:
                raise ValueError(f"Missing timing sweep: {fuel}/{mode}/{field}")
            plot_timing(ax, subset, field)
            ax.set_ylim(min(times) / 1.25, max(times) * 1.25)
            ax.set_xlabel(label)
            if field == "phi":
                ax.set_xticks([.7, 1., 1.4])
            elif field == "temperature":
                ax.set_xticks([300, 400, 500])
            if j == 0:
                ax.set_ylabel("Tiempo [s]")
            else:
                ax.tick_params(axis="y", labelleft=False)
            if i == 0:
                ax.set_title(MODE_NAMES[j].replace(" + ", "\n+ "), fontsize=9.5)
        row_label(axes[i, 0], fuel)
    fig.legend(handles=timing_handles(), ncol=2, loc="upper center", frameon=False)
    description = [r"la composición a 1 atm y 300 K", r"la presión a $\phi=1$ y 300 K",
                   r"la temperatura de entrada a $\phi=1$ y 1 atm"][sweep_index]
    save(fig, out, name,
         "Tendencia del coste con " + description + r" para los cuatro modelos de transporte. "
         r"Filas: mecanismos; columnas: promediado y multicomponente, "
         r"cada uno sin y con Soret. Se usa una escala temporal logarítmica común a "
         r"las cuatro columnas de cada mecanismo, lo que permite comparar tanto "
         r"el nivel de tiempo como su variación en el barrido. Puntos pequeños: las cinco "
         r"ejecuciones de producción, desplazadas ligeramente en horizontal; líneas: "
         r"medianas; barras: cuartiles 25 y 75, que describen la dispersión de las cinco mediciones.")


def mesh_figure(campaign, records, out):
    fig, axes = plt.subplots(2, 3, figsize=(8.8, 5.8))
    fig.subplots_adjust(left=.13, right=.98, top=.87, bottom=.10,
                        wspace=.49, hspace=.57)
    summary = []
    for i, fuel in enumerate(("CH4", "H2")):
        refs = []
        for backend in SOLVERS:
            r = pp.reference_record(records, fuel, backend)
            f = pp.profile(campaign, r)
            z = pp.aligned(f) * 1.e3
            progress = (f["T"]-f["T"][0]) / (f["T"][-1]-f["T"][0])
            active = (progress >= .05) & (progress <= .95)
            refs.append((r, f, z, active))
        lo = min(z[a].min() for _, _, z, a in refs)
        hi = max(z[a].max() for _, _, z, a in refs)
        padding = .3 * (hi-lo)
        lo, hi = lo-padding, hi+padding
        spacing_in_front = []
        for r, f, z, active in refs:
            backend = r["backend"]
            color = COLORS[backend]
            ls = "-" if backend == "native" else "--"
            dz = np.diff(f["z"]) * 1.e6
            mid = .5 * (z[:-1]+z[1:])
            axes[i, 0].plot(z, f["T"], color=color, ls=ls)
            axes[i, 0].plot(z, np.full_like(z, .05 if backend == "native" else .13),
                            "|", color=color, ms=3, alpha=.5,
                            transform=axes[i, 0].get_xaxis_transform())
            mask = (z >= lo) & (z <= hi)
            axes[i, 1].plot(z[mask], f["T"][mask], color=color, ls=ls, lw=.8,
                            marker="o" if backend == "native" else "x", ms=2,
                            mfc="none", mew=.45)
            axes[i, 2].semilogy(mid, dz, color=color, ls=ls)
            spacing_in_front.extend(dz[(mid >= lo) & (mid <= hi)])
            summary.append(dict(case_id=r["case_id"], backend=backend,
                                repetition=r["repetition"], folder=r["folder"],
                                profile_sha256=r["profile_sha256"], nodes=len(z),
                                width_mm=r["width"]*1000, dz_min_um=float(dz.min()),
                                dz_max_um=float(dz.max()), nodes_5_95=int(active.sum())))
        note = "\n".join(f'{NAMES[r["backend"]]}: {len(z)} nodos'
                         for r, _, z, _ in refs)
        widths = [r["width"] * 1000 for r, _, _, _ in refs]
        note += f"\nDominio: {widths[0]:g} mm" if np.isclose(*widths) else (
            f"\nDominios: {widths[0]:g} / {widths[1]:g} mm")
        axes[i, 0].text(.98, .46, note, transform=axes[i, 0].transAxes,
                        ha="right", fontsize=6.8,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1))
        for j, ax in enumerate(axes[i]):
            ax.set_xlabel(r"$x-x_{1/2}$ [mm]")
            ax.set_ylabel(r"$T$ [K]" if j < 2 else r"$\Delta x$ [$\mu$m]")
            if j:
                ax.set_xlim(lo, hi)
            if i == 0:
                ax.set_title(("Dominio completo", "Nodos del frente", "Espaciado local")[j],
                             fontsize=10)
        axes[i, 2].set_ylim(min(spacing_in_front)/1.3, max(spacing_in_front)*1.3)
        row_label(axes[i, 0], fuel)
    fig.legend(handles=[Line2D([], [], color=COLORS[b], ls="-" if b == "native" else "--",
                              label=NAMES[b]) for b in SOLVERS],
               ncol=2, loc="upper center", frameon=False)
    pp.write_csv(out / "mallas_seleccionadas.csv", summary)
    save(fig, out, "10_mallas_finales",
         r"Mallas finales de dos estados presentes también en los perfiles: "
         r"CH$_4$/GRI-Mech 3.0 promediado sin Soret e H$_2$/h2o2 multicomponente con "
         r"Soret, ambos a $\phi=1$, 1 atm y 300 K. Cada malla corresponde a una ejecución "
         r"individual: la primera repetición de producción de cada solver. "
         r"Izquierda: temperatura en todo el dominio; las dos bandas inferiores marcan "
         r"los nodos de KFlame y Cantera. Centro: ampliación del frente con todos sus "
         r"nodos originales. Derecha: distancia entre nodos consecutivos, en escala "
         r"logarítmica y sobre la misma ventana espacial del panel central. "
         r"El origen se traslada al punto medio térmico $x_{1/2}$ de cada perfil, "
         r"conservando la forma del perfil y el espaciado. La anotación informa "
         r"el número de nodos y la longitud del dominio final.")


def coverage_audit(campaign, records, rows, out):
    """Check coverage against the actual experimental design and archive it."""
    all_ids = {r["id"] for r in rows}
    composition = {r["id"] for r in rows if r["temperature"] == 300 and r["pressure_atm"] == 1}
    central = {r["id"] for r in rows if r["phi"] == 1 and r["id"] in composition}
    representative_ids = {r["id"] for r in rows if representative(r)}
    ref_ids = central & representative_ids
    spatial_ids = {r["id"] for r in rows if representative(r) and r["phi"] == 1
                   and (r["pressure_atm"], r["temperature"]) in STATES}
    assert len(all_ids) in (56, 104)
    assert [len(s) for s in (composition, central, representative_ids, ref_ids, spatial_ids)] == [40, 8, 26, 2, 6]
    for cid in all_ids:
        for backend in SOLVERS:
            rr = [r for r in records if r["phase"] == "main" and r.get("usable")
                  and r["case_id"] == cid and r["variant"] == backend]
            if len(rr) != 5 or len({r["repetition"] for r in rr}) != 5:
                raise ValueError(f"Missing production repetitions: {cid}/{backend}")
    profiles = read_csv(out / "perfiles_seleccionados.csv")
    cost = read_csv(out / "coste_casos_seleccionados.csv")
    assert {(r["case_id"], r["solver"]) for r in profiles} == {
        (r["case_id"], r["backend"]) for r in cost}
    assert {r["case_id"] for r in profiles} == central
    mesh = []
    byfolder = {r["folder"]: r for r in records}
    for p in profiles:
        r = byfolder[p["folder"]]
        f = pp.profile(campaign, r)
        dz = np.diff(f["z"]) * 1.e6
        theta = (f["T"] - f["T"][0]) / (f["T"][-1] - f["T"][0])
        mesh.append(dict(case_id=r["case_id"], fuel=r["condition"]["fuel"],
                         transport=r["condition"]["transport"], soret=r["condition"]["soret"],
                         backend=p["solver"], repetition=r["repetition"], folder=r["folder"],
                         profile_sha256=r["profile_sha256"], nodes=len(f["z"]),
                         width_mm=float(np.ptp(f["z"])) * 1000,
                         dz_min_um=float(dz.min()),
                         front_nodes_percent=100 * float(np.mean((theta >= .05) & (theta <= .95)))))
    pp.write_csv(out / "mallas_ocho_casos.csv", mesh)
    table = [r"\begin{table}[!htbp]", r"\centering\small",
             r"\caption{Mallas de los ocho estados centrales ($\phi=1$, 1 atm, 300 K). "
             r"Cada celda numérica presenta KFlame / Cantera. P: promediado; M: multicomponente; "
             r"+S: Soret. $N$: nodos; $\Delta x_{\min}$: separación mínima; $L$: longitud final; "
             r"$N_f/N$: porcentaje de nodos entre el 5 y el 95\,\% del salto térmico. "
             r"Se usan exactamente los perfiles de la primera repetición representados en las "
             r"figuras de las cuatro combinaciones, reproducidos por los diagnósticos de coste.}",
             r"\label{tab:rev-mallas-ocho-casos}",
             r"\begin{tabular}{@{}llrrrr@{}}\toprule",
             r"Mecanismo & Transp. & $N$ & $\Delta x_{\min}$ [$\mu$m] & $L$ [mm] & $N_f/N$ [\%] \\",
             r"\midrule"]
    for fuel in ("CH4", "H2"):
        for i, mode in enumerate(MODES):
            rr = {r["backend"]: r for r in mesh if r["fuel"] == fuel
                  and (r["transport"], r["soret"]) == mode}
            cells = ["GRI-Mech 3.0" if fuel == "CH4" else "h2o2", ["P", "P+S", "M", "M+S"][i]]
            for key, fmt in (("nodes", ".0f"), ("dz_min_um", ".2f"), ("width_mm", ".0f"),
                             ("front_nodes_percent", ".1f")):
                cells.append(" / ".join(format(rr[b][key], fmt) for b in SOLVERS))
            table.append(" & ".join(cells) + r" \\")
        if fuel == "CH4":
            table.append(r"\midrule")
    table.extend([r"\bottomrule\end{tabular}", r"\end{table}"])
    (out / "tabla_mallas_ocho_casos.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    coverage = [
        ("02_perfiles_CH4", {c for c in central if c.startswith("CH4_")}, "Cuatro transportes, primera repetición, ambos solvers"),
        ("02_perfiles_H2", {c for c in central if c.startswith("H2_")}, "Cuatro transportes, primera repetición, ambos solvers"),
        ("06_transporte", composition, "Cuatro contrastes; cuatro transportes, cinco composiciones"),
        ("07_tiempos", representative_ids, "Tres barridos con transporte de referencia; 09 completa los transportes"),
        ("08_ablaciones", ref_ids, "Dos optimizaciones específicas de KFlame; cinco pares por ablación"),
        ("09_tiempos_transportes", composition, "Cuatro transportes; cinco tiempos por solver y condición"),
        ("10_mallas_finales", ref_ids, "Detalle de dos referencias; cuadro adjunto amplía a ocho estados centrales"),
        ("11_sensibilidad_L3", spatial_ids, "Tres estados por mecanismo con su transporte de referencia; L2–L5 y dominio"),
        ("12_concordancia_global", all_ids, f"{len(rows)//2} condiciones por mecanismo; cada punto resume cinco pares"),
        ("13_respuesta_fisica", all_ids, "Cuatro transportes en los tres barridos" if full_transport(rows) else "Composición: cuatro transportes; presión y temperatura: transporte de referencia"),
        ("14_speedup_global", all_ids, "Misma cobertura física que 13, intervalos de cinco pares"),
        ("15_precision_speedup", all_ids, "Mecanismos separados; cuatro transportes identificados"),
        ("16_origen_coste", central, "Ocho estados, 16 ejecuciones instrumentadas que reproducen los perfiles"),
        ("tabla_mallas_ocho_casos", central, "Mismos 16 perfiles de 02 y diagnósticos de 16")]
    if full_transport(rows):
        for j, name in ((1, "09_tiempos_presion"), (2, "09_tiempos_temperatura")):
            coverage.append((name, {r['id'] for r in rows if SWEEPS[j][2](r)},
                             "Cuatro transportes, ambos mecanismos, cinco pares por condición"))
    (out / "cobertura_figuras.json").write_text(json.dumps([
        dict(figure=n, conditions=sorted(ids), count=len(ids), scope=scope)
        for n, ids, scope in coverage], ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "cobertura_figuras.md").write_text(
        "# Cobertura comprobada de las figuras\n\n"
        f"{len(rows)} condiciones principales, cinco repeticiones aceptadas por solver y condición. "
        "Los ocho estados centrales coinciden entre perfiles, mallas y coste.\n\n"
        "| Figura | Condiciones únicas | Alcance |\n|---|---:|---|\n" +
        "\n".join(f"| {n} | {len(ids)} | {scope} |" for n, ids, scope in coverage) +
        ("\n\nLa ampliación completa los cuatro transportes en composición, presión y temperatura. "
         "La sensibilidad y las ablaciones mantienen sus casos específicos del diseño original.\n"
         if full_transport(rows) else
        "\n\nEn presión, temperatura y sensibilidad espacial se simuló promediado sin Soret "
        "para CH4 y multicomponente con Soret para H2. Ampliar esos barridos a cuatro "
        "transportes sería un estudio adicional. Las ablaciones estudian una optimización "
        "por mecanismo. No faltan simulaciones para completar la cobertura del diseño actual.\n"),
        encoding="utf-8")


def inventory(out, report, source_files):
    (out / "pies_figuras.md").write_text("\n\n".join(
        f"## {name}\n\n{caption}" for name, caption in CAPTIONS.items()) + "\n", encoding="utf-8")
    old = [
        ("02", "Dos páginas, CH4 y H2; cuatro transportes y ejes secundarios"),
        ("06", "Contraste multicomponente/promediado sin y con Soret; efecto Soret en ambos"),
        ("07", "Tiempos absolutos de los tres barridos; complemento del speed-up"),
        ("08", "Cinco pares y mediana; CH4 arriba, H2 abajo"),
        ("09", "Detalle de tendencias temporales por transporte; cuatro columnas"),
        ("10", "Detalle de dos referencias y cuadro complementario de los ocho estados centrales"),
        ("11", "Dos indicadores y tres estados por mecanismo; cuatro paneles"),
        ("12", "Violines en log; todas las condiciones por mecanismo"),
        ("13", "Tres barridos; composición con los cuatro transportes, leyenda global"),
        ("14", "Curvas e intervalos en todas las columnas; sin mapa de color"),
        ("15", "Relación concordancia/tiempo; mecanismos separados y cuatro transportes identificados"),
        ("16", "Ocho estados centrales, cuatro transportes por mecanismo; barras enfrentadas en segundos")]
    (out / "inventario_figuras_revision.md").write_text(
        "# Revisión gráfica del capítulo 4\n\n"
        "Figuras regeneradas desde perfiles y registros guardados, con sus pies LaTeX "
        "y auditoría de cobertura. El postprocesado no ejecuta llamas.\n\n"
        "| Figura | Revisión |\n|---|---|\n" +
        "\n".join(f"| {a} | {b} |" for a, b in old) +
        "\n\nLa comparación de tiempos conserva la misma escala a izquierda y derecha "
        "dentro de cada mecanismo. CH4 e H2 tienen escalas distintas, indicadas en segundos.\n"
        "\n02_perfiles_concordancia.pdf reúne las dos páginas de perfiles. "
        "revision_graficos.pdf reúne la selección integrada en el capítulo; "
        "07 y 08 se conservan como gráficos auxiliares. Los PNG se exportan a 300 dpi.\n"
        "\nLos pies de cada figura definen variables, muestreo y límites de interpretación. "
        "perfiles_seleccionados.csv documenta cada perfil; concordancia_por_condicion.csv "
        "documenta la agregación de cinco pares; coste_casos_seleccionados.csv conserva "
        "los temporizadores originales. sensibilidad_indicadores.csv incluye nodos y dominio.\n",
        encoding="utf-8")
    audit = [
        ("02_perfiles_CH4 / H2", "Conservadas", "Los mismos ocho estados centrales, cuatro transportes por mecanismo; perfiles originales y ejes comunes por mecanismo."),
        ("06_transporte", "Conservada", "Cuatro contrastes de transporte sobre las 40 condiciones de composición; los datos físicos originales no cambiaron."),
        ("07_tiempos", "Solo archivo auxiliar", "Se excluye del capítulo y del conjunto seleccionado: los tres detalles 09 ya presentan esos tiempos y los demás transportes."),
        ("08_ablaciones", "Sustituida por un cuadro en el capítulo", "Cinco pares por optimización; el cuadro permite conservar la evidencia causal y sus límites sin repetirla en una figura aislada."),
        ("09_tiempos_transportes", "Verificada", "Cinco composiciones por transporte, ambos mecanismos, cinco observaciones por solver."),
        ("09_tiempos_presion / temperatura", "Añadidas en la ampliación", "Cuatro transportes por mecanismo en cada barrido; cinco observaciones, medianas y cuartiles."),
        ("10_mallas_finales", "Conservada y complementada", "Dos referencias detalladas y cuadro de los ocho estados centrales; coincidencia de los perfiles con 02 y 16 comprobada."),
        ("11_sensibilidad_L3", "Verificada con cuadro corregido", "Seis estados, L2–L5 y dominio ampliado. El cuadro conserva signo de temperatura, nodos y cambios de dominio sin redondearlos a cero; su alcance sigue siendo el del estudio espacial ejecutado."),
        ("12_concordancia_global", "Actualizada", "Una mediana por condición a partir de cinco pares; incluye todas las condiciones del informe y actualiza el tamaño de muestra."),
        ("13_respuesta_fisica", "Actualizada", "Cuatro transportes en composición, presión y temperatura cuando la ampliación está completa."),
        ("14_speedup_global", "Actualizada", "Cuatro curvas en cada barrido ampliado; intervalos pareados y misma escala vertical entre barridos de cada mecanismo."),
        ("15_precision_speedup", "Actualizada", "Todas las condiciones, mecanismos separados, transportes identificados y extremos calculados desde los datos actuales."),
        ("16_origen_coste", "Conservada tras comprobar sus fuentes", "16 diagnósticos instrumentados de ocho estados centrales; barras absolutas que cierran el tiempo total. Los nuevos tiempos de producción no se presentan como un desglose instrumentado.")]
    (out/'revision_grafica_por_figura.md').write_text(
        '# Revisión figura por figura\n\n| Figura | Decisión | Comprobación y alcance |\n|---|---|---|\n' +
        '\n'.join(f'| {n} | {decision} | {scope} |' for n,decision,scope in audit)+'\n', encoding='utf-8')
    manifest = {"sources": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in source_files},
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "figures": OUTPUTS, "dpi": 300, "font": "DejaVu Serif",
                "cost_units": "seconds", "cost_stacks_close": True}
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    order = ["02_perfiles_CH4", "02_perfiles_H2", "11_sensibilidad_L3",
             "12_concordancia_global", "13_respuesta_fisica", "06_transporte",
             "09_tiempos_transportes", "14_speedup_global",
             "15_precision_speedup", "16_origen_coste", "10_mallas_finales"]
    for name in ("09_tiempos_presion", "09_tiempos_temperatura"):
        if name in OUTPUTS:
            order.insert(order.index("14_speedup_global"), name)
    writer = PdfWriter()
    for name in order:
        writer.append(str(out / (name + ".pdf")))
    writer.write(str(out / "revision_graficos.pdf"))
    assert len(PdfReader(out / "revision_graficos.pdf").pages) == len(order)
    preview = [
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}",
        r"\usepackage[spanish]{babel}\usepackage{graphicx,amsmath,booktabs}",
        r"\usepackage[margin=2cm]{geometry}\usepackage[font=small]{caption}",
        r"\providecommand{\SweepReportRoot}{..}",
        r"\begin{document}",
        *[r"\input{" + name + r".tex}\clearpage" for name in order],
        r"\input{tabla_mallas_ocho_casos.tex}\clearpage",
        r"\end{document}",
    ]
    (out / "revision_con_pies.tex").write_text("\n".join(preview) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("runs/thesis_flames_L3"))
    ap.add_argument("--report", type=Path, help="Offline CSV report; diagnostics remain in the original report")
    ap.add_argument("--sensitivity", type=Path,
                    default=Path("runs/thesis_flames_L3/report/sensibilidad_L3.json"))
    ap.add_argument("--output", type=Path,
                    default=Path("runs/thesis_flames_L3/report/revision_figures"))
    args = ap.parse_args(argv)
    OUTPUTS.clear()
    CAPTIONS.clear()
    campaign, out = args.input.resolve(), args.output.resolve()
    default_report = "report_expanded" if (campaign / "transport_extension.json").exists() else "report"
    report = args.report.resolve() if args.report else campaign / default_report
    out.mkdir(parents=True, exist_ok=True)
    style()
    config, errors, performance = [
        read_csv(report / name) for name in
        ("configuracion.csv", "errores.csv", "rendimiento.csv")]
    diagnostic_path = campaign / "report/desglose_diagnostico.csv"
    diagnostic = read_csv(diagnostic_path)
    rows = summary_rows(performance, config)
    assert len(config) in (56, 104) and len(rows) == len(config) and len(errors) == 5*len(config), "Finish all five pairs before generating the final figures; the offline report retains pending cases."
    records = pp.load_records(campaign)
    profile_figures(campaign, records, out)
    errors_figure(errors, config, out)
    physical_figure(rows, out)
    speedup_figure(rows, out)
    accuracy_speedup(rows, errors, out)
    cost_figure(diagnostic, out)
    sensitivity_figure(json.loads(args.sensitivity.read_text(encoding="utf-8")), out)
    transport_figure(rows, out)
    ablation_figure(records, out)
    timing_figures(rows, out)
    mesh_figure(campaign, records, out)
    coverage_audit(campaign, records, rows, out)
    sources = [report / n for n in
               ("configuracion.csv", "errores.csv", "rendimiento.csv")]
    sources.append(diagnostic_path)
    sources.append(args.sensitivity)
    inventory(out, report, sources)
    print(f"{len(OUTPUTS)} figures, vector PDF + PNG 300 dpi; data checks passed: {out}")


if __name__ == "__main__":
    main()
