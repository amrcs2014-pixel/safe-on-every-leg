"""Tables and figures of the manuscript, written from the result files.

Usage, from the code folder: python -m experiments.paper [objectives] [cost] [stops] [algorithms] [plain] [convergence] [figures]
Without arguments everything is written. Tables go to manuscript/generated/*.tex as table bodies
(rows only), figures to manuscript/figures/*.pdf.
"""
import sys

import numpy as np
import pandas as pd

from .common import ROOT, read
from .final_summary import CASE, complete_replicates, indicators, per_run
from .plain_summary import ALONSO_2014, BRKGA_S, GRASP_2015, TREE_SEARCH

OUT = ROOT / "manuscript" / "generated"
FIGURES = ROOT / "manuscript" / "figures"
TYPES = {1: 3, 2: 5, 3: 8, 4: 10, 5: 12, 6: 15, 7: 20, 8: 30, 9: 40, 10: 50, 11: 60, 12: 70, 13: 80, 14: 90, 15: 100}
OBJECTIVE_NAMES = {"centre": "Centre of the space, departure", "departure": "Safety margin, departure",
                   "every_leg": "Safety margin, every leg"}
ALGORITHM_NAMES = {"nsga2": "NSGA-II", "nsga3": "NSGA-III", "moead": "MOEA/D", "sms_emoa": "SMS-EMOA"}
EXAMPLE_CLASS, EXAMPLE_INSTANCE = 3, 20  # the case drawn in the two example figures


def write(name: str, rows: list[list]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(" & ".join(str(cell) for cell in row) + r" \\" for row in rows) + "\n"
    (OUT / f"{name}.tex").write_text(text, encoding="utf-8")
    print(f"--- {name}.tex")
    print(text)


def num(value, digits=1) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def case_means(name: str, runs=None) -> pd.DataFrame:
    d = complete_replicates(read(f"final_{name}.csv"))
    if runs is not None:
        d = d[d["run"].isin(runs)]
    runs = per_run(d)
    print(f"final_{name}.csv: {d.groupby(CASE).ngroups} cases, complete replicates {sorted(d['run'].unique())}")
    return runs.groupby(CASE + ["objective", "algorithm"]).mean(numeric_only=True)


def objectives():
    means = case_means("main").groupby("objective").mean()
    rows = []
    for objective, label in OBJECTIVE_NAMES.items():
        m = means.loc[objective]
        rows.append([label, num(m["legal_share"]), num(m["fails_en_route"]), num(m["best_departure_fails"]),
                     num(m["best_legal"], 2), num(m["best_legal_few"], 2), num(m["best_legal_free"], 2)])
    write("objectives", rows)


def cost():
    every = case_means("main").xs("every_leg", level="objective").droplevel("algorithm")
    columns = ["best_any", "best_legal", "best_margin", "best_any_few", "best_legal_few", "best_any_free", "best_legal_free"]
    by_class = every[columns].groupby("br_class").mean()
    rows = [[f"BR{c}", TYPES[c]] + [num(v, 2) for v in by_class.loc[c]] for c in by_class.index]
    rows.append([r"\midrule Mean", ""] + [num(v, 2) for v in by_class.mean()])
    write("cost", rows)


def stops():
    five_runs = read("final_stops.csv")["run"].unique()
    one = case_means("main", runs=five_runs).xs("every_leg", level="objective").droplevel("algorithm")
    five = case_means("stops").xs("every_leg", level="objective").droplevel("algorithm")
    columns = ["best_legal", "best_legal_few", "best_legal_free"]
    a = one[columns].groupby("br_class").mean()
    b = five[columns].groupby("br_class").mean()
    rows = [[f"BR{c}", TYPES[c]] + [num(v, 2) for pair in zip(a.loc[c], b.loc[c]) for v in pair] for c in b.index]
    shared = a.loc[b.index]
    rows.append([r"\midrule Mean", ""] + [num(v, 2) for pair in zip(shared.mean(), b.mean()) for v in pair])
    write("stops", rows)


def algorithms():
    d = complete_replicates(read("final_algorithms.csv"))
    scores = indicators(d)
    means = scores.groupby(CASE + ["algorithm"])[["hypervolume", "igd_plus", "plans", "on_reference"]].mean()
    rank_hv = means["hypervolume"].unstack("algorithm").rank(axis=1, ascending=False).mean()
    rank_igd = means["igd_plus"].unstack("algorithm").rank(axis=1, ascending=True).mean()
    overall = means.groupby("algorithm").mean()
    singles = per_run(d).groupby(CASE + ["algorithm"]).mean(numeric_only=True).groupby("algorithm").mean()
    rows = []
    for algorithm, label in ALGORITHM_NAMES.items():
        if algorithm in overall.index:
            m = overall.loc[algorithm]
            rows.append([label, num(m["hypervolume"], 3), num(rank_hv[algorithm], 2), num(m["igd_plus"], 3), num(rank_igd[algorithm], 2),
                         num(m["plans"], 0), num(singles.loc[algorithm, "best_legal"], 2), num(singles.loc[algorithm, "best_legal_few"], 2)])
    write("algorithms", rows)


def plain():
    small, large = read("plain.csv"), read("plain_large.csv")
    rows = []
    means = []
    for d in (small, large):
        budgets = sorted(d["per_type"].unique())
        fill = d.pivot_table(index="br_class", columns="per_type", values="fill") * 100
        instances = d.groupby("br_class")["instance"].nunique()
        published = (BRKGA_S, TREE_SEARCH, GRASP_2015, ALONSO_2014)
        for c in fill.index:
            rows.append([f"BR{c}", TYPES[c], num(fill.loc[c, budgets[-1]], 1)] + [num(p[c - 1], 1) for p in published])
        classes = list(fill.index)
        means.append([f"Mean BR{classes[0]}--{classes[-1]}", "", num(fill[budgets[-1]].mean(), 1)]
                     + [num(np.mean([p[c - 1] for c in classes]), 1) for p in published])
    means[0][0] = r"\midrule " + means[0][0]
    write("plain", rows + means)


def convergence():
    d = read("convergence_types.csv")
    d = d[d["source"] == "archive"]
    rows = []
    for column, digits in (("best_legal", 2), ("area_legal", 2)):
        t = d.pivot_table(index=["br_class", "types"], columns="per_type", values=column) * 100
        for (c, types), values in t.iterrows():
            rows.append([f"BR{c}", types, column.replace("_", " ")] + [num(values[b], digits) for b in (1000, 2000, 4000, 8000, 16000)]
                        + [num(values[16000] - values[8000], digits)])
    write("convergence", rows)


def figures():
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42})
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Convergence: area dominated by legal plans against the budget, one line per class.
    d = read("convergence_types.csv")
    t = d[d["source"] == "archive"].pivot_table(index="types", columns="per_type", values="area_legal") * 100
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for types, style in zip(t.index, ("o-", "s-", "^-", "v-", "D--", "x--")):
        ax.plot(t.columns, t.loc[types], style, color="black", markersize=3, linewidth=0.8, markerfacecolor="white", label=f"{types} box types")
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(t.columns))
    ax.set_xticklabels([f"{b:,}" for b in t.columns], rotation=45, ha="right")
    ax.set_xlabel("Evaluations per box type")
    ax.set_ylabel(r"Area of legal plans ($\times 100$)")
    ax.legend(ncol=1, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=2.4)
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("--- figures/convergence.pdf")

    # One case in detail: a plan that is legal at departure and not on a later leg, and the two fronts.
    from legsafe import decode, leg_moments, support_loads

    from .common import BETAS, instance
    from .final import VEHICLE, measures, search

    job, run_number = (EXAMPLE_CLASS, EXAMPLE_INSTANCE, BETAS[0]), 0
    plans = {}
    for objective in ("centre", "every_leg"):
        inst, rngs = instance(job, 2)
        keys, _ = search(inst, rngs[run_number], objective)
        plans[objective] = [(k, measures(inst, decode(inst, k))) for k in keys]
    inst, _ = instance(job, 2)

    failing = [(k, m) for k, m in plans["centre"] if m["sigma_departure"] >= 0 and m["sigma"] < 0]
    print(f"example case BR{job[0]} instance {job[1]}, weights {job[2]}: centre search, {len(plans['centre'])} plans, "
          f"{sum(m['sigma_departure'] >= 0 for _, m in plans['centre'])} legal at departure, {len(failing)} of them illegal later")
    if failing:
        k, m = max(failing, key=lambda pair: pair[1]["utilisation"])
        layout = decode(inst, k)
        front, rear = support_loads(VEHICLE, leg_moments(layout.mass, *layout.centre.T, layout.rank, inst.n_stops))
        edges = np.arange(inst.n_stops + 1)
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        ax.stairs(100 * front / VEHICLE.max_front, edges, baseline=None, color="black", linewidth=1.0, label="Front axle")
        ax.stairs(100 * rear / VEHICLE.max_rear, edges, baseline=None, color="black", linewidth=1.0, linestyle="--", label="Rear axle")
        ax.axhline(100, color="gray", linewidth=0.6)
        ax.set_xlabel("Leg (0 is the leg from the depot)")
        ax.set_ylabel("Axle load (% of its limit)")
        ax.set_xticks(edges[:-1] + 0.5)
        ax.set_xticklabels(edges[:-1])
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(FIGURES / "route.pdf")
        plt.close(fig)
        print(f"--- figures/route.pdf: plan with utilisation {m['utilisation']:.4f}, {m['obstacles']} obstacles, margin at departure "
              f"{m['sigma_departure']:.4f}, worst margin {m['sigma']:.4f} on leg {m['worst_leg']}; front axle load {front.round(3).tolist()} t, "
              f"rear {rear.round(3).tolist()} t; peak front {100 * front.max() / VEHICLE.max_front:.1f}% of the limit")

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    every = pd.DataFrame([m for _, m in plans["every_leg"]])
    for mask, marker, label in ((every["sigma"] < 0, "x", "Illegal on some leg"), ((every["sigma"] >= 0) & (every["sigma"] < 0.05), "o", r"Legal, $\sigma<0.05$"),
                                (every["sigma"] >= 0.05, "s", r"$\sigma\geq0.05$")):
        ax.scatter(every.loc[mask, "obstacles"], 100 * every.loc[mask, "utilisation"], s=9, marker=marker, linewidths=0.6,
                   facecolors="none" if marker != "x" else "gray", edgecolors="black" if marker != "x" else None, color="gray" if marker == "x" else None,
                   label=label)
    ax.set_xlabel("Unloading obstacles")
    ax.set_ylabel("Utilisation (%)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "front.pdf")
    plt.close(fig)
    print(f"--- figures/front.pdf: every-leg search, {len(every)} plans, {int((every['sigma'] >= 0).sum())} legal, "
          f"{int((every['sigma'] >= 0.05).sum())} with a margin of 5%")


def vehicles():
    """Vehicle sensitivity study: one row per vehicle."""
    from .final import VEHICLES

    d = complete_replicates(read("final_vehicles.csv").assign(algorithm=lambda f: f["algorithm"] + "|" + f["vehicle"]))
    d["algorithm"] = d["algorithm"].str.split("|").str[0]
    labels = {"reference": "Reference", "overhang_1.0": "Axle moved back", "overhang_0.0": "Axle at the door",
              "high": "Higher centres"}
    rows = []
    for name, label in labels.items():
        part = d[d["vehicle"] == name]
        means = per_run(part).groupby(CASE + ["objective"]).mean(numeric_only=True)
        centre, departure, every = (means.xs(o, level="objective") for o in ("centre", "departure", "every_leg"))
        v = VEHICLES[name]
        legal = part[(part["objective"] == "every_leg") & (part["sigma"] >= -1e-12)]
        rows.append([label, num(v.length - v.a_rear, 2), num(centre["fails_en_route"].mean()), num(departure["fails_en_route"].mean()),
                     num(every["best_legal"].mean(), 2), num((every["best_any_few"] - every["best_legal_few"]).mean(), 2),
                     num((every["best_any_free"] - every["best_legal_free"]).mean(), 2),
                     num(100 * (legal["rollover"] < legal["axle"]).mean()), num(100 * legal["rear_share"].min(), 0)])
    write("vehicles", rows)


def timing():
    """Run times of the four searches: wall-clock seconds per run by class, relative time, share spent in selection."""
    d = read("timing_e4.csv")
    wall = d.pivot_table(index="algorithm", columns="br_class", values="wall")
    relative = d.pivot_table(index=["br_class", "run"], columns="algorithm", values="wall")
    relative = relative.div(relative["nsga2"], axis=0).mean()
    selection = 1 - d.groupby("algorithm")["evaluation_wall"].sum() / d.groupby("algorithm")["wall"].sum()
    rows = []
    for algorithm, label in ALGORITHM_NAMES.items():
        rows.append([label] + [num(wall.loc[algorithm, c], 1) for c in wall.columns] + [num(relative[algorithm], 2), num(100 * selection[algorithm], 0)])
    write("timing", rows)


PARTS = {"objectives": objectives, "cost": cost, "stops": stops, "algorithms": algorithms, "plain": plain,
         "convergence": convergence, "vehicles": vehicles, "timing": timing, "figures": figures}

if __name__ == "__main__":
    for part in sys.argv[1:] or list(PARTS):
        PARTS[part]()
