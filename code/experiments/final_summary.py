"""Summaries of the final experiments. Usage, from the code folder:
  python -m experiments.final_summary objectives NAME   third objectives compared (E2, E3, E5), from results/final_NAME.csv
  python -m experiments.final_summary algorithms NAME   search algorithms compared (E4), from results/final_NAME.csv

A case is an instance with one weight setting and one number of stops. Runs of a case are averaged
first; tests then compare case means, so that the unit of every test is a case and not a run.
"""
import numpy as np
import pandas as pd

from legsafe.moea import hypervolume, igd_plus
from legsafe.nsga2 import nondominated_fronts

from .common import read
from .final import CAP

CASE = ["br_class", "instance", "alpha", "beta", "stops"]
RUN = CASE + ["objective", "algorithm", "run"]
FEW = 20  # obstacles


def complete_replicates(d: pd.DataFrame) -> pd.DataFrame:
    """The rows of the run numbers that every case has for every search in the file."""
    searches = d.groupby(["objective", "algorithm"]).ngroups
    per_number = d.groupby("run").apply(lambda part: part.groupby(CASE + ["objective", "algorithm"]).ngroups, include_groups=False)
    keep = per_number[per_number == d.groupby(CASE).ngroups * searches].index
    return d[d["run"].isin(keep)]


def per_run(d: pd.DataFrame) -> pd.DataFrame:
    """One row per run with the scores of its archive."""
    legal, margin = d["sigma"] >= -1e-12, d["sigma"] >= CAP - 1e-9
    depart = d["sigma_departure"] >= -1e-12
    few, free = d["obstacles"] <= FEW, d["obstacles"] == 0

    def best(mask):
        return d["utilisation"].where(mask).groupby([d[c] for c in RUN]).max() * 100

    def share(mask, of):
        return (mask & of).groupby([d[c] for c in RUN]).sum() / of.groupby([d[c] for c in RUN]).sum()

    groups = d.groupby(RUN)
    out = pd.DataFrame({
        "plans": groups.size(),
        "best_any": best(legal | ~legal), "best_any_few": best(few), "best_any_free": best(free),
        "best_legal": best(legal), "best_legal_few": best(legal & few), "best_legal_free": best(legal & free),
        "best_margin": best(margin), "best_margin_few": best(margin & few), "best_margin_free": best(margin & free),
        "best_departure": best(depart), "best_departure_few": best(depart & few), "best_departure_free": best(depart & free),
        "legal_share": share(legal, legal | ~legal) * 100,
        "fails_en_route": share(depart & ~legal, depart) * 100,  # of the plans that are legal at departure
        "rollover_binds": share(d["rollover"] < d["axle"], legal | ~legal) * 100,
        "rollover_lowest": groups["rollover"].min(),
        "seconds": groups["seconds"].first(),
    })
    top = d.loc[d["utilisation"].where(depart).groupby([d[c] for c in RUN]).idxmax().dropna()]
    out["best_departure_fails"] = (top.set_index(RUN)["sigma"] < -1e-12).astype(float) * 100
    heavy = d[d["utilisation"] >= d.groupby(RUN)["utilisation"].transform("max") - 0.05]  # plans within 5 points of the fullest
    out["cg_height_full_plans"] = heavy.groupby(RUN)["cg_height"].mean()
    return out


def paired(case_means: pd.DataFrame, column: str, a: str, b: str, level="objective") -> str:
    """Difference a minus b of a score over the cases that have it for both, with a signed-rank test."""
    from scipy.stats import wilcoxon

    wide = case_means[column].unstack(level)
    both = wide[[a, b]].dropna()
    diff = both[a] - both[b]
    if len(diff) == 0:
        return f"{column}: no case has it for both"
    p = wilcoxon(diff).pvalue if (diff.abs() > 1e-9).any() else float("nan")
    return (f"{column}: {a} minus {b} = {diff.mean():+.2f} (median {diff.median():+.2f}; better in {(diff > 1e-9).sum()}, "
            f"worse in {(diff < -1e-9).sum()} of {len(diff)} cases; p = {p:.4f})")


def objectives_main(name):
    d = complete_replicates(read(f"final_{name}.csv"))
    runs = per_run(d)
    counts = runs.groupby(["objective", "algorithm"]).size()
    print(f"{d.groupby(CASE).ngroups} cases, {d['run'].nunique()} runs each; runs per search: {counts.to_dict()}\n")
    case_means = runs.groupby(CASE + ["objective"]).mean(numeric_only=True)
    has = runs.notna().groupby(CASE + ["objective"]).mean()
    scores = [c for c in runs.columns if c != "seconds"]
    print("Mean over cases of the mean over runs, by third objective of the search (utilisation in %, shares in %):")
    print(case_means[scores].groupby("objective").mean().round(2).T.to_string())
    print("\nShare of runs that have such a plan (%):")
    print((has[[c for c in scores if c.startswith("best_")]].groupby("objective").mean() * 100).round(1).T.to_string())
    print("\nBy class, search on every leg:")
    every = case_means.xs("every_leg", level="objective")
    print(every[["best_any", "best_legal", "best_margin", "best_any_few", "best_legal_few", "best_any_free", "best_legal_free",
                 "legal_share", "rollover_lowest", "rollover_binds"]].groupby("br_class").mean().round(2).to_string())
    every = every.assign(cost=every["best_any"] - every["best_legal"], cost_margin=every["best_any"] - every["best_margin"],
                         cost_few=every["best_any_few"] - every["best_legal_few"], cost_free=every["best_any_free"] - every["best_legal_free"],
                         price_few=every["best_legal"] - every["best_legal_few"], price_free=every["best_legal"] - every["best_legal_free"])
    print("\nCost of safety on every leg, in points of utilisation, within the archives of the every-leg search:")
    for column, label in (("cost", "legal on every leg, any unloading effort"), ("cost_margin", "5% margin on every leg, any unloading effort"),
                          ("cost_few", "legal, at most 20 obstacles (against any plan with at most 20)"),
                          ("cost_free", "legal, no obstacles (against any plan without obstacles)"),
                          ("price_few", "at most 20 obstacles, among legal plans"), ("price_free", "no obstacles, among legal plans")):
        values = every[column].dropna()
        print(f"  {label}: mean {values.mean():.2f}, median {values.median():.2f}, largest {values.max():.2f}, "
              f"zero in {(values.abs() < 1e-9).mean():.0%} of {len(values)} cases")
    print("  by class:")
    print(every[["cost", "cost_margin", "cost_few", "cost_free", "price_few", "price_free"]].groupby("br_class").mean().round(2).to_string())

    others = [o for o in ("departure", "centre") if o in runs.index.get_level_values("objective")]
    if others:
        print("\nPlans that are legal at departure and illegal on a later leg, by third objective of the search:")
        print(case_means[["fails_en_route", "best_departure_fails", "legal_share"]].groupby("objective").mean().round(1).to_string())
        print(case_means["fails_en_route"].unstack("objective").groupby("br_class").mean().round(1).to_string())
        print("\nSearch on every leg against the other searches, case by case:")
        for other in others:
            for column in ("best_legal", "best_legal_few", "best_legal_free", "best_any", "legal_share", "cg_height_full_plans"):
                print("  " + paired(case_means, column, "every_leg", other))
    print("\nRollover margin over all plans, by third objective of the search:")
    columns = [c for c in ("rollover", "rollover_low", "rollover_high") if c in d.columns]
    print(d.groupby("objective")[columns].agg(["min", "median"]).round(3).to_string())
    print("share of plans whose rollover margin is below zero (%):", (d[columns].lt(0).groupby(d["objective"]).mean() * 100).round(2).to_dict("index"))
    print("share of plans whose rollover margin is below their axle margin (%):",
          (d["rollover"].lt(d["axle"]).groupby(d["objective"]).mean() * 100).round(2).to_dict())
    print("share of plans whose worst margin is below the cap and set by rollover (%):",
          ((d["rollover"].lt(d["axle"]) & d["sigma"].lt(CAP)).groupby(d["objective"]).mean() * 100).round(2).to_dict())
    missing = case_means[["best_legal", "best_legal_few", "best_legal_free", "best_margin"]].isna().groupby("objective").sum()
    print(f"\nCases, of {d.groupby(CASE).ngroups}, in which no run has such a plan:")
    print(missing.to_string())
    print(f"\nSeconds per run: {runs.groupby('objective')['seconds'].mean().round(1).to_dict()} (not a timing result if the machine was shared)")


def search_objectives(d: pd.DataFrame) -> np.ndarray:
    """The three objectives as the every-leg search minimises them."""
    return np.column_stack([-d["utilisation"], d["obstacles"], -np.minimum(d["sigma"], CAP)])


def indicators(d: pd.DataFrame, margin=0.1) -> pd.DataFrame:
    """Hypervolume and IGD+ of every run, on objectives scaled by the reference front of its case.

    The reference front of a case is the set of nondominated plans over all runs of all algorithms.
    Objectives are scaled so that this front spans [0, 1]; the hypervolume is taken up to 1 + margin.
    An objective in which the whole reference front has one value (all its plans reach the margin
    cap, say) is scaled by the range of all plans of the case instead.
    """
    rows = []
    for case, part in d.groupby(CASE):
        f = search_objectives(part)
        front = np.unique(f[nondominated_fronts(f)[0]], axis=0)
        low = front.min(axis=0)
        span, wider = front.max(axis=0) - low, f.max(axis=0) - low
        span = np.where(span > 1e-12, span, np.where(wider > 1e-12, wider, 1.0))
        reference = (front - low) / span
        best = hypervolume(reference, np.full(3, 1 + margin))
        for run, plans in part.groupby(["algorithm", "run"]):
            scaled = (search_objectives(plans) - low) / span
            rows.append(dict(zip(CASE + ["algorithm", "run"], case + run), reference_plans=len(front), plans=len(plans),
                             hypervolume=hypervolume(scaled, np.full(3, 1 + margin)) / best, igd_plus=igd_plus(scaled, reference),
                             on_reference=int((scaled[:, None, :] == reference[None, :, :]).all(axis=2).any(axis=1).sum()),
                             seconds=plans["seconds"].iloc[0]))
    return pd.DataFrame(rows)


def algorithms_main(name):
    from scipy.stats import friedmanchisquare, wilcoxon

    d = complete_replicates(read(f"final_{name}.csv"))
    scores = indicators(d)
    algorithms = sorted(scores["algorithm"].unique())
    print(f"{scores.groupby(CASE).ngroups} cases, {scores['run'].nunique()} runs each, algorithms {algorithms}\n")
    case_means = scores.groupby(CASE + ["algorithm"])[["hypervolume", "igd_plus", "plans", "on_reference", "seconds"]].mean()
    print("Mean over cases of the mean over runs:")
    print(case_means.groupby("algorithm").mean().round(4).to_string())
    print("\nBy class:")
    for column in ("hypervolume", "igd_plus"):
        print(column)
        print(case_means[column].unstack("algorithm").groupby("br_class").mean().round(4).to_string())
    for column, larger_is_better in (("hypervolume", True), ("igd_plus", False)):
        wide = case_means[column].unstack("algorithm")
        ranks = wide.rank(axis=1, ascending=not larger_is_better)
        print(f"\n{column}: mean rank over {len(wide)} cases (1 is best): {ranks.mean().round(2).to_dict()}")
        if len(algorithms) > 2 and len(wide) > 1:
            print(f"Friedman test: p = {friedmanchisquare(*[wide[a] for a in algorithms]).pvalue:.4g}")
        tests = []
        for i, a in enumerate(algorithms):
            for b in algorithms[i + 1:]:
                diff = wide[a] - wide[b]
                p = wilcoxon(diff).pvalue if (diff.abs() > 1e-12).any() else float("nan")
                tests.append([a, b, diff.mean(), int((diff > 0).sum() if larger_is_better else (diff < 0).sum()),
                              int((diff < 0).sum() if larger_is_better else (diff > 0).sum()), p])
        tests = pd.DataFrame(tests, columns=["a", "b", "mean a minus b", "cases a better", "cases b better", "p"]).sort_values("p")
        m = tests["p"].notna().sum()  # Holm: the k-th smallest p is multiplied by m - k + 1, and adjusted values do not decrease
        tests["p, Holm"] = np.minimum(1, np.maximum.accumulate(tests["p"].to_numpy() * (m - np.arange(len(tests)))))
        print(tests.round(4).to_string(index=False))

    runs = per_run(d)
    by_algorithm = runs.groupby(CASE + ["algorithm"]).mean(numeric_only=True).groupby("algorithm").mean()
    print("\nSingle scores, for reading only (algorithms are ranked by the indicators above):")
    print(by_algorithm[["best_any", "best_legal", "best_legal_few", "best_legal_free", "best_margin", "legal_share", "plans"]].round(2).to_string())
    print(f"\nSeconds per run: {scores.groupby('algorithm')['seconds'].mean().round(1).to_dict()} (not a timing result if the machine was shared)")


def stops_main(name):
    """The search on every leg with stops merged into five against one customer per box type, case by case."""
    from scipy.stats import wilcoxon

    five = complete_replicates(read(f"final_{name}.csv"))
    one = complete_replicates(read("final_main.csv"))
    # The same run numbers on both sides, so that both are means over equally many runs from the same seeds.
    one = one[(one["objective"] == "every_leg") & one["br_class"].isin(five["br_class"].unique()) & one["run"].isin(five["run"].unique())]
    key = ["br_class", "instance", "alpha", "beta"]
    a = per_run(five).groupby(key).mean(numeric_only=True)
    b = per_run(one).groupby(key).mean(numeric_only=True)
    print(f"{len(a)} cases with {int(five['stops'].iloc[0])} stops ({five['run'].nunique()} runs) against the same {len(b)} cases "
          f"with one customer per box type ({one['run'].nunique()} runs); classes {sorted(int(c) for c in five['br_class'].unique())}\n")
    rows = []
    for column in ("best_any", "best_legal", "best_margin", "best_any_few", "best_legal_few", "best_any_free", "best_legal_free",
                   "legal_share", "fails_en_route", "plans"):
        both = pd.DataFrame({"five": a[column], "one": b[column]}).dropna()
        diff = both["five"] - both["one"]
        rows.append(dict(score=column, cases=len(both), five=both["five"].mean(), one=both["one"].mean(), difference=diff.mean(),
                         better=int((diff > 1e-9).sum()), worse=int((diff < -1e-9).sum()),
                         p=wilcoxon(diff).pvalue if (diff.abs() > 1e-9).any() else float("nan")))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nBy class (five stops | one customer per box type):")
    columns = ["best_legal", "best_legal_few", "best_legal_free"]
    print(pd.concat([a[columns].groupby("br_class").mean().add_suffix(", five"), b[columns].groupby("br_class").mean().add_suffix(", one")],
                    axis=1).round(2).to_string())
    for label, frame in (("five stops", a), ("one customer per box type", b)):
        cost = pd.DataFrame({"legal": frame["best_any"] - frame["best_legal"], "margin": frame["best_any"] - frame["best_margin"],
                             "legal, at most 20 obstacles": frame["best_any_few"] - frame["best_legal_few"],
                             "legal, no obstacles": frame["best_any_free"] - frame["best_legal_free"],
                             "at most 20 obstacles, among legal": frame["best_legal"] - frame["best_legal_few"],
                             "no obstacles, among legal": frame["best_legal"] - frame["best_legal_free"]})
        print(f"\nCost in points of utilisation, {label}: mean {cost.mean().round(2).to_dict()}")
    columns = [c for c in ("rollover", "rollover_low", "rollover_high") if c in five.columns]
    print("\nRollover margin of the plans with five stops (reference heights, lower heights, higher heights):")
    print(five[columns].agg(["min", "median"]).round(3).to_string())
    print("share of plans with a negative rollover margin (%):", (five[columns].lt(0).mean() * 100).round(2).to_dict())
    print("share of plans whose rollover margin is below their axle margin (%):",
          {c: round(100 * float(five[c].lt(five["axle"]).mean()), 2) for c in columns})
    legal = five[five["axle"] >= 0]
    print(f"of the {len(legal)} plans that respect the axle limits, share with a negative rollover margin (%):",
          (legal[columns].lt(0).mean() * 100).round(2).to_dict())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["objectives", "algorithms", "stops"])
    parser.add_argument("name")
    args = parser.parse_args()
    {"objectives": objectives_main, "algorithms": algorithms_main, "stops": stops_main}[args.what](args.name)
