"""Summary of the convergence study. Usage, from the code folder: python -m experiments.convergence_summary

Every checkpoint is scored on the population's first front and on the archive of the same search.
"""
import pandas as pd

from .common import RUN, read

UTILISATION = {
    "best_any": "best plan, legal or not", "best_legal": "best legal plan",
    "best_legal_few": "best legal plan, at most 20 obstacles", "best_legal_free": "best legal plan, no obstacles",
    "best_margin_few": "best plan with a 5% margin, at most 20 obstacles",
}
AREAS = {"area_legal": "area dominated by legal plans", "area_margin": "area dominated by plans with a 5% margin"}
KEY = RUN + ["run"]


def main():
    d = read("convergence.csv")
    budgets = sorted(d["evaluations"].unique())
    print(f"{d.groupby(RUN).ngroups} cases x {d['run'].nunique()} runs = {d.groupby(KEY).ngroups} runs; "
          f"box types per case: {sorted(int(t) for t in d['types'].unique())}; budgets {[int(b) for b in budgets]}\n")

    def wide(column, source, index=KEY):
        return d[d["source"] == source].pivot_table(index=index, columns="evaluations", values=column) * 100

    print("Mean over runs, by number of evaluations (utilisation in %, areas x 100).")
    for column, label in {**UTILISATION, **AREAS}.items():
        population, archive = wide(column, "population"), wide(column, "archive")
        complete = population.notna().all(axis=1) & archive.notna().all(axis=1)
        population, archive = population[complete], archive[complete]
        dropped = (~complete).sum()
        print(f"\n{label}" + (f" ({dropped} runs without such a plan at some budget are left out)" if dropped else ""))
        print(pd.DataFrame({
            "population": population.mean(), "archive": archive.mean(), "archive minus population": (archive - population).mean(),
            "archive, gain of the doubling": archive.mean().diff(),
            "archive, below its last budget": archive.mean().iloc[-1] - archive.mean(),
            "archive, runs still improving after": [(archive[budgets[-1]] > archive[b] + 1e-9).mean() for b in budgets],
        }).round(2).T.to_string())

    by_class = ["br_class"] + KEY[1:]
    for column, label in (("area_legal", "area dominated by legal plans (x 100)"),
                          ("best_legal_few", "best legal plan with at most 20 obstacles (%)")):
        for source in ("population", "archive"):
            table = wide(column, source, by_class).dropna().groupby("br_class").mean()
            print(f"\n{label}, {source}, by benchmark class:")
            print(table.round(2).to_string())
            print("gain of each doubling:")
            print(table.diff(axis=1).iloc[:, 1:].round(2).to_string())

    print("\nArchive at one budget against the population at twice that budget, area dominated by legal plans (x 100):")
    population, archive = wide("area_legal", "population"), wide("area_legal", "archive")
    for small, large in zip(budgets, budgets[1:]):
        diff = archive[small] - population[large]
        print(f"  archive at {small} minus population at {large}: mean {diff.mean():+.2f}, archive at least as good in {(diff >= -1e-9).mean():.0%} of runs")

    print("\nPlans held, by benchmark class and budget (mean):")
    for source, column, label in (("population", "plans", "members of the population's first front"),
                                  ("population", "distinct", "distinct objective vectors among them"),
                                  ("archive", "plans", "plans in the archive")):
        print(f"{label}:")
        print(d[d["source"] == source].pivot_table(index="br_class", columns="evaluations", values=column).round(0).to_string())

    print("\nSeconds per run by benchmark class and budget:")
    print(d[d["source"] == "archive"].pivot_table(index="br_class", columns="evaluations", values="seconds").round(0).to_string())

    spread = (d.groupby(["source"] + RUN + ["evaluations"])["area_legal"].agg(lambda s: s.max() - s.min())
              .groupby(["source", "evaluations"]).mean().unstack("evaluations") * 100)
    print("\nMean spread of the legal area between the runs of one case (x 100), by budget:")
    print(spread.round(2).to_string())


def levelling(gains: pd.Series, threshold: float):
    """Smallest budget after which every further doubling gains less than the threshold; None if there is none."""
    budgets = list(gains.index)
    for position, budget in enumerate(budgets[:-1]):
        if (gains.iloc[position + 1:] < threshold).all():
            return budget
    return None


def types_main():
    """The study with budgets per box type (convergence_types.csv)."""
    d = read("convergence_types.csv")
    budgets = sorted(d["per_type"].unique())
    by_class = ["br_class", "types"] + KEY[1:]
    print(f"{d.groupby(RUN).ngroups} cases x {d['run'].nunique()} runs = {d.groupby(KEY).ngroups} runs; "
          f"budgets in evaluations per box type: {[int(b) for b in budgets]}\n")
    print("Evaluations at each budget, by class (box types in brackets):")
    table = d.pivot_table(index=["br_class", "types"], columns="per_type", values="evaluations")
    print(table.astype(int).to_string(), "\n")

    def wide(column, source):
        return d[d["source"] == source].pivot_table(index=by_class, columns="per_type", values=column) * 100

    for column, label in (("area_legal", "area dominated by legal plans (x 100)"),
                          ("best_legal_few", "best legal plan with at most 20 obstacles (%)"),
                          ("best_legal", "best legal plan (%)")):
        archive = wide(column, "archive")
        complete = archive.notna().all(axis=1)
        mean = archive[complete].groupby(["br_class", "types"]).mean()
        gain = mean.diff(axis=1).iloc[:, 1:]
        print(f"{label}, archive, by class and evaluations per box type"
              + (f" ({(~complete).sum()} runs without such a plan at some budget are left out)" if (~complete).any() else "") + ":")
        print(mean.round(2).to_string())
        print("gain of each doubling:")
        print(gain.round(2).to_string())
        for threshold in (0.5, 0.25):
            points = {f"BR{c} ({t})": levelling(gain.loc[(c, t)], threshold) for c, t in gain.index}
            print(f"budget per type after which every doubling gains less than {threshold}: "
                  + ", ".join(f"{k}: {'not reached' if v is None else int(v)}" for k, v in points.items()))
        population = wide(column, "population")[complete].groupby(["br_class", "types"]).mean()
        print("archive minus population:")
        print((mean - population).round(2).to_string(), "\n")

    last = d[d["per_type"] == budgets[-1]]
    archive, population = last[last["source"] == "archive"], last[last["source"] == "population"]
    print("At the last budget, by class:")
    summary = pd.DataFrame({
        "evaluations": archive.groupby("br_class")["evaluations"].mean(),
        "seconds per run": archive.groupby("br_class")["seconds"].mean(),
        "ms per evaluation": 1000 * archive.groupby("br_class")["seconds"].mean() / archive.groupby("br_class")["evaluations"].mean(),
        "plans in the archive": archive.groupby("br_class")["plans"].mean(),
        "distinct plans in the population": population.groupby("br_class")["distinct"].mean(),
        "spread of legal area between seeds x 100": archive.groupby(RUN)["area_legal"].agg(lambda s: s.max() - s.min()).groupby("br_class").mean() * 100,
    })
    print(summary.round(2).to_string(), "\n")

    print("Distinct objective vectors among the 100 members of the population, by class and budget per type:")
    print(d[d["source"] == "population"].pivot_table(index=["br_class", "types"], columns="per_type", values="distinct").round(0).to_string())


VARIANTS = {  # file suffix: (the variant, the search it is compared with)
    "distinct": ("copies ranked last", "copies kept"),
    "one_key": ("one key mutated per child", "present mutation rate"),
    "seeded": ("search started from a search for fill", "present search"),
    "two_ends": ("search started from both ends of the front", "present search"),
}


def compare_main(variant="distinct"):
    """A variant of the search against the present one, run by run (budgets per box type)."""
    from scipy.stats import wilcoxon

    new, old = VARIANTS[variant]
    keep, drop = read("convergence_types.csv"), read(f"convergence_types_{variant}.csv")
    budgets = sorted(keep["per_type"].unique())
    by_class = ["br_class", "types"] + KEY[1:]
    print(f"{drop.groupby(KEY).ngroups} runs with {new} against {keep.groupby(KEY).ngroups} runs with {old}; "
          f"same cases, seeds and budgets per box type. Scores are those of the archive.\n")

    def wide(frame, column, source="archive"):
        return frame[frame["source"] == source].pivot_table(index=by_class, columns="per_type", values=column) * 100

    for column, label in (("area_legal", "area dominated by legal plans (x 100)"),
                          ("best_legal_few", "best legal plan with at most 20 obstacles (%)"),
                          ("best_legal", "best legal plan (%)")):
        a, b = wide(drop, column), wide(keep, column)
        a, b = a.align(b, join="inner", axis=0)
        complete = a.notna().all(axis=1) & b.notna().all(axis=1)
        diff = (a - b)[complete]
        print(f"{label}: {new} minus {old}, mean by class and budget per box type"
              + (f" ({(~complete).sum()} runs left out)" if (~complete).any() else "") + ":")
        print(diff.groupby(["br_class", "types"]).mean().round(2).to_string())
        def test(values):
            return wilcoxon(values).pvalue if (values.abs() > 1e-9).any() else float("nan")

        groups = diff.groupby(["br_class", "types"])
        better, worse = groups.agg(lambda s: (s > 1e-9).sum()), groups.agg(lambda s: (s < -1e-9).sum())
        print(f"runs better / worse, of {groups.size().iloc[0]} per class:")
        print((better.astype(str) + "/" + worse.astype(str)).to_string())
        print("p of a paired Wilcoxon signed-rank test (not corrected for the number of tests):")
        print(groups.agg(test).round(3).to_string())
        large = diff.index.get_level_values("types") > 20
        for name, part in (("all classes", diff), ("up to 20 box types", diff[~large]), ("more than 20 box types", diff[large])):
            print(f"{name}, {len(part)} runs, by budget:")
            print(pd.DataFrame({"mean": part.mean(), "median": part.median(), "better": (part > 1e-9).sum(),
                                "worse": (part < -1e-9).sum(), "p": part.agg(test)}).round(3).T.to_string())
        mean = a[complete].groupby(["br_class", "types"]).mean()
        print(f"with {new}: value, then gain of each doubling:")
        print(mean.round(2).to_string())
        print(mean.diff(axis=1).iloc[:, 1:].round(2).to_string(), "\n")

    print(f"Distinct objective vectors among the 100 members of the population, {new}:")
    print(drop[drop["source"] == "population"].pivot_table(index=["br_class", "types"], columns="per_type", values="distinct").round(0).to_string())
    print(f"\nThe same, {old}:")
    print(keep[keep["source"] == "population"].pivot_table(index=["br_class", "types"], columns="per_type", values="distinct").round(0).to_string())

    last_drop, last_keep = (f[(f["per_type"] == budgets[-1]) & (f["source"] == "archive")] for f in (drop, keep))
    print(f"\nAt the last budget, by class ({new} | {old}):")
    print(pd.DataFrame({
        "plans in the archive": last_drop.groupby("br_class")["plans"].mean(),
        "plans in the archive, old": last_keep.groupby("br_class")["plans"].mean(),
        "seconds per run": last_drop.groupby("br_class")["seconds"].mean(),
        "seconds per run, old": last_keep.groupby("br_class")["seconds"].mean(),
        "spread between seeds x 100": last_drop.groupby(RUN)["area_legal"].agg(lambda s: s.max() - s.min()).groupby("br_class").mean() * 100,
        "spread between seeds x 100, old": last_keep.groupby(RUN)["area_legal"].agg(lambda s: s.max() - s.min()).groupby("br_class").mean() * 100,
    }).round(2).to_string())


def stops_main(name="convergence_types_stops.csv"):
    """The same boxes on routes with fewer stops against one customer per box type (budgets per box type)."""
    from scipy.stats import wilcoxon

    short = read(name)
    full = read("convergence_types.csv")
    full = full[full["br_class"].isin(short["br_class"].unique())].assign(stops=lambda frame: frame["types"])
    d = pd.concat([full, short], ignore_index=True)
    d["stops"] = d["stops"].astype(int)
    budgets = sorted(d["per_type"].unique())
    case = KEY[1:]  # instance, weight setting and run: the same for every number of stops
    index = ["br_class", "types", "stops"]
    print(f"{d.groupby(index + case).ngroups} runs: classes {sorted(int(c) for c in d['br_class'].unique())}, "
          f"{d.groupby(['br_class'] + case).ngroups // d['br_class'].nunique()} runs per class and number of stops; "
          f"the rows with as many stops as box types are the earlier study. Scores are those of the archive.\n")
    archive = d[d["source"] == "archive"]

    def test(values):
        return wilcoxon(values).pvalue if (values.abs() > 1e-9).any() else float("nan")

    for column, label in (("best_legal", "best legal plan (%)"),
                          ("best_legal_free", "best legal plan without obstacles (%)"),
                          ("best_legal_few", "best legal plan with at most 20 obstacles (%)"),
                          ("area_legal", "area dominated by legal plans (x 100)"),
                          ("best_any", "best plan, legal or not (%)")):
        table = archive.set_index(index + case + ["per_type"])[column].unstack("per_type") * 100
        counts = table.notna().groupby(index).sum()
        missing = (counts < counts.max().max()).any().any()
        print(f"{label}, mean by class, stops and evaluations per box type" + (" (over the runs that have such a plan)" if missing else "") + ":")
        mean = table.groupby(index).mean()
        print(mean.round(2).to_string())
        if missing:
            print("runs that have such a plan:")
            print(counts.to_string())
        print("gain of each doubling:")
        print(mean.diff(axis=1).iloc[:, 1:].round(2).to_string())
        rows = []
        for (br_class, types), part in table.groupby(["br_class", "types"]):
            base = part.xs(types, level="stops").droplevel(["br_class", "types"])
            for stops in sorted(s for s in part.index.get_level_values("stops").unique() if s != types):
                diff = part.xs(stops, level="stops").droplevel(["br_class", "types"]) - base
                for budget in (budgets[2], budgets[5], budgets[-1]):
                    at = diff[budget].dropna()
                    rows.append(dict(br_class=br_class, types=types, stops=stops, per_type=budget, runs=len(at), mean=at.mean(),
                                     median=at.median(), better=int((at > 1e-9).sum()), worse=int((at < -1e-9).sum()), p=test(at)))
        print("against one customer per box type, run by run (runs with such a plan at that budget in both); "
              "p from a paired Wilcoxon signed-rank test, not corrected:")
        print(pd.DataFrame(rows).set_index(["br_class", "types", "stops", "per_type"]).round(3).to_string(), "\n")

    last = archive[archive["per_type"] == budgets[-1]]
    print("At the last budget, by class and stops:")
    print(pd.DataFrame({
        "plans in the archive": last.groupby(index)["plans"].mean(),
        "seconds per run": last.groupby(index)["seconds"].mean(),
        "spread of the best legal plan between seeds (points)":
            last.groupby(index + RUN[1:])["best_legal"].agg(lambda s: s.max() - s.min()).groupby(index).mean() * 100,
    }).round(2).to_string())
    population = d[d["source"] == "population"]
    print("\nDistinct objective vectors among the 100 members of the population:")
    print(population.pivot_table(index=index, columns="per_type", values="distinct").round(0).to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=["budget", "types"], default="budget")
    parser.add_argument("--compare", choices=list(VARIANTS), nargs="?", const="distinct",
                        help="a variant of the search against the present one, run by run")
    parser.add_argument("--stops", action="store_true", help="routes with fewer stops against one customer per box type")
    args = parser.parse_args()
    if args.stops:
        stops_main()
    elif args.compare:
        compare_main(args.compare)
    else:
        types_main() if args.grid == "types" else main()
