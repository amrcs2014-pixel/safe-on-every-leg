"""Summary of the E5 pilot: what legality on every leg costs in utilisation.

Usage, from the code folder:
  python -m experiments.e5_summary [--search NAME] [--against NAME] [--classes 1 2 3]
NAME is a configuration of experiments.e5_pilot: untuned, tuned, extended or extended-long.
--search picks the every-leg search that is summarised, --against a second one to pair it with.
"""
import argparse

import numpy as np
import pandas as pd

from .common import MAX_FRONT, MAX_REAR, OUT, REFERENCE_TARE, RUN, TARE
from .e5_pilot import CONFIGS
from .tune_e5 import dominated_area

LEVELS = [0.0, 0.02, 0.05, 0.10, 0.15]
CAPS = (("any number of obstacles", np.inf), ("at most 20 obstacles", 20), ("no obstacles", 0))


def best(d: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Best utilisation per run among the plans selected by the mask (NaN where a run has none)."""
    return d[mask].groupby(RUN)["utilisation"].max()


def paired(label: str, new: pd.Series, old: pd.Series, runs: pd.Index):
    new, old = new.reindex(runs), old.reindex(runs)
    has_new, has_old = new.notna(), old.notna()
    diff = (new - old)[has_new & has_old] * 100
    print(f"  {label}: a plan is found by both in {(has_new & has_old).sum()} runs, only the first in "
          f"{(has_new & ~has_old).sum()}, only the second in {(~has_new & has_old).sum()}; where both, "
          f"mean {diff.mean():+.2f} points (better in {(diff > 1e-9).mean():.0%}, worse in {(diff < -1e-9).mean():.0%})")


def main(classes=None, search="untuned", against=None):
    tare_front, tare_rear = TARE[REFERENCE_TARE]
    leg = pd.read_csv(OUT / CONFIGS[search][1])
    dep = pd.read_csv(OUT / "e2_pilot.csv")
    other = pd.read_csv(OUT / CONFIGS[against][1]) if against else None
    if classes:
        leg, dep = leg[leg["br_class"].isin(classes)], dep[dep["br_class"].isin(classes)]
        other = other[other["br_class"].isin(classes)] if against else None
    dep["sigma"] = np.minimum((MAX_FRONT - dep["front_peak"] - tare_front) / MAX_FRONT,
                              (MAX_REAR - dep["rear_peak"] - tare_rear) / MAX_REAR)
    dep["depot_margin"] = np.minimum((MAX_FRONT - dep["front_depot"] - tare_front) / MAX_FRONT,
                                     (MAX_REAR - dep["rear_depot"] - tare_rear) / MAX_REAR)
    both = pd.concat([dep.assign(search="departure"), leg.assign(search="every leg")], ignore_index=True)
    runs = leg.groupby(RUN).ngroups
    name = f"{search} every-leg search"
    print(f"Tare: {REFERENCE_TARE}. {runs} runs; {len(dep)} plans from the departure-state search, "
          f"{len(leg)} from the {name} ({leg.groupby(RUN)['seconds'].first().mean():.0f} s per run).\n")

    print("Share of each search's nondominated plans (mean over runs):")
    for label, d in (("departure-state objective", dep), ("every-leg objective", leg)):
        legal = d.assign(a=d["depot_margin"] >= 0, b=d["sigma"] >= 0).groupby(RUN)[["a", "b"]].mean().mean()
        print(f"  {label:27s} legal at the depot {legal['a']:.1%} | legal on every leg {legal['b']:.1%}")
    print()

    for label, cap in CAPS:
        ok = both["obstacles"] <= cap
        table = pd.DataFrame({
            "any": best(both, ok),
            "depot_legal": best(both, ok & (both["depot_margin"] >= 0)),
            "safe_departure_search": best(both, ok & (both["sigma"] >= 0) & (both["search"] == "departure")),
            "safe_leg_search": best(both, ok & (both["sigma"] >= 0) & (both["search"] == "every leg")),
            **{f"sigma>={lv:g}": best(both, ok & (both["sigma"] >= lv)) for lv in LEVELS},
        })
        covered = table["any"].notna()
        c = ((table["any"] - table["sigma>=0"])[covered]).dropna() * 100
        print(f"=== Best utilisation per run, {label} ({covered.sum()} of {runs} runs have such a plan) ===")
        print(f"  runs with a plan legal on every leg: {table['sigma>=0'].notna().sum()}")
        print(f"  cost of legality on every leg (percentage points of utilisation), over runs with a legal plan:")
        print(f"    mean {c.mean():.2f} | median {c.median():.2f} | 90th percentile {c.quantile(0.9):.2f} | max {c.max():.2f} "
              f"| zero in {(c <= 1e-9).mean():.0%} of runs")
        paired("best legal plan, every-leg search against departure search",
               table["safe_leg_search"], table["safe_departure_search"], table.index)
        print("  mean best utilisation by required margin (runs with a plan at that level in brackets):")
        print("    any: %.3f | depot-legal: %.3f | " % (table["any"].mean(), table["depot_legal"].mean())
              + " | ".join(f"sigma>={lv:g}: {table[f'sigma>={lv:g}'].mean():.3f} ({table[f'sigma>={lv:g}'].notna().sum()})"
                           for lv in LEVELS))
        by_class = table.assign(cost=(table["any"] - table["sigma>=0"]) * 100).groupby("br_class")[["any", "sigma>=0", "cost"]].mean()
        print(by_class.round(3).T.to_string(), "\n")

    if against:
        print(f"=== {search} against {against} every-leg search, same cases ===")
        all_runs = leg.groupby(RUN).size().index
        for level in (0.0, 0.05):
            for label, cap in CAPS:
                paired(f"best plan with a margin of at least {level:g}, {label}",
                       best(leg, (leg["sigma"] >= level) & (leg["obstacles"] <= cap)),
                       best(other, (other["sigma"] >= level) & (other["obstacles"] <= cap)), all_runs)
        paired("best plan, legal or not", best(leg, leg["sigma"] > -np.inf), best(other, other["sigma"] > -np.inf), all_runs)

        def area(d):
            legal = d[d["sigma"] >= 0]
            return legal.groupby(RUN).apply(lambda g: dominated_area(g["utilisation"], g["obstacles"]), include_groups=False)

        new, old = area(leg).reindex(all_runs).fillna(0), area(other).reindex(all_runs).fillna(0)
        diff = new - old
        print(f"  area dominated by legal plans: {search} {new.mean():.3f}, {against} {old.mean():.3f}, "
              f"mean difference {diff.mean():+.3f} (better in {(diff > 1e-9).mean():.0%} of runs, worse in {(diff < -1e-9).mean():.0%})")
        seconds = [d.groupby(RUN)["seconds"].first().mean() for d in (leg, other)]
        print(f"  seconds per run: {search} {seconds[0]:.0f}, {against} {seconds[1]:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, nargs="+")
    parser.add_argument("--search", choices=list(CONFIGS), default="untuned")
    parser.add_argument("--against", choices=list(CONFIGS))
    args = parser.parse_args()
    main(args.classes, args.search, args.against)
