"""Summary of a tuning grid.

Usage, from the code folder:
  python -m experiments.tune_summary [--classes 2 4]                search settings
  python -m experiments.tune_summary --set decoder [--classes 2 4]  plain against extended decoder
"""
import argparse

import pandas as pd

from .common import OUT, RUN

SETS = {
    "search": ("tune_e5.csv", ["cap", "seeded", "pop_size"]),
    "decoder": ("tune_decoder.csv", ["extended", "plain_start"]),
}
SCORES = ["area_legal", "best_legal", "best_legal_few", "best_any", "legal_share", "seconds"]
PAIRED = {"best_any": "best plan, legal or not", "best_legal": "best legal plan",
          "best_legal_few": "best legal plan, at most 20 obstacles", "best_legal_free": "best legal plan, no obstacles"}


def main(classes=None, name="search"):
    file, factors = SETS[name]
    d = pd.read_csv(OUT / file)
    if classes:
        d = d[d["br_class"].isin(classes)]
    d["cap"] = d["cap"].fillna(0).map(lambda c: "none" if c == 0 else f"{c:g}")
    d["has_legal_few"] = d["best_legal_few"].notna()
    cases = d.groupby(RUN).ngroups
    print(f"{cases} cases, {d['config'].nunique()} configurations, {len(d)} runs\n")

    table = d.groupby(["config"] + factors)[SCORES + ["has_legal_few"]].mean()
    # A case with no legal low-obstacle plan scores zero here, so missing plans are penalised.
    table["best_legal_few_or_0"] = d.assign(v=d["best_legal_few"].fillna(0)).groupby(["config"] + factors)["v"].mean()
    print("Mean per configuration, best area first:")
    print(table.sort_values("area_legal", ascending=False).round(3).to_string(), "\n")

    print("Main effects on the dominated area of legal plans:")
    for factor in factors:
        print(f"  {factor}: " + ", ".join(f"{k}={v:.3f}" for k, v in d.groupby(factor)["area_legal"].mean().items()))

    baseline = d[d["config"] == 0].set_index(RUN)
    print("\nPaired against configuration 0, per case (dominated area of legal plans):")
    for number, group in d.groupby("config"):
        diff = group.set_index(RUN)["area_legal"] - baseline["area_legal"]
        print(f"  config {number:2d}: mean {diff.mean():+.3f} | better in {(diff > 1e-9).sum():2d}, worse in {(diff < -1e-9).sum():2d} of {len(diff)}")

    for score, label in PAIRED.items():
        if score not in d or name == "search":
            continue
        print(f"\nPaired against configuration 0, {label} (points of utilisation):")
        for number, group in d.groupby("config"):
            if number == 0:
                continue
            new, old = group.set_index(RUN)[score], baseline[score]
            both = new.notna() & old.notna()
            diff = (new - old)[both] * 100
            print(f"  config {number:2d}: found by both in {both.sum():2d} cases, mean {diff.mean():+.2f}, better in {(diff > 1e-9).sum()}, "
                  f"worse in {(diff < -1e-9).sum()} | found only by this one {(new.notna() & old.isna()).sum()}, "
                  f"only by configuration 0 {(new.isna() & old.notna()).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, nargs="+")
    parser.add_argument("--set", choices=list(SETS), default="search")
    args = parser.parse_args()
    main(args.classes, args.set)
