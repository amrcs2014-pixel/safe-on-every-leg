"""Summary of the E2 pilot. Usage, from the code folder: python -m experiments.e2_summary"""
import pandas as pd

from .common import MAX_FRONT, MAX_REAR, OUT, RUN, TARE


def per_run_share(d: pd.DataFrame, flag: pd.Series) -> float:
    """Mean over runs of the share of a run's plans that carry the flag, so large fronts do not dominate."""
    return d.assign(flag=flag).groupby(RUN)["flag"].mean().mean()


def main():
    d = pd.read_csv(OUT / "e2_pilot.csv")
    d["balanced"] = d[["rd_x", "rd_y"]].max(axis=1) <= 0.05
    d["few_obstacles"] = d["obstacles"] <= 20
    rise = d["front_peak"] - d["front_depot"]
    runs = d.groupby(RUN).ngroups
    print(f"{runs} runs, {len(d)} nondominated plans, {d.groupby(RUN)['seconds'].first().mean():.0f} s per run on average\n")

    print("Plans found with the departure-state objective (means over plans):")
    print(d.groupby("br_class").agg(
        stops=("stops", "first"), plans=("plan", "size"),
        utilisation=("utilisation", "mean"), best_utilisation=("utilisation", "max"),
        obstacles=("obstacles", "mean"), rd_x=("rd_x", "mean"), rd_y=("rd_y", "mean"), rd_z=("rd_z", "mean"),
        balanced_share=("balanced", "mean"),
    ).round(3).to_string(), "\n")

    later = d["front_peak_leg"] > 0
    print("Cargo-induced axle loads, independent of any limit:")
    print(f"  front-axle load rises on at least one leg: {per_run_share(d, d['legs_front_rises'] > 0):.1%} of plans")
    print(f"  front-axle load peaks after the depot:     {per_run_share(d, later):.1%} of plans")
    print(f"  size of that rise: median {rise[later].median():.2f} t, 90th percentile {rise[later].quantile(0.9):.2f} t, "
          f"max {rise[later].max():.2f} t")
    print(f"  rear-axle load peaks after the depot:      {(d['rear_peak'] > d['rear_depot'] + 1e-9).mean():.1%} of plans "
          f"(Corollary 1.2 predicts none)\n")

    for name, (tare_front, tare_rear) in TARE.items():
        legal = (d["front_depot"] + tare_front <= MAX_FRONT) & (d["rear_depot"] + tare_rear <= MAX_REAR)
        fails_later = legal & (d["front_peak"] + tare_front > MAX_FRONT)
        groups = [d[c] for c in RUN]
        share = (fails_later.groupby(groups).sum() / legal.groupby(groups).sum()).dropna()
        print(f"Tare: {name} (front {tare_front:.3f} t, rear {tare_rear:.3f} t)")
        print(f"  legal at the depot:                            {per_run_share(d, legal):.1%} of plans")
        print(f"  legal at the depot, illegal on a later leg:    {per_run_share(d, fails_later):.1%} of plans")
        print(f"  ... as a share of the depot-legal plans:       {share.mean():.1%} "
              f"(mean over {len(share)} runs with a depot-legal plan; {(share > 0).mean():.0%} of those runs affected)")
        for label, subset in (("balanced at departure (x and y within 5%)", d["balanced"]),
                              ("at most 20 obstacles", d["few_obstacles"])):
            both = subset & legal
            if both.any():
                print(f"  ... among depot-legal plans {label}: {(both & fails_later).sum() / both.sum():.1%} "
                      f"({both.sum()} plans)")
        table = pd.DataFrame({"legal": legal, "fails_later": fails_later})
        print(table.groupby(d["br_class"]).mean().round(3).T.to_string())
        print(table.groupby([d["alpha"], d["beta"]]).mean().round(3).T.to_string(), "\n")


if __name__ == "__main__":
    main()
