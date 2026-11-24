"""Summary of experiment E1: exact fronts on small instances against the decoders and the search.

For every instance and required margin, these sets of plans are compared with the proven front of
the grid model (e1_exact.csv):
  plain decoder, all its plans      e1_reach.csv, complete enumeration of the key space
  plain decoder, search             e1_heuristic.csv, five runs of the tuned search
  extended decoder, search          e1_extended.csv, five runs with the same number of evaluations
  extended decoder, all its plans   e1_reach_extended.csv, only where the key space was enumerated

Usage, from the code folder: python -m experiments.e1_summary
"""
import numpy as np
import pandas as pd

from .common import OUT
from .e1_exact import MARGINS

CASE = ["space", "n_types", "seed"]
LABELS = dict(
    none="no plan at the required margin", points="exact points found", whole="whole exact front found",
    area="dominated area, share of exact", top_gap="gap in best volume (points)",
    zero_missed="no plan without obstacles", zero_gap="gap in best volume without obstacles (points)",
)


def nondominated(points):
    """(volume, obstacles) pairs with more volume and fewer obstacles preferred, best volume first."""
    front, fewest = [], np.inf
    for volume, obstacles in sorted(set(points), key=lambda p: (-p[0], p[1])):
        if obstacles < fewest:
            front.append((volume, obstacles))
            fewest = obstacles
    return front


def front_of(frame, margin=None):
    frame = frame[frame["volume"] >= 0]
    if margin is not None:
        frame = frame[frame["sigma"] >= margin - 1e-9]
    return nondominated(zip(frame["volume"], frame["obstacles"]))


def area(front, obstacle_ref):
    """Area dominated in the (volume, obstacles) plane, with reference point (0, obstacle_ref)."""
    total, edge = 0.0, obstacle_ref
    for volume, obstacles in front:  # volume falls, obstacles fall
        total += volume * (edge - obstacles)
        edge = obstacles
    return total


def covered(points, front):
    """True when no point lies outside the region dominated by `front`."""
    return all(any(v <= fv and o >= fo for fv, fo in front) for v, o in points)


def compare(found, front, cells, ref):
    """How a set of plans compares with the exact front; volumes in points of utilisation."""
    zero = max((v for v, o in front if o == 0), default=None)
    zero_found = max((v for v, o in found if o == 0), default=None)
    return dict(
        none=not found,
        points=len(set(front) & set(found)) / len(front),
        whole=set(front) <= set(found),
        area=area(found, ref) / area(front, ref) if area(front, ref) else np.nan,
        top_gap=(front[0][0] - found[0][0]) / cells * 100 if found else np.nan,
        zero_missed=zero_found is None if zero is not None else np.nan,
        zero_gap=(zero - zero_found) / cells * 100 if zero is not None and zero_found is not None else np.nan,
    )


def table(frame):
    frame = frame.astype({c: float for c in LABELS})
    groups = frame.groupby("who", sort=False)
    out = groups[list(LABELS)].mean().rename(columns=LABELS).round(3)
    out.insert(0, "whole front found, count", groups["whole"].sum().astype(int))
    out.insert(0, "fronts", groups.size())
    return out.T.to_string()


def read(name):
    path = OUT / name
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "margin" in frame:
        frame["margin"] = frame["margin"].astype(str)
    return frame


def main():
    exact, reach, plain = read("e1_exact.csv"), read("e1_reach.csv"), read("e1_heuristic.csv")
    extended, reach_extended = read("e1_extended.csv"), read("e1_reach_extended.csv")

    one = exact.drop_duplicates(CASE)
    per_front = exact.drop_duplicates(CASE + ["margin"])
    print(f"{len(one)} instances: {one['types'].min()}-{one['types'].max()} box types (stops), "
          f"{one['boxes'].min()}-{one['boxes'].max()} boxes (median {one['boxes'].median():.0f}), "
          f"{one['cells'].min()} or {one['cells'].max()} grid cells")
    print("fronts proved exact: " + ", ".join(
        f"margin {m}: {per_front[per_front['margin'] == m]['closed'].sum()} of {len(one)}" for m in MARGINS))
    print(f"time per exact front: median {per_front['seconds'].median():.0f} s, max {per_front['seconds'].max():.0f} s; "
          f"search per run: plain decoder {plain.drop_duplicates(CASE + ['run'])['seconds'].mean():.1f} s"
          + (f", extended decoder {extended.drop_duplicates(CASE + ['run'])['seconds'].mean():.1f} s" if extended is not None else "")
          + f"; plain key space: {reach.drop_duplicates(CASE)['plans'].min()}-{reach.drop_duplicates(CASE)['plans'].max()} plans")
    if reach_extended is not None:
        sizes = reach_extended.drop_duplicates(CASE)["plans"]
        print(f"extended key space enumerated on {len(sizes)} instances ({sizes.sum():,} plans, at most {sizes.max():,} each)")
    print()

    sizes = per_front[per_front["closed"]].pivot_table(index="points", columns="margin", values="seed", aggfunc="size", fill_value=0)
    print("Number of points on the exact front (proved fronts):")
    print(sizes[list(MARGINS)].to_string(), "\n")

    def of(frame, key):
        return None if frame is None else frame[(frame[CASE] == key).all(axis=1)]

    rows, search_reach, enumerated = [], [], []
    for key, e in exact.groupby(CASE):
        r, h, x, rx = (of(frame, key) for frame in (reach, plain, extended, reach_extended))
        cells = e["cells"].iloc[0]
        ref = 1 + max(frame["obstacles"].max() for frame in (e, r, h, x, rx) if frame is not None and len(frame))
        for name, margin in MARGINS.items():
            em = e[e["margin"] == name]
            if not em["closed"].all():
                continue
            front = front_of(em)
            base = dict(margin=name, multi=len(front) > 1)

            def add(who, found, into=rows):
                assert covered(found, front), f"{who} beats the exact front at {key}, margin {name}"
                into.append(dict(**base, who=who, **compare(found, front, cells, ref)))

            reachable = front_of(r[r["margin"] == name])
            union = front_of(h, margin)
            assert covered(union, reachable), f"search beats the plain decoder's reach at {key}, margin {name}"
            add("plain decoder, all its plans", reachable)
            add("plain decoder, search, five runs", union)
            if reachable:
                search_reach.append(dict(margin=name, decoder="plain", whole=set(reachable) <= set(union)))
            if x is not None:
                union_x = front_of(x, margin)
                add("extended decoder, search, five runs", union_x)
                runs = pd.DataFrame([compare(front_of(x[x["run"] == run], margin), front, cells, ref)
                                     for run in sorted(x["run"].unique())]).astype(float)
                rows.append(dict(**base, who="extended decoder, search, one run", **runs.mean().to_dict()))
                if rx is not None and len(rx):
                    reachable_x = front_of(rx[rx["margin"] == name])
                    assert covered(union_x, reachable_x), f"search beats the extended decoder's reach at {key}, margin {name}"
                    assert covered(reachable, reachable_x), f"plain decoder beats the extended one at {key}, margin {name}"
                    add("extended decoder, all its plans", reachable_x, enumerated)
                    add("extended decoder, search, five runs", union_x, enumerated)
                    add("plain decoder, all its plans", reachable, enumerated)
                    if reachable_x:
                        search_reach.append(dict(margin=name, decoder="extended", whole=set(reachable_x) <= set(union_x)))
    result = pd.DataFrame(rows)
    print("Checked on every proved front: no plan of either decoder beats the exact front, and no search plan beats its decoder's reach.\n")

    for name in MARGINS:
        print(f"Required margin {name}, all proved fronts:")
        print(table(result[result["margin"] == name]), "\n")
    print("Fronts with more than one point, all margins together:")
    print(table(result[result["multi"]]), "\n")

    sr = pd.DataFrame(search_reach)
    print("Search against what its decoder can build (whole reachable front found in five runs):")
    for (decoder, name), group in sr.groupby(["decoder", "margin"], sort=False):
        print(f"  {decoder} decoder, margin {name}: {group['whole'].sum()} of {len(group)}")
    print()

    if enumerated:
        en = pd.DataFrame(enumerated)
        print("Only the instances whose extended key space was enumerated:")
        for name in MARGINS:
            print(f"Required margin {name}:")
            print(table(en[en["margin"] == name]), "\n")

    closed = exact[(exact["volume"] >= 0) & exact["closed"]]
    top = closed.groupby(CASE + ["margin"]).apply(lambda g: pd.Series({
        "best": g["volume"].max(), "best_free": g.loc[g["obstacles"] == 0, "volume"].max(), "cells": g["cells"].iloc[0],
    }), include_groups=False).unstack("margin")
    full = top.dropna()
    print(f"Exact cost of a required margin, {len(full)} instances with all three fronts proved (points of utilisation):")
    for name in ("0", "0.05"):
        any_obstacles = (full[("best", "none")] - full[("best", name)]) / full[("cells", "none")] * 100
        no_obstacles = (full[("best_free", "none")] - full[("best_free", name)]) / full[("cells", "none")] * 100
        print(f"  margin {name}: any number of obstacles: mean {any_obstacles.mean():.2f}, non-zero in {(any_obstacles > 0).sum()} instances; "
              f"no obstacles: mean {no_obstacles.mean():.2f}, non-zero in {(no_obstacles > 0).sum()}, max {no_obstacles.max():.1f}")


if __name__ == "__main__":
    main()
