"""Experiment E1, decoder reach: the best plans the decoder can build, by enumerating its key space.

A plan is decided by the order of the box types and by the first orientation tried for each type;
with the extended decoder, also by the number of boxes of each type left behind and by the free
length at the front wall. The key space is therefore finite. Comparing its nondominated plans with
the exact fronts and with the plans the search found separates what the decoder cannot represent
from what the search misses.

Usage, from the code folder:
  python -m experiments.e1_reach --workers 8                           -> results/e1_reach.csv
  python -m experiments.e1_reach --workers 8 --extended --limit 3e6    -> results/e1_reach_extended.csv
The key space is enumerated only on instances where it has at most --limit plans.
"""
import argparse
import itertools
import math
from functools import partial

import numpy as np

from legsafe import axle_margin, decode, n_keys, unloading_obstacles
from legsafe.decoder import level_key, offset_levels

from .common import run_parallel
from .e1_exact import JOBS, MARGINS, SPACES, describe
from .e5_pilot import VEHICLE, legs
from .small import DELTA, small_instance


def choices(inst, extended: bool):
    """For every key after the order keys, values that cover all the decoder can do with it."""
    lists = [[(i + 0.5) / len(t.orientations()) for i in range(len(t.orientations()))] for t in inst.types]
    if extended:
        lists += [[level_key(v, t.count) for v in range(t.count + 1)] for t in inst.types]
        top = offset_levels(inst, DELTA)[1]
        lists.append([level_key(v, top) for v in range(top + 1)])
    return lists


def key_space(inst, extended: bool) -> int:
    return math.factorial(len(inst.types)) * math.prod(len(c) for c in choices(inst, extended))


def run_reach(job, extended=False):
    inst = small_instance(job[2], job[1], space=SPACES[job[0]])
    n_types = len(inst.types)
    best = {name: {} for name in MARGINS}  # margin level -> obstacles -> largest volume
    keys = np.empty(n_keys(inst, extended))
    for order in itertools.permutations(range(n_types)):
        keys[list(order)] = np.arange(n_types) / n_types
        for choice in itertools.product(*choices(inst, extended)):
            keys[n_types:] = choice
            layout = decode(inst, keys, DELTA)
            volume = round(layout.size.prod(axis=1).sum() / DELTA**3)
            obstacles = unloading_obstacles(layout)
            sigma = axle_margin(VEHICLE, legs(inst, layout)).min()
            for name, margin in MARGINS.items():
                if (margin is None or sigma >= margin - 1e-9) and volume > best[name].get(obstacles, -1):
                    best[name][obstacles] = volume
    plans = key_space(inst, extended)
    rows = []
    for name, by_obstacles in best.items():
        fewest = np.inf
        for volume, obstacles in sorted(((v, o) for o, v in by_obstacles.items()), key=lambda p: (-p[0], p[1])):
            if obstacles < fewest:
                rows.append(dict(**describe(job, inst), plans=plans, margin=name, volume=volume, obstacles=obstacles))
                fewest = obstacles
        if not by_obstacles:
            rows.append(dict(**describe(job, inst), plans=plans, margin=name, volume=-1, obstacles=-1))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--limit", type=float, default=math.inf)
    args = parser.parse_args()
    sizes = {job: key_space(small_instance(job[2], job[1], space=SPACES[job[0]]), args.extended) for job in JOBS}
    chosen = sorted((job for job in JOBS if sizes[job] <= args.limit), key=lambda job: -sizes[job])
    print(f"{len(chosen)} of {len(JOBS)} instances, {sum(sizes[j] for j in chosen):,} plans", flush=True)
    run_parallel(partial(run_reach, extended=args.extended), chosen, args.workers,
                 "e1_reach_extended.csv" if args.extended else "e1_reach.csv")
