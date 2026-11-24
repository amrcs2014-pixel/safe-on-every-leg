"""Tuning of the every-leg search on instances that the pilots do not use (indices 10-12).

Every configuration gets the same number of evaluations. Each run is scored on its plans that are
legal on every leg: best utilisation, best utilisation with at most 20 obstacles, and the area they
dominate in the (utilisation, obstacles) plane.

Usage, from the code folder:
  python -m experiments.tune_e5 --workers 8                search settings    -> results/tune_e5.csv
  python -m experiments.tune_e5 --workers 8 --set decoder  plain or extended  -> results/tune_decoder.csv
"""
import argparse
import time
from functools import partial
from itertools import product

import numpy as np

from legsafe import axle_margin, decode, unloading_obstacles, utilisation

from .common import BETAS, instance, run_parallel
from .e5_pilot import EXTENDED, TUNED, VEHICLE, legs, search

CLASSES, INSTANCES = [2, 4, 6], [10, 11, 12]
OBSTACLE_REF = 100
CONFIGS = [
    dict(cap=cap, seeded=seeded, pop_size=pop, generations=gens)
    for cap, seeded, (pop, gens) in product([None, 0.05, 0.10], [0.0, 0.1], [(50, 200), (100, 100)])
]
# The tuned search with the plain decoder, and with the extended decoder from a random start or
# from a start in which half the members behave as the plain decoder.
SETS = {
    "search": (CONFIGS, "tune_e5.csv"),
    "decoder": ([TUNED, dict(EXTENDED, plain_start=0.0), EXTENDED], "tune_decoder.csv"),
}


def dominated_area(util, obstacles) -> float:
    """Share of the (utilisation, obstacles) box [0,1] x [0,OBSTACLE_REF] dominated by the plans."""
    area, best = 0.0, 0.0
    for o, u in sorted(zip(obstacles, util)):
        if o >= OBSTACLE_REF:
            break
        if u > best:
            area += (u - best) * (OBSTACLE_REF - o) / OBSTACLE_REF
            best = u
    return area


def run(task):
    job, number, config = task
    inst, (_, rng) = instance(job, 2)
    started = time.perf_counter()
    keys = search(inst, rng, config)
    seconds = time.perf_counter() - started

    layouts = [decode(inst, k) for k in keys]
    util = np.array([utilisation(inst, lay) for lay in layouts])
    obstacles = np.array([unloading_obstacles(lay) for lay in layouts])
    legal = np.array([axle_margin(VEHICLE, legs(inst, lay)).min() >= 0 for lay in layouts])
    few = legal & (obstacles <= 20)
    free = legal & (obstacles == 0)
    return [dict(
        br_class=job[0], instance=job[1], alpha=job[2][0], beta=job[2][1], config=number,
        **dict(dict(extended=False, plain_start=0.0), **config),
        seconds=seconds, plans=len(keys), legal_share=legal.mean(),
        best_any=util.max(),
        best_legal=util[legal].max() if legal.any() else np.nan,
        best_legal_few=util[few].max() if few.any() else np.nan,
        best_legal_free=util[free].max() if free.any() else np.nan,
        area_legal=dominated_area(util[legal], obstacles[legal]),
    )]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--set", choices=list(SETS), default="search")
    args = parser.parse_args()
    configs, out_name = SETS[args.set]
    tasks = [((c, i, b), n, config) for c in CLASSES for i in INSTANCES for b in BETAS[::2] for n, config in enumerate(configs)]
    run_parallel(run, tasks, args.workers, out_name)
