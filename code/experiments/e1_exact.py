"""Experiment E1: exact fronts on small instances against the tuned heuristic search.

The exact model works on a 0.5 m grid, so its fronts are exact for the problem restricted to that
grid; every plan the decoder builds lies on the grid. Loading spaces are one or one and a half
metres wide, because wider ones leave room for one lane per customer and the objectives stop
conflicting.

Usage, from the code folder:
  python -m experiments.e1_exact exact --workers 8      -> results/e1_exact.csv
  python -m experiments.e1_exact heuristic --workers 8  -> results/e1_heuristic.csv  (plain decoder)
  python -m experiments.e1_exact extended --workers 8   -> results/e1_extended.csv   (extended decoder)
"""
import argparse
import os
import sys
import time
from functools import partial

import numpy as np

from legsafe import decode, unloading_obstacles
from legsafe.exact import GridModel

from .common import REFERENCE_TARE, TARE, run_parallel, truck
from .e5_pilot import EXTENDED, TUNED, margin_of, search
from .small import DELTA, small_instance

SPACES = {"narrow": (6.0, 1.0, 1.0), "tall": (6.0, 1.0, 1.5)}
MARGINS = {"none": None, "0": 0.0, "0.05": 0.05}
TIME_LIMIT = 120.0
HEURISTIC_SEEDS = 5
JOBS = ([("narrow", n, seed) for n in (4, 5, 6) for seed in range(1, 11)]
        + [("tall", n, seed) for n in (4, 5, 6) for seed in range(1, 6)])


def silence():
    """HiGHS prints debug lines straight to the process's output; send a worker's output to nowhere."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    sys.stdout.flush()
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)


def vehicle_for(inst):
    """The reference truck with its rollover data, its centre line at the middle of the narrow cargo space."""
    return truck(TARE[REFERENCE_TARE], rollover=True, width=inst.space[1])


def describe(job, inst):
    space, n_types, seed = job
    return dict(space=space, n_types=n_types, seed=seed, types=len(inst.types),
                boxes=sum(t.count for t in inst.types), cells=round(np.prod(SPACES[space]) / DELTA**3))


def run_exact(job, time_limit=TIME_LIMIT):
    inst = small_instance(job[2], job[1], space=SPACES[job[0]])
    vehicle = vehicle_for(inst)
    model = GridModel(inst, vehicle, DELTA)
    rows = []
    for name, margin in MARGINS.items():
        started = time.perf_counter()
        points, closed = model.front(margin, time_limit)
        seconds = time.perf_counter() - started
        for point in points or [None]:
            rows.append(dict(
                **describe(job, inst), margin=name, closed=closed, seconds=seconds, points=len(points),
                volume=point.volume if point else -1, obstacles=point.obstacles if point else -1,
                proven=point.proven if point else False,
                sigma=margin_of(inst, point.layout, vehicle) if point else np.nan,
            ))
    return rows


def run_heuristic(job, config=TUNED):
    inst = small_instance(job[2], job[1], space=SPACES[job[0]])
    vehicle = vehicle_for(inst)
    rows = []
    for run in range(HEURISTIC_SEEDS):
        rng = np.random.default_rng([job[2], job[1], run, 77])
        started = time.perf_counter()
        keys = search(inst, rng, config, DELTA, vehicle=vehicle)
        seconds = time.perf_counter() - started
        for k in keys:
            layout = decode(inst, k, DELTA)
            rows.append(dict(
                **describe(job, inst), run=run, seconds=seconds,
                volume=round(layout.size.prod(axis=1).sum() / DELTA**3), obstacles=unloading_obstacles(layout),
                sigma=margin_of(inst, layout, vehicle),
            ))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["exact", "heuristic", "extended"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="seconds per solve of the exact model")
    args = parser.parse_args()
    if args.part == "exact":
        run_parallel(partial(run_exact, time_limit=args.time_limit), JOBS, args.workers, "e1_exact.csv", initializer=silence)
    elif args.part == "heuristic":
        run_parallel(run_heuristic, JOBS, args.workers, "e1_heuristic.csv")
    else:
        run_parallel(partial(run_heuristic, config=EXTENDED), JOBS, args.workers, "e1_extended.csv")
