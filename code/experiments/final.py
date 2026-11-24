"""The final experiments: one runner for the questions E2 to E5.

A run is one search on one case with one seed. What changes between experiments is the third
objective of the search and the search algorithm; decoder, variation operators, starting
population, budget and archive are the same. The result of a run is its archive of nondominated
plans, written plan by plan with every measure the analyses need, whatever the search optimised:

  objective  every_leg  worst-leg safety margin, truncated at CAP           (the model of the paper)
             departure  the same margin at departure only
             centre     distance of the centre of gravity from the centre of the space at departure,
                        mean over the three axes, as in Elahi et al. (2026)
  algorithm  nsga2, nsga3, moead, sms_emoa

Runs of one case with the same run number share their seed, so that searches that differ in
objective or algorithm start from the same population.

Usage, from the code folder:
  python -m experiments.final --name NAME --workers 6 [--objectives ...] [--algorithms ...]
      [--classes 1 ... 7] [--instances 20 ... 29] [--runs 10] [--stops 5] [--per-type 5000]
      -> results/final_NAME.csv, one row per plan of every archive
"""
import argparse
import time
from functools import partial

import numpy as np

from legsafe import (axle_margin, centre_deviation, decode, leg_moments, n_keys, nsga2, rollover_margin,
                     unloading_obstacles, utilisation)
from legsafe.moea import moead, nsga3, sms_emoa

from .common import BETAS, REFERENCE_TARE, TARE, instance, run_parallel, truck
from .e5_pilot import fill_first

VEHICLE = truck(TARE[REFERENCE_TARE], rollover=True)
POP_SIZE = 105       # the number of reference directions for three objectives and 13 divisions
DIVISIONS = 13
PER_TYPE = 5000      # evaluations per box type
CAP = 0.05           # the search sees min(margin, CAP)
PLAIN_START = 0.5    # share of the starting population that leaves no box behind and no free length
FILL_SEEDS = 25      # plans put into the starting population when a search for fill comes first (--fill; not used in the paper)
OBJECTIVES = ("every_leg", "departure", "centre")
ALGORITHMS = {"nsga2": nsga2, "nsga3": nsga3, "moead": moead, "sms_emoa": sms_emoa}


def measures(inst, layout) -> dict:
    """Everything recorded for a plan."""
    lm = leg_moments(layout.mass, *layout.centre.T, layout.rank, inst.n_stops)
    axle, roll = axle_margin(VEHICLE, lm), rollover_margin(VEHICLE, lm)
    per_leg = np.minimum(axle, roll)
    mass = float(layout.mass.sum())
    rd = centre_deviation(VEHICLE, layout.mass, *layout.centre.T) if mass > 0 else (0.5, 0.5, 0.5)
    return dict(
        utilisation=utilisation(inst, layout), obstacles=unloading_obstacles(layout),
        sigma=float(per_leg.min()), sigma_departure=float(per_leg[0]), worst_leg=int(per_leg.argmin()),
        axle=float(axle.min()), axle_departure=float(axle[0]), rollover=float(roll.min()),
        rd_x=rd[0], rd_y=rd[1], rd_z=rd[2], boxes=len(layout.lo), mass=mass,
        cg_height=float(layout.mass @ layout.centre[:, 2] / mass) if mass > 0 else 0.0,  # of the cargo, above the floor
    )


# Other heights for the rollover margin, in metres: (cargo floor, centre of gravity of the empty vehicle).
HEIGHTS = {"low": (1.00, 0.80), "high": (1.30, 1.20)}
OTHER_VEHICLES = {name: truck(TARE[REFERENCE_TARE], rollover=True, deck_height=deck, tare_cg_height=tare)
                  for name, (deck, tare) in HEIGHTS.items()}


def rollover_at_other_heights(inst, layout) -> dict:
    """The lowest rollover margin of a plan when the two assumed heights are lower or higher."""
    lm = leg_moments(layout.mass, *layout.centre.T, layout.rank, inst.n_stops)
    return {f"rollover_{name}": float(rollover_margin(v, lm).min()) for name, v in OTHER_VEHICLES.items()}


def third_objective(m: dict, objective: str) -> float:
    if objective == "every_leg":
        return -min(m["sigma"], CAP)
    if objective == "departure":
        return -min(m["sigma_departure"], CAP)
    return (m["rd_x"] + m["rd_y"] + m["rd_z"]) / 3


def search(inst, rng, objective="every_leg", algorithm="nsga2", per_type=PER_TYPE, fill=0, observer=None):
    """Keys and objectives of the archive of one run.

    `fill` evaluations per box type are spent first on a search for fill alone, whose best plans join the
    starting population. The pilot found that this raises the fill of the fullest plans on large instances
    and lowers the quality of the rest of the front, so the experiments of the paper run without it.
    """
    n_types = len(inst.types)

    def evaluate(keys):
        m = measures(inst, decode(inst, keys))
        return -m["utilisation"], m["obstacles"], third_objective(m, objective)

    generations = max(1, round(per_type * n_types / POP_SIZE))
    initial = np.hstack([rng.random((round(PLAIN_START * POP_SIZE), 2 * n_types)),
                         np.zeros((round(PLAIN_START * POP_SIZE), n_types + 1))])
    used = 0
    if fill:
        first = max(1, round(fill * n_types / POP_SIZE))
        seeds = fill_first(inst, rng, first, POP_SIZE, FILL_SEEDS)
        initial[:len(seeds), :2 * n_types] = seeds
        used = first + 1
    if generations <= used:
        raise ValueError("the budget does not cover the search for fill")
    shifted = None if observer is None else (lambda generation, front, archive: observer(generation + used, front, archive))
    common = dict(initial=initial, observer=shifted)
    if algorithm == "nsga2":
        return nsga2(evaluate, n_keys(inst), POP_SIZE, generations - used, rng, archive=True, **common)
    if algorithm == "sms_emoa":
        return sms_emoa(evaluate, n_keys(inst), POP_SIZE, generations - used, rng, **common)
    return ALGORITHMS[algorithm](evaluate, n_keys(inst), POP_SIZE, generations - used, rng, divisions=DIVISIONS, **common)


def run(task, per_type=PER_TYPE, fill=0, n_runs=10):
    job, stops, number, objective, algorithm = task
    inst, rngs = instance(job, n_runs, stops)
    started = time.perf_counter()
    keys, _ = search(inst, rngs[number], objective, algorithm, per_type, fill)
    seconds = time.perf_counter() - started
    head = dict(br_class=job[0], instance=job[1], alpha=job[2][0], beta=job[2][1], stops=inst.n_stops, types=len(inst.types),
                run=number, objective=objective, algorithm=algorithm, seconds=round(seconds, 2))
    rows = []
    for plan, k in enumerate(keys):
        layout = decode(inst, k)
        rows.append(dict(head, plan=plan, **measures(inst, layout), **rollover_at_other_heights(inst, layout)))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--instances", type=int, nargs=2, default=[20, 29], metavar=("FIRST", "LAST"))
    parser.add_argument("--weights", type=int, nargs="+", default=[0, 1, 2], help="positions in the list of weight settings")
    parser.add_argument("--stops", type=int, nargs="+", default=[0], help="0 for one customer per box type, else that many stops")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--first-run", type=int, default=0,
                        help="run numbers start here; runs below it are left out, so that more runs can be added to a file later")
    parser.add_argument("--objectives", nargs="+", choices=OBJECTIVES, default=["every_leg"])
    parser.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS), default=["nsga2"])
    parser.add_argument("--per-type", type=int, default=PER_TYPE)
    parser.add_argument("--fill", type=int, default=0, help="evaluations per box type of a search for fill that comes first")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    # One whole replicate after the other, the largest classes first within a replicate: a run that is
    # stopped early still leaves complete replicates of every case.
    tasks = [((c, i, BETAS[w]), s or None, n, objective, algorithm)
             for n in range(args.first_run, args.runs) for c in sorted(args.classes, reverse=True) for s in args.stops
             for i in range(args.instances[0], args.instances[1] + 1) for w in args.weights
             for objective in args.objectives for algorithm in args.algorithms]
    run_parallel(partial(run, per_type=args.per_type, fill=args.fill, n_runs=args.runs),
                 tasks, args.workers, f"final_{args.name}.csv")
