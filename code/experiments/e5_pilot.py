"""Pilot of experiment E5: search with the worst-leg axle margin as the third objective, on the
instances of the E2 pilot, to measure what safety on every leg costs in utilisation.

The rollover margin is not included yet: the truck's centre-of-gravity height is not known.

Usage, from the code folder: python -m experiments.e5_pilot --workers 8 --config NAME
  untuned        first search settings, plain decoder        -> results/e5_pilot.csv
  tuned          tuned search, plain decoder                 -> results/e5_tuned.csv
  extended       tuned search, extended decoder              -> results/e5_extended.csv
  extended-long  the same with twice the evaluations         -> results/e5_extended_long.csv
  archived       extended, reporting the search's archive    -> results/e5_archived.csv
  distinct       archived, with copies ranked last           -> results/e5_distinct.csv
  one-key        archived, one key mutated per child         -> results/e5_one_key.csv
  seeded         archived, started from a search for fill    -> results/e5_seeded.csv
"""
import argparse
import time
from functools import partial

import numpy as np

from legsafe import axle_margin, decode, leg_moments, n_keys, nsga2, rollover_margin, support_loads, unloading_obstacles, utilisation

from .common import GENERATIONS, POP_SIZE, REFERENCE_TARE, TARE, instance, jobs, run_parallel, truck

VEHICLE = truck(TARE[REFERENCE_TARE])
CARGO_ONLY = truck()

# cap: the search sees min(sigma, cap), so that loading less cargo is not rewarded beyond that margin.
# seeded: share of the starting population ordered so that later deliveries are placed first.
UNTUNED = dict(cap=None, seeded=0.0, pop_size=POP_SIZE, generations=GENERATIONS)
# Chosen on held-out instances by experiments.tune_e5; same number of evaluations as UNTUNED.
TUNED = dict(cap=0.05, seeded=0.0, pop_size=2 * POP_SIZE, generations=GENERATIONS // 2)
# extended: the decoder may leave boxes behind and stand clear of the front wall (3T + 1 keys).
# plain_start: share of the starting population that does neither, as the plain decoder would.
EXTENDED = dict(TUNED, extended=True, plain_start=0.5)
EXTENDED_LONG = dict(EXTENDED, generations=2 * EXTENDED["generations"])
# archive: report every nondominated plan met during the search, not only the final population's.
ARCHIVED = dict(EXTENDED, archive=True)
# distinct: members with the objectives of an earlier member are ranked last in the population.
DISTINCT = dict(ARCHIVED, distinct=True)
# mutation_keys: mean number of keys resampled in a child. Without it the rate per key is
# max(1 / keys, 0.05), which is one key per child up to 20 keys and 5% of the keys beyond.
ONE_KEY = dict(ARCHIVED, mutation_keys=1.0)
# fill_first: evaluations per box type spent first on a search for fill alone (plain decoder);
# fill_seeds: number of its best plans that start the three-objective search. Same total budget.
SEEDED = dict(ARCHIVED, fill_first=1000, fill_seeds=25)
# The same for both ends of the front, with shorter first parts and fewer seeds: a search for fill
# and a search for the fewest unloading obstacles (access_first, access_seeds).
TWO_ENDS = dict(ARCHIVED, fill_first=500, fill_seeds=10, access_first=500, access_seeds=10)
CONFIGS = {
    "untuned": (UNTUNED, "e5_pilot.csv"),
    "tuned": (TUNED, "e5_tuned.csv"),
    "extended": (EXTENDED, "e5_extended.csv"),
    "extended-long": (EXTENDED_LONG, "e5_extended_long.csv"),
    "archived": (ARCHIVED, "e5_archived.csv"),
    "distinct": (DISTINCT, "e5_distinct.csv"),
    "one-key": (ONE_KEY, "e5_one_key.csv"),
    "seeded": (SEEDED, "e5_seeded.csv"),
    "two-ends": (TWO_ENDS, "e5_two_ends.csv"),
}


def legs(inst, layout):
    return leg_moments(layout.mass, *layout.centre.T, layout.rank, inst.n_stops)


def delivery_order_seeds(inst, rng, count: int) -> np.ndarray:
    """Key vectors that place later deliveries first, with random orientation keys."""
    rank = np.array([t.rank for t in inst.types])
    order = 1.0 - rank / (rank.max() + 1)
    return np.hstack([np.tile(order, (count, 1)), rng.random((count, len(rank)))])


def starting_members(inst, rng, config):
    """Members that replace the head of the random starting population, or None."""
    n_types = len(inst.types)
    extended = config.get("extended", False)
    head = []
    n_seeded = round(config["seeded"] * config["pop_size"])
    if n_seeded:
        head.append(delivery_order_seeds(inst, rng, n_seeded))
    n_plain = round(config.get("plain_start", 0.0) * config["pop_size"]) if extended else 0
    if n_plain:
        head.append(rng.random((n_plain, 2 * n_types)))
    if not head:
        return None
    head = np.vstack(head)
    return np.hstack([head, np.zeros((len(head), n_types + 1))]) if extended else head


def best_first(inst, rng, generations: int, pop_size: int, count: int, goal: str = "fill",
               extended: bool = False, offset_step=None) -> np.ndarray:
    """Keys of the `count` best plans, each with a different score, met by a search with one goal.

    goal 'fill'    the fullest plans;
    goal 'access'  the plans with the fewest unloading obstacles and, among those, the fullest.

    The search is the NSGA-II with one objective; members with the score of an earlier member are
    ranked last, which is what keeps a one-objective population apart. With the extended decoder,
    half of its starting population leaves no box behind and no free length, as in the main search.
    """
    best = {}  # score -> keys, the `count` lowest scores met

    def evaluate(keys):
        layout = decode(inst, keys, offset_step)
        fill = utilisation(inst, layout)
        score = -fill if goal == "fill" else unloading_obstacles(layout) - fill  # fill is below one obstacle
        if score not in best and (len(best) < count or score < max(best)):
            best[score] = keys.copy()
            if len(best) > count:
                del best[max(best)]
        return (score,)

    n_types = len(inst.types)
    initial = np.hstack([rng.random((pop_size // 2, 2 * n_types)), np.zeros((pop_size // 2, n_types + 1))]) if extended else None
    nsga2(evaluate, n_keys(inst, extended), pop_size, generations, rng, initial=initial, distinct=True)
    return np.array([best[score] for score in sorted(best)])


def fill_first(inst, rng, generations: int, pop_size: int, count: int) -> np.ndarray:
    """Plain keys of the `count` fullest plans, each with a different fill, of a search for fill alone."""
    return best_first(inst, rng, generations, pop_size, count, "fill")


def margin_of(inst, layout, vehicle=None) -> float:
    """Worst margin over the legs: axle loads on VEHICLE, or axle loads and rollover on `vehicle`."""
    lm = legs(inst, layout)
    if vehicle is None:
        return float(axle_margin(VEHICLE, lm).min())
    return float(min(axle_margin(vehicle, lm).min(), rollover_margin(vehicle, lm).min()))


def search(inst, rng, config, offset_step=None, observer=None, vehicle=None):
    """The three-objective search of a configuration; returns the keys of its result.

    With `fill_first` (evaluations per box type) the search starts with a search for fill alone and
    puts its `fill_seeds` best plans into the starting population; with `access_first` and
    `access_seeds` it does the same with a search for the fewest unloading obstacles. The evaluations
    of these first parts come out of the budget, and the observer counts them: it is called with the
    number of the generation at which a search without first parts has used as many evaluations.

    The margin is that of the axle loads on VEHICLE, or, with `vehicle`, the smaller of its axle and
    rollover margins.
    """
    cap = config["cap"]
    pop, generations = config["pop_size"], config["generations"]
    extended = config.get("extended", False)
    n_types = len(inst.types)

    def evaluate(keys):
        layout = decode(inst, keys, offset_step)
        sigma = margin_of(inst, layout, vehicle)
        return -utilisation(inst, layout), unloading_obstacles(layout), -(sigma if cap is None else min(sigma, cap))

    initial = starting_members(inst, rng, config)
    used = 0  # generations' worth of evaluations spent before the three-objective search starts
    seeds = []
    for goal, wide in (("fill", False), ("access", extended)):
        if config.get(f"{goal}_first"):
            first = max(1, round(config[f"{goal}_first"] * n_types / pop))
            found = best_first(inst, rng, first, pop, config[f"{goal}_seeds"], goal, wide, offset_step)
            seeds.append(np.hstack([found, np.zeros((len(found), n_types + 1))]) if extended and not wide else found)
            used += first + 1
    if seeds:
        seeds = np.vstack(seeds)
        if initial is None:
            initial = seeds
        else:
            initial[:len(seeds)] = seeds  # the seeds take the place of random members of the plain half
        if generations <= used:
            raise ValueError("the budget does not cover the searches that come first")

    n = n_keys(inst, extended)
    changed = config.get("mutation_keys")
    shifted = None if observer is None else (lambda generation, front, archive: observer(generation + used, front, archive))
    keys, _ = nsga2(evaluate, n, pop, generations - used, rng,
                    mutation_prob=None if changed is None else changed / n,
                    initial=initial, observer=shifted,
                    archive=config.get("archive", False), distinct=config.get("distinct", False))
    return keys


def run(job, config=UNTUNED):
    br_class, index, (alpha, beta) = job
    inst, (_, rng) = instance(job, 2)  # the first seed belongs to the E2 search

    started = time.perf_counter()
    keys = search(inst, rng, config)
    seconds = time.perf_counter() - started

    rows = []
    for plan, k in enumerate(keys):
        layout = decode(inst, k)
        lm = legs(inst, layout)
        margin = axle_margin(VEHICLE, lm)
        front, rear = support_loads(CARGO_ONLY, lm)
        rows.append(dict(
            br_class=br_class, instance=index, alpha=alpha, beta=beta, plan=plan,
            stops=inst.n_stops, boxes=len(layout.lo), mass=layout.mass.sum(), seconds=seconds,
            utilisation=utilisation(inst, layout), obstacles=unloading_obstacles(layout),
            sigma=margin.min(), worst_leg=int(margin.argmin()), depot_margin=margin[0],
            front_depot=front[0], front_peak=front.max(), rear_depot=rear[0], rear_peak=rear.max(),
        ))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", choices=list(CONFIGS), default="untuned")
    args = parser.parse_args()
    config, out = CONFIGS[args.config]
    run_parallel(partial(run, config=config), jobs(args.classes, args.instances), args.workers, out)
