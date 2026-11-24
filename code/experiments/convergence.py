"""Convergence of the every-leg search with the extended decoder: how its plans improve with the
number of evaluations, on instances that no pilot and no tuning run has used (indices 13 and 14).

One long run per case and seed is recorded at checkpoints, so every smaller budget is the same
run stopped earlier. Each checkpoint is scored twice: on the first front of the population, which
is what a search without an archive returns, and on the archive of all nondominated plans met so
far. The archive only records, so both scores belong to the same search.

Usage, from the code folder:
  python -m experiments.convergence --workers 8               -> results/convergence.csv
      classes 1, 3, 5, 7; the same budgets for every class, up to 80,100 evaluations
  python -m experiments.convergence --workers 8 --grid types  -> results/convergence_types.csv
      classes 1, 3, 5, 7, 10, 15; budgets per box type, up to 16,000 evaluations per type
Add --variant NAME for another search on the same cases and seeds; the file name then ends in _NAME.csv:
  distinct  copies are ranked last in the population
  one_key   one key is mutated per child on average, however many keys a plan has
  seeded    the first 1,000 evaluations per box type search for fill alone and seed the rest
  two_ends  500 evaluations per box type search for fill, 500 for the fewest obstacles; both seed the rest
Add --up-to N to stop at N evaluations per box type. Add --stops N [N ...] for the same boxes on routes shortened to N stops by merging consecutive stops;
the file name then ends in _stops.csv and has a column with the number of stops.
"""
import argparse
import time
from functools import partial

import numpy as np

from .common import BETAS, instance, run_parallel
from .e5_pilot import ARCHIVED, DISTINCT, ONE_KEY, SEEDED, TWO_ENDS, search
from .tune_e5 import dominated_area

INSTANCES, RUNS = [13, 14], 3
CLASSES, CHECKPOINTS = [1, 3, 5, 7], [25, 50, 100, 200, 400, 800]  # generations
TYPE_CLASSES, PER_TYPE = [1, 3, 5, 7, 10, 15], [125, 250, 500, 1000, 2000, 4000, 8000, 16000]  # evaluations per box type
CONFIG = dict(ARCHIVED, generations=CHECKPOINTS[-1])
VARIANTS = {"distinct": DISTINCT, "one_key": ONE_KEY, "seeded": SEEDED, "two_ends": TWO_ENDS}


def scores(f: np.ndarray, cap: float) -> dict:
    """Scores of a nondominated set, from the objectives the search sees."""
    util, obstacles, sigma = -f[:, 0], f[:, 1], -f[:, 2]
    legal, margin = sigma >= -1e-12, sigma >= cap - 1e-9

    def best(mask):
        return util[mask].max() if mask.any() else np.nan

    return dict(
        plans=len(f), distinct=len(np.unique(f, axis=0)), best_any=util.max(),
        best_legal=best(legal), best_legal_few=best(legal & (obstacles <= 20)), best_legal_free=best(legal & (obstacles == 0)),
        best_margin=best(margin), best_margin_few=best(margin & (obstacles <= 20)), best_margin_free=best(margin & (obstacles == 0)),
        area_legal=dominated_area(util[legal], obstacles[legal]), area_margin=dominated_area(util[margin], obstacles[margin]),
    )


def run(task, per_type=False, variant=None, budgets=PER_TYPE):
    job, number, stops = task
    inst, rngs = instance(job, RUNS, stops)
    pop = CONFIG["pop_size"]
    if per_type:  # generation at which each budget per box type is reached
        checkpoints = {max(1, round(budget * len(inst.types) / pop)): budget for budget in budgets}
    else:
        checkpoints = {generation: None for generation in CHECKPOINTS}
    config = dict(VARIANTS[variant] if variant else CONFIG, generations=max(checkpoints))
    rows = []
    started = time.perf_counter()

    def observer(generation, front, archive):
        if generation in checkpoints:
            for source, f in (("population", front), ("archive", archive)):
                row = dict(br_class=job[0], instance=job[1], alpha=job[2][0], beta=job[2][1], run=number, types=len(inst.types),
                           generation=generation, evaluations=pop * (generation + 1), source=source)
                if per_type:
                    row["per_type"] = checkpoints[generation]
                if stops is not None:
                    row["stops"] = inst.n_stops
                rows.append(dict(**row, seconds=time.perf_counter() - started, **scores(f, config["cap"])))

    search(inst, rngs[number], config, observer=observer)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grid", choices=["budget", "types"], default="budget")
    parser.add_argument("--variant", choices=list(VARIANTS), help="another search on the same cases and seeds")
    parser.add_argument("--classes", type=int, nargs="+")
    parser.add_argument("--stops", type=int, nargs="+", help="routes shortened to these numbers of stops")
    parser.add_argument("--up-to", type=int, default=PER_TYPE[-1], help="grid 'types': largest budget per box type")
    args = parser.parse_args()
    per_type = args.grid == "types"
    classes = args.classes or (TYPE_CLASSES if per_type else CLASSES)
    tasks = [((c, i, b), n, s) for c in sorted(classes, reverse=True) for s in args.stops or [None]
             for i in INSTANCES for b in BETAS[::2] for n in range(RUNS)]
    name = ("convergence" + ("_types" if per_type else "") + (f"_{args.variant}" if args.variant else "")
            + ("_stops" if args.stops else "") + ".csv")
    budgets = [b for b in PER_TYPE if b <= args.up_to]
    run_parallel(partial(run, per_type=per_type, variant=args.variant, budgets=budgets), tasks, args.workers, name)
