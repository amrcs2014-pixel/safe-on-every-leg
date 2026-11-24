"""The decoder with fill as the only goal, on two kinds of instance.

plain     The container loading problem of the literature: the Bischoff-Ratcliff files as published,
          one stop, no weights. Published results for these classes are for this problem, so the run
          shows how the block-building decoder with the random-key search compares with dedicated
          loading heuristics. Every box is fully supported, the variant reported as 'with full support'.
weighted  The instances of the other experiments (weights, payload limit, stops), with the unloading
          obstacles and the safety margin ignored. Set against the three-objective search on the same
          instances, this shows how far that search is from the fill the decoder can reach.

Both use the plain decoder (2T keys). The search is the NSGA-II of the other experiments with one
objective. With one objective it has no crowding to keep the population apart, so members with the
fill of an earlier member are ranked last (`distinct`); `--keep-copies` runs it without that.

Usage, from the code folder:
  python -m experiments.plain --workers 6 [--classes 1 ... 15] [--instances 10] [--first 0]
      -> results/plain.csv, one row per instance and checkpoint
  python -m experiments.plain --workers 6 --setting weighted --classes 10 15 [--stops 5]
      -> results/plain_weighted.csv; the cases and seeds of the convergence study
"""
import argparse
import time
from functools import partial

import numpy as np

from legsafe import decode, n_keys, nsga2, read_br, utilisation

from .common import BETAS, br_file, instance, run_parallel

POP_SIZE = 100
PER_TYPE = [125, 250, 500, 1000, 2000, 4000]  # evaluations per box type
WEIGHTED_INSTANCES = [13, 14]  # as in experiments.convergence


def fill_search(inst, rng, per_type, distinct, row):
    """Rows with the best fill at each budget, and the number of boxes of the final plan."""
    checkpoints = {max(1, round(budget * len(inst.types) / POP_SIZE)): budget for budget in per_type}
    rows = []
    started = time.perf_counter()

    def evaluate(keys):
        return (-utilisation(inst, decode(inst, keys)),)

    def observer(generation, front, archive):
        if generation in checkpoints:
            rows.append(dict(row, types=len(inst.types), boxes=sum(t.count for t in inst.types),
                             per_type=checkpoints[generation], evaluations=POP_SIZE * (generation + 1),
                             seconds=time.perf_counter() - started, fill=-front[:, 0].min()))

    keys, f = nsga2(evaluate, n_keys(inst, extended=False), POP_SIZE, max(checkpoints), rng, observer=observer, distinct=distinct)
    loaded = len(decode(inst, keys[np.argmin(f[:, 0])]).lo)
    return [dict(r, loaded=loaded) for r in rows]


def run(task, distinct=True, per_type=PER_TYPE):
    br_class, index = task
    inst = read_br(br_file(br_class))[index]
    rng = np.random.default_rng(np.random.SeedSequence([br_class, index, 2026]))
    return fill_search(inst, rng, per_type, distinct, dict(br_class=br_class, instance=index))


def run_weighted(task, distinct=True, per_type=PER_TYPE):
    job, stops = task
    inst, rngs = instance(job, 1, stops)
    row = dict(br_class=job[0], instance=job[1], alpha=job[2][0], beta=job[2][1], stops=inst.n_stops)
    return fill_search(inst, rngs[0], per_type, distinct, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["plain", "weighted"], default="plain")
    parser.add_argument("--classes", type=int, nargs="+")
    parser.add_argument("--instances", type=int, default=10, help="plain: number of instances per class")
    parser.add_argument("--first", type=int, default=0, help="plain: index of the first instance of each class")
    parser.add_argument("--stops", type=int, nargs="+", help="weighted: routes shortened to these numbers of stops, besides the full route")
    parser.add_argument("--budgets", type=int, nargs="+", default=PER_TYPE, help="evaluations per box type")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-copies", action="store_true", help="do not rank members with equal fill last")
    parser.add_argument("--out")
    args = parser.parse_args()
    common = dict(distinct=not args.keep_copies, per_type=sorted(args.budgets))
    if args.setting == "plain":
        classes = args.classes or list(range(1, 16))
        tasks = [(c, i) for c in sorted(classes, reverse=True) for i in range(args.first, args.first + args.instances)]
        run_parallel(partial(run, **common), tasks, args.workers, args.out or "plain.csv")
    else:
        classes = args.classes or [10, 15]
        tasks = [((c, i, b), s) for c in sorted(classes, reverse=True) for s in [None] + (args.stops or [])
                 for i in WEIGHTED_INSTANCES for b in BETAS[::2]]
        run_parallel(partial(run_weighted, **common), tasks, args.workers, args.out or "plain_weighted.csv")
