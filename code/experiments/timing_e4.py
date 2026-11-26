"""Run times of the four searches of experiment E4, measured one run at a time.

Each run records its wall-clock time, the processor time of the process, and the share of either
spent inside the evaluation of plans (decoding plus objectives); the rest is the selection of the
search, which is where the four algorithms differ. Run it on an otherwise idle machine; the
processor time is less affected by other jobs than the wall-clock time.

Usage, from the code folder: python -m experiments.timing_e4 [--runs 2]  -> results/timing_e4.csv
"""
import argparse
import csv
import platform
import time

import numpy as np

from legsafe import decode, n_keys
from legsafe.compiled import warm_up
from legsafe.moea import moead, nsga3, sms_emoa
from legsafe.nsga2 import nsga2

from .common import BETAS, OUT, instance
from .final import DIVISIONS, PER_TYPE, PLAIN_START, POP_SIZE, VEHICLE, measures, third_objective

CASES = [(c, 20, BETAS[0]) for c in (1, 3, 5, 7)]
ALGORITHMS = ("nsga2", "nsga3", "moead", "sms_emoa")


def timed_run(job, number, algorithm, n_runs):
    inst, rngs = instance(job, n_runs)
    rng = rngs[number]
    n_types = len(inst.types)
    spent = {"wall": 0.0, "cpu": 0.0, "count": 0}

    def evaluate(keys):
        wall, cpu = time.perf_counter(), time.process_time()
        m = measures(inst, decode(inst, keys), VEHICLE)
        spent["wall"] += time.perf_counter() - wall
        spent["cpu"] += time.process_time() - cpu
        spent["count"] += 1
        return -m["utilisation"], m["obstacles"], third_objective(m, "every_leg")

    generations = max(1, round(PER_TYPE * n_types / POP_SIZE))
    initial = np.hstack([rng.random((round(PLAIN_START * POP_SIZE), 2 * n_types)),
                         np.zeros((round(PLAIN_START * POP_SIZE), n_types + 1))])
    wall, cpu = time.perf_counter(), time.process_time()
    if algorithm == "nsga2":
        _, f = nsga2(evaluate, n_keys(inst), POP_SIZE, generations, rng, initial=initial, archive=True)
    elif algorithm == "sms_emoa":
        _, f = sms_emoa(evaluate, n_keys(inst), POP_SIZE, generations, rng, initial=initial)
    else:
        search = nsga3 if algorithm == "nsga3" else moead
        _, f = search(evaluate, n_keys(inst), POP_SIZE, generations, rng, initial=initial, divisions=DIVISIONS)
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    return dict(br_class=job[0], instance=job[1], alpha=job[2][0], beta=job[2][1], types=n_types, run=number,
                algorithm=algorithm, evaluations=spent["count"], plans=len(f),
                wall=wall, cpu=cpu, evaluation_wall=spent["wall"], evaluation_cpu=spent["cpu"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    warm_up()
    for algorithm in ALGORITHMS:  # compile what each search compiles before anything is timed
        timed_run((1, 20, BETAS[0]), 0, algorithm, args.runs)
    print(f"{platform.processor()}; Python {platform.python_version()}", flush=True)
    rows = []
    with open(OUT / "timing_e4.csv", "w", newline="") as out:
        writer = None
        for number in range(args.runs):
            for job in CASES:
                for algorithm in ALGORITHMS:
                    row = timed_run(job, number, algorithm, args.runs)
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(out, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    out.flush()
                    print(f"BR{job[0]} run {number} {algorithm}: {row['wall']:.1f} s wall, {row['cpu']:.1f} s processor, "
                          f"{row['evaluations']} evaluations", flush=True)
