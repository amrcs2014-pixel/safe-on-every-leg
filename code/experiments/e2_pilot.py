"""Pilot of experiment E2: optimise with the departure-state balance objective of Elahi et al. (2026),
then record the cargo-induced axle loads of the resulting plans on every leg of the route.

Usage, from the code folder: python -m experiments.e2_pilot --classes 1 2 3 4 5 6 7 --instances 10 --workers 8
"""
import argparse
import time

import numpy as np

from legsafe import centre_deviation, decode, leg_moments, nsga2, support_loads, unloading_obstacles, utilisation

from .common import GENERATIONS, POP_SIZE, instance, jobs, run_parallel, truck

# Tare is left out so that the recorded loads are those of the cargo alone; e2_summary adds it.
CARGO_ONLY = truck()


def run(job):
    br_class, index, (alpha, beta) = job
    inst, (rng,) = instance(job, 1)

    def evaluate(keys):
        layout = decode(inst, keys)
        deviation = centre_deviation(CARGO_ONLY, layout.mass, *layout.centre.T)
        return -utilisation(inst, layout), unloading_obstacles(layout), float(np.mean(deviation))

    started = time.perf_counter()
    keys, _ = nsga2(evaluate, 2 * len(inst.types), POP_SIZE, GENERATIONS, rng)
    seconds = time.perf_counter() - started

    rows = []
    for plan, k in enumerate(keys):
        layout = decode(inst, k)
        cx, cy, cz = layout.centre.T
        rdx, rdy, rdz = centre_deviation(CARGO_ONLY, layout.mass, cx, cy, cz)
        front, rear = support_loads(CARGO_ONLY, leg_moments(layout.mass, cx, cy, cz, layout.rank, inst.n_stops))
        rows.append(dict(
            br_class=br_class, instance=index, alpha=alpha, beta=beta, plan=plan,
            stops=inst.n_stops, boxes=len(layout.lo), mass=layout.mass.sum(), seconds=seconds,
            utilisation=utilisation(inst, layout), obstacles=unloading_obstacles(layout),
            rd_x=rdx, rd_y=rdy, rd_z=rdz,
            front_depot=front[0], front_peak=front.max(), front_peak_leg=int(front.argmax()),
            legs_front_rises=int((np.diff(front) > 1e-9).sum()),
            rear_depot=rear[0], rear_peak=rear.max(),
        ))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--instances", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_parallel(run, jobs(args.classes, args.instances), args.workers, "e2_pilot.csv")
