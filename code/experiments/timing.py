"""Time of one plan evaluation, split into its parts, on one instance per benchmark class.

The reference implementation (plain Python and NumPy) is timed against the compiled one on the
same key vectors, and the two are checked to give the same objectives.

Usage, from the code folder: python -m experiments.timing [--plans 1000] [--classes 1 3 5 7]
"""
import argparse
import time

import numpy as np

from legsafe import axle_margin, decode, leg_moments, n_keys, unloading_obstacles, utilisation
from legsafe.decoder import decode_reference
from legsafe.legs import leg_moments_reference
from legsafe.objectives import unloading_obstacles_reference

from .common import instance
from .e5_pilot import VEHICLE

INSTANCE, WEIGHTS = 13, (2, 2)
FAST = (decode, unloading_obstacles, leg_moments)
REFERENCE = (decode_reference, unloading_obstacles_reference, leg_moments_reference)


def parts(inst, keys, functions):
    """Seconds spent in each part of evaluating the key vectors, the objectives, and the mean number of boxes."""
    decoder, obstacle_count, moments = functions
    spent = dict(decode=0.0, utilisation=0.0, obstacles=0.0, margin=0.0)
    objectives, boxes = [], 0
    for k in keys:
        t0 = time.perf_counter()
        layout = decoder(inst, k)
        t1 = time.perf_counter()
        filled = utilisation(inst, layout)
        t2 = time.perf_counter()
        obstacles = obstacle_count(layout)
        t3 = time.perf_counter()
        sigma = axle_margin(VEHICLE, moments(layout.mass, *layout.centre.T, layout.rank, inst.n_stops)).min()
        t4 = time.perf_counter()
        for name, seconds in zip(spent, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            spent[name] += seconds
        objectives.append((filled, obstacles, sigma))
        boxes += len(layout.lo)
    return spent, np.array(objectives), boxes / len(keys)


def main(classes, plans):
    print(f"milliseconds per plan, mean over {plans} random key vectors of the extended decoder")
    print(f"{'class':>5} {'types':>5} {'boxes':>6} {'version':>10} {'decode':>8} {'utilis.':>8} {'obstacles':>9} {'margin':>8} {'total':>8} {'speed-up':>9}")
    for br_class in classes:
        inst, _ = instance((br_class, INSTANCE, WEIGHTS), 1)
        keys = np.random.default_rng(br_class).random((plans, n_keys(inst)))
        keys[::2, 2 * len(inst.types):] = 0  # half of the plans load every box that fits
        totals, results = {}, {}
        for version, functions in (("reference", REFERENCE), ("compiled", FAST)):
            parts(inst, keys[:20], functions)  # warm-up
            spent, results[version], boxes = parts(inst, keys, functions)
            ms = {name: 1000 * seconds / plans for name, seconds in spent.items()}
            totals[version] = sum(ms.values())
            speed = f"{totals['reference'] / totals[version]:8.1f}x" if version == "compiled" else ""
            print(f"{br_class:>5} {len(inst.types):>5} {boxes:>6.0f} {version:>10} {ms['decode']:>8.3f} {ms['utilisation']:>8.3f} "
                  f"{ms['obstacles']:>9.3f} {ms['margin']:>8.3f} {totals[version]:>8.3f} {speed:>9}")
        assert np.array_equal(results["reference"], results["compiled"]), "the two versions disagree"
    print("the two versions gave the same objectives for every plan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", type=int, default=1000)
    parser.add_argument("--classes", type=int, nargs="+", default=[1, 3, 5, 7])
    args = parser.parse_args()
    main(args.classes, args.plans)
