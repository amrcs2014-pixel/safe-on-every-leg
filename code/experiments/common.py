"""Setup shared by the pilot experiments: instances, reference truck and the parallel runner."""
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from legsafe import Vehicle, extend, merge_stops, read_br
from legsafe.compiled import warm_up

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "br" / "orlib"
LARGE = ROOT / "data" / "br" / "or-datasets" / "original"
OUT = ROOT / "results"
RUN = ["br_class", "instance", "alpha", "beta"]

# Rigid truck of the worked example in Krebs and Ehmke (2021), Fig. 5: front axle 0.6 m ahead of the
# cargo space, wheelbase 4.455 m, load capacity 10,100 kg. The BR loading space (5.87 x 2.33 x 2.20 m)
# is taken as the usable part of that truck's 6.04 x 2.5 x 2.5 m body, flush with its front wall.
# Plated axle weights are from the IVECO technical description of the Eurocargo ML150E21 chassis cab
# (26 Nov 2025) and equal the limits used by Krebs and Ehmke.
PAYLOAD = 10.1
MAX_FRONT, MAX_REAR = 5.3, 10.7
GEOMETRY = dict(length=5.87, width=2.33, height=2.20, a_front=-0.6, a_rear=-0.6 + 4.455,
                max_front=MAX_FRONT, max_rear=MAX_REAR)

# Tare loads on the front and rear axle, in tonnes.
TARE = {
    # Convention of the worked example in Krebs and Ehmke (2021): limits apply to the cargo alone.
    "cargo only, as in the published example": (0.0, 0.0),
    # IVECO technical description, ML150E21 chassis cab, wheelbase 4455 mm, day cab: kerb weight
    # 2,993 kg front and 1,543 kg rear, including driver, full tank, tools and spare wheel.
    "chassis cab, manufacturer kerb weights": (2.993, 1.543),
    # Plus a 364 kg body at the middle of the 6.04 m body. That mass is what remains of the 15 t gross
    # weight after the kerb weight and the 10.1 t load capacity stated by Krebs and Ehmke.
    "chassis cab and body": (2.993 + 0.364 * (1 - 3.62 / 4.455), 1.543 + 0.364 * 3.62 / 4.455),
}
REFERENCE_TARE = "chassis cab and body"

# The paper does not report the target utilisation used in its weight generator; 0.9 is ASSUMED.
TAU = 0.9
P_NONSTACKABLE = 0.01
BETAS = [(2, 2), (5, 2), (2, 5)]
POP_SIZE, GENERATIONS = 50, 200


# Data for the rollover margin. Track: rear track of the IVECO technical description (1,818 mm).
# The description gives a frame height of 0.94 to 1.03 m for the empty vehicle; the height of the
# cargo floor (frame, subframe and floor) and the height of the empty vehicle's centre of gravity
# are ASSUMED. The design lateral acceleration is the 4 m/s2 that the calculation method of
# UN Regulation No. 111 sets for tank vehicles; it is a parameter of the study, not a limit that
# applies to a box truck.
ROLLOVER = dict(track=1.818, deck_height=1.15, tare_cg_height=1.0, lateral_accel=4.0 / 9.81)


def truck(tare=(0.0, 0.0), rollover=False, **changes) -> Vehicle:
    data = dict(GEOMETRY, tare_front=tare[0], tare_rear=tare[1], **(ROLLOVER if rollover else {}))
    return Vehicle(**dict(data, **changes))


def br_file(br_class: int) -> Path:
    """Classes 1-7 from the OR-Library files; 8-15 from the OR-Datasets copy, which has the same format
    and whose files for classes 1-7 hold the same numbers as the OR-Library ones."""
    return DATA / f"thpack{br_class}.txt" if br_class <= 7 else LARGE / f"BR{br_class}.txt"


def instance(job, n_seeds: int, stops: int | None = None):
    """The extended instance of a job and independent seeds for the searches run on it.

    With `stops`, the route is shortened to that many stops by merging consecutive ones; boxes,
    weights and seeds stay those of the instance with one customer per box type.
    """
    br_class, index, (alpha, beta) = job
    seeds = np.random.SeedSequence([br_class, index, alpha, beta]).spawn(1 + n_seeds)
    raw = read_br(br_file(br_class))[index]
    inst = extend(raw, PAYLOAD, TAU, alpha, beta, P_NONSTACKABLE, np.random.default_rng(seeds[0]))
    if stops is not None:
        inst = merge_stops(inst, stops)
    return inst, [np.random.default_rng(s) for s in seeds[1:]]


def read(name: str):
    """A results file with its numbers read back exactly. The default parser of pandas is faster
    but can be off by one unit in the last place, which shows when a new run is compared with a file.
    A file that is kept compressed, as NAME.gz, is read from there when NAME itself is absent."""
    import pandas as pd

    path = OUT / name
    if not path.exists() and path.with_name(path.name + ".gz").exists():
        path = path.with_name(path.name + ".gz")
    return pd.read_csv(path, float_precision="round_trip")


def jobs(classes, n_instances):
    return [(c, i, b) for c in classes for i in range(n_instances) for b in BETAS]


def run_parallel(run, job_list, workers: int, out_name: str, initializer=None):
    OUT.mkdir(exist_ok=True)
    warm_up()  # compile once here; the workers then load the compiled code from the cache
    started = time.perf_counter()
    with Pool(workers, initializer=initializer) as pool, open(OUT / out_name, "w", newline="") as f:
        writer = None
        for done, rows in enumerate(pool.imap_unordered(run, job_list), 1):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
            writer.writerows(rows)
            f.flush()
            if done % 10 == 0 or done == len(job_list):
                print(f"{done}/{len(job_list)} runs, {time.perf_counter() - started:.0f} s", flush=True)
