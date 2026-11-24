"""The compiled kernels of the decoder, the obstacle count and the leg sums are built on first use
and cached on disk. Calling `warm_up` in the parent process builds them once, so that worker
processes load them from the cache instead of each compiling its own copy."""
import numpy as np

from .decoder import decode, n_keys
from .instances import BoxType, Instance
from .legs import leg_moments
from .objectives import unloading_obstacles


def warm_up() -> None:
    inst = Instance("warm-up", (2.0, 1.0, 1.0), (
        BoxType((1.0, 0.5, 0.5), (True, True, True), 3, mass=0.1, rank=1),
        BoxType((0.5, 0.5, 0.5), (False, False, True), 2, mass=0.1, rank=2, stackable=False),
    ), payload=1.0)
    for extended in (False, True):
        layout = decode(inst, np.full(n_keys(inst, extended), 0.25))
        unloading_obstacles(layout)
        leg_moments(layout.mass, *layout.centre.T, layout.rank, inst.n_stops)
