"""Small instances on a 0.5 m grid for the exact model (experiment E1)."""
import numpy as np

from legsafe import BoxType, Instance, extend

from .common import P_NONSTACKABLE, PAYLOAD, TAU

DELTA = 0.5
SPACE = (6.0, 2.0, 1.0)


def small_instance(seed: int, n_types: int, alpha: float = 2, beta: float = 2, space=SPACE) -> Instance:
    """Random box types with sides of 0.5-1.5 m, added until their volume would exceed the space."""
    rng = np.random.default_rng([seed, n_types])
    types = []
    for _ in range(n_types):
        dims = (DELTA * rng.integers(1, 4), DELTA * rng.integers(1, 3), DELTA * rng.integers(1, 3))
        vertical = (bool(rng.integers(2)), bool(rng.integers(2)), True)
        types.append([dims, vertical, 0])
    volume_left = np.prod(space)
    while True:
        t = rng.integers(n_types)
        volume = np.prod(types[t][0])
        if volume > volume_left:
            break
        types[t][2] += 1
        volume_left -= volume
    raw = Instance(f"small-{seed}-{n_types}", tuple(space),
                   tuple(BoxType(d, v, c) for d, v, c in types if c > 0))
    return extend(raw, PAYLOAD, TAU, alpha, beta, P_NONSTACKABLE, rng)
