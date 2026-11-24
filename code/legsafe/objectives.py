import numpy as np

from .decoder import Layout, njit
from .instances import Instance

EPS = 1e-9


def utilisation(inst: Instance, layout: Layout) -> float:
    return float(layout.size.prod(axis=1).sum() / np.prod(inst.space))


def unloading_obstacles_reference(layout: Layout) -> int:
    """Pairs (earlier box, later box) where the later one is between the earlier one and the door, or on top of it.

    Follows Elahi et al. (2026), equations 26-47; the door is at the high-x end.
    """
    lo, hi = layout.lo, layout.lo + layout.size
    later = layout.rank[None, :] > layout.rank[:, None]

    def overlap(axis):
        return (lo[None, :, axis] < hi[:, None, axis] - EPS) & (lo[:, None, axis] < hi[None, :, axis] - EPS)

    in_front = (lo[None, :, 0] >= hi[:, None, 0] - EPS) & overlap(1) & overlap(2)
    on_top = (lo[None, :, 2] >= hi[:, None, 2] - EPS) & overlap(0) & overlap(1)
    return int((later & (in_front | on_top)).sum())


def _count(lo, hi, rank):
    """The count of `unloading_obstacles_reference`, pair by pair."""
    total = 0
    for i in range(len(rank)):
        for j in range(len(rank)):
            if rank[j] <= rank[i]:
                continue
            if not (lo[j, 1] < hi[i, 1] - EPS and lo[i, 1] < hi[j, 1] - EPS):
                continue  # both cases need an overlap across the width
            over_x = lo[j, 0] < hi[i, 0] - EPS and lo[i, 0] < hi[j, 0] - EPS
            over_z = lo[j, 2] < hi[i, 2] - EPS and lo[i, 2] < hi[j, 2] - EPS
            in_front = lo[j, 0] >= hi[i, 0] - EPS and over_z
            on_top = lo[j, 2] >= hi[i, 2] - EPS and over_x
            if in_front or on_top:
                total += 1
    return total


if njit is not None:
    _count = njit(cache=True)(_count)


def unloading_obstacles(layout: Layout) -> int:
    """Pairs (earlier box, later box) where the later one is between the earlier one and the door, or on top of it.

    Follows Elahi et al. (2026), equations 26-47; the door is at the high-x end. Returns the count
    of `unloading_obstacles_reference`.
    """
    if njit is None:
        return unloading_obstacles_reference(layout)
    return int(_count(layout.lo, layout.lo + layout.size, layout.rank))
