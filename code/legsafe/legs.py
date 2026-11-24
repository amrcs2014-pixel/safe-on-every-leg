from typing import NamedTuple

import numpy as np

from .vehicle import Vehicle

try:
    from numba import njit
except ImportError:  # without numba, the reference implementation is used
    njit = None


class LegMoments(NamedTuple):
    """Mass and first moments of the cargo on board during legs 0..K-1."""

    mass: np.ndarray
    mx: np.ndarray
    my: np.ndarray
    mz: np.ndarray


class LegSafety(NamedTuple):
    sigma: float
    worst_leg: int
    axle: np.ndarray
    rollover: np.ndarray


def leg_moments_reference(mass, cx, cy, cz, rank, n_stops: int) -> LegMoments:
    """Leg k carries the boxes with delivery rank > k; all legs cost O(n + K) via suffix sums over ranks."""
    mass = np.asarray(mass, dtype=float)
    rank = np.asarray(rank, dtype=int)

    def suffix(weights):
        buckets = np.bincount(rank, weights=weights, minlength=n_stops + 1)
        return np.cumsum(buckets[:0:-1])[::-1]

    return LegMoments(
        suffix(mass),
        suffix(mass * np.asarray(cx, dtype=float)),
        suffix(mass * np.asarray(cy, dtype=float)),
        suffix(mass * np.asarray(cz, dtype=float)),
    )


def _suffix_sums(mass, cx, cy, cz, rank, n_stops):
    """The four suffix sums of `leg_moments_reference`, added up in the same order."""
    buckets = np.zeros((4, n_stops + 1))
    for i in range(len(mass)):
        r = rank[i]
        if r < 0 or r > n_stops:
            raise ValueError("delivery rank outside 0..n_stops")
        buckets[0, r] += mass[i]
        buckets[1, r] += mass[i] * cx[i]
        buckets[2, r] += mass[i] * cy[i]
        buckets[3, r] += mass[i] * cz[i]
    sums = np.empty((4, n_stops))
    for q in range(4):
        running = 0.0
        for k in range(n_stops, 0, -1):
            running += buckets[q, k]
            sums[q, k - 1] = running
    return sums


if njit is not None:
    _suffix_sums = njit(cache=True)(_suffix_sums)


def leg_moments(mass, cx, cy, cz, rank, n_stops: int) -> LegMoments:
    """Leg k carries the boxes with delivery rank > k; all legs cost O(n + K) via suffix sums over ranks.

    Returns the values of `leg_moments_reference`.
    """
    if njit is None:
        return leg_moments_reference(mass, cx, cy, cz, rank, n_stops)
    sums = _suffix_sums(np.asarray(mass, dtype=float), np.asarray(cx, dtype=float), np.asarray(cy, dtype=float),
                        np.asarray(cz, dtype=float), np.asarray(rank, dtype=np.int64), n_stops)
    return LegMoments(sums[0], sums[1], sums[2], sums[3])


def support_loads(v: Vehicle, lm: LegMoments):
    front = v.tare_front + (v.a_rear * lm.mass - lm.mx) / v.wheelbase
    rear = v.tare_rear + (lm.mx - v.a_front * lm.mass) / v.wheelbase
    return front, rear


def axle_margin(v: Vehicle, lm: LegMoments) -> np.ndarray:
    """Smallest normalised slack over the vehicle's axle limits on each leg; negative means illegal."""
    front, rear = support_loads(v, lm)
    slacks = [(v.max_front - front) / v.max_front, (v.max_rear - rear) / v.max_rear]
    if v.min_share > 0:
        required = v.min_share * (v.tare + lm.mass)
        carried = rear if v.min_share_support == "rear" else front
        slacks.append((carried - required) / required)
    return np.min(slacks, axis=0)


def rollover_margin(v: Vehicle, lm: LegMoments) -> np.ndarray:
    """Static stability factor of vehicle plus cargo over the design lateral acceleration, minus one."""
    restoring = v.track / 2 * (v.tare + lm.mass) - np.abs(lm.my - v.width / 2 * lm.mass)
    overturning = v.tare * v.tare_cg_height + v.deck_height * lm.mass + lm.mz
    return restoring / overturning / v.lateral_accel - 1


def worst_leg_margin(v: Vehicle, lm: LegMoments) -> LegSafety:
    axle = axle_margin(v, lm)
    rollover = rollover_margin(v, lm)
    per_leg = np.minimum(axle, rollover)
    worst = int(np.argmin(per_leg))
    return LegSafety(float(per_leg[worst]), worst, axle, rollover)


def centre_deviation(v: Vehicle, mass, cx, cy, cz):
    """Depot-state balance as in Elahi et al. (2026): relative centre-of-gravity deviation from the geometric centre, per axis."""
    mass = np.asarray(mass, dtype=float)
    total = mass.sum()
    dims = (v.length, v.width, v.height)
    return tuple(
        abs(float(mass @ np.asarray(c, dtype=float)) / total - d / 2) / d
        for c, d in zip((cx, cy, cz), dims)
    )
