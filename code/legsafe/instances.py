from dataclasses import dataclass
from itertools import permutations

import numpy as np


@dataclass(frozen=True)
class BoxType:
    dims: tuple[float, float, float]
    vertical_ok: tuple[bool, bool, bool]
    count: int
    mass: float = 0.0
    rank: int = 1
    stackable: bool = True

    @property
    def volume(self) -> float:
        return self.dims[0] * self.dims[1] * self.dims[2]

    def orientations(self) -> list[tuple[float, float, float]]:
        """Distinct (x, y, z) extents with an admissible dimension vertical."""
        found = []
        for a, b, c in permutations(range(3)):
            if self.vertical_ok[c]:
                o = (self.dims[a], self.dims[b], self.dims[c])
                if o not in found:
                    found.append(o)
        return found


@dataclass(frozen=True)
class Instance:
    name: str
    space: tuple[float, float, float]
    types: tuple[BoxType, ...]
    payload: float = float("inf")

    @property
    def n_stops(self) -> int:
        return max(t.rank for t in self.types)


def read_br(path) -> list[Instance]:
    """Bischoff-Ratcliff file (OR-Library thpack format); centimetres are converted to metres."""
    with open(path) as f:
        tokens = iter(f.read().split())
    instances = []
    for _ in range(int(next(tokens))):
        number, _seed = int(next(tokens)), next(tokens)
        space = tuple(int(next(tokens)) / 100 for _ in range(3))
        types = []
        for _ in range(int(next(tokens))):
            row = [int(next(tokens)) for _ in range(8)]
            types.append(BoxType(
                dims=(row[1] / 100, row[3] / 100, row[5] / 100),
                vertical_ok=(bool(row[2]), bool(row[4]), bool(row[6])),
                count=row[7],
            ))
        instances.append(Instance(f"{number}", space, tuple(types)))
    return instances


def extend(inst: Instance, payload: float, tau: float, alpha: float, beta: float,
           p_nonstackable: float, rng: np.random.Generator) -> Instance:
    """Weights, stackability and delivery ranks as in Algorithm 4 of Elahi et al. (2026); payload in tonnes."""
    volume = inst.space[0] * inst.space[1] * inst.space[2]
    d_mean = payload / (tau * volume)
    d_min = 0.5 * d_mean
    d_max = d_min + (d_mean - d_min) / (alpha / (alpha + beta))

    by_demand = sorted(range(len(inst.types)), key=lambda t: (-inst.types[t].count, -inst.types[t].volume))
    rank = {t: r + 1 for r, t in enumerate(by_demand)}

    types = []
    for t, box in enumerate(inst.types):
        density = d_min + rng.beta(alpha, beta) * (d_max - d_min)
        types.append(BoxType(
            dims=box.dims,
            vertical_ok=box.vertical_ok,
            count=box.count,
            mass=density * box.volume,
            rank=rank[t],
            stackable=rng.random() >= p_nonstackable,
        ))
    return Instance(inst.name, inst.space, tuple(types), payload)


def merge_stops(inst: Instance, stops: int) -> Instance:
    """The same boxes on a route with fewer stops: consecutive stops are merged into `stops` groups
    of nearly equal size, in delivery order, as if several box types went to one customer.

    A plan of the original instance is also a plan of the merged one, with no more unloading
    obstacles and no smaller worst-leg margin: the order of deliveries is kept, and the legs of the
    merged route are legs of the original one.
    """
    before = inst.n_stops
    if not 1 <= stops <= before:
        raise ValueError(f"1 to {before} stops expected")
    types = tuple(BoxType(t.dims, t.vertical_ok, t.count, t.mass, (t.rank * stops + before - 1) // before, t.stackable)
                  for t in inst.types)
    return Instance(inst.name, inst.space, types, inst.payload)
