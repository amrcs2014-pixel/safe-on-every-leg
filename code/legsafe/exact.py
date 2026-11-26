from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .decoder import Layout
from .instances import Instance
from .vehicle import Vehicle


class Solution(NamedTuple):
    layout: Layout
    volume: int
    obstacles: int
    proven: bool


class GridModel:
    """Exact model on a grid of step `delta`, after Junqueira et al. (2012).

    One binary per (box type, orientation, grid position). Boxes rest with their whole base on the
    floor or on stackable boxes that leave no earlier, so a later box is never above an earlier one
    and unloading obstacles are only later boxes between an earlier box and the door.
    """

    def __init__(self, inst: Instance, vehicle: Vehicle, delta: float):
        def units(value):
            scaled = value / delta
            assert abs(scaled - round(scaled)) < 1e-9, "dimensions must be multiples of delta"
            return int(round(scaled))

        self.inst, self.vehicle, self.delta = inst, vehicle, delta
        self.grid = tuple(units(s) for s in inst.space)
        kind, lo, size = [], [], []
        for t, box in enumerate(inst.types):
            for orientation in box.orientations():
                l, w, h = (units(d) for d in orientation)
                for x in range(self.grid[0] - l + 1):
                    for y in range(self.grid[1] - w + 1):
                        for z in range(self.grid[2] - h + 1):
                            kind.append(t)
                            lo.append((x, y, z))
                            size.append((l, w, h))
        self.kind, self.lo, self.size = np.array(kind), np.array(lo), np.array(size)
        self.hi = self.lo + self.size
        self.n = len(kind)
        self.rank = np.array([t.rank for t in inst.types])[self.kind]
        self.mass = np.array([t.mass for t in inst.types])[self.kind]
        self.stackable = np.array([t.stackable for t in inst.types])[self.kind]
        self.volume = self.size.prod(axis=1)
        self.centre_x = (self.lo[:, 0] + self.size[:, 0] / 2) * delta
        self.centre_y = (self.lo[:, 1] + self.size[:, 1] / 2) * delta
        self.centre_z = (self.lo[:, 2] + self.size[:, 2] / 2) * delta
        self.index = {(k, *l, *s): i for i, (k, l, s) in enumerate(zip(kind, lo, size))}
        self.max_pairs = sum(t.count for t in inst.types) ** 2
        self.hard = self._hard_rows()
        self.blocking = self._blocking_rows()

    # --- constraint blocks: (matrix over [x, c], lower bounds, upper bounds) --------------------

    def _overlap(self, a: int, axis: int) -> np.ndarray:
        return np.clip(np.minimum(self.hi[:, axis], self.hi[a, axis]) - np.maximum(self.lo[:, axis], self.lo[a, axis]), 0, None)

    def _hard_rows(self):
        n, (gx, gy, gz) = self.n, self.grid
        rows, cols, vals, lower, upper = [], [], [], [], []

        def add(columns, values, lb, ub):
            rows.extend([len(lower)] * len(columns))
            cols.extend(columns)
            vals.extend(values)
            lower.append(lb)
            upper.append(ub)

        for t, box in enumerate(self.inst.types):
            members = np.flatnonzero(self.kind == t)
            add(members, np.ones(len(members)), 0, box.count)

        cells = [[] for _ in range(gx * gy * gz)]
        for a in range(n):
            (x0, y0, z0), (x1, y1, z1) = self.lo[a], self.hi[a]
            for x in range(x0, x1):
                for y in range(y0, y1):
                    for z in range(z0, z1):
                        cells[(x * gy + y) * gz + z].append(a)
        for members in cells:
            if len(members) > 1:
                add(members, np.ones(len(members)), 0, 1)

        for a in np.flatnonzero(self.lo[:, 2] > 0):
            area = self._overlap(a, 0) * self._overlap(a, 1)
            below = (self.hi[:, 2] == self.lo[a, 2]) & (area > 0) & self.stackable & (self.rank >= self.rank[a])
            supporters = np.flatnonzero(below)
            add([*supporters, a], [*area[supporters], -self.size[a, 0] * self.size[a, 1]], 0, np.inf)

        add(range(n), self.mass, 0, self.inst.payload)
        matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lower), 2 * n))
        return matrix, np.array(lower, float), np.array(upper, float)

    def _blocking_rows(self):
        """c_a >= (later boxes between a and the door) when a is placed."""
        n = self.n
        rows, cols, vals, lower = [], [], [], []
        self.counted = np.zeros(n, bool)
        for a in range(n):
            later = self.rank > self.rank[a]
            if not later.any():
                continue
            blockers = np.flatnonzero(later & (self.lo[:, 0] >= self.hi[a, 0])
                                      & (self._overlap(a, 1) > 0) & (self._overlap(a, 2) > 0))
            if not len(blockers):
                continue
            big = sum(t.count for t in self.inst.types if t.rank > self.rank[a])
            row = len(lower)
            rows.extend([row] * (len(blockers) + 2))
            cols.extend([n + a, a, *blockers])
            vals.extend([1.0, -big, *(-np.ones(len(blockers)))])
            lower.append(-big)
            self.counted[a] = True
        matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lower), 2 * n))
        return matrix, np.array(lower, float), np.full(len(lower), np.inf)

    def _axle_rows(self, margin: float):
        v, n = self.vehicle, self.n
        front = self.mass * (v.a_rear - self.centre_x) / v.wheelbase
        rear = self.mass * (self.centre_x - v.a_front) / v.wheelbase
        rows, upper = [], []
        for k in range(self.inst.n_stops):
            on_board = self.rank > k
            rows += [np.where(on_board, front, 0.0), np.where(on_board, rear, 0.0)]
            upper += [(1 - margin) * v.max_front - v.tare_front, (1 - margin) * v.max_rear - v.tare_rear]
        matrix = sparse.hstack([sparse.csr_matrix(np.array(rows)), sparse.csr_matrix((len(rows), n))]).tocsr()
        return matrix, np.full(len(rows), -np.inf), np.array(upper)

    def _rollover_rows(self, margin: float):
        """rho_k >= margin on every leg, as two linear rows per leg (one per side of the vehicle).

        rho_k >= margin  <=>  t/2 (T + M) -+ (mu_y - W/2 M) >= (1 + margin) a_y (T h_T + h_d M + mu_z),
        with the vehicle's centre line at W/2 of the cargo space.
        """
        v, n = self.vehicle, self.n
        factor = (1 + margin) * v.lateral_accel
        lateral = self.mass * (self.centre_y - v.width / 2)
        common = self.mass * (v.track / 2 - factor * (v.deck_height + self.centre_z))
        bound = factor * v.tare * v.tare_cg_height - v.track / 2 * v.tare
        rows = []
        for k in range(self.inst.n_stops):
            on_board = self.rank > k
            for side in (1.0, -1.0):
                rows.append(np.where(on_board, common - side * lateral, 0.0))
        matrix = sparse.hstack([sparse.csr_matrix(np.array(rows)), sparse.csr_matrix((len(rows), n))]).tocsr()
        return matrix, np.full(len(rows), bound), np.full(len(rows), np.inf)

    def _margin_rows(self, margin: float):
        """The rows that impose sigma >= margin: axle limits, and rollover when the vehicle has its data."""
        blocks = [self._axle_rows(margin)]
        if self.vehicle.track is not None:
            blocks.append(self._rollover_rows(margin))
        return blocks

    # --- solving -------------------------------------------------------------------------------

    def solve(self, margin: float | None = None, max_obstacles: int | None = None, time_limit: float = 120.0):
        """Most volume, then fewest obstacles, with sigma >= margin and obstacles <= max_obstacles.

        Returns (solution or None, closed). `closed` is True when the solver proved optimality or
        infeasibility; a None solution with closed False means the time limit ran out first.
        """
        n = self.n
        blocks = [self.hard, self.blocking]
        if margin is not None:
            blocks.extend(self._margin_rows(margin))
        if max_obstacles is not None:
            total = sparse.csr_matrix((np.ones(n), (np.zeros(n, int), n + np.arange(n))), shape=(1, 2 * n))
            blocks.append((total, np.array([0.0]), np.array([float(max_obstacles)])))
        matrix = sparse.vstack([b[0] for b in blocks]).tocsr()
        lower, upper = np.concatenate([b[1] for b in blocks]), np.concatenate([b[2] for b in blocks])

        weight = self.max_pairs + 1  # one unit of volume outweighs any number of obstacles
        cost = np.concatenate([-weight * self.volume.astype(float), np.ones(n)])
        bounds = Bounds(np.zeros(2 * n), np.concatenate([np.ones(n), np.where(self.counted, np.inf, 0.0)]))
        result = milp(cost, constraints=LinearConstraint(matrix, lower, upper),
                      integrality=np.concatenate([np.ones(n), np.zeros(n)]), bounds=bounds,
                      options={"time_limit": time_limit, "mip_rel_gap": 0.0})
        closed = result.status in (0, 2)
        if result.x is None:
            return None, closed
        chosen = np.flatnonzero(result.x[:n] > 0.5)
        return Solution(self.layout(chosen), int(self.volume[chosen].sum()),
                        int(round(result.x[n:].sum())), result.status == 0), closed

    def front(self, margin: float | None = None, time_limit: float = 120.0):
        """Nondominated (volume, obstacles) plans with sigma >= margin, by the epsilon-constraint method.

        Returns (points, closed); the front is exact only when closed is True.
        """
        points, bound = [], None
        while True:
            solution, closed = self.solve(margin, bound, time_limit)
            if solution is not None:
                points.append(solution)
            if not closed:
                return points, False
            if solution is None or solution.obstacles == 0:
                return points, True
            bound = solution.obstacles - 1

    # --- conversions ---------------------------------------------------------------------------

    def layout(self, chosen) -> Layout:
        return Layout(self.lo[chosen] * self.delta, self.size[chosen] * self.delta,
                      self.kind[chosen], self.rank[chosen], self.mass[chosen])

    def violation(self, layout: Layout, margin: float | None = None) -> float:
        """Largest violation of the model's constraints by a layout; zero means the layout is feasible."""
        x = np.zeros(2 * self.n)
        for k, lo, size in zip(layout.type, layout.lo, layout.size):
            key = (int(k), *(int(round(v / self.delta)) for v in lo), *(int(round(v / self.delta)) for v in size))
            x[self.index[key]] += 1
        blocks = [self.hard] + (self._margin_rows(margin) if margin is not None else [])
        worst = max(0.0, float(x.max()) - 1)
        for matrix, lower, upper in blocks:
            value = matrix @ x
            worst = max(worst, float(np.max(lower - value, initial=0)), float(np.max(value - upper, initial=0)))
        return worst
