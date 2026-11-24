"""Three further multi-objective searches on random keys, for the comparison with NSGA-II.

NSGA-III (Deb and Jain 2014), MOEA/D with the Tchebycheff function (Zhang and Li 2007) and
SMS-EMOA (Beume, Naujoks and Emmerich 2007). They share with `nsga2` the representation, the
variation operators (uniform crossover, resampling mutation), the starting population and the
unbounded archive that only records, so that a difference between two of them comes from how
they select. All minimise. `generations` counts blocks of `pop_size` evaluations after the
starting population, also for the two searches that replace one member at a time.

Quality indicators for sets of three objectives are at the end: hypervolume and IGD+.
"""
import numpy as np

from .decoder import njit
from .nsga2 import merge_nondominated, nondominated_fronts

TINY = 1e-12


def das_dennis(divisions: int, objectives: int = 3) -> np.ndarray:
    """All vectors with entries k / divisions that sum to one (Das and Dennis 1998)."""
    def compositions(total, parts):
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first,) + rest
    return np.array(list(compositions(divisions, objectives)), dtype=float) / divisions


def _mutation_rate(mutation_prob, n_keys):
    return max(1.0 / n_keys, 0.05) if mutation_prob is None else mutation_prob


def _start(evaluate, n_keys, pop_size, rng, initial):
    x = rng.random((pop_size, n_keys))
    if initial is not None:
        x[:len(initial)] = initial
    return x, np.array([evaluate(ind) for ind in x], dtype=float)


def _child(p1, p2, rng, crossover_prob, mutation_prob):
    """One child: the first parent, with half of its keys from the second if crossover applies, then mutation."""
    child = p1.copy()
    if rng.random() < crossover_prob:
        mix = rng.random(len(p1)) < 0.5
        child[mix] = p2[mix]
    mutate = rng.random(len(p1)) < mutation_prob
    child[mutate] = rng.random(int(mutate.sum()))
    return child


def _first_front(f):
    return f[np.sort(nondominated_fronts(f)[0])]


def _finish(kept_x, kept_f):
    order = np.lexsort(kept_f.T[::-1])
    return kept_x[order], kept_f[order]


# ----------------------------------------------------------------------------------------------
# NSGA-III

def _intercepts(translated):
    """Axis intercepts of the plane through the extreme points; the largest values where that plane is no use."""
    m = translated.shape[1]
    worst = np.maximum(translated.max(axis=0), TINY)
    weights = np.full((m, m), 1e-6)
    np.fill_diagonal(weights, 1.0)
    extreme = translated[np.argmin((translated[:, None, :] / weights[None, :, :]).max(axis=2), axis=0)]
    try:
        plane = np.linalg.solve(extreme, np.ones(m))
    except np.linalg.LinAlgError:
        return worst
    if not np.all(np.isfinite(plane)) or np.any(plane <= TINY):
        return worst
    intercepts = 1.0 / plane
    return np.where((intercepts > TINY) & (intercepts <= worst * (1 + 1e-9)), intercepts, worst)


def _associate(normalised, directions):
    """For every point, the nearest reference direction and the distance to its line."""
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    along = normalised @ unit.T
    distance = np.sqrt(np.maximum((normalised ** 2).sum(axis=1)[:, None] - along ** 2, 0.0))
    nearest = distance.argmin(axis=1)
    return nearest, distance[np.arange(len(normalised)), nearest]


def _niching(f, fronts, pop_size, directions, rng):
    """Members kept by NSGA-III from the fronts of a merged population."""
    kept, last = [], None
    for front in fronts:
        if len(kept) + len(front) <= pop_size:
            kept.extend(front.tolist())
        else:
            last = front
            break
    if last is None or len(kept) == pop_size:
        return np.array(kept, dtype=int)
    considered = np.array(kept + last.tolist(), dtype=int)
    translated = f[considered] - f[considered].min(axis=0)
    nearest, distance = _associate(translated / _intercepts(translated), directions)
    count = np.bincount(nearest[:len(kept)], minlength=len(directions))
    candidates = {}  # direction -> positions in `considered` of the members of the last front near it
    for position in range(len(kept), len(considered)):
        candidates.setdefault(int(nearest[position]), []).append(position)
    while len(kept) < pop_size:
        open_directions = np.array(sorted(candidates))
        least = open_directions[count[open_directions] == count[open_directions].min()]
        direction = int(rng.choice(least))
        positions = candidates[direction]
        if count[direction] == 0:
            pick = min(range(len(positions)), key=lambda k: distance[positions[k]])
        else:
            pick = int(rng.integers(len(positions)))
        kept.append(int(considered[positions.pop(pick)]))
        count[direction] += 1
        if not positions:
            del candidates[direction]
    return np.array(kept, dtype=int)


def nsga3(evaluate, n_keys: int, pop_size: int, generations: int, rng: np.random.Generator,
          crossover_prob: float = 0.8, mutation_prob: float | None = None, initial=None, observer=None,
          divisions: int = 13):
    """NSGA-III; returns the keys and objectives of the archive of all nondominated plans met."""
    mutation_prob = _mutation_rate(mutation_prob, n_keys)
    x, f = _start(evaluate, n_keys, pop_size, rng, initial)
    directions = das_dennis(divisions, f.shape[1])
    kept_x, kept_f = merge_nondominated(np.empty((0, n_keys)), np.empty((0, f.shape[1])), x, f)
    for generation in range(1, generations + 1):
        a, b = rng.integers(pop_size, size=(2, pop_size))
        children = np.array([_child(x[i], x[j], rng, crossover_prob, mutation_prob) for i, j in zip(a, b)])
        child_f = np.array([evaluate(ind) for ind in children], dtype=float)
        kept_x, kept_f = merge_nondominated(kept_x, kept_f, children, child_f)
        x, f = np.vstack([x, children]), np.vstack([f, child_f])
        keep = _niching(f, nondominated_fronts(f), pop_size, directions, rng)
        x, f = x[keep], f[keep]
        if observer is not None:
            observer(generation, _first_front(f), kept_f)
    return _finish(kept_x, kept_f)


# ----------------------------------------------------------------------------------------------
# MOEA/D

def moead(evaluate, n_keys: int, pop_size: int, generations: int, rng: np.random.Generator,
          crossover_prob: float = 0.8, mutation_prob: float | None = None, initial=None, observer=None,
          divisions: int = 13, neighbours: int = 20):
    """MOEA/D with the Tchebycheff function on objectives scaled to the range of the population.

    `pop_size` must be the number of weight vectors that `divisions` gives. Both parents come from
    the neighbourhood of the subproblem, and a child replaces every neighbour it is no worse than.
    """
    mutation_prob = _mutation_rate(mutation_prob, n_keys)
    x, f = _start(evaluate, n_keys, pop_size, rng, initial)
    weights = np.maximum(das_dennis(divisions, f.shape[1]), 1e-6)
    if len(weights) != pop_size:
        raise ValueError(f"{divisions} divisions give {len(weights)} weight vectors, not {pop_size}")
    near = np.argsort(((weights[:, None, :] - weights[None, :, :]) ** 2).sum(axis=2), axis=1, kind="stable")[:, :neighbours]
    ideal = f.min(axis=0)
    kept_x, kept_f = merge_nondominated(np.empty((0, n_keys)), np.empty((0, f.shape[1])), x, f)
    for generation in range(1, generations + 1):
        scale = np.maximum(f.max(axis=0) - ideal, TINY)
        children, child_f = np.empty((pop_size, n_keys)), np.empty((pop_size, f.shape[1]))
        for done, i in enumerate(rng.permutation(pop_size)):
            first, second = rng.choice(near[i], 2, replace=False)
            child = _child(x[first], x[second], rng, crossover_prob, mutation_prob)
            fc = np.asarray(evaluate(child), dtype=float)
            children[done], child_f[done] = child, fc
            ideal = np.minimum(ideal, fc)
            group = near[i]
            new = (weights[group] * (fc - ideal) / scale).max(axis=1)
            old = (weights[group] * (f[group] - ideal) / scale).max(axis=1)
            better = group[new <= old]
            x[better], f[better] = child, fc
        kept_x, kept_f = merge_nondominated(kept_x, kept_f, children, child_f)
        if observer is not None:
            observer(generation, _first_front(f), kept_f)
    return _finish(kept_x, kept_f)


# ----------------------------------------------------------------------------------------------
# SMS-EMOA

def _hv3(points, ref):
    """Volume dominated by points of three objectives up to a reference point all of them improve on."""
    n = len(points)
    order = np.argsort(points[:, 2])
    sx, sy = np.empty(n), np.empty(n)  # staircase of the slices passed so far: x rising, y falling
    tx, ty = np.empty(n), np.empty(n)
    count = 0
    volume = 0.0
    for a in range(n):
        i = order[a]
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        covered = False
        for k in range(count):
            if sx[k] <= x and sy[k] <= y:
                covered = True
                break
        if not covered:
            m = 0
            placed = False
            for k in range(count):
                if x <= sx[k] and y <= sy[k]:
                    continue
                if not placed and x < sx[k]:
                    tx[m], ty[m] = x, y
                    m += 1
                    placed = True
                tx[m], ty[m] = sx[k], sy[k]
                m += 1
            if not placed:
                tx[m], ty[m] = x, y
                m += 1
            for k in range(m):
                sx[k], sy[k] = tx[k], ty[k]
            count = m
        area = 0.0
        for k in range(count):
            right = sx[k + 1] if k + 1 < count else ref[0]
            area += (right - sx[k]) * (ref[1] - sy[k])
        upper = points[order[a + 1], 2] if a + 1 < n else ref[2]
        volume += area * (upper - z)
    return volume


def _contributions(points, ref):
    """What each point adds to the dominated volume of the set.

    The part of a point's own box, from the point to the reference, that no other point covers. The
    others are moved into that box first, which leaves few of them on the staircase of `_hv3`.
    """
    n = len(points)
    out = np.empty(n)
    inside = np.empty((n - 1, 3))
    for i in range(n):
        m = 0
        for j in range(n):
            if j != i:
                for c in range(3):
                    inside[m, c] = max(points[j, c], points[i, c])
                m += 1
        box = (ref[0] - points[i, 0]) * (ref[1] - points[i, 1]) * (ref[2] - points[i, 2])
        out[i] = box - _hv3(inside, ref)
    return out


def _leaver(f, margin, u):
    """Position of the member that SMS-EMOA removes; `u` in [0, 1) settles ties.

    While some member is dominated: the member of the last front that most others dominate. Else a
    member with the objectives of an earlier one, if there is any. Else the member that adds least to
    the dominated volume, on objectives scaled to the range of the set, up to 1 + margin.
    """
    n = len(f)
    dominates = np.zeros((n, n), np.bool_)
    dominated_by = np.zeros(n, np.int64)
    for i in range(n):
        for j in range(n):
            if (i != j and f[i, 0] <= f[j, 0] and f[i, 1] <= f[j, 1] and f[i, 2] <= f[j, 2]
                    and (f[i, 0] < f[j, 0] or f[i, 1] < f[j, 1] or f[i, 2] < f[j, 2])):
                dominates[i, j] = True
                dominated_by[j] += 1
    pick = np.empty(n, np.int64)
    m = 0
    if dominated_by.max() > 0:
        waiting = dominated_by.copy()
        level = np.full(n, -1)
        done, current = 0, -1
        while done < n:
            current += 1
            for k in range(n):
                if level[k] == -1 and waiting[k] == 0:
                    level[k] = current
                    done += 1
            for k in range(n):
                if level[k] == current:
                    for j in range(n):
                        if dominates[k, j] and level[j] == -1:
                            waiting[j] -= 1
        most = -1
        for k in range(n):
            if level[k] == current and dominated_by[k] > most:
                most = dominated_by[k]
        for k in range(n):
            if level[k] == current and dominated_by[k] == most:
                pick[m] = k
                m += 1
        return pick[int(u * m)]
    for j in range(1, n):
        for i in range(j):
            if f[i, 0] == f[j, 0] and f[i, 1] == f[j, 1] and f[i, 2] == f[j, 2]:
                pick[m] = j
                m += 1
                break
    if m > 0:
        return pick[int(u * m)]
    scaled = np.empty((n, 3))
    for c in range(3):
        low, high = f[:, c].min(), f[:, c].max()
        span = high - low if high - low > TINY else 1.0
        for k in range(n):
            scaled[k, c] = (f[k, c] - low) / span
    return int(np.argmin(_contributions(scaled, np.full(3, 1.0 + margin))))


if njit is not None:
    _hv3 = njit(cache=True)(_hv3)
    _contributions = njit(cache=True)(_contributions)
    _leaver = njit(cache=True)(_leaver)


def hypervolume(points: np.ndarray, ref) -> float:
    """Volume dominated by a set of points with three objectives, minimised, up to `ref`."""
    points, ref = np.asarray(points, dtype=float), np.asarray(ref, dtype=float)
    inside = points[(points < ref).all(axis=1)]
    return float(_hv3(np.ascontiguousarray(inside), ref)) if len(inside) else 0.0


def sms_emoa(evaluate, n_keys: int, pop_size: int, generations: int, rng: np.random.Generator,
             crossover_prob: float = 0.8, mutation_prob: float | None = None, initial=None, observer=None,
             margin: float = 0.1):
    """SMS-EMOA for three objectives: one child at a time, and one member leaves.

    While some member is dominated, the member of the last front that most others dominate leaves.
    Once all are nondominated, the one that adds least to the dominated volume leaves; the volume is
    measured on objectives scaled to the range of the population, up to 1 + `margin` in each.
    """
    mutation_prob = _mutation_rate(mutation_prob, n_keys)
    x, f = _start(evaluate, n_keys, pop_size, rng, initial)
    if f.shape[1] != 3:
        raise ValueError("three objectives expected")
    kept_x, kept_f = merge_nondominated(np.empty((0, n_keys)), np.empty((0, 3)), x, f)
    x, f = np.vstack([x, np.empty((1, n_keys))]), np.vstack([f, np.empty((1, 3))])  # one spare row for the child
    for generation in range(1, generations + 1):
        children, child_f = np.empty((pop_size, n_keys)), np.empty((pop_size, 3))
        for done in range(pop_size):
            first, second = rng.choice(pop_size, 2, replace=False)
            child = _child(x[first], x[second], rng, crossover_prob, mutation_prob)
            fc = np.asarray(evaluate(child), dtype=float)
            children[done], child_f[done] = child, fc
            x[pop_size], f[pop_size] = child, fc
            out = int(_leaver(f, margin, rng.random()))
            x[out], f[out] = x[pop_size], f[pop_size]
        kept_x, kept_f = merge_nondominated(kept_x, kept_f, children, child_f)
        if observer is not None:
            observer(generation, _first_front(f[:pop_size]), kept_f)
    return _finish(kept_x, kept_f)


# ----------------------------------------------------------------------------------------------
# Indicators

def igd_plus(front: np.ndarray, reference: np.ndarray) -> float:
    """IGD+ (Ishibuchi et al. 2015): mean over the reference points of the distance to the nearest
    point of the front, counting only the objectives in which that point is worse."""
    worse = np.maximum(front[None, :, :] - reference[:, None, :], 0.0)
    return float(np.sqrt((worse ** 2).sum(axis=2)).min(axis=1).mean())
