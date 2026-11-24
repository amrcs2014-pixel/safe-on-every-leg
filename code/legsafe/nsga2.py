import numpy as np


def nondominated_fronts(f: np.ndarray) -> list[np.ndarray]:
    """Fronts of a minimisation problem, best first."""
    dominates = (f[:, None, :] <= f[None, :, :]).all(axis=2) & (f[:, None, :] < f[None, :, :]).any(axis=2)
    dominated_by = dominates.sum(axis=0)
    fronts = []
    current = np.flatnonzero(dominated_by == 0)
    while current.size:
        fronts.append(current)
        dominated_by[current] = -1
        dominated_by -= dominates[current].sum(axis=0)
        current = np.flatnonzero(dominated_by == 0)
    return fronts


def crowding_distance(f: np.ndarray) -> np.ndarray:
    n, m = f.shape
    distance = np.zeros(n)
    for j in range(m):
        order = np.argsort(f[:, j], kind="stable")
        span = f[order[-1], j] - f[order[0], j]
        distance[order[0]] = distance[order[-1]] = np.inf
        if span > 0 and n > 2:
            distance[order[1:-1]] += (f[order[2:], j] - f[order[:-2], j]) / span
    return distance


def merge_nondominated(archive_x: np.ndarray, archive_f: np.ndarray, x: np.ndarray, f: np.ndarray):
    """The nondominated members of an archive and of newcomers, one per objective vector.

    The archive must be nondominated already. A newcomer with the objectives of an archive member is
    dropped, so the archive keeps the plan it met first.
    """
    if len(archive_f):
        covered = (archive_f[:, None, :] <= f[None, :, :]).all(axis=2).any(axis=0)
        x, f = x[~covered], f[~covered]
    if not len(f):
        return archive_x, archive_f
    _, first = np.unique(f, axis=0, return_index=True)
    x, f = x[np.sort(first)], f[np.sort(first)]
    best = np.sort(nondominated_fronts(f)[0])
    x, f = x[best], f[best]
    if len(archive_f):
        beaten = (f[:, None, :] <= archive_f[None, :, :]).all(axis=2).any(axis=0)
        archive_x, archive_f = archive_x[~beaten], archive_f[~beaten]
    return np.vstack([archive_x, x]), np.vstack([archive_f, f])


def nsga2(evaluate, n_keys: int, pop_size: int, generations: int, rng: np.random.Generator,
          crossover_prob: float = 0.8, mutation_prob: float | None = None, initial=None, observer=None,
          archive: bool = False, distinct: bool = False):
    """NSGA-II on random keys in [0, 1); returns the keys and objectives of the final nondominated set.

    Rows of `initial`, if given, replace the first members of the random starting population.

    With `distinct`, a member whose objectives equal those of an earlier member counts as a copy.
    Copies are ranked behind every other member, so they survive only when there are too few
    distinct members to fill the population, and they rarely win a tournament.

    With `archive`, every nondominated plan met during the search is kept in an unbounded archive,
    and the archive is returned in place of the first front of the final population. The archive
    only records: the search runs exactly as it does without it.

    `observer(generation, front, archive)`, if given, is called after every generation with the
    objectives of the population's first front and of the archive (None without an archive).
    """
    if mutation_prob is None:
        mutation_prob = max(1.0 / n_keys, 0.05)

    def rank_and_crowd(f):
        rank = np.empty(len(f), int)
        crowd = np.zeros(len(f))
        members = np.arange(len(f))
        if distinct:
            members = np.sort(np.unique(f, axis=0, return_index=True)[1])  # the first member of each objective vector
        fronts = nondominated_fronts(f[members])
        rank[:] = len(fronts)  # copies, if any are set aside, come after the last front
        for level, front in enumerate(fronts):
            rank[members[front]] = level
            crowd[members[front]] = crowding_distance(f[members[front]])
        return rank, crowd

    x = rng.random((pop_size, n_keys))
    if initial is not None:
        x[:len(initial)] = initial
    f = np.array([evaluate(ind) for ind in x], dtype=float)
    rank, crowd = rank_and_crowd(f)
    kept_x, kept_f = np.empty((0, n_keys)), np.empty((0, f.shape[1]))
    if archive:
        kept_x, kept_f = merge_nondominated(kept_x, kept_f, x, f)

    for generation in range(1, generations + 1):
        a, b = rng.integers(pop_size, size=(2, 2 * pop_size))
        better = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
        parents = np.where(better, a, b)
        p1, p2 = x[parents[:pop_size]], x[parents[pop_size:]]

        mix = rng.random((pop_size, n_keys)) < 0.5
        crossed = rng.random(pop_size) < crossover_prob
        children = np.where(mix & crossed[:, None], p2, p1)
        mutate = rng.random((pop_size, n_keys)) < mutation_prob
        children = np.where(mutate, rng.random((pop_size, n_keys)), children)
        child_f = np.array([evaluate(ind) for ind in children], dtype=float)
        if archive:
            kept_x, kept_f = merge_nondominated(kept_x, kept_f, children, child_f)

        x = np.vstack([x, children])
        f = np.vstack([f, child_f])
        rank, crowd = rank_and_crowd(f)
        keep = np.lexsort((-crowd, rank))[:pop_size]
        x, f, rank, crowd = x[keep], f[keep], rank[keep], crowd[keep]
        if observer is not None:
            observer(generation, f[rank == 0], kept_f if archive else None)

    if archive:
        order = np.lexsort(kept_f.T[::-1])
        return kept_x[order], kept_f[order]
    best = rank == 0
    _, unique = np.unique(f[best], axis=0, return_index=True)
    return x[best][unique], f[best][unique]
