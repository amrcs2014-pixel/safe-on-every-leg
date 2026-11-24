import numpy as np
import pytest

from legsafe.moea import (_contributions, _leaver, _niching, das_dennis, hypervolume, igd_plus, moead, nsga3, sms_emoa)
from legsafe.nsga2 import nondominated_fronts, nsga2

SEARCHES = {"nsga3": nsga3, "moead": moead, "sms_emoa": sms_emoa}
POP = 28  # the number of directions for six divisions and three objectives


def sphere(k):
    """DTLZ2 with three objectives: the best plans lie on the unit sphere, with every key after the second at 0.5."""
    g = float(((k[2:] - 0.5) ** 2).sum())
    a, b = k[0] * np.pi / 2, k[1] * np.pi / 2
    return (1 + g) * np.cos(a) * np.cos(b), (1 + g) * np.cos(a) * np.sin(b), (1 + g) * np.sin(a)


def run(name, seed=0, generations=60, observer=None, n_keys=6):
    return SEARCHES[name](sphere, n_keys, POP, generations, np.random.default_rng(seed), observer=observer, divisions=6) \
        if name != "sms_emoa" else sms_emoa(sphere, n_keys, POP, generations, np.random.default_rng(seed), observer=observer)


def exact_volume(points, ref):
    """Dominated volume by cutting the space at every coordinate of the points."""
    cuts = [np.unique(np.append(points[:, j], ref[j])) for j in range(3)]
    volume = 0.0
    for a in range(len(cuts[0]) - 1):
        for b in range(len(cuts[1]) - 1):
            for c in range(len(cuts[2]) - 1):
                corner = np.array([cuts[0][a], cuts[1][b], cuts[2][c]])
                if (points <= corner).all(axis=1).any():
                    volume += (cuts[0][a + 1] - corner[0]) * (cuts[1][b + 1] - corner[1]) * (cuts[2][c + 1] - corner[2])
    return volume


@pytest.mark.parametrize("divisions, expected", [(1, 3), (6, 28), (12, 91), (13, 105)])
def test_das_dennis_gives_every_lattice_point_of_the_simplex(divisions, expected):
    w = das_dennis(divisions)
    assert len(w) == expected == len(np.unique(w, axis=0))
    assert np.allclose(w.sum(axis=1), 1) and (w >= 0).all()


def test_hypervolume_of_simple_sets():
    ref = np.ones(3)
    assert hypervolume(np.zeros((1, 3)), ref) == pytest.approx(1.0)
    assert hypervolume(np.array([[0.5, 0.5, 0.5]]), ref) == pytest.approx(0.125)
    assert hypervolume(np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]), ref) == pytest.approx(0.125)  # a dominated point adds nothing
    assert hypervolume(np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]), ref) == pytest.approx(0.25 + 0.25 - 0.125)
    assert hypervolume(np.array([[1.0, 0.0, 0.0], [2.0, 2.0, 2.0]]), ref) == 0.0  # no point improves on the reference


@pytest.mark.parametrize("seed", range(8))
def test_hypervolume_matches_the_volume_counted_cell_by_cell(seed):
    rng = np.random.default_rng(seed)
    points = rng.random((int(rng.integers(1, 12)), 3)).round(1 if seed % 2 else 6)  # rounding gives equal coordinates
    ref = np.full(3, 1.1)
    assert hypervolume(points, ref) == pytest.approx(exact_volume(points, ref))


@pytest.mark.parametrize("seed", range(4))
def test_contributions_are_what_the_volume_loses_without_the_point(seed):
    points = np.random.default_rng(seed).random((9, 3))
    points[3] = points[2] + 0.01  # dominated
    ref = np.full(3, 1.2)
    contributions = _contributions(points, ref)
    for i in range(len(points)):
        assert contributions[i] == pytest.approx(exact_volume(points, ref) - exact_volume(np.delete(points, i, axis=0), ref), abs=1e-12)
    assert contributions[3] == pytest.approx(0.0, abs=1e-12) and (contributions > -1e-12).all()


def test_member_that_leaves_is_the_most_dominated_of_the_last_front_then_a_copy_then_the_least_useful():
    front = np.array([[0.0, 1.0, 0.5], [0.5, 0.5, 0.5], [1.0, 0.0, 0.5], [0.4, 0.6, 0.45]])
    behind = np.array([[0.6, 0.6, 0.6], [0.9, 0.9, 0.9], [0.95, 0.2, 0.95]])  # the second lies behind the first
    f = np.vstack([front, behind])
    assert all(_leaver(f, 0.1, u) == 5 for u in (0.0, 0.5, 0.99))  # last front is {5}; 6 is in the second front

    copies = np.vstack([front, front[1:2], front[3:4]])
    assert {int(_leaver(copies, 0.1, u)) for u in (0.0, 0.49, 0.5, 0.99)} == {4, 5}

    rng = np.random.default_rng(1)
    for _ in range(5):
        points = rng.random((12, 3))
        points = points[[not ((points <= p).all(axis=1) & (points < p).any(axis=1)).any() for p in points]]
        low, span = points.min(axis=0), points.max(axis=0) - points.min(axis=0)
        scaled = (points - low) / span
        ref = np.full(3, 1.1)
        exact = [exact_volume(scaled, ref) - exact_volume(np.delete(scaled, i, axis=0), ref) for i in range(len(scaled))]
        assert _leaver(points, 0.1, 0.3) == int(np.argmin(exact))


def test_igd_plus_is_zero_for_a_front_that_reaches_or_beats_the_reference():
    reference = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.5]])
    assert igd_plus(reference, reference) == 0.0
    assert igd_plus(reference - 0.1, reference) == 0.0
    assert igd_plus(reference + np.array([0.3, 0.0, 0.4]), reference) == pytest.approx(0.5)


def test_niching_keeps_the_better_fronts_whole_and_fills_up_from_the_last():
    rng = np.random.default_rng(0)
    f = rng.random((40, 3))
    fronts = nondominated_fronts(f)
    kept = _niching(f, fronts, 20, das_dennis(6), rng)
    assert len(kept) == 20 == len(set(kept.tolist()))
    whole, size = [], 0
    for front in fronts:
        if size + len(front) > 20:
            break
        whole.extend(front.tolist())
        size += len(front)
    assert set(whole) <= set(kept.tolist())
    rank = np.empty(40, int)
    for level, front in enumerate(fronts):
        rank[front] = level
    assert rank[kept].max() <= rank[np.setdiff1d(np.arange(40), kept)].min()


@pytest.mark.parametrize("name", SEARCHES)
def test_search_returns_a_nondominated_archive_close_to_the_true_front(name):
    seen = []
    x, f = run(name, observer=lambda generation, front, archive: seen.append((generation, len(front), archive.copy())))
    assert [g for g, _, _ in seen] == list(range(1, 61))
    assert all(1 <= size <= POP for _, size, _ in seen)
    assert len(nondominated_fronts(f)[0]) == len(f) == len(np.unique(f, axis=0))
    assert np.allclose(np.array([sphere(k) for k in x]), f)
    assert np.array_equal(f, seen[-1][2][np.lexsort(seen[-1][2].T[::-1])])
    radius = np.sqrt((f ** 2).sum(axis=1))
    assert radius.min() >= 1 - 1e-9 and np.median(radius) < 1.05
    ref = np.full(3, 1.1)
    start = np.array([sphere(k) for k in np.random.default_rng(0).random((POP, 6))])
    assert hypervolume(f, ref) > max(0.60, hypervolume(start, ref) + 0.2)  # the whole octant gives 0.807
    volumes = [hypervolume(archive, ref) for _, _, archive in seen]
    assert all(later >= earlier - 1e-12 for earlier, later in zip(volumes, volumes[1:]))  # an archive never loses volume


@pytest.mark.parametrize("name", SEARCHES)
def test_search_is_repeatable_and_depends_on_its_seed(name):
    first, again, other = run(name, 1, 8), run(name, 1, 8), run(name, 2, 8)
    assert np.array_equal(first[0], again[0]) and np.array_equal(first[1], again[1])
    assert first[1].shape != other[1].shape or not np.array_equal(first[1], other[1])


@pytest.mark.parametrize("name", SEARCHES)
def test_search_starts_from_the_members_it_is_given(name):
    head = np.full((5, 6), 0.5)
    head[:, 0] = np.linspace(0, 1, 5)
    kwargs = {} if name == "sms_emoa" else {"divisions": 6}
    _, f = SEARCHES[name](sphere, 6, POP, 0, np.random.default_rng(3), initial=head, **kwargs)
    assert (np.abs(np.sqrt((f ** 2).sum(axis=1)) - 1) < 1e-9).any()  # a given member lies on the true front


def test_moead_needs_a_population_of_the_size_of_its_weight_set():
    with pytest.raises(ValueError):
        moead(sphere, 6, 30, 1, np.random.default_rng(0), divisions=6)


def test_all_four_searches_reach_similar_quality_on_the_sphere():
    ref = np.full(3, 1.1)
    volumes = {name: hypervolume(run(name, 5, 80)[1], ref) for name in SEARCHES}
    _, f = nsga2(sphere, 6, POP, 80, np.random.default_rng(5), archive=True)
    volumes["nsga2"] = hypervolume(f, ref)
    assert min(volumes.values()) > 0.65 and max(volumes.values()) - min(volumes.values()) < 0.12
