from pathlib import Path

import numpy as np
import pytest
from checks import assert_feasible

from legsafe import decode, extend, merge_stops, n_keys, nsga2, read_br, unloading_obstacles, utilisation
from legsafe.decoder import level, level_key, offset_levels
from legsafe.nsga2 import merge_nondominated, nondominated_fronts

DATA = Path(__file__).resolve().parents[2] / "data" / "br" / "orlib"

def instance(br_class, index, seed=0, payload=10.1):
    raw = read_br(DATA / f"thpack{br_class}.txt")[index]
    return extend(raw, payload=payload, tau=0.9, alpha=2, beta=5, p_nonstackable=0.1,
                  rng=np.random.default_rng(seed))


def test_br_file_is_read_as_published():
    instances = read_br(DATA / "thpack1.txt")
    assert len(instances) == 100
    first = instances[0]
    assert first.space == (5.87, 2.33, 2.20)
    assert [t.count for t in first.types] == [40, 33, 39]
    assert first.types[0].dims == (1.08, 0.76, 0.30)
    assert first.types[0].vertical_ok == (False, False, True)


@pytest.mark.parametrize("br_class, n_types", [(1, 3), (7, 20), (8, 30), (15, 100)])
def test_every_class_has_its_file_and_number_of_box_types(br_class, n_types):
    from experiments.common import br_file

    instances = read_br(br_file(br_class))
    assert len(instances) == 100
    assert all(len(inst.types) == n_types and inst.space == (5.87, 2.33, 2.20) for inst in instances)


def test_extension_ranks_types_by_demand_and_hits_target_density():
    raw = read_br(DATA / "thpack5.txt")[0]
    rng = np.random.default_rng(1)
    inst = extend(raw, payload=10.1, tau=0.9, alpha=2, beta=2, p_nonstackable=0.01, rng=rng)
    by_rank = sorted(inst.types, key=lambda t: t.rank)
    assert [t.rank for t in by_rank] == list(range(1, len(inst.types) + 1))
    assert all(a.count >= b.count for a, b in zip(by_rank, by_rank[1:]))
    volume = np.prod(inst.space)
    densities = np.array([t.mass / t.volume for t in inst.types])
    mean_density = 10.1 / (0.9 * volume)
    assert densities.min() >= 0.5 * mean_density
    assert densities.max() <= 1.5 * mean_density + 1e-12  # Beta(2,2): d_max = 1.5 x mean


def test_merging_stops_keeps_the_boxes_and_the_order_of_deliveries():
    inst = instance(7, 0)
    assert merge_stops(inst, inst.n_stops).types == inst.types
    for stops in (1, 3, 7, 19):
        merged = merge_stops(inst, stops)
        assert [(t.dims, t.count, t.mass, t.stackable) for t in merged.types] == \
               [(t.dims, t.count, t.mass, t.stackable) for t in inst.types]
        before, after = (np.array([t.rank for t in i.types]) for i in (inst, merged))
        assert sorted(set(after)) == list(range(1, stops + 1)) and merged.n_stops == stops
        assert (np.diff(after[np.argsort(before)]) >= 0).all()   # no delivery overtakes another
        sizes = np.bincount(after)[1:]
        assert sizes.max() - sizes.min() <= 1                    # groups of nearly equal size
    with pytest.raises(ValueError):
        merge_stops(inst, inst.n_stops + 1)


@pytest.mark.parametrize("stops", [1, 4, 10])
@pytest.mark.parametrize("seed", range(4))
def test_a_plan_stays_a_plan_when_stops_are_merged_and_gets_no_worse(stops, seed):
    from experiments.e5_pilot import VEHICLE, legs
    from legsafe import axle_margin
    from legsafe.decoder import Layout

    inst = instance(7, seed, seed)
    merged = merge_stops(inst, stops)
    layout = decode(inst, np.random.default_rng(300 + seed).random(n_keys(inst)))
    rank = np.array([t.rank for t in merged.types])[layout.type]
    same = Layout(layout.lo, layout.size, layout.type, rank, layout.mass)
    assert_feasible(merged, same)
    assert unloading_obstacles(same) <= unloading_obstacles(layout)
    assert axle_margin(VEHICLE, legs(merged, same)).min() >= axle_margin(VEHICLE, legs(inst, layout)).min() - 1e-12


def test_instances_with_fewer_stops_share_boxes_weights_and_seeds_with_the_original():
    from experiments.common import instance as job_instance

    job = (10, 13, (2, 2))
    full, seeds = job_instance(job, 2)
    short, short_seeds = job_instance(job, 2, stops=10)
    assert full.n_stops == 50 and short.n_stops == 10
    assert [t.mass for t in short.types] == [t.mass for t in full.types]
    assert all(a.random() == b.random() for a, b in zip(seeds, short_seeds))


@pytest.mark.parametrize("br_class", [1, 3, 5, 7])
@pytest.mark.parametrize("seed", range(5))
def test_decoded_layout_is_feasible(br_class, seed):
    inst = instance(br_class, seed, seed)
    rng = np.random.default_rng(100 + seed)
    layout = decode(inst, rng.random(2 * len(inst.types)))
    assert len(layout.lo) > 0
    assert_feasible(inst, layout)


@pytest.mark.parametrize("br_class", [1, 4, 7])
@pytest.mark.parametrize("seed", range(5))
def test_extended_layout_is_feasible_and_follows_its_keys(br_class, seed):
    inst = instance(br_class, seed, seed)
    n_types = len(inst.types)
    keys = np.random.default_rng(200 + seed).random(n_keys(inst))
    layout = decode(inst, keys)
    assert len(layout.lo) > 0
    assert_feasible(inst, layout)

    step, top = offset_levels(inst)
    assert (layout.lo[:, 0] >= step * level(keys[3 * n_types], top) - 1e-9).all()
    for i, box in enumerate(inst.types):
        assert (layout.type == i).sum() <= box.count - level(keys[2 * n_types + i], box.count)


def test_extended_keys_below_one_half_give_the_plain_layout():
    inst = instance(3, 0)
    rng = np.random.default_rng(5)
    plain = rng.random(n_keys(inst, extended=False))
    expected = decode(inst, plain)
    for extra in (np.zeros(len(inst.types) + 1), 0.499 * rng.random(len(inst.types) + 1)):
        layout = decode(inst, np.concatenate([plain, extra]))
        assert np.array_equal(layout.lo, expected.lo) and np.array_equal(layout.type, expected.type)


def test_front_wall_key_moves_the_load_back_and_leave_behind_keys_remove_boxes():
    inst = instance(2, 0)
    n_types = len(inst.types)
    keys = np.zeros(n_keys(inst))
    keys[:2 * n_types] = np.random.default_rng(1).random(2 * n_types)
    step, top = offset_levels(inst)
    assert top == 6 and step == pytest.approx(inst.space[0] / 12)

    keys[3 * n_types] = level_key(2, top)
    assert decode(inst, keys).lo[:, 0].min() == pytest.approx(2 * step)
    quarter, top_quarter = offset_levels(inst, 0.25)
    assert quarter == 0.25 and top_quarter == 11
    keys[3 * n_types] = level_key(2, top_quarter)
    assert decode(inst, keys, offset_step=0.25).lo[:, 0].min() == pytest.approx(0.5)

    keys[3 * n_types] = level_key(2, top)
    keys[2 * n_types] = level_key(inst.types[0].count, inst.types[0].count)
    assert (decode(inst, keys).type != 0).all()


@pytest.mark.parametrize("top", [0, 1, 3, 12])
def test_level_keys_map_back_to_their_levels(top):
    assert [level(level_key(value, top), top) for value in range(top + 1)] == list(range(top + 1))
    assert level(0.0, top) == level(0.4999, top) == 0
    assert level(0.9999, top) == top


@pytest.mark.parametrize("seed", range(5))
def test_unloading_obstacles_match_pairwise_count(seed):
    inst = instance(4, seed, seed)
    layout = decode(inst, np.random.default_rng(seed).random(2 * len(inst.types)))
    lo, hi = layout.lo, layout.lo + layout.size

    def overlaps(i, j, axis):
        return lo[j, axis] < hi[i, axis] - 1e-9 and lo[i, axis] < hi[j, axis] - 1e-9

    count = 0
    for i in range(len(lo)):
        for j in range(len(lo)):
            if layout.rank[j] <= layout.rank[i]:
                continue
            in_front = lo[j, 0] >= hi[i, 0] - 1e-9 and overlaps(i, j, 1) and overlaps(i, j, 2)
            on_top = lo[j, 2] >= hi[i, 2] - 1e-9 and overlaps(i, j, 0) and overlaps(i, j, 1)
            count += in_front or on_top
    assert unloading_obstacles(layout) == count


def test_no_later_box_sits_above_an_earlier_one():
    inst = instance(6, 0)
    layout = decode(inst, np.random.default_rng(3).random(2 * len(inst.types)))
    lo, hi = layout.lo, layout.lo + layout.size
    later = layout.rank[None, :] > layout.rank[:, None]
    above = lo[None, :, 2] >= hi[:, None, 2] - 1e-9
    same_column = np.ones_like(later)
    for axis in (0, 1):
        same_column &= (lo[None, :, axis] < hi[:, None, axis] - 1e-9) & (lo[:, None, axis] < hi[None, :, axis] - 1e-9)
    assert not (later & above & same_column).any()


def test_utilisation_is_packed_volume_over_space_volume():
    inst = instance(1, 0)
    layout = decode(inst, np.full(2 * len(inst.types), 0.5))
    assert 0.3 < utilisation(inst, layout) <= 1.0


def test_nondominated_fronts_on_a_known_set():
    f = np.array([[1, 4], [2, 2], [4, 1], [3, 3], [5, 5]], dtype=float)
    fronts = nondominated_fronts(f)
    assert sorted(fronts[0]) == [0, 1, 2]
    assert sorted(fronts[1]) == [3]
    assert sorted(fronts[2]) == [4]


def test_nsga2_returns_a_nondominated_set_near_the_true_front():
    x, f = nsga2(lambda k: (k[0], (1 - k[0]) + k[1]), n_keys=2, pop_size=20, generations=40,
                 rng=np.random.default_rng(0))
    assert len(nondominated_fronts(f)[0]) == len(f)
    assert (f.sum(axis=1) < 1.1).all()


def toy(k):
    return k[0], (1 - k[0]) + k[1], round(4 * k[2]) / 4


def test_nsga2_observer_sees_every_generation_and_the_extreme_never_gets_worse():
    seen = []
    nsga2(toy, n_keys=3, pop_size=20, generations=30, rng=np.random.default_rng(0),
          observer=lambda generation, front, archive: seen.append((generation, front[:, 0].min(), archive)))
    assert [g for g, _, _ in seen] == list(range(1, 31))
    assert all(archive is None for _, _, archive in seen)
    best = [b for _, b, _ in seen]
    assert all(later <= earlier for earlier, later in zip(best, best[1:]))


def test_merge_keeps_the_nondominated_and_the_first_of_equal_plans():
    archive_x, archive_f = np.array([[0.1], [0.2]]), np.array([[1.0, 4.0], [3.0, 2.0]])
    x = np.array([[0.3], [0.4], [0.5], [0.6], [0.7]])
    f = np.array([[1.0, 4.0], [2.0, 3.0], [2.0, 3.0], [3.0, 1.0], [5.0, 5.0]])
    merged_x, merged_f = merge_nondominated(archive_x, archive_f, x, f)
    assert merged_f.tolist() == [[1.0, 4.0], [2.0, 3.0], [3.0, 1.0]]
    assert merged_x.ravel().tolist() == [0.1, 0.4, 0.6]


def test_archive_records_without_changing_the_search():
    plain, archived = [], []
    nsga2(toy, n_keys=3, pop_size=20, generations=40, rng=np.random.default_rng(3),
          observer=lambda generation, front, archive: plain.append(front.copy()))
    x, f = nsga2(toy, n_keys=3, pop_size=20, generations=40, rng=np.random.default_rng(3), archive=True,
                 observer=lambda generation, front, archive: archived.append((front.copy(), archive.copy())))
    assert all(np.array_equal(a, b) for a, (b, _) in zip(plain, archived))  # same search with and without

    for front, archive in archived:
        assert len(nondominated_fronts(archive)[0]) == len(archive)
        assert len(np.unique(archive, axis=0)) == len(archive)
        covered = (archive[:, None, :] <= front[None, :, :]).all(axis=2).any(axis=0)
        assert covered.all()  # no plan of the population is better than the archive
    assert np.array_equal(np.sort(f, axis=0), np.sort(archived[-1][1], axis=0))
    assert np.allclose(np.array([toy(k) for k in x]), f)
    assert len(f) > 20  # more plans than the population can hold


def coarse(k):
    """Few distinct objective vectors, so that copies are common."""
    return round(3 * k[0]) / 3, round(3 * (1 - k[0]) + 2 * k[1]) / 3


def test_copies_are_ranked_last_only_when_asked():
    def copies(distinct):
        seen = []
        nsga2(coarse, n_keys=2, pop_size=20, generations=15, rng=np.random.default_rng(1), distinct=distinct,
              observer=lambda generation, front, archive: seen.append(len(front) - len(np.unique(front, axis=0))))
        return seen

    assert max(copies(distinct=False)) > 0   # the plain search fills its first front with copies
    assert max(copies(distinct=True)) == 0   # no two members of the first front share their objectives


def test_one_key_per_child_is_the_present_rate_up_to_twenty_keys_and_a_lower_one_beyond():
    from experiments.e5_pilot import ARCHIVED, ONE_KEY, search

    def archive(br_class, config):
        seen = []
        search(instance(br_class, 0), np.random.default_rng(4), dict(config, pop_size=12, generations=4),
               observer=lambda generation, front, archive: seen.append(archive.copy()))
        return seen[-1]

    assert np.array_equal(archive(1, ARCHIVED), archive(1, ONE_KEY))       # 10 keys: one key per child either way
    assert not np.array_equal(archive(7, ARCHIVED), archive(7, ONE_KEY))   # 61 keys: three keys against one


def test_search_started_from_fill_keeps_the_fill_it_was_given_and_counts_those_evaluations():
    from experiments.e5_pilot import ARCHIVED, fill_first, search, starting_members

    inst = instance(5, 2)
    config = dict(ARCHIVED, pop_size=12, generations=30, fill_first=10, fill_seeds=4)
    first = round(10 * len(inst.types) / 12)

    rng = np.random.default_rng(8)
    starting_members(inst, rng, config)
    seeds = fill_first(inst, rng, first, 12, 4)
    fills = [utilisation(inst, decode(inst, k)) for k in seeds]
    assert len(seeds) == 4 and fills == sorted(fills, reverse=True) and len(set(fills)) == 4

    seen = []
    search(inst, np.random.default_rng(8), config, observer=lambda generation, front, archive: seen.append((generation, archive.copy())))
    assert [g for g, _ in seen] == list(range(first + 2, 31))   # the search for fill used first + 1 generations' worth
    assert -seen[0][1][:, 0].min() >= fills[0] - 1e-12          # the best seed is in the archive from the start
    assert -seen[-1][1][:, 0].min() >= fills[0] - 1e-12

    with pytest.raises(ValueError):
        search(inst, np.random.default_rng(8), dict(config, generations=first), observer=None)


def test_search_for_access_returns_the_plans_with_fewest_obstacles_and_then_the_fullest():
    from experiments.e5_pilot import ARCHIVED, TWO_ENDS, best_first, search

    inst = instance(5, 2)
    seeds = best_first(inst, np.random.default_rng(3), 12, 12, 5, "access", extended=True)
    assert seeds.shape == (5, n_keys(inst))
    scores = [(unloading_obstacles(decode(inst, k)), -utilisation(inst, decode(inst, k))) for k in seeds]
    assert scores == sorted(scores) and len(set(scores)) == 5

    config = dict(TWO_ENDS, pop_size=12, generations=40, fill_first=10, fill_seeds=3, access_first=10, access_seeds=3)
    first = round(10 * len(inst.types) / 12)
    seen = []
    search(inst, np.random.default_rng(8), config, observer=lambda generation, front, archive: seen.append(generation))
    assert seen == list(range(2 * (first + 1) + 1, 41))   # both first parts count against the budget
    assert ARCHIVED.get("access_first") is None


def test_mutation_rate_is_the_share_of_keys_resampled():
    def changed(mutation_prob):
        start = np.full((40, 50), 0.5)
        children = []
        nsga2(lambda k: (children.append(k.copy()), (k[0], k[1]))[1], n_keys=50, pop_size=40, generations=1,
              rng=np.random.default_rng(0), crossover_prob=0.0, mutation_prob=mutation_prob, initial=start)
        return np.mean(np.array(children[40:]) != 0.5)   # the first 40 evaluations are the starting population

    assert changed(0.0) == 0.0
    assert changed(1 / 50) == pytest.approx(1 / 50, abs=0.01)
    assert changed(0.2) == pytest.approx(0.2, abs=0.03)


def test_ranking_copies_last_keeps_the_population_full_and_the_result_nondominated():
    x, f = nsga2(coarse, n_keys=2, pop_size=20, generations=15, rng=np.random.default_rng(2), distinct=True)
    assert len(np.unique(f, axis=0)) == len(f) == len(nondominated_fronts(f)[0])
    assert np.allclose(np.array([coarse(k) for k in x]), f)
