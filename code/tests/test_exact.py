from itertools import product

import numpy as np
import pytest
from checks import assert_feasible

from experiments.common import REFERENCE_TARE, TARE, truck
from experiments.small import DELTA, small_instance
from legsafe import BoxType, Instance, axle_margin, decode, leg_moments, n_keys, unloading_obstacles
from legsafe.decoder import level_key, offset_levels
from legsafe.exact import GridModel

VEHICLE = truck(TARE[REFERENCE_TARE])
NARROW = (6.0, 1.0, 1.0)


def sigma(inst, layout):
    return axle_margin(VEHICLE, leg_moments(layout.mass, *layout.centre.T, layout.rank, inst.n_stops)).min()


def test_boxes_that_tile_the_space_fill_it():
    inst = Instance("tiling", (2.0, 1.0, 1.0), (BoxType((1.0, 1.0, 1.0), (True, True, True), 2, mass=0.1),), payload=10.0)
    model = GridModel(inst, VEHICLE, DELTA)
    solution, closed = model.solve()
    assert closed and solution.proven
    assert solution.volume == np.prod(model.grid)
    assert_feasible(inst, solution.layout)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_decoded_layout_is_feasible_for_the_model(seed):
    inst = small_instance(seed, 5, space=NARROW)
    model = GridModel(inst, VEHICLE, DELTA)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        assert model.violation(decode(inst, rng.random(2 * len(inst.types)))) < 1e-9


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_extended_layout_is_feasible_for_the_model(seed):
    inst = small_instance(seed, 5, space=NARROW)
    model = GridModel(inst, VEHICLE, DELTA)
    rng = np.random.default_rng(seed)
    for _ in range(100):
        assert model.violation(decode(inst, rng.random(n_keys(inst)), DELTA)) < 1e-9


def test_extended_decoder_builds_a_legal_plan_the_plain_decoder_cannot():
    """Exact front with every leg legal: (46, 3), (44, 0). The second point needs one box left
    behind and half a metre free at the front wall."""
    inst = small_instance(3, 6, space=NARROW)
    n_types = len(inst.types)
    rank = np.array([t.rank for t in inst.types])
    last = int(rank.argmax())
    _, top = offset_levels(inst, DELTA)
    keys = np.zeros(n_keys(inst))
    keys[:n_types] = 1.0 - rank / (rank.max() + 1)  # later deliveries first
    keys[2 * n_types + last] = level_key(1, inst.types[last].count)
    keys[3 * n_types] = level_key(1, top)

    def outcome(k):
        layout = decode(inst, k, DELTA)
        return round(layout.size.prod(axis=1).sum() / DELTA**3), unloading_obstacles(layout), sigma(inst, layout) >= 0

    choices = [[(i + 0.5) / len(t.orientations()) for i in range(len(t.orientations()))] for t in inst.types]
    found = set()
    for choice in product(*choices):
        keys[n_types:2 * n_types] = choice
        found.add(outcome(keys))
        found.add(("plain", *outcome(keys[:2 * n_types])))
    assert (44, 0, True) in found
    assert not any(o[0] == "plain" and o[2] == 0 and o[3] for o in found)


@pytest.mark.parametrize("seed", [1, 3])
def test_exact_solution_is_feasible_and_no_worse_than_the_decoder(seed):
    inst = small_instance(seed, 5, space=NARROW)
    model = GridModel(inst, VEHICLE, DELTA)
    solution, closed = model.solve(time_limit=60)
    assert closed and solution.proven
    assert_feasible(inst, solution.layout)
    assert solution.obstacles == unloading_obstacles(solution.layout)

    rng = np.random.default_rng(seed)
    best = max(decode(inst, rng.random(2 * len(inst.types))).size.prod(axis=1).sum() for _ in range(300))
    assert solution.volume * DELTA**3 >= best - 1e-9


def test_margin_constraint_makes_every_leg_legal():
    inst = small_instance(3, 5, space=NARROW)
    model = GridModel(inst, VEHICLE, DELTA)
    for margin in (0.0, 0.05):
        solution, closed = model.solve(margin=margin, time_limit=60)
        assert closed
        assert sigma(inst, solution.layout) >= margin - 1e-9
        assert model.violation(solution.layout, margin) < 1e-9


def test_front_trades_volume_against_obstacles_when_legality_is_required():
    inst = small_instance(3, 6, space=NARROW)
    model = GridModel(inst, VEHICLE, DELTA)
    free, closed_free = model.front(None, time_limit=60)
    legal, closed_legal = model.front(0.0, time_limit=60)
    assert closed_free and closed_legal
    assert [(p.volume, p.obstacles) for p in free] == [(46, 0)]
    assert [(p.volume, p.obstacles) for p in legal] == [(46, 3), (44, 0)]
    for point in legal:
        assert point.obstacles == unloading_obstacles(point.layout)
        assert sigma(inst, point.layout) >= -1e-9


def rollover_vehicle(inst, **changes):
    return truck(TARE[REFERENCE_TARE], rollover=True, width=inst.space[1], **changes)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("quantile", [0.3, 0.7])
def test_rollover_rows_hold_exactly_when_the_rollover_margin_reaches_the_level(seed, quantile):
    from legsafe import rollover_margin

    inst = small_instance(seed, 5, space=(6.0, 1.0, 1.5))
    v = rollover_vehicle(inst, deck_height=1.6, tare_cg_height=1.5, lateral_accel=0.45)  # tall, so that the rows bind
    model = GridModel(inst, v, DELTA)
    rng = np.random.default_rng(seed)
    layouts = [decode(inst, rng.random(n_keys(inst)), DELTA) for _ in range(200)]
    rhos = [rollover_margin(v, leg_moments(l.mass, *l.centre.T, l.rank, inst.n_stops)).min() for l in layouts]
    level = float(np.quantile(rhos, quantile))  # a level that some plans reach and others do not
    matrix, lower, _ = model._rollover_rows(level)
    held = 0
    for layout, rho in zip(layouts, rhos):
        x = np.zeros(2 * model.n)
        for k, lo, size in zip(layout.type, layout.lo, layout.size):
            x[model.index[(int(k), *(int(round(c / DELTA)) for c in lo), *(int(round(c / DELTA)) for c in size))]] += 1
        rows_hold = bool((matrix @ x >= lower - 1e-9).all())
        if abs(rho - level) > 1e-9:
            assert rows_hold == (rho >= level)
        held += rows_hold
    assert 0 < held < len(layouts)


def test_margin_constraint_with_rollover_makes_every_leg_legal_on_both_counts():
    from legsafe import rollover_margin

    inst = small_instance(2, 4, space=(6.0, 1.0, 1.5))
    v = rollover_vehicle(inst, deck_height=1.6, tare_cg_height=1.5, lateral_accel=0.45)
    loose, _ = GridModel(inst, truck(TARE[REFERENCE_TARE], width=inst.space[1]), DELTA).solve(0.0)
    solution, closed = GridModel(inst, v, DELTA).solve(0.0)
    assert closed and solution is not None
    lm = leg_moments(solution.layout.mass, *solution.layout.centre.T, solution.layout.rank, inst.n_stops)
    assert rollover_margin(v, lm).min() >= -1e-9 and axle_margin(v, lm).min() >= -1e-9
    assert solution.volume <= loose.volume
