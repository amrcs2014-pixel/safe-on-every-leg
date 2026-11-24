import numpy as np
import pytest

from legsafe import (
    Vehicle,
    axle_margin,
    centre_deviation,
    leg_moments,
    rollover_margin,
    support_loads,
    worst_leg_margin,
)

# Made-up vehicle for testing the formulas; not data for the paper.
TEST_VEHICLE = Vehicle(
    length=7.0, width=2.4, height=2.4,
    a_front=-1.0, a_rear=4.0,
    tare_front=3.0, tare_rear=2.0,
    max_front=7.0, max_rear=11.5,
    track=2.0, deck_height=1.2, tare_cg_height=1.0,
    lateral_accel=0.5, min_share=0.25, min_share_support="rear",
)


def random_load(rng, n_stops, n=30, vehicle=TEST_VEHICLE):
    mass = rng.uniform(0.02, 1.5, n)
    cx = rng.uniform(0, vehicle.length, n)
    cy = rng.uniform(0, vehicle.width, n)
    cz = rng.uniform(0, vehicle.height, n)
    rank = rng.integers(1, n_stops + 1, n)
    return mass, cx, cy, cz, rank


@pytest.mark.parametrize("seed", range(20))
def test_leg_moments_match_direct_sums(seed):
    rng = np.random.default_rng(seed)
    n_stops = int(rng.integers(1, 9))
    mass, cx, cy, cz, rank = random_load(rng, n_stops)
    lm = leg_moments(mass, cx, cy, cz, rank, n_stops)
    for k in range(n_stops):
        on_board = rank > k
        assert lm.mass[k] == pytest.approx(mass[on_board].sum())
        assert lm.mx[k] == pytest.approx((mass * cx)[on_board].sum())
        assert lm.my[k] == pytest.approx((mass * cy)[on_board].sum())
        assert lm.mz[k] == pytest.approx((mass * cz)[on_board].sum())


def test_support_loads_carry_total_weight():
    rng = np.random.default_rng(1)
    lm = leg_moments(*random_load(rng, 5), 5)
    front, rear = support_loads(TEST_VEHICLE, lm)
    assert front + rear == pytest.approx(TEST_VEHICLE.tare + lm.mass)


def test_unloading_from_overhang_raises_front_load():
    # 4 t at x=1 (second drop) and 2 t at x=5.5, behind the rear support (first drop).
    lm = leg_moments([4.0, 2.0], [1.0, 5.5], [1.2, 1.2], [0.5, 0.5], [2, 1], 2)
    front, rear = support_loads(TEST_VEHICLE, lm)
    assert front == pytest.approx([4.8, 5.4])
    assert rear == pytest.approx([6.2, 3.6])


@pytest.mark.parametrize("seed", range(50))
def test_proposition_1_load_change_at_a_stop(seed):
    rng = np.random.default_rng(seed)
    n_stops = 6
    mass, cx, cy, cz, rank = random_load(rng, n_stops)
    v = TEST_VEHICLE
    front, rear = support_loads(v, leg_moments(mass, cx, cy, cz, rank, n_stops))
    for k in range(n_stops - 1):
        unloaded = rank == k + 1
        if not unloaded.any():
            continue
        m_u = mass[unloaded].sum()
        x_u = (mass * cx)[unloaded].sum() / m_u
        assert front[k + 1] - front[k] == pytest.approx(m_u / v.wheelbase * (x_u - v.a_rear))
        assert rear[k + 1] - rear[k] == pytest.approx(m_u / v.wheelbase * (v.a_front - x_u))
        assert not (front[k + 1] > front[k] and rear[k + 1] > rear[k])


@pytest.mark.parametrize("seed", range(50))
def test_corollary_no_overhang_means_loads_never_rise(seed):
    rng = np.random.default_rng(seed)
    n_stops = 6
    mass, cx, cy, cz, rank = random_load(rng, n_stops)
    v = TEST_VEHICLE
    cx = rng.uniform(max(0.0, v.a_front), min(v.length, v.a_rear), len(cx))
    front, rear = support_loads(v, leg_moments(mass, cx, cy, cz, rank, n_stops))
    assert np.all(np.diff(front) <= 1e-12)
    assert np.all(np.diff(rear) <= 1e-12)


@pytest.mark.parametrize("seed", range(50))
def test_proposition_2_centre_of_gravity_shift(seed):
    rng = np.random.default_rng(seed)
    n_stops = 6
    mass, cx, cy, cz, rank = random_load(rng, n_stops)
    lm = leg_moments(mass, cx, cy, cz, rank, n_stops)
    xbar = lm.mx / lm.mass
    for k in range(n_stops - 1):
        unloaded = rank == k + 1
        if not unloaded.any() or lm.mass[k + 1] == 0:
            continue
        m_u = mass[unloaded].sum()
        x_u = (mass * cx)[unloaded].sum() / m_u
        assert xbar[k + 1] - xbar[k] == pytest.approx(m_u / lm.mass[k + 1] * (xbar[k] - x_u))


@pytest.mark.parametrize("seed", range(50))
def test_proposition_2_forward_drift_when_ranks_are_ordered_along_x(seed):
    rng = np.random.default_rng(seed)
    n_stops = 6
    mass, cx, cy, cz, rank = random_load(rng, n_stops)
    order = np.argsort(rank, kind="stable")
    cx_sorted = np.empty_like(cx)
    cx_sorted[order] = np.sort(cx)[::-1]  # earliest drops get the rearmost centroids
    lm = leg_moments(mass, cx_sorted, cy, cz, rank, n_stops)
    loaded = lm.mass > 0
    xbar = lm.mx[loaded] / lm.mass[loaded]
    assert np.all(np.diff(xbar) <= 1e-12)


@pytest.mark.parametrize("seed", range(50))
def test_corollary_min_share_slack_decomposition(seed):
    rng = np.random.default_rng(seed)
    n_stops = 6
    mass, cx, cy, cz, rank = random_load(rng, n_stops)
    v = TEST_VEHICLE
    lm = leg_moments(mass, cx, cy, cz, rank, n_stops)
    _, rear = support_loads(v, lm)
    share = (cx - v.a_front) / v.wheelbase
    for k in range(n_stops):
        on_board = rank > k
        slack = (v.tare_rear - v.min_share * v.tare) + (mass * (share - v.min_share))[on_board].sum()
        assert rear[k] - v.min_share * (v.tare + lm.mass[k]) == pytest.approx(slack)


@pytest.mark.parametrize("seed", range(20))
def test_rollover_margin_matches_static_stability_factor(seed):
    rng = np.random.default_rng(seed)
    n_stops = 4
    mass, cx, cy, cz, rank = random_load(rng, n_stops)
    v = TEST_VEHICLE
    lm = leg_moments(mass, cx, cy, cz, rank, n_stops)
    rho = rollover_margin(v, lm)
    for k in range(n_stops):
        on_board = rank > k
        total = v.tare + mass[on_board].sum()
        cg_height = (v.tare * v.tare_cg_height + (mass * (v.deck_height + cz))[on_board].sum()) / total
        lateral_offset = abs((mass * (cy - v.width / 2))[on_board].sum()) / total
        ssf = (v.track / 2 - lateral_offset) / cg_height
        assert rho[k] == pytest.approx(ssf / v.lateral_accel - 1)


def test_lower_centre_of_gravity_raises_rollover_margin():
    low = leg_moments([5.0], [2.0], [1.2], [0.4], [1], 1)
    high = leg_moments([5.0], [2.0], [1.2], [1.2], [1], 1)
    assert rollover_margin(TEST_VEHICLE, low)[0] > rollover_margin(TEST_VEHICLE, high)[0]


def test_worst_leg_margin_is_the_minimum_over_legs_and_limits():
    rng = np.random.default_rng(7)
    lm = leg_moments(*random_load(rng, 5), 5)
    result = worst_leg_margin(TEST_VEHICLE, lm)
    per_leg = np.minimum(axle_margin(TEST_VEHICLE, lm), rollover_margin(TEST_VEHICLE, lm))
    assert result.sigma == pytest.approx(per_leg.min())
    assert result.worst_leg == int(per_leg.argmin())


def test_centre_deviation_is_zero_for_a_centred_load():
    v = TEST_VEHICLE
    dev = centre_deviation(v, [1.0, 1.0], [v.length / 2] * 2, [v.width / 2] * 2, [v.height / 2] * 2)
    assert dev == pytest.approx((0.0, 0.0, 0.0))
