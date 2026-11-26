import numpy as np
import pytest

from experiments import final
from experiments.common import instance
from legsafe import decode, n_keys
from legsafe.nsga2 import nondominated_fronts

JOB = (2, 20, (2, 2))


def small_run(objective, algorithm, number=0):
    return final.run((JOB, None, number, objective, algorithm), per_type=130, fill=40, n_runs=2)


def test_reference_vehicle_has_the_rollover_data():
    v = final.VEHICLE
    assert v.track == 1.818 and v.deck_height == 1.15 and v.tare_cg_height == 1.0
    assert v.lateral_accel == pytest.approx(4 / 9.81) and v.tare == pytest.approx(2.993 + 1.543 + 0.364)


@pytest.mark.parametrize("objective", final.OBJECTIVES)
@pytest.mark.parametrize("algorithm", list(final.ALGORITHMS))
def test_every_search_writes_an_archive_with_consistent_measures(objective, algorithm):
    rows = small_run(objective, algorithm)
    assert rows and {r["objective"] for r in rows} == {objective} and {r["algorithm"] for r in rows} == {algorithm}
    assert [r["plan"] for r in rows] == list(range(len(rows)))
    for r in rows:
        assert 0 < r["utilisation"] <= 1 and r["obstacles"] >= 0 and r["boxes"] > 0
        assert r["sigma"] == pytest.approx(min(r["axle"], r["rollover"]))
        assert r["sigma"] <= r["sigma_departure"] + 1e-12 and r["axle"] <= r["axle_departure"] + 1e-12
        assert r["mass"] <= 10.1 + 1e-9 and all(0 <= r[k] <= 0.5 + 1e-9 for k in ("rd_x", "rd_y", "rd_z"))
    f = np.array([[-r["utilisation"], r["obstacles"], final.third_objective(r, objective)] for r in rows])
    assert len(nondominated_fronts(f)[0]) == len(f) == len(np.unique(f, axis=0))


def test_third_objective_follows_its_definition():
    m = dict(sigma=-0.2, sigma_departure=0.3, rd_x=0.1, rd_y=0.2, rd_z=0.3)
    assert final.third_objective(m, "every_leg") == pytest.approx(0.2)
    assert final.third_objective(m, "departure") == pytest.approx(-final.CAP)
    assert final.third_objective(m, "centre") == pytest.approx(0.2)


def test_runs_are_repeatable_and_differ_by_run_number():
    def plans(rows):
        return [(r["utilisation"], r["obstacles"], r["sigma"]) for r in rows]

    assert plans(small_run("every_leg", "nsga2")) == plans(small_run("every_leg", "nsga2"))
    assert plans(small_run("every_leg", "nsga2")) != plans(small_run("every_leg", "nsga2", number=1))


def test_measures_of_an_empty_plan_are_those_of_the_empty_vehicle():
    inst, _ = instance(JOB, 1)
    n_types = len(inst.types)
    keys = np.concatenate([np.zeros(2 * n_types), np.full(n_types, 0.9999), [0.0]])  # every box is left behind
    layout = decode(inst, keys)
    assert len(layout.lo) == 0 and len(keys) == n_keys(inst)
    m = final.measures(inst, layout)
    assert m["utilisation"] == 0 and m["mass"] == 0 and m["boxes"] == 0 and m["obstacles"] == 0
    assert m["sigma"] == pytest.approx(min(m["axle"], m["rollover"])) and m["rd_x"] == 0.5


def test_a_budget_smaller_than_the_search_for_fill_is_refused():
    inst, (rng,) = instance(JOB, 1)
    with pytest.raises(ValueError):
        final.search(inst, rng, per_type=400, fill=1000)


def test_search_reports_generations_in_evaluations_of_the_whole_run():
    inst, (rng,) = instance(JOB, 1)
    seen = []
    final.search(inst, rng, per_type=130, fill=40, observer=lambda generation, front, archive: seen.append(generation))
    total = round(130 * len(inst.types) / final.POP_SIZE)
    first = max(1, round(40 * len(inst.types) / final.POP_SIZE))
    assert seen == list(range(first + 2, total + 1))


def test_vehicle_variants_keep_the_tare_and_its_centre_and_move_the_rear_axle():
    reference = final.VEHICLES["reference"]
    centre = reference.a_front + reference.wheelbase * reference.tare_rear / reference.tare
    from experiments.common import PAYLOAD

    slack = [1 - load / limit for load, limit in zip(final.axle_loads_uniform(reference, PAYLOAD), (reference.max_front, reference.max_rear))]
    for overhang in (1.0, 0.0):
        v = final.VEHICLES[f"overhang_{overhang}"]
        assert v.length - v.a_rear == pytest.approx(overhang)
        assert v.tare == pytest.approx(reference.tare) and v.a_front == reference.a_front
        assert v.a_front + v.wheelbase * v.tare_rear / v.tare == pytest.approx(centre)
        assert (v.track, v.deck_height) == (reference.track, reference.deck_height)
        loads = final.axle_loads_uniform(v, PAYLOAD)
        assert [1 - load / limit for load, limit in zip(loads, (v.max_front, v.max_rear))] == pytest.approx(slack)
        assert v.tare_front < v.max_front and v.tare_rear < v.max_rear  # the empty vehicle is legal
    high = final.VEHICLES["high"]
    assert (high.deck_height, high.tare_cg_height, high.a_rear) == (1.30, 1.20, reference.a_rear)


def test_rear_share_is_the_rear_load_over_the_gross_weight():
    from legsafe import leg_moments, support_loads

    inst, (rng,) = instance(JOB, 1)
    layout = decode(inst, rng.random(n_keys(inst)))
    for name, v in final.VEHICLES.items():
        m = final.measures(inst, layout, v)
        front, rear = support_loads(v, leg_moments(layout.mass, *layout.centre.T, layout.rank, inst.n_stops))
        assert m["rear_share"] == pytest.approx((rear / (front + rear)).min())
        assert m["rear_share_departure"] == pytest.approx(rear[0] / (front[0] + rear[0]))
        assert front[0] + rear[0] == pytest.approx(v.tare + layout.mass.sum())
