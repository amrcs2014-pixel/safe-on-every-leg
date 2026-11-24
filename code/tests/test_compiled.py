"""The compiled decoder, obstacle count and leg sums must return exactly what the reference
implementations return: same arrays, bit for bit."""
from pathlib import Path

import numpy as np
import pytest

from experiments.small import DELTA, small_instance
from legsafe import decode, extend, leg_moments, n_keys, read_br, unloading_obstacles
from legsafe.compiled import warm_up
from legsafe.decoder import decode_reference
from legsafe.legs import leg_moments_reference
from legsafe.objectives import unloading_obstacles_reference

DATA = Path(__file__).resolve().parents[2] / "data" / "br" / "orlib"


def benchmark(br_class, index):
    raw = read_br(DATA / f"thpack{br_class}.txt")[index]
    return extend(raw, payload=10.1, tau=0.9, alpha=2, beta=5, p_nonstackable=0.1, rng=np.random.default_rng(index))


def key_vectors(inst, seed, count=40):
    """Plain and extended keys; some extended ones load everything, some leave everything behind."""
    rng = np.random.default_rng(seed)
    n_types = len(inst.types)
    for trial in range(count):
        yield rng.random(n_keys(inst, extended=False))
        keys = rng.random(n_keys(inst, extended=True))
        if trial % 4 == 0:
            keys[2 * n_types:] = 0.0
        if trial % 20 == 1:
            keys[2 * n_types:3 * n_types] = 0.999
        yield keys


def assert_same(inst, keys, offset_step=None):
    fast, reference = decode(inst, keys, offset_step), decode_reference(inst, keys, offset_step)
    for a, b in zip(fast, reference):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    assert unloading_obstacles(fast) == unloading_obstacles_reference(reference)
    moments = leg_moments(fast.mass, *fast.centre.T, fast.rank, inst.n_stops)
    expected = leg_moments_reference(reference.mass, *reference.centre.T, reference.rank, inst.n_stops)
    for a, b in zip(moments, expected):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("br_class", range(1, 8))
def test_compiled_evaluation_equals_the_reference_on_benchmark_instances(br_class):
    inst = benchmark(br_class, 3)
    for keys in key_vectors(inst, br_class):
        assert_same(inst, keys)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_compiled_evaluation_equals_the_reference_on_grid_instances(seed):
    inst = small_instance(seed, 6, space=(6.0, 1.0, 1.5))
    for keys in key_vectors(inst, seed):
        assert_same(inst, keys, DELTA)


def test_leg_sums_refuse_a_rank_beyond_the_last_stop():
    with pytest.raises(ValueError):
        leg_moments([1.0], [0.5], [0.5], [0.5], [3], n_stops=2)


def test_warm_up_runs():
    warm_up()
