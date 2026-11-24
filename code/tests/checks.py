import numpy as np
import pytest

EPS = 1e-7


def assert_feasible(inst, layout):
    """Independent check of a layout: containment, counts, payload, orientations, overlap, support and stacking rules."""
    lo, hi = layout.lo, layout.lo + layout.size
    n = len(lo)
    assert (lo >= -EPS).all() and (hi <= np.array(inst.space) + EPS).all()
    assert layout.mass.sum() <= inst.payload + EPS
    assert all((layout.type == t).sum() <= box.count for t, box in enumerate(inst.types))
    for i in range(n):
        assert tuple(round(v, 9) for v in layout.size[i]) in [
            tuple(round(v, 9) for v in o) for o in inst.types[layout.type[i]].orientations()
        ]

    overlap = np.ones((n, n), bool)
    for axis in range(3):
        overlap &= (lo[None, :, axis] < hi[:, None, axis] - EPS) & (lo[:, None, axis] < hi[None, :, axis] - EPS)
    np.fill_diagonal(overlap, False)
    assert not overlap.any()

    stackable = np.array([t.stackable for t in inst.types])[layout.type]
    for i in np.flatnonzero(lo[:, 2] > EPS):
        below = np.abs(hi[:, 2] - lo[i, 2]) < EPS
        wx = np.clip(np.minimum(hi[:, 0], hi[i, 0]) - np.maximum(lo[:, 0], lo[i, 0]), 0, None)
        wy = np.clip(np.minimum(hi[:, 1], hi[i, 1]) - np.maximum(lo[:, 1], lo[i, 1]), 0, None)
        area = np.where(below, wx * wy, 0.0)
        supporters = area > EPS
        assert area.sum() == pytest.approx(layout.size[i, 0] * layout.size[i, 1])
        assert (layout.rank[supporters] >= layout.rank[i]).all()
        assert stackable[supporters].all()
