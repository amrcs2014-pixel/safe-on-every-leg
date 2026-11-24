from typing import NamedTuple

import numpy as np

from .instances import Instance

try:
    from numba import njit
except ImportError:  # without numba, decode falls back to the reference implementation
    njit = None

EPS = 1e-9
FLOOR = 10**9


class Layout(NamedTuple):
    """One row per loaded box: minimum corner, extents, box type, delivery rank and mass."""

    lo: np.ndarray
    size: np.ndarray
    type: np.ndarray
    rank: np.ndarray
    mass: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return self.lo + self.size / 2


OFFSET_STEPS = 12  # default step of the free length at the front wall: this share of the length
MAX_OFFSET = 0.5   # largest free length at the front wall, as a share of the length


def n_keys(inst: Instance, extended: bool = True) -> int:
    """Length of a key vector: 2T for the plain decoder, 3T + 1 for the extended one."""
    return (3 if extended else 2) * len(inst.types) + extended


def level(key: float, top: int) -> int:
    """0 for the lower half of the key range, then 0..top in equal steps over the upper half."""
    return min(top, int(max(0.0, 2.0 * key - 1.0) * (top + 1)))


def level_key(value: int, top: int) -> float:
    """A key that `level` maps to `value`."""
    return 0.25 if value == 0 else 0.5 + (value + 0.5) / (2 * (top + 1))


def offset_levels(inst: Instance, offset_step: float | None = None) -> tuple[float, int]:
    """Step of the free length at the front wall and the largest number of steps."""
    step = offset_step or inst.space[0] / OFFSET_STEPS
    return step, int(MAX_OFFSET * inst.space[0] / step + EPS)


def decode_reference(inst: Instance, keys, offset_step: float | None = None) -> Layout:
    """Block building over guillotine residual spaces, in plain Python.

    This function defines the decoder; `decode` is a faster implementation that returns the same
    layout.

    keys[:T] order the box types, keys[T:2T] pick each type's first orientation. A block of one type
    fills the front-bottom-left corner of a space; what is left is cut into a space above the block,
    one beside it and one behind it. Boxes are fully supported, and a space above a block accepts
    only types that leave no later than that block.

    With 2T keys every box that fits is loaded, starting at the front wall. With 3T + 1 keys the plan
    may also leave cargo behind and stand clear of the front wall: keys[2T:3T] set how many boxes of
    each type stay behind, and keys[3T] sets the free length at the front wall in steps of
    `offset_step`. Both are zero for keys below one half.
    """
    types = inst.types
    n_types = len(types)
    keys = np.asarray(keys)
    extended = len(keys) == n_keys(inst, True)
    assert extended or len(keys) == n_keys(inst, False), "2T or 3T + 1 keys expected"
    order = np.argsort(keys[:n_types], kind="stable")
    orientations = [t.orientations() for t in types]
    remaining = [t.count for t in types]
    offset = 0.0
    if extended:
        remaining = [t.count - level(keys[2 * n_types + i], t.count) for i, t in enumerate(types)]
        step, top = offset_levels(inst, offset_step)
        offset = step * level(keys[3 * n_types], top)
    payload_left = inst.payload
    smallest = min(min(t.dims) for t in types)

    blocks = []
    stack = [(offset, 0.0, 0.0, inst.space[0] - offset, inst.space[1], inst.space[2], FLOOR)]
    while stack:
        x, y, z, dx, dy, dz, base_rank = stack.pop()
        chosen = None
        for t in order:
            box = types[t]
            if remaining[t] == 0 or box.rank > base_rank:
                continue
            cap = remaining[t] if box.mass <= 0 else min(remaining[t], int(payload_left / box.mass + EPS))
            if cap == 0:
                continue
            options = orientations[t]
            first = int(keys[n_types + t] * len(options)) % len(options)
            for j in range(len(options)):
                l, w, h = options[(first + j) % len(options)]
                if l <= dx + EPS and w <= dy + EPS and h <= dz + EPS:
                    chosen = (t, cap, l, w, h)
                    break
            if chosen:
                break
        if not chosen:
            continue

        t, cap, l, w, h = chosen
        nz = min(int((dz + EPS) / h), cap) if types[t].stackable else 1
        ny = min(int((dy + EPS) / w), cap // nz)
        nx = min(int((dx + EPS) / l), cap // (nz * ny))
        bx, by, bz = nx * l, ny * w, nz * h
        blocks.append((t, x, y, z, l, w, h, nx, ny, nz))
        remaining[t] -= nx * ny * nz
        payload_left -= nx * ny * nz * types[t].mass

        if dx - bx >= dy - by:
            behind = (x + bx, y, z, dx - bx, dy, dz, base_rank)
            beside = (x, y + by, z, bx, dy - by, dz, base_rank)
        else:
            behind = (x + bx, y, z, dx - bx, by, dz, base_rank)
            beside = (x, y + by, z, dx, dy - by, dz, base_rank)
        above = (x, y, z + bz, bx, by, dz - bz, types[t].rank)
        candidates = [behind, beside] + ([above] if types[t].stackable else [])
        stack.extend(s for s in candidates if min(s[3], s[4], s[5]) >= smallest - EPS)

    return _expand_reference(inst, blocks)


def _expand_reference(inst: Instance, blocks) -> Layout:
    lo, size, kind = [], [], []
    for t, x, y, z, l, w, h, nx, ny, nz in blocks:
        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        corners = np.column_stack([x + ix.ravel() * l, y + iy.ravel() * w, z + iz.ravel() * h])
        lo.append(corners)
        size.append(np.tile((l, w, h), (len(corners), 1)))
        kind.append(np.full(len(corners), t))
    if not lo:
        empty = np.empty((0, 3))
        return Layout(empty, empty, np.empty(0, int), np.empty(0, int), np.empty(0))
    kind = np.concatenate(kind)
    rank = np.array([t.rank for t in inst.types])[kind]
    mass = np.array([t.mass for t in inst.types])[kind]
    return Layout(np.concatenate(lo), np.concatenate(size), kind, rank, mass)


class _Tables(NamedTuple):
    """The data of an instance that the decoder reads, as arrays."""

    options: np.ndarray  # (T, 6, 3): extents of the orientations of each type, in the order they are tried
    n_options: np.ndarray
    count: np.ndarray
    mass: np.ndarray
    rank: np.ndarray
    stackable: np.ndarray
    space: np.ndarray
    smallest: float


_TABLES: dict[int, tuple[Instance, _Tables]] = {}


def _tables(inst: Instance) -> _Tables:
    entry = _TABLES.get(id(inst))
    if entry is not None and entry[0] is inst:
        return entry[1]
    types = inst.types
    options = np.zeros((len(types), 6, 3))
    n_options = np.empty(len(types), np.int64)
    for i, box in enumerate(types):
        found = box.orientations()
        n_options[i] = len(found)
        options[i, :len(found)] = found
    tables = _Tables(
        options, n_options,
        np.array([t.count for t in types], np.int64), np.array([t.mass for t in types], float),
        np.array([t.rank for t in types], np.int64), np.array([t.stackable for t in types], np.bool_),
        np.array(inst.space, float), float(min(min(t.dims) for t in types)),
    )
    if len(_TABLES) >= 256:
        _TABLES.clear()
    _TABLES[id(inst)] = (inst, tables)  # holding the instance keeps its id from being reused
    return tables


def _build(order, first, remaining, payload_left, options, n_options, mass, rank, stackable, space, offset, smallest):
    """The blocks of a plan, one row each: type, corner, box extents and number of boxes per axis.

    Same steps, in the same order and with the same arithmetic, as the loop of `decode_reference`.
    """
    n_types = len(order)
    most = remaining.sum()  # a block holds at least one box
    blocks = np.empty((most, 10))
    stack = np.empty((2 * most + 3, 7))  # a block takes one space and leaves at most three
    stack[0, 0], stack[0, 1], stack[0, 2] = offset, 0.0, 0.0
    stack[0, 3], stack[0, 4], stack[0, 5], stack[0, 6] = space[0] - offset, space[1], space[2], FLOOR
    top, made = 1, 0
    while top > 0:
        top -= 1
        x, y, z = stack[top, 0], stack[top, 1], stack[top, 2]
        dx, dy, dz, base_rank = stack[top, 3], stack[top, 4], stack[top, 5], stack[top, 6]
        chosen, cap = -1, 0
        l = w = h = 0.0
        for i in range(n_types):
            t = order[i]
            if remaining[t] == 0 or rank[t] > base_rank:
                continue
            if mass[t] <= 0 or payload_left == np.inf:
                cap = remaining[t]
            else:
                cap = min(remaining[t], int(payload_left / mass[t] + EPS))
            if cap == 0:
                continue
            n = n_options[t]
            for j in range(n):
                o = (first[t] + j) % n
                if options[t, o, 0] <= dx + EPS and options[t, o, 1] <= dy + EPS and options[t, o, 2] <= dz + EPS:
                    chosen = t
                    l, w, h = options[t, o, 0], options[t, o, 1], options[t, o, 2]
                    break
            if chosen >= 0:
                break
        if chosen < 0:
            continue

        t = chosen
        nz = min(int((dz + EPS) / h), cap) if stackable[t] else 1
        ny = min(int((dy + EPS) / w), cap // nz)
        nx = min(int((dx + EPS) / l), cap // (nz * ny))
        bx, by, bz = nx * l, ny * w, nz * h
        blocks[made, 0], blocks[made, 1], blocks[made, 2], blocks[made, 3] = t, x, y, z
        blocks[made, 4], blocks[made, 5], blocks[made, 6] = l, w, h
        blocks[made, 7], blocks[made, 8], blocks[made, 9] = nx, ny, nz
        made += 1
        remaining[t] -= nx * ny * nz
        payload_left -= nx * ny * nz * mass[t]

        # Spaces behind and beside the block, then the one above it, which is therefore filled first.
        if dx - bx >= dy - by:
            behind_dy, beside_dx = dy, bx
        else:
            behind_dy, beside_dx = by, dx
        if min(dx - bx, behind_dy, dz) >= smallest - EPS:
            stack[top, 0], stack[top, 1], stack[top, 2] = x + bx, y, z
            stack[top, 3], stack[top, 4], stack[top, 5], stack[top, 6] = dx - bx, behind_dy, dz, base_rank
            top += 1
        if min(beside_dx, dy - by, dz) >= smallest - EPS:
            stack[top, 0], stack[top, 1], stack[top, 2] = x, y + by, z
            stack[top, 3], stack[top, 4], stack[top, 5], stack[top, 6] = beside_dx, dy - by, dz, base_rank
            top += 1
        if stackable[t] and min(bx, by, dz - bz) >= smallest - EPS:
            stack[top, 0], stack[top, 1], stack[top, 2] = x, y, z + bz
            stack[top, 3], stack[top, 4], stack[top, 5], stack[top, 6] = bx, by, dz - bz, rank[t]
            top += 1
    return blocks[:made]


def _expand(blocks, type_rank, type_mass):
    """One row per box from one row per block, in the order of `_expand_reference`."""
    n = 0
    for b in range(len(blocks)):
        n += int(blocks[b, 7]) * int(blocks[b, 8]) * int(blocks[b, 9])
    lo, size = np.empty((n, 3)), np.empty((n, 3))
    kind, rank, mass = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n)
    i = 0
    for b in range(len(blocks)):
        t = int(blocks[b, 0])
        for ix in range(int(blocks[b, 7])):
            for iy in range(int(blocks[b, 8])):
                for iz in range(int(blocks[b, 9])):
                    lo[i, 0] = blocks[b, 1] + ix * blocks[b, 4]
                    lo[i, 1] = blocks[b, 2] + iy * blocks[b, 5]
                    lo[i, 2] = blocks[b, 3] + iz * blocks[b, 6]
                    size[i, 0], size[i, 1], size[i, 2] = blocks[b, 4], blocks[b, 5], blocks[b, 6]
                    kind[i], rank[i], mass[i] = t, type_rank[t], type_mass[t]
                    i += 1
    return lo, size, kind, rank, mass


if njit is not None:
    _build = njit(cache=True)(_build)
    _expand = njit(cache=True)(_expand)


def decode(inst: Instance, keys, offset_step: float | None = None) -> Layout:
    """Block building over guillotine residual spaces.

    keys[:T] order the box types, keys[T:2T] pick each type's first orientation. A block of one type
    fills the front-bottom-left corner of a space; what is left is cut into a space above the block,
    one beside it and one behind it. Boxes are fully supported, and a space above a block accepts
    only types that leave no later than that block.

    With 2T keys every box that fits is loaded, starting at the front wall. With 3T + 1 keys the plan
    may also leave cargo behind and stand clear of the front wall: keys[2T:3T] set how many boxes of
    each type stay behind, and keys[3T] sets the free length at the front wall in steps of
    `offset_step`. Both are zero for keys below one half.

    Returns the layout of `decode_reference`, which states the procedure step by step.
    """
    if njit is None:
        return decode_reference(inst, keys, offset_step)
    tables = _tables(inst)
    n_types = len(inst.types)
    keys = np.asarray(keys, dtype=float)
    extended = len(keys) == n_keys(inst, True)
    assert extended or len(keys) == n_keys(inst, False), "2T or 3T + 1 keys expected"
    order = np.argsort(keys[:n_types], kind="stable")
    first = (keys[n_types:2 * n_types] * tables.n_options).astype(np.int64) % tables.n_options
    remaining = tables.count.copy()
    offset = 0.0
    if extended:
        behind = (np.maximum(0.0, 2.0 * keys[2 * n_types:3 * n_types] - 1.0) * (tables.count + 1)).astype(np.int64)
        remaining -= np.minimum(tables.count, behind)
        step, top = offset_levels(inst, offset_step)
        offset = step * level(keys[3 * n_types], top)
    blocks = _build(order, first, remaining, float(inst.payload), tables.options, tables.n_options, tables.mass,
                    tables.rank, tables.stackable, tables.space, float(offset), tables.smallest)
    return Layout(*_expand(blocks, tables.rank, tables.mass))
