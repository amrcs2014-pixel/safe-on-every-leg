"""Five further figures of the paper. Usage, from the code folder:
  python -m experiments.figures_extra [mechanism] [plans] [price] [exact] [algorithms]
Without arguments all five are drawn into manuscript/figures/.

Colours: the first slots of the categorical palette of the dataviz reference (blue, orange, aqua),
which separate for colour-blind readers when no more than three are used together; the delivery
stops of a plan are shown on a sequential blue scale, because their order is what matters.
"""
import sys

import numpy as np
import pandas as pd

from .common import BETAS, read
from .paper import EXAMPLE_CLASS, EXAMPLE_INSTANCE, FIGURES

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
WARNING_INK = "#a8441b"  # dark orange for lettering: contrast 6:1 on white, where the series orange has 3:1
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def setup():
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "lines.linewidth": 1.2})
    FIGURES.mkdir(parents=True, exist_ok=True)
    return plt


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    print(f"--- figures/{name}.pdf")


# --------------------------------------------------------------------------------------------
def mechanism():
    """Side view of the reference truck with an illustrative load, before and after the first stop."""
    plt = setup()
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    from .final import VEHICLE as v

    length, height = v.length, v.height
    # Illustrative load: three customers, the first one nearest the door (masses in t, centres in m).
    loads = [(1, 3.5, 3.78, 5.87), (2, 3.3, 1.75, 3.78), (3, 2.6, 0.0, 1.75)]
    shades = {1: "#cfe0f5", 2: "#7fb0e8", 3: "#2a78d6"}

    def axle_loads(on_board):
        cargo = [(m, (lo + hi) / 2) for stop, m, lo, hi in loads if stop in on_board]
        front = v.tare_front + sum(m * (v.a_rear - x) for m, x in cargo) / v.wheelbase
        rear = v.tare_rear + sum(m * (x - v.a_front) for m, x in cargo) / v.wheelbase
        return front, rear

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.2), sharey=True)
    for ax, (title, on_board) in zip(axes, (("a", {1, 2, 3}), ("b", {2, 3}))):
        ax.add_patch(Rectangle((-2.1, 0.05), 1.6, 1.9, facecolor="white", edgecolor=MUTED, linewidth=0.8))  # cab
        ax.add_patch(Rectangle((-0.05, 0), length + 0.1, height + 0.05, facecolor="none", edgecolor=MUTED, linewidth=0.8))
        ax.plot([-2.3, length + 0.2], [-0.12, -0.12], color=MUTED, linewidth=1.6)  # chassis
        for stop, m, lo, hi in loads:
            if stop in on_board:
                ax.add_patch(Rectangle((lo + 0.04, 0.02), hi - lo - 0.08, 1.9 if stop != 3 else 2.1,
                                       facecolor=shades[stop], edgecolor="white", linewidth=1.0))
                ax.text((lo + hi) / 2, 1.0, f"Stop {stop}\n{m:.1f} t", ha="center", va="center", fontsize=8,
                        color="white" if stop == 3 else INK)
        for x in (v.a_front, v.a_rear):
            ax.add_patch(Circle((x, -0.45), 0.33, facecolor="white", edgecolor=INK, linewidth=1.0))
        front, rear = axle_loads(on_board)
        for x, load, limit, name in ((v.a_front, front, v.max_front, "front"), (v.a_rear, rear, v.max_rear, "rear")):
            over = load > limit + 1e-9
            colour = WARNING_INK if over else INK
            ax.add_patch(FancyArrowPatch((x, -1.0), (x, -1.75), arrowstyle="-|>", mutation_scale=8, color=colour, linewidth=1.2))
            ax.text(x, -1.95, f"{name.capitalize()} axle\n{load:.2f} t ({100 * load / limit:.0f}%)", ha="center", va="top",
                    fontsize=8, color=colour)
        ax.annotate("", xy=(length, height + 0.35), xytext=(v.a_rear, height + 0.35),
                    arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=0.7))
        ax.text((length + v.a_rear) / 2, height + 0.45, "overhang", ha="center", va="bottom", fontsize=8, color=MUTED)
        ax.text(length + 0.15, 1.0, "door", rotation=90, va="center", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left", fontweight="bold")
        ax.set_xlim(-2.5, length + 0.6)
        ax.set_ylim(-2.9, height + 0.9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    save(fig, "mechanism")
    for on_board in ({1, 2, 3}, {2, 3}, {3}):
        f, r = axle_loads(on_board)
        print(f"  customers {sorted(on_board)} on board: front {f:.3f} t ({100 * f / v.max_front:.1f}%), rear {r:.3f} t ({100 * r / v.max_rear:.1f}%)")
    plt.close(fig)


# --------------------------------------------------------------------------------------------
def plans():
    """3D views of two plans for the example case: legal at departure and failing later, and legal on every leg."""
    plt = setup()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from legsafe import decode

    from .common import instance
    from .final import measures, search

    job = (EXAMPLE_CLASS, EXAMPLE_INSTANCE, BETAS[0])
    found = {}
    for objective in ("centre", "every_leg"):
        inst, rngs = instance(job, 2)
        keys, _ = search(inst, rngs[0], objective)
        found[objective] = [(decode(inst, k), measures(inst, decode(inst, k))) for k in keys]
    inst, _ = instance(job, 2)
    failing = max(((l, m) for l, m in found["centre"] if m["sigma_departure"] >= 0 and m["sigma"] < 0), key=lambda p: p[1]["utilisation"])
    legal = max(((l, m) for l, m in found["every_leg"] if m["sigma"] >= 0 and m["obstacles"] <= failing[1]["obstacles"]),
                key=lambda p: p[1]["utilisation"])

    stops = inst.n_stops
    ramp = plt.get_cmap("Blues")
    colour_of = {s: ramp(0.25 + 0.7 * (s - 1) / max(1, stops - 1)) for s in range(1, stops + 1)}

    def cuboid(lo, size):
        x, y, z = lo
        dx, dy, dz = size
        p = np.array([[x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
                      [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [x, y + dy, z + dz]])
        return [p[[0, 1, 2, 3]], p[[4, 5, 6, 7]], p[[0, 1, 5, 4]], p[[2, 3, 7, 6]], p[[1, 2, 6, 5]], p[[0, 3, 7, 4]]]

    fig = plt.figure(figsize=(6.2, 2.6))
    panels = ((failing, "a"), (legal, "b"))
    for position, ((layout, m), title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 2, position, projection="3d")
        faces, colours = [], []
        for lo, size, rank in zip(layout.lo, layout.size, layout.rank):
            for face in cuboid(lo, size):
                faces.append(face)
                colours.append(colour_of[int(rank)])
        ax.add_collection3d(Poly3DCollection(faces, facecolors=colours, edgecolors="white", linewidths=0.4))
        ax.set_xlim(0, inst.space[0])
        ax.set_ylim(0, inst.space[1])
        ax.set_zlim(0, inst.space[2])
        ax.set_box_aspect(inst.space)
        ax.view_init(elev=24, azim=-58)
        ax.set_xticks(range(0, int(inst.space[0]) + 1))
        ax.set_yticks([0, round(inst.space[1], 2)])
        ax.set_zticks([0, 1, 2])
        ax.set_xlabel("Length (m), door at right", labelpad=-6, fontsize=8)
        ax.set_ylabel("Width", labelpad=-9, fontsize=8)
        ax.set_zlabel("Height", labelpad=-9, fontsize=8)
        ax.tick_params(labelsize=8, pad=-3)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("white")
            axis.pane.set_edgecolor(GRID)
        ax.set_title(title, fontsize=9, color=INK, loc="left", fontweight="bold", pad=-2)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colour_of[s], edgecolor="none") for s in range(1, stops + 1)]
    fig.legend(handles, [f"{s}" for s in range(1, stops + 1)], title="Delivery stop", loc="lower center", ncol=stops,
               fontsize=8, title_fontsize=8, bbox_to_anchor=(0.5, -0.04), handlelength=1.2, columnspacing=0.8)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.08, wspace=0.0)
    save(fig, "plans")
    for (layout, m), title in panels:
        print(f"  {title}: utilisation {m['utilisation']:.4f}, obstacles {m['obstacles']}, sigma {m['sigma']:.4f}, "
              f"at departure {m['sigma_departure']:.4f}, worst leg {m['worst_leg']}, boxes {m['boxes']}")
    plt.close(fig)


# --------------------------------------------------------------------------------------------
def price():
    """Utilisation lost by requiring a margin on every leg, against the margin, by limit on unloading obstacles."""
    plt = setup()
    from .final_summary import CASE, complete_replicates

    d = complete_replicates(read("final_main.csv"))
    d = d[d["objective"] == "every_leg"]
    levels = np.round(np.arange(0, 0.0501, 0.005), 4)
    limits = (("Any number of obstacles", None, BLUE, "o", "-"), ("At most 20 obstacles", 20, ORANGE, "s", "--"),
              ("No obstacles", 0, AQUA, "^", ":"))
    rows = []
    for key, run in d.groupby(CASE + ["run"]):
        for label, limit, *_ in limits:
            plans = run if limit is None else run[run["obstacles"] <= limit]
            if plans.empty:
                continue
            best = plans["utilisation"].max()
            for s in levels:
                ok = plans[plans["sigma"] >= s - 1e-12]
                if len(ok):
                    rows.append(dict(zip(CASE + ["run"], key), limit=label, level=s, cost=100 * (best - ok["utilisation"].max())))
    cost = pd.DataFrame(rows).groupby(CASE + ["limit", "level"])["cost"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    for label, _, colour, marker, style in limits:
        part = cost[cost["limit"] == label].groupby("level")["cost"]
        mean, q1, q3 = part.mean(), part.quantile(0.25), part.quantile(0.75)
        ax.fill_between(100 * mean.index, q1, q3, color=colour, alpha=0.10, linewidth=0)
        ax.plot(100 * mean.index, mean, color=colour, marker=marker, linestyle=style, markersize=3.5, label=label)
        ax.text(100 * mean.index[-1] + 0.12, mean.iloc[-1], label, color=INK, fontsize=8, va="center")
        print(f"  {label}: cost at 0% {mean.iloc[0]:.2f}, at 5% {mean.iloc[-1]:.2f} (quartiles {q1.iloc[-1]:.2f}-{q3.iloc[-1]:.2f})")
    ax.set_xlabel(r"Required margin on every leg, $\sigma \geq$ (%)")
    ax.set_ylabel("Utilisation lost (points)")
    ax.set_xlim(0, 5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    fig.tight_layout()
    save(fig, "price")
    plt.close(fig)


# --------------------------------------------------------------------------------------------
def exact():
    """Exact front against what the two decoders reach, for one small instance."""
    plt = setup()
    key = dict(space="narrow", n_types=5, seed=4)
    margin = "0.05"
    e, reach, ext = read("e1_exact.csv"), read("e1_reach.csv"), read("e1_extended.csv")

    def pick(frame):
        mask = np.ones(len(frame), bool)
        for k, v in key.items():
            mask &= frame[k].astype(str).to_numpy() == str(v)
        return frame[mask]

    exact_points = pick(e)
    exact_points = exact_points[exact_points["margin"].astype(str) == margin][["volume", "obstacles"]]
    cells = int(pick(e)["cells"].iloc[0])
    plain = pick(reach)
    plain = plain[plain["margin"].astype(str) == margin][["volume", "obstacles"]]
    extended = pick(ext)
    extended = extended[extended["sigma"] >= 0.05 - 1e-9][["volume", "obstacles"]]

    def front(points):
        points = points.drop_duplicates().sort_values(["obstacles", "volume"], ascending=[True, False])
        kept, best = [], -1
        for o, v in zip(points["obstacles"], points["volume"]):
            if v > best:
                kept.append((o, v))
                best = v
        return np.array(kept)

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for points, label, colour, marker, size in ((front(exact_points), "Exact front", INK, "s", 30),
                                                (front(extended), "Extended decoder, search", BLUE, "o", 18),
                                                (front(plain), "Plain decoder, all its plans", ORANGE, "^", 18)):
        x, y = points[:, 0], 100 * points[:, 1] / cells
        ax.step(np.append(x, x[-1] + 2), np.append(y, y[-1]), where="post", color=colour, linewidth=0.9)
        ax.scatter(x, y, s=size, marker=marker, color=colour, edgecolor="white", linewidth=0.6, zorder=3, label=label)
        print(f"  {label}: " + ", ".join(f"({int(o)} obstacles, {100 * v / cells:.1f}%)" for o, v in points))
    ax.set_xlabel("Unloading obstacles")
    ax.set_ylabel("Utilisation (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    save(fig, "exact")
    plt.close(fig)


# --------------------------------------------------------------------------------------------
def algorithms():
    """Hypervolume and IGD+ of the four searches, one point per case (mean over its runs)."""
    plt = setup()
    from .final_summary import CASE, complete_replicates, indicators
    from .paper import ALGORITHM_NAMES

    scores = indicators(complete_replicates(read("final_algorithms.csv")))
    means = scores.groupby(CASE + ["algorithm"])[["hypervolume", "igd_plus"]].mean().reset_index()
    order = list(ALGORITHM_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5))
    rng = np.random.default_rng(0)
    for ax, (column, label, better) in zip(axes, (("hypervolume", "Hypervolume (share of reference)", "a"),
                                                  ("igd_plus", r"IGD$^{+}$", "b"))):
        data = [means.loc[means["algorithm"] == a, column].to_numpy() for a in order]
        ax.boxplot(data, widths=0.5, showfliers=False, medianprops=dict(color=INK, linewidth=1.6),
                   boxprops=dict(color=MUTED, linewidth=0.8), whiskerprops=dict(color=MUTED, linewidth=0.8),
                   capprops=dict(color=MUTED, linewidth=0.8))
        for position, values in enumerate(data, start=1):
            ax.scatter(position + rng.uniform(-0.12, 0.12, len(values)), values, s=8, color=INK, alpha=0.55, linewidth=0, zorder=3)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels([ALGORITHM_NAMES[a] for a in order])
        ax.set_ylabel(label)
        ax.set_title(better, fontsize=9, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    fig.tight_layout()
    save(fig, "algorithms")
    plt.close(fig)


PARTS = {"mechanism": mechanism, "plans": plans, "price": price, "exact": exact, "algorithms": algorithms}

if __name__ == "__main__":
    for part in sys.argv[1:] or list(PARTS):
        PARTS[part]()
