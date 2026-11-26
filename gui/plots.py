"""
gui/plots.py  -  Charts for the project (matplotlib).

Run from the project root (the folder that has main.py):
    python -m gui.plots

It makes 3 pictures in the results/ folder:
    results/strategy_comparison.png   bar charts of the 5 metrics
    results/routes_seed0.png          the route of every strategy on map 0
    results/nectar_per_map.png        nectar of every strategy on each map
"""

import os

import matplotlib.pyplot as plt

from evaluation.compare import compare_strategies
from evaluation.metrics import METRIC_NAMES

FLOWER_SIZES = {5: "small", 10: "medium", 20: "large"}


def plot_comparison(summary, path="results/strategy_comparison.png", show=False):
    """One small bar chart for each of the 5 metrics (average over the maps)."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    names = list(summary["Strategy"])

    for ax, metric in zip(axes, METRIC_NAMES):
        values = list(summary[metric])
        ax.bar(range(len(names)), values)
        ax.set_title(metric)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        for i, v in enumerate(values):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)

    axes[-1].axis("off")                       # 5 metrics, so the 6th box is empty
    fig.suptitle("Strategy comparison (average over the maps)")
    fig.tight_layout()
    save_or_show(fig, path, show)


def plot_route(env, result, title, ax):
    """Draw the garden and the bee's route on one axis."""
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)        # y goes down, like on the screen
    ax.set_aspect("equal")
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.tick_params(labelsize=6)
    ax.grid(True, linewidth=0.3)

    for (x, y) in env.obstacles:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="gray"))

    xs = [p[0] for p in result["route"]]
    ys = [p[1] for p in result["route"]]
    ax.plot(xs, ys, color="orange", linewidth=2)

    for (x, y), value in env.flowers.items():
        visited = (x, y) in result["visited"]
        ax.plot(x, y, "o", markersize=12,
                color="purple" if visited else "lightgray")
        ax.text(x, y, str(value), ha="center", va="center", fontsize=6, color="white")

    ax.plot(env.hive[0], env.hive[1], "s", markersize=12, color="brown")
    ax.set_title(f"{title}\nnectar {result['nectar']}, energy {result['energy_used']}",
                 fontsize=9)


def plot_all_routes(env, results, seed, path="results/routes_seed0.png", show=False):
    """The routes of all strategies on the same map, side by side."""
    names = [name for (s, name) in results if s == seed]
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 5))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        plot_route(env, results[(seed, name)], name, ax)

    fig.suptitle(f"Routes on map {seed}  (purple = visited flower, gray = skipped)")
    fig.tight_layout()
    save_or_show(fig, path, show)


def plot_nectar_per_map(per_map, path="results/nectar_per_map.png", show=False):
    """Grouped bars: nectar of every strategy on every map."""
    seeds = sorted(per_map["Seed"].unique())
    names = list(per_map["Strategy"].unique())
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(names):
        values = []
        for s in seeds:
            row = per_map[(per_map["Seed"] == s) & (per_map["Strategy"] == name)]
            values.append(float(row["Total Nectar Collected"].iloc[0]))
        positions = [k + i * width for k in range(len(seeds))]
        ax.bar(positions, values, width=width, label=name)

    ax.set_xticks([k + 0.4 - width / 2 for k in range(len(seeds))])
    ax.set_xticklabels([f"Map {s}" for s in seeds])
    ax.set_ylabel("Total nectar")
    ax.set_title("Nectar collected on each map")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_or_show(fig, path, show)


def save_or_show(fig, path, show):
    if path:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(path, dpi=120)
        print("Saved", path)
    if show:
        plt.show()
    plt.close(fig)


def main():
    from environment.garden import BeeGarden

    seeds = (0, 1, 2)
    per_map, summary, results = compare_strategies(seeds=seeds, size=15, n_flowers=10)

    plot_comparison(summary)
    plot_nectar_per_map(per_map)

    env = BeeGarden(15, 15, n_flowers=10, seed=seeds[0])
    plot_all_routes(env, results, seeds[0])


if __name__ == "__main__":
    main()