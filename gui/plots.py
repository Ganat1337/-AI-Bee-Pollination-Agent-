import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _finish(fig, save_path, show):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_energy(energy_hist, max_energy=None, save_path=None, show=True):
    """Energy of the bee at every step."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(energy_hist)), energy_hist, color="#16a34a", linewidth=2)
    if max_energy is not None:
        ax.axhline(max_energy, color="#9ca3af", linestyle="--", label="Energy limit")
        ax.legend()
    ax.set_title("Energy over time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy")
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show)


def plot_path(garden, path, collected=None, save_path=None, show=True):
    """Garden map: obstacles, flowers, hive and the route (list of (row, col))."""
    collected = set(collected or [])
    fig, ax = plt.subplots(figsize=(6, 6))

    for r in range(garden.height):
        for c in range(garden.width):
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#f7f9f2",
                                   edgecolor="#cbd5c0", linewidth=1))
    for r, c in garden.obstacles:
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#9ca3af",
                               edgecolor="#4b5563", linewidth=1))

    for r, c in garden.flowers:
        color = "#86efac" if (r, c) in collected else "#f59e0b"
        ax.scatter(c + 0.5, r + 0.5, s=220, color=color,
                   edgecolors="#b45309", linewidths=1.5, zorder=3)

    hr, hc = garden.hive
    ax.scatter(hc + 0.5, hr + 0.5, s=320, marker="s", color="#eab308",
               edgecolors="#713f12", linewidths=2, zorder=4)

    if path:
        ax.plot([c + 0.5 for _r, c in path], [r + 0.5 for r, _c in path],
                "-", color="#ef4444", linewidth=2.5, alpha=0.8, zorder=2)
        ax.scatter(path[-1][1] + 0.5, path[-1][0] + 0.5, s=200, color="#ef4444",
                   edgecolors="#7f1d1d", linewidths=1.5, zorder=5)

    ax.set_xlim(0, garden.width)
    ax.set_ylim(garden.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Bee path")
    return _finish(fig, save_path, show)


def plot_convergence(histories, ylabel="Fitness", title="Convergence",
                     save_path=None, show=True):
    """histories = {"Genetic": [..], "Hill climbing": [..]}"""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, values in histories.items():
        ax.plot(range(len(values)), values, linewidth=2, label=name)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, save_path, show)


def plot_comparison(results, ylabel="Value", title="Comparison", save_path=None, show=True):
    """results = {"A*": 42, "Genetic": 38, ...} -> bar chart."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(results)
    values = [results[n] for n in names]
    bars = ax.bar(names, values, color="#f59e0b", edgecolor="#b45309")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:g}",
                ha="center", va="bottom")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, save_path, show)


if __name__ == "__main__":
    from environment.garden import BeeGarden
    from gui.simulation_ui import BeeAgent

    garden = BeeGarden(width=15, height=15, n_flowers=10, seed=0)
    bee = BeeAgent(garden)
    while not bee.done:
        bee.step()

    plot_energy(bee.energy_hist, bee.max_energy, show=False)
    plot_path(garden, bee.trail, bee.collected, show=False)
    plot_comparison({"Bee (greedy A*)": len(bee.collected)},
                    ylabel="Flowers collected", title="Flowers collected", show=False)
    plt.show()