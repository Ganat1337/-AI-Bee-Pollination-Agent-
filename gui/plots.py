"""
gui/plots.py - Game-styled Matplotlib visualizations for the Bee Garden,
training/optimization curves, and multi-strategy benchmark comparisons.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_garden_route(env, route=None, visited=None, title="Bee Meadow Garden"):
    """Draw the garden with grass tiles, grey rocks, blooming flowers, hive, and route."""
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#163024")
    ax.set_facecolor("#4fa844")

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.tick_params(colors="#d6e8d5", labelsize=9)
    ax.set_aspect("equal")
    ax.set_title(title, color="#fef3c7", fontsize=13, fontweight="bold", pad=12)

    # Checkerboard meadow grass tiles
    for x in range(env.width):
        for y in range(env.height):
            color = "#5cb850" if (x + y) % 2 == 0 else "#4fa844"
            rect = patches.Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=color, edgecolor="#439339", linewidth=0.5, zorder=0
            )
            ax.add_patch(rect)

    # Grey boulders (obstacles)
    for ox, oy in env.obstacles:
        rock = patches.RegularPolygon(
            (ox, oy), numVertices=7, radius=0.38,
            orientation=0.2, facecolor="#8b929a", edgecolor="#454a52", linewidth=1.2, zorder=2
        )
        ax.add_patch(rock)
        highlight = patches.RegularPolygon(
            (ox - 0.06, oy - 0.08), numVertices=5, radius=0.18,
            facecolor="#bcc3cc", edgecolor="none", zorder=2.1
        )
        ax.add_patch(highlight)

    # Planned / flown route
    if route:
        rx = [p[0] for p in route]
        ry = [p[1] for p in route]
        ax.plot(rx, ry, color="#78350f", linewidth=4.5, alpha=0.35, zorder=2.5)
        ax.plot(rx, ry, color="#fef08a", linewidth=2.2, linestyle="--", zorder=2.6, label="Flight Route")

    # Blooming flowers
    visited_set = set(visited or [])
    for (fx, fy), nectar in env.flowers.items():
        if nectar >= 20:
            petal_color, center_color, rad = "#e11d48", "#7f1d1d", 0.36
        elif nectar >= 10:
            petal_color, center_color, rad = "#f97316", "#7c2d12", 0.31
        else:
            petal_color, center_color, rad = "#fde047", "#d97706", 0.27

        if (fx, fy) in visited_set:
            ring = patches.Circle((fx, fy), rad + 0.09, fill=False, edgecolor="#fef08a", linewidth=2, zorder=3)
            ax.add_patch(ring)

        # Petals
        n_petals = 8 if nectar >= 20 else (7 if nectar >= 10 else 6)
        for i in range(n_petals):
            ang = 2 * np.pi * i / n_petals
            px = fx + np.cos(ang) * rad * 0.55
            py = fy + np.sin(ang) * rad * 0.55
            petal = patches.Circle((px, py), rad * 0.42, facecolor=petal_color, edgecolor="#451a03", linewidth=0.5, zorder=3.2)
            ax.add_patch(petal)

        center = patches.Circle((fx, fy), rad * 0.48, facecolor=center_color, edgecolor="#fffbeb", linewidth=0.7, zorder=3.5)
        ax.add_patch(center)
        ax.text(fx, fy, str(nectar), ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=4)

    # Hive at (0, 0)
    hx, hy = env.hive
    hive = patches.RegularPolygon(
        (hx, hy), numVertices=6, radius=0.4,
        facecolor="#f59e0b", edgecolor="#78350f", linewidth=1.5, zorder=4.5, label="Hive (0,0)"
    )
    ax.add_patch(hive)
    ax.text(hx, hy, "H", ha="center", va="center", color="#451a03", fontsize=9, fontweight="bold", zorder=5)

    for spine in ax.spines.values():
        spine.set_edgecolor("#6f4a2e")
        spine.set_linewidth(4)

    plt.tight_layout()
    plt.show()


def plot_training_curve(
    history,
    title="Training Progress",
    xlabel="Episode / Iteration",
    ylabel="Score / Fitness",
    window=None,
    path=None,
    show=True,
):
    """Plot Q-Learning reward curve (with moving average) or Hill Climbing / GA fitness history."""
    if not history:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fig.patch.set_facecolor("#fffcf4")
    ax.set_facecolor("#fffcf4")

    if window and window > 1:
        avg = []
        for i in range(len(history)):
            chunk = history[max(0, i - window + 1): i + 1]
            avg.append(sum(chunk) / len(chunk))
        ax.plot(history, color="#16a34a", alpha=0.28, linewidth=1.1, label="Episode Score")
        ax.plot(avg, color="#d97706", linewidth=2.3, label=f"Moving Avg ({window})")
    else:
        ax.plot(history, color="#d97706", linewidth=2.2, label="Best Fitness")

    ax.set_title(title, color="#3b2a1a", fontweight="bold")
    ax.set_xlabel(xlabel, color="#5c4732")
    ax.set_ylabel(ylabel, color="#5c4732")
    ax.grid(alpha=0.25, color="#cbb99d")
    ax.legend(frameon=True, facecolor="#fffcf4", edgecolor="#cbb99d")

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_strategy_comparison(summary_df, path=None, show=True):
    """Bar chart comparing Total Nectar, Travel Distance, and Energy across strategies."""
    strategies = summary_df["Strategy"].tolist()
    nectar = summary_df["Total Nectar Collected"].tolist()
    distance = summary_df["Travel Distance"].tolist()
    energy = summary_df["Energy Consumption"].tolist()

    x = np.arange(len(strategies))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor("#fffcf4")
    ax.set_facecolor("#fffcf4")

    ax.bar(x - w, nectar, width=w, color="#f59e0b", edgecolor="#78350f", label="Avg Nectar")
    ax.bar(x, distance, width=w, color="#16a34a", edgecolor="#14532d", label="Avg Distance")
    ax.bar(x + w, energy, width=w, color="#64748b", edgecolor="#334155", label="Avg Energy")

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=10, ha="right", color="#3b2a1a", fontweight="bold")
    ax.set_ylabel("Value", color="#5c4732")
    ax.set_title("Strategy Leaderboard Comparison", color="#3b2a1a", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, color="#cbb99d")
    ax.legend(frameon=True, facecolor="#fffcf4", edgecolor="#cbb99d")

    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    from environment.garden import BeeGarden
    from learning.q_learning import q_learning_strategy

    env = BeeGarden(15, 15, n_flowers=10, seed=0)
    res = q_learning_strategy(env)
    plot_garden_route(env, route=res["route"], visited=res["visited"], title="Q-Learning Route (Seed 0)")
    plot_training_curve(res["history"], title="Q-Learning Curve", window=100)