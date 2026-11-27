"""
gui/plots.py  -  Drawing helpers + comparison charts (matplotlib).

This file has 3 jobs:
    1. run_everything(env)   runs the 4 strategies + the exact "Optimal" answer
    2. show_comparison(...)  one window that compares all of them
    3. helpers that gui/simulation_ui.py also uses (colors, garden drawing, energy)

Run from the project root (the folder that has main.py):
    python -m gui.plots            (garden seed 0)
    python -m gui.plots 5          (garden seed 5)
"""

import os
import sys
import time

import matplotlib.pyplot as plt

from environment.garden import BeeGarden
from search.astar import build_distance_matrix
from optimization import hill_climbing_strategy, genetic_algorithm_strategy
from learning.q_learning import q_learning_strategy
from evaluation.search_based import search_based_strategy

SIZE = 15
N_FLOWERS = 10

# (name, function) - same order as evaluation/compare.py
STRATEGIES = [
    ("A* + Nearest Flower", search_based_strategy),
    ("Hill Climbing", hill_climbing_strategy),
    ("Genetic Algorithm", genetic_algorithm_strategy),
    ("Q-Learning", q_learning_strategy),
]
OPTIMAL = "Optimal (exact)"          # the best possible answer (see optimal_solution)

COLORS = {
    "A* + Nearest Flower": "#1f77b4",
    "Hill Climbing": "#ff7f0e",
    "Genetic Algorithm": "#2ca02c",
    "Q-Learning": "#9467bd",
    OPTIMAL: "#d4a017",
}
SHORT = {
    "A* + Nearest Flower": "A*",
    "Hill Climbing": "Hill\nClimbing",
    "Genetic Algorithm": "Genetic",
    "Q-Learning": "Q-Learning",
    OPTIMAL: "Optimal",
}


# ----------------------------------------------------------------------
# 1. Running the strategies
# ----------------------------------------------------------------------
def optimal_solution(env, matrix):
    """
    The best possible answer, found by checking every possible set of flowers.
    (This is a DP over subsets: "cost[group][last]" = the least energy needed to
    collect that group of flowers and end at the flower "last".)
    The bee must be able to come home, and we keep the group with the most nectar.
    It is only fast for a small number of flowers (like our 10).
    """
    start = time.perf_counter()
    flowers = list(env.flowers)
    n = len(flowers)
    if n > 14:
        return None
    collect = env.collect_cost
    hive = env.hive
    limit = env.energy_limit

    INF = float("inf")
    cost = {}                                   # (group, last) -> least energy
    parent = {}                                 # (group, last) -> the flower before it
    for j in range(n):
        cost[(1 << j, j)] = matrix[(hive, flowers[j])][1] + collect

    for group in range(1, 1 << n):
        for last in range(n):
            here = cost.get((group, last), INF)
            if here == INF:
                continue
            for nxt in range(n):
                if group & (1 << nxt):
                    continue
                new_group = group | (1 << nxt)
                new_cost = here + matrix[(flowers[last], flowers[nxt])][1] + collect
                if new_cost < cost.get((new_group, nxt), INF):
                    cost[(new_group, nxt)] = new_cost
                    parent[(new_group, nxt)] = last

    best = (0, 0, None, None)                   # (nectar, -energy, group, last)
    for (group, last), c in cost.items():
        if c + matrix[(flowers[last], hive)][1] > limit:
            continue                            # could not come back home
        nectar = sum(env.flowers[flowers[k]] for k in range(n) if group & (1 << k))
        if (nectar, -c) > best[:2]:
            best = (nectar, -c, group, last)

    order = []                                  # rebuild the visiting order
    group, last = best[2], best[3]
    while group:
        order.append(flowers[last])
        previous = parent.get((group, last))
        group &= ~(1 << last)
        last = previous
    order.reverse()

    stops = [hive] + order + [hive]
    route = [hive]
    for a, b in zip(stops, stops[1:]):
        route.extend(matrix[(a, b)][0][1:])

    distance = len(route) - 1
    return {
        "nectar": best[0],
        "distance": distance,
        "energy_used": distance + collect * len(order),
        "returned": True,
        "visited": list(order),
        "order": list(order),
        "route": route,
        "history": [],
        "seconds": time.perf_counter() - start,
    }


def run_everything(env):
    """Run the 4 strategies and the optimal answer. Returns {name: result}."""
    matrix = build_distance_matrix(env)
    results = {}
    for name, strategy in STRATEGIES:
        results[name] = strategy(env, matrix=matrix)
    best = optimal_solution(env, matrix)
    if best is not None:
        results[OPTIMAL] = best
    return results


def best_strategy(results):
    """The best of the 4 strategies: most nectar, then least energy, then fastest."""
    names = [name for name, _ in STRATEGIES if name in results]
    return min(names, key=lambda n: (-results[n]["nectar"],
                                     results[n]["energy_used"],
                                     results[n]["seconds"]))


# ----------------------------------------------------------------------
# 2. Energy step by step (used by the animation)
# ----------------------------------------------------------------------
def energy_profile(env, result):
    """
    Walk along the route and write down the state after every step:
    every step costs 1 energy, collecting nectar costs 2 more.
    """
    route = result["route"]
    planned = set(result["visited"])
    energy = env.energy_limit
    nectar = 0
    done = set()

    states = [{"pos": route[0], "energy": energy, "nectar": 0,
               "collected": frozenset(), "just": False}]
    for pos in route[1:]:
        energy -= 1
        just = False
        if pos in planned and pos not in done:
            done.add(pos)
            energy -= env.collect_cost
            nectar += env.flowers[pos]
            just = True
        states.append({"pos": pos, "energy": energy, "nectar": nectar,
                       "collected": frozenset(done), "just": just})
    return states


# ----------------------------------------------------------------------
# 3. Drawing the garden (same look in the animation and in the comparison)
# ----------------------------------------------------------------------
def draw_garden_base(ax, env):
    """Empty garden: white grid, gray obstacles and the yellow hive."""
    ax.set_facecolor("#f8f8f4")
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)         # y grows downward, like a screen
    ax.set_aspect("equal")
    ax.set_xticks([i - 0.5 for i in range(env.width + 1)])
    ax.set_yticks([i - 0.5 for i in range(env.height + 1)])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.grid(True, color="#d5dccb", linewidth=0.8)

    for (x, y) in env.obstacles:
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#9a9fb5", zorder=1))

    hx, hy = env.hive
    ax.add_patch(plt.Rectangle((hx - 0.35, hy - 0.35), 0.7, 0.7, facecolor="#e8b000",
                               edgecolor="#6b4a00", linewidth=2, zorder=4))


def draw_route_map(ax, env, result, title):
    """Garden + the route of one strategy (green flower = visited, orange = skipped)."""
    draw_garden_base(ax, env)
    xs = [p[0] for p in result["route"]]
    ys = [p[1] for p in result["route"]]
    ax.plot(xs, ys, color="#e88080", linewidth=2, zorder=2)

    for (x, y), value in env.flowers.items():
        visited = (x, y) in result["visited"]
        ax.scatter([x], [y], s=110, zorder=3, linewidths=1.2,
                   facecolors="#90e0a0" if visited else "#f59e0b",
                   edgecolors="#a05a00")
    ax.set_title(title, fontsize=9)


# ----------------------------------------------------------------------
# 4. The comparison window
# ----------------------------------------------------------------------
def show_comparison(env, results, seed, save_path=None):
    """
    Top row   : bar charts of the metrics (the best of the 4 has a thick border)
    Bottom row: the route of every strategy on the same garden
    Returns the figure.
    """
    names = [n for n in list(COLORS) if n in results]
    best = best_strategy(results)
    optimal = results.get(OPTIMAL)

    fig = plt.figure(figsize=(17, 9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.4,
                             left=0.04, right=0.98, top=0.88, bottom=0.04)
    top = outer[0].subgridspec(1, 4, wspace=0.3)
    bottom = outer[1].subgridspec(1, len(names), wspace=0.08)

    metrics = [
        ("Total Nectar (higher is better)", "nectar"),
        ("Energy Used (lower is better)", "energy_used"),
        ("Travel Distance (lower is better)", "distance"),
        ("Execution Time in seconds", "seconds"),
    ]
    for i, (title, key) in enumerate(metrics):
        ax = fig.add_subplot(top[0, i])
        values = [results[n][key] for n in names]
        bars = ax.bar(range(len(names)), values, color=[COLORS[n] for n in names])
        for bar, n in zip(bars, names):
            if n == best:
                bar.set_edgecolor("black")
                bar.set_linewidth(3)
        for x, v in enumerate(values):
            label = f"{v:.3f}" if key == "seconds" else f"{v:g}"
            ax.text(x, v, label, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([SHORT[n] for n in names], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.15)

    for i, n in enumerate(names):
        ax = fig.add_subplot(bottom[0, i])
        r = results[n]
        tag = "  (BEST of the 4)" if n == best else ""
        draw_route_map(ax, env, r,
                       f"{n}{tag}\nnectar {r['nectar']}  |  energy {r['energy_used']}"
                       f"  |  flowers {len(r['visited'])}")

    text = f"Comparison on garden (seed {seed})   -   best of the 4: {best} with {results[best]['nectar']} nectar"
    if optimal is not None:
        gap = optimal["nectar"] - results[best]["nectar"]
        text += f"   -   optimal answer: {optimal['nectar']} nectar (gap = {gap})"
    fig.suptitle(text, fontsize=13)

    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, dpi=110)
        print("Saved", save_path)
    return fig


def plot_all_maps(seeds=(0, 1, 2), save_path=None):
    """Nectar of every strategy on several maps, plus the average."""
    table = {}                                   # name -> list of nectar (one per map)
    for seed in seeds:
        env = BeeGarden(SIZE, SIZE, n_flowers=N_FLOWERS, seed=seed)
        results = run_everything(env)
        for name, r in results.items():
            table.setdefault(name, []).append(r["nectar"])

    names = list(table)
    groups = [f"Map {s}" for s in seeds] + ["Average"]
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, name in enumerate(names):
        values = table[name] + [sum(table[name]) / len(table[name])]
        positions = [k + i * width for k in range(len(groups))]
        bars = ax.bar(positions, values, width=width, label=name, color=COLORS[name])
        for p, v in zip(positions, values):
            ax.text(p, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks([k + 0.4 - width / 2 for k in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set_ylabel("Total nectar")
    ax.set_title("Nectar collected on each map (Optimal = the best possible)")
    ax.legend(fontsize=8)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=110)
        print("Saved", save_path)
    return fig


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = BeeGarden(SIZE, SIZE, n_flowers=N_FLOWERS, seed=seed)
    results = run_everything(env)

    print(f"\nGarden seed {seed}")
    for name, r in results.items():
        print(f"{name:22s} nectar {r['nectar']:3d} | energy {r['energy_used']:3d} | "
              f"distance {r['distance']:3d} | time {r['seconds']:.4f}s")
    print("Best of the 4:", best_strategy(results))

    show_comparison(env, results, seed, save_path=f"results/comparison_seed{seed}.png")
    plot_all_maps(save_path="results/nectar_all_maps.png")
    plt.show()


if __name__ == "__main__":
    main()