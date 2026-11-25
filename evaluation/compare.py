"""
evaluation/compare.py  -  Compare the strategies on the same maps (Team Leader)

Every strategy runs on exactly the same gardens (same seeds) with the same
Distance Matrix, so the comparison is fair. The matrix is built once and shared,
so "Execution Time" measures only the strategy itself.

Returns:
    per_map  : one row per (map, strategy)
    summary  : one row per strategy (average over the maps)
    results  : the full result dictionaries, key = (seed, strategy name)
              (Team 4 can use them to draw routes and learning curves)
"""

import os

import pandas as pd

from environment.garden import BeeGarden
from search.astar import build_distance_matrix
from optimization import hill_climbing_strategy, genetic_algorithm_strategy
from learning.q_learning import q_learning_strategy
from evaluation.search_based import search_based_strategy
from evaluation.metrics import compute_metrics, check_result

# (type of strategy, name, function)
STRATEGIES = [
    ("Search-Based", "A* + Nearest Flower", search_based_strategy),
    ("Optimized", "Hill Climbing", hill_climbing_strategy),
    ("Optimized", "Genetic Algorithm", genetic_algorithm_strategy),
    ("Learning-Based", "Q-Learning", q_learning_strategy),
]


def compare_strategies(seeds=(0, 1, 2), size=15, n_flowers=10, save_csv=True):
    rows = []
    results = {}

    for seed in seeds:
        env = BeeGarden(size, size, n_flowers=n_flowers, seed=seed)
        matrix = build_distance_matrix(env)           # built once for all strategies

        for kind, name, strategy in STRATEGIES:
            result = strategy(env, matrix=matrix)

            problems = check_result(env, result)
            if problems:
                print(f"WARNING (seed {seed}, {name}):", "; ".join(problems))

            row = {"Seed": seed, "Type": kind, "Strategy": name}
            row.update(compute_metrics(result))
            rows.append(row)
            results[(seed, name)] = result

    per_map = pd.DataFrame(rows)

    summary = per_map.groupby(["Type", "Strategy"], sort=False).agg({
        "Total Nectar Collected": "mean",
        "Travel Distance": "mean",
        "Energy Consumption": "mean",
        "Successful Hive Returns": "sum",
        "Execution Time (s)": "mean",
        "Seed": "count",
    }).rename(columns={"Seed": "Runs"}).round(2).reset_index()

    if save_csv:
        os.makedirs("results", exist_ok=True)
        per_map.to_csv("results/comparison_per_map.csv", index=False)
        summary.to_csv("results/comparison_summary.csv", index=False)

    return per_map, summary, results