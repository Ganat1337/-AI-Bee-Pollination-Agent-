"""
main.py  -  Run the whole project: build the gardens, run every strategy, compare.

Run from the project root:
    python main.py
"""

import os

from evaluation.compare import compare_strategies
from learning.q_learning import plot_learning_curve

SEEDS = (0, 1, 2)       # the 3 maps used for the comparison
SIZE = 15               # 15x15 garden
N_FLOWERS = 10


def main():
    per_map, summary, results = compare_strategies(
        seeds=SEEDS, size=SIZE, n_flowers=N_FLOWERS
    )

    print(f"\nMaps: seeds {list(SEEDS)} | garden {SIZE}x{SIZE} | {N_FLOWERS} flowers\n")

    print("=== Results on every map ===")
    print(per_map.to_string(index=False))

    print("\n=== Average over the maps ===")
    print(summary.to_string(index=False))

    # Learning curve of the Q-Learning bee on the first map
    os.makedirs("results", exist_ok=True)
    history = results[(SEEDS[0], "Q-Learning")]["history"]
    plot_learning_curve(history, path="results/learning_curve.png")
    print("\nLearning curve saved to results/learning_curve.png")
    print("Tables saved to results/comparison_per_map.csv and results/comparison_summary.csv")


if __name__ == "__main__":
    main()