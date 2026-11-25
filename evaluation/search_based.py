"""
evaluation/search_based.py  -  Search-Based strategy (Team Leader)

The bee uses A* distances (from the Distance Matrix) to choose its next flower.
Two simple rules to choose from:
    rule="nearest"     -> the closest flower it can still afford
    rule="best_ratio"  -> the flower with the most nectar per unit of energy

Before going to a flower the bee checks that it can still come back:
    energy - trip to flower - collect cost  >=  trip back to the hive

Returns the same result dictionary as the other strategies:
    res = search_based_strategy(env, matrix=matrix)
"""

import time

from search.astar import build_distance_matrix


def search_based_strategy(env, matrix=None, rule="nearest"):
    start = time.perf_counter()
    if matrix is None:
        matrix = build_distance_matrix(env)
    collect = getattr(env, "collect_cost", 2)

    pos = env.hive
    energy = env.energy_limit
    visited = []
    nectar = 0
    distance = 0

    while True:
        best_flower = None
        best_score = None

        for flower in env.flowers:
            if flower in visited:
                continue
            go = matrix[(pos, flower)][1]             # A* cost to reach the flower
            back = matrix[(flower, env.hive)][1]      # A* cost to return from it
            if energy - go - collect < back:          # not safe: could not come back
                continue

            if rule == "nearest":
                score = -go                           # closer is better
            else:
                score = env.flowers[flower] / (go + collect)   # nectar per energy

            if best_score is None or score > best_score:
                best_flower = flower
                best_score = score

        if best_flower is None:                       # nothing affordable left
            break

        go = matrix[(pos, best_flower)][1]
        energy -= go + collect
        distance += go
        nectar += env.flowers[best_flower]
        visited.append(best_flower)
        pos = best_flower

    if pos != env.hive:                               # final trip back home
        distance += matrix[(pos, env.hive)][1]

    stops = [env.hive] + visited + [env.hive]
    route = [env.hive]
    for a, b in zip(stops, stops[1:]):
        route.extend(matrix[(a, b)][0][1:])

    energy_used = distance + collect * len(visited)
    return {
        "nectar": nectar,
        "distance": distance,
        "energy_used": energy_used,
        "returned": energy_used <= env.energy_limit,
        "visited": visited,
        "order": list(visited),
        "route": route,
        "history": [],                                # no training, nothing to plot
        "seconds": time.perf_counter() - start,
    }


if __name__ == "__main__":
    from environment.garden import BeeGarden

    env = BeeGarden(15, 15, n_flowers=10, seed=0)
    for rule in ("nearest", "best_ratio"):
        res = search_based_strategy(env, rule=rule)
        print(rule, "-> nectar:", res["nectar"], "| energy used:", res["energy_used"],
              "| visited:", len(res["visited"]), "| returned:", res["returned"])