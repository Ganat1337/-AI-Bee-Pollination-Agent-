import itertools
import random
import time

from search.astar import build_distance_matrix
COLLECT_COST = 2
def get_collect_cost(env):
    return getattr(env, "collect_cost", COLLECT_COST)


def evaluate_order(env, order, matrix=None):
   
    if matrix is None:
        matrix = build_distance_matrix(env)
    collect = get_collect_cost(env)

    energy = env.energy_limit
    pos = env.hive
    steps = 0
    nectar = 0
    visited = []

    for flower in order:
        go = matrix[(pos, flower)][1]
        back = matrix[(flower, env.hive)][1]
        if energy - go - collect < back:    
            break
        energy -= go + collect
        steps += go
        nectar += env.flowers[flower]
        visited.append(flower)
        pos = flower

    steps += matrix[(pos, env.hive)][1]        
    energy_used = steps + collect * len(visited)
    return {
        "nectar": nectar,
        "distance": steps,
        "energy_used": energy_used,
        "returned": energy_used <= env.energy_limit,
        "visited": visited,
    }


def make_fitness(env, matrix=None):
   
    if matrix is None:
        matrix = build_distance_matrix(env)
    penalty = 0.5 / (env.energy_limit + 1)

    def fit(order):
        r = evaluate_order(env, order, matrix)
        return r["nectar"] - penalty * r["energy_used"]

    return fit


def build_full_route(env, visited, matrix):
  
    stops = [env.hive] + list(visited) + [env.hive]
    route = [env.hive]
    for a, b in zip(stops, stops[1:]):
        path = matrix[(a, b)][0]
        route.extend(path[1:])
    return route


def make_result(env, order, matrix, history, seconds):

    r = evaluate_order(env, order, matrix)
    r.update({
        "order": list(order),
        "route": build_full_route(env, r["visited"], matrix),
        "history": history,
        "seconds": seconds,
    })
    return r



def order_neighbors(order):
    n = len(order)
    for i, j in itertools.combinations(range(n), 2):      
        s = order[:]
        s[i], s[j] = s[j], s[i]
        yield s
    for i in range(n):                                    
        for j in range(n):
            if i != j:
                s = order[:]
                s.insert(j, s.pop(i))
                yield s


def climb(order, fit):
    cur, cur_fit = order[:], fit(order)
    while True:
        best_nb, best_f = None, cur_fit
        for nb in order_neighbors(cur):
            f = fit(nb)
            if f > best_f:
                best_nb, best_f = nb, f
        if best_nb is None:               
            return cur, cur_fit
        cur, cur_fit = best_nb, best_f


def hill_climbing_strategy(env, restarts=30, seed=42, matrix=None):
    
    start = time.perf_counter()
    if matrix is None:
        matrix = build_distance_matrix(env)
    fit = make_fitness(env, matrix)
    rng = random.Random(seed)             

    flowers = list(env.flowers.keys())
    best, best_fit, history = flowers[:], float("-inf"), []
    for _ in range(restarts):
        s = flowers[:]
        rng.shuffle(s)                  
        sol, f = climb(s, fit)
        if f > best_fit:
            best, best_fit = sol, f
        history.append(best_fit)          
    return make_result(env, best, matrix, history, time.perf_counter() - start)


if __name__ == "__main__":
    from environment.garden import BeeGarden
    env = BeeGarden()
    res = hill_climbing_strategy(env)
    print("Hill Climbing order:", res["order"])
    print("Visited:", res["visited"])
    print("Nectar:", res["nectar"], "| Energy used:", res["energy_used"],
          "| Returned:", res["returned"])
