import random
import time

from search.astar import build_distance_matrix
from optimization.hill_climbing import make_fitness, make_result   


def order_crossover(p1, p2, rng):
   
    n = len(p1)
    if n < 2:
        return p1[:]
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]
    rest = iter(x for x in p2 if x not in child)
    for k in range(n):
        if child[k] is None:
            child[k] = next(rest)
    return child


def mutate(sol, rng, rate=0.15):
    sol = sol[:]
    if len(sol) >= 2 and rng.random() < rate:
        i, j = rng.sample(range(len(sol)), 2)
        sol[i], sol[j] = sol[j], sol[i]
    return sol


def tournament(pop, fits, rng, k=3):
    picks = rng.sample(range(len(pop)), min(k, len(pop)))
    return pop[max(picks, key=lambda i: fits[i])]


def genetic_algorithm_strategy(env, pop_size=100, generations=300, seed=42, matrix=None):

    start = time.perf_counter()
    if matrix is None:
        matrix = build_distance_matrix(env)
    fit = make_fitness(env, matrix)
    rng = random.Random(seed)

    flowers = list(env.flowers.keys())
    if len(flowers) < 2:                 
        return make_result(env, flowers, matrix, [fit(flowers)], time.perf_counter() - start)

    pop = []
    for _ in range(pop_size):
        s = flowers[:]
        rng.shuffle(s)
        pop.append(s)

    history = []
    for _ in range(generations):
        fits = [fit(s) for s in pop]
        history.append(max(fits))
        ranked = sorted(zip(fits, pop), key=lambda x: x[0], reverse=True)
        new_pop = [ranked[0][1][:], ranked[1][1][:]]         
        while len(new_pop) < pop_size:
            p1 = tournament(pop, fits, rng)
            p2 = tournament(pop, fits, rng)
            new_pop.append(mutate(order_crossover(p1, p2, rng), rng))
        pop = new_pop

    best = max(pop, key=fit)
    return make_result(env, best, matrix, history, time.perf_counter() - start)


if __name__ == "__main__":
    from environment.garden import BeeGarden
    env = BeeGarden()
    res = genetic_algorithm_strategy(env)
    print("GA order:", res["order"])
    print("Visited:", res["visited"])
    print("Nectar:", res["nectar"], "| Energy used:", res["energy_used"],
          "| Returned:", res["returned"])
