"""
learning/q_learning.py  -  Team 3: Q-Learning (Learning-Based strategy)

The bee learns which flower to visit next (or when to go back to the hive).

Main idea (Q-Learning):
    The bee keeps a table "q" that says how good each action is in each state.
    Every trip it updates the table, so after many trips it knows the best choices.

    State  = (where the bee is, which flowers it visited, energy left)
    Action = the next flower to visit, or the hive position to go home
    Reward = nectar collected - COST_WEIGHT * energy spent

Same style as the other teams:
    res = q_learning_strategy(env, matrix=matrix)
returns the same keys as hill_climbing_strategy / genetic_algorithm_strategy:
nectar, distance, energy_used, returned, visited, order, route, history, seconds
(here "history" = total reward of each training trip = the Learning Curve)

Run from the project root:   python -m learning.q_learning
"""

import random
import time

import matplotlib.pyplot as plt

from search.astar import build_distance_matrix

COLLECT_COST = 2        # energy to collect nectar (same as Team 2)
COST_WEIGHT = 0.5       # how much each unit of energy hurts the reward
ALPHA = 0.1             # learning rate: how fast the bee changes its mind
GAMMA = 0.95            # how much the bee cares about future rewards
EPS_START = 1.0         # at first the bee explores randomly 100% of the time...
EPS_MIN = 0.05          # ...and later only 5% of the time
EPS_DECAY = 0.999       # exploring gets smaller after every trip


def get_collect_cost(env):
    return getattr(env, "collect_cost", COLLECT_COST)


def get_state(pos, visited, energy):
    """The situation of the bee, written as a tuple so it can be a dictionary key."""
    return (pos, tuple(sorted(visited)), energy // 10)


def get_valid_actions(env, matrix, pos, visited, energy):
    """Flowers the bee can visit AND still return home from, plus 'go home'."""
    collect = get_collect_cost(env)
    actions = []
    for flower in env.flowers:
        if flower in visited:
            continue                              # already collected
        go = matrix[(pos, flower)][1]             # energy to reach the flower
        back = matrix[(flower, env.hive)][1]      # energy to return from it
        if energy - go - collect >= back:         # safe: can still come back
            actions.append(flower)
    if pos != env.hive:
        actions.append(env.hive)                  # going home is always allowed
    return actions


def choose_action(q, state, actions, epsilon, rng):
    """Explore (random) with probability epsilon, otherwise pick the best known action."""
    if rng.random() < epsilon:
        return rng.choice(actions)

    best_value = max(q.get((state, a), 0.0) for a in actions)
    best_actions = []
    for a in actions:
        if q.get((state, a), 0.0) == best_value:
            best_actions.append(a)
    return rng.choice(best_actions)


def take_step(env, matrix, pos, energy, action):
    """Move the bee to 'action' and return: new energy, steps walked, nectar, reward, done."""
    steps = matrix[(pos, action)][1]
    energy = energy - steps

    if action == env.hive:                        # going home ends the trip
        reward = -COST_WEIGHT * steps
        return energy, steps, 0, reward, True

    collect = get_collect_cost(env)
    energy = energy - collect                     # collecting costs energy too
    nectar = env.flowers[action]
    reward = nectar - COST_WEIGHT * (steps + collect)
    return energy, steps, nectar, reward, False


def update_q(q, state, action, reward, next_state, next_actions):
    """The Q-Learning formula: move Q a little toward (reward + best future value)."""
    if len(next_actions) == 0:                    # the trip ended: no future
        future = 0
    else:
        future = max(q.get((next_state, a), 0.0) for a in next_actions)

    old = q.get((state, action), 0.0)
    q[(state, action)] = old + ALPHA * (reward + GAMMA * future - old)


def run_trip(env, matrix, q, epsilon, rng, learn=True):
    """One full trip: leave the hive, collect some flowers, come back."""
    pos = env.hive
    energy = env.energy_limit
    visited = []
    nectar = 0
    distance = 0
    total_reward = 0

    while True:
        state = get_state(pos, visited, energy)
        actions = get_valid_actions(env, matrix, pos, visited, energy)
        if len(actions) == 0:                     # nothing possible, stay at the hive
            break

        action = choose_action(q, state, actions, epsilon, rng)
        energy, steps, gained, reward, done = take_step(env, matrix, pos, energy, action)

        distance += steps
        nectar += gained
        total_reward += reward
        pos = action
        if action != env.hive:
            visited.append(action)

        if learn:
            next_state = get_state(pos, visited, energy)
            if done:
                next_actions = []
            else:
                next_actions = get_valid_actions(env, matrix, pos, visited, energy)
            update_q(q, state, action, reward, next_state, next_actions)

        if done:
            break

    return {
        "reward": total_reward,
        "nectar": nectar,
        "distance": distance,
        "energy_left": energy,
        "visited": visited,
        "returned": pos == env.hive,
    }


def build_full_route(env, visited, matrix):
    """Every cell the bee walks through: hive -> flowers -> hive."""
    stops = [env.hive] + list(visited) + [env.hive]
    route = [env.hive]
    for a, b in zip(stops, stops[1:]):
        route.extend(matrix[(a, b)][0][1:])
    return route


def q_learning_strategy(env, episodes=5000, seed=42, matrix=None):
    start = time.perf_counter()
    if matrix is None:
        matrix = build_distance_matrix(env)
    rng = random.Random(seed)
    q = {}                                        # the Q-table (starts empty)
    epsilon = EPS_START
    history = []                                  # reward of every training trip

    # 1) training: many trips, the bee learns from each one
    for _ in range(episodes):
        result = run_trip(env, matrix, q, epsilon, rng, learn=True)
        history.append(result["reward"])
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

    # 2) final trip with no random exploring: the best the bee learned
    final = run_trip(env, matrix, q, 0, rng, learn=False)

    return {
        "nectar": final["nectar"],
        "distance": final["distance"],
        "energy_used": env.energy_limit - final["energy_left"],
        "returned": final["returned"],
        "visited": final["visited"],
        "order": list(final["visited"]),
        "route": build_full_route(env, final["visited"], matrix),
        "history": history,
        "seconds": time.perf_counter() - start,
    }


def plot_learning_curve(history, window=100, path="learning_curve.png", show=False):
    """Learning Curve: reward per episode + moving average."""
    average = []
    for i in range(len(history)):
        chunk = history[max(0, i - window + 1): i + 1]
        average.append(sum(chunk) / len(chunk))

    plt.figure(figsize=(9, 5))
    plt.plot(history, alpha=0.25, label="Reward per episode")
    plt.plot(average, linewidth=2, label=f"Moving average ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title("Q-Learning Learning Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    from environment.garden import BeeGarden

    env = BeeGarden()
    res = q_learning_strategy(env)
    plot_learning_curve(res["history"], show=True)
    print("Q-Learning order:", res["order"])
    print("Visited:", res["visited"])
    print("Nectar:", res["nectar"], "| Energy used:", res["energy_used"],
          "| Returned:", res["returned"])
