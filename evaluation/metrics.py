"""
evaluation/metrics.py  -  The 5 metrics required by the project description.

Every strategy returns a "result" dictionary with the same keys
(nectar, distance, energy_used, returned, visited, route, seconds, ...).
compute_metrics() turns that dictionary into the 5 numbers we compare.
check_result() makes sure a result follows the garden rules.
"""

METRIC_NAMES = [
    "Total Nectar Collected",
    "Travel Distance",
    "Energy Consumption",
    "Successful Hive Returns",
    "Execution Time (s)",
]


def compute_metrics(result):
    """The 5 metrics from the project description."""
    return {
        "Total Nectar Collected": result["nectar"],
        "Travel Distance": result["distance"],
        "Energy Consumption": result["energy_used"],
        "Successful Hive Returns": 1 if result["returned"] else 0,
        "Execution Time (s)": round(result["seconds"], 4),
    }


def check_result(env, result):
    """Look for rule violations. Returns a list of problems (empty list = all good)."""
    problems = []
    route = result["route"]
    collect = getattr(env, "collect_cost", 2)

    if route[0] != env.hive or route[-1] != env.hive:
        problems.append("route does not start and end at the hive")

    for a, b in zip(route, route[1:]):
        one_step = abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        if not one_step or b in env.obstacles:
            problems.append(f"bad step {a} -> {b}")
            break

    if len(route) - 1 != result["distance"]:
        problems.append("distance does not match the route length")

    expected_energy = result["distance"] + collect * len(result["visited"])
    if result["energy_used"] != expected_energy:
        problems.append("energy_used does not match distance + collect cost")

    if result["energy_used"] > env.energy_limit:
        problems.append("used more energy than the limit")

    if not result["returned"]:
        problems.append("the bee did not return to the hive")

    return problems