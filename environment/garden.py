import random
from collections import deque

from environment.entities import BeeEnvironment


class BeeGarden(BeeEnvironment):
    def __init__(
        self,
        width=8,
        height=8,
        n_flowers=4,
        obstacle_ratio=0.17,
        seed=0,
        energy_limit=None,
        collect_cost=2,
    ):
        super().__init__(width, height)
        if energy_limit is None:
            energy_limit = 25 if width <= 8 else 60
        self.energy_limit = energy_limit
        self.collect_cost = collect_cost

        if width <= 0 or height <= 0:
            raise ValueError("Garden dimensions must be positive.")
        if n_flowers < 0:
            raise ValueError("The number of flowers cannot be negative.")
        if not 0 <= obstacle_ratio <= 1:
            raise ValueError("Obstacle ratio must be between 0 and 1.")

        rng = random.Random(seed)
        cells = [
            (x, y)
            for x in range(width)
            for y in range(height)
            if (x, y) != self.hive
        ]
        obstacle_count = round(width * height * obstacle_ratio)
        if obstacle_count + n_flowers > len(cells):
            raise ValueError("Not enough cells for the requested obstacles and flowers.")

        for _ in range(100):
            obstacles = set(rng.sample(cells, obstacle_count))
            free_cells = [cell for cell in cells if cell not in obstacles]
            flower_positions = rng.sample(free_cells, n_flowers)
            flowers = {
                pos: (5, 10, 20)[index % 3]
                for index, pos in enumerate(flower_positions)
            }

            self.obstacles = obstacles
            self.flowers = flowers
            reachable = self._get_reachable_positions()
            if all(flower in reachable for flower in flowers):
                return

        raise ValueError("Could not generate a garden with all flowers reachable.")

    def get_neighbors(self, pos):
        """Get valid neighboring positions (up, down, left, right) avoiding obstacles and borders."""
        x, y = pos
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in self.obstacles:
                    neighbors.append((nx, ny))
        return neighbors

    def _get_reachable_positions(self):
        reachable = {self.hive}
        queue = deque([self.hive])

        while queue:
            current = queue.popleft()
            for neighbor in self.get_neighbors(current):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append(neighbor)

        return reachable


if __name__ == "__main__":
    from search.astar import a_star_search

    for seed in range(3):
        garden = BeeGarden(width=15, height=15, n_flowers=10, seed=seed)
        repeated_garden = BeeGarden(width=15, height=15, n_flowers=10, seed=seed)

        deterministic = (
            garden.hive == repeated_garden.hive
            and garden.obstacles == repeated_garden.obstacles
            and garden.flowers == repeated_garden.flowers
        )
        print(f"Seed {seed} determinism: {'PASS' if deterministic else 'FAIL'}")

        reachable = garden._get_reachable_positions()
        all_reachable = all(flower in reachable for flower in garden.flowers)
        print(f"Seed {seed} flower reachability: {'PASS' if all_reachable else 'FAIL'}")

        current = garden.hive
        remaining = set(garden.flowers)
        tour_cost = 0
        while remaining:
            options = [
                (a_star_search(garden, current, flower)[1], flower)
                for flower in remaining
            ]
            step_cost, next_flower = min(options)
            tour_cost += step_cost + garden.collect_cost
            current = next_flower
            remaining.remove(next_flower)

        return_cost = a_star_search(garden, current, garden.hive)[1]
        tour_cost += return_cost
        binding = tour_cost > garden.energy_limit
        print(f"Seed {seed} energy binding: {'PASS' if binding else 'FAIL'}")
        if not binding:
            print(
                f"  visit-all cost={tour_cost}, "
                f"energy_limit={garden.energy_limit}"
            )