from environment.entities import BeeEnvironment

class BeeGarden(BeeEnvironment):
    def __init__(self, width=8, height=8):
        super().__init__(width, height)
        # Flowers with different nectar values
        self.flowers = {
            (1, 5): 5,   # Small flower (nectar = 5)
            (6, 2): 10,  # Medium flower (nectar = 10)
            (7, 7): 20   # Large flower (nectar = 20)
        }

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
if __name__ == "__main__":
    garden = BeeGarden()
    print("Flowers:", garden.flowers)
    print("Valid neighbors for (0, 0):", garden.get_neighbors((0, 0)))        