class BeeEnvironment:
    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        self.hive = (0, 0)
        self.obstacles = set()
        self.flowers = {}
        self.energy_limit = 25 if width <= 8 else 60
        self.collect_cost = 2