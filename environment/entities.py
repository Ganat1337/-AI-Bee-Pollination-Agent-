class BeeEnvironment:
    def __init__(self, width=8, height=8):
        self.width = width
        self.height = height
        # Hive position and obstacles
        self.hive = (0, 0)
        self.obstacles = {(2, 2), (2, 3), (4, 5), (5, 5)}
        self.energy_limit = 100