"""
gui/simulation_ui.py  -  Pygame window that shows the bee moving in the garden.

Run from the project root (the folder that has main.py):
    python -m gui.simulation_ui

Keys:
    1 2 3 4     choose the strategy (A*, Hill Climbing, Genetic, Q-Learning)
    SPACE       pause / play
    R           restart the bee
    UP / DOWN   faster / slower
    N           next map (next seed)
    ESC         quit
"""

import pygame

from environment.garden import BeeGarden
from search.astar import build_distance_matrix
from optimization import hill_climbing_strategy, genetic_algorithm_strategy
from learning.q_learning import q_learning_strategy
from evaluation.search_based import search_based_strategy

SIZE = 15
N_FLOWERS = 10
CELL = 40                    # size of one cell in pixels
PANEL = 280                  # width of the info panel on the right

# the strategies, same order as in evaluation/compare.py
STRATEGIES = [
    ("A* + Nearest Flower", search_based_strategy),
    ("Hill Climbing", hill_climbing_strategy),
    ("Genetic Algorithm", genetic_algorithm_strategy),
    ("Q-Learning", q_learning_strategy),
]

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRASS_1 = (126, 200, 80)          # two greens, drawn like a checkerboard
GRASS_2 = (112, 188, 70)
ROCK = (125, 125, 130)
ROCK_LIGHT = (165, 165, 170)
HIVE_COLOR = (235, 170, 40)
HIVE_DARK = (170, 110, 20)
BEE_YELLOW = (255, 205, 0)
WING = (200, 235, 255)
TRAIL = (255, 255, 255)
PANEL_BG = (250, 244, 220)
STEM = (40, 130, 40)
FLOWER_CENTER = (255, 215, 0)
# petal colors and size for each nectar amount (small / medium / large flower)
FLOWER_PETALS = {5: (255, 170, 200), 10: (255, 110, 110), 20: (190, 100, 230)}
FLOWER_RADIUS = {5: 7, 10: 9, 20: 11}
WILTED = (200, 190, 170)          # a flower that was already collected


def run_all_strategies(env):
    """Run every strategy on this garden and return {name: result}."""
    matrix = build_distance_matrix(env)
    results = {}
    for name, strategy in STRATEGIES:
        results[name] = strategy(env, matrix=matrix)
    return results


class Simulation:
    def __init__(self, seed=0):
        self.seed = seed
        self.speed = 6                 # steps per second
        self.paused = False
        self.load_map(seed)

    def load_map(self, seed):
        self.seed = seed
        self.env = BeeGarden(SIZE, SIZE, n_flowers=N_FLOWERS, seed=seed)
        self.results = run_all_strategies(self.env)
        self.strategy_index = 0
        self.restart()

    def restart(self):
        name = STRATEGIES[self.strategy_index][0]
        self.result = self.results[name]
        self.route = self.result["route"]
        self.step = 0                          # index in the route
        self.energy = self.env.energy_limit
        self.nectar = 0
        self.collected = set()
        self.trail = [self.route[0]]
        self.facing = 1                        # 1 = looking right, -1 = looking left
        self.timer = 0.0
        self.paused = False

    def choose_strategy(self, index):
        self.strategy_index = index
        self.restart()

    def finished(self):
        return self.step >= len(self.route) - 1

    def move_one_step(self):
        """Move the bee one cell on its route and update energy and nectar."""
        if self.finished():
            return
        self.step += 1
        pos = self.route[self.step]
        dx = pos[0] - self.route[self.step - 1][0]
        if dx != 0:
            self.facing = 1 if dx > 0 else -1
        self.trail.append(pos)
        self.energy -= 1                       # every step costs 1 energy

        # collect nectar if this is a flower the strategy chose to visit
        if (pos in self.env.flowers and pos in self.result["visited"]
                and pos not in self.collected):
            self.collected.add(pos)
            self.energy -= self.env.collect_cost
            self.nectar += self.env.flowers[pos]

    def update(self, seconds):
        if self.paused or self.finished():
            return
        self.timer += seconds
        while self.timer >= 1.0 / self.speed:
            self.timer -= 1.0 / self.speed
            self.move_one_step()


def cell_rect(pos):
    x, y = pos
    return pygame.Rect(x * CELL, y * CELL, CELL, CELL)


def draw_flower(screen, center, value, collected):
    """A flower with a stem, 6 petals and a yellow center."""
    cx, cy = center
    r = FLOWER_RADIUS[value]
    petal = WILTED if collected else FLOWER_PETALS[value]
    middle = (150, 130, 90) if collected else FLOWER_CENTER

    pygame.draw.line(screen, STEM, (cx, cy), (cx, cy + 17), 3)         # stem
    pygame.draw.ellipse(screen, STEM, (cx, cy + 9, 9, 5))              # leaf

    offsets = [(-1, 0), (1, 0), (-0.5, -0.87), (0.5, -0.87), (-0.5, 0.87), (0.5, 0.87)]
    for ox, oy in offsets:
        pygame.draw.circle(screen, petal, (int(cx + ox * r * 0.8), int(cy + oy * r * 0.8)), int(r * 0.6))
    pygame.draw.circle(screen, middle, (cx, cy), int(r * 0.5))


def draw_hive(screen, rect):
    """A beehive made of 3 stacked rounded layers with a dark door."""
    cx = rect.centerx
    for i, width in enumerate((30, 34, 26)):
        y = rect.bottom - 9 - i * 10
        pygame.draw.ellipse(screen, HIVE_COLOR, (cx - width // 2, y - 6, width, 13))
        pygame.draw.ellipse(screen, HIVE_DARK, (cx - width // 2, y - 6, width, 13), 2)
    pygame.draw.ellipse(screen, BLACK, (cx - 4, rect.bottom - 14, 8, 8))   # door


def draw_rock(screen, rect):
    """An obstacle: a gray rock with a light spot."""
    body = rect.inflate(-6, -12).move(0, 3)
    pygame.draw.ellipse(screen, ROCK, body)
    pygame.draw.ellipse(screen, ROCK_LIGHT, (body.x + 6, body.y + 4, body.width // 2, body.height // 3))
    pygame.draw.ellipse(screen, BLACK, body, 1)


def draw_bee(screen, center, facing):
    """A bee: yellow body, black stripes, two wings, a head and a small stinger."""
    cx, cy = center
    a, b = 13, 9                                    # half width / half height of the body

    # wings (a little see-through)
    wing_surface = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
    pygame.draw.ellipse(wing_surface, WING + (200,), (8, 0, 12, 16))
    pygame.draw.ellipse(wing_surface, WING + (200,), (18, 2, 12, 16))
    screen.blit(wing_surface, (cx - CELL // 2, cy - CELL // 2 - 8))

    # body
    pygame.draw.ellipse(screen, BEE_YELLOW, (cx - a, cy - b, 2 * a, 2 * b))
    for dx in (-5, 3):                              # stripes (only inside the body)
        chord = int(b * (1 - (dx / a) ** 2) ** 0.5)
        pygame.draw.line(screen, BLACK, (cx + dx, cy - chord), (cx + dx, cy + chord), 4)
    pygame.draw.ellipse(screen, BLACK, (cx - a, cy - b, 2 * a, 2 * b), 2)

    # head and eye on the side the bee is looking at
    hx = cx + facing * (a + 2)
    pygame.draw.circle(screen, BLACK, (hx, cy), 6)
    pygame.draw.circle(screen, WHITE, (hx + facing * 2, cy - 2), 2)
    # stinger on the other side
    pygame.draw.polygon(screen, BLACK, [(cx - facing * a, cy - 2), (cx - facing * (a + 5), cy), (cx - facing * a, cy + 2)])


def draw_garden(screen, sim, font):
    env = sim.env

    # grass (checkerboard of two greens)
    for x in range(env.width):
        for y in range(env.height):
            color = GRASS_1 if (x + y) % 2 == 0 else GRASS_2
            pygame.draw.rect(screen, color, cell_rect((x, y)))

    # obstacles
    for pos in env.obstacles:
        draw_rock(screen, cell_rect(pos))

    # the path the bee walked (white dots)
    for pos in sim.trail:
        pygame.draw.circle(screen, TRAIL, cell_rect(pos).center, 3)

    # the hive
    draw_hive(screen, cell_rect(env.hive))

    # flowers, with the nectar number under each one
    for pos, value in env.flowers.items():
        rect = cell_rect(pos)
        center = (rect.centerx, rect.centery - 6)
        draw_flower(screen, center, value, pos in sim.collected)
        text = font.render(str(value), True, WHITE)
        shadow = font.render(str(value), True, BLACK)
        screen.blit(shadow, (rect.right - 13, rect.bottom - 17))
        screen.blit(text, (rect.right - 14, rect.bottom - 18))

    # the bee
    draw_bee(screen, cell_rect(sim.route[sim.step]).center, sim.facing)


def draw_panel(screen, sim, font, big_font):
    env = sim.env
    x0 = env.width * CELL
    pygame.draw.rect(screen, PANEL_BG, pygame.Rect(x0, 0, PANEL, env.height * CELL))

    name = STRATEGIES[sim.strategy_index][0]
    lines = [
        ("Bee Pollination Agent", big_font),
        (f"Map (seed): {sim.seed}", font),
        ("", font),
        (f"Strategy: {name}", big_font),
        (f"Energy: {sim.energy} / {env.energy_limit}", font),
        (f"Nectar: {sim.nectar}", font),
        (f"Steps: {sim.step} / {len(sim.route) - 1}", font),
        (f"Flowers: {len(sim.collected)} / {len(sim.result['visited'])}", font),
        (f"Speed: {sim.speed} steps/sec", font),
        ("", font),
    ]
    if sim.finished():
        lines.append(("Back at the hive!" if sim.result["returned"] else "Did not return", big_font))
        lines.append((f"Final nectar: {sim.result['nectar']}", font))
    elif sim.paused:
        lines.append(("PAUSED", big_font))

    lines += [
        ("", font),
        ("Keys:", big_font),
        ("1-4  choose strategy", font),
        ("SPACE  pause / play", font),
        ("R  restart", font),
        ("UP/DOWN  speed", font),
        ("N  next map", font),
        ("ESC  quit", font),
    ]

    y = 10
    for text, f in lines:
        screen.blit(f.render(text, True, BLACK), (x0 + 12, y))
        y += 24 if f is font else 30

    # energy bar
    bar = pygame.Rect(x0 + 12, env.height * CELL - 30, PANEL - 24, 16)
    pygame.draw.rect(screen, WHITE, bar)
    left = max(sim.energy, 0) / env.energy_limit
    pygame.draw.rect(screen, (60, 170, 60), (bar.x, bar.y, int(bar.width * left), bar.height))
    pygame.draw.rect(screen, BLACK, bar, 1)


def main():
    pygame.init()
    sim = Simulation(seed=0)
    width = SIZE * CELL + PANEL
    height = SIZE * CELL
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("AI Bee Pollination Agent")
    font = pygame.font.SysFont("arial", 16)
    big_font = pygame.font.SysFont("arial", 18, bold=True)
    clock = pygame.time.Clock()

    running = True
    while running:
        seconds = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sim.paused = not sim.paused
                elif event.key == pygame.K_r:
                    sim.restart()
                elif event.key == pygame.K_UP:
                    sim.speed = min(sim.speed + 2, 60)
                elif event.key == pygame.K_DOWN:
                    sim.speed = max(sim.speed - 2, 1)
                elif event.key == pygame.K_n:
                    sim.load_map(sim.seed + 1)
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    sim.choose_strategy(event.key - pygame.K_1)

        sim.update(seconds)
        draw_garden(screen, sim, font)
        draw_panel(screen, sim, font, big_font)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()