"""
gui/simulation_ui.py - Interactive 2D Garden Game UI using pygame-ce.

Run from the project root:
    python -m gui.simulation_ui
"""

import math
import sys
import pygame

from environment.garden import BeeGarden
from search.astar import build_distance_matrix
from evaluation.search_based import search_based_strategy
from evaluation.metrics import check_result
from evaluation.compare import compare_strategies
from optimization import hill_climbing_strategy, genetic_algorithm_strategy
from learning.q_learning import q_learning_strategy
from gui.plots import plot_training_curve, plot_strategy_comparison


# Game Palette (Meadow Woodland + Warm Cream HUD + Honey Gold)
BG_WOODLAND = (22, 48, 36)
HEADER_BG = (30, 64, 48)
WOOD_DARK = (88, 57, 34)
WOOD_MID = (111, 74, 46)
WOOD_LIGHT = (141, 98, 65)
GRASS_LIGHT = (92, 184, 80)
GRASS_DARK = (79, 168, 68)
GRASS_BLADE = (43, 110, 36)

HUD_CREAM = (255, 252, 244)
HUD_BORDER = (203, 185, 157)
HUD_CREAM_ALT = (245, 236, 217)
TEXT_DARK = (59, 42, 26)
TEXT_MUTED = (107, 86, 62)

GOLD = (245, 158, 11)
GOLD_BRIGHT = (251, 191, 36)
GOLD_BORDER = (180, 83, 9)
EMERALD = (22, 163, 74)
EMERALD_DARK = (20, 83, 45)
ROSE = (225, 29, 72)

STRATEGIES = [
    ("astar_nearest", "A* Nearest Flower"),
    ("astar_ratio", "A* Best Nectar Ratio"),
    ("hill_climbing", "Hill Climbing"),
    ("genetic", "Genetic Algorithm"),
    ("q_learning", "Q-Learning"),
]

SPEED_OPTIONS = [
    ("0.5x", 18),
    ("1x", 10),
    ("2x", 5),
    ("4x", 2),
]


class Button:
    def __init__(self, rect, label, active=False, accent=False):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.active = active
        self.accent = accent

    def draw(self, surface, font):
        mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)

        if self.accent:
            bg = GOLD_BRIGHT if hovered else GOLD
            border = GOLD_BORDER
            fg = TEXT_DARK
        elif self.active:
            bg = (254, 243, 199)
            border = GOLD
            fg = TEXT_DARK
        else:
            bg = HUD_CREAM_ALT if hovered else (255, 255, 255)
            border = HUD_BORDER
            fg = TEXT_DARK

        pygame.draw.rect(surface, bg, self.rect, border_radius=7)
        pygame.draw.rect(surface, border, self.rect, width=2, border_radius=7)

        txt = font.render(self.label, True, fg)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


def build_timeline(env, result):
    """Precompute step-by-step position, energy used, nectar collected, and visited set."""
    route = result["route"]
    if not route:
        return []

    visit_order = list(result["visited"])
    next_idx = 0
    collected = set()
    nectar = 0
    collect_energy = 0
    timeline = []

    for step_idx, pos in enumerate(route):
        if next_idx < len(visit_order) and pos == visit_order[next_idx]:
            collected.add(pos)
            nectar += env.flowers.get(pos, 0)
            collect_energy += env.collect_cost
            next_idx += 1

        timeline.append({
            "pos": pos,
            "steps": step_idx,
            "energy_used": step_idx + collect_energy,
            "nectar": nectar,
            "visited": set(collected),
        })
    return timeline


def draw_rock(surface, cx, cy, cell_size, ox, oy):
    """Draw a sculpted grey boulder with facets, shadow, and companion pebble."""
    r = cell_size * 0.37
    variant = (ox * 31 + oy * 17) % 3

    # Ground shadow
    shadow_rect = pygame.Rect(cx - r * 0.9, cy + r * 0.25, r * 1.85, r * 0.8)
    pygame.draw.ellipse(surface, (42, 92, 36), shadow_rect)

    if variant == 0:
        pts = [
            (cx - r * 0.90, cy + r * 0.42),
            (cx - r * 0.72, cy - r * 0.45),
            (cx - r * 0.10, cy - r * 0.76),
            (cx + r * 0.68, cy - r * 0.52),
            (cx + r * 0.90, cy + r * 0.30),
            (cx + r * 0.24, cy + r * 0.66),
            (cx - r * 0.45, cy + r * 0.64),
        ]
    elif variant == 1:
        pts = [
            (cx - r * 0.88, cy + r * 0.38),
            (cx - r * 0.58, cy - r * 0.60),
            (cx + r * 0.18, cy - r * 0.74),
            (cx + r * 0.85, cy - r * 0.28),
            (cx + r * 0.82, cy + r * 0.44),
            (cx + r * 0.10, cy + r * 0.66),
            (cx - r * 0.54, cy + r * 0.60),
        ]
    else:
        pts = [
            (cx - r * 0.84, cy + r * 0.40),
            (cx - r * 0.78, cy - r * 0.34),
            (cx - r * 0.20, cy - r * 0.75),
            (cx + r * 0.56, cy - r * 0.62),
            (cx + r * 0.88, cy + r * 0.18),
            (cx + r * 0.46, cy + r * 0.64),
            (cx - r * 0.36, cy + r * 0.62),
        ]

    # Base grey stone
    pygame.draw.polygon(surface, (139, 146, 154), pts)

    # Sunlit top-left facet
    highlight_pts = [
        (cx - r * 0.62, cy - r * 0.34),
        (cx - r * 0.10, cy - r * 0.68),
        (cx + r * 0.46, cy - r * 0.44),
        (cx + r * 0.04, cy - r * 0.02),
        (cx - r * 0.50, cy + r * 0.04),
    ]
    pygame.draw.polygon(surface, (188, 195, 204), highlight_pts)

    # Shaded bottom-right facet
    shade_pts = [
        (cx + r * 0.04, cy - r * 0.02),
        (cx + r * 0.80, cy + r * 0.22),
        (cx + r * 0.26, cy + r * 0.60),
        (cx - r * 0.40, cy + r * 0.56),
        (cx - r * 0.50, cy + r * 0.04),
    ]
    pygame.draw.polygon(surface, (100, 107, 115), shade_pts)

    # Crisp dark rock outline and crack
    pygame.draw.polygon(surface, (69, 74, 82), pts, width=2)
    pygame.draw.lines(
        surface, (73, 79, 87), False,
        [(cx - r * 0.15, cy - r * 0.15), (cx + r * 0.14, cy + r * 0.10), (cx + r * 0.05, cy + r * 0.36)],
        width=1,
    )

    # Companion pebble
    peb_rect = pygame.Rect(cx + r * 0.52, cy + r * 0.38, r * 0.46, r * 0.34)
    pygame.draw.ellipse(surface, (156, 163, 175), peb_rect)
    pygame.draw.ellipse(surface, (75, 85, 99), peb_rect, width=1)


def draw_flower(surface, cx, cy, cell_size, nectar, visited, font):
    """Draw a blooming flower with leaves, layered petals, center disc, and nectar value."""
    if nectar >= 20:
        outer_r = cell_size * 0.40
        n_petals = 8
        petal_col = (225, 29, 72)
        petal_edge = (136, 19, 55)
        inner_col = (251, 113, 133)
        center_col = (127, 29, 29)
    elif nectar >= 10:
        outer_r = cell_size * 0.35
        n_petals = 7
        petal_col = (249, 115, 22)
        petal_edge = (154, 52, 18)
        inner_col = (253, 186, 116)
        center_col = (124, 45, 18)
    else:
        outer_r = cell_size * 0.30
        n_petals = 6
        petal_col = (253, 224, 71)
        petal_edge = (161, 98, 7)
        inner_col = (254, 249, 195)
        center_col = (217, 119, 6)

    if visited:
        pygame.draw.circle(surface, (254, 240, 138), (int(cx), int(cy)), int(outer_r + 4), width=2)

    # Green leaves peeking below petals
    leaf_r = int(outer_r * 0.35)
    pygame.draw.circle(surface, (21, 128, 61), (int(cx - outer_r * 0.58), int(cy + outer_r * 0.52)), leaf_r)
    pygame.draw.circle(surface, (22, 163, 74), (int(cx + outer_r * 0.58), int(cy + outer_r * 0.50)), leaf_r)

    # Outer petals
    petal_r = max(4, int(outer_r * 0.38))
    for i in range(n_petals):
        ang = 2 * math.pi * i / n_petals
        px = int(cx + math.cos(ang) * outer_r * 0.58)
        py = int(cy + math.sin(ang) * outer_r * 0.58)
        pygame.draw.circle(surface, petal_col, (px, py), petal_r)
        pygame.draw.circle(surface, petal_edge, (px, py), petal_r, width=1)

    # Inner petals for medium & large flowers
    if nectar >= 10:
        in_r = max(3, int(outer_r * 0.24))
        for i in range(5):
            ang = 0.3 + 2 * math.pi * i / 5
            px = int(cx + math.cos(ang) * outer_r * 0.38)
            py = int(cy + math.sin(ang) * outer_r * 0.38)
            pygame.draw.circle(surface, inner_col, (px, py), in_r)

    # Center disc + nectar number
    c_r = max(6, int(outer_r * 0.46))
    pygame.draw.circle(surface, center_col, (int(cx), int(cy)), c_r)
    pygame.draw.circle(surface, (255, 251, 235), (int(cx), int(cy)), c_r, width=1)

    txt = font.render(str(nectar), True, (255, 255, 255))
    surface.blit(txt, txt.get_rect(center=(int(cx), int(cy))))


def draw_hive(surface, cx, cy, cell_size):
    """Draw a tiered golden beehive skep at (0,0)."""
    r = cell_size * 0.42
    tiers = [
        (0.42, 0.90, 0.34, (217, 119, 6)),
        (0.10, 0.78, 0.32, (245, 158, 11)),
        (-0.22, 0.60, 0.28, (251, 191, 36)),
        (-0.48, 0.34, 0.18, (253, 230, 138)),
    ]
    for y_off, rx_mul, ry_mul, col in tiers:
        rect = pygame.Rect(
            cx - r * rx_mul,
            cy + r * y_off - r * ry_mul,
            r * rx_mul * 2,
            r * ry_mul * 2,
        )
        pygame.draw.ellipse(surface, col, rect)
        pygame.draw.ellipse(surface, (120, 53, 15), rect, width=1)

    # Hive entrance arch
    door = pygame.Rect(cx - r * 0.22, cy + r * 0.16, r * 0.44, r * 0.50)
    pygame.draw.rect(surface, (69, 26, 3), door, border_top_left_radius=6, border_top_right_radius=6)


def draw_bee(surface, cx, cy, cell_size, angle_deg, anim_tick, is_playing):
    """Draw a detailed illustrative bee sprite facing its flight direction."""
    sprite_size = max(48, int(cell_size * 1.45))
    bee_surf = pygame.Surface((sprite_size, sprite_size), pygame.SRCALPHA)
    mx = sprite_size / 2
    my = sprite_size / 2
    s = cell_size / 42.0

    # Ground shadow
    pygame.draw.ellipse(
        bee_surf, (20, 48, 24, 90),
        pygame.Rect(mx - 12 * s, my + 4 * s, 22 * s, 11 * s)
    )

    # Stinger
    pygame.draw.polygon(
        bee_surf, (41, 37, 36),
        [(mx - 12 * s, my), (mx - 18 * s, my - 2 * s), (mx - 18 * s, my + 2 * s)]
    )

    # Abdomen (yellow oval with 3 dark stripes)
    ab_rect = pygame.Rect(mx - 13 * s, my - 7.5 * s, 21 * s, 15 * s)
    pygame.draw.ellipse(bee_surf, (250, 204, 21), ab_rect)
    for sx in (-8.0, -2.5, 2.5):
        pygame.draw.line(
            bee_surf, (41, 37, 36),
            (mx + sx * s, my - 6 * s),
            (mx + sx * s, my + 6 * s),
            max(2, int(2.6 * s)),
        )
    pygame.draw.ellipse(bee_surf, (41, 37, 36), ab_rect, width=max(1, int(1.4 * s)))

    # Thorax
    pygame.draw.circle(bee_surf, (69, 26, 3), (int(mx + 6 * s), int(my)), max(3, int(5.6 * s)))

    # Fluttering translucent wings
    flap = math.sin(anim_tick * 0.55) * 2.5 * s if is_playing else 0.0
    wing_w = 13 * s
    wing_h = 8 * s - abs(flap) * 0.5
    top_wing = pygame.Rect(mx - 6 * s, my - 13 * s + flap, wing_w, max(4, wing_h))
    bot_wing = pygame.Rect(mx - 6 * s, my + 5 * s - flap, wing_w, max(4, wing_h))
    pygame.draw.ellipse(bee_surf, (240, 249, 255, 220), top_wing)
    pygame.draw.ellipse(bee_surf, (125, 211, 252, 230), top_wing, width=1)
    pygame.draw.ellipse(bee_surf, (240, 249, 255, 220), bot_wing)
    pygame.draw.ellipse(bee_surf, (125, 211, 252, 230), bot_wing, width=1)

    # Head & expressive eyes
    hx = int(mx + 11.5 * s)
    hy = int(my)
    pygame.draw.circle(bee_surf, (41, 37, 36), (hx, hy), max(3, int(4.4 * s)))
    eye_r = max(1, int(1.5 * s))
    pygame.draw.circle(bee_surf, (255, 255, 255), (int(hx + 1.5 * s), int(hy - 2 * s)), eye_r)
    pygame.draw.circle(bee_surf, (255, 255, 255), (int(hx + 1.5 * s), int(hy + 2 * s)), eye_r)
    pygame.draw.circle(bee_surf, (20, 20, 20), (int(hx + 2 * s), int(hy - 2 * s)), max(1, int(0.8 * s)))
    pygame.draw.circle(bee_surf, (20, 20, 20), (int(hx + 2 * s), int(hy + 2 * s)), max(1, int(0.8 * s)))

    # Antennae
    pygame.draw.lines(
        bee_surf, (41, 37, 36), False,
        [(hx + 2 * s, hy - 2 * s), (hx + 6 * s, hy - 5 * s), (hx + 8.5 * s, hy - 4 * s)],
        width=max(1, int(1.2 * s)),
    )
    pygame.draw.lines(
        bee_surf, (41, 37, 36), False,
        [(hx + 2 * s, hy + 2 * s), (hx + 6 * s, hy + 5 * s), (hx + 8.5 * s, hy + 4 * s)],
        width=max(1, int(1.2 * s)),
    )

    rotated = pygame.transform.rotozoom(bee_surf, -angle_deg, 1.0)
    surface.blit(rotated, rotated.get_rect(center=(int(cx), int(cy))))


def run_simulation_ui():
    pygame.init()
    screen_w, screen_h = 1140, 740
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Bee Meadow Quest - AI Pollination Agent")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("arial", 21, bold=True)
    font_head = pygame.font.SysFont("arial", 15, bold=True)
    font_body = pygame.font.SysFont("arial", 13)
    font_small = pygame.font.SysFont("arial", 11, bold=True)
    font_mono = pygame.font.SysFont("consolas", 14, bold=True)

    # Simulation state
    size = 15
    n_flowers = 10
    seed = 0
    strategy_idx = 4  # Q-Learning by default
    speed_idx = 1     # 1x

    env = None
    matrix = None
    result = None
    problems = []
    timeline = []
    step_idx = 0
    playing = False
    tick_counter = 0

    def rebuild_garden_and_run(new_garden=True):
        nonlocal env, matrix, result, problems, timeline, step_idx, playing
        if new_garden or env is None:
            env = BeeGarden(width=size, height=size, n_flowers=n_flowers, seed=seed)
            matrix = build_distance_matrix(env)

        key = STRATEGIES[strategy_idx][0]
        if key == "astar_nearest":
            result = search_based_strategy(env, matrix=matrix, rule="nearest")
        elif key == "astar_ratio":
            result = search_based_strategy(env, matrix=matrix, rule="best_ratio")
        elif key == "hill_climbing":
            result = hill_climbing_strategy(env, matrix=matrix)
        elif key == "genetic":
            result = genetic_algorithm_strategy(env, matrix=matrix)
        else:
            result = q_learning_strategy(env, matrix=matrix)

        problems = check_result(env, result)
        timeline = build_timeline(env, result)
        step_idx = 0
        playing = True

    rebuild_garden_and_run(new_garden=True)

    # Sidebar buttons (Left control panel)
    btn_8x8 = Button((24, 102, 130, 32), "8x8 (4 Flowers)")
    btn_15x15 = Button((164, 102, 146, 32), "15x15 (10 Flowers)")
    btn_seed_prev = Button((24, 152, 42, 30), "-")
    btn_seed_next = Button((268, 152, 42, 30), "+")

    strategy_buttons = []
    for i, (_, label) in enumerate(STRATEGIES):
        strategy_buttons.append(Button((24, 224 + i * 38, 286, 32), label))

    btn_play_run = Button((24, 426, 286, 38), "Play Bee Flight", accent=True)
    btn_plot_curve = Button((24, 476, 138, 32), "Training Curve")
    btn_benchmark = Button((172, 476, 138, 32), "Compare All")

    # Playback controls below board
    btn_reset = Button((356, 672, 72, 34), "Reset")
    btn_prev = Button((436, 672, 72, 34), "< Step")
    btn_toggle = Button((516, 672, 92, 34), "Pause", accent=True)
    btn_next = Button((616, 672, 72, 34), "Step >")
    btn_speed = Button((700, 672, 90, 34), "Speed 1x")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if step_idx >= len(timeline) - 1:
                        step_idx = 0
                    playing = not playing
                elif event.key == pygame.K_r:
                    step_idx = 0
                    playing = False
                elif event.key == pygame.K_RIGHT:
                    playing = False
                    step_idx = min(len(timeline) - 1, step_idx + 1)
                elif event.key == pygame.K_LEFT:
                    playing = False
                    step_idx = max(0, step_idx - 1)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if btn_8x8.clicked(pos):
                    size, n_flowers = 8, 4
                    rebuild_garden_and_run(new_garden=True)
                elif btn_15x15.clicked(pos):
                    size, n_flowers = 15, 10
                    rebuild_garden_and_run(new_garden=True)
                elif btn_seed_prev.clicked(pos):
                    seed = max(0, seed - 1)
                    rebuild_garden_and_run(new_garden=True)
                elif btn_seed_next.clicked(pos):
                    seed += 1
                    rebuild_garden_and_run(new_garden=True)
                elif btn_play_run.clicked(pos):
                    step_idx = 0
                    playing = True
                elif btn_plot_curve.clicked(pos):
                    if result["history"]:
                        strat_name = STRATEGIES[strategy_idx][1]
                        win = 100 if STRATEGIES[strategy_idx][0] == "q_learning" else None
                        plot_training_curve(
                            result["history"],
                            title=f"{strat_name} Progress (Seed {seed})",
                            window=win,
                            show=True,
                        )
                elif btn_benchmark.clicked(pos):
                    _, summary, _ = compare_strategies(seeds=(0, 1, 2), size=size, n_flowers=n_flowers)
                    plot_strategy_comparison(summary, show=True)
                elif btn_reset.clicked(pos):
                    playing = False
                    step_idx = 0
                elif btn_prev.clicked(pos):
                    playing = False
                    step_idx = max(0, step_idx - 1)
                elif btn_toggle.clicked(pos):
                    if step_idx >= len(timeline) - 1:
                        step_idx = 0
                    playing = not playing
                elif btn_next.clicked(pos):
                    playing = False
                    step_idx = min(len(timeline) - 1, step_idx + 1)
                elif btn_speed.clicked(pos):
                    speed_idx = (speed_idx + 1) % len(SPEED_OPTIONS)
                else:
                    for idx, b in enumerate(strategy_buttons):
                        if b.clicked(pos):
                            strategy_idx = idx
                            rebuild_garden_and_run(new_garden=False)
                            break

        # Advance animation
        tick_counter += 1
        step_delay = SPEED_OPTIONS[speed_idx][1]
        if playing and tick_counter % step_delay == 0:
            if step_idx < len(timeline) - 1:
                step_idx += 1
            else:
                playing = False

        # Current frame step
        cur = timeline[step_idx] if timeline else {
            "pos": env.hive, "steps": 0, "energy_used": 0, "nectar": 0, "visited": set()
        }
        prev_pos = timeline[step_idx - 1]["pos"] if step_idx > 0 else None

        # ---------------- DRAW ----------------
        screen.fill(BG_WOODLAND)

        # Header Bar
        pygame.draw.rect(screen, HEADER_BG, (0, 0, screen_w, 54))
        pygame.draw.line(screen, (15, 36, 26), (0, 54), (screen_w, 54), 2)
        title_surf = font_title.render("Bee Meadow Quest  |  AI Pollination Agent", True, (253, 230, 138))
        screen.blit(title_surf, (24, 15))
        hint_surf = font_body.render("Space: Play/Pause  |  Left/Right: Step  |  R: Reset", True, (167, 243, 208))
        screen.blit(hint_surf, (screen_w - hint_surf.get_width() - 24, 18))

        # Left Control Panel
        panel_rect = pygame.Rect(14, 68, 306, 650)
        pygame.draw.rect(screen, HUD_CREAM, panel_rect, border_radius=12)
        pygame.draw.rect(screen, HUD_BORDER, panel_rect, width=2, border_radius=12)

        screen.blit(font_head.render("Garden Map Setup", True, TEXT_DARK), (24, 78))
        btn_8x8.active = (size == 8)
        btn_15x15.active = (size == 15)
        btn_8x8.draw(screen, font_body)
        btn_15x15.draw(screen, font_body)

        btn_seed_prev.draw(screen, font_head)
        btn_seed_next.draw(screen, font_head)
        seed_box = pygame.Rect(74, 152, 186, 30)
        pygame.draw.rect(screen, (255, 255, 255), seed_box, border_radius=6)
        pygame.draw.rect(screen, HUD_BORDER, seed_box, width=1, border_radius=6)
        seed_txt = font_mono.render(f"Map Seed: {seed}", True, TEXT_DARK)
        screen.blit(seed_txt, seed_txt.get_rect(center=seed_box.center))

        screen.blit(font_head.render("Bee Brain Strategy", True, TEXT_DARK), (24, 198))
        for i, b in enumerate(strategy_buttons):
            b.active = (i == strategy_idx)
            b.draw(screen, font_body)

        btn_play_run.draw(screen, font_head)
        btn_plot_curve.draw(screen, font_body)
        btn_benchmark.draw(screen, font_body)

        # Flight Path Sequence Log inside Left Panel
        log_rect = pygame.Rect(24, 522, 286, 182)
        pygame.draw.rect(screen, HUD_CREAM_ALT, log_rect, border_radius=8)
        pygame.draw.rect(screen, HUD_BORDER, log_rect, width=1, border_radius=8)
        screen.blit(font_head.render("Pollination Sequence", True, TEXT_DARK), (34, 530))

        order_items = ["Hive (0,0)"] + [f"({x},{y}) +{env.flowers[(x, y)]}" for (x, y) in result["visited"]] + ["Hive (0,0)"]
        line_y = 556
        cur_line = ""
        for item in order_items:
            candidate = f"{cur_line} -> {item}" if cur_line else item
            if font_small.size(candidate)[0] > 266:
                screen.blit(font_small.render(cur_line, True, TEXT_MUTED), (34, line_y))
                line_y += 17
                cur_line = "-> " + item
            else:
                cur_line = candidate
        if cur_line and line_y < 690:
            screen.blit(font_small.render(cur_line, True, TEXT_MUTED), (34, line_y))

        # Top Game Info HUD Strip (Right Side above Board)
        hud_rect = pygame.Rect(338, 68, 782, 74)
        pygame.draw.rect(screen, HUD_CREAM, hud_rect, border_radius=12)
        pygame.draw.rect(screen, HUD_BORDER, hud_rect, width=2, border_radius=12)

        # HUD 1: Nectar Score
        screen.blit(font_small.render("Nectar Score", True, TEXT_MUTED), (356, 78))
        screen.blit(font_title.render(f"{cur['nectar']} / {result['nectar']}", True, GOLD_BORDER), (356, 100))

        # HUD 2: Energy Bar
        energy_left = max(0, env.energy_limit - cur["energy_used"])
        energy_ratio = energy_left / max(1, env.energy_limit)
        screen.blit(font_small.render(f"Bee Energy: {energy_left}/{env.energy_limit}", True, TEXT_MUTED), (490, 78))
        bar_bg = pygame.Rect(490, 104, 150, 18)
        pygame.draw.rect(screen, HUD_CREAM_ALT, bar_bg, border_radius=9)
        fill_col = EMERALD if energy_ratio > 0.4 else (GOLD if energy_ratio > 0.15 else ROSE)
        if energy_ratio > 0:
            pygame.draw.rect(
                screen, fill_col,
                pygame.Rect(490, 104, max(8, int(150 * energy_ratio)), 18),
                border_radius=9,
            )
        pygame.draw.rect(screen, HUD_BORDER, bar_bg, width=1, border_radius=9)

        # HUD 3: Flowers
        screen.blit(font_small.render("Flowers", True, TEXT_MUTED), (668, 78))
        screen.blit(font_title.render(f"{len(cur['visited'])}/{len(env.flowers)}", True, TEXT_DARK), (668, 100))

        # HUD 4: Distance
        screen.blit(font_small.render("Distance", True, TEXT_MUTED), (766, 78))
        screen.blit(font_title.render(f"{cur['steps']} steps", True, TEXT_DARK), (766, 100))

        # HUD 5: Status & Rules
        screen.blit(font_small.render("Mission Status", True, TEXT_MUTED), (898, 78))
        if step_idx < len(timeline) - 1:
            status_txt, status_col = "In Flight", GOLD_BORDER
        elif result["returned"] and not problems:
            status_txt, status_col = "Safe at Hive (PASS)", EMERALD_DARK
        else:
            status_txt, status_col = "Failed / Limit", ROSE
        screen.blit(font_head.render(status_txt, True, status_col), (898, 103))

        # Garden Board Area (Centered)
        board_avail = 480
        cell_size = board_avail // env.width
        grid_px = cell_size * env.width
        board_x = 338 + (782 - grid_px) // 2
        board_y = 160 + (496 - grid_px) // 2

        # Wooden garden fence frame
        fence_rect = pygame.Rect(board_x - 14, board_y - 14, grid_px + 28, grid_px + 28)
        pygame.draw.rect(screen, WOOD_MID, fence_rect, border_radius=10)
        pygame.draw.rect(screen, WOOD_DARK, fence_rect, width=3, border_radius=10)
        for px, py in [
            (fence_rect.left + 3, fence_rect.top + 3),
            (fence_rect.right - 11, fence_rect.top + 3),
            (fence_rect.left + 3, fence_rect.bottom - 11),
            (fence_rect.right - 11, fence_rect.bottom - 11),
        ]:
            pygame.draw.rect(screen, WOOD_LIGHT, (px, py, 8, 8), border_radius=2)

        # Grass Meadow Tiles
        for gx in range(env.width):
            for gy in range(env.height):
                tx = board_x + gx * cell_size
                ty = board_y + gy * cell_size
                tile_col = GRASS_LIGHT if (gx + gy) % 2 == 0 else GRASS_DARK
                pygame.draw.rect(screen, tile_col, (tx, ty, cell_size, cell_size))
                pygame.draw.rect(screen, (67, 147, 57), (tx, ty, cell_size, cell_size), width=1)

                # Grass tuft details on empty lawn cells
                if (gx, gy) not in env.obstacles and (gx, gy) not in env.flowers and (gx, gy) != env.hive:
                    h = (gx * 7 + gy * 13) % 5
                    if h == 0:
                        pygame.draw.line(screen, GRASS_BLADE, (tx + cell_size * 0.25, ty + cell_size * 0.72), (tx + cell_size * 0.20, ty + cell_size * 0.54), 1)
                        pygame.draw.line(screen, GRASS_BLADE, (tx + cell_size * 0.28, ty + cell_size * 0.72), (tx + cell_size * 0.34, ty + cell_size * 0.52), 1)
                    elif h == 3:
                        pygame.draw.circle(screen, (217, 249, 157), (int(tx + cell_size * 0.35), int(ty + cell_size * 0.36)), 2)

        # Grey Rocks
        for ox, oy in env.obstacles:
            draw_rock(screen, board_x + (ox + 0.5) * cell_size, board_y + (oy + 0.5) * cell_size, cell_size, ox, oy)

        # Planned & Traversed Route Trail
        if result["route"] and len(result["route"]) > 1:
            all_pts = [(board_x + (px + 0.5) * cell_size, board_y + (py + 0.5) * cell_size) for (px, py) in result["route"]]
            if step_idx < len(all_pts) - 1:
                pygame.draw.lines(screen, (254, 249, 195), False, all_pts[step_idx:], width=2)
            if step_idx >= 1:
                pygame.draw.lines(screen, (120, 53, 15), False, all_pts[:step_idx + 1], width=5)
                pygame.draw.lines(screen, (254, 240, 138), False, all_pts[:step_idx + 1], width=3)

        # Blooming Flowers
        for (fx, fy), nectar in env.flowers.items():
            draw_flower(
                screen,
                board_x + (fx + 0.5) * cell_size,
                board_y + (fy + 0.5) * cell_size,
                cell_size,
                nectar,
                (fx, fy) in cur["visited"],
                font_small,
            )

        # Beehive
        draw_hive(
            screen,
            board_x + (env.hive[0] + 0.5) * cell_size,
            board_y + (env.hive[1] + 0.5) * cell_size,
            cell_size,
        )

        # Real Bee Sprite
        bx, by = cur["pos"]
        angle = 25.0
        if prev_pos and prev_pos != cur["pos"]:
            angle = math.degrees(math.atan2(by - prev_pos[1], bx - prev_pos[0]))
        draw_bee(
            screen,
            board_x + (bx + 0.5) * cell_size,
            board_y + (by + 0.5) * cell_size,
            cell_size,
            angle,
            tick_counter,
            playing,
        )

        # Bottom Playback Control Bar
        bar_rect = pygame.Rect(338, 662, 782, 56)
        pygame.draw.rect(screen, HUD_CREAM, bar_rect, border_radius=12)
        pygame.draw.rect(screen, HUD_BORDER, bar_rect, width=2, border_radius=12)

        btn_toggle.label = "Pause" if playing else "Play"
        btn_speed.label = f"Speed {SPEED_OPTIONS[speed_idx][0]}"
        btn_reset.draw(screen, font_body)
        btn_prev.draw(screen, font_body)
        btn_toggle.draw(screen, font_head)
        btn_next.draw(screen, font_body)
        btn_speed.draw(screen, font_body)

        info_str = f"Tile ({bx},{by})  |  Step {cur['steps']}/{max(0, len(timeline) - 1)}  |  Exec: {result['seconds']*1000:.1f} ms"
        info_surf = font_mono.render(info_str, True, TEXT_DARK)
        screen.blit(info_surf, (805, 681))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run_simulation_ui()