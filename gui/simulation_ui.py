"""
gui/simulation_ui.py  -  Animated dashboard of the bee (matplotlib window).

Run from the project root (the folder that has main.py):
    python -m gui.simulation_ui          (garden seed 0)
    python -m gui.simulation_ui 5        (garden seed 5)

What you see:
    left   : the garden game: the bee flying, flowers turn gray when collected
    right  : energy over time, how the algorithm improves, and a dashboard
    bottom : choose the algorithm, Pause, Restart, New garden, Compare all, speed

No pygame needed, only matplotlib.
"""

import random
import sys

import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon
from matplotlib.widgets import Button, RadioButtons, Slider

from environment.garden import BeeGarden
from gui.plots import (COLORS, OPTIMAL, SHORT, SIZE, N_FLOWERS, STRATEGIES,
                       best_strategy, energy_profile,
                       run_everything, show_comparison)

START_DELAY = 250          # milliseconds between two steps

# what to write above the "how the algorithm improves" chart
HISTORY_INFO = {
    "A* + Nearest Flower": "no iterations (it always goes to the nearest flower)",
    "Hill Climbing": "best score after each restart",
    "Genetic Algorithm": "best score of each generation",
    "Q-Learning": "reward of each training trip (smoothed)",
    OPTIMAL: "no iterations (it checks every possible group of flowers)",
}


# colors of the game look
GRASS_1 = "#7ec850"                  # two greens, like a checkerboard
GRASS_2 = "#72bd48"
GRASS_TUFT = "#5aa636"
ROCK = "#85858c"
ROCK_DARK = "#4a4a4f"
ROCK_LIGHT = "#b4b4ba"
HIVE_1 = "#ebaa28"
HIVE_2 = "#f5c850"
HIVE_DARK = "#8c5a10"
BEE_YELLOW = "#ffcd00"
WING = "#c8ebff"
STEM = "#2f8a2f"
FLOWER_CENTER = "#ffd700"
WILTED = "#cfc5b0"                   # a flower that was already collected
WILTED_EDGE = "#a89c85"
WILTED_CENTER = "#96825a"
FLOWER_PETALS = {5: "#ffaac8", 10: "#ff6e6e", 20: "#be64e6"}     # by nectar amount
FLOWER_EDGES = {5: "#e07aa0", 10: "#d04646", 20: "#8e3cb4"}
FLOWER_RADIUS = {5: 0.17, 10: 0.22, 20: 0.28}                    # bigger flower = more nectar
PETAL_OFFSETS = [(-1, 0), (1, 0), (-0.5, -0.87), (0.5, -0.87), (-0.5, 0.87), (0.5, 0.87)]


def draw_game_garden(ax, env):
    """Grass with little tufts, rocks (obstacles) and the beehive."""
    ax.clear()
    w, h = env.width, env.height

    # grass: two greens like a checkerboard
    green1 = np.array(mcolors.to_rgb(GRASS_1))
    green2 = np.array(mcolors.to_rgb(GRASS_2))
    grass = np.zeros((h, w, 3))
    for x in range(w):
        for y in range(h):
            grass[y, x] = green1 if (x + y) % 2 == 0 else green2
    ax.imshow(grass, extent=(-0.5, w - 0.5, h - 0.5, -0.5), interpolation="nearest", zorder=0)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():                  # a dark green frame around the garden
        spine.set_color("#3a6b2a")
        spine.set_linewidth(3)

    # small grass tufts (the same places every time, so the garden does not flicker)
    rng = random.Random(7)
    tufts = []
    for x in range(w):
        for y in range(h):
            for _ in range(2):
                tx = x + rng.uniform(-0.4, 0.4)
                ty = y + rng.uniform(-0.3, 0.4)
                for dx in (-0.06, 0.0, 0.06):
                    tufts.append([(tx, ty), (tx + dx * 2, ty - 0.13)])
    ax.add_collection(LineCollection(tufts, colors=GRASS_TUFT, linewidths=0.9, zorder=0.5))

    # obstacles are rocks with a shadow, a light spot and a small crack
    for (x, y) in env.obstacles:
        r = random.Random(x * 100 + y)
        size = r.uniform(0.9, 1.08)
        ax.add_patch(Ellipse((x + 0.05, y + 0.26), 0.82 * size, 0.2, facecolor="black",
                             alpha=0.25, zorder=1))
        ax.add_patch(Ellipse((x, y + 0.06), 0.82 * size, 0.64 * size, facecolor=ROCK,
                             edgecolor=ROCK_DARK, linewidth=1.2, zorder=2))
        ax.add_patch(Ellipse((x - 0.13, y - 0.07), 0.3 * size, 0.16, facecolor=ROCK_LIGHT, zorder=3))
        ax.add_line(Line2D([x + 0.1, x + 0.17, x + 0.12], [y - 0.02, y + 0.08, y + 0.17],
                           color=ROCK_DARK, linewidth=0.9, zorder=3))

    # the hive: 3 stripes of honey color, a roof and a dark door (bottom layer first)
    hx, hy = env.hive
    ax.add_patch(Ellipse((hx, hy + 0.36), 0.95, 0.16, facecolor="black", alpha=0.25, zorder=3))
    for i, (width, dy) in enumerate(((0.85, 0.24), (0.95, -0.02), (0.74, -0.26))):
        ax.add_patch(Ellipse((hx, hy + dy), width, 0.32, facecolor=HIVE_1 if i % 2 == 0 else HIVE_2,
                             edgecolor=HIVE_DARK, linewidth=1.8, zorder=4))
    ax.add_patch(Ellipse((hx, hy - 0.42), 0.18, 0.1, facecolor=HIVE_DARK, zorder=4))
    ax.add_patch(Ellipse((hx, hy + 0.28), 0.2, 0.22, facecolor="black", edgecolor=HIVE_DARK,
                         linewidth=1.5, zorder=5))


def make_flower(ax, pos, value):
    """A flower: shadow, stem, 2 leaves, 6 petals, a center and a nectar badge."""
    x, y = pos[0], pos[1] - 0.12
    r = FLOWER_RADIUS[value]
    ax.add_patch(Ellipse((x, y + 0.45), 0.34, 0.08, facecolor="black", alpha=0.2, zorder=3))
    ax.add_line(Line2D([x, x], [y, y + 0.42], color=STEM, linewidth=3, zorder=3))
    ax.add_patch(Ellipse((x + 0.11, y + 0.27), 0.22, 0.1, angle=-30, facecolor=STEM, zorder=3))
    ax.add_patch(Ellipse((x - 0.11, y + 0.18), 0.22, 0.1, angle=30, facecolor=STEM, zorder=3))

    petals = []
    for ox, oy in PETAL_OFFSETS:
        petal = Circle((x + ox * r * 0.8, y + oy * r * 0.8), r * 0.6, facecolor=FLOWER_PETALS[value],
                       edgecolor=FLOWER_EDGES[value], linewidth=0.8, zorder=4)
        ax.add_patch(petal)
        petals.append(petal)
    center = Circle((x, y), r * 0.5, facecolor=FLOWER_CENTER, edgecolor="#b8860b",
                    linewidth=0.8, zorder=5)
    ax.add_patch(center)

    # nectar amount in a small white badge
    ax.add_patch(Circle((pos[0] + 0.3, pos[1] + 0.32), 0.14, facecolor="white", edgecolor="black",
                        linewidth=1, zorder=6))
    ax.text(pos[0] + 0.3, pos[1] + 0.32, str(value), fontsize=6, fontweight="bold", color="black",
            ha="center", va="center", zorder=7)
    return {"pos": pos, "value": value, "petals": petals, "center": center}


class BeeApp:
    def __init__(self, seed=0):
        self.playing = True
        self.step = 0
        self.name = STRATEGIES[0][0]

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("Bee Pollination Simulation")

        # --- the plots -------------------------------------------------
        self.ax_garden = self.fig.add_axes([0.03, 0.17, 0.42, 0.76])
        self.ax_energy = self.fig.add_axes([0.54, 0.74, 0.43, 0.17])
        self.ax_history = self.fig.add_axes([0.54, 0.51, 0.43, 0.15])
        self.ax_bar = self.fig.add_axes([0.54, 0.40, 0.43, 0.03])
        self.ax_info = self.fig.add_axes([0.54, 0.13, 0.43, 0.24])
        self.ax_info.axis("off")
        self.info = self.ax_info.text(0.0, 1.0, "", va="top", family="monospace", fontsize=10)

        # the energy bar (gray = full battery, orange = energy left)
        self.ax_bar.barh([0], [1], color="#e5e7eb", edgecolor="#777777")
        self.bar_fill = self.ax_bar.barh([0], [1], color="#f59e0b")[0]
        self.ax_bar.set_xlim(0, 1)
        self.ax_bar.set_xticks([])
        self.ax_bar.set_yticks([])
        self.ax_bar.set_title("Dashboard", fontsize=12)

        # --- the controls ----------------------------------------------
        labels = [name for name, _ in STRATEGIES] + [OPTIMAL]
        self.ax_radio = self.fig.add_axes([0.03, 0.01, 0.19, 0.15])
        self.ax_radio.set_title("Algorithm", fontsize=9, loc="left")
        self.radio = RadioButtons(self.ax_radio, labels, active=0)
        self.radio.on_clicked(self.on_choose)

        self.btn_play = Button(self.fig.add_axes([0.26, 0.045, 0.09, 0.06]), "Pause")
        self.btn_restart = Button(self.fig.add_axes([0.36, 0.045, 0.09, 0.06]), "Restart")
        self.btn_new = Button(self.fig.add_axes([0.46, 0.045, 0.09, 0.06]), "New garden")
        self.btn_compare = Button(self.fig.add_axes([0.56, 0.045, 0.10, 0.06]), "Compare all")
        self.btn_play.on_clicked(self.on_play)
        self.btn_restart.on_clicked(self.on_restart)
        self.btn_new.on_clicked(self.on_new)
        self.btn_compare.on_clicked(self.on_compare)

        self.slider = Slider(self.fig.add_axes([0.76, 0.06, 0.18, 0.03]), "Delay (ms)",
                             20, 1000, valinit=START_DELAY, valstep=10)
        self.slider.on_changed(self.on_speed)

        # --- timer: calls self.tick() again and again ------------------
        self.timer = self.fig.canvas.new_timer(interval=START_DELAY)
        self.timer.add_callback(self.tick)

        self.load_garden(seed)
        self.timer.start()

    # ------------------------------------------------------------------
    # loading a garden and choosing an algorithm
    # ------------------------------------------------------------------
    def load_garden(self, seed):
        self.ax_garden.set_title("Computing all the algorithms...", fontsize=12)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        while True:
            try:
                self.env = BeeGarden(SIZE, SIZE, n_flowers=N_FLOWERS, seed=seed)
                break
            except ValueError:               # a garden that could not be built
                seed += 1
        self.seed = seed
        self.results = run_everything(self.env)
        self.best = best_strategy(self.results)

        # the energy line of every algorithm (for the comparison on the chart)
        self.profiles = {n: energy_profile(self.env, r) for n, r in self.results.items()}
        self.max_steps = max(len(p) for p in self.profiles.values())

        if self.name not in self.results:
            self.name = STRATEGIES[0][0]
        self.choose(self.name)

    def choose(self, name):
        """Show one algorithm from the start."""
        self.name = name
        self.result = self.results[name]
        self.states = self.profiles[name]
        self.step = 0
        self.playing = True
        self.btn_play.label.set_text("Pause")

        self.setup_garden()
        self.setup_energy_chart()
        self.setup_history_chart()
        self.render()

    def setup_garden(self):
        ax = self.ax_garden
        draw_game_garden(ax, self.env)

        # the white dots show where the bee has already been
        self.trail, = ax.plot([], [], "o", color="white", markersize=4, alpha=0.85, zorder=3)

        self.flowers = [make_flower(ax, pos, value) for pos, value in self.env.flowers.items()]

        # a yellow dashed ring around the place the bee is flying to
        self.target_ring = Circle((0, 0), 0.5, fill=False, edgecolor="#fff176", linewidth=2.5,
                                  linestyle="--", zorder=9)
        ax.add_patch(self.target_ring)

        # the bee: shadow, wings, body, 2 stripes, head, eye, 2 antennae and a stinger
        self.bee = {
            "shadow": Ellipse((0, 0), 0.6, 0.12, facecolor="black", alpha=0.25, zorder=6),
            "wing1": Ellipse((0, 0), 0.3, 0.44, facecolor=WING, edgecolor="#88aabb", alpha=0.85, zorder=6),
            "wing2": Ellipse((0, 0), 0.3, 0.44, facecolor=WING, edgecolor="#88aabb", alpha=0.85, zorder=6),
            "body": Ellipse((0, 0), 0.76, 0.52, facecolor=BEE_YELLOW, edgecolor="black", linewidth=1.5, zorder=7),
            "stripe1": Line2D([], [], color="black", linewidth=3.5, zorder=8),
            "stripe2": Line2D([], [], color="black", linewidth=3.5, zorder=8),
            "ant1": Line2D([], [], color="black", linewidth=1.2, zorder=8),
            "ant2": Line2D([], [], color="black", linewidth=1.2, zorder=8),
            "head": Circle((0, 0), 0.17, facecolor="black", zorder=8),
            "eye": Circle((0, 0), 0.05, facecolor="white", zorder=9),
            "sting": Polygon(np.zeros((3, 2)), closed=True, facecolor="black", zorder=7),
        }
        for part in self.bee.values():
            if isinstance(part, Line2D):
                ax.add_line(part)
            else:
                ax.add_patch(part)

        # which way the bee looks (1 = right, -1 = left) for every step
        self.facings = [1]
        for i in range(1, len(self.states)):
            dx = self.states[i]["pos"][0] - self.states[i - 1]["pos"][0]
            self.facings.append(1 if dx > 0 else -1 if dx < 0 else self.facings[-1])

    def place_bee(self, x, y, f, flap):
        """Move all the parts of the bee to the cell (x, y). f = which way it looks."""
        b = self.bee
        b["shadow"].set_center((x, y + 0.4))
        tilt = 25 if flap else -10                          # the wings move a little each step
        b["wing1"].set_center((x - 0.1, y - 0.33))
        b["wing1"].angle = tilt
        b["wing2"].set_center((x + 0.12, y - 0.31))
        b["wing2"].angle = -tilt
        b["body"].set_center((x, y))
        for line, dx in ((b["stripe1"], -0.14), (b["stripe2"], 0.08)):
            chord = 0.26 * (1 - (dx / 0.38) ** 2) ** 0.5     # stripe stays inside the body
            line.set_data([x + dx, x + dx], [y - chord, y + chord])
        b["head"].set_center((x + f * 0.42, y))
        b["eye"].set_center((x + f * 0.46, y - 0.05))
        b["ant1"].set_data([x + f * 0.46, x + f * 0.58], [y - 0.13, y - 0.32])
        b["ant2"].set_data([x + f * 0.5, x + f * 0.68], [y - 0.1, y - 0.22])
        b["sting"].set_xy([(x - f * 0.38, y - 0.05), (x - f * 0.52, y), (x - f * 0.38, y + 0.05)])

    def setup_energy_chart(self):
        ax = self.ax_energy
        ax.clear()
        for n, profile in self.profiles.items():           # faint lines: the other algorithms
            if n == self.name:
                continue
            ax.plot([s["energy"] for s in profile], color=COLORS[n], alpha=0.35,
                    linewidth=1, label=SHORT[n].replace("\n", " "))
        self.energy_line, = ax.plot([], [], color=COLORS[self.name], linewidth=2.8,
                                    label=SHORT[self.name].replace("\n", " ") + " (now)")
        ax.axhline(self.env.energy_limit, color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0, self.max_steps + 2)
        ax.set_ylim(0, self.env.energy_limit + 4)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Energy", fontsize=9)
        ax.set_title("Energy over time (faint lines = the other algorithms)", fontsize=11)
        ax.legend(fontsize=7, loc="upper right", ncol=3)
        ax.grid(True, alpha=0.3)

    def setup_history_chart(self):
        ax = self.ax_history
        ax.clear()
        history = list(self.result.get("history", []))
        ax.set_title(f"How the algorithm improves: {HISTORY_INFO[self.name]}", fontsize=10)
        ax.grid(True, alpha=0.3)

        if not history:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, f"{self.name} has no iterations to show",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            return

        if self.name == "Q-Learning":                      # 5000 noisy points -> smooth them
            window = 100
            smooth = np.convolve(history, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(history)), smooth, color=COLORS[self.name])
            ax.set_xlabel("Training trip", fontsize=9)
        else:
            ax.plot(range(1, len(history) + 1), history, color=COLORS[self.name], linewidth=2)
            ax.set_xlabel("Restart" if self.name == "Hill Climbing" else "Generation", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)

    # ------------------------------------------------------------------
    # drawing the current step
    # ------------------------------------------------------------------
    def render(self):
        env = self.env
        state = self.states[self.step]
        shown = self.states[: self.step + 1]

        # garden
        self.trail.set_data([s["pos"][0] for s in shown], [s["pos"][1] for s in shown])
        for flower in self.flowers:
            collected = flower["pos"] in state["collected"]
            for petal in flower["petals"]:
                petal.set_facecolor(WILTED if collected else FLOWER_PETALS[flower["value"]])
                petal.set_edgecolor(WILTED_EDGE if collected else FLOWER_EDGES[flower["value"]])
            flower["center"].set_facecolor(WILTED_CENTER if collected else FLOWER_CENTER)
        self.place_bee(state["pos"][0], state["pos"][1], self.facings[self.step], self.step % 2 == 0)

        # the ring: next flower on the list, or the hive when all are collected
        todo = [f for f in self.result.get("order", self.result["visited"]) if f not in state["collected"]]
        goal = todo[0] if todo else env.hive
        self.target_ring.set_center(goal)
        self.target_ring.set_visible(self.step < len(self.states) - 1)
        self.ax_garden.set_title(f"Bee Pollination Garden (seed {self.seed})  -  {self.name}",
                                 fontsize=12)

        # energy chart
        self.energy_line.set_data(range(len(shown)), [s["energy"] for s in shown])

        # energy bar
        self.bar_fill.set_width(max(state["energy"], 0) / env.energy_limit)

        # dashboard text
        total_steps = len(self.states) - 1
        planned = self.result.get("order", self.result["visited"])
        waiting = [f for f in planned if f not in state["collected"]]
        if waiting:
            nxt = f"{waiting[0]}  (nectar {env.flowers[waiting[0]]})"
        else:
            nxt = f"hive {env.hive}"

        if self.step >= total_steps:
            status = "Finished - back at the hive" if self.result["returned"] else "Finished - did not return"
        elif not self.playing:
            status = "Paused"
        elif state["just"]:
            status = "Collecting nectar"
        elif not waiting:
            status = "Returning to the hive"
        else:
            status = "Flying to the next flower"

        optimal = self.results.get(OPTIMAL)
        lines = [
            f"Algorithm   : {self.name}",
            f"State       : {status}",
            f"Step        : {self.step} / {total_steps}",
            f"Energy      : {state['energy']:.1f}/{env.energy_limit:.1f}",
            f"Energy used : {env.energy_limit - state['energy']:.1f}",
            f"Collected   : {len(state['collected'])}/{len(env.flowers)} flowers",
            f"Nectar      : {state['nectar']}  (final: {self.result['nectar']})",
            f"Remaining   : {len(env.flowers) - len(state['collected'])} flowers",
            f"Next stop   : {nxt}",
            f"Path length : {total_steps} cells",
            f"Run time    : {self.result['seconds']:.4f} s",
            "",
            f"Best of the 4: {self.best} ({self.results[self.best]['nectar']} nectar)",
        ]
        if optimal is not None:
            gap = optimal["nectar"] - self.result["nectar"]
            lines.append(f"Optimal      : {optimal['nectar']} nectar  (this algorithm: gap {gap})")
        self.info.set_text("\n".join(lines))

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # timer and buttons
    # ------------------------------------------------------------------
    def tick(self):
        if not self.playing:
            return
        if self.step < len(self.states) - 1:
            self.step += 1
        if self.step >= len(self.states) - 1:        # reached the end
            self.playing = False
            self.btn_play.label.set_text("Play")
        self.render()

    def on_choose(self, label):
        self.choose(label)

    def on_play(self, event):
        if self.step >= len(self.states) - 1:        # at the end: start again
            self.step = 0
            self.playing = True
        else:
            self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        self.render()

    def on_restart(self, event):
        self.choose(self.name)

    def on_new(self, event):
        self.load_garden(random.randint(0, 9999))

    def on_compare(self, event):
        fig = show_comparison(self.env, self.results, self.seed)
        fig.show()

    def on_speed(self, value):
        self.timer.interval = int(value)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    app = BeeApp(seed)
    plt.show()


if __name__ == "__main__":
    main()