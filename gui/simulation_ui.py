"""
Interactive simulation for the AI Bee Pollination Agent (put this file in gui/).

Uses your real project code:
    environment.garden.BeeGarden      -> the world (hive, obstacles, flowers, energy_limit, collect_cost)
    search.astar.a_star_search        -> path planning, returns (path, cost)

Run from anywhere:
    python gui/simulation_ui.py

Panels: garden + path trail | energy over time | dashboard.
Controls: Pause/Play, Restart (same garden), New garden, speed slider.
"""
import math
import os
import random
import sys

# Make "environment" and "search" importable even when running `python gui/simulation_ui.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, Slider

from environment.garden import BeeGarden
from search.astar import a_star_search

GARDEN_KWARGS = dict(width=15, height=15, n_flowers=10)
START_SEED = 0


# ----------------------------------------------------------------------------
# Bee agent: greedy "nearest flower I can still afford, then get home"
# ----------------------------------------------------------------------------
class BeeAgent:
    def __init__(self, garden):
        self.g = garden
        self.max_energy = float(garden.energy_limit)
        self.energy = self.max_energy
        self.collect_cost = float(getattr(garden, "collect_cost", 0))
        self.pos = garden.hive
        self.remaining = set(garden.flowers)
        self.collected = []
        self.plan = []
        self.step_cost = 1.0
        self.target = None
        self.state = "Planning"
        self.done = False
        self.steps = 0
        self.trail = [self.pos]
        self.energy_hist = [self.energy]

    def _route(self, a, b):
        """(path without start, cost) or (None, None) if there is no usable path."""
        if a == b:
            return [], 0.0
        try:
            path, cost = a_star_search(self.g, a, b)
        except Exception:
            return None, None
        if not path or cost is None or not math.isfinite(cost):
            return None, None
        path = list(path)
        if path and path[0] == a:
            path = path[1:]
        return path, float(cost)

    def _choose(self):
        best = None
        for f in self.remaining:
            to_f, c1 = self._route(self.pos, f)
            if to_f is None:
                continue
            back, c2 = self._route(f, self.g.hive)
            if back is None:
                continue
            if c1 + self.collect_cost + c2 <= self.energy + 1e-9:
                if best is None or c1 < best[0]:
                    best = (c1, f, to_f)
        if best:
            c1, f, path = best
            self.target, self.plan = f, path
            self.step_cost = c1 / max(1, len(path))
            self.state = "To flower"
            return
        if self.pos != self.g.hive:
            path, cost = self._route(self.pos, self.g.hive)
            if path is None:
                self.done, self.state = True, "Stuck"
                return
            self.target, self.plan = None, path
            self.step_cost = cost / max(1, len(path))
            self.state = "Returning"
            return
        self.done, self.state = True, "Finished"

    def _arrive(self):
        if self.target is not None and self.pos == self.target:
            self.remaining.discard(self.pos)
            self.collected.append(self.pos)
            self.energy -= self.collect_cost
            self.target = None
            self.state = "Collecting"
        elif self.pos == self.g.hive:
            self.state = "At hive"

    def step(self):
        if self.done:
            return
        if not self.plan:
            self._choose()
            if self.done:
                self.energy_hist.append(self.energy)
                return
            if not self.plan:  # already standing on the target
                self._arrive()
                self.energy_hist.append(self.energy)
                return
        self.pos = self.plan.pop(0)
        self.energy -= self.step_cost
        self.steps += 1
        self.trail.append(self.pos)
        if not self.plan:
            self._arrive()
        self.energy_hist.append(self.energy)


# ----------------------------------------------------------------------------
# Interactive UI
# ----------------------------------------------------------------------------
class Simulation:
    def __init__(self, seed=START_SEED):
        self.seed = seed
        self.paused = False

        self.fig = plt.figure(figsize=(12, 7))
        try:
            self.fig.canvas.manager.set_window_title("Bee Pollination Simulation")
        except Exception:
            pass
        gs = GridSpec(2, 2, figure=self.fig, width_ratios=[1.3, 1], height_ratios=[1, 1],
                      left=0.05, right=0.97, top=0.93, bottom=0.18, wspace=0.18, hspace=0.35)
        self.ax_g = self.fig.add_subplot(gs[:, 0])
        self.ax_e = self.fig.add_subplot(gs[0, 1])
        self.ax_i = self.fig.add_subplot(gs[1, 1])

        self.btn_pause = Button(self.fig.add_axes([0.06, 0.04, 0.11, 0.06]), "Pause")
        self.btn_restart = Button(self.fig.add_axes([0.19, 0.04, 0.11, 0.06]), "Restart")
        self.btn_new = Button(self.fig.add_axes([0.32, 0.04, 0.14, 0.06]), "New garden")
        self.slider = Slider(self.fig.add_axes([0.62, 0.055, 0.25, 0.03]),
                             "Delay (ms)", 30, 800, valinit=250, valstep=10)
        self.btn_pause.on_clicked(self.toggle_pause)
        self.btn_restart.on_clicked(lambda _e: self.restart(new_garden=False))
        self.btn_new.on_clicked(lambda _e: self.restart(new_garden=True))
        self.slider.on_changed(self.set_speed)

        self.build(new_garden=False)
        self.anim = FuncAnimation(self.fig, self.update, interval=250, cache_frame_data=False)

    # -- setup ---------------------------------------------------------------
    def build(self, new_garden):
        if new_garden:
            self.seed = random.randrange(10_000)
        self.garden = BeeGarden(seed=self.seed, **GARDEN_KWARGS)
        self.flower_positions = list(self.garden.flowers)
        self.bee = BeeAgent(self.garden)
        self._draw_static()
        self._draw_energy_axes()
        self._draw_dashboard_axes()
        self.redraw()

    def _draw_static(self):
        ax, g = self.ax_g, self.garden
        ax.clear()
        for r in range(g.height):
            for c in range(g.width):
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#f7f9f2",
                                       edgecolor="#cbd5c0", linewidth=1))
        for obs in g.obstacles:
            r, c = obs
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor="#9ca3af",
                                   edgecolor="#4b5563", linewidth=1))
        hr, hc = g.hive
        ax.scatter(hc + 0.5, hr + 0.5, s=320, marker="s", color="#eab308",
                   edgecolors="#713f12", linewidths=2, zorder=4)

        fx = np.array([[c + 0.5, r + 0.5] for r, c in self.flower_positions]).reshape(-1, 2)
        self.flower_sc = ax.scatter(fx[:, 0], fx[:, 1], s=220, color="#f59e0b",
                                    edgecolors="#b45309", linewidths=1.5, zorder=3)
        (self.trail_line,) = ax.plot([], [], "-", color="#ef4444", alpha=0.6,
                                     linewidth=2, zorder=2)
        self.bee_sc = ax.scatter([], [], s=260, marker="o", color="#ef4444",
                                 edgecolors="#7f1d1d", linewidths=1.5, zorder=5)
        ax.set_xlim(0, g.width)
        ax.set_ylim(g.height, 0)  # row 0 at the top
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Bee Pollination Garden (seed {self.seed})")

    def _draw_energy_axes(self):
        ax = self.ax_e
        ax.clear()
        ax.set_title("Energy over time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy")
        ax.set_ylim(0, self.bee.max_energy * 1.05)
        ax.axhline(self.bee.max_energy, color="#9ca3af", linestyle="--", linewidth=1)
        ax.grid(alpha=0.3)
        (self.energy_line,) = ax.plot([], [], color="#16a34a", linewidth=2)

    def _draw_dashboard_axes(self):
        ax = self.ax_i
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Dashboard")
        ax.add_patch(Rectangle((0.05, 0.82), 0.9, 0.1, facecolor="#e5e7eb", edgecolor="#6b7280"))
        self.bar_fg = Rectangle((0.05, 0.82), 0.9, 0.1, facecolor="#16a34a")
        ax.add_patch(self.bar_fg)
        self.info_text = ax.text(0.05, 0.75, "", va="top", family="monospace", fontsize=11)

    # -- drawing -------------------------------------------------------------
    def redraw(self):
        g, b = self.garden, self.bee

        self.flower_sc.set_facecolors(
            ["#86efac" if p in b.collected else "#f59e0b" for p in self.flower_positions])
        self.trail_line.set_data([c + 0.5 for _r, c in b.trail], [r + 0.5 for r, _c in b.trail])
        self.bee_sc.set_offsets([[b.pos[1] + 0.5, b.pos[0] + 0.5]])

        self.energy_line.set_data(range(len(b.energy_hist)), b.energy_hist)
        self.ax_e.set_xlim(0, max(50, len(b.energy_hist) + 5))

        energy = max(0.0, b.energy)
        ratio = energy / b.max_energy if b.max_energy else 0.0
        self.bar_fg.set_width(0.9 * ratio)
        self.bar_fg.set_facecolor("#16a34a" if ratio > 0.5 else "#f59e0b" if ratio > 0.25 else "#dc2626")
        self.info_text.set_text(
            f"State      : {b.state}\n"
            f"Step       : {b.steps}\n"
            f"Energy     : {energy:.1f}/{b.max_energy:.1f}\n"
            f"Energy used: {b.max_energy - energy:.1f}\n"
            f"Collected  : {len(b.collected)}/{len(self.flower_positions)} flowers\n"
            f"Remaining  : {len(b.remaining)} flowers\n"
            f"Path length: {len(b.trail) - 1} cells"
        )
        self.fig.canvas.draw_idle()

    # -- callbacks -----------------------------------------------------------
    def update(self, _frame):
        if not self.paused and not self.bee.done:
            self.bee.step()
            self.redraw()

    def toggle_pause(self, _event):
        self.paused = not self.paused
        self.btn_pause.label.set_text("Play" if self.paused else "Pause")
        self.fig.canvas.draw_idle()

    def restart(self, new_garden):
        self.paused = False
        self.btn_pause.label.set_text("Pause")
        self.build(new_garden)

    def set_speed(self, value):
        self.anim.event_source.interval = int(value)


if __name__ == "__main__":
    sim = Simulation()
    plt.show()