"""
gui/simulation_ui.py  -  Animated dashboard of the bee (matplotlib window).

Run from the project root (the folder that has main.py):
    python -m gui.simulation_ui          (garden seed 0)
    python -m gui.simulation_ui 5        (garden seed 5)

What you see:
    left   : the garden and the bee moving (green flower = collected)
    right  : energy over time, how the algorithm improves, and a dashboard
    bottom : choose the algorithm, Pause, Restart, New garden, Compare all, speed

No pygame needed, only matplotlib.
"""

import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from environment.garden import BeeGarden
from gui.plots import (COLORS, OPTIMAL, SHORT, SIZE, N_FLOWERS, STRATEGIES,
                       best_strategy, draw_garden_base, energy_profile,
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
        ax.clear()
        draw_garden_base(ax, self.env)
        self.route_line, = ax.plot([], [], color="#e88080", linewidth=2, zorder=2)

        self.flower_list = list(self.env.flowers)
        fx = [p[0] for p in self.flower_list]
        fy = [p[1] for p in self.flower_list]
        self.flower_dots = ax.scatter(fx, fy, s=300, zorder=3, linewidths=1.5,
                                      facecolors="#f59e0b", edgecolors="#a05a00")
        for (x, y), value in self.env.flowers.items():
            ax.text(x, y, str(value), ha="center", va="center", fontsize=7, zorder=4)

        start = self.states[0]["pos"]
        self.bee, = ax.plot([start[0]], [start[1]], "o", color="#e53935", markersize=14,
                            markeredgecolor="#7f0000", markeredgewidth=1.5, zorder=5)

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
        xs = [s["pos"][0] for s in shown]
        ys = [s["pos"][1] for s in shown]
        self.route_line.set_data(xs, ys)
        self.bee.set_data([state["pos"][0]], [state["pos"][1]])
        colors = ["#7ddc8f" if f in state["collected"] else "#f59e0b" for f in self.flower_list]
        self.flower_dots.set_facecolor(colors)
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