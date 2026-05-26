# visualize.py
# Matplotlib preview with crash markers.

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import math

def _color_for_fill(frac: float) -> str:
    if frac <= 0.33: return "#2ecc71"
    if frac <= 0.66: return "#f1c40f"
    return "#e74c3c"

def preview(sim):
    if not sim.frames:
        raise RuntimeError("No frames to preview. Did you call sim.run(...) before preview()?")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, sim.city.w)
    ax.set_ylim(0, sim.city.h)
    ax.set_aspect("equal")
    ax.set_title("Trash Collection — City Preview")

    lane_off = float(sim.cfg.get("LANE_OFFSET_M", 2.0))
    for r in sim.city.roads:
        (x1,y1),(x2,y2) = r.polyline
        # centerline
        ax.plot([x1, x2],[y1, y2],"-",linewidth=1.5,color="#bfc5c9",zorder=0)
        # lanes (offset by ±lane_off)
        L = max(1e-9, math.hypot(x2-x1, y2-y1))
        ux, uy = (x2-x1)/L, (y2-y1)/L
        nx, ny = -uy, ux
        # left lane
        ax.plot([x1 + nx*lane_off, x2 + nx*lane_off],
                [y1 + ny*lane_off, y2 + ny*lane_off],
                "-", linewidth=1.5, color="#8aa0b1", alpha=0.9, zorder=0)
        # right lane
        ax.plot([x1 - nx*lane_off, x2 - nx*lane_off],
                [y1 - ny*lane_off, y2 - ny*lane_off],
                "-", linewidth=1.5, color="#8aa0b1", alpha=0.9, zorder=0)


    ax.scatter([sim.city.depot[0]], [sim.city.depot[1]], c="blue", marker="D", s=80, label="Depot", zorder=3)

    trucks_scatter = ax.scatter([], [], c="#6c5ce7", marker="^", s=90, label="Trucks", zorder=4)
    bin_scatter    = ax.scatter([], [], c=[], marker="s", s=60, label="Bins", zorder=2)
    crash_scatter  = ax.scatter([], [], c="red", marker="x", s=120, label="Crash", zorder=5)
    target_lines   = []

    bin_g = mlines.Line2D([], [], color="#2ecc71", marker="s", linestyle="None", markersize=8, label="Bin (low)")
    bin_y = mlines.Line2D([], [], color="#f1c40f", marker="s", linestyle="None", markersize=8, label="Bin (mid)")
    bin_r = mlines.Line2D([], [], color="#e74c3c", marker="s", linestyle="None", markersize=8, label="Bin (high)")
    truck_patch = mlines.Line2D([], [], color="#6c5ce7", marker="^", linestyle="None", markersize=8, label="Truck")
    crash_patch = mlines.Line2D([], [], color="red", marker="x", linestyle="None", markersize=8, label="Crash")
    ax.legend(handles=[bin_g, bin_y, bin_r, truck_patch, crash_patch], loc="upper right")

    bin_texts = []; truck_texts = []

    # collect all crash events once (static markers over time)
    crash_xy = [(e["x"], e["y"]) for e in sim.events if e.get("type") == "crash"]

    def update(i):
        nonlocal target_lines, bin_texts, truck_texts
        frame = sim.frames[i]
        ax.set_title(f"t = {frame['t']:.0f}s")

        for t in bin_texts + truck_texts:
            t.remove()
        bin_texts.clear(); truck_texts.clear()
        for ln in target_lines:
            ln.remove()
        target_lines.clear()

        bx = [b["x"] for b in frame["bins"]]
        by = [b["y"] for b in frame["bins"]]
        colors = [_color_for_fill(b["fill"]/max(1,b["cap"])) for b in frame["bins"]]
        bin_scatter.set_offsets(list(zip(bx, by)))
        bin_scatter.set_color(colors)
        for b in frame["bins"]:
            bin_texts.append(ax.text(b["x"], b["y"] + 2.0, f'{b["fill"]}/{b["cap"]}', ha="center", va="bottom", fontsize=7, color="#333"))

        tx = [t["x"] for t in frame["trucks"]]
        ty = [t["y"] for t in frame["trucks"]]
        trucks_scatter.set_offsets(list(zip(tx, ty)))
        for t in frame["trucks"]:
            truck_texts.append(
                ax.text(t["x"], t["y"] + 2.2, f'load {t["load"]} | E {int(t["energy"])}', ha="center", va="bottom", fontsize=7, color="#222")
            )
            tgt = t["target"]
            if tgt is not None:
                ln = ax.plot([t["x"], tgt["x"]], [t["y"], tgt["y"]], "--", color="#95a5a6", linewidth=1, alpha=0.6)[0]
                target_lines.append(ln)

        if crash_xy:
            crash_scatter.set_offsets(crash_xy)

        return trucks_scatter, bin_scatter, crash_scatter

    ani = FuncAnimation(fig, update, frames=len(sim.frames), interval=100, blit=False, repeat=False)
    plt.show()
