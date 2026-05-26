#!/usr/bin/env python3
# viz_rich.py — rich matplotlib visualization for the waste-collection sim
from __future__ import annotations
from typing import List
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

def color_fill(frac: float):
    f = _clamp01(frac)
    if f <= 1/3:   return (0.18, 0.80, 0.44)
    if f <= 2/3:   return (0.95, 0.77, 0.06)
    return (0.91, 0.30, 0.24)

def color_energy(frac: float):
    f = _clamp01(frac)
    return (1.0 - f, f, 0.0)

@dataclass
class EventBuffer:
    max_items: int = 5
    window_s: float = 10.0
    items: List[dict] = field(default_factory=list)

    def add(self, ev: dict):
        self.items.append(ev)
        if len(self.items) > 50 * self.max_items:
            self.items = self.items[-50 * self.max_items:]

    def recent(self, now: float) -> List[dict]:
        out = [e for e in self.items if (now - float(e.get("t", 0))) <= self.window_s]
        return out[-self.max_items:]

class RichViz:
    def __init__(self, sim, show_ids: bool = True, draw_routes: bool = True):
        self.sim = sim
        self.city = sim.city
        self.cfg = sim.cfg
        self.show_ids = show_ids
        self.draw_routes = draw_routes

        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(0, self.city.w)
        self.ax.set_ylim(0, self.city.h)
        self.ax.grid(True, alpha=0.2)

        self.paused = False
        self.step_once = False
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw_static()

        self.bin_patches: List[Circle] = []
        self.bin_labels: List = []          # "b3" etc. (optional)
        self.bin_fill_labels: List = []     # "64%" live fullness text
        self.cooldown_rings: List[Circle] = []
        self.truck_labels: List = []
        self.route_collections: List[LineCollection] = []
        self.target_segments: List[LineCollection] = []
        self.alert_rings: List[Circle] = []
        self._alert_cd = {}
        self._near_cd = {}

        self._init_bins()
        self.trucks_scat = None
        self._init_trucks()

        self.hud_title = self.ax.text(0.01, 0.99, "", transform=self.ax.transAxes,
                                      ha="left", va="top", fontsize=11, color="#111")
        self.hud_kpi   = self.ax.text(0.01, 0.95, "", transform=self.ax.transAxes,
                                      ha="left", va="top", fontsize=9, color="#333")
        self.hud_ev = self.ax.text(0.99, 0.99, "", transform=self.ax.transAxes,
                                   ha="right", va="top", fontsize=9, color="#222",
                                   bbox=dict(boxstyle="round,pad=0.3", fc="#f8f8f8", ec="#ddd"))

        self.evbuf = EventBuffer(max_items=6, window_s=8.0)
        self.fig.tight_layout()

        self._id_to_bin = {b.id: b for b in self.sim.bins}

    def _on_key(self, ev):
        if ev.key == "q":
            plt.close(self.fig)
        elif ev.key == "p":
            self.paused = not self.paused
        elif ev.key == "n":
            self.step_once = True
        elif ev.key == "r":
            self.draw_routes = not self.draw_routes

    def _draw_static(self):
        segs = []
        for r in self.city.roads:
            (x1, y1), (x2, y2) = r.polyline
            segs.append([(x1, y1), (x2, y2)])
        lc = LineCollection(segs, linewidths=2.0, alpha=0.28, colors=[(0.75, 0.75, 0.75, 1.0)])
        self.ax.add_collection(lc)
        dx, dy = self.city.depot
        dep = Rectangle((dx - 2.5, dy - 2.5), 5.0, 5.0,
                        facecolor=(0.2, 0.7, 0.2, 0.9), edgecolor='k', lw=1.0, alpha=0.9, zorder=2)
        self.ax.add_patch(dep)
        self.ax.text(dx + 4.5, dy + 4.5, "Depot", fontsize=9, color='black', zorder=3)

    def _init_bins(self):
        for b in self.sim.bins:
            frac = (b.fill / max(1, b.capacity))
            c = Circle((b.pos[0], b.pos[1]), radius=2.0,
                       facecolor=color_fill(frac), edgecolor='k', lw=0.6, alpha=0.95, zorder=2)
            self.ax.add_patch(c)
            self.bin_patches.append(c)

            # Live fullness label (centered on the circle)
            perc = int(round(100.0 * b.fill / max(1, b.capacity)))
            lbl_full = self.ax.text(
                b.pos[0], b.pos[1],
                f"{perc}%",
                ha="center", va="center",
                fontsize=7, color="#111", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc=(1,1,1,0.6), ec=(0,0,0,0.15), lw=0.5)
            )
            self.bin_fill_labels.append(lbl_full)

            # Cooldown ring
            ring = Circle((b.pos[0], b.pos[1]), radius=2.6,
                          facecolor=(0,0,0,0), edgecolor=(0.2,0.6,1.0,0.0), lw=1.2, zorder=1)
            self.ax.add_patch(ring)
            self.cooldown_rings.append(ring)

            # Optional ID label slightly offset
            if self.show_ids:
                txt = self.ax.text(b.pos[0] + 2.2, b.pos[1] + 2.2, f"{b.id}",
                                   fontsize=8, color="#333", alpha=0.85)
                self.bin_labels.append(txt)

    def _init_trucks(self):
        xs = [t.pos[0] for t in self.sim.trucks]
        ys = [t.pos[1] for t in self.sim.trucks]
        cols = [color_energy(t.energy / max(1.0, self.cfg["ENERGY_MAX"])) for t in self.sim.trucks]
        self.trucks_scat = self.ax.scatter(xs, ys, s=85, marker='^', c=cols,
                                           edgecolors='k', linewidths=0.6, zorder=5)
        if self.show_ids:
            for t in self.sim.trucks:
                txt = self.ax.text(t.pos[0] + 2.0, t.pos[1] + 2.0, f"{t.tid}",
                                   fontsize=9, color="#111", zorder=6)
                self.truck_labels.append(txt)
        for t  in self.sim.trucks:
            ring = Circle((t.pos[0], t.pos[1]), radius=3.2, facecolor=(0,0,0,0), edgecolor=(1,0,0,0), lw=2.0, zorder=6)
            self.ax.add_patch(ring)
            self.alert_rings.append(ring)
            self._alert_cd[t.tid] = 0
            self._near_cd[t.tid] = 0
        for _ in self.sim.trucks:
            rc = LineCollection([], colors=[(0.0,0.0,0.0,0.28)], linewidths=1.6, zorder=3)
            self.ax.add_collection(rc)
            self.route_collections.append(rc)
            tc = LineCollection([], colors=[(0.6,0.6,0.6,0.65)], linewidths=1.0, linestyles='dashed', zorder=3)
            self.ax.add_collection(tc)
            self.target_segments.append(tc)

    def update_from_state(self):
        tnow = float(self.sim.t)
        self._update_bins(tnow)
        self._update_trucks()
        self._update_routes()
        self._update_hud(tnow)

    def update_from_frame(self, frame_i: int):
        fr = self.sim.frames[frame_i]
        tnow = float(fr["t"])
        for patch, b, lbl_full in zip(self.bin_patches, fr["bins"], self.bin_fill_labels):
            frac = (b["fill"] / max(1, b["cap"]))
            patch.set_facecolor(color_fill(frac))
            perc = int(round(100.0 * b["fill"] / max(1, b["cap"])))
            lbl_full.set_text(f"{perc}%")
            # (Bins are static; no need to move label position)

        tx = [t["x"] for t in fr["trucks"]]
        ty = [t["y"] for t in fr["trucks"]]
        cols = []
        for t in fr["trucks"]:
            efrac = t["energy"] / max(1.0, self.cfg["ENERGY_MAX"])
            cols.append(color_energy(efrac))
        self.trucks_scat.set_offsets(list(zip(tx, ty)))
        self.trucks_scat.set_color(cols)
        if self.show_ids:
            for lbl, t in zip(self.truck_labels, fr["trucks"]):
                lbl.set_position((t["x"] + 2.0, t["y"] + 2.0))
                lbl.set_text(f'{t["id"]}')
        for tc, t in zip(self.target_segments, fr["trucks"]):
            segs = []
            if t["target"] is not None and self.draw_routes:
                a = (t["x"], t["y"])
                b = (t["target"]["x"], t["target"]["y"])
                segs.append([a, b])
            tc.set_segments(segs)
        for rc in self.route_collections:
            rc.set_segments([])

        self._harvest_new_events_until(tnow)
        self._update_hud(tnow)
        self._update_cooldown_rings(tnow)

    def _update_bins(self, now: float):
        for patch, b, lbl_full in zip(self.bin_patches, self.sim.bins, self.bin_fill_labels):
            frac = (b.fill / max(1, b.capacity))
            patch.set_facecolor(color_fill(frac))
            perc = int(round(100.0 * b.fill / max(1, b.capacity)))
            lbl_full.set_text(f"{perc}%")
        self._update_cooldown_rings(now)

    def _update_cooldown_rings(self, now: float):
        cool_s = float(self.cfg.get("SERVICE_COOLDOWN_S", 300.0))
        show_s = max(3.0, cool_s / 6.0)
        for ring, b in zip(self.cooldown_rings, self.sim.bins):
            last = float(getattr(b, "last_service_t", -1e9))
            dt = now - last
            alpha = 0.0
            if dt >= 0.0 and dt <= show_s:
                alpha = max(0.15, 1.0 - dt / show_s) * 0.8
            ring.set_edgecolor((0.2, 0.6, 1.0, alpha))

    def _update_trucks(self):
        xs = [t.pos[0] for t in self.sim.trucks]
        ys = [t.pos[1] for t in self.sim.trucks]
        cols = [color_energy(t.energy / max(1.0, self.cfg["ENERGY_MAX"])) for t in self.sim.trucks]
        self.trucks_scat.set_offsets(list(zip(xs, ys)))
        self.trucks_scat.set_color(cols)
        if self.show_ids:
            for lbl, t in zip(self.truck_labels, self.sim.trucks):
                lbl.set_position((t.pos[0] + 2.0, t.pos[1] + 2.0))
                lbl.set_text(f"{t.tid}")
        for ring, t in zip(self.alert_rings, self.sim.trucks):
            ring.center = (t.pos[0], t.pos[1])
            if self._alert_cd.get(t.tid, 0) > 0:
                self._alert_cd[t.tid] -= 1
                ring.set_edgecolor((1.0, 0.0, 0.0, 0.9))
                ring.set_linewidth(2.6)
            elif self._near_cd.get(t.tid, 0) > 0:
                self._near_cd[t.tid] -= 1
                ring.set_edgecolor((1.0, 0.5, 0.0, 0.7))
                ring.set_linewidth(2.0)
            else:
                ring.set_edgecolor((1.0, 0.0, 0.0, 0.0))

    def _update_routes(self):
        for rc, tc, t in zip(self.route_collections, self.target_segments, self.sim.trucks):
            tseg = []
            if t.target is not None and self.draw_routes:
                tseg.append([(t.pos[0], t.pos[1]), (t.target[0], t.target[1])])
            tc.set_segments(tseg)

            segs = []
            pts = t.route_pts or []
            if self.draw_routes and len(pts) >= 1:
                idx = max(0, min(int(getattr(t, "route_i", 0) or 0), len(pts) - 1))
                if idx < len(pts):
                    nx, ny = pts[idx]
                    segs.append([(t.pos[0], t.pos[1]), (nx, ny)])
                    for i in range(idx, len(pts) - 1):
                        ax, ay = pts[i]
                        bx, by = pts[i + 1]
                        segs.append([(ax, ay), (bx, by)])
            rc.set_segments(segs)

    def _harvest_new_events_until(self, now: float):
        for e in self.sim.events:
            if "_seen" in e:
                continue
            if float(e.get("t", 0)) <= now:
                self.evbuf.add(e)
                e["_seen"] = True
                if e.get("type") == "crash":
                    tid = str(e.get("truck"))
                    self._alert_cd[tid] = max(self._alert_cd.get(tid, 0), 18)
                elif e.get("type") == "near_miss":
                    tid = str(e.get("truck"))
                    self._near_cd[tid] = max(self._near_cd.get(tid, 0), 14)

    def _format_ticker(self, now: float) -> str:
        lines = []
        for e in self.evbuf.recent(now):
            et = e.get("type")
            if et == "assign":
                lines.append(f'ASSIGN  T={e.get("truck")} -> {e.get("bin")}')
            elif et == "pickup":
                amt = int(e.get("amount", 0))
                lines.append(f'PICKUP  T={e.get("truck")} @ {e.get("bin")} (+{amt})')
            elif et == "drop":
                lines.append(f'DUMP    T={e.get("truck")}')
            elif et == "recharge":
                lines.append(f'RECHARGE T={e.get("truck")}')
            elif et == "overflow":
                lines.append(f'OVERFLOW bin={e.get("bin")}')
            elif et == "crash":
                lines.append(f'CRASH   T={e.get("truck")} x {e.get("with")}')
            elif et == "near_miss":
                lines.append(f'NEAR MISS T={e.get("truck")} ~ {e.get("with")}')
        return "\n".join(lines) if lines else "—"

    def _update_hud(self, now: float):
        self.hud_title.set_text(f"t = {int(now)} s")
        total_collected = sum(int(e.get("amount", 0)) for e in self.sim.events if e.get("type") == "pickup")
        km = sum(float(getattr(t, "km_total", 0.0)) for t in self.sim.trucks)
        evs = len(self.sim.events)
        kpi = [f"Collected: {total_collected}  |  Fleet km: {km:.1f}  |  Events: {evs}"]
        per = []
        for t in self.sim.trucks:
            efrac = t.energy / max(1.0, self.cfg["ENERGY_MAX"])
            per.append(f'{t.tid}: load {t.load}/{self.cfg.get("TRUCK_CAPACITY", 0)}  E {int(efrac*100)}%  {t.state}')
        kpi.append("\n".join(per))
        self.hud_kpi.set_text("\n".join(kpi))
        self._harvest_new_events_until(now)
        self.hud_ev.set_text(self._format_ticker(now))

def live(sim, steps: int, show_ids: bool = True, interval_ms: int = 40):
    viz = RichViz(sim, show_ids=show_ids, draw_routes=True)

    def _tick(_i):
        if viz.paused and not viz.step_once:
            return []
        viz.step_once = False
        sim.step()
        viz.update_from_state()
        return []

    viz._ani = FuncAnimation(
        viz.fig,
        _tick,
        frames=range(int(steps)),
        interval=max(1, int(interval_ms)),
        blit=False,
        repeat=False,
    )
    plt.show()

def playback(sim, show_ids: bool = True, interval_ms: int = 60):
    if not sim.frames:
        raise RuntimeError("No frames to play back. Run sim.run(...) first.")
    viz = RichViz(sim, show_ids=show_ids, draw_routes=True)
    viz._ani = FuncAnimation(
        viz.fig,
        lambda i: (viz.update_from_frame(i), [])[1],
        frames=len(sim.frames),
        interval=max(1, int(interval_ms)),
        blit=False,
        repeat=False,
    )
    plt.show()

if __name__ == "__main__":
    import argparse, os
    from config import CONFIG
    from city import City
    from sim import Simulation

    parser = argparse.ArgumentParser(description="Visualize waste collection simulation")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--planner", choices=["graph", "grid"], default="graph")
    parser.add_argument("--mode", choices=["live", "playback"], default="live")
    parser.add_argument("--ids", action="store_true")
    parser.add_argument("--interval", type=int, default=40)
    parser.add_argument("--dqn-weights", type=str, default=None)

    args = parser.parse_args()
    cfg = CONFIG.copy()

    if args.dqn_weights:
        import torch
        from dqn_env import TruckEnv
        from dqn_agent import DQNAgent

        if not os.path.isfile(args.dqn_weights):
            raise SystemExit(f"Weight file not found: {args.dqn_weights}")

        env = TruckEnv(cfg)
        agent = DQNAgent(env.obs_dim, env.action_space.n, cfg)
        sd = torch.load(args.dqn_weights, map_location="cpu")
        agent.q_net.load_state_dict(sd)
        agent.target_net.load_state_dict(sd)
        agent.eps = 0.0

        # reset so assignments/obs exist
        obs_all = env.reset()

        if args.mode == "live":
            viz = RichViz(env.sim, show_ids=args.ids, draw_routes=True)

            state = {"obs_all": obs_all}

            def _tick(_i):
                if viz.paused and not viz.step_once:
                    return []
                viz.step_once = False
                acts = [agent.act_eval(state["obs_all"][i]) for i in range(env.n_agents)]
                state["obs_all"], _r, _done, _ = env.step(acts)
                viz.update_from_state()
                return []

            viz._ani = FuncAnimation(
                viz.fig,
                _tick,
                frames=range(int(args.steps)),
                interval=max(1, int(args.interval)),
                blit=False,
                repeat=False,
            )
            plt.show()
        else:
            while env.current_step < args.steps:
                acts = [agent.act_eval(obs_all[i]) for i in range(env.n_agents)]
                obs_all, _r, done, _ = env.step(acts)
                if all(done):
                    break
            playback(env.sim, show_ids=args.ids, interval_ms=args.interval)
    else:
        city = City(cfg)
        sim = Simulation(cfg=cfg, city=city, planner=args.planner)

        if args.mode == "live":
            live(sim, steps=args.steps, show_ids=args.ids, interval_ms=args.interval)
        else:
            sim.run(args.steps)
            playback(sim, show_ids=args.ids, interval_ms=args.interval)
