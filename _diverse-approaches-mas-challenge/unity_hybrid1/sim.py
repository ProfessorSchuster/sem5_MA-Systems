# sim.py
from typing import List, Dict
import random, math

from agents import Truck, BinObj, dist
from dispatch import auction


class Simulation:
    def __init__(self, cfg, city, planner="graph", grid_passable=None):
        self.cfg = cfg
        self.city = city
        self.t = 0.0
        self.planner = planner
        self.grid_passable = grid_passable

        # Bins aus der City übernehmen
        self.bins: List[BinObj] = [
            BinObj(b["id"], b["pos"], b["capacity"], b["fill"], b.get("curb"))
            for b in city.bins
        ]

        # Trucks mit leichtem Spawn-Jitter ums Depot
        spawn_jitter = float(self.cfg.get("SPAWN_JITTER_M", 0.6))
        n_trucks = int(cfg["N_TRUCKS"])
        self.trucks: List[Truck] = []
        for i in range(n_trucks):
            ang = (i * 2.0 * math.pi) / max(1, n_trucks)
            px = city.depot[0] + spawn_jitter * math.cos(ang)
            py = city.depot[1] + spawn_jitter * math.sin(ang)
            t = Truck(tid=f"T{i}", pos=(px, py), cfg=cfg, energy=cfg["ENERGY_MAX"])
            self.trucks.append(t)

        self.frames: List[Dict] = []
        self.events: List[Dict] = []

        # NEW: kick-start one assignment so trucks have routes before first step
        try:
            assigns = auction(self.bins, self.trucks, self.t, self.cfg, self._plan_route)
            for ev in assigns:
                self.events.append({"t": self.t, **ev})
        except Exception:
            # safe-guard: never break construction
            pass

    def _rnd(self):
        return random.Random(int(self.t) ^ self.cfg["SEED"])

    def _fill_bins(self):
        lo, hi = self.cfg["BIN_FILL_PER_STEP"]
        rnd = self._rnd()
        overflows = 0
        for b in self.bins:
            before = b.fill
            if b.step_fill(lo, hi, rnd) and before < b.capacity:
                overflows += 1
                self.events.append({"t": self.t, "type": "overflow", "bin": b.id})
        return overflows

    # ---------- Two-lane helper ----------
    def _apply_lane_offset(self, route: List[tuple], goal: tuple) -> List[tuple]:
        """Shift the route off the road centerline to one lane.
        - RIGHT_HAND_TRAFFIC -> offset to the right of motion
        - LEFT_HAND_TRAFFIC  -> offset to the left of motion
        The final point (goal) is kept unoffset so trucks can pull to curb/depot.
        """
        lane_offset = float(self.cfg.get("LANE_OFFSET_M", 0.0))
        if lane_offset <= 1e-6 or len(route) < 2:
            return route  # nothing to do

        right_hand = bool(self.cfg.get("RIGHT_HAND_TRAFFIC", True))
        sign = +1.0 if right_hand else -1.0  # +1 = shift to the right of motion

        pts = []
        n = len(route)
        for i in range(n):
            x, y = route[i]

            # Keep the exact goal position unshifted so service / depot snapping works
            if i == n - 1:
                pts.append((goal[0], goal[1]))
                continue

            # Determine forward direction at this point
            x2, y2 = route[i + 1]
            dx, dy = (x2 - x), (y2 - y)
            L = math.hypot(dx, dy)
            if L <= 1e-6:
                # degenerate; do not offset
                pts.append((x, y))
                continue

            # right-hand normal = (dy, -dx) / |v|
            nx, ny = (dy / L, -dx / L)
            offx = x + sign * lane_offset * nx
            offy = y + sign * lane_offset * ny
            pts.append((offx, offy))

        return pts

    def _plan_route(self, start, goal):
        """Return a lane-aware route from start to goal."""
        # GRID planner stays as-is (grid cell centers already separate flows)
        if self.planner == "grid" and self.grid_passable is not None:
            try:
                from grid_planner import astar, manhattan_path
            except Exception:
                from .grid_planner import astar, manhattan_path
            s = (int(round(start[0])), int(round(start[1])))
            g = (int(round(goal[0])), int(round(goal[1])))
            path = astar(s, g, self.grid_passable) or manhattan_path(s, g)
            return [(float(x), float(y)) for (x, y) in path]

        # Graph planner -> path along road nodes
        base = self.city.plan_route(start, goal)

        # Ensure goal is explicitly present at the end (many callers rely on exact equality)
        if not base or base[-1] != goal:
            base = base + [goal]

        # Apply two-lane offset along the path, but keep the final point (goal) at the center (curb/depot)
        lane = self._apply_lane_offset(base, goal)

        # Small de-dup (keep continuity)
        dedup = []
        for p in lane:
            if not dedup or (abs(dedup[-1][0] - p[0]) > 1e-6 or abs(dedup[-1][1] - p[1]) > 1e-6):
                dedup.append(p)
        return dedup

    def _safety_pass(self):
        near_r  = float(self.cfg.get("NEAR_MISS_RADIUS_M", 2.0))
        crash_r = float(self.cfg.get("CRASH_RADIUS_M", 1.2))
        near_pen  = float(self.cfg.get("NEAR_MISS_PENALTY", 3.0))
        crash_pen = float(self.cfg.get("CRASH_PENALTY", 300.0))

        T = self.trucks
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                ti, tj = T[i], T[j]
                d = dist(ti.pos, tj.pos)
                if d <= crash_r:
                    self.events.append({"t": self.t, "type": "crash_penalty", "truck": ti.tid, "amount": crash_pen})
                    self.events.append({"t": self.t, "type": "crash_penalty", "truck": tj.tid, "amount": crash_pen})
                elif d <= near_r:
                    self.events.append({"t": self.t, "type": "near_miss_penalty", "truck": ti.tid, "amount": near_pen})
                    self.events.append({"t": self.t, "type": "near_miss_penalty", "truck": tj.tid, "amount": near_pen})

    def step(self):
        dt = self.cfg["DT"]

        # 1) Bins füllen (inkl. Overflow-Events)
        self._fill_bins()

        # 2) Zuweisungen via Auktion (Hybrid-Logik steckt in dispatch.auction)
        assigns = auction(self.bins, self.trucks, self.t, self.cfg, self._plan_route)
        for ev in assigns:
            self.events.append({"t": self.t, **ev})

        # 3) Trucks einen Tick bewegen/bedienen
        step_events = []
        for t in self.trucks:
            for ev in t.step(dt, self.bins, self.city.depot, self._plan_route):
                ev["t"] = self.t
                if ev.get("type") == "pickup":
                    bid = ev.get("bin")
                    b = next((bb for bb in self.bins if bb.id == bid), None)
                    if b is not None:
                        b.last_service_t = self.t
                step_events.append(ev)

        # 3b) Safety-Shaping (optional fürs Log/Visuals)
        self._safety_pass()
        new_evs = [e for e in self.events if e.get("t") == self.t and e.get("type") in ("near_miss_penalty", "crash_penalty")]
        step_events.extend(new_evs)

        # 4) Frame loggen
        frame = {
            "t": self.t,
            "trucks": [
                {
                    "id": t.tid,
                    "x": t.pos[0],
                    "y": t.pos[1],
                    "energy": t.energy,
                    "load": t.load,
                    "state": t.state,
                    "target": (None if t.target is None else {"x": t.target[0], "y": t.target[1]}),
                } for t in self.trucks
            ],
            "bins": [
                {"id": b.id, "x": b.pos[0], "y": b.pos[1], "fill": b.fill, "cap": b.capacity}
                for b in self.bins
            ],
        }
        self.frames.append(frame)

        # 5) Zeit fortschreiben
        self.t += dt

    def run(self, steps: int):
        for _ in range(steps):
            self.step()
