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

        # Trucks: spawn ON roads, at least 20m from depot, spaced apart
        n_trucks = int(cfg["N_TRUCKS"])
        self.trucks: List[Truck] = []
        spawn_pts = self._pick_spawn_points(
            n=n_trucks,
            min_depot_dist=float(self.cfg.get("SPAWN_MIN_FROM_DEPOT_M", 20.0)),
            min_between=float(self.cfg.get("SPAWN_MIN_SEPARATION_M", 12.0)),
        )
        for i, (px, py) in enumerate(spawn_pts):
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
            pass

    def _rnd(self):
        return random.Random(int(self.t) ^ self.cfg["SEED"])

    def _fill_bins(self):
        lo, hi = self.cfg["BIN_FILL_PER_STEP"]
        rnd = self._rnd()

        p = float(self.cfg.get("BIN_FILL_PROB", 1.0))
        mult = float(self.cfg.get("BIN_FILL_MULT", 1.0))
        lo_eff = max(0, int(round(lo * mult)))
        hi_eff = max(lo_eff, int(round(hi * mult)))

        overflows = 0
        for b in self.bins:
            if rnd.random() > p:
                continue
            before = b.fill
            if b.step_fill(lo_eff, hi_eff, rnd) and before < b.capacity:
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
            return route

        right_hand = bool(self.cfg.get("RIGHT_HAND_TRAFFIC", True))
        sign = +1.0 if right_hand else -1.0

        pts = []
        n = len(route)
        for i in range(n):
            x, y = route[i]

            if i == n - 1:
                pts.append((goal[0], goal[1]))
                continue

            x2, y2 = route[i + 1]
            dx, dy = (x2 - x), (y2 - y)
            L = math.hypot(dx, dy)
            if L <= 1e-6:
                pts.append((x, y))
                continue

            nx, ny = (dy / L, -dx / L)
            offx = x + sign * lane_offset * nx
            offy = y + sign * lane_offset * ny
            pts.append((offx, offy))

        return pts

    def _pick_spawn_points(self, n: int, min_depot_dist: float, min_between: float):
        """Pick n positions along road centerlines, each ≥ min_depot_dist from depot
        and ≥ min_between from each other. Deterministic per-seed."""
        candidates = []
        depot = self.city.depot

        for r in self.city.roads:
            (x1, y1), (x2, y2) = r.polyline
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 1e-6:
                continue
            samples = max(1, min(6, int(L // 25)))
            for i in range(samples):
                t = 0.2 + 0.6 * ((i + 0.5) / samples)
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                if dist((px, py), depot) >= min_depot_dist:
                    candidates.append((px, py))

        rnd = self._rnd()
        rnd.shuffle(candidates)

        picks = []
        for p in candidates:
            if all(dist(p, q) >= min_between for q in picks):
                picks.append(p)
                if len(picks) >= n:
                    break

        if len(picks) < n and candidates:
            remaining = [p for p in candidates if p not in picks]
            remaining.sort(key=lambda p: dist(p, depot), reverse=True)
            for p in remaining:
                picks.append(p)
                if len(picks) >= n:
                    break

        while len(picks) < n:
            k = len(picks)
            ang = (k * 2.0 * math.pi) / max(1, n)
            px = depot[0] + (min_depot_dist + 1.0) * math.cos(ang)
            py = depot[1] + (min_depot_dist + 1.0) * math.sin(ang)
            picks.append((px, py))

        return picks

    def _plan_route(self, start, goal):
        """Return a lane-aware route from start to goal."""
        if self.planner == "grid" and self.grid_passable is not None:
            try:
                from grid_planner import astar, manhattan_path
            except Exception:
                from .grid_planner import astar, manhattan_path
            s = (int(round(start[0])), int(round(start[1])))
            g = (int(round(goal[0])), int(round(goal[1])))
            path = astar(s, g, self.grid_passable) or manhattan_path(s, g)
            return [(float(x), float(y)) for (x, y) in path]

        base = self.city.plan_route(start, goal)

        if not base or base[-1] != goal:
            base = base + [goal]

        lane = self._apply_lane_offset(base, goal)

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

    # ---------- NEW: intersection arbiter ----------
    def _intersection_arbitration(self):
        """Allow only one truck to enter the same upcoming node per tick.
        Deterministic ROW (lower tid wins). Others yield for 1 tick."""
        approach_r = float(self.cfg.get("SAFE_STOP_M", 4.0)) + float(self.cfg.get("LANE_LOOKAHEAD_M", 3.0))
        buckets = {}
        for t in self.trucks:
            if not t.route_pts or t.route_i >= len(t.route_pts):
                continue
            nxt = t.route_pts[t.route_i]
            if dist(t.pos, nxt) <= approach_r:
                buckets.setdefault(nxt, []).append((t, nxt))

        for nxt, contenders in buckets.items():
            if len(contenders) <= 1:
                continue
            contenders.sort(key=lambda it: it[0].tid)  # lower id wins
            winner, *_losers = contenders
            for (loser, _n) in contenders[1:]:
                loser.yield_steps = max(getattr(loser, "yield_steps", 0), 1)

    def step(self):
        dt = self.cfg["DT"]

        # 1) Fill bins (with overflow events)
        self._fill_bins()

        # 2) Auction assignments (hybrid)
        assigns = auction(self.bins, self.trucks, self.t, self.cfg, self._plan_route)
        for ev in assigns:
            self.events.append({"t": self.t, **ev})

        # 2b) Depot gate (prevent pileups inside depot radius)
        gate_r = float(self.cfg.get("DEPOT_GATE_RADIUS_M", 6.0))
        gate_n = int(self.cfg.get("DEPOT_MAX_INSIDE", 1))
        inside = []
        outside = []
        for i, t in enumerate(self.trucks):
            (inside if dist(t.pos, self.city.depot) <= gate_r else outside).append((i, t))
        inside.sort(key=lambda it: it[1].tid)
        for rank, (idx, loser) in enumerate(inside):
            if rank >= gate_n:
                self.trucks[idx].yield_steps = max(self.trucks[idx].yield_steps, 1)

        # 2c) Simple car-following: if too close, loser yields 1 tick
        safe_stop = float(self.cfg.get("SAFE_STOP_M", 3.5))
        T = self.trucks
        for i in range(len(T)):
            for j in range(i+1, len(T)):
                ti, tj = T[i], T[j]
                if dist(ti.pos, tj.pos) < safe_stop:
                    loser = tj if ti.tid < tj.tid else ti
                    loser.yield_steps = max(loser.yield_steps, 1)

        # 2d) NEW: per-node arbitration just before movement
        self._intersection_arbitration()

        # 3) Move trucks
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

        # 3b) Safety log (as before)
        self._safety_pass()
        new_evs = [e for e in self.events if e.get("t") == self.t and e.get("type") in ("near_miss_penalty", "crash_penalty")]
        step_events.extend(new_evs)

        # 4) Frame
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

        # 5) Time
        self.t += dt

    def run(self, steps: int):
        for _ in range(steps):
            self.step()
