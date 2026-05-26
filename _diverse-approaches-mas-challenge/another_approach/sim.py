# sim.py
from typing import List, Dict
import random, json, math
from agents import Truck, BinObj, dist
from negotiation import auction

class Simulation:
    def __init__(self, cfg, city):
        self.cfg = cfg
        self.city = city
        self.t = 0.0

        self.bins: List[BinObj] = [
            BinObj(b["id"], b["pos"], b["capacity"], b["fill"], b.get("curb")) for b in city.bins
        ]

        self.trucks: List[Truck] = []
        lane_off = float(cfg.get("LANE_OFFSET_M", 2.0))
        safe_gap = float(cfg.get("SAFE_GAP_M", 3.0)) + 2.0 * float(cfg.get("TRUCK_RADIUS_M", 1.5))
        roads = city.roads[:]
        if not roads:
            raise RuntimeError("No roads to spawn trucks on.")
        for i in range(cfg["N_TRUCKS"]):
            r = roads[i % len(roads)]
            (x1,y1),(x2,y2) = r.polyline
            tpos = [0.2, 0.5, 0.8][(i // len(roads)) % 3]
            cx = x1 + tpos*(x2-x1); cy = y1 + tpos*(y2-y1)
            L = math.hypot(x2-x1,y2-y1); ux,uy = (x2-x1)/max(1e-9,L), (y2-y1)/max(1e-9,L)
            nx,ny = -uy, ux
            side = 1 if (i % 2 == 0) else -1
            px = cx + nx*lane_off*side; py = cy + ny*lane_off*side
            t = Truck(tid=f"T{i}", pos=(px, py), cfg=cfg, energy=cfg["ENERGY_MAX"])
            t.lane_side = side
            self.trucks.append(t)

        rnd = random.Random(cfg.get("SEED", 42))
        for k in range(len(self.trucks)):
            for j in range(k):
                if dist(self.trucks[k].pos, self.trucks[j].pos) < safe_gap:
                    dx = (rnd.random()-0.5)*safe_gap; dy = (rnd.random()-0.5)*safe_gap
                    pk = self.trucks[k].pos; self.trucks[k].pos = (pk[0]+dx, pk[1]+dy)

        self.frames: List[Dict] = []
        self.events: List[Dict] = []
        self.day_costs = {"wage_eur":0.0,"energy_eur":0.0,"maintenance_eur":0.0,"penalties_eur":0.0}

    def _rnd(self):
        return random.Random(int(self.t) ^ self.cfg["SEED"])

    def _fill_bins(self):
        lo, hi = self.cfg["BIN_FILL_PER_STEP"]
        rnd = self._rnd(); overflows = 0
        for b in self.bins:
            before = b.fill
            if b.step_fill(lo, hi, rnd) and before < b.capacity:
                overflows += 1; self.events.append({"t": self.t, "type": "overflow", "bin": b.id})
        return overflows

    def _wage_tick(self):
        dt_hours = self.cfg["DT"] / 3600.0
        self.day_costs["wage_eur"] += len(self.trucks) * self.cfg["WAGE_PER_HOUR"] * dt_hours

    def _detect_crashes(self):
        R = float(self.cfg.get("TRUCK_RADIUS_M", 1.5))
        crash_R = 2.0 * R * 0.98
        n = len(self.trucks)
        pen_each = float(self.cfg.get("CRASH_PENALTY_EUR", 5000.0))
        lock = int(self.cfg.get("CRASH_LOCK_STEPS", 25))
        for i in range(n):
            for j in range(i+1, n):
                a = self.trucks[i]; b = self.trucks[j]
                if dist(a.pos, b.pos) < crash_R:
                    self.events.append({"t": self.t, "type": "crash", "a": a.tid, "b": b.tid,
                                        "x": (a.pos[0]+b.pos[0])/2.0, "y": (a.pos[1]+b.pos[1])/2.0})
                    self.day_costs["penalties_eur"] += pen_each
                    for t in (a, b):
                        # clear routing + release any held token on crash
                        t.assigned_bin = None; t.assign_hold_steps = 0
                        t.target = None; t.route_pts = []; t.route_i = 0
                        node_idx = self.city.nearest_waypoint_idx(t.pos)
                        self.city.intersections.release(node_idx, t.tid)
                        if t.node_token is not None:
                            self.city.intersections.release(t.node_token, t.tid)
                        t.node_token = None
                        t.segment_frozen = False
                        t.frozen_node_idx = None
                        t.crash_lock_steps = max(t.crash_lock_steps, lock)

    def step(self):
        dt = self.cfg["DT"]
        new_ov = self._fill_bins()
        if new_ov > 0:
            self.day_costs["penalties_eur"] += new_ov * self.cfg["OVERFLOW_PENALTY_EUR"]

        auction(self.bins, self.trucks, self.t, self.cfg, self.city.plan_route)

        step_events = []
        for t in self.trucks:
            evs = t.step(dt, self.bins, self.city.depot, self.city.plan_route, self.trucks, self.city)
            step_events.extend(evs)
        self.events.extend(step_events)

        self._detect_crashes()
        self._wage_tick()
        self.day_costs["energy_eur"] = sum(t.costs_eur["energy"] for t in self.trucks)
        self.day_costs["maintenance_eur"] = sum(t.costs_eur["maint"] for t in self.trucks)

        frame = {
            "t": self.t,
            "trucks": [
                {"id": t.tid, "x": t.pos[0], "y": t.pos[1], "energy": t.energy, "load": t.load,
                 "state": t.state, "target": (None if t.target is None else {"x": t.target[0], "y": t.target[1]})}
                for t in self.trucks
            ],
            "bins": [{"id": b.id, "x": b.pos[0], "y": b.pos[1], "fill": b.fill, "cap": b.capacity} for b in self.bins],
            "events": step_events + [e for e in self.events if e.get("t")==self.t and e.get("type")=="crash"],
        }
        self.frames.append(frame)
        self.t += dt

    def run(self, steps:int):
        for _ in range(steps):
            self.step()

    def summary_costs(self)->Dict:
        total = sum(self.day_costs.values())
        return {**self.day_costs, "total_eur": total}

    def export_json(self, path: str):
        def json_safe(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    if callable(v): continue
                    try: json.dumps(v); out[k]=v
                    except TypeError: out[k]=str(v)
                return out
            return obj
        out = {"frames": self.frames, "events": self.events, "costs": self.summary_costs(), "cfg": json_safe(self.cfg)}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"✅ Exported simulation JSON to {path}")
