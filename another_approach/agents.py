# agents.py
# Robust two-lane controller with pure-pursuit, signed-progress junction release,
# stall failsafe, crash locks, and assignment freeze while in junctions.
# FIXES:
# - Gate intersections ONLY when the segment target is a real road node (snap eps).
# - Always release any held node token when routes/assignments are cleared.
# - Respect NO_LANECHANGE_NEAR_NODE_M (prevents side-swapping right at nodes).
# - Consistent token cleanup at end-of-route.

from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import math

Point = Tuple[float, float]

def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _unit(v: Tuple[float, float]) -> Tuple[float, float]:
    x, y = v
    n = math.hypot(x, y)
    return (0.0, 0.0) if n < 1e-9 else (x / n, y / n)

def _perp(v: Tuple[float, float]) -> Tuple[float, float]:
    x, y = v
    return (-y, x)

@dataclass
class BinObj:
    id: str
    pos: Point
    capacity: int
    fill: int = 0
    curb: Optional[Point] = None

    def step_fill(self, lo: int, hi: int, rnd) -> bool:
        if self.fill < self.capacity:
            self.fill = min(self.capacity, self.fill + rnd.randint(lo, hi))
        return self.fill >= self.capacity

@dataclass
class Truck:
    tid: str
    pos: Point
    cfg: dict
    energy: float
    load: int = 0

    # motion state
    v: float = 0.0
    heading: float = 0.0
    desired_v: float = 0.0

    target: Optional[Point] = None
    assigned_bin: Optional[str] = None
    state: str = "idle"

    # two-lane
    lane_side: int = 1
    lane_change_cooldown: int = 0

    # crash + intersection
    crash_lock_steps: int = 0
    node_token: Optional[int] = None
    segment_frozen: bool = False
    frozen_node_idx: Optional[int] = None
    stall_steps: int = 0  # junction stall failsafe

    # bookkeeping
    km_total: float = 0.0
    kwh_total: float = 0.0
    costs_eur: Dict[str, float] = field(default_factory=lambda: {
        "wage": 0.0, "energy": 0.0, "maint": 0.0
    })
    route_pts: List[Point] = field(default_factory=list)
    route_i: int = 0

    # anti-churn / batching
    route_freeze_steps: int = 0
    assign_hold_steps: int = 0
    go_depot_lock_steps: int = 0
    stops_since_depot: int = 0

    # ---------- helpers ----------
    def _can_return_to_depot(self, depot: Point) -> bool:
        meters_left = self.energy / max(1e-9, self.cfg["ENERGY_PER_M"])
        return meters_left >= (dist(self.pos, depot) + self.cfg["ENERGY_RESERVE_M"])

    def _segment_vectors(self, prev_pt: Point, nxt_pt: Point):
        ux, uy = _unit((nxt_pt[0] - prev_pt[0], nxt_pt[1] - prev_pt[1]))
        nx, ny = _perp((ux, uy))
        L = max(1e-9, math.hypot(nxt_pt[0] - prev_pt[0], nxt_pt[1] - prev_pt[1]))
        return (ux, uy), (nx, ny), L

    def _pure_pursuit_point(self, prev_pt: Point, nxt_pt: Point, pos: Point) -> Point:
        (ux, uy), (_nx, _ny), L = self._segment_vectors(prev_pt, nxt_pt)
        rx, ry = (pos[0] - prev_pt[0], pos[1] - prev_pt[1])
        s = max(0.0, min(L, rx * ux + ry * uy))
        look = float(self.cfg.get("LOOKAHEAD_M", 3.5))
        s2 = min(L, s + look)
        return (prev_pt[0] + ux * s2, prev_pt[1] + uy * s2)

    def _lane_target_point(self, prev_pt: Point, nxt_pt: Point, pos: Point, inside_junction: bool) -> Point:
        base = self._pure_pursuit_point(prev_pt, nxt_pt, pos)
        if inside_junction and self.cfg.get("JUNCTION_SINGLE_LANE", True):
            return base
        (ux, uy), (nx, ny), L = self._segment_vectors(prev_pt, nxt_pt)
        blend_m = float(self.cfg.get("LANE_BLEND_M", 10.0))
        d_prev = dist(pos, prev_pt); d_next = dist(pos, nxt_pt)
        scale = min(d_prev, d_next) / max(1e-9, blend_m)
        scale = max(0.0, min(1.0, scale))
        off = float(self.cfg.get("LANE_OFFSET_M", 2.0)) * float(self.lane_side) * scale
        return (base[0] + nx * off, base[1] + ny * off)

    # ---- token utils ----
    def _release_token_if_any(self, city) -> None:
        """Always-safe token release helper."""
        if self.node_token is not None:
            city.intersections.release(self.node_token, self.tid)
            self.node_token = None
        self.segment_frozen = False
        self.frozen_node_idx = None
        self.stall_steps = 0

    # ---------- assignment / targets ----------
    def assign_target(self, route_pts, bin_id, final_target, force: bool = False):
        # NOTE: token release is handled by callers at decision points,
        # so we keep this pure and not city-dependent.
        if (self.cfg.get("ASSIGN_FREEZE_IN_JUNCTION", True)
                and self.segment_frozen and not force):
            return
        self.route_pts = [p for i, p in enumerate(route_pts) if i == 0 or dist(route_pts[i-1], p) > 1e-3]
        self.route_i = 1 if (self.route_pts and dist(self.pos, self.route_pts[0]) < 0.5) else 0
        self.assigned_bin = bin_id
        self.target = final_target
        self.state = "moving"
        self.route_freeze_steps = int(self.cfg.get("ROUTE_FREEZE_STEPS", 6))
        if bin_id is not None:
            self.assign_hold_steps = int(self.cfg.get("ASSIGN_HOLD_STEPS", 8))
        if final_target is not None and bin_id is None:
            self.go_depot_lock_steps = max(self.go_depot_lock_steps, int(self.cfg.get("DEPOT_LOCK_STEPS", 8)))

    # ---------- junction gating with signed progress + stall failsafe ----------
    def _signed_progress_past_node(self, prev_pt: Point, nxt_pt: Point) -> float:
        (ux, uy), _n, _L = self._segment_vectors(prev_pt, nxt_pt)
        dx, dy = (self.pos[0] - nxt_pt[0], self.pos[1] - nxt_pt[1])
        return dx * ux + dy * uy

    def _junction_state(self, city, prev_pt: Point, nxt_pt: Point):
        """
        Returns: (inside, node_idx_or_None, d_to_target, s_past, is_node_segment)
        We only consider junction gating if nxt_pt is a snapped road node.
        """
        snap_eps = float(self.cfg.get("NODE_SNAP_EPS", 1.0))
        node_idx = city.nearest_waypoint_idx(nxt_pt)
        node_pt = city.waypoints[node_idx]
        is_node_segment = (dist(nxt_pt, node_pt) <= snap_eps)

        d_to_target = dist(self.pos, nxt_pt)
        s_past = self._signed_progress_past_node(prev_pt, nxt_pt)

        approach_R = float(self.cfg.get("INTERSECTION_APPROACH_M", 7.0))
        clear_R    = float(self.cfg.get("INTERSECTION_CLEAR_M", 8.0))

        # "inside" only meaningful for node segments
        inside = False
        if is_node_segment:
            inside = (d_to_target <= clear_R) or self.segment_frozen or (self.node_token == node_idx)
            if d_to_target <= approach_R:
                inside = True

        return inside, (node_idx if is_node_segment else None), d_to_target, s_past, is_node_segment

    def _intersection_gate(self, city, prev_pt: Point, nxt_pt: Point) -> bool:
        inside, node_idx, d_to_target, s_past, is_node_segment = self._junction_state(city, prev_pt, nxt_pt)
        if not is_node_segment:
            # Not a node-bound segment (e.g., curb/goal mid-road) → never gate
            return True

        approach_R = float(self.cfg.get("INTERSECTION_APPROACH_M", 7.0))
        clear_R    = float(self.cfg.get("INTERSECTION_CLEAR_M", 8.0))

        # stall failsafe: if frozen without token too long, unfreeze
        if self.segment_frozen and self.node_token is None:
            self.stall_steps += 1
            if self.stall_steps > int(self.cfg.get("JUNCTION_STALL_STEPS", 30)):
                self.segment_frozen = False
                self.stall_steps = 0
        else:
            self.stall_steps = 0

        # entering approach zone: request token
        if d_to_target <= approach_R and self.node_token is None:
            if city.intersections.request(node_idx, self.tid):
                self.node_token = node_idx
                self.segment_frozen = True
                self.frozen_node_idx = node_idx
            else:
                return False  # wait

        # if holding, allow through; release when signed progress beyond node
        if self.node_token == node_idx:
            if s_past > clear_R:
                city.intersections.release(node_idx, self.tid)
                self.node_token = None
                self.segment_frozen = False
                self.frozen_node_idx = None
            return True

        return True

    # ---------- local traffic checks ----------
    def _front_blocked(self, trucks: List["Truck"], prev_pt: Point, nxt_pt: Point) -> bool:
        """
        Dynamic car-following with braking distance + emergency head-on check.
        Stops when:
        - Another truck is ahead within dynamic stopping distance, OR
        - A head-on (or near head-on) truck is inside an emergency radius.
        """
        # geometry along segment
        dir_vec = _unit((nxt_pt[0] - prev_pt[0], nxt_pt[1] - prev_pt[1]))
        nx, ny = _perp(dir_vec)

        # params
        base_gap   = float(self.cfg.get("SAFE_GAP_M", 3.0))          # buffer
        R         = float(self.cfg.get("TRUCK_RADIUS_M", 1.5))
        dt        = float(self.cfg.get("DT", 1.0))
        dec       = float(self.cfg.get("TRUCK_DEC_MPS2", 4.0))
        lane_w    = float(self.cfg.get("LANE_OFFSET_M", 2.0)) * 2.0  # approx lane-to-lane width
        headon_R  = max(2.5*R, 4.0)                                  # emergency radius

        # dynamic stopping distance for current speed
        v = max(0.0, self.v)
        brake_dist = (v * dt) + (v * v) / max(1e-6, 2.0 * dec)       # reaction + braking
        same_lane_thresh = lane_w * 0.7 + R                          # lateral closeness to treat as same lane

        for other in trucks:
            if other.tid == self.tid:
                continue

            # relative geometry
            dx = other.pos[0] - self.pos[0]
            dy = other.pos[1] - self.pos[1]
            d  = math.hypot(dx, dy)
            if d > max(20.0, brake_dist + base_gap + 2*R):
                continue

            # longitudinal vs lateral components
            fwd = dx * dir_vec[0] + dy * dir_vec[1]       # forward (+ ahead, - opposite)
            lat = abs(dx * nx + dy * ny)

            # (A) head-on/emergency: someone is coming towards us in (nearly) same line
            # approximate relative closing speed with headings if available, else use distance
            closing = 0.0
            try:
                closing = v + max(0.0, other.v)
            except Exception:
                closing = v
            near_headon = (fwd < 0.0) and (lat < same_lane_thresh)
            if near_headon and d < max(headon_R, 0.5 * closing + 2*R):
                return True

            # (B) car-following: someone ahead in our lane within stopping distance + buffer
            if fwd > 0.0 and lat < same_lane_thresh:
                dyn_safe = base_gap + 2*R + brake_dist
                if fwd < dyn_safe:
                    return True

        return False


    def _lane_change_possible(self, trucks: List["Truck"], prev_pt: Point, nxt_pt: Point, inside_junction: bool) -> bool:
        if inside_junction:
            return False
        # New: respect "no lane change near nodes"
        no_change_R = float(self.cfg.get("NO_LANECHANGE_NEAR_NODE_M", 20.0))
        if min(dist(self.pos, prev_pt), dist(self.pos, nxt_pt)) <= no_change_R:
            return False

        if self.lane_change_cooldown > 0:
            return False
        alt_side = -self.lane_side
        (ux, uy), (nx, ny), _L = self._segment_vectors(prev_pt, nxt_pt)
        off = float(self.cfg.get("LANE_OFFSET_M", 2.0)) * float(alt_side)
        probe = (self.pos[0] + nx * off, self.pos[1] + ny * off)
        safe = float(self.cfg.get("SAFE_GAP_M", 3.0)) + 2.0 * float(self.cfg.get("TRUCK_RADIUS_M", 1.5))
        for other in trucks:
            if other.tid == self.tid:
                continue
            if dist(probe, other.pos) < safe:
                return False
        return True

    def _traffic_controls(self, dt: float, trucks: List["Truck"], city) -> None:
        max_v = float(self.cfg["TRUCK_SPEED_MPS"])
        acc = float(self.cfg.get("TRUCK_ACC_MPS2", 2.5))
        dec = float(self.cfg.get("TRUCK_DEC_MPS2", 4.0))
        self.desired_v = max_v

        if not self.route_pts or self.route_i >= len(self.route_pts):
            self.desired_v = 0.0
            return

        prev_pt = self.pos if self.route_i == 0 else self.route_pts[self.route_i - 1]
        nxt_pt = self.route_pts[self.route_i]

        # crash lock → stop
        if self.crash_lock_steps > 0:
            self.desired_v = 0.0
        else:
            # intersection gating (only if segment ends on a node)
            if not self._intersection_gate(city, prev_pt, nxt_pt):
                self.desired_v = 0.0
            else:
                inside, _node_idx, _d_node, _s_past, _is_node_segment = self._junction_state(city, prev_pt, nxt_pt)
                # car-following
                if self._front_blocked(trucks, prev_pt, nxt_pt):
                    if self._lane_change_possible(trucks, prev_pt, nxt_pt, inside):
                        self.lane_side *= -1
                        self.lane_change_cooldown = int(self.cfg.get("LANE_CHANGE_COOLDOWN_STEPS", 12))
                    else:
                        self.desired_v = 0.0
                if self.lane_change_cooldown > 0:
                    self.lane_change_cooldown -= 1

        # smooth speed
        if self.v < self.desired_v:
            self.v = min(self.desired_v, self.v + acc * dt)
        else:
            self.v = max(self.desired_v, self.v - dec * dt)

    # ---------- movement ----------
    def _move_along_route(self, dt: float, trucks: List["Truck"], city, bins: List[BinObj]) -> float:
        if not self.route_pts or self.route_i >= len(self.route_pts):
            self.state = "idle"; return 0.0

        self._traffic_controls(dt, trucks, city)

        prev_pt = self.pos if self.route_i == 0 else self.route_pts[self.route_i - 1]
        nxt_pt = self.route_pts[self.route_i]
        inside, node_idx, d_to_node, s_past, is_node_segment = self._junction_state(city, prev_pt, nxt_pt)

        # docking: steer to actual bin near curb
        lane_tgt = self._lane_target_point(prev_pt, nxt_pt, self.pos, inside)
        dock_switch_m = float(self.cfg.get("DOCK_SWITCH_M", 8.0))
        if self.assigned_bin:
            b = next((bb for bb in bins if bb.id == self.assigned_bin), None)
            if b is not None and b.curb is not None:
                if dist(nxt_pt, b.curb) < 1.0 and dist(self.pos, b.curb) <= dock_switch_m:
                    lane_tgt = b.pos

        # integrate
        dx, dy = (lane_tgt[0] - self.pos[0], lane_tgt[1] - self.pos[1])
        d = math.hypot(dx, dy)
        step = min(d, max(0.0, self.v) * dt) if d > 1e-6 else 0.0
        if step > 0.0:
            nx = self.pos[0] + dx / d * step
            ny = self.pos[1] + dy / d * step
            self.pos = (nx, ny)
            self.heading = math.atan2(dy, dx)
            self.km_total += step / 1000.0
            e_used = step * self.cfg["ENERGY_PER_M"]
            self.energy -= e_used
            self.kwh_total += e_used
            self.costs_eur["energy"] += e_used * self.cfg["ENERGY_EUR_PER_UNIT"]
            self.costs_eur["maint"] += (step / 1000.0) * self.cfg["MAINT_EUR_PER_KM"]

        # segment advance (not while frozen)
        # segment advance — allow crossing a node while holding its token
        (ux, uy), (_nx, _ny), L = self._segment_vectors(prev_pt, nxt_pt)
        rx, ry = (self.pos[0] - prev_pt[0], self.pos[1] - prev_pt[1])
        s = max(0.0, min(L, rx * ux + ry * uy))

        # Are we on a node-bound segment and holding that node's token?
        _on_node_seg = False
        _node_idx_for_seg = None
        snap_eps = float(self.cfg.get("NODE_SNAP_EPS", 1.0))
        if city is not None:
            _node_idx_for_seg = city.nearest_waypoint_idx(nxt_pt)
            _on_node_seg = (dist(nxt_pt, city.waypoints[_node_idx_for_seg]) <= snap_eps)

        # We normally block advancement when frozen; but if we're at the end of a node segment
        # and we *hold that node's token*, advance and release the token immediately.
        advance_ok = (L - s) < 0.5 and (
            not self.segment_frozen
            or (_on_node_seg and self.node_token == _node_idx_for_seg)
        )

        if advance_ok:
            # Release any token tied to the node we're crossing
            if _on_node_seg and self.node_token == _node_idx_for_seg:
                city.intersections.release(_node_idx_for_seg, self.tid)
                self.node_token = None
                self.segment_frozen = False
                self.frozen_node_idx = None

            # Also release if we somehow still hold a token for the previous node
            prev_idx = city.nearest_waypoint_idx(prev_pt)
            if self.node_token == prev_idx:
                city.intersections.release(prev_idx, self.tid)
                self.node_token = None
                self.segment_frozen = False
                self.frozen_node_idx = None

            self.route_i += 1
            if self.route_i >= len(self.route_pts):
                self.state = "idle"
                self._release_token_if_any(city)
        else:
            self.state = "moving"


        return self.costs_eur["energy"] + self.costs_eur["maint"]

    # ---------- RL and baseline hooks ----------
    def apply_action(self, action: int, bins: List[BinObj], depot: Point, cfg: dict,
                     trucks: List["Truck"], city) -> float:
        dt = cfg["DT"]; reward = 0.0
        # tickdowns
        if self.route_freeze_steps > 0: self.route_freeze_steps -= 1
        if self.assign_hold_steps > 0:  self.assign_hold_steps  -= 1
        if self.go_depot_lock_steps > 0: self.go_depot_lock_steps -= 1
        if self.crash_lock_steps > 0:   self.crash_lock_steps   -= 1
        if self.crash_lock_steps > 0: self.v = 0.0; self.desired_v = 0.0

        wage_cost = (cfg["WAGE_PER_HOUR"] / 3600.0) * dt
        self.costs_eur["wage"] += wage_cost; reward -= wage_cost

        thr = cfg.get("APPROACH_RADIUS_M", 3.0)
        srv_thr = float(cfg.get("SERVICE_RADIUS_M",
            math.hypot(cfg.get("LANE_OFFSET_M",2.0), cfg.get("SIDEWALK_OFFSET_M",2.0)) + 1.0))

        # depot ops
        if dist(self.pos, depot) < 1.0:
            if self.load > 0:
                dumped = self.load; self.load = 0
                reward += 0.03 * dumped
                if self.stops_since_depot >= 2: reward += 0.3
                self.stops_since_depot = 0
            if action == 3 and self.energy < cfg["ENERGY_MAX"]:
                self.energy = cfg["ENERGY_MAX"]; reward += 0.1

        # service ops
        if self.assigned_bin:
            b = next((bb for bb in bins if bb.id == self.assigned_bin), None)
            if b and dist(self.pos, b.pos) < max(thr, srv_thr) and b.fill > 0 and self.load < cfg["TRUCK_CAPACITY"]:
                take = min(cfg["TRUCK_CAPACITY"] - self.load, b.fill)
                if take > 0:
                    self.load += take; b.fill -= take
                    reward += 0.02 * take; self.stops_since_depot += 1
            if b:
                if self.load >= cfg["TRUCK_CAPACITY"]:
                    # ensure token is not held when we repath to depot
                    self._release_token_if_any(city)
                    self.assigned_bin = None; self.assign_hold_steps = 0
                    self.target = depot
                    route = cfg.get("plan_route_fn")(self.pos, self.target) if "plan_route_fn" in cfg else [self.pos, self.target]
                    self.assign_target(route, None, self.target, force=True)
                elif b.fill == 0:
                    # bin emptied -> fully clear and release any token
                    self.assigned_bin = None; self.assign_hold_steps = 0
                    self.target = None; self.route_pts = []; self.route_i = 0
                    self._release_token_if_any(city)

        # autoplan depot
        if (self.load > 0) and (not self.route_pts) and (self.target is None):
            must_recharge = not self._can_return_to_depot(depot)
            near_full = self.load >= float(cfg.get("NEAR_FULL_FRAC", 0.9)) * cfg["TRUCK_CAPACITY"]
            if near_full or must_recharge or self.go_depot_lock_steps > 0:
                self._release_token_if_any(city)
                self.target = depot
                route = cfg.get("plan_route_fn")(self.pos, self.target) if "plan_route_fn" in cfg else [self.pos, self.target]
                self.assign_target(route, None, self.target, force=True)

        # actions
        if action == 1:  # return to depot
            self._release_token_if_any(city)
            self.assigned_bin = None; self.assign_hold_steps = 0
            self.target = depot
            route = cfg.get("plan_route_fn")(self.pos, self.target) if "plan_route_fn" in cfg else [self.pos, self.target]
            self.assign_target(route, None, self.target, force=True)

        if action == 4 and (self.route_pts or self.assigned_bin or self.load > 0 or self.route_freeze_steps > 0):
            action = 0

        if action != 4:
            if self.route_pts:
                _ = self._move_along_route(dt, trucks, city, bins); reward -= _
            elif self.target is not None:
                route = cfg.get("plan_route_fn")(self.pos, self.target) if "plan_route_fn" in cfg else [self.pos, self.target]
                self.assign_target(route, self.assigned_bin, self.target, force=False)
                _ = self._move_along_route(dt, trucks, city, bins); reward -= _
        else:
            self.state = "idle"; self.v = 0.0; self.desired_v = 0.0

        if self.energy <= 0 and dist(self.pos, depot) > 1.0:
            reward -= cfg.get("OUTAGE_PENALTY_EUR", 1000.0)

        if self.assigned_bin:
            b = next((bb for bb in bins if bb.id == self.assigned_bin), None)
            if b: reward += 0.001 * max(0.0, 1.0 - dist(self.pos, b.pos) / 200.0)
        return reward

    def step(self, dt: float, bins: List[BinObj], depot: Point, plan_route, trucks: List["Truck"], city) -> List[dict]:
        events = []
        if self.route_freeze_steps > 0: self.route_freeze_steps -= 1
        if self.assign_hold_steps > 0:  self.assign_hold_steps  -= 1
        if self.go_depot_lock_steps > 0: self.go_depot_lock_steps -= 1
        if self.crash_lock_steps > 0:   self.crash_lock_steps   -= 1
        if self.crash_lock_steps > 0: self.v = 0.0; self.desired_v = 0.0

        if dist(self.pos, depot) < 1.0:
            if self.load > 0:
                events.append({"type": "drop", "truck": self.tid, "amount": self.load})
                self.load = 0; self.stops_since_depot = 0
            if self.energy < self.cfg["ENERGY_MAX"]:
                self.energy = self.cfg["ENERGY_MAX"]; events.append({"type": "recharge", "truck": self.tid})

        thr = self.cfg.get("APPROACH_RADIUS_M", 3.0)
        srv_thr = float(self.cfg.get("SERVICE_RADIUS_M",
            math.hypot(self.cfg.get("LANE_OFFSET_M",2.0), self.cfg.get("SIDEWALK_OFFSET_M",2.0)) + 1.0))
        for b in bins:
            if self.assigned_bin == b.id and dist(self.pos, b.pos) < max(thr, srv_thr) and b.fill > 0:
                take = min(self.cfg["TRUCK_CAPACITY"] - self.load, b.fill)
                if take > 0:
                    self.load += take; b.fill -= take; self.stops_since_depot += 1
                    events.append({"type": "pickup", "truck": self.tid, "bin": b.id, "amount": take})
                if self.load >= self.cfg["TRUCK_CAPACITY"]:
                    self._release_token_if_any(city)
                    self.assigned_bin = None; self.assign_hold_steps = 0
                    route = plan_route(self.pos, depot); self.assign_target(route, None, depot, force=True)
                elif b.fill == 0:
                    self.assigned_bin = None; self.assign_hold_steps = 0
                    self.target = None; self.route_pts = []; self.route_i = 0
                    self._release_token_if_any(city)
                break

        if self.route_pts:
            self._move_along_route(dt, trucks, city, bins)
        elif self.target:
            route = plan_route(self.pos, self.target)
            self.assign_target(route, self.assigned_bin, self.target, force=False)
            self._move_along_route(dt, trucks, city, bins)

        return events
