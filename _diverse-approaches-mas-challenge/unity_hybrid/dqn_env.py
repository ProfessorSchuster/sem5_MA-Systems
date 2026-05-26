# dqn_env.py — RL wrapper Environment with ROW, pull-aside, anti-idle, sane near-miss
import math
import numpy as np
try:
    from gymnasium import spaces
except Exception:
    class _Disc:
        def __init__(self, n): self.n = n
    class _Box:
        def __init__(self, low, high, shape, dtype): self.shape = shape
    class _spaces:
        Discrete=_Disc; Box=_Box
    spaces = _spaces()

from sim import Simulation
from city import City
from dispatch import auction
from agents import dist

class TruckEnv:
    """
    Actions (Discrete(5)):
      0: MOVE/CONTINUE
      1: RETURN_TO_DEPOT (sets route to depot, then MOVE)
      2: (unused -> MOVE)
      3: RECHARGE (only at depot; else MOVE)
      4: WAIT
    """
    def __init__(self, cfg):
        base_cfg = cfg.copy()
        self.city = City(base_cfg)
        self.cfg = base_cfg
        self.sim = Simulation(self.cfg, self.city)
        self.n_agents = int(self.cfg["N_TRUCKS"])
        self.max_steps = int(self.cfg.get("STEPS_PER_DAY", 1200))
        self.current_step = 0

        # obs = [x/w, y/h, load%, energy%, assigned_d, assigned_fill, (3 bins * (d,f)), nearest_norm, headway_norm, truck_id_norm]
        self.obs_dim = (4 + 2 + 3 * 2) + 2 + 1
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.reward_scale = float(self.cfg.get("REWARD_SCALE", 0.01))
        self.max_penalties_per_tick = self.cfg.get("MAX_PENALTIES_PER_TICK", None)

        # per-truck state
        self._yield_cd = {t.tid: 0 for t in self.sim.trucks}
        self._idle_ticks = {t.tid: 0 for t in self.sim.trucks}
        self._stuck_ticks = {t.tid: 0 for t in self.sim.trucks}

        # car-following / smoothing
        self._warmup_ticks = int(self.cfg.get("WARMUP_TICKS", 2))
        self._depot_unblock_r = float(self.cfg.get("DEPOT_UNBLOCK_RADIUS_M", 3.0))
        self._min_speed_scale = float(self.cfg.get("MIN_SPEED_SCALE", 0.35))
        self._speed_lp = {t.tid: 1.0 for t in self.sim.trucks}
        self._stopped = {t.tid: False for t in self.sim.trucks}
        self._release_hysteresis_m = 0.6
        self._speed_smooth_alpha = 0.6

        # near-miss bookkeeping
        self._pair_last_d = {}       # (tid_lo, tid_hi) -> last distance
        self._near_cd_pairs = {}     # cooldown for near-miss events

        # pull-aside cooldown
        self._pull_cd = {t.tid: 0 for t in self.sim.trucks}

    # ---------- logging helper ----------
    def _log_frame(self):
        frame = {
            "t": self.sim.t,
            "trucks": [
                {
                    "id": t.tid,
                    "x": t.pos[0],
                    "y": t.pos[1],
                    "energy": t.energy,
                    "load": t.load,
                    "state": t.state,
                    "target": (None if t.target is None else {"x": t.target[0], "y": t.target[1]}),
                } for t in self.sim.trucks
            ],
            "bins": [
                {"id": b.id, "x": b.pos[0], "y": b.pos[1], "fill": b.fill, "cap": b.capacity}
                for b in self.sim.bins
            ],
            "events": [],
        }
        self.sim.frames.append(frame)

    def reset(self):
        self.city = City(self.cfg)
        self.sim = Simulation(self.cfg, self.city)
        self.current_step = 0
        self._yield_cd = {t.tid: 0 for t in self.sim.trucks}
        self._idle_ticks = {t.tid: 0 for t in self.sim.trucks}
        self._stuck_ticks = {t.tid: 0 for t in self.sim.trucks}
        self._speed_lp = {t.tid: 1.0 for t in self.sim.trucks}
        self._stopped = {t.tid: False for t in self.sim.trucks}
        self._pair_last_d = {}
        self._near_cd_pairs = {}
        self._pull_cd = {t.tid: 0 for t in self.sim.trucks}

        auction(self.sim.bins, self.sim.trucks, self.sim.t, self.cfg, self.sim._plan_route)
        self._bootstrap_plans_if_needed()
        self._log_frame()
        return self._get_obs_all()

    # ---------- movement bootstrap ----------
    def _best_bin_for(self, truck):
        nonempty = [b for b in self.sim.bins if b.fill > 0]
        if not nonempty:
            return None
        return min(nonempty, key=lambda b: (-(b.fill / max(1, b.capacity)), dist(truck.pos, b.curb or b.pos)))

    def _bootstrap_plans_if_needed(self):
        for t in self.sim.trucks:
            has_route = bool(t.route_pts) or (t.target is not None)
            if t.assigned_bin or has_route:
                continue
            b = self._best_bin_for(t)
            if b is None: continue
            curb = b.curb or b.pos
            route = self.sim._plan_route(t.pos, curb)
            if not route or route[-1] != curb: route = route + [curb]
            t.assign_target(route, b.id, curb)

    # ---------- helpers ----------
    # ---------- proactive TTC yield (pre-move) ----------
    def _predict_step_pos(self, t):
        """Predict where 't' would move this tick based on current speed_scale."""
        dt = float(self.cfg["DT"])
        v = float(self.cfg.get("TRUCK_SPEED_MPS", 2.0)) * float(max(0.0, t.speed_scale))
        if v <= 1e-6:
            return t.pos
        if t.route_pts and t.route_i < len(t.route_pts):
            tx, ty = t.route_pts[t.route_i]
        elif t.target is not None:
            tx, ty = t.target
        else:
            return t.pos
        dx, dy = (tx - t.pos[0]), (ty - t.pos[1])
        L = math.hypot(dx, dy)
        if L <= 1e-6:
            return t.pos
        step = min(L, v * dt)
        return (t.pos[0] + dx / L * step, t.pos[1] + dy / L * step)

    def _proactive_ttc_yield(self):
        """Before moving, if two trucks are on a collision course soon, force the loser to WAIT this tick."""
        if not bool(self.cfg.get("PROACTIVE_YIELD", True)):
            return set()

        crash_r = float(self.cfg.get("CRASH_RADIUS_M", 1.2))
        buf = float(self.cfg.get("CRASH_BUFFER_M", 0.25))
        thr = crash_r + buf
        ttc_horizon = float(self.cfg.get("TTC_YIELD_S", 1.2))

        losers = set()
        T = self.sim.trucks
        dt = float(self.cfg["DT"])
        speed_mps = float(self.cfg.get("TRUCK_SPEED_MPS", 2.0))

        for i in range(len(T)):
            for j in range(i+1, len(T)):
                a, b = T[i], T[j]

                # current relative geometry
                ax, ay = a.pos; bx, by = b.pos
                dx0, dy0 = (bx - ax), (by - ay)
                d0 = math.hypot(dx0, dy0)
                if d0 > max(6.0, thr*3.0):   # too far to worry this tick
                    continue

                # predicted next positions (using current speed_scale)
                apx, apy = self._predict_step_pos(a)
                bpx, bpy = self._predict_step_pos(b)
                d1 = math.hypot((bpx - apx), (bpy - apy))

                # quick early stop: if after one tick distance already below threshold
                if d1 < thr:
                    # choose ROW winner deterministically (same rule as elsewhere)
                    win_tid = self._safety_row_winner(a.tid, b.tid)
                    loser = b if win_tid == a.tid else a
                    losers.add(loser.tid)
                    continue

                # coarse TTC check along line of centers using scalar closing rate
                # approximate velocity norms
                va = speed_mps * float(max(0.0, a.speed_scale))
                vb = speed_mps * float(max(0.0, b.speed_scale))
                if va <= 1e-6 and vb <= 1e-6:
                    continue

                # unit line-of-centers
                if d0 <= 1e-6:
                    continue
                ux, uy = (dx0 / d0, dy0 / d0)

                # forward direction unit vectors
                def _ahead_unit(t):
                    if t.route_pts and t.route_i < len(t.route_pts):
                        tx, ty = t.route_pts[t.route_i]
                    elif t.target is not None:
                        tx, ty = t.target
                    else:
                        return (0.0, 0.0)
                    dx, dy = (tx - t.pos[0]), (ty - t.pos[1])
                    L = math.hypot(dx, dy)
                    if L <= 1e-6: return (0.0, 0.0)
                    return (dx / L, dy / L)

                axu, ayu = _ahead_unit(a)
                bxu, byu = _ahead_unit(b)
                if (axu, ayu) == (0.0, 0.0) and (bxu, byu) == (0.0, 0.0):
                    continue

                # relative closing speed along line of centers
                vrel = (va * (axu*ux + ayu*uy)) - (vb * (bxu*ux + byu*uy))
                # If vrel < 0, they are approaching (b towards a); if > 0, separating.
                closing = (-vrel) > 1e-6
                if not closing:
                    continue

                ttc = (d0 - thr) / max(1e-6, (-vrel))  # seconds until we reach threshold
                if 0.0 <= ttc <= ttc_horizon:
                    win_tid = self._safety_row_winner(a.tid, b.tid)
                    loser = b if win_tid == a.tid else a
                    losers.add(loser.tid)

        return losers
    
    def _safety_row_winner(self, tid_a: str, tid_b: str) -> str:
        rule = str(self.cfg.get("ROW_RULE", "lower_id_wins"))
        if rule == "lower_id_wins":
            return tid_a if tid_a < tid_b else tid_b
        return tid_a

    def _ahead_vec(self, t):
        if t.route_pts and t.route_i < len(t.route_pts):
            tx, ty = t.route_pts[t.route_i]
        elif t.target is not None:
            tx, ty = t.target
        else:
            return (0.0, 0.0)
        dx, dy = (tx - t.pos[0]), (ty - t.pos[1])
        L = math.hypot(dx, dy)
        if L <= 1e-6: return (0.0, 0.0)
        return (dx / L, dy / L)

    def _step_cost_eur(self):
        sim = self.sim
        wage = sum(t.costs_eur.get("wage", 0.0) for t in sim.trucks)
        energy = sum(t.costs_eur.get("energy", 0.0) for t in sim.trucks)
        maint = sum(t.costs_eur.get("maint", 0.0) for t in sim.trucks)
        t_now = sim.t
        overflow_fee = sum(float(self.cfg.get("OVERFLOW_PENALTY_EUR", 0.0))
                           for e in sim.events if e.get("type") == "overflow" and e.get("t") == t_now)
        crash_fee = sum(float(self.cfg.get("CRASH_PENALTY", 0.0))
                        for e in sim.events if e.get("type") == "crash" and e.get("t") == t_now)
        near_fee = sum(float(self.cfg.get("NEAR_MISS_PENALTY", 0.0))
                       for e in sim.events if e.get("type") == "near_miss" and e.get("t") == t_now)
        return wage + energy + maint + overflow_fee + crash_fee + near_fee

    # ---------- near-miss & collision (sane) ----------
    def _apply_collisions(self, rewards):
        if self.current_step < self._warmup_ticks:
            self._update_pair_bookkeeping(); return

        crash_r = float(self.cfg.get("CRASH_RADIUS_M", 1.2))
        near_r  = float(self.cfg.get("NEAR_MISS_RADIUS_M", 1.2))
        crash_pen = float(self.cfg.get("CRASH_PENALTY", 300.0)) * self.reward_scale
        near_pen  = float(self.cfg.get("NEAR_MISS_PENALTY", 3.0)) * self.reward_scale
        yield_steps = int(self.cfg.get("YIELD_STEPS", 3))
        approach_eps = float(self.cfg.get("NEAR_MISS_APPROACH_EPS", 0.08))

        T = self.sim.trucks
        for i in range(len(T)):
            for j in range(i+1, len(T)):
                ta, tb = T[i], T[j]
                d = dist(ta.pos, tb.pos)
                key = (ta.tid, tb.tid) if ta.tid < tb.tid else (tb.tid, ta.tid)
                last_d = self._pair_last_d.get(key, None)
                cd = self._near_cd_pairs.get(key, 0)
                stopped_pair = (self._speed_lp.get(ta.tid, 0.0) <= 0.02 or self._stopped.get(ta.tid, False)) and \
                               (self._speed_lp.get(tb.tid, 0.0) <= 0.02 or self._stopped.get(tb.tid, False))

                if d < crash_r:
                    rewards[i] -= crash_pen; rewards[j] -= crash_pen
                    winner = self._safety_row_winner(ta.tid, tb.tid)
                    loser  = tb if winner == ta.tid else ta
                    self._yield_cd[loser.tid] = max(self._yield_cd.get(loser.tid, 0), yield_steps)
                    self.sim.events.append({"t": self.sim.t, "type": "crash", "truck": ta.tid, "with": tb.tid})
                    self.sim.events.append({"t": self.sim.t, "type": "crash", "truck": tb.tid, "with": ta.tid})
                    self._near_cd_pairs[key] = max(cd, int(self.cfg.get("NEAR_MISS_COOLDOWN_STEPS", 10)))

                elif d < near_r:
                    approaching = (last_d is not None) and ((last_d - d) > approach_eps)
                    if cd <= 0 and approaching and not stopped_pair:
                        rewards[i] -= near_pen; rewards[j] -= near_pen
                        winner = self._safety_row_winner(ta.tid, tb.tid)
                        loser  = tb if winner == ta.tid else ta
                        self._yield_cd[loser.tid] = max(self._yield_cd.get(loser.tid, 0), yield_steps)
                        self.sim.events.append({"t": self.sim.t, "type": "near_miss", "truck": loser.tid, "with": (ta.tid if loser is tb else tb.tid)})
                        self._near_cd_pairs[key] = int(self.cfg.get("NEAR_MISS_COOLDOWN_STEPS", 10))

        self._update_pair_bookkeeping()

    def _update_pair_bookkeeping(self):
        for k in list(self._near_cd_pairs.keys()):
            if self._near_cd_pairs[k] > 0: self._near_cd_pairs[k] -= 1
            if self._near_cd_pairs[k] < 0: self._near_cd_pairs[k] = 0
        T = self.sim.trucks
        for i in range(len(T)):
            for j in range(i+1, len(T)):
                ta, tb = T[i], T[j]
                key = (ta.tid, tb.tid) if ta.tid < tb.tid else (tb.tid, ta.tid)
                self._pair_last_d[key] = dist(ta.pos, tb.pos)

    # ---------- car-following ----------
    def _ahead_vec(self, t):
        if t.route_pts and t.route_i < len(t.route_pts):
            tx, ty = t.route_pts[t.route_i]
        elif t.target is not None:
            tx, ty = t.target
        else:
            return (0.0, 0.0)
        dx, dy = (tx - t.pos[0]), (ty - t.pos[1])
        L = math.hypot(dx, dy)
        if L <= 1e-6: return (0.0, 0.0)
        return (dx / L, dy / L)

    def _car_following_speed_scale(self, t_self, all_trucks):
        if self.current_step < self._warmup_ticks: return 1.0, None
        if dist(t_self.pos, self.city.depot) <= self._depot_unblock_r: return 1.0, None

        cone_deg = float(self.cfg.get("FORWARD_CONE_DEG", 25.0))
        safe_stop = float(self.cfg.get("SAFE_STOP_M", 3.5))
        safe_slow = float(self.cfg.get("SAFE_SLOW_M", 7.0)); safe_slow = max(safe_slow, safe_stop + 1e-3)

        vx, vy = self._ahead_vec(t_self)
        if vx == 0.0 and vy == 0.0: return 1.0, None

        best_d = None; cos_cut = math.cos(math.radians(cone_deg))
        for t in all_trucks:
            if t is t_self: continue
            dx, dy = (t.pos[0] - t_self.pos[0]), (t.pos[1] - t_self.pos[1])
            d = math.hypot(dx, dy)
            if d <= 1e-6: continue
            if (vx*dx + vy*dy) / d < cos_cut: continue
            best_d = d if (best_d is None or d < best_d) else best_d

        if best_d is None: return 1.0, None
        if best_d <= safe_stop: return 0.0, best_d
        if best_d <= safe_slow: return ((best_d - safe_stop) / (safe_slow - safe_stop) * 0.5), best_d
        return 1.0, best_d

    # ---------- neighbor features ----------
    def _nearest_truck_norm(self, me):
        best = float('inf')
        for t in self.sim.trucks:
            if t is me: continue
            d = dist(me.pos, t.pos); best = d if d < best else best
        w, h = self.cfg["MAP_SIZE"]; diag = math.hypot(w, h)
        return float(min(1.0, best / max(1e-6, diag)))

    def _headway_norm(self, me):
        cone_deg = float(self.cfg.get("FORWARD_CONE_DEG", 25.0))
        vx, vy = self._ahead_vec(me)
        if vx == 0.0 and vy == 0.0: return 1.0
        cos_cut = math.cos(math.radians(cone_deg))
        best_d = None
        for t in self.sim.trucks:
            if t is me: continue
            dx, dy = (t.pos[0] - me.pos[0]), (t.pos[1] - me.pos[1])
            d = math.hypot(dx, dy)
            if d <= 1e-6: continue
            if (vx*dx + vy*dy) / d < cos_cut: continue
            best_d = d if (best_d is None or d < best_d) else best_d
        if best_d is None: return 1.0
        w, h = self.cfg["MAP_SIZE"]; diag = math.hypot(w, h)
        return float(min(1.0, best_d / max(1e-6, diag)))

    # ---------- pull-aside ----------
    def _pull_aside(self, loser):
        if self._pull_cd.get(loser.tid, 0) > 0: return False
        vx, vy = self._ahead_vec(loser)
        if vx == 0.0 and vy == 0.0: return False
        # lateral unit normal (right-hand traffic pulls to right by default)
        right_hand = True
        side_cfg = str(self.cfg.get("PULL_SIDE", "right")).lower()
        sign = +1.0 if side_cfg == "right" else -1.0
        nx, ny = (vy, -vx)  # right-side normal
        L = math.hypot(nx, ny); 
        if L <= 1e-6: return False
        nx, ny = nx / L, ny / L
        off = float(self.cfg.get("PULL_ASIDE_M", 2.2))
        target = (loser.pos[0] + sign * off * nx, loser.pos[1] + sign * off * ny)
        route = self.sim._plan_route(loser.pos, target)
        if not route or route[-1] != target: route = route + [target]
        # keep their assignment; temporary final target is the side pocket
        loser.assign_target(route, loser.assigned_bin, target)
        self._pull_cd[loser.tid] = int(self.cfg.get("PULL_COOLDOWN", 18))
        self.sim.events.append({"t": self.sim.t, "type": "pull_aside", "truck": loser.tid})
        return True

    # ---------- standoff resolver ----------
    def _facing_each_other(self, ta, tb, cone_deg):
        va = self._ahead_vec(ta); vb = self._ahead_vec(tb)
        if va == (0.0, 0.0) or vb == (0.0, 0.0): return False
        ax, ay = va; bx, by = vb
        dax, day = (tb.pos[0] - ta.pos[0]), (tb.pos[1] - ta.pos[1])
        dbx, dby = (ta.pos[0] - tb.pos[0]), (ta.pos[1] - tb.pos[1])
        da = math.hypot(dax, day); db = math.hypot(dbx, dby)
        if da <= 1e-6 or db <= 1e-6: return False
        cos_cut = math.cos(math.radians(cone_deg))
        cond_a = (ax*dax + ay*day) / da >= cos_cut
        cond_b = (bx*dbx + by*dby) / db >= cos_cut
        return cond_a and cond_b

    def _resolve_standoffs(self):
        safe_stop = float(self.cfg.get("SAFE_STOP_M", 3.5))
        cone = float(self.cfg.get("ROW_FACE_CONE_DEG", self.cfg.get("FORWARD_CONE_DEG", 25.0)))
        rng = safe_stop * float(self.cfg.get("ROW_RANGE_MULT", 1.15))
        need_ticks = int(self.cfg.get("STANDOFF_TICKS", 3))
        yield_steps = int(self.cfg.get("STANDOFF_YIELD_STEPS", 6))
        go_speed = float(self.cfg.get("ROW_WINNER_SPEED", 0.9))

        # update stuck counters
        for t in self.sim.trucks:
            has_route = bool(t.route_pts) or (t.target is not None)
            if has_route and (self._speed_lp.get(t.tid, 1.0) <= 0.02 or self._stopped.get(t.tid, False)):
                self._stuck_ticks[t.tid] = self._stuck_ticks.get(t.tid, 0) + 1
            else:
                self._stuck_ticks[t.tid] = 0

        T = self.sim.trucks
        for i in range(len(T)):
            for j in range(i+1, len(T)):
                ta, tb = T[i], T[j]
                d = dist(ta.pos, tb.pos)
                if d > rng: continue
                if self._stuck_ticks.get(ta.tid, 0) < need_ticks or self._stuck_ticks.get(tb.tid, 0) < need_ticks:
                    continue

                # treat both head-on and side-by-side stalls the same: pick a ROW winner deterministically
                winner_tid = self._safety_row_winner(ta.tid, tb.tid)
                winner = ta if winner_tid == ta.tid else tb
                loser  = tb if winner is ta else ta

                # winner goes (release + speed boost); loser yields and pulls aside if possible
                self._stopped[winner.tid] = False
                self._speed_lp[winner.tid] = max(self._speed_lp.get(winner.tid, 0.0), go_speed)
                self._yield_cd[loser.tid] = max(self._yield_cd.get(loser.tid, 0), yield_steps)
                self._pull_aside(loser)

                # suppress near-miss spam for this pair for a bit
                key = (winner.tid, loser.tid) if winner.tid < loser.tid else (loser.tid, winner.tid)
                self._near_cd_pairs[key] = max(self._near_cd_pairs.get(key, 0), int(self.cfg.get("NEAR_MISS_COOLDOWN_STEPS", 10)))
                self.sim.events.append({"t": self.sim.t, "type": "row_release", "go": winner.tid, "wait": loser.tid})

                self._stuck_ticks[winner.tid] = 0
                self._stuck_ticks[loser.tid] = 0

    def step(self, actions):
        dt = self.cfg["DT"]
        rewards = [0.0] * self.n_agents

        # potential baseline for shaping
        fill_prev = sum(b.fill for b in self.sim.bins)

        # 1) bins fill
        lo, hi = self.cfg["BIN_FILL_PER_STEP"]
        rnd = self.sim._rnd()
        p = float(self.cfg.get("BIN_FILL_PROB", 1.0))
        mult = float(self.cfg.get("BIN_FILL_MULT", 1.0))
        lo_eff = max(0, int(round(lo * mult))); hi_eff = max(lo_eff, int(round(hi * mult)))
        for b in self.sim.bins:
            if rnd.random() > p: continue
            before = b.fill
            if b.step_fill(lo_eff, hi_eff, rnd) and before < b.capacity:
                self.sim.events.append({"t": self.sim.t, "type": "overflow", "bin": b.id})

        # 2) auction + bootstrap
        auction(self.sim.bins, self.sim.trucks, self.sim.t, self.cfg, self.sim._plan_route)
        self._bootstrap_plans_if_needed()

        # --- team reward reference
        cost_prev = self._step_cost_eur()

        # 3) mask actions (default move when routed)
        raw_actions = list(actions)
        masked_actions = []
        allow_wait_when_routed = bool(self.cfg.get("ALLOW_WAIT_WHEN_ROUTED", False))

        for idx, truck in enumerate(self.sim.trucks):
            a = raw_actions[idx]
            at_depot = dist(truck.pos, self.city.depot) < 1.0
            has_route = bool(truck.route_pts) or (truck.target is not None)

            if self._yield_cd.get(truck.tid, 0) > 0:
                a = 4
            else:
                if a == 1 and not at_depot:
                    truck.assigned_bin = None
                    truck.target = self.city.depot
                    route = self.sim._plan_route(truck.pos, self.city.depot)
                    if not route or route[-1] != self.city.depot: route = route + [self.city.depot]
                    truck.assign_target(route, None, self.city.depot)
                    a = 0
                elif a == 3 and not at_depot:
                    a = 0

                if has_route and not allow_wait_when_routed:
                    a = 0
                elif not has_route and a == 0:
                    a = 4

            masked_actions.append(a)

        # cooldowns & pull cooldown decay
        for t in self.sim.trucks:
            if self._yield_cd.get(t.tid, 0) > 0: self._yield_cd[t.tid] -= 1
            if self._pull_cd.get(t.tid, 0) > 0: self._pull_cd[t.tid] -= 1

        # 3b) car-following with smoothing + anti-idle wake
        wake_ticks = int(self.cfg.get("ANTI_IDLE_WAKE_TICKS", 12))
        for t in self.sim.trucks:
            has_route = bool(t.route_pts) or (t.target is not None)
            if not has_route or self.current_step < self._warmup_ticks:
                self._stopped[t.tid] = False; self._speed_lp[t.tid] = 1.0; t.speed_scale = 1.0; continue

            scale_raw, best_d = self._car_following_speed_scale(t, self.sim.trucks)

            if self._stopped[t.tid]:
                stop_thr = float(self.cfg.get("SAFE_STOP_M", 3.5)) + self._release_hysteresis_m
                if best_d is not None and best_d >= stop_thr:
                    self._stopped[t.tid] = False
                else:
                    scale_raw = 0.0

            if (not self._stopped[t.tid]) and scale_raw <= 0.0:
                self._stopped[t.tid] = True

            # anti-idle: if no blocker in cone for a while, wake up
            if has_route and self._stopped[t.tid] and (best_d is None) and self._idle_ticks.get(t.tid, 0) >= wake_ticks:
                self._stopped[t.tid] = False
                scale_raw = max(scale_raw, float(self.cfg.get("MIN_SPEED_SCALE", self._min_speed_scale)))
                self.sim.events.append({"t": self.sim.t, "type": "wake_move", "truck": t.tid})

            if not self._stopped[t.tid] and scale_raw > 0.0:
                scale_raw = max(scale_raw, float(self.cfg.get("MIN_SPEED_SCALE", self._min_speed_scale)))

            prev = self._speed_lp[t.tid]; alpha = self._speed_smooth_alpha
            smoothed = (1 - alpha) * prev + alpha * float(min(max(0.0, scale_raw), 1.0))
            self._speed_lp[t.tid] = smoothed; t.speed_scale = smoothed

        # 3c) standoff resolution (ROW + pull-aside)
        self._resolve_standoffs()

        # 3d) proactive TTC yield: force losers to WAIT this tick
        losers = self._proactive_ttc_yield()
        if losers:
            for idx, truck in enumerate(self.sim.trucks):
                if truck.tid in losers:
                    # enforce WAIT now (override previous decision)
                    self._yield_cd[truck.tid] = max(self._yield_cd.get(truck.tid, 0), 1)
            # update masked actions accordingly
            for idx, truck in enumerate(self.sim.trucks):
                if self._yield_cd.get(truck.tid, 0) > 0:
                    # ensure we still have this local var; if not, move this block
                    try:
                        masked_actions[idx] = 4
                    except NameError:
                        pass


        # 4) apply actions
        for idx, truck in enumerate(self.sim.trucks):
            if masked_actions[idx] != 4:
                for ev in truck.step(dt, self.sim.bins, self.city.depot, self.sim._plan_route):
                    ev["t"] = self.sim.t
                    if ev.get("type") == "pickup":
                        bid = ev.get("bin")
                        b = next((bb for bb in self.sim.bins if bb.id == bid), None)
                        if b is not None: b.last_service_t = self.sim.t
                    self.sim.events.append(ev)
                self._idle_ticks[truck.tid] = 0
            else:
                self._idle_ticks[truck.tid] = self._idle_ticks.get(truck.tid, 0) + 1

        # 5) safety shaping
        self._apply_collisions(rewards)

        # --- team reward from € delta + potential fill bonus
        cost_now = self._step_cost_eur()
        r_team = -(cost_now - cost_prev)
        fill_now = sum(b.fill for b in self.sim.bins)
        beta = float(self.cfg.get("POTENTIAL_FILL_BONUS", 0.05))
        r_pot = beta * (fill_prev - fill_now)
        rewards = [ri + r_team + r_pot for ri in rewards]

        # 7) log + advance
        self._log_frame()
        self.sim.t += dt; self.current_step += 1

        obs = self._get_obs_all()
        done_flag = self.current_step >= self.max_steps
        dones = [done_flag] * self.n_agents
        info = {"costs": {}, "t": self.sim.t, "r_team": r_team}
        return obs, rewards, dones, info

    # ---------- observation builder ----------
    def _norm_d(self, x1, y1, x2, y2):
        w, h = self.cfg["MAP_SIZE"]
        dx = (x2 - x1) / max(1e-9, w); dy = (y2 - y1) / max(1e-9, h)
        d = math.hypot(dx, dy); return float(min(1.0, d))

    def _get_obs_all(self):
        return [self._get_obs(i, tr) for i, tr in enumerate(self.sim.trucks)]

    def _get_obs(self, idx, truck):
        w, h = self.cfg["MAP_SIZE"]; x, y = truck.pos
        load = truck.load / self.cfg["TRUCK_CAPACITY"]
        energy = truck.energy / self.cfg["ENERGY_MAX"]

        assigned_d, assigned_fill = 0.0, 0.0
        if truck.assigned_bin:
            b = next((bb for bb in self.sim.bins if bb.id == truck.assigned_bin), None)
            if b:
                px, py = (b.curb or b.pos)
                assigned_d = self._norm_d(x, y, px, py)
                assigned_fill = b.fill / b.capacity

        bins = sorted(self.sim.bins, key=lambda bb: math.hypot((bb.curb or bb.pos)[0] - x, (bb.curb or bb.pos)[1] - y))[:3]
        b_feats = []
        for b in bins:
            px, py = (b.curb or b.pos)
            d = self._norm_d(x, y, px, py); f = b.fill / b.capacity
            b_feats += [d, f]
        while len(b_feats) < 6: b_feats.append(0.0)

        nearest_norm = self._nearest_truck_norm(truck)
        headway_norm = self._headway_norm(truck)
        truck_id_norm = idx / max(1, self.n_agents - 1) if self.n_agents > 1 else 0.0

        base = [x / w, y / h, load, energy, assigned_d, assigned_fill] + b_feats
        return np.array(base + [nearest_norm, headway_norm, truck_id_norm], dtype=np.float32)
