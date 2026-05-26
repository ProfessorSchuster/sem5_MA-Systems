# dqn_env.py — RL wrapper Environment around your new Simulation/City
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

        # obs = [x/w, y/h, load%, energy%, assigned_d, assigned_fill, (3 bins * (d,f)), truck_id_norm, nearest_norm, headway_norm]
        self.obs_dim = (4 + 2 + 3 * 2) + 1
        self.obs_dim += 2  # nearest_norm, headway_norm
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.reward_scale = float(self.cfg.get("REWARD_SCALE", 0.01))
        self.max_penalties_per_tick = self.cfg.get("MAX_PENALTIES_PER_TICK", None)

        self._yield_cd = {t.tid: 0 for t in self.sim.trucks}
        self._idle_ticks = {t.tid: 0 for t in self.sim.trucks}

        # Deadlock-avoidance / safety knobs
        self._warmup_ticks = int(self.cfg.get("WARMUP_TICKS", 2))
        self._depot_unblock_r = float(self.cfg.get("DEPOT_UNBLOCK_RADIUS_M", 3.0))
        self._min_speed_scale = float(self.cfg.get("MIN_SPEED_SCALE", 0.15))

        # Smooth + decisive car-following
        self._speed_lp = {t.tid: 1.0 for t in self.sim.trucks}
        self._stopped = {t.tid: False for t in self.sim.trucks}
        self._release_hysteresis_m = 0.6
        self._speed_smooth_alpha = 0.6

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
        self._speed_lp = {t.tid: 1.0 for t in self.sim.trucks}
        self._stopped = {t.tid: False for t in self.sim.trucks}

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
            if b is None:
                continue
            curb = b.curb or b.pos
            route = self.sim._plan_route(t.pos, curb)
            if not route or route[-1] != curb:
                route = route + [curb]
            t.assign_target(route, b.id, curb)

    # ---------- helpers ----------
    def _safety_row_winner(self, tid_a: str, tid_b: str) -> str:
        rule = str(self.cfg.get("ROW_RULE", "lower_id_wins"))
        if rule == "lower_id_wins":
            return tid_a if tid_a < tid_b else tid_b
        return tid_a

    def _step_cost_eur(self):
        sim = self.sim
        wage = sum(t.costs_eur.get("wage", 0.0) for t in sim.trucks)
        energy = sum(t.costs_eur.get("energy", 0.0) for t in sim.trucks)
        maint = sum(t.costs_eur.get("maint", 0.0) for t in sim.trucks)
        t_now = sim.t
        overflow_fee = sum(
            float(self.cfg.get("OVERFLOW_PENALTY_EUR", 0.0))
            for e in sim.events if e.get("type") == "overflow" and e.get("t") == t_now
        )
        crash_fee = sum(
            float(self.cfg.get("CRASH_PENALTY", 0.0))
            for e in sim.events if e.get("type") == "crash" and e.get("t") == t_now
        )
        near_fee = sum(
            float(self.cfg.get("NEAR_MISS_PENALTY", 0.0))
            for e in sim.events if e.get("type") == "near_miss" and e.get("t") == t_now
        )
        return wage + energy + maint + overflow_fee + crash_fee + near_fee

    def _apply_collisions(self, rewards):
        if self.current_step < self._warmup_ticks:
            return
        crash_r = float(self.cfg.get("CRASH_RADIUS_M", 1.2))
        near_r  = float(self.cfg.get("NEAR_MISS_RADIUS_M", 2.0))
        crash_pen = float(self.cfg.get("CRASH_PENALTY", 300.0)) * self.reward_scale
        near_pen  = float(self.cfg.get("NEAR_MISS_PENALTY", 3.0)) * self.reward_scale
        yield_steps = int(self.cfg.get("YIELD_STEPS", 3))

        T = self.sim.trucks
        for i in range(len(T)):
            for j in range(i+1, len(T)):
                ta, tb = T[i], T[j]
                d = dist(ta.pos, tb.pos)
                if d <= 1e-6:
                    d = 0.0
                if d < crash_r:
                    rewards[i] -= crash_pen
                    rewards[j] -= crash_pen
                    winner = self._safety_row_winner(ta.tid, tb.tid)
                    loser  = tb if winner == ta.tid else ta
                    self._yield_cd[loser.tid] = max(self._yield_cd.get(loser.tid, 0), yield_steps)
                    self.sim.events.append({"t": self.sim.t, "type": "crash", "truck": ta.tid, "with": tb.tid})
                    self.sim.events.append({"t": self.sim.t, "type": "crash", "truck": tb.tid, "with": ta.tid})
                elif d < near_r:
                    rewards[i] -= near_pen
                    rewards[j] -= near_pen
                    winner = self._safety_row_winner(ta.tid, tb.tid)
                    loser  = tb if winner == ta.tid else ta
                    self._yield_cd[loser.tid] = max(self._yield_cd.get(loser.tid, 0), yield_steps)
                    self.sim.events.append({"t": self.sim.t, "type": "near_miss", "truck": loser.tid, "with": (ta.tid if loser is tb else tb.tid)})

    # ---------- proactive car-following ----------
    def _ahead_vec(self, t):
        if t.route_pts and t.route_i < len(t.route_pts):
            tx, ty = t.route_pts[t.route_i]
        elif t.target is not None:
            tx, ty = t.target
        else:
            return (0.0, 0.0)
        dx, dy = (tx - t.pos[0]), (ty - t.pos[1])
        L = math.hypot(dx, dy)
        if L <= 1e-6:
            return (0.0, 0.0)
        return (dx / L, dy / L)

    def _car_following_speed_scale(self, t_self, all_trucks):
        if self.current_step < self._warmup_ticks:
            return 1.0, None
        if dist(t_self.pos, self.city.depot) <= self._depot_unblock_r:
            return 1.0, None

        cone_deg = float(self.cfg.get("FORWARD_CONE_DEG", 25.0))
        safe_stop = float(self.cfg.get("SAFE_STOP_M", 3.5))
        safe_slow = float(self.cfg.get("SAFE_SLOW_M", 7.0))
        safe_slow = max(safe_slow, safe_stop + 1e-3)

        vx, vy = self._ahead_vec(t_self)
        if vx == 0.0 and vy == 0.0:
            return 1.0, None

        best_d = None
        cos_cut = math.cos(math.radians(cone_deg))
        for t in all_trucks:
            if t is t_self:
                continue
            dx, dy = (t.pos[0] - t_self.pos[0]), (t.pos[1] - t_self.pos[1])
            d = math.hypot(dx, dy)
            if d <= 1e-6:
                continue
            if (vx*dx + vy*dy) / d < cos_cut:
                continue
            best_d = d if (best_d is None or d < best_d) else best_d

        if best_d is None:
            return 1.0, None
        if best_d <= safe_stop:
            return 0.0, best_d
        if best_d <= safe_slow:
            return ((best_d - safe_stop) / (safe_slow - safe_stop) * 0.5), best_d
        return 1.0, best_d

    # ---------- neighbor features ----------
    def _nearest_truck_norm(self, me):
        best = float('inf')
        for t in self.sim.trucks:
            if t is me:
                continue
            d = dist(me.pos, t.pos)
            if d < best:
                best = d
        w, h = self.cfg["MAP_SIZE"]
        diag = math.hypot(w, h)
        return float(min(1.0, best / max(1e-6, diag)))

    def _headway_norm(self, me):
        cone_deg = float(self.cfg.get("FORWARD_CONE_DEG", 25.0))
        vx, vy = self._ahead_vec(me)
        if vx == 0.0 and vy == 0.0:
            return 1.0
        cos_cut = math.cos(math.radians(cone_deg))
        best_d = None
        for t in self.sim.trucks:
            if t is me:
                continue
            dx, dy = (t.pos[0] - me.pos[0]), (t.pos[1] - me.pos[1])
            d = math.hypot(dx, dy)
            if d <= 1e-6:
                continue
            if (vx*dx + vy*dy) / d < cos_cut:
                continue
            best_d = d if (best_d is None or d < best_d) else best_d
        if best_d is None:
            return 1.0
        w, h = self.cfg["MAP_SIZE"]
        diag = math.hypot(w, h)
        return float(min(1.0, best_d / max(1e-6, diag)))

    def step(self, actions):
        dt = self.cfg["DT"]
        rewards = [0.0] * self.n_agents

        # Potential shaping baseline
        fill_prev = sum(b.fill for b in self.sim.bins)

        # 1) bins fill + overflow events
        lo, hi = self.cfg["BIN_FILL_PER_STEP"]
        rnd = self.sim._rnd()
        p = float(self.cfg.get("BIN_FILL_PROB", 1.0))
        mult = float(self.cfg.get("BIN_FILL_MULT", 1.0))
        lo_eff = max(0, int(round(lo * mult)))
        hi_eff = max(lo_eff, int(round(hi * mult)))

        for b in self.sim.bins:
            if rnd.random() > p:
                continue
            before = b.fill
            if b.step_fill(lo_eff, hi_eff, rnd) and before < b.capacity:
                self.sim.events.append({"t": self.sim.t, "type": "overflow", "bin": b.id})

        # 2) auction + bootstrap
        auction(self.sim.bins, self.sim.trucks, self.sim.t, self.cfg, self.sim._plan_route)
        self._bootstrap_plans_if_needed()

        # --- Team reward from € delta
        cost_prev = self._step_cost_eur()

        # 3) action masking (safety + DEFAULT-MOVE WHEN ROUTED)
        raw_actions = list(actions)
        masked_actions = []
        allow_wait_when_routed = bool(self.cfg.get("ALLOW_WAIT_WHEN_ROUTED", False))

        for idx, truck in enumerate(self.sim.trucks):
            a = raw_actions[idx]
            at_depot = dist(truck.pos, self.city.depot) < 1.0
            has_route = bool(truck.route_pts) or (truck.target is not None)

            # must yield -> WAIT
            if self._yield_cd.get(truck.tid, 0) > 0:
                a = 4
            else:
                # special actions
                if a == 1 and not at_depot:
                    truck.assigned_bin = None
                    truck.target = self.city.depot
                    route = self.sim._plan_route(truck.pos, self.city.depot)
                    if not route or route[-1] != self.city.depot:
                        route = route + [self.city.depot]
                    truck.assign_target(route, None, self.city.depot)
                    a = 0  # then move

                elif a == 3 and not at_depot:
                    a = 0  # cannot recharge on road; just move

                # DEFAULT MOVE WHEN ROUTED
                if has_route and not allow_wait_when_routed:
                    a = 0
                elif not has_route and a == 0:
                    # don't MOVE if we have nowhere to go
                    a = 4

            masked_actions.append(a)

        # Decrement yield cooldowns
        for t in self.sim.trucks:
            if self._yield_cd.get(t.tid, 0) > 0:
                self._yield_cd[t.tid] -= 1

        # 3b) car-following speed scaling with hysteresis & smoothing
        for t in self.sim.trucks:
            has_route = bool(t.route_pts) or (t.target is not None)
            if not has_route or self.current_step < self._warmup_ticks:
                self._stopped[t.tid] = False
                self._speed_lp[t.tid] = 1.0
                t.speed_scale = 1.0
                continue

            scale_raw, best_d = self._car_following_speed_scale(t, self.sim.trucks)

            if self._stopped[t.tid]:
                stop_thr = float(self.cfg.get("SAFE_STOP_M", 3.5)) + self._release_hysteresis_m
                if best_d is not None and best_d >= stop_thr:
                    self._stopped[t.tid] = False
                else:
                    scale_raw = 0.0

            if (not self._stopped[t.tid]) and scale_raw <= 0.0:
                self._stopped[t.tid] = True

            if not self._stopped[t.tid] and scale_raw > 0.0:
                scale_raw = max(scale_raw, float(self.cfg.get("MIN_SPEED_SCALE", 0.15)))

            prev = self._speed_lp[t.tid]
            alpha = self._speed_smooth_alpha
            smoothed = (1 - alpha) * prev + alpha * float(min(max(0.0, scale_raw), 1.0))
            self._speed_lp[t.tid] = smoothed
            t.speed_scale = smoothed

        # 4) apply actions
        for idx, truck in enumerate(self.sim.trucks):
            if masked_actions[idx] != 4:
                for ev in truck.step(dt, self.sim.bins, self.city.depot, self.sim._plan_route):
                    ev["t"] = self.sim.t
                    if ev.get("type") == "pickup":
                        bid = ev.get("bin")
                        b = next((bb for bb in self.sim.bins if bb.id == bid), None)
                        if b is not None:
                            b.last_service_t = self.sim.t
                    self.sim.events.append(ev)
                self._idle_ticks[truck.tid] = 0
            else:
                self._idle_ticks[truck.tid] = self._idle_ticks.get(truck.tid, 0) + 1

        # 5) safety shaping (adds events used by € cost)
        self._apply_collisions(rewards)

        # --- compute team reward from € delta
        cost_now = self._step_cost_eur()
        r_team = -(cost_now - cost_prev)

        # Potential-based shaping: reward reduction in total fill
        fill_now = sum(b.fill for b in self.sim.bins)
        beta = float(self.cfg.get("POTENTIAL_FILL_BONUS", 0.05))
        r_pot = beta * (fill_prev - fill_now)

        rewards = [ri + r_team + r_pot for ri in rewards]

        # 7) log + advance time
        self._log_frame()
        self.sim.t += dt
        self.current_step += 1

        obs = self._get_obs_all()
        done_flag = self.current_step >= self.max_steps
        dones = [done_flag] * self.n_agents
        info = {"costs": {}, "t": self.sim.t, "r_team": r_team}
        return obs, rewards, dones, info

    # ---------- observation builder ----------
    def _norm_d(self, x1, y1, x2, y2):
        w, h = self.cfg["MAP_SIZE"]
        dx = (x2 - x1) / max(1e-9, w)
        dy = (y2 - y1) / max(1e-9, h)
        d = math.hypot(dx, dy)
        return float(min(1.0, d))

    def _get_obs_all(self):
        return [self._get_obs(i, tr) for i, tr in enumerate(self.sim.trucks)]

    def _get_obs(self, idx, truck):
        w, h = self.cfg["MAP_SIZE"]
        x, y = truck.pos
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
            d = self._norm_d(x, y, px, py)
            f = b.fill / b.capacity
            b_feats += [d, f]
        while len(b_feats) < 6:
            b_feats.append(0.0)

        nearest_norm = self._nearest_truck_norm(truck)
        headway_norm = self._headway_norm(truck)

        truck_id_norm = idx / max(1, self.n_agents - 1) if self.n_agents > 1 else 0.0
        base = [x / w, y / h, load, energy, assigned_d, assigned_fill] + b_feats
        return np.array(base + [nearest_norm, headway_norm, truck_id_norm], dtype=np.float32)
