# dispatch.py — cost-first hybrid assigner
from typing import List, Callable, Dict, Optional, Tuple
from agents import Truck, BinObj, dist
import math

def compute_due_time(bin: BinObj, inflow_rate: float, now: float) -> float:
    remain = max(0, bin.capacity - bin.fill)
    eta = remain / max(1e-9, inflow_rate)
    return now + eta

def _route_length_m(route: List[Tuple[float, float]]) -> float:
    if not route or len(route) < 2:
        return 0.0
    d = 0.0
    for i in range(len(route)-1):
        x1,y1 = route[i]
        x2,y2 = route[i+1]
        d += math.hypot(x2-x1, y2-y1)
    return d

def _travel_cost_eur(cfg: dict, meters: float) -> float:
    maint = (meters / 1000.0) * float(cfg["MAINT_EUR_PER_KM"])
    e_used = meters * float(cfg["ENERGY_PER_M"])
    energy = e_used * float(cfg["ENERGY_EUR_PER_UNIT"])
    return maint + energy

def _avoided_overflow_eur(cfg: dict, now: float, eta_s: float, due_s: float) -> float:
    """Credit (negative cost) if arriving before due time; scale by how much risk window we shave off."""
    fee = float(cfg.get("OVERFLOW_PENALTY_EUR", 0.0))
    if fee <= 0.0:
        return 0.0
    if eta_s >= due_s:
        return 0.0
    horizon = float(cfg.get("URGENCY_HORIZON_S", 300.0))
    margin = max(0.0, min(1.0, (due_s - eta_s) / max(1e-9, horizon)))
    return fee * margin

def auction(bins: List[BinObj], trucks: List[Truck], now: float, cfg: dict, plan_route: Callable):
    """Hybrid cost-minimizing assigner (greedy but costed in €)."""
    lo, hi = cfg["BIN_FILL_PER_STEP"]
    p     = float(cfg.get("BIN_FILL_PROB", 1.0))
    mult  = float(cfg.get("BIN_FILL_MULT", 1.0))
    dt    = float(cfg.get("DT", 1.0))
    # expected increment PER TICK
    inflow_rate = (0.5 * (lo + hi)) * p * mult / max(1e-9, dt)

    cooldown    = float(cfg.get("SERVICE_COOLDOWN_S", 60.0))
    reassign_m  = float(cfg.get("REASSIGN_MARGIN", 0.85))
    ttl_steps   = int(cfg.get("ASSIGN_TTL_STEPS", 20))
    speed_mps   = float(cfg.get("TRUCK_SPEED_MPS", 2.0))
    defer_thr   = float(cfg.get("DEFER_FILL_FRAC", 0.10))
    hot_threshold = 0.80

    # Snapshot existing ownership
    current_owner: Dict[str, str] = {}
    last_dist: Dict[str, float] = {}
    last_t: Dict[str, float] = {}
    for t in trucks:
        if t.assigned_bin:
            b_id = t.assigned_bin
            current_owner[b_id] = t.tid
            last_t[b_id] = getattr(next((bb for bb in bins if bb.id == b_id), None), "last_assign_t", now)
            last_dist[b_id] = getattr(next((bb for bb in bins if bb.id == b_id), None), "last_assign_dist", float("inf"))

    # Candidates: non-empty and not in cooldown
    cand = [b for b in bins if b.fill > 0 and (now - getattr(b, 'last_service_t', -1e9)) >= cooldown]
    if not cand:
        return []

    # HOT bins first
    cand.sort(key=lambda b: ( (b.fill / max(1,b.capacity)) < hot_threshold, -b.fill / max(1,b.capacity) ))

    # Helper: can a contender steal?
    def can_steal(bin_obj: BinObj, contender: Truck) -> bool:
        b_id = bin_obj.id
        if b_id not in current_owner:
            return True
        owner_tid = current_owner[b_id]
        owner = next((t for t in trucks if t.tid == owner_tid), None)
        if owner is None:
            return True
        d_owner = dist(owner.pos, bin_obj.curb or bin_obj.pos)
        d_cont  = dist(contender.pos, bin_obj.curb or bin_obj.pos)
        age_steps = (now - last_t.get(b_id, now)) / max(1.0, float(cfg.get("DT", 1.0)))
        baseline = last_dist.get(b_id, d_owner)
        if d_cont <= reassign_m * float(baseline):
            return True
        if age_steps >= ttl_steps:
            return d_cont <= d_owner
        return False

    # Build per-truck feasible target set with € costs
    def truck_bin_cost(t: Truck, b: BinObj) -> float:
        curb = b.curb or b.pos
        route = plan_route(t.pos, curb)
        if not route or route[-1] != curb:
            route = route + [curb]
        meters = _route_length_m(route)
        travel_eur = _travel_cost_eur(cfg, meters)
        eta_s = meters / max(1e-6, speed_mps)
        due_s = compute_due_time(b, inflow_rate, now)
        # Defer bias only when truly low fill and not hot
        fill_frac = b.fill / max(1, b.capacity)
        defer_bias = 0.0
        if (fill_frac < defer_thr) and (fill_frac < hot_threshold):
            defer_bias = travel_eur * 0.5
        credit = _avoided_overflow_eur(cfg, now, now + eta_s, due_s)
        return travel_eur - credit + defer_bias

    free = [t for t in trucks if not t.assigned_bin and not t.route_pts and t.target is None]
    assigned_events = []
    taken: set = set()

    while free and len(taken) < len(cand):
        best: Tuple[Optional[Truck], Optional[BinObj], float] = (None, None, float("inf"))
        for t in free:
            for b in cand:
                if b.id in taken:
                    continue
                if b.id in current_owner and not can_steal(b, t):
                    continue
                c = truck_bin_cost(t, b)
                if c < best[2]:
                    best = (t, b, c)
        t_best, b_best, _ = best
        if t_best is None or b_best is None:
            break
        curb = b_best.curb or b_best.pos
        route = plan_route(t_best.pos, curb)
        if not route or route[-1] != curb:
            route = route + [curb]
        t_best.assign_target(route, b_best.id, curb)
        assigned_events.append({"type": "assign", "truck": t_best.tid, "bin": b_best.id})
        current_owner[b_best.id] = t_best.tid
        b_best.last_assign_t = now
        b_best.last_assign_dist = dist(t_best.pos, curb)
        taken.add(b_best.id)
        free.remove(t_best)

    return assigned_events
