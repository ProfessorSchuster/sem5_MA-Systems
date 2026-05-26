# smoke_stage2.py
from config import CONFIG
from city import City
from sim import Simulation
from visualize import preview
from agents import dist

def main():
    cfg = CONFIG.copy()
    cfg.update({
        "N_TRUCKS": 4,
        "BIN_FILL_PER_STEP": (0, 1),     # keep background quiet for test
        "INTERSECTION_APPROACH_M": 6.0,  # GATING ON
        "INTERSECTION_CLEAR_M": 9.0,
        "ASSIGN_FREEZE_IN_JUNCTION": True,
        "JUNCTION_STALL_STEPS": 10,
        "SERVICE_RADIUS_M": 7.0,
        "DOCK_SWITCH_M": 16.0,
    })

    city = City(cfg)
    cfg["plan_route_fn"] = city.plan_route
    sim = Simulation(cfg, city)

    t = sim.trucks[0]
    depot = city.depot
    target_bin = max(sim.bins, key=lambda b: dist(b.pos, depot))
    target_bin.fill = max(target_bin.fill, int(0.9 * target_bin.capacity))
    for b in sim.bins:
        if b is not target_bin:
            b.fill = min(b.fill, int(0.2 * b.capacity))

    curb = getattr(target_bin, "curb", target_bin.pos)
    first_route = city.plan_route(t.pos, curb)
    t.assign_target(first_route, target_bin.id, curb, force=True)

    first_pick, first_drop = False, False
    for _ in range(cfg["STEPS_PER_DAY"]):
        if not first_pick:
            t.assign_hold_steps = 999  # protect first leg from auction
        sim.step()
        if not first_pick and t.load > 0:
            first_pick = True
            t.assigned_bin = None
            t.assign_hold_steps = 999
            t.target = depot
            route_home = city.plan_route(t.pos, depot)
            t.assign_target(route_home, None, depot, force=True)
        if first_pick and not first_drop:
            if any(e.get("type")=="drop" and e.get("truck")==t.tid for e in sim.events):
                first_drop = True
                t.assign_hold_steps = 0  # release; done with the guarantee

    pickups = sum(1 for e in sim.events if e.get("type")=="pickup")
    drops   = sum(1 for e in sim.events if e.get("type")=="drop")
    print(f"Stage2 smoke: pickups={pickups} drops={drops}")
    preview(sim)

if __name__ == "__main__":
    main()
