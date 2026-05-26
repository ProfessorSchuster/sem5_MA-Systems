# eval_compare.py — compare baseline (auction) vs DQN shared-policy
import argparse, copy, csv, os, statistics as stats
from typing import Dict, Any, List
from config import CONFIG
from city import City
from sim import Simulation
from dispatch import auction

import torch
from dqn_env import TruckEnv
from dqn_agent import DQNAgent

def _seeded_cfg(base: Dict[str, Any], seed: int) -> Dict[str, Any]:
    cfg = copy.deepcopy(base); cfg["SEED"] = int(seed); return cfg

def _kpis(sim: Simulation) -> Dict[str, float]:
    pickup = sum(e.get("amount", 0) for e in sim.events if e.get("type") == "pickup")
    overflows = sum(1 for e in sim.events if e.get("type") == "overflow")
    dumps = sum(1 for e in sim.events if e.get("type") == "drop")
    km = sum(float(getattr(t, "km_total", 0.0)) for t in sim.trucks)
    wage = sum(t.costs_eur.get("wage", 0.0) for t in sim.trucks)
    energy = sum(t.costs_eur.get("energy", 0.0) for t in sim.trucks)
    maint  = sum(t.costs_eur.get("maint", 0.0) for t in sim.trucks)
    penalties = overflows * sim.cfg.get("OVERFLOW_PENALTY_EUR", 0.0)
    total = wage + energy + maint + penalties
    return {
        "pickup_units": float(pickup),
        "overflows": float(overflows),
        "dumps": float(dumps),
        "fleet_km": float(km),
        "wage_eur": float(wage),
        "energy_eur": float(energy),
        "maintenance_eur": float(maint),
        "penalties_eur": float(penalties),
        "total_eur": float(total),
        "steps": float(len(sim.frames)),
    }

def run_baseline(cfg: Dict[str, Any], steps: int) -> Simulation:
    city = City(cfg); sim = Simulation(cfg=cfg, city=city, planner="graph")
    for _ in range(steps):
        sim._fill_bins()
        assigns = auction(sim.bins, sim.trucks, sim.t, cfg, sim._plan_route)
        for ev in assigns:
            sim.events.append({"t": sim.t, "type": "assign", "truck": ev["truck"], "bin": ev["bin"]})
        for t in sim.trucks:
            for ev in t.step(cfg["DT"], sim.bins, city.depot, sim._plan_route):
                ev["t"] = sim.t
                if ev.get("type") == "pickup":
                    bid = ev.get("bin")
                    b = next((bb for bb in sim.bins if bb.id == bid), None)
                    if b is not None:
                        b.last_service_t = sim.t
                sim.events.append(ev)
        frame = {"t": sim.t,
                 "trucks": [{"id": t.tid, "x": t.pos[0], "y": t.pos[1],
                             "energy": t.energy, "load": t.load, "state": t.state,
                             "target": (None if t.target is None else {"x": t.target[0], "y": t.target[1]})}
                            for t in sim.trucks],
                 "bins": [{"id": b.id, "x": b.pos[0], "y": b.pos[1], "fill": b.fill, "cap": b.capacity}
                          for b in sim.bins]}
        sim.frames.append(frame)
        sim.t += cfg["DT"]
    return sim

def load_dqn_agent(cfg: Dict[str, Any], weights_path: str):
    env = TruckEnv(cfg)
    agent = DQNAgent(env.obs_dim, env.action_space.n, cfg)
    sd = torch.load(weights_path, map_location="cpu")
    agent.q_net.load_state_dict(sd)
    agent.target_net.load_state_dict(sd)
    agent.eps = 0.0
    return env, agent

def run_dqn_eval(cfg: Dict[str, Any], steps: int, weights_path: str) -> Simulation:
    env, agent = load_dqn_agent(cfg, weights_path)
    obs_all = env.reset()
    done = [False] * env.n_agents
    while not all(done):
        acts = [agent.act_eval(obs_all[i]) for i in range(env.n_agents)]
        obs_all, _r, done, _info = env.step(acts)
        if env.current_step >= steps:
            break
    return env.sim

def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    out = {}
    keys = ["total_eur","wage_eur","energy_eur","maintenance_eur",
            "penalties_eur","pickup_units","overflows","dumps","fleet_km","steps"]
    for k in keys:
        vals = [r[k] for r in rows]
        m = stats.mean(vals)
        s = (stats.pstdev(vals) if len(vals) > 1 else 0.0)
        out[k+"_mean"] = m; out[k+"_std"] = s
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    base = copy.deepcopy(CONFIG)
    base["STEPS_PER_DAY"] = max(base.get("STEPS_PER_DAY", args.steps), args.steps)

    bl_rows, rl_rows = [], []
    print(f"\n== Compare on {args.trials} seeds, {args.steps} steps each ==")
    for i in range(args.trials):
        seed = base.get("SEED", 42) + i
        print(f"\n[trial {i+1}/{args.trials}] seed={seed}")
        cfg_bl = _seeded_cfg(base, seed)
        sim_bl = run_baseline(cfg_bl, args.steps)
        kpi_bl = _kpis(sim_bl); bl_rows.append(kpi_bl)
        print("  baseline:", {k: round(v,2) for k,v in kpi_bl.items()})

        cfg_rl = _seeded_cfg(base, seed)
        sim_rl = run_dqn_eval(cfg_rl, args.steps, args.weights)
        kpi_rl = _kpis(sim_rl); rl_rows.append(kpi_rl)
        print("  dqn:", {k: round(v,2) for k,v in kpi_rl.items()})

    def fmt(d): return {k: round(v,2) for k,v in d.items()}
    print("\n== Averages ==")
    print("baseline:", fmt(aggregate(bl_rows)))
    print("dqn:     ", fmt(aggregate(rl_rows)))

    if args.csv:
        fields = ["policy","trial","seed","total_eur","wage_eur","energy_eur","maintenance_eur",
                  "penalties_eur","pickup_units","overflows","dumps","fleet_km","steps"]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i,(b,r) in enumerate(zip(bl_rows, rl_rows)):
                seed = base.get("SEED",42) + i
                w.writerow({"policy":"baseline","trial":i+1,"seed":seed, **b})
                w.writerow({"policy":"dqn","trial":i+1,"seed":seed, **r})
        print("CSV ->", os.path.abspath(args.csv))

if __name__ == "__main__":
    main()
