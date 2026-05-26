#!/usr/bin/env python3
# eval_dqn_cli.py — robust greedy eval for shared DQN
import os, sys, json, argparse
import torch
from collections import Counter

# Make local modules importable regardless of where you run this
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import CONFIG
from dqn_env import TruckEnv
from dqn_agent import DQNAgent


def _strip_module_prefix(state_dict):
    """Handle DataParallel checkpoints."""
    if not state_dict:
        return state_dict
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    if not has_module:
        return state_dict
    return {k[len("module."):]: v for k, v in state_dict.items()}


def load_agent(cfg: dict, weights_path: str):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"weights not found: {weights_path}")
    env = TruckEnv(cfg)
    agent = DQNAgent(env.obs_dim, env.action_space.n, cfg)
    sd = torch.load(weights_path, map_location="cpu")
    sd = _strip_module_prefix(sd)
    agent.q_net.load_state_dict(sd, strict=True)
    agent.target_net.load_state_dict(sd, strict=True)
    agent.eps = 0.0
    return env, agent


def _count_crashes_and_near_misses(events):
    """Returns (crash_incidents, near_miss_incidents) across sim/dqn variants."""
    # crashes
    crashes_ev = [e for e in events if e.get("type") == "crash"]
    if crashes_ev:
        seen = set(); crash_inc = 0
        for e in crashes_ev:
            a = str(e.get("truck")); b = str(e.get("with"))
            t = int(round(float(e.get("t", 0))))
            key = (min(a,b), max(a,b), t)
            if key not in seen:
                seen.add(key); crash_inc += 1
    else:
        pens = [e for e in events if e.get("type") == "crash_penalty"]
        by_t = Counter(int(round(float(e.get("t", 0)))) for e in pens)
        crash_inc = sum(n // 2 for n in by_t.values())

    # near-misses
    near_ev = [e for e in events if e.get("type") == "near_miss"]
    if near_ev:
        near_inc = len(near_ev)
    else:
        pens = [e for e in events if e.get("type") == "near_miss_penalty"]
        by_t = Counter(int(round(float(e.get("t", 0)))) for e in pens)
        near_inc = sum(n // 2 for n in by_t.values())

    return int(crash_inc), int(near_inc)


def kpis(sim):
    pickup = sum(e.get("amount", 0) for e in sim.events if e.get("type") == "pickup")
    overflows = sum(1 for e in sim.events if e.get("type") == "overflow")
    dumps = sum(1 for e in sim.events if e.get("type") == "drop")
    km = sum(float(getattr(t, "km_total", 0.0)) for t in sim.trucks)
    wage = sum(t.costs_eur.get("wage", 0.0) for t in sim.trucks)
    energy = sum(t.costs_eur.get("energy", 0.0) for t in sim.trucks)
    maint = sum(t.costs_eur.get("maint", 0.0) for t in sim.trucks)
    penalties = overflows * sim.cfg.get("OVERFLOW_PENALTY_EUR", 0.0)
    total = wage + energy + maint + penalties

    crashes, near_misses = _count_crashes_and_near_misses(sim.events)

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
        "crashes": float(crashes),
        "near_misses": float(near_misses),
        "steps": float(len(sim.frames)),
    }


def rollout_greedy(env: TruckEnv, agent: DQNAgent, max_steps=None):
    obs_all = env.reset()
    done = [False] * env.n_agents
    totals = [0.0] * env.n_agents
    limit = max_steps or int(env.cfg.get("STEPS_PER_DAY", 1200))

    while not all(done):
        acts = [agent.act_eval(obs_all[i]) for i in range(env.n_agents)]
        obs_all, r, done, _info = env.step(acts)
        for i in range(env.n_agents):
            totals[i] += float(r[i])
        if env.current_step >= limit:
            break

    avg = sum(totals) / max(1, len(totals))
    return avg, env.sim


def main():
    ap = argparse.ArgumentParser(description="Greedy DQN evaluation (shared policy)")
    ap.add_argument("--weights", required=True, help="Path to .pt weights")
    ap.add_argument("--steps", type=int, default=None, help="Override CONFIG['STEPS_PER_DAY']")
    ap.add_argument("--out-json", default=None, help="Optional: write sim frames/events here")
    args = ap.parse_args()

    cfg = CONFIG.copy()
    if args.steps is not None:
        cfg["STEPS_PER_DAY"] = int(args.steps)

    weights_path = os.path.abspath(args.weights)
    print(f"[eval] cwd={os.getcwd()}")
    print(f"[eval] using weights: {weights_path}")
    print(f"[eval] steps: {cfg['STEPS_PER_DAY']} | trucks={cfg['N_TRUCKS']} | bins={cfg['N_BINS']}")

    env, agent = load_agent(cfg, weights_path)
    avg_r, sim = rollout_greedy(env, agent, max_steps=cfg["STEPS_PER_DAY"])
    m = kpis(sim)

    print("\n== DQN Greedy Eval ==")
    print("avg_step_reward:", round(avg_r, 4))
    for k in ["pickup_units","overflows","dumps","fleet_km","wage_eur","energy_eur",
              "maintenance_eur","penalties_eur","total_eur","crashes","near_misses","steps"]:
        print(f"{k}: {round(m[k], 2)}")

    if args.out_json:
        blob = {"cfg": cfg, "frames": sim.frames, "events": sim.events}
        outp = os.path.abspath(args.out_json)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=2)
        print("Wrote", outp)


if __name__ == "__main__":
    main()
