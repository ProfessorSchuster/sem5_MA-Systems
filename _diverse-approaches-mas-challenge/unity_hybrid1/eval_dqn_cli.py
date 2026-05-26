# eval_dqn_cli.py
import argparse, json, os
from config import CONFIG
from eval_dqn import load_agents, rollout_greedy

def kpis(sim):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to .pt weights")
    ap.add_argument("--steps", type=int, default=None, help="Override CONFIG['STEPS_PER_DAY']")
    ap.add_argument("--out-json", default=None, help="Optional: write sim frames/events here")
    args = ap.parse_args()

    cfg = CONFIG.copy()
    if args.steps is not None:
        cfg["STEPS_PER_DAY"] = int(args.steps)

    env, agent = load_agents(cfg, args.weights)
    avg_r, sim, _ = rollout_greedy(env, agent)
    m = kpis(sim)

    print("\n== DQN Greedy Eval ==")
    print("avg_step_reward:", round(avg_r, 4))
    for k, v in m.items():
        print(f"{k}: {round(v, 2)}")

    if args.out_json:
        blob = {"cfg": cfg, "frames": sim.frames, "events": sim.events}
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False, indent=2)
        print("Wrote", os.path.abspath(args.out_json))

if __name__ == "__main__":
    main()
