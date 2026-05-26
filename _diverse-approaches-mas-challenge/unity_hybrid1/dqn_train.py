# dqn_train.py — shared-policy trainer that uses TruckEnv
import os, time, torch
from tqdm import trange
from dqn_env import TruckEnv
from dqn_agent import DQNAgent

def _proxy_cost(env) -> float:
    """Compute econ-like cost from the live Simulation inside env (no summary_costs needed)."""
    sim = env.sim
    # wages already accounted in Truck.step's costs_eur["wage"] per truck
    wage = sum(t.costs_eur.get("wage", 0.0) for t in sim.trucks)
    energy = sum(t.costs_eur.get("energy", 0.0) for t in sim.trucks)
    maint  = sum(t.costs_eur.get("maint", 0.0) for t in sim.trucks)
    overflows = sum(1 for e in sim.events if e.get("type") == "overflow")
    penalties = overflows * float(env.cfg.get("OVERFLOW_PENALTY_EUR", 0.0))
    return float(wage + energy + maint + penalties)

def _eval_greedy_cost(cfg, agent, steps=None):
    env = TruckEnv(cfg)
    if steps is None:
        steps = int(cfg.get("STEPS_PER_DAY", 1200))
    old_eps = agent.eps
    agent.eps = 0.0
    obs_all = env.reset()
    done = [False]*env.n_agents
    while not all(done):
        acts = [agent.act_eval(obs_all[i]) for i in range(env.n_agents)]
        obs_all, _r, done, _info = env.step(acts)
        if env.current_step >= steps:
            break
    agent.eps = old_eps
    return _proxy_cost(env)

def train(cfg, episodes=200, verbose=True, save_checkpoints=True, eval_every=5):
    env = TruckEnv(cfg)
    agent = DQNAgent(env.obs_dim, env.action_space.n, cfg)

    rewards_hist = []
    best_cost = float("inf")
    best_path = None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    last_path = None

    for ep in trange(episodes, desc="Training Episodes", unit="ep"):
        obs_all = env.reset()
        total_rewards = [0.0]*env.n_agents
        done = [False]*env.n_agents

        while not all(done):
            acts = [agent.act(obs_all[i]) for i in range(env.n_agents)]
            next_obs_all, rewards, done, _info = env.step(acts)

            for i in range(env.n_agents):
                agent.store(obs_all[i], acts[i], rewards[i], next_obs_all[i], done[i])
                total_rewards[i] += rewards[i]

            agent.update()
            obs_all = next_obs_all

        avg_reward = sum(total_rewards)/env.n_agents
        rewards_hist.append(avg_reward)
        if verbose:
            print(f"Ep {ep} avg reward={avg_reward:.3f} eps={agent.eps:.2f}")

        if (ep + 1) % eval_every == 0:
            cost = _eval_greedy_cost(cfg, agent, steps=cfg.get("STEPS_PER_DAY", None))
            if verbose:
                print(f"[eval @ ep {ep+1}] proxy_total_eur={cost:.2f} (best={best_cost:.2f})")
            if cost < best_cost and save_checkpoints:
                os.makedirs("models", exist_ok=True)
                best_cost = cost
                best_path = f"models/dqn_shared_best_{stamp}.pt"
                torch.save(agent.q_net.state_dict(), best_path)

        if save_checkpoints:
            os.makedirs("models", exist_ok=True)
            last_path = f"models/dqn_shared_last_{stamp}.pt"
            torch.save(agent.q_net.state_dict(), last_path)

    paths = []
    if save_checkpoints:
        if last_path: paths.append(last_path)
        if best_path: paths.append(best_path)
    return [agent], rewards_hist, paths

if __name__ == "__main__":
    import argparse, json, os
    from config import CONFIG

    ap = argparse.ArgumentParser(description="Train shared-policy DQN for truck env")
    ap.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    ap.add_argument("--steps", type=int, default=None, help="Override CONFIG['STEPS_PER_DAY'] during training")
    ap.add_argument("--save-every", type=int, default=5, help="Greedy eval cadence (also prints cost)")
    ap.add_argument("--outdir", type=str, default="models", help="Where to save checkpoints")
    ap.add_argument("--verbose", action="store_true", help="Print per-episode logs")
    args = ap.parse_args()

    cfg = CONFIG.copy()
    if args.steps is not None:
        cfg["STEPS_PER_DAY"] = int(args.steps)

    os.makedirs(args.outdir, exist_ok=True)

    agents, rewards_hist, paths = train(
        cfg,
        episodes=args.episodes,
        verbose=args.verbose or True,
        save_checkpoints=True,
        eval_every=max(1, int(args.save_every)),
    )

    # Just to be explicit in STDOUT:
    print("\n=== Training finished ===")
    if paths:
        print("Saved checkpoints:")
        for p in paths:
            print(" -", p)
    else:
        print("No checkpoints saved.")
    print("Episodes:", args.episodes)
    print("Steps/episode:", cfg["STEPS_PER_DAY"])
