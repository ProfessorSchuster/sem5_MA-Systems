# dqn_train.py — shared-policy trainer that uses TruckEnv
import os, time, signal, torch
from tqdm import trange
from dqn_env import TruckEnv
from dqn_agent import DQNAgent

_STOP = False
def _handle_sigint(*_a):
    global _STOP
    _STOP = True
signal.signal(signal.SIGINT, _handle_sigint)

def _proxy_cost(env) -> float:
    sim = env.sim
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

def train(cfg, episodes=200, verbose=True, save_checkpoints=True, eval_every=None, resume_path=None, outdir=None, forever=False):
    outdir = outdir or cfg.get("TRAIN_SAVE_DIR", "models")
    os.makedirs(outdir, exist_ok=True)
    eval_every = eval_every or int(cfg.get("TRAIN_EVAL_EVERY_EP", 10))

    env = TruckEnv(cfg)
    agent = DQNAgent(env.obs_dim, env.action_space.n, cfg)

    if resume_path and os.path.isfile(resume_path):
        sd = torch.load(resume_path, map_location="cpu")
        agent.q_net.load_state_dict(sd)
        agent.target_net.load_state_dict(sd)
        agent.eps = max(agent.eps, cfg.get("EPS_END", 0.05))  # keep some exploration
        print(f"[resume] loaded weights: {resume_path}")

    rewards_hist = []
    best_cost = float("inf")
    stamp = time.strftime("%Y%m%d-%H%M%S")

    ep_iter = (iter(int, 1) if forever else range(episodes))
    for ep in (trange(episodes, desc="Training", unit="ep") if not forever else trange(999999999, desc="Training∞", unit="ep")):
        if _STOP:
            break
        obs_all = env.reset()
        total_rewards = [0.0]*env.n_agents
        done = [False]*env.n_agents

        while not all(done):
            if _STOP:
                break
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
            print(f"Ep {ep} avg_reward={avg_reward:.4f} eps={agent.eps:.3f}")

        # greedy eval + checkpointing
        if save_checkpoints:
            last_path = os.path.join(outdir, f"dqn_shared_last_{stamp}.pt")
            torch.save(agent.q_net.state_dict(), last_path)

            if (ep + 1) % max(1, int(eval_every)) == 0:
                cost = _eval_greedy_cost(cfg, agent, steps=cfg.get("STEPS_PER_DAY", None))
                if verbose:
                    print(f"[eval @ ep {ep+1}] greedy_proxy_total_eur={cost:.2f} (best={best_cost:.2f})")
                if cost < best_cost:
                    best_cost = cost
                    best_path = os.path.join(outdir, f"dqn_shared_best_{stamp}.pt")
                    torch.save(agent.q_net.state_dict(), best_path)

        if forever and _STOP:
            break

    return [agent], rewards_hist, []  # paths already printed

if __name__ == "__main__":
    import argparse
    from config import CONFIG

    ap = argparse.ArgumentParser(description="Train shared-policy DQN for truck env")
    ap.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    ap.add_argument("--steps", type=int, default=None, help="Override CONFIG['STEPS_PER_DAY'] during training")
    ap.add_argument("--save-every", type=int, default=None, help="Greedy eval cadence (episodes)")
    ap.add_argument("--outdir", type=str, default=None, help="Where to save checkpoints")
    ap.add_argument("--resume", type=str, default=None, help="Resume from .pt weights")
    ap.add_argument("--forever", action="store_true", help="Train until Ctrl-C")
    ap.add_argument("--verbose", action="store_true", help="Print per-episode logs")
    args = ap.parse_args()

    cfg = CONFIG.copy()
    if args.steps is not None:
        cfg["STEPS_PER_DAY"] = int(args.steps)

    train(
        cfg,
        episodes=args.episodes,
        verbose=args.verbose or True,
        save_checkpoints=True,
        eval_every=(args.save_every or cfg.get("TRAIN_EVAL_EVERY_EP", 10)),
        resume_path=args.resume,
        outdir=(args.outdir or cfg.get("TRAIN_SAVE_DIR", "models")),
        forever=args.forever,
    )
