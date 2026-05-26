# eval_dqn.py — simple greedy rollout helper
import torch
from collections import Counter
from dqn_env import TruckEnv
from dqn_agent import DQNAgent

def load_agents(cfg, path):
    env = TruckEnv(cfg)
    agent = DQNAgent(env.obs_dim, env.action_space.n, cfg)
    sd = torch.load(path, map_location="cpu")
    agent.q_net.load_state_dict(sd)
    agent.target_net.load_state_dict(sd)
    agent.eps = 0.0
    return env, agent

def _count_crashes_and_near_misses(events):
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

    near_ev = [e for e in events if e.get("type") == "near_miss"]
    if near_ev:
        near_inc = len(near_ev)
    else:
        pens = [e for e in events if e.get("type") == "near_miss_penalty"]
        by_t = Counter(int(round(float(e.get("t", 0)))) for e in pens)
        near_inc = sum(n // 2 for n in by_t.values())

    return int(crash_inc), int(near_inc)

def rollout_greedy(env, agent):
    obs_all = env.reset()
    total = [0.0]*env.n_agents
    done = [False]*env.n_agents
    last_info = {}
    while not all(done):
        acts = [agent.act_eval(obs_all[i]) for i in range(env.n_agents)]
        obs_all, r, done, info = env.step(acts)
        last_info = info
        for i in range(env.n_agents):
            total[i] += r[i]
    avg = sum(total)/len(total)
    crashes, near_misses = _count_crashes_and_near_misses(env.sim.events)
    stats = {"crashes": crashes, "near_misses": near_misses}
    return avg, env.sim, last_info, stats
