# config.py — unified config with crash + hybrid + lanes knobs (cost-first)

CONFIG = {
    # World (graph)
    "MAP_SIZE": (247.0, 232.2),
    "SEED": 42,

    # Road layout mode
    "ROAD_LAYOUT": "manual",  # 'manual' | 'grid'
    "GRID_ROWS": 6,
    "GRID_COLS": 6,
    "GRID_SPACING": 20.0,
    "GRID_MARGIN": 10.0,

    # Waypoints (Unity x,z -> sim x,y)
    "WAYPOINTS": [
        (0.0,0.0),(0.0,60.0),(0.0,120.0),(0.0,160.0),(0.0,220.0),
        (60.0,0.0),(60.0,60.0),(60.0,120.0),
        (120.0,0.0),(120.0,60.0),(120.0,120.0),(120.0,160.0),(120.0,220.0),
        (180.0,0.0),(180.0,60.0),(180.0,120.0),
        (240.0,0.0),(240.0,60.0),(240.0,120.0),(240.0,160.0),(240.0,220.0),

    ],

    # Roads (pairs of waypoint indices)
    "ROADS": [
        [0,1],[1,2],[2,3],[3,4],
        [5,6],[6,7],
        [8,9],[9,10],[10,11],[11,12],
        [13,14],[14,15],
        [16,17],[17,18],[18,19],[19,20],
        [0,5],[5,8],[8,13],[13,16],
        [1,6],[6,9],[9,14],[14,17],
        [2,7],[7,10],[10,15],[15,18],
        [3,11],[11,19]
    ],

    "DEPOT": (120.324, 124.700),
    "SIDEWALK_OFFSET_M": 2.0,
    "ROAD_HALF_WIDTH": 4.0,

    # Bins / trucks
    "N_BINS": 12,
    "BIN_CAPACITY": 100,
    "BIN_FILL_PER_STEP": (0, 1),
    "N_TRUCKS": 8,
    "TRUCK_CAPACITY": 300,
    "TRUCK_SPEED_MPS": 2.0,
    "APPROACH_RADIUS_M": 1.2,
    "SPAWN_JITTER_M": 12.0,   # keep trucks well separated at t=0

    # Energy & costs (the single objective is to minimize € total)
    "ENERGY_MAX": 100.0,
    "ENERGY_PER_M": 0.06,
    "ENERGY_RESERVE_M": 30.0,
    "WAGE_PER_HOUR": 25.0,
    "ENERGY_EUR_PER_UNIT": 0.30,
    "MAINT_EUR_PER_KM": 0.06,

    # Fees (how other goals enter the objective)
    "OVERFLOW_PENALTY_EUR": 2000.0,
    "OUTAGE_PENALTY_EUR": 1000.0,

    # Assignment “look-ahead” feel (still useful for manager heuristics)
    "URGENCY_HORIZON_S": 600,
    "OPPORTUNISTIC_FILL_FRAC": 0.55,

    # Anti-churn windows
    "ROUTE_FREEZE_STEPS": 6,
    "ASSIGN_HOLD_STEPS": 10,
    "DEPOT_LOCK_STEPS": 8,
    "NEAR_FULL_FRAC": 0.90,

    # Coverage bias (anti-bunching)
    "COVERAGE_BIAS": 0.95,

    # Time
    "DT": 1.0,
    "STEPS_PER_DAY": 1200,

    # ---- DQN (shared policy) ----
    # (No reward shaping — env uses pure -Δ€ team reward)
    "MAX_PENALTIES_PER_TICK": None,

    "HIDDEN": 256,
    "LR": 5e-4,
    "GAMMA": 0.99,
    "EPS_START": 0.8,
    "EPS_END": 0.05,
    "EPS_DECAY": 0.997,
    "BUFFER_SIZE": 100_000,
    "BATCH_SIZE": 128,
    "TAU": 0.005,

    "DQN_WEIGHTS_DIR": "dqn_weights",

    # ---- Turn behaviour (routing) ----
    "UTURN_PENALTY": 200.0,
    "FORBID_UTURN_IF_ALTERNATIVE": True,

    # Grid planner (optional)
    "GRID_SIZE": 150,
    "STREETS_MASK_PNG": None,
    "STREETS_MASK_THRESHOLD": 0.5,
    "STREETS_MASK_INVERT_Y": False,
    "DILATE_PASSES": 2,

    # ---- Safety / crash model ----
    "CRASH_RADIUS_M": 0.6,         # smaller hitbox
    "NEAR_MISS_RADIUS_M": 1.2,
    "CRASH_PENALTY": 4000.0,       # € fee counted in the objective
    "NEAR_MISS_PENALTY": 8.0,      # € fee counted in the objective
    "YIELD_STEPS": 3,
    "ROW_RULE": "lower_id_wins",

    # ---- Hybrid manager knobs (still hybrid with DQN) ----
    "HYBRID_MODE": True,
    "REASSIGN_MARGIN": 0.85,
    "ASSIGN_TTL_STEPS": 20,
    "DEFER_FILL_FRAC": 0.35,

    # Allow purposeful idle when low urgency (kept; DQN will decide in € terms)
    "ALLOW_WAIT_WHEN_LOW_URGENCY": True,

    # ---- Two-lane traffic model ----
    "RIGHT_HAND_TRAFFIC": True,
    "LANE_OFFSET_M": 1.6,
    "LANE_LOOKAHEAD_M": 3.0,

    # ---- Proactive car-following (collision avoidance) ----
    "SAFE_STOP_M": 3.5,
    "SAFE_SLOW_M": 7.0,
    "FORWARD_CONE_DEG": 35.0,

    "WARMUP_TICKS": 10,
    # Fixed key name to match env: (was DEPOT_UNBLOCKING_RADIUS_M)
    "DEPOT_UNBLOCK_RADIUS_M": 3.0,
    "MIN_SPEED_SCALE": 0.35
}
