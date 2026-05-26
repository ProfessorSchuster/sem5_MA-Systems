# config.py — stable/safe preset + small arbitration helpers

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

    # Waypoints (Unity x,z -> sim x,y)  **UNCHANGED**
    "WAYPOINTS": [
        (0.0,0.0),(0.0,60.0),(0.0,120.0),(0.0,160.0),(0.0,220.0),
        (60.0,0.0),(60.0,60.0),(60.0,120.0),
        (120.0,0.0),(120.0,60.0),(120.0,120.0),(120.0,160.0),(120.0,220.0),
        (180.0,0.0),(180.0,60.0),(180.0,120.0),
        (240.0,0.0),(240.0,60.0),(240.0,120.0),(240.0,160.0),(240.0,220.0),
    ],

    # Roads (pairs of waypoint indices)  **UNCHANGED**
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
    "BIN_FILL_MULT": 1.0,
    "BIN_FILL_PROB": 0.75,
    "N_TRUCKS": 8,
    "TRUCK_CAPACITY": 300,

    # SAFER speed (down from 2.0)
    "TRUCK_SPEED_MPS": 1.4,

    "APPROACH_RADIUS_M": 1.2,
    "SPAWN_JITTER_M": 12.0,   # keep trucks well separated at t=0

    # Energy & costs
    "ENERGY_MAX": 100.0,
    "ENERGY_PER_M": 0.06,
    "ENERGY_RESERVE_M": 30.0,
    "WAGE_PER_HOUR": 25.0,
    "ENERGY_EUR_PER_UNIT": 0.30,
    "MAINT_EUR_PER_KM": 0.06,

    # Fees
    "OVERFLOW_PENALTY_EUR": 2500.0,   # up from 2000
    "OUTAGE_PENALTY_EUR": 1000.0,

    # Assignment look-ahead
    "URGENCY_HORIZON_S": 900,
    "OPPORTUNISTIC_FILL_FRAC": 0.55,

    # Service cooldown
    "SERVICE_COOLDOWN_S": 180.0,

    # Anti-churn windows
    "ROUTE_FREEZE_STEPS": 6,
    "ASSIGN_HOLD_STEPS": 10,
    "DEPOT_LOCK_STEPS": 8,
    "NEAR_FULL_FRAC": 0.90,

    # Coverage bias
    "COVERAGE_BIAS": 0.95,

    # Time
    "DT": 1.0,
    "STEPS_PER_DAY": 1200,

    # ---- DQN (shared policy) ----
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
    "CRASH_RADIUS_M": 0.6,             # smaller hitbox
    "NEAR_MISS_RADIUS_M": 1.2,
    "CRASH_PENALTY": 6000.0,
    "NEAR_MISS_PENALTY": 12.0,
    "YIELD_STEPS": 4,
    "ROW_RULE": "lower_id_wins",

    # ---- Hybrid manager knobs ----
    "HYBRID_MODE": True,
    "REASSIGN_MARGIN": 0.85,
    "ASSIGN_TTL_STEPS": 20,

    # DEFERRAL more eager to service (down from 0.35)
    "DEFER_FILL_FRAC": 0.18,

    "ALLOW_WAIT_WHEN_LOW_URGENCY": True,

    # ---- Two-lane traffic model ----
    "RIGHT_HAND_TRAFFIC": True,
    "LANE_OFFSET_M": 1.6,
    "LANE_LOOKAHEAD_M": 3.0,

    # ---- Proactive car-following (collision avoidance) ----
    # Wider cone + earlier braking for stability
    "SAFE_STOP_M": 4.5,                # up from 4.0/3.5
    "SAFE_SLOW_M": 9.5,                # up from 7.0
    "FORWARD_CONE_DEG": 32.0,          # up from 25.0

    "WARMUP_TICKS": 8,
    "DEPOT_UNBLOCK_RADIUS_M": 3.0,

    # Floor speed when allowed to move (slightly softer)
    "MIN_SPEED_SCALE": 0.30,

    # ---- Depot gate ----
    "DEPOT_GATE_RADIUS_M": 7.0,        # up from 6.0
    "DEPOT_MAX_INSIDE": 1,

    # ---- Standoff resolver / anti-stall (if you’re using the richer dqn_env) ----
    "STANDOFF_TICKS": 3,
    "STANDOFF_YIELD_STEPS": 6,
    "ROW_WINNER_SPEED": 0.9,
    "ROW_FACE_CONE_DEG": 35.0,
    "ROW_RANGE_MULT": 1.15,

    # Pull-aside maneuver
    "PULL_ASIDE_M": 2.2,
    "PULL_COOLDOWN": 18,
    "PULL_SIDE": "right",

    # Near-miss sanity
    "NEAR_MISS_COOLDOWN_STEPS": 10,
    "NEAR_MISS_APPROACH_EPS": 0.08,

    # Anti-idle wake-up
    "ANTI_IDLE_WAKE_TICKS": 12,

    # (Optional) Proactive TTC yield — harmless if your env ignores it
    "PROACTIVE_YIELD": True,
    "TTC_YIELD_S": 1.6,
    "CRASH_BUFFER_M": 0.35,

    # ---- Training defaults ----
    "TRAIN_EVAL_EVERY_EP": 10,
    "TRAIN_SAVE_DIR": "models",
}
