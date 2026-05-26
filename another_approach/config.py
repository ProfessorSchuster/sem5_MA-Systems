# config.py
# Global parameters for simulation, training, and optimization

CONFIG = {
    # ---------- run modes ----------
    "RUN_SIM": True,
    "RUN_VIZ": True,
    "RUN_EXPORT": True,
    "RUN_DQN_TRAIN": False,
    "RUN_HYPEROPT": False,

    # ---------- world (grid road graph) ----------
    "WAYPOINTS": [
        (0.0,0.0),(0.0,60.0),(0.0,120.0),(0.0,160.0),(0.0,220.0),
        (60.0,0.0),(60.0,60.0),(60.0,120.0),
        (120.0,0.0),(120.0,60.0),(120.0,120.0),(120.0,160.0),(120.0,220.0),
        (180.0,0.0),(180.0,60.0),(180.0,120.0),
        (240.0,0.0),(240.0,60.0),(240.0,120.0),(240.0,160.0),(240.0,220.0),
    ],
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
    "MAP_SIZE": (240.0, 220.0),
    "SEED": 42,
    "DEPOT": (120.0, 120.0),
    "SIDEWALK_OFFSET_M": 2.0,

    # ---------- bins ----------
    "N_BINS": 12,             # MVP: a few more bins, but we control inflow in the runner
    "BIN_CAPACITY": 100,
    "BIN_FILL_PER_STEP": (0, 1),

    # ---------- trucks & traffic ----------
    "N_TRUCKS": 4,            # MVP: single truck
    "TRUCK_CAPACITY": 300,
    "TRUCK_SPEED_MPS": 8.0,   # brisk so it reaches the far bin and returns comfortably
    "TRUCK_ACC_MPS2": 2.5,
    "TRUCK_DEC_MPS2": 4.0,
    "TRUCK_RADIUS_M": 1.5,
    "SAFE_GAP_M": 3.0,

    # two-lane + controller
    "LANE_OFFSET_M": 3.0,
    "LANE_BLEND_M": 16.0,
    "LOOKAHEAD_M": 3.0,
    "INTERSECTION_APPROACH_M": 6.0,
    "INTERSECTION_CLEAR_M": 9.0,
    "NO_LANECHANGE_NEAR_NODE_M": 40.0,
    "LANE_CHANGE_COOLDOWN_STEPS": 12,
    "JUNCTION_SINGLE_LANE": True,
    "ASSIGN_FREEZE_IN_JUNCTION": True, # with gating off, no freeze
    "JUNCTION_STALL_STEPS": 12,

    "ENERGY_MAX": 100.0,
    "ENERGY_PER_M": 0.06,
    "ENERGY_RESERVE_M": 30.0,
    "APPROACH_RADIUS_M": 3.0,
    "SERVICE_RADIUS_M": 12.0,   # slightly generous so docking actually happens
    "DOCK_SWITCH_M": 16.0,     # steer to the bin earlier when close

    # ---------- costs (€) ----------
    "WAGE_PER_HOUR": 25.0,
    "ENERGY_EUR_PER_UNIT": 0.30,
    "MAINT_EUR_PER_KM": 0.06,
    "OVERFLOW_PENALTY_EUR": 200.0,
    "OUTAGE_PENALTY_EUR": 1000.0,
    "CRASH_PENALTY_EUR": 5000.0,
    "CRASH_LOCK_STEPS": 25,

    # ---------- negotiation / dispatch ----------
    "URGENCY_HORIZON_S": 120,
    "OPPORTUNISTIC_FILL_FRAC": 0.60,

    # ---------- RL ----------
    "DT": 1.0,
    "STEPS_PER_DAY": 1000,     # enough time to reach far bin and return
    "GAMMA": 0.9609734014656335,
    "LR": 0.0005084369486447114,
    "EPS_START": 1.0,
    "EPS_END": 0.03711065334643615,
    "EPS_DECAY": 0.9917886334001346,
    "BUFFER_SIZE": 50000,
    "BATCH_SIZE": 128,
    "TARGET_UPDATE": 100,
    "REWARD_SCALE": 0.01,
    "MAX_PENALTIES_PER_TICK": 8,

    # ---------- anti-churn ----------
    "ROUTE_FREEZE_STEPS": 6,
    "ASSIGN_HOLD_STEPS": 8,
    "DEPOT_LOCK_STEPS": 8,
    "NEAR_FULL_FRAC": 0.90,

    # ---------- export ----------
    "JSON_EXPORT_PATH": "sim_day.json"
}
