# src/compute_cpts.py  — v2 JSON-safe
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, List, Any
import pandas as pd
import numpy as np

RAW = Path("C:/Users/pyppr/Create3_BayesianFusion/data/raw/ir_pid_bumper_log.csv")
OUT_CSV = Path("C:/Users/pyppr/Create3_BayesianFusion/data/processed/cpt_tables.csv")
OUT_JSON = Path("C:/Users/pyppr/Create3_BayesianFusion/data/processed/cpt_tables.json")

# -----------------------
# Config (tweak as needed)
# -----------------------
CFG = {
    "alpha": 1.0,                       # Laplace smoothing
    "trend_deadband": 0.005,            # meters; below this is 'steady'
    "gap": {"n": 4, "k": 3, "min_delta": 0.01},  # windowed sharp increase
    "weak_door": {"n": 8, "need_depart": 4, "needs_gap": True},
    # If None, thresholds learned from data tertiles; else [near_ok, ok_far] in meters
    "wall_dist_thresholds_m": None
}

# -----------------------
# Helpers
# -----------------------
def laplace_counts_to_probs(counts: Dict[Tuple, int], values: List[str], alpha: float):
    """Return smoothed probabilities P(X=v | parents) for every parent assignment seen."""
    parent_to_val = {}
    for (parents, val), c in counts.items():
        parent_to_val.setdefault(parents, {v: 0 for v in values})
        parent_to_val[parents][val] = parent_to_val[parents].get(val, 0) + c

    parent_to_prob = {}
    for parents, vc in parent_to_val.items():
        total = sum(vc.values())
        k = len(values)
        probs = {v: (vc.get(v, 0) + alpha) / (total + alpha * k) for v in values}
        s = sum(probs.values())
        probs = {v: p / s for v, p in probs.items()}
        parent_to_prob[parents] = probs
    return parent_to_prob

def discretize_wall_dist(avg_m: float, thr: List[float]) -> str:
    if avg_m < thr[0]: return "near"
    if avg_m < thr[1]: return "ok"
    return "far"

def discretize_trend(delta_m: float, deadband: float) -> str:
    if delta_m < -deadband: return "approach"
    if delta_m >  deadband: return "depart"
    return "steady"

def gap_detect(deltas: List[float], n: int, k: int, min_delta: float) -> bool:
    recent = deltas[-n:] if len(deltas) >= n else deltas
    hits = sum(1 for d in recent if d > min_delta)
    return hits >= k

def weak_label_door(trend_hist: List[str], gap_hist: List[bool], n: int, need_depart: int, needs_gap: bool) -> bool:
    recent_t = trend_hist[-n:] if len(trend_hist) >= n else trend_hist
    recent_g = gap_hist[-n:] if len(gap_hist) >= n else gap_hist
    cond1 = recent_t.count("depart") >= need_depart
    cond2 = any(recent_g) if needs_gap else True
    return cond1 and cond2

# -----------------------
# Load & feature engineer
# -----------------------
def load_and_engineer() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    # Expect columns: Time, IR_Left, IR_Right, PID_P, PID_I, PID_D, Bumper, Location
    for c in ["IR_Left","IR_Right","PID_P","PID_I","PID_D","Time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Bumper" in df.columns:
        df["Bumper"] = df["Bumper"].astype(bool)
    else:
        df["Bumper"] = False

    # distance proxy (meters) — if IR is cm, /100
    df["dist_m"] = (df["IR_Left"] + df["IR_Right"]) / 2.0 / 100.0

    # thresholds for near/ok/far
    if CFG["wall_dist_thresholds_m"] is None:
        q1, q2 = df["dist_m"].quantile([0.33, 0.66]).tolist()
        thr = [max(0.02, q1), max(q1 + 1e-3, q2)]
    else:
        thr = CFG["wall_dist_thresholds_m"]
    df["WallDist"] = [discretize_wall_dist(x, thr) for x in df["dist_m"]]

    # trend & gap
    df["delta_dist"] = df["dist_m"].diff().fillna(0.0)
    df["Trend"] = [discretize_trend(d, CFG["trend_deadband"]) for d in df["delta_dist"]]

    deltas = df["delta_dist"].tolist()
    gaps = []
    trend_hist, gap_hist = [], []
    for i, _ in enumerate(deltas):
        trend_hist.append(df["Trend"].iloc[i])
        gaps.append(gap_detect(deltas[: i + 1], CFG["gap"]["n"], CFG["gap"]["k"], CFG["gap"]["min_delta"]))
        gap_hist.append(gaps[-1])
    df["GapDetected"] = gaps

    # weak door label
    door_flags = []
    trend_hist = []
    gap_hist = []
    for i in range(len(df)):
        trend_hist.append(df["Trend"].iloc[i])
        gap_hist.append(df["GapDetected"].iloc[i])
        pattern = weak_label_door(trend_hist, gap_hist,
                                  CFG["weak_door"]["n"],
                                  CFG["weak_door"]["need_depart"],
                                  CFG["weak_door"]["needs_gap"])
        loc_hint = False
        if "Location" in df.columns:
            val = df["Location"].iloc[i]
            loc_hint = isinstance(val, str) and val.lower() == "door_start"
        door_flags.append(bool(pattern or loc_hint))
    df["DoorPassed"] = door_flags

    return df[["Time","WallDist","Trend","GapDetected","Bumper","DoorPassed"]].dropna().reset_index(drop=True)

# -----------------------
# Learn CPTs
# -----------------------
def learn_cpts(df: pd.DataFrame) -> Dict[str, Any]:
    alpha = CFG["alpha"]

    # Domains
    dom_W = ["near","ok","far"]
    dom_T = ["approach","steady","depart"]
    dom_G = [False, True]
    dom_B = [False, True]
    dom_D = [False, True]

    # P(WallDist)
    counts_W = { ((), w): 0 for w in dom_W }
    for w in df["WallDist"]:
        counts_W[((), w)] += 1
    P_W = laplace_counts_to_probs(counts_W, dom_W, alpha)

    # P(Bumper | WallDist)
    counts_B = {}
    for _, row in df.iterrows():
        parents = (row["WallDist"],)
        key = (parents, bool(row["Bumper"]))
        counts_B[key] = counts_B.get(key, 0) + 1
    P_B_given_W = laplace_counts_to_probs(counts_B, dom_B, alpha)

    # P(DoorPassed | WallDist, Trend, GapDetected)
    counts_D = {}
    for _, row in df.iterrows():
        parents = (row["WallDist"], row["Trend"], bool(row["GapDetected"]))
        key = (parents, bool(row["DoorPassed"]))
        counts_D[key] = counts_D.get(key, 0) + 1
    P_D_given_WTG = laplace_counts_to_probs(counts_D, dom_D, alpha)

    # Optional helpers
    counts_T = {}
    for _, row in df.iterrows():
        parents = (row["WallDist"],)
        key = (parents, row["Trend"])
        counts_T[key] = counts_T.get(key, 0) + 1
    P_T_given_W = laplace_counts_to_probs(counts_T, dom_T, alpha)

    counts_G = {}
    for _, row in df.iterrows():
        parents = (row["Trend"],)
        key = (parents, bool(row["GapDetected"]))
        counts_G[key] = counts_G.get(key, 0) + 1
    P_G_given_T = laplace_counts_to_probs(counts_G, dom_G, alpha)

    # Keep parent name order for JSON materialization
    return {
        "domains": {
            "WallDist": dom_W, "Trend": dom_T, "GapDetected": dom_G,
            "Bumper": dom_B, "DoorPassed": dom_D
        },
        "tables": {
            "P(WallDist)": {"parents": [], "var": "WallDist", "data": P_W},
            "P(Bumper|WallDist)": {"parents": ["WallDist"], "var": "Bumper", "data": P_B_given_W},
            "P(DoorPassed|WallDist,Trend,GapDetected)": {
                "parents": ["WallDist","Trend","GapDetected"], "var": "DoorPassed", "data": P_D_given_WTG
            },
            "P(Trend|WallDist)": {"parents": ["WallDist"], "var": "Trend", "data": P_T_given_W},
            "P(GapDetected|Trend)": {"parents": ["Trend"], "var": "GapDetected", "data": P_G_given_T},
        }
    }

# -----------------------
# Save as CSV & JSON
# -----------------------
def flatten_for_csv(cpts: Dict[str, Any]) -> pd.DataFrame:
    rows = []

    def emit(table_name: str, parent_names: List[str], var_name: str, parent_to_prob: Dict[Tuple, Dict[Any, float]]):
        for parents, probs in parent_to_prob.items():
            given = {}
            # parents is a tuple aligned with parent_names
            for i, p in enumerate(parents):
                if parent_names:
                    given[parent_names[i]] = p
            for val, pr in probs.items():
                rows.append({
                    "table": table_name,
                    "given": json.dumps(given),
                    "variable": var_name,
                    "value": val,
                    "probability": round(float(pr), 6)
                })

    T = cpts["tables"]
    emit("P(WallDist)", [], "WallDist", T["P(WallDist)"]["data"])
    emit("P(Bumper|WallDist)", ["WallDist"], "Bumper", T["P(Bumper|WallDist)"]["data"])
    emit("P(DoorPassed|WallDist,Trend,GapDetected)", ["WallDist","Trend","GapDetected"],
         "DoorPassed", T["P(DoorPassed|WallDist,Trend,GapDetected)"]["data"])
    emit("P(Trend|WallDist)", ["WallDist"], "Trend", T["P(Trend|WallDist)"]["data"])
    emit("P(GapDetected|Trend)", ["Trend"], "GapDetected", T["P(GapDetected|Trend)"]["data"])

    return pd.DataFrame(rows)

def materialize_json_friendly(cpts: Dict[str, Any]) -> Dict[str, Any]:
    """Turn tuple-keyed tables into JSON-serializable lists."""
    out = {"domains": cpts["domains"], "tables": {}}
    for name, meta in cpts["tables"].items():
        parents = meta["parents"]
        var = meta["var"]
        data = meta["data"]
        entries = []
        for parent_tuple, probs in data.items():
            given = {}
            for i, p in enumerate(parent_tuple):
                if parents:
                    given[parents[i]] = p
            entries.append({"given": given, "probs": {str(k): float(v) for k, v in probs.items()}})
        out["tables"][name] = {"parents": parents, "var": var, "entries": entries}
    return out

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = load_and_engineer()
    cpts = learn_cpts(df)

    # CSV (flat)
    flat = flatten_for_csv(cpts)
    flat.to_csv(OUT_CSV, index=False)

    # JSON (tuple-free)
    jsonable = materialize_json_friendly(cpts)
    with open(OUT_JSON, "w") as f:
        json.dump(jsonable, f, indent=2)

    print(f"✅ Learned CPTs written to:\n  - {OUT_CSV}\n  - {OUT_JSON}")

if __name__ == "__main__":
    main()
