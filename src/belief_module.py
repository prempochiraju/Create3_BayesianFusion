"""
belief_module.py
----------------
- Loads CPTs learned by compute_cpts.py (JSON at data/processed/cpt_tables.json)
- Provides:
    belief(sensor_readings, config) -> belief_map
    batch_belief_from_csv(raw_csv_path, ...) -> pd.DataFrame with beliefs vs time

CPT JSON format expected (from compute_cpts.py v2):
{
  "domains": { ... },
  "tables": {
    "P(WallDist)": {"parents": [], "var": "WallDist",
                    "entries":[{"given":{}, "probs":{"near":..,"ok":..,"far":..}}, ...]},
    "P(Bumper|WallDist)": {"parents":["WallDist"], "var":"Bumper", "entries":[ ... ]},
    "P(DoorPassed|WallDist,Trend,GapDetected)": {...},
    "P(Trend|WallDist)": {...},
    "P(GapDetected|Trend)": {...}
  }
}
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import pandas as pd
import numpy as np
import os

# -------------------------------------------------------------------
# User defaults (adjust paths if needed)
# -------------------------------------------------------------------
DEFAULT_CPT_JSON = "C:/Users/pyppr/Create3_BayesianFusion/data/processed/cpt_tables.json"

# -------------------------------------------------------------------
# Config container
# -------------------------------------------------------------------
@dataclass
class BeliefConfig:
    cpt_json: str = DEFAULT_CPT_JSON
    # Discretization (mirror compute_cpts.py)
    trend_deadband: float = 0.005  # meters
    # If thresholds None, we fall back to [0.06, 0.20] m
    wall_dist_thresholds_m: Optional[List[float]] = None
    # Gap detector (same semantics as in learner)
    gap_n: int = 4
    gap_k: int = 3
    gap_min_delta: float = 0.01
    # IR scaling to meters; if IR is in cm, /100.0 (your logger used this)
    ir_to_m_factor: float = 1.0 / 100.0

# -------------------------------------------------------------------
# Utilities to materialize CPT lookups
# -------------------------------------------------------------------
def _load_cpts(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        return json.load(f)

def _entries_to_lookup(entries: List[Dict[str, Any]]) -> Dict[Tuple, Dict[str, float]]:
    """
    Turn [{"given": {...}, "probs": {...}}, ...] into {(parent_tuple): {val: p, ...}, ...}
    Parent tuple key is sorted by parent name to be deterministic.
    """
    table: Dict[Tuple, Dict[str, float]] = {}
    for row in entries:
        given = row.get("given", {})
        key = tuple((k, given[k]) for k in sorted(given.keys()))
        # Ensure all probs are float-serializable
        probs = {str(k): float(v) for k, v in row.get("probs", {}).items()}
        table[key] = probs
    return table

def _table(cpts: Dict[str, Any], name: str):
    t = cpts["tables"][name]
    parents = t["parents"]
    var = t["var"]
    lookup = _entries_to_lookup(t["entries"])
    return parents, var, lookup

def _query_lookup(lookup: Dict[Tuple, Dict[str, float]], assignment: Dict[str, Any]) -> Dict[str, float]:
    """
    Given a lookup keyed by sorted parent tuples, fetch the probability row
    for the provided parent assignment. If not found, back off by removing
    parents from the right (simple smoothing fallback).
    """
    if not lookup:
        return {}
    keys_sorted = sorted(assignment.keys())
    # try exact match
    parent_tuple = tuple((k, assignment[k]) for k in keys_sorted)
    if parent_tuple in lookup:
        return lookup[parent_tuple]
    # Backoff: iteratively drop last key
    for drop in range(1, len(keys_sorted) + 1):
        reduced = tuple((k, assignment[k]) for k in keys_sorted[:-drop])
        if reduced in lookup:
            return lookup[reduced]
    # Last resort: try empty tuple (prior-like row)
    return lookup.get((), {})

# -------------------------------------------------------------------
# Discretization helpers (mirror compute_cpts.py)
# -------------------------------------------------------------------
def _discretize_wall_dist(avg_m: float, thr: List[float]) -> str:
    if avg_m < thr[0]: return "near"
    if avg_m < thr[1]: return "ok"
    return "far"

def _discretize_trend(delta_m: float, deadband: float) -> str:
    if delta_m < -deadband: return "approach"
    if delta_m >  deadband: return "depart"
    return "steady"

def _gap_detect(deltas: List[float], n: int, k: int, min_delta: float) -> bool:
    recent = deltas[-n:] if len(deltas) >= n else deltas
    hits = sum(1 for d in recent if d > min_delta)
    return hits >= k

# -------------------------------------------------------------------
# Core belief engine (evidence-conditioned P(W|T,G))
# -------------------------------------------------------------------
class BeliefEngine:
    def __init__(self, cfg: BeliefConfig):
        self.cfg = cfg
        self.cpts = _load_cpts(cfg.cpt_json)
        self.domains = self.cpts["domains"]

        # Materialize lookups
        self.parents_W,  self.var_W,  self.P_W               = _table(self.cpts, "P(WallDist)")
        self.parents_BW, self.var_BW, self.P_B_given_W       = _table(self.cpts, "P(Bumper|WallDist)")
        self.parents_D,  self.var_D,  self.P_D_given_WTG     = _table(self.cpts, "P(DoorPassed|WallDist,Trend,GapDetected)")
        self.parents_TW, self.var_TW, self.P_T_given_W       = _table(self.cpts, "P(Trend|WallDist)")
        self.parents_GT, self.var_GT, self.P_G_given_T       = _table(self.cpts, "P(GapDetected|Trend)")

        # Streaming state for trend/gap
        self.prev_dist_m: Optional[float] = None
        self.delta_hist: List[float] = []

        # Thresholds: fallback if not provided
        self.thr = self.cfg.wall_dist_thresholds_m or [0.06, 0.20]

        # Prior over W (from P(WallDist))
        self.prior_W = next(iter(self.P_W.values())) if self.P_W else {"near": 1/3, "ok": 1/3, "far": 1/3}

    # --------------- feature prep ---------------
    def _ensure_features(self, reading: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts either:
          - raw: {"IR_Left":..., "IR_Right":..., "Bumper":...}
          - engineered: {"dist_m":..., "Trend":..., "GapDetected":..., "Bumper":...}
        Returns a dict with {"WallDist","Trend","GapDetected","Bumper","dist_m"}.
        """
        r = dict(reading)

        # distance (meters)
        if "dist_m" not in r:
            if "IR_Left" in r and "IR_Right" in r:
                r["dist_m"] = (float(r["IR_Left"]) + float(r["IR_Right"])) / 2.0 * self.cfg.ir_to_m_factor
            else:
                raise ValueError("Need either dist_m or both IR_Left and IR_Right")

        # trend from delta if not provided
        if "Trend" not in r:
            delta = 0.0 if self.prev_dist_m is None else (r["dist_m"] - self.prev_dist_m)
            r["delta_dist"] = delta
            r["Trend"] = _discretize_trend(delta, self.cfg.trend_deadband)
        self.prev_dist_m = r["dist_m"]

        # gap from history if not provided
        if "GapDetected" not in r:
            self.delta_hist.append(r.get("delta_dist", 0.0))
            r["GapDetected"] = _gap_detect(self.delta_hist, self.cfg.gap_n, self.cfg.gap_k, self.cfg.gap_min_delta)

        # bumper default
        if "Bumper" not in r:
            r["Bumper"] = False

        # discretized wall-distance label (not used directly in inference but handy to expose)
        if "WallDist" not in r:
            r["WallDist"] = _discretize_wall_dist(r["dist_m"], self.thr)

        return r

    # --------------- posterior over W ---------------
    def _posterior_W_given_TG(self, T: str, G: bool) -> Dict[str, float]:
        """
        P(W | T,G) ∝ P(W) * P(T|W) * P(G|T)
        Normalize at the end. This sharpens the wall-distance belief using evidence.
        """
        # P(G|T)
        p_g_given_t_row = _query_lookup(self.P_G_given_T, {"Trend": T})
        # Keys are strings ("True"/"False"); ensure robust access
        p_g = float(p_g_given_t_row.get(str(G), p_g_given_t_row.get(G, 1.0))) if p_g_given_t_row else 1.0

        # For each W: prior * P(T|W) * P(G|T)
        unnorm: Dict[str, float] = {}
        for w_val, p_w in self.prior_W.items():
            p_t_given_w_row = _query_lookup(self.P_T_given_W, {"WallDist": w_val})
            p_t_given_w = float(p_t_given_w_row.get(T, 0.0)) if p_t_given_w_row else 0.0
            unnorm[w_val] = float(p_w) * p_t_given_w * p_g

        total = sum(unnorm.values())
        if total <= 0:
            # fallback to prior if degenerate
            return {k: float(v) for k, v in self.prior_W.items()}
        return {k: v / total for k, v in unnorm.items()}

    # --------------- public belief ---------------
    def belief(self, sensor_readings: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute a belief map:
            - features: engineered inputs (dist_m, Trend, GapDetected, etc.)
            - P(WallDist): posterior P(W|T,G)
            - P(DoorPassed): ∑_W P(Door|W,T,G) P(W|T,G)
        """
        r = self._ensure_features(sensor_readings)
        T = str(r["Trend"])
        G = bool(r["GapDetected"])

        # evidence-conditioned posterior over W
        post_W = self._posterior_W_given_TG(T, G)

        # Mix into P(Door)
        p_door = 0.0
        detail_rows = []
        for w_val, p_w in post_W.items():
            cond = _query_lookup(self.P_D_given_WTG, {"GapDetected": G, "Trend": T, "WallDist": w_val})
            # keys in table are strings "True"/"False"
            p_true = float(cond.get("True", cond.get(True, 0.0))) if cond else 0.0
            contrib = p_true * p_w
            p_door += contrib
            detail_rows.append({
                "W": w_val, "P(W|T,G)": float(p_w),
                "P(Door|W,T,G)": float(p_true), "contrib": float(contrib)
            })

        return {
            "features": {
                "dist_m": float(r["dist_m"]),
                "Trend": T,
                "GapDetected": G,
                "Bumper": bool(r["Bumper"]),
                "WallDist_disc": str(r["WallDist"]),
            },
            "P(WallDist)": {k: float(v) for k, v in post_W.items()},
            "P(DoorPassed)": float(p_door),
            "explain": {"mixing": detail_rows}
        }

# -------------------------------------------------------------------
# Batch helper for timelines
# -------------------------------------------------------------------
def batch_belief_from_csv(
    raw_csv_path: str,
    cfg: Optional[BeliefConfig] = None,
    save_csv_to: Optional[str] = None
) -> pd.DataFrame:
    """
    Reads your raw logger CSV:
      Time, IR_Left, IR_Right, PID_P, PID_I, PID_D, Bumper, Location
    Returns a DataFrame with:
      [Time, dist_m, Trend, GapDetected, P_Door, P_W_near, P_W_ok, P_W_far]
    """
    cfg = cfg or BeliefConfig()
    eng = BeliefEngine(cfg)

    df = pd.read_csv(raw_csv_path)
    # Normalize types
    for c in ["IR_Left", "IR_Right", "Time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Bumper" in df.columns:
        df["Bumper"] = df["Bumper"].astype(bool)
    else:
        df["Bumper"] = False

    times = df["Time"].tolist()
    out_rows = []
    for i, row in df.iterrows():
        reading = {
            "IR_Left": row.get("IR_Left", np.nan),
            "IR_Right": row.get("IR_Right", np.nan),
            "Bumper": bool(row.get("Bumper", False)),
        }
        b = eng.belief(reading)
        out_rows.append({
            "Time": float(times[i]) if times[i] == times[i] else float(i),  # safe if NaN
            "dist_m": b["features"]["dist_m"],
            "Trend": b["features"]["Trend"],
            "GapDetected": b["features"]["GapDetected"],
            "P_Door": b["P(DoorPassed)"],
            "P_W_near": b["P(WallDist)"].get("near", np.nan),
            "P_W_ok":   b["P(WallDist)"].get("ok",   np.nan),
            "P_W_far":  b["P(WallDist)"].get("far",  np.nan),
        })

    out = pd.DataFrame(out_rows)
    if save_csv_to:
        Path(os.path.dirname(save_csv_to)).mkdir(parents=True, exist_ok=True)
        out.to_csv(save_csv_to, index=False)
    return out

# -------------------------------------------------------------------
# Public API (required signature)
# -------------------------------------------------------------------
def belief(sensor_readings: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Public API: def belief(sensor_readings, config) -> belief_map
    sensor_readings: dict with either IR inputs or engineered features.
    config: optional dict overriding BeliefConfig fields (keys match dataclass).
    """
    cfg = BeliefConfig(**(config or {}))
    eng = BeliefEngine(cfg)
    return eng.belief(sensor_readings)
