# src/sample_run.py  — annotated timeline
from pathlib import Path
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from belief_module import batch_belief_from_csv, BeliefConfig

RAW_CSV   = "C:/Users/pyppr/Create3_BayesianFusion/data/raw/ir_pid_bumper_log.csv"
BELIEF_CSV= "C:/Users/pyppr/Create3_BayesianFusion/results/plots/belief_timeline.csv"
PLOT_PNG  = "C:/Users/pyppr/Create3_BayesianFusion/results/plots/belief_timeline_annotated.png"
PLOT_HTML = "C:/Users/pyppr/Create3_BayesianFusion/results/plots/belief_timeline_annotated.html"
CPT_JSON  = "C:/Users/pyppr/Create3_BayesianFusion/data/processed/cpt_tables.json"

PEAK_MIN = 0.55  # only mark door peaks above this prob
NEIGHBOR = 2     # local-maximum window (±2 samples)

def find_local_peaks(y, min_val=0.5, neighbor=2):
    peaks = []
    n = len(y)
    for i in range(neighbor, n - neighbor):
        window = y[i - neighbor: i + neighbor + 1]
        if y[i] == max(window) and y[i] >= min_val:
            peaks.append(i)
    return peaks

def main():
    os.makedirs(os.path.dirname(PLOT_PNG), exist_ok=True)

    # compute beliefs timeline
    cfg = BeliefConfig(cpt_json=CPT_JSON)
    df = batch_belief_from_csv(RAW_CSV, cfg=cfg, save_csv_to=BELIEF_CSV)

    # load raw for bumper overlay
    raw = pd.read_csv(RAW_CSV)
    raw["Time"] = pd.to_numeric(raw["Time"], errors="coerce")
    bumper_times = raw.loc[raw.get("Bumper", False) == True, "Time"].dropna().tolist()

    # find peaks
    peaks_idx = find_local_peaks(df["P_Door"].values, min_val=PEAK_MIN, neighbor=NEIGHBOR)
    peak_times = df["Time"].iloc[peaks_idx].tolist()
    peak_vals  = df["P_Door"].iloc[peaks_idx].tolist()

    # ---- Matplotlib annotated figure ----
    plt.figure(figsize=(11, 4))
    plt.plot(df["Time"], df["P_Door"], label="P(DoorPassed)", linewidth=1.8)
    plt.plot(df["Time"], df["P_W_near"], label="P(W=near)", linestyle="--", alpha=0.7)
    plt.plot(df["Time"], df["P_W_ok"],   label="P(W=ok)",   linestyle="--", alpha=0.7)
    plt.plot(df["Time"], df["P_W_far"],  label="P(W=far)",  linestyle="--", alpha=0.7)

    # mark peaks
    if peak_times:
        plt.scatter(peak_times, peak_vals, marker="o", s=40, label="Door peaks")

    # vertical lines for bumpers
    for t in bumper_times:
        plt.axvline(t, color="red", alpha=0.25, linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title("Door-passed belief (evidence-conditioned) with peaks & bumper hits")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=160)
    print(f"✅ Saved annotated plot: {PLOT_PNG}")
    print(f"✅ Saved belief CSV: {BELIEF_CSV}")

    # ---- Optional Plotly annotated HTML ----
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Time"], y=df["P_Door"], name="P(DoorPassed)"))
        fig.add_trace(go.Scatter(x=df["Time"], y=df["P_W_near"], name="P(W=near)", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=df["Time"], y=df["P_W_ok"],   name="P(W=ok)",   line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=df["Time"], y=df["P_W_far"],  name="P(W=far)",  line=dict(dash="dash")))

        # peaks
        if peak_times:
            fig.add_trace(go.Scatter(
                x=peak_times, y=peak_vals, mode="markers", name="Door peaks"
            ))

        # bumper vertical shapes
        for t in bumper_times:
            fig.add_vline(x=t, line_color="red", opacity=0.25)

        fig.update_layout(
            title="Door-passed belief (evidence-conditioned)",
            xaxis_title="Time (s)", yaxis_title="Probability",
            template="plotly_white", width=980, height=420
        )
        fig.write_html(PLOT_HTML, include_plotlyjs="cdn")
        print(f"✅ Saved interactive HTML: {PLOT_HTML}")
    except Exception as e:
        print("Plotly export skipped:", e)

if __name__ == "__main__":
    main()
