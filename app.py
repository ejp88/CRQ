"""
Cyber Risk Quantification (CRQ) Dashboard
-------------------------------------------
Streamlit front-end for the FAIR-lite engine. Three stakeholder views:
  - Board:  headline $ exposure range, trend, risk appetite threshold
  - Audit:  methodology, input assumptions, traceability per scenario
  - Executive: scenario ranking, remediation cost/benefit tradeoff

Run with:  streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fair_engine import (
    apply_remediation,
    load_scenarios,
    run_all,
    run_simulation,
)

st.set_page_config(page_title="Cyber Risk Quantification Dashboard", layout="wide")

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
ITERATIONS = 10000
RISK_APPETITE_THRESHOLD = 15_000_000  # example board-set annual loss tolerance

# ---------------------------------------------------------------------------
# Data loading (cached so the sim doesn't rerun on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_data
def get_results(seed=42):
    scenarios = load_scenarios(SCENARIO_DIR)
    results = run_all(scenarios, iterations=ITERATIONS, seed=seed)
    return scenarios, results


scenarios, results = get_results()


def fmt_usd(x):
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    return f"${x/1_000:.0f}K"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Cyber Risk Quantification Dashboard")
st.caption(
    "FAIR-lite model | Monte Carlo simulation (10,000 iterations/scenario) | "
    "Demo data — inputs are illustrative, not production figures"
)

tab_board, tab_audit, tab_exec = st.tabs(["📊 Board View", "🔍 Audit / Methodology", "🎯 Executive View"])

# ---------------------------------------------------------------------------
# BOARD VIEW
# ---------------------------------------------------------------------------
with tab_board:
    st.subheader("Annual Loss Exposure by Risk Scenario")

    total_mean = sum(r.ale_mean for r in results.values())
    total_p90 = sum(r.ale_p90 for r in results.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Expected Annual Loss (mean)", fmt_usd(total_mean))
    c2.metric("Total 90th Percentile Exposure", fmt_usd(total_p90))
    c3.metric("Board Risk Appetite Threshold", fmt_usd(RISK_APPETITE_THRESHOLD),
              delta=f"{'Over' if total_p90 > RISK_APPETITE_THRESHOLD else 'Within'} appetite at P90",
              delta_color="inverse")

    # Range chart: P10 - P90 band per scenario, sorted by mean exposure
    sorted_names = sorted(results.keys(), key=lambda n: -results[n].ale_mean)
    fig = go.Figure()
    for name in sorted_names:
        r = results[name]
        fig.add_trace(go.Bar(
            name=name,
            y=[name],
            x=[r.ale_p90 - r.ale_p10],
            base=[r.ale_p10],
            orientation="h",
            marker=dict(color="rgba(180, 40, 40, 0.55)"),
            text=f"P50: {fmt_usd(r.ale_p50)}",
            textposition="inside",
            showlegend=False,
        ))
    fig.add_vline(x=RISK_APPETITE_THRESHOLD, line_dash="dash", line_color="black",
                  annotation_text="Risk Appetite Threshold", annotation_position="top")
    fig.update_layout(
        title="Exposure Range by Scenario (P10 – P90 band, annual $)",
        xaxis_title="Annual Loss Exposure ($)",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Simulated Quarterly Trend (illustrative)")
    quarters = ["Q1", "Q2", "Q3", "Q4 (current)"]
    rng = np.random.default_rng(1)
    trend = total_mean * (1 + np.cumsum(rng.normal(0.02, 0.05, len(quarters)) - 0.02))
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=quarters, y=trend, mode="lines+markers", line=dict(color="#b42828", width=3)))
    trend_fig.add_hline(y=RISK_APPETITE_THRESHOLD, line_dash="dash", annotation_text="Risk Appetite")
    trend_fig.update_layout(height=300, yaxis_title="Total Mean Annual Loss ($)", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(trend_fig, use_container_width=True)
    st.caption("Note: quarterly trend is simulated for demo purposes. In production this would pull from historical model runs.")

# ---------------------------------------------------------------------------
# AUDIT VIEW
# ---------------------------------------------------------------------------
with tab_audit:
    st.subheader("Methodology")
    st.markdown("""
    This model applies a simplified **FAIR (Factor Analysis of Information Risk)** approach:

    - **Loss Event Frequency (LEF)** = Threat Event Frequency × Vulnerability
    - **Annualized Loss Expectancy (ALE)** = LEF × Loss Magnitude
    - Each input is expressed as a **range** (min / most likely / max) using a modified PERT distribution,
      not a single point estimate — this reflects genuine uncertainty rather than false precision.
    - **Monte Carlo simulation** (10,000 iterations per scenario) generates a full distribution of
      possible annual outcomes, from which we derive percentile-based exposure (P50, P90, VaR95)
      and a loss exceedance curve.
    """)

    st.subheader("Scenario-Level Assumptions & Traceability")
    for name, s in scenarios.items():
        r = results[name]
        with st.expander(f"{name} — Mean ALE: {fmt_usd(r.ale_mean)}"):
            st.markdown(f"**Description:** {s.description}")
            colA, colB, colC = st.columns(3)
            colA.metric("Threat Event Frequency", f"{s.threat_event_frequency.most_likely:.0f}/yr",
                        help=f"Range: {s.threat_event_frequency.min:.0f} – {s.threat_event_frequency.max:.0f}")
            colB.metric("Vulnerability", f"{s.vulnerability.most_likely:.0%}",
                        help=f"Range: {s.vulnerability.min:.0%} – {s.vulnerability.max:.0%}")
            colC.metric("Loss Magnitude", fmt_usd(s.loss_magnitude.most_likely),
                        help=f"Range: {fmt_usd(s.loss_magnitude.min)} – {fmt_usd(s.loss_magnitude.max)}")
            st.markdown(f"**Data source / rationale:** {s.source_notes}")

            thresholds, exceed_prob = r.exceedance_curve()
            curve_fig = go.Figure()
            curve_fig.add_trace(go.Scatter(x=thresholds, y=exceed_prob, mode="lines", line=dict(color="#2b5fb4")))
            curve_fig.update_layout(
                title="Loss Exceedance Curve", xaxis_title="Annual Loss ($)",
                yaxis_title="Probability of Exceeding", height=280, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(curve_fig, use_container_width=True)

    st.info(
        "Audit note: every output number traces back to an explicit, documented input range and a "
        "reproducible simulation (fixed seed available on request). No black-box scoring."
    )

# ---------------------------------------------------------------------------
# EXECUTIVE VIEW
# ---------------------------------------------------------------------------
with tab_exec:
    st.subheader("Risk Ranking & Remediation Tradeoff")
    st.caption("Model the effect of a control investment on exposure before committing budget.")

    scenario_names = list(scenarios.keys())
    selected = st.selectbox("Select a scenario to model remediation against:", scenario_names)
    reduction_pct = st.slider("Vulnerability reduction from control investment (%)", 0, 80, 25, step=5)

    base_result = results[selected]
    remediated_scenario = apply_remediation(scenarios[selected], reduction_pct)
    remediated_result = run_simulation(remediated_scenario, iterations=ITERATIONS, seed=42)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Mean ALE", fmt_usd(base_result.ale_mean))
    c2.metric("Projected Mean ALE (post-investment)", fmt_usd(remediated_result.ale_mean),
              delta=f"-{fmt_usd(base_result.ale_mean - remediated_result.ale_mean)}", delta_color="normal")
    reduction_dollars = base_result.ale_mean - remediated_result.ale_mean
    c3.metric("Annual Risk Reduction", fmt_usd(reduction_dollars))

    compare_fig = go.Figure()
    compare_fig.add_trace(go.Bar(name="Current", x=["Mean ALE"], y=[base_result.ale_mean], marker_color="#b42828"))
    compare_fig.add_trace(go.Bar(name="Post-Investment", x=["Mean ALE"], y=[remediated_result.ale_mean], marker_color="#2b8a3e"))
    compare_fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(compare_fig, use_container_width=True)

    st.markdown(
        f"**Talking point for budget conversations:** a control investment that reduces vulnerability "
        f"by {reduction_pct}% for *{selected}* is projected to reduce mean annual loss exposure by "
        f"**{fmt_usd(reduction_dollars)}/year**. Compare this figure directly against the cost of the "
        f"control investment to build a defensible ROI case."
    )

    st.subheader("Full Scenario Ranking")
    df = pd.DataFrame([
        {
            "Scenario": name,
            "Mean ALE": r.ale_mean,
            "P50": r.ale_p50,
            "P90": r.ale_p90,
            "VaR 95": r.var_95,
        }
        for name, r in results.items()
    ]).sort_values("Mean ALE", ascending=False)
    df_display = df.copy()
    for col in ["Mean ALE", "P50", "P90", "VaR 95"]:
        df_display[col] = df_display[col].apply(fmt_usd)
    st.dataframe(df_display, use_container_width=True, hide_index=True)
