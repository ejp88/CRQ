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
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fair_engine import (
    MIN_EVENTS_FOR_HISTORY,
    apply_remediation,
    load_scenarios,
    run_historical_simulation,
    run_simulation,
)
from loss_store import (
    build_histories,
    clean_losses,
    default_data_dir,
    filter_window,
    load_losses,
    load_settings,
    save_losses,
    save_settings,
)

st.set_page_config(page_title="Cyber Risk Quantification Dashboard", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------------------
# Look & feel: dark "security operations" theme
# ---------------------------------------------------------------------------
CYAN = "#00e5ff"
MAGENTA = "#ff3b6b"
GREEN = "#00ff9c"
AMBER = "#ffb020"
TEXT = "#e6edf3"
MUTED = "#8b9bb4"
GRID = "rgba(0, 229, 255, 0.10)"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"], .stMarkdown, p, label {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }
    .stApp {
        background:
            radial-gradient(1200px 500px at 85% -10%, rgba(0, 229, 255, 0.10), transparent 60%),
            radial-gradient(900px 500px at -10% 110%, rgba(255, 59, 107, 0.08), transparent 60%),
            linear-gradient(rgba(0, 229, 255, 0.035) 1px, transparent 1px) 0 0 / 44px 44px,
            linear-gradient(90deg, rgba(0, 229, 255, 0.035) 1px, transparent 1px) 0 0 / 44px 44px,
            #0a0e17;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2.2rem; max-width: 1400px; }

    /* Hero header */
    .crq-eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; letter-spacing: 0.22em;
        color: #00e5ff; text-transform: uppercase; margin-bottom: 0.2rem;
    }
    .crq-title {
        font-size: 2.3rem; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 0.3rem 0;
        background: linear-gradient(90deg, #ffffff 0%, #00e5ff 70%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .crq-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.4rem 0 1.4rem 0; }
    .crq-badge {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.06em;
        color: #8b9bb4; border: 1px solid rgba(0, 229, 255, 0.25); border-radius: 999px;
        padding: 0.2rem 0.7rem; background: rgba(0, 229, 255, 0.05);
    }
    .crq-badge b { color: #00e5ff; font-weight: 600; }

    /* Section headings */
    h2, h3, [data-testid="stHeading"] h3 {
        font-weight: 600 !important; letter-spacing: -0.005em;
    }
    h3::before, [data-testid="stHeading"] h3::before {
        content: "//"; font-family: 'JetBrains Mono', monospace; color: #00e5ff; margin-right: 0.5rem;
        opacity: 0.8;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(17, 26, 43, 0.95), rgba(10, 16, 28, 0.95));
        border: 1px solid rgba(0, 229, 255, 0.22);
        border-left: 3px solid #00e5ff;
        border-radius: 10px; padding: 1rem 1.1rem;
        box-shadow: 0 0 18px rgba(0, 229, 255, 0.07), inset 0 0 24px rgba(0, 229, 255, 0.03);
        transition: box-shadow .2s ease, border-color .2s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 229, 255, 0.6); box-shadow: 0 0 24px rgba(0, 229, 255, 0.18);
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.1em; color: #8b9bb4;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', 'Consolas', monospace; font-weight: 600; color: #ffffff;
        text-shadow: 0 0 14px rgba(0, 229, 255, 0.35);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem; border-bottom: 1px solid rgba(0, 229, 255, 0.2);
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; letter-spacing: 0.08em;
        text-transform: uppercase; color: #8b9bb4; padding: 0.7rem 1.1rem;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        color: #00e5ff !important; background: rgba(0, 229, 255, 0.07);
        text-shadow: 0 0 10px rgba(0, 229, 255, 0.6);
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #00e5ff; box-shadow: 0 0 10px #00e5ff; }

    /* Expanders, alerts, tables, charts */
    [data-testid="stExpander"] {
        border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 10px;
        background: rgba(17, 26, 43, 0.6);
    }
    [data-testid="stExpander"] summary:hover { color: #00e5ff; }
    [data-testid="stAlert"] {
        background: rgba(0, 229, 255, 0.06); border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 10px;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(0, 229, 255, 0.2); border-radius: 10px; overflow: hidden;
    }
    [data-testid="stPlotlyChart"] {
        border: 1px solid rgba(0, 229, 255, 0.15); border-radius: 12px; padding: 0.4rem;
        background: rgba(10, 16, 28, 0.55);
    }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #a9b8cf; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 229, 255, 0.3); border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def style_fig(fig, height=None, **layout):
    """Apply the dark cyber theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[CYAN, MAGENTA, GREEN, AMBER, "#7c5cff", "#4da3ff"],
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT),
        title_font=dict(size=15, color=TEXT),
        hoverlabel=dict(bgcolor="#0a0e17", bordercolor=CYAN, font=dict(family="JetBrains Mono, monospace")),
        **layout,
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig

SCENARIO_DIR = os.path.join(os.path.dirname(__file__), "scenarios")
ITERATIONS = 10000
RISK_APPETITE_THRESHOLD = 15_000_000  # example board-set annual loss tolerance

# ---------------------------------------------------------------------------
# Data loading (cached so the sim doesn't rerun on every widget interaction)
# ---------------------------------------------------------------------------

DATA_DIR = default_data_dir()


@st.cache_data(show_spinner="Running simulations…")
def get_results(loss_df, window_years, as_of, seed=42):
    """
    Scenarios with enough recorded losses in the look-back window are forecast
    from that history; the rest fall back to the expert ranges in scenarios/*.yaml.
    `as_of` is part of the cache key so the look-back window rolls forward daily.
    """
    scenarios = load_scenarios(SCENARIO_DIR)
    histories = build_histories(loss_df, scenarios, window_years, as_of)
    results = {}
    for name, s in scenarios.items():
        if name in histories:
            results[name] = run_historical_simulation(histories[name], iterations=ITERATIONS, seed=seed)
        else:
            results[name] = run_simulation(s, iterations=ITERATIONS, seed=seed)
    return scenarios, histories, results


saved_losses = load_losses(DATA_DIR)
settings = load_settings(DATA_DIR)
window_years = int(settings["window_years"])
scenarios, histories, results = get_results(saved_losses, window_years, str(date.today()))


def fmt_usd(x):
    if x == 0:
        return "$0"
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    return f"${x/1_000:.0f}K"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="crq-eyebrow">Cyber Risk Intelligence</div>
    <div class="crq-title">Cyber Risk Quantification</div>
    <div class="crq-badges">
        <span class="crq-badge"><b>MODEL</b> FAIR-lite</span>
        <span class="crq-badge"><b>ENGINE</b> Monte Carlo · {ITERATIONS:,} iterations / scenario</span>
        <span class="crq-badge"><b>DATA</b> Demo — illustrative inputs, not production figures</span>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_board, tab_loss, tab_audit, tab_exec = st.tabs(
    ["Board View", "Loss Data", "Audit / Methodology", "Executive View"]
)

# ---------------------------------------------------------------------------
# BOARD VIEW
# ---------------------------------------------------------------------------
with tab_board:
    st.subheader("Annual Loss Exposure by Risk Scenario")
    st.caption(
        f"{len(histories)} of {len(results)} scenarios are forecast from your recorded loss history; "
        f"the rest use expert ranges. Add or update losses in the Loss Data tab."
    )

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
            marker=dict(color="rgba(0, 229, 255, 0.55)", line=dict(color=CYAN, width=1.5)),
            text=f"P50: {fmt_usd(r.ale_p50)}",
            textposition="inside",
            showlegend=False,
        ))
    fig.add_vline(x=RISK_APPETITE_THRESHOLD, line_dash="dash", line_color=MAGENTA,
                  annotation_text="Risk Appetite Threshold", annotation_position="top",
                  annotation_font_color=MAGENTA)
    fig.update_layout(
        title="Exposure Range by Scenario (P10 – P90 band, annual $)",
        xaxis_title="Annual Loss Exposure ($)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    style_fig(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Simulated Quarterly Trend (illustrative)")
    quarters = ["Q1", "Q2", "Q3", "Q4 (current)"]
    rng = np.random.default_rng(1)
    trend = total_mean * (1 + np.cumsum(rng.normal(0.02, 0.05, len(quarters)) - 0.02))
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=quarters, y=trend, mode="lines+markers",
        line=dict(color=CYAN, width=3), marker=dict(size=9, color="#0a0e17", line=dict(color=CYAN, width=2)),
        fill="tozeroy", fillcolor="rgba(0, 229, 255, 0.08)",
    ))
    trend_fig.add_hline(y=RISK_APPETITE_THRESHOLD, line_dash="dash", line_color=MAGENTA,
                        annotation_text="Risk Appetite", annotation_font_color=MAGENTA)
    trend_fig.update_layout(yaxis_title="Total Mean Annual Loss ($)", margin=dict(l=10, r=10, t=20, b=10))
    style_fig(trend_fig, height=300)
    st.plotly_chart(trend_fig, use_container_width=True)
    st.caption("Note: quarterly trend is simulated for demo purposes. In production this would pull from historical model runs.")

# ---------------------------------------------------------------------------
# LOSS DATA (enter / update historical losses, recalculate forecasts)
# ---------------------------------------------------------------------------
with tab_loss:
    st.subheader("Historical Loss Data")
    st.caption(
        "Record actual loss events here. Add a row with the + at the bottom of the table, paste rows "
        "straight from Excel, or select rows and press Delete. Click Save to recalculate every forecast. "
        f"A scenario is forecast from your data once it has at least {MIN_EVENTS_FOR_HISTORY} events inside "
        "the look-back window; otherwise it keeps using its expert ranges."
    )

    flash = st.session_state.pop("loss_flash", None)
    if flash:
        st.success(flash)

    window_input = st.number_input(
        "Look-back window (years)", min_value=1, max_value=10, value=window_years, step=1,
        help="Only losses dated within this many years of today feed the forecast. "
             "Event frequency is calculated as events per year over this window.",
    )

    editor_key = f"loss_editor_{st.session_state.get('loss_editor_version', 0)}"
    edited = st.data_editor(
        saved_losses,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key=editor_key,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
            "Scenario": st.column_config.SelectboxColumn("Scenario", options=list(scenarios), required=True),
            "Loss ($)": st.column_config.NumberColumn("Loss ($)", min_value=0, step=1000, format="$%d", required=True),
            "Notes": st.column_config.TextColumn("Notes"),
        },
    )

    if st.button("Save & recalculate forecasts", type="primary"):
        clean, dropped = clean_losses(edited, scenarios)
        try:
            save_losses(clean, DATA_DIR)
            save_settings({"window_years": int(window_input)}, DATA_DIR)
        except OSError as exc:
            st.error(f"Could not save to {DATA_DIR}: {exc}")
        else:
            msg = f"Saved {len(clean)} loss events. Forecasts recalculated."
            if dropped:
                msg += f" {dropped} incomplete row(s) were skipped (each needs a date, scenario and loss above $0)."
            st.session_state["loss_flash"] = msg
            st.session_state["loss_editor_version"] = st.session_state.get("loss_editor_version", 0) + 1
            st.rerun()

    st.subheader("Forecast Basis (from saved data)")
    in_window = filter_window(saved_losses, window_years)
    basis_rows = []
    for name in scenarios:
        rows = in_window[in_window["Scenario"] == name]
        from_history = name in histories
        basis_rows.append({
            "Scenario": name,
            "Forecast basis": "Your loss history" if from_history else "Expert ranges",
            f"Events (last {window_years} yr)": len(rows),
            "Total losses": fmt_usd(rows["Loss ($)"].sum()),
            "Forecast mean annual loss": fmt_usd(results[name].ale_mean),
            "Forecast P90": fmt_usd(results[name].ale_p90),
        })
    st.dataframe(pd.DataFrame(basis_rows), use_container_width=True, hide_index=True)

    if len(in_window):
        by_year = (
            in_window.assign(Year=in_window["Date"].dt.year)
            .groupby(["Year", "Scenario"], as_index=False)["Loss ($)"].sum()
        )
        hist_fig = go.Figure()
        for name, grp in by_year.groupby("Scenario"):
            hist_fig.add_trace(go.Bar(name=name, x=grp["Year"].astype(str), y=grp["Loss ($)"]))
        hist_fig.update_layout(
            barmode="stack", title="Recorded Losses by Year", yaxis_title="Loss ($)",
            margin=dict(l=10, r=10, t=50, b=10),
        )
        style_fig(hist_fig, height=320)
        st.plotly_chart(hist_fig, use_container_width=True)
    else:
        st.info("No losses recorded in the look-back window yet — all scenarios are using their expert ranges.")

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
            if name in histories:
                h = histories[name]
                colA.metric(f"Loss events (last {window_years} yr)", f"{h.n_events}",
                            help=f"Observed rate: {h.observed_rate:.2f} events/yr")
                colB.metric("Average loss per event", fmt_usd(h.mean))
                colC.metric("Largest recorded loss", fmt_usd(h.largest))
                st.markdown(
                    "**Data source / rationale:** forecast from your recorded loss history (Loss Data tab). "
                    "Event frequency: Poisson with the yearly rate uncertain (Gamma posterior from the "
                    "observed count and window). Severity: lognormal fitted to the recorded losses, capped at "
                    "10× the largest recorded loss. The expert ranges in the scenario file are not used."
                )
            else:
                colA.metric("Threat Event Frequency", f"{s.threat_event_frequency.most_likely:.0f}/yr",
                            help=f"Range: {s.threat_event_frequency.min:.0f} – {s.threat_event_frequency.max:.0f}")
                colB.metric("Vulnerability", f"{s.vulnerability.most_likely:.0%}",
                            help=f"Range: {s.vulnerability.min:.0%} – {s.vulnerability.max:.0%}")
                colC.metric("Loss Magnitude", fmt_usd(s.loss_magnitude.most_likely),
                            help=f"Range: {fmt_usd(s.loss_magnitude.min)} – {fmt_usd(s.loss_magnitude.max)}")
                st.markdown(f"**Data source / rationale:** {s.source_notes}")

            thresholds, exceed_prob = r.exceedance_curve()
            curve_fig = go.Figure()
            curve_fig.add_trace(go.Scatter(
                x=thresholds, y=exceed_prob, mode="lines", line=dict(color=CYAN, width=2.5),
                fill="tozeroy", fillcolor="rgba(0, 229, 255, 0.08)",
            ))
            curve_fig.update_layout(
                title="Loss Exceedance Curve", xaxis_title="Annual Loss ($)",
                yaxis_title="Probability of Exceeding", margin=dict(l=10, r=10, t=40, b=10),
            )
            style_fig(curve_fig, height=280)
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
    if selected in histories:
        st.caption("This scenario is forecast from your loss history, so the reduction is applied to its event frequency.")
        remediated_result = run_historical_simulation(
            histories[selected], iterations=ITERATIONS, seed=42, frequency_reduction_pct=reduction_pct
        )
    else:
        remediated_scenario = apply_remediation(scenarios[selected], reduction_pct)
        remediated_result = run_simulation(remediated_scenario, iterations=ITERATIONS, seed=42)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Mean ALE", fmt_usd(base_result.ale_mean))
    c2.metric("Projected Mean ALE (post-investment)", fmt_usd(remediated_result.ale_mean),
              delta=f"-{fmt_usd(base_result.ale_mean - remediated_result.ale_mean)}", delta_color="normal")
    reduction_dollars = base_result.ale_mean - remediated_result.ale_mean
    c3.metric("Annual Risk Reduction", fmt_usd(reduction_dollars))

    compare_fig = go.Figure()
    compare_fig.add_trace(go.Bar(name="Current", x=["Mean ALE"], y=[base_result.ale_mean], marker_color=MAGENTA))
    compare_fig.add_trace(go.Bar(name="Post-Investment", x=["Mean ALE"], y=[remediated_result.ale_mean], marker_color=GREEN))
    compare_fig.update_layout(barmode="group", margin=dict(l=10, r=10, t=20, b=10))
    style_fig(compare_fig, height=320)
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
