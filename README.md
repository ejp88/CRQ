# Cyber Risk Quantification (CRQ) Dashboard — FAIR-lite

A lightweight, working implementation of FAIR-based cyber risk quantification:
Monte Carlo simulation over uncertain inputs, producing defensible dollar-denominated
risk exposure for board, audit, and executive stakeholders — instead of a red/yellow/green
heat map with no math behind it.

## Why this exists

Most cyber risk reporting is qualitative (High/Med/Low) and doesn't hold up well when
audit or the board asks "how did you get that number?" This project applies the FAIR
(Factor Analysis of Information Risk) methodology to translate technical risk into
financial exposure ranges that speak the same language as enterprise risk management.

## Methodology

```
Loss Event Frequency (LEF)      = Threat Event Frequency (TEF) x Vulnerability
Annualized Loss Expectancy (ALE) = LEF x Loss Magnitude (LM)
```

Every input (TEF, Vulnerability, Loss Magnitude) is expressed as a **range**
(min / most likely / max) rather than a single guess, using a modified PERT
distribution. A Monte Carlo simulation (10,000 iterations per scenario) samples
from these distributions to produce a full loss exceedance curve, not a single
misleadingly-precise number.

This is intentionally a **lite** version of FAIR — a full FAIR implementation
decomposes Vulnerability further into Threat Capability vs. Resistance Strength,
and Loss Magnitude into primary/secondary loss categories. This project simplifies
those into single ranges to keep it buildable and demoable, while preserving the
core probabilistic reasoning that differentiates FAIR from a scorecard.

## Project structure

```
cyber_risk_crq/
├── fair_engine.py        # Core FAIR math + Monte Carlo simulation engine
├── app.py                 # Streamlit dashboard (Board / Audit / Executive views)
├── requirements.txt
├── scenarios/             # One YAML config per risk scenario (edit these to recalibrate)
│   ├── ransomware.yaml
│   ├── third_party_breach.yaml
│   ├── insider_exfiltration.yaml
│   ├── availability_disruption.yaml
│   └── cloud_misconfiguration.yaml
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or run the engine standalone from the command line for a quick numeric summary:

```bash
python3 fair_engine.py
```

## The three views

- **Board view** — total exposure (mean and P90), a risk-appetite threshold line,
  and a trended chart. Built for a 5-minute board conversation, not a data dump.
- **Audit / Methodology view** — every input assumption, its source/rationale, and
  the resulting loss exceedance curve per scenario. This is the traceability layer
  that answers "how did you get that number?" before it's asked.
- **Executive view** — scenario ranking plus a remediation slider: model how much a
  control investment (e.g. +25% vulnerability reduction from improved EDR coverage)
  reduces annual loss exposure, to build a defensible budget/ROI case.

## Calibrating inputs

Input ranges in `scenarios/*.yaml` are illustrative for this demo. In production,
each range should be grounded in:
- **Threat Event Frequency** — industry incident benchmarks (Verizon DBIR, industry
  ISAC data) adjusted for your sector and threat exposure.
- **Vulnerability** — your actual control maturity (EDR coverage %, patch SLAs,
  segmentation, DLP coverage) — ideally sourced from CMDB/tooling, not estimated.
- **Loss Magnitude** — internal Business Impact Analysis (BIA) figures, cyber
  insurance actuarial data, and IBM/Ponemon Cost of a Data Breach benchmarks.

## Extending this

Natural next steps beyond this lite version:
- Wire scenario inputs to live tooling (Tenable/Qualys for vulnerability data,
  a CMDB for asset criticality) instead of manually-edited YAML.
- Add asset-level granularity instead of scenario-level.
- Add historical model runs so the board trend chart reflects real quarter-over-quarter
  changes instead of simulated ones.
- Decompose Vulnerability and Loss Magnitude into full FAIR sub-factors for scenarios
  that warrant deeper rigor (e.g. your top 1-2 board-level risks).
