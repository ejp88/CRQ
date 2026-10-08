"""
FAIR-lite Cyber Risk Quantification Engine
--------------------------------------------
Implements a simplified FAIR (Factor Analysis of Information Risk) model:

    Loss Event Frequency (LEF) = Threat Event Frequency (TEF) x Vulnerability
    Annualized Loss Expectancy (ALE) = LEF x Loss Magnitude (LM)

Each input is a PERT distribution (min, most_likely, max) rather than a
point estimate, reflecting real-world uncertainty. Monte Carlo simulation
runs thousands of simulated "years" to produce a distribution of possible
annual loss outcomes -- i.e. a loss exceedance curve -- rather than a single
misleadingly-precise number.

Usage:
    from fair_engine import RiskScenario, run_simulation, load_scenarios

    scenarios = load_scenarios("scenarios/")
    results = {name: run_simulation(s, iterations=10000) for name, s in scenarios.items()}
"""

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PertRange:
    min: float
    most_likely: float
    max: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Modified PERT distribution sample.
        Lambda=4 is the standard PERT shape parameter (weights the most
        likely value heavily without ignoring tail extremes).
        """
        lam = 4.0
        if self.max == self.min:
            return np.full(n, self.min)
        alpha = 1 + lam * (self.most_likely - self.min) / (self.max - self.min)
        beta = 1 + lam * (self.max - self.most_likely) / (self.max - self.min)
        return self.min + rng.beta(alpha, beta, size=n) * (self.max - self.min)


@dataclass
class RiskScenario:
    name: str
    description: str
    threat_event_frequency: PertRange
    vulnerability: PertRange
    loss_magnitude: PertRange
    source_notes: str = ""


@dataclass
class SimulationResult:
    scenario_name: str
    annual_losses: np.ndarray  # one simulated annual loss per iteration
    ale_mean: float = field(init=False)
    ale_p10: float = field(init=False)
    ale_p50: float = field(init=False)
    ale_p90: float = field(init=False)
    ale_p95: float = field(init=False)
    var_95: float = field(init=False)  # value at risk (95th percentile loss)
    prob_material_loss: float = field(init=False)  # P(loss > threshold)

    def __post_init__(self):
        self.ale_mean = float(np.mean(self.annual_losses))
        self.ale_p10 = float(np.percentile(self.annual_losses, 10))
        self.ale_p50 = float(np.percentile(self.annual_losses, 50))
        self.ale_p90 = float(np.percentile(self.annual_losses, 90))
        self.ale_p95 = float(np.percentile(self.annual_losses, 95))
        self.var_95 = self.ale_p95

    def exceedance_curve(self, n_points: int = 100):
        """
        Returns (loss_values, probability_of_exceeding) for plotting a loss
        exceedance curve -- the standard FAIR/actuarial output format.
        """
        sorted_losses = np.sort(self.annual_losses)[::-1]
        n = len(sorted_losses)
        thresholds = np.linspace(sorted_losses.min(), sorted_losses.max(), n_points)
        exceed_prob = [np.mean(self.annual_losses >= t) for t in thresholds]
        return thresholds, exceed_prob

    def prob_exceeds(self, threshold: float) -> float:
        return float(np.mean(self.annual_losses >= threshold))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _pert_from_dict(d: dict) -> PertRange:
    return PertRange(min=float(d["min"]), most_likely=float(d["most_likely"]), max=float(d["max"]))


def load_scenario(filepath: str) -> RiskScenario:
    with open(filepath, "r") as f:
        raw = yaml.safe_load(f)
    return RiskScenario(
        name=raw["scenario"],
        description=raw.get("description", "").strip(),
        threat_event_frequency=_pert_from_dict(raw["threat_event_frequency"]),
        vulnerability=_pert_from_dict(raw["vulnerability"]),
        loss_magnitude=_pert_from_dict(raw["loss_magnitude"]),
        source_notes=raw.get("source_notes", "").strip(),
    )


def load_scenarios(directory: str) -> Dict[str, RiskScenario]:
    scenarios = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
        s = load_scenario(path)
        scenarios[s.name] = s
    return scenarios


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(scenario: RiskScenario, iterations: int = 10000, seed: int = None) -> SimulationResult:
    rng = np.random.default_rng(seed)

    tef = scenario.threat_event_frequency.sample(iterations, rng)
    vuln = np.clip(scenario.vulnerability.sample(iterations, rng), 0, 1)
    lm = scenario.loss_magnitude.sample(iterations, rng)

    # Loss Event Frequency: expected number of successful loss events per year
    lef = tef * vuln

    # For each simulated year, the number of loss events is itself uncertain.
    # Model event count as Poisson(lef) per iteration, then sum loss magnitude
    # draws for however many events occur that "year".
    annual_losses = np.zeros(iterations)
    event_counts = rng.poisson(lef)
    max_events = int(event_counts.max()) if iterations > 0 else 0

    if max_events > 0:
        # Pre-sample a pool of loss magnitudes to draw from per event
        loss_pool = scenario.loss_magnitude.sample(iterations * max(max_events, 1), rng)
        pool_idx = 0
        for i in range(iterations):
            k = event_counts[i]
            if k > 0:
                annual_losses[i] = loss_pool[pool_idx: pool_idx + k].sum()
                pool_idx += k

    return SimulationResult(scenario_name=scenario.name, annual_losses=annual_losses)


def run_all(scenarios: Dict[str, RiskScenario], iterations: int = 10000, seed: int = None) -> Dict[str, SimulationResult]:
    return {name: run_simulation(s, iterations=iterations, seed=seed) for name, s in scenarios.items()}


# ---------------------------------------------------------------------------
# Empirical model: forecast from recorded (historical) loss events
# ---------------------------------------------------------------------------

MIN_EVENTS_FOR_HISTORY = 3  # fewer recorded events than this -> fall back to expert ranges
SEVERITY_CAP_MULTIPLE = 10  # lognormal tail guard: no single event above 10x the largest recorded loss


@dataclass
class LossHistory:
    """Recorded loss events for one scenario over a look-back window."""
    name: str
    losses: np.ndarray  # one dollar amount per recorded event (all > 0)
    window_years: float

    @property
    def n_events(self) -> int:
        return len(self.losses)

    @property
    def total(self) -> float:
        return float(self.losses.sum())

    @property
    def mean(self) -> float:
        return float(self.losses.mean())

    @property
    def largest(self) -> float:
        return float(self.losses.max())

    @property
    def observed_rate(self) -> float:
        """Observed loss events per year over the window."""
        return self.n_events / self.window_years


def run_historical_simulation(
    history: LossHistory,
    iterations: int = 10000,
    seed: int = None,
    frequency_reduction_pct: float = 0.0,
) -> SimulationResult:
    """
    Forecast annual loss from recorded events.

    Frequency: the true yearly event rate is uncertain with only a few years of
    data, so each simulated year first draws a rate from Gamma(k + 0.5, 1/T)
    (Jeffreys-prior posterior for k events observed over T years), then draws an
    event count from Poisson(rate).
    Severity: lognormal fitted to the recorded losses, capped at
    SEVERITY_CAP_MULTIPLE x the largest recorded loss.

    frequency_reduction_pct scales the event rate down, to model a control
    investment (the empirical analogue of apply_remediation).
    """
    rng = np.random.default_rng(seed)

    rate = rng.gamma(shape=history.n_events + 0.5, scale=1.0 / history.window_years, size=iterations)
    rate = rate * (1 - frequency_reduction_pct / 100.0)
    counts = rng.poisson(rate)

    logs = np.log(history.losses)
    mu = float(logs.mean())
    sigma = max(float(logs.std(ddof=1)), 0.1)  # floor so identical losses don't collapse the spread

    n_total = int(counts.sum())
    severities = np.minimum(rng.lognormal(mu, sigma, n_total), history.largest * SEVERITY_CAP_MULTIPLE)
    year_idx = np.repeat(np.arange(iterations), counts)
    annual_losses = np.bincount(year_idx, weights=severities, minlength=iterations).astype(float)

    return SimulationResult(scenario_name=history.name, annual_losses=annual_losses)


# ---------------------------------------------------------------------------
# Remediation tradeoff modeling
# ---------------------------------------------------------------------------

def apply_remediation(scenario: RiskScenario, vulnerability_reduction_pct: float) -> RiskScenario:
    """
    Returns a new scenario with vulnerability reduced by the given percentage,
    modeling the effect of a control investment (e.g. improved EDR coverage,
    DLP rollout, CSPM tooling). Used to show risk-reduction-per-dollar tradeoffs.
    """
    factor = 1 - (vulnerability_reduction_pct / 100.0)
    v = scenario.vulnerability
    new_v = PertRange(
        min=max(v.min * factor, 0),
        most_likely=max(v.most_likely * factor, 0),
        max=max(v.max * factor, 0),
    )
    return RiskScenario(
        name=scenario.name,
        description=scenario.description,
        threat_event_frequency=scenario.threat_event_frequency,
        vulnerability=new_v,
        loss_magnitude=scenario.loss_magnitude,
        source_notes=scenario.source_notes,
    )


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = load_scenarios(os.path.join(os.path.dirname(__file__), "scenarios"))
    results = run_all(scenarios, iterations=10000, seed=42)

    print(f"{'Scenario':<40} {'Mean ALE':>15} {'P50':>15} {'P90':>15} {'VaR95':>15}")
    print("-" * 105)
    total_mean = 0
    for name, r in sorted(results.items(), key=lambda x: -x[1].ale_mean):
        print(f"{name:<40} {r.ale_mean:>15,.0f} {r.ale_p50:>15,.0f} {r.ale_p90:>15,.0f} {r.var_95:>15,.0f}")
        total_mean += r.ale_mean
    print("-" * 105)
    print(f"{'TOTAL (sum of means)':<40} {total_mean:>15,.0f}")
