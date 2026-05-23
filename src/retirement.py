from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RetirementInputs:
    current_age: int
    retirement_age: int
    life_expectancy: int
    current_assets_wan: float          # 萬元
    monthly_contribution_wan: float    # 萬元/月
    monthly_expense_wan: float         # 退休後每月支出，萬元
    mean_annual_return: float          # 年均報酬率，e.g. 0.07
    annual_return_std: float           # 年化波動率（標準差），e.g. 0.15
    inflation_rate: float              # 通膨率，e.g. 0.025
    n_simulations: int = 1000


@dataclass
class MonteCarloResult:
    success_rate: float
    n_simulations: int
    depleted_count: int
    ages: list
    percentile_paths: dict             # {10: list, 25: list, 50: list, 75: list, 90: list}
    median_at_retirement: float
    p10_at_retirement: float
    p90_at_retirement: float
    median_final: float
    p10_final: float
    p90_final: float
    survival_rates: list               # 每個年齡仍有正資產的模擬比例（0~1）


def run_monte_carlo(inputs: RetirementInputs) -> MonteCarloResult:
    rng = np.random.default_rng()
    n = inputs.n_simulations
    accum_years = inputs.retirement_age - inputs.current_age
    retire_years = inputs.life_expectancy - inputs.retirement_age
    total_years = accum_years + retire_years

    annual_contrib = inputs.monthly_contribution_wan * 12
    adj_expense = inputs.monthly_expense_wan * (1 + inputs.inflation_rate) ** accum_years
    base_withdrawal = adj_expense * 12
    g = inputs.inflation_rate

    paths = np.zeros((n, total_years + 1))
    paths[:, 0] = inputs.current_assets_wan

    if total_years > 0:
        returns = rng.normal(inputs.mean_annual_return, inputs.annual_return_std, (n, total_years))

        for yr in range(accum_years):
            paths[:, yr + 1] = np.maximum(
                paths[:, yr] * (1 + returns[:, yr]) + annual_contrib, 0.0
            )

        for yr in range(retire_years):
            col = accum_years + yr
            withdrawal = base_withdrawal * (1 + g) ** yr
            paths[:, col + 1] = np.maximum(
                paths[:, col] * (1 + returns[:, col]) - withdrawal, 0.0
            )

    ages = list(range(inputs.current_age, inputs.life_expectancy + 1))
    retirement_values = paths[:, accum_years]
    final_values = paths[:, -1]
    depleted_count = int(np.sum(final_values <= 0))

    percentile_paths = {
        p: np.percentile(paths, p, axis=0).tolist()
        for p in [10, 25, 50, 75, 90]
    }
    survival_rates = (paths > 0).mean(axis=0).tolist()

    return MonteCarloResult(
        success_rate=float((n - depleted_count) / n),
        n_simulations=n,
        depleted_count=depleted_count,
        ages=ages,
        percentile_paths=percentile_paths,
        median_at_retirement=float(np.median(retirement_values)),
        p10_at_retirement=float(np.percentile(retirement_values, 10)),
        p90_at_retirement=float(np.percentile(retirement_values, 90)),
        median_final=float(np.median(final_values)),
        p10_final=float(np.percentile(final_values, 10)),
        p90_final=float(np.percentile(final_values, 90)),
        survival_rates=survival_rates,
    )
