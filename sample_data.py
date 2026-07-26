from __future__ import annotations

import numpy as np
import pandas as pd


def make_rotation_sample() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    inertia = np.linspace(0.010, 0.090, 15)
    kappa = 0.080
    theory = 2 * np.pi * np.sqrt(inertia / kappa)
    systematic_residual = 0.06 * np.sin(45 * inertia) + 0.45 * inertia
    noise = rng.normal(0.0, 0.012, size=len(inertia))
    measured = theory + systematic_residual + noise
    return pd.DataFrame({"x": inertia, "y_exp": measured})


def make_multivariable_sample() -> pd.DataFrame:
    """시간과 전력을 사용하는 합성 제거율 데이터."""
    rng = np.random.default_rng(2026)
    times = np.arange(0.0, 181.0, 20.0)
    powers = np.array([100.0, 130.0, 160.0, 185.0])
    rows = []
    beta0 = -6.05
    beta1 = 0.0084

    for power in powers:
        k = np.exp(beta0 + beta1 * power)
        theory = 100.0 * (1.0 - np.exp(-k * times))
        residual = (
            4.0 * np.sin(times / 55.0)
            * ((power - powers.min()) / (powers.max() - powers.min()))
        )
        noise = rng.normal(0.0, 0.8, size=len(times))
        measured = np.clip(theory + residual + noise, 0.0, 100.0)
        for time, value in zip(times, measured):
            rows.append({"time": time, "power": power, "y_exp": value})

    return pd.DataFrame(rows)
