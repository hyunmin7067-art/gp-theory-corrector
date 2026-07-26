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
