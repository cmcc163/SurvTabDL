"""Generate reproducible, entirely simulated survival observations."""
import numpy as np
import pandas as pd


def make_survival_data(n_samples=300, random_state=42):
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(n_samples, 6))
    group = rng.integers(0, 2, n_samples)
    log_risk = 0.8 * x[:, 0] - 0.6 * x[:, 1] + 0.4 * group
    event_time = rng.exponential(8 / np.exp(log_risk))
    censor_time = rng.uniform(1, 15, n_samples)
    X = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(6)])
    X["group"] = np.where(group, "B", "A")
    return X, np.minimum(event_time, censor_time), (event_time <= censor_time).astype(int)
