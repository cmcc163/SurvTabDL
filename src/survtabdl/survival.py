"""Cox partial likelihood and train-fitted Breslow baseline hazard."""
import numpy as np
import torch


def validate_target(time, event, n=None):
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event)
    if time.ndim != 1 or event.ndim != 1 or len(time) != len(event):
        raise ValueError("time and event must be one-dimensional arrays of equal length.")
    if n is not None and len(time) != n:
        raise ValueError("X, time and event must have the same number of rows.")
    if not np.isfinite(time).all() or (time <= 0).any():
        raise ValueError("Follow-up times must be finite and positive.")
    if not np.isin(event, [0, 1]).all() or not event.any():
        raise ValueError("event must contain only 0/1 and at least one observed event.")
    return time, event.astype(bool)


def cox_loss(log_risk, time, event):
    """Negative Cox partial log-likelihood with Breslow handling of ties.

    The risk set includes every supplied subject at risk at each event time.
    No quadratic risk-set matrix is allocated.
    """
    order = torch.argsort(time, descending=True, stable=True)
    t, r, e = time[order], log_risk.reshape(-1)[order], event[order].bool()
    _, counts = torch.unique_consecutive(t, return_counts=True)
    ends = torch.cumsum(counts, 0) - 1
    log_denominator = torch.repeat_interleave(torch.logcumsumexp(r, 0)[ends], counts)
    return (log_denominator[e] - r[e]).mean()


class BreslowBaseline:
    """Baseline cumulative hazard fitted exclusively on training subjects."""
    def fit(self, log_risk, time, event):
        r = np.asarray(log_risk, dtype=np.float64)
        self.offset_ = float(np.max(r))
        order = np.argsort(time, kind="stable")
        t, e = np.asarray(time)[order], np.asarray(event)[order]
        w = np.exp(r[order] - self.offset_)
        unique, starts = np.unique(t, return_index=True)
        deaths = np.add.reduceat(e.astype(float), starts)
        at_risk = np.cumsum(w[::-1])[::-1][starts]
        mask = deaths > 0
        self.times_ = unique[mask]
        self.hazard_ = np.cumsum(deaths[mask] / np.maximum(at_risk[mask], np.finfo(float).tiny))
        return self

    def survival(self, log_risk, times):
        times = np.atleast_1d(np.asarray(times, dtype=float))
        if times.ndim != 1 or not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("Prediction times must be finite, nonnegative and one-dimensional.")
        idx = np.searchsorted(self.times_, times, side="right") - 1
        h = np.where(idx >= 0, self.hazard_[np.maximum(idx, 0)], 0)
        # Evaluate on the log scale to avoid overflow in the cumulative hazard.
        with np.errstate(divide="ignore", over="ignore"):
            log_h = np.log(h)[None, :] + np.asarray(log_risk)[:, None] - self.offset_
            return np.exp(-np.exp(np.clip(log_h, -745, 710)))
