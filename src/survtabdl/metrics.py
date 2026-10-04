"""Harrell concordance for right-censored survival outcomes."""
import numpy as np
from .survival import validate_target


def concordance_index(time, event, risk):
    """Higher risk should correspond to an earlier observed event.

    Equal-time event/censor pairs are comparable; equal-time event/event pairs
    are excluded. Risk ties receive half credit. Complexity is O(n log n).
    """
    t, e = validate_target(time, event)
    r = np.asarray(risk, dtype=float)
    if r.shape != t.shape or not np.isfinite(r).all():
        raise ValueError("risk must be a finite vector matching time.")
    levels, rank = np.unique(r, return_inverse=True)
    tree = np.zeros(len(levels) + 1, dtype=np.int64)
    def add(i):
        i += 1
        while i < len(tree):
            tree[i] += 1
            i += i & -i
    def prefix(i):
        total = 0
        while i:
            total += tree[i]
            i -= i & -i
        return total
    order = np.argsort(-t, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(t[order])) + 1)
    total, comparable, concordant = 0, 0, 0.0
    for group in groups:
        censored = group[~e[group]]
        for i in censored:
            add(rank[i])
            total += 1
        for i in group[e[group]]:
            less = prefix(rank[i])
            equal = prefix(rank[i] + 1) - less
            concordant += less + 0.5 * equal
            comparable += total
        for i in group[e[group]]:
            add(rank[i])
            total += 1
    if not comparable:
        raise ValueError("No comparable pairs for concordance.")
    return float(concordant / comparable)
