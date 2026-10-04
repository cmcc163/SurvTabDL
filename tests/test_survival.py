import numpy as np
import pytest
import torch
from survtabdl.survival import cox_loss, BreslowBaseline
from survtabdl import concordance_index


def test_tied_loss_matches_explicit_risk_sets_and_is_permutation_invariant():
    t = torch.tensor([1., 2., 2., 3., 4.], dtype=torch.float64)
    e = torch.tensor([1, 1, 0, 1, 0])
    r = torch.tensor([0.2, -0.3, 0.8, 0.4, 0.1], dtype=torch.float64, requires_grad=True)
    expected = torch.stack([torch.logsumexp(r[t >= t[i]], 0) - r[i] for i in range(5) if e[i]]).mean()
    loss = cox_loss(r, t, e)
    torch.testing.assert_close(loss, expected)
    idx = torch.tensor([3, 2, 4, 1, 0])
    torch.testing.assert_close(cox_loss(r[idx], t[idx], e[idx]), expected)
    loss.backward()
    assert torch.isfinite(r.grad).all()


def test_breslow_known_solution_shift_invariance():
    t, e, r = np.array([1., 2., 2., 3.]), np.array([1, 1, 1, 0]), np.zeros(4)
    b = BreslowBaseline().fit(r, t, e)
    np.testing.assert_allclose(b.hazard_, [0.25, 0.25 + 2 / 3])
    s = b.survival(r, [0, 1, 2, 4])
    np.testing.assert_allclose(s[:, 0], 1)
    np.testing.assert_allclose(s, BreslowBaseline().fit(r + 1000, t, e).survival(r + 1000, [0, 1, 2, 4]))
    assert (np.diff(s, axis=1) <= 0).all()


def test_concordance_and_ties():
    assert concordance_index([1, 2, 3], [1, 1, 0], [3, 2, 1]) == 1
    assert concordance_index([1, 2, 3], [1, 1, 0], [1, 2, 3]) == 0
    assert concordance_index([1, 1], [1, 0], [2, 1]) == 1
    assert concordance_index([1, 2], [1, 0], [1, 1]) == 0.5
    with pytest.raises(ValueError):
        concordance_index([1, 1], [1, 1], [1, 2])
