import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from survtabdl import SurvivalEstimator, make_survival_data
from survtabdl.preprocessing import TabularPreprocessor

torch.set_num_threads(1)


@pytest.mark.parametrize("name", ["MLP", "Cox", "FT-Transformer", "SAINT", "NODE", "TabNet", "XGBoost"])
def test_train_probability_and_saved_prediction(name, tmp_path):
    X, t, e = make_survival_data(100)
    params = {"hidden_dim": 8, "dim": 4, "depth": 1, "heads": 1,
              "head_hidden": 8, "layer_dim": 4, "num_layers": 1,
              "tree_depth": 2, "n_d": 4, "n_steps": 2}
    if name == "XGBoost":
        params = {"max_depth": 2}
    model = SurvivalEstimator(name, ["group"], params=params, epochs=3, reference_size=3)
    model.fit(X.iloc[:80], t[:80], e[:80], validation_data=(X.iloc[80:], t[80:], e[80:]))
    pred = model.predict(X.iloc[80:])
    assert pred.shape == (20,) and np.isfinite(pred).all()
    probs = model.predict_event_probability(X.iloc[80:], [0, 1, 5, 10])
    assert probs.shape == (20, 4)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert (np.diff(probs, axis=1) >= 0).all()
    model.save(tmp_path / "model.joblib")
    loaded = model.load(tmp_path / "model.joblib")
    np.testing.assert_allclose(loaded.predict(X.iloc[80:]), pred)
    np.testing.assert_allclose(loaded.predict_event_probability(X.iloc[80:], [5, 10]), probs[:, 2:])
    assert clone(model).model == name
    if name == "SAINT":
        np.testing.assert_allclose(model.predict(X.iloc[80:81]), pred[:1], rtol=1e-6, atol=1e-6)


def test_train_only_preprocessing_unknowns_and_schema():
    X = pd.DataFrame({"v": [1., 3., np.nan], "g": ["A", "B", "A"]})
    p = TabularPreprocessor(["g"]).fit(X)
    out = p.transform(pd.DataFrame({"v": [np.nan, 100.], "g": ["new", None]}))
    np.testing.assert_allclose(out[:, 1], 0)
    np.testing.assert_allclose(out[0, 0], 0)
    np.testing.assert_allclose(p.medians_, [2])
    with pytest.raises(ValueError):
        p.transform(X[["g", "v"]])


def test_invalid_survival_input():
    X, t, e = make_survival_data(20)
    with pytest.raises(ValueError):
        SurvivalEstimator().fit(X, t, np.zeros(20))
    with pytest.raises(ValueError):
        SurvivalEstimator().fit(X, -t, e)
