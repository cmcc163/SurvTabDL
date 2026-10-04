"""Offline adapter contract test; no real pretrained weights are used."""
import sys
import types
import numpy as np
from survtabdl import SurvivalEstimator, make_survival_data


class FakeFoundationModel:
    def __init__(self, **kwargs):
        pass


class FakeEmbedding:
    def __init__(self, **kwargs):
        pass

    def get_embeddings(self, X_train, y_train, X, data_source):
        # Match the upstream ensemble/subject/embedding axis convention.
        return np.stack([X[:, :3], X[:, :3]], axis=0)


def test_optional_tabpfn_adapter_contract(monkeypatch, tmp_path):
    module = types.ModuleType("tabpfn_extensions")
    module.TabPFNClassifier = module.TabPFNRegressor = FakeFoundationModel
    embedding = types.ModuleType("tabpfn_extensions.embedding")
    embedding.TabPFNEmbedding = FakeEmbedding
    monkeypatch.setitem(sys.modules, "tabpfn_extensions", module)
    monkeypatch.setitem(sys.modules, "tabpfn_extensions.embedding", embedding)
    X, time, event = make_survival_data(80)
    model = SurvivalEstimator("TabPFN", ["group"], epochs=2)
    model.fit(X.iloc[:60], time[:60], event[:60])
    pred = model.predict(X.iloc[60:])
    assert pred.shape == (20,) and np.isfinite(pred).all()
    model.save(tmp_path / "embedding.joblib")
    np.testing.assert_allclose(model.load(tmp_path / "embedding.joblib").predict(X.iloc[60:]), pred)
