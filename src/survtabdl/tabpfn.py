"""Optional TabPFN event/time embeddings followed by a Cox-trained MLP."""
import numpy as np


class TabPFNFeatures:
    def __init__(self, device="cpu", random_state=42):
        self.device = device
        self.random_state = random_state

    def _extract(self, extractor, target, source):
        import torch
        with torch.no_grad():
            value = extractor.get_embeddings(self.x_, target, self.query_, data_source=source)
        if isinstance(value, (list, tuple)):
            value = value[0]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 3:
            value = value.mean(axis=0)
        if value.ndim != 2 or len(value) != len(self.query_):
            raise RuntimeError("Unsupported TabPFN embedding shape.")
        return value.astype(np.float32)

    def fit_transform(self, X, time, event):
        try:
            from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
            from tabpfn_extensions.embedding import TabPFNEmbedding
        except ImportError as exc:
            raise ImportError('Install SurvTabDL with the "tabpfn" extra.') from exc
        if len(X) < 10 or np.bincount(event.astype(int), minlength=2).min() < 5:
            raise ValueError("TabPFN embedding extraction requires at least five subjects in each event class.")
        self.x_, self.event_, self.time_ = X.copy(), event.astype(int), np.log1p(time)
        self.clf_ = TabPFNEmbedding(tabpfn_clf=TabPFNClassifier(device=self.device, random_state=self.random_state, n_estimators=4), n_fold=5)
        self.reg_ = TabPFNEmbedding(tabpfn_reg=TabPFNRegressor(device=self.device, random_state=self.random_state, n_estimators=4), n_fold=5)
        self.query_ = X
        return np.concatenate([self._extract(self.clf_, self.event_, "train"), self._extract(self.reg_, self.time_, "train")], axis=1)

    def transform(self, X):
        self.query_ = X
        return np.concatenate([self._extract(self.clf_, self.event_, "test"), self._extract(self.reg_, self.time_, "test")], axis=1)
