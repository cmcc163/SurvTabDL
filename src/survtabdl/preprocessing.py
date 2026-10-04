"""Fold-local numeric imputation, scaling and categorical encoding."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TabularPreprocessor:
    def __init__(self, categorical_features=()):
        self.categorical_features = tuple(categorical_features or ())

    def _frame(self, X, fit=False):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            X = np.asarray(X)
            if X.ndim != 2:
                raise ValueError("X must be a two-dimensional table.")
            frame = pd.DataFrame(X)
        if frame.columns.has_duplicates:
            raise ValueError("Feature names must be unique.")
        if fit:
            self.columns_ = list(frame.columns)
        elif list(frame.columns) != self.columns_:
            raise ValueError("Prediction columns must match training names and order.")
        if len(frame) == 0 or len(frame.columns) == 0:
            raise ValueError("X must contain rows and features.")
        return frame

    def fit(self, X):
        frame = self._frame(X, fit=True)
        if not set(self.categorical_features).issubset(frame.columns):
            raise ValueError("Unknown categorical feature name/index.")
        self.cat_idx_ = [self.columns_.index(c) for c in self.categorical_features]
        self.num_idx_ = [i for i in range(len(self.columns_)) if i not in self.cat_idx_]
        self.maps_ = {}
        for c in self.categorical_features:
            self.maps_[c] = {v: i + 1 for i, v in enumerate(pd.unique(frame[c].dropna()))}
        self.cat_dims_ = [len(self.maps_[c]) + 1 for c in self.categorical_features]
        if self.num_idx_:
            numeric = frame.iloc[:, self.num_idx_].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            if np.isinf(numeric).any():
                raise ValueError("Numeric features must not contain infinity.")
            if np.isnan(numeric).all(axis=0).any():
                raise ValueError("A numeric feature is entirely missing in training data.")
            self.medians_ = np.nanmedian(numeric, axis=0)
            self.scaler_ = StandardScaler().fit(np.where(np.isnan(numeric), self.medians_, numeric))
        return self

    def transform(self, X):
        frame = self._frame(X)
        out = np.zeros(frame.shape, dtype=np.float32)
        for c, i in zip(self.categorical_features, self.cat_idx_):
            out[:, i] = frame[c].map(self.maps_[c]).fillna(0).to_numpy()
        if self.num_idx_:
            numeric = frame.iloc[:, self.num_idx_].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            if np.isinf(numeric).any():
                raise ValueError("Numeric features must not contain infinity.")
            out[:, self.num_idx_] = self.scaler_.transform(np.where(np.isnan(numeric), self.medians_, numeric))
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)
