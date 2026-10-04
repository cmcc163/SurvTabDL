"""Unified train/predict/persist interface for right-censored outcomes."""
import copy
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from .networks import build_network
from .preprocessing import TabularPreprocessor
from .survival import BreslowBaseline, cox_loss, validate_target


class SurvivalEstimator(BaseEstimator):
    """Fit a proportional-hazards tabular model.

    Parameters
    ----------
    model : str
        MLP, TabNet, NODE, FT-Transformer, SAINT, TabPFN, Cox or XGBoost.
    categorical_features : sequence
        DataFrame column names, or integer column indices for array input.
    params : dict, optional
        Architecture-specific settings; see README.
    batch_size : int or None
        None uses full training risk sets. Mini-batches approximate Cox risk
        sets. SAINT also attends across subjects within each training batch.
    reference_size : int
        Fixed train-derived context for SAINT inference. Each query is scored
        separately with that context, so predictions do not depend on other
        test subjects or caller batch size.
    """
    def __init__(self, model="MLP", categorical_features=None, params=None,
                 epochs=100, learning_rate=1e-3, weight_decay=1e-4,
                 batch_size=None, patience=15, device="cpu", random_state=42,
                 reference_size=32):
        self.model = model
        self.categorical_features = categorical_features
        self.params = params
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.reference_size = reference_size

    def fit(self, X, time, event, validation_data=None):
        """Fit preprocessing, log-hazard model and baseline on training data.

        validation_data is an optional (X_val, time_val, event_val) tuple.
        It selects the best epoch but never fits preprocessing or the baseline.
        Time units are supplied by the caller and preserved in predictions.
        """
        from . import MODELS
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.epochs < 1 or self.patience < 1 or self.reference_size < 1:
            raise ValueError("epochs, patience and reference_size must be positive.")
        if self.batch_size is not None and self.batch_size < 2:
            raise ValueError("batch_size must be None or at least 2.")
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        time, event = validate_target(time, event, len(X))
        self.preprocessor_ = TabularPreprocessor(self.categorical_features)
        z = self.preprocessor_.fit_transform(X)
        self.n_features_in_ = z.shape[1]
        self.device_ = torch.device(self.device)
        params = dict(self.params or {})
        val = None
        if validation_data is not None:
            xv, tv, ev = validation_data
            tv, ev = validate_target(tv, ev, len(xv))
            val = (self.preprocessor_.transform(xv), tv, ev)
        self.embedding_ = None
        if self.model == "TabPFN":
            from .tabpfn import TabPFNFeatures
            self.embedding_ = TabPFNFeatures(device=self.device, random_state=self.random_state)
            z = self.embedding_.fit_transform(z, time, event)
            if val is not None:
                val = (self.embedding_.transform(val[0]), val[1], val[2])
        self.history_ = []
        if self.model == "XGBoost":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise ImportError('Install SurvTabDL with the "xgboost" extra.') from exc
            labels = np.where(event, time, -time)
            config = {"max_depth": 3, "eta": self.learning_rate, **params,
                      "objective": "survival:cox", "seed": self.random_state,
                      "eval_metric": "cox-nloglik", "nthread": 1}
            train = xgb.DMatrix(z, label=labels)
            evals = []
            if val is not None:
                evals = [(xgb.DMatrix(val[0], label=np.where(val[2], val[1], -val[1])), "validation")]
            self.network_ = xgb.train(config, train, num_boost_round=self.epochs,
                                      evals=evals, early_stopping_rounds=self.patience if evals else None,
                                      verbose_eval=False)
        else:
            cat_idx = [] if self.embedding_ else self.preprocessor_.cat_idx_
            cat_dims = [] if self.embedding_ else self.preprocessor_.cat_dims_
            self.network_ = build_network("MLP" if self.embedding_ else self.model,
                                           z.shape[1], cat_idx, cat_dims, params).to(self.device_)
            self.reference_ = z[:min(self.reference_size, len(z))].copy()
            self._train(z, time, event, val)
        scores = self._risk(z)
        if not np.isfinite(scores).all():
            raise RuntimeError("Training produced nonfinite risk scores.")
        self.baseline_ = BreslowBaseline().fit(scores, time, event)
        return self

    def _train(self, z, time, event, val):
        x = torch.as_tensor(z, device=self.device_)
        t = torch.as_tensor(time, dtype=torch.float64, device=self.device_)
        e = torch.as_tensor(event, device=self.device_)
        if self.model == "NODE":
            # Initialize tree thresholds from the training cohort only.
            with torch.no_grad():
                self.network_(x)
        optimizer = torch.optim.AdamW(self.network_.parameters(), lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        best, state, stale = float("inf"), None, 0
        for epoch in range(self.epochs):
            self.network_.train()
            if self.batch_size is None or self.batch_size >= len(x):
                batches = [torch.arange(len(x), device=self.device_)]
            else:
                batches = torch.randperm(len(x), device=self.device_).split(self.batch_size)
                if len(batches[-1]) == 1:
                    batches = list(batches[:-2]) + [torch.cat(batches[-2:])]
            losses = []
            for idx in batches:
                if not e[idx].any():
                    continue
                optimizer.zero_grad()
                risk = self.network_(x[idx]).reshape(-1)
                loss = cox_loss(risk, t[idx], e[idx])
                if self.model == "TabNet":
                    loss = loss - (self.params or {}).get("lambda_sparse", 1e-3) * self.network_.sparsity_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; check data and learning rate.")
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            train_loss = float(np.mean(losses))
            monitor = train_loss
            if val is not None:
                risk = torch.as_tensor(self._risk(val[0]), dtype=torch.float64)
                monitor = cox_loss(risk, torch.as_tensor(val[1]), torch.as_tensor(val[2])).item()
            self.history_.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": monitor if val else None})
            if monitor < best:
                best, stale = monitor, 0
                state = copy.deepcopy(self.network_.state_dict())
                self.best_epoch_ = epoch + 1
            else:
                stale += 1
            # Without validation, retain the last epoch rather than comparing
            # dropout-perturbed training objectives across epochs.
            if val is None:
                state = copy.deepcopy(self.network_.state_dict())
                self.best_epoch_ = epoch + 1
            elif stale >= self.patience:
                break
        self.network_.load_state_dict(state)
        self.network_.eval()

    def _risk(self, z):
        if self.model == "XGBoost":
            import xgboost as xgb
            limit = (0, self.network_.best_iteration + 1) if hasattr(self.network_, "best_iteration") else (0, 0)
            return self.network_.predict(xgb.DMatrix(z), output_margin=True, iteration_range=limit).astype(float)
        self.network_.eval()
        with torch.no_grad():
            if self.model == "SAINT":
                context = torch.as_tensor(self.reference_, device=self.device_)
                result = [self.network_(torch.cat([context, torch.as_tensor(row[None], device=self.device_)], dim=0))[-1].item() for row in z]
                return np.asarray(result)
            parts = [self.network_(torch.as_tensor(part, device=self.device_)).reshape(-1).cpu().numpy()
                     for part in np.array_split(z, max(1, (len(z) + 255) // 256))]
            return np.concatenate(parts).astype(float)

    def predict(self, X):
        """Return log-relative hazards; larger values indicate higher risk."""
        if not hasattr(self, "baseline_"):
            raise NotFittedError("Call fit before predict.")
        z = self.preprocessor_.transform(X)
        if self.embedding_ is not None:
            z = self.embedding_.transform(z)
        return self._risk(z)

    def predict_survival(self, X, times):
        """Return a (n_subjects, n_times) matrix of survival probabilities."""
        return self.baseline_.survival(self.predict(X), times)

    def predict_event_probability(self, X, times):
        """Return cumulative event probabilities at caller-specified times."""
        return 1 - self.predict_survival(X, times)

    def score(self, X, time, event):
        from .metrics import concordance_index
        return concordance_index(time, event, self.predict(X))

    def save(self, path):
        """Save model, preprocessing and baseline together to a trusted file."""
        if not hasattr(self, "baseline_"):
            raise NotFittedError("Call fit before save.")
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = copy.deepcopy(self)
        if isinstance(obj.network_, torch.nn.Module):
            obj.network_.cpu()
            obj.device_ = torch.device("cpu")
        joblib.dump(obj, path)

    @staticmethod
    def load(path, device="cpu"):
        """Load a trusted SurvTabDL file. Never load untrusted pickle files."""
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, SurvivalEstimator):
            raise TypeError("Not a SurvTabDL estimator.")
        obj.device_ = torch.device(device)
        if isinstance(obj.network_, torch.nn.Module):
            obj.network_.to(obj.device_).eval()
        return obj
