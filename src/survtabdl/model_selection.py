"""Train-fold preprocessing during cross-validation and optional tuning."""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from .survival import validate_target


def cross_validate(estimator, X, time, event, n_splits=5, random_state=42):
    """Return fold C-indices and out-of-fold log hazards.

    Each fold starts with a fresh estimator, including fresh preprocessing,
    model weights, and baseline hazard. C-indices are summarized per fold;
    raw hazards from independently fitted folds are not pooled for scoring.
    """
    time, event = validate_target(time, event, len(X))
    if np.bincount(event.astype(int), minlength=2).min() < n_splits:
        raise ValueError("Each event class must contain at least n_splits subjects.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores, oof = [], np.zeros(len(X))
    take = lambda idx: X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx]
    for train, test in cv.split(np.zeros(len(time)), event):
        model = clone(estimator).fit(take(train), time[train], event[train])
        oof[test] = model.predict(take(test))
        scores.append(model.score(take(test), time[test], event[test]))
    return {"test_c_index": np.asarray(scores), "mean_c_index": float(np.mean(scores)), "oof_log_risk": oof}


def tune(estimator, X, time, event, search_space, n_trials=20, n_splits=5, storage=None, study_name=None):
    """Tune estimator settings with a user-supplied Optuna search function.

    search_space(trial) returns constructor settings such as learning_rate or
    params. SQLite storage may be provided as sqlite:///trials.db. Evaluate
    the resulting model on a separately held-out test set.
    """
    try:
        import optuna
    except ImportError as exc:
        raise ImportError('Install SurvTabDL with the "tuning" extra.') from exc
    study = optuna.create_study(direction="maximize", storage=storage, study_name=study_name,
                                load_if_exists=study_name is not None,
                                sampler=optuna.samplers.TPESampler(seed=estimator.random_state))
    def objective(trial):
        settings = search_space(trial)
        trial.set_user_attr("estimator_settings", settings)
        candidate = clone(estimator).set_params(**settings)
        return cross_validate(candidate, X, time, event, n_splits=n_splits,
                              random_state=estimator.random_state)["mean_c_index"]
    study.optimize(objective, n_trials=n_trials)
    best = clone(estimator).set_params(**study.best_trial.user_attrs["estimator_settings"])
    return best.fit(X, time, event), study
