import numpy as np
from survtabdl import SurvivalEstimator, make_survival_data, cross_validate


def test_cross_validation_produces_complete_oof_scores():
    X, time, event = make_survival_data(90)
    model = SurvivalEstimator("MLP", ["group"], epochs=2, params={"hidden_dim": 8})
    result = cross_validate(model, X, time, event, n_splits=3)
    assert result["test_c_index"].shape == (3,)
    assert result["oof_log_risk"].shape == (90,)
    assert np.isfinite(result["oof_log_risk"]).all()
    assert not hasattr(model, "preprocessor_")


def test_tuning_restores_settings_and_sqlite_trials(tmp_path):
    import pytest
    pytest.importorskip("optuna")
    from survtabdl import tune
    X, time, event = make_survival_data(60)
    model = SurvivalEstimator("MLP", ["group"], epochs=2)
    best, study = tune(model, X, time, event,
                       lambda trial: {"learning_rate": trial.suggest_float("lr", 0.001, 0.002)},
                       n_trials=1, n_splits=2, storage=f"sqlite:///{(tmp_path / 'trials.db').as_posix()}",
                       study_name="demo")
    assert len(study.trials) == 1
    assert np.isfinite(best.predict(X)).all()
    assert best.learning_rate == study.best_trial.params["lr"]
