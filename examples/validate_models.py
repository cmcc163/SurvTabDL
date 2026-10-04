"""Integration smoke test, including optional real TabPFN weights."""
import argparse
import json
import platform
import tempfile
from pathlib import Path
import numpy as np
from survtabdl import SurvivalEstimator, make_survival_data
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-tabpfn", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("demo_output/validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    X, t, e = make_survival_data(60)
    names = ["MLP", "Cox", "FT-Transformer", "SAINT", "NODE", "TabNet", "XGBoost"]
    if args.with_tabpfn:
        names.append("TabPFN")
    report = {"python": platform.python_version(), "torch": torch.__version__, "models": {}}
    with tempfile.TemporaryDirectory(prefix="survtabdl-check-") as directory:
        for name in names:
            params = {"hidden_dim": 8, "dim": 4, "depth": 1, "heads": 1,
                      "head_hidden": 8, "layer_dim": 4, "num_layers": 1,
                      "tree_depth": 2, "n_d": 4, "n_steps": 2}
            if name == "XGBoost":
                params = {"max_depth": 2}
            model = SurvivalEstimator(name, ["group"], params=params, epochs=2, reference_size=3)
            model.fit(X.iloc[:45], t[:45], e[:45])
            query = X.iloc[45:48]
            pred = model.predict(query)
            probability = model.predict_event_probability(query, [5, 10])
            assert np.isfinite(pred).all()
            assert ((probability >= 0) & (probability <= 1)).all()
            assert (np.diff(probability, axis=1) >= 0).all()
            path = Path(directory) / f"{name}.joblib"
            model.save(path)
            restored = model.load(path)
            np.testing.assert_allclose(restored.predict(query), pred, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(restored.predict_event_probability(query, [5, 10]), probability,
                                       rtol=1e-6, atol=1e-6)
            report["models"][name] = {"passed": True, "max_prediction_difference": float(np.max(np.abs(restored.predict(query) - pred)))}
            print(f"{name}: passed")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
