"""Train, save and restore a model on an entirely simulated split."""
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from survtabdl import SurvivalEstimator, make_survival_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="MLP")
    args = parser.parse_args()
    X, time, event = make_survival_data()
    train, test = train_test_split(np.arange(len(X)), test_size=0.2, stratify=event, random_state=42)
    model = SurvivalEstimator(args.model, categorical_features=["group"], epochs=30)
    model.fit(X.iloc[train], time[train], event[train])
    model.save("demo_output/model.joblib")
    restored = SurvivalEstimator.load("demo_output/model.joblib")
    np.testing.assert_allclose(model.predict(X.iloc[test]), restored.predict(X.iloc[test]))
    print(f"Test C-index: {model.score(X.iloc[test], time[test], event[test]):.3f}")
    print(model.predict_event_probability(X.iloc[test[:3]], [5, 10]))


if __name__ == "__main__":
    main()
