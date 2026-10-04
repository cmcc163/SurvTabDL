"""Minimal synthetic demo and CSV training entry point."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from . import MODELS, SurvivalEstimator, make_survival_data


def main():
    parser = argparse.ArgumentParser(description="SurvTabDL survival training")
    parser.add_argument("command", choices=["demo", "train"])
    parser.add_argument("--model", choices=MODELS, default="MLP")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--time-column", default="time")
    parser.add_argument("--event-column", default="event")
    parser.add_argument("--categorical", nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("demo_output/model.joblib"))
    args = parser.parse_args()
    if args.command == "demo":
        X, time, event = make_survival_data()
        categorical = ["group"]
    else:
        if args.data is None:
            parser.error("train requires --data")
        frame = pd.read_csv(args.data)
        time, event = frame.pop(args.time_column).to_numpy(), frame.pop(args.event_column).to_numpy()
        X, categorical = frame, args.categorical
    train, test = train_test_split(range(len(X)), test_size=0.2, random_state=42, stratify=event)
    model = SurvivalEstimator(args.model, categorical_features=categorical, epochs=args.epochs)
    model.fit(X.iloc[train], time[train], event[train])
    model.save(args.output)
    print(f"{args.model}: test C-index={model.score(X.iloc[test], time[test], event[test]):.3f}")
    print(f"Saved: {args.output.resolve()}")
