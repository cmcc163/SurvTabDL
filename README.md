# SurvTabDL

Tabular deep learning for right-censored survival prediction.

Models: **MLP, TabNet, NODE, FT-Transformer, SAINT, TabPFN**, with **Cox** and **XGBoost** comparators. All return log-relative hazards and survival/event probabilities through one interface.

## Install

Python 3.10 or later (the manuscript environment uses Python 3.10.1).

```bash
pip install -e ".[tabnet,xgboost]"
```

Optional TabPFN embedding backend:

```bash
pip install -e ".[tabpfn]"
```

## Run the simulated example

```bash
survtabdl demo --model MLP --epochs 30
survtabdl demo --model SAINT --epochs 30
```

The demo generates data, splits training/test subjects, fits preprocessing and a survival model, reports test concordance, and saves the model.

## Python API

```python
from sklearn.model_selection import train_test_split
from survtabdl import SurvivalEstimator, make_survival_data

X, time, event = make_survival_data(n_samples=300)
train, test = train_test_split(range(len(X)), test_size=0.2,
                              random_state=42, stratify=event)
model = SurvivalEstimator("SAINT", categorical_features=["group"], epochs=30)
model.fit(X.iloc[train], time[train], event[train])
risk = model.predict(X.iloc[test])
survival = model.predict_survival(X.iloc[test], times=[5, 10])
probability = model.predict_event_probability(X.iloc[test], times=[5, 10])
model.save("model.joblib")
restored = SurvivalEstimator.load("model.joblib")
```

`X` is a DataFrame or 2D array. `time` contains positive follow-up times; `event` uses 1 for events and 0 for censoring. Prediction horizons use the same units as `time`. Prediction columns must match training names and order. For array input, supply categorical column indices. Optional early stopping: `fit(X, time, event, validation_data=(X_val, time_val, event_val))`.

## Train from CSV

```bash
survtabdl train --data example.csv --model MLP --time-column time --event-column event --categorical group --output model.joblib
```

All other CSV columns become predictors. The command reserves 20% for testing.

## Model settings

Pass architecture settings through `params={...}`:

| Model | Main settings |
| --- | --- |
| MLP / TabPFN survival head | `hidden_dim`, `n_layers`, `dropout` |
| SAINT / FT-Transformer | `dim`, `depth`, `heads`, `dropout`, `embedding_hidden`, `head_hidden` |
| NODE | `layer_dim`, `num_layers`, `tree_depth` |
| TabNet | `n_d`, `n_steps`, `virtual_batch_size`, `lambda_sparse` |
| XGBoost | `max_depth`, `eta`, other native booster settings |

Neural models use the Cox partial likelihood with Breslow ties. `batch_size=None` uses full risk sets; a finite batch size uses approximate batch-local risk sets. Cox is a regularized linear log-hazard head optimized with the same objective. Numeric inputs are median-imputed and standardized using training data; categorical inputs use train-fitted integer codes, with 0 reserved for missing/unseen categories. MLP, NODE and Cox treat these codes as numeric values; one-hot encode nominal predictors externally for these models when appropriate.

FT-Transformer is the manuscript's feature-token column-attention implementation using the SAINT component library, rather than a byte-for-byte reproduction of the original FT-Transformer implementation. SAINT adds row attention; inference uses a fixed training reference context (`reference_size=32`) and scores queries individually. This package is intended for new training and does not import manuscript checkpoints automatically.

TabPFN combines event-classifier and log-follow-up-regressor embeddings with a Cox-trained MLP. It is not a native censored-outcome TabPFN model. The optional backend downloads upstream weights on first use. Offline tests use a stand-in embedding backend; `examples/validate_models.py --with-tabpfn` also exercises real weights.

## Cross-validation and tuning

`cross_validate(model, X, time, event, n_splits=5)` reports fold C-indices and out-of-fold scores with fresh preprocessing in each fold. Install `.[tuning]` to use `tune`: supply an Optuna search function, optionally a SQLite `storage`, and the desired number of trials (default 20). Reserve an independent test set for the tuned model.

## Project files

- `src/survtabdl/`: estimator, architectures, preprocessing, survival utilities and CLI.
- `examples/`: runnable simulated workflow.
- `tests/`: survival mathematics and model integration checks.
- `licenses/`, `THIRD_PARTY.md`: upstream attribution.

```bash
pip install -e ".[dev,tabnet,xgboost]"
pytest
python -m build
```
