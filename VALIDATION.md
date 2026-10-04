# Validation

Validated locally in the Conda `Codex` environment with Python 3.11.15 and PyTorch 2.4.1 (CPU).

- 15 automated tests passed: tied Cox risk sets, Breslow probabilities, concordance, preprocessing, cross-validation, SQLite Optuna trials and persistence.
- All eight models passed the simulated integration workflow, including real TabPFN 2.2.1 / tabpfn-extensions 0.1.6 weights.
- Saved/restored predictions matched exactly for all eight models in this workflow.
- Source distribution and wheel built successfully.

Python 3.10.1, GPU training, large-cohort performance and equivalence to the manuscript's stored checkpoints have not been tested for this standalone package. The smoke tests verify execution and consistency, not scientific equivalence or clinical validation.

Reproduce:

```bash
pip install -e ".[dev,tabnet,xgboost,tabpfn,tuning]"
pytest
python examples/validate_models.py --with-tabpfn
python -m build
```
