"""Public survival prediction API."""
from .estimator import SurvivalEstimator
from .metrics import concordance_index
from .synthetic import make_survival_data
from .model_selection import cross_validate, tune

__version__ = "0.1.0"
MODELS = ("MLP", "TabNet", "NODE", "FT-Transformer", "SAINT", "TabPFN", "Cox", "XGBoost")
__all__ = ["SurvivalEstimator", "concordance_index", "make_survival_data", "MODELS", "cross_validate", "tune"]
