"""Validator-gated correction stage for the receipt pipeline."""

from .acceptance import evaluate_candidate
from .profile import CorrectionProfile, StrategyConfig, load_correction_profile

__all__ = [
    "CorrectionProfile",
    "StrategyConfig",
    "evaluate_candidate",
    "load_correction_profile",
]
