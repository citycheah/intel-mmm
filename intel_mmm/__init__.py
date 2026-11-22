"""Federated Marketing Mix Modeling (3-Tier Ridge -> PyMC-Marketing + nutpie) & Theme Attribution."""

from intel_mmm.data import (
    PreparedMMMData,
    check_theme_collinearity_guard,
    derive_roi_guardrail,
    load_mmm_panel,
    load_theme_panel,
)
from intel_mmm.ladder import (
    TierResult,
    fit_tier1_spend_proportional,
    fit_tier2_ridge_adstock_hill,
    fit_tier3_bayesian_mmm,
    geometric_adstock_vec,
)
from intel_mmm.pipeline import run_mmm_pipeline
from intel_mmm.themes import fit_theme_attribution

__all__ = [
    "PreparedMMMData",
    "TierResult",
    "check_theme_collinearity_guard",
    "derive_roi_guardrail",
    "load_mmm_panel",
    "load_theme_panel",
    "geometric_adstock_vec",
    "fit_tier1_spend_proportional",
    "fit_tier2_ridge_adstock_hill",
    "fit_tier3_bayesian_mmm",
    "fit_theme_attribution",
    "run_mmm_pipeline",
]
