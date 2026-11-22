"""Separate Theme Attribution model over `gold_theme_spend_weekly` (avoids channel+theme collinearity)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from intel_mmm.data import load_theme_panel


def fit_theme_attribution(
    theme_source: Any,
    outcomes_df: pd.DataFrame,
    *,
    brand_id: str,
    country_iso: str,
    alpha: float = 5.0,
) -> list[dict[str, Any]]:
    """Fits a regularized non-negative Ridge model on weekly creative theme spend separately from channel MMM."""
    merged, theme_cols = load_theme_panel(
        theme_source,
        outcomes_df,
        brand_id=brand_id,
        country_iso=country_iso,
    )
    if merged.empty or not theme_cols:
        return []

    X_themes = merged[theme_cols].to_numpy(dtype=np.float64) / 1000.0  # per $1k spend
    X_ctrl = merged[["promo_flag", "holiday_flag", "macro_seasonality"]].to_numpy(dtype=np.float64)
    y = merged["outcome_value"].to_numpy(dtype=np.float64)

    # Demean controls effect before fitting non-negative theme coefficients
    ctrl_model = Ridge(alpha=1.0).fit(X_ctrl, y)
    y_resid = y - ctrl_model.predict(X_ctrl)
    y_resid_centered = y_resid - np.mean(y_resid)

    # Non-negative Ridge over scaled theme spend
    model = Ridge(alpha=alpha, positive=True, fit_intercept=True).fit(X_themes, y_resid_centered)
    coefs = np.asarray(model.coef_, dtype=np.float64)

    total_spend = float(np.sum(merged[theme_cols].to_numpy(dtype=np.float64)))
    raw_contribs = np.maximum(0.0, np.sum(X_themes * coefs[None, :], axis=0))
    total_contrib = float(np.sum(raw_contribs))

    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(theme_cols):
        sp = float(merged[label].sum())
        contrib = float(raw_contribs[idx])
        rows.append(
            {
                "theme_label": label,
                "total_spend_usd": round(sp, 2),
                "spend_share_pct": round((sp / total_spend * 100.0) if total_spend > 0 else 0.0, 2),
                "incremental_per_1k_usd": round(float(coefs[idx]), 4),
                "estimated_contribution": round(contrib, 2),
                "contribution_share_pct": round((contrib / total_contrib * 100.0) if total_contrib > 0 else 0.0, 2),
            }
        )
    rows.sort(key=lambda r: r["estimated_contribution"], reverse=True)
    return rows
