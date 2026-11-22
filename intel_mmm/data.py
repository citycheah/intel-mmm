"""Data loader, interpolation exclusion, collider guard, and ROI labeling for `intel-mmm`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from intel_core.storage import read_parquet_table

CHANNEL_COLUMNS = [
    "own_spend_meta_usd",
    "own_spend_youtube_usd",
    "own_spend_tiktok_usd",
    "own_spend_display_usd",
]

FORBIDDEN_COLLIDER_COLUMNS = {"competitor_sov_pct", "competitor_sos_pct"}


@dataclass
class PreparedMMMData:
    df: pd.DataFrame
    brand_id: str
    country_iso: str
    channel_cols: list[str]
    control_cols: list[str]
    outcome_metric_name: str
    outcome_source: str
    is_proxy_outcome: bool
    is_spend_modeled: bool
    roas_claim_blocked: bool
    roi_label: str
    disclaimer: str
    excluded_interpolated_rows: int


def derive_roi_guardrail(
    *,
    outcome_metric_name: str,
    is_spend_modeled: bool,
) -> tuple[bool, str, str]:
    """Enforces strict ROI claim guardrails for proxy metrics (Google Trends) & EU DSA modeled spend."""
    metric = str(outcome_metric_name or "conversions").strip().lower()
    if metric == "google_trends_index":
        return (
            True,
            "proxy_index_only",
            "Outcome is `google_trends_index` (search interest proxy). Financial ROAS claims are blocked; report directional elasticity and relative index lift only.",
        )
    if is_spend_modeled:
        return (
            False,
            "modeled_reach_cpm_proxy",
            "Ad spend is modeled from EU DSA reach/CPM benchmarks (not advertiser ledger spend). Interpret marginal returns as directional efficiency proxies.",
        )
    return (
        False,
        "incremental_roas",
        "Incremental channel ROAS calibrated against 1st-party/DGP conversion outcomes.",
    )


def load_mmm_panel(
    panel_source: str | Path | pd.DataFrame,
    *,
    brand_id: str,
    country_iso: str,
    extra_regressors: list[str] | None = None,
) -> PreparedMMMData:
    """Loads `gold_mmm_weekly_panel`, enforces collider & interpolation exclusions, and computes ROI guardrails."""
    if isinstance(panel_source, pd.DataFrame):
        df = panel_source.copy()
    else:
        df = read_parquet_table(panel_source, table_name="gold_mmm_weekly_panel")

    for forbidden in FORBIDDEN_COLLIDER_COLUMNS:
        if extra_regressors and forbidden in extra_regressors:
            raise ValueError(
                f"Collider bias violation: `{forbidden}` cannot be used as an MMM regressor. Use exogenous `competitor_spend_usd`."
            )

    sub = df[(df["brand_id"] == brand_id) & (df["country_iso"] == country_iso)].copy()
    if sub.empty:
        raise ValueError(f"No rows in gold_mmm_weekly_panel for brand={brand_id!r}, country={country_iso!r}")

    sub["week_start"] = pd.to_datetime(sub["week_start"])
    sub = sub.sort_values("week_start").reset_index(drop=True)

    # Exclude flat-lined interpolated rows from causal MMM regression fit
    interp_mask = sub.get("is_weekly_interpolated", pd.Series(False, index=sub.index)).astype(bool)
    excluded_count = int(interp_mask.sum())
    fit_df = sub.loc[~interp_mask].reset_index(drop=True)
    if fit_df.empty:
        raise ValueError("All rows were excluded because is_weekly_interpolated=True; cannot fit causal MMM.")

    outcome_metric = str(fit_df["outcome_metric_name"].iloc[0])
    outcome_source = str(fit_df["outcome_source"].iloc[0])
    is_modeled = bool(fit_df["is_spend_modeled"].astype(bool).any())

    roas_blocked, roi_label, disclaimer = derive_roi_guardrail(
        outcome_metric_name=outcome_metric,
        is_spend_modeled=is_modeled,
    )

    active_channels = [c for c in CHANNEL_COLUMNS if c in fit_df.columns and float(fit_df[c].sum()) > 0.0]
    control_cols = [
        c
        for c in ("promo_flag", "holiday_flag", "price_index", "macro_seasonality", "competitor_spend_usd")
        if c in fit_df.columns
    ]
    if extra_regressors:
        for reg in extra_regressors:
            if reg in FORBIDDEN_COLLIDER_COLUMNS:
                raise ValueError(f"Forbidden collider regressor: {reg}")
            if reg not in control_cols and reg in fit_df.columns:
                control_cols.append(reg)

    return PreparedMMMData(
        df=fit_df,
        brand_id=brand_id,
        country_iso=country_iso,
        channel_cols=active_channels,
        control_cols=control_cols,
        outcome_metric_name=outcome_metric,
        outcome_source=outcome_source,
        is_proxy_outcome=(outcome_metric == "google_trends_index"),
        is_spend_modeled=is_modeled,
        roas_claim_blocked=roas_blocked,
        roi_label=roi_label,
        disclaimer=disclaimer,
        excluded_interpolated_rows=excluded_count,
    )


def check_theme_collinearity_guard(regressors: list[str]) -> None:
    """Prevents mixing channel spend and theme spend in the same design matrix."""
    has_channel = any(r.startswith("own_spend_") for r in regressors)
    has_theme = any(r.startswith("theme_spend_") for r in regressors)
    if has_channel and has_theme:
        raise ValueError(
            "Compositional collinearity guard: cannot include both `own_spend_*` channels and `theme_spend_*` in the same regression."
        )


def load_theme_panel(
    theme_source: str | Path | pd.DataFrame,
    outcomes_df: pd.DataFrame,
    *,
    brand_id: str,
    country_iso: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Builds a separate weekly theme-spend panel joined to outcomes for structural theme attribution."""
    if isinstance(theme_source, pd.DataFrame):
        tdf = theme_source.copy()
    else:
        tdf = read_parquet_table(theme_source, table_name="gold_theme_spend_weekly")

    sub = tdf[(tdf["brand_id"] == brand_id) & (tdf["country_iso"] == country_iso)].copy()
    if sub.empty:
        return pd.DataFrame(), []

    sub["week_start"] = pd.to_datetime(sub["week_start"])
    piv = sub.pivot_table(
        index="week_start",
        columns="theme_label",
        values="theme_spend_usd",
        aggfunc="sum",
        fill_value=0.0,
    )
    theme_cols = [str(c) for c in piv.columns]
    piv = piv.reset_index()

    out = outcomes_df[["week_start", "outcome_value", "promo_flag", "holiday_flag", "macro_seasonality"]].copy()
    out["week_start"] = pd.to_datetime(out["week_start"])
    merged = pd.merge(out, piv, on="week_start", how="inner").sort_values("week_start").reset_index(drop=True)
    return merged, theme_cols
