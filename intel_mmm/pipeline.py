"""Orchestration pipeline for 3-Tier Channel MMM + Separate Theme Attribution + Artifact Exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from intel_core.storage import write_parquet_table

from intel_mmm.data import load_mmm_panel
from intel_mmm.ladder import (
    TierResult,
    fit_tier1_spend_proportional,
    fit_tier2_ridge_adstock_hill,
    fit_tier3_bayesian_mmm,
)
from intel_mmm.themes import fit_theme_attribution


def run_mmm_pipeline(
    panel_source: str | Path | pd.DataFrame,
    *,
    brand_id: str,
    country_iso: str,
    theme_source: str | Path | pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    run_bayesian: bool = True,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    random_seed: int = 42,
    extra_regressors: list[str] | None = None,
) -> dict[str, Any]:
    """Fits the 3-Tier MMM ladder, enforces ROI guardrails, runs separate Theme Attribution, and writes artifacts."""
    data = load_mmm_panel(
        panel_source,
        brand_id=brand_id,
        country_iso=country_iso,
        extra_regressors=extra_regressors,
    )

    t1 = fit_tier1_spend_proportional(data)
    t2 = fit_tier2_ridge_adstock_hill(data)
    tiers: list[TierResult] = [t1, t2]
    champion = t2
    if run_bayesian:
        t3 = fit_tier3_bayesian_mmm(
            data,
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
        )
        tiers.append(t3)
        champion = t3

    theme_attrib: list[dict[str, Any]] = []
    if theme_source is not None:
        theme_attrib = fit_theme_attribution(
            theme_source,
            data.df,
            brand_id=brand_id,
            country_iso=country_iso,
        )

    posterior_rows: list[dict[str, Any]] = []
    for t in tiers:
        for ch, cmeta in t.channels.items():
            posterior_rows.append(
                {
                    "brand_id": data.brand_id,
                    "country_iso": data.country_iso,
                    "tier": t.tier_name,
                    "channel": ch,
                    "outcome_metric_name": data.outcome_metric_name,
                    "is_proxy_outcome": data.is_proxy_outcome,
                    "is_spend_modeled": data.is_spend_modeled,
                    "roas_claim_blocked": data.roas_claim_blocked,
                    "roi_label": data.roi_label,
                    "total_spend_usd": cmeta["total_spend_usd"],
                    "total_contribution": cmeta["total_contribution"],
                    "roas_mean": cmeta["roas_mean"],
                    "roas_hdi_low": cmeta["roas_hdi_90"][0],
                    "roas_hdi_high": cmeta["roas_hdi_90"][1],
                    "adstock_lambda": cmeta["adstock_lambda"],
                    "ec50_usd": cmeta["ec50_usd"],
                    "slope": cmeta["slope"],
                    "beta": cmeta["beta"],
                }
            )
    posteriors_df = pd.DataFrame(posterior_rows)

    response_curves = {
        "brand_id": data.brand_id,
        "country_iso": data.country_iso,
        "outcome_metric_name": data.outcome_metric_name,
        "outcome_source": data.outcome_source,
        "is_proxy_outcome": data.is_proxy_outcome,
        "is_spend_modeled": data.is_spend_modeled,
        "roas_claim_blocked": data.roas_claim_blocked,
        "roi_label": data.roi_label,
        "disclaimer": data.disclaimer,
        "champion_tier": champion.tier_name,
        "channels": champion.channels,
        "theme_attribution": theme_attrib,
    }

    diagnostics_payload = {
        "brand_id": data.brand_id,
        "country_iso": data.country_iso,
        "outcome_metric_name": data.outcome_metric_name,
        "is_proxy_outcome": data.is_proxy_outcome,
        "is_spend_modeled": data.is_spend_modeled,
        "roas_claim_blocked": data.roas_claim_blocked,
        "roi_label": data.roi_label,
        "excluded_interpolated_rows": data.excluded_interpolated_rows,
        "champion_tier": champion.tier_name,
        "ladder_comparison": [
            {"tier": t.tier_name, **t.metrics, **t.diagnostics}
            for t in tiers
        ],
    }

    out_paths: dict[str, str] = {}
    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        write_parquet_table(posteriors_df, out_root / "roas_posteriors")
        rc_path = out_root / "response_curves.json"
        rc_path.write_text(json.dumps(response_curves, indent=2), encoding="utf-8")
        diag_path = out_root / "mmm_diagnostics.json"
        diag_path.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")
        out_paths = {
            "roas_posteriors_path": str(out_root / "roas_posteriors"),
            "response_curves_path": str(rc_path),
            "mmm_diagnostics_path": str(diag_path),
        }

    return {
        "response_curves": response_curves,
        "diagnostics": diagnostics_payload,
        "posteriors_df": posteriors_df,
        **out_paths,
    }
