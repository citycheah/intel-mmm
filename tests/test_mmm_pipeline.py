"""Tests for `intel-mmm` 3-Tier MMM ladder, collider/interpolation guards, proxy ROI block, and Bayesian recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from intel_core.optimize import simulate_budget_reallocation

from intel_mmm import (
    check_theme_collinearity_guard,
    load_mmm_panel,
    run_mmm_pipeline,
)

EU_LAKEHOUSE = Path(__file__).resolve().parents[2] / "data" / "eu_auto_lakehouse"


def test_collider_bias_and_compositional_collinearity_guards():
    with pytest.raises(ValueError, match="Compositional collinearity guard"):
        check_theme_collinearity_guard(["own_spend_meta_usd", "theme_spend_ev_range"])

    if not EU_LAKEHOUSE.exists():
        pytest.skip("eu_auto_lakehouse fixture not present")

    with pytest.raises(ValueError, match="Collider bias violation"):
        load_mmm_panel(
            EU_LAKEHOUSE / "gold" / "gold_mmm_weekly_panel",
            brand_id="BMW",
            country_iso="DE",
            extra_regressors=["competitor_sov_pct"],
        )


def test_interpolated_week_exclusion_and_proxy_roi_guardrails():
    if not EU_LAKEHOUSE.exists():
        pytest.skip("eu_auto_lakehouse fixture not present")

    # 1. Google Trends Proxy outcome blocks financial ROAS claims
    res_proxy = run_mmm_pipeline(
        EU_LAKEHOUSE / "gold" / "gold_mmm_proxy_weekly_panel",
        brand_id="BMW",
        country_iso="DE",
        run_bayesian=False,
    )
    assert res_proxy["response_curves"]["is_proxy_outcome"] is True
    assert res_proxy["response_curves"]["roas_claim_blocked"] is True
    assert res_proxy["response_curves"]["roi_label"] == "proxy_index_only"
    assert "blocked" in res_proxy["response_curves"]["disclaimer"].lower()

    # 2. Interpolated weeks are excluded before regression fit
    panel_df = pd.read_parquet(EU_LAKEHOUSE / "gold" / "gold_mmm_weekly_panel")
    sub = panel_df[(panel_df["brand_id"] == "BMW") & (panel_df["country_iso"] == "DE")].copy()
    sub.iloc[:6, sub.columns.get_loc("is_weekly_interpolated")] = True
    prep = load_mmm_panel(sub, brand_id="BMW", country_iso="DE")
    assert prep.excluded_interpolated_rows == 6
    assert len(prep.df) == len(sub) - 6

    # 3. Modeled EU DSA spend switches ROI label to directional modeled proxy
    sub_mod = sub.copy()
    sub_mod["is_spend_modeled"] = True
    res_mod = run_mmm_pipeline(sub_mod, brand_id="BMW", country_iso="DE", run_bayesian=False)
    assert res_mod["response_curves"]["is_spend_modeled"] is True
    assert res_mod["response_curves"]["roi_label"] == "modeled_reach_cpm_proxy"


def test_full_bayesian_mmm_recovers_dgp_and_powers_optimizer(tmp_path: Path):
    if not EU_LAKEHOUSE.exists():
        pytest.skip("eu_auto_lakehouse fixture not present")

    out_dir = tmp_path / "mmm_out"
    res = run_mmm_pipeline(
        EU_LAKEHOUSE / "gold" / "gold_mmm_weekly_panel",
        brand_id="BMW",
        country_iso="DE",
        theme_source=EU_LAKEHOUSE / "gold" / "gold_theme_spend_weekly",
        output_dir=out_dir,
        run_bayesian=True,
        draws=400,
        tune=400,
        chains=2,
        random_seed=42,
    )

    assert Path(res["roas_posteriors_path"]).exists()
    assert Path(res["response_curves_path"]).exists()
    assert Path(res["mmm_diagnostics_path"]).exists()

    ladder = {r["tier"]: r for r in res["diagnostics"]["ladder_comparison"]}
    t1 = ladder["tier1_spend_proportional"]
    t3 = ladder["tier3_bayesian_pymc_nutpie"]

    assert t3["rmse"] < t1["rmse"]
    assert t3["r2"] >= 0.90
    assert t3["max_rhat"] < 1.05, f"Unconverged MCMC chains: max_rhat={t3['max_rhat']}"

    # Verify 90% Bayesian HDI covers true DGP ROAS across all 4 channels
    truth = json.loads((EU_LAKEHOUSE / "eu_auto_dgp_truth.json").read_text(encoding="utf-8"))[
        "true_marginal_roas_at_mean"
    ]
    for ch, ch_truth in truth.items():
        true_roas = float(ch_truth["true_avg_roas_per_1k_usd"])
        low, high = res["response_curves"]["channels"][ch]["roas_hdi_90"]
        assert low <= true_roas <= high, (
            f"Channel {ch} true ROAS {true_roas} outside 90% HDI [{low}, {high}]"
        )

    # Verify separate theme attribution results
    themes = res["response_curves"]["theme_attribution"]
    assert len(themes) >= 4
    assert sum(t["contribution_share_pct"] for t in themes) == pytest.approx(100.0, abs=0.5)

    # Verify pure-Python optimizer in intel-core consumes response_curves.json cleanly
    rc_loaded = json.loads(Path(res["response_curves_path"]).read_text(encoding="utf-8"))
    sim = simulate_budget_reallocation(rc_loaded, optimize=True)
    assert sim["expected_outcome_lift_pct"] >= 0.0


def test_self_contained_guardrails_and_ridge_ladder(tmp_path: Path):
    """Self-contained unit test for GitHub Actions CI when external eu_auto_lakehouse is absent."""
    import numpy as np

    rng = np.random.default_rng(42)
    weeks = pd.date_range("2025-01-06", periods=52, freq="W-MON")
    meta = rng.uniform(8000, 20000, size=52)
    yt = rng.uniform(6000, 16000, size=52)
    tt = rng.uniform(4000, 12000, size=52)
    disp = rng.uniform(3000, 9000, size=52)
    comp = rng.uniform(15000, 35000, size=52)
    season = np.sin(2.0 * np.pi * np.arange(52) / 52.0)
    outcome = 1200.0 + 0.04 * meta + 0.03 * yt + 0.02 * tt + 0.015 * disp + 60.0 * season + rng.normal(0, 15, size=52)

    df = pd.DataFrame(
        {
            "brand_id": "BMW",
            "country_iso": "DE",
            "week_start": weeks.strftime("%Y-%m-%d"),
            "is_weekly_interpolated": [i < 4 for i in range(52)],
            "outcome_metric_name": "conversions",
            "outcome_source": "synthetic_dgp",
            "is_spend_modeled": False,
            "own_spend_meta_usd": meta,
            "own_spend_youtube_usd": yt,
            "own_spend_tiktok_usd": tt,
            "own_spend_display_usd": disp,
            "promo_flag": (np.arange(52) % 8 == 0).astype(float),
            "holiday_flag": (np.arange(52) % 13 == 0).astype(float),
            "price_index": np.full(52, 1.0),
            "macro_seasonality": season,
            "competitor_spend_usd": comp,
            "competitor_sov_pct": np.full(52, 0.28),
            "outcome_value": outcome,
        }
    )

    with pytest.raises(ValueError, match="Collider bias violation"):
        load_mmm_panel(
            df,
            brand_id="BMW",
            country_iso="DE",
            extra_regressors=["competitor_sov_pct"],
        )

    prep = load_mmm_panel(df, brand_id="BMW", country_iso="DE")
    assert prep.excluded_interpolated_rows == 4
    assert len(prep.df) == 48

    proxy_df = df.copy()
    proxy_df["outcome_metric_name"] = "google_trends_index"
    proxy_df["outcome_source"] = "google_trends_proxy"
    res_proxy = run_mmm_pipeline(
        proxy_df,
        brand_id="BMW",
        country_iso="DE",
        output_dir=tmp_path / "proxy_out",
        run_bayesian=False,
    )
    assert res_proxy["response_curves"]["is_proxy_outcome"] is True
    assert res_proxy["response_curves"]["roas_claim_blocked"] is True
    assert res_proxy["response_curves"]["roi_label"] == "proxy_index_only"

    res_conv = run_mmm_pipeline(
        df,
        brand_id="BMW",
        country_iso="DE",
        output_dir=tmp_path / "conv_out",
        run_bayesian=False,
    )
    rc_loaded = json.loads(Path(res_conv["response_curves_path"]).read_text(encoding="utf-8"))
    sim = simulate_budget_reallocation(rc_loaded, optimize=True)
    assert sim["expected_outcome_lift_pct"] >= 0.0
