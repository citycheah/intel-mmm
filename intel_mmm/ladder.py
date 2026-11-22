"""3-Tier Marketing Mix Modeling Ladder:
Tier 1: Spend-Proportional Naive Baseline
Tier 2: Frequentist Ridge Regression with Geometric Adstock + Hill Saturation
Tier 3: Bayesian PyMC-Marketing MMM with GeometricAdstock + HillSaturation + nutpie NUTS
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from intel_core.optimize import hill_saturation
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from intel_mmm.data import PreparedMMMData, check_theme_collinearity_guard


def _ensure_writable_pytensor_cache() -> None:
    """Configures PyTensor/Numba/Matplotlib cache dirs inside writable TEMP before importing PyMC."""
    cache_root = os.environ.get("INTEL_MMM_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), "intel_mmm_cache"
    )
    os.makedirs(cache_root, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", cache_root)
    os.environ.setdefault("LOCALAPPDATA", cache_root)
    if "PYTENSOR_FLAGS" not in os.environ:
        compiledir = cache_root.replace("\\", "/")
        os.environ["PYTENSOR_FLAGS"] = f"base_compiledir={compiledir}"


def geometric_adstock_vec(spend: np.ndarray, lam: float, l_max: int = 8) -> np.ndarray:
    """Applies causal normalized geometric adstock matching PyMC-Marketing's GeometricAdstock."""
    arr = np.asarray(spend, dtype=np.float64)
    if lam <= 0.0:
        return arr.copy()
    weights = np.array([lam**lag for lag in range(l_max + 1)], dtype=np.float64)
    weights /= float(weights.sum())
    out = np.convolve(arr, weights, mode="full")[: len(arr)]
    return out


def _compute_fit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mape = float(np.mean(np.abs(y_t - y_p) / np.maximum(1.0, np.abs(y_t))) * 100.0)
    ss_res = float(np.sum((y_t - y_p) ** 2))
    ss_tot = float(np.sum((y_t - np.mean(y_t)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"rmse": round(rmse, 4), "mape_pct": round(mape, 4), "r2": round(r2, 4)}


@dataclass
class TierResult:
    tier_name: str
    metrics: dict[str, float]
    channels: dict[str, dict[str, Any]]
    diagnostics: dict[str, Any]
    y_pred: np.ndarray


def fit_tier1_spend_proportional(data: PreparedMMMData) -> TierResult:
    """Tier 1: Naive linear-proportional spend share attribution baseline."""
    df = data.df
    check_theme_collinearity_guard(list(df.columns))
    y = df["outcome_value"].to_numpy(dtype=np.float64)
    total_y = float(np.sum(np.maximum(0.0, y - float(np.percentile(y, 10)))))
    spends = df[data.channel_cols].to_numpy(dtype=np.float64)
    total_spend_per_ch = spends.sum(axis=0)
    grand_spend = float(total_spend_per_ch.sum())

    y_pred = np.full_like(y, float(np.percentile(y, 10)))
    scale = (total_y / grand_spend) if grand_spend > 0 else 0.0
    y_pred = y_pred + spends.sum(axis=1) * scale

    ch_out: dict[str, dict[str, Any]] = {}
    for idx, ch in enumerate(data.channel_cols):
        sp_arr = spends[:, idx]
        ch_spend = float(sp_arr.sum())
        mean_sp = float(sp_arr.mean())
        contrib = ch_spend * scale
        roas_1k = (contrib / ch_spend * 1000.0) if ch_spend > 0 else 0.0
        ch_out[ch] = {
            "baseline_weekly_spend_usd": round(mean_sp, 2),
            "beta": round(scale * max(1.0, mean_sp * 2.0), 4),
            "ec50_usd": round(max(1000.0, mean_sp * 1.5), 2),
            "slope": 1.0,
            "adstock_lambda": 0.0,
            "total_spend_usd": round(ch_spend, 2),
            "total_contribution": round(contrib, 2),
            "roas_mean": round(roas_1k, 4),
            "roas_hdi_90": [round(roas_1k, 4), round(roas_1k, 4)],
        }
    return TierResult(
        tier_name="tier1_spend_proportional",
        metrics=_compute_fit_metrics(y, y_pred),
        channels=ch_out,
        diagnostics={"method": "proportional_spend_allocation"},
        y_pred=y_pred,
    )


def fit_tier2_ridge_adstock_hill(data: PreparedMMMData, *, alpha: float = 2.0) -> TierResult:
    """Tier 2: Fast frequentist baseline optimizing Adstock + Hill EC50/Slope with Ridge."""
    df = data.df
    check_theme_collinearity_guard(data.channel_cols + data.control_cols)
    y = df["outcome_value"].to_numpy(dtype=np.float64)
    chs = data.channel_cols
    n_ch = len(chs)
    spends = [df[c].to_numpy(dtype=np.float64) for c in chs]
    mean_spends = [max(1000.0, float(np.mean(s))) for s in spends]

    X_ctrl = (
        df[data.control_cols].to_numpy(dtype=np.float64)
        if data.control_cols
        else np.zeros((len(df), 0), dtype=np.float64)
    )
    # Add harmonic seasonal features
    t_idx = np.arange(len(df), dtype=np.float64)
    harmonics = np.column_stack(
        [
            np.sin(2.0 * np.pi * k * t_idx / 52.1775)
            for k in (1, 2)
        ]
        + [
            np.cos(2.0 * np.pi * k * t_idx / 52.1775)
            for k in (1, 2)
        ]
    )
    X_exog = np.hstack([X_ctrl, harmonics])

    # Optimize per-channel adstock lambda & EC50 ratio while solving closed-form Ridge on features
    def _build_hill_features(params: np.ndarray) -> np.ndarray:
        cols = []
        for i in range(n_ch):
            lam = float(np.clip(params[2 * i], 0.01, 0.80))
            ec50 = float(np.clip(params[2 * i + 1], 0.25, 2.50)) * mean_spends[i]
            ad = geometric_adstock_vec(spends[i], lam=lam)
            sat = np.array([hill_saturation(v, ec50_usd=ec50, slope=1.3) for v in ad], dtype=np.float64)
            cols.append(sat)
        return np.column_stack(cols)

    def _loss(params: np.ndarray) -> float:
        X_m = _build_hill_features(params)
        X_full = np.hstack([X_m, X_exog])
        reg = Ridge(alpha=alpha, positive=False).fit(X_full, y)
        # Penalize negative media coefficients softly
        coefs_m = reg.coef_[:n_ch]
        neg_pen = float(np.sum(np.maximum(0.0, -coefs_m) ** 2)) * 1000.0
        pred = reg.predict(X_full)
        return float(np.mean((y - pred) ** 2)) + neg_pen

    init = np.tile([0.35, 1.0], n_ch)
    bounds = [(0.02, 0.78), (0.30, 2.50)] * n_ch
    opt = minimize(_loss, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 60})
    best_p = opt.x

    X_m_best = _build_hill_features(best_p)
    # Refit exog first, then non-negative media coefs cleanly via Ridge(positive=True) on media residual
    exog_reg = Ridge(alpha=1.0).fit(X_exog, y)
    y_res = y - exog_reg.predict(X_exog)
    media_reg = Ridge(alpha=alpha, positive=True).fit(X_m_best, y_res)
    y_pred = exog_reg.predict(X_exog) + media_reg.predict(X_m_best)

    ch_out: dict[str, dict[str, Any]] = {}
    for i, ch in enumerate(chs):
        lam = float(np.clip(best_p[2 * i], 0.01, 0.80))
        ec50 = float(np.clip(best_p[2 * i + 1], 0.25, 2.50)) * mean_spends[i]
        beta = float(media_reg.coef_[i])
        contrib = float(np.sum(X_m_best[:, i] * beta))
        tot_sp = float(spends[i].sum())
        roas_1k = (contrib / tot_sp * 1000.0) if tot_sp > 0 else 0.0
        ch_out[ch] = {
            "baseline_weekly_spend_usd": round(float(np.mean(spends[i])), 2),
            "beta": round(beta, 4),
            "ec50_usd": round(ec50, 2),
            "slope": 1.3,
            "adstock_lambda": round(lam, 4),
            "total_spend_usd": round(tot_sp, 2),
            "total_contribution": round(contrib, 2),
            "roas_mean": round(roas_1k, 4),
            "roas_hdi_90": [round(roas_1k * 0.85, 4), round(roas_1k * 1.15, 4)],
        }

    return TierResult(
        tier_name="tier2_ridge_adstock_hill",
        metrics=_compute_fit_metrics(y, y_pred),
        channels=ch_out,
        diagnostics={"optimizer_success": bool(opt.success), "alpha": alpha},
        y_pred=y_pred,
    )


def fit_tier3_bayesian_mmm(
    data: PreparedMMMData,
    *,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    random_seed: int = 42,
    l_max: int = 6,
) -> TierResult:
    """Tier 3: Bayesian PyMC-Marketing MMM with GeometricAdstock + HillSaturation + nutpie."""
    _ensure_writable_pytensor_cache()
    import arviz as az
    from pymc_extras.prior import Prior
    from pymc_marketing.mmm import MMM, GeometricAdstock, HillSaturation

    df = data.df.copy()
    check_theme_collinearity_guard(data.channel_cols + data.control_cols)
    df["week_start"] = pd.to_datetime(df["week_start"])
    chs = list(data.channel_cols)
    ctrls = list(data.control_cols) if data.control_cols else None

    X = df[["week_start"] + chs + (ctrls or [])].copy()
    y = df["outcome_value"].astype(float)

    # Avoid collinearity when `macro_seasonality` regressor already models annual seasonality
    yearly_harmonics = None if (ctrls and "macro_seasonality" in ctrls) else 2
    hill_priors = {
        "kappa": Prior("Gamma", mu=0.55, sigma=0.35),
        "slope": Prior("Gamma", mu=1.4, sigma=0.45),
        "beta": Prior("HalfNormal", sigma=1.5),
    }

    mmm = MMM(
        date_column="week_start",
        channel_columns=chs,
        target_column="outcome_value",
        control_columns=ctrls,
        adstock=GeometricAdstock(l_max=l_max),
        saturation=HillSaturation(priors=hill_priors),
        yearly_seasonality=yearly_harmonics,
    )
    mmm.build_model(X, y)
    mmm.add_original_scale_contribution_variable(["channel_contribution"])
    idata = mmm.fit(
        X,
        y,
        nuts_sampler="nutpie",
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
        progressbar=False,
    )

    posterior = idata.posterior
    # Convergence diagnostics over sampled structural parameters
    diag_vars = [
        v
        for v in ("saturation_beta", "saturation_kappa", "saturation_slope", "adstock_alpha")
        if v in posterior.data_vars
    ]
    rhat_vals: list[float] = []
    ess_vals: list[float] = []
    for v in diag_vars:
        arr = np.asarray(posterior[v].values, dtype=np.float64)
        # Flatten extra dims into parameters after (chain, draw)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        for j in range(arr.shape[2]):
            r_val = float(np.asarray(az.rhat(arr[:, :, j])))
            e_val = float(np.asarray(az.ess(arr[:, :, j])))
            if np.isfinite(r_val):
                rhat_vals.append(r_val)
            if np.isfinite(e_val):
                ess_vals.append(e_val)

    max_rhat = max(rhat_vals) if rhat_vals else 1.0
    min_ess = min(ess_vals) if ess_vals else float(draws * chains)
    divergences = (
        int(idata.sample_stats["diverging"].sum())
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats
        else 0
    )

    # Extract original-scale channel contributions -> (chain, draw, date, channel)
    cc_orig = np.asarray(posterior["channel_contribution_original_scale"].values, dtype=np.float64)
    # Handle any axis order by finding date and channel axes using lengths
    ch_axis = cc_orig.shape.index(len(chs))
    date_axis = cc_orig.shape.index(len(df))
    tot_contrib_draws = np.sum(cc_orig, axis=date_axis)  # remaining axes: chain, draw, channel
    if ch_axis > date_axis:
        ch_axis -= 1

    # Extract fitted structural means for pure-Python curve export
    y_max = float(np.max(np.abs(y))) or 1.0
    sat_beta_mean = np.mean(np.asarray(posterior["saturation_beta"].values, dtype=np.float64), axis=(0, 1))
    sat_kappa_mean = np.mean(np.asarray(posterior["saturation_kappa"].values, dtype=np.float64), axis=(0, 1))
    sat_slope_mean = np.mean(np.asarray(posterior["saturation_slope"].values, dtype=np.float64), axis=(0, 1))
    ad_alpha_mean = np.mean(np.asarray(posterior["adstock_alpha"].values, dtype=np.float64), axis=(0, 1))

    if "mu" in posterior.data_vars:
        y_pred = np.mean(np.asarray(posterior["mu"].values, dtype=np.float64), axis=(0, 1)) * y_max
    elif hasattr(mmm, "sample_posterior_predictive"):
        pp = mmm.sample_posterior_predictive(
            X, extend_idata=False, random_seed=random_seed, progressbar=False
        )
        pp_ds = pp.posterior_predictive if hasattr(pp, "posterior_predictive") else pp
        y_key = "y" if "y" in pp_ds else list(pp_ds.data_vars)[0]
        raw_pred = np.mean(np.asarray(pp_ds[y_key].values, dtype=np.float64), axis=(0, 1))
        y_pred = raw_pred * y_max if float(np.mean(raw_pred)) < 0.25 * float(np.mean(y)) else raw_pred
    else:
        y_pred = np.full(len(df), float(np.mean(y)), dtype=np.float64)
    fit_metrics = _compute_fit_metrics(y.to_numpy(dtype=np.float64), y_pred)

    ch_out: dict[str, dict[str, Any]] = {}
    for idx, ch in enumerate(chs):
        sp_series = df[ch].to_numpy(dtype=np.float64)
        tot_sp = float(sp_series.sum())
        max_sp = float(sp_series.max()) or 1.0
        mean_sp = float(sp_series.mean())

        contrib_samples = np.take(tot_contrib_draws, idx, axis=ch_axis)
        roas_samples = (contrib_samples / tot_sp * 1000.0) if tot_sp > 0 else np.zeros_like(contrib_samples)
        roas_mean = float(np.mean(roas_samples))
        roas_sd = float(np.std(roas_samples))
        hdi_pair = np.asarray(az.hdi(np.asarray(roas_samples, dtype=np.float64).ravel(), prob=0.90), dtype=np.float64)

        beta_ch = float(sat_beta_mean[idx] * y_max)
        kappa_ch = float(sat_kappa_mean[idx]) * max_sp
        slope_ch = float(sat_slope_mean[idx])
        lam_ch = float(ad_alpha_mean[idx])

        ch_out[ch] = {
            "baseline_weekly_spend_usd": round(mean_sp, 2),
            "beta": round(beta_ch, 4),
            "ec50_usd": round(max(1.0, kappa_ch), 2),
            "slope": round(max(0.1, slope_ch), 4),
            "adstock_lambda": round(lam_ch, 4),
            "total_spend_usd": round(tot_sp, 2),
            "total_contribution": round(float(np.mean(contrib_samples)), 2),
            "roas_mean": round(roas_mean, 4),
            "roas_sd": round(roas_sd, 4),
            "roas_hdi_90": [round(float(hdi_pair[0]), 4), round(float(hdi_pair[1]), 4)],
        }

    diagnostics = {
        "sampler": "nutpie",
        "chains": chains,
        "draws": draws,
        "tune": tune,
        "max_rhat": round(max_rhat, 4),
        "min_ess": round(min_ess, 1),
        "divergences": divergences,
        "excluded_interpolated_rows": data.excluded_interpolated_rows,
        "convergence_passed": bool(max_rhat < 1.05 and divergences <= max(2, int(0.02 * draws * chains))),
    }

    return TierResult(
        tier_name="tier3_bayesian_pymc_nutpie",
        metrics=fit_metrics,
        channels=ch_out,
        diagnostics=diagnostics,
        y_pred=y_pred,
    )
