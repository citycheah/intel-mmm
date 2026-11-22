# `intel-mmm`

3-Tier Marketing Mix Modeling (`Spend-Proportional` $\rightarrow$ `Ridge + Adstock/Hill` $\rightarrow$ `PyMC-Marketing + nutpie` Rust sampler) and structurally separated Creative Theme Attribution for the **Creative & Marketing Intelligence Suite**.

## Guardrails

- Excludes `is_weekly_interpolated = True` rows from causal fit.
- Rejects `competitor_sov_pct` collider regressors (uses exogenous `competitor_spend_usd`).
- Keeps `gold_theme_spend_weekly` structurally separate from channel regressors in `gold_mmm_weekly_panel`.
- Blocks financial ROAS claims when `outcome_metric_name == 'google_trends_index'` (`proxy_index_only`) or forces a reach/CPM disclaimer when `is_spend_modeled == True`.
