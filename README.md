# `intel-mmm`

3-Tier Marketing Mix Modeling (`Spend-Proportional` $\rightarrow$ `Ridge + Adstock/Hill` $\rightarrow$ `PyMC-Marketing + nutpie` Rust sampler) and structurally separated Creative Theme Attribution for the **Creative & Marketing Intelligence Suite**.

## Guardrails

- Excludes `is_weekly_interpolated = True` rows from causal fit.
- Rejects `competitor_sov_pct` collider regressors (uses exogenous `competitor_spend_usd`).
- Keeps `gold_theme_spend_weekly` structurally separate from channel regressors in `gold_mmm_weekly_panel`.
- Blocks financial ROAS claims when `outcome_metric_name == 'google_trends_index'` (`proxy_index_only`) or forces a reach/CPM disclaimer when `is_spend_modeled == True`.

## Creative & Marketing Intelligence Suite (`v0.1.0`) — Module Map

Part of the 7-repository **`creative-intel-suite`** (flagship UI & copilot in [`brand-analytics-agent`](https://github.com/citycheah/brand-analytics-agent)):

| Component | Role in `creative-intel-suite` (`v0.1.0`) | Stack |
| :--- | :--- | :--- |
| **[`brand-analytics-agent`](https://github.com/citycheah/brand-analytics-agent)** | **Flagship Product Shell** — 5-page Next.js 15 Portal, FastAPI REST/SSE & Google ADK Copilot | Next.js 15, FastAPI, Google ADK, DuckDB |
| **[`intel-core`](https://github.com/citycheah/intel-core)** | **Core SDK** — Canonical contracts, DuckDB HITL views, FinOps, Hill optimizer, Golden Vision evals | Pydantic v2, PyArrow, DuckDB, Gemini |
| **[`intel-etl-spark`](https://github.com/citycheah/intel-etl-spark)** | **Medallion Batch ETL** — Bronze->Silver->Gold Spark jobs, EU Auto DGP & DuckDB local parity | PySpark 3.5, DuckDB, PyArrow |
| **[`intel-mmm`](https://github.com/citycheah/intel-mmm)** | **Causal Incrementality** — 3-tier Bayesian MMM ladder, separate theme attribution & ROAS guardrails | PyMC-Marketing, `nutpie`, Ridge |
| **[`creative-intelligence-toolkit`](https://github.com/citycheah/creative-intelligence-toolkit)** | **Ingestion Connectors** — Meta EU DSA Ad Library, Google Trends, YouTube/TikTok/IG/FB & alerts | Gemini, Playwright, `yt-dlp`, `intel-core` |
| **[`creative-analytics-grouping`](https://github.com/citycheah/creative-analytics-grouping)** | **Theme & Fatigue Engine** — Anti-join unseen dissection, 30-D UMAP+HDBSCAN & wearout tracking | Vertex/Gemini, HDBSCAN, UMAP |
| **[`competitive-velocity-ml-engine`](https://github.com/citycheah/competitive-velocity-ml-engine)** | **Velocity & Drift Engine** — YouTube $t_0$ breakout prediction, dbt PIT features & PSI drift | dbt-duckdb, LightGBM, SHAP |
