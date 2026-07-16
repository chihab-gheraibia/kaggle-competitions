# House Prices — Advanced Regression Techniques

Predicting home sale prices in Ames, Iowa from 79 explanatory features.

**Competition:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
**Result:** 0.1114 CV RMSE (log scale) — stacked meta-learner
**Note:** this is a "Getting Started" competition (no medals awarded), used as a
skill-building exercise targeting a strong leaderboard percentile.

## Feature Engineering

- **"NA means None" fix** for 15 columns where missing data actually means the
  house lacks that feature entirely (no pool, no garage, no basement) —
  the most common trap on this specific dataset
- **Neighborhood price encoding** — leakage-safe, out-of-fold target encoding
  of `Neighborhood` by average `log(SalePrice)`, since location is one of the
  strongest predictors and one-hot alone underuses it
- **Composite features** — TotalSF, TotalBath, TotalPorchSF, house/remodel/
  garage age
- **Ordinal encoding** for quality columns (Ex > Gd > TA > Fa > Po > None)
  instead of unordered one-hot
- **Outlier removal** — 2 documented high-GrLivArea/low-price houses
- **Log-transform target** to match the competition's RMSLE metric directly

## Models

7 models trained, tested for genuine ensemble diversity:

| Model | CV RMSE |
|---|---|
| Ridge | 0.1255 |
| Lasso | 0.1215 |
| ElasticNet | 0.1228 |
| KNN | 0.1897 |
| SVR | 0.1837 |
| XGBoost (Optuna-tuned) | 0.1156 |
| LightGBM (Optuna-tuned) | 0.1173 |
| CatBoost | 0.1158 |
| Simple average (all 7) | 0.1186 |
| **Stacked meta-learner (6 models)** | **0.1114** |

## Takeaways

KNN and SVR contributed essentially zero weight in the stacked ensemble
(meta-learner effectively ignored them) despite being genuinely different
model types — a reminder that diversity only helps when the additional
model is competitive, not just different. Dropping them and adding
CatBoost + Optuna-tuned hyperparameters instead meaningfully beat simple
averaging (0.1114 vs 0.1186).

Meta-learner weights (Ridge, Lasso, ElasticNet, XGBoost, LightGBM, CatBoost):
`[-0.033, 0.185, 0.193, 0.260, 0.198, 0.200]` — boosting models carry the
most weight, but the regularized linear models contribute meaningfully too.

## Links

- [Kaggle Notebook](#) <!-- add once published -->
