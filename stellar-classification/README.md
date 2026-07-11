# Predicting Stellar Class — Playground Series S6E6

Classifying celestial objects into GALAXY, QSO (quasar), or STAR from photometric
and spectral measurements.

**Competition:** [Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6)
**Result:** 0.9542 cross-validated balanced accuracy (5-fold stratified)

## Feature Engineering

- **SDSS color indices** (`u-g`, `g-r`, `r-i`, `i-z`, `u-r`, `g-i`) — more
  informative than raw magnitudes alone, standard technique for separating
  stellar/galactic/quasar populations
- **Redshift transforms** (log1p, squared) + near-zero-redshift flags — stars
  sit at redshift ≈ 0, QSOs at much higher redshift
- **Stellar locus residual** — classic star/galaxy separation trick: fits a
  quadratic to the tight color-color curve stars trace, measures each
  object's distance from it
- **Interaction terms** between spectral_type/galaxy_population and redshift

## Models

Three gradient boosting models, 5-fold stratified cross-validation:

| Model | CV Balanced Accuracy |
|---|---|
| XGBoost | 0.9533 |
| LightGBM | 0.9540 |
| CatBoost | 0.9468 |
| Simple 3-model average | 0.9527 |
| Weighted 3-model average | 0.9538 |
| **XGBoost + LightGBM (final)** | **0.9542** |

## Takeaways

CatBoost consistently underperformed and dragged down any ensemble that
included it — even weighted. Dropping it and averaging just the two stronger
models beat every 3-model combination. Ensembling only helps when member
models are comparably strong.

Confusion matrix showed STAR as the weakest class (92.1% accuracy), mostly
confused with GALAXY rather than QSO — physically sensible, since low-redshift
galaxies can look photometrically similar to stars. Targeted feature
engineering (stellar locus residual) didn't meaningfully close this gap,
suggesting it reflects the dataset's natural noise floor rather than a
missing feature.

**Next steps:** Optuna hyperparameter tuning, stacking with a meta-learner
instead of averaging.

## Links

- [Kaggle Notebook](https://www.kaggle.com/code/chihebghraibia/notebook758e4105ca)
- [Discussion post](#) <!-- add once posted -->
