# Spaceship Titanic — Ensemble Model

Predicting which passengers were transported to an alternate dimension during
the Spaceship Titanic's collision with a spacetime anomaly.

**Competition:** [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
**Result:** 0.8095 cross-validated accuracy (5-fold stratified)

## Feature Engineering

- **Cabin** (`Deck/Num/Side`) split into three separate features — deck
  location and ship side may correlate with the anomaly's effect
- **PassengerId** (`gggg_pp`) parsed for `GroupSize` and `IsAlone` — passengers
  traveling together likely share outcomes
- **Name** surnames extracted to catch families/groups sharing a surname but
  assigned different group IDs
- **Spending columns** summed into `TotalSpend`, log-transformed to handle
  skew, with a `NoSpend` flag (CryoSleep passengers can't spend, so this
  interacts meaningfully with that field)
- **Age** bucketed into groups (Child / Teen / Adult / MidAge / Senior) to
  capture non-linear effects

## Models

Three tree-based models trained with 5-fold stratified cross-validation,
combined via out-of-fold probability averaging:

| Model | CV Accuracy |
|---|---|
| Random Forest | 0.8001 (+/- 0.0085) |
| XGBoost | 0.8094 (+/- 0.0061) |
| LightGBM | 0.8097 (+/- 0.0065) |
| **Ensemble** | **0.8095** |

## Takeaways

XGBoost and LightGBM performed almost identically and clearly outperformed
Random Forest alone. The ensemble matched the best single model rather than
exceeding it — suggesting the two boosting models make very similar errors.
A more diverse ensemble member (e.g. linear or distance-based) might add more
value than another tree-based model.

**Next steps:** hyperparameter tuning (Optuna), stacking with a meta-learner
instead of simple averaging.

## Links

- [Kaggle Notebook](https://www.kaggle.com/code/chihebghraibia/notebook4fd941547d)
- [Discussion post](https://www.kaggle.com/competitions/spaceship-titanic/discussion/724336)
