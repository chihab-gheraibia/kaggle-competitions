# Predicting Student Health Risk — Playground Series S6E7

Classifying students into three health risk categories — **at-risk**, **fit**,
or **unhealthy** — from lifestyle and physiological features.

**Competition:** [Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7)
**Result:** 0.9495 cross-validated balanced accuracy (5-fold stratified)

## The Core Problem: Severe Class Imbalance

Target distribution: **85.9% at-risk, 8.4% unhealthy, only 5.8% fit**. A
model optimizing raw accuracy defaults toward the majority class and looks
deceptively strong while badly underserving the minority classes — exactly
what happened in the unweighted baseline.

## Feature Engineering

- **Missingness flags** for `stress_level` and `smoking_alcohol` before
  imputing — whether someone skipped a question can itself carry signal
- **Health composite ratios** — steps per exercise minute, calories per
  step, water intake relative to BMI, a sleep-efficiency proxy
- **Clinical BMI categories** (underweight/normal/overweight/obese)
- **Risk flags** — elevated resting heart rate (>80bpm), sleep deficit (<6h)
- **Interaction terms** — stress level × physical activity, smoking/alcohol
  × exercise duration

## Models & The Class Weighting Fix

XGBoost + LightGBM, trained twice — once unweighted to establish the
imbalance problem, once with `compute_sample_weight('balanced')` to fix it:

| Class | Unweighted | Class-Weighted |
|---|---|---|
| at-risk | 0.9920 | 0.9371 |
| fit | 0.8279 | 0.9484 |
| unhealthy | 0.8075 | 0.9630 |
| **Balanced accuracy** | 0.8758 | **0.9495** |

## Takeaways

Class weighting alone was worth far more than any individual feature
engineered above — it moved balanced accuracy from 0.8758 to 0.9495 by
forcing the model to stop coasting on the majority class. All three classes
now sit in a much tighter, more honest 0.93-0.96 range instead of one
class propping up the average.

CatBoost was tested as a third model but consistently underperformed and
was excluded from the final ensemble, echoing the same lesson from the
Stellar Class competition.

## Links

- [Kaggle Notebook](#) <!-- add once published -->
