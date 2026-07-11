# Spaceship Titanic - Extended Pipeline (XGBoost + LightGBM + Ensemble)
# Run inside a Kaggle Notebook. If xgboost/lightgbm aren't pre-installed,
# uncomment the pip install lines below (Kaggle usually has both already).

# !pip install -q xgboost lightgbm

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
train = pd.read_csv('/kaggle/input/spaceship-titanic/train.csv')
test = pd.read_csv('/kaggle/input/spaceship-titanic/test.csv')

# ---------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')

    df['Group'] = df['PassengerId'].str.split('_').str[0]
    group_sizes = df['Group'].value_counts()
    df['GroupSize'] = df['Group'].map(group_sizes)
    df['IsAlone'] = (df['GroupSize'] == 1).astype(int)

    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpend'] = df[spend_cols].sum(axis=1)
    df['NoSpend'] = (df['TotalSpend'] == 0).astype(int)
    df['LogTotalSpend'] = np.log1p(df['TotalSpend'])

    # Name -> Surname (people traveling together often share a surname)
    df['Surname'] = df['Name'].str.split().str[-1]
    surname_counts = df['Surname'].value_counts()
    df['SurnameGroupSize'] = df['Surname'].map(surname_counts)

    # Age buckets - kids and elderly can behave differently
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                             labels=['Child', 'Teen', 'Adult', 'MidAge', 'Senior'])

    df['CryoSleep'] = df['CryoSleep'].fillna(False)

    return df

train = engineer_features(train)
test = engineer_features(test)

# ---------------------------------------------------------
# 3. Handle missing values
# ---------------------------------------------------------
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
            'CabinNum', 'TotalSpend', 'LogTotalSpend', 'GroupSize', 'SurnameGroupSize']
cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'AgeGroup']

for col in num_cols:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)

for col in cat_cols:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col] = test[col].fillna(mode_val)

# ---------------------------------------------------------
# 4. Encode categoricals
# ---------------------------------------------------------
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col].astype(str), test[col].astype(str)])
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

features = num_cols + cat_cols + ['NoSpend', 'IsAlone']
X = train[features]
y = train['Transported'].astype(int)
X_test = test[features]

# ---------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=500, max_depth=12, min_samples_split=5,
    random_state=42, n_jobs=-1
)

xgb_model = XGBClassifier(
    n_estimators=600, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss', n_jobs=-1
)

lgbm_model = LGBMClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)

models = {'RandomForest': rf_model, 'XGBoost': xgb_model, 'LightGBM': lgbm_model}

# ---------------------------------------------------------
# 6. Cross-validated out-of-fold predictions for each model
#    (needed to build a proper ensemble + get honest CV score)
# ---------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = {name: np.zeros(len(X)) for name in models}
test_preds = {name: np.zeros(len(X_test)) for name in models}

for name, model in models.items():
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)[:, 1]
        oof_preds[name][val_idx] = val_pred
        test_preds[name] += model.predict_proba(X_test)[:, 1] / cv.get_n_splits()

        fold_acc = accuracy_score(y_val, val_pred > 0.5)
        fold_scores.append(fold_acc)

    print(f"{name}: CV Accuracy = {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")

# ---------------------------------------------------------
# 7. Ensemble (simple average of the three models' probabilities)
# ---------------------------------------------------------
ensemble_oof = np.mean([oof_preds[name] for name in models], axis=0)
ensemble_acc = accuracy_score(y, ensemble_oof > 0.5)
print(f"\nEnsemble CV Accuracy: {ensemble_acc:.4f}")

ensemble_test = np.mean([test_preds[name] for name in models], axis=0)
final_predictions = (ensemble_test > 0.5)

# ---------------------------------------------------------
# 8. Create submission
# ---------------------------------------------------------
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Transported': final_predictions
})
submission.to_csv('submission.csv', index=False)
print("\nsubmission.csv created — ready to submit!")
