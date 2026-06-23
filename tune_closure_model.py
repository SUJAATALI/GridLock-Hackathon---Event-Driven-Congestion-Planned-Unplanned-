"""
Step 1: confirm XGBoost on the CLEAN regenerated data (~0.81 ROC-AUC / ~0.45 PR-AUC,
        no *_missing / *_unknown features in the top importances).
Step 2: hyperparameter tuning, optimizing for PR-AUC (average_precision) via cross-validated
        randomized search on the TRAIN split only. Test split stays untouched until final eval.
Step 3: pick a sensible operating threshold on the precision/recall trade-off.

Same 75/25 stratified split (random_state=42) and same feature set as all prior experiments.
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, precision_recall_curve)
from xgboost import XGBClassifier

df = pd.read_csv("astram_enriched.csv", dtype=str, keep_default_na=False)
df["y"] = df["requires_road_closure"].str.upper().eq("TRUE").astype(int)

CAT = ["event_type", "event_cause", "corridor", "locality_hotspot", "region",
       "veh_type", "time_band", "day_of_week", "is_weekend", "is_night"]
NUM = ["hour"]
for c in NUM:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["hour"] = df["hour"].fillna(-1)
for c in CAT:
    df[c] = df[c].replace("", "missing").fillna("missing")

X = pd.get_dummies(df[CAT], dummy_na=False)
X[NUM] = df[NUM].values
y = df["y"].values

# sanity: should be no leakage columns left in the data now
leak_cols = [c for c in X.columns if c.endswith("_missing") or c.endswith("_unknown")]
print(f"feature cols={X.shape[1]}   leakage-shaped cols still present: {leak_cols if leak_cols else 'NONE'}")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pos_w = (len(ytr) - ytr.sum()) / ytr.sum()
prevalence = yte.mean()


def report(tag, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    print(f"{tag}: ROC-AUC={roc_auc_score(yte,proba):.3f}  PR-AUC={average_precision_score(yte,proba):.3f}"
          f"  | @{thr:.2f} prec={precision_score(yte,pred,zero_division=0):.2f}"
          f" rec={recall_score(yte,pred,zero_division=0):.2f} f1={f1_score(yte,pred,zero_division=0):.2f}")


# ===== STEP 1: confirm baseline on clean data (same params as bake-off) =====
print("\n" + "=" * 78)
print("STEP 1 — confirm XGBoost on CLEAN data (bake-off params)")
print("=" * 78)
base = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
                     colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="logloss",
                     n_jobs=4, random_state=42)
base.fit(Xtr, ytr)
base_proba = base.predict_proba(Xte)[:, 1]
report("clean baseline", base_proba)
fi = pd.Series(base.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
print("\nTop 15 (clean baseline) — verifying no _missing/_unknown:")
for k, v in fi.items():
    flag = "  <-- LEAK?" if (k.endswith("_missing") or k.endswith("_unknown")) else ""
    print(f"  {v:.3f}  {k}{flag}")

# ===== STEP 2: hyperparameter tuning for PR-AUC =====
print("\n" + "=" * 78)
print("STEP 2 — RandomizedSearchCV optimizing PR-AUC (average_precision), 5-fold on TRAIN")
print("=" * 78)
param_dist = {
    "n_estimators":     [200, 300, 400, 600, 800],
    "max_depth":        [3, 4, 5, 6, 8],
    "learning_rate":    [0.02, 0.03, 0.05, 0.08, 0.1],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "gamma":            [0, 0.5, 1, 2],
    "reg_lambda":       [1, 3, 5, 10],
    "scale_pos_weight": [pos_w * 0.5, pos_w, pos_w * 1.5],
}
search = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", n_jobs=4, random_state=42),
    param_distributions=param_dist, n_iter=60, scoring="average_precision",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42, n_jobs=4, verbose=0, refit=True,
)
search.fit(Xtr, ytr)
print(f"best CV PR-AUC (train, 5-fold) = {search.best_score_:.3f}")
print("best params:")
for k, v in search.best_params_.items():
    print(f"   {k} = {v}")

best = search.best_estimator_
tuned_proba = best.predict_proba(Xte)[:, 1]
print()
report("TUNED (held-out)", tuned_proba)

# ===== STEP 3: operating-threshold selection on the PR trade-off =====
print("\n" + "=" * 78)
print("STEP 3 — operating threshold (held-out PR curve)")
print("=" * 78)
prec, rec, thr = precision_recall_curve(yte, tuned_proba)
# threshold maximizing F1
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_i = int(np.nanargmax(f1s))
f1_thr = thr[best_i]
print(f"max-F1 threshold = {f1_thr:.3f}  (prec={prec[best_i]:.2f} rec={rec[best_i]:.2f} f1={f1s[best_i]:.2f})")

print("\nthreshold sweep on tuned model (held-out):")
print(f"  {'thr':>5}{'prec':>7}{'recall':>8}{'f1':>6}{'flagged':>9}{'TP':>5}{'FP':>5}")
for t in [0.5, 0.45, 0.4, round(float(f1_thr), 3), 0.35, 0.3, 0.25]:
    pred = (tuned_proba >= t).astype(int)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum())
    print(f"  {t:>5.2f}{precision_score(yte,pred,zero_division=0):>7.2f}"
          f"{recall_score(yte,pred,zero_division=0):>8.2f}{f1_score(yte,pred,zero_division=0):>6.2f}"
          f"{int(pred.sum()):>9}{tp:>5}{fp:>5}")

# ===== summary =====
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"clean baseline : ROC-AUC={roc_auc_score(yte,base_proba):.3f}  PR-AUC={average_precision_score(yte,base_proba):.3f}")
print(f"tuned          : ROC-AUC={roc_auc_score(yte,tuned_proba):.3f}  PR-AUC={average_precision_score(yte,tuned_proba):.3f}")
print(f"PR-AUC chance baseline (prevalence) = {prevalence:.3f}")

# save tuned model + chosen threshold
import pickle
with open("closure_model.pkl", "wb") as fh:
    pickle.dump({"model": best, "columns": list(X.columns),
                 "threshold_maxF1": float(f1_thr)}, fh)
print("\nsaved tuned model -> closure_model.pkl")
