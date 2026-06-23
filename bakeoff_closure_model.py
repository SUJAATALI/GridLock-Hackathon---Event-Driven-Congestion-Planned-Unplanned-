"""
Model bake-off for requires_road_closure on astram_enriched.csv.
Three gradient-boosted models, SAME 75/25 stratified split, SAME features,
SAME class-imbalance handling. No tuning -- this is a fair-fight comparison.

  - XGBoost            : one-hot features, scale_pos_weight
  - LightGBM           : NATIVE categorical handling (no one-hot), scale_pos_weight
  - HistGradientBoosting: one-hot features, class_weight=balanced  (current baseline / control)

Reports per model on the held-out 25%: ROC-AUC, PR-AUC (vs 0.083 prevalence baseline),
and precision/recall/F1 at thresholds 0.5 / 0.4 / 0.3. Keeps the best PR-AUC model and
prints its top-15 feature importances.
"""
# HISTORICAL EXPERIMENT SCRIPT — references the original 55-col CSV (before weather removal).
# Retained for audit / 'receipts' only; not part of the live pipeline. Will error if re-run on the
# current 53-col astram_enriched.csv (rain_mm/was_raining removed 2026-06-21).

import os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score, f1_score, confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "astram_enriched.csv")

df = pd.read_csv(F, dtype=str, keep_default_na=False)

# ---- target ----
df["y"] = df["requires_road_closure"].str.upper().eq("TRUE").astype(int)
y = df["y"].values
print(f"rows={len(df)}  positives(closure)={df.y.sum()} ({100*df.y.mean():.1f}%)")

# ---- features (same set the baseline uses) ----
CAT = ["event_type", "event_cause", "corridor", "locality_hotspot", "region",
       "veh_type", "time_band", "day_of_week", "is_weekend", "is_night", "was_raining"]
NUM = ["hour", "rain_mm"]
# the four high-cardinality cats LightGBM gets to handle natively
NATIVE_CAT = ["corridor", "locality_hotspot", "region", "event_cause"]

for c in NUM:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["hour"] = df["hour"].fillna(-1)
df["rain_mm"] = df["rain_mm"].fillna(0.0)
for c in CAT:
    df[c] = df[c].replace("", "missing").fillna("missing")

# one-hot matrix (XGBoost + HistGB)
X_oh = pd.get_dummies(df[CAT], dummy_na=False)
X_oh[NUM] = df[NUM].values

# raw matrix with category dtype (LightGBM native)
X_native = df[CAT + NUM].copy()
for c in CAT:
    X_native[c] = X_native[c].astype("category")

print(f"one-hot feature matrix: {X_oh.shape[1]} cols  |  native matrix: {X_native.shape[1]} cols "
      f"({len(NATIVE_CAT)} hi-card cats handled natively)")

# ---- ONE shared split (indices), reused by every model ----
idx = np.arange(len(df))
itr, ite = train_test_split(idx, test_size=0.25, random_state=42, stratify=y)
ytr, yte = y[itr], y[ite]
pos_w = (len(ytr) - ytr.sum()) / ytr.sum()
print(f"train={len(itr)} test={len(ite)}  scale_pos_weight={pos_w:.2f}\n")


def evaluate(name, proba):
    """Return a list of result rows (one per threshold) + print nothing here."""
    auc = roc_auc_score(yte, proba)
    ap = average_precision_score(yte, proba)
    rows = []
    for thr in (0.5, 0.4, 0.3):
        pred = (proba >= thr).astype(int)
        rows.append(dict(
            model=name, auc=auc, ap=ap, thr=thr,
            precision=precision_score(yte, pred, zero_division=0),
            recall=recall_score(yte, pred, zero_division=0),
            f1=f1_score(yte, pred, zero_division=0),
        ))
    return rows, auc, ap


results = []
importances = {}   # name -> pd.Series

# ===== 1. XGBoost (one-hot) =====
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                    subsample=0.9, colsample_bytree=0.8, scale_pos_weight=pos_w,
                    eval_metric="logloss", n_jobs=4, random_state=42)
xgb.fit(X_oh.iloc[itr], ytr)
p = xgb.predict_proba(X_oh.iloc[ite])[:, 1]
r, auc, ap = evaluate("XGBoost", p); results += r
importances["XGBoost"] = pd.Series(xgb.feature_importances_, index=X_oh.columns)
print(f"[done] XGBoost              ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")

# ===== 2. LightGBM (native categorical) =====
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                     subsample=0.9, colsample_bytree=0.8, scale_pos_weight=pos_w,
                     n_jobs=4, random_state=42, verbose=-1)
lgb.fit(X_native.iloc[itr], ytr, categorical_feature=CAT)
p = lgb.predict_proba(X_native.iloc[ite])[:, 1]
r, auc, ap = evaluate("LightGBM", p); results += r
importances["LightGBM"] = pd.Series(lgb.feature_importances_, index=X_native.columns)
print(f"[done] LightGBM            ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")

# ===== 3. HistGradientBoosting (one-hot, control) =====
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.08,
                                     class_weight="balanced", random_state=42)
hgb.fit(X_oh.iloc[itr], ytr)
p = hgb.predict_proba(X_oh.iloc[ite])[:, 1]
r, auc, ap = evaluate("HistGB(ctrl)", p); results += r
# HistGB has no native feature_importances_; skip (would need permutation importance)
print(f"[done] HistGradientBoosting ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")

# ===== comparison table =====
res = pd.DataFrame(results)
prevalence = yte.mean()
print("\n" + "=" * 78)
print(f"HELD-OUT TEST COMPARISON (25% unseen, n={len(ite)}, prevalence={prevalence:.3f})")
print("=" * 78)
hdr = f"{'model':<14}{'ROC-AUC':>9}{'PR-AUC':>8}{'thr':>6}{'prec':>7}{'recall':>8}{'F1':>7}"
print(hdr); print("-" * len(hdr))
for name in ["XGBoost", "LightGBM", "HistGB(ctrl)"]:
    sub = res[res.model == name]
    for i, (_, row) in enumerate(sub.iterrows()):
        m = name if i == 0 else ""
        a = f"{row.auc:.3f}" if i == 0 else ""
        pr = f"{row.ap:.3f}" if i == 0 else ""
        print(f"{m:<14}{a:>9}{pr:>8}{row.thr:>6.1f}"
              f"{row.precision:>7.2f}{row.recall:>8.2f}{row.f1:>7.2f}")
    print("-" * len(hdr))
print(f"(PR-AUC chance baseline = {prevalence:.3f})")

# ===== pick winner by PR-AUC =====
by_ap = res.groupby("model").ap.first().sort_values(ascending=False)
winner = by_ap.index[0]
print("\n" + "=" * 78)
print("WINNER BY PR-AUC (rare positives -> PR-AUC matters more than ROC-AUC):")
for name, ap in by_ap.items():
    star = "  <-- WINNER" if name == winner else ""
    print(f"   {name:<14} PR-AUC={ap:.3f}{star}")

# ===== winner feature importances (top 15) =====
print("\n" + "=" * 78)
print(f"TOP 15 FEATURES DRIVING ROAD CLOSURE  ({winner})")
print("=" * 78)
if winner in importances:
    fi = importances[winner].sort_values(ascending=False).head(15)
    fimax = fi.max()
    for k, v in fi.items():
        bar = "#" * int(round(30 * v / fimax)) if fimax > 0 else ""
        print(f"  {v:>10.3f}  {k:<28} {bar}")
else:
    print("  (no native feature_importances_ for this model)")
