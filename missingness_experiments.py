"""
Leakage investigation for the *_missing dummy features in the XGBoost closure model.

Concern: time_band_missing / veh_type_missing ranked high in the bake-off, but a live
event always has a timestamp -> the model must not "cheat" off missingness.

Same setup as the bake-off: XGBoost, 75/25 stratified split (random_state=42),
metrics = ROC-AUC, PR-AUC, precision/recall@0.5.

  Baseline : all rows, all features (incl. *_missing dummies)   <- current bake-off model
  Exp A    : drop the 116 rows whose time features are blank, then retrain
  Exp B    : keep all rows, remove every *_missing dummy column, then retrain
  Exp C    : both A and B together (the cleanest, leakage-free version)
"""
# HISTORICAL EXPERIMENT SCRIPT — references the original 55-col CSV (before weather removal).
# Retained for audit / 'receipts' only; not part of the live pipeline. Will error if re-run on the
# current 53-col astram_enriched.csv (rain_mm/was_raining removed 2026-06-21).

import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score)
from xgboost import XGBClassifier

df = pd.read_csv("astram_enriched.csv", dtype=str, keep_default_na=False)
df["y"] = df["requires_road_closure"].str.upper().eq("TRUE").astype(int)

CAT = ["event_type", "event_cause", "corridor", "locality_hotspot", "region",
       "veh_type", "time_band", "day_of_week", "is_weekend", "is_night", "was_raining"]
NUM = ["hour", "rain_mm"]

for c in NUM:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["hour"] = df["hour"].fillna(-1)
df["rain_mm"] = df["rain_mm"].fillna(0.0)
# the blank time/veh fields become a literal "missing" category -> get_dummies makes *_missing cols
for c in CAT:
    df[c] = df[c].replace("", "missing").fillna("missing")

# rows whose time features were stored blank (now == "missing" in time_band)
blank_rows = (df["time_band"] == "missing")
print(f"rows total={len(df)}  blank-time-feature rows={blank_rows.sum()}  "
      f"closure rate in those={df.loc[blank_rows,'y'].mean():.3f} vs rest={df.loc[~blank_rows,'y'].mean():.3f}")

XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.08,
                  subsample=0.9, colsample_bytree=0.8,
                  eval_metric="logloss", n_jobs=4, random_state=42)


def run(label, drop_blank_rows, drop_missing_dummies):
    d = df.copy()
    if drop_blank_rows:
        d = d[~(d["time_band"] == "missing")].copy()
    y = d["y"].values
    X = pd.get_dummies(d[CAT], dummy_na=False)
    X[NUM] = d[NUM].values
    if drop_missing_dummies:
        drop_cols = [c for c in X.columns if c.endswith("_missing")]
        X = X.drop(columns=drop_cols)
    # same split recipe as bake-off
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pos_w = (len(ytr) - ytr.sum()) / ytr.sum()
    m = XGBClassifier(scale_pos_weight=pos_w, **XGB_PARAMS)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    res = dict(label=label, n_rows=len(d), n_feats=X.shape[1],
               auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p),
               prec=precision_score(yte, pred, zero_division=0),
               rec=recall_score(yte, pred, zero_division=0),
               prevalence=yte.mean())
    n_missing_dummies = len([c for c in X.columns if c.endswith("_missing")])
    res["missing_dummies_present"] = n_missing_dummies
    return res, m, X.columns


results = []
r, _, _ = run("Baseline", False, False); results.append(r)
r, _, _ = run("Exp A (drop 116 rows)", True, False); results.append(r)
r, _, _ = run("Exp B (drop *_missing)", False, True); results.append(r)
r_c, model_c, cols_c = run("Exp C (both)", True, True); results.append(r_c)

# ---- comparison table ----
res = pd.DataFrame(results)
print("\n" + "=" * 92)
print("BEFORE / AFTER  (XGBoost, held-out 25%, metrics @ threshold 0.5)")
print("=" * 92)
hdr = (f"{'setup':<26}{'rows':>6}{'feats':>7}{'ROC-AUC':>9}{'PR-AUC':>8}"
       f"{'prec':>7}{'recall':>8}{'_missing cols':>15}")
print(hdr); print("-" * len(hdr))
for _, x in res.iterrows():
    print(f"{x.label:<26}{x.n_rows:>6}{x.n_feats:>7}{x.auc:>9.3f}{x.ap:>8.3f}"
          f"{x.prec:>7.2f}{x.rec:>8.2f}{int(x.missing_dummies_present):>15}")
print("-" * len(hdr))
base = res.iloc[0]; c = res.iloc[3]
print(f"PR-AUC chance baseline ~= {base.prevalence:.3f}")
print(f"\nDROP from Baseline -> Exp C:  ROC-AUC {base.auc:.3f} -> {c.auc:.3f} "
      f"(-{base.auc-c.auc:.3f})   PR-AUC {base.ap:.3f} -> {c.ap:.3f} (-{base.ap-c.ap:.3f})")

# ---- Exp C top-15 feature importances ----
print("\n" + "=" * 92)
print("TOP 15 FEATURES — Exp C (cleanest, leakage-free)")
print("=" * 92)
fi = pd.Series(model_c.feature_importances_, index=cols_c).sort_values(ascending=False).head(15)
fimax = fi.max()
for k, v in fi.items():
    bar = "#" * int(round(30 * v / fimax)) if fimax > 0 else ""
    print(f"  {v:>10.3f}  {k:<30} {bar}")
