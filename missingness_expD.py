"""
Exp D: FIX the enrichment bug instead of deleting rows.

The 116 rows with blank time features have valid start_datetime in BOTH the original
and enriched files -> the blanks are an enrichment bug. Here we re-derive hour/time_band/
day_of_week/is_weekend/is_night from the (valid) start_datetime, which (a) removes the
leakage signal (no more "missing") and (b) keeps all 8173 rows, including 102 planned events.

Compared against the prior runs, same XGBoost setup, same 75/25 split, same metrics.
"""
# HISTORICAL EXPERIMENT SCRIPT — references the original 55-col CSV (before weather removal).
# Retained for audit / 'receipts' only; not part of the live pipeline. Will error if re-run on the
# current 53-col astram_enriched.csv (rain_mm/was_raining removed 2026-06-21).

import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from xgboost import XGBClassifier

df = pd.read_csv("astram_enriched.csv", dtype=str, keep_default_na=False)
df["y"] = df["requires_road_closure"].str.upper().eq("TRUE").astype(int)

# ---- re-derive time features from start_datetime ----
# ROOT CAUSE of the bug: the 116 blanked rows have NO milliseconds ('...46+00', len 22) while
# normal rows do ('...46.111+00'). pd.to_datetime on the whole column locks to one inferred
# format and turns the odd-format rows into NaT. format="ISO8601" parses each row regardless.
dt = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True, format="ISO8601").dt.tz_convert("Asia/Kolkata")
def band(h):
    if pd.isna(h): return "missing"
    h = int(h)
    if 0 <= h < 6:  return "late_night"
    if 6 <= h < 10: return "early_morning"
    if 10 <= h < 13: return "morning_rush"
    if 13 <= h < 17: return "afternoon"
    if 17 <= h < 21: return "evening"
    return "night"
fixed_band = dt.dt.hour.apply(band)
fixed_hour = dt.dt.hour
fixed_dow = dt.dt.day_name().fillna("missing")
fixed_wknd = dt.dt.dayofweek.isin([5, 6]).map({True: "yes", False: "no"}).where(dt.notna(), "missing")
fixed_night = dt.dt.hour.apply(lambda h: "yes" if pd.notna(h) and (h >= 22 or h < 6) else ("no" if pd.notna(h) else "missing"))

n_recovered = (df["time_band"].replace("", "missing").eq("missing") & fixed_band.ne("missing")).sum()
print(f"rows with blank time_band recovered by re-derivation: {n_recovered}")
print(f"unparseable timestamps remaining after fix: {(fixed_band=='missing').sum()}")

CAT = ["event_type", "event_cause", "corridor", "locality_hotspot", "region",
       "veh_type", "time_band", "day_of_week", "is_weekend", "is_night", "was_raining"]
NUM = ["hour", "rain_mm"]
XGB_PARAMS = dict(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.9,
                  colsample_bytree=0.8, eval_metric="logloss", n_jobs=4, random_state=42)


def prep(fix_time):
    d = df.copy()
    if fix_time:
        d["time_band"] = fixed_band.values
        d["day_of_week"] = fixed_dow.values
        d["is_weekend"] = fixed_wknd.values
        d["is_night"] = fixed_night.values
        d["hour"] = fixed_hour.values  # numeric, possibly NaN
    d["hour"] = pd.to_numeric(d["hour"], errors="coerce").fillna(-1)
    d["rain_mm"] = pd.to_numeric(d["rain_mm"], errors="coerce").fillna(0.0)
    for c in CAT:
        d[c] = d[c].replace("", "missing").fillna("missing")
    return d


def fit_eval(d, drop_missing_dummies):
    y = d["y"].values
    X = pd.get_dummies(d[CAT], dummy_na=False)
    X[NUM] = d[NUM].values
    if drop_missing_dummies:
        X = X.drop(columns=[c for c in X.columns if c.endswith("_missing")])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pos_w = (len(ytr) - ytr.sum()) / ytr.sum()
    m = XGBClassifier(scale_pos_weight=pos_w, **XGB_PARAMS)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p),
                prec=precision_score(yte, pred, zero_division=0),
                rec=recall_score(yte, pred, zero_division=0),
                n_rows=len(d), n_feats=X.shape[1]), m, X.columns


rows = []
# reuse prior numbers for context by re-running them here for an apples-to-apples table
r,_,_  = fit_eval(prep(False), False); rows.append(("Baseline (buggy + _missing)", r))
r,_,_  = fit_eval(prep(False), True);  rows.append(("Exp B (drop _missing only)", r))
# Exp C (drop the 116 rows + drop _missing)
dC = prep(False); dC = dC[dC["time_band"] != "missing"].copy()
r,_,_  = fit_eval(dC, True); rows.append(("Exp C (drop 116 rows)", r))
# Exp D (re-derive time feats, keep all rows; _missing dummies now essentially gone)
rD, modelD, colsD = fit_eval(prep(True), True); rows.append(("Exp D (re-derive, keep rows)", rD))

print("\n" + "=" * 96)
print("XGBoost, held-out 25%, metrics @ 0.5  —  leakage fixes compared")
print("=" * 96)
hdr = f"{'setup':<32}{'rows':>6}{'feats':>7}{'ROC-AUC':>9}{'PR-AUC':>8}{'prec':>7}{'recall':>8}"
print(hdr); print("-" * len(hdr))
for name, r in rows:
    print(f"{name:<32}{r['n_rows']:>6}{r['n_feats']:>7}{r['auc']:>9.3f}{r['ap']:>8.3f}{r['prec']:>7.2f}{r['rec']:>8.2f}")
print("-" * len(hdr))

print("\n" + "=" * 96)
print("TOP 15 FEATURES — Exp D (bug fixed, all rows kept, leakage-free)")
print("=" * 96)
fi = pd.Series(modelD.feature_importances_, index=colsD).sort_values(ascending=False).head(15)
fimax = fi.max()
for k, v in fi.items():
    bar = "#" * int(round(30 * v / fimax)) if fimax > 0 else ""
    print(f"  {v:>10.3f}  {k:<30} {bar}")
