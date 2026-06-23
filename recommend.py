"""
MODEL 3 — Recommendation layer (rules-based, transparent, honestly backtested).

Combines:
  Model 1 (closure_model.pkl)  -> per-event closure probability
  Model 2 (hotspot_risk_table.csv) -> corridor x hour x dow expected-event risk

For an event it returns:
  - manpower  : officer count, tiered by closure risk x cause severity
  - barricade : yes/no + junction (driven by closure-risk; junction from event if known)
  - prepositioning : top high-risk corridor x hour cells to deploy to in advance

NO "optimal" claims. Every threshold is documented below and is a policy choice, not a
learned optimum. Validation = backtest on the SAME held-out 25% used to train Model 1:
"for events we flag high closure-risk, did they historically actually close / last longer?"

Public API for the dashboard:
    recommend(event: dict) -> dict          # event in -> recommendation out
    prepositioning(top_n=10) -> list[dict]   # standing deploy-here suggestions
"""
import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = pickle.load(open(os.path.join(_HERE, "closure_model.pkl"), "rb"))
_CLF, _COLS = _MODEL["model"], _MODEL["columns"]
_HOTSPOT = pd.read_csv(os.path.join(_HERE, "hotspot_risk_table.csv"))

# ---- DOCUMENTED POLICY CONSTANTS (tunable by BTP; not learned "optimums") ----
CLOSURE_THRESHOLD = 0.40          # committed operating point of Model 1 (recall-leaning)
# closure-risk tiers (probability from Model 1)
RISK_HIGH, RISK_MED = 0.40, 0.20
# cause severity = empirical historical closure rate buckets (from astram_enriched.csv):
#   high   >=25% close: vip_movement, public_event, protest, tree_fall, construction, procession
#   medium 8-25% close: road_conditions, water_logging, others
#   low    <8%   close: vehicle_breakdown, accident, congestion, pot_holes
SEVERITY = {
    "vip_movement": "high", "public_event": "high", "protest": "high",
    "tree_fall": "high", "construction": "high", "procession": "high",
    "road_conditions": "medium", "water_logging": "medium", "others": "medium",
    "vehicle_breakdown": "low", "accident": "low", "congestion": "low", "pot_holes": "low",
}
# manpower lookup: (risk_tier, severity) -> officers. Transparent grid.
MANPOWER = {
    ("high", "high"): 8, ("high", "medium"): 6, ("high", "low"): 5,
    ("med", "high"): 5,  ("med", "medium"): 4,  ("med", "low"): 3,
    ("low", "high"): 3,  ("low", "medium"): 2,  ("low", "low"): 2,
}

_CAT = ["event_type", "event_cause", "corridor", "locality_hotspot", "region",
        "veh_type", "time_band", "day_of_week", "is_weekend", "is_night"]
_NUM = ["hour"]


def _featurize(events: pd.DataFrame) -> pd.DataFrame:
    """Turn raw event rows into the exact one-hot matrix Model 1 expects."""
    d = events.copy()
    for c in _NUM:
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["hour"] = d["hour"].fillna(-1)
    for c in _CAT:
        d[c] = d.get(c, "missing").replace("", "missing").fillna("missing")
    X = pd.get_dummies(d[_CAT], dummy_na=False)
    X[_NUM] = d[_NUM].values
    return X.reindex(columns=_COLS, fill_value=0)   # align to training columns


def closure_proba(events: pd.DataFrame) -> np.ndarray:
    return _CLF.predict_proba(_featurize(events))[:, 1]


# human-readable labels for each feature group (shown in the dashboard "Why this risk" panel)
_GROUP_LABEL = {
    "event_cause": "cause", "event_type": "type", "corridor": "corridor",
    "locality_hotspot": "location", "region": "region", "veh_type": "vehicle",
    "time_band": "time band", "day_of_week": "day", "is_weekend": "weekend",
    "is_night": "night", "hour": "hour",
}


def explain(event: dict, top_k: int = 3) -> list:
    """Per-prediction explainability: which factors pushed THIS event's closure risk
    up or down. Uses XGBoost SHAP contributions (pred_contribs) in log-odds space,
    aggregated back to the original feature group (so the 200+ one-hot columns roll
    up to readable factors like 'cause: tree_fall'). Returns top-k by absolute impact.
    Falls back to [] if the booster doesn't support contributions."""
    try:
        import xgboost as xgb
        X = _featurize(pd.DataFrame([event]))
        booster = _CLF.get_booster()
        dm = xgb.DMatrix(X, feature_names=list(_COLS))
        contribs = booster.predict(dm, pred_contribs=True)[0]  # len = n_features + 1 (last = bias)
    except Exception:
        return []
    col_contrib = dict(zip(_COLS, contribs[:-1]))
    # roll up one-hot columns to their parent feature group
    group_sum = {}
    for col, val in col_contrib.items():
        if col in _NUM:
            g = col
        else:
            g = next((c for c in _CAT if col.startswith(c + "_")), None)
        if g is None:
            continue
        group_sum[g] = group_sum.get(g, 0.0) + float(val)
    ranked = sorted(group_sum.items(), key=lambda kv: -abs(kv[1]))
    out = []
    for g, v in ranked:
        if abs(v) < 1e-6:
            continue
        value = str(event.get(g, "")).strip()
        if not value or value.lower() in ("missing", "null", "nan", "none"):
            continue   # skip blank/"missing" factors — they read as data artifacts to operators
        label = _GROUP_LABEL.get(g, g)
        out.append({"factor": f"{label}: {value}",
                    "direction": "up" if v > 0 else "down",
                    "weight": round(float(v), 3)})
        if len(out) >= top_k:
            break
    return out


def _risk_tier(p):
    return "high" if p >= RISK_HIGH else ("med" if p >= RISK_MED else "low")


def recommend(event: dict) -> dict:
    """event in -> recommendation out. event is a dict of the enriched-schema fields."""
    row = pd.DataFrame([event])
    p = float(closure_proba(row)[0])
    cause = str(event.get("event_cause", "missing"))
    sev = SEVERITY.get(cause, "medium")
    tier = _risk_tier(p)
    officers = MANPOWER[(tier, sev)]

    # barricade: driven by closure risk crossing the committed threshold
    will_close = p >= CLOSURE_THRESHOLD
    junction = str(event.get("junction", "") or "").strip()
    junction = junction if junction and junction.upper() != "NULL" else None
    barricade = {
        "needed": bool(will_close),
        "junction": junction if will_close else None,
        "note": ("barricade at listed junction" if (will_close and junction)
                 else "barricade recommended; junction not in record — set on site" if will_close
                 else "no barricade"),
    }

    return {
        "closure_probability": round(p, 3),
        "closure_risk_tier": tier,
        "cause_severity": sev,
        "manpower_officers": officers,
        "barricade": barricade,
        "drivers": explain(event),
        "rationale": (f"closure risk {p:.0%} ({tier}) x cause '{cause}' severity={sev} "
                      f"-> {officers} officers; "
                      f"{'barricade (risk>=%.2f)' % CLOSURE_THRESHOLD if will_close else 'no barricade'}"),
    }


def prepositioning(top_n: int = 10, named_only: bool = True) -> list:
    """Standing 'deploy here in advance' list from Model 2's hotspot risk table.
    Keyed on `location` (locality_hotspot) — named arterials + split-out former Non-corridor areas."""
    h = _HOTSPOT.copy()
    if named_only:
        h = h[h["named"]] if "named" in h.columns else h[h["location"] != "Other"]
    h = h.sort_values("risk_per_day", ascending=False).head(top_n)
    return [dict(location=r.location, day=r.dow_name, hour=int(r.hour),
                 expected_events=round(float(r.risk_per_day), 2))
            for r in h.itertuples()]


# ============================ BACKTEST (honest validation) ============================
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(os.path.join(_HERE, "astram_enriched.csv"), dtype=str, keep_default_na=False)
    df["y"] = df["requires_road_closure"].str.upper().eq("TRUE").astype(int)
    s = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True, format="ISO8601")
    cd = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True, format="ISO8601")
    df["dur_min"] = (cd - s).dt.total_seconds() / 60

    # rebuild the SAME held-out 25% Model 1 never trained on (random_state=42, stratified)
    idx = np.arange(len(df))
    _, ite = train_test_split(idx, test_size=0.25, random_state=42, stratify=df["y"].values)
    test = df.iloc[ite].copy()
    test["p"] = closure_proba(test)
    test["flag_high"] = test["p"] >= CLOSURE_THRESHOLD

    print("=" * 78)
    print("BACKTEST on held-out 25% (events Model 1 never saw) — n =", len(test))
    print("=" * 78)
    flagged = test[test["flag_high"]]
    notflag = test[~test["flag_high"]]
    print(f"\nQ1: For events we FLAG high-closure-risk (p>= {CLOSURE_THRESHOLD}), did they actually close?")
    print(f"   flagged events                : {len(flagged)}")
    print(f"   of those, ACTUALLY closed      : {int(flagged.y.sum())}  "
          f"({flagged.y.mean():.0%} actual closure rate)")
    print(f"   NOT-flagged events closure rate: {notflag.y.mean():.1%}   (n={len(notflag)})")
    print(f"   base rate (all test events)    : {test.y.mean():.1%}")
    lift = flagged.y.mean() / test.y.mean()
    print(f"   -> flagged events close {lift:.1f}x more often than the average event (honest lift)")

    print(f"\nQ2: Do flagged events last LONGER?  (duration is administrative / only "
          f"{100*test.dur_min.notna().mean():.0f}% populated, p90 huge — treat as COARSE signal only)")
    fd = flagged.dur_min.dropna(); nd = notflag.dur_min.dropna()
    if len(fd) and len(nd):
        print(f"   flagged median duration   : {fd.median():.0f} min   (n={len(fd)})")
        print(f"   not-flagged median duration: {nd.median():.0f} min   (n={len(nd)})")
        print(f"   NOTE: medians are noisy and durations are admin timestamps; we do NOT "
              f"claim precise duration prediction.")

    # confusion on the recommendation itself
    tp = int(((test.flag_high) & (test.y == 1)).sum()); fp = int(((test.flag_high) & (test.y == 0)).sum())
    fn = int(((~test.flag_high) & (test.y == 1)).sum()); tn = int(((~test.flag_high) & (test.y == 0)).sum())
    print(f"\nQ3: Recommendation confusion @ threshold {CLOSURE_THRESHOLD}:")
    print(f"   TP={tp} (flagged & closed)   FP={fp} (flagged, no close)")
    print(f"   FN={fn} (missed closures)    TN={tn}")
    print(f"   precision={tp/(tp+fp):.2f}  recall={tp/(tp+fn):.2f}  "
          f"-> we catch {tp/(tp+fn):.0%} of real closures, {fp} false alarms is the cost")

    # demo: a few example recommendations
    print("\n" + "=" * 78)
    print("EXAMPLE RECOMMENDATIONS (event in -> recommendation out)")
    print("=" * 78)
    for _, ev in test.sort_values("p", ascending=False).head(2).iterrows():
        r = recommend(ev.to_dict())
        print(f"\n[{ev.event_cause} on {ev.corridor}] -> {r['manpower_officers']} officers, "
              f"barricade={r['barricade']['needed']}")
        print("   ", r["rationale"])
    low = test.sort_values("p").iloc[0]
    rl = recommend(low.to_dict())
    print(f"\n[{low.event_cause} on {low.corridor}] -> {rl['manpower_officers']} officers, "
          f"barricade={rl['barricade']['needed']}")
    print("   ", rl["rationale"])

    print("\n" + "=" * 78)
    print("PRE-POSITIONING (standing deploy-here list from Model 2, named locations):")
    print("=" * 78)
    for p in prepositioning(top_n=8):
        print(f"   {p['location']:<22} {p['day']} {p['hour']:02d}:00  "
              f"~{p['expected_events']} events/occurrence")
