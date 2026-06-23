"""
MODEL 2 — Corridor x Hour x Day-of-week hotspot forecaster.
"Where / when are traffic events likely?"  -> expected-event-count (risk score) per cell.

Approach (simple + explainable): exposure-based Poisson rates.
  cell = (corridor, hour, day_of_week)
  exposure(window) = number of calendar days in the window matching that day_of_week
                     (each matching day = one chance for an event at that corridor+hour)
  rate = events / exposure  = expected events on a matching day
Sparse cells are stabilized by SHRINKAGE toward the coarser corridor x hour rate
(empirical-Bayes style: blend with k pseudo-observations of the parent rate).
predicted test count = shrunk_train_rate * test_exposure.

Validation: TIME split — train on the earlier ~80% of days, test on the later ~20%.
Compare the model against two naive baselines on held-out per-cell counts:
  - flat   : every cell gets the global average rate (the "naive average")
  - corridor-marginal : per-corridor rate, but no hour/day structure
Metrics: MAE, RMSE, Poisson deviance (lower=better), Spearman rank corr (higher=better),
and recall@top-K busy cells (do the predicted hotspots capture the real events?).
"""
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr

GROUP = "locality_hotspot"  # v2: group on the split locality (named arterials keep their name;
                            # the old "Non-corridor" 3124-blob is now real areas: Jayanagar, KR Puram...)
NAMED_BLOB = "Other"        # the residual unnamed bucket within locality_hotspot (~504 events)
SHRINK_K = 5.0              # pseudo-count for shrinkage toward parent (location x hour) rate

df = pd.read_csv("astram_enriched.csv", dtype=str, keep_default_na=False)
dt = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True, format="ISO8601").dt.tz_convert("Asia/Kolkata")
df["date"] = dt.dt.normalize()
df["hour"] = dt.dt.hour
df["dow"] = dt.dt.dayofweek            # 0=Mon..6=Sun
df = df.dropna(subset=["hour", "dow"]).copy()
df["hour"] = df["hour"].astype(int); df["dow"] = df["dow"].astype(int)

# ---- TIME split: earlier 80% of the calendar span = train, later 20% = test ----
days = pd.Series(sorted(df["date"].unique()))
cut = days.iloc[int(len(days) * 0.80)]
tr = df[df["date"] < cut].copy()
te = df[df["date"] >= cut].copy()
print(f"days={len(days)}  cutoff={cut.date()}  train_events={len(tr)}  test_events={len(te)}")

corridors = sorted(df[GROUP].unique())
hours = list(range(24))
dows = list(range(7))


def dow_exposure(frame):
    """#days in the frame's date range matching each day-of-week -> dict dow->#days."""
    d = pd.Series(sorted(frame["date"].unique()))
    wd = pd.to_datetime(d).dt.dayofweek
    return wd.value_counts().to_dict()


exp_tr = dow_exposure(tr)              # train slots per dow
exp_te = dow_exposure(te)             # test slots per dow
n_train_days = tr["date"].nunique()

# ---- full cell grid ----
grid = pd.MultiIndex.from_product([corridors, hours, dows], names=[GROUP, "hour", "dow"]).to_frame(index=False)

# train counts per cell
tc = tr.groupby([GROUP, "hour", "dow"]).size().rename("train_count")
grid = grid.merge(tc, on=[GROUP, "hour", "dow"], how="left").fillna({"train_count": 0})
# parent (corridor x hour) train count, summed over dow -> rate per train day
pc = tr.groupby([GROUP, "hour"]).size().rename("parent_count")
grid = grid.merge(pc, on=[GROUP, "hour"], how="left").fillna({"parent_count": 0})

grid["exp_tr"] = grid["dow"].map(exp_tr).fillna(0)
grid["exp_te"] = grid["dow"].map(exp_te).fillna(0)

# rates
grid["cell_rate"] = grid["train_count"] / grid["exp_tr"].replace(0, np.nan)
grid["parent_rate"] = grid["parent_count"] / n_train_days        # per any train day (avg over dow)
# shrink cell rate toward parent rate
grid["shrunk_rate"] = (grid["train_count"] + SHRINK_K * grid["parent_rate"]) / (grid["exp_tr"] + SHRINK_K)

# global naive rate per (corridor,hour,day) slot
global_rate = len(tr) / (len(corridors) * 24 * n_train_days)
# corridor-marginal rate: per-corridor events per (hour,day) slot
corr_rate = (tr.groupby(GROUP).size() / (24 * n_train_days)).to_dict()
grid["corr_rate"] = grid[GROUP].map(corr_rate).fillna(0)

# ---- predictions for the test window ----
grid["pred_model"] = grid["shrunk_rate"] * grid["exp_te"]
grid["pred_flat"] = global_rate * grid["exp_te"]
grid["pred_corr"] = grid["corr_rate"] * grid["exp_te"]

# actual test counts
ac = te.groupby([GROUP, "hour", "dow"]).size().rename("actual")
grid = grid.merge(ac, on=[GROUP, "hour", "dow"], how="left").fillna({"actual": 0})


def poisson_dev(actual, pred):
    pred = np.clip(pred, 1e-9, None)
    a = np.where(actual > 0, actual * np.log(actual / pred), 0.0)
    return 2 * np.sum(a - (actual - pred))


def metrics(name, pred):
    a = grid["actual"].values
    mae = np.mean(np.abs(a - pred))
    rmse = np.sqrt(np.mean((a - pred) ** 2))
    dev = poisson_dev(a, pred)
    rho = spearmanr(pred, a).correlation
    return dict(model=name, MAE=mae, RMSE=rmse, PoissonDev=dev, Spearman=rho)


rows = [metrics("Model (cell+shrink)", grid["pred_model"].values),
        metrics("Naive corridor-only", grid["pred_corr"].values),
        metrics("Naive flat average", grid["pred_flat"].values)]
res = pd.DataFrame(rows)

print("\n" + "=" * 84)
print("HELD-OUT VALIDATION (later 20% of dates) — per corridor x hour x dow cell")
print("=" * 84)
hdr = f"{'model':<24}{'MAE':>9}{'RMSE':>9}{'PoissonDev':>13}{'Spearman':>11}"
print(hdr); print("-" * len(hdr))
for _, r in res.iterrows():
    print(f"{r.model:<24}{r.MAE:>9.4f}{r.RMSE:>9.4f}{r.PoissonDev:>13.1f}{r.Spearman:>11.3f}")
print("-" * len(hdr))
m, fl = res.iloc[0], res.iloc[2]
print(f"Model vs naive-flat:  MAE {fl.MAE:.4f}->{m.MAE:.4f} ({100*(fl.MAE-m.MAE)/fl.MAE:+.1f}%)   "
      f"PoissonDev {fl.PoissonDev:.0f}->{m.PoissonDev:.0f} ({100*(fl.PoissonDev-m.PoissonDev)/fl.PoissonDev:+.1f}%)")

# ---- recall@top-K: do predicted hotspots capture the real test events? ----
print("\nRecall@top-K busy cells (share of actual test events falling in the K highest-risk cells):")
total_actual = grid["actual"].sum()
for K in (20, 50, 100):
    top_model = grid.nlargest(K, "pred_model")["actual"].sum() / total_actual
    top_corr = grid.nlargest(K, "pred_corr")["actual"].sum() / total_actual
    # flat can't rank (ties) -> a random-ish K/N share as reference
    ref = K / len(grid)
    print(f"  K={K:>3}:  model={top_model:5.1%}   corridor-only={top_corr:5.1%}   (flat≈{ref:4.1%} of cells)")

# ---- final risk table on FULL data (the deliverable) ----
# location centroids so the dashboard can plot each named area at its own point
loc_centroid = (df.assign(lat=pd.to_numeric(df["latitude"], errors="coerce"),
                          lon=pd.to_numeric(df["longitude"], errors="coerce"))
                .groupby(GROUP).agg(lat=("lat", "median"), lon=("lon", "median")))

print("\n" + "=" * 84)
print("TOP 15 HIGH-RISK  location x hour x day-of-week  (full-data model, expected events/occurrence)")
print("(location = locality_hotspot: named arterials + the split-out former Non-corridor areas)")
print("=" * 84)
exp_all = dow_exposure(df)
full_cell = df.groupby([GROUP, "hour", "dow"]).size().rename("count").reset_index()
full_cell["exposure"] = full_cell["dow"].map(exp_all)
full_cell["risk_per_day"] = full_cell["count"] / full_cell["exposure"]
DOW = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
full_cell["dow_name"] = full_cell["dow"].map(DOW)
# named view = exclude the residual "Other" bucket (too generic to send a patrol to)
named = full_cell[full_cell[GROUP] != NAMED_BLOB]
top = named.sort_values("risk_per_day", ascending=False).head(15)
print(f"  {'location':<22}{'dow':>5}{'hour':>6}{'tot_events':>12}{'exp.events/occurrence':>23}")
for _, r in top.iterrows():
    print(f"  {r[GROUP]:<22}{r.dow_name:>5}{int(r.hour):>6}{int(r['count']):>12}{r.risk_per_day:>23.2f}")

# ---- also show location x hour (collapsed over dow) — denser, demo-friendly ----
print("\nTOP 12 high-risk location x hour (collapsed over day-of-week, named only):")
ch = df[df[GROUP] != NAMED_BLOB].groupby([GROUP, "hour"]).size().rename("count").reset_index()
ch["risk_per_day"] = ch["count"] / df["date"].nunique()
top_ch = ch.sort_values("count", ascending=False).head(12)
print(f"  {'location':<22}{'hour':>6}{'tot_events':>12}{'avg/day':>10}")
for _, r in top_ch.iterrows():
    print(f"  {r[GROUP]:<22}{int(r.hour):>6}{int(r['count']):>12}{r.risk_per_day:>10.2f}")

# save the full risk table with a stable 'location' column (+ centroids + named flag)
out = full_cell.merge(loc_centroid, left_on=GROUP, right_index=True, how="left")
out = out.rename(columns={GROUP: "location"})
out["named"] = out["location"] != NAMED_BLOB
out[["location", "hour", "dow", "dow_name", "count", "exposure", "risk_per_day",
     "lat", "lon", "named"]].to_csv("hotspot_risk_table.csv", index=False)
print(f"\nsaved full risk table (grouped on {GROUP}) -> hotspot_risk_table.csv "
      f"[{out['location'].nunique()} locations]")
