import pandas as pd, numpy as np
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 100)

F = "/Users/agharisu/.meshclaw/uploads/65b5d9d708fa4500a3b6ae3d9f62b4b2_Astram_event_data_anonymized_-_Astram_event_data_anonymizedb40ac87.csv"
df = pd.read_csv(F, na_values=["NULL","null","",""], keep_default_na=True, low_memory=False)
df = df.dropna(how="all")  # drop fully-empty rows
print(f"ROWS={len(df)}  COLS={df.shape[1]}\n")

def pct(n): return f"{100*n/len(df):.1f}%"

# 1. event_type split (planned vs unplanned)
print("=== 1. event_type split ===")
print(df["event_type"].value_counts(dropna=False), "\n")

# 2. event_cause distribution
print("=== 2. event_cause distribution ===")
print(df["event_cause"].value_counts(dropna=False).head(20), "\n")

# 3. NULL rates on key fields
print("=== 3. NULL rates (key fields) ===")
for c in ["resolved_datetime","end_datetime","closed_datetime","cargo_material","age_of_truck",
          "route_path","assigned_to_police_id","reason_breakdown","veh_type","description",
          "requires_road_closure","corridor","priority","junction","zone","police_station"]:
    if c in df: print(f"  {c:24s} NULL={pct(df[c].isna().sum())}")
print()

# 4. Duration distribution: resolved - start
print("=== 4. Duration (resolved_datetime - start_datetime) ===")
for c in ["start_datetime","resolved_datetime","end_datetime","closed_datetime"]:
    df[c+"_p"] = pd.to_datetime(df[c], errors="coerce", utc=True)
dur = (df["resolved_datetime_p"] - df["start_datetime_p"]).dt.total_seconds()/60.0
print(f"  rows with both start+resolved: {dur.notna().sum()} ({pct(dur.notna().sum())})")
valid = dur[dur.notna()]
print(f"  negatives: {(valid<0).sum()}   zeros: {(valid==0).sum()}")
pos = valid[valid>0]
print(f"  positive durations: {len(pos)}")
if len(pos):
    print(pos.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(1))
# fallback: closed - start
dur2 = (df["closed_datetime_p"] - df["start_datetime_p"]).dt.total_seconds()/60.0
print(f"\n  [alt] closed-start available rows: {dur2.notna().sum()} ({pct(dur2.notna().sum())})")
print()

# status breakdown (open vs resolved vs closed)
print("=== status breakdown ===")
print(df["status"].value_counts(dropna=False), "\n")

# 5. corridor cardinality & concentration
print("=== 5. corridor cardinality/concentration ===")
print(f"  unique corridors: {df['corridor'].nunique(dropna=True)}")
print(df["corridor"].value_counts(dropna=False).head(15), "\n")

# 6. temporal span & density
print("=== 6. temporal span & density ===")
s = df["start_datetime_p"]
print(f"  min={s.min()}  max={s.max()}  span_days={(s.max()-s.min()).days}")
print(f"  events/day (mean over span): {len(df)/max((s.max()-s.min()).days,1):.1f}")
print("  by month:")
print(s.dt.to_period("M").value_counts().sort_index(), "\n")

# 7. geo spread
print("=== 7. geo spread ===")
print(f"  zone unique: {df['zone'].nunique()}   police_station unique: {df['police_station'].nunique()}")
print(df["zone"].value_counts(dropna=False).head(12), "\n")
lat = pd.to_numeric(df["latitude"], errors="coerce"); lon = pd.to_numeric(df["longitude"], errors="coerce")
print(f"  lat range {lat.min():.3f}..{lat.max():.3f}   lon range {lon.min():.3f}..{lon.max():.3f}")
print(f"  rows with valid coords: {pct(((lat.between(12,14))&(lon.between(77,78))).sum())}\n")

# 8. assigned_to_police_id manpower signal
print("=== 8. assigned_to_police_id ===")
ap = df["assigned_to_police_id"]
print(f"  non-null: {pct(ap.notna().sum())}  unique values: {ap.nunique()}")
print()

# extras useful for modelling
print("=== EXTRA: priority / requires_road_closure ===")
print(df["priority"].value_counts(dropna=False))
print(df["requires_road_closure"].value_counts(dropna=False))
print("\n=== EXTRA: veh_type ===")
print(df["veh_type"].value_counts(dropna=False).head(10))
