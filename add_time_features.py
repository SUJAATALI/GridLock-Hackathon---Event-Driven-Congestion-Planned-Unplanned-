import pandas as pd, numpy as np, os

_HERE=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.join(_HERE,"astram_enriched.csv")
OUT=os.path.join(_HERE,"astram_enriched.csv")  # update in place

df=pd.read_csv(SRC,dtype=str,keep_default_na=False)
# format="ISO8601" parses each row regardless of whether it carries milliseconds.
# Without it, pandas locks to one inferred format and silently NaTs the 116 no-millisecond
# rows ('...46+00'), which blanked their time features and leaked as a *_missing signal.
dt=pd.to_datetime(df["start_datetime"],errors="coerce",utc=True,format="ISO8601").dt.tz_convert("Asia/Kolkata")

df["hour"]      = dt.dt.hour.astype("Int64").astype(str).replace("<NA>","")
df["day_of_week"]= dt.dt.day_name().fillna("")
df["is_weekend"]= dt.dt.dayofweek.isin([5,6]).map({True:"yes",False:"no"}).where(dt.notna(),"")
df["is_night"]  = dt.dt.hour.apply(lambda h: "yes" if pd.notna(h) and (h>=22 or h<6) else ("no" if pd.notna(h) else ""))
def band(h):
    if pd.isna(h): return ""
    h=int(h)
    if 0<=h<6:  return "late_night"      # 02:00 freight peak lives here
    if 6<=h<10: return "early_morning"
    if 10<=h<13:return "morning_rush"     # 10-12 peak
    if 13<=h<17:return "afternoon"
    if 17<=h<21:return "evening"
    return "night"
df["time_band"]=dt.dt.hour.apply(band)

df.to_csv(OUT,index=False)

print(f"Wrote {OUT}  ({len(df)} rows, {df.shape[1]} cols)")
print("\nNew time columns sample:")
print(df[["start_datetime","hour","day_of_week","is_weekend","is_night","time_band"]].head(6).to_string(index=False))
print("\ntime_band distribution:")
print(df["time_band"].value_counts().to_string())
print("\nis_weekend:"); print(df["is_weekend"].value_counts().to_string())
# quick payoff check: closure rate by time_band
df["_c"]=df["requires_road_closure"].str.upper().eq("TRUE")
print("\nClosure rate by time_band:")
print((df.groupby("time_band")["_c"].mean()*100).round(1).to_string())
