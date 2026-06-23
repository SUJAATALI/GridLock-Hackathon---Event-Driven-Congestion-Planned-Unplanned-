import pandas as pd, numpy as np, os

# Adds the `region` column (8 city zones from a 3x3 grid over the data's lat/long box).
# Region is derived ENTIRELY from the provided dataset's latitude/longitude — no external data.

_HERE=os.path.dirname(os.path.abspath(__file__))
SRC=os.path.join(_HERE,"astram_with_locality_hotspot.csv")
OUT=os.path.join(_HERE,"astram_enriched.csv")

df=pd.read_csv(SRC,dtype=str,keep_default_na=False)
lat=pd.to_numeric(df["latitude"],errors="coerce"); lon=pd.to_numeric(df["longitude"],errors="coerce")

# ---- 3x3 region assignment (data bounding box) ----
lat_edges=np.linspace(12.801,13.268,4); lon_edges=np.linspace(77.309,77.769,4)
ns=["South","Central","North"]; ew=["West","Central","East"]
def cell_idx(la,lo):
    if pd.isna(la) or pd.isna(lo): return None
    i=min(max(np.searchsorted(lat_edges,la,side="right")-1,0),2)
    j=min(max(np.searchsorted(lon_edges,lo,side="right")-1,0),2)
    return (i,j)
idx=[cell_idx(a,o) for a,o in zip(lat,lon)]
df["region"]=[(f"{ns[t[0]]}-{ew[t[1]]}" if t else "Unknown") for t in idx]

df.to_csv(OUT,index=False)

print(f"Wrote {OUT}  ({len(df)} rows, {df.shape[1]} cols)")
print("\nregion distribution:")
print(df["region"].value_counts().to_string())
