import pandas as pd, numpy as np, re

SRC="/Users/agharisu/.meshclaw/uploads/877ef744d1874837afe96fa41f9f6910_Astram_event_data_anonymized_-_Astram_event_data_anonymizedb40ac87__2_.csv"
OUT="/Users/agharisu/.meshclaw/workspace/blr-traffic-hackathon/astram_with_locality_hotspot.csv"

raw=pd.read_csv(SRC,dtype=str,keep_default_na=False)        # keep raw exactly as-is
df=raw.copy()
NA={"NULL","null","","NaN","nan"}

def locality_from_address(addr):
    """Pick the locality segment from the address (the part just before 'Bengaluru')."""
    if not addr or addr in NA: return None
    parts=[p.strip() for p in re.split(r"[,]", addr) if p.strip()]
    # find first segment that mentions Bengaluru/Bangalore -> locality is the one before it
    for i,p in enumerate(parts):
        if re.search(r"bengaluru|bangalore", p, re.I):
            if i-1>=0: return parts[i-1]
            break
    # fallback: second segment if present (skip the street name)
    return parts[1] if len(parts)>1 else (parts[0] if parts else None)

lat=pd.to_numeric(df["latitude"],errors="coerce")
lon=pd.to_numeric(df["longitude"],errors="coerce")
corr=df["corridor"].astype(str).str.strip()
is_named = (~corr.isin(NA)) & (corr.str.lower()!="non-corridor")

# ---- build locality_hotspot ----
hotspot=pd.Series(index=df.index, dtype=object)

# 1) named corridors keep their corridor name
hotspot[is_named]=corr[is_named]

# 2) non-corridor -> 1km grid clustering + address-derived name
nc_mask = ~is_named & lat.notna() & lon.notna()
GRID=0.009  # ~1 km
cell=(lat/GRID).round().astype("Int64").astype(str)+"_"+(lon/GRID).round().astype("Int64").astype(str)
addr_loc=df["address"].apply(locality_from_address)
ps=df["police_station"].where(~df["police_station"].isin(NA))

tmp=pd.DataFrame({"cell":cell,"loc":addr_loc,"ps":ps})[nc_mask]
# name each cell = most common address-locality in it (fallback: most common police_station)
def cell_name(g):
    s=g["loc"].dropna()
    if len(s):
        m=s.value_counts()
        if m.iloc[0]>=1: return m.index[0]
    s2=g["ps"].dropna()
    return s2.value_counts().index[0] if len(s2) else None
names=tmp.groupby("cell").apply(cell_name)
cell_size=tmp["cell"].value_counts()

OTHER_CUTOFF=5
def resolve(c):
    if pd.isna(c): return "Other"
    if cell_size.get(c,0) < OTHER_CUTOFF: return "Other"
    nm=names.get(c)
    return nm if nm else "Other"
hotspot[nc_mask]=tmp["cell"].map(resolve)

# 3) anything left (non-corridor w/o coords, or blank corridor) -> Other
hotspot=hotspot.fillna("Other")
df["locality_hotspot"]=hotspot.values

df.to_csv(OUT,index=False)

# ---- report ----
print(f"Rows: {len(df)}   Output: {OUT}")
print(f"Named-corridor rows (kept corridor name): {is_named.sum()}")
print(f"Non-corridor rows (got locality):         {(~is_named).sum()}")
print(f"\nUnique locality_hotspot values: {df['locality_hotspot'].nunique()}")
ncv=df.loc[~is_named.values,"locality_hotspot"].value_counts()
print(f"\nTop non-corridor localities derived:")
print(ncv.head(20).to_string())
print(f"\n'Other' bucket size (non-corridor): {ncv.get('Other',0)} ({100*ncv.get('Other',0)/(~is_named).sum():.0f}% of non-corridor)")
print("\nSample rows:")
print(df.loc[~is_named.values,["latitude","longitude","police_station","corridor","locality_hotspot"]].head(8).to_string(index=False))
