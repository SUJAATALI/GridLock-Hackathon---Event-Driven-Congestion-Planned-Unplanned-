"""
Event Congestion Command Center — backend (Flask).
Ties Model 1 (closure risk) + Model 2 (hotspots) + Model 3 (recommendation) together
and serves a single-page control-room dashboard on localhost.

Endpoints:
  GET  /                  -> the dashboard page
  GET  /api/meta          -> dropdown options (causes, corridors w/ centroids) + map center
  GET  /api/hotspots      -> corridor x hour x dow risk cells (for heatmap) + named top list
  GET  /api/prepositioning-> top deploy-here cells (Model 2 via recommend.prepositioning)
  GET  /api/sample-events -> a deterministic replay set for the live demo (no typing needed)
  POST /api/predict       -> {event fields} -> Model1 closure risk + Model3 recommendation

Design rule: NO flaky external call on the critical path. The map uses Leaflet + OSM tiles
in the browser; all model logic is local. MapmyIndia is optional and only used client-side
if a key is present — the page always works without it.
"""
import os, math
import pandas as pd, numpy as np
from flask import Flask, jsonify, request, send_from_directory

import recommend  # Model 1 + Model 3 API, and reads hotspot_risk_table.csv (Model 2)

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"))

# ---- load enriched data once for metadata / corridor centroids / sample events ----
_DF = pd.read_csv(os.path.join(HERE, "astram_enriched.csv"), dtype=str, keep_default_na=False)
_DF["_lat"] = pd.to_numeric(_DF["latitude"], errors="coerce")
_DF["_lon"] = pd.to_numeric(_DF["longitude"], errors="coerce")

MAP_CENTER = [float(_DF["_lat"].median()), float(_DF["_lon"].median())]

# bucket rare causes (per the brief) but keep the meaningful ones
_CAUSE_COUNTS = _DF["event_cause"].value_counts()
CAUSES = [c for c in _CAUSE_COUNTS.index if _CAUSE_COUNTS[c] >= 15]

# LOCATION dropdown is keyed on locality_hotspot (the split column): named arterials keep their
# name, and the former "Non-corridor (3124)" blob is now real areas (Jayanagar, KR Puram, ...).
# Each location carries its own centroid + dominant corridor/region so the form needs only
# cause + location + time.
_LOC = "locality_hotspot"
_cent = (_DF.assign(lat=_DF["_lat"], lon=_DF["_lon"])
         .groupby(_LOC)
         .agg(lat=("lat", "median"), lon=("lon", "median"), n=("lat", "size")))
def _mode(s):
    m = s[s != ""].mode()
    return m.iloc[0] if len(m) else "missing"
_region_by_loc = _DF.groupby(_LOC)["region"].apply(_mode).to_dict()
_corr_by_loc = _DF.groupby(_LOC)["corridor"].apply(_mode).to_dict()
CORRIDORS = []   # kept name for frontend compat; entries are now locations
for c, r in _cent.sort_values("n", ascending=False).iterrows():
    if int(r.n) < 5:          # tiny clusters add noise to the dropdown; "Other" still covers them
        continue
    CORRIDORS.append(dict(corridor=c, lat=round(float(r.lat), 5), lon=round(float(r.lon), 5),
                          n=int(r.n), region=_region_by_loc.get(c, "missing"),
                          locality_hotspot=c, parent_corridor=_corr_by_loc.get(c, "Non-corridor")))

DOW_NAME = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
            4: "Friday", 5: "Saturday", 6: "Sunday"}


def time_band(h):
    if h is None:
        return "missing"
    if 0 <= h < 6:  return "late_night"
    if 6 <= h < 10: return "early_morning"
    if 10 <= h < 13: return "morning_rush"
    if 13 <= h < 17: return "afternoon"
    if 17 <= h < 21: return "evening"
    return "night"


def build_event(payload: dict) -> dict:
    """Form gives cause + location + hour + dow (+ optional lat/lon). Derive the full
    enriched feature row the models expect, exactly mirroring the enrichment scripts.
    The form's `corridor` field now carries a LOCATION (locality_hotspot value)."""
    location = payload.get("corridor", "Other")          # frontend still posts as `corridor`
    cinfo = next((c for c in CORRIDORS if c["corridor"] == location), None)
    hour = payload.get("hour")
    hour = int(hour) if hour not in (None, "") else None
    dow = payload.get("dow")
    dow = int(dow) if dow not in (None, "") else 1
    lat = payload.get("lat") or (cinfo["lat"] if cinfo else MAP_CENTER[0])
    lon = payload.get("lon") or (cinfo["lon"] if cinfo else MAP_CENTER[1])
    return {
        "event_type": payload.get("event_type", "unplanned"),
        "event_cause": payload.get("event_cause", "others"),
        "corridor": (cinfo["parent_corridor"] if cinfo else "Non-corridor"),
        "locality_hotspot": location,
        "region": payload.get("region") or (cinfo["region"] if cinfo else "missing"),
        "veh_type": payload.get("veh_type", "missing"),
        "time_band": time_band(hour),
        "day_of_week": DOW_NAME.get(dow, "Tuesday"),
        "is_weekend": "yes" if dow in (5, 6) else "no",
        "is_night": "yes" if (hour is not None and (hour >= 22 or hour < 6)) else "no",
        "hour": hour if hour is not None else -1,
        "junction": payload.get("junction", "NULL"),
        "_lat": float(lat), "_lon": float(lon),
    }


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/meta")
def meta():
    return jsonify(dict(center=MAP_CENTER, causes=CAUSES, corridors=CORRIDORS,
                        mapmyindia_key=os.environ.get("MAPMYINDIA_KEY", "")))


@app.route("/api/hotspots")
def hotspots():
    # Model 2 v2 table is keyed on `location` (locality_hotspot) and carries its own centroid.
    h = recommend._HOTSPOT.copy()
    h = h[(h["risk_per_day"] > 0) & h["lat"].notna()]
    cells = [dict(corridor=r.location, hour=int(r.hour), dow_name=r.dow_name,
                  risk=round(float(r.risk_per_day), 3),
                  lat=round(float(r.lat), 5), lon=round(float(r.lon), 5),
                  named=bool(r.named))
             for r in h.itertuples()]
    return jsonify(cells)


@app.route("/api/prepositioning")
def prepositioning():
    return jsonify(recommend.prepositioning(top_n=10, named_only=True))


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True) or {}
    ev = build_event(payload)
    rec = recommend.recommend(ev)
    rec["lat"], rec["lon"] = ev["_lat"], ev["_lon"]
    rec["event_cause"] = ev["event_cause"]
    rec["corridor"] = ev["locality_hotspot"]   # surface the location name for display
    return jsonify(rec)


@app.route("/api/sample-events")
def sample_events():
    """Deterministic replay set spanning the risk spectrum — for a hands-free live demo."""
    samples = [
        dict(event_cause="tree_fall", corridor="Mysore Road", hour=3, dow=1, event_type="unplanned"),
        dict(event_cause="construction", corridor="ORR East 2", hour=11, dow=3, event_type="planned"),
        dict(event_cause="vehicle_breakdown", corridor="Jayanagar", hour=4, dow=2, event_type="unplanned"),
        dict(event_cause="water_logging", corridor="Bellary Road 1", hour=10, dow=2, event_type="unplanned"),
        dict(event_cause="public_event", corridor="Whitefield", hour=12, dow=6, event_type="planned"),
        dict(event_cause="pot_holes", corridor="Koramangala", hour=15, dow=4, event_type="unplanned"),
        dict(event_cause="accident", corridor="Bannerghata Road", hour=18, dow=0, event_type="unplanned"),
        dict(event_cause="vip_movement", corridor="KR Puram", hour=9, dow=3, event_type="planned"),
    ]
    out = []
    for s in samples:
        ev = build_event(s)
        rec = recommend.recommend(ev)
        rec.update(lat=ev["_lat"], lon=ev["_lon"], event_cause=ev["event_cause"],
                   corridor=ev["corridor"], hour=s["hour"], dow_name=DOW_NAME[s["dow"]])
        out.append(rec)
    return jsonify(out)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Event Congestion Command Center -> http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
