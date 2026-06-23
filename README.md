# 🚦 Event Congestion Command Center

**Flipkart Gridlock 2.0 × Bengaluru Traffic Police — Theme 2: Event-Driven Congestion**

> Predicting traffic-event impact for the Bengaluru Traffic Police, built on **8,173 real events** from the ASTraM dataset. No simulations, no synthetic data.

---

## Why we built this

Every day, Bengaluru's traffic police respond to rallies, festivals, breakdowns, tree-falls, and accidents — and **94% of these events hit without warning**. Today the response is experience-driven: a senior officer's gut tells them how many people to send and where to barricade. We thought the history might already hold the answer.

So we trained three models on 8,173 past events to do three things:

1. **Predict** how likely an event is to close a road
2. **Forecast** where and when events are likely to happen
3. **Recommend** officers and barricades to deploy

Everything you'll see is grounded in real ASTraM data — and we report the honest numbers, not the flattering ones.

---

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
# → open http://127.0.0.1:5050
```

Click **▶ Replay sample events** to watch the system in action, or use the **Report Event** form to score a new event yourself.

---

## Approach

### 1. Data enrichment — three small scripts

The raw ASTraM file has 46 columns, many of them 98–99% empty. We added useful columns without inventing data:

| Script | Adds | Why |
|---|---|---|
| `add_locality_hotspot.py` | `locality_hotspot` | 38% of events sat in a useless "Non-corridor" blob. We snapped GPS to a 1 km grid and named each cluster from the address field. |
| `add_region.py` | `region` | A 3×3 coarse-geo grid over the data's bounding box. Cheap, useful both for the model and the map drill-down. |
| `add_time_features.py` | `hour`, `day_of_week`, `is_weekend`, `is_night`, `time_band` | Surfaces the bimodal rhythm — 2 AM freight peak + 10 AM morning rush. |

Result: `astram_enriched.csv` — 53 columns, 8,173 rows, model-ready.

### 2. Three models, one command center

| Model | Question | Method |
|---|---|---|
| **Closure Risk** | Will this event shut the road? | Tuned XGBoost classifier |
| **Hotspot Forecaster** | Where and when do events cluster? | Poisson rates with empirical Bayes shrinkage |
| **Recommendation** | What should the police do? | Transparent 3×3 policy grid (closure risk × cause severity) |

### 3. Per-prediction explainability

Every prediction surfaces its top 3 drivers (▲ raises risk, ▼ lowers risk). Operators see *why*, not just a number.

---

## The numbers (honest)

Held-out 25% test set — the model never saw any of it.

| Metric | Score | What it means |
|---|---|---|
| **ROC-AUC** | 0.80 | Correctly ranks a closure above a non-closure 4 times out of 5 |
| **PR-AUC** | 0.44 | ~5× better than random (closures are 8.3% of events) |
| **Recall @ 0.40** | 54% | Catches over half of real closures |
| **Recommendation backtest** | 4.2× | Flagged events close 4.2× more often than baseline |

We also re-validated under a strict **time-based split** (train Nov→Mar, test Mar→Apr — chronologically held-out). The model held — PR-AUC actually nudged up to **0.46**. No temporal leakage.

---

## What we tested and *didn't* build

Same discipline as what we kept. Features earn their place.

- **LP officer-count optimizer.** Not built. An optimizer needs a real objective. The data has no ground truth for "optimal" deployment — so optimization on top of a forecast would just be math hiding the same uncertainty. We chose a transparent policy + a backtest instead.
- **Duration regressor.** Not built. `resolved_datetime` is 99% NULL; `closed_datetime` has a heavy administrative tail (events held open for weeks). A duration model on those timestamps would give false precision.
- **Mappls routed diversion.** Architected, not built. Mappls's routing API can't be told "this road is blocked" — so the returned route could pass through the closure itself. We kept it as the next ML layer in the roadmap.

---

## What we caught (the hardest one to own)

While inspecting feature importances, two columns — `time_band_missing` and `veh_type_missing` — ranked higher than they should. We thought we'd found a leak.

The cause turned out to be ours: **our own enrichment script had a timestamp-parsing bug.** 116 events with timestamps lacking milliseconds were being silently coerced to `NaT` by pandas, which then propagated to blank time features. The model was learning a fingerprint of our bug, not real signal.

We fixed it at the source (`format='ISO8601'`), regenerated the CSV, retrained, and reported the post-fix number. PR-AUC went from a worry-inflated 0.45 to an honest 0.44 — barely changed, but provably clean.

> *We'd rather show a true 0.44 than a fake 0.99.*

---

## Why we tune for recall, not accuracy

At our **0.40 threshold**: precision ~0.34, recall ~0.54. We deliberately picked the lower threshold — for the police, a missed road closure is far more costly than a false-alarm patrol. The threshold is a dial BTP can tune from operational experience.

---

## Tech stack

Python · Flask · XGBoost · scikit-learn · pandas · NumPy · Leaflet · OpenStreetMap

Classical ML on purpose. 8,173 rows isn't deep-learning territory; gradient-boosted trees fit cleanly and the feature importances are readable.

---

## What's next (not built, honest about it)

- **Blast Radius** — congestion spread visualization (needs live road-flow data)
- **Diversion routing** — proper "avoid-this-edge" routing on OSM or live Mappls flow
- **Live ASTraM integration** — replace the manual form with the live BTP feed
- **Bayesian hyperparameter search** — Optuna; we expect a feature ceiling rather than a tuning one
- **Learning loop** — refine the recommendation policy from real deployment outcomes

---

## Team

**Abhijeet Kumar · Sujaatali Agharia · Muskan Jarwal**

---

*Honest models, real ASTraM data, a demo that runs live.*
