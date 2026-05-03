# NYC Taxi ETA Predictor — Gobblecube AI Builder Challenge

> Predict trip duration in seconds given pickup zone, dropoff zone, timestamp, and passenger count.
> Scored on MAE against a held-out 2024 NYC Yellow Taxi slice.

---

## Results

| Approach | Dev MAE | vs Baseline |
|---|---|---|
| Global mean prediction | ~580s | — |
| Zone-pair lookup (no ML) | 297.0s | −61.6s |
| XGBoost baseline (starter repo) | 358.6s | 0s (reference) |
| **LightGBM + feature engineering (ours, v1)** | **281.0s** | **−77.6s (21.6% better)** |

*Dev set = last 2 weeks of Dec 2023, ~1.23M trips. Eval set = held-out 2024 slice graded by Gobblecube.*

---

## Approach

### Step 1 — Exploratory Data Analysis

Before building any model, I ran a full EDA (`eda.py`) on the 36.7M training trips to understand what signals actually exist in the data.

**Key findings that shaped the model:**

**1. Target is strongly right-skewed (skew = 2.23)**
```
Min: 30s  |  p25: 466s  |  Median: 766s  |  Mean: 989s  |  p95: 2581s  |  Max: 10799s
```
77.6% of trips fall in the 5–30 minute range. Long tail of outliers (airport runs, outer borough trips). This motivated exploring log-transform of the target in future iterations.

**2. Time-of-day is the strongest single signal — 437s range across hours**
```
Fastest: 3am  → 733s avg
Slowest: 3pm  → 1170s avg
Range:   437s  (bigger than our entire improvement over baseline)
```
Afternoon congestion (2–4pm) is the dominant driver of long trips. Rush hour flags and cyclical hour encoding are high-priority features.

**3. Same-zone trips are fundamentally different**
```
Same zone (4.8% of trips):  avg 450s
Diff zone (95.2% of trips): avg 1016s
Difference: 566s
```
The `is_same_zone` binary feature carries enormous signal.

**4. Passenger count is a real signal (not noise)**
```
1 passenger:  968s avg
2 passengers: 1079s avg  (+11%)
4 passengers: 1125s avg  (+16%)
```
Groups tend to take airport/outer-borough trips. Kept as a feature.

**5. Zone-pair coverage: 63.6% of possible pairs seen in training**
```
44,697 / 70,225 pairs covered
51.9% of seen pairs have < 10 trips  ← sparse, need smoothing
2.3% of dev pairs never seen in training  ← need global fallback
```

**6. Minimal train/dev distribution shift**
```
Train mean: 989.4s  |  Dev mean: 998.3s  |  Difference: 8.9s
```
Dev is the last 2 weeks of Dec 2023 (holiday season). Slightly longer trips but not a major shift.

**7. Zone-pair lookup alone beats the GBT baseline**
```
Zone-pair median lookup: 297.0s MAE
XGBoost GBT baseline:   358.6s MAE
```
This told me the baseline was ignoring its most valuable signal. The lookup table is not a trick — it's a well-calibrated feature built purely from training data.

---

### Step 2 — Feature Engineering

Replaced the 6-feature baseline with 28 features grouped into four categories:

**Zone-pair statistics** (from training data only — no leakage)
- `pair_median` — median duration for this exact (pickup, dropoff) pair
- `pu_zone_median` — median duration starting from this pickup zone
- `do_zone_median` — median duration ending at this dropoff zone
- `pu_zone_hr_med` — median duration from this pickup zone at this specific hour
- `is_same_zone` — binary flag (pickup == dropoff)

**Time features**
- Raw: `hour`, `minute`, `dow`, `month`
- Cyclical: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` (avoids 23→0 discontinuity)
- `minute_of_day` (0–1439, finer granularity than hour)
- Flags: `is_rush_hour`, `is_weekend`, `is_night`, `is_holiday`

**Geographic features** *(zone centroids — partially working in v1, full in v2)*
- `pu_lat`, `pu_lon`, `do_lat`, `do_lon` — zone centroid coordinates
- `haversine_km` — straight-line distance between zone centroids
- `delta_lat`, `delta_lon` — directional displacement

**Passenger**
- `passenger_count`

---

### Step 3 — Model

Switched from XGBoost to **LightGBM** for faster training on 37M rows and better handling of high-cardinality categoricals.

```
Model:           LightGBM Regressor
n_estimators:    3000 (with early stopping, rounds=100)
learning_rate:   0.05
num_leaves:      255
subsample:       0.8
colsample_bytree: 0.8
device:          GPU (Kaggle T4)
Best iteration:  53  (early stopping kicked in — lookup features dominate)
Training time:   278s (vs 602s for XGBoost baseline)
```

The model stops at iteration 53 because `pair_median` essentially provides a near-optimal signal and additional trees have little marginal gain. This is expected behaviour, not a bug.

**Model artifact** (`model.pkl`) bundles everything needed for offline inference:
- Trained LightGBM model
- Zone-pair lookup dict
- Zone-level median dicts (pickup, dropoff, pickup×hour)
- Global fallback median
- Zone centroid coordinates

Zero external API calls at inference time. Inference latency: ~2–3ms per request (well within 200ms limit).

---

## What I Tried That Didn't Work (Yet)

**Zone centroid download** — The TLC shapefile (`taxi_zones.zip`) failed to extract correctly on the first run because the `.shp` file wasn't at the expected path inside the zip. This means haversine distance, delta_lat, and delta_lon all defaulted to zero — those features contributed nothing in v1. Fixed with `glob`-based path search + hardcoded fallback dict for v2.

---

## What I'd Do Next

In priority order, based on expected MAE reduction:

1. **Fix zone centroids → retrain** — haversine distance is a strong physical signal; expect ~10–15s improvement
2. **Log-transform the target** — `log1p(duration)` → predict → `expm1()` back. Skew of 2.23 means this could reduce MAE by 10–20s by preventing the model from over-weighting rare long trips
3. **Weather join** — Fetch NOAA hourly data for NYC (Central Park station). Precipitation and snow dramatically slow NYC traffic. Embed as a precomputed `{YYYY-MM-DDTHH: weather}` dict in `model.pkl` — no API calls at inference
4. **OSRM road distances** — Precompute a 265×265 matrix of actual road distances using OSRM (training-time only). Straight-line haversine misses the grid layout of Manhattan and river crossings
5. **More aggressive hyperparameter search** — The current model stops at iteration 53. A lower `learning_rate` (0.01–0.03) with more early-stopping patience might unlock more trees
6. **Ensemble** — Blend LightGBM with a dedicated zone-pair lookup correction model

---

## Repository Structure

```
├── predict.py          # Submission interface (fixed signature)
├── train.py            # Our training script (replaces baseline.py)
├── eda.py              # Exploratory data analysis
├── grade.py            # Local scoring harness (unchanged from starter)
├── baseline.py         # Original starter baseline (reference only)
├── model.pkl           # Trained model + all lookup artifacts
├── requirements.txt    # Dependencies
├── Dockerfile          # Container definition
└── data/
    ├── train.parquet   # 36.7M trips, Jan–Dec 17 2023
    ├── dev.parquet     # 1.23M trips, Dec 18–31 2023
    ├── schema.md
    └── download_data.py
```

---

## Reproducing the Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NYC TLC data (~500 MB, one-time)
python data/download_data.py

# 3. (Optional) Run EDA
python eda.py

# 4. Train the model
python train.py
# Produces model.pkl, prints Dev MAE

# 5. Grade locally
python grade.py
# Should print ~281s on 50k dev sample

# 6. Build and test Docker image
docker build -t my-eta .
docker run --rm -v $(pwd)/data:/work my-eta /work/dev.parquet /work/preds.csv
```

**Training environment:** Kaggle Notebook, T4 GPU, Python 3.11

---

## Git History

The commit log shows the full iteration trajectory:
- `chore`: verified baseline at 358.6s Dev MAE
- `analysis`: EDA — time signal 437s range, skew=2.23, lookup MAE=297s
- `feat`: LightGBM + 28 features → 281.0s Dev MAE (−77.6s)
- *(in progress)*: fix zone centroids, log-transform target → targeting <260s

---

*Submitted by Rohit Chiluka | agentic-hiring@gobblecube.ai*
