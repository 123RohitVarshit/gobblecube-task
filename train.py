#!/usr/bin/env python
"""
Phase 2: LightGBM + Rich Feature Engineering
Features added over baseline:
  - Zone-pair lookup table (median duration per pickup/dropoff pair)
  - Zone centroid lat/lon from NYC TLC shapefile
  - Haversine distance between zone centroids
  - Cyclical time encoding (hour_sin/cos, dow_sin/cos)
  - Rush hour, weekend, night, holiday flags
  - Zone-level aggregations (per-zone median, per-zone-hour median)
  - LightGBM with GPU + early stopping

Target: <270s Dev MAE (vs 358.6s baseline)
"""
from __future__ import annotations

import io, math, pickle, time, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import requests

DATA_DIR   = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
NYC_LAT, NYC_LON = 40.7128, -74.0060


# ---------------------------------------------------------------------------
# Zone centroids
# ---------------------------------------------------------------------------

def get_zone_centroids() -> dict[int, tuple[float, float]]:
    """Download NYC TLC shapefile and return {zone_id: (lat, lon)}."""
    try:
        import geopandas as gpd
        url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
        print("  Downloading zone shapefile...")
        r = requests.get(url, timeout=60)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall("/tmp/taxi_zones")
        gdf = gpd.read_file("/tmp/taxi_zones/taxi_zones.shp").to_crs("EPSG:4326")
        gdf["centroid"] = gdf.geometry.centroid
        centroids = {
            int(row["LocationID"]): (float(row["centroid"].y), float(row["centroid"].x))
            for _, row in gdf.iterrows()
        }
        print(f"  {len(centroids)} zone centroids computed")
        return centroids
    except Exception as e:
        print(f"  WARNING: centroid download failed ({e}). Using NYC default.")
        return {}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def engineer_features(
    df: pd.DataFrame,
    centroids: dict,
    pair_lookup: dict,
    zone_pu_med: dict,
    zone_do_med: dict,
    zone_hour_med: dict,
    global_med: float,
) -> pd.DataFrame:

    ts  = pd.to_datetime(df["requested_at"])
    pu  = df["pickup_zone"].astype(int)
    do_ = df["dropoff_zone"].astype(int)

    hour   = ts.dt.hour
    minute = ts.dt.minute
    dow    = ts.dt.dayofweek
    month  = ts.dt.month

    # Cyclical
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin  = np.sin(2 * np.pi * dow / 7)
    dow_cos  = np.cos(2 * np.pi * dow / 7)

    # Flags
    is_rush    = (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))) & (dow < 5)
    is_weekend = dow >= 5
    is_night   = (hour >= 22) | (hour <= 5)

    try:
        import holidays
        us_hols = holidays.US(years=[2022, 2023, 2024])
        is_holiday = ts.dt.date.map(lambda d: int(d in us_hols))
    except Exception:
        is_holiday = pd.Series(0, index=df.index)

    # Centroids
    def lat(z): return centroids.get(int(z), (NYC_LAT, NYC_LON))[0]
    def lon(z): return centroids.get(int(z), (NYC_LAT, NYC_LON))[1]

    pu_lat = pu.map(lat).values
    pu_lon = pu.map(lon).values
    do_lat = do_.map(lat).values
    do_lon = do_.map(lon).values
    hav    = haversine_vec(pu_lat, pu_lon, do_lat, do_lon)

    # Lookup tables
    pair_med   = np.array([pair_lookup.get((p, d), global_med) for p, d in zip(pu, do_)])
    pu_med_arr = pu.map(lambda z: zone_pu_med.get(z, global_med)).values
    do_med_arr = do_.map(lambda z: zone_do_med.get(z, global_med)).values
    zh_med_arr = np.array([
        zone_hour_med.get((p, h), zone_pu_med.get(p, global_med))
        for p, h in zip(pu, hour)
    ])

    return pd.DataFrame({
        "pickup_zone":      pu.values,
        "dropoff_zone":     do_.values,
        "hour":             hour.values,
        "minute":           minute.values,
        "dow":              dow.values,
        "month":            month.values,
        "hour_sin":         hour_sin.values,
        "hour_cos":         hour_cos.values,
        "dow_sin":          dow_sin.values,
        "dow_cos":          dow_cos.values,
        "minute_of_day":    (hour * 60 + minute).values,
        "is_rush_hour":     is_rush.astype(int).values,
        "is_weekend":       is_weekend.astype(int).values,
        "is_night":         is_night.astype(int).values,
        "is_holiday":       is_holiday.values,
        "is_same_zone":     (pu == do_).astype(int).values,
        "pu_lat":           pu_lat,
        "pu_lon":           pu_lon,
        "do_lat":           do_lat,
        "do_lon":           do_lon,
        "haversine_km":     hav,
        "delta_lat":        do_lat - pu_lat,
        "delta_lon":        do_lon - pu_lon,
        "pair_median":      pair_med,
        "pu_zone_median":   pu_med_arr,
        "do_zone_median":   do_med_arr,
        "pu_zone_hr_med":   zh_med_arr,
        "passenger_count":  df["passenger_count"].astype(int).values,
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    dev   = pd.read_parquet(DATA_DIR / "dev.parquet")
    print(f"  train: {len(train):,} rows | dev: {len(dev):,} rows")

    print("\nComputing zone centroids...")
    centroids = get_zone_centroids()

    print("\nBuilding lookup tables from training data...")
    pair_stats    = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
    pair_lookup   = {(int(p), int(d)): float(m)
                     for (p, d), m in pair_stats.median().items()}
    global_med    = float(train["duration_seconds"].median())
    zone_pu_med   = train.groupby("pickup_zone")["duration_seconds"].median().to_dict()
    zone_do_med   = train.groupby("dropoff_zone")["duration_seconds"].median().to_dict()

    tmp           = train.copy()
    tmp["hour"]   = pd.to_datetime(tmp["requested_at"]).dt.hour
    zone_hour_med = tmp.groupby(["pickup_zone", "hour"])["duration_seconds"].median().to_dict()

    print(f"  {len(pair_lookup):,} zone-pair medians")
    print(f"  {len(zone_hour_med):,} zone-hour medians")
    print(f"  global median: {global_med:.1f}s")

    fe_args = (centroids, pair_lookup, zone_pu_med, zone_do_med, zone_hour_med, global_med)

    print("\nEngineering features...")
    X_train = engineer_features(train, *fe_args)
    y_train = train["duration_seconds"].values
    X_dev   = engineer_features(dev, *fe_args)
    y_dev   = dev["duration_seconds"].values
    print(f"  {X_train.shape[1]} features | {X_train.shape[0]:,} train rows")

    print("\nTraining LightGBM (GPU)...")
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=255,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        device="gpu",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  trained in {time.time() - t0:.0f}s | best iter: {model.best_iteration_}")

    preds = model.predict(X_dev)
    mae   = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nDev MAE: {mae:.1f}s  (baseline: 358.6s | improvement: {358.6 - mae:.1f}s)")

    artifact = {
        "model":          model,
        "centroids":      centroids,
        "pair_lookup":    pair_lookup,
        "zone_pu_med":    zone_pu_med,
        "zone_do_med":    zone_do_med,
        "zone_hour_med":  zone_hour_med,
        "global_med":     global_med,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Saved model.pkl ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

