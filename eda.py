"""
EDA for NYC Taxi ETA Challenge
Run this as cells in the Kaggle notebook BEFORE training.
Answers:
  1. What does trip duration look like? (distribution, outliers)
  2. Which zones are busiest?
  3. How strong is the time-of-day signal?
  4. Does passenger_count actually matter?
  5. Zone-pair coverage: how many unseen pairs will the eval have?
  6. What's the dev vs train distribution shift?
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/kaggle/working/gobblecube-task/data")

print("Loading data...")
train = pd.read_parquet(DATA_DIR / "train.parquet")
dev   = pd.read_parquet(DATA_DIR / "dev.parquet")

train["ts"] = pd.to_datetime(train["requested_at"])
dev["ts"]   = pd.to_datetime(dev["requested_at"])

print(f"Train: {len(train):,} rows | Dev: {len(dev):,} rows")
print(f"Train date range: {train['ts'].min()} → {train['ts'].max()}")
print(f"Dev   date range: {dev['ts'].min()}   → {dev['ts'].max()}")

# ─────────────────────────────────────────────────────────────────
# 1. TRIP DURATION DISTRIBUTION
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("1. TRIP DURATION (target variable)")
print("="*60)
dur = train["duration_seconds"]
print(f"  Min:    {dur.min():.0f}s  ({dur.min()/60:.1f} min)")
print(f"  Max:    {dur.max():.0f}s  ({dur.max()/60:.1f} min)")
print(f"  Mean:   {dur.mean():.1f}s  ({dur.mean()/60:.1f} min)")
print(f"  Median: {dur.median():.1f}s  ({dur.median()/60:.1f} min)")
print(f"  Std:    {dur.std():.1f}s")
print(f"  Skew:   {dur.skew():.2f}  (>0 means right-skewed / long tail)")
print("\n  Percentiles:")
for p in [5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    p{p:2d}: {np.percentile(dur, p):.0f}s  ({np.percentile(dur, p)/60:.1f} min)")

print(f"\n  Trips under 2 min : {(dur < 120).sum():,}  ({(dur < 120).mean()*100:.1f}%)")
print(f"  Trips over 1 hour : {(dur > 3600).sum():,}  ({(dur > 3600).mean()*100:.1f}%)")
print(f"  Trips 5-30 min    : {((dur >= 300) & (dur <= 1800)).sum():,}  ({((dur >= 300) & (dur <= 1800)).mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# 2. ZONE ANALYSIS
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. ZONE ANALYSIS")
print("="*60)

n_pu_zones = train["pickup_zone"].nunique()
n_do_zones = train["dropoff_zone"].nunique()
print(f"  Unique pickup zones:  {n_pu_zones} / 265")
print(f"  Unique dropoff zones: {n_do_zones} / 265")

# Top 10 busiest pickup zones
print("\n  Top 10 pickup zones (by trip count):")
top_pu = train["pickup_zone"].value_counts().head(10)
for zone, count in top_pu.items():
    print(f"    Zone {zone:3d}: {count:,} trips ({count/len(train)*100:.1f}%)")

# Zone-pair coverage
all_pairs   = train.groupby(["pickup_zone", "dropoff_zone"]).size()
n_pairs     = len(all_pairs)
total_pairs = 265 * 265
print(f"\n  Zone pairs seen in training: {n_pairs:,} / {total_pairs:,} ({n_pairs/total_pairs*100:.1f}%)")

# Pairs with very few trips (unreliable median)
sparse_pairs = (all_pairs < 10).sum()
print(f"  Pairs with < 10 trips: {sparse_pairs:,} ({sparse_pairs/n_pairs*100:.1f}% of seen pairs)")

# How many dev pairs are unseen in train?
dev_pairs = set(zip(dev["pickup_zone"], dev["dropoff_zone"]))
train_pairs = set(zip(train["pickup_zone"], train["dropoff_zone"]))
unseen = dev_pairs - train_pairs
print(f"\n  Dev pairs not in train: {len(unseen):,} / {len(dev_pairs):,} ({len(unseen)/len(dev_pairs)*100:.1f}%)")
print(f"  → These need the global/zone fallback in our lookup table")

# ─────────────────────────────────────────────────────────────────
# 3. TIME SIGNAL
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. TIME-OF-DAY SIGNAL")
print("="*60)

train["hour"] = train["ts"].dt.hour
train["dow"]  = train["ts"].dt.dayofweek

hour_means = train.groupby("hour")["duration_seconds"].mean()
print("\n  Mean trip duration by hour:")
for h, m in hour_means.items():
    bar = "█" * int(m / 30)
    print(f"    {h:02d}h: {m:5.0f}s  {bar}")

dow_means = train.groupby("dow")["duration_seconds"].mean()
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("\n  Mean trip duration by day of week:")
for d, m in dow_means.items():
    print(f"    {days[d]}: {m:.0f}s")

month_means = train.groupby(train["ts"].dt.month)["duration_seconds"].mean()
print("\n  Mean trip duration by month:")
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for mo, m in month_means.items():
    print(f"    {months[mo-1]}: {m:.0f}s")

# ─────────────────────────────────────────────────────────────────
# 4. PASSENGER COUNT
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4. PASSENGER COUNT")
print("="*60)
pax_stats = train.groupby("passenger_count")["duration_seconds"].agg(["mean", "count"])
print(pax_stats.to_string())
print("\n  → If means are similar across counts, passenger_count is a WEAK feature")

# ─────────────────────────────────────────────────────────────────
# 5. SAME-ZONE TRIPS
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. SAME-ZONE TRIPS (pickup == dropoff)")
print("="*60)
same_zone = train[train["pickup_zone"] == train["dropoff_zone"]]
diff_zone = train[train["pickup_zone"] != train["dropoff_zone"]]
print(f"  Same-zone trips: {len(same_zone):,} ({len(same_zone)/len(train)*100:.1f}%)")
print(f"    Avg duration: {same_zone['duration_seconds'].mean():.0f}s")
print(f"  Diff-zone trips: {len(diff_zone):,}")
print(f"    Avg duration: {diff_zone['duration_seconds'].mean():.0f}s")

# ─────────────────────────────────────────────────────────────────
# 6. TRAIN vs DEV DISTRIBUTION SHIFT
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("6. TRAIN vs DEV DISTRIBUTION SHIFT")
print("="*60)
print(f"  Train mean: {train['duration_seconds'].mean():.1f}s")
print(f"  Dev mean:   {dev['duration_seconds'].mean():.1f}s")
print(f"  Difference: {abs(train['duration_seconds'].mean() - dev['duration_seconds'].mean()):.1f}s")
print(f"\n  Train median: {train['duration_seconds'].median():.1f}s")
print(f"  Dev median:   {dev['duration_seconds'].median():.1f}s")
print("\n  Dev is last 2 weeks of Dec 2023 — may have holiday effects")
dev["hour"] = dev["ts"].dt.hour
dev["dow"]  = dev["ts"].dt.dayofweek
print(f"  Dev hour distribution (top 5):")
print(dev["hour"].value_counts().head(5).to_string())

# ─────────────────────────────────────────────────────────────────
# 7. QUICK ZONE-PAIR LOOKUP BASELINE
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("7. ZONE-PAIR LOOKUP TABLE (naive 10-line baseline)")
print("="*60)
pair_median = train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"].median()
global_median = train["duration_seconds"].median()

dev_preds = dev.apply(
    lambda r: pair_median.get((r["pickup_zone"], r["dropoff_zone"]), global_median),
    axis=1
)
lookup_mae = np.mean(np.abs(dev_preds.values - dev["duration_seconds"].values))
print(f"  Zone-pair lookup MAE on Dev: {lookup_mae:.1f}s")
print(f"  XGBoost baseline MAE on Dev: ~358.6s")
print(f"  Lookup beats baseline by:    {358.6 - lookup_mae:.1f}s")
print(f"\n  → This confirms the naive lookup is our floor to beat")

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EDA SUMMARY — KEY FINDINGS")
print("="*60)
print(f"""
  1. Duration is RIGHT-SKEWED: median={train['duration_seconds'].median():.0f}s, 
     mean={train['duration_seconds'].mean():.0f}s. Long tail of rare long trips.
     → Consider log-transforming target or using quantile loss.

  2. STRONG TIME SIGNAL: trips vary by {hour_means.max()-hour_means.min():.0f}s across hours.
     Worst hour (rush) vs best hour: big gap.
     → hour, dow, rush_hour flags are high-value features.

  3. ZONE-PAIR LOOKUP gives {lookup_mae:.0f}s MAE — already better than GBT baseline.
     → Must be a feature in our model, not just a fallback.

  4. {len(unseen)/len(dev_pairs)*100:.1f}% of dev zone pairs unseen in train → need fallback.

  5. MONTH SIGNAL exists: {month_means.max()-month_means.min():.0f}s range across months.
     Dec (holiday season) skews longer.

  6. PASSENGER COUNT: check output above. Usually weak signal.
""")
