"""
Submission interface — fixed signature, our internals.
Loads model.pkl once at import; predict() must stay under 200ms on CPU.
"""
from __future__ import annotations

import math
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _ART = pickle.load(_f)

_MODEL         = _ART["model"]
_CENTROIDS     = _ART["centroids"]
_PAIR_LOOKUP   = _ART["pair_lookup"]
_PU_MED        = _ART["zone_pu_med"]
_DO_MED        = _ART["zone_do_med"]
_ZONE_HR_MED   = _ART["zone_hour_med"]
_GLOBAL_MED    = _ART["global_med"]

_NYC_LAT, _NYC_LON = 40.7128, -74.0060

try:
    import holidays as _hols
    _US_HOLIDAYS = _hols.US(years=[2022, 2023, 2024, 2025])
except Exception:
    _US_HOLIDAYS = set()


def _centroid(zone: int) -> tuple[float, float]:
    return _CENTROIDS.get(zone, (_NYC_LAT, _NYC_LON))


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def predict(request: dict) -> float:
    """
    Input:
        pickup_zone     int   NYC taxi zone 1-265
        dropoff_zone    int
        requested_at    str   ISO 8601 datetime
        passenger_count int
    Output:
        predicted trip duration in seconds (float)
    """
    ts  = datetime.fromisoformat(request["requested_at"])
    pu  = int(request["pickup_zone"])
    do  = int(request["dropoff_zone"])
    pax = int(request["passenger_count"])

    hour   = ts.hour
    minute = ts.minute
    dow    = ts.weekday()
    month  = ts.month

    pu_lat, pu_lon = _centroid(pu)
    do_lat, do_lon = _centroid(do)
    hav = _haversine(pu_lat, pu_lon, do_lat, do_lon)

    pair_med  = _PAIR_LOOKUP.get((pu, do), _GLOBAL_MED)
    pu_med    = _PU_MED.get(pu, _GLOBAL_MED)
    do_med    = _DO_MED.get(do, _GLOBAL_MED)
    zh_med    = _ZONE_HR_MED.get((pu, hour), pu_med)

    is_rush    = int(((7 <= hour <= 9) or (17 <= hour <= 19)) and dow < 5)
    is_weekend = int(dow >= 5)
    is_night   = int(hour >= 22 or hour <= 5)
    is_holiday = int(ts.date() in _US_HOLIDAYS)
    is_same    = int(pu == do)

    x = np.array([[
        pu, do,
        hour, minute, dow, month,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * dow / 7),
        math.cos(2 * math.pi * dow / 7),
        hour * 60 + minute,
        is_rush, is_weekend, is_night, is_holiday, is_same,
        pu_lat, pu_lon, do_lat, do_lon,
        hav,
        do_lat - pu_lat, do_lon - pu_lon,
        pair_med, pu_med, do_med, zh_med,
        pax,
    ]], dtype=np.float32)

    return float(_MODEL.predict(x)[0])

