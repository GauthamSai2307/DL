#!/usr/bin/env python3
"""Dependency-light vegetable price forecasting for 6w and 8w horizons.

Uses data.gov.in API resource 9ef84268-d588-465a-a308-a864a43d0070 and trains a
simple global autoregressive linear model (SGD) using historical mandi prices
across districts/markets/commodities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlencode
from urllib.request import urlopen

RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
DEFAULT_API_KEY = "579b464db66ec23bdd0000016ca6bbf5f0fa4f4c4f4f307878fd8875"

LAGS = [1, 2, 3, 7, 14, 21, 28]
MIN_HISTORY = 60
HASH_BUCKETS = 64


@dataclass
class Metrics:
    horizon_days: int
    mape: float
    accuracy_pct: float
    rmse: float
    evaluated_series: int
    evaluated_points: int


def fetch_records(api_key: str, limit: int = 1000, max_pages: int | None = None) -> List[dict]:
    records: List[dict] = []
    offset = 0
    pages = 0
    while True:
        q = urlencode({"api-key": api_key, "format": "json", "limit": limit, "offset": offset})
        with urlopen(f"{BASE_URL}?{q}", timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        batch = payload.get("records", [])
        if not batch:
            break
        records.extend(batch)
        pages += 1
        offset += limit
        if max_pages is not None and pages >= max_pages:
            break
    return records


def find_key(sample: dict, candidates: List[str]) -> str:
    keys = {k.strip().lower(): k for k in sample.keys()}
    for c in candidates:
        if c in keys:
            return keys[c]
    raise KeyError(f"Missing key from {candidates}")


def parse_date(value: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except Exception:
            continue
    return None


def preprocess(records: List[dict]):
    if not records:
        raise RuntimeError("No records fetched.")

    sample = records[0]
    k_date = find_key(sample, ["arrival_date", "date", "reported_date"])
    k_district = find_key(sample, ["district"])
    k_market = find_key(sample, ["market", "mandi"])
    k_commodity = find_key(sample, ["commodity"])
    k_variety = find_key(sample, ["variety"])
    k_price = find_key(sample, ["modal_price", "modal price", "price"])

    grouped: Dict[Tuple[str, datetime], List[float]] = defaultdict(list)
    meta: Dict[str, dict] = {}

    for r in records:
        dt = parse_date(str(r.get(k_date, "")))
        if dt is None:
            continue
        try:
            price = float(str(r.get(k_price, "")).replace(",", ""))
        except Exception:
            continue
        if price <= 0:
            continue
        district = str(r.get(k_district, "")).strip()
        market = str(r.get(k_market, "")).strip()
        commodity = str(r.get(k_commodity, "")).strip()
        variety = str(r.get(k_variety, "")).strip()
        sid = f"{district}|{market}|{commodity}|{variety}"
        grouped[(sid, dt.date())].append(price)
        meta[sid] = {
            "district": district,
            "market": market,
            "commodity": commodity,
            "variety": variety,
        }

    series: Dict[str, Dict[datetime.date, float]] = defaultdict(dict)
    for (sid, d), vals in grouped.items():
        series[sid][d] = sum(vals) / len(vals)

    return series, meta


def date_features(d):
    return [d.weekday() / 6.0, d.month / 12.0, ((d.timetuple().tm_yday - 1) % 366) / 365.0]


def hash_features(meta: dict, buckets: int = HASH_BUCKETS) -> List[float]:
    out = [0.0] * buckets
    for part in (meta["district"], meta["market"], meta["commodity"], meta["variety"]):
        idx = hash(part) % buckets
        out[idx] += 1.0
    return out


def get_series_sorted(series_map):
    return sorted(series_map.items(), key=lambda x: x[0])


def build_daily_series(points: Dict):
    dates = sorted(points.keys())
    if not dates:
        return []
    out = []
    cur = dates[0]
    end = dates[-1]
    last = points[cur]
    while cur <= end:
        if cur in points:
            last = points[cur]
        out.append((cur, last))
        cur += timedelta(days=1)
    return out


def make_samples(series, meta):
    X, y = [], []
    for sid, points in get_series_sorted(series):
        seq = build_daily_series(points)
        if len(seq) < MIN_HISTORY:
            continue
        cat = hash_features(meta[sid])
        prices = [p for _, p in seq]
        dates = [d for d, _ in seq]
        for i in range(max(LAGS), len(seq)):
            feat = []
            for lag in LAGS:
                feat.append(prices[i - lag])
            feat.append(sum(prices[max(0, i - 7) : i]) / min(7, i))
            feat.append(sum(prices[max(0, i - 14) : i]) / min(14, i))
            feat.extend(date_features(dates[i]))
            feat.extend(cat)
            X.append(feat)
            y.append(prices[i])
    return X, y


def fit_linear_sgd(X: List[List[float]], y: List[float], epochs: int = 6, lr: float = 1e-7):
    if not X:
        raise RuntimeError("No training samples generated.")
    dim = len(X[0])
    w = [0.0] * dim
    b = 0.0
    n = len(X)
    for _ in range(epochs):
        for i in range(n):
            xi = X[i]
            yi = y[i]
            pred = b + sum(a * c for a, c in zip(w, xi))
            err = pred - yi
            b -= lr * err
            for j in range(dim):
                w[j] -= lr * err * xi[j]
    return w, b


def predict_one(w, b, feat):
    v = b
    for a, c in zip(w, feat):
        v += a * c
    return max(0.0, v)


def recursive_forecast(series, meta, w, b, horizon, start_cutoff=None):
    rows = []
    for sid, points in get_series_sorted(series):
        seq = build_daily_series(points)
        if len(seq) < MIN_HISTORY:
            continue
        if start_cutoff:
            seq = [(d, p) for d, p in seq if d <= start_cutoff]
        if len(seq) < MIN_HISTORY:
            continue

        cat = hash_features(meta[sid])
        dates = [d for d, _ in seq]
        prices = [p for _, p in seq]
        cur = dates[-1]
        for step in range(1, horizon + 1):
            d = cur + timedelta(days=step)
            feat = [prices[-lag] if len(prices) >= lag else prices[0] for lag in LAGS]
            feat.append(sum(prices[-7:]) / min(7, len(prices)))
            feat.append(sum(prices[-14:]) / min(14, len(prices)))
            feat.extend(date_features(d))
            feat.extend(cat)
            pred = predict_one(w, b, feat)
            prices.append(pred)
            rows.append(
                {
                    "series_id": sid,
                    "district": meta[sid]["district"],
                    "market": meta[sid]["market"],
                    "commodity": meta[sid]["commodity"],
                    "variety": meta[sid]["variety"],
                    "date": d.isoformat(),
                    "predicted_price": round(pred, 2),
                    "step": step,
                }
            )
    return rows


def evaluate(series, meta, horizon):
    all_dates = [d for pts in series.values() for d in pts.keys()]
    max_date = max(all_dates)
    test_start = max_date - timedelta(days=horizon - 1)

    train_series, test_actual = defaultdict(dict), {}
    for sid, pts in series.items():
        for d, p in pts.items():
            if d < test_start:
                train_series[sid][d] = p
            else:
                test_actual[(sid, d.isoformat())] = p

    X, y = make_samples(train_series, meta)
    w, b = fit_linear_sgd(X, y)
    preds = recursive_forecast(train_series, meta, w, b, horizon)

    eval_rows = []
    abs_pct, sq = [], []
    series_seen = set()
    for r in preds:
        key = (r["series_id"], r["date"])
        if key in test_actual and test_actual[key] > 0:
            actual = test_actual[key]
            pred = r["predicted_price"]
            ape = abs(actual - pred) / actual
            abs_pct.append(ape)
            sq.append((actual - pred) ** 2)
            series_seen.add(r["series_id"])
            z = dict(r)
            z["actual_price"] = round(actual, 2)
            z["ape"] = round(ape, 6)
            eval_rows.append(z)

    mape = (sum(abs_pct) / len(abs_pct) * 100.0) if abs_pct else 100.0
    rmse = math.sqrt(sum(sq) / len(sq)) if sq else 0.0
    metrics = Metrics(
        horizon_days=horizon,
        mape=round(mape, 4),
        accuracy_pct=round(max(0.0, 100.0 - mape), 4),
        rmse=round(rmse, 4),
        evaluated_series=len(series_seen),
        evaluated_points=len(eval_rows),
    )
    return metrics, eval_rows, (w, b)


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.getenv("DATA_GOV_API_KEY", DEFAULT_API_KEY)
    records = fetch_records(api_key, limit=args.page_size, max_pages=args.max_pages)
    series, meta = preprocess(records)

    m6, eval6, _ = evaluate(series, meta, 42)
    m8, eval8, _ = evaluate(series, meta, 56)

    X_all, y_all = make_samples(series, meta)
    w_all, b_all = fit_linear_sgd(X_all, y_all)
    forecast6 = recursive_forecast(series, meta, w_all, b_all, 42)
    forecast8 = recursive_forecast(series, meta, w_all, b_all, 56)

    summary = {
        "dataset_rows": len(records),
        "series_count": len(series),
        "metrics": {"6_weeks": m6.__dict__, "8_weeks": m8.__dict__},
        "better_horizon": "6_weeks" if m6.accuracy_pct >= m8.accuracy_pct else "8_weeks",
    }

    write_csv(output_dir / "evaluation_6_weeks.csv", eval6)
    write_csv(output_dir / "evaluation_8_weeks.csv", eval8)
    write_csv(output_dir / "forecast_next_6_weeks.csv", forecast6)
    write_csv(output_dir / "forecast_next_8_weeks.csv", forecast8)
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Forecast vegetable prices for 6w and 8w horizons")
    p.add_argument("--api-key", default=None)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--page-size", type=int, default=1000)
    p.add_argument("--max-pages", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
