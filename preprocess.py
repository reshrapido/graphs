"""
preprocess.py
-------------
Run this locally on the 220 MB raw CSV.
Produces a tiny aggregated JSON that the dashboard can load.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# CONFIG – keep in sync with the dashboard
# ------------------------------------------------------------------
RAW_CSV = "data.csv"                 # path to your 220 MB file
OUTPUT_JSON = "dashboard_data.json"  # this is the only file you upload

STATUS_COL = "modified_order_status"
NET_STATUSES = {"dropped"}
COBRM = "COBRM"
COBRA = "COBRA"
OCARA = "OCARA"
CHUNK_ROWS = 200_000

RAW_COLS = [
    "order_id", "yyyymmdd", "hour", "pickup_cluster", "drop_cluster",
    "modified_order_status", "order_status", "propagation_request_epoch", "pool_orders",
]
SUM_COLS = [
    "gross_orders", "propagated_orders", "net_rides", "pooled_rides",
    "solo_rides", "cobrm_orders", "cobra_orders", "ocara_orders", "vehicle_trips",
]


def _sniff_sep(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def _reduce_chunk(ch):
    n = len(ch)

    def col(name, default=np.nan):
        return ch[name] if name in ch.columns else pd.Series([default] * n, index=ch.index)

    out = pd.DataFrame(index=ch.index)
    out["order_id"] = ch["order_id"]
    out["yyyymmdd"] = ch["yyyymmdd"]
    out["hour"] = pd.to_numeric(col("hour"), errors="coerce")
    out["pickup_cluster"] = col("pickup_cluster").fillna("(unknown)")
    out["drop_cluster"] = col("drop_cluster").fillna("(unknown)")
    status_src = col(STATUS_COL) if STATUS_COL in ch.columns else col("order_status", "")
    out["status"] = status_src.fillna("").astype(str).str.strip().str.upper()
    out["propagated"] = (
        pd.to_numeric(col("propagation_request_epoch"), errors="coerce").fillna(0) > 0
    ).astype("int8")
    if "pool_orders" in ch.columns:
        s = ch["pool_orders"].fillna("").astype(str)
        total = s.str.count("orderId")
        invalid = s.str.count(r"isValid\W*false")
        out["pool_size"] = (total - invalid).clip(lower=0).astype("int16")
    else:
        out["pool_size"] = 0
    return out


def _aggregate_orders(o):
    o = o.copy()
    o["is_net"] = o["status"].isin({x.upper() for x in NET_STATUSES})
    o["is_pooled"] = o["is_net"] & (o["pool_size"] >= 2)
    o["is_solo"] = o["is_net"] & (o["pool_size"] < 2)
    o["is_cobrm"] = o["status"] == COBRM
    o["is_cobra"] = o["status"] == COBRA
    o["is_ocara"] = o["status"] == OCARA
    o["trip_w"] = np.where(o["is_net"], 1.0 / o["pool_size"].clip(lower=1), 0.0)

    return (
        o.groupby(["date", "hour", "pickup_cluster", "drop_cluster"], as_index=False)
        .agg(
            gross_orders=("order_id", "size"),
            propagated_orders=("propagated", "sum"),
            net_rides=("is_net", "sum"),
            pooled_rides=("is_pooled", "sum"),
            solo_rides=("is_solo", "sum"),
            cobrm_orders=("is_cobrm", "sum"),
            cobra_orders=("is_cobra", "sum"),
            ocara_orders=("is_ocara", "sum"),
            vehicle_trips=("trip_w", "sum"),
        )
    )


def main():
    if not os.path.isfile(RAW_CSV):
        raise SystemExit(f"Raw CSV not found: {RAW_CSV}")

    sep = _sniff_sep(RAW_CSV)
    header = list(pd.read_csv(RAW_CSV, sep=sep, nrows=0).columns)

    print(f"Reading {RAW_CSV} …")
    usecols = [c for c in RAW_COLS if c in header]
    parts = []
    for i, ch in enumerate(pd.read_csv(
        RAW_CSV, sep=sep, usecols=usecols, dtype=str,
        chunksize=CHUNK_ROWS, on_bad_lines="skip", engine="c",
    )):
        parts.append(_reduce_chunk(ch))
        if (i + 1) % 5 == 0:
            print(f"  processed {(i + 1) * CHUNK_ROWS:,} rows …")

    orders = pd.concat(parts, ignore_index=True)
    print(f"Raw rows: {len(orders):,}")

    orders = orders.dropna(subset=["order_id"])
    before = len(orders)
    orders = orders.drop_duplicates("order_id", keep="first")
    print(f"Duplicate order IDs dropped: {before - len(orders):,}")

    orders["date"] = pd.to_datetime(orders["yyyymmdd"], format="%Y%m%d", errors="coerce")
    bad_dates = int(orders["date"].isna().sum())
    orders = orders.dropna(subset=["date"])
    orders["hour"] = orders["hour"].fillna(-1).astype(int)

    status_counts = orders["status"].value_counts().to_dict()

    print("Aggregating …")
    agg = _aggregate_orders(orders)

    # Convert date to string for clean JSON
    agg["date"] = agg["date"].dt.strftime("%Y-%m-%d")

    payload = {
        "meta": {
            "mode": "pre-aggregated",
            "raw_rows": int(before),
            "dup_removed": int(before - len(orders)),
            "bad_dates": bad_dates,
            "status_counts": status_counts,
            "generated_at": pd.Timestamp.now().isoformat(),
        },
        "data": agg.to_dict(orient="records"),
    }

    Path(OUTPUT_JSON).write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = Path(OUTPUT_JSON).stat().st_size / 1024
    print(f"Wrote {OUTPUT_JSON}  ({size_kb:.1f} KB)")
    print("You can now commit only this JSON file.")


if __name__ == "__main__":
    main()