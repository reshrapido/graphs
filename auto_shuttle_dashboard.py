"""
Auto Shuttle pooling dashboard
==============================

Setup:
    pip install streamlit pandas numpy plotly

Run:
    streamlit run auto_shuttle_dashboard.py

Then paste the path of your CSV in the sidebar (or set DEFAULT_CSV below).

Accepted inputs (auto-detected):
  1. Raw order-level export from orders.order_logs_fact (comma or tab separated).
     Only the columns listed in RAW_COLS are read, so the huge JSON columns
     (map_riders etc.) and all personal data are skipped. Read in chunks, so a
     220 MB file is fine.
  2. The aggregated CSV from the GROUP BY query (columns: yyyymmdd,
     pickup_cluster, drop_cluster, gross_orders, propagated_orders, net_rides,
     pooled_rides, solo_rides).

Definitions (edit the constants below if they are wrong):
  gross orders      every order row
  propagated        propagation_request_epoch is filled
  net rides         modified_order_status in NET_STATUSES ('dropped')
  cancel buckets    COBRM = cancelled before rider mapped, COBRA = cancelled before
                    rider arrived, OCARA = cancelled after rider arrived
  pooled / solo     net rides whose pool_orders lists >= 2 / < 2 valid orders
  vehicle trips     net rides weighted by 1 / pool size (a pooled pair = 1 trip)
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG - check these against your data
# ----------------------------------------------------------------------------
DEFAULT_CSV = "data.csv"
STATUS_COL = "modified_order_status"      # falls back to order_status if missing
NET_STATUSES = {"dropped"}                 # status values that count as a completed ride
COBRM = "COBRM"                           # cancelled before rider mapped
COBRA = "COBRA"                           # cancelled before rider arrived
OCARA = "OCARA"                           # cancelled after rider arrived
JIT_START = pd.Timestamp("2026-09-17")     # first day of JIT
ROLLBACK_START = pd.Timestamp("2026-09-28")  # first day of the 3-minute pooling window
PHASE_NAMES = ("8-min pooling", "JIT", "3-min pooling")
PHASE_COLORS = {"8-min pooling": "#4C78A8", "JIT": "#E45756", "3-min pooling": "#54A24B"}
CHUNK_ROWS = 200_000

RAW_COLS = [
    "order_id", "yyyymmdd", "hour", "pickup_cluster", "drop_cluster",
    "modified_order_status", "order_status", "propagation_request_epoch", "pool_orders",
]
SUM_COLS = [
    "gross_orders", "propagated_orders", "net_rides", "pooled_rides",
    "solo_rides", "cobrm_orders", "cobra_orders", "ocara_orders", "vehicle_trips",
]
COUNT_LABELS = {
    "gross_orders": "Gross orders",
    "propagated_orders": "Propagated orders",
    "net_rides": "Net rides",
    "pooled_rides": "Pooled rides",
    "solo_rides": "Solo rides",
}
RATE_LABELS = {
    "propagation_rate": "Propagation rate (propagated / gross)",
    "completion_rate": "Completion rate (net / gross)",
    "ride_per_propagated": "Net rides / propagated",
    "cobrm_rate": "Cancelled before rider mapped (COBRM) / gross",
    "cobra_rate": "Cancelled before rider arrived (COBRA) / gross",
    "ocara_rate": "Cancelled after rider arrived (OCARA) / gross",
    "pool_share": "Pool share (pooled / net)",
    "pax_per_trip": "Passengers per vehicle trip",
}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ----------------------------------------------------------------------------
# LOADING
# ----------------------------------------------------------------------------
def _sniff_sep(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        header = f.readline()
    return "\t" if header.count("\t") > header.count(",") else ","


def _reduce_chunk(ch):
    """Collapse a raw chunk to one small row per order."""
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


@st.cache_data(show_spinner="Reading CSV (a big file takes a minute the first time)...")
def load_data(path, mtime):
    sep = _sniff_sep(path)
    header = list(pd.read_csv(path, sep=sep, nrows=0).columns)
    meta = {"warnings": [], "status_counts": {}}

    # ---- already aggregated -------------------------------------------------
    if "gross_orders" in header:
        meta["mode"] = "aggregated CSV"
        df = pd.read_csv(path, sep=sep, dtype={"yyyymmdd": str})
        df["date"] = pd.to_datetime(df["yyyymmdd"], format="%Y%m%d", errors="coerce")
        meta["bad_dates"] = int(df["date"].isna().sum())
        df = df.dropna(subset=["date"])
        if "hour" not in df.columns:
            df["hour"] = -1
        for c in ("pickup_cluster", "drop_cluster"):
            if c not in df.columns:
                df[c] = "(all)"
        for c in SUM_COLS:
            if c not in df.columns:
                df[c] = np.nan
        meta["raw_rows"] = len(df)
        meta["dup_removed"] = 0
        return df[["date", "hour", "pickup_cluster", "drop_cluster"] + SUM_COLS], meta

    # ---- raw order-level ----------------------------------------------------
    meta["mode"] = "raw order-level CSV"
    required = ["order_id", "yyyymmdd"]
    missing = [c for c in required if c not in header]
    if STATUS_COL not in header and "order_status" not in header:
        missing.append(STATUS_COL)
    elif STATUS_COL not in header:
        meta["warnings"].append(f"Column '{STATUS_COL}' not found - using order_status, so "
                                "the COBRM / COBRA / OCARA split will be empty.")
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    for c in ("pickup_cluster", "propagation_request_epoch", "pool_orders", "hour"):
        if c not in header:
            meta["warnings"].append(f"Column '{c}' not found - related metrics will be empty.")
    usecols = [c for c in RAW_COLS if c in header]

    parts = []
    for ch in pd.read_csv(
        path, sep=sep, usecols=usecols, dtype=str, chunksize=CHUNK_ROWS,
        on_bad_lines="skip", engine="c",
    ):
        parts.append(_reduce_chunk(ch))
    orders = pd.concat(parts, ignore_index=True)

    meta["raw_rows"] = len(orders)
    orders = orders.dropna(subset=["order_id"])
    before = len(orders)
    orders = orders.drop_duplicates("order_id", keep="first")
    meta["dup_removed"] = before - len(orders)

    orders["date"] = pd.to_datetime(orders["yyyymmdd"], format="%Y%m%d", errors="coerce")
    meta["bad_dates"] = int(orders["date"].isna().sum())
    orders = orders.dropna(subset=["date"])
    orders["hour"] = orders["hour"].fillna(-1).astype(int)
    meta["status_counts"] = orders["status"].value_counts().to_dict()

    return _aggregate_orders(orders), meta


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def _div(a, b):
    return a / b.replace(0, np.nan)


def add_ratios(d):
    d = d.copy()
    d["propagation_rate"] = _div(d["propagated_orders"], d["gross_orders"])
    d["completion_rate"] = _div(d["net_rides"], d["gross_orders"])
    d["ride_per_propagated"] = _div(d["net_rides"], d["propagated_orders"])
    d["cobrm_rate"] = _div(d["cobrm_orders"], d["gross_orders"])
    d["cobra_rate"] = _div(d["cobra_orders"], d["gross_orders"])
    d["ocara_rate"] = _div(d["ocara_orders"], d["gross_orders"])
    d["pool_share"] = _div(d["pooled_rides"], d["net_rides"])
    d["pax_per_trip"] = _div(d["net_rides"], d["vehicle_trips"])
    return d


def timeseries(df, hourly=False):
    keys = ["date", "hour"] if hourly else ["date"]
    g = df.groupby(keys, as_index=False)[SUM_COLS].sum(min_count=1)
    g["ts"] = g["date"] + pd.to_timedelta(g["hour"], unit="h") if hourly else g["date"]
    return add_ratios(g).sort_values("ts")


def phase_of(dates, jit, rb):
    return np.select(
        [dates < jit, dates < rb], [PHASE_NAMES[0], PHASE_NAMES[1]], default=PHASE_NAMES[2]
    )


def shade(fig, xmin, xmax, jit, rb, annotate=True, **kw):
    """Colour the three policy phases behind a time-series figure."""
    end = xmax + pd.Timedelta(days=1)
    spans = [
        (PHASE_NAMES[0], xmin, jit),
        (PHASE_NAMES[1], jit, rb),
        (PHASE_NAMES[2], rb, end),
    ]
    for name, a, b in spans:
        a, b = max(a, xmin), min(b, end)
        if a >= b:
            continue
        x0, x1 = a.to_pydatetime(), b.to_pydatetime()
        fig.add_vrect(
            x0=x0, x1=x1, fillcolor=PHASE_COLORS[name], opacity=0.12,
            layer="below", line_width=0, **kw,
        )
        if annotate:
            fig.add_annotation(
                x=x0 + (x1 - x0) / 2, y=1.02, yref="paper", text=name,
                showarrow=False, font=dict(size=11, color=PHASE_COLORS[name]),
            )
    return fig


def pct(x):
    return None if pd.isna(x) else round(float(x) * 100, 1)


def has(df, col):
    return col in df.columns and df[col].notna().any()


def summarize(df, lo, hi):
    """Per-day averages and rates for a date range."""
    d = df[(df["date"] >= lo) & (df["date"] <= hi)]
    days = d["date"].nunique()
    if days == 0:
        return None
    s = d[SUM_COLS].sum(min_count=1)
    r = add_ratios(pd.DataFrame([s])).iloc[0]
    out = {
        "Days with data": days,
        "Gross orders / day": s["gross_orders"] / days,
        "Propagated / day": s["propagated_orders"] / days,
        "Net rides / day": s["net_rides"] / days,
        "Pooled rides / day": s["pooled_rides"] / days,
        "Solo rides / day": s["solo_rides"] / days,
        "Propagation rate %": pct(r["propagation_rate"]),
        "Completion rate %": pct(r["completion_rate"]),
        "Cancelled before rider mapped (COBRM) %": pct(r["cobrm_rate"]),
        "Cancelled before rider arrived (COBRA) %": pct(r["cobra_rate"]),
        "Cancelled after rider arrived (OCARA) %": pct(r["ocara_rate"]),
        "Pool share %": pct(r["pool_share"]),
        "Passengers per trip": r["pax_per_trip"],
    }
    return out


# ----------------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Auto Shuttle pooling", layout="wide")
st.title("Auto Shuttle pooling dashboard")

with st.sidebar:
    st.header("Data")
    path = st.text_input("CSV path", value=DEFAULT_CSV)

if not os.path.isfile(path):
    st.info("Enter the path of your CSV in the sidebar (or set DEFAULT_CSV in the script).")
    st.stop()

try:
    data, meta = load_data(path, os.path.getmtime(path))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the file: {e}")
    st.stop()

if data.empty:
    st.error("No usable rows after parsing. Check the yyyymmdd column.")
    st.stop()

for w in meta["warnings"]:
    st.warning(w)

dmin, dmax = data["date"].min(), data["date"].max()
has_hour = (data["hour"] >= 0).any()
has_pool = has(data, "pooled_rides") and data["pooled_rides"].sum() > 0

with st.sidebar:
    st.header("Filters")
    dr = st.date_input("Date range", (dmin.date(), dmax.date()),
                       min_value=dmin.date(), max_value=dmax.date())
    if not isinstance(dr, (tuple, list)) or len(dr) != 2:
        st.stop()
    d0, d1 = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])

    all_pick = sorted(data["pickup_cluster"].unique())
    sel_pick = st.multiselect("Pickup area", all_pick, default=all_pick)
    all_drop = sorted(data["drop_cluster"].unique())
    with st.expander("Drop area filter"):
        sel_drop = st.multiselect("Drop area", all_drop, default=all_drop)

    if has_hour:
        h0, h1 = st.slider("Hour of day", 0, 23, (0, 23))
        gran = st.radio("Granularity", ["Daily", "Hourly"], horizontal=True)
    else:
        h0, h1, gran = 0, 23, "Daily"
    hourly = gran == "Hourly"

    st.header("Policy phases")
    jit = pd.Timestamp(st.date_input("JIT starts", JIT_START.date()))
    rb = pd.Timestamp(st.date_input("3-min pooling starts", ROLLBACK_START.date()))

base = data[data["pickup_cluster"].isin(sel_pick) & data["drop_cluster"].isin(sel_drop)]
if has_hour:
    base = base[(base["hour"] >= h0) & (base["hour"] <= h1)]
view = base[(base["date"] >= d0) & (base["date"] <= d1)]

if view.empty:
    st.warning("No data for this combination of filters.")
    st.stop()

with st.expander("Data check (open this first)"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Input", meta["mode"])
    c2.metric("Rows read", f"{meta['raw_rows']:,}")
    c3.metric("Duplicate order IDs dropped", f"{meta['dup_removed']:,}")
    c4.metric("Rows with bad dates", f"{meta['bad_dates']:,}")
    st.write(f"Dates: {dmin.date()} to {dmax.date()}  |  Pickup areas: {len(all_pick)}  |  "
             f"Net-ride statuses: {sorted(NET_STATUSES)}")
    if meta["status_counts"]:
        st.write("Status values found (expected: dropped, COBRM, COBRA, OCARA; "
                 "anything else is counted in gross orders only):")
        st.dataframe(pd.Series(meta["status_counts"], name="orders").to_frame())
    if meta["dup_removed"] > 0:
        st.warning("Duplicate order IDs were found, so the file may hold several rows per order "
                   "(event log). Only the first row per order was kept - check the SQL filter.")
    if not has_pool:
        st.warning("No pooled rides detected. Check that the pool_orders column is present.")

ts = timeseries(view, hourly)
xmin, xmax = ts["ts"].min().normalize(), ts["ts"].max().normalize()

k1, k2, k3, k4, k5 = st.columns(5)
tot = view[SUM_COLS].sum(min_count=1)
tr = add_ratios(pd.DataFrame([tot])).iloc[0]
k1.metric("Gross orders", f"{tot['gross_orders']:,.0f}")
k2.metric("Propagated", f"{tot['propagated_orders']:,.0f}")
k3.metric("Net rides", f"{tot['net_rides']:,.0f}")
k4.metric("Pool share", "n/a" if pd.isna(tr["pool_share"]) else f"{tr['pool_share']:.1%}")
k5.metric("Completion rate", "n/a" if pd.isna(tr["completion_rate"]) else f"{tr['completion_rate']:.1%}")

tab_trend, tab_pool, tab_phase, tab_week, tab_range, tab_area = st.tabs(
    ["Trends", "Pool vs solo", "Phase comparison", "Weekday overlay", "Range A vs B", "By area"]
)

# ---------------------------------------------------------------- Trends
with tab_trend:
    smooth = False
    if not hourly:
        smooth = st.checkbox("7-day rolling mean", value=False)
    plot = ts.copy()
    if smooth:
        num = [c for c in plot.columns if c not in ("ts", "date", "hour")]
        plot[num] = plot[num].rolling(7, min_periods=1).mean()

    count_cols = [c for c in ["gross_orders", "propagated_orders", "net_rides"] if has(plot, c)]
    long = plot.melt(id_vars="ts", value_vars=count_cols, var_name="metric", value_name="value")
    long["metric"] = long["metric"].map(COUNT_LABELS)
    fig = px.line(long, x="ts", y="value", color="metric", markers=not hourly)
    fig.update_layout(height=420, legend_title_text="", yaxis_title="orders", xaxis_title="")
    st.plotly_chart(shade(fig, xmin, xmax, jit, rb))

    rate_opts = [c for c in RATE_LABELS if c not in ("pool_share", "pax_per_trip") and has(plot, c)]
    chosen = st.multiselect("Rates", rate_opts, default=rate_opts[:2],
                            format_func=lambda c: RATE_LABELS[c])
    if chosen:
        rl = plot.melt(id_vars="ts", value_vars=chosen, var_name="metric", value_name="value")
        rl["metric"] = rl["metric"].map(RATE_LABELS)
        fig = px.line(rl, x="ts", y="value", color="metric", markers=not hourly)
        fig.update_layout(height=380, legend_title_text="", yaxis_tickformat=".0%",
                          yaxis_title="", xaxis_title="")
        st.plotly_chart(shade(fig, xmin, xmax, jit, rb))

# ------------------------------------------------------------- Pool vs solo
# ------------------------------------------------------------- Pool vs solo
with tab_pool:
    if not has_pool:
        st.info("No pool data available.")
    else:
        mode = st.radio("Show", ["Share (%)", "Counts"], horizontal=True)
        long = ts.melt(id_vars="ts", value_vars=["pooled_rides", "solo_rides"],
                       var_name="type", value_name="rides")
        long["type"] = long["type"].map({"pooled_rides": "Pooled", "solo_rides": "Solo"})

        if mode.startswith("Share"):
            # Normalize to percentage of net rides per time bucket
            totals = long.groupby("ts")["rides"].transform("sum")
            long = long.copy()
            long["rides"] = (long["rides"] / totals.replace(0, np.nan) * 100).fillna(0)

        fig = px.bar(
            long, x="ts", y="rides", color="type",
            color_discrete_map={"Pooled": "#54A24B", "Solo": "#E45756"},
            barmode="stack",
        )
        fig.update_layout(
            height=430,
            legend_title_text="",
            xaxis_title="",
            yaxis_title="% of net rides" if mode.startswith("Share") else "rides",
        )
        if mode.startswith("Share"):
            fig.update_layout(yaxis_ticksuffix="%")
        st.plotly_chart(shade(fig, xmin, xmax, jit, rb))

        if has(ts, "pax_per_trip"):
            fig = px.line(ts, x="ts", y="pax_per_trip", markers=not hourly)
            fig.update_layout(height=300, xaxis_title="", yaxis_title="passengers per vehicle trip")
            st.plotly_chart(shade(fig, xmin, xmax, jit, rb))

# ------------------------------------------------------- Phase comparison
with tab_phase:
    daily = timeseries(view, hourly=False)
    daily["phase"] = phase_of(daily["date"], jit, rb)
    g = daily.groupby("phase")[SUM_COLS].sum(min_count=1)
    days = daily.groupby("phase")["date"].nunique()
    order = [p for p in PHASE_NAMES if p in g.index]
    g, days = g.loc[order], days.loc[order]
    rat = add_ratios(g)
    table = pd.DataFrame({
        "Days": days,
        "Gross / day": (g["gross_orders"] / days).round(1),
        "Propagated / day": (g["propagated_orders"] / days).round(1),
        "Net rides / day": (g["net_rides"] / days).round(1),
        "Propagation rate %": (rat["propagation_rate"] * 100).round(1),
        "Completion rate %": (rat["completion_rate"] * 100).round(1),
        "COBRM %": (rat["cobrm_rate"] * 100).round(1),
        "COBRA %": (rat["cobra_rate"] * 100).round(1),
        "OCARA %": (rat["ocara_rate"] * 100).round(1),
        "Pool share %": (rat["pool_share"] * 100).round(1),
        "Solo share %": (100 - rat["pool_share"] * 100).round(1),
        "Passengers / trip": rat["pax_per_trip"].round(2),
    })
    st.dataframe(table.dropna(axis=1, how="all"))

    per_day = pd.DataFrame({
        "Gross": g["gross_orders"] / days,
        "Propagated": g["propagated_orders"] / days,
        "Net rides": g["net_rides"] / days,
    }).reset_index().melt(id_vars="phase", var_name="metric", value_name="per day")
    fig = px.bar(per_day, x="metric", y="per day", color="phase", barmode="group",
                 color_discrete_map=PHASE_COLORS, category_orders={"phase": list(PHASE_NAMES)})
    fig.update_layout(height=380, xaxis_title="", legend_title_text="")
    st.plotly_chart(fig)
    st.caption("Averages are totals divided by days that have data in the selected range. "
               "Rates are computed from totals, not averaged day by day.")

# --------------------------------------------------------- Weekday overlay
with tab_week:
    wd = timeseries(view, hourly=False)
    opts = {**COUNT_LABELS, **RATE_LABELS}
    avail = [c for c in opts if has(wd, c)]
    metric = st.selectbox("Metric", avail, index=avail.index("net_rides") if "net_rides" in avail else 0,
                          format_func=lambda c: opts[c])
    wd["weekday"] = wd["date"].dt.dayofweek.map(dict(enumerate(WEEKDAYS)))
    wk_start = wd["date"] - pd.to_timedelta(wd["date"].dt.dayofweek, unit="D")
    wd["week"] = wk_start.dt.strftime("%Y-%m-%d")
    wd["week"] = "Week of " + wd["week"]
    fig = px.line(wd, x="weekday", y=metric, color="week", markers=True,
                  category_orders={"weekday": WEEKDAYS})
    fig.update_layout(height=450, xaxis_title="", yaxis_title=opts[metric], legend_title_text="")
    if metric in RATE_LABELS and metric != "pax_per_trip":
        fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig)
    st.caption("Each line is one calendar week (Monday start), so a Tuesday under one policy "
               "sits directly over a Tuesday under another.")

# ------------------------------------------------------------ Range A vs B
with tab_range:
    st.write("Pick two date ranges. Pickup/drop area and hour filters from the sidebar still apply.")
    a_hi = min(max(jit - pd.Timedelta(days=1), dmin), dmax)
    a_lo = max(a_hi - pd.Timedelta(days=13), dmin)
    b_lo = min(max(rb, dmin), dmax)
    b_hi = min(b_lo + pd.Timedelta(days=13), dmax)
    ca, cb = st.columns(2)
    ra = ca.date_input("Range A", (a_lo.date(), a_hi.date()),
                       min_value=dmin.date(), max_value=dmax.date(), key="ra")
    rb_ = cb.date_input("Range B", (b_lo.date(), b_hi.date()),
                        min_value=dmin.date(), max_value=dmax.date(), key="rb")
    if len(ra) == 2 and len(rb_) == 2:
        sa = summarize(base, pd.Timestamp(ra[0]), pd.Timestamp(ra[1]))
        sb = summarize(base, pd.Timestamp(rb_[0]), pd.Timestamp(rb_[1]))
        if sa is None or sb is None:
            st.warning("One of the ranges has no data.")
        else:
            rows = []
            for k in sa:
                a, b = sa[k], sb[k]
                if a is None or b is None or pd.isna(a) or pd.isna(b):
                    continue
                if k.endswith("%"):
                    change = f"{b - a:+.1f} pts"
                elif a != 0:
                    change = f"{(b - a) / a:+.1%}"
                else:
                    change = "n/a"
                rows.append({"Metric": k, "Range A": round(a, 2), "Range B": round(b, 2),
                             "Change (B vs A)": change})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

            keys = ["Gross orders / day", "Propagated / day", "Net rides / day"]
            comp = pd.DataFrame({"metric": keys,
                                 "Range A": [sa[k] for k in keys],
                                 "Range B": [sb[k] for k in keys]})
            comp = comp.melt(id_vars="metric", var_name="range", value_name="per day")
            fig = px.bar(comp, x="metric", y="per day", color="range", barmode="group")
            fig.update_layout(height=360, xaxis_title="", legend_title_text="")
            st.plotly_chart(fig)

# ----------------------------------------------------------------- By area
with tab_area:
    n_top = st.slider("Number of areas (ranked by net rides)", 3, 30, 9)
    opts = {**COUNT_LABELS, "pool_share": RATE_LABELS["pool_share"],
            "completion_rate": RATE_LABELS["completion_rate"],
            "propagation_rate": RATE_LABELS["propagation_rate"]}
    avail = [c for c in opts if has(view, c) or c in RATE_LABELS]
    metric = st.selectbox("Metric", avail, index=avail.index("net_rides") if "net_rides" in avail else 0,
                          format_func=lambda c: opts[c], key="area_metric")
    top = view.groupby("pickup_cluster")["net_rides"].sum().nlargest(n_top).index
    ad = view[view["pickup_cluster"].isin(top)]
    ad = add_ratios(ad.groupby(["date", "pickup_cluster"], as_index=False)[SUM_COLS].sum(min_count=1))
    fig = px.line(ad, x="date", y=metric, facet_col="pickup_cluster", facet_col_wrap=3,
                  markers=True, height=max(320, 230 * int(np.ceil(len(top) / 3))))
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    if metric in RATE_LABELS:
        fig.update_layout(yaxis_tickformat=".0%")
    shade(fig, view["date"].min(), view["date"].max(), jit, rb, annotate=False, row="all", col="all")
    st.plotly_chart(fig)

    if has_pool:
        st.subheader("Pool share by area and phase")
        ph = view.copy()
        ph["phase"] = phase_of(ph["date"], jit, rb)
        ph = ph.groupby(["pickup_cluster", "phase"], as_index=False)[SUM_COLS].sum(min_count=1)
        ph = add_ratios(ph)
        ph = ph[ph["pickup_cluster"].isin(top)]
        piv = ph.pivot(index="pickup_cluster", columns="phase", values="pool_share")
        piv = piv[[p for p in PHASE_NAMES if p in piv.columns]]
        fig = px.imshow(piv, text_auto=".0%", aspect="auto", color_continuous_scale="RdYlGn",
                        zmin=0, zmax=1)
        fig.update_layout(height=max(300, 32 * len(piv)), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig)