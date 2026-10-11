"""
Auto Shuttle pooling dashboard
==============================

Setup (with uv):
    uv add streamlit pandas numpy plotly

Run:
    uv run streamlit run auto_shuttle_dashboard.py

First generate the data file locally:
    uv run preprocess.py
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DEFAULT_JSON = "dashboard_data.json"
STATUS_COL = "modified_order_status"
NET_STATUSES = {"dropped"}
COBRM = "COBRM"
COBRA = "COBRA"
OCARA = "OCARA"
JIT_START = pd.Timestamp("2026-09-17")
ROLLBACK_START = pd.Timestamp("2026-09-28")
PHASE_NAMES = ("8-min pooling", "JIT", "3-min pooling")
PHASE_COLORS = {
    "8-min pooling": "#4C78A8",
    "JIT": "#E45756",
    "3-min pooling": "#54A24B",
}

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
# LOADING (JSON only)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading pre-aggregated data…")
def load_json(path: str, mtime: float):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    meta = payload["meta"]
    df = pd.DataFrame(payload["data"])
    df["date"] = pd.to_datetime(df["date"])

    for c in SUM_COLS:
        if c not in df.columns:
            df[c] = 0
    if "hour" not in df.columns:
        df["hour"] = -1

    return df, meta


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
        [dates < jit, dates < rb],
        [PHASE_NAMES[0], PHASE_NAMES[1]],
        default=PHASE_NAMES[2],
    )


def shade(fig, xmin, xmax, jit, rb, annotate=True, **kw):
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
    d = df[(df["date"] >= lo) & (df["date"] <= hi)]
    days = d["date"].nunique()
    if days == 0:
        return None
    s = d[SUM_COLS].sum(min_count=1)
    r = add_ratios(pd.DataFrame([s])).iloc[0]
    return {
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


# ----------------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Auto Shuttle pooling", layout="wide")
st.title("Auto Shuttle pooling dashboard")

# Load data (hard-coded path)
path = DEFAULT_JSON
if not os.path.isfile(path):
    st.error(
        f"`{path}` not found. Run `uv run preprocess.py` first and make sure "
        "the file is next to this script."
    )
    st.stop()

try:
    data, meta = load_json(path, os.path.getmtime(path))
except Exception as e:
    st.error(f"Could not load the JSON: {e}")
    st.stop()

if data.empty:
    st.error("No usable rows in the JSON.")
    st.stop()

dmin, dmax = data["date"].min(), data["date"].max()
has_hour = (data["hour"] >= 0).any()
has_pool = has(data, "pooled_rides") and data["pooled_rides"].sum() > 0

with st.sidebar:
    st.header("Filters")
    dr = st.date_input(
        "Date range",
        (dmin.date(), dmax.date()),
        min_value=dmin.date(),
        max_value=dmax.date(),
    )
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

base = data[
    data["pickup_cluster"].isin(sel_pick) & data["drop_cluster"].isin(sel_drop)
]
if has_hour:
    base = base[(base["hour"] >= h0) & (base["hour"] <= h1)]
view = base[(base["date"] >= d0) & (base["date"] <= d1)]

if view.empty:
    st.warning("No data for this combination of filters.")
    st.stop()

# ---- KPI row ----
ts = timeseries(view, hourly)
xmin, xmax = ts["ts"].min().normalize(), ts["ts"].max().normalize()

k1, k2, k3, k4, k5 = st.columns(5)
tot = view[SUM_COLS].sum(min_count=1)
tr = add_ratios(pd.DataFrame([tot])).iloc[0]
k1.metric("Gross orders", f"{tot['gross_orders']:,.0f}")
k2.metric("Propagated", f"{tot['propagated_orders']:,.0f}")
k3.metric("Net rides", f"{tot['net_rides']:,.0f}")
k4.metric(
    "Pool share",
    "n/a" if pd.isna(tr["pool_share"]) else f"{tr['pool_share']:.1%}",
)
k5.metric(
    "Completion rate",
    "n/a" if pd.isna(tr["completion_rate"]) else f"{tr['completion_rate']:.1%}",
)

# ---- Tabs ----
tab_trend, tab_pool, tab_phase, tab_range, tab_area = st.tabs(
    ["Trends", "Pool vs solo", "Phase comparison", "Range A vs B", "By area"]
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
    st.plotly_chart(shade(fig, xmin, xmax, jit, rb), use_container_width=True)

# ------------------------------------------------------------- Pool vs solo
with tab_pool:
    if not has_pool:
        st.info("No pool data available.")
    else:
        # Pie chart of Pooled vs Solo
        pooled = view["pooled_rides"].sum()
        solo = view["solo_rides"].sum()
        pie_df = pd.DataFrame({
            "type": ["Pooled", "Solo"],
            "rides": [pooled, solo],
        })
        fig = px.pie(
            pie_df,
            names="type",
            values="rides",
            color="type",
            color_discrete_map={"Pooled": "#54A24B", "Solo": "#E45756"},
            hole=0.35,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=420, showlegend=True, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

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
    st.caption(
        "Averages are totals divided by days that have data in the selected range. "
        "Rates are computed from totals, not averaged day by day."
    )

# ------------------------------------------------------------ Range A vs B
with tab_range:
    st.write(
        "Pick two date ranges. Pickup/drop area and hour filters from the sidebar still apply."
    )
    a_hi = min(max(jit - pd.Timedelta(days=1), dmin), dmax)
    a_lo = max(a_hi - pd.Timedelta(days=13), dmin)
    b_lo = min(max(rb, dmin), dmax)
    b_hi = min(b_lo + pd.Timedelta(days=13), dmax)
    ca, cb = st.columns(2)
    ra = ca.date_input(
        "Range A",
        (a_lo.date(), a_hi.date()),
        min_value=dmin.date(),
        max_value=dmax.date(),
        key="ra",
    )
    rb_ = cb.date_input(
        "Range B",
        (b_lo.date(), b_hi.date()),
        min_value=dmin.date(),
        max_value=dmax.date(),
        key="rb",
    )
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
                rows.append({
                    "Metric": k,
                    "Range A": round(a, 2),
                    "Range B": round(b, 2),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True)

# ----------------------------------------------------------------- By area
with tab_area:
    # ---- Net rides contribution by cluster (from order data) ----
    area_contrib = (
        view.groupby("pickup_cluster", as_index=False)
        .agg(
            gross_rides=("gross_orders", "sum"),
            net_rides=("net_rides", "sum"),
        )
        .sort_values("gross_rides", ascending=False)   # table ordered by gross
        .reset_index(drop=True)
    )
    area_contrib = area_contrib.rename(columns={"pickup_cluster": "Cluster"})

    # Try to attach route counts where the name matches
    route_json_path = "route_cluster_counts.json"
    if os.path.isfile(route_json_path):
        with open(route_json_path, "r", encoding="utf-8") as f:
            route_data = json.load(f)
        route_df = pd.DataFrame(route_data["unique_routes_per_cluster"])
        route_df["key"] = route_df["cluster"].str.strip().str.lower()
        area_contrib["key"] = area_contrib["Cluster"].str.strip().str.lower()
        area_contrib = area_contrib.merge(
            route_df[["key", "unique_routes"]],
            on="key",
            how="left",
        )
        area_contrib["unique_routes"] = area_contrib["unique_routes"].fillna(0).astype(int)
        area_contrib = area_contrib.drop(columns=["key"])
    else:
        area_contrib["unique_routes"] = 0

    # ---------- Slider ----------
    n_top = st.slider("Show top N clusters (rest grouped as Others)", 5, 30, 12)

    # ---------- Title + Pie chart ----------
    st.subheader("Net rides contribution by cluster")

    pie_df = area_contrib.copy()
    if len(pie_df) > n_top:
        top = pie_df.head(n_top)
        others = pd.DataFrame({
            "Cluster": ["Others"],
            "net_rides": [pie_df["net_rides"].iloc[n_top:].sum()],
            "gross_rides": [pie_df["gross_rides"].iloc[n_top:].sum()],
            "unique_routes": [pie_df["unique_routes"].iloc[n_top:].sum()],
        })
        pie_df = pd.concat([top, others], ignore_index=True)

    fig = px.pie(
        pie_df,
        names="Cluster",
        values="net_rides",
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=520, showlegend=True, legend_title_text="Cluster")
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Table (already sorted by gross_rides desc) ----------
    st.subheader("Cluster level data")
    table_df = area_contrib.rename(columns={
        "gross_rides": "Gross rides",
        "net_rides": "Net rides",
        "unique_routes": "Route count",
    })
    st.dataframe(
        table_df[["Cluster", "Route count", "Gross rides", "Net rides"]],
        hide_index=True,
        use_container_width=True,
    )