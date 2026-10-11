"""
route_cluster_counts.py
-----------------------
Improved mapping – far fewer Unknowns.
Writes route_cluster_counts.json for the dashboard.
"""

import json
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------
# Exact station -> cluster mapping (expanded)
# ------------------------------------------------------------------
STATION_CLUSTER_MAP = {
    # --- existing ---
    "opp. st john's wood sub station": "Tavarakere",
    "wework entrance (near centro mall)": "WeWork",
    "welingkar institute of management development & research bangalore": "Electronic City",
    "eco space entrance": "Ecospace",
    "tata advanced systems limited": "Electronic City",
    "samosa party ecospace": "Ecospace",
    "green glen layout near sobha apartments": "Green Glen",
    "green glen layout near tipsy bull bar": "Green Glen",
    "tavarkere mori (health india hospital)": "Tavarakere",
    "sneha car zone (opp. egl entrance)": "EGL",
    "yokogawa ltd. (infosys gate 4)": "Electronic City",
    "intel exit gate": "Intel",
    "(opp) eco space - bellandur bus stop": "Ecospace",
    "adobe/oracle – prestige tech park": "Adobe",
    "adobe/oracle - prestige tech park": "Adobe",
    "the bay at ecoworld": "Ecoworld",
    "(opp.) manipal hospital": "Manipal",
    "salarpuria softzone": "Salarpuria",
    "(opp.) green trends unisex salon": "Tavarakere",
    "bhel office gate": "Electronic City",
    "jpmorgan chase & co., tower c, gate 2": "JPMC/Wells Fargo",
    "croma, electronic city": "Neeladri Nagar",
    "infosys cresent building": "Electronic City",
    "wework entrance": "WeWork",
    "near infosys foundation metro": "Electronic City",
    "near royal paradise coliving pg": "Tavarakere",
    "intuit rmz eco space": "Ecospace",
    "infosys foundation metro (near infosys campus)": "Electronic City",
    "salarpuria softzone skywalk - towards sarjapur road": "Salarpuria",
    "christ university": "Christ/IBC",
    "(opp.) adobe/oracle - prestige tech park": "Adobe",
    "deloitte – prestige/helios tech park": "Deloitte",
    "deloitte - prestige/helios tech park": "Deloitte",
    "vaishnavi tech square (navi)": "Navi",
    "fashion factory, neeladri nagar": "Neeladri Nagar",
    "ecoworld building 1–4 pickup": "Ecoworld",
    "infosys foundation metro (near seasons restaurant)": "Electronic City",
    "vaishnavi tech park": "Vaishnavi",
    "(opp.) st john's wood sub station": "Electronic City",
    "intel shuttle stop": "Intel",
    "ibc knowledge park gate": "Christ/IBC",
    "(opp.) wells fargo tower 1": "JPMC/Wells Fargo",
    "(opp) christ university": "Christ/IBC",
    "wells fargo tower 1": "JPMC/Wells Fargo",
    "the hub (near reliance digital)": "HUB",

    # --- newly added (high-frequency unknowns) ---
    "jyoti niwas college": "Koramangala",
    "madiwala market": "Koramangala",
    "sakra hospital": "Marathahalli",
    "btm metro": "BTM",
    "brigade tech gardens": "Whitefield",
    "accenture -a": "Ecospace",
    "accenture -c": "Ecospace",
    "bellandur gate rd": "Bellandur",
    "vyedhi hosptal": "Whitefield",
    "vydehi hospital": "Whitefield",
    "itpl": "Whitefield",
    "sigma tech park": "Whitefield",
    "sjr i park": "Whitefield",
    "nr layout": "Whitefield",
    "sathya sai hospital metro": "Whitefield",
    "bellandur post office": "Bellandur",
    "eco world honeywell campus": "Ecoworld",
    "wipro gate bus stop": "Electronic City",
    "e-city bda complex": "Electronic City",
    "prestige silver oak": "Whitefield",
    "nallur halli": "Whitefield",
    "indiqube orion": "HSR",
    "indiqube orion, hsr sec 1": "HSR",
    "lti block 1  whitefield": "Whitefield",
    "birla circle": "Whitefield",
    "iptl": "Whitefield",
    "wipro junction": "Electronic City",
    "indra canteen(iti layout)": "Whitefield",
    "bommanahalli metro": "Bommanahalli",
    "silkboard metro": "Silkboard",
    "nexus vega city mall": "Bannerghatta",
    "aet junction": "Bellandur",
    "esteem enclave arekere": "Arekere",
    "ews colony, btm second stage": "BTM",
    "wilson garden lakkasandra": "Wilson Garden",
    "lal bagh metro": "Lalbagh",
    "marathahalli bridge": "Marathahalli",
    "silverwood regency apartment, hosa rd": "Hosa Road",
    "cloudnine hospital, sarjapur rd": "Sarjapur",
    "suncity apartments, ibbluru": "Sarjapur",
    "ozone residenza, haralur rd": "Haralur",
    "sarjapura junction": "Sarjapur",
    "sarjapur social": "Sarjapur",
    "munnekolal": "Marathahalli",
    "m.v.j college": "Whitefield",
    "huskur road metro": "Electronic City",
    "nexus koramangala": "Koramangala",
    "apollo cradle, koramangala": "Koramangala",
    "ibc knowledge park": "Christ/IBC",
    "jayanagar metro": "Jayanagar Metro",
    "jain university, kr layout, 25th main road": "Jain",
    "cpwd complex hsr": "HSR",
    "egl centre avenue": "EGL",
    "near anugraha apartments": "Tavarakere",
    "stanza living podolsk house": "Tavarakere",
    "tavarekere aladamara bus stop": "Tavarakere",
    "concentrix ecospace": "Ecospace",
    "raheja residency, koramangala": "Koramangala",
    "balaji theatre, 4th main road, tavarakere": "Tavarakere",
    "bethany kids centre, 4th cross road maruti nagar": "Tavarakere",
    "bee9 juice corner": "Tavarakere",
    "avs convent high school, chocolate factory road, tavarakere": "Tavarakere",
    "gangotri circle, chikka madiwala": "Tavarakere",
    "lenskart, maruti nagar road": "Tavarakere",
    "green glen layout": "Green Glen",
    "sobha dahlia, green glen layout": "Green Glen",
}


def get_station_cluster(name):
    if pd.isna(name):
        return None

    n = str(name).strip().lower()

    # 1. Exact mapping
    if n in STATION_CLUSTER_MAP:
        return STATION_CLUSTER_MAP[n]

    # 2. Keyword fallbacks
    keyword_map = [
        (["christ", "ibc"], "Christ/IBC"),
        (["wework", "centro mall"], "WeWork"),
        (["intel"], "Intel"),
        (["ecoworld", "eco world"], "Ecoworld"),
        (["eco space", "ecospace"], "Ecospace"),
        (["adobe", "oracle"], "Adobe"),
        (["deloitte"], "Deloitte"),
        (["vaishnavi tech square"], "Navi"),
        (["vaishnavi"], "Vaishnavi"),
        (["wells fargo", "jpmorgan"], "JPMC/Wells Fargo"),
        (["green glen"], "Green Glen"),
        (["tavarkere", "tavarekere", "st john", "royal", "stanza living",
          "bee9", "anugraha", "chocolate", "balaji", "bethany", "gangotri", "maruti"], "Tavarakere"),
        (["infosys", "bhel", "tata", "yokogawa", "welingkar", "wellingkar"], "Electronic City"),
        (["fashion", "croma", "neeladri"], "Neeladri Nagar"),
        (["salarpuria"], "Salarpuria"),
        (["manipal"], "Manipal"),
        (["egl"], "EGL"),
        (["hub", "reliance digital"], "HUB"),
        (["koramangala", "jyoti niwas", "madiwala", "apollo cradle", "nexus koramangala"], "Koramangala"),
        (["hsr", "indiqube orion", "cpwd"], "HSR"),
        (["jain"], "Jain"),
        (["jayanagar"], "Jayanagar Metro"),
        (["btm"], "BTM"),
        (["silkboard"], "Silkboard"),
        (["whitefield", "itpl", "brigade tech", "sigma tech", "sjr i park",
          "vydehi", "vyedhi", "sathya sai", "lti", "birla", "iptl"], "Whitefield"),
        (["marathahalli", "sakra", "munnekolal"], "Marathahalli"),
        (["bellandur"], "Bellandur"),
        (["sarjapur", "sarjapura", "cloudnine", "suncity", "ibbluru"], "Sarjapur"),
        (["haralur"], "Haralur"),
        (["bommanahalli"], "Bommanahalli"),
        (["arekere", "esteem enclave"], "Arekere"),
        (["wilson garden", "lakkasandra"], "Wilson Garden"),
        (["lal bagh", "lalbagh"], "Lalbagh"),
        (["hosa rd", "hosa road"], "Hosa Road"),
        (["bannerghatta", "vega city"], "Bannerghatta"),
        (["diamond district", "sterling mac", "lucknow"], "Diamond District"),
    ]

    for keywords, cluster in keyword_map:
        if any(k in n for k in keywords):
            return cluster

    return "Unknown"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
CSV_PATH = "query_result_2026-10-11T09_28_09.413309+05_30.csv"
OUTPUT_JSON = "route_cluster_counts.json"


def main():
    print(f"Reading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    df["start_cluster"] = df["data__routestart__name"].apply(get_station_cluster)
    df["end_cluster"]   = df["data__routeend__name"].apply(get_station_cluster)

    # Unique routes that start OR end in the cluster
    melted = pd.concat([
        df[["data__id", "start_cluster"]].rename(columns={"start_cluster": "cluster"}),
        df[["data__id", "end_cluster"]].rename(columns={"end_cluster": "cluster"}),
    ])
    unique_counts = (
        melted.groupby("cluster")["data__id"]
        .nunique()
        .sort_values(ascending=False)
        .rename_axis("cluster")
        .reset_index(name="unique_routes")
    )

    # Also keep start / end breakdowns
    start_counts = (
        df["start_cluster"].value_counts()
        .rename_axis("cluster").reset_index(name="route_count")
    )
    end_counts = (
        df["end_cluster"].value_counts()
        .rename_axis("cluster").reset_index(name="route_count")
    )

    payload = {
        "unique_routes_per_cluster": unique_counts.to_dict(orient="records"),
        "start_cluster_counts": start_counts.to_dict(orient="records"),
        "end_cluster_counts": end_counts.to_dict(orient="records"),
    }

    Path(OUTPUT_JSON).write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUTPUT_JSON}")
    print("\n=== Unique routes per cluster ===")
    print(unique_counts.to_string(index=False))

    # Show remaining Unknowns so you can keep improving
    still_unknown_start = df[df["start_cluster"] == "Unknown"]["data__routestart__name"].value_counts()
    still_unknown_end   = df[df["end_cluster"] == "Unknown"]["data__routeend__name"].value_counts()
    if len(still_unknown_start) or len(still_unknown_end):
        print("\n=== Still Unknown (start) ===")
        print(still_unknown_start.head(20).to_string() if len(still_unknown_start) else "None")
        print("\n=== Still Unknown (end) ===")
        print(still_unknown_end.head(20).to_string() if len(still_unknown_end) else "None")
    else:
        print("\nNo Unknowns left!")


if __name__ == "__main__":
    main()