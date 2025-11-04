import streamlit as st
from mapper_core import (_norm_cols, _require_cols, nearest_assignment, _palette)
import pandas as pd
import numpy as np
import pydeck as pdk

USE_SIDEBAR_DEFAULT = True

st.set_page_config(page_title="Network Mapper (Sidebar‑ready)", layout="wide")
st.title("📍 Network Mapper — Bolt‑on (Sidebar‑ready)")

with st.sidebar:
    use_sidebar = st.checkbox("Use sidebar controls", value=USE_SIDEBAR_DEFAULT)

ui = st.sidebar if use_sidebar else st

if use_sidebar:
    ui.header("Uploads")
    loc_file = ui.file_uploader("Locations CSV (CITY, STATE, LAT, LONG)", type=["csv"], key="nm_locs_sb")
    cust_file = ui.file_uploader("Customers CSV (LAT, LONG)", type=["csv"], key="nm_custs_sb")
else:
    c1, c2 = st.columns(2)
    with c1:
        loc_file = st.file_uploader("Locations CSV (CITY, STATE, LAT, LONG)", type=["csv"], key="nm_locs_main")
    with c2:
        cust_file = st.file_uploader("Customers CSV (LAT, LONG)", type=["csv"], key="nm_custs_main")

if not loc_file or not cust_file:
    st.info("Upload both CSV files to continue.")
    st.stop()

loc_df = pd.read_csv(loc_file)
loc_df, loc_lut = _norm_cols(loc_df)
_require_cols(loc_df, {"city", "state", "lat", "long"}, "Locations CSV")
loc_df = loc_df.rename(columns={
    loc_lut["city"]: "CITY",
    loc_lut["state"]: "STATE",
    loc_lut["lat"]: "Lat",
    loc_lut["long"]: "Lon",
})
loc_df["Name"] = loc_df["CITY"].astype(str).str.strip() + ", " + loc_df["STATE"].astype(str).str.strip()
loc_df = loc_df[["Name", "CITY", "STATE", "Lat", "Lon"]].dropna()

cust_df = pd.read_csv(cust_file)
cust_df, cust_lut = _norm_cols(cust_df)
_require_cols(cust_df, {"lat", "long"}, "Customers CSV")
cust_df = cust_df.rename(columns={cust_lut["lat"]: "Lat", cust_lut["long"]: "Lon"})
if "id" in {c.lower() for c in cust_df.columns}:
    cust_df = cust_df.rename(columns={cust_lut["id"]: "ID"})
elif "name" in {c.lower() for c in cust_df.columns}:
    cust_df = cust_df.rename(columns={cust_lut["name"]: "ID"})
else:
    cust_df["ID"] = np.arange(len(cust_df)) + 1
cust_df = cust_df[["ID", "Lat", "Lon"]].dropna()

ui.header("Warehouse selection")
selected_names = ui.multiselect("Choose warehouses to include",
                                options=loc_df["Name"].tolist(),
                                default=loc_df["Name"].tolist(),
                                key="nm_ws")
hubs_df = loc_df[loc_df["Name"].isin(selected_names)].reset_index(drop=True)
if hubs_df.empty:
    st.warning("Select at least one warehouse.")
    st.stop()

if use_sidebar:
    ui.header("Map options")
    hub_radius_m = ui.number_input("Hub marker radius (m)", min_value=1000, max_value=100000, value=15000, step=1000, key="nm_hubr")
    cust_radius_m = ui.number_input("Customer marker radius (m)", min_value=500, max_value=20000, value=3000, step=500, key="nm_custr")
    show_rings = ui.checkbox("Show coverage ring (miles)", value=False, key="nm_rings")
    ring_miles = ui.number_input("Coverage radius (miles)", min_value=10, max_value=1000, value=350, step=10, key="nm_ringmi") if show_rings else 350
else:
    with st.expander("Map options", expanded=False):
        c3, c4, c5 = st.columns(3)
        with c3:
            hub_radius_m = st.number_input("Hub marker radius (m)", min_value=1000, max_value=100000, value=15000, step=1000, key="nm_hubr")
        with c4:
            cust_radius_m = st.number_input("Customer marker radius (m)", min_value=500, max_value=20000, value=3000, step=500, key="nm_custr")
        with c5:
            show_rings = st.checkbox("Show coverage ring (miles)", value=False, key="nm_rings")
        ring_miles = st.number_input("Coverage radius (miles)", min_value=10, max_value=1000, value=350, step=10, key="nm_ringmi") if show_rings else 350

idx, dist_m = nearest_assignment(cust_df, hubs_df)
assign = cust_df.copy()
assign["AssignedHubIdx"] = idx
assign["AssignedHub"] = assign["AssignedHubIdx"].map(lambda i: hubs_df.loc[int(i), "Name"] if pd.notna(i) else None)
assign["DistanceMi"] = dist_m / 1609.34

st.subheader("Assignments summary")
per_hub = assign.groupby("AssignedHub", dropna=False).agg(
    Customers=("ID", "count"),
    AvgDistanceMi=("DistanceMi", "mean"),
    MedianDistanceMi=("DistanceMi", "median"),
    MaxDistanceMi=("DistanceMi", "max")
).reset_index().sort_values("Customers", ascending=False)
st.dataframe(per_hub, width='stretch')

dl_cols = ["ID", "Lat", "Lon", "AssignedHub", "DistanceMi"]
st.download_button("⬇️ Download customer assignments CSV",
                   data=assign[dl_cols].to_csv(index=False).encode("utf-8"),
                   file_name="customer_assignments.csv",
                   mime="text/csv")

layers = []
hub_layer = pdk.Layer(
    "ScatterplotLayer",
    data=hubs_df.assign(Tooltip=hubs_df["Name"]),
    get_position="[Lon, Lat]",
    get_radius=hub_radius_m,
    pickable=True,
    filled=True,
    get_fill_color=[0,0,0],
    get_line_color=[255,255,255],
    line_width_min_pixels=1,
)
layers.append(hub_layer)

if show_rings:
    ring_df = hubs_df.assign(RadiusM=ring_miles * 1609.34)
    ring_layer = pdk.Layer(
        "ScatterplotLayer",
        data=ring_df,
        get_position="[Lon, Lat]",
        get_radius="RadiusM",
        stroked=True,
        filled=False,
        line_width_min_pixels=1,
        get_line_color=[0,0,0],
        pickable=False,
    )
    layers.append(ring_layer)

colors = _palette(len(hubs_df))
name_to_idx = {name: i for i, name in enumerate(hubs_df["Name"].tolist())}
assign["Color"] = assign["AssignedHub"].map(lambda n: colors[name_to_idx.get(n, 0)] if pd.notna(n) else [127,127,127])
cust_layer = pdk.Layer(
    "ScatterplotLayer",
    data=assign.assign(Tooltip=assign["ID"].astype(str) + " → " + assign["AssignedHub"].astype(str) + " • " + assign["DistanceMi"].round(1).astype(str) + " mi"),
    get_position="[Lon, Lat]",
    get_radius=cust_radius_m,
    pickable=True,
    filled=True,
    get_fill_color="Color",
    get_line_color=[255,255,255],
    line_width_min_pixels=0.5,
)
layers.append(cust_layer)

center_lat = float(pd.concat([hubs_df["Lat"], assign["Lat"]]).mean())
center_lon = float(pd.concat([hubs_df["Lon"], assign["Lon"]]).mean())
deck = pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=4.0, pitch=0),
    layers=layers,
    tooltip={"text": "{Tooltip}"}
)
st.subheader("Map")
st.pydeck_chart(deck, width='stretch')

st.caption("Place this file in `pages/` to appear as a page. Toggle sidebar mode above.")
