import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# =========================
# Config
# =========================
# If True, all controls render in the Streamlit sidebar.
# You can also toggle it in the UI.
USE_SIDEBAR_DEFAULT = True

# =========================
# Helpers (self-contained)
# =========================
def _norm_cols(df):
    renamed = {c: c.strip() for c in df.columns}
    df = df.rename(columns=renamed)
    lut = {c.lower(): c for c in df.columns}
    return df, lut

def _require_cols(df, required, label):
    have = {c.lower() for c in df.columns}
    missing = [c for c in required if c not in have]
    if missing:
        raise ValueError(f"{label}: missing required columns: {', '.join(missing)}")

EARTH_R_M = 6371000.0
def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2.0 * EARTH_R_M * np.arcsin(np.sqrt(a))

def nearest_assignment(customers, hubs):
    if customers.empty or hubs.empty:
        return pd.Series(index=customers.index, dtype=int), pd.Series(index=customers.index, dtype=float)
    c_lat = customers["Lat"].to_numpy()
    c_lon = customers["Lon"].to_numpy()
    h_lat = hubs["Lat"].to_numpy()
    h_lon = hubs["Lon"].to_numpy()
    dm = haversine_m(c_lat[:, None], c_lon[:, None], h_lat[None, :], h_lon[None, :])
    idx = dm.argmin(axis=1)
    dist_m = dm[np.arange(dm.shape[0]), idx]
    return pd.Series(idx, index=customers.index), pd.Series(dist_m, index=customers.index)

def _palette(n):
    base = [
        [31, 119, 180],[255, 127, 14],[44, 160, 44],[214, 39, 40],[148, 103, 189],
        [140, 86, 75],[227, 119, 194],[127, 127, 127],[188, 189, 34],[23, 190, 207],
    ]
    if n <= len(base):
        return base[:n]
    colors = []
    for i in range(n):
        c = base[i % len(base)].copy()
        c = [min(255, int(v * (0.9 + 0.2 * ((i // len(base)) % 2)))) for v in c]
        colors.append(c)
    return colors

# =========================
# UI
# =========================
def main():
    st.set_page_config(page_title="Network Mapper (Bolt‑on, Sidebar‑ready)", layout="wide")
    st.title("📍 Network Mapper — Bolt‑on (Sidebar‑ready)")

    # Sidebar toggle to support both layouts
    with st.sidebar:
        use_sidebar = st.checkbox("Use sidebar controls", value=USE_SIDEBAR_DEFAULT, help="Toggle this if your app uses a sidebar layout.")

    ui = st.sidebar if use_sidebar else st  # route controls to sidebar or main

    # Uploaders
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

    # Locations
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

    # Customers
    cust_df = pd.read_csv(cust_file)
    cust_df, cust_lut = _norm_cols(cust_df)
    _require_cols(cust_df, {"lat", "long"}, "Customers CSV")
    cust_df = cust_df.rename(columns={
        cust_lut["lat"]: "Lat",
        cust_lut["long"]: "Lon",
    })
    if "id" in {c.lower() for c in cust_df.columns}:
        cust_df = cust_df.rename(columns={cust_lut["id"]: "ID"})
    elif "name" in {c.lower() for c in cust_df.columns}:
        cust_df = cust_df.rename(columns={cust_lut["name"]: "ID"})
    else:
        cust_df["ID"] = np.arange(len(cust_df)) + 1
    cust_df = cust_df[["ID", "Lat", "Lon"]].dropna()

    # Warehouse select
    ui.header("Warehouse selection")
    selected_names = ui.multiselect(
        "Choose warehouses to include", options=loc_df["Name"].tolist(), default=loc_df["Name"].tolist(), key="nm_ws"
    )
    hubs_df = loc_df[loc_df["Name"].isin(selected_names)].reset_index(drop=True)
    if hubs_df.empty:
        st.warning("Select at least one warehouse.")
        st.stop()

    # Map options
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

    # Assignments
    idx, dist_m = nearest_assignment(cust_df, hubs_df)
    assign = cust_df.copy()
    assign["AssignedHubIdx"] = idx
    assign["AssignedHub"] = assign["AssignedHubIdx"].map(lambda i: hubs_df.loc[int(i), "Name"] if pd.notna(i) else None)
    assign["DistanceMi"] = dist_m / 1609.34

    # Summary
    st.subheader("Assignments summary")
    per_hub = assign.groupby("AssignedHub", dropna=False).agg(
        Customers=("ID", "count"),
        AvgDistanceMi=("DistanceMi", "mean"),
        MedianDistanceMi=("DistanceMi", "median"),
        MaxDistanceMi=("DistanceMi", "max")
    ).reset_index().sort_values("Customers", ascending=False)
    st.dataframe(per_hub, use_container_width=True)

    # Download
    dl_cols = ["ID", "Lat", "Lon", "AssignedHub", "DistanceMi"]
    st.download_button("⬇️ Download customer assignments CSV",
                       data=assign[dl_cols].to_csv(index=False).encode("utf-8"),
                       file_name="customer_assignments.csv",
                       mime="text/csv")

    # Map
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
    st.pydeck_chart(deck, use_container_width=True)

    st.caption("Tip: Place this file in `pages/` to expose it as a separate page. Toggle the sidebar mode above if your app predominantly uses the sidebar for controls.")

if __name__ == "__main__":
    main()
