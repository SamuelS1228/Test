import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

EARTH_R_M = 6371000.0

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
