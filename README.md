# Network Mapper — Bolt‑on (Sidebar‑ready)

This version routes all controls to the Streamlit sidebar when the toggle is enabled, so it plays nicely with apps that use sidebar-heavy layouts.

## Files
- `network_mapper_sidebar.py`
- `requirements.txt`

## Run
```bash
pip install -r requirements.txt
streamlit run network_mapper_sidebar.py
```

## Use inside an existing app
- Copy `network_mapper_sidebar.py` into your repo (root or `pages/`).
- Launch your app. If under `pages/`, it appears as a separate page.
- Use the **“Use sidebar controls”** toggle (in the sidebar) to switch layouts.

## Required CSVs
- Locations: `CITY, STATE, LAT, LONG`
- Customers: `LAT, LONG` (+ optional `ID`/`NAME`)
