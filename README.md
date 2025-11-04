# Network Mapper — Updated for Streamlit width API

This package removes deprecated `use_container_width` and uses the new `width` parameter:
- For full-width elements, we set `width='stretch'`.
- For content-sized elements, use `width='content'` (not used by default here).

## Files
- `mapper_core.py` — shared helpers
- `network_mapper.py` — standalone bolt‑on page
- `network_mapper_sidebar.py` — sidebar‑ready bolt‑on page
- `requirements.txt`

## Main module / entrypoint tips
- If your platform expects a specific path (e.g., `/mount/src/test/network_mapper.py`), either:
  1) Rename/copy `network_mapper.py` to that path, or
  2) Update the platform's *Main module file* setting to point to `network_mapper.py` or `pages/Network Mapper.py`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run network_mapper.py          # standard version
# or
streamlit run network_mapper_sidebar.py  # sidebar-first version
```
