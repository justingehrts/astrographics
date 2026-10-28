# Fonts

Proxima Nova (station group typeface), copied unchanged from the `fonts/` folder of
`justingehrts/graph-maker` (branch `weather-impact-2`). Used for the graphic's labels.

| Text | File |
|---|---|
| Sun / planet / Moon labels | `proximanova-bold.ttf` |
| Bright star names | `proximanova-medium.ttf` |
| Star catalog IDs ("HIP nnn") | `proximanova-regular.ttf` |
| Asterism names, altitude and compass labels | `proximanova-semibold.ttf` |
| (spare) | `proximanova-extrabold.ttf` |

Proxima Nova is a commercial font; the owner confirmed the station group is licensed to use and
distribute it. If a file is missing, `label_font()` in `streamlit_app.py` falls back to the default
sans-serif at the same weight.
