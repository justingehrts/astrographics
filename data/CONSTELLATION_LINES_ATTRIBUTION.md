# Constellation line data attribution

`constellation_lines.csv` in this directory is derived from two sources:

**Which stars to connect** (the constellation stick-figure selection):

Marc van der Sluys, *ConstellationLines* dataset, v1.0
https://github.com/MarcvdSluys/ConstellationLines/
DOI: 10.5281/zenodo.10397197

Licensed under [Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

That source dataset identifies stars by their number in the *Bright Star
Catalogue* (BSC / Yale / Harvard Revised, "HR"), not by Hipparcos (HIP)
number. This app's star field (`stars_df` in `streamlit_app.py`) is
Hipparcos-indexed, so the star IDs in this file were converted from HR to
HIP numbers using the cross-reference below — the *line selection* (which
stars form each constellation) is unchanged from the CC BY 4.0 source
above; only the numbering scheme was translated.

**HR-to-HIP cross-reference used for that conversion:**

AstroNexus, *HYG Database*, v4.1 (`hip`/`hr` columns only)
https://github.com/astronexus/HYG-Database
Licensed under [Creative Commons Attribution-ShareAlike 4.0 International
(CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

12 of 840 star references (about 1.4%, affecting 21 of 750 line segments)
had no HR->HIP match in that cross-reference and were dropped; everything
else converted cleanly (verified against known values, e.g. Sirius
HR 2491 -> HIP 32349, Arcturus HR 5340 -> HIP 69673).

Each row lists a constellation abbreviation, the number of stars in its
stick-figure path, and that many Hipparcos (HIP) catalog star numbers to
be connected in order.
