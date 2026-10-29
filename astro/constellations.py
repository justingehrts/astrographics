"""Constellation stick figures, drawn as connected lines through the
Hipparcos catalog already loaded for the star field.

Line data: Marc van der Sluys, "ConstellationLines" (CC BY 4.0), with its
Bright Star Catalogue (HR) star numbers converted to Hipparcos (HIP)
numbers via the HYG Database's hip/hr cross-reference (CC BY-SA 4.0) --
bundled at data/constellation_lines.csv; see
data/CONSTELLATION_LINES_ATTRIBUTION.md for the full citation.
"""
import csv
import os

_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "constellation_lines.csv")


# IAU constellation names by the abbreviations used in the bundled data.
NAMES = {
    "And": "Andromeda", "Ant": "Antlia", "Aps": "Apus", "Aqr": "Aquarius", "Aql": "Aquila",
    "Ara": "Ara", "Ari": "Aries", "Aur": "Auriga", "Boo": "Boötes", "Cae": "Caelum",
    "Cam": "Camelopardalis", "Cnc": "Cancer", "CVn": "Canes Venatici", "CMa": "Canis Major",
    "CMi": "Canis Minor", "Cap": "Capricornus", "Car": "Carina", "Cas": "Cassiopeia",
    "Cen": "Centaurus", "Cep": "Cepheus", "Cet": "Cetus", "Cha": "Chamaeleon", "Cir": "Circinus",
    "Col": "Columba", "Com": "Coma Berenices", "CrA": "Corona Australis", "CrB": "Corona Borealis",
    "Crv": "Corvus", "Crt": "Crater", "Cru": "Crux", "Cyg": "Cygnus", "Del": "Delphinus",
    "Dor": "Dorado", "Dra": "Draco", "Equ": "Equuleus", "Eri": "Eridanus", "For": "Fornax",
    "Gem": "Gemini", "Gru": "Grus", "Her": "Hercules", "Hor": "Horologium", "Hya": "Hydra",
    "Hyi": "Hydrus", "Ind": "Indus", "Lac": "Lacerta", "Leo": "Leo", "LMi": "Leo Minor",
    "Lep": "Lepus", "Lib": "Libra", "Lup": "Lupus", "Lyn": "Lynx", "Lyr": "Lyra", "Men": "Mensa",
    "Mic": "Microscopium", "Mon": "Monoceros", "Mus": "Musca", "Nor": "Norma", "Oct": "Octans",
    "Oph": "Ophiuchus", "Ori": "Orion", "Pav": "Pavo", "Peg": "Pegasus", "Per": "Perseus",
    "Phe": "Phoenix", "Pic": "Pictor", "Psc": "Pisces", "PsA": "Piscis Austrinus", "Pup": "Puppis",
    "Pyx": "Pyxis", "Ret": "Reticulum", "Sge": "Sagitta", "Sgr": "Sagittarius", "Sco": "Scorpius",
    "Scl": "Sculptor", "Sct": "Scutum", "Ser": "Serpens", "Sex": "Sextans", "Tau": "Taurus",
    "Tel": "Telescopium", "Tri": "Triangulum", "TrA": "Triangulum Australe", "Tuc": "Tucana",
    "UMa": "Ursa Major", "UMi": "Ursa Minor", "Vel": "Vela", "Vir": "Virgo", "Vol": "Volans",
    "Vul": "Vulpecula",
}


def load_constellations(path=_DATA_PATH):
    """Returns {full name: [(hip_a, hip_b), ...]}: each constellation's line
    segments, from its ordered "connect the dots" star path in the bundled
    dataset. Rows sharing a name (Serpens' two parts, Crux's two bars) are merged."""
    figures = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header row
        abbr = ""
        for row in reader:
            abbr = row[0].strip() or abbr  # a blank name continues the previous figure (Crux's 2nd bar)
            hip_ids = [int(v) for v in row[2:] if v.strip()]
            figures.setdefault(NAMES.get(abbr, abbr), []).extend(zip(hip_ids, hip_ids[1:]))
    return figures


def load_constellation_segments(path=_DATA_PATH):
    """Returns a list of (hip_a, hip_b) Hipparcos-ID pairs, one per line
    segment to draw."""
    return [seg for segments in load_constellations(path).values() for seg in segments]
