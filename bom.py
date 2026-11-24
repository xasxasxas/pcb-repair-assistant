"""Bill of materials (BOM) from one board photo.

Combines the part detector (counts per part type) with IC marking OCR (part
numbers of the chips it could identify). Identified chips get their own line;
the remaining ICs, and every other part type, are listed as counts.

The BOM lists what the AI found in the photo, not the board's full parts list:
the detector finds about half of the smallest parts on dense boards, so the
quantities are a starting point for the technician to check.
"""
import pandas as pd

COLUMNS = ["Line", "Part type", "Part number", "Manufacturer", "Description", "Qty", "Source"]
ORDER = ["IC", "Transistor/Diode", "Connector", "Switch", "Inductor", "Capacitor", "Resistor"]


def build_bom(parts, chips=()):
    """parts: detector output (dicts with 'name'); chips: read_ics output (dicts with 'match')."""
    counts = pd.Series([p["name"] for p in parts], dtype=str).value_counts().to_dict()
    rows = []
    ids = pd.Series([c["match"]["key"] for c in chips if c.get("match")], dtype=str).value_counts()
    meta = {c["match"]["key"]: c["match"] for c in chips if c.get("match")}
    for key, n in ids.items():
        m = meta[key]
        rows.append({"Part type": "IC", "Part number": key, "Manufacturer": m["manufacturer"],
                     "Description": m["description"], "Qty": int(n), "Source": "Detector + chip OCR"})
    if "IC" in counts:
        counts["IC"] -= int(ids.sum()) if len(ids) else 0
    for name in sorted(counts, key=lambda k: (ORDER.index(k) if k in ORDER else len(ORDER), k)):
        if counts[name] > 0:
            rows.append({"Part type": name,
                         "Part number": "" if name != "IC" or not len(ids) else "(not identified)",
                         "Manufacturer": "", "Description": "", "Qty": int(counts[name]),
                         "Source": "Detector"})
    df = pd.DataFrame(rows, columns=COLUMNS[1:])
    df.insert(0, "Line", range(1, len(df) + 1))
    return df
