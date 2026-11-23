"""AI part 1b - IC marking OCR: read the part number printed on each detected chip
(Tesseract) and look it up in a local parts catalogue (data/ic_catalogue.csv).

Chip markings are hard for OCR: small, low-contrast laser etching, light text on
a dark package, and the chip can sit at any of four rotations on the board. So
for every IC crop we:
  1. upscale it (Tesseract wants text roughly 20-40 px tall),
  2. normalise contrast and make the text dark-on-light,
  3. read it at 0, 90, 180 and 270 degrees,
  4. keep the rotation whose words Tesseract is most confident about
     (a catalogue hit wins outright),
  5. match the text against the parts catalogue, allowing small OCR slips.
Only chips big enough to carry legible text are attempted.
"""
import csv
import difflib
import re
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageOps

from ocr_utils import _single_thread_tesseract

ROOT = Path(__file__).parent
CATALOGUE = ROOT / "data" / "ic_catalogue.csv"

MIN_SIDE = 60           # px in the photo; smaller chips are not attempted
TARGET_LONG = 900       # upscale crops so the long side is about this
FUZZY = 0.83            # catalogue match threshold (difflib ratio) for keys of 6+ chars
PART_RE = re.compile(r"\b(?=[A-Z0-9\-]*\d)(?=[A-Z0-9\-]*[A-Z])[A-Z0-9][A-Z0-9\-]{4,}\b")


def load_catalogue(path=CATALOGUE):
    with open(path, newline="", encoding="utf-8") as f:
        return [{**r, "norm": norm(r["key"])} for r in csv.DictReader(f)]


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


# Preprocessing variants, tried in order until one gives an exact catalogue hit.
# Each catches chips the others miss (measured on tests/ic_markings.json).
VARIANTS = [
    {"inner": 0.0, "target": 900, "median": 0, "psm": 6},     # whole crop
    {"inner": 0.08, "target": 1200, "median": 3, "psm": 6},   # trim pins/edges, smooth package texture
    {"inner": 0.08, "target": 900, "median": 0, "psm": 11},   # sparse-text mode for scattered markings
]


def prepare(crop: Image.Image, inner=0.0, target=TARGET_LONG, median=0, **_) -> Image.Image:
    if inner:
        w, h = crop.size
        dx, dy = int(w * inner), int(h * inner)
        crop = crop.crop((dx, dy, w - dx, h - dy))
    g = ImageOps.grayscale(crop)
    s = min(8.0, max(1.0, target / max(g.size)))
    g = g.resize((round(g.width * s), round(g.height * s)), Image.LANCZOS)
    if median:
        g = g.filter(ImageFilter.MedianFilter(median))
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    # markings are usually light on a dark package: Tesseract reads dark-on-light best
    if np.median(np.asarray(g)) < 128:
        g = ImageOps.invert(g)
    return g


def _read(img: Image.Image, psm=6):
    """Tesseract words with confidences."""
    with _single_thread_tesseract():
        d = pytesseract.image_to_data(img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words, lines = [], {}
    for txt, conf, b, p, ln in zip(d["text"], d["conf"], d["block_num"], d["par_num"], d["line_num"]):
        t = txt.strip()
        if not t:
            continue
        c = float(conf)
        words.append((t, c))
        lines.setdefault((b, p, ln), []).append(t)
    return words, [" ".join(v) for v in lines.values()]


# Characters OCR confuses on chip markings; both sides are folded the same way
# before fuzzy matching (exact matching still uses the raw text).
LOOKALIKE = str.maketrans("OQDILSBZG", "000115826")


def match_catalogue(text: str, catalogue):
    """Best catalogue entry found in the OCR text: exact substring of the
    space-free text first, then a fuzzy match over same-length windows of the
    look-alike-folded text (one slip allowed in a 6-character key)."""
    joined = norm(text)
    if not joined:
        return None, 0.0
    # fuzzy matching stays inside one printed line: junk scattered over many
    # lines must not add up to a part number
    lines = [norm(ln) for ln in text.splitlines() if norm(ln)]
    best, best_r = None, 0.0
    for entry in catalogue:
        k = entry["norm"]
        if k in joined:
            r = 1.0 + len(k) / 100            # prefer the longest exact key
        elif len(k) >= 6 and re.search(r"\d", k) and re.search(r"[A-Z]", k):
            r = _fuzzy(k, lines)
            if r < FUZZY:
                continue
        else:
            continue                          # short or all-letter keys: exact only
        if r > best_r:
            best, best_r = entry, r
    return best, best_r


def _fuzzy(key, lines):
    """Best similarity of `key` to a same-length window of one line, comparing
    look-alike-folded text. The raw window must hold at least half of the key's
    digits, so letters alone cannot be folded into a part number."""
    kf, need = key.translate(LOOKALIKE), (sum(c.isdigit() for c in key) + 1) // 2
    best = 0.0
    for ln in lines:
        if len(ln) < len(key) - 1:
            continue
        for i in range(max(1, len(ln) - len(key) + 1)):
            win = ln[i:i + len(key)]
            if sum(c.isdigit() for c in win) < need:
                continue
            best = max(best, difflib.SequenceMatcher(None, kf, win.translate(LOOKALIKE)).ratio())
    return best


def read_ic(crop: Image.Image, catalogue=None):
    """Read one IC crop. Returns dict(text, lines, rotation, part, match, score)."""
    catalogue = catalogue if catalogue is not None else load_catalogue()
    best = None
    for v in VARIANTS:
        base = prepare(crop, **v)
        for rot in (0, 90, 180, 270):
            img = base.rotate(rot, expand=True, fillcolor=255)
            words, lines = _read(img, v["psm"])
            good = [(t, c) for t, c in words if c >= 40 and len(norm(t)) >= 2]
            score = sum(len(norm(t)) * c / 100 for t, c in good)
            text = "\n".join(lines)
            entry, ratio = match_catalogue(text, catalogue)
            if entry:
                score += 100 * ratio
            cand = {"text": text, "lines": lines, "rotation": rot, "score": round(score, 1),
                    "match": entry and {k: entry[k] for k in ("key", "manufacturer", "description")},
                    "match_ratio": round(ratio, 2)}
            if best is None or cand["score"] > best["score"]:
                best = cand
            if ratio >= 1.0:                # exact catalogue hit: done
                break
        if best["match_ratio"] >= 1.0:
            break
    # part number: catalogue key if matched, else the most part-number-like token read
    if best["match"]:
        best["part"] = best["match"]["key"]
    else:
        tokens = PART_RE.findall(best["text"].upper())
        best["part"] = max(tokens, key=len) if tokens else None
    return best


def read_ics(img: Image.Image, parts, max_ics=6, catalogue=None):
    """Read markings on the largest detected ICs in a board photo."""
    catalogue = catalogue if catalogue is not None else load_catalogue()
    ics = [p for p in parts if p["name"] == "IC"
           and min(p["box"][2] - p["box"][0], p["box"][3] - p["box"][1]) >= MIN_SIDE]
    ics.sort(key=lambda p: -(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
    rgb = img.convert("RGB")
    out = []
    for p in ics[:max_ics]:
        r = read_ic(rgb.crop(tuple(round(v) for v in p["box"])), catalogue)
        out.append({**r, "box": p["box"]})
    return out
