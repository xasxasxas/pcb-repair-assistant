"""
PCB Repair Assistant - everything the app does with a board photo and a job.

  1. TAG OCR         read the yellow asset tag (Tesseract)             AI, pretrained
  2. PART DETECTOR   find and classify components (YOLOv8s, tiled)     AI, fine-tuned by us
  3. CHIP OCR        read the part number printed on the big chips     AI, pretrained
  4. BOM             parts list built from 2 and 3
  5. EXPLANATION     why the Random Forest gave this BER risk / hours  ML, trained by us

The models are created and scored in train.py. The screens are in app.py.
"""
import contextlib
import csv
import difflib
import json
import os
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage

ROOT = Path(__file__).parent
MODELS = ROOT / "models"


# =====================================================================
# 1. TAG OCR - read board type, revision and serial number from the photo
# =====================================================================
# Raw OCR on a whole board photo fails (15 of 50 serials): silkscreen text and
# chip markings drown out the tag. So we first FIND the yellow tag by colour,
# crop it, clean it up, and only then run OCR (49 of 50).

def preprocess(img):
    """Grayscale -> upscale -> contrast stretch -> Otsu threshold (black text on white)."""
    g = ImageOps.grayscale(img)
    if g.width < 1600:
        s = 1600 / g.width
        g = g.resize((int(g.width * s), int(g.height * s)), Image.LANCZOS)
    arr = np.asarray(ImageOps.autocontrast(g, cutoff=2))
    # Otsu: pick the threshold that best separates dark and light pixels
    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    total, sum_all = arr.size, np.dot(np.arange(256), hist)
    w_b = sum_b = 0.0
    best_t, best_var = 128, 0.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0 or w_b == total:
            continue
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / (total - w_b)
        var = w_b * (total - w_b) * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return Image.fromarray(((arr > best_t) * 255).astype(np.uint8))


def _best_region(mask, min_frac=0.004, min_fill=0.6, aspect=None):
    """Largest connected region that looks like a filled rectangle (a tag)."""
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=4))
    lab, _ = ndimage.label(mask)
    best, best_area = None, min_frac * mask.size
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        area = (lab[sl] == i).sum()
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        ratio = max(w, h) / max(1, min(w, h))
        if aspect and not (aspect[0] <= ratio <= aspect[1]):
            continue                     # e.g. long gold edge connectors are yellow too
        if area > best_area and area / (h * w) >= min_fill:
            best, best_area = sl, area
    return best


def find_label(img):
    """Crop the photo to the asset tag: a yellow tag first, else a white sticker.
    Returns the whole photo if nothing tag-like is found."""
    rgb = img.convert("RGB")
    scale = min(1.0, 700 / max(rgb.size))
    small = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale)))) if scale < 1 else rgb
    a = np.asarray(small).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    yellow = (r > 170) & (g > 140) & (b < 110) & (r - b > 100) & (g - b > 70)
    white = (a.min(axis=2) > 170) & ((a.max(axis=2) - a.min(axis=2)) < 40)
    sl = _best_region(yellow, aspect=(1.6, 3.6)) or _best_region(white, min_frac=0.01)
    if sl is None:
        return img
    pad = 4
    y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, small.height)
    x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, small.width)
    return rgb.crop((int(x0 / scale), int(y0 / scale), int(x1 / scale), int(y1 / scale)))


_env_lock = threading.Lock()


@contextlib.contextmanager
def _single_thread_tesseract():
    """Tesseract's own multi-threading makes small images much slower, so it is
    limited to one thread, and only while Tesseract runs."""
    with _env_lock:
        old = os.environ.get("OMP_THREAD_LIMIT")
        os.environ["OMP_THREAD_LIMIT"] = "1"
        try:
            yield
        finally:
            if old is None:
                os.environ.pop("OMP_THREAD_LIMIT", None)
            else:
                os.environ["OMP_THREAD_LIMIT"] = old


def _tesseract(img, config):
    with _single_thread_tesseract():
        return pytesseract.image_to_string(img, config=config)


def read_text(img, clean=True):
    """OCR text of the tag. clean=False runs raw OCR on the whole photo (for comparison)."""
    if not clean:
        return _tesseract(img.convert("RGB"), "--psm 11")
    text = _tesseract(preprocess(find_label(img)), "--psm 6")
    if not re.search(r"S\s*[/I1L|]?\s*N\s*[:.\-]?\s*[0-9OQ]{4}", text.upper()):
        # tag not isolated (e.g. a yellow tag on a yellow board): read the whole photo too
        text += "\n" + _tesseract(img.convert("RGB"), "--psm 11")
    return text


def parse_label(text, known_boards):
    """Pull board type, revision and serial number out of the OCR text,
    forgiving common OCR slips (O read as 0, S/N read as S|N, ...)."""
    up = text.upper()
    serial = rev = board = None
    m = re.search(r"S\s*[/I1L|]?\s*N\s*[:.\-]?\s*([0-9OQ]{4,6})", up)
    if m:
        serial = m.group(1).replace("O", "0").replace("Q", "0")
    m = re.search(r"REV\.?\s*([A-Z])\b", up)
    if m:
        rev = m.group(1)
    for tok in re.findall(r"[A-Z0-9]{2,}\s*[-_]\s*[A-Z0-9]{2,}", up):
        tok = re.sub(r"\s*[-_]\s*", "-", tok)
        match = difflib.get_close_matches(tok, known_boards, n=1, cutoff=0.6)
        if not match:                    # OCR often swaps letters for look-alike digits
            match = difflib.get_close_matches(tok.translate(str.maketrans("0158", "OISB")),
                                              known_boards, n=1, cutoff=0.6)
        if match:
            board = match[0]
            break
    return {"board_type": board, "revision": rev, "serial": serial}


# =====================================================================
# 2. PART DETECTOR - YOLOv8s, fine-tuned on tiles of real PCB photos
# =====================================================================
# The network looks at a fixed 1024 px square. In one pass the whole photo is
# squeezed into it and a small resistor becomes a few blurry pixels. Tiled
# detection enlarges the photo, cuts it into overlapping tiles and detects in
# each tile, so small parts keep enough pixels to be recognised.

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


class PartDetector:
    def __init__(self, model=MODELS / "parts.onnx", meta=MODELS / "parts.json"):
        import onnxruntime as ort
        self.meta = json.loads(Path(meta).read_text())
        self.names = self.meta["names"]              # the 7 part types
        self.size = int(self.meta["imgsz"])          # network input size (1024)
        # zoom the model was trained at: tiles were cut from photos resized to 2048 px
        self.zoom = self.meta["tiling"]["long_side"] / self.size if "tiling" in self.meta else 1.5
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def _run(self, canvas):
        """Run the network on one square RGB image. Returns raw output (4 + classes, N)."""
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return self.session.run(None, {self.input: x})[0][0]

    def _named(self, boxes, scores, cls, iou):
        keep = nms(boxes, scores, cls, iou)
        return [{"class_id": int(cls[i]), "name": self.names[int(cls[i])], "score": float(scores[i]),
                 "box": tuple(map(float, boxes[i]))} for i in keep]

    def detect(self, img, score=0.30, iou=0.45):
        """Single pass: the whole photo, letterboxed into the network's input square."""
        w, h = img.size
        s = min(self.size / w, self.size / h)
        nw, nh = round(w * s), round(h * s)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (px, py))
        return self._named(*decode(self._run(canvas), s, px, py, w, h, score), iou)

    def detect_tiled(self, img, zoom=None, overlap=0.25, score=0.30, iou=0.45):
        """Tiled detection (what the app uses).

        * the photo is enlarged `zoom` times relative to a single pass and cut
          into overlapping tiles of the network's input size
        * a box touching an inner tile edge is dropped: the part is cut off
          there, and the neighbouring tile sees it whole
        * one extra pass over the whole photo catches large parts (big chips,
          connectors) that do not fit inside one tile
        * duplicates from overlapping tiles are merged with NMS
        """
        img = img.convert("RGB")
        w, h = img.size
        f = (zoom or self.zoom) * self.size / max(w, h)          # original px -> enlarged px
        W, H = max(1, round(w * f)), max(1, round(h * f))
        big = img.resize((W, H), Image.BILINEAR)
        T, edge = self.size, 4
        stride = max(1, int(T * (1 - overlap)))
        boxes, scores, cls = [], [], []
        for y0 in tile_starts(H, T, stride):
            for x0 in tile_starts(W, T, stride):
                tile = Image.new("RGB", (T, T), (114, 114, 114))
                tile.paste(big.crop((x0, y0, min(x0 + T, W), min(y0 + T, H))), (0, 0))
                b, s_, c_ = decode(self._run(tile), 1.0, 0, 0, T, T, score)
                for i in nms(b, s_, c_, iou):
                    (x1, y1, x2, y2), sc, c = b[i], s_[i], c_[i]
                    cut_off = ((x0 > 0 and x1 <= edge) or (y0 > 0 and y1 <= edge) or
                               (x0 + T < W and x2 >= T - edge) or (y0 + T < H and y2 >= T - edge))
                    if not cut_off:
                        boxes.append((min((x1 + x0) / f, w), min((y1 + y0) / f, h),
                                      min((x2 + x0) / f, w), min((y2 + y0) / f, h)))
                        scores.append(sc)
                        cls.append(c)
        for d in self.detect(img, score, iou):                   # the whole-photo pass
            boxes.append(d["box"]); scores.append(d["score"]); cls.append(d["class_id"])
        if not boxes:
            return []
        return self._named(np.array(boxes, float), np.array(scores, float), np.array(cls, int), iou)

    def draw(self, img, parts):
        """Photo with one box per part, plus the colour used for each part type."""
        out = img.convert("RGB").copy()
        d = ImageDraw.Draw(out)
        colour = {n: COLORS[i % len(COLORS)] for i, n in enumerate(self.names)}
        for p in parts:
            d.rectangle(p["box"], outline=colour[p["name"]], width=max(2, round(max(out.size) / 500)))
        return out, {n: colour[n] for n in sorted({p["name"] for p in parts})}


def decode(out, s, px, py, w, h, score=0.30):
    """Raw YOLOv8 output (4 + classes, N) -> (boxes, scores, classes) in original
    pixels, keeping boxes above the score threshold. (s, px, py) undo the letterbox."""
    boxes, scores = out[:4].T, out[4:].T
    cls, conf = scores.argmax(1), scores.max(1)
    keep = conf > score
    boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
    if not len(conf):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    cx, cy, bw, bh = boxes.T
    x1 = np.clip((cx - bw / 2 - px) / s, 0, w); y1 = np.clip((cy - bh / 2 - py) / s, 0, h)
    x2 = np.clip((cx + bw / 2 - px) / s, 0, w); y2 = np.clip((cy + bh / 2 - py) / s, 0, h)
    ok = ((x2 - x1) >= 1) & ((y2 - y1) >= 1)
    return np.stack([x1, y1, x2, y2], 1)[ok], conf[ok], cls[ok]


def nms(boxes, scores, cls, iou=0.45):
    """Non-maximum suppression: go through boxes from most to least confident
    and drop any box that overlaps an already-kept box of the same part type
    by more than `iou`. Returns the indices kept."""
    boxes = np.asarray(boxes, float)
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order = np.argsort(-np.asarray(scores), kind="stable")
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(int(i))
        inter = (np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None) *
                 np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None))
        union = area[i] + area[rest] - inter
        overlap = np.where(union > 0, inter / np.where(union > 0, union, 1), 0) > iou
        order = rest[~(overlap & (cls[rest] == cls[i]))]
    return keep


def tile_starts(length, tile, stride):
    """Where tiles start along one side: regular steps, last tile flush with the end."""
    if length <= tile:
        return [0]
    return list(range(0, length - tile, stride)) + [length - tile]


# =====================================================================
# 3. CHIP OCR - read the part number on each large chip
# =====================================================================
# Chip markings are tiny, grey laser etching on black, and the chip can sit at
# any of four rotations, so plain OCR reads almost nothing (4 of 36 chips).
# We enlarge and clean each crop, read it at 4 rotations with up to 3 clean-ups,
# and match the text against a parts catalogue (24 of 36, no false claims).

MIN_SIDE = 60            # px in the photo; smaller chips are not attempted
FUZZY = 0.83             # how similar OCR text must be to a catalogue part number
VARIANTS = [             # clean-ups, tried in order until one gives an exact catalogue hit
    {"inner": 0.0, "target": 900, "median": 0, "psm": 6},      # whole crop
    {"inner": 0.08, "target": 1200, "median": 3, "psm": 6},    # trim the pins, smooth the package texture
    {"inner": 0.08, "target": 900, "median": 0, "psm": 11},    # sparse-text mode for scattered markings
]
LOOKALIKE = str.maketrans("OQDILSBZG", "000115826")   # characters OCR mixes up on chips


def norm(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def load_catalogue(path=ROOT / "data" / "ic_catalogue.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        return [{**r, "norm": norm(r["key"])} for r in csv.DictReader(f)]


def prepare(crop, inner=0.0, target=900, median=0, **_):
    """Enlarge, stretch contrast, sharpen, and make the text dark on light."""
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
    return ImageOps.invert(g) if np.median(np.asarray(g)) < 128 else g


def _read_words(img, psm):
    """Tesseract words with their confidence, and the text grouped into lines."""
    with _single_thread_tesseract():
        d = pytesseract.image_to_data(img, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words, lines = [], {}
    for txt, conf, b, p, ln in zip(d["text"], d["conf"], d["block_num"], d["par_num"], d["line_num"]):
        if txt.strip():
            words.append((txt.strip(), float(conf)))
            lines.setdefault((b, p, ln), []).append(txt.strip())
    return words, [" ".join(v) for v in lines.values()]


def match_catalogue(text, catalogue):
    """Best catalogue part found in the OCR text, and how sure the match is.
    An exact match anywhere in the text wins. Otherwise a fuzzy match is allowed,
    but only for part numbers of 6+ characters with both letters and digits,
    and only within ONE printed line, so noise spread over several lines can
    never add up to a part number."""
    joined = norm(text)
    if not joined:
        return None, 0.0
    lines = [norm(ln) for ln in text.splitlines() if norm(ln)]
    best, best_r = None, 0.0
    for entry in catalogue:
        k = entry["norm"]
        if k in joined:
            r = 1.0 + len(k) / 100                   # prefer the longest exact match
        elif len(k) >= 6 and re.search(r"\d", k) and re.search(r"[A-Z]", k):
            r = _fuzzy(k, lines)
            if r < FUZZY:
                continue
        else:
            continue
        if r > best_r:
            best, best_r = entry, r
    return best, best_r


def _fuzzy(key, lines):
    """Similarity of `key` to the best same-length window of one line, after
    folding look-alike characters. The window must contain real digits, so
    letters alone cannot be 'corrected' into a part number."""
    kf, need = key.translate(LOOKALIKE), (sum(c.isdigit() for c in key) + 1) // 2
    best = 0.0
    for ln in lines:
        if len(ln) < len(key) - 1:
            continue
        for i in range(max(1, len(ln) - len(key) + 1)):
            win = ln[i:i + len(key)]
            if sum(c.isdigit() for c in win) >= need:
                best = max(best, difflib.SequenceMatcher(None, kf, win.translate(LOOKALIKE)).ratio())
    return best


def read_ic(crop, catalogue):
    """Read one chip crop. Returns dict(text, rotation, match): the best reading
    over the clean-ups and rotations; `match` is the catalogue part or None."""
    best = None
    for v in VARIANTS:
        base = prepare(crop, **v)
        for rot in (0, 90, 180, 270):
            words, lines = _read_words(base.rotate(rot, expand=True, fillcolor=255), v["psm"])
            text = "\n".join(lines)
            entry, ratio = match_catalogue(text, catalogue)
            # score = how much confident text was read, plus a big bonus for a catalogue hit
            score = sum(len(norm(t)) * c / 100 for t, c in words if c >= 40 and len(norm(t)) >= 2)
            score += 100 * ratio if entry else 0
            if best is None or score > best["score"]:
                best = {"text": text, "rotation": rot, "score": score, "ratio": ratio,
                        "match": entry and {k: entry[k] for k in ("key", "manufacturer", "description")}}
            if ratio >= 1.0:
                return best                              # exact catalogue hit: done
    return best


def read_ics(img, parts, catalogue, max_ics=6):
    """Read the markings on the largest detected ICs of a board photo."""
    def area(p):
        return (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1])
    ics = [p for p in parts if p["name"] == "IC"
           and min(p["box"][2] - p["box"][0], p["box"][3] - p["box"][1]) >= MIN_SIDE]
    rgb = img.convert("RGB")
    return [{**read_ic(rgb.crop(tuple(round(v) for v in p["box"])), catalogue), "box": p["box"]}
            for p in sorted(ics, key=area, reverse=True)[:max_ics]]


# =====================================================================
# 4. BOM - parts list built from the detector and the chip OCR
# =====================================================================

def build_bom(parts, chips=()):
    """One line per identified chip (part number, maker, function), then one
    line per part type with its count. It lists what the AI found in the photo:
    the detector misses about half of the smallest parts, so the quantities are
    a starting point to check, not a complete BOM."""
    counts = pd.Series([p["name"] for p in parts], dtype=str).value_counts().to_dict()
    found = [c["match"] for c in chips if c.get("match")]
    rows = []
    for key in dict.fromkeys(m["key"] for m in found):           # unique, in order
        m = next(x for x in found if x["key"] == key)
        rows.append({"Part type": "IC", "Part number": key, "Manufacturer": m["manufacturer"],
                     "Description": m["description"], "Qty": sum(x["key"] == key for x in found)})
    if "IC" in counts:
        counts["IC"] -= len(found)
    order = ["IC", "Transistor/Diode", "Connector", "Switch", "Inductor", "Capacitor", "Resistor"]
    for name in sorted(counts, key=order.index):
        if counts[name] > 0:
            rows.append({"Part type": name, "Part number": "(not identified)" if name == "IC" and found else "",
                         "Manufacturer": "", "Description": "", "Qty": int(counts[name])})
    return pd.DataFrame(rows, columns=["Part type", "Part number", "Manufacturer", "Description", "Qty"])


# =====================================================================
# 5. EXPLANATION - why the Random Forest gave this prediction
# =====================================================================

FEATURES = ["board_type", "fault_category", "age_years", "previous_repairs"]


def contributions(pipe, X):
    """Split one Random Forest prediction into a share per input.

    A forest is many decision trees. Each tree sends the job down a path of
    yes/no questions; every question moves that tree's estimate from the parent
    node's value to the child's, and we credit that change to the input the
    question was about. Averaged over all trees:

        prediction = starting point + sum of the inputs' contributions

    exactly, with no approximation.

    pipe: Pipeline(prep, RandomForest). X: one-row DataFrame.
    Returns (starting point, {input: contribution}, prediction).
    """
    prep, forest = pipe.named_steps["prep"], pipe.named_steps["model"]
    classifier = hasattr(forest, "predict_proba")
    row = prep.transform(X)
    row = row.toarray() if hasattr(row, "toarray") else np.asarray(row)
    # one-hot columns such as 'cat__board_type_RF-COMM' belong to the input 'board_type'
    owner = [next(f for f in FEATURES if n.split("__", 1)[1].startswith(f))
             for n in prep.get_feature_names_out()]
    contrib = dict.fromkeys(FEATURES, 0.0)
    start = 0.0
    for est in forest.estimators_:
        t = est.tree_
        val = t.value[:, 0, :]
        # classifier: probability of BER at each node; regressor: mean hours at each node
        val = (val / val.sum(1, keepdims=True))[:, 1] if classifier else val[:, 0]
        node = 0
        start += val[0]
        while t.children_left[node] != -1:                       # walk down to a leaf
            f = t.feature[node]
            child = t.children_left[node] if row[0, f] <= t.threshold[node] else t.children_right[node]
            contrib[owner[f]] += val[child] - val[node]
            node = child
    n = len(forest.estimators_)
    contrib = {k: float(v / n) for k, v in contrib.items()}
    return float(start / n), contrib, float(start / n + sum(contrib.values()))
