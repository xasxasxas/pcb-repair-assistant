"""
PCB Repair Assistant - how the models were made and scored.

  python train.py data                    1. make the synthetic repair records (data/repair_records.csv)
  python train.py ml                      2. train + evaluate the two ML models (models/*.joblib, metrics.json)
  python train.py tiles RF100 OUT         3. cut the board photos into training tiles for the detector
  python train.py eval-detector RF100     4. score the detector on the 22 unseen test boards
  python train.py eval-chips              5. score chip-marking OCR on the hand-labelled chips

The detector itself is trained on a free Kaggle GPU with train_detector_kaggle.ipynb,
using the tiles from step 3. RF100 is the unzipped Roboflow 100 "printed circuit
board" dataset (CC BY 4.0). Steps 1 and 2 run in seconds on a laptop.
"""
import argparse
import collections
import json
import random
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
SEED = 42


# =====================================================================
# 1. SYNTHETIC REPAIR RECORDS
# =====================================================================
# Everything here is invented and modelled on general electronics-repair
# patterns; no real workshop data is used. Each record has a technician-style
# note, the true fault, and an outcome that depends on age, repair history,
# board type and fault, plus random noise (so the models cannot be perfect).

BOARD_TYPES = {                      # complexity scales repair time; ber_bias scales scrap risk
    "PSU-CTRL":    {"complexity": 1.0, "ber_bias": 0.0},
    "MCU-MAIN":    {"complexity": 1.4, "ber_bias": 0.3},
    "SENSOR-IF":   {"complexity": 1.1, "ber_bias": 0.1},
    "RF-COMM":     {"complexity": 1.8, "ber_bias": 0.7},
    "MOTOR-DRV":   {"complexity": 1.2, "ber_bias": 0.2},
    "DISPLAY-DRV": {"complexity": 1.3, "ber_bias": 0.2},
}
FAULTS = ["short_circuit", "open_circuit", "component_failure",
          "corrosion", "connector_fault", "power_supply"]
FAULT_HOURS = {"short_circuit": 2.5, "open_circuit": 2.0, "component_failure": 1.8,
               "corrosion": 4.0, "connector_fault": 1.2, "power_supply": 2.2}
FAULT_BER = {"short_circuit": 0.2, "open_circuit": 0.0, "component_failure": -0.2,
             "corrosion": 1.4, "connector_fault": -0.8, "power_supply": 0.0}
REFS = {"ic": ["U1", "U2", "U3", "U5", "U7", "U12"], "cap": ["C4", "C9", "C12", "C21", "C33"],
        "res": ["R7", "R15", "R22", "R40"], "fet": ["Q1", "Q2", "Q5"],
        "conn": ["J1", "J2", "J4", "P3"], "diode": ["D2", "D6", "D9"]}


def ref(kind):
    return random.choice(REFS[kind])


# Phrases a technician might write for each fault. Each lambda returns a fresh phrase.
PHRASES = {
    "short_circuit": [
        lambda: f"{ref('cap')} shorted to ground",
        lambda: "dead short on the 5V rail",
        lambda: "supply goes into current limit at power up",
        lambda: f"{ref('fet')} drain shorted to source",
        lambda: "solder bridge between adjacent pins",
        lambda: "near zero ohms between VCC and ground",
        lambda: f"{ref('ic')} running hot, internal short suspected",
        lambda: "burn mark under the shorted capacitor",
    ],
    "open_circuit": [
        lambda: "open track near the mounting hole",
        lambda: f"cracked solder joint on {ref('res')}",
        lambda: "no continuity on the trace to the connector",
        lambda: f"{ref('res')} measures open",
        lambda: "lifted pad, track broken",
        lambda: "via has no continuity between layers",
        lambda: "intermittent signal, hairline crack in track",
        lambda: "dry joint found under magnification",
    ],
    "component_failure": [
        lambda: f"{ref('ic')} failed, no output on the pin",
        lambda: f"electrolytic {ref('cap')} bulging, high ESR",
        lambda: f"{ref('diode')} failed open on diode test",
        lambda: "crystal oscillator not starting",
        lambda: f"op amp {ref('ic')} output stuck high",
        lambda: f"{ref('fet')} gate damaged, not switching",
        lambda: "relay coil open, relay not clicking",
        lambda: "optocoupler not responding",
    ],
    "corrosion": [
        lambda: "visible corrosion near the connector",
        lambda: "green deposits on the pins",
        lambda: "salt residue across the lower edge",
        lambda: "moisture ingress, tracks eaten away",
        lambda: "white crusty oxidation around the vias",
        lambda: "conformal coating peeled, corrosion underneath",
        lambda: f"corroded legs on {ref('ic')}",
        lambda: "board shows water damage and rust on the shield",
    ],
    "connector_fault": [
        lambda: f"connector {ref('conn')} pins bent",
        lambda: f"loose contact in {ref('conn')}",
        lambda: "cable harness intermittent at the plug",
        lambda: "connector housing cracked",
        lambda: f"pin 3 of {ref('conn')} pushed back",
        lambda: "header pins worn, poor mating",
        lambda: "ribbon cable socket latch broken",
        lambda: "contact resistance high on the plug",
    ],
    "power_supply": [
        lambda: "no output on the 5V rail",
        lambda: "3.3V regulator output low",
        lambda: f"regulator {ref('ic')} is hot, output drooping",
        lambda: "fuse blown on the input",
        lambda: "ripple on the supply too high",
        lambda: "switching regulator not starting up",
        lambda: "input voltage present but no rail out",
        lambda: "power LED off, supply dead",
    ],
}
# Neutral filler that appears in any note (makes the text task realistic)
FILLER_OPEN = ["board received from unit,", "checked on the bench,", "initial inspection done,",
               "unit reported fault,", "visual check complete,", ""]
FILLER_CLOSE = ["will replace and retest", "replaced and tested ok", "needs further testing",
                "parts ordered", "awaiting spare", "passes power up test now", ""]


def make_note(fault):
    """A technician-style note: 1-2 phrases for the true fault and, a quarter of
    the time, one phrase from another fault (realistic confusion)."""
    opener = random.choice(FILLER_OPEN)
    parts = [p() for p in random.sample(PHRASES[fault], k=random.choice([1, 2, 2]))]
    if random.random() < 0.25:
        other = random.choice(FAULTS)
        if other != fault:
            parts.insert(random.randint(0, len(parts)), random.choice(PHRASES[other])())
    closer = random.choice(FILLER_CLOSE)
    if closer:
        parts.append(closer)
    note = (opener + " " + ", ".join(parts)).strip() + "."
    return note[0].upper() + note[1:]


def make_records(n=600):
    """600 repair jobs. Fixed seed, so the file is always the same."""
    random.seed(SEED)
    np.random.seed(SEED)
    rows = []
    for i in range(n):
        board = random.choice(list(BOARD_TYPES))
        fault = random.choice(FAULTS)
        age = int(np.clip(np.random.gamma(3, 2.5), 1, 25))
        prev = int(np.clip(np.random.poisson(0.15 * age), 0, 8))
        info = BOARD_TYPES[board]
        # scrap risk rises with age, repair history, corrosion and complex boards
        logit = (-3.2 + 0.16 * age + 0.45 * prev + FAULT_BER[fault]
                 + info["ber_bias"] + np.random.normal(0, 0.6))
        ber = np.random.rand() < 1 / (1 + np.exp(-logit))
        hours = (FAULT_HOURS[fault] * info["complexity"] + 0.05 * age
                 + 0.3 * prev + np.random.normal(0, 0.5))
        rows.append({"job_id": f"J{1000 + i}", "board_type": board, "age_years": age,
                     "previous_repairs": prev, "note": make_note(fault), "fault_category": fault,
                     "outcome": "BER" if ber else "Repairable",
                     "repair_hours": round(float(np.clip(hours, 0.5, 12)), 1)})
    return pd.DataFrame(rows)


# =====================================================================
# 2. THE TWO ML MODELS
# =====================================================================
# Model 1, fault classifier: note text -> fault category (TF-IDF + Logistic Regression).
# Model 2, repair predictor: board type, age, previous repairs, fault ->
#          BER risk (Random Forest classifier) and repair hours (Random Forest regressor).
# Both are scored on a 20% test split the model never saw during training.

TAB_FEATURES = ["board_type", "fault_category", "age_years", "previous_repairs"]
# Filler words with no fault information. "no" and "not" are kept on purpose:
# "no continuity" and "not responding" matter.
STOP_WORDS = ["a", "an", "the", "is", "are", "was", "on", "to", "of", "and", "in", "at", "near",
              "from", "with", "for", "it", "this", "will", "be", "now", "done", "board", "unit",
              "checked", "bench"]


def tfidf():
    """Words and word pairs, weighted high when frequent in this note but rare overall."""
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words=STOP_WORDS)


def train_fault_classifier(df, out):
    X, y = df["note"], df["fault_category"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # compare three algorithms with 5-fold cross-validation on the training part
    comparison = []
    for name, clf in {"Naive Bayes": MultinomialNB(),
                      "Logistic Regression": LogisticRegression(max_iter=2000, C=5),
                      "Linear SVM": LinearSVC(C=1)}.items():
        pipe = Pipeline([("tfidf", tfidf()), ("clf", clf)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1_macro")
        pred = pipe.fit(X_tr, y_tr).predict(X_te)
        comparison.append({"model": name,
                           "cv_f1_mean": round(float(cv.mean()), 3), "cv_f1_std": round(float(cv.std()), 3),
                           "test_accuracy": round(float(accuracy_score(y_te, pred)), 3),
                           "test_f1_macro": round(float(f1_score(y_te, pred, average="macro")), 3)})

    # keep Logistic Regression: it gives probabilities and per-word weights that explain each prediction
    final = Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=2000, C=5))])
    final.fit(X_tr, y_tr)
    labels = list(final.classes_)
    cm = confusion_matrix(y_te, final.predict(X_te), labels=labels)
    vocab = np.array(final.named_steps["tfidf"].get_feature_names_out())
    coefs = final.named_steps["clf"].coef_
    keywords = {lab: vocab[np.argsort(coefs[i])[::-1][:8]].tolist() for i, lab in enumerate(labels)}

    final.fit(X, y)                                  # refit on all data for the app
    joblib.dump(final, out / "fault_classifier.joblib")
    return {"comparison": comparison, "chosen": "Logistic Regression", "labels": labels,
            "confusion_matrix": cm.tolist(), "test_size": int(len(y_te)), "top_keywords": keywords}


def tabular_preprocessor():
    """Board type and fault become one-hot columns; age and repair count pass through."""
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["board_type", "fault_category"]),
                              ("num", "passthrough", ["age_years", "previous_repairs"])])


def grouped_importance(pipe):
    """Feature importance, with the one-hot columns summed back into the 4 inputs."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    out = dict.fromkeys(TAB_FEATURES, 0.0)
    for n, v in zip(names, pipe.named_steps["model"].feature_importances_):
        out[next(f for f in TAB_FEATURES if f in n)] += float(v)
    return {k: round(v, 3) for k, v in sorted(out.items(), key=lambda x: -x[1])}


def train_outcome_models(df, out):
    X = df[TAB_FEATURES]
    y_cls = (df["outcome"] == "BER").astype(int)
    y_reg = df["repair_hours"]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_cls, y_reg, test_size=0.2, stratify=y_cls, random_state=SEED)

    # a) Repairable vs BER. Baseline to beat: always answer "Repairable".
    clf = Pipeline([("prep", tabular_preprocessor()),
                    ("model", RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                                     class_weight="balanced", random_state=SEED))])
    cv = cross_val_score(clf, X_tr, yc_tr, cv=5, scoring="roc_auc")
    clf.fit(X_tr, yc_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, yc_tr)

    # b) Repair hours. Baseline to beat: always answer the average.
    reg = Pipeline([("prep", tabular_preprocessor()),
                    ("model", RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                                    random_state=SEED))])
    reg.fit(X_tr, yr_tr)
    hours_pred = reg.predict(X_te)
    base_reg = DummyRegressor().fit(X_tr, yr_tr)

    metrics = {
        "outcome": {
            "cv_roc_auc_mean": round(float(cv.mean()), 3), "cv_roc_auc_std": round(float(cv.std()), 3),
            "test_accuracy": round(float(accuracy_score(yc_te, pred)), 3),
            "test_f1_ber": round(float(f1_score(yc_te, pred)), 3),
            "test_roc_auc": round(float(roc_auc_score(yc_te, proba)), 3),
            "baseline_accuracy": round(float(base.score(X_te, yc_te)), 3),
            "confusion_matrix": confusion_matrix(yc_te, pred).tolist(),
            "labels": ["Repairable", "BER"],
            "feature_importance": grouped_importance(clf),
        },
        "hours": {
            "test_mae": round(float(mean_absolute_error(yr_te, hours_pred)), 2),
            "test_r2": round(float(r2_score(yr_te, hours_pred)), 3),
            "baseline_mae": round(float(mean_absolute_error(yr_te, base_reg.predict(X_te))), 2),
            "feature_importance": grouped_importance(reg),
        },
        "ber_rate": round(float(y_cls.mean()), 3),
    }
    joblib.dump(clf.fit(X, y_cls), out / "outcome_model.joblib")     # refit on all data for the app
    joblib.dump(reg.fit(X, y_reg), out / "hours_model.joblib")
    return metrics


def train_ml(out=MODELS):
    import sklearn
    out.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "repair_records.csv")
    fault = train_fault_classifier(df, out)
    outcome = train_outcome_models(df, out)
    metrics = {"n_records": int(len(df)), "sklearn_version": sklearn.__version__,
               "fault_classifier": fault, **outcome}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# =====================================================================
# 3. TRAINING TILES FOR THE PART DETECTOR
# =====================================================================
# The Roboflow labels have 23 names; we merge them into 7 part types and drop
# non-parts (pads, test points, "unknown"). Roboflow's split is by board, so no
# board appears in both training and testing.

MERGE = {
    "IC": "IC", "iC": "IC", "Clock": "IC",
    "Capacitor": "Capacitor", "Electrolytic Capacitor": "Capacitor", "Capacitor Jumper": "Capacitor",
    "Resistor": "Resistor", "Resistor Network": "Resistor", "Resistor Jumper": "Resistor",
    "Connector": "Connector", "Pins": "Connector", "Jumper": "Connector",
    "Transistor": "Transistor/Diode", "Diode": "Transistor/Diode", "Led": "Transistor/Diode",
    "Inductor": "Inductor", "Ferrite Bead": "Inductor",
    "Button": "Switch", "Switch": "Switch",
}
CLASSES = ["Capacitor", "Connector", "IC", "Inductor", "Resistor", "Switch", "Transistor/Diode"]
KEEP_FRAC = 0.6              # keep a label in a tile if at least 60% of the part is inside it
BACKGROUND_KEEP = 0.15       # share of empty tiles kept as "nothing here" examples


def load_boards(src, split):
    """One (board name, image path, labelled parts) per photographed board.
    Parts are (class id, x1, y1, x2, y2) in pixels. Roboflow stores several
    augmented copies of each photo; only the first is used."""
    import re
    import yaml
    names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    seen, boards = set(), []
    for img_path in sorted((src / split / "images").glob("*.jpg")):
        board = re.sub(r"\.rf\.[0-9a-f]+$", "", img_path.stem)
        if board in seen:
            continue
        seen.add(board)
        w, h = Image.open(img_path).size
        parts = []
        for line in (src / split / "labels" / (img_path.stem + ".txt")).read_text().splitlines():
            if not line.strip():
                continue
            c, *v = line.split()
            merged = MERGE.get(names[int(c)])
            if merged is None:
                continue
            v = [float(x) for x in v]
            if len(v) > 4:                           # a polygon: use its bounding box
                xs, ys = v[0::2], v[1::2]
                v = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys)]
            cx, cy, bw, bh = v
            parts.append((CLASSES.index(merged), (cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h))
        boards.append((board, img_path, parts))
    return boards


def make_tiles(src, out, long_side=2048, tile=1024, overlap=0.25):
    """Resize each photo to `long_side`, cut it into overlapping tiles and write
    them as a YOLO dataset (plus a zip with '/' paths, which Kaggle needs)."""
    import yaml
    import zipfile
    from pcb_ai import tile_starts
    random.seed(0)
    if out.exists():
        shutil.rmtree(out)
    stride = int(tile * (1 - overlap))
    stats = {"boards": {}, "tiles": {}, "boxes": collections.Counter()}
    for split in ["train", "valid", "test"]:
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        boards, n_tiles = load_boards(src, split), 0
        for board, img_path, parts in boards:
            img = Image.open(img_path).convert("RGB")
            f = long_side / max(img.size)
            W, H = round(img.width * f), round(img.height * f)
            big = img.resize((W, H), Image.LANCZOS)
            for y0 in tile_starts(H, tile, stride):
                for x0 in tile_starts(W, tile, stride):
                    tw, th = min(tile, W - x0), min(tile, H - y0)
                    lines = []
                    for c, x1, y1, x2, y2 in parts:
                        x1, y1, x2, y2 = x1 * f - x0, y1 * f - y0, x2 * f - x0, y2 * f - y0
                        area = max(1e-6, (x2 - x1) * (y2 - y1))
                        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(tw, x2), min(th, y2)
                        if cx2 <= cx1 or cy2 <= cy1 or (cx2 - cx1) * (cy2 - cy1) / area < KEEP_FRAC:
                            continue
                        lines.append(f"{c} {(cx1 + cx2) / 2 / tile:.6f} {(cy1 + cy2) / 2 / tile:.6f} "
                                     f"{(cx2 - cx1) / tile:.6f} {(cy2 - cy1) / tile:.6f}")
                        stats["boxes"][CLASSES[c]] += 1
                    if not lines and random.random() > BACKGROUND_KEEP:
                        continue
                    canvas = Image.new("RGB", (tile, tile), (114, 114, 114))
                    canvas.paste(big.crop((x0, y0, x0 + tw, y0 + th)), (0, 0))
                    canvas.save(out / split / "images" / f"{board}_{x0}_{y0}.jpg", quality=92)
                    (out / split / "labels" / f"{board}_{x0}_{y0}.txt").write_text(
                        "\n".join(lines) + ("\n" if lines else ""))
                    n_tiles += 1
        stats["boards"][split], stats["tiles"][split] = len(boards), n_tiles
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "train/images", "val": "valid/images",
         "test": "test/images", "names": dict(enumerate(CLASSES))}, sort_keys=False))
    summary = {**stats, "boxes": dict(stats["boxes"]), "long_side": long_side, "tile": tile, "overlap": overlap,
               "source": "Roboflow 100 printed-circuit-board v4 (CC BY 4.0)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with zipfile.ZipFile(out.with_suffix(".zip"), "w", zipfile.ZIP_STORED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    return summary


# =====================================================================
# 4. SCORING THE DETECTOR ON WHOLE TEST BOARDS
# =====================================================================
# AP@50 (average precision): for one part type, sort all predicted boxes by
# confidence and trace precision against recall; AP is the area under that
# curve. A box counts as correct if it overlaps an unmatched labelled part of
# the same type by IoU >= 0.5. mAP@50 is the mean over part types.

APP_SCORE = 0.30             # the confidence threshold the app shows boxes at


def iou_one_to_many(b, boxes):
    if len(boxes) == 0:
        return np.zeros(0)
    iw = np.clip(np.minimum(b[2], boxes[:, 2]) - np.maximum(b[0], boxes[:, 0]), 0, None)
    ih = np.clip(np.minimum(b[3], boxes[:, 3]) - np.maximum(b[1], boxes[:, 1]), 0, None)
    union = (b[2] - b[0]) * (b[3] - b[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - iw * ih
    return iw * ih / np.maximum(union, 1e-9)


def score_detections(preds, boards):
    """preds: per board, the detector's output. Returns per-type AP@50, and
    precision / recall at the app's confidence threshold."""
    ap, prec, rec = {}, {}, {}
    for c, name in enumerate(CLASSES):
        gts = [np.array([g[1:] for g in parts if g[0] == c]).reshape(-1, 4) for _, _, parts in boards]
        npos = sum(len(g) for g in gts)
        if npos == 0:
            continue
        cand = sorted(((d["score"], k, d["box"]) for k, ds in enumerate(preds) for d in ds
                       if d["class_id"] == c), key=lambda t: -t[0])
        used = [np.zeros(len(g), bool) for g in gts]
        tp = np.zeros(len(cand))
        for n, (_, k, box) in enumerate(cand):                   # most confident first
            ious = iou_one_to_many(box, gts[k])
            if len(ious):
                j = int(np.argmax(np.where(used[k], -1, ious)))
                if ious[j] >= 0.5 and not used[k][j]:
                    used[k][j] = True
                    tp[n] = 1
        ctp = np.cumsum(tp)
        recall = ctp / npos
        precision = ctp / np.arange(1, len(cand) + 1) if len(cand) else np.zeros(0)
        # area under the precision-recall curve (101-point interpolation, as in COCO)
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate([[1.0], precision, [0.0]]))))
        x = np.linspace(0, 1, 101)
        ap[name] = round(float(np.trapezoid(np.interp(x, mrec, mpre), x)), 3)
        shown = np.array([t[0] for t in cand]) >= APP_SCORE
        prec[name] = float(tp[shown].sum() / max(shown.sum(), 1))
        rec[name] = float(tp[shown].sum() / npos)
    return ap, prec, rec


def eval_detector(src, zooms=(1.5, 2.0, 2.5)):
    """Single pass vs tiled detection on the unseen test boards (full-resolution photos)."""
    from pcb_ai import PartDetector
    boards = load_boards(src, "test")
    counts = {n: sum(g[0] == c for _, _, parts in boards for g in parts) for c, n in enumerate(CLASSES)}
    det = PartDetector()
    results = []
    for mode, zoom in [("single", None)] + [("tiled", z) for z in zooms]:
        t0 = time.time()
        preds = [det.detect(Image.open(p), score=0.05) if mode == "single"
                 else det.detect_tiled(Image.open(p), zoom=zoom, score=0.05) for _, p, _ in boards]
        secs = (time.time() - t0) / len(boards)
        ap, prec, rec = score_detections(preds, boards)
        common = [k for k in ap if counts[k] >= 100]      # rare types (<100 test parts) are too noisy
        results.append({
            "mode": mode, **({"zoom": zoom} if zoom else {}),
            "map50": round(float(np.mean(list(ap.values()))), 3),
            "map50_common": round(float(np.mean([ap[k] for k in common])), 3),
            "ap50_part_weighted": round(sum(ap[k] * counts[k] for k in ap) / sum(counts[k] for k in ap), 3),
            "per_class_ap50": ap,
            "precision@0.30": round(float(np.mean(list(prec.values()))), 3),
            "recall@0.30": round(float(np.mean(list(rec.values()))), 3),
            "parts_shown_per_board": round(sum(sum(d["score"] >= APP_SCORE for d in ds) for ds in preds)
                                           / len(boards), 1),
            "seconds_per_board": round(secs, 2)})
    return {"test_boards": len(boards), "labelled_parts": sum(counts.values()), "results": results}


# =====================================================================
# 5. SCORING CHIP-MARKING OCR
# =====================================================================
# data/ic_markings.json holds 91 chip crops from the demo boards, read by eye:
# 36 "readable" (a person can read the part number) and 55 others (sticker,
# heatsink, unmarked, too small, or not a chip). The others test whether the
# reader invents a part when there is nothing to read.

def eval_chips():
    import pytesseract
    from pcb_ai import load_catalogue, match_catalogue, norm, read_ic
    items = json.loads((DATA / "ic_markings.json").read_text())["items"]
    catalogue = load_catalogue()

    def plain_tesseract(crop, catalogue):                # the baseline: no clean-up, no rotations
        text = pytesseract.image_to_string(crop, config="--psm 6")
        entry, _ = match_catalogue(text, catalogue)
        return {"text": text, "match": entry and {"key": entry["key"]}}

    def score(reader):
        res, t0 = [], time.time()
        for it in items:
            crop = Image.open(ROOT / "sample_images" / it["image"]).convert("RGB").crop(tuple(it["box"]))
            match = reader(crop, catalogue)["match"]
            key = norm(match["key"]) if match else None
            res.append({"image": it["image"], "kind": it["kind"], "expected": it["expected"],
                        "match": match and match["key"],
                        "identified": bool(key) and any(key in norm(e) or norm(e) in key for e in it["expected"])})
        readable = [x for x in res if x["kind"] == "readable"]
        other = [x for x in res if x["kind"] != "readable"]
        return {"readable_chips": len(readable),
                "part_identified": sum(x["identified"] for x in readable),
                "wrong_part_on_readable": sum(bool(x["match"]) and not x["identified"] for x in readable),
                "other_crops": len(other),
                "false_identifications": sum(bool(x["match"]) for x in other),
                "seconds_per_chip": round((time.time() - t0) / len(items), 2),
                "details": res}

    return {"pipeline": score(read_ic), "baseline_plain_tesseract": score(plain_tesseract)}


# =====================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["data", "ml", "tiles", "eval-detector", "eval-chips"])
    ap.add_argument("paths", nargs="*", type=Path, help="RF100 folder (and output folder for tiles)")
    args = ap.parse_args()

    if args.step == "data":
        df = make_records()
        df.to_csv(DATA / "repair_records.csv", index=False)
        print(f"wrote {len(df)} repair records")
        print(df["fault_category"].value_counts().to_string())
        print(df["outcome"].value_counts().to_string())

    elif args.step == "ml":
        m = train_ml()
        print(pd.DataFrame(m["fault_classifier"]["comparison"]).to_string(index=False))
        o, h = m["outcome"], m["hours"]
        print(f"\nBER:   accuracy {o['test_accuracy']} (baseline {o['baseline_accuracy']}), ROC AUC {o['test_roc_auc']}")
        print(f"Hours: error {h['test_mae']} h (baseline {h['baseline_mae']} h), R2 {h['test_r2']}")

    elif args.step == "tiles":
        print(json.dumps(make_tiles(args.paths[0], args.paths[1]), indent=2))

    elif args.step == "eval-detector":
        report = eval_detector(args.paths[0])
        print(f"{report['test_boards']} test boards, {report['labelled_parts']} labelled parts")
        print("mode    zoom  mAP50(5 common)  recall  precision  parts/board  s/board")
        for r in report["results"]:
            print(f"{r['mode']:<7} {r.get('zoom', ''):>4}  {r['map50_common']:>15.3f}  {r['recall@0.30']:.3f}  "
                  f"{r['precision@0.30']:>9.3f}  {r['parts_shown_per_board']:>11}  {r['seconds_per_board']:>7}")
        # models/tiling_eval.json keeps these rows next to the previous model's, for comparison

    elif args.step == "eval-chips":
        report = eval_chips()
        for name, r in report.items():
            print(f"{name:26} identified {r['part_identified']}/{r['readable_chips']}  wrong part "
                  f"{r['wrong_part_on_readable']}  false claims {r['false_identifications']}/{r['other_crops']}  "
                  f"{r['seconds_per_chip']} s/chip")
        (MODELS / "ic_ocr_eval.json").write_text(json.dumps(report, indent=1))
