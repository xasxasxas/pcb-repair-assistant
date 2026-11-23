"""
Compare single-pass vs tiled (sliced) detection on the held-out test boards.

Uses the ORIGINAL full-resolution Roboflow 100 test photos (not the 1280 px
training copies), so tiling can use all the detail in the photo. Ground truth
is merged into the same 7 part types as training (prepare_dataset.MERGE).

  python detector/eval_tiling.py path/to/unzipped/rf100 [--out models/tiling_eval.json]

Metric: AP@50 per class (101-point interpolation, as in COCO / Ultralytics),
mAP@50 = mean over classes; plus precision / recall at the app's 0.30 threshold.
The absolute numbers differ slightly from Ultralytics' `val` (different
low-score cut-off and letterbox details); compare modes with each other.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from detector_utils import PartDetector  # noqa: E402
from prepare_dataset import CLASSES, MERGE, source_name, to_box  # noqa: E402

EVAL_SCORE = 0.05     # low cut-off for the precision-recall curve
APP_SCORE = 0.30      # what the app shows


def load_test_set(src: Path):
    names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    seen, items = set(), []
    for img_path in sorted((src / "test" / "images").glob("*.jpg")):
        board = source_name(img_path)
        if board in seen:            # Roboflow keeps augmented copies; one per board
            continue
        seen.add(board)
        w, h = Image.open(img_path).size
        gt = []
        for line in (src / "test" / "labels" / (img_path.stem + ".txt")).read_text().splitlines():
            if not line.strip():
                continue
            c, *box = line.split()
            merged = MERGE.get(names[int(c)])
            if merged is None:
                continue
            cx, cy, bw, bh = to_box(box)
            gt.append((CLASSES.index(merged), (cx - bw / 2) * w, (cy - bh / 2) * h,
                       (cx + bw / 2) * w, (cy + bh / 2) * h))
        items.append((board, img_path, gt))
    return items


def iou_one_to_many(b, boxes):
    if len(boxes) == 0:
        return np.zeros(0)
    boxes = np.asarray(boxes)
    iw = np.clip(np.minimum(b[2], boxes[:, 2]) - np.maximum(b[0], boxes[:, 0]), 0, None)
    ih = np.clip(np.minimum(b[3], boxes[:, 3]) - np.maximum(b[1], boxes[:, 1]), 0, None)
    inter = iw * ih
    union = (b[2] - b[0]) * (b[3] - b[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(union, 1e-9)


def evaluate(preds, items, nc, thr=0.5):
    """preds: list (per image) of detections. Returns per-class AP@50 and P/R at APP_SCORE."""
    ap, prec, rec = {}, {}, {}
    for c in range(nc):
        gts = {k: np.array([g[1:] for g in items[k][2] if g[0] == c]).reshape(-1, 4) for k in range(len(items))}
        npos = sum(len(v) for v in gts.values())
        cand = sorted(((d["score"], k, d["box"]) for k, ds in enumerate(preds) for d in ds
                       if d["class_id"] == c), key=lambda t: -t[0])
        used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
        tp = np.zeros(len(cand)); sc = np.array([t[0] for t in cand])
        for n, (_, k, box) in enumerate(cand):
            ious = iou_one_to_many(box, gts[k])
            if len(ious):
                j = int(np.argmax(np.where(used[k], -1, ious)))
                if ious[j] >= thr and not used[k][j]:
                    used[k][j] = True
                    tp[n] = 1
        if npos == 0:
            continue
        ctp = np.cumsum(tp)
        recall = ctp / npos
        precision = ctp / np.arange(1, len(cand) + 1) if len(cand) else np.zeros(0)
        # 101-point interpolated AP on the precision envelope
        mrec = np.concatenate([[0.0], recall, [1.0]])
        mpre = np.concatenate([[1.0], precision, [0.0]])
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        ap[CLASSES[c]] = round(float(np.trapezoid(np.interp(x, mrec, mpre), x)), 3)
        at = sc >= APP_SCORE
        prec[CLASSES[c]] = float(tp[at].sum() / max(at.sum(), 1))
        rec[CLASSES[c]] = float(tp[at].sum() / npos)
    return ap, prec, rec


def run(det, items, mode, **kw):
    preds, t0 = [], time.time()
    for _, path, _ in items:
        img = Image.open(path)
        preds.append(det.detect(img, score=EVAL_SCORE) if mode == "single"
                     else det.detect_tiled(img, score=EVAL_SCORE, **kw))
    secs = (time.time() - t0) / len(items)
    ap, p, r = evaluate(preds, items, len(CLASSES))
    n_app = sum(sum(d["score"] >= APP_SCORE for d in ds) for ds in preds) / len(items)
    counts = {CLASSES[c]: sum(g[0] == c for *_, gt in items for g in gt) for c in range(len(CLASSES))}
    common = [k for k in ap if counts[k] >= 100]          # rare classes (<100 test parts) make mAP noisy
    weighted = sum(ap[k] * counts[k] for k in ap) / sum(counts[k] for k in ap)
    return {"mode": mode, **kw, "map50": round(float(np.mean(list(ap.values()))), 3),
            "map50_common": round(float(np.mean([ap[k] for k in common])), 3),
            "ap50_part_weighted": round(float(weighted), 3), "per_class_ap50": ap,
            "precision@0.30": round(float(np.mean(list(p.values()))), 3),
            "recall@0.30": round(float(np.mean(list(r.values()))), 3),
            "parts_shown_per_board": round(n_app, 1), "seconds_per_board": round(secs, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rf100", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--zooms", default="1.5,2")
    ap.add_argument("--with-tile-only", action="store_true", help="also try tiles without the full-image pass")
    args = ap.parse_args()
    items = load_test_set(args.rf100)
    n_gt = sum(len(g) for *_, g in items)
    print(f"{len(items)} test boards, {n_gt} labelled parts")
    det = PartDetector()
    results = [run(det, items, "single")]
    print(json.dumps(results[-1]))
    for z in [float(v) for v in args.zooms.split(",")]:
        for full in ((True, False) if args.with_tile_only else (True,)):
            results.append(run(det, items, "tiled", zoom=z, full_pass=full))
            print(json.dumps(results[-1]))
    report = {"test_boards": len(items), "labelled_parts": n_gt, "results": results}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    print("\nmode            zoom full  mAP50  mAP50(common)  weighted  P@.30  R@.30  parts/board  s/board")
    for r in results:
        print(f"{r['mode']:<15} {r.get('zoom', 1):>4} {str(r.get('full_pass', '')):>5}  {r['map50']:.3f}  "
              f"{r['map50_common']:>13.3f}  {r['ap50_part_weighted']:>8.3f}  "
              f"{r['precision@0.30']:.3f}  {r['recall@0.30']:.3f}  {r['parts_shown_per_board']:>11}  {r['seconds_per_board']:>7}")


if __name__ == "__main__":
    main()
