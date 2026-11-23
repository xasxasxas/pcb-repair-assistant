"""
Build a TILED training set for the part detector (Track A).

Why: the single-pass detector shrinks a ~2000-2900 px board photo to 1024 px,
so small SMD resistors/capacitors end up a few pixels wide and are missed.
Training (and detecting) on tiles keeps parts at a usable pixel size.

What it does, per split (train / valid / test, same board-level split as
prepare_dataset.py, one photo per board):
  1. resize the original photo so its long side is LONG_SIDE px
     (the same scale the app uses at detection time: zoom = LONG_SIDE / imgsz)
  2. cut it into TILE x TILE tiles with OVERLAP overlap
  3. keep a label in a tile if at least KEEP_FRAC of its box lies inside the
     tile (the box is clipped to the tile); tiles with no parts are mostly
     dropped (a few kept as background examples)
  4. valid/ and test/ also keep the WHOLE photos in *_full/ folders, so the
     final score can be measured on whole boards with the app's tiled detector
     (detector/eval_tiling.py), not just on tiles

Usage:
  python detector/prepare_tiles.py path/to/unzipped/rf100 [out_dir] [--long-side 2048] [--tile 1024]
"""
import argparse
import collections
import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image

from prepare_dataset import CLASSES, MERGE, source_name, to_box

KEEP_FRAC = 0.6
BACKGROUND_KEEP = 0.15      # share of empty tiles kept as negatives


def tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    return starts + [length - tile]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path, nargs="?", default=Path(__file__).parent / "dataset_tiles")
    ap.add_argument("--long-side", type=int, default=2048)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip for Kaggle upload")
    args = ap.parse_args()
    random.seed(0)

    names = yaml.safe_load((args.src / "data.yaml").read_text())["names"]
    if args.out.exists():
        shutil.rmtree(args.out)
    T, stride = args.tile, int(args.tile * (1 - args.overlap))
    stats = {"boards": {}, "tiles": {}, "boxes": collections.Counter()}

    for split in ["train", "valid", "test"]:
        (args.out / split / "images").mkdir(parents=True)
        (args.out / split / "labels").mkdir(parents=True)
        seen, n_tiles = set(), 0
        for img_path in sorted((args.src / split / "images").glob("*.jpg")):
            board = source_name(img_path)
            if board in seen:
                continue
            seen.add(board)
            img = Image.open(img_path).convert("RGB")
            w0, h0 = img.size
            # boxes in original pixels
            boxes = []
            for line in (args.src / split / "labels" / (img_path.stem + ".txt")).read_text().splitlines():
                if not line.strip():
                    continue
                c, *b = line.split()
                merged = MERGE.get(names[int(c)])
                if merged is None:
                    continue
                cx, cy, bw, bh = to_box(b)
                boxes.append((CLASSES.index(merged), (cx - bw / 2) * w0, (cy - bh / 2) * h0,
                              (cx + bw / 2) * w0, (cy + bh / 2) * h0))
            if split != "train":
                full = args.out / f"{split}_full"
                (full / "images").mkdir(parents=True, exist_ok=True)
                (full / "labels").mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, full / "images" / f"{board}.jpg")
                shutil.copy(args.src / split / "labels" / (img_path.stem + ".txt"),
                            full / "labels" / f"{board}.txt")

            f = args.long_side / max(w0, h0)
            W, H = round(w0 * f), round(h0 * f)
            big = img.resize((W, H), Image.LANCZOS)
            for y0 in tile_starts(H, T, stride):
                for x0 in tile_starts(W, T, stride):
                    tw, th = min(T, W - x0), min(T, H - y0)
                    lines = []
                    for c, x1, y1, x2, y2 in boxes:
                        x1, y1, x2, y2 = x1 * f - x0, y1 * f - y0, x2 * f - x0, y2 * f - y0
                        area = max(1e-6, (x2 - x1) * (y2 - y1))
                        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(tw, x2), min(th, y2)
                        if cx2 <= cx1 or cy2 <= cy1:
                            continue
                        if (cx2 - cx1) * (cy2 - cy1) / area < KEEP_FRAC:
                            continue
                        # labels are normalised to the T x T tile (edge tiles are padded)
                        lines.append(f"{c} {(cx1 + cx2) / 2 / T:.6f} {(cy1 + cy2) / 2 / T:.6f} "
                                     f"{(cx2 - cx1) / T:.6f} {(cy2 - cy1) / T:.6f}")
                        stats["boxes"][CLASSES[c]] += 1
                    if not lines and random.random() > BACKGROUND_KEEP:
                        continue
                    tile = Image.new("RGB", (T, T), (114, 114, 114))
                    tile.paste(big.crop((x0, y0, x0 + tw, y0 + th)), (0, 0))
                    stem = f"{board}_{x0}_{y0}"
                    tile.save(args.out / split / "images" / f"{stem}.jpg", quality=92)
                    (args.out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                    n_tiles += 1
        stats["boards"][split] = len(seen)
        stats["tiles"][split] = n_tiles

    (args.out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(args.out.resolve()), "train": "train/images", "val": "valid/images",
        "test": "test/images", "names": dict(enumerate(CLASSES)),
    }, sort_keys=False))
    summary = {**stats, "boxes": dict(stats["boxes"]), "long_side": args.long_side, "tile": T,
               "overlap": args.overlap, "classes": CLASSES,
               "source": "Roboflow 100 printed-circuit-board v4 (CC BY 4.0)"}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if args.zip:
        write_zip(args.out, args.out.with_suffix(".zip"))


def write_zip(folder: Path, dest: Path):
    """Zip for upload to Kaggle. Paths inside use '/' (Windows' Compress-Archive
    writes '\\', which Kaggle rejects as a forbidden character)."""
    import zipfile
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder).as_posix())
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
