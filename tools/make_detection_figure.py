"""Before/after figure for the briefing: previous model (single pass) vs the
current model (tiled), on the same demo board.

  python tools/make_detection_figure.py OLD_ONNX OLD_JSON [--board board_1686]
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detector_utils import PartDetector  # noqa: E402


def panel(det, img, parts, title, width):
    im, _ = det.draw(img, parts)
    im = im.resize((width, round(im.height * width / im.width)))
    bar = 46
    out = Image.new("RGB", (im.width, im.height + bar), "white")
    out.paste(im, (0, bar))
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(out).text((10, 10), title, fill="black", font=font)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old_onnx")
    ap.add_argument("old_json")
    ap.add_argument("--board", default="board_1686")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "tiling_before_after.jpg")
    a = ap.parse_args()
    img = Image.open(ROOT / "sample_images" / f"{a.board}.jpg").convert("RGB")
    old, new = PartDetector(a.old_onnx, a.old_json), PartDetector()
    t = new.meta.get("tiling")                      # same rule as app.tile_zoom
    zoom = t["long_side"] / new.size if t else 1.5
    p_old, p_new = old.detect(img), new.detect_tiled(img, zoom=zoom)
    left = panel(old, img, p_old, f"Before: previous model, single pass ({len(p_old)} parts)", 900)
    right = panel(new, img, p_new, f"Now: retrained on tiles, tiled ({len(p_new)} parts)", 900)
    fig = Image.new("RGB", (left.width + right.width + 20, max(left.height, right.height)), "white")
    fig.paste(left, (0, 0))
    fig.paste(right, (left.width + 20, 0))
    fig.save(a.out, quality=88)
    print(a.out, len(p_old), len(p_new))


if __name__ == "__main__":
    main()
