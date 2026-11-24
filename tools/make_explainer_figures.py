"""Figures for docs/TRACKS_EXPLAINED.md (Track A results and the Track B idea).

Needs matplotlib, onnxruntime and pytesseract (any env that has them; the app's
own venv deliberately doesn't ship matplotlib).

  python tools/make_explainer_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from detector_utils import tile_starts  # noqa: E402
from ic_ocr import VARIANTS, load_catalogue, prepare, read_ic  # noqa: E402

OUT = ROOT / "docs" / "figures"
SAMPLES = ROOT / "sample_images"
BLUE, ORANGE, GREEN, RED, GREY = "#2a78d6", "#e8833a", "#2e9e5b", "#d64541", "#6b7280"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT / name)


# ---------------------------------------------------------------- Track A
def fig_tiles(board="board_8601"):
    """How the app cuts a photo into tiles (zoom 2, 1024 px tiles, 25% overlap)."""
    img = Image.open(SAMPLES / f"{board}.jpg").convert("RGB")
    w, h = img.size
    f = 2 * 1024 / max(w, h)                       # original px -> enlarged px
    W, H = round(w * f), round(h * f)
    xs, ys = tile_starts(W, 1024, 768), tile_starts(H, 1024, 768)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1, 1]})
    a = axes[0]
    a.imshow(img)
    a.set_title(f"1. Single pass: the whole {w}×{h} photo\nis squeezed into one 1024 px input", loc="left")
    a.axis("off")
    a = axes[1]
    a.imshow(img)
    cols = [BLUE, ORANGE, GREEN, RED, "#8e44ad", "#16a085"]
    k = 0
    for y in ys:
        for x in xs:
            c = cols[k % len(cols)]
            a.add_patch(Rectangle((x / f, y / f), 1024 / f, 1024 / f, fill=False, lw=2.5, ec=c))
            a.text(x / f + 12, y / f + 40, f"tile {k + 1}", color="white", fontsize=10,
                   bbox=dict(fc=c, ec="none", pad=2))
            k += 1
    a.set_title(f"2. Tiled: photo enlarged 2× to {W}×{H}, cut into {k} overlapping\n"
                "1024 px tiles (25% overlap), each detected separately", loc="left")
    a.axis("off")
    save(fig, "a1_tiling_grid.png")


def fig_what_model_sees(board="board_1686", box=(300, 300, 460, 420)):
    """Same patch of board at the resolution each mode gives the network."""
    img = Image.open(SAMPLES / f"{board}.jpg").convert("RGB")
    w, h = img.size
    crop = img.crop(box)
    single = 1024 / max(w, h)                      # scale in a single pass
    tiled = 2 * 1024 / max(w, h)                   # scale inside a tile
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    full = axes[0]
    full.imshow(img)
    full.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, ec=RED, lw=3))
    full.set_title("Zybo demo board\n(red box shown zoomed)", loc="left")
    full.axis("off")
    for a, s, name in ((axes[1], single, "Single pass"), (axes[2], tiled, "Tiled")):
        small = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.BILINEAR)
        a.imshow(small.resize(crop.size, Image.NEAREST))     # blow back up so pixels show
        a.set_title(f"{name}: what the network gets\n({small.width}×{small.height} px for this area)", loc="left")
        a.axis("off")
    fig.suptitle("A tiny resistor is a few blurry pixels in a single pass; tiling gives it about twice the pixels "
                 "in each direction", x=0.01, ha="left", fontsize=12)
    save(fig, "a2_what_the_model_sees.png")


def fig_results():
    ev = json.loads((ROOT / "models" / "tiling_eval.json").read_text())["results"]

    def pick(model, mode, zoom=None):
        return next(r for r in ev if r["model"].startswith(model) and r["mode"] == mode
                    and (zoom is None or r.get("zoom") == zoom))
    rows = [("Old model\nsingle pass", pick("YOLOv8n", "single")),
            ("Old model\ntiled (step 1)", pick("YOLOv8n", "tiled", 1.5)),
            ("New model\ntiled (step 2)", pick("YOLOv8s", "tiled", 2.0))]
    metrics = [("mAP@50\n(5 common types)", "map50_common"), ("Resistor AP", None),
               ("Capacitor AP", None), ("Recall", "recall@0.30"), ("Precision", "precision@0.30")]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    x = np.arange(len(metrics))
    width = 0.26
    for i, ((label, r), c) in enumerate(zip(rows, [GREY, ORANGE, BLUE])):
        vals = [r["per_class_ap50"]["Resistor"] if m == "Resistor AP" else
                r["per_class_ap50"]["Capacitor"] if m == "Capacitor AP" else r[k] for m, k in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, color=c, label=label.replace("\n", " "))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, [m for m, _ in metrics])
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("score (higher is better)")
    ax.legend(frameon=False, loc="upper left", ncol=3)
    ax.set_title("Track A on the 22 unseen test boards (4,709 hand-labelled parts)", loc="left")
    save(fig, "a3_detector_results.png")


def fig_chip_ocr(board="board_8601", box=(816, 176, 1005, 365)):
    img = Image.open(SAMPLES / f"{board}.jpg").convert("RGB")
    crop = img.crop(box)
    cat = load_catalogue()
    r = read_ic(crop, cat)
    prep = prepare(crop, **VARIANTS[0])
    fig = plt.figure(figsize=(14, 4.4))
    gs = fig.add_gridspec(2, 7, width_ratios=[1.3, 1.3, 0.55, 0.55, 0.55, 0.55, 1.9])
    a = fig.add_subplot(gs[:, 0]); a.imshow(img); a.axis("off")
    a.add_patch(Rectangle(box[:2], box[2] - box[0], box[3] - box[1], fill=False, ec=RED, lw=3))
    a.set_title("1. Detector finds\nthe big chips", loc="left")
    a = fig.add_subplot(gs[:, 1]); a.imshow(crop); a.axis("off")
    a.set_title("2. Crop: grey laser\netching on black", loc="left")
    for i, rot in enumerate((0, 90, 180, 270)):
        a = fig.add_subplot(gs[:, 2 + i])
        a.imshow(prep.rotate(rot, expand=True, fillcolor=255), cmap="gray"); a.axis("off")
        a.set_title(("3. Enlarge, contrast,\nsharpen, invert;\nread at 4 rotations\n" if i == 0 else "\n\n\n")
                    + f"{rot}°", loc="left", fontsize=10, color=GREEN if rot == r["rotation"] else "black")
    a = fig.add_subplot(gs[:, 6]); a.axis("off")
    text = "\n".join(r["lines"][:5])
    m = r["match"]
    a.text(0, 0.98, "4. OCR text (best rotation):", va="top", fontsize=11)
    a.text(0, 0.86, text, va="top", family="monospace", fontsize=9,
           bbox=dict(fc="#f3f4f6", ec="none", pad=6))
    a.text(0, 0.28, "5. Matched in the parts catalogue:", va="top", fontsize=11)
    a.text(0, 0.17, f"{m['key']} · {m['manufacturer']}\n{m['description']}" if m else "Not identified",
           va="top", fontsize=11, color=GREEN, weight="bold")
    save(fig, "a4_chip_ocr_steps.png")


# ---------------------------------------------------------------- Track B (concept)
def box(ax, xy, wh, text, fc="#eef4fc", ec=BLUE, fs=10.5):
    ax.add_patch(FancyBboxPatch(xy, *wh, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec=ec, lw=1.5))
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.5, color=GREY))


def fig_golden_concept(board="board_1686"):
    """Diagram of PatchCore. The 'faulty board' is a demo photo with a part digitally
    erased, and the heat map is drawn by hand at that spot: it illustrates the idea
    and is NOT model output."""
    img = Image.open(SAMPLES / f"{board}.jpg").convert("RGB").resize((560, 373))
    arr = np.asarray(img).copy()
    y0, y1, x0, x1 = 250, 299, 114, 141               # erase a slide switch: paint bare board colour over it
    arr[y0:y1, x0:x1] = np.median(arr[y0 - 22:y0 - 6, x0:x1].reshape(-1, 3), 0)
    faulty = Image.fromarray(arr)
    yy, xx = np.mgrid[0:373, 0:560]
    heat = np.exp(-(((xx - (x0 + x1) / 2) / 28) ** 2 + ((yy - (y0 + y1) / 2) / 24) ** 2))
    heat += 0.08 * np.random.RandomState(0).rand(373, 560)

    fig = plt.figure(figsize=(14, 7.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")
    ax.text(0.2, 7.3, "Training: show it ONLY good boards of one design", fontsize=13, weight="bold")
    for i in range(3):
        sub = fig.add_axes([0.015 + i * 0.012, 0.66 - i * 0.025, 0.16, 0.2]); sub.imshow(img); sub.axis("off")
    ax.text(0.2, 6.75, "hundreds of photos of GOOD boards", fontsize=10.5, color=GREEN)
    arrow(ax, (3.0, 6.0), (3.7, 6.0))
    box(ax, (3.75, 5.45), (2.3, 1.1), "Pretrained CNN\n(ResNet-18, ImageNet)\ncuts each photo\ninto patch features")
    arrow(ax, (6.1, 6.0), (6.8, 6.0))
    box(ax, (6.85, 5.25), (3.0, 1.5), "Memory bank\n'what every patch of a\ngood board looks like'\n(keep a representative 1-10%)",
        fc="#eaf7ef", ec=GREEN)
    ax.text(10.2, 6.2, "No fault labels needed:\nit never sees a broken board\nwhile learning.", fontsize=10.5,
            color=GREY)

    ax.plot([0.2, 13.8], [4.45, 4.45], color="#d1d5db", lw=1)
    ax.text(0.2, 4.1, "Inspection: a new board of the same design", fontsize=13, weight="bold")
    sub = fig.add_axes([0.015, 0.1, 0.2, 0.37]); sub.imshow(faulty); sub.axis("off")
    sub.set_title("new board\n(one slide switch is missing)", fontsize=10, loc="left")
    arrow(ax, (3.1, 2.0), (3.7, 2.0))
    box(ax, (3.75, 1.45), (2.3, 1.1), "Same CNN:\npatch features")
    arrow(ax, (6.1, 2.0), (6.8, 2.0))
    box(ax, (6.85, 1.2), (3.0, 1.6), "For every patch: distance\nto its NEAREST good patch\nin the memory bank\n(small = normal, big = odd)",
        fc="#eaf7ef", ec=GREEN)
    arrow(ax, (9.9, 2.0), (10.3, 2.0))
    sub = fig.add_axes([0.745, 0.1, 0.24, 0.37])
    sub.imshow(faulty); sub.imshow(heat, cmap="jet", alpha=0.45); sub.axis("off")
    sub.set_title("heat map: hottest patch = defect\nboard score = hottest patch → OK / DEFECT",
                  fontsize=10, loc="left")
    ax.text(0.2, 0.15, "Illustration only: the missing part was erased digitally and the heat map is drawn by hand "
            "to show the idea. Real results come after training on the VisA PCB images.",
            fontsize=9.5, color=RED, style="italic")
    save(fig, "b1_golden_board_idea.png")


def fig_a_vs_b():
    fig, ax = plt.subplots(figsize=(13, 3.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
    box(ax, (0.2, 0.3), (6.0, 3.0),
        "TRACK A: object detection (done)\n\nQuestion: \"WHAT parts are on this board?\"\n"
        "Learns from: photos with every part boxed by hand\n"
        "Output: boxes + part type (IC, resistor, ...)\n+ chip part numbers by OCR\n"
        "Works on: any board", fc="#eef4fc", ec=BLUE, fs=11)
    box(ax, (6.8, 0.3), (6.0, 3.0),
        "TRACK B: anomaly detection (proposed)\n\nQuestion: \"Does this board look WRONG?\"\n"
        "Learns from: good boards only, no labels\n"
        "Output: defect score + heat map of where\n(missing part, scratch, burn, bent pin)\n"
        "Works on: one board design per model", fc="#eaf7ef", ec=GREEN, fs=11)
    save(fig, "b0_track_a_vs_b.png")


if __name__ == "__main__":
    fig_tiles()
    fig_what_model_sees()
    fig_results()
    fig_chip_ocr()
    fig_a_vs_b()
    fig_golden_concept()
