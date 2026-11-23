"""AI part 3 - Part detector: YOLOv8n (fine-tuned on open PCB photos), run
with ONNX Runtime on CPU. Same pre/post-processing as the Android app."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "parts.onnx"
META = ROOT / "models" / "parts.json"

# Fixed categorical colours, assigned by class name (alphabetical), never by rank
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


class PartDetector:
    def __init__(self, model=MODEL, meta=META):
        import onnxruntime as ort
        m = json.loads(Path(meta).read_text())
        self.names = m["names"]
        self.size = int(m.get("imgsz", 640))
        self.meta = m
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    @staticmethod
    def available():
        return MODEL.exists() and META.exists()

    def letterbox(self, img):
        w, h = img.size
        s = min(self.size / w, self.size / h)
        nw, nh = round(w * s), round(h * s)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (px, py))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, s, px, py

    def detect(self, img, score=0.30, iou=0.45):
        x, s, px, py = self.letterbox(img)
        out = self.session.run(None, {self.input: x})[0][0]      # (4 + nc, N)
        dets = decode(out, s, px, py, img.size[0], img.size[1], score, iou)
        for det in dets:
            det["name"] = self.names[det["class_id"]]
        return dets

    def detect_tiled(self, img, zoom=2.0, overlap=0.25, score=0.30, iou=0.45, full_pass=True):
        """Sliced inference for small parts.

        A single pass shrinks the whole photo to the model's input size, so a
        0402 resistor becomes a few pixels. Here the photo is enlarged `zoom`
        times relative to that single pass and cut into overlapping tiles of
        the model's input size; each tile is detected separately and the boxes
        are merged back into original-image coordinates.

        * boxes touching an inner tile edge are dropped (the part is cut off;
          a neighbouring tile sees it whole, because the overlap is larger
          than a typical small part)
        * `full_pass` adds the normal single-pass detections, which handle
          large parts (big ICs, connectors) that no single tile contains whole
        * duplicates are merged with class-wise NMS
        """
        img = img.convert("RGB")
        w, h = img.size
        f = zoom * self.size / max(w, h)                 # original px -> enlarged px
        W, H = max(1, round(w * f)), max(1, round(h * f))
        big = img.resize((W, H), Image.BILINEAR)
        T = self.size
        stride = max(1, int(T * (1 - overlap)))
        edge = 4                                          # px tolerance for "touches the edge"
        boxes, scores, cls = [], [], []
        for y0 in tile_starts(H, T, stride):
            for x0 in tile_starts(W, T, stride):
                tile = Image.new("RGB", (T, T), (114, 114, 114))
                tile.paste(big.crop((x0, y0, min(x0 + T, W), min(y0 + T, H))), (0, 0))
                x = np.asarray(tile, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
                out = self.session.run(None, {self.input: x})[0][0]
                for d in decode(out, 1.0, 0, 0, T, T, score, iou):
                    bx1, by1, bx2, by2 = d["box"]
                    # inner edges only: an edge that is also the image border is fine
                    if ((x0 > 0 and bx1 <= edge) or (y0 > 0 and by1 <= edge) or
                            (x0 + T < W and bx2 >= T - edge) or (y0 + T < H and by2 >= T - edge)):
                        continue
                    boxes.append((min((bx1 + x0) / f, w), min((by1 + y0) / f, h),
                                  min((bx2 + x0) / f, w), min((by2 + y0) / f, h)))
                    scores.append(d["score"]); cls.append(d["class_id"])
        if full_pass:
            for d in self.detect(img, score, iou):
                boxes.append(d["box"]); scores.append(d["score"]); cls.append(d["class_id"])
        if not boxes:
            return []
        boxes, scores, cls = np.array(boxes, float), np.array(scores, float), np.array(cls, int)
        keep = nms(boxes, scores, cls, iou)
        return [{"class_id": int(cls[i]), "score": float(scores[i]), "box": tuple(map(float, boxes[i])),
                 "name": self.names[int(cls[i])]} for i in keep]

    def draw(self, img, dets):
        out = img.convert("RGB").copy()
        d = ImageDraw.Draw(out)
        # colour fixed per part type (model class order), so it never shifts between photos or modes
        colour = {n: COLORS[i % len(COLORS)] for i, n in enumerate(self.names)}
        lw = max(2, round(max(out.size) / 500))
        for x in dets:
            d.rectangle(x["box"], outline=colour[x["name"]], width=lw)
        return out, {n: colour[n] for n in sorted({x["name"] for x in dets})}


def decode(out, s, px, py, w, h, score=0.30, iou=0.45):
    """YOLOv8 raw output (4 + nc, N) -> boxes in original pixels, per-class NMS.
    Mirrors YoloDecoder.java (checked by a shared parity fixture)."""
    boxes, scores = out[:4].T, out[4:].T
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf > score
    boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
    cx, cy, bw, bh = boxes.T if len(conf) else (np.zeros(0),) * 4
    x1 = np.clip((cx - bw / 2 - px) / s, 0, w); y1 = np.clip((cy - bh / 2 - py) / s, 0, h)
    x2 = np.clip((cx + bw / 2 - px) / s, 0, w); y2 = np.clip((cy + bh / 2 - py) / s, 0, h)
    ok = ((x2 - x1) >= 1) & ((y2 - y1) >= 1)
    x1, y1, x2, y2, cls, conf = x1[ok], y1[ok], x2[ok], y2[ok], cls[ok], conf[ok]
    boxes = np.stack([x1, y1, x2, y2], 1) if len(conf) else np.zeros((0, 4))
    return [{"class_id": int(cls[i]), "score": float(conf[i]), "box": tuple(float(v) for v in boxes[i])}
            for i in nms(boxes, conf, cls, iou)]


def nms(boxes, scores, cls, iou=0.45):
    """Greedy class-wise non-maximum suppression, highest score first (stable
    order on ties). A box is dropped when its IoU with a kept box of the same
    class is above `iou`. Same rule as YoloDecoder.java."""
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
        over = np.where(union > 0, inter / np.where(union > 0, union, 1), 0) > iou
        order = rest[~(over & (cls[rest] == cls[i]))]
    return keep


def tile_starts(length, tile, stride):
    """Tile origins covering [0, length): regular steps, last tile flush with the end."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    return starts + [length - tile]
