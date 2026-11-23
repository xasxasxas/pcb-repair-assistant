"""
Fine-tune YOLOv8n to detect PCB components, evaluate it on the held-out test
boards, export to ONNX and install it into the web app and the Android app.

Runs on CPU (GitHub Actions: .github/workflows/train-detector.yml).
  python detector/train.py --epochs 60 --hours 4.0
Smoke test (no pretrained weights, tiny):  python detector/train.py --smoke
CPU budget: about 3.5 min per epoch at 1024 px on a 4-core runner.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASET = HERE / "dataset"
TARGETS = [ROOT / "models", ROOT / "android" / "app" / "src" / "main" / "assets" / "models"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hours", type=float, default=4.0, help="training time budget")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--weights", default="yolov8n.pt", help="COCO-pretrained start point")
    ap.add_argument("--smoke", action="store_true", help="2-minute pipeline check, from scratch")
    ap.add_argument("--data", type=Path, default=DATASET,
                    help="dataset dir: detector/dataset (whole boards) or a prepare_tiles.py output")
    ap.add_argument("--device", default="cpu", help="'cpu', '0' for one GPU, '0,1' for two (Kaggle T4 x2)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--install", action="store_true", default=True,
                    help="copy the ONNX model into models/ and the Android assets")
    ap.add_argument("--no-install", dest="install", action="store_false")
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils import nms

    # Dense boards carry up to ~900 parts. Ultralytics' defaults (300 boxes per
    # image, ~0.05 s NMS budget per image) truncate detections on CPU and
    # under-report mAP, so evaluation gets a larger box cap and NMS time budget.
    _nms = nms.non_max_suppression
    nms.non_max_suppression = lambda *a, **k: _nms(*a, **{**k, "max_time_img": 2.0})

    data_yaml = args.data / "data.yaml"
    cfg = yaml.safe_load(data_yaml.read_text())
    cfg["path"] = str(args.data.resolve())
    run_yaml = HERE / "data_run.yaml"
    run_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))
    names = [cfg["names"][i] for i in sorted(cfg["names"])]

    if args.smoke:
        args.weights, args.epochs, args.imgsz, args.hours = "yolov8n.yaml", 1, 320, 0.05

    t0 = time.time()
    model = YOLO(args.weights)
    model.train(
        data=str(run_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, patience=20, time=args.hours, plots=False, project=str(HERE / "runs"), name="parts",
        exist_ok=True, seed=0, deterministic=True, fliplr=0.5, flipud=0.5, degrees=5, close_mosaic=5,
        fraction=0.15 if args.smoke else 1.0, val=True, amp=False, max_det=1000,
    )
    train_minutes = (time.time() - t0) / 60
    best = HERE / "runs" / "parts" / "weights" / "best.pt"
    model = YOLO(str(best))

    # Honest final score: the 22 test boards were never used for training or model selection
    m = model.val(data=str(run_yaml), split="test", imgsz=args.imgsz, batch=args.batch, device=args.device,
                  plots=False, project=str(HERE / "runs"), name="test", exist_ok=True, max_det=1000)
    per_class = {names[c]: round(float(m.box.ap50[i]), 3) for i, c in enumerate(m.box.ap_class_index)}

    onnx = Path(model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True, dynamic=False))
    summary = json.loads((args.data / "summary.json").read_text())
    boards = summary.get("images") or summary["boards"]
    meta = {
        "names": names,
        "imgsz": args.imgsz,
        "metrics": {
            "map50": round(float(m.box.map50), 3),
            "map50_95": round(float(m.box.map), 3),
            "precision": round(float(m.box.mp), 3),
            "recall": round(float(m.box.mr), 3),
        },
        "per_class_ap50": per_class,
        "training": {"base": args.weights, "epochs_requested": args.epochs, "imgsz": args.imgsz,
                     "minutes": round(train_minutes, 1), "smoke_test": args.smoke},
        "dataset": (f"Trained on {boards['train']} boards, validated on {boards['valid']}, "
                    f"tested on {boards['test']} unseen boards. Source: {summary['source']}."),
    }
    if "long_side" in summary:
        # Trained on tiles: the app must detect on tiles at the same pixel scale.
        meta["tiling"] = {"long_side": summary["long_side"], "tile": summary["tile"],
                          "overlap": summary["overlap"], "metrics_are": "per tile (see tiling_eval.json for whole boards)"}
        meta["training"]["device"] = args.device
    if not args.install:
        out = HERE / "runs" / "export"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx, out / "parts.onnx")
        (out / "parts.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return
    for t in TARGETS:
        t.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx, t / "parts.onnx")
        (t / "parts.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
