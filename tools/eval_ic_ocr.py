"""
Score IC marking OCR on the hand-labelled chip crops (tests/ic_markings.json).

  python tools/eval_ic_ocr.py [--out models/ic_ocr_eval.json]

For the 'readable' chips (a person can read the part number):
  * marking read   - the expected part number appears in the OCR text
                     (spaces/punctuation ignored)
  * part identified - the catalogue match is the right part
For the other crops (sticker/heatsink/unmarked/too small, or not a chip):
  * false identifications - the reader claims a catalogue part anyway
Baseline: plain Tesseract on the raw crop, no upscaling, no rotation search.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pytesseract
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ic_ocr import load_catalogue, match_catalogue, norm, read_ic  # noqa: E402
from ocr_utils import _single_thread_tesseract  # noqa: E402


def baseline(crop, catalogue):
    with _single_thread_tesseract():
        text = pytesseract.image_to_string(crop, config="--psm 6")
    entry, _ = match_catalogue(text, catalogue)
    return {"text": text, "match": entry and {"key": entry["key"]}}


def correct_match(match, expected):
    if not match:
        return False
    k = norm(match["key"])
    return any(k in norm(e) or norm(e) in k for e in expected)


def evaluate(items, reader, catalogue):
    res, t0 = [], time.time()
    for it in items:
        crop = Image.open(ROOT / "sample_images" / it["image"]).convert("RGB").crop(tuple(it["box"]))
        r = reader(crop, catalogue)
        res.append({"image": it["image"], "kind": it["kind"], "expected": it["expected"],
                    "read": r["text"].strip()[:120], "match": r["match"] and r["match"]["key"],
                    "rotation": r.get("rotation"),
                    "marking_read": any(norm(e) in norm(r["text"]) for e in it["expected"]),
                    "identified": correct_match(r["match"], it["expected"])})
    secs = (time.time() - t0) / len(items)
    readable = [x for x in res if x["kind"] == "readable"]
    other = [x for x in res if x["kind"] != "readable"]
    return {
        "readable_chips": len(readable),
        "marking_read": sum(x["marking_read"] for x in readable),
        "part_identified": sum(x["identified"] for x in readable),
        "wrong_part_on_readable": sum(bool(x["match"]) and not x["identified"] for x in readable),
        "other_crops": len(other),
        "false_identifications": sum(bool(x["match"]) for x in other),
        "seconds_per_chip": round(secs, 2),
        "details": res,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    items = json.loads((ROOT / "tests" / "ic_markings.json").read_text())["items"]
    catalogue = load_catalogue()
    report = {"pipeline": evaluate(items, read_ic, catalogue),
              "baseline_plain_tesseract": evaluate(items, baseline, catalogue)}
    for name, r in report.items():
        print(f"{name:26} read {r['marking_read']}/{r['readable_chips']}  identified {r['part_identified']}/"
              f"{r['readable_chips']}  wrong part {r['wrong_part_on_readable']}  false IDs on other crops "
              f"{r['false_identifications']}/{r['other_crops']}  {r['seconds_per_chip']} s/chip")
    for x in report["pipeline"]["details"]:
        if x["kind"] == "readable" and not x["identified"]:
            print("  MISS", x["image"], x["expected"], "->", x["match"], "|", x["read"].replace("\n", " / ")[:90])
    for x in report["pipeline"]["details"]:
        if x["kind"] != "readable" and x["match"]:
            print("  FALSE", x["image"], x["kind"], "->", x["match"], "|", x["read"].replace("\n", " / ")[:90])
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
