"""
Checks that every part of the project works.   Run:  pytest test_project.py

  data        the synthetic records are well-formed and reproducible
  ML models   reach their quality targets and behave sensibly
  tag OCR     reads the serial number on the 50 demo photos
  detector    decodes correctly and finds parts; tiling finds more
  chip OCR    names known chips and never invents one
  BOM / why   the parts list adds up; explanations add up to the prediction
  web app     runs end to end without errors
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

import pcb_ai
import train

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
SAMPLES = ROOT / "sample_images"
REGISTRY = pd.read_csv(ROOT / "data" / "board_registry.csv", dtype={"serial": str})
BOARDS = sorted(REGISTRY["board_type"].unique())
METRICS = json.loads((MODELS / "metrics.json").read_text())
CORROSION = "Green deposits on the pins, salt residue across the lower edge, moisture ingress."


def job(board_type="RF-COMM", fault="corrosion", age=10, prev=0):
    return pd.DataFrame([{"board_type": board_type, "fault_category": fault,
                          "age_years": age, "previous_repairs": prev}])


# ---------------------------------------------------------------- data
def test_repair_records_are_well_formed():
    df = pd.read_csv(ROOT / "data" / "repair_records.csv")
    assert len(df) == 600 and df.notna().all().all() and df["job_id"].is_unique
    assert set(df["fault_category"]) == set(train.FAULTS)
    assert set(df["outcome"]) == {"Repairable", "BER"}
    assert df["fault_category"].value_counts(normalize=True).min() > 0.10     # no tiny class
    assert 0.15 < (df["outcome"] == "BER").mean() < 0.45


def test_repair_records_are_reproducible():
    saved = pd.read_csv(ROOT / "data" / "repair_records.csv")
    pd.testing.assert_frame_equal(train.make_records(), saved)


def test_registry_matches_the_demo_photos():
    assert len(REGISTRY) == 50 and REGISTRY["serial"].is_unique
    assert all((SAMPLES / name).exists() for name in REGISTRY["image"])


# ---------------------------------------------------------------- ML models
@pytest.fixture(scope="module")
def models():
    return [joblib.load(MODELS / f"{n}.joblib") for n in ("fault_classifier", "outcome_model", "hours_model")]


def test_fault_classifier_quality():
    fc = METRICS["fault_classifier"]
    chosen = next(c for c in fc["comparison"] if c["model"] == fc["chosen"])
    assert chosen["cv_f1_mean"] >= 0.90 and chosen["test_f1_macro"] >= 0.90


def test_repair_predictor_beats_its_baselines():
    o, h = METRICS["outcome"], METRICS["hours"]
    assert o["test_roc_auc"] >= 0.75 and o["test_accuracy"] >= o["baseline_accuracy"]
    assert h["test_mae"] < 0.6 * h["baseline_mae"] and h["test_r2"] >= 0.8


@pytest.mark.parametrize("note,expected", [
    ("Dead short on the 3.3V rail, capacitor C12 shorted to ground.", "short_circuit"),
    ("Cracked solder joint on R7, no continuity on the trace.", "open_circuit"),
    ("Op amp U5 output stuck high, relay coil open.", "component_failure"),
    (CORROSION, "corrosion"),
    ("Connector J2 pins bent, loose contact at the plug.", "connector_fault"),
    ("Fuse blown on the input, 3.3V regulator output low.", "power_supply"),
])
def test_fault_examples(models, note, expected):
    assert models[0].predict([note])[0] == expected


def test_ber_risk_rises_with_age_and_corrosion(models):
    risk = lambda **kw: models[1].predict_proba(job(**kw))[0, 1]
    assert risk(age=20) > risk(age=2)
    assert risk(fault="corrosion") > risk(fault="connector_fault")


def test_retraining_reproduces_the_saved_metrics(tmp_path):
    fresh = train.train_ml(tmp_path)
    assert fresh["fault_classifier"]["comparison"] == METRICS["fault_classifier"]["comparison"]
    assert fresh["outcome"]["test_roc_auc"] == METRICS["outcome"]["test_roc_auc"]


# ---------------------------------------------------------------- tag OCR
@pytest.fixture(scope="module")
def ocr_results():
    def read(row):
        img = Image.open(SAMPLES / row["image"])
        return (pcb_ai.parse_label(pcb_ai.read_text(img, clean=True), BOARDS),
                pcb_ai.parse_label(pcb_ai.read_text(img, clean=False), BOARDS))
    rows = REGISTRY.to_dict("records")
    with ThreadPoolExecutor(4) as ex:
        return list(zip(rows, ex.map(read, rows)))


def test_tag_ocr_reads_the_serial_on_the_demo_photos(ocr_results):
    clean = sum(c["serial"] == row["serial"] for row, (c, _) in ocr_results)
    raw = sum(r["serial"] == row["serial"] for row, (_, r) in ocr_results)
    print(f"serials read: {clean}/50 with clean-up, {raw}/50 raw")
    assert clean >= 48 and clean > raw + 20
    # a missed read is acceptable (the technician picks the serial); a wrong one is not
    assert not [row["image"] for row, (c, _) in ocr_results if c["serial"] not in (None, row["serial"])]


@pytest.mark.parametrize("text,expected", [
    ("PSU-CTRL\nREV A\nS/N 2291", ("PSU-CTRL", "A", "2291")),
    ("RF - C0MM REVC S|N: 5O18", ("RF-COMM", "C", "5018")),          # OCR look-alikes are forgiven
    ("nothing useful here", (None, None, None)),
])
def test_parse_label(text, expected):
    p = pcb_ai.parse_label(text, BOARDS)
    assert (p["board_type"], p["revision"], p["serial"]) == expected


# ---------------------------------------------------------------- part detector
@pytest.fixture(scope="module")
def detector():
    return pcb_ai.PartDetector()


def test_decode_undoes_the_letterbox():
    out = np.zeros((6, 1), np.float32)
    out[:, 0] = [320, 320, 100, 50, 0.9, 0.1]          # a 1280x640 photo at 640: scale 0.5, 160 px padding
    boxes, scores, cls = pcb_ai.decode(out, 0.5, 0, 160, 1280, 640)
    assert np.allclose(boxes[0], (540, 270, 740, 370)) and cls[0] == 0


def test_nms_drops_duplicates_of_the_same_type_only():
    boxes = np.array([[80, 80, 120, 120], [82, 81, 122, 121], [82, 81, 122, 121]], float)
    keep = pcb_ai.nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]))
    assert keep == [0, 2]


def test_tiles_cover_the_whole_photo():
    assert pcb_ai.tile_starts(800, 1024, 768) == [0]
    for length in (1025, 2048, 2900):
        s = pcb_ai.tile_starts(length, 1024, 768)
        assert s[0] == 0 and s[-1] == length - 1024 and all(b - a <= 768 for a, b in zip(s, s[1:]))


def test_tiled_detection_finds_more_parts_than_a_single_pass(detector):
    single = tiled = 0
    for p in sorted(SAMPLES.glob("*.jpg"))[:5]:
        img = Image.open(p)
        parts = detector.detect_tiled(img)
        single, tiled = single + len(detector.detect(img)), tiled + len(parts)
        w, h = img.size
        assert all(0 <= x["box"][0] < x["box"][2] <= w and 0 <= x["box"][1] < x["box"][3] <= h for x in parts)
    assert tiled > 1.3 * single, (single, tiled)


def test_recorded_detector_scores_back_the_app():
    rows = json.loads((MODELS / "tiling_eval.json").read_text())["results"]
    used = next(r for r in rows if r.get("in_app"))
    assert used["map50_common"] == max(r["map50_common"] for r in rows) >= 0.50
    assert used["zoom"] == pcb_ai.PartDetector().zoom


# ---------------------------------------------------------------- chip OCR
CATALOGUE = pcb_ai.load_catalogue()
CHIPS = json.loads((ROOT / "data" / "ic_markings.json").read_text())["items"]


def matched(text):
    entry, _ = pcb_ai.match_catalogue(text, CATALOGUE)
    return entry and entry["key"]


def test_catalogue_matching_rules():
    assert matched("XILINX\nZYNQ\nXC7Z010 TM\nCLG400") == "XC7Z010"            # exact, ignoring spaces
    assert matched("hynix 222K\nHYSDVE41 622AT-\nKOREA") == "HY5DV641622"      # S/5 and E/6 misread
    assert matched("Ly\nAY Feu\nB84\nyf,\n99\nSS") is None                     # junk over several lines
    assert matched("VOVEDOTVHE\nSVNOSH") is None                               # letters never become digits


def test_chip_ocr_reads_clear_chips_and_claims_nothing_on_non_chips():
    def crop(it):
        return Image.open(SAMPLES / it["image"]).convert("RGB").crop(tuple(it["box"]))
    clear = [it for it in CHIPS if it["kind"] == "readable" and it["expected"][0] in
             {"XC7A100T", "5CEBA4F23", "82437FX", "AD9882", "GM1601", "MB91305"}]
    hits = sum((pcb_ai.read_ic(crop(it), CATALOGUE)["match"] or {}).get("key") in it["expected"] for it in clear)
    assert hits >= len(clear) - 1
    for it in [it for it in CHIPS if it["kind"] == "not_ic"][:6]:
        assert pcb_ai.read_ic(crop(it), CATALOGUE)["match"] is None, it


def test_recorded_chip_ocr_scores_back_the_app():
    ev = json.loads((MODELS / "ic_ocr_eval.json").read_text())
    p, b = ev["pipeline"], ev["baseline_plain_tesseract"]
    assert p["part_identified"] > 2 * b["part_identified"]
    assert p["wrong_part_on_readable"] == 0 and p["false_identifications"] == 0


# ---------------------------------------------------------------- BOM and explanations
def test_bom_lists_identified_chips_then_counts():
    parts = [{"name": n} for n in ["IC"] * 3 + ["Resistor"] * 5 + ["Capacitor"] * 2]
    chips = [{"match": {"key": "82437FX", "manufacturer": "Intel", "description": "system controller"}},
             {"match": None}]
    bom = pcb_ai.build_bom(parts, chips)
    assert list(bom["Part type"]) == ["IC", "IC", "Capacitor", "Resistor"]
    assert list(bom["Part number"])[:2] == ["82437FX", "(not identified)"]
    assert bom["Qty"].sum() == len(parts)
    assert len(pcb_ai.build_bom([], [])) == 0


def test_explanations_add_up_exactly_to_the_prediction(models):
    for X in (job("MCU-MAIN", "corrosion", 17, 2), job("RF-COMM", "short_circuit", 2, 0)):
        start, parts, total = pcb_ai.contributions(models[1], X)
        assert total == pytest.approx(models[1].predict_proba(X)[0, 1], abs=1e-9)
        assert start + sum(parts.values()) == pytest.approx(total)
        assert pcb_ai.contributions(models[2], X)[2] == pytest.approx(models[2].predict(X)[0], abs=1e-9)
    old, new = pcb_ai.contributions(models[1], job(age=18))[1], pcb_ai.contributions(models[1], job(age=1))[1]
    assert old["age_years"] > 0 > new["age_years"]            # old boards push towards scrapping


# ---------------------------------------------------------------- web app
def test_web_app_runs_the_whole_bench_flow():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 3
    serial = next(s for s in at.selectbox if s.label.startswith("Serial number"))
    assert serial.value is not None, "OCR should auto-fill the serial of the first demo photo"

    next(s for s in at.selectbox if s.label == "Example note").set_value(CORROSION).run()
    assert not at.exception, at.exception
    assert any(m.value == "Corrosion" for m in at.metric)
    assert any("Why this recommendation" in m.value for m in at.markdown)

    at.button[0].click().run()
    assert not at.exception, at.exception
    entry = at.session_state.log[0]
    assert entry["fault"] == "Corrosion" and entry["why"] and entry["parts_found"] > 0
