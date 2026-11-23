"""IC marking OCR: catalogue matching rules and accuracy on hand-labelled chip crops."""
import json

import pytest
from PIL import Image

from conftest import ROOT
from ic_ocr import load_catalogue, match_catalogue, read_ic

CAT = load_catalogue()
ITEMS = json.loads((ROOT / "tests" / "ic_markings.json").read_text())["items"]


def key(text):
    e, _ = match_catalogue(text, CAT)
    return e and e["key"]


def test_exact_match_ignores_spaces_and_punctuation():
    assert key("XILINX\nZYNQ\nXC7Z010 TM\nCLG400") == "XC7Z010"
    assert key("Micronas SDA 6001 B12") == "SDA6001"


def test_fuzzy_match_tolerates_ocr_lookalikes():
    assert key("hynix 222K\nHYSDVE41 622AT-\nKOREA") == "HY5DV641622"   # S/5, E/6 misread


def test_longest_exact_key_wins_between_siblings():
    assert key("VT8233A\n0232CE TAIWAN") == "VT8233A"


def test_junk_spread_over_lines_is_not_a_part():
    assert key("Ly\nAY Feu\nB84\nyf,\n99\nSS") is None
    assert key("VOVEDOTVHE\nSVNOSH") is None             # letters alone never fold into a part


def test_all_letter_keys_need_an_exact_match():
    assert key("MASC0T V2") is None
    assert key("MRT MASCOT V2") == "MASCOT"


def test_labels_are_consistent():
    kinds = {it["kind"] for it in ITEMS}
    assert kinds == {"readable", "unreadable", "not_ic"}
    assert all(it["expected"] for it in ITEMS if it["kind"] == "readable")
    assert not any(it["expected"] for it in ITEMS if it["kind"] != "readable")


def _crop(it):
    return Image.open(ROOT / "sample_images" / it["image"]).convert("RGB").crop(tuple(it["box"]))


def test_reads_clear_chips_and_claims_nothing_on_non_chips():
    """Quick subset (full run: tools/eval_ic_ocr.py)."""
    clear = [it for it in ITEMS if it["kind"] == "readable" and it["expected"][0] in
             {"XC7A100T", "5CEBA4F23", "82437FX", "AD9882", "GM1601", "MB91305"}]
    hits = 0
    for it in clear:
        m = read_ic(_crop(it), CAT)["match"]
        hits += bool(m) and m["key"] in it["expected"]
    assert hits >= len(clear) - 1, hits
    for it in [it for it in ITEMS if it["kind"] == "not_ic"][:6]:
        assert read_ic(_crop(it), CAT)["match"] is None, it


def test_recorded_evaluation_backs_the_app():
    path = ROOT / "models" / "ic_ocr_eval.json"
    if not path.exists():
        pytest.skip("IC OCR evaluation not run")
    ev = json.loads(path.read_text())
    p, b = ev["pipeline"], ev["baseline_plain_tesseract"]
    assert p["part_identified"] > 2 * b["part_identified"]
    assert p["wrong_part_on_readable"] == 0 and p["false_identifications"] <= 1
